#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <omp.h>
#include <mpi.h>
#include "RandomForest.h"

using namespace std;

RandomForest::RandomForest(int n) : num_trees(n) {}
RandomForest::~RandomForest() { for(auto t : trees) delete t; }

// --- TRAINING IBRIDO ---
void RandomForest::train(const Dataset& data, size_t depth) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // DETERMINA IL NUMERO DI CLASSI (se non lo sai già)
    int local_max = 0;
    for(int label : data.labels) if(label > local_max) local_max = label;
    
    // Sincronizza il massimo tra tutti i processi MPI
    MPI_Allreduce(&local_max, &this->num_classes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    // Se le etichette partono da 1, num_classes deve essere max_label + 1
    this->num_classes++;

    // Dividiamo il lavoro: quanti alberi deve fare questo processo?
    int trees_per_proc = num_trees / size;
    int start_tree = rank * trees_per_proc;
    int end_tree = (rank == size - 1) ? num_trees : (rank + 1) * trees_per_proc;
    int local_num_trees = end_tree - start_tree;

    if (rank == 0) {
        cout << "Starting hybrid training: " << size << " MPI ranks, " 
             << omp_get_max_threads() << " threads each." << endl;
    }

    double local_sum_depth = 0;
    int n_rows = data.rows;
    int n_cols = data.cols;
    
    // In questa versione, il vettore 'trees' del singolo processo 
    // conterrà solo gli alberi locali.
    trees.resize(local_num_trees);

    // Parallelismo OpenMP sugli alberi LOCALI assegnati al rank
    #pragma omp parallel for schedule(dynamic) reduction(+:local_sum_depth)
    for (int i = 0; i < local_num_trees; i++) {
        // Ogni albero usa un seed diverso basato sul rank e sull'indice locale
        int global_tree_idx = start_tree + i;
        
        Dataset bootstrap_data;
        bootstrap_data.rows = n_rows;
        bootstrap_data.cols = n_cols;
        bootstrap_data.features_flat.resize(n_rows * n_cols);
        bootstrap_data.labels.resize(n_rows);

        std::mt19937 gen(41 + global_tree_idx); 
        std::uniform_int_distribution<> dis(0, n_rows - 1);
        
        vector<int> random_indices(n_rows);
        for(int j=0; j<n_rows; j++) random_indices[j] = dis(gen);

        for(int j=0; j<n_rows; j++) bootstrap_data.labels[j] = data.labels[random_indices[j]];

        for (int c = 0; c < n_cols; c++) {
            int offset = c * n_rows;
            for (int r = 0; r < n_rows; r++) {
                bootstrap_data.features_flat[offset + r] = data.features_flat[offset + random_indices[r]];
            }
        }

        DecisionTree* tree = new DecisionTree(depth, 2); 
        local_sum_depth += tree->fit(bootstrap_data);
        trees[i] = tree;

        if (rank == 0 && (i+1) % 10 == 0) 
            cout << "Rank 0: Albero " << i+1 << " / " << local_num_trees << " completato." << endl;
    }

    // Riduzione MPI per calcolare la profondità media globale
    double global_sum_depth = 0;
    MPI_Reduce(&local_sum_depth, &global_sum_depth, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        mean_depth = global_sum_depth / num_trees;
    }
}
// function to return the mean depth of the forest
double RandomForest::get_average_depth(){
    return mean_depth;
}

// --- PREDICTION IBRIDA ---
std::vector<int> RandomForest::predict(const Dataset& data) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_rows = data.rows;
    int n_cols = data.cols;
    
    // Assumiamo di conoscere il numero di classi (es. 10). 
    // In un caso reale, questo valore verrebbe dal training o dal dataset.
    int n_classes = this->num_classes; 

    // Matrice dei voti locale: ogni processo conta i voti dei suoi alberi
    // Usiamo un vettore "flat" per facilitare la MPI_Reduce
    std::vector<int> local_votes_matrix(n_rows * n_classes, 0);

    // OpenMP parallelizza sulle righe del dataset
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_rows; i++) {
        std::vector<double> row(n_cols);
        for(int c = 0; c < n_cols; c++) {
            row[c] = data.features_flat[c * n_rows + i];
        }

        // Ogni processo interroga SOLO i suoi alberi locali
        for (const auto& tree : trees) {
            int p = tree->predict(row);
            
            // Protezione atomica perché più thread scrivono sulla stessa riga della matrice locale
            #pragma omp atomic
            local_votes_matrix[i * n_classes + p]++;
        }
    }

    // Vettore per i risultati finali (usato solo dal Master)
    std::vector<int> final_predictions;
    std::vector<int> global_votes_matrix;

    if (rank == 0) {
        global_votes_matrix.resize(n_rows * n_classes, 0);
        final_predictions.resize(n_rows);
    }

    // --- RIDUZIONE MPI ---
    // Sommiamo le matrici dei voti di tutti i processi sul Rank 0
    
    MPI_Reduce(local_votes_matrix.data(), 
               rank == 0 ? global_votes_matrix.data() : nullptr, 
               n_rows * n_classes, 
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Solo il Master calcola i vincitori finali
    if (rank == 0) {
        for (int i = 0; i < n_rows; i++) {
            int best_class = -1;
            int max_votes = -1;
            for (int c = 0; c < n_classes; c++) {
                int votes = global_votes_matrix[i * n_classes + c];
                if (votes > max_votes) {
                    max_votes = votes;
                    best_class = c;
                }
            }
            final_predictions[i] = best_class;
        }
    }

    return final_predictions; 
}