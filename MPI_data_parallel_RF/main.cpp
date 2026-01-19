#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#define _XOPEN_SOURCE 700
#include <chrono>
#include <algorithm>
#include <map>
#include <cmath>
#include <sys/stat.h>
#include <set>
#include <limits>
#include <random>  // Per std::default_random_engine
#include <algorithm> // Per std::shuffle
#include <omp.h>
#include <mpi.h>

#include "Tree.h"
#include "Data.h"
#include "RandomForest.h"

using namespace std;

// --- CALCOLO METRICHE SEMPLIFICATO ---
void print_metrics(const vector<int>& predictions, const vector<int>& true_labels) {
    if (predictions.size() != true_labels.size()) return;

    int n_samples = predictions.size();
    int correct_total = 0;
    
    // Mappe per contare TP, Totali Reali (TP+FN), Totali Predetti (TP+FP)
    set<int> classes;
    map<int, int> tp_counts;
    map<int, int> true_counts;
    map<int, int> pred_counts;

    for (int i = 0; i < n_samples; i++) {
        int t = true_labels[i];
        int p = predictions[i];
        
        classes.insert(t);
        classes.insert(p);
        
        true_counts[t]++;
        pred_counts[p]++;
        
        if (t == p) {
            correct_total++;
            tp_counts[t]++;
        }
    }

    // 1. Accuracy
    double accuracy = (double)correct_total / n_samples;

    // 2. Macro Recall e F1
    double sum_recall = 0.0;
    double sum_f1 = 0.0;

    for (int c : classes) {
        int tp = tp_counts[c];
        int actual_positives = true_counts[c]; // TP + FN
        int predicted_positives = pred_counts[c]; // TP + FP

        // Recall = TP / (TP + FN)
        double recall = (actual_positives == 0) ? 0.0 : (double)tp / actual_positives;
        
        // Precision = TP / (TP + FP) (Serve per F1)
        double precision = (predicted_positives == 0) ? 0.0 : (double)tp / predicted_positives;

        // F1 Score
        double f1 = 0.0;
        if (precision + recall > 0) {
            f1 = 2.0 * (precision * recall) / (precision + recall);
        }

        sum_recall += recall;
        sum_f1 += f1;
    }

    double macro_recall = classes.empty() ? 0.0 : sum_recall / classes.size();
    double macro_f1 = classes.empty() ? 0.0 : sum_f1 / classes.size();

    // Print results
    cout << "-----------------------------------" << endl;
    cout << "Accuracy:      " << fixed << accuracy * 100.0 << "%" << endl;
    cout << "Macro-Recall:  " << fixed << macro_recall << endl;
    cout << "Macro-F1:      " << fixed << macro_f1 << endl;
    cout << "-----------------------------------" << endl;
}

// Function to save predictions to CSV
void save_predictions_to_csv(const vector<int>& predictions, const string& filename) {
    ofstream outfile(filename);
    if (!outfile.is_open()) return;
    outfile << "Predicted_Label" << endl;
    for (size_t i = 0; i < predictions.size(); i++) {
        outfile << predictions[i] << "\n";
    }
    outfile.close();
}
// Function to check if a file is empty or does not exist
bool is_file_empty(const string& filename) {
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        return true; // File does not exist
    }
    return st.st_size == 0; // Returns true if file size is 0
}

void save_times_to_csv(const string& dataset, int size, int num_trees, double avg_depth, double training, double prediction, const string& filename) {
    // Open the file in APPEND mode (std::ios::app)
    bool write_header = is_file_empty(filename);
    ofstream outfile(filename, std::ios::app);
    
    if (!outfile.is_open()) return;

    // Write the header only if the file was just created
    if (write_header) {
        outfile << "Dataset,Num_Processes,Num_Trees,Avg_Depth,Training_Time,Prediction_Time,Total_Time" << endl;
    }

    // Calculate total execution time
    double total_time = training + prediction;

    // Append the new measurements to the end of the file
    outfile << dataset << ","
            << size << "," 
            << num_trees << ","
            << avg_depth << "," 
            << training << "," 
            << prediction << "," 
            << total_time << "\n";

    outfile.close();
}

int main(int argc, char* argv[]) {
    // 1. INIZIALIZZAZIONE MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check input
    if (argc < 4) {
        if (rank == 0) {
            cout << "Uso: " << argv[0] << " <file_csv> <num_alberi> <depth>" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    Dataset trainData, testData;
    int num_trees;
    size_t depth;
    string filename = "not_defined";

    // 2. CARICAMENTO DATI
    // In un sistema cluster, spesso è più semplice che ogni rank legga il file (se il filesystem è condiviso)
    // o che il Rank 0 legga e faccia Bcast. Qui manteniamo la lettura per ogni rank per semplicità.
    if (argc == 5){
        string train_name = argv[1];
        string test_name = argv[2];
        trainData = load_csv_dataset(train_name);
        testData = load_csv_dataset(test_name);
        num_trees = stoi(argv[3]);
        depth = stoi(argv[4]);
    } else {
        filename = argv[1];
        Dataset allData = load_csv_dataset(filename);
        int seed = 45;
        double train_ratio = 0.8;
        split_dataset(allData, trainData, testData, seed, train_ratio);
        num_trees = stoi(argv[2]);
        depth = stoi(argv[3]);
    }

    // 3. LOGGING (Solo Rank 0)
    if (rank == 0) {
        cout << "------------------------------------------------" << endl;
        cout << "MPI Processes (Nodes): " << size << endl;
        cout << "OpenMP Threads per process: " << omp_get_max_threads() << endl;
        cout << "Total hardware threads: " << size * omp_get_max_threads() << endl;
        cout << "------------------------------------------------" << endl;
    }

    RandomForest rf(num_trees);

    // 4. TRAINING 
    // Nota: se non hai ancora un 'train_hybrid', qui ogni processo allenerà num_trees.
    // Sarebbe meglio allenare num_trees/size per ogni processo.
    auto training_start = chrono::high_resolution_clock::now();
    
    rf.train(trainData, depth); // Chiamata alla tua funzione ibrida
    
    // Sincronizziamo i processi per misurare il tempo di training totale
    MPI_Barrier(MPI_COMM_WORLD);
    auto training_end = chrono::high_resolution_clock::now();

    if (rank == 0) {
        chrono::duration<double> elapsed = training_end - training_start;
        cout << "Tempo di Training Totale: " << elapsed.count() << " secondi." << endl;
        cout << "Average depth: " << rf.get_average_depth() << endl;
    }

    // 5. PREDICTION
    if (rank == 0) cout << "------------------------------------------------" << endl;
    auto start_pred = chrono::high_resolution_clock::now();

    // Ogni rank calcola una parte, ma solo il Rank 0 riceve il vettore completo (grazie a MPI_Reduce/Gather interna)
    vector<int> predictions = rf.predict(testData); 

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_pred = chrono::high_resolution_clock::now();

    // 6. OUTPUT E METRICHE (Solo Rank 0)
    if (rank == 0) {
        chrono::duration<double> elapsed_pred = end_pred - start_pred;
        cout << "Tempo di Predizione Totale: " << elapsed_pred.count() << " secondi." << endl;

        print_metrics(predictions, testData.labels);
        save_predictions_to_csv(predictions, "predictions.csv");
        
        double train_time = chrono::duration<double>(training_end - training_start).count();
        save_times_to_csv(filename, size, num_trees, rf.get_average_depth(), train_time, elapsed_pred.count(), "times.csv");
    }

    // 7. FINALIZZAZIONE
    MPI_Finalize();
    return 0;
}