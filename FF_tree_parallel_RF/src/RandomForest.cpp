#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include "RandomForest.h"

// FastFlow
#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>
using namespace std;

RandomForest::RandomForest(int n) : num_trees(n) {}
RandomForest::~RandomForest() { for(auto t : trees) delete t; }

void RandomForest::train_fastflow(const Dataset& data, size_t depth, size_t num_workers) {
    cout << "Starting FastFlow training with " << num_trees << " trees of depth " << depth << endl;
    
    int n_rows = data.rows;
    int n_cols = data.cols;
    trees.resize(num_trees);

    // vector to store depths of each tree for later reduction (openMP equivalent of reduction)
    std::vector<double> depths_vector(num_trees, 0.0);

    // initialize parallel for
    ff::ParallelFor pf(num_workers);

    /* pf.parallel_for(start, end, step, chunk, lambda)
       - chunk = 1 to simulate dynamic scheduling (each task is one iteration)
    */
    pf.parallel_for(0, num_trees, 1, 1, [&](const long i) {
        // --- BOOTSTRAP SAMPLING  ---
        Dataset bootstrap_data;
        bootstrap_data.rows = n_rows;
        bootstrap_data.cols = n_cols;
        bootstrap_data.features_flat.resize(n_rows * n_cols);
        bootstrap_data.labels.resize(n_rows);
        
        // Random sampling with replacement
        std::mt19937 gen(41 + i);
        std::uniform_int_distribution<> dis(0, n_rows - 1);
        
        vector<int> random_indices(n_rows);
        for(int j=0; j<n_rows; j++) random_indices[j] = dis(gen);

        for(int j=0; j<n_rows; j++) bootstrap_data.labels[j] = data.labels[random_indices[j]];

        for (int c = 0; c < n_cols; c++) {
            int src_offset = c * n_rows;
            int dst_offset = c * n_rows;
            for (int r = 0; r < n_rows; r++) {
                bootstrap_data.features_flat[dst_offset + r] = data.features_flat[src_offset + random_indices[r]];
            }
        }

        // --- TRAINING ---
        DecisionTree* tree = new DecisionTree(depth, 2); 
        int tree_depth = tree->fit(bootstrap_data);
        
        // save the trained tree and its depth
        trees[i] = tree;
        depths_vector[i] = (double)tree_depth;

        // Log (opzionale)
        if ((i + 1) % 10 == 0) {
            #pragma omp critical
            cout << "Albero " << i + 1 << " / " << num_trees << " completato." << endl;
        }
    }, num_workers);

    // Reduction finale: sommiamo i risultati raccolti nel vettore
    double total_depth = 0;
    for (double d : depths_vector) total_depth += d;
    mean_depth = total_depth / num_trees;
}

// function to return the mean depth of the forest
double RandomForest::get_average_depth(){
    return mean_depth;
}


std::vector<int> RandomForest::predict_fastflow(const Dataset& data, int num_workers) {
    int n_rows = data.rows;
    int n_cols = data.cols;

    // each worker has its own voting area to avoid contention it maps each worker to a vector of maps (one per row)
    std::vector<std::vector<std::map<int, int>>> worker_votes(num_workers, std::vector<std::map<int, int>>(n_rows));

    ff::ParallelFor pf(num_workers);

    /* parallelize on trees, differently from the training here the chunk is 0 since
         each task is very small (predicting one row) and we want to minimize scheduling overhead.
         For the trees we could have cases where one tree is much faster than another (unbalanced trees).
         
         to create a private voting area for each worker we use the my_id parameter of the lambda which is present
         only in the parallel_for_thid version.
    */

    pf.parallel_for_thid(0, num_trees, 1, 0, [&](const long i, const int my_id) {
        
        for (int r = 0; r < n_rows; r++) {
            std::vector<double> row(n_cols);
            for(int c = 0; c < n_cols; c++) 
                row[c] = data.features_flat[c * n_rows + r];

            int p = trees[i]->predict(row);
            
            // each worker writes on its private area 'my_id'
            worker_votes[my_id][r][p]++;
        }
    }, num_workers);

    // --- Merge votes from all workers ---
    std::vector<int> final_predictions(n_rows);
    for (int r = 0; r < n_rows; r++) {
        std::map<int, int> global_row_votes;
        for (int w = 0; w < num_workers; w++) {
            for (auto const& [label, count] : worker_votes[w][r]) {
                global_row_votes[label] += count;
            }
        }
        // winner
        int best_class = -1, max_v = -1;
        for (auto const& [cls, count] : global_row_votes) {
            if (count > max_v) { max_v = count; best_class = cls; }
        }
        final_predictions[r] = best_class;
    }
    return final_predictions;
}
