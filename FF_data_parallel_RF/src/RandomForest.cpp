#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include "RandomForest.h"
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

    ff::parallel_for(0, num_trees, 1, 1, [&](const long i) {
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

    // final reduction to compute mean depth
    double total_depth = 0;
    for (double d : depths_vector) total_depth += d;
    mean_depth = total_depth / num_trees;
}

// function to return the mean depth of the forest
double RandomForest::get_average_depth(){
    return mean_depth;
}

std::vector<int> RandomForest::predict_fastflow(const Dataset& data, int num_threads) {
    int n_rows = data.rows;
    int n_cols = data.cols;
    
    // Vector to save predictions
    std::vector<int> predictions(n_rows);

    // Data parallelism, each thread handles a subset of rows
    // for example thread 0 handles rows 0:10000, thread 1 handles 10001:20000, etc.
    ff::parallel_for(0, n_rows, 1, 1, [&](const long i) {
        
        // Extract the row (rebuild the row from column-major format)
        std::vector<double> row(n_cols);
        for(int c = 0; c < n_cols; c++) {
            row[c] = data.features_flat[c * n_rows + i];
        }

        // Local voting map
        std::map<int, int> local_votes;
        
        // For each tree, get prediction
        for (const auto& tree : trees) {
            int prediction = tree->predict(row);
            local_votes[prediction]++;
        }
        
        // Determine the class with the most votes
        int best_class = -1; 
        int max_votes = -1;
        
        for (auto const& [cls, count] : local_votes) {
            if (count > max_votes) {
                max_votes = count;
                best_class = cls;
            }
        }
        
        // safely write the prediction because each thread writes to a unique index
        predictions[i] = best_class;
    },num_threads);

    return predictions;
}