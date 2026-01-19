#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <omp.h>
#include "RandomForest.h"
using namespace std;

RandomForest::RandomForest(int n) : num_trees(n) {}
RandomForest::~RandomForest() { for(auto t : trees) delete t; }

// Train the Random Forest
void RandomForest::train(const Dataset& data, size_t depth) {
    cout << "Starting training with " << num_trees << " trees of depth " << depth << endl;
    
    double current_mean_depth = 0;
    int n_rows = data.rows;
    int n_cols = data.cols;
    trees.resize(num_trees);

    // Parallelization with OpenMP on the different trees
    #pragma omp parallel for schedule(dynamic) reduction(+:current_mean_depth)
    for (int i = 0; i < num_trees; i++) {
        // Creates bootstrap sample
        Dataset bootstrap_data;
        bootstrap_data.rows = n_rows;
        bootstrap_data.cols = n_cols;
        bootstrap_data.features_flat.resize(n_rows * n_cols);
        bootstrap_data.labels.resize(n_rows);

        std::mt19937 gen(41 + i); 
        std::uniform_int_distribution<> dis(0, n_rows - 1);
        
        // generate random indices for bootstrap
        vector<int> random_indices(n_rows);
        for(int j=0; j<n_rows; j++) random_indices[j] = dis(gen);

        // copy labels
        for(int j=0; j<n_rows; j++) bootstrap_data.labels[j] = data.labels[random_indices[j]];

        // efficiently copy features (column-major)
        for (int c = 0; c < n_cols; c++) {
            int src_offset = c * n_rows;
            int dst_offset = c * n_rows;
            
            for (int r = 0; r < n_rows; r++) {
                int original_idx = random_indices[r];
                bootstrap_data.features_flat[dst_offset + r] = data.features_flat[src_offset + original_idx];
            }
        }

        // Train the tree with bootstrap sample
        DecisionTree* tree = new DecisionTree(depth, 2); 
        int tree_depth = tree->fit(bootstrap_data);
        trees[i] = tree;

        current_mean_depth += tree_depth;
        if ((i+1) % 10 == 0) cout << "Albero " << i+1 << " / " << num_trees << " completato." << endl;
    }
    mean_depth = current_mean_depth / num_trees;
}

// function to return the mean depth of the forest
double RandomForest::get_average_depth(){
    return mean_depth;
}

std::vector<int> RandomForest::predict(const Dataset& data) {
    int n_rows = data.rows;
    int n_cols = data.cols;
    
    // Vector to save predictions
    std::vector<int> predictions(n_rows);

    // Data parallelism, each thread handles a subset of rows
    // for example thread 0 handles rows 0:10000, thread 1 handles 10001:20000, etc.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_rows; i++) {
        
        // estract the row (rebuild the row from column-major format)
        std::vector<double> row(n_cols);
        for(int c = 0; c < n_cols; c++) {
            row[c] = data.features_flat[c * n_rows + i];
        }

        // Map to count votes per class for this row
        std::map<int, int> local_votes;
        
        // each tree makes a prediction
        for (const auto& tree : trees) {
            int prediction = tree->predict(row);
            local_votes[prediction]++;
        }
        
        // find the class with the maximum votes
        int best_class = -1; 
        int max_votes = -1;
        
        for (auto const& [cls, count] : local_votes) {
            if (count > max_votes) {
                max_votes = count;
                best_class = cls;
            }
        }
        
        // write the prediction for this row (thread-safe because each thread writes to a unique index)
        predictions[i] = best_class;
    }

    return predictions;
}