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
    // Prepare a structure to hold votes
    // Each entry in global_votes corresponds to a row in data
    std::vector<std::map<int, int>> global_votes(data.rows);
    
    int n_rows = data.rows;
    int n_cols = data.cols;

    // Parallel region on trees
    #pragma omp parallel
    {
        // Each thread has its own local votes to avoid contention
        std::vector<std::map<int, int>> local_votes(n_rows);

        // each thread processes a subset of trees
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_trees; i++) {
            
            // For each tree assigned to this thread we must iterate through the entire dataset
            for (int r = 0; r < n_rows; r++) {
                
                // rebuild the row vector from the flattened features
                std::vector<double> row(n_cols);
                for(int c = 0; c < n_cols; c++) {
                    row[c] = data.features_flat[c * n_rows + r];
                }

                int prediction = trees[i]->predict(row);
                
                // write the local vote (no LOCK needed)
                local_votes[r][prediction]++;
            }
        }

        //Merge all local votes into global votes (CRITICAL SECTION)
        #pragma omp critical
        {
            for (int r = 0; r < n_rows; r++) {
                for (auto const& [label, count] : local_votes[r]) {
                    global_votes[r][label] += count;
                }
            }
        }
    } // end of the parlallel region

    // calculate final predictions based on global votes
    std::vector<int> predictions;
    predictions.reserve(n_rows);

    for (int i = 0; i < n_rows; i++) {
        int best_class = -1, max_votes = -1;
        for (auto const& [cls, count] : global_votes[i]) {
            if (count > max_votes) {
                max_votes = count;
                best_class = cls;
            }
        }
        predictions.push_back(best_class);
    }

    return predictions;
}
