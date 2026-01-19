#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <vector>
#include "Data.h"

// Training node
struct TrainNode {
    bool is_leaf = false;
    int label = -1;
    int feature_index = 0;
    double threshold = 0.0;
    
    TrainNode* left = nullptr;
    TrainNode* right = nullptr;
    
    ~TrainNode() { delete left; delete right; }
};

// Prediction node (flattened)
struct FlatNode {
    bool is_leaf = false;
    int label = -1;
    int feature_index = 0;
    double threshold = 0.0;
    
    // indices of children in the flat array
    int left_idx = -1;
    int right_idx = -1;
};

class DecisionTree {
    // Training phase
    TrainNode* root = nullptr;
    
    // Prediction Phase (flattened)
    std::vector<FlatNode> flat_nodes;

    int max_depth;
    int min_size;

    // Training helper
    double gini_index(const std::vector<int>& labels, const std::vector<int>& indices);
    
    // function to find best split
    void get_best_split(const std::vector<double>& features_flat, int n_total_rows,
                        const std::vector<int>& labels,
                        const std::vector<int>& node_indices, 
                        int& best_feat, double& best_thresh, double& best_gini, 
                        std::vector<int>& left_idx, std::vector<int>& right_idx);
                        
    // recursive build (returns node pointer and number of nodes created)
    std::pair<TrainNode*, int> build_recursive(const std::vector<double>& features_flat, int n_total_rows,
                               const std::vector<int>& labels, 
                               const std::vector<int>& node_indices, 
                               int depth);

    // function to flatten the tree
    int flatten_tree(TrainNode* node);

    // Prediction uses flattened tree
    int predict_one_flat(int node_idx, const std::vector<double>& row) const;

public:
    DecisionTree(int depth = 10, int min_samples = 2);
    ~DecisionTree();

    int get_depth();
    int fit(const Dataset& train_data);
    int predict(const std::vector<double>& row) const;
};

#endif