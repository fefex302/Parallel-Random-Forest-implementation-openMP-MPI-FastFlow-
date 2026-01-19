#include "Tree.h"
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <numeric>

using namespace std;

// Constructor and Destructor
DecisionTree::DecisionTree(int depth, int min_samples) : max_depth(depth), min_size(min_samples) {}
DecisionTree::~DecisionTree() { delete root; }

// Optimized best split search using flat feature storage
void DecisionTree::get_best_split(const vector<double>& features_flat, int n_total_rows,
                                  const vector<int>& labels,
                                  const vector<int>& node_indices, 
                                  int& best_feat, double& best_thresh, double& best_gini, 
                                  vector<int>& left_idx, vector<int>& right_idx) {
    
    // initialize bests
    best_gini = numeric_limits<double>::max();
    int n_subset = node_indices.size();
    if (n_subset < 2) return;
    
    // number of features, inferred from flat storage
    int n_cols = features_flat.size() / n_total_rows;

    // total_counts maps class label -> count for the current node
    map<int, int> total_counts;
    for (int idx : node_indices) total_counts[labels[idx]]++;

    vector<int> sorted_indices = node_indices; 

    for (int f = 0; f < n_cols; f++) {
        // cache optimization: pointer to the start of the column 
        const double* col_ptr = &features_flat[f * n_total_rows];

        // The sort is now extremely fast because the lambda reads sequential memory
        stable_sort(sorted_indices.begin(), sorted_indices.end(), [col_ptr](int a, int b) {
            return col_ptr[a] < col_ptr[b];
        });

        // initialize counts for left and right
        map<int, int> left_counts;
        map<int, int> right_counts = total_counts;
        int n_left = 0;
        int n_right = n_subset;
        double sum_sq_left = 0.0;
        double sum_sq_right = 0.0;
        for(auto const& [l, c] : right_counts) sum_sq_right += (double)c*c;

        // in this loop we consider splits between i and i+1 in sorted order, in this way Gini can be updated incrementally
        // avoiding re-computation from scratch
        for (int i = 0; i < n_subset - 1; i++) {
            int idx = sorted_indices[i];
            int label = labels[idx];
            
            double val = col_ptr[idx];
            double next_val = col_ptr[sorted_indices[i+1]];

            double c_r = right_counts[label];
            sum_sq_right -= c_r * c_r;
            right_counts[label]--;
            sum_sq_right += (c_r - 1.0) * (c_r - 1.0);
            n_right--;

            double c_l = left_counts[label];
            sum_sq_left -= c_l * c_l;
            left_counts[label]++;
            sum_sq_left += (c_l + 1.0) * (c_l + 1.0);
            n_left++;

            if (val == next_val) continue;

            double gini_left = 1.0 - (sum_sq_left / ((double)n_left * n_left));
            double gini_right = 1.0 - (sum_sq_right / ((double)n_right * n_right));
            double weighted = ((double)n_left / n_subset) * gini_left + ((double)n_right / n_subset) * gini_right;

            if (weighted < best_gini) {
                best_gini = weighted;
                best_feat = f;
                best_thresh = (val + next_val) / 2.0;
            }
        }
    }

    // After finding the best split, we need to fill left_idx and right_idx
    if (best_gini != numeric_limits<double>::max()) {
        left_idx.reserve(n_subset); 
        right_idx.reserve(n_subset);
        
        // pointer to the best feature column
        const double* best_col_ptr = &features_flat[best_feat * n_total_rows];
        
        // 
        for (int idx : node_indices) {
            if (best_col_ptr[idx] < best_thresh)
                left_idx.push_back(idx);
            else
                right_idx.push_back(idx);
        }
        
    }
}
// --- BUILD RECURSIVE: builds the tree using TrainNode pointers ---
// features_flat: flattened feature matrix
// n_total_rows: total number of rows in the dataset
// labels: vector of labels
// node_indices: indices of the current node's samples  
// depth: current depth in the tree
pair<TrainNode*,int> DecisionTree::build_recursive(const vector<double>& features_flat, int n_total_rows,
                                         const vector<int>& labels, 
                                         const vector<int>& node_indices, 
                                         int depth) {
    TrainNode* node = new TrainNode();

    // starts by 
    bool all_same = true;
    int first_label = labels[node_indices[0]];
    for (size_t i = 1; i < node_indices.size(); i++) {
        if (labels[node_indices[i]] != first_label) { all_same = false; break; }
    }

    // Base case: create a leaf when:
    // 1. max depth reached
    // 2. min samples reached 
    // 3. all labels are the same (no need to split further)
    if (depth >= max_depth || node_indices.size() <= (size_t)min_size || all_same) {
        node->is_leaf = true;

        // majority class voting
        map<int, int> counts;
        for (int idx : node_indices) counts[labels[idx]]++;
        int most_freq = -1, max_c = -1;

        // save into most_freq the label with the highest count
        for (auto p : counts) if (p.second > max_c) { max_c = p.second; most_freq = p.first; }
        node->label = most_freq;
        return make_pair(node, depth);
    }

    // Find the best Split, best_feat will be the feature index, best_thresh the threshold
    // then we initialize the left and right indices 
    int best_feat = 0;
    double best_thresh = 0.0, best_gini = 1.0;
    vector<int> left_idx, right_idx;

    // find the best split
    get_best_split(features_flat, n_total_rows, labels, node_indices, best_feat, best_thresh, best_gini, left_idx, right_idx);

    // if no valid split found, make leaf (this can happen if all features have same value but different labels)
    if (left_idx.empty() || right_idx.empty()) {
        node->is_leaf = true;
        map<int, int> counts;
        for (int idx : node_indices) counts[labels[idx]]++;
        int most_freq = -1, max_c = -1;
        for (auto p : counts) if (p.second > max_c) { max_c = p.second; most_freq = p.first; }
        node->label = most_freq;
        return make_pair(node, depth);
    }

    // create decision node
    node->feature_index = best_feat;
    node->threshold = best_thresh;
    
    // continue recursion
    auto res_left = build_recursive(features_flat, n_total_rows, labels, left_idx, depth + 1);
    auto res_right = build_recursive(features_flat, n_total_rows, labels, right_idx, depth + 1);
    node->left = res_left.first;
    node->right = res_right.first;
    int max_depth = std::max(res_left.second, res_right.second);

    return make_pair(node, max_depth);
}

// Flatten the tree into flat_nodes vector
int DecisionTree::flatten_tree(TrainNode* node) {
    if (!node) return -1;

    // creates a new FlatNode and gets its index
    int current_idx = flat_nodes.size();
    flat_nodes.push_back(FlatNode());

    // copy data from TrainNode to FlatNode
    flat_nodes[current_idx].is_leaf = node->is_leaf;
    flat_nodes[current_idx].label = node->label;
    flat_nodes[current_idx].feature_index = node->feature_index;
    flat_nodes[current_idx].threshold = node->threshold;

    // Children recursion
    if (node->left) {
        int left_idx = flatten_tree(node->left);
        flat_nodes[current_idx].left_idx = left_idx;
    }
    if (node->right) {
        int right_idx = flatten_tree(node->right);
        flat_nodes[current_idx].right_idx = right_idx;
    }

    return current_idx;
}

int DecisionTree::fit(const Dataset& train_data) {
    // Clean previous trees
    delete root; 
    root = nullptr;
    flat_nodes.clear();
    
    // Training
    vector<int> all_indices(train_data.rows);
    iota(all_indices.begin(), all_indices.end(), 0);
    std::pair<TrainNode*, int> result = build_recursive(train_data.features_flat, train_data.rows, train_data.labels, all_indices, 0);

    // Convert into a flattened structure for prediction
    flat_nodes.reserve(2048); 
    flatten_tree(result.first);

    // return the depth
    return result.second;

}

// Prediction using flattened tree
int DecisionTree::predict_one_flat(int node_idx, const vector<double>& row) const {
    const FlatNode& node = flat_nodes[node_idx];

    if (node.is_leaf) return node.label;

    if (row[node.feature_index] < node.threshold) 
        return predict_one_flat(node.left_idx, row);
    else 
        return predict_one_flat(node.right_idx, row);
}

int DecisionTree::predict(const vector<double>& row) const { 
    if (flat_nodes.empty()) return 0;
    return predict_one_flat(0, row);
}

int DecisionTree::get_depth() {
    return max_depth;
}