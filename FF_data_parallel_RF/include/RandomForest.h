#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "Tree.h"
#include "Data.h"
#include <vector>

class RandomForest {
    int num_trees;
    double mean_depth;
    std::vector<DecisionTree*> trees;

public:
    RandomForest(int n);
    ~RandomForest();
    double get_average_depth();
    void train_fastflow(const Dataset& data, size_t depth, size_t num_workers);
    std::vector<int> predict_fastflow(const Dataset& data, int num_threads);
};

#endif