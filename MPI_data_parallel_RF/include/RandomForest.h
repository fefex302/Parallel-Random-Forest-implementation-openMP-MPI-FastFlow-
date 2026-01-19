#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "Tree.h"
#include "Data.h"
#include <vector>

class RandomForest {
    int num_trees;
    int num_classes;
    double mean_depth;
    std::vector<DecisionTree*> trees;

public:
    RandomForest(int n);
    ~RandomForest();
    double get_average_depth();
    void train(const Dataset& data, size_t depth);
    std::vector<int> predict(const Dataset& data);
};

#endif