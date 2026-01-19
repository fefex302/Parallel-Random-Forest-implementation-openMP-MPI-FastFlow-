#include "Data.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>

using namespace std;

Dataset load_csv_dataset(const string& filename) {
    Dataset data;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(1);
    }

    // Temporary storage in row-major format
    vector<vector<double>> temp_rows;
    
    while (getline(file, line)) {
        stringstream ss(line);
        string val_str;
        vector<double> row_features;
        
        while (getline(ss, val_str, ',')) {
            row_features.push_back(stod(val_str));
        }
        
        // If a row have at least one feature the last value is the label
        if (!row_features.empty()) {
            // take last value as label
            int label = (int)row_features.back();

            // remove label from features
            row_features.pop_back();
            
            // Store the row
            temp_rows.push_back(row_features);
            // Store the label
            data.labels.push_back(label);
        }
    }
    
    data.rows = temp_rows.size();

    // if any row exists, infer number of columns from the first row
    if (data.rows > 0) data.cols = temp_rows[0].size();

    // Now convert to column-major format
    // [Col0_R0, Col0_R1, ..., Col1_R0, Col1_R1, ...]
    data.features_flat.resize(data.rows * data.cols);
    
    // Convert from row-major to column-major
    for (int c = 0; c < data.cols; c++) {
        for (int r = 0; r < data.rows; r++) {
            // column c row r goes to index c * rows + r in flat array
            data.features_flat[c * data.rows + r] = temp_rows[r][c];
        }
    }
    
    cout << "Loaded dataset: " << data.rows << " rows, " << data.cols << " columns." << endl;
    return data;
}

// split_dataset divides the dataset into training and test sets
void split_dataset(const Dataset& all_data, Dataset& train, Dataset& test, unsigned seed, float train_ratio) {
    cout << "Splitting dataset with train ratio: " << train_ratio << endl;
    int total_rows = all_data.rows;
    int n_cols = all_data.cols;
    int train_rows = (int)(total_rows * train_ratio);
    int test_rows = total_rows - train_rows;

    train.rows = train_rows; train.cols = n_cols;
    test.rows = test_rows; test.cols = n_cols;
    
    // memory allocation
    train.features_flat.resize(train_rows * n_cols);
    train.labels.resize(train_rows);
    test.features_flat.resize(test_rows * n_cols);
    test.labels.resize(test_rows);
    cout << "Allocated memory for train and test sets." << endl;

    // Create shuffled indices for splitting
    vector<int> indices(total_rows);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    cout << "Shuffled indices for splitting." << endl;

    // Copy labels
    // the data in cell at index "i" of train becomes the data in cell at index "indices[i]" of all_data, this allows shuffling,
    // e.g. if "i" = 0 indices[i] = 5, then train.labels[0] = all_data.labels[5]
    for(int i=0; i<total_rows; i++) {
        if(i < train_rows) train.labels[i] = all_data.labels[indices[i]];
        // else part goes to test set
        else test.labels[i - train_rows] = all_data.labels[indices[i]];
    }

    cout << "Copied labels to train and test sets." << endl;

    // Copy features column by column (faster for sequential writing)
    for (int c = 0; c < n_cols; c++) {
        // since we are working with column-major, calculate offsets
        int src_offset = c * total_rows;
        int train_offset = c * train_rows;
        int test_offset = c * test_rows;

        // Copy data for this column
        for (int i = 0; i < total_rows; i++) {
            int original_idx = indices[i];
            double val = all_data.features_flat[src_offset + original_idx];
            
            if (i < train_rows) {
                train.features_flat[train_offset + i] = val;
            } else {
                test.features_flat[test_offset + (i - train_rows)] = val;
            }
        }
    }

    cout << "Split completed: " << train.rows << " training, " << test.rows << " test." << endl;
}