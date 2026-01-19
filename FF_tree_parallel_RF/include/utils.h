#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <map>
#include <cmath>
#include <set>
#include <limits>
#include <sys/stat.h>
#include <random>
#include "Tree.h"
#include "Data.h"
#include "omp.h"
#include "RandomForest.h"
#include <ff/ff.hpp>
using namespace std;


// Compute Metrics
void print_metrics(const vector<int>& predictions, const vector<int>& true_labels) {
    if (predictions.size() != true_labels.size()) return;

    int n_samples = predictions.size();
    int correct_total = 0;
    
    // Maps to count true positives, actual positives, and predicted positives per class
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

    // Accuracy
    double accuracy = (double)correct_total / n_samples;

    // Macro Recall and F1
    double sum_recall = 0.0;
    double sum_f1 = 0.0;

    for (int c : classes) {
        int tp = tp_counts[c];
        int actual_positives = true_counts[c]; // TP + FN
        int predicted_positives = pred_counts[c]; // TP + FP

        // Recall = TP / (TP + FN)
        double recall = (actual_positives == 0) ? 0.0 : (double)tp / actual_positives;
        
        // Precision = TP / (TP + FP)
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

#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <sys/stat.h>

using namespace std;

// Function to check if a file is empty or does not exist
bool is_file_empty(const string& filename) {
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        return true; // File does not exist
    }
    return st.st_size == 0; // Returns true if file size is 0
}

void save_times_to_csv(const string& dataset, int num_trees, double avg_depth, double training, double prediction, const string& filename) {
    // Open the file in APPEND mode (std::ios::app)
    bool write_header = is_file_empty(filename);
    ofstream outfile(filename, std::ios::app);
    
    if (!outfile.is_open()) return;

    // Write the header only if the file was just created
    if (write_header) {
        outfile << "Dataset,Num_Threads,Num_Trees,Avg_Depth,Training_Time,Prediction_Time,Total_Time" << endl;
    }

    // Calculate total execution time
    double total_time = training + prediction;

    // Append the new measurements to the end of the file
    outfile << dataset << ","
            << omp_get_max_threads() << "," 
            << num_trees << ","
            << avg_depth << "," 
            << training << "," 
            << prediction << "," 
            << total_time << "\n";

    outfile.close();
}
