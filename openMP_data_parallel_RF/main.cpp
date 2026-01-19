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
#include <random>  // Per std::default_random_engine
#include <algorithm> // Per std::shuffle
#include <omp.h>
#include "Tree.h"
#include "Data.h"
#include "RandomForest.h"
#include "utils.h"

using namespace std;

int main(int argc, char* argv[]) {
    // Check input
    if (argc < 4) {
        cout << "Uso: " << argv[0] << " <file_csv> <num_alberi> <depth>" << endl;
        return 1;
    }
    
    Dataset trainData, testData;
    int num_trees;
    size_t depth;
    string filename = "not_defined";
    // for comparison on the same split of scikit-learn
    if (argc == 5){
        string train_name = argv[1];
        string test_name = argv[2];
        trainData = load_csv_dataset(train_name);
        testData = load_csv_dataset(test_name);
        num_trees = stoi(argv[3]);
        depth = stoi(argv[4]);
    }

    // else normal use
    else{
        filename = argv[1];
        Dataset allData = load_csv_dataset(filename);
        int seed = 45;
        double train_ratio = 0.8; // 80% train, 20% test
        split_dataset(allData, trainData, testData, seed, train_ratio);
        num_trees = stoi(argv[2]);
        depth = stoi(argv[3]);
    }

    //print thread infos    
    // how many threads are available
    cout << "------------------------------------------------" << endl;
    cout << "Available threads (Hardware): " 
              << omp_get_num_procs() << endl;
              
    // how many threads will be used for parallel regions
    cout << "Allocated threads: " 
              << omp_get_max_threads() << endl;

    RandomForest rf(num_trees);

    // training
    cout << "------------------------------------------------" << endl;
    auto training_start = chrono::high_resolution_clock::now();
    
    rf.train(trainData, depth); 
    
    auto training_end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = training_end - training_start;
    cout << "Tempo di Training: " << elapsed.count() << " secondi." << endl;
    double avg_depth = rf.get_average_depth();
    cout << "average depth: "<< avg_depth << endl;

    // prediction
    cout << "------------------------------------------------" << endl;
    auto start_pred = chrono::high_resolution_clock::now();

    vector<int> predictions = rf.predict(testData);
    auto end_pred = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_pred = end_pred - start_pred;
    cout << "Tempo di Predizione: " << elapsed_pred.count() << " secondi." << endl;

    // metrics
    print_metrics(predictions, testData.labels);

    // save predictions to csv
    save_predictions_to_csv(predictions,"predictions.csv");

    //save training and prediction time to csv file
    save_times_to_csv(filename, num_trees, avg_depth, elapsed.count(), elapsed_pred.count(), "times.csv");
    return 0;
}