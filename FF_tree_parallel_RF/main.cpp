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
#include <random>  // Per std::default_random_engine
#include <omp.h>
#include "Tree.h"
#include "Data.h"
#include "RandomForest.h"
#include "utils.h"
#include <ff/ff.hpp>


using namespace std;

int main(int argc, char* argv[]) {
    // Check input
    if (argc < 5) {
        cout << "Uso: " << argv[0] << " <num_threads> <file_csv> <num_alberi> <depth>" << endl;
        return 1;
    }
    
    Dataset trainData, testData;
    int num_trees;
    int num_threads;
    size_t depth;
    string filename = "not_defined";
    // for comparison on the same split of scikit-learn
    if (argc == 6){
        num_threads = stoi(argv[1]);
        string train_name = argv[2];
        string test_name = argv[3];
        trainData = load_csv_dataset(train_name);
        testData = load_csv_dataset(test_name);
        num_trees = stoi(argv[4]);
        depth = stoi(argv[5]);
    }

    // else normal use
    else{
        num_threads = stoi(argv[1]);
        filename = argv[2];
        Dataset allData = load_csv_dataset(filename);
        int seed = 45;
        double train_ratio = 0.8; // 80% train, 20% test
        split_dataset(allData, trainData, testData, seed, train_ratio);
        num_trees = stoi(argv[3]);
        depth = stoi(argv[4]);
    }

    num_threads = std::max(1, num_threads - 2);

    //print thread infos    
    // how many threads are available
    cout << "------------------------------------------------" << endl;
    cout << "Available threads (Hardware): " 
              << omp_get_num_procs() << endl;
              
    // how many threads will be used for parallel regions
    cout << "Allocated threads: " 
              << num_threads << endl;

    RandomForest rf(num_trees);

    // training
    cout << "------------------------------------------------" << endl;
    auto training_start = chrono::high_resolution_clock::now();
    
    rf.train_fastflow(trainData, depth, num_threads); 
    
    auto training_end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = training_end - training_start;
    cout << "Tempo di Training: " << elapsed.count() << " secondi." << endl;
    double avg_depth = rf.get_average_depth();
    cout << "average depth: "<< avg_depth << endl;

    // prediction
    cout << "------------------------------------------------" << endl;
    auto start_pred = chrono::high_resolution_clock::now();

    vector<int> predictions = rf.predict_fastflow(testData, num_threads);
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