#!/bin/bash
RESULTS_FILE="openMP_parallel_tree_times.csv"
DATASET1="../iris_dataset/iris.csv"
DATASET2="../magic_dataset/magic04.csv"
#DATASET3 ="../covertype_dataset/covtype.csv"
# Remove the old CSV if it exists
rm -f $RESULTS_FILE

# Clean and compile
echo "Compiling"
make clean && make -B
if [ $? -ne 0 ]; then
    echo "Error during compilation"
    exit 1
fi

# Clean previous results
echo "Benchmark Random Forest" > risultati.txt
date >> risultati.txt
# varying number of threads
for t in 1 2 4 8 16 32; do
    echo "=================================="
    echo "TESTING with $t THREAD..."
    echo "=================================="

    # cluster command
    # vary the number of trees
    for n_trees in 1 50 100 250; do
        echo "Number of Trees: $n_trees" >> risultati.txt
        # vary dataset
        for i in 1 2; do
            if [ $i -eq 1 ]; then
                echo "Dataset: iris.csv" >> risultati.txt
            srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=$t --time=00:10:00 ./RandomForest $t ../iris_dataset/iris.csv $n_trees 100 >> risultati.txt

            elif [ $i -eq 2 ]; then
                echo "Dataset: magic04.csv" >> risultati.txt
            srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=$t --time=00:10:00 ./RandomForest $t ../magic_dataset/magic04.csv $n_trees 100 >> risultati.txt
            # else
            #     echo "Dataset: covtype.csv" >> risultati.txt
            #     if [ $t -eq 1 ]; then 
            #         echo "Skipping covtype with 1 thread due to long execution time." >> risultati.txt
            #         continue
            #     fi
            # srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=$t --time=00:10:00 ./RandomForest ../covertype_dataset/covtype.csv $n_trees 100 >> risultati.txt
            fi
        done
    done
    
done

echo "benchmark completed. Results saved in risultati.txt"