#!/bin/bash
RESULTS_FILE="openMP_parallel_tree_times.csv"
DATASET1 ="../iris_dataset/iris.csv"
DATASET2 ="../magic_dataset/magic04.csv"
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
# varying number of nodes, weak scaling == increasing problem size 
for n in 1 2 4 6 8; do
    echo "=================================="
    echo "TESTING with $n NODES..."
    echo "=================================="
    
    # OpenMP environment variable
    export OMP_NUM_THREADS=32
    
    # cluster command

    echo "Testing with OMP_NUM_THREADS=32" >> risultati.txt
    srun --nodes=$n --ntasks-per-node=1 --cpus-per-task=32 --time=00:20:00 --mpi=pmix ./RandomForest_hybrid ../magic_dataset/magic04.csv $((64 * n)) 100 >> risultati.txt

    srun --nodes=$n --ntasks-per-node=1 --cpus-per-task=32 --time=00:20:00 --mpi=pmix ./RandomForest_hybrid ../covertype_dataset/covtype.csv $((64 * n)) 100 >> risultati.txt
    
done

echo "benchmark completed. Results saved in risultati.txt"  