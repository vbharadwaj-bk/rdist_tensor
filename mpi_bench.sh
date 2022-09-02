. modules.sh
export OMP_NUM_THREADS=1

for NODE_REQUEST in 1 2 4
do

    for TASKS_PER_NODE in 1 2 4 8 16 32 64 128
    do
        srun -N $NODE_REQUEST --ntasks-per-node $TASKS_PER_NODE python mpi_bench.py -o data/mpi_benchmarks.out
    done

done
