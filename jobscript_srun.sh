#!/bin/bash
#SBATCH -N 8
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 02:00:00

. modules.sh
export OMP_NUM_THREADS=1
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

# Exact computation
#export OMP_NUM_THREADS=1
#TENSOR=$SCRATCH/tensors/uber.tns_converted.hdf5
#OUTPUT="data/uber.out"
#srun -N 1 -n 1 python decompose_sparse.py -i $TENSOR -g "1,1,1,1" -t 25 -iter 30 -o $OUTPUT -op "accumulator_stationary" -s 131000 
#srun -N 1 -n 8 python decompose_sparse.py -i $TENSOR -g "2,1,2,2" -t 25 -iter 30 -o $OUTPUT
#srun -N 1 -n 27 python decompose_sparse.py -i $TENSOR -g "3,1,3,3" -t 25 -iter 20 -o $OUTPUT 
#srun -N 1 -n 64 python decompose_sparse.py -i $TENSOR -g "4,1,4,4" -t 25 -iter 50 -o $OUTPUT 

#TENSOR=$SCRATCH/tensors/nell-1.tns_converted.hdf5
#OUTPUT="data/nell-1.out"
#export OMP_NUM_THREADS=1
#srun -N 1 -n 128 python decompose_sparse.py -g "8,4,4" -i $TENSOR -t 25 -iter 20 -o $OUTPUT 
#srun -N 1 -n 1 python decompose_sparse.py -g "1,1,1" -i $TENSOR -t 2 -iter 15 -o $OUTPUT -s 2000000

# Sampled tests
#TENSOR=$SCRATCH/tensors/uber.tns_converted.hdf5
#OUTPUT="data/uber.out"
#srun -N 1 -n 1 python decompose_sparse.py -i $TENSOR -g "1,1,1,1" -t 25 -iter 20 -o $OUTPUT #-s 131000 

# Large-scale test
TENSOR=$SCRATCH/tensors/amazon-reviews.tns_converted.hdf5
OUTPUT="data/amazon.out"
FACTOR_FILE="data/amazon_factors.hdf5"
srun -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "8,8,8" -t 25 -iter 15 -o $OUTPUT -op "accumulator_stationary" -f $FACTOR_FILE -s 131000

# Test writing the factors to an output file
#TENSOR=$SCRATCH/tensors/uber.tns_converted.hdf5
#OUTPUT="data/uber.out"
#FACTOR_FILE="data/uber_factors.hdf5"
#srun -N 1 -n 2 python decompose_sparse.py -i $TENSOR -g "2,1,1,1" -t 25 -iter 25 -o $OUTPUT -op "accumulator_stationary" -f $FACTOR_FILE -s 131000
