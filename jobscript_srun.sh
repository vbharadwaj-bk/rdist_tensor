#!/bin/bash
#SBATCH -N 8
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 02:00:00

. modules.sh
export OMP_NUM_THREADS=1

#gdb --args 
TENSOR=$SCRATCH/tensors/uber.tns_converted.hdf5
OUTPUT="data/uber.out"
srun -N 1 -u -n 1 python decompose_sparse.py -i $TENSOR  \
	-g "1,1,1,1" -t 25 -iter 20 \
	-o $OUTPUT -op "accumulator_stationary" -s 131000 

#srun -N 1 -u -n 1 python decompose_sparse.py -i $TENSOR  \
#	-g "1,1,1,1" -t 25 -iter 20 \
#	-o $OUTPUT -op "exact" 

#TENSOR=$SCRATCH/tensors/nell-1.tns_converted.hdf5
#OUTPUT="data/nell-1.out"
#FACTOR_FILE="data/nell_factors.hdf5"
#srun -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "8,8,8" -t 25 -iter 15 -o $OUTPUT -op "accumulator_stationary" -f $FACTOR_FILE -s 500000

#TENSOR=$SCRATCH/tensors/amazon-reviews.tns_converted.hdf5
#OUTPUT="data/amazon.out"
#FACTOR_FILE="data/amazon_factors.hdf5"
#srun -u -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "8,8,8" -t 25 -iter 15 -o $OUTPUT -op "accumulator_stationary" -f $FACTOR_FILE -s 131000

#TENSOR=$SCRATCH/tensors/reddit-2015.tns_converted.hdf5
#OUTPUT="data/reddit.out"
#FACTOR_FILE="data/reddit_factors.hdf5"
#srun -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "8,8,8" \
#	-t 25 -iter 15 -o $OUTPUT -op "accumulator_stationary" -f $FACTOR_FILE -s 131000 \
#	-p "log_count"

#TENSOR=$SCRATCH/tensors/enron.tns_converted.hdf5
#OUTPUT="data/enron.out"
#FACTOR_FILE="data/enron_factors.hdf5"
#srun -N 1 -n 128 python decompose_sparse.py -i $TENSOR -g "2,4,4,4" \
#	-t 25 -iter 15 -o $OUTPUT -op "accumulator_stationary" -f $FACTOR_FILE -s 131000 \
#	-p "log_count"

