#!/bin/bash
#SBATCH -N 32
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 00:12:00

. modules.sh
export OMP_NUM_THREADS=1

TENSOR_DIR=$SCRATCH/tensors
FACTOR_DIR=$SCRATCH/factor_files
#TENSOR=$TENSOR_DIR/uber.tns_converted.hdf5
#OUTPUT="data/uber.out"
#srun -N 1 -u -n 2 python decompose_sparse.py -i $TENSOR  \
#	-g "1,2,1,1" -iter 50 \
#    -o $OUTPUT -op "accumulator_stationary" \
#    -t "25" \
#    -s "131000" \
#	-pre_optim 1
#gdb --args 

#srun -N 1 -u -n 1 python decompose_sparse.py -i $TENSOR  \
#	-g "1,1,1,1" -t 25 -iter 20 \
#	-o $OUTPUT -op "exact" 

#TENSOR=$TENSOR_DIR/nell-1.tns_converted.hdf5
#OUTPUT="data/nell-1.out"
#FACTOR_FILE="data/nell_factors.hdf5"
#srun -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "8,8,8" -t 25 -iter 15 -o $OUTPUT -op "accumulator_stationary" -f $FACTOR_FILE -s 500000

#TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
#OUTPUT="data/amazon.out"
#FACTOR_FILE="data/amazon_factors.hdf5"
#srun -u -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "8,8,8" \
#    -t "25" -iter 500 -o $OUTPUT -op "accumulator_stationary" \
#    -s "250072" -rs 22343

#TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
#OUTPUT="data/amazon.out"
#FACTOR_FILE="data/amazon_factors.hdf5"
#srun -u -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "8,8,8" \
#    -t "25" -iter 500 -o $OUTPUT -op "exact"

#TENSOR=$TENSOR_DIR/reddit-2015.tns_converted.hdf5
#OUTPUT="data/scaling_runs/reddit_tensor_stationary.out"
#FACTOR_FILE="data/reddit_factors.hdf5"
#srun -N 32 -n 4096 -u python decompose_sparse.py -i $TENSOR -g "64,2,32" \
#	-t 50 -iter 50 -o $OUTPUT -op "tensor_stationary" -p "log_count" -rs 55 -s 150000 

#TENSOR=$TENSOR_DIR/tensors/enron.tns_converted.hdf5
#OUTPUT="data/enron.out"
#FACTOR_FILE="data/enron_factors.hdf5"
#srun -N 1 -n 128 python decompose_sparse.py -i $TENSOR -g "2,4,4,4" \
#	-t 25 -iter 15 -o $OUTPUT -op "accumulator_stationary" -f $FACTOR_FILE -s 131000 \
#	-p "log_count"

TENSOR=$TENSOR_DIR/caida_small.hdf5
OUTPUT="data/scaling_runs/caida_small_as.out"
FACTOR_FILE=$FACTOR_DIR/caida_factors.hdf5
srun -N 4 -u -n 512 python decompose_sparse.py -i $TENSOR  \
	-g "16,16,2" -iter 50 \
    -o $OUTPUT -op "exact" -t "15" -p "log_count" #\
	#-s "80000" -pre_optim 1
