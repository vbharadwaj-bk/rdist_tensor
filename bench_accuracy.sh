#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 02:00:00

. modules.sh
export OMP_NUM_THREADS=1

TENSOR_DIR=tensors

#TENSOR=$TENSOR_DIR/uber.tns_converted.hdf5
#OUTPUT="data/uber.out"
#srun -N 1 -u -n 64 python decompose_sparse.py -i $TENSOR  \
#	-g "4,1,4,4" -iter 500 -o $OUTPUT -op "accumulator_stationary" \
#    -t "25,50,100,200" \
#    -s "131000,150000,170000,200000,230000,2600000,300000"

TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
OUTPUT="data/amazon.out"
FACTOR_FILE="data/amazon_factors.hdf5"
srun -u -N 4 -n 256 -c 2 python decompose_sparse.py -i $TENSOR -g "8,8,4" \
    -iter 500 -o $OUTPUT -op "accumulator_stationary" \
    -t "25,50,100,200" \
    -s "131000,150000,170000,200000,230000,2600000,300000"