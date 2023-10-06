#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q debug 
#SBATCH -t 00:30:00
#SBATCH -A m1982

. env.sh

export NODE_COUNT=4
export OMP_NUM_THREADS=16

export TRIAL_COUNT=5
export TENSOR=amazon

export CORES_PER_NODE=128
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))

for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
do
    srun -N $NODE_COUNT -n $((NODE_COUNT * RANKS_PER_NODE)) -c $((OMP_NUM_THREADS * 2)) python decompose.py -i $TENSOR \
                        --trank 25 \
                        -s 65536 \
                        -iter 40 \
                        -alg sts_cp \
                        -dist accumulator_stationary \
                        -o data/fit_progress_vs_time \
                        -r 5 \
                        -m nnodes_$NODE_COUNT 
done
