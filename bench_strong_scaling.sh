#!/bin/bash
#SBATCH -N 16
#SBATCH -C cpu
#SBATCH -q regular 
#SBATCH -t 00:04:00

. env.sh

export OMP_MAX_ACTIVE_LEVELS=1
export TRIAL_COUNT=1
export TENSOR=patents
export DISTRIBUTION=tensor_stationary
export ITERATIONS=20
export RANK=25
export OMP_NUM_THREADS=16

export CORES_PER_NODE=128
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))

for N in 16
do
    export NODE_COUNT=$N
    for ALG in cp_arls_lev sts_cp 
    do
        for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
        do
            srun -N $NODE_COUNT -n $((NODE_COUNT * RANKS_PER_NODE)) -c $((OMP_NUM_THREADS * 2)) python decompose.py -i $TENSOR \
                        --trank $RANK \
                        -s 65536 \
                        -iter $ITERATIONS \
                        -alg $ALG \
                        -dist $DISTRIBUTION \
                        -r $TRIAL_COUNT \
                        -m nnodes_$NODE_COUNT \
                        -o data/strong_scaling
        done
    done
done