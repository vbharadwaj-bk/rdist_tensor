#!/bin/bash
#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q debug 
#SBATCH -t 00:30:00

. env.sh

export OMP_MAX_ACTIVE_LEVELS=1
export TRIAL_COUNT=3
export TENSOR=amazon
export DISTRIBUTION=accumulator_stationary
export ITERATIONS=20
export OMP_NUM_THREADS=16
export ALG=sts_cp

export CORES_PER_NODE=128
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))

for TENSOR in amazon 
do
    for N in 1
    do
        export NODE_COUNT=$N
        for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
        do
            srun -N $NODE_COUNT -n $((NODE_COUNT * RANKS_PER_NODE)) -c $((OMP_NUM_THREADS * 2)) python decompose.py -i $TENSOR \
                        --trank $((16 * NODE_COUNT)) \
                        -s 65536 \
                        -iter $ITERATIONS \
                        -alg $ALG \
                        -dist $DISTRIBUTION \
                        -r $TRIAL_COUNT \
                        -m nnodes_$NODE_COUNT \
                        -o data/weak_scaling
        done
    done
done