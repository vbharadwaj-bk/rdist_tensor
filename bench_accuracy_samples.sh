#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular 
#SBATCH -t 04:00:00

. env.sh

export OMP_MAX_ACTIVE_LEVELS=1
export TRIAL_COUNT=5
export ITERATIONS=40
export RANK=25
export OMP_NUM_THREADS=16

export CORES_PER_NODE=128
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))

export NODE_COUNT=4

for ALG in cp_arls_lev sts_cp
do
    for SAMPLE_COUNT in 32768 65536 98304 131072 163840 196608
    do
        for TENSOR in amazon patents 
        do
            for DISTRIBUTION in accumulator_stationary 
            do
                for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
                do
                    srun -N $NODE_COUNT -n $((NODE_COUNT * RANKS_PER_NODE)) -c $((OMP_NUM_THREADS * 2)) python decompose.py -i $TENSOR \
                                --trank $RANK \
                                -s $SAMPLE_COUNT \
                                -iter $ITERATIONS \
                                -alg $ALG \
                                -dist $DISTRIBUTION \
                                -r $TRIAL_COUNT \
                                -o data/sample_count_accuracy
                done
            done
        done
    done
done