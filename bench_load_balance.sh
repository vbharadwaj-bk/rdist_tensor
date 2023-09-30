#!/bin/bash
#SBATCH -N 16
#SBATCH -C cpu
#SBATCH -q regular 
#SBATCH -t 01:30:00

. env.sh

export OMP_MAX_ACTIVE_LEVELS=1
export TRIAL_COUNT=5
export ITERATIONS=20
export RANK=25
export OMP_NUM_THREADS=16

export CORES_PER_NODE=128
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))

export ALG=sts_cp
for NODE_COUNT in 4 2
    do
    for TENSOR in amazon patents reddit 
    do
        for DISTRIBUTION in tensor_stationary accumulator_stationary 
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
                            -o data/load_balance
            done
        done
    done
done