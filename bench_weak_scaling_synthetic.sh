#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -A m1982

export NODE_COUNT=1
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))
export OMP_NUM_THREADS=16
export ALG=sts_cp
export ITERATIONS=20
export DISTRIBUTION=accumulator_stationary
export TRIAL_COUNT=1
export TENSOR="random"

#srun -N $NODE_COUNT -n $((NODE_COUNT * RANKS_PER_NODE)) -c $((OMP_NUM_THREADS * 2)) python decompose.py \

# For local testing only!
export OMP_NUM_THREADS=1
srun -np 8 python decompose.py \
            -i $TENSOR \
            --trank 1 \
            -s 65536 \
            -iter $ITERATIONS \
            -alg $ALG \
            -dist $DISTRIBUTION \
            -r $TRIAL_COUNT \
            -m nnodes_$NODE_COUNT \
            #-o data/weak_scaling_synthetic