#!/bin/bash
#SBATCH -N 16
#SBATCH -C cpu
#SBATCH -q regular 
#SBATCH -t 01:00:00
#SBATCH -A m1982 
#SBATCH --reservation=ipdps_perlmutter

. env.sh

export OMP_MAX_ACTIVE_LEVELS=1
export TRIAL_COUNT=4
export DISTRIBUTION=tensor_stationary
export ITERATIONS=20
export OMP_NUM_THREADS=4

export OMP_PLACES=threads
export CORES_PER_NODE=128
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))
export N=16

for TENSOR in patents 
do 
    for RANK in 25 50 75 
    do
        export NODE_COUNT=$N
        for ALG in cp_arls_lev 
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
                            -o data/baseline_runtime_comparison
            done
        done
    done
done