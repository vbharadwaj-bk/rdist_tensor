#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular 
#SBATCH -t 03:00:00
#SBATCH -A m1982

TENSOR_LOC=/pscratch/sd/v/vbharadw/tensors
SPLATT_LOC=/global/cfs/projectdirs/m1982/vbharadw/splatt/build/Linux-x86_64/bin

TRIAL_COUNT=4
TOL=1e-8
MAX_ITER=40

TENSOR=patents-reordered.bin
OUT_FILE=outputs/patents_accuracy.txt

export N=4
export OMP_NUM_THREADS=8

export CORES_PER_NODE=128
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))

echo "----" + $(date) + "----" >> $OUT_FILE
for RANK in 50 
do
    for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
    do
        srun    -N $N \
                -n $((N * RANKS_PER_NODE)) \
                -c $((OMP_NUM_THREADS * 2)) \
                -u $SPLATT_LOC/splatt cpd \
                $TENSOR_LOC/$TENSOR -r $RANK \
                --nowrite -i $MAX_ITER \
                --tol $TOL >> $OUT_FILE
    done
done
