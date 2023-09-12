#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 04:00:00

TENSOR_LOC=/pscratch/sd/v/vbharadw/tensors
SPLATT_LOC=/global/cfs/projectdirs/m1982/vbharadw/splatt/build/Linux-x86_64/bin

TRIAL_COUNT=5
TOL=1e-5
MAX_ITER=80

export OMP_NUM_THREADS=2

TENSOR=reddit-2015-spl-binary.bin
OUT_FILE=outputs/reddit_trace.txt
echo "----" + $(date) + "----" >> $OUT_FILE
for RANK in 100 
do
    for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
    do
        srun -N 4 -n 256 -c 4 -u $SPLATT_LOC/splatt cpd $TENSOR_LOC/$TENSOR -r $RANK --nowrite -i $MAX_ITER --tol $TOL >> $OUT_FILE
    done
done