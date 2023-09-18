. env.sh

export TRIAL_COUNT=1
export TENSOR=uber
export ITERATIONS=20
export RANK=100
export OMP_NUM_THREADS=16

export CORES_PER_NODE=128
export RANKS_PER_NODE=$((CORES_PER_NODE / OMP_NUM_THREADS))

for NODE_COUNT in 1
do
    for ALG in sts_cp 
    do
        for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
        do
            srun -N $NODE_COUNT -n $((NODE_COUNT * RANKS_PER_NODE)) -c $((OMP_NUM_THREADS * 2)) python decompose.py -i $TENSOR \
                        --trank $RANK \
                        -s 65536 \
                        -iter $ITERATIONS \
                        -alg $ALG \
                        -dist accumulator_stationary \
                        -o data/strong_scaling \
                        -r $TRIAL_COUNT \
                        -m nnodes_$NODE_COUNT
        done
    done
done