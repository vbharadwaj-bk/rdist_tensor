. env.sh

export TRIAL_COUNT=1
export OMP_NUM_THREADS=16

export TENSOR=uber
export ITERATIONS=20
export RANK=100

for ALG in cp_arls_lev 
do
    for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
    do
        srun -N 1 -n 8 -c 32 python decompose.py -i $TENSOR \
                    --trank $RANK \
                    -s 65536 \
                    -iter $ITERATIONS \
                    -alg $ALG \
                    -dist accumulator_stationary \
                    -o data/strong_scaling \
                    -r $TRIAL_COUNT
    done
done