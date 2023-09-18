. env.sh

export TRIAL_COUNT=5
export OMP_NUM_THREADS=16

export TENSOR=reddit
export ITERATIONS=80

for ALG in cp_arls_lev sts_cp
do
    for RANK in 25 50 75
    do
        for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
        do
            srun -N 4 -n 32 -c 32 python decompose.py -i $TENSOR \
                        --trank $RANK \
                        -s 65536 \
                        -iter $ITERATIONS \
                        -alg $ALG \
                        -dist accumulator_stationary \
                        -o data/accuracy_benchmarks \
                        -r 5
        done
    done
done