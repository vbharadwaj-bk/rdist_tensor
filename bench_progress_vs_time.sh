. env.sh

# Test run on a small tensor
#python decompose.py -i uber \
#                    --trank 25 \
#                    -s 65536 \
#                    -iter 10 \
#                    -alg sts_cp \
#                    -dist accumulator_stationary \
#                    -o data/fit_progress_vs_time \
#                    -r 1


export OMP_NUM_THREADS=16
export TRIAL_COUNT=5

for (( trial=1; trial<=$TRIAL_COUNT; trial++ )) 
do
    srun -N 4 -n 32 -c 32 python decompose.py -i reddit \
                        --trank 100 \
                        -s 98304 \
                        -iter 80 \
                        -alg cp_arls_lev \
                        -dist accumulator_stationary \
                        -o data/fit_progress_vs_time \
                        -r 5
done
