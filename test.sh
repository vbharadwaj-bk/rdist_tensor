. env.sh
export OMP_NUM_THREADS=16

srun -n 8 -c 32 python decompose.py -i uber \
                    --trank 25 \
                    -s 65536 \
                    -iter 10 \
                    -alg sts_cp \
                    -dist accumulator_stationary \
                    -r 1
                    #-o data/fit_progress_vs_time \