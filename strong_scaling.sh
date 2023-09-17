. env.sh

export OMP_NUM_THREADS=8
srun -n 16 python decompose.py -i uber \
                    --trank 25 \
                    -s 65536 \
                    -iter 40 \
                    -alg sts_cp \
                    -dist tensor_stationary \
                    -r 1
#                    -o data/fit_progress_vs_time \
