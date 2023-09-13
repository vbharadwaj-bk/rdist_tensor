export OMP_NUM_THREADS=32

python decompose.py -i uber \
                    --trank 25 \
                    -s 65536 \
                    -iter 5 \
                    -alg sts_cp \
                    -dist accumulator_stationary \
                    -o data/fit_progress_vs_time \
                    -r 2