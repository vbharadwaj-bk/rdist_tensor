. env.sh
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -N 1 -n 8 -c 32 --cpu_bind=cores python decompose.py -i uber \
                    --trank 100 \
                    -s 65536 \
                    -iter 20 \
                    -alg sts_cp \
                    -dist accumulator_stationary \
                    -r 1
                    #-o data/fit_progress_vs_time \