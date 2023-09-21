. env.sh
export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#srun -N 4 -n 64 -c 16 --cpu_bind=cores python decompose.py -i nell1 \
#                    --trank 25 \
#                    -s 131072 \
#                    -iter 40 \
#                    -alg sts_cp \
#                    -dist accumulator_stationary \
#                    -r 1
                    #-o data/fit_progress_vs_time \

python decompose.py -i uber \
                    --trank 25 \
                    -iter 40 \
                    -alg sts_cp \
                    -s 131072 \
                    -dist accumulator_stationary \
                    -r 1
