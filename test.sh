#!/bin/bash
#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q debug 
#SBATCH -t 00:08:00

. env.sh
export OMP_NUM_THREADS=16
export OMP_PLACES=threads

#python decompose.py -i uber \
#                    --trank 25 \
#                    -s 65000 \
#                    -iter 40 \
#                    -alg sts_cp \
#                    -dist accumulator_stationary \
#                    -r 1 #\
#                    -p exact
                    #-o data/fit_progress_vs_time \

#srun -N 4 -n 32 -c 32 python decompose.py -i caida \
#                    --trank 25 \
#                    -iter 10 \
#                    -alg cp_arls_lev \
#                    -s 4000000 \
#                    -dist accumulator_stationary \
#                    -p exact


srun -N 4 -n 32 -c 32 python decompose.py -i amazon \
                    --trank 25 \
                    -iter 40 \
                    -alg sts_cp \
                    -s 65536 \
                    -dist accumulator_stationary 


