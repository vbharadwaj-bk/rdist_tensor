#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 03:00:00
export OMP_NUM_THREADS=16

. modules.sh

#./grb_test /global/cfs/projectdirs/m1982/caida-telescope/data

#ls /global/cfs/projectdirs/m1982/caida-telescope/data/2022-03-23/net44.1648018800 | wc -l

#srun -N 1 -n 1 -u ./grb_test /global/cfs/projectdirs/m1982/caida-telescope/data/2022-03-23/net44.1648018800

#srun -N 1 -n 1 -u ./grb_test /global/cfs/projectdirs/mp156/vivek/caida/net44.1648018800

echo "Starting..."
srun -N 1 -n 1 -u ./grb_test \
    /global/cfs/projectdirs/m1982/caida-telescope/data/2022-03-23/net44.1648018800 \
    /global/cfs/projectdirs/m1982/caida-telescope/data/2022-03-23/net44.1648022400 \
    /global/cfs/projectdirs/m1982/caida-telescope/data/2022-03-23/net44.1648026000