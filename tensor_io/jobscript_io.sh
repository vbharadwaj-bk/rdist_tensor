#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu 
#SBATCH -q regular
#SBATCH -J convert_tensors_hdf5
#SBATCH --mail-user=vivek_bharadwaj@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -A m1982
#SBATCH -t 08:00:00

#OpenMP settings:
#export OMP_NUM_THREADS=1

. modules.sh
python preprocess_frostt_datasets.py