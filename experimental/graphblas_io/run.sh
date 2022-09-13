export OMP_NUM_THREADS=1

#./grb_test /global/cfs/projectdirs/m1982/caida-telescope/data

#ls /global/cfs/projectdirs/m1982/caida-telescope/data/2022-03-23/net44.1648018800 | wc -l

srun -N 1 -n 1 -u ./grb_test /global/cfs/projectdirs/m1982/caida-telescope/data/2022-03-23/net44.1648018800

#srun -N 1 -n 1 -u ./grb_test /global/cfs/projectdirs/mp156/vivek/caida/net44.1648018800
