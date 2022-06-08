#. modules.sh
srun -N 1 -n 64 python decompose_sparse.py -t 25 -iter 30 -o data/test.out
