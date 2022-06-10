. modules.sh
export OMP_NUM_THREADS=1

# Exact computation
TENSOR=$SCRATCH/tensors/uber.tns_converted.hdf5
OUTPUT="data/uber.out"
srun -N 1 -n 1 python decompose_sparse.py -i $TENSOR -g "1,1,1,1" -t 25 -iter 20 -o $OUTPUT 
srun -N 1 -n 8 python decompose_sparse.py -i $TENSOR -g "2,1,2,2" -t 25 -iter 20 -o $OUTPUT 
srun -N 1 -n 27 python decompose_sparse.py -i $TENSOR -g "3,1,3,3" -t 25 -iter 20 -o $OUTPUT 
srun -N 1 -n 64 python decompose_sparse.py -i $TENSOR -g "4,1,4,4" -t 25 -iter 20 -o $OUTPUT 

TENSOR=$SCRATCH/tensors/nell-1.tns_converted.hdf5
OUTPUT="data/nell-1.out"
#srun -N 2 -n 256 python decompose_sparse.py -g "8,8,4" -i $TENSOR -t 25 -iter 20 -o $OUTPUT 

#srun -N 1 -n 1 python decompose_sparse.py -g "1,1,1" -i $TENSOR -t 25 -iter 1 -o $OUTPUT -s 131000


# Sampled tests
TENSOR="tensors/uber.tns_converted.hdf5"
OUTPUT="data/uber.out"
#srun -N 1 -n 1 python decompose_sparse.py -i $TENSOR -g "1,1,1,1" -t 25 -iter 20 -o $OUTPUT -s 131000 

