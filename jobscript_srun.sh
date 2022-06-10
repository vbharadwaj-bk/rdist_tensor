. modules.sh

TENSOR=tensors/uber.tns_converted.hdf5
srun -N 1 -n 1 python decompose_sparse.py \
	-i $TENSOR
	-g "1,1,1,1" 
	-t 25 -iter 20 -o data/test.out
