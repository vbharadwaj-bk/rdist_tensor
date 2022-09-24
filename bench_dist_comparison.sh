. modules.sh
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

TENSOR_DIR=tensors
FACTOR_DIR=$SCRATCH/factor_files

#TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
#OUTPUT="data/dist_comparison/amazon.out"
#srun -u -N 4 -n 256 python decompose_sparse.py -i $TENSOR -g "16,4,4" \
#    -t "25" -iter 30 -o $OUTPUT -op "accumulator_stationary" \
#    -s "131000"

TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
OUTPUT="data/dist_comparison/amazon.out"
srun -u -N 4 -n 256 python decompose_sparse.py -i $TENSOR -g "16,4,4" \
    -t "25" -iter 30 -o $OUTPUT -op "tensor_stationary" \
    -s "131000"
