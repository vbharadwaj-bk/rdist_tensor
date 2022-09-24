. modules.sh
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

TENSOR_DIR=tensors
FACTOR_DIR=$SCRATCH/factor_files

#TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
#OUTPUT="data/baseline_runs/amazon.out"
#srun -u -N 4 -n 256 python decompose_sparse.py -i $TENSOR -g "16,4,4" \
#    -t "25" -iter 30 -o $OUTPUT -op "exact"

#TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
#OUTPUT="data/baseline_runs/amazon.out"
#srun -u -N 4 -n 256 python decompose_sparse.py -i $TENSOR -g "16,4,4" \
#    -t "25" -iter 500 -o $OUTPUT -op "accumulator_stationary" \
#    -s "131000"

#TENSOR=$TENSOR_DIR/reddit-2015.tns_converted.hdf5
#OUTPUT="data/baseline_runs/reddit.out"
#srun -u -N 4 -n 256 python decompose_sparse.py -i $TENSOR -g "16,1,16" \
#    -t "25" -iter 30 -o $OUTPUT -op "exact"

#TENSOR=$TENSOR_DIR/reddit-2015.tns_converted.hdf5
#OUTPUT="data/baseline_runs/reddit.out"
#srun -u -N 4 -n 256 python decompose_sparse.py -i $TENSOR -g "8,8,4" \
#    -t "25" -iter 500 -o $OUTPUT -op "accumulator_stationary" \
#    -s "131000"
