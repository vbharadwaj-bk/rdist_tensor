. modules.sh
export OMP_NUM_THREADS=1

TENSOR_DIR=$SCRATCH/tensors
FACTOR_DIR=$SCRATCH/factor_files

TENSOR=$TENSOR_DIR/reddit-2015.tns_converted.hdf5
OUTPUT="data/scaling_runs/reddit.out"
for SEED in 189907 #1240003 734272 1215049 1222111
do
    srun -u -N 4 -n 256 python decompose_sparse.py -i $TENSOR -g "16,2,8" \
        -t "25" -iter 50 -o $OUTPUT -op "accumulator_stationary" \
        --reuse -s "131072" -rs $SEED -p "log_count" 
done