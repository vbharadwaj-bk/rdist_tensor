. env.sh
export OMP_NUM_THREADS=1

TENSOR_DIR=tensors
FACTOR_DIR=$SCRATCH/factor_files

# [ 189907, 1240003,  734272, 1215049, 1222111, 1225969, 1721519,
#        263888, 1578654,  918989]

TENSOR=$TENSOR_DIR/uber.tns_converted.hdf5
OUTPUT="data/baseline_runs/uber.out"
for SEED in 1240003 # 734272 1215049 1222111 189909 
do
    srun -u -N 1 -n 1 python decompose_sparse.py -i $TENSOR -g "1,1,1,1" \
        -t "75" -iter 40 -o $OUTPUT -op "accumulator_stationary" \
        --no-reuse -s "131072" -e 5 -rs $SEED 
done

#TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
#OUTPUT="data/baseline_runs/amazon_noreuse.out"
#for SEED in 189907 # 1240003 734272 1215049 1222111
#do
#    srun -u -N 1 -n 64 python decompose_sparse.py -i $TENSOR -g "4,4,4" \
#        -t "25" -iter 40 -o $OUTPUT -op "accumulator_stationary" \
#        --no-reuse -s "131072" -rs $SEED 
#done

#TENSOR=$TENSOR_DIR/reddit-2015.tns_converted.hdf5
#OUTPUT="data/baseline_runs/reddit.out"
#for SEED in 189907 1240003 734272 1215049 1222111
#do
#    srun -u -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "16,4,8" \
#        -t "25" -iter 500 -o $OUTPUT -op "accumulator_stationary" \
#        -s "131072" -rs $SEED -p "log_count" 
#done

#TENSOR=$TENSOR_DIR/amazon-reviews.tns_converted.hdf5
#OUTPUT="data/baseline_runs/amazon.out"
#srun -u -N 4 -n 512 python decompose_sparse.py -i $TENSOR -g "16,4,4" \
#    -t "25" -iter 30 -o $OUTPUT -op "exact"

#TENSOR=$TENSOR_DIR/reddit-2015.tns_converted.hdf5
#OUTPUT="data/baseline_runs/reddit.out"
#srun -u -N 4 -n 256 python decompose_sparse.py -i $TENSOR -g "8,8,4" \
#    -t "25" -iter 500 -o $OUTPUT -op "accumulator_stationary" \
#    -s "131000"
