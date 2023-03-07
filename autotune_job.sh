. modules.sh
#export OMP_NUM_THREADS=1

TENSOR_DIR=tensors

TENSOR=$TENSOR_DIR/uber.tns_converted.hdf5
OUTPUT="data/uber.out"
python exafac_controller.py -i $TENSOR  \
	-g "4,1,4,4" -iter 500 -o $OUTPUT -op "accumulator_stationary" \
    -t "25,50,100,200" \
    -s "131000,150000,170000,200000,230000,2600000,300000"