TENSOR_NAME=amazon-reviews
#TENSOR_NAME=reddit-2015

#$SPLATT_LOC/splatt cpd \
#        $BIN_TENSOR_LOC/$TENSOR_NAME.bin \
#        -r 25 -t 128 --nowrite 


#$SPLATT_LOC/splatt check \
#        $BIN_TENSOR_LOC/$TENSOR_NAME.bin

srun -N 1 -n 64 $SPLATT_LOC/splatt cpd \
        $BIN_TENSOR_LOC/$TENSOR_NAME.bin \
        -r 25 -t 2 --nowrite 