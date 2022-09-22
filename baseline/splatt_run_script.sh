TENSOR_NAME=amazon-reviews
#TENSOR_NAME=reddit-2015

$SPLATT_LOC/splatt cpd \
        $BIN_TENSOR_LOC/$TENSOR_NAME.bin \
        -r 25 -t 128 