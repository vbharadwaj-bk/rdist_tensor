#TENSOR_NAME=amazon-reviews
#TENSOR_NAME=reddit-2015
TENSOR_NAME=uber

$SPLATT_LOC/splatt convert --type=bin \
        $RAW_TENSOR_LOC/$TENSOR_NAME.tns \
        $BIN_TENSOR_LOC/$TENSOR_NAME.bin