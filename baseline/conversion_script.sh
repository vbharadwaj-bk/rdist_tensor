TENSOR_NAME=amazon-reviews

./$SPLATT_LOC/splatt convert --type=bin \
        $RAW_TENSOR_LOC/$TENSOR_NAME.tns \
        $BIN_TENSOR_LOC/$TENSOR_NAME.bin