TENSOR_NAME=patents-reordered

TENSOR_LOC=/pscratch/sd/v/vbharadw/tensors
SPLATT_LOC=/global/cfs/projectdirs/m1982/vbharadw/splatt/build/Linux-x86_64/bin

$SPLATT_LOC/splatt convert --type=bin \
        $TENSOR_LOC/$TENSOR_NAME.tns \
        $TENSOR_LOC/$TENSOR_NAME.bin