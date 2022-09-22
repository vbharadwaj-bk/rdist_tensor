module load cmake

#conda activate tensor_env
conda deactivate
module load python
module load cray-hdf5
module load fast-mkl-amd
module load cray-openshmemx
module load cray-pmi cray-pmi-lib
module unload darshan
module load craype-hugepages1G

export CC=CC
export CXX=CC

export HDF5_USE_FILE_LOCKING=FALSE

