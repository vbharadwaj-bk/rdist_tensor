module load cmake
module load cudatoolkit
#conda activate gpu_linalg

conda deactivate
module load python
#conda activate tensor_env
export HDF5_USE_FILE_LOCKING=FALSE
module load cray-hdf5

export CC=CC
export CXX=CC
