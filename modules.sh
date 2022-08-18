module load cmake
module load cudatoolkit
#conda activate gpu_linalg

#conda activate tensor_env
conda deactivate
module load python
conda activate lazy-h5py
module load cray-hdf5

export CC=CC
export CXX=CC

export HDF5_USE_FILE_LOCKING=FALSE