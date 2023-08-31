conda create --name rdist_tensor python=3.9

conda activate rdist_tensor
pip install mpi4py

module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

module load cray-hdf5-parallel
conda install -c defaults --override-channels numpy cython

#HDF5_MPI=ON CC=cc pip install -v --force-reinstall --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py
pip install h5py
