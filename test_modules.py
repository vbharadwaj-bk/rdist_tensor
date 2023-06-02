from mpi4py import MPI
import numpy as np
import argparse
import gc
import cppimport
import cppimport.import_hook

def test_prefixes(axes):
    lst = [1]
    for i in range(len(axes) - 1):
        lst.append(lst[-1] * axes[-1-i])
        print(axes[-1-i-1])

    lst.reverse()
    return np.array(lst, dtype=np.int32)


def test_grid():
    from exafac.cpp_ext.py_module import Grid, TensorGrid, DistMat1D, LowRankTensor, ExactALS, TensorStationaryOpt0
    from exafac.sparse_tensor_e import DistSparseTensorE
    from exafac.grid import Grid as GridPy
    from exafac.grid import TensorGrid as TensorGridPy

    dims = [2, 2, 1, 1]
    proc_dims = np.array(dims, dtype=np.int32)

    grid = Grid(proc_dims)

    # Get the MPI rank with mpi4py
    rank = MPI.COMM_WORLD.Get_rank()

    #sparse_tensor = DistSparseTensorE('/pscratch/sd/v/vbharadw/tensors/amazon-reviews.tns_converted.hdf5', grid) 
    #sparse_tensor = DistSparseTensorE('/pscratch/sd/v/vbharadw/tensors/uber.tns_converted.hdf5', grid) 
    sparse_tensor = DistSparseTensorE('../tensors/uber.tns_converted.hdf5', grid) 
    low_rank_tensor = LowRankTensor(25, sparse_tensor.tensor_grid)
    #low_rank_tensor.initialize_factors_deterministic()
    low_rank_tensor.initialize_factors_gaussian_random()
    tensor_stationary_opt0 = TensorStationaryOpt0(sparse_tensor.sparse_tensor, low_rank_tensor) 

    fit = tensor_stationary_opt0.compute_exact_fit()
    if rank == 0:
        print(f"Initial Fit: {fit}")
    tensor_stationary_opt0.execute_ALS_rounds(1, 5)

    fit = tensor_stationary_opt0.compute_exact_fit()
    if rank == 0:
        print(f"Final Fit: {fit}")


if __name__=='__main__':
    num_procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    parser = argparse.ArgumentParser()
    args = None
    try:
        if rank == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)

    if rank == 0:
        print("Loading Python modules...")
        import exafac.cpp_ext.py_module

    MPI.COMM_WORLD.Barrier()

    # Start running tests 
    test_grid()
