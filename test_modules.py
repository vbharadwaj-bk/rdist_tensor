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
    from exafac.cpp_ext.py_module import Grid, TensorGrid, DistMat1D, LowRankTensor, ExactALS
    from exafac.sparse_tensor_e import DistSparseTensorE
    from exafac.grid import Grid as GridPy
    from exafac.grid import TensorGrid as TensorGridPy

    dims = [3, 2, 1, 1]
    proc_dims = np.array(dims, dtype=np.int32)

    #print(test_prefixes(dims))
    #exit(1)

    grid = Grid(proc_dims)

    test = GridPy(dims)

    sparse_tensor = DistSparseTensorE('../tensors/uber.tns_converted.hdf5', grid) 
    low_rank_tensor = LowRankTensor(5, sparse_tensor.tensor_grid)
    exact_als = ExactALS(sparse_tensor, low_rank_tensor) 

    #low_rank_tensor.test_gram_matrix_computation()

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
