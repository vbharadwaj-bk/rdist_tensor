from sparse_tensor import *
import numpy as np
from numpy.random import Generator, Philox

from mpi4py import MPI
import argparse

from low_rank_tensor import *
from grid import *

def test_tensor_evaluation():
    ground_truth = DistSparseTensor("tensors/test.tns_converted.hdf5")
    grid = Grid([1, 1, 1])
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    ground_truth.redistribute_nonzeros(tensor_grid)

    ten_to_optimize = DistLowRank(tensor_grid, 10, None)
    ten_to_optimize.initialize_factors_deterministic(42)
    #tensor_values = ten_to_optimize.compute_tensor_values(ground_truth.tensor_idxs)

    world_comm = MPI.COMM_WORLD
    counts = np.zeros(world_comm.Get_size(), dtype=np.ulonglong)
    world_comm.Gather([cl(len(ground_truth.tensor_idxs[0])), MPI.UINT64_T], [counts, MPI.UINT64_T], root=0)

    disps = np.cumsum(counts)
    disps[1:] = disps[:-1]
    disps[0] = 0

    dim = ground_truth.dim
    idx_buffers = [np.zeros(ground_truth.nnz, dtype=np.ulonglong) for i in range(dim)]

    for i in range(dim):
        world_comm.Gatherv(ground_truth.tensor_idxs[i], [idx_buffers[i], counts, disps, MPI.UINT64_T], root=0)

    values = np.zeros(ground_truth.nnz, dtype=np.double) 
    #world_comm.Gatherv(tensor_values, [values, counts, disps, MPI.DOUBLE], root=0)

    rank = world_comm.Get_rank()

    if rank == 0:
        overall_dict = {}
        for i in range(ground_truth.nnz):
            coord = tuple([idx_buffers[j][i] for j in range(dim)])
            overall_dict[coord] = values[i]
        print("Finished!")

    print(ten_to_optimize.factors[0].data)

if __name__=='__main__':
	test_tensor_evaluation()
