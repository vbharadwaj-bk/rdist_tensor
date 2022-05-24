from sparse_tensor import *
import numpy as np
from numpy.random import Generator, Philox

from mpi4py import MPI
import argparse

from low_rank_tensor import *
from grid import *

def test_allgather():
    x = np.array(list(range(25)), dtype=np.double).reshape((5, 5))
    y = np.zeros((5, 5), dtype=np.double) 
    MPI.COMM_WORLD.Allgather([x, MPI.DOUBLE], [y, MPI.DOUBLE])
    print(f"Initial: {x}")
    print(f"Initial: {y}")

def test_mttkrp():
    ground_truth = DistSparseTensor("tensors/uber.tns_converted.hdf5")
    grid_dims = [2, 2, 2, 1]
    grid = Grid(grid_dims)
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    ground_truth.redistribute_nonzeros(tensor_grid)
    #ground_truth.values = np.ones_like(ground_truth.values, dtype=np.double)

    rank = 5
    mode_to_leave = 1

    ten_to_optimize = DistLowRank(tensor_grid, rank, None)
    ten_to_optimize.initialize_factors_deterministic(42)

    gathered_matrices, _ = ten_to_optimize.allgather_factors([True] * len(grid_dims))

    world_comm = MPI.COMM_WORLD
    mttkrp_unreduced = np.zeros((tensor_grid.intervals[mode_to_leave], rank)) 

    ground_truth.mttkrp(gathered_matrices, mode_to_leave, mttkrp_unreduced)  
    mttkrp_reduced = np.zeros_like(ten_to_optimize.factors[mode_to_leave].data, dtype=np.double) 

    grid.slices[mode_to_leave].Reduce_scatter([mttkrp_unreduced, MPI.DOUBLE], 
            [mttkrp_reduced, MPI.DOUBLE])

    print(get_sum_distributed(mttkrp_reduced, world_comm)) 
    print("Finished!")

def test_tensor_evaluation():
    ground_truth = DistSparseTensor("tensors/uber.tns_converted.hdf5")
    grid = Grid([2, 2, 2, 1])
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    ground_truth.redistribute_nonzeros(tensor_grid)

    ten_to_optimize = DistLowRank(tensor_grid, 10, None)
    ten_to_optimize.initialize_factors_deterministic(42)
    tensor_values = ten_to_optimize.compute_tensor_values(ground_truth.tensor_idxs)

    world_comm = MPI.COMM_WORLD
    counts = np.zeros(world_comm.Get_size(), dtype=np.ulonglong)
    world_comm.Gather([cl(len(ground_truth.tensor_idxs[0])), MPI.UINT64_T], [counts, MPI.UINT64_T], root=0)

    disps = np.cumsum(counts)
    disps[1:] = disps[:-1]
    disps[0] = 0

    dim = ground_truth.dim
    idx_buffers = [np.zeros(ground_truth.nnz, dtype=np.ulonglong) for i in range(dim)]

    for i in range(dim):
        send_buf = ground_truth.tensor_idxs[i] + tensor_grid.start_coords[i][grid.coords[i]]
        #send_buf = ground_truth.tensor_idxs[i] 
        world_comm.Gatherv(send_buf, [idx_buffers[i], counts, disps, MPI.UINT64_T], root=0)

    values = np.zeros(ground_truth.nnz, dtype=np.double) 
    world_comm.Gatherv(tensor_values, [values, counts, disps, MPI.DOUBLE], root=0)

    rank = world_comm.Get_rank()

    if rank == 0:
        overall_dict = {}
        for i in range(ground_truth.nnz):
            coord = tuple([idx_buffers[j][i] for j in range(dim)])
            overall_dict[coord] = values[i]
            rows = [np.array(list(range(10 * int(i), 10 * (int(i) + 1))), dtype=np.double) for i in coord]
            accum = np.ones(10)
            for j in range(dim):
                accum *= rows[j]
            cval = np.sum(accum)

            if np.abs(cval - values[i]) > 1e-6: 
                print(f"Error: {cval}, {values[i]}")

        print("Finished!")

if __name__=='__main__':
    #test_allgather()
	#test_tensor_evaluation()
    test_mttkrp()