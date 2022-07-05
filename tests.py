from sparse_tensor import *
import numpy as np
from numpy.random import Generator, Philox
from sampling import *

from mpi4py import MPI
import argparse

from low_rank_tensor import *
from grid import *

import cppimport.import_hook
import cpp_ext.bloom_filter as bf

def test_sampling():
    world_comm = MPI.COMM_WORLD
    ground_truth = DistSparseTensor("tensors/test.tns_converted.hdf5")
    grid_dims = [1, 1, 1]
    grid = Grid(grid_dims)
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    mode_to_leave = 2
    sample_idxs = [np.array([0, 0]), np.array([0, 1])]
    samples = ground_truth.sample_nonzeros(sample_idxs, mode_to_leave)
    samples.print_contents()

def test_sampling_distributed():
    ground_truth = DistSparseTensor("tensors/test.tns_converted.hdf5")
    grid_dims = [2, 2, 2]
    grid = Grid(grid_dims)
    #tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    #ground_truth.redistribute_nonzeros(tensor_grid)

    #mode_to_sample = 0
    
    #ten_to_optimize = DistLowRank(tensor_grid, rank, None)
    #ten_to_optimize.initialize_factors_deterministic(42)

    #ten_to_optimize.factors[0].compute_leverage_scores()

    row_probs = [0.0, 0.0]

    if grid.rank == 0 or grid.rank == 1:
        row_probs = [0.25, 0.25]

    get_samples_distributed(grid.comm, row_probs, 100)


def test_mttkrp():
    world_comm = MPI.COMM_WORLD
    ground_truth = DistSparseTensor("tensors/uber.tns_converted.hdf5")
    grid_dims = [4, 4, 4, 1]
    grid = Grid(grid_dims)
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)

    ground_truth.redistribute_nonzeros(tensor_grid)
    #ground_truth.values = np.ones_like(ground_truth.values, dtype=np.double)

    rank = 1
    mode_to_leave = 1

    ten_to_optimize = DistLowRank(tensor_grid, rank, None)
    ten_to_optimize.initialize_factors_deterministic(42)

    gathered_matrices, _ = ten_to_optimize.allgather_factors([True] * len(grid_dims))

    mttkrp_unreduced = np.zeros((tensor_grid.intervals[mode_to_leave], rank)) 
    mttkrp_final_buffer = np.zeros_like(mttkrp_unreduced)

    ground_truth.mttkrp(gathered_matrices, mode_to_leave, mttkrp_unreduced)
    mttkrp_reduced = np.zeros_like(ten_to_optimize.factors[mode_to_leave].data, dtype=np.double) 

    grid.slices[mode_to_leave].Reduce_scatter([mttkrp_unreduced, MPI.DOUBLE], 
            [mttkrp_reduced, MPI.DOUBLE])

    if grid.rank == 0:
        print(f'First value: {mttkrp_reduced[0]}')

    print("Finished!")

def test_tensor_evaluation():
    ground_truth = DistSparseTensor("tensors/test.tns_converted.hdf5")
    grid = Grid([1, 1, 1])
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

            rows = [np.cos((np.array(list(range(10 * int(i), 10 * (int(i) + 1))), dtype=np.double) + 42) * 5) for i in coord]
            accum = np.ones(10)
            for j in range(dim):
                accum *= rows[j]
            cval = np.sum(accum)

            if np.abs(cval - values[i]) > 1e-6: 
                print(f"Error: {cval}, {values[i]}")

        print("Finished!")

def test_bloom_filter():
    ground_truth = DistSparseTensor("tensors/test.tns_converted.hdf5")
    grid = Grid([1, 1, 1])
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    ground_truth.redistribute_nonzeros(tensor_grid)

    idx_filter = bf.IndexFilter(ground_truth.tensor_idxs, 0.01)

    test_probes = []
    for j in range(3):
        test_probes.append(np.array([2], dtype=np.ulonglong))

    print(idx_filter.check_idxs(test_probes))

def test_hdf5():
    pass


if __name__=='__main__':
    #test_allgather()
	#test_tensor_evaluation()
    #test_mttkrp()
    #test_bloom_filter()
    #test_sampling()
    #test_sampling_distributed()
    test_hdf5()

