import numpy as np
import numpy.linalg as la
import os
import h5py

from exafac.cpp_ext.py_module import Grid, TensorGrid, SparseTensor
from mpi4py import MPI

import cppimport.import_hook

def round_to_nearest(n, m):
    return (n + m - 1) // m * m

def prime_factorization(n):
    prime_factors = []

    i = 2
    while i**2 <= n:
        if n % i:
            i += 1
        else:
            n /= i

            prime_factors.append(i)

    if n > 1:
        prime_factors.append(n)

    return sorted(prime_factors)

def get_best_mpi_dim(nprocs, global_dims):
    '''
    Adapted from
    https://github.com/ShadenSmith/splatt/blob/6cb86283c1fbfddcc67c2564e025691de4f784cf/src/mpi/mpi_io.c#L537
    '''
    N= len(global_dims)
    factors = prime_factorization(nprocs)
    total_size = 0

    mpi_dims = np.ones(N, dtype=np.uint64)
    diffs = np.zeros(N, dtype=np.uint64)

    total_size = np.sum(global_dims) 
    target = total_size // nprocs

    for i in reversed(range(len(factors))):
        furthest = 0

        for j in range(N):
            curr = global_dims[j] / mpi_dims[j]
            if curr > target:
                diffs[j] = curr - target
            else:
                diffs[j] = 0 

            if diffs[j] > diffs[furthest]:
                furthest = j

        mpi_dims[furthest] *= factors[i]

    return mpi_dims

class DistSparseTensorE:
    def __init__(self, filename, grid, preprocessing="None"):
        f = h5py.File(filename, 'r')
        world_comm = MPI.COMM_WORLD
        self.world_size = world_comm.Get_size()
        self.rank = world_comm.Get_rank()

        self.max_idxs = f['MAX_MODE_SET'][:]
        self.min_idxs = f['MIN_MODE_SET'][:]
        self.dim = len(self.max_idxs)
        self.tensor_dims = np.array(self.max_idxs - self.min_idxs + 1, dtype=np.int32)
        self.grid = grid

        if self.grid is None:
            optimal_grid_dims = get_best_mpi_dim(self.world_size, self.tensor_dims)

            if self.rank == 0:
                print(f"Optimal grid dimensions: {optimal_grid_dims}")

            self.grid = Grid(optimal_grid_dims)

        self.tensor_grid = TensorGrid(self.tensor_dims, self.grid)

        if self.grid.get_dimension() != self.dim:
            raise ValueError("Grid dimension must match tensor dimension")

        # The tensor must have at least one mode
        self.nnz = len(f['MODE_0']) 

        padded_nnz_ct = round_to_nearest(self.nnz, self.world_size) 

        local_nnz_ct = padded_nnz_ct // self.world_size
        start_nnz = min(local_nnz_ct * self.rank, self.nnz)
        end_nnz = min(local_nnz_ct * (self.rank + 1), self.nnz)

        self.tensor_idxs = np.zeros((end_nnz-start_nnz, self.dim), dtype=np.uint32)

        if self.rank == 0:
            print("Loading sparse tensor...")

        for i in range(self.dim): 
            self.tensor_idxs[:, i] = (f[f'MODE_{i}'][start_nnz:end_nnz] - self.min_idxs[i])

        # Assumption: all values are double format
        self.values = f['VALUES'][start_nnz:end_nnz]

        self.sparse_tensor = SparseTensor(
                self.tensor_grid, 
                self.tensor_idxs, 
                self.values,
                preprocessing)

        MPI.COMM_WORLD.Barrier()
        if self.rank == 0:
            print("Finished constructing sparse tensor...")


class RandomSparseTensor: 
    def __init__(self, grid, I, N, Q):
        world_comm = MPI.COMM_WORLD
        self.world_size = world_comm.Get_size()
        self.rank = world_comm.Get_rank()

        self.max_idxs = np.array([I-1] * N, dtype=np.int32)
        self.min_idxs = np.array([0] * N, dtype=np.int32) 
        self.dim = N 
        self.tensor_dims = np.array(self.max_idxs - self.min_idxs + 1, dtype=np.int32)
        self.grid = grid

        if self.grid is None:
            optimal_grid_dims = get_best_mpi_dim(self.world_size, self.tensor_dims)

            if self.rank == 0:
                print(f"Optimal grid dimensions: {optimal_grid_dims}")

            self.grid = Grid(optimal_grid_dims)

        self.tensor_grid = TensorGrid(self.tensor_dims, self.grid)

        if self.grid.get_dimension() != self.dim:
            raise ValueError("Grid dimension must match tensor dimension")

        if self.rank == 0:
            print("Generating random sparse tensor...")

        self.sparse_tensor = SparseTensor(
                self.tensor_grid,
                I,
                N,
                Q) 
  
        self.nnz = None 

        MPI.COMM_WORLD.Barrier()
        if self.rank == 0:
            print("Finished generating random sparse tensor...")

        exit(1)


