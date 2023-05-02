import numpy as np
import numpy.linalg as la
import os
import h5py

from exafac.grid import Grid, TensorGrid
from mpi4py import MPI
from exafac.common import * 

import cppimport.import_hook

class DistSparseTensorE:
    def __init__(self, filename):
        f = h5py.File(filename, 'r')
        world_comm = MPI.COMM_WORLD
        self.world_size = world_comm.Get_size()
        self.rank = world_comm.Get_rank()

        self.max_idxs = f['MAX_MODE_SET'][:]
        self.min_idxs = f['MIN_MODE_SET'][:]
        self.dim = len(self.max_idxs)

        # The tensor must have at least one mode
        self.nnz = len(f['MODE_0']) 

        padded_nnz_ct = round_to_nearest(self.nnz, self.world_size) 

        local_nnz_ct = padded_nnz_ct // self.world_size
        start_nnz = min(local_nnz_ct * self.rank, self.nnz)
        end_nnz = min(local_nnz_ct * (self.rank + 1), self.nnz)

        self.tensor_idxs = []

        if self.rank == 0:
            print("Loading sparse tensor...")

        for i in range(self.dim): 
            self.tensor_idxs.append(f[f'MODE_{i}'][start_nnz:end_nnz] - self.min_idxs[i])

        self.values = f['VALUES'][start_nnz:end_nnz]

        MPI.COMM_WORLD.Barrier()
        if self.rank == 0:
            print("Finished loading sparse tensor...")


