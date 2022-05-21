import numpy as np
from numpy.random import Generator, Philox

import numpy.linalg as la
from grid import Grid
from local_kernels import *
from sketching import *
from common import *

import mpi4py
from mpi4py import MPI

# Matrix is partitioned into block rows across processors
# This class is designed so that each slice of the processor
# grid holds a chunk of matrices. The slice dimension is
# the axis along which to ``align" the distribution of the factor 
class DistMat1D:
    def __init__(self, cols, tensor_grid, slice_dim):
        self.grid = tensor_grid.grid
        self.rows = tensor_grid.tensor_dims[slice_dim] 
        self.cols = cl(cols)

        # TODO: This computation is pretty redundant compared to the one
        # already in grid.py...
        self.padded_rows = round_to_nearest_np_arr(self.rows, self.grid.world_size)
        self.local_rows_padded = self.padded_rows // cl(self.grid.world_size)
        self.local_window_size = self.local_rows_padded * self.cols

        self.slice_dim = slice_dim

        self.row_position = cl(self.grid.slices[slice_dim].Get_rank() + \
            self.grid.coords[slice_dim] * self.grid.slices[slice_dim].Get_size())

        # Compute the true count of the rows that this processor owns 
        if(self.row_position * self.local_rows_padded > self.rows):
            self.rowct = cl(0)
        else:
            self.rowct = min(self.rows - self.row_position * self.local_rows_padded, self.local_rows_padded)

        self.data = np.zeros((self.local_rows_padded, self.cols), dtype=np.double)   
        
        #self.row_idxs = np.array(list(range(self.rowct)), dtype=np.int64)

        # TODO: Should store the padding offset here, add a view
        # into the matrix that represents the true data
        #print(f"Rank: {grid.rank}\t{self.grid.slices[0].Get_rank()}")
        #print(f"Row position: {self.row_position}")

    def initialize_deterministic(self, offset):
        value_start = self.row_position * self.local_window_size 
        self.data = np.array(range(value_start, value_start + self.local_window_size)).reshape(self.local_rows_padded, self.cols)
        #self.data = np.cos((self.data + offset) * 5)
 
    def initialize_random(self, random_seed=42):
        #gen = Philox(random_seed)
        #gen = gen.advance(self.row_position * self.local_window_size)
        #rg = Generator(gen)
        self.data = np.random.rand(*self.data.shape) - 0.5

    def compute_gram_matrix(self):
        if self.rowct == 0:
            ncol = np.shape(self.data)[1]
            self.gram = np.zeros((ncol, ncol))
        else:
            data_trunc = self.data[:self.rowct]
            self.gram = (data_trunc.T @ data_trunc)

        self.grid.comm.Allreduce(MPI.IN_PLACE, self.gram)

    def allgather_factor(self, with_leverage=False):
        slice_dim = self.slice_dim
        slice_size = cl(self.grid.slices[slice_dim].Get_size())
        buffer_rowct = self.local_rows_padded * slice_size
        buffer_data = np.zeros((buffer_rowct, self.cols))
        buffer_leverage = np.zeros(buffer_rowct)

        self.grid.slices[slice_dim].Allgather([self.data, MPI.DOUBLE], 
                [buffer_data, MPI.DOUBLE])


        # Handle the overhang when mode length is not divisible by
        # the processor count

        if buffer_rowct * cl(self.grid.coords[slice_dim]) > self.rows:
            truncated_rowct = 0
        else:
            truncated_rowct = min(buffer_rowct, self.rows - buffer_rowct * cl(self.grid.coords[slice_dim]))

        buffer_data = buffer_data[:truncated_rowct] 

        self.gathered_factor = buffer_data

        if with_leverage:
            self.grid.slices[slice_dim].Allgather([self.leverage_scores, MPI.DOUBLE], 
                    [buffer_leverage, MPI.DOUBLE])
            buffer_leverage = buffer_leverage[:truncated_rowct]
            self.gathered_leverage = buffer_leverage 

    #=================================================================
    # METHODS RELATED TO SKETCHING GO HERE!!!
    #=================================================================

    def compute_leverage_scores(self):
        gram_inv = la.pinv(self.gram)
 
        self.leverage_scores = np.sum((self.data @ gram_inv) * self.data, axis=1)

        # Leverage weight is the sum of the leverage scores held by  
        normalization_factor = np.array(np.sum(self.leverage_scores))
        self.grid.comm.Allreduce(MPI.IN_PLACE, normalization_factor)

        if normalization_factor != 0:
            self.leverage_scores /= normalization_factor 

        self.leverage_weight = np.array(np.sum(self.leverage_scores))
