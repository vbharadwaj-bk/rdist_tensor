import numpy as np
from numpy.random import Generator, Philox

import numpy.linalg as la
from grid import Grid
from local_kernels import *
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

        self.gathered_factor = None
        self.gathered_leverage = None
        self.col_norms = np.zeros(cols, dtype=np.double)

    def initialize_deterministic(self, offset):
        value_start = self.row_position * self.local_window_size 
        self.data = np.array(range(value_start, value_start + self.local_window_size), dtype=np.double).reshape(self.local_rows_padded, self.cols)
        self.data = np.cos((self.data + offset) * 5)
 
    def initialize_random(self, random_seed=42):
        self.data = np.random.rand(*self.data.shape, dtype=np.double) - 0.5

    def normalize_cols(self):
        normsq_cols = la.norm(self.data, axis=0) ** 2
        self.grid.comm.Allreduce(MPI.IN_PLACE, normsq_cols)
        self.col_norms = np.sqrt(normsq_cols)
        #self.col_norms = np.ones(self.cols, dtype=np.double)
        self.data = self.data @ np.diag(self.col_norms ** -1)


    def compute_gram_matrix(self):
        if self.rowct == 0:
            ncol = np.shape(self.data)[1]
            self.gram = np.zeros((ncol, ncol))
        else:
            data_trunc = self.data[:self.rowct]
            self.gram = (data_trunc.T @ data_trunc)

        self.grid.comm.Allreduce(MPI.IN_PLACE, self.gram)

    def allgather_factor(self, truncate=False):
        slice_dim = self.slice_dim
        slice_size = cl(self.grid.slices[slice_dim].Get_size())
        buffer_rowct = self.local_rows_padded * slice_size

        if self.gathered_factor is None:
            self.gathered_factor = np.zeros((buffer_rowct, self.cols), dtype=np.double)

        self.grid.slices[slice_dim].Allgather([self.data, MPI.DOUBLE], 
                [self.gathered_factor, MPI.DOUBLE])

        # Handle the overhang when mode length is not divisible by
        # the processor count

        #if truncate:
        #    if buffer_rowct * cl(self.grid.coords[slice_dim]) > self.rows:
        #        truncated_rowct = 0
        #    else:
        #        truncated_rowct = min(buffer_rowct, self.rows - buffer_rowct * cl(self.grid.coords[slice_dim]))

        #    buffer_data = buffer_data[:truncated_rowct] 
        #self.gathered_factor = buffer_data

    #=================================================================
    # METHODS RELATED TO SKETCHING 
    #=================================================================

    def compute_leverage_scores(self):
        gram_inv = la.pinv(self.gram)
 
        self.leverage_scores = np.sum((self.data @ gram_inv) * self.data, axis=1)
        #self.leverage_scores = np.ones(self.data.shape[0], dtype=np.double) 

        self.leverage_scores = np.maximum(self.leverage_scores, 0.0)

        # Leverage weight is the sum of the leverage scores held by  
        normalization_factor = np.array(np.sum(self.leverage_scores))
        self.grid.comm.Allreduce(MPI.IN_PLACE, normalization_factor)

        if normalization_factor != 0:
            self.leverage_scores /= normalization_factor 

        self.leverage_weight = np.array(np.sum(self.leverage_scores))

    def allgather_leverage_scores(self):
        slice_dim = self.slice_dim
        slice_size = cl(self.grid.slices[slice_dim].Get_size())
        buffer_rowct = self.local_rows_padded * slice_size

        if self.gathered_leverage is None:
            self.gathered_leverage = np.zeros(buffer_rowct, dtype=np.double)

        self.grid.slices[slice_dim].Allgather([self.leverage_scores, MPI.DOUBLE], 
                [self.gathered_leverage, MPI.DOUBLE])