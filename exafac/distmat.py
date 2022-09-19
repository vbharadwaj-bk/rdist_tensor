from re import U
import numpy as np
from numpy.random import Generator, Philox

import numpy.linalg as la
from exafac.common import *
from exafac.sampling import *

import mpi4py
from mpi4py import MPI
import h5py

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

        #self.row_position = cl(self.grid.slices[slice_dim].Get_rank() + \
        #    self.grid.coords[slice_dim] * self.grid.slices[slice_dim].Get_size())

        self.row_position  = self.grid.row_positions[slice_dim][self.grid.rank]

        # Compute the true count of the rows that this processor owns 
        if(self.row_position * self.local_rows_padded > self.rows):
            self.rowct = cl(0)
        else:
            self.rowct = min(self.rows - self.row_position * self.local_rows_padded, self.local_rows_padded)

        self.data = np.zeros((self.local_rows_padded, self.cols), dtype=np.double)

        self.gathered_factor = None
        self.gathered_leverage = None
        self.col_norms = np.zeros(cols, dtype=np.double)

        # Mapping of processes to row order and vice-versa 
        self.proc_to_row_order = np.empty(self.grid.world_size, dtype=np.uint64)
        self.row_order_to_proc = np.empty(self.grid.world_size, dtype=np.uint64)

        self.grid.comm.Allgather(
			[self.row_position, MPI.UINT64_T],
			[self.proc_to_row_order, MPI.UINT64_T]	
		)

		# Invert the permutation here
        for i in range(len(self.proc_to_row_order)):
            self.row_order_to_proc[self.proc_to_row_order[i]] = i 

        self.row_order_to_proc = self.row_order_to_proc.astype(int)

    def permute_to_grid(self, new_grid):
        row_positions = new_grid.row_positions[self.slice_dim]
        target = np.where(row_positions == self.row_position)[0][0]

        #print(f"{self.grid.rank}, {self.permuted_rank}")

        #if self.permuted_rank != self.grid.rank: 

        MPI.COMM_WORLD.Sendrecv_replace([self.data, MPI.DOUBLE], 
            target) 

        MPI.COMM_WORLD.Barrier()

    def permute_from_grid(self, new_grid):
        row_pos = new_grid.row_positions[self.slice_dim][self.grid.rank] 
        row_positions = self.grid.row_positions[self.slice_dim]
        target = np.where(row_positions == row_pos)[0][0]

        MPI.COMM_WORLD.Sendrecv_replace([self.data, MPI.DOUBLE], 
            target) 

        MPI.COMM_WORLD.Barrier()

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

    def allgather_factor(self, world, truncate=False): 
        slice_size = cl(world.Get_size())

        buffer_rowct = self.local_rows_padded * slice_size
        self.gathered_factor = np.zeros((buffer_rowct, self.cols), dtype=np.double)

        world.Allgather([self.data, MPI.DOUBLE], 
                [self.gathered_factor, MPI.DOUBLE])

    def write_factor_to_file(self, hdf5_file, factor_name):
        # TODO: Undo the permutation used for load-balancing here! For now, we will 
        # write the padded factor to memory
        dset = hdf5_file.create_dataset(factor_name, (self.padded_rows, self.cols), dtype='f8')

        start = self.local_rows_padded * self.row_position
        end = self.local_rows_padded * (self.row_position + cl(1))

        dset[start:end, :] = self.data

    #=================================================================
    # METHODS RELATED TO SKETCHING 
    #=================================================================

    def compute_leverage_scores(self):
        gram_inv = la.pinv(self.gram)
 
        self.leverage_scores = np.sum((self.data @ gram_inv) * self.data, axis=1)
        self.leverage_scores = np.maximum(self.leverage_scores, 0.0)

        # Leverage weight is the sum of the leverage scores held by  
        normalization_factor = np.array(np.sum(self.leverage_scores))
        self.grid.comm.Allreduce(MPI.IN_PLACE, normalization_factor)

        if normalization_factor != 0:
            self.leverage_scores /= normalization_factor 

        #self.leverage_weight = np.array(np.sum(self.leverage_scores))

    def allgather_leverage_scores(self):
        slice_dim = self.slice_dim
        slice_size = cl(self.grid.slices[slice_dim].Get_size())
        buffer_rowct = self.local_rows_padded * slice_size

        if self.gathered_leverage is None:
            self.gathered_leverage = np.zeros(buffer_rowct, dtype=np.double)

        self.grid.slices[slice_dim].Allgather([self.leverage_scores, MPI.DOUBLE], 
                [self.gathered_leverage, MPI.DOUBLE])

    def sample_and_gather_singlemode(self, local_probs, world, sample_count):
        grid = self.grid
        base_idx = self.row_position * self.local_rows_padded 
        local_samples, local_counts, local_probs = get_samples_distributed_compressed(
                self.grid.comm,
                local_probs,
                sample_count)


        sampled_rows = self.data[local_samples]

        # TODO: Need to eliminate the explicit typecasting! 
        all_samples = allgatherv(world, base_idx.astype(np.uint32) + local_samples, MPI.UINT32_T)
        all_counts = allgatherv(world, local_counts, MPI.UINT64_T)
        all_probs = allgatherv(world, local_probs, MPI.DOUBLE)
        all_rows = allgatherv(world, sampled_rows, MPI.DOUBLE)

        return all_samples, all_counts, all_probs, all_rows 

    def sample_and_gather_rows(self, local_probs, world, num_modes, mode_to_leave, sample_count):
        assert(num_modes >= 1)

        # TODO: Should probably reuse these buffers!
        self.gathered_samples = []

        for i in range(num_modes):
            if (mode_to_leave is not None) and i == mode_to_leave:
                self.gathered_samples.append(None)
            else:
                self.gathered_samples.append(self.sample_and_gather_singlemode(local_probs, world, sample_count))