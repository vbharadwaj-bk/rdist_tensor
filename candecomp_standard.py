import mpi4py
import numpy as np
from numpy.random import Generator, Philox

import numpy.linalg as la
from grid import Grid
from local_kernels import *

from mpi4py import MPI

def round_to_nearest(n, m):
    return (n + m - 1) // m * m

def compute_residual(ground_truth, current):
  return np.linalg.norm(ground_truth - current)

def get_norm_distributed(buf, world):
    val = la.norm(buf) ** 2
    result = np.zeros(1)
    world.Allreduce([val, MPI.DOUBLE], [result, MPI.DOUBLE]) 
    return np.sqrt(result)

# Matrix is partitioned into block rows across processors
# This class is designed so that each slice of the processor
# grid holds a chunk of matrices. The slice dimension is
# the third parameter 
class DistMat1D:
    def __init__(self, rows, cols, grid, slice_dim):
        self.rows = rows 
        self.padded_rows = round_to_nearest(rows, grid.world_size)
        self.cols = cols
        self.local_rows_padded = self.padded_rows // grid.world_size
        self.local_window_size = self.local_rows_padded * self.cols

        self.grid = grid

        self.row_position = grid.slices[slice_dim].Get_rank() + \
            grid.coords[slice_dim] * grid.slices[slice_dim].Get_size()

        # Compute the true count of the rows that this processor owns 
        self.rowct = min(self.rows - self.row_position * self.local_rows_padded, self.local_rows_padded)
        self.rowct = max(self.rowct, 0)

        self.data = np.zeros((self.local_rows_padded, self.cols))   
        self.row_idxs = np.array(list(range(self.rowct)), dtype=np.int64)

        # TODO: Should store the padding offset here, add a view
        # into the matrix that represents the true data
        #print(f"Rank: {grid.rank}\t{self.grid.slices[0].Get_rank()}")
        #print(f"Row position: {self.row_position}")

    def initialize_deterministic(self, offset):
        value_start = self.row_position * self.local_window_size 
        self.data = np.array(range(value_start, value_start + self.local_window_size)).reshape(self.local_rows_padded, self.cols)
        self.data = np.cos((self.data + offset) * 5)
 
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

    def compute_leverage_scores(self):
        gram_inv = la.pinv(self.gram)

        # TODO: This function can be made more efficient using
        # BLAS calls!

        self.leverage_scores = np.zeros(self.rowct)
        for i in range(self.rowct):
            row = self.data[[i]]
            self.leverage_scores[i] = row @ gram_inv @ row.T 

        # Leverage weight is the sum of the leverage scores held by  
        normalization_factor = np.array(np.sum(self.leverage_scores))
        self.grid.comm.Allreduce(MPI.IN_PLACE, normalization_factor)

        if normalization_factor != 0:
            self.leverage_scores /= normalization_factor 

        self.leverage_weight = np.array(np.sum(self.leverage_scores))

    def leverage_sample(self, num_samples):
        # This function is probably over-engineered in the spirit of wanting 
        # a good theoretical guarantee

        world_size = self.grid.comm.Get_size()

        # Step 1: Allgather the leverage weights across all processors
        all_leverage_weights = np.zeros(world_size)
        self.grid.comm.Allgather([self.leverage_weight, MPI.DOUBLE], 
                [all_leverage_weights, MPI.DOUBLE])

        all_leverage_weights = np.maximum(np.minimum(all_leverage_weights, 1.0), 0.0)

        # Step 2: A single processor computes the number of samples
        # that everybody should draw using a multinomial distribution
        # and then scatters. This could be made more efficient,
        # but want to get a correct result first 

        self.global_sample_cts = np.zeros(world_size, dtype=np.int64)

        if self.grid.rank == 0:
            rng = np.random.default_rng() 
            self.global_sample_cts = rng.multinomial(num_samples, all_leverage_weights) 
        self.grid.comm.Bcast([self.global_sample_cts, MPI.LONG], root=0)

        # Step 4: Everybody draws the specified number of samples 
        # from their local dataset
        self.sample_idxs = np.random.choice(self.row_idxs, p=self.leverage_scores / self.leverage_weight, size=self.global_sample_cts[self.grid.rank])
        self.leverage_samples = self.data[self.sample_idxs] 

    def allgatherv_leverage(self, dim, coord_in_dim):
        # TODO: Need to offset the indices here by the base index 
        world = self.grid.slices[dim]
        slice_size = world.Get_size()
        base_position = slice_size * coord_in_dim 
        self.sample_idxs += (self.row_position - base_position) * self.local_rows_padded

        # We can make this process more efficient, but this is okay for now.
        # Gather up the count of samples within each slice 
        my_sample_ct = np.array([len(self.sample_idxs)], dtype=np.int64)
        sample_cts = np.zeros(slice_size, dtype=np.int64) 
        world.Allgather([my_sample_ct, MPI.LONG], [sample_cts, MPI.LONG])

        # Aggregates rows and sample indices for each leverage score on each 
        total_samples = np.sum(sample_cts)
        disps = np.insert(np.cumsum(sample_cts)[:-1], [0], [0]) 
        self.sketch_indices = np.zeros(total_samples, dtype=np.int64)

        world.Allgatherv([self.sample_idxs, MPI.LONG],
            [self.sketch_indices, sample_cts, disps, MPI.LONG] 
        )

        self.sketch_samples = np.zeros((total_samples, self.cols), dtype=np.double)

        world.Allgatherv([self.leverage_samples, MPI.DOUBLE],
            [self.sketch_samples, sample_cts * self.cols, disps * self.cols, MPI.DOUBLE] 
        )


# Initializes a distributed tensor of a known low rank
class DistLowRank:
    def __init__(self, grid, mode_sizes, rank, singular_values): 
        self.rank = rank
        self.grid = grid
        self.mode_sizes = mode_sizes
        self.dim = len(mode_sizes)
        self.factors = [DistMat1D(mode_sizes[i], rank, grid, i)
                    for i in range(self.dim)]

        self.singular_values = np.array(singular_values)
        
    # Replicate data on each slice and all-gather all factors accoording to the
    # provided true / false array 
    def allgather_factors(self, which_factors):
        gathered_matrices = []
        for i in range(self.dim):
            if which_factors[i]:
                slice_size = self.grid.slices[i].Get_size()
                buffer_rowct = self.factors[i].local_rows_padded * slice_size
                buffer = np.zeros((buffer_rowct, self.rank))
                self.grid.slices[i].Allgather([self.factors[i].data, MPI.DOUBLE], 
                        [buffer, MPI.DOUBLE])

                # Handle the overhang when mode length is not divisible by
                # the processor count 
                truncated_rowct = min(buffer_rowct, self.mode_sizes[i] - buffer_rowct * self.grid.coords[i])
                truncated_rowct = max(truncated_rowct, 0) 
                buffer = buffer[:truncated_rowct]

                self.factors[i].local_rows_padded * slice_size 
                gathered_matrices.append(buffer)

        return gathered_matrices

    def leverage_sketch(self, factors_to_sketch, matricized_tensor):
        lhs = np.ones((self.sketch_indices, self.rank))
        mat_tensor_idxs = np.zeros(len(self.sketch_indices), dtype=np.int64)
        for i in range(len(factors_to_sketch)):
            lhs *= factors_to_sketch[i][sample_idxs[i]]
            mat_tensor_idxs *= factors_to_sketch[i].shape[0]
            mat_tensor_idxs += self.sketch_indices[i]

        rhs = matricized_tensor[mat_tensor_idxs]

        return lhs, rhs

    def materialize_tensor(self):
        gathered_matrices = self.allgather_factors([True] * self.dim)
        #self.local_materialized = np.einsum('r,ir,jr,kr->ijk', self.singular_values, *gathered_matrices)
        self.local_materialized = tensor_from_factors_sval(self.singular_values, gathered_matrices) 

    # Materialize the tensor starting only from the given singular
    # value 
    def partial_materialize(self, singular_value_start):
        assert(singular_value_start < self.rank)

        gathered_matrices = self.allgather_factors([True] * self.dim)
        gathered_matrices = [mat[:, singular_value_start:] for mat in gathered_matrices]
        
        return tensor_from_factors_sval(self.singular_values[singular_value_start:], gathered_matrices) 

    def initialize_factors_deterministic(self, offset):
        for factor in self.factors:
            factor.initialize_deterministic(offset)

    def initialize_factors_random(self, random_seed=42):
        for factor in self.factors:
            factor.initialize_random(random_seed=random_seed)

    # Computes a distributed MTTKRP of all but one of this 
    # class's factors with a given dense tensor. Also performs 
    # gram matrix computation. 
    def optimize_factor(self, local_ten, mode_to_leave, sketching=False):
        factors_to_gather = [True] * self.dim
        factors_to_gather[mode_to_leave] = False

        selected_indices = np.array(list(range(self.dim)))[factors_to_gather]
        selected_factors = [self.factors[idx] for idx in selected_indices] 

        # Compute gram matrices of all factors but the one we are currently
        # optimizing for, perform leverage-score based sketching if necessary 
        for idx in selected_indices:
            factor = self.factors[idx]
            factor.compute_gram_matrix()

            if sketching:
                factor.compute_leverage_scores()
                factor.leverage_sample(10)
                factor.allgatherv_leverage(idx, self.grid.coords[idx]) 

        gram_prod = selected_factors[0].gram

        for i in range(1, len(selected_factors)):
            gram_prod = np.multiply(gram_prod, selected_factors[i].gram)

        # Compute inverse of the gram matrix 
        krp_gram_inv = la.pinv(gram_prod)
        gathered_matrices = self.allgather_factors(factors_to_gather)

        # Compute a local MTTKRP
        matricized_tensor = matricize_tensor(local_ten, mode_to_leave)
        mttkrp_unreduced = matricized_tensor.T @ krp(gathered_matrices)

        # Padding before the reduce-scatter
        padded_rowct = self.factors[mode_to_leave].local_rows_padded * self.grid.slices[mode_to_leave].Get_size()

        reduce_scatter_buffer = np.zeros((padded_rowct, self.rank))
        reduce_scatter_buffer[:len(mttkrp_unreduced)] = mttkrp_unreduced

        mttkrp_reduced = np.zeros_like(self.factors[mode_to_leave].data)

        self.grid.slices[mode_to_leave].Reduce_scatter_block([reduce_scatter_buffer, MPI.DOUBLE], 
                [mttkrp_reduced, MPI.DOUBLE])  

        res = (krp_gram_inv @ mttkrp_reduced.T).T.copy()
        self.factors[mode_to_leave].data = res


    def als_fit(self, local_ground_truth, num_iterations):
        # Should initialize the singular values more intelligently, but this is fine
        # for now:

        self.singular_values = np.ones(self.rank)
        self.materialize_tensor()
        loss = get_norm_distributed(local_ground_truth - self.local_materialized, self.grid.comm)

        for iter in range(num_iterations):
            self.materialize_tensor()
            loss = get_norm_distributed(local_ground_truth - self.local_materialized, self.grid.comm)

            if self.grid.rank == 0:
                print("Residual after iteration {}: {}".format(iter, loss)) 

            for mode_to_optimize in range(self.dim):
                self.optimize_factor(local_ground_truth, mode_to_optimize)

def test_reduce_scatter():
    rank = MPI.COMM_WORLD.Get_rank()
    mat = np.array(range(54 * 5), dtype=np.double).reshape((54, 5))

    if rank == 0:
        print(mat)
        print("=====================================")

    recvbuf = np.zeros((2, 5))
    MPI.COMM_WORLD.Reduce_scatter_block([mat, MPI.DOUBLE], [recvbuf, MPI.DOUBLE])
    print(f'{rank}, {recvbuf}')
