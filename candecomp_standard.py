import mpi4py
import numpy as np
from numpy.random import Generator, Philox

import numpy.linalg as la
from grid import Grid
from local_kernels import *
from sketching import *
from mpi4py import MPI

import time
import json

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
# the axis along which to ``align" the distribution of the factor 
class DistMat1D:
    def __init__(self, rows, cols, grid, slice_dim):
        self.rows = rows 
        self.padded_rows = round_to_nearest(rows, grid.world_size)
        self.cols = cols
        self.local_rows_padded = self.padded_rows // grid.world_size
        self.local_window_size = self.local_rows_padded * self.cols

        self.grid = grid
        self.slice_dim = slice_dim

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

    def allgather_factor(self, with_leverage=False):
        slice_dim = self.slice_dim
        slice_size = self.grid.slices[slice_dim].Get_size()
        buffer_rowct = self.local_rows_padded * slice_size
        buffer_data = np.zeros((buffer_rowct, self.cols))
        buffer_leverage = np.zeros(buffer_rowct)

        self.grid.slices[slice_dim].Allgather([self.data, MPI.DOUBLE], 
                [buffer_data, MPI.DOUBLE])


        # Handle the overhang when mode length is not divisible by
        # the processor count 
        truncated_rowct = min(buffer_rowct, self.rows - buffer_rowct * self.grid.coords[slice_dim])
        truncated_rowct = max(truncated_rowct, 0) 
        buffer_data = buffer_data[:truncated_rowct] 

        self.gathered_factor = buffer_data

        if with_leverage:
            self.grid.slices[slice_dim].Allgather([self.leverage_scores, MPI.DOUBLE], 
                    [buffer_leverage, MPI.DOUBLE])
            buffer_leverage = buffer_leverage[:truncated_rowct]
            self.gathered_leverage = buffer_leverage 

def start_clock():
    return time.time()

def stop_clock_and_add(t0, dict, key):
    t1 = time.time()
    dict[key] += t1 - t0 

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
    def allgather_factors(self, which_factors, with_leverage=False):
        gathered_matrices = []
        gathered_leverage = []
        for i in range(self.dim):
            if which_factors[i]:
                self.factors[i].allgather_factor(with_leverage)
                gathered_matrices.append(self.factors[i].gathered_factor)

                if with_leverage: 
                    gathered_leverage.append(self.factors[i].gathered_leverage)

        if with_leverage:
            return gathered_matrices, gathered_leverage
        else:
            return gathered_matrices, None

    def materialize_tensor(self):
        gathered_matrices, _ = self.allgather_factors([True] * self.dim)
        #self.local_materialized = np.einsum('r,ir,jr,kr->ijk', self.singular_values, *gathered_matrices)
        self.local_materialized = tensor_from_factors_sval(self.singular_values, gathered_matrices) 

    # Materialize the tensor starting only from the given singular
    # value 
    def partial_materialize(self, singular_value_start):
        assert(singular_value_start < self.rank)

        gathered_matrices, _ = self.allgather_factors([True] * self.dim)
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
    def optimize_factor(self, local_ten, mode_to_leave, timer_dict, sketching_pct=None):

        sketching = sketching_pct is not None
        factors_to_gather = [True] * self.dim
        factors_to_gather[mode_to_leave] = False

        selected_indices = np.array(list(range(self.dim)))[factors_to_gather]
        selected_factors = [self.factors[idx] for idx in selected_indices] 

        # Compute gram matrices of all factors but the one we are currently
        # optimizing for, perform leverage-score based sketching if necessary 
        for idx in selected_indices:
            factor = self.factors[idx]
            
            start = start_clock()
            factor.compute_gram_matrix()
            stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

            if sketching_pct is not None:
                start = start_clock() 
                factor.compute_leverage_scores()
                stop_clock_and_add(start, timer_dict, "Leverage Score Computation")


        start = start_clock() 
        gram_prod = selected_factors[0].gram

        for i in range(1, len(selected_factors)):
            gram_prod = np.multiply(gram_prod, selected_factors[i].gram)

        # Compute inverse of the gram matrix 
        krp_gram_inv = la.pinv(gram_prod)
        stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")


        start = start_clock() 
        gathered_matrices, gathered_leverage = self.allgather_factors(factors_to_gather, with_leverage=sketching) 
        stop_clock_and_add(start, timer_dict, "Slice Replication")

        # Compute a local MTTKRP
        start = start_clock() 
        matricized_tensor = matricize_tensor(local_ten, mode_to_leave)
        mttkrp_unreduced = None

        if sketching_pct is None:
            mttkrp_unreduced = matricized_tensor.T @ krp(gathered_matrices)
        else:
            lhs, rhs = LeverageProdSketch(gathered_matrices, gathered_leverage, matricized_tensor, sketching_pct) 
            mttkrp_unreduced =  rhs.T @ lhs
        stop_clock_and_add(start, timer_dict, "MTTKRP")

        # Padding before reduce-scatter. Is there a smarter way to do this? 

        start = start_clock() 
        padded_rowct = self.factors[mode_to_leave].local_rows_padded * self.grid.slices[mode_to_leave].Get_size()

        reduce_scatter_buffer = np.zeros((padded_rowct, self.rank))
        reduce_scatter_buffer[:len(mttkrp_unreduced)] = mttkrp_unreduced

        mttkrp_reduced = np.zeros_like(self.factors[mode_to_leave].data)

        self.grid.slices[mode_to_leave].Reduce_scatter_block([reduce_scatter_buffer, MPI.DOUBLE], 
                [mttkrp_reduced, MPI.DOUBLE])  
        stop_clock_and_add(start, timer_dict, "Slice Reduce-Scatter")


        start = start_clock() 
        res = (krp_gram_inv @ mttkrp_reduced.T).T.copy()
        self.factors[mode_to_leave].data = res
        stop_clock_and_add(start, timer_dict, "Gram-Times-MTTKRP")

        return timer_dict

    def als_fit(self, local_ground_truth, num_iterations, sketching_pct, output_file):
        # Should initialize the singular values more intelligently, but this is fine
        # for now:

        statistics = {
                        "Gram Matrix Computation": 0.0,
                        "Leverage Score Computation": 0.0,
                        "Slice Replication": 0.0,
                        "MTTKRP": 0.0,
                        "Slice Reduce-Scatter": 0.0,
                        "Gram-Times-MTTKRP": 0.0
                        }

        statistics["Mode Sizes"] = self.mode_sizes
        statistics["Tensor Target Rank"] = self.rank
        statistics["Processor Count"] = self.grid.world_size
        statistics["Grid Dimensions"] = self.grid.axesLengths

        self.singular_values = np.ones(self.rank)
        self.materialize_tensor()
        loss = get_norm_distributed(local_ground_truth - self.local_materialized, self.grid.comm)

        for iter in range(num_iterations):
            self.materialize_tensor()
            loss = get_norm_distributed(local_ground_truth - self.local_materialized, self.grid.comm)

            if self.grid.rank == 0:
                print("Residual after iteration {}: {}".format(iter, loss)) 

            for mode_to_optimize in range(self.dim):
                self.optimize_factor(local_ground_truth, mode_to_optimize, statistics, sketching_pct=sketching_pct)

        if self.grid.rank == 0:
            f = open(output_file, 'a')
            json_obj = json.dumps(statistics, indent=4)
            f.write(json_obj + ",\n")
            f.close()