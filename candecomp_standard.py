import mpi4py
import numpy as np
import numpy.linalg as la
from grid import Grid
from mpi4py import MPI

def round_to_nearest(n, m):
    return (n + m - 1) // m * m

def matricize_tensor(input_ten, column_mode):
  modes = list(range(len(input_ten.shape)))
  modes.remove(column_mode)
  modes.append(column_mode)

  mode_sizes_perm = [input_ten.shape[mode] for mode in modes]
  height, width = np.prod(mode_sizes_perm[:-1]), mode_sizes_perm[-1]

  return input_ten.transpose(modes).reshape(height, width)

def compute_residual(ground_truth, current):
  return np.linalg.norm(ground_truth - current)

def get_norm_distributed(buf, world):
    val = la.norm(buf) ** 2
    result = np.zeros(1)
    world.Allreduce([val, MPI.DOUBLE], [result, MPI.DOUBLE]) 
    return np.sqrt(result)

def krp(factor_matrices):
  height = factor_matrices[0].shape[0] * factor_matrices[1].shape[0]
  width = factor_matrices[0].shape[1]
  return np.einsum('ir,jr->ijr', *factor_matrices).reshape(height, width)

def tensor_from_factors(factor_matrices):
  return np.einsum('ir,jr,kr->ijk', *factor_matrices)

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

        self.data = np.zeros((self.local_rows_padded, self.cols)) 

        # TODO: Should store the padding offset here, add a view
        # into the matrix that represents the true data
        #print(f"Rank: {grid.rank}\t{self.grid.slices[0].Get_rank()}")
        #print(f"Row position: {self.row_position}")

    def initialize_deterministic(self, offset):
        value_start = self.row_position * self.local_window_size 
        self.data = np.array(range(value_start, value_start + self.local_window_size)).reshape(self.local_rows_padded, self.cols)
        self.data = np.cos(self.data + offset)

    def compute_gram_matrix(self):
        rowct = min(self.rows - self.row_position * self.local_rows_padded, self.local_rows_padded)
        rowct = max(rowct, 0)

        if rowct == 0:
            ncol = np.shape(self.data)[1]
            self.gram = np.zeros((ncol, ncol))
        else:
            data_trunc = self.data[:rowct]
            self.gram = (data_trunc.T @ data_trunc)

        self.grid.comm.Allreduce(MPI.IN_PLACE, self.gram)

        #if self.grid.rank == 0:
        #    print(self.gram)

    def compute_leverage_scores(reduced_gram_prod):
        gram_inv = la.inv(reduced_gram_prod)

        # TODO: This function can be made more efficient using
        # BLAS calls!

        self.leverage_scores = np.zeros(len(self.data))
        for i in range(len(self.data)):
            row = self.data[[i]]
            self.leverage_scores[i] = row @ gram_inv @ row.T 

        leverage_weight = np.array(np.sum(leverage_scores))
        grid.comm.Allreduce(MPI.IN_PLACE, leverage_weight)
        self.leverage_scores /= leverage_weight

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

    def materialize_tensor(self):
        gathered_matrices = self.allgather_factors([True] * self.dim)
        self.local_materialized = np.einsum('r,ir,jr,kr->ijk', self.singular_values, *gathered_matrices)
        #print(f'Shape: {self.local_materialized.shape}')


    def initialize_factors_deterministic(self, offset):
        for factor in self.factors:
            factor.initialize_deterministic(offset)

    # Computes a distributed MTTKRP of all but one of this 
    # class's factors with a given dense tensor. Also performs 
    # gram matrix computation. 
    def optimize_factor(self, local_ten, mode_to_leave, sketching=False):
        factors_to_gather = [True] * self.dim
        factors_to_gather[mode_to_leave] = False

        selected_factors = [self.factors[i] for i in range(len(self.factors)) 
                            if factors_to_gather[i]] 

        # Compute gram matrices of all factors but the one we are currently
        # optimizing for 

        for factor in selected_factors:
            factor.compute_gram_matrix()

        gram_prod = selected_factors[0].gram

        for i in range(1, len(selected_factors)):
            gram_prod = np.multiply(gram_prod, selected_factors[i].gram)

        krp_gram_inv = la.inv(gram_prod)
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
