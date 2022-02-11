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

def get_norm_distributed(buf, world):
    val = la.norm(buf) ** 2
    result = np.zeros(1)
    world.Allreduce([val, MPI.DOUBLE], [result, MPI.DOUBLE]) 
    return np.sqrt(result)

def krp(factor_matrices):
  height = factor_matrices[0].shape[0] * factor_matrices[1].shape[0]
  width = factor_matrices[0].shape[1]
  return np.einsum('ir,jr->ijr', *factor_matrices).reshape(height, width)

# Matrix is partitioned into block rows across processors
# This class is designed so that each slice of the processor
# grid holds a chunk of matrices. The slice dimension is
# the third parameter 
class DistMat1D:
    def __init__(self, rows, cols, grid, slice_dim):
        self.padded_rows = round_to_nearest(rows, grid.world_size)
        self.cols = cols
        self.local_rows_padded = self.padded_rows // grid.world_size
        self.local_window_size = self.local_rows_padded * self.cols

        # Should add in some explicit zero-padding logic 

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
        gram = (self.data.T @ self.data)
        self.grid.comm.Allreduce(MPI.IN_PLACE, gram)
        return gram

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
                buffer = np.zeros((self.factors[i].local_rows_padded * slice_size, self.rank))
                self.grid.slices[i].Allgather([self.factors[i].data, MPI.DOUBLE], 
                        [buffer, MPI.DOUBLE])
                gathered_matrices.append(buffer)

        return gathered_matrices

    def materialize_tensor(self):
        gathered_matrices = self.allgather_factors([True] * self.dim)
        self.local_materialized = np.einsum('r,ir,jr,kr->ijk', self.singular_values, *gathered_matrices)

    def initialize_factors_deterministic(self, offset):
        for factor in self.factors:
            factor.initialize_deterministic(offset)

    # Computes a distributed MTTKRP of all but one of this 
    # class's factors with a given dense tensor. Also performs 
    # gram matrix computation. 
    def optimize_factor(self, local_ten, mode_to_leave):
        factors_to_gather = [True] * self.dim
        factors_to_gather[mode_to_leave] = False

        # Compute gram matrices of all factors but the one we are currently
        # optimizing for 
        gram_matrices = [self.factors[i].compute_gram_matrix()
                         for i in range(len(self.factors)) 
                         if factors_to_gather[i]] 

        # Code for sketching based on leverage scores goes here!

        krp_gram_inv = la.inv(la.multi_dot(gram_matrices))
        gathered_matrices = self.allgather_factors(factors_to_gather)

        # Compute a local MTTKRP
        matricized_tensor = matricize_tensor(local_ten, mode_to_leave)
        mttkrp_unreduced = matricized_tensor.T @ krp(gathered_matrices)
        mttkrp_reduced = np.zeros_like(self.factors[mode_to_leave].data)

        self.grid.slices[mode_to_leave].Reduce_scatter_block([mttkrp_unreduced, MPI.DOUBLE], 
                [mttkrp_reduced, MPI.DOUBLE]) 

        return mttkrp_reduced @ krp_gram_inv 

    def als_fit(self, local_ground_truth, num_iterations):
        mttkrp = self.optimize_factor(local_ground_truth, 2)
        result = get_norm_distributed(mttkrp, grid.comm) 

        if self.grid.rank == 0:
            print(result)

def test_reduce_scatter():
    rank = MPI.COMM_WORLD.Get_rank()
    mat = np.array(range(54 * 5), dtype=np.double).reshape((54, 5))

    if rank == 0:
        print(mat)
        print("=====================================")

    recvbuf = np.zeros((2, 5))
    MPI.COMM_WORLD.Reduce_scatter_block([mat, MPI.DOUBLE], [recvbuf, MPI.DOUBLE])
    print(f'{rank}, {recvbuf}')


if __name__=='__main__':
    # For testing purposes, initialize a cubic grid
    num_procs = MPI.COMM_WORLD.Get_size()
    grid = Grid([int(np.cbrt(num_procs))] * 3)
    ground_truth = DistLowRank(grid, [27, 27, 27], 1, [1.0])
    ground_truth.initialize_factors_deterministic(0.1)
    ground_truth.materialize_tensor()

    ten_to_optimize = DistLowRank(grid, [27, 27, 27], 1, None)
    ten_to_optimize.initialize_factors_deterministic(0.05)

    #test_reduce_scatter()
    ten_to_optimize.als_fit(ground_truth.local_materialized, num_iterations=20)

    #dist_norm = get_norm_distributed(lowRankTensor.local_materialized, grid.comm)
    #dist_norm = get_norm_distributed(lowRankTensor.factors[0].data, grid.comm)
    #print(lowRankTensor.factors[0].data)
    #if grid.rank == 0:
    #    print(dist_norm)