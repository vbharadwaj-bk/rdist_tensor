import numpy as np
from numpy.random import default_rng
import h5py

import numpy as np
import numpy.linalg as la
from grid import Grid, TensorGrid
from mpi4py import MPI
from common import *
import cppimport.import_hook
import cpp_ext.redistribute_tensor as rd
import cpp_ext.tensor_kernels as tensor_kernels 
import cpp_ext.bloom_filter as bf
import cpp_ext.filter_nonzeros as nz_filter

from sampling import broadcast_common_seed

def allocate_recv_buffers(dim, count, lst_idx, lst_values, idx_t, val_t):
    for i in range(dim):
        lst_idx.append(np.empty(count, dtype=str_to_type[idx_t]))

    lst_values.append(np.empty(count, dtype=str_to_type[val_t]))

class DistSparseTensor:
    def __init__(self, tensor_file, preprocessing=None):
        self.type = "SPARSE_TENSOR"

        f = h5py.File(tensor_file, 'r')
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
        for i in range(self.dim):
            self.tensor_idxs.append(f[f'MODE_{i}'][start_nnz:end_nnz] - 1)

        # TODO: Need to remove this downcast! 
        # ============================================================== 
        for i in range(self.dim):
            self.tensor_idxs[i] = self.tensor_idxs[i].astype(np.uint32, copy=False)
        # ============================================================== 

        self.values = f['VALUES'][start_nnz:end_nnz]

        if preprocessing is not None:
            if preprocessing == "log_count":
                self.values = np.log(self.values + 1.0)
            else:
                print(f"Unknown preprocessing option '{preprocessing}' specified!")
                exit(1)

        local_norm = la.norm(self.values) ** 2
        result = np.zeros(1, dtype=np.double)
        world_comm.Allreduce([local_norm, MPI.DOUBLE], [result, MPI.DOUBLE]) 
        self.tensor_norm = np.sqrt(result.item())

        self.offsets = np.zeros(self.dim, dtype=np.uint64)

        # TODO: THESE ARE FORCED, REMOVE THEM!
        self.idx_dtype=np.uint32
        self.val_dtype=np.double

    def random_permute(self):
        '''
        Applies a random permutation to the indices of the sparse
        tensor. We could definitely make this more efficient, but meh 
        '''
        # TODO: Save the permutation so that it can be inverted later! 
        rng = np.random.default_rng(seed=broadcast_common_seed(MPI.COMM_WORLD))
        for i in range(self.dim):
            idxs = np.array(list(range(self.max_idxs[i])), dtype=np.ulonglong)
            perm = rng.permutation(idxs)
            self.tensor_idxs[i] = perm[self.tensor_idxs[i]]

    def gather_tensor(self):
        '''
        Warning: This function is for debugging purposes only!
        '''
        pass         

    def redistribute_nonzeros(self, tensor_grid, debug=False):
        '''
        Redistribute the nonzeros according to the provided tensor grid.
        '''
        assert( tensor_grid.grid.dim == self.dim)
        self.tensor_grid = tensor_grid
        grid = tensor_grid.grid
        prefix_array = grid.get_prefix_array()

        recv_buffers = []
        recv_values = []

        redistribute_nonzeros = get_templated_function(rd, "redistribute_nonzeros", 
                [self.idx_dtype, self.val_dtype])
        redistribute_nonzeros(tensor_grid.intervals, \
            self.tensor_idxs, \
            self.values, \
            grid.world_size, \
            prefix_array, recv_buffers, recv_values, \
            allocate_recv_buffers)

        self.tensor_idxs = recv_buffers
        self.values = recv_values[0]

        if debug:
            for i in range(len(recv_buffers[0])):
                for j in range(self.dim):
                    start = tensor_grid.start_coords[j][grid.coords[j]]
                    end= tensor_grid.start_coords[j][grid.coords[j] + 1]
                    val = recv_buffers[j][i]
                    assert(start <= val and val < end)

            if self.rank == 0:
                print("Finished debug test!")

        for j in range(self.dim):
            self.offsets[j] = self.tensor_grid.start_coords[j][grid.coords[j]]
            self.tensor_idxs[j] -= self.offsets[j] 

        # TODO: Need to add the index filter back in!
        #self.idx_filter = bf.IndexFilter(self.tensor_idxs, 0.01)

        # TODO: This takes up a lot of extra space! Should amortize away 
        self.offset_idxs = [self.tensor_idxs[j] 
            + self.offsets[j].astype(np.uint32) for j in range(self.dim)]

        self.mode_hashes = [np.zeros(interval, dtype=np.uint64) for interval in tensor_grid.intervals]

        # Compute hashes of the indices that this processor will reference 
        compute_mode_hashes = get_templated_function(nz_filter, "compute_mode_hashes", [np.uint32])
        compute_mode_hashes(tensor_grid.intervals.astype(np.uint32), self.mode_hashes)

    def mttkrp(self, factors, mode):
        '''
        For convenience, factors is sized equal to the dimension of the
        tensor, so we replace the factor at the mode to replace with the
        output buffer.

        Mode is the index of the mode to isolate along the column axis
        when matricizing the tensor
        ''' 
        factors[mode] *= 0.0
        tensor_kernels.sp_mttkrp(mode, factors, self.tensor_idxs, self.values)

    def sampled_mttkrp(self, mode, factors, sampled_idxs, sampled_lhs, sampled_rhs, weights):
        factors[mode] *= 0.0 
        tensor_kernels.sampled_mttkrp(mode, factors, sampled_idxs, sampled_lhs, sampled_rhs, weights)

    def sample_nonzeros(self, samples, weights, mode):
        return nz_filter.sample_nonzeros(self.tensor_idxs, self.values, samples, weights, mode)
