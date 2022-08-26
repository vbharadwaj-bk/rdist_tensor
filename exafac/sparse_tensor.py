import numpy as np
from numpy.random import default_rng
import h5py

import numpy as np
import numpy.linalg as la
from exafac.grid import Grid, TensorGrid
from mpi4py import MPI
from exafac.common import *

import cppimport.import_hook
import exafac.cpp_ext.redistribute_tensor as rd
import exafac.cpp_ext.tensor_kernels as tensor_kernels 
#import exafac.cpp_ext.bloom_filter as bf
import exafac.cpp_ext.filter_nonzeros as nz_filter

from exafac.sampling import broadcast_common_seed

class TensorSampler:
    def __init__(self, id):
        self.name = id 

class HashedSampleSet(TensorSampler):
    def __init__(self, idxs_mat, offsets, values):
        super().__init__("HashedSampleSet")
        self.idxs_mat = idxs_mat
        self.offsets = offsets
        self.values = values

class HashedTensorTuples(TensorSampler):
    def __init__(self, mat_idxs, values):
        super().__init__("HashedTensorTuples")
        self.slicer = nz_filter.TensorSlicer(mat_idxs, values)

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

        if self.rank == 0:
            print("Loading sparse tensor...")

        for i in range(self.dim):
            self.tensor_idxs.append(f[f'MODE_{i}'][start_nnz:end_nnz] - 1)

        self.values = f['VALUES'][start_nnz:end_nnz]

        MPI.COMM_WORLD.Barrier()
        if self.rank == 0:
            print("Finished loading sparse tensor...")


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

        # TODO: Eventually, will support more than these two datatypes 
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

        for j in range(self.dim):
            self.offsets[j] = self.tensor_grid.start_coords[j][grid.coords[j]]
            self.tensor_idxs[j] -= self.offsets[j] 

        # TODO: Need to add the index filter back in!
        #self.idx_filter = bf.IndexFilter(self.tensor_idxs, 0.01)

        # TODO: This takes up a lot of extra space! Should amortize away 
        self.offset_idxs = [self.tensor_idxs[j] 
            + self.offsets[j].astype(np.uint32) for j in range(self.dim)]

        # Compute hashes of the indices that this processor will reference 
        self.mat_idxs = np.zeros((
            len(self.tensor_idxs[0]),
            self.dim
            ),
            dtype=self.tensor_idxs[0].dtype
        )

        for i in range(self.dim):
            self.mat_idxs[:, i] = self.offset_idxs[i]

        #self.sampler = HashedSampleSet(self.mat_idxs, self.offsets, self.values)
        self.slicer = nz_filter.TensorSlicer(self.mat_idxs, self.values)


    def sampled_mttkrp(self, mode, factors, sampled_idxs, sampled_lhs, sampled_rhs, weights):
        factors[mode] *= 0.0 
        tensor_kernels.sampled_mttkrp(mode, factors, sampled_idxs, sampled_lhs, sampled_rhs, weights)

    def sample_nonzeros(self, samples, weights, mode):
        return nz_filter.sample_nonzeros(self.tensor_idxs, self.values, samples, weights, mode)