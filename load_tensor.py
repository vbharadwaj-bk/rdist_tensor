import numpy as np
import h5py

import numpy as np
from grid import Grid, TensorGrid
from mpi4py import MPI
from common import *
import cppimport.import_hook
import cpp_ext.redistribute_tensor as rd

def allocate_recv_buffers(dim, count, lst):
    for i in range(dim):
        lst.append(np.zeros(count, dtype=np.ulonglong))

    lst.append(np.zeros(count, dtype=np.double))

class DistSparseTensor:
    def __init__(self, tensor_file):
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

        self.values = f['VALUES'][start_nnz:end_nnz]

    def random_permute(self):
        '''
        Applies a random permutation to the indices of the sparse
        tensor. 
        '''
        pass

def test_tensor_redistribute():
    x = DistSparseTensor("tensors/nips.tns_converted.hdf5")
    grid = Grid([1, 1, 1, 1])
    prefix_array = grid.get_prefix_array()
    tensor_grid = TensorGrid(x.max_idxs, grid=grid)
    
    #grid.test_prefix_array()

    recv_buffers = []
    rd.redistribute_nonzeros(tensor_grid.intervals, x.tensor_idxs, x.values, grid.world_size, prefix_array, recv_buffers, allocate_recv_buffers)

    for i in range(len(recv_buffers[0])):
        for j in range(x.dim):
            start = tensor_grid.start_coords[j][grid.coords[j]]
            end= tensor_grid.start_coords[j][grid.coords[j] + 1]
            val = recv_buffers[j][i]
            #print(f"Start: {start}, Val: {val}, End: {end}")
            assert(start <= val and val < end) 

if __name__=='__main__':
    test_tensor_redistribute()