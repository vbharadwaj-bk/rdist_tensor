import numpy as np
import h5py

import numpy as np
from grid import Grid
from mpi4py import MPI
from common import *

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
            self.tensor_idxs.append(f[f'MODE_{i}'][start_nnz:end_nnz])

        self.values = f['VALUES'][start_nnz:end_nnz]



    def random_permute(self):
        '''
        Applies a random permutation to the indices of the sparse
        tensor. 
        '''
        pass


if __name__=='__main__':
    x = DistSparseTensor("tensors/test.tns_converted.hdf5")