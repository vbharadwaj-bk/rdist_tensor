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

def allocate_recv_buffers(dim, count, lst_idx, lst_values):
    for i in range(dim):
        lst_idx.append(np.zeros(count, dtype=np.ulonglong))

    lst_values.append(np.zeros(count, dtype=np.double))

class DistSparseTensor:
    def __init__(self, tensor_file):
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

        self.values = f['VALUES'][start_nnz:end_nnz]

        local_norm = la.norm(self.values) ** 2
        result = np.zeros(1, dtype=np.double)
        world_comm.Allreduce([local_norm, MPI.DOUBLE], [result, MPI.DOUBLE]) 
        self.tensor_norm = np.sqrt(result.item())

    def random_permute(self, seed=42):
        '''
        Applies a random permutation to the indices of the sparse
        tensor. We could definitely make this more efficient, but meh 
        '''
        if seed is None:
            assert False
            # TODO: Select a random seed here and broadcast it to all other
            # processes!
 
            # Fill the seed here after broadcasting, make it consistent on all processors 

        rng = np.random.default_rng(seed)
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
        grid = tensor_grid.grid
        prefix_array = grid.get_prefix_array()

        recv_buffers = []
        recv_values = []

        rd.redistribute_nonzeros(tensor_grid.intervals, \
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

        # For debugging purposes only!
        self.original_tensor_idxs = [el.copy() for el in self.tensor_idxs]

        for j in range(self.dim):
            self.tensor_idxs[j] -= tensor_grid.start_coords[j][grid.coords[j]]

        print(f"Count: {len(self.tensor_idxs[0])}")

    def mttkrp(self, factors, mode, buffer):
        '''
        For convenience, factors is sized equal to the dimension of the
        tensor, so we replace the factor at the mode to replace with the
        output buffer.

        Mode is the index of the mode to isolate along the column axis
        when matricizing the tensor
        '''
        #print(factors)
        factor_args = []
        for i in range(len(factors)):
            if i==mode:
                factor_args.append(buffer)
            else:
                factor_args.append(factors[i])

        self.debug_list = []

        # This is a manual MTTKRP
        #interactions = 0
        #test_accum = 0.0
        #for i in range(len(self.values)):
        #    coord = [self.tensor_idxs[j][i] for j in range(self.dim)]
        #    accum = np.ones(factors[0].shape[1])
        #    for j in range(self.dim):
        #        if j != mode:
        #            accum *= factor_args[j][coord[j]]

        #    factor_args[mode][coord[mode]] += self.values[i] * accum

        #    if self.original_tensor_idxs[1][i] == 0:
        #        interactions += 1
                #print(f'{self.rank} {self.values[i]} {accum}')
        #        temp = 1.0
        #        for j in range(self.dim): 
        #            if j != mode:
        #                temp *= factor_args[j][coord[j]][0]
        #        test_accum += self.values[i] * temp 

        #print(f"Test Accumulator: {test_accum}")

        #print(f"Interaction Count: {interactions}")
        tensor_kernels.sp_mttkrp(mode, factor_args, \
            self.tensor_idxs, self.values) 

def test_tensor_redistribute():
    x = DistSparseTensor("tensors/nips.tns_converted.hdf5")
    grid = Grid([4, 4, 4, 1])
    tensor_grid = TensorGrid(x.max_idxs, grid=grid)
    x.redistribute_nonzeros(tensor_grid, debug=True)

def test_mttkrp():
    x = DistSparseTensor("tensors/nips.tns_converted.hdf5")
    factors = []

    rank = np.array([25], dtype=np.ulonglong)[0] 
    for i in range(x.dim):
        factors.append(np.array(list(range(x.max_idxs[i] * rank)), dtype=np.double).reshape((x.max_idxs[i], rank)))
    
    #for factor in factors:
    #    print(factor)

    m = np.zeros_like(factors[1], dtype=np.double)

    print("Starting MTTKRP...")
    start = time.time()
    x.mttkrp(factors, 1, m)
    interval = time.time() - start

    print(f"MTTKRP Completed in {interval} seconds!")
    #print(m)
    #grid = Grid([1, 1, 1])
    #tensor_grid = TensorGrid(x.max_idxs, grid=grid)

if __name__=='__main__':
    #test_tensor_redistribute()
    test_mttkrp()