import numpy as np
from numpy.random import default_rng
import h5py

import numpy as np
import numpy.linalg as la
#import cppimport.import_hook
#import cpp_ext.tensor_kernels as tensor_kernels 

class SparseTensor:
    '''
    For single-node testing purposes only (in Jupyter). Does not have
    an MPI component.
    '''
    def __init__(self, tensor_file):
        self.type = "SPARSE_TENSOR"

        f = h5py.File(tensor_file, 'r')
        self.max_idxs = f['MAX_MODE_SET'][:]
        self.min_idxs = f['MIN_MODE_SET'][:]
        self.dim = len(self.max_idxs)

        # The tensor must have at least one mode
        self.nnz = len(f['MODE_0']) 

        self.tensor_idxs = []
        for i in range(self.dim):
            self.tensor_idxs.append(f[f'MODE_{i}'][:] - 1)

        self.values = f['VALUES'][:]
        self.tensor_norm = la.norm(self.values) 
 
    def random_permute(self, seed=42):
        '''
        Applies a random permutation to the indices of the sparse
        tensor. We could definitely make this more efficient, but meh 
        '''
        rng = np.random.default_rng(seed)
        for i in range(self.dim):
            idxs = np.array(list(range(self.max_idxs[i])), dtype=np.ulonglong)
            perm = rng.permutation(idxs)
            self.tensor_idxs[i] = perm[self.tensor_idxs[i]]