import numpy as np
import numpy.linalg as la

# The einsum only works for 3-dim. tensors right now, but fine
def tensor_from_factors(factor_matrices):
    return np.einsum('ir,jr,kr->ijk', *factor_matrices)

    # Generate a synthetic tensor (of a known rank) that we will decompose

def generateLowrankDenseTensor(mode_sizes, rank):
    dims = len(mode_sizes)
    factor_matrices = [(np.random.rand(msize, rank) - 0.5) for msize in mode_sizes]

    # Scale a random subset of rows of each factor matrix by 10 to make the
    # resulting factor matrix more interesting
    scale_fraction = 0.1
    scale_factor = 8

    for i in range(dims):
        rows = np.array(np.random.choice(list(range(mode_sizes[i])), 
                                        size=int(scale_fraction * mode_sizes[i])), 
                        dtype=np.int32)
        factor_matrices[i][rows] *= scale_factor

    return tensor_from_factors(factor_matrices)

# Computes the KRP of exactly two factor matrices, should generalize for higher
# dimensions
def krp(factor_matrices):
    height = factor_matrices[0].shape[0] * factor_matrices[1].shape[0]
    width = factor_matrices[0].shape[1]
    return np.einsum('ir,jr->ijr', *factor_matrices).reshape(height, width)

def matricize_tensor(input_ten, column_mode):
    modes = list(range(len(input_ten.shape)))
    modes.remove(column_mode)
    modes.append(column_mode)

    mode_sizes_perm = [input_ten.shape[mode] for mode in modes]
    height, width = np.prod(mode_sizes_perm[:-1]), mode_sizes_perm[-1]

    return input_ten.transpose(modes).reshape(height, width)

def compute_residual(ground_truth, current):
    return np.linalg.norm(ground_truth - current)

# Performs the default Khatri-Rhao product and performs no sampling
# on the matricized tensor
def NoSketching(factor_matrices, matricized_tensor):
    return krp(factor_matrices), matricized_tensor

def als(input_ten, target_rank, num_iterations, verbose=False, residual_interval=1, sketcher=NoSketching):
    iters = []
    residuals = []
    ms = list(np.shape(input_ten))
    factors = [np.random.rand(msize, target_rank) - 0.5 
                        for msize in ms]

    for iter in range(num_iterations):
        if iter % residual_interval == 0:
            residual = compute_residual(input_ten, tensor_from_factors(factors))
        if verbose:
            print("Residual after iteration {}: {}".format(iter, residual))

        iters.append(iter)
        residuals.append(residual)

        factors[2] = la.lstsq(*sketcher([factors[0], factors[1]], matricize_tensor(input_ten, 2)), rcond=None)[0].T
        factors[1] = la.lstsq(*sketcher([factors[0], factors[2]], matricize_tensor(input_ten, 1)), rcond=None)[0].T
        factors[0] = la.lstsq(*sketcher([factors[1], factors[2]], matricize_tensor(input_ten, 0)), rcond=None)[0].T

    return iters, residuals, factors

if __name__=='__main__':
    test = generateLowrankDenseTensor([80, 80, 80], 40)

    ranks = [40]
    num_iterations=40

    for target_rank in ranks:
        iters, res, _ = als(test, target_rank=target_rank, 
                        num_iterations=num_iterations)
        print(res) 