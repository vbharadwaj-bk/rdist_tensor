import numpy as np

# Einstein summation characters for free modes. 
# Don't need that many; don't need to support tensors that
# are too large 
einchars = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

def krp(factor_matrices):
    # TODO: Need a more intelligent implementation of this! 
    height = np.prod([factor.shape[0] for factor in factor_matrices])
    width = factor_matrices[0].shape[1]

    einsum_lhs = []
    einsum_rhs = ''
    for i in range(len(factor_matrices)):
        einsum_lhs.append(f'{einchars[i]}r')
        einsum_rhs += einchars[i]

    einsum_rhs += 'r'
    einsum_equation = ','.join(einsum_lhs) + "->" + einsum_rhs

    return np.einsum(einsum_equation, *factor_matrices).reshape(height, width)

def tensor_from_factors(factor_matrices):

    einsum_lhs = []
    einsum_rhs = ''
    for i in range(len(factor_matrices)):
        einsum_lhs.append(f'{einchars[i]}r')
        einsum_rhs += einchars[i]
 
    einsum_equation = ','.join(einsum_lhs) + "->" + einsum_rhs

    return np.einsum(einsum_equation, *factor_matrices)

def tensor_from_factors_sval(singular_values, factor_matrices):

    einsum_lhs = ['r']
    einsum_rhs = ''
    for i in range(len(factor_matrices)):
        einsum_lhs.append(f'{einchars[i]}r')
        einsum_rhs += einchars[i]
 
    einsum_equation = ','.join(einsum_lhs) + "->" + einsum_rhs
    return np.einsum(einsum_equation, singular_values, *factor_matrices)


def matricize_tensor(input_ten, column_mode):
  modes = list(range(len(input_ten.shape)))
  modes.remove(column_mode)
  modes.append(column_mode)

  mode_sizes_perm = [input_ten.shape[mode] for mode in modes]
  height, width = np.prod(mode_sizes_perm[:-1]), mode_sizes_perm[-1]

  return input_ten.transpose(modes).reshape(height, width)
