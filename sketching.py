import numpy as np

def LeverageProdSketch(factor_matrices, leverage_scores, matricized_tensor, sample_fraction):
    height = np.prod([len(factor) for factor in factor_matrices])
    num_samples = int(np.round(sample_fraction * height))

    probs=None
    if leverage_scores is not None:
        probs = leverage_scores
    else:
        probs = [np.ones(factor.shape[0]) / factor.shape[0] for factor in factor_matrices]

    # Sample from multinomial distribution based on leverage score probabilities
    sample_idxs = [np.random.choice(list(range(len(prob_vector))), p=prob_vector, size=num_samples) for prob_vector in probs]

    lhs = np.ones((num_samples, factor_matrices[0].shape[1]))
    mat_tensor_idxs = np.zeros(num_samples, dtype=np.int64)
    for i in range(len(factor_matrices)):
        lhs *= factor_matrices[i][sample_idxs[i]]
        mat_tensor_idxs *= factor_matrices[i].shape[0]
        mat_tensor_idxs += sample_idxs[i]

    rhs = matricized_tensor[mat_tensor_idxs]

    return lhs, rhs


