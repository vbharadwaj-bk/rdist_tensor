import numpy as np
import numpy.linalg as la
import progressbar

import matplotlib.pyplot as plt

docstring = '''
This whole thing is a version-controlled dump of
a Jupyter notebook - but the core functions are
valid.
'''

def plot_rowGnorms_sq(mat, G, q=None, label=None):
    if q is None:
        q = np.ones(G.shape[0])
    Q = np.diag(q)
    norms = np.diag(A @ Q @ G @ Q @ A.T)
    norms = norms / np.sum(norms)
    plt.plot(norms, label=label)

def plot_test():
	n = 64
	r = 5

	A = np.random.rand(n, r) 

	G = np.random.rand(r, r)
	G = G.T @ G

	q = np.random.rand(r)

	plot_rowGnorms_sq(A, G)

def preprocess_mat(A):
    # Compute several QR decompositions of subsegments of the TS matrix A; should be
    # replaced by parallel tall-skinny QR decomposition
    R_lists = []
    full_height = A.shape[0]
    current_qr_height = full_height
    while current_qr_height > 0:
        R_lst_current = []
        
        weight = 0
        for i in range(0, full_height, current_qr_height):
            R_lst_current.append(la.qr(A[i:i + current_qr_height])[1])
            weight += la.norm(R_lst_current[-1], 'fro') ** 2
            
        R_lists.append(R_lst_current)
        current_qr_height = current_qr_height // 2
        
    return R_lists

def draw_samples(A, G, q_values, R_lists):
    lam, V = la.eigh(G)
    n = A.shape[0]
    r = A.shape[1]
    
    samples = np.zeros(n)
    sample_idxs = []
    sample_rows = np.zeros((len(q_values), r))

    for i in progressbar.progressbar(range(len(q_values))):
        q = q_values[i]
        q = q / la.norm(q)
        weights = [lam[i] * la.norm(R_lists[0][0] @ np.diag(V[:, i]) @ q) ** 2 for i in range(r)]
        eigen_choice = np.random.choice(list(range(r)), p=weights / np.sum(weights))
        
        # Draw samples through a recursive process
        level = 1
        position = 0
        eigen_diag = np.diag(V[:, eigen_choice])
        prev_weight = weights[eigen_choice] / lam[eigen_choice]
        while level < len(R_lists):
            p1_weight = la.norm(R_lists[level][position * 2] @ eigen_diag @ q) ** 2
            p2_weight = prev_weight - p1_weight
            p2_gt_weight = la.norm(R_lists[level][position * 2 + 1] @ eigen_diag @ q) ** 2

            choice = np.random.binomial(1, 1 - (p1_weight / prev_weight))
            
            if choice == 0:
                prev_weight = p1_weight
            elif choice == 1:
                prev_weight = p2_weight
            else:
                assert(False)
            position = position * 2 + choice
            level += 1
        
        weight_choice = position  
        samples[weight_choice] += 1
        sample_idxs.append(weight_choice)
        sample_rows[i] = A[weight_choice]
        
    return samples, sample_idxs, sample_rows

def test_sample_draw():
	num_samples = 10000
	q_values = [q] * num_samples

	R_lists = preprocess_mat(A)
	samples = draw_samples(A, G, q_values, R_lists)[0]
	samples /= np.sum(samples)
	plot_rowGnorms_sq(A, G, q=q, label="Ground Truth Row Norm Distribution")
	plt.plot(samples, label="Fake TSQR Sampler Distribution")
	plt.legend()
	plt.title("30k Sample Distribution w/ Constant Query Vector q")
	plt.xlabel("Row Index")
	plt.ylabel("Probability Density")
	samples[0]

def chain_mult(bufs, dim):
    temp = np.ones(dim)
    for buf in bufs:
        temp = temp * buf
    
    return temp

def draw_exact_mttkrp_samples(mats, num_samples):
    r = mats[0].shape[1]
    
    preprocess_mats = [preprocess_mat(mat) for mat in mats]
    gram_matrices = [mat.T @ mat for mat in mats]
    
    krp_gram = chain_mult(gram_matrices, r)
    phi = la.pinv(krp_gram)
    
    samples = np.ones((num_samples, r)) 
    sample_idxs = []
    for i in range(len(mats)):
        post_gram = chain_mult(gram_matrices[(i+1):], r)
        G = post_gram * phi
        
        _, mat_idxs, mat_draws = draw_samples(mats[i], G, samples, preprocess_mats[i])
        samples *= mat_draws
        sample_idxs.append(mat_idxs)
        
    rows = [mat.shape[0] for mat in mats]
    prods = np.ones(len(rows), dtype=np.int32)
    
    prods[1:] = np.cumprod(rows)[:-1]
    prods = prods[::-1]
    sample_idxs = np.array(sample_idxs)
    lin_idxs = sample_idxs.T @ prods
        
    return lin_idxs, samples

def krp(mats):
    if len(mats) == 1:
        return mats[0]
    else:
        running_mat = np.einsum('ik,jk->ijk', mats[0], mats[1]).reshape((mats[0].shape[0] * mats[1].shape[0], mats[0].shape[1]))
        
        for i in range(2, len(mats)):
            running_mat = np.einsum('ik,jk->ijk', running_mat, mats[i]).reshape((running_mat.shape[0] * mats[i].shape[0], mats[0].shape[1]))
            
        return running_mat

def test_mttkrp_lev_sampler():
	n = 8
	r = 5
	dim = 3

	mats = [np.random.rand(n, r) for i in range(dim)]
	krp_materialized = krp(mats)

	lin_idxs = draw_exact_mttkrp_samples(mats, 100000)[0]
	hist = np.bincount(lin_idxs)


	krp_q = la.qr(krp_materialized)[0]

	krp_norms = la.norm(krp_q, axis=1) ** 2
	plt.plot(krp_norms / np.sum(krp_norms), label="Ground Truth PDF")
	plt.plot(hist / np.sum(hist), label="PDF of our Sampler")
	plt.xlabel("KRP Row Index")
	plt.ylabel("Probability Density")
	plt.legend()
	plt.title("100k Sample Comparison to True PDF of MTTKRP Lev. Scores")
	plt.show()