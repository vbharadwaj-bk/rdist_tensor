from tokenize import group
from distmat import *
import numpy as np
import numpy.linalg as la

from sampling import *
import cppimport.import_hook
import cpp_ext.tensor_kernels as tensor_kernels 

# ========================================
# This is deprecated test code that just tests the sampled
# MTTKRP and RHS nonzero filtering functions
# to make sure everthing is working correctly

#gmttkrp = gathered_matrices[mode_to_leave].copy()
#total_ten_entries = 1	
#for i in range(dim):
#	total_ten_entries *= factors[i].data.shape[0] 

#total_ten_entries = total_ten_entries // factors[mode_to_leave].data.shape[0]
#samples = [np.zeros(total_ten_entries, dtype=np.ulonglong) for i in range(dim - 1)]
#for i in range(total_ten_entries):
#	val = i
#	for j in range(dim):
#		if j != mode_to_leave:
#			if j > mode_to_leave:
#				samples[j-1][i] = val % factors[j].data.shape[0]
#			else:
#				samples[j][i] = val % factors[j].data.shape[0]
#			val = val // factors[j].data.shape[0]

#sampled_rhs = local_ten.sample_nonzeros(samples, mode_to_leave)
#sampled_rhs.print_contents()

#local_ten.sampled_mttkrp(mode_to_leave, gathered_matrices, samples, sampled_rhs)
#print(la.norm(gmttkrp - gathered_matrices[mode_to_leave]))

# ========================================

def initial_setup(ten_to_optimize):
	# Initial allgather of tensor factors 
	for mode in range(ten_to_optimize.dim):
		ten_to_optimize.factors[mode].allgather_factor()
		ten_to_optimize.factors[mode].compute_gram_matrix()
		ten_to_optimize.factors[mode].compute_leverage_scores()
		ten_to_optimize.factors[mode].allgather_leverage_scores()

# Computes a distributed MTTKRP of all but one of this 
# class's factors with a given dense tensor. Also performs 
# gram matrix computation. 
def optimize_factor(factors, grid, local_ten, mode_to_leave, timer_dict):
	mode_to_leave = 1
	dim = len(factors)
	factors_to_gather = [True] * dim 
	factors_to_gather[mode_to_leave] = False

	selected_indices = np.array(list(range(dim)))[factors_to_gather]
	selected_factors = [factors[idx] for idx in selected_indices] 	

	start = start_clock() 
	gram_prod = selected_factors[0].gram
	for i in range(1, len(selected_factors)):
		gram_prod = np.multiply(gram_prod, selected_factors[i].gram)

	# Compute inverse of the gram matrix 
	krp_gram_inv = la.pinv(gram_prod)
	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

	start = start_clock()
	gathered_matrices = [factor.gathered_factor for factor in factors]

	# Let's do the single-node case before we get
	# to the multi-node case... note, we need to compress 
	# factor matrices and perform several modifications for 
	# the multi-node case

	s = 100000
	samples = [get_samples(factors[i].gathered_leverage, s) \
		for i in range(dim) if i != mode_to_leave]

	# To generate samples for this simplest implementation, each
	# processor will take an equal number of samples from the locally
	# owned portion of the data	

	sampled_rhs = local_ten.sample_nonzeros(samples, mode_to_leave)
	local_ten.sampled_mttkrp(mode_to_leave, gathered_matrices, samples, sampled_rhs)

	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "MTTKRP")

	start = start_clock()
	mttkrp_reduced = np.zeros_like(factors[mode_to_leave].data)
	grid.slices[mode_to_leave].Reduce_scatter([gathered_matrices[mode_to_leave], MPI.DOUBLE], 
			[mttkrp_reduced, MPI.DOUBLE])  
	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Slice Reduce-Scatter")
	start = start_clock() 
	res = (krp_gram_inv @ mttkrp_reduced.T).T.copy()
	factors[mode_to_leave].data = res

	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Gram-Times-MTTKRP")
	
	start = start_clock()  
	factors[mode_to_leave].allgather_factor()
	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Slice All-gather")

	start = start_clock()
	factors[mode_to_leave].compute_gram_matrix()
	stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

	return timer_dict