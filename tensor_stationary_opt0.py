from distmat import *
import numpy as np
import numpy.linalg as la
import json

import cppimport.import_hook
import cpp_ext.tensor_kernels as tensor_kernels 

# Computes a distributed MTTKRP of all but one of this 
# class's factors with a given dense tensor. Also performs 
# gram matrix computation. 
def optimize_factor(factors, grid, local_ten, mode_to_leave, timer_dict):
	dim = len(factors)
	factors_to_gather = [True] * dim 
	factors_to_gather[mode_to_leave] = False

	selected_indices = np.array(list(range(dim)))[factors_to_gather]
	selected_factors = [factors[idx] for idx in selected_indices] 

	# Compute gram matrices of all factors but the one we are currently
	# optimizing for, perform leverage-score based sketching if necessary 
	for idx in selected_indices:
		factor = factors[idx]
		
		start = start_clock()
		factor.compute_gram_matrix()
		stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

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

	# The gathered factor to optimize is overwritten 
	local_ten.mttkrp(gathered_matrices, mode_to_leave)
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

	return timer_dict