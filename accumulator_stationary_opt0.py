from distmat import *
import numpy as np
import numpy.linalg as la

from sampling import *
import cppimport.import_hook
import cpp_ext.tensor_kernels as tensor_kernels 

def allgatherv(world, local_buffer):
	'''
	If the local buffer is one-dimensional, perform
	a 1-dimensional all-gather. If two-dimensional,
	the resulting maatrices are stacked on top of
	each other, e.g. allgather(A1, A2) = [A1 A2].T	
	'''
	my_buffer_len = local_buffer.shape[0]
	buffer_lengths = np.zeros(world.Get_size(), dtype=np.ulonglong) 

	world.Allgather([nonzero_count, MPI.DOUBLE], 
			[buffer_lengths, MPI.COMM_WORLD.Get_size()])	

	print(buffer_lengths)


def initial_setup(ten_to_optimize):
	# Initial allgather of tensor factors 
	for mode in range(ten_to_optimize.dim):
		ten_to_optimize.factors[mode].allgather_factor()
		ten_to_optimize.factors[mode].compute_gram_matrix()
		ten_to_optimize.factors[mode].compute_leverage_scores()

def gather_distributed_samples_lhs(factors, dist_sample_count, mode_to_leave, grid):
	'''
	With this approach, we gather fresh samples for every
	tensor mode. 
	'''
	gathered_samples = []
	for i in range(len(factors)):
		if i != mode_to_leave:
			factor = factors[i]
			base_idx = factor.row_position * factor.local_rows_padded,

			local_samples, local_probs = get_samples_distributed(
					grid.world,
					factors[i].leverage_scores,
					dist_sample_count,
					base_idx)

			local_sample_count = len(local_samples)

# Computes a distributed MTTKRP of all but one of this 
# class's factors with a given dense tensor. Also performs 
# gram matrix computation. 
def optimize_factor(arg_dict, ten_to_optimize, grid, local_ten, mode_to_leave, timer_dict):
	factors = ten_to_optimize.factors
	dim = len(factors)
	r = factors[0].data.shape[1]
	factors_to_gather = [True] * dim 
	factors_to_gather[mode_to_leave] = False

	selected_indices = np.array(list(range(dim)))[factors_to_gather]
	selected_factors = [factors[idx] for idx in selected_indices] 

	start = start_clock() 
	col_norms = [factor.col_norms for factor in selected_factors]
	gram_matrices = [factor.gram for factor in selected_factors]
	gram_prod = chain_multiply_buffers(gram_matrices) 
	singular_values = chain_multiply_buffers(col_norms) 

	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

	start = start_clock()
	gathered_matrices = [factor.gathered_factor for factor in factors]

	s = arg_dict['sample_count'] 
	#samples_and_weights = [get_samples(factors[i].gathered_leverage, s) \
	#	for i in range(dim) if i != mode_to_leave]

	sampled_lhs, samples, weights = None 
	exit(1)

	weight_prods = np.zeros(s, dtype=np.double)
	weight_prods -= 0.5 * np.log(s)
	for i in range(dim - 1):
		weight_prods -= 0.5 * np.log(samples_and_weights[i][1]) 

	weight_prods = np.exp(weight_prods)
	samples = [el[0] for el in samples_and_weights]

	# For debugging purposes, want a buffer for the sampled LHS	
	sampled_lhs = np.zeros((s, r), dtype=np.double)	
	sampled_rhs = local_ten.sample_nonzeros(samples, weight_prods, mode_to_leave)
	local_ten.sampled_mttkrp(mode_to_leave, gathered_matrices, samples, sampled_lhs, sampled_rhs, weight_prods)

	sampled_rhs.print_contents()

	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "MTTKRP")

	start = start_clock()
	mttkrp_reduced = np.zeros_like(factors[mode_to_leave].data)
	grid.slices[mode_to_leave].Reduce_scatter([gathered_matrices[mode_to_leave], MPI.DOUBLE], 
			[mttkrp_reduced, MPI.DOUBLE])  
	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Slice Reduce-Scatter")

	start = start_clock()
	lstsq_soln = la.lstsq(gram_prod, mttkrp_reduced.T, rcond=None)
	res = (np.diag(singular_values ** -1) @ lstsq_soln[0]).T.copy()
	factors[mode_to_leave].data = res
	factors[mode_to_leave].normalize_cols()

	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Gram LSTSQ Solve")

	start = start_clock()
	factors[mode_to_leave].compute_gram_matrix()
	stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

	start = start_clock()  
	factors[mode_to_leave].allgather_factor()
	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Slice All-gather")

	start = start_clock()  
	factors[mode_to_leave].compute_leverage_scores()
	factors[mode_to_leave].allgather_leverage_scores()
	stop_clock_and_add(start, timer_dict, "Leverage Score Computation")

	return timer_dict
