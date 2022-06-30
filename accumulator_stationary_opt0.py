from distmat import *
import numpy as np
import numpy.linalg as la

from sampling import *
from sparse_tensor import allocate_recv_buffers
import cppimport.import_hook
import cpp_ext.tensor_kernels as tensor_kernels 
import cpp_ext.filter_nonzeros as nz_filter 

def initial_setup(ten_to_optimize):
	# Initial allgather of tensor factors 
	for mode in range(ten_to_optimize.dim):
		ten_to_optimize.factors[mode].allgather_factor()
		ten_to_optimize.factors[mode].compute_gram_matrix()
		ten_to_optimize.factors[mode].compute_leverage_scores()

def allgatherv(world, local_buffer, mpi_dtype):
	'''
	If the local buffer is one-dimensional, perform
	a 1-dimensional all-gather. If two-dimensional,
	the resulting matrices are stacked on top of
	each other, e.g. allgather(A1, A2) = [A1 A2].T.
	Could generalize to handle higher-dimensional
	tensors, but this is fine for now.
	'''
	if local_buffer.ndim != 1 and local_buffer.ndim != 2:
		print("Input buffer must be 1 or 2-dimensional")
		exit(1)

	sendcount = np.array([local_buffer.shape[0]], dtype=np.ulonglong)
	sendcounts = np.empty(world.Get_size(), dtype=np.ulonglong) 

	world.Allgather([sendcount, MPI.UNSIGNED_LONG_LONG], 
			[sendcounts, MPI.UNSIGNED_LONG_LONG])

	offsets = np.empty(len(sendcounts), dtype=np.ulonglong)
	offsets[0] = 0
	offsets[1:] = np.cumsum(sendcounts[:-1])

	if local_buffer.ndim == 1:
		shape = np.sum(sendcounts)
	elif local_buffer.ndim == 2:
		cols = local_buffer.shape[1]
		shape = (np.sum(sendcounts), cols)
		sendcounts *= cols
		offsets *= cols 

	recv_buffer = np.zeros(shape, dtype=local_buffer.dtype)
	world.Allgatherv([local_buffer, mpi_dtype], 
		[recv_buffer, sendcounts, offsets, mpi_dtype]	
		)

	return recv_buffer

def gather_samples_lhs(factors, dist_sample_count, mode_to_leave, grid):
	'''
	With this approach, we gather fresh samples for every
	tensor mode. 
	'''
	samples = []
	weight_prods = np.zeros(dist_sample_count, dtype=np.double)
	weight_prods -= 0.5 * np.log(dist_sample_count)
	r = factors[0].data.shape[1]
	lhs_buffer = np.ones((dist_sample_count, r), dtype=np.double) 

	for i in range(len(factors)):
		# TODO: Need to chain-multiply the
		# gathered LHS samples here!	
		if i == mode_to_leave:
			continue

		factor = factors[i]
		base_idx = factor.row_position * factor.local_rows_padded

		local_samples, local_probs = get_samples_distributed(
				grid.comm,
				factors[i].leverage_scores,
				dist_sample_count)

		sampled_rows = factors[i].data[local_samples]	
		
		all_samples = allgatherv(grid.comm, base_idx + local_samples, MPI.UNSIGNED_LONG_LONG)
		all_probs = allgatherv(grid.comm, local_probs, MPI.DOUBLE)
		all_rows = allgatherv(grid.comm, sampled_rows, MPI.DOUBLE)

		# All processors apply a consistent random
		# permutation to everything they receive 
		#seed = broadcast_common_seed(grid.comm)
		#for shuffle_el in [all_samples, all_probs, all_rows]:
		#	rng = default_rng(seed=seed)
		#	rng.shuffle(shuffle_el)

		samples.append(all_samples)
		weight_prods -= 0.5 * np.log(all_probs) 
		lhs_buffer *= all_rows

	return samples, np.exp(weight_prods), lhs_buffer	

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
	#s = 200 
	#samples_and_weights = [get_samples(factors[i].gathered_leverage, s) \
	#	for i in range(dim) if i != mode_to_leave]

	sample_idxs, weights, lhs_buffer = gather_samples_lhs(factors, s, mode_to_leave, grid)
	lhs_buffer = np.einsum('i,ij->ij', weights, lhs_buffer)

	# Should probably offload this to distmat.py file;
	# only have to do this once
	recv_idx, recv_values = [], []
	row_order_to_proc = np.empty(grid.world_size, dtype=np.ulonglong)

	grid.comm.Allgather(
		[factors[mode_to_leave].row_position, MPI.UNSIGNED_LONG_LONG],
		[row_order_to_proc, MPI.UNSIGNED_LONG_LONG]	
	)

	#print(lhs_buffer)
	#print("Made it here first!")

	nz_filter.sample_nonzeros_redistribute(
		local_ten.tensor_idxs, 
		local_ten.values, 
		sample_idxs,
		weights,
		mode_to_leave,
		factors[mode_to_leave].local_rows_padded,
		row_order_to_proc.astype(int), 
		recv_idx,
		recv_values,
		allocate_recv_buffers 
		)

	result_buffer = np.zeros_like(factors[mode_to_leave].data)

	tensor_kernels.sampled_mttkrp_with_lhs_assembled(
		lhs_buffer,
		recv_idx[0],
		recv_idx[1],
		recv_values,
		result_buffer
		)

	print(f"MTTKRP Reduced Norm: {la.norm(result_buffer)}")

	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "MTTKRP")

	start = start_clock()
	lstsq_soln = la.lstsq(gram_prod, result_buffer.T, rcond=None)
	res = (np.diag(singular_values ** -1) @ lstsq_soln[0]).T.copy()	

	factors[mode_to_leave].data = res
	factors[mode_to_leave].normalize_cols()

	MPI.COMM_WORLD.Barrier()
	stop_clock_and_add(start, timer_dict, "Gram LSTSQ Solve")

	start = start_clock()
	factors[mode_to_leave].compute_gram_matrix()
	stop_clock_and_add(start, timer_dict, "Gram Matrix Computation")

	start = start_clock() 
	factors[mode_to_leave].compute_leverage_scores()
	factors[mode_to_leave].allgather_leverage_scores()
	stop_clock_and_add(start, timer_dict, "Leverage Score Computation")

	return timer_dict
