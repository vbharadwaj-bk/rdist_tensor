from distmat import *
import numpy as np
import numpy.linalg as la

from sampling import *
from sparse_tensor import allocate_recv_buffers
from alternating_optimizer import *
from common import *

import cppimport.import_hook
import cpp_ext.tensor_kernels as tensor_kernels 
import cpp_ext.filter_nonzeros as nz_filter 

def gather_samples_lhs(factors, dist_sample_count, mode_to_leave, grid, timers):
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
		if i == mode_to_leave:
			continue

		factor = factors[i]
		base_idx = factor.row_position * factor.local_rows_padded

		local_samples, local_counts, local_probs = get_samples_distributed_compressed(
				grid.comm,
				factors[i].leverage_scores,
				dist_sample_count)

		inflated_samples = np.zeros(dist_sample_count, dtype=np.uint64)
		sampled_rows = factors[i].data[local_samples]

		start = start_clock() 
		all_samples = allgatherv(grid.comm, base_idx + local_samples, MPI.UINT64_T)
		all_counts = allgatherv(grid.comm, local_counts, MPI.UINT64_T)
		all_probs = allgatherv(grid.comm, local_probs, MPI.DOUBLE)

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, timers, "Sample Allgather")

		start = start_clock() 
		all_rows = allgatherv(grid.comm, sampled_rows, MPI.DOUBLE)

		# All processors apply a consistent random
		# permutation to everything they receive 
		rng = default_rng(seed=broadcast_common_seed(grid.comm))
		perm = rng.permutation(dist_sample_count)

		tensor_kernels.inflate_samples_multiply(
				all_samples, all_counts, all_probs, all_rows,
				inflated_samples, weight_prods, lhs_buffer,
				perm
				)

		samples.append(inflated_samples)

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, timers, "LHS Assembly")

	return samples, np.exp(weight_prods), lhs_buffer	


class AccumulatorStationaryOpt1(AlternatingOptimizer):
	def __init__(self, ten_to_optimize, ground_truth, sample_count):
		super().__init__(ten_to_optimize, ground_truth)
		self.sample_count = sample_count
		self.info['Sample Count'] = self.sample_count
		self.info["Algorithm Name"] = "Accumulator Stationary Opt0"	
		self.info["Nonzeros Sampled Per Round"] = []

	def initial_setup(self):
		# Initial allgather of tensor factors 
		for mode in range(self.ten_to_optimize.dim):
			self.ten_to_optimize.factors[mode].allgather_factor()
			self.ten_to_optimize.factors[mode].compute_gram_matrix()
			self.ten_to_optimize.factors[mode].compute_leverage_scores()

	# Computes a distributed MTTKRP of all but one of this 
	# class's factors with a given dense tensor. Also performs 
	# gram matrix computation. 
	def optimize_factor(self, mode_to_leave):
		factors = self.ten_to_optimize.factors
		grid = self.ten_to_optimize.grid
		s = self.sample_count 

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
		stop_clock_and_add(start, self.timers, "Gram Matrix Computation")

		sample_idxs, weights, lhs_buffer = gather_samples_lhs(factors, s, mode_to_leave, grid, self.timers)

		# Should probably offload this to distmat.py file;
		# only have to do this once
		recv_idx, recv_values = [], []

		# This is an expensive operation, but we can optimize it away later
		offset_idxs = [self.ground_truth.tensor_idxs[j] 
				+ self.ground_truth.offsets[j] for j in range(self.dim)]

		start = start_clock() 
		nz_filter.sample_nonzeros_redistribute(
			offset_idxs, 
			self.ground_truth.values, 
			sample_idxs,
			weights,
			mode_to_leave,
			factors[mode_to_leave].local_rows_padded,
			factors[mode_to_leave].row_order_to_proc, 
			recv_idx,
			recv_values,
			allocate_recv_buffers)
	
		total_nnz_sampled = grid.comm.allreduce(len(recv_idx[0]))
		self.info["Nonzeros Sampled Per Round"].append(total_nnz_sampled)

		offset = factors[mode_to_leave].row_position * factors[mode_to_leave].local_rows_padded
		recv_idx[1] -= offset 

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Nonzero Filtering + Redistribute")

		# Perform the weight update after the nonzero
		# sampling so that repeated rows are combined 

		start = start_clock()
		lhs_buffer = np.einsum('i,ij->ij', weights, lhs_buffer)

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "LHS Assembly")

		start = start_clock()
		result_buffer = np.zeros_like(factors[mode_to_leave].data)
	
		tensor_kernels.spmm(
			lhs_buffer,
			recv_idx[0],
			recv_idx[1],
			recv_values,
			result_buffer
			)

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "MTTKRP")

		start = start_clock()
		lstsq_soln = la.lstsq(gram_prod, result_buffer.T, rcond=None)
		res = (np.diag(singular_values ** -1) @ lstsq_soln[0]).T.copy()	

		factors[mode_to_leave].data = res
		factors[mode_to_leave].normalize_cols()

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Gram LSTSQ Solve")

		start = start_clock()
		factors[mode_to_leave].compute_gram_matrix()
		stop_clock_and_add(start, self.timers, "Gram Matrix Computation")

		start = start_clock() 
		factors[mode_to_leave].compute_leverage_scores()
		stop_clock_and_add(start, self.timers, "Leverage Score Computation")
