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

def gather_samples_lhs(factors, dist_sample_count, mode_to_leave, grid, timers, reuse_samples):
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
		
		if reuse_samples:
			all_samples, all_counts, all_probs, all_rows = factors[i].gathered_samples[0]
		else:
			all_samples, all_counts, all_probs, all_rows = factors[i].gathered_samples[mode_to_leave]

		inflated_samples = np.zeros(dist_sample_count, dtype=np.uint64)

		# All processors apply a consistent random
		# permutation to everything they receive 
		start = start_clock() 

		rng = default_rng(seed=broadcast_common_seed(grid.comm))
		perm = rng.permutation(dist_sample_count)


		inflate_samples_multiply = get_templated_function(tensor_kernels, 
                "inflate_samples_multiply", 
                [np.uint64])

		inflate_samples_multiply(
				all_samples, all_counts, all_probs, all_rows,
				inflated_samples, weight_prods, lhs_buffer,
				perm
				)

		samples.append(inflated_samples)

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, timers, "LHS Assembly")

	return samples, np.exp(weight_prods), lhs_buffer	

class AccumulatorStationaryOpt1(AlternatingOptimizer):
	def __init__(self, ten_to_optimize, ground_truth, sample_count, reuse_samples=True):
		super().__init__(ten_to_optimize, ground_truth)
		self.sample_count = sample_count
		self.reuse_samples = reuse_samples
		self.info['Sample Count'] = self.sample_count
		self.info["Algorithm Name"] = "Accumulator Stationary Opt1"	
		self.info["Nonzeros Sampled Per Round"] = []
		self.info["Samples Reused Between Rounds"] = self.reuse_samples 

	def initial_setup(self):
		'''
		TODO: Need to time these functions.
		'''
		ten = self.ten_to_optimize
		for mode in range(ten.dim):
			factor = ten.factors[mode]	
			factor.compute_gram_matrix()
			factor.compute_leverage_scores()

			if self.reuse_samples:
				factor.sample_and_gather_rows(factor.leverage_scores, None, 
						1, None, self.sample_count)
			else:
				factor.sample_and_gather_rows(factor.leverage_scores, None, 
						self.dim, mode, self.sample_count)

	# Computes a distributed MTTKRP of all but one of this 
	# class's factors with a given dense tensor. Also performs 
	# gram matrix computation. 
	def optimize_factor(self, mode_to_leave):
		factors = self.ten_to_optimize.factors
		factor = factors[mode_to_leave]
		grid = self.ten_to_optimize.grid
		s = self.sample_count 

		dim = len(factors)
		r = factor.data.shape[1]
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

		sample_idxs, weights, lhs_buffer = gather_samples_lhs(factors, s, 
				mode_to_leave, grid, self.timers, self.reuse_samples)

		recv_idx, recv_values = [], []

		sample_nonzeros_redistribute = get_templated_function(nz_filter, 
                "sample_nonzeros_redistribute", 
                [np.uint64, np.double])

		start = start_clock() 
		sample_nonzeros_redistribute(
			self.ground_truth.offset_idxs, 
			self.ground_truth.values, 
			sample_idxs,
			weights,
			mode_to_leave,
			factor.local_rows_padded,
			factor.row_order_to_proc, 
			recv_idx,
			recv_values,
			allocate_recv_buffers)
	
		total_nnz_sampled = grid.comm.allreduce(len(recv_idx[0]))
		self.info["Nonzeros Sampled Per Round"].append(total_nnz_sampled)

		offset = factor.row_position * factor.local_rows_padded
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
		result_buffer = np.zeros_like(factor.data)


		spmm = get_templated_function(tensor_kernels, 
                "spmm", 
                [np.uint64, np.double])

		spmm(
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

		factor.data = res
		factor.normalize_cols()

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Gram LSTSQ Solve")

		start = start_clock()
		factor.compute_gram_matrix()
		stop_clock_and_add(start, self.timers, "Gram Matrix Computation")

		start = start_clock() 
		factor.compute_leverage_scores()
		stop_clock_and_add(start, self.timers, "Leverage Score Computation")

		# Gather up samples here for future ALS iterations 
		factor.gathered_samples = []

		start = start_clock() 
		if self.reuse_samples:
			factor.sample_and_gather_rows(factor.leverage_scores, None, 1, 
				None, self.sample_count)
		else:
			factor.sample_and_gather_rows(factor.leverage_scores, None, self.dim, 
				mode_to_leave, self.sample_count)
	
		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Sample Allgather")
