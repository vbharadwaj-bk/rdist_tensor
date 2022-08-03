from exafac.optim.alternating_optimizer import *
from exafac.common import *

import numpy as np
import numpy.linalg as la
import json

import cppimport.import_hook
import exafac.cpp_ext.tensor_kernels as tensor_kernels 

# Each processor sends some data to another processor
# and receives some data from another processor 
def send_recv_multiple_els(buf_lst, dest):
	# Exchange a list of buffer shapes. Assume that each buffer is the same
	# datatype, though 
	shapes = []
	for i in range(len(buf_lst)):
		shapes.append(buf_lst.shape)

	MPI.isend(shapes, dest)
	MPI.irecv()

def gather_samples_lhs(factors, 
			dist_sample_count, 
			mode_to_leave, 
			grid, 
			timers, 
			reuse_samples):
	start = start_clock() 
	samples = []
	inflated_sample_ids = []
	mode_rows = []
	weight_prods = np.zeros(dist_sample_count, dtype=np.double)
	weight_prods -= 0.5 * np.log(dist_sample_count)	

	for i in range(len(factors)):
		if i == mode_to_leave:
			continue

		factor = factors[i]
		
		if reuse_samples:
			all_samples, all_counts, all_probs, all_rows = factor.gathered_samples[0]
		else:
			all_samples, all_counts, all_probs, all_rows = factor.gathered_samples[mode_to_leave]

		inflated_samples = np.zeros(dist_sample_count, dtype=np.uint32)
		sample_ids = np.zeros(dist_sample_count, dtype=np.int64)

		# All processors apply a consistent random
		# permutation to everything they receive 

		rng = default_rng(seed=broadcast_common_seed(grid.comm))
		perm = rng.permutation(dist_sample_count)

		inflate_samples_multiply = get_templated_function(tensor_kernels, 
                "inflate_samples_multiply", 
                [np.uint32])

		inflate_samples_multiply(
				all_samples, all_counts, all_probs, all_rows,
				inflated_samples, weight_prods,
				perm,
				sample_ids
				)

		samples.append(inflated_samples)
		inflated_sample_ids.append(sample_ids)
		mode_rows.append(all_rows)

	stop_clock_and_add(start, timers, "Sample Inflation")

	return samples, np.exp(weight_prods), inflated_sample_ids, mode_rows

class DistributedGridOptimizer(AlternatingOptimizer):
	def	__init__(self, low_rank_ten, ground_truth, comm_scheduler, sample_scheduler):
		super().__init__(low_rank_ten, ground_truth)
		self.comm_scheduler = comm_scheduler
		self.sample_scheduler = sample_scheduler
		self.reuse_samples = True 

	def initial_setup(self):
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

	def optimize_factor(self, mode_to_leave):
		factors = self.ten_to_optimize.factors
		factor = factors[mode_to_leave]
		base_grid = self.ten_to_optimize.grid
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

		redis_grid = self.comm_scheduler.get_grid(mode_to_leave) 

		sample_idxs, weights, inflated_sample_ids, mode_rows = gather_samples_lhs(factors, s, 
				mode_to_leave, base_grid, self.timers, self.reuse_samples)

		recv_idx, recv_values = [], []

		sample_nonzeros_redistribute = get_templated_function(nz_filter, 
                "sample_nonzeros_redistribute", 
                [np.uint32, np.double])

		sample_mat = np.zeros((len(sample_idxs[0]), self.dim), dtype=np.uint32)

		for i in range(self.dim):
			if i < mode_to_leave:
				sample_mat[:, i] = sample_idxs[i]
			elif i > mode_to_leave:
				sample_mat[:, i] = sample_idxs[i-1]

		start = start_clock() 
		sample_nonzeros_redistribute(
			self.ground_truth.mat_idxs, 
			self.ground_truth.offsets, 
			self.ground_truth.values, 
			sample_mat,
			self.ground_truth.mode_hashes,
			weights,
			mode_to_leave,	
			factor.local_rows_padded,
			factor.row_order_to_proc, 
			recv_idx,
			recv_values,
			allocate_recv_buffers)

		total_nnz_sampled = grid.comm.allreduce(len(recv_idx[0]))
		#self.info["Nonzeros Sampled Per Round"].append(total_nnz_sampled)	

		offset = factor.row_position * factor.local_rows_padded
		recv_idx[1] -= offset 

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Nonzero Filtering + Redistribute")

		# Perform the weight update after the nonzero
		# sampling so that repeated rows are combined 

		start = start_clock()
		result_buffer = np.zeros_like(factor.data)

		spmm_compressed = get_templated_function(tensor_kernels, 
                "spmm_compressed", 
                [np.uint32, np.double])

		spmm_compressed(
			inflated_sample_ids,
			mode_rows,
			weights,
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

