from distmat import *
from alternating_optimizer import *

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


	#s = 100000
	#samples = [get_samples(factors[i].gathered_leverage, s) \
	#	for i in range(dim) if i != mode_to_leave]

	# To generate samples for this simplest implementation, each
	# processor will take an equal number of samples from the locally
	# owned portion of the data	
	#sampled_rhs = local_ten.sample_nonzeros(samples, mode_to_leave)
	#local_ten.sampled_mttkrp(mode_to_leave, gathered_matrices, samples, sampled_rhs)
	#local_ten.mttkrp(gathered_matrices, mode_to_leave)

# ========================================
class TensorStationaryOpt0(AlternatingOptimizer):
	def __init__(self, ten_to_optimize, ground_truth, sample_count):
		super().__init__(ten_to_optimize, ground_truth)
		self.sample_count = sample_count

	def initial_setup(self):
		# Initial allgather of tensor factors 
		for mode in range(self.ten_to_optimize.dim):
			self.ten_to_optimize.factors[mode].allgather_factor()
			self.ten_to_optimize.factors[mode].compute_gram_matrix()
			self.ten_to_optimize.factors[mode].compute_leverage_scores()
			self.ten_to_optimize.factors[mode].allgather_leverage_scores()

	# Computes a distributed MTTKRP of all but one of this 
	# class's factors with a given dense tensor. Also performs 
	# gram matrix computation. 
	def optimize_factor(self, mode_to_leave):
		factors = self.ten_to_optimize.factors
		s = self.sample_count
		grid = self.ten_to_optimize.grid

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

		start = start_clock()
		gathered_matrices = [factor.gathered_factor for factor in factors]

		samples_and_weights = [get_samples(factors[i].gathered_leverage, s) \
			for i in range(dim) if i != mode_to_leave]

		weight_prods = np.zeros(s, dtype=np.double)
		weight_prods -= 0.5 * np.log(s)
		for i in range(dim - 1):
			weight_prods -= 0.5 * np.log(samples_and_weights[i][1]) 

		weight_prods = np.exp(weight_prods)
		samples = [el[0] for el in samples_and_weights]

		# For debugging purposes, want a buffer for the sampled LHS	
		sampled_lhs = np.zeros((s, r), dtype=np.double)	
		sampled_rhs = self.local_ten.sample_nonzeros(samples, weight_prods, mode_to_leave)
		self.local_ten.sampled_mttkrp(mode_to_leave, gathered_matrices, samples, sampled_lhs, sampled_rhs, weight_prods)
		
		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "MTTKRP")

		start = start_clock()
		mttkrp_reduced = np.zeros_like(factors[mode_to_leave].data)
		grid.slices[mode_to_leave].Reduce_scatter([gathered_matrices[mode_to_leave], MPI.DOUBLE], 
				[mttkrp_reduced, MPI.DOUBLE])  
		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Slice Reduce-Scatter")

		#print(f"MTTKRP Unreduced Norm: {la.norm(gathered_matrices[mode_to_leave])}")
		#print(f"LHS Buffer Norm: {la.norm(sampled_lhs)}")

		start = start_clock()
		lstsq_soln = la.lstsq(gram_prod, mttkrp_reduced.T, rcond=None)
		res = (np.diag(singular_values ** -1) @ lstsq_soln[0]).T.copy()

		factors[mode_to_leave].data = res
		factors[mode_to_leave].normalize_cols()

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Gram LSTSQ Solve")

		start = start_clock()
		factors[mode_to_leave].compute_gram_matrix()
		stop_clock_and_add(start, self.timers, "Gram Matrix Computation")

		start = start_clock()  
		factors[mode_to_leave].allgather_factor()
		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Slice All-gather")

		start = start_clock()  
		factors[mode_to_leave].compute_leverage_scores()
		factors[mode_to_leave].allgather_leverage_scores()
		stop_clock_and_add(start, self.timers, "Leverage Score Computation")