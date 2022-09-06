from exafac.optim.alternating_optimizer import *
from exafac.common import *
from exafac.grid import *

import numpy as np
import numpy.linalg as la
import json

import cppimport.import_hook
import exafac.cpp_ext.tensor_kernels as tensor_kernels 

class ExactALS(AlternatingOptimizer):
	def	__init__(self, low_rank_ten, ground_truth):
		super().__init__(low_rank_ten, ground_truth)
		self.info["Algorithm Name"] = "Exact ALS"	

	def initial_setup(self):
		# Initial allgather of tensor factors 
		for mode in range(self.ten_to_optimize.dim):
			self.ten_to_optimize.factors[mode].normalize_cols()
			#self.ten_to_optimize.factors[mode].allgather_factor()
			self.ten_to_optimize.factors[mode].compute_gram_matrix()

	def optimize_factor(self, mode_to_leave):
		factors = self.ten_to_optimize.factors
		grid = self.ten_to_optimize.grid

		original_tensor_grid = self.ground_truth.tensor_grid

		new_grid = Grid([8, 1, 2, 4]) 
		new_tensor_grid = TensorGrid(self.ground_truth.tensor_grid.tensor_dims, new_grid)

		# This tests nonzero redistribution to an arbitrary grid 
		self.ground_truth.redistribute_nonzeros(new_tensor_grid)
		for factor in factors:
			factor.permute_to_grid(new_tensor_grid)

		dim = len(factors)
		factors_to_gather = [True] * dim 
		factors_to_gather[mode_to_leave] = False

		selected_indices = np.array(list(range(dim)))[factors_to_gather]
		selected_factors = [factors[idx] for idx in selected_indices] 

		start = start_clock() 
		col_norms = [factor.col_norms for factor in selected_factors]
		gram_matrices = [factor.gram for factor in selected_factors]
		gram_prod = chain_multiply_buffers(gram_matrices) 
		singular_values = chain_multiply_buffers(col_norms) 

		# Compute inverse of the gram matrix 
		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Gram Matrix Computation")

		for i in range(dim):
			factors[i].allgather_factor(new_grid.slices[i])

		start = start_clock()
		gathered_matrices = [factor.gathered_factor for factor in factors]

		sp_mttkrp = get_templated_function(tensor_kernels, 
				"sp_mttkrp", 
				[np.uint32, np.double])

		gathered_matrices[mode_to_leave] *= 0.0
		sp_mttkrp(mode_to_leave, 
			gathered_matrices, 
			self.ground_truth.tensor_idxs, 
			self.ground_truth.values)

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "MTTKRP")

		start = start_clock()
		mttkrp_reduced = np.zeros_like(factors[mode_to_leave].data)
		new_grid.slices[mode_to_leave].Reduce_scatter([gathered_matrices[mode_to_leave], MPI.DOUBLE], 
				[mttkrp_reduced, MPI.DOUBLE])

		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Slice Reduce-Scatter")
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

		#start = start_clock()  
		#factors[mode_to_leave].allgather_factor()
		MPI.COMM_WORLD.Barrier()
		stop_clock_and_add(start, self.timers, "Slice All-gather")

		for factor in factors:
			factor.permute_from_grid(new_tensor_grid)

		self.ground_truth.redistribute_nonzeros(original_tensor_grid)
