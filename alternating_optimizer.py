from distmat import	*
from low_rank_tensor import *

import numpy as	np
import numpy.linalg	as la
import json

import cppimport.import_hook
import cpp_ext.tensor_kernels as tensor_kernels	
from sampling import get_random_seed

class AlternatingOptimizer:
	def	__init__(self, ten_to_optimize, ground_truth):
		self.ten_to_optimize =	ten_to_optimize
		self.ground_truth =	ground_truth
		self.dim = self.ten_to_optimize.dim
		self.grid = self.ten_to_optimize.grid

		# Timers common to the superset of our optimization
		# methods 
		self.timers = {
						"Gram Matrix Computation": 0.0,
						"Leverage Score	Computation": 0.0,
						"Slice All-gather":	0.0,
						"MTTKRP": 0.0,
						"Slice Reduce-Scatter":	0.0,
						"Gram LSTSQ Solve":	0.0,
						"Leverage Score Computation": 0.0
						}

		self.info = {}
		self.info["Mode Sizes"] = self.ten_to_optimize.mode_sizes.tolist()
		self.info["Tensor Target Rank"] = self.ten_to_optimize.rank
		self.info["Processor Count"] = self.ten_to_optimize.grid.world_size
		self.info["Timers"] = self.timers

		# Fields that the subclass needs to fill in 
		self.algorithm_name = None 

	# The two methods below should be implemented
	# by concrete subclasses 
	def initial_setup(self):
		'''
	 	Called just prior to optimization	
		'''
		raise NotImplementedError	

	def optimize_factor(self, mode_to_optimize):
		'''
	 	Optimize a single factor while keeping the
		others constant.	
		'''
		raise NotImplementedError	

	def zero_timers(self):
		for key in self.timers:
			self.timers[key] = 0.0

	def	fit(self, num_iterations, output_file, compute_accuracy_interval=0):
		assert(compute_accuracy_interval >= 0)
		self.zero_timers()
		self.info["Iteration Count"] = num_iterations 

		low_rank_ten = self.ten_to_optimize
		ground_truth = self.ground_truth
		grid = low_rank_ten.grid

		loss_iterations	= []
		loss_values	= []

		self.initial_setup()
		for	iter in	range(num_iterations):
			if (compute_accuracy_interval != 0 and iter % compute_accuracy_interval == 0) \
					or iter == 0 \
					or iter == num_iterations - 1:
				loss = low_rank_ten.compute_loss(ground_truth)
				loss_iterations.append(iter)
				loss_values.append(loss.item())

				if grid.rank == 0:
					print("Estimated Fit after iteration {}: {}".format(iter, loss)) 

			# Optimize each factor while keeping the others constant	
			for	mode_to_optimize in	range(self.dim):
				self.optimize_factor(mode_to_optimize)

		self.info["Loss Iterations"] = loss_iterations	
		self.info["Loss Values"] = loss_values	

		if self.grid.rank == 0:
			f =	open(output_file, 'a')
			json_obj = json.dumps(self.info, indent=4)
			f.write(json_obj + ",\n")
			f.close()
			#print(statistics)