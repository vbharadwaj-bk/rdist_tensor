from exafac.distmat import	*
from exafac.low_rank_tensor import *

import numpy as	np
import numpy.linalg	as la
import json

import cppimport.import_hook
import os
from datetime import datetime

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
			"Slice All-gather":	0.0,
			"MTTKRP": 0.0,
			"Slice Reduce-Scatter":	0.0,
			"Sample Allgather": 0.0,
			"Sample Inflation": 0.0,
			"Nonzero Filtering + Redistribute": 0.0,
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

	def	fit(self, max_iterations, output_file, factor_file=None, epoch_interval=5):
		assert(epoch_interval>= 0)

		self.zero_timers()
		self.info["Max Iteration Count"] = max_iterations 

		now = datetime.now()
		self.info["Experiment Time"] = now.strftime("%d/%m/%Y %H:%M:%S")
		self.info["Epoch Interval"] = epoch_interval 
		self.info["Random Seed"] = get_init_seed()

		env_variables = ["OMP_NUM_THREADS", "MKL_NUM_THREADS"]

		for var in env_variables:
			if var in os.environ:
				self.info[var] = os.environ[var]
			else:
				self.info[var] = "UNSET" 


		low_rank_ten = self.ten_to_optimize
		ground_truth = self.ground_truth
		grid = low_rank_ten.grid

		loss_iterations	= []
		loss_values	= []

		pi = 3                # Same variable meaning as CP-ARLS-LEV 
		tol = 1e-5

		self.initial_setup()

		for	iter in	range(max_iterations):
			if grid.rank == 0:
				print(f"Starting iteration {iter}...")
			if (epoch_interval != 0 and iter % epoch_interval == 0) \
					or iter == 0 \
					or iter == max_iterations - 1:
				loss = low_rank_ten.compute_loss(ground_truth)
				loss_iterations.append(iter)
				loss_values.append(loss.item())

				if grid.rank == 0:
					print("Estimated Fit after iteration {}: {}".format(iter, loss)) 

				# Stopping condition 
				if len(loss_values) > pi: 
					recent_max_fit = np.max(loss_values[-pi:])
					old_max_fit = np.max(loss_values[:-pi])

					if recent_max_fit <= old_max_fit + tol:
						if grid.rank == 0:
							print(f"Loss failed to decrease after {iter} iterations. Stopping...")
						break

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

		if factor_file is not None:
			if self.grid.rank == 0:
				print(f"Writing factors to output file {factor_file}")
			self.ten_to_optimize.write_to_file(factor_file, metadata=self.info)
			#pass
