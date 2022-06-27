import numpy as np
import mpi4py
from mpi4py import MPI

from common import *

from numpy.random import default_rng

__seed = None 
__seed_rng = None 

def initialize_seed_generator(seed):
	global __seed
	global __seed_rng
	__seed = seed
	__seed_rng = default_rng(seed=seed)

def get_random_seed():
	global __seed
	global __seed_rng
	if __seed is None or __seed_rng is None:
		print("Error, need to initialize seed RNG!")
		exit(1)
	return __seed_rng.integers(1000000000)

def get_samples(row_probs, num_samples):
	row_range = list(range(len(row_probs)))
	sample_idxs = np.random.choice(row_range, p=row_probs, size=num_samples) 
	sampled_probs = row_probs[sample_idxs]

	return sample_idxs, sampled_probs

def broadcast_common_seed(world):
	seed = world.bcast(get_random_seed(), root=0)
	return seed

def get_samples_distributed(world, row_probs, dist_sample_count):
	rng = default_rng(seed=broadcast_common_seed(world))

	local_weight = np.sum(row_probs)
	processor_weights = np.zeros(world.Get_size(), dtype=np.double)
	world.Allgather([local_weight, MPI.DOUBLE], 
			[processor_weights, MPI.DOUBLE])	

	sample_counts = rng.multinomial(dist_sample_count, processor_weights)
	local_sample_count = sample_counts[world.rank]

	# Take local samples at random
	row_range = list(range(cl(len(row_probs))))
	sample_idxs = rng.choice(row_range, p=row_probs / local_weight, size=local_sample_count) 
	sampled_probs = row_probs[sample_idxs]

	return sample_idxs, sampled_probs