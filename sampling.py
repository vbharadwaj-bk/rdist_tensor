import numpy as np
import mpi4py
from mpi4py import MPI

from common import *

from numpy.random import default_rng

__seed_rng = None 

def initialize_seed_generator(seed):
	global __seed_rng
	__seed_rng = default_rng(seed=seed)

def get_random_seed():
	global __seed_rng
	if __seed_rng is None:
		print("Error, need to initialize seed RNG!")
		exit(1)

	#print("Seed generator called!")
	return __seed_rng.integers(1000000000)

def get_samples(row_probs, num_samples):
	# TODO: Need to make this method distributed!
	row_range = list(range(len(row_probs)))
	seed = get_random_seed() 
	rng = default_rng(seed=seed)
	sample_idxs = rng.choice(row_range, p=row_probs, size=num_samples) 
	sampled_probs = row_probs[sample_idxs]

	#print(f'Seed: {seed}, Samples: {sample_idxs}')

	return sample_idxs, sampled_probs

def broadcast_common_seed(world):
	seed = world.bcast(get_random_seed(), root=0)
	return seed

def get_samples_distributed(world, row_probs, dist_sample_count):
	seed = broadcast_common_seed(world) 
	rng = default_rng(seed=seed)

	local_weight = np.sum(row_probs)
	processor_weights = np.zeros(world.Get_size(), dtype=np.double)
	world.Allgather([local_weight, MPI.DOUBLE], 
			[processor_weights, MPI.DOUBLE])	

	# Sum of all local weights should be 1, but we will divide by the sum to
	# avoid round-off errors 
	total_weight = np.sum(processor_weights)
	processor_weights /= total_weight

	sample_counts = rng.multinomial(dist_sample_count, processor_weights)
	local_sample_count = sample_counts[world.rank]

	# Take local samples at random
	row_range = list(range(cl(len(row_probs))))
	sample_idxs = rng.choice(row_range, p=row_probs / local_weight, size=local_sample_count) 
	sampled_probs = row_probs[sample_idxs]

	#print(f'Seed: {seed}, Samples: {sample_idxs}')

	return sample_idxs, sampled_probs