import numpy as np
import mpi4py
from mpi4py import MPI

from numpy.random import default_rng

def get_samples(row_probs, num_samples):
	row_range = list(range(len(row_probs)))
	sample_idxs = np.random.choice(row_range, p=row_probs, size=num_samples) 
	sampled_probs = row_probs[sample_idxs]

	return sample_idxs, sampled_probs 

def get_samples_distributed(world, row_probs, dist_sample_count, base_idx):
	seed_rng = np.random.default_rng()
	if world.rank == 0:
		seed = seed_rng.integers(50000000)
	else:
		seed = None	
	seed = world.bcast(seed, root=0)
	rng = np.random.default_rng(seed=seed)

	local_weight = np.sum(row_probs)
	processor_weights = np.zeros(world.Get_size(), dtype=np.double)
	world.Allgather([local_weight, MPI.DOUBLE], 
			[processor_weights, MPI.DOUBLE])	

	sample_counts = rng.multinomial(dist_sample_count, processor_weights)
	local_sample_count = sample_counts[world.rank]

	# Take local samples at random
	row_range = list(range(base_idx, base_idx + len(row_probs)))
	sample_idxs = np.random.choice(row_range, p=row_probs, size=local_sample_count) 
	sampled_probs = row_probs[sample_idxs]

	return sample_idxs, sampled_probs