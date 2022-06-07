import numpy as np

def get_samples(row_probs, num_samples):
	row_range = list(range(len(row_probs)))
	sample_idxs = np.random.choice(row_range, p=row_probs, size=num_samples) 
	sampled_probs = row_probs[sample_idxs]

	return sample_idxs, sampled_probs 

