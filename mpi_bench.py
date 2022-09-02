from mpi4py import MPI
import numpy as np
import argparse
import json
import time
import os

from exafac.common import *

world = MPI.COMM_WORLD
num_ranks = world.Get_size()
rank = world.Get_rank()

def bench_reduce_scatter(exp_list):
	num_trials = 4
	experiment = {
		"operation" : "reduce-scatter",
		"num_nodes": int(os.getenv("SLURM_NNODES")),
		"num_mpi_ranks": num_ranks,
		"bytes_per_word": 8, 
		"num_trials":  num_trials,
		"words_communicated": [],
		"mean_times" : [],
		"std_times" : [],
	}
	for i in range(11, 28):
		total_words = 2 ** i
		words_per_proc = total_words // num_ranks

		if words_per_proc * num_ranks != total_words:
			continue

		pre_buf = np.ones(total_words, dtype=np.double)
		post_buf = np.zeros(words_per_proc, dtype=np.double) 

		times = []
		for i in range(num_trials):
			MPI.COMM_WORLD.Barrier()	
			start = time.time()

			world.Reduce_scatter([pre_buf, MPI.DOUBLE], 
					[post_buf, MPI.DOUBLE])

			MPI.COMM_WORLD.Barrier()	
			elapsed = time.time() - start
			times.append(elapsed)

		experiment["words_communicated"].append(total_words)
		experiment["mean_times"].append(np.mean(times))
		experiment["std_times"].append(np.std(times))
		
	exp_list.append(experiment)

if __name__=='__main__':
	experiments = [] 
	if rank == 0:
		parser = argparse.ArgumentParser()
		parser.add_argument('-o','--output_file', type=str, help='Output file', required=True)
		args = parser.parse_args()
		f =	open(args.output_file, 'r')
		try:
			loaded_exps = json.load(open(args.output_file, 'r'))
			experiments = loaded_exps
		except :
			pass
		f.close()

	bench_reduce_scatter(experiments)

	if rank == 0:	
		f =	open(args.output_file, 'w')
		json_str = json.dumps(experiments, indent=4)
		f.write(json_str)
		f.close()
