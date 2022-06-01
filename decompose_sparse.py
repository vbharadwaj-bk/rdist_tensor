import numpy as np
from numpy.random import Generator, Philox

from mpi4py import MPI
import argparse

from low_rank_tensor import *
from grid import *
from sparse_tensor import *

def compute_best_residual(ground_truth, target_rank):
    # This assumes that the factors are normalized
    partial_ten = ground_truth.partial_materialize(target_rank)
    return get_norm_distributed(partial_ten, MPI.COMM_WORLD)

# Decomposes a synthetic (hyper)cubic tensor initialized
# deterministically with singular values that decay exponentially 
if __name__=='__main__':
    # For testing purposes, initialize a cubic grid
    num_procs = MPI.COMM_WORLD.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=int) 
    parser.add_argument("-p", "--skrp", help="Fraction of samples to take from the full height of the Khatri-Rhao Product", required=False, type=float)
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser.add_argument("-rs", help="Random seed", required=False, type=int, default=42)
    parser.add_argument("-o", "--output", help="Output file to print benchmark statistics", required=True)

    args = None
    try:
        if MPI.COMM_WORLD.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)

    ground_truth = DistSparseTensor("/global/cscratch1/sd/vbharadw/tensors/nell-1.tns_converted.hdf5")
    grid = Grid([4, 8, 8])
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    ground_truth.random_permute()
    ground_truth.redistribute_nonzeros(tensor_grid)

    ten_to_optimize = DistLowRank(tensor_grid, args.trank, None)
    ten_to_optimize.initialize_factors_deterministic(args.rs) 

    if grid.rank == 0:
        print(f"Starting benchmark...")

    ten_to_optimize.als_fit(ground_truth, output_file=args.output, num_iterations=args.iter, sketching_pct=args.skrp, compute_accuracy=True)