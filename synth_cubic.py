import numpy as np
from numpy.random import Generator, Philox

from mpi4py import MPI
import argparse

from candecomp_standard import DistLowRank, get_norm_distributed
from grid import Grid

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
    parser.add_argument("-d", "--dim", help="Tensor Dimension", required=True, type=int)
    parser.add_argument("-s", "--sidelen", help="Length of each side of tensor", required=True, type=int)
    parser.add_argument("-g", "--grank", help="Rank of ground truth", required=True, type=int)
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

    gridlen = np.round(np.power(num_procs, 1 / args.dim))
 
    assert(gridlen ** args.dim == num_procs)

    grid = Grid([gridlen] * args.dim)

    # Exponentially decaying singular values starting from a maximum of the target
    # rank value

    singular_values = np.exp(0 - np.array(range(args.grank))) * args.grank

    ground_truth = DistLowRank(grid, [args.sidelen] * args.dim, args.grank, singular_values)
    #ground_truth.initialize_factors_deterministic(0.1) 
    ground_truth.initialize_factors_random(args.rs)
    ground_truth.materialize_tensor()

    best_resnorm = compute_best_residual(ground_truth, 3)
    #print(f'Best Residual Norm: {best_resnorm}')

    ten_to_optimize = DistLowRank(grid, [args.sidelen] * args.dim, args.trank, None)
    #ten_to_optimize.initialize_factors_deterministic(0.05)
    ten_to_optimize.initialize_factors_random(args.rs) 

    ten_to_optimize.als_fit(ground_truth.local_materialized, output_file=args.output, num_iterations=args.iter, sketching_pct=args.skrp)