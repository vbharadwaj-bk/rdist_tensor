import numpy as np
from mpi4py import MPI
import argparse

from candecomp_standard import DistLowRank
from grid import Grid

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
    parser.add_argument("-p", "--skrp", help="Fraction of samples to take from the full height of the Khatri-Rhao Product", required=True, type=float)
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)

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

    ground_truth = DistLowRank(grid, [args.sidelen] * args.dim, args.grank, [1.0, 0.8, 0.6, 0.4, 0.2])
    ground_truth.initialize_factors_deterministic(0.1)
    ground_truth.materialize_tensor()

    ten_to_optimize = DistLowRank(grid, [args.sidelen] * args.dim, args.trank, None)
    ten_to_optimize.initialize_factors_deterministic(0.05)

    ten_to_optimize.als_fit(ground_truth.local_materialized, num_iterations=args.iter)

