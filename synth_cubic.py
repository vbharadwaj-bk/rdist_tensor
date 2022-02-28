import numpy as np
from mpi4py import MPI
import argparse


# Decomposes a synthetic (hyper)cubic tensor initialized
# deterministically with singular values that decay exponentially 


if __name__=='__main__':
    # For testing purposes, initialize a cubic grid
    num_procs = MPI.COMM_WORLD.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dim", help="Tensor Dimension", required=True)
    parser.add_argument("-s", "--sidelen", help="Length of each side of tensor", required=True)
    parser.add_argument("-g", "--grank", help="Rank of ground truth", required=True)
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True)
    parser.add_argument("-p", "--skrp", help="Percent of samples to take from the full height of the Khatri-Rhao Product", required=True)

    args = None
    try:
        if MPI.COMM_WORLD.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)

    #grid = Grid([int(np.cbrt(num_procs))] * 3)
    #ground_truth = DistLowRank(grid, [35] * 3, 5, [1.0, 0.8, 0.6, 0.4, 0.2])
    #ground_truth.initialize_factors_deterministic(0.1)
    #ground_truth.materialize_tensor()

    #ten_to_optimize = DistLowRank(grid, [35] * 3, 2, None)
    #ten_to_optimize.initialize_factors_deterministic(0.05)

    #ten_to_optimize.als_fit(ground_truth.local_materialized, num_iterations=5)

