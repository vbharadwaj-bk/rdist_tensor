import numpy as np
from numpy.random import Generator, Philox

from mpi4py import MPI
import argparse

from low_rank_tensor import *
from grid import *
from sparse_tensor import *
from sampling import *

if __name__=='__main__':
    num_procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='HDF5 of Input Tensor', required=True)
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=int) 
    parser.add_argument("-s", "--samples", help="Number of samples taken from the KRP", required=False, type=int)
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser.add_argument("-rs", help="Random seed", required=False, type=int, default=42)
    parser.add_argument("-o", "--output", help="Output file to print benchmark statistics", required=True)
    parser.add_argument('-g','--grid', type=str, help='Grid Shape (Comma separated)', required=True)

    args = None
    try:
        if rank == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)

    # Let every process have a different random
    # seed based on its MPI rank; may be a better
    # way to initialize, though... 
    initialize_seed_generator(args.rs + rank)

    grid_dimensions = [int(el) for el in args.grid.split(',')]
    # TODO: Should assert that the grid has the proper dimensions here!

    ground_truth = DistSparseTensor(args.input)
    grid = Grid(grid_dimensions)
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    ground_truth.random_permute()
    ground_truth.redistribute_nonzeros(tensor_grid)

    ten_to_optimize = DistLowRank(tensor_grid, args.trank)
    ten_to_optimize.initialize_factors_deterministic(args.rs) 

    if grid.rank == 0:
        print(f"Starting tensor decomposition...")

    ten_to_optimize.als_fit(ground_truth, output_file=args.output, num_iterations=args.iter, num_samples=args.samples, compute_accuracy=True)