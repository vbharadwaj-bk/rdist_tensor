from mpi4py import MPI
import numpy as np
import argparse
import gc
import cppimport.import_hook

if __name__=='__main__':
    num_procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='HDF5 of Input Tensor', required=True)
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=str)
    parser.add_argument("-s", "--samples", help="Number of samples taken from the KRP", required=False, type=str)
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser.add_argument("-rs", help="Random seed", required=False, type=int, default=42)
    parser.add_argument("-o", "--output", help="Output file to print benchmark statistics", required=True)
    parser.add_argument('-g','--grid', type=str, help='Grid Shape (Comma separated)', required=True)
    parser.add_argument('-op','--optimizer', type=str, help='Optimizer to use for tensor decomposition', required=False, default='exact')
    parser.add_argument("-f", "--factor_file", help="File to print the output factors", required=False, type=str)
    parser.add_argument("-p", "--preprocessing", help="Preprocessing algorithm to apply to the tensor", required=False, type=str)
    parser.add_argument("-e", "--epoch_iter", help="Number of iterations per accuracy evaluation epoch", required=False, type=int, default=5)
    parser.add_argument("--reuse", help="Whether or not to reuse samples between optimization rounds", action=argparse.BooleanOptionalAction)

    args = None
    try:
        if rank == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)

    if rank == 0:
        #import cpp_ext.bloom_filter 
        import exafac.cpp_ext.filter_nonzeros
        import exafac.cpp_ext.redistribute_tensor
        import exafac.cpp_ext.tensor_kernels

    MPI.COMM_WORLD.Barrier()

    if rank == 0:
        print("Loading Python modules...")

    from exafac.low_rank_tensor import *
    from exafac.grid import *
    from exafac.sparse_tensor import *
    from exafac.sampling import *

    #from exafac.optim.tensor_stationary_opt0 import TensorStationaryOpt0
    #from accumulator_stationary_opt0 import AccumulatorStationaryOpt0
    from exafac.optim.tensor_stationary_opt1 import TensorStationaryOpt1
    from exafac.optim.accumulator_stationary_opt1 import AccumulatorStationaryOpt1
    from exafac.optim.exact_als import ExactALS
    #from exafac.optim.dist_grid_optimizer import DistributedGridOptimizer

    # Let every process have a different random
    # seed based on its MPI rank; may be a better
    # way to initialize, though... 
    initialize_seed_generator(args.rs + rank)

    if rank == 0:
        print("Seed generator Initialized...")

    grid_dimensions = [int(el) for el in args.grid.split(',')]
    # TODO: Should assert that the grid has the proper dimensions here!

    if rank == 0:
        print("Initializing Sparse Tensor...")
    ground_truth = DistSparseTensor(args.input, preprocessing=args.preprocessing)
    grid = Grid(grid_dimensions)
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    ground_truth.random_permute()
    ground_truth.redistribute_nonzeros(tensor_grid)

    #new_grid = Grid([8, 1, 2, 4]) 
    #new_tensor_grid = TensorGrid(ground_truth.tensor_grid.tensor_dims, new_grid)
    #ground_truth.redistribute_nonzeros(new_tensor_grid)

    if args.samples is None:
        sample_counts = [None]
    else:
        sample_counts = args.samples.split(",")

    for sample_count in [el for el in sample_counts]: 
        for trank in [int(el) for el in args.trank.split(",")]:
            gc.collect()

            if sample_count is not None:
                sample_count = int(sample_count)

            ten_to_optimize = DistLowRank(tensor_grid, trank) 
            ten_to_optimize.initialize_factors_gaussian() 

            optimizer = None
            if args.optimizer == 'exact':
                optimizer = ExactALS(ten_to_optimize, ground_truth)
            elif args.optimizer == 'tensor_stationary':
                assert(args.samples is not None and sample_count >= 0)
                optimizer = TensorStationaryOpt1(ten_to_optimize, ground_truth, sample_count, reuse_samples=args.reuse)
            elif args.optimizer == 'accumulator_stationary':
                assert(args.samples is not None and sample_count >= 0)
                optimizer = AccumulatorStationaryOpt1(ten_to_optimize, ground_truth, sample_count, reuse_samples=args.reuse)
            elif args.optimizer == 'generic_grid':
                assert(args.samples is not None and sample_count >= 0)
                optimizer = DistributedGridOptimizer(ten_to_optimize, ground_truth, sample_count, None, None)

            else:
                print(f"Error, invalid optimizer specified: '{args.op}'")
                exit(1) 

            if grid.rank == 0:
                print(f"Starting tensor decomposition...")

            #ten_to_optimize.grid = new_grid
            #ten_to_optimize.tensor_grid = new_tensor_grid

            optimizer.fit(output_file=args.output,
                    factor_file = args.factor_file,
                    max_iterations=args.iter, 
                    epoch_interval=args.epoch_iter)

            if grid.rank == 0:
                print(f"Finished tensor decomposition...") 
    
    if grid.rank == 0:
        print("Finalized SHMEM...")
