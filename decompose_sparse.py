from mpi4py import MPI
import numpy as np
import argparse

if __name__=='__main__':
    num_procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='HDF5 of Input Tensor', required=True)
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=int) 
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser.add_argument("-rs", help="Random seed", required=False, type=int, default=42)
    parser.add_argument("-o", "--output", help="Output file to print benchmark statistics", required=True)
    parser.add_argument('-g','--grid', type=str, help='Grid Shape (Comma separated)', required=True)
    parser.add_argument('-op','--optimizer', type=str, help='Optimizer to use for tensor decomposition', required=False, default='exact')
    parser.add_argument("-s", "--samples", help="Number of samples taken from the KRP", required=False, type=int)
    parser.add_argument("-f", "--factor_file", help="File to print the output factors", required=False, type=str)
    parser.add_argument("-p", "--preprocessing", help="Preprocessing algorithm to apply to the tensor", required=False, type=str)

    args = None
    try:
        if rank == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)

    if rank == 0:
        import cppimport.import_hook
        import cpp_ext.bloom_filter 
        import cpp_ext.filter_nonzeros
        import cpp_ext.redistribute_tensor
        import cpp_ext.tensor_kernels

    MPI.COMM_WORLD.Barrier()

    from low_rank_tensor import *
    from grid import *
    from sparse_tensor import *
    from sampling import *

    # List of optimizers
    from tensor_stationary_opt0 import TensorStationaryOpt0
    from accumulator_stationary_opt0 import AccumulatorStationaryOpt0
    from accumulator_stationary_opt1 import AccumulatorStationaryOpt1
    from exact_als import ExactALS

    # Let every process have a different random
    # seed based on its MPI rank; may be a better
    # way to initialize, though... 
    initialize_seed_generator(args.rs + rank)

    grid_dimensions = [int(el) for el in args.grid.split(',')]
    # TODO: Should assert that the grid has the proper dimensions here!

    ground_truth = DistSparseTensor(args.input, preprocessing=args.preprocessing)
    grid = Grid(grid_dimensions)
    tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
    ground_truth.random_permute()
    ground_truth.redistribute_nonzeros(tensor_grid) 

    ten_to_optimize = DistLowRank(tensor_grid, args.trank) 
    ten_to_optimize.initialize_factors_deterministic(args.rs) 
    #ten_to_optimize.initialize_factors_gaussian() 
    #ten_to_optimize.initialize_factors_rrf(ground_truth, 200000) 


    optimizer = None
    if args.optimizer == 'exact':
        assert(args.samples is None)
        optimizer = ExactALS(ten_to_optimize, ground_truth)
    elif args.optimizer == 'tensor_stationary':
        assert(args.samples is not None and args.samples >= 0)
        optimizer = TensorStationaryOpt0(ten_to_optimize, ground_truth, args.samples)
    elif args.optimizer == 'accumulator_stationary':
        assert(args.samples is not None and args.samples >= 0)
        optimizer = AccumulatorStationaryOpt1(ten_to_optimize, ground_truth, args.samples)
    else:
        print(f"Error, invalid optimizer specified: '{args.op}'")
        exit(1) 

    if grid.rank == 0:
        print(f"Starting tensor decomposition...")
 
    optimizer.fit(output_file=args.output,
            factor_file = args.factor_file,
            num_iterations=args.iter, 
            compute_accuracy_interval=0)