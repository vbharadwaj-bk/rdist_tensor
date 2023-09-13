from mpi4py import MPI
import numpy as np
import argparse
import cppimport
import cppimport.import_hook

def decompose(args):
    from exafac.cpp_ext.py_module import Grid, TensorGrid, DistMat1D, LowRankTensor, ExactALS, TensorStationaryOpt0, AccumulatorStationaryOpt0, AccumulatorStationaryOpt1, test_distributed_exact_leverage 

    from exafac.sparse_tensor_e import DistSparseTensorE
    from exafac.grid import Grid as GridPy
    from exafac.grid import TensorGrid as TensorGridPy

    grid = None 
    rank = MPI.COMM_WORLD.Get_rank()

    tensors = {
        'uber': {
            "path": '/pscratch/sd/v/vbharadw/tensors/uber.tns_converted.hdf5',
            "preprocessing": None
        },
        'amazon': {
            "path": '/pscratch/sd/v/vbharadw/tensors/amazon-reviews.tns_converted.hdf5',
            "preprocessing": None
        },
        'reddit': {
            "path": '/pscratch/sd/v/vbharadw/tensors/reddit-2015.tns_converted.hdf5',
            "preprocessing": "log_count"
        } 
    }

    path = tensors[args.input]['path']

    sparse_tensor = DistSparseTensorE(path, grid, preprocessing=tensors[args.input]['preprocessing']) 
    low_rank_tensor = LowRankTensor(args.trank, sparse_tensor.tensor_grid)    
    low_rank_tensor.initialize_factors_gaussian_random()

    # We currently don't support the "algorithm" argument. 

    if args.optimizer == 'exact':
        optimizer = ExactALS(sparse_tensor.sparse_tensor, low_rank_tensor) 
    elif args.optimizer == 'tensor_stationary':
        optimizer = TensorStationaryOpt0(sparse_tensor.sparse_tensor, low_rank_tensor)
    elif args.optimizer == 'accumulator_stationary':
        optimizer = AccumulatorStationaryOpt1(sparse_tensor.sparse_tensor, low_rank_tensor) 
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    optimizer.initialize_ground_truth_for_als()

    fit = optimizer.compute_exact_fit()
    if rank == 0:
        print(f"Initial Fit: {fit}")
    optimizer.execute_ALS_rounds(80, 65536, 5)

    #optimizer.execute_ALS_rounds(5)

    fit = optimizer.compute_exact_fit()
    if rank == 0:
        print(f"Final Fit: {fit}")


if __name__=='__main__':
    num_procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    MPI.COMM_WORLD.Barrier()

    # Arguments for decomposition
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='Tensor name to decompose', required=True)
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=int)
    parser.add_argument("-s", "--samples", help="Number of samples taken from the KRP", required=False, type=int)
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser.add_argument('-dist','--distribution', type=str, help='Data distribution (tensor_stationary / accumulator_stationary)', required=False)
    parser.add_argument('-alg','--algorithm', type=str, help='', required=False)
    parser.add_argument("-o", "--output", help="Output file to print benchmark statistics", required=True)
    #parser.add_argument("-rs", help="Random seed", required=False, type=int, default=42)
    #parser.add_argument("-f", "--factor_file", help="File to print the output factors", required=False, type=str)
    #parser.add_argument("-p", "--preprocessing", help="Preprocessing algorithm to apply to the tensor", required=False, type=str)
    #parser.add_argument("-e", "--epoch_iter", help="Number of iterations per accuracy evaluation epoch", required=False, type=int, default=5)

    args = None
    try:
        if rank == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)

    if rank == 0:
        print("Loading Python modules...")
        import exafac.cpp_ext.py_module
        print("Modules loaded!")

    # Start running tests 
    decompose(args)
