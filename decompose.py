from mpi4py import MPI
import numpy as np
import os, json, time, argparse
from datetime import datetime 
import cppimport
import cppimport.import_hook

def decompose(args, output_filename, trial_num):
    from exafac.cpp_ext.py_module import Grid, TensorGrid, DistMat1D, LowRankTensor, ExactALS, AccumulatorStationary, CP_ARLS_LEV, STS_CP
    from exafac.sparse_tensor_e import DistSparseTensorE

    grid = None 
    rank = MPI.COMM_WORLD.Get_rank()

    tensors = {
        'uber': {
            "path": '/pscratch/sd/v/vbharadw/tensors/uber.tns_converted.hdf5',
            "preprocessing": "none"
        },
        'amazon': {
            "path": '/pscratch/sd/v/vbharadw/tensors/amazon-reviews.tns_converted.hdf5',
            "preprocessing": "none" 
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

    if args.algorithm == 'exact':
        # Ignore the distribution for exact ALS, this is alawys tensor stationary. 
        optimizer = ExactALS(sparse_tensor.sparse_tensor, low_rank_tensor) 
    else:
        if args.algorithm == 'cp_arls_lev':
            sampler = CP_ARLS_LEV(low_rank_tensor)
        elif args.algorithm == 'sts_cp':
            sampler = STS_CP(low_rank_tensor)
        else:
            raise ValueError(f"Unknown algorithm {args.algorithm}")

        if args.distribution == "accumulator_stationary":
            optimizer = AccumulatorStationary(sparse_tensor.sparse_tensor, low_rank_tensor, sampler)
        elif args.distribution == "tensor_stationary":
            optimizer = TensorStationary(sparse_tensor, low_rank_tensor, sampler)
        else:
            raise ValueError(f"Unrecognized distribution {args.distribution}") 
    optimizer.initialize_ground_truth_for_als()

    initial_fit = optimizer.compute_exact_fit()
    if rank == 0:
        print(f"Initial Fit: {initial_fit}")

    if args.algorithm != 'exact':
        optimizer.execute_ALS_rounds(args.iter, args.samples, args.epoch_iter)
    else:
        print("Error, need to fix this!")

    optimizer_stats = json.loads(optimizer.get_statistics_json())

    final_fit = optimizer.compute_exact_fit()

    now = datetime.now()
    output_dict = {
        'time': now.strftime('%m/%d/%Y, %H:%M:%S'),
        'metadata': args.metadata,
        'input': args.input,
        'target_rank': args.trank,
        'iterations': args.iter,
        'algorithm': args.algorithm,
        'data_distribution': args.distribution,
        'sample_count': args.samples,
        'accuracy_epoch_length': args.epoch_iter,
        'trial_count': args.repetitions,
        'trial_num': trial_num,
        'initial_fit': initial_fit,
        'final_fit': final_fit,
        'stats': optimizer_stats,
    }
    print(json.dumps(output_dict, indent=4))
    print(f"Final Fit: {final_fit}")

    if rank == 0 and output_filename is not None:
        with open(os.path.join(args.output_folder, output_filename), 'w') as f:
            f.write(json.dumps(output_dict, indent=4)) 


if __name__=='__main__':
    num_procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    MPI.COMM_WORLD.Barrier()

    # Arguments for decomposition
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='Tensor name to decompose', required=True)
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=int)
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser.add_argument('-dist','--distribution', type=str, help='Data distribution (tensor_stationary / accumulator_stationary)')
    parser.add_argument('-alg','--algorithm', type=str, help='Algorithm to perform decomposition')
    parser.add_argument("-s", "--samples", help="Number of samples taken from the KRP", required=False, type=int)
    parser.add_argument("-o", "--output_folder", help="Folder name to print statistics", required=False)
    parser.add_argument("-e", "--epoch_iter", help="Number of iterations per accuracy evaluation epoch", required=False, type=int, default=5)
    parser.add_argument("-r", "--repetitions", help="Number of repetitions for multiple trials", required=False, type=int)
    parser.add_argument("-m", "--metadata", help="A string piece of metadata to include output json", required=False, type=str, default="")
    #parser.add_argument("-rs", help="Random seed", required=False, type=int, default=42)
    #parser.add_argument("-f", "--factor_file", help="File to print the output factors", required=False, type=str)
    #parser.add_argument("-p", "--preprocessing", help="Preprocessing algorithm to apply to the tensor", required=False, type=str)

    args = None
    try:
        if rank == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)

    remaining_trials = None
    output_filename = None 
    trial_num = None

    if rank == 0 and args.output_folder is not None:
        print("Loading Python modules...")
        import exafac.cpp_ext.py_module
        print("Modules loaded!")

        filename_prefix = '_'.join([args.input, str(args.trank), 
                                    str(args.iter), args.distribution, 
                                    args.algorithm, str(args.samples), 
                                    str(args.epoch_iter)])

        files = os.listdir(args.output_folder)
        filtered_files = [f for f in files if filename_prefix in f]

        trial_nums = []
        for f in filtered_files:
            with open(os.path.join(args.output_folder, f), 'r') as f_handle:
                exp = json.load(f_handle)
                trial_nums.append(exp["trial_num"])

        remaining_trials = [i for i in range(args.repetitions) if i not in trial_nums]

        if len(remaining_trials) > 0:
            trial_num = remaining_trials[0] 
            output_filename = f'{filename_prefix}_{trial_num}.out'

    remaining_trials = MPI.COMM_WORLD.bcast(remaining_trials, root=0)
    
    if remaining_trials is not None and len(remaining_trials) == 0:
        if rank == 0:
            print("No trials left to perform!")
        exit(0)
    else:
        decompose(args, output_filename, trial_num)
