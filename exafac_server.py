from multiprocessing.connection import Client, Listener
import argparse
from rpc_utilities import *
import sys
from mpi4py import MPI

class Exafac_Server:
    def __init__(self, ctrl_hostname, ctrl_port, server_port):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        if self.rank == 0:
            hostname, _ = run_shell_cmd("hostname") 
            hostname = hostname.rstrip("\n")

            self.to_controller = Client((ctrl_hostname, ctrl_port))
            self.to_controller.send(hostname)
            self.listener = Listener((hostname, server_port))
            self.from_controller = self.listener.accept()

        self.comm.Barrier()

    def status_print(self, msg):
        if self.rank == 0:
            print(msg)

    def recv_command(self):
        if self.rank == 0:
            cmd = self.from_controller.recv() 
        else:
            cmd = None
        cmd = self.comm.bcast(cmd, root=0)
        return cmd

    def serve(self):
        while True:
            cmd = self.recv_command() 

            if cmd['type'] == 'initialize':
                self.initialize(cmd['payload'])
            elif cmd['type'] == 'terminate':
                self.status_print("Terminating exafac server...")
                break

    def initialize(self, args):
        if args is None:
            exit(1)

        # Let every process have a different random
        # seed based on its MPI rank; may be a better
        # way to initialize, though... 
        initialize_seed_generator(args.rs + self.rank)
        self.status_print("Seed generator Initialized...")

        grid_dimensions = [int(el) for el in args.grid.split(',')]
        self.status_print("Initializing Sparse Tensor...")
        ground_truth = DistSparseTensor(args.input, preprocessing=args.preprocessing)
        grid = Grid(grid_dimensions)
        tensor_grid = TensorGrid(ground_truth.max_idxs, grid=grid)
        ground_truth.random_permute()
        ground_truth.redistribute_nonzeros(tensor_grid)

        if args.samples is None:
            sample_counts = [None]
        else:
            sample_counts = args.samples.split(",")

        for sample_count in [el for el in sample_counts]: 
            for trank in [int(el) for el in args.trank.split(",")]:
                # gc.collect()

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
                else:
                    print(f"Error, invalid optimizer specified: '{args.op}'")
                    exit(1) 


                self.status_print(f"Exafac initialized...")

                #pre_info = None
                #if args.pre_optim > 0:
                #    pre_optim = ExactALS(ten_to_optimize, ground_truth)
                #    if grid.rank == 0:
                #        print("Starting pre-optimization...")
                #    pre_optim.fit(output_file=None, factor_file=None, 
                #        max_iterations=args.pre_optim,epoch_interval=1,
                #        early_stop=False)
                #    pre_info = pre_optim.info


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print("Loading C++ extensions and Python modules...")
        import exafac.cpp_ext.filter_nonzeros
        import exafac.cpp_ext.redistribute_tensor
        import exafac.cpp_ext.tensor_kernels
    comm.Barrier()

    from exafac.low_rank_tensor import *
    from exafac.grid import *
    from exafac.sparse_tensor import *
    from exafac.sampling import *

    from exafac.optim.tensor_stationary_opt1 import TensorStationaryOpt1
    from exafac.optim.accumulator_stationary_opt1 import AccumulatorStationaryOpt1
    from exafac.optim.exact_als import ExactALS

    server = Exafac_Server(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    server.serve()