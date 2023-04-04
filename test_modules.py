from mpi4py import MPI
import numpy as np
import argparse
import gc
import cppimport
import cppimport.import_hook

def test_grid():
    from exafac.cpp_ext.py_module import Grid 

    proc_dims = np.array([2, 2, 2], dtype=np.int32)
    grid = Grid(proc_dims)

if __name__=='__main__':
    num_procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    parser = argparse.ArgumentParser()
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

    MPI.COMM_WORLD.Barrier()

    # Start running tests 
    test_grid()
