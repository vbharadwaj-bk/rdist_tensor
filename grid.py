import mpi4py
from mpi4py import MPI
import numpy as np

from common import *

class Grid:
    def __init__(self, proc_count_dims):
        world_comm = MPI.COMM_WORLD
        # TODO: May want to perform manual rank reordering here with
        # MPI_Comm_split to order the grid for the best locality

        self.world_size = world_comm.Get_size()
        self.dim = len(proc_count_dims)
        self.comm = MPI.Intracomm(world_comm).Create_cart(dims=proc_count_dims, reorder=False)
        self.rank = self.comm.Get_rank()
        self.axesLengths = proc_count_dims
        self.coords = self.comm.Get_coords(self.rank)

        self.axes = []
        self.slices = []
        for i in range(self.dim):
            remain_dims_axis = [False] * self.dim
            remain_dims_axis[i] = True

            remain_dims_slice = [True] * self.dim
            remain_dims_slice[i] = False

            self.axes.append(self.comm.Sub(remain_dims_axis))
            self.slices.append(self.comm.Sub(remain_dims_slice))

class TensorGrid:
    def __init__(self, tensor_dims, grid=None):
        '''
        If grid is None, then we initialize the grid to the optimal 
        dimensions to match the supplied tensor dimensions 
        ''' 
        if grid is None:
            assert False
        else:
            self.start_coords = []
            self.intervals = []

            for i in range(len(tensor_dims)):
                dim = tensor_dims[i]
                proc_count = grid.axesLengths[i]
                interval = round_to_nearest(dim, proc_count) // proc_count
                self.intervals.append(interval)

                coords = list(range(0, dim, interval))
                coords.append(dim)
                self.start_coords.append(coords)

        if grid.rank == 0:
            print(self.start_coords)

if __name__=='__main__':
    grid = Grid([3, 3, 3])
    tGrid = TensorGrid([5, 5, 5], grid=grid)