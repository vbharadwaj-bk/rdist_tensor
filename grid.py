import mpi4py
from mpi4py import MPI
import numpy as np

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

if __name__=='__main__':
    grid = Grid([3, 3, 3])

