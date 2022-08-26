import mpi4py
from mpi4py import MPI
import numpy as np

from exafac.common import *

class Grid:
    def __init__(self, proc_count_dims):
        world_comm = MPI.COMM_WORLD
        # TODO: May want to perform manual rank reordering here with
        # MPI_Comm_split to order the grid for the best locality

        self.world_size = world_comm.Get_size()
        self.dim = len(proc_count_dims)
        self.comm = MPI.Intracomm(world_comm).Create_cart(dims=proc_count_dims, reorder=False)
        self.rank = self.comm.Get_rank()
        self.axesLengths = np.array(proc_count_dims, dtype=np.int)
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

    def get_prefix_array(self):
        lst = [one_const]
        for i in range(self.dim - 1):
            lst.append(lst[-1] * self.axesLengths[-1-i])

        lst.reverse()
        return np.array(lst, dtype=np.int)

    def test_prefix_array(self):
        prefix_array = self.get_prefix_array()
        if self.rank == 0:
            print(f"Prefix Array: {prefix_array}")

        for i in range(self.world_size):
            coords = self.comm.Get_coords(i)

            s = 0
            for j in range(self.dim):
                s += coords[j] * prefix_array[j] 

            if self.rank == 0: 
                print(f"True Rank: {i}, Coords: {coords}, Pfx. Rank: {s}")
                pass

class TensorGrid:
    def __init__(self, tensor_dims, grid=None):
        '''
        If grid is None, then we initialize the grid to the optimal 
        dimensions to match the supplied tensor dimensions.

        start_coords: A collection of arrays representing tick marks
        along the axes of the tensor.
        ''' 
        if grid is None:
            assert False
        else:
            self.tensor_dims = tensor_dims
            self.start_coords = []
            self.intervals = []
            self.grid = grid

            self.bound_starts = []
            self.bound_ends = []

            for i in range(self.grid.dim):
                dim = tensor_dims[i]
                proc_count = np.array([grid.axesLengths[i]], dtype=np.uint64)[0]
                interval = round_to_nearest_np_arr(dim, grid.world_size) // proc_count
                self.intervals.append(interval)

                coords = list(range(0, dim, interval))
                while len(coords) <= proc_count:
                    coords.append(dim)

                coords = np.array(coords, dtype=np.uint64)
                self.start_coords.append(coords)

                self.bound_starts.append(coords[grid.coords[i]])
                self.bound_ends.append(coords[grid.coords[i] + 1])


if __name__=='__main__':
    #grid = Grid([3, 3, 3])
    #tGrid = TensorGrid([5, 1, 20], grid=grid)
    pass