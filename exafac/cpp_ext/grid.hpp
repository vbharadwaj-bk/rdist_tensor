#pragma once

#include <iostream>
#include <vector>
#include <mpi.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.h"

using namespace std;
namespace py = pybind11;

class __attribute__((visibility("hidden"))) Grid {
public:
    MPI_Comm overall_world;
    int world_size;

    Buffer<int> proc_dims;
    int dim;

    MPI_Comm world;   // This world is a Cartesian topology
    int rank;
    Buffer<int> coords;

    vector<MPI_Comm> axes;
    vector<MPI_Comm> slices;

    vector<vector<int>> row_positions;
    vector<vector<int>> row_order_to_procs;

    Grid(py::array_t<int> proc_dims_py) :
        proc_dims(proc_dims_py, true),
        dim((int) proc_dims.shape[0]),
        coords({(uint64_t) dim})
        {
        overall_world = MPI_COMM_WORLD;

        Buffer<int> periodic({(uint64_t) dim});
        std::fill(periodic(), periodic(dim), 0);

        // Gives MPI the flexibility to reorder
        int reorder = 1; 

        MPI_Cart_create(
            overall_world,
            dim,
            proc_dims(),
            periodic(),
            reorder,
            &world
        );

        MPI_Comm_rank(world, &rank);
        MPI_Comm_size(world, &world_size);

        int proc_count = std::accumulate(proc_dims(), proc_dims(dim), 1, 
                std::multiplies<int>());

        if(proc_count != world_size) {
            throw std::runtime_error("The number of MPI ranks does not match "
                "the number of processes specified by the grid.");
        }

        // These two arrays are dummy variables that are not used 
        Buffer<int> dims({(uint64_t) dim});
        Buffer<int> periods({(uint64_t) dim});
        MPI_Cart_get(world, dim, dims(), periods(), coords());

        for(int i = 0; i < dim; i++) {
            Buffer<int> remain_dims({(uint64_t) dim});

            std::fill(remain_dims(), remain_dims(dim), 0);
            remain_dims[i] = 1;
            axes.push_back(0);
            MPI_Cart_sub(world, remain_dims(), &(axes[i]));

            std::fill(remain_dims(), remain_dims(dim), 1);
            remain_dims[i] = 0;
            slices.push_back(0);
            MPI_Cart_sub(world, remain_dims(), &(slices[i]));
        }

        for(int slice_dim = 0; slice_dim < dim; slice_dim++) {
            int slice_rank, slice_size;
            MPI_Comm_rank(slices[slice_dim], &slice_rank); 
            MPI_Comm_size(slices[slice_dim], &slice_size);
            int row_position = slice_rank + coords[slice_dim] * slice_size;

            row_positions.emplace_back(world_size, 0);
            MPI_Allgather(&row_position,
                1,
                MPI_INT,
                row_positions[slice_dim].data(),
                1,
                MPI_INT,
                world 
                );

            row_order_to_procs.emplace_back(world_size, 0);

            for(int i = 0; i < world_size; i++) {
                row_order_to_procs[slice_dim][row_positions[slice_dim][i]] = i;
            }
        }
    }

    int get_dimension() {
        return dim;
    }

    void get_prefix_array(Buffer<int> &prefix_array) {
        prefix_array[0] = 1;
        for(int i = 1; i < dim; i++) {
            prefix_array[i] = 
                prefix_array[i-1] * proc_dims[dim - i];
        }
        std::reverse(prefix_array(), prefix_array(dim));
    }
};

class __attribute__((visibility("hidden"))) TensorGrid {
/*
* Partitions a Cartesian domain among a grid of processors. 
*/
public:
    Buffer<int> tensor_dims;
    Grid &grid;

    vector<int> padded_row_counts;

    vector<vector<int>> start_coords;
    vector<int> bound_starts, bound_ends;

    int dim;
    int rank;

    TensorGrid(py::array_t<int> tensor_d, Grid &g)
    :
    tensor_dims(tensor_d),
    grid(g),
    dim(g.dim),
    rank(g.rank)
    {
        for(int i = 0; i < grid.dim; i++) {
            int dim = tensor_dims[i];
            int proc_count = grid.proc_dims[i];

            int padded_row_count = (int) round_to_nearest_integer((uint64_t) dim, 
                    (uint64_t) grid.world_size) / proc_count; 
            padded_row_counts.push_back(padded_row_count);

            start_coords.emplace_back();
            for(int j = 0; j < dim; j += padded_row_count) {
                start_coords[i].push_back(j);
            }

            while(start_coords[i].size() < (uint64_t) proc_count) {
                start_coords[i].push_back(dim);
            }

            bound_starts.push_back(start_coords[i][grid.coords[i]]);
            bound_ends.push_back(start_coords[i][grid.coords[i] + 1]);
        }
    }
};


