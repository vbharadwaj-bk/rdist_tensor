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
        proc_dims(proc_dims_py),
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

        // These two arrays are dummy variables that are not used 
        Buffer<int> dims({(uint64_t) dim});
        Buffer<int> periods({(uint64_t) dim});
        MPI_Cart_get(world, dim, dims(), periods(), coords());

        //for(int i = 0; i < dim; i++) {
        //    cout << coords[i] << endl;
        //}

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

            cout << "[";
            for(int i = 0; i < row_positions[slice_dim].size(); i++) {
                cout << row_positions[slice_dim][i] << " ";
            }
            cout << "]" << endl;
        }
    }
};
