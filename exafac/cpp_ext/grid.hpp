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
    MPI_Comm world;   // This world is a Cartesian topology
    int rank;
    int world_size;

    Buffer<int> proc_dims;
    int dim;

    Grid(py::array_t<int> proc_dims_py) :
        proc_dims(proc_dims_py) {
        dim = (int) proc_dims.shape[0];
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
    }
};
