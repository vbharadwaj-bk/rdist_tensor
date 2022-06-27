//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "common.h"
#include 

using namespace std;
namespace py = pybind11;

/*
 * Count up the nonzeros in preparation to allocate receive buffers. 
 * 
 */
void redistribute_nonzeros(
        py::array_t<unsigned long long> intervals_py, 
        py::list coords_py,
        py::array_t<double> values_py,
        unsigned long long proc_count, 
        py::array_t<int> prefix_mult_py,
        py::list recv_idx_py,
        py::list recv_values_py,
        py::function allocate_recv_buffers 
        ) {

    // Unpack the parameters 
    NumpyArray<unsigned long long> intervals(intervals_py); 
    NumpyList<unsigned long long> coords(coords_py); 
    NumpyArray<double> values(values_py); 
    NumpyArray<int> prefix_mult(prefix_mult_py);

    unsigned long long nnz = coords.infos[0].shape[0];
    int dim = prefix_mult.info.shape[0];

    // Initialize AlltoAll data structures 
    vector<uint64_t> send_counts(proc_count, 0);
    vector<int> processor_assignments(nnz, -1);

    // TODO: Could parallelize using OpenMP if we want faster IO 
    for(uint64_t i = 0; i < nnz; i++) {
        uint64_t processor = 0;
        for(int j = 0; j < dim; j++) {
            processor += prefix_mult.ptr[j] * (coords.ptrs[j][i] / intervals.ptr[j]); 
        }
        send_counts[processor]++;
        processor_assignments[i] = processor;
    }

    redistribute_nonzeros(
		dim,
		proc_count,
		nnz,
        coords,
        values,
		processor_assignments,
		send_counts,
        recv_idx_py,
        recv_values_py,
        allocate_recv_buffers 
    );
}

PYBIND11_MODULE(redistribute_tensor, m) {
    m.def("redistribute_nonzeros", &redistribute_nonzeros);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-fopenmp']
%>
*/