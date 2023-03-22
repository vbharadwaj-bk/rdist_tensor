//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "common.h"
#include "tensor_alltoallv.h"

using namespace std;
namespace py = pybind11;

/*
 * Count up the nonzeros in preparation to allocate receive buffers.  
 */
template<typename IDX_T, typename VAL_T>
void redistribute_nonzeros(
        py::array_t<uint64_t> intervals_py, 
        py::list coords_py,
        py::array_t<VAL_T> values_py,
        uint64_t proc_count, 
        py::array_t<int> prefix_mult_py,
        py::list recv_idx_py,
        py::list recv_values_py,
        py::function allocate_recv_buffers 
        ) {

    // Unpack the parameters 
    NumpyArray<uint64_t> intervals(intervals_py); 
    NumpyList<IDX_T> coords(coords_py); 
    NumpyArray<VAL_T> values(values_py); 
    NumpyArray<int> prefix_mult(prefix_mult_py);

    uint64_t nnz = coords.infos[0].shape[0];
    int dim = prefix_mult.info.shape[0];

    // Initialize AlltoAll data structures 
    vector<uint64_t> send_counts(proc_count, 0);
    vector<int> processor_assignments(nnz, -1);

    cout << "Starting redistribute counter..." << endl;

    #pragma omp parallel for
    for(uint64_t i = 0; i < nnz; i++) {
        uint64_t processor = 0;
        for(int j = 0; j < dim; j++) {
            processor += prefix_mult.ptr[j] * (coords.ptrs[j][i] / intervals.ptr[j]); 
        }

        #pragma omp atomic 
        send_counts[processor]++;
        processor_assignments[i] = processor;
    }  

    cout << "Ended..." << endl;

    tensor_alltoallv(
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
    m.def("redistribute_nonzeros_u32_double", &redistribute_nonzeros<uint32_t, double>);
    m.def("redistribute_nonzeros_u64_double", &redistribute_nonzeros<uint64_t, double>);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-fopenmp', '-O3']
cfg['extra_link_args'] = ['-openmp', '-O3']
cfg['dependencies'] = ['common.h', 'tensor_alltoallv.h']
%>
*/