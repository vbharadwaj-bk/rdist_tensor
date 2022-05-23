//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "common.h"

using namespace std;
namespace py = pybind11;

/*
 * This is currently a single-threaded CPU MTTKRP.
 * We are also assuming that the factor matrices are stored in
 * row-major order. 
 */
void sp_mttkrp(
        int mode,
        py::list factors_py,
        py::list idxs_py,
        py::array_t<double> values_py
        ) {

    NumpyList<double> factors(factors_py);
    NumpyList<unsigned long long> idxs(idxs_py);
    NumpyArray<double> values(values_py);

    int dim = factors.length;
    unsigned long long nnz = idxs.infos[0].shape[0];
    int col_count = factors.infos[0].shape[1];
    double* result_ptr = factors.ptrs[mode];

    // =======================================================
    // The code below actually implements the MTTKRP! 
    // =======================================================

    vector<double> accum_buffer(col_count, 1.0);
    double* accum_ptr = accum_buffer.data();

    for(unsigned long long i = 0; i < nnz; i++) {
        for(int k = 0; k < col_count; k++) {
            accum_ptr[k] = values.ptr[i];
        }

        for(int j = 0; j < dim; j++) {
            if(j != mode) {
                double* row_ptr = factors.ptrs[j] + (idxs.ptrs[j][i] * col_count);
                for(int k = 0; k < col_count; k++) {
                    accum_ptr[k] *= row_ptr[k]; 
                }
            }
        }

        unsigned long long out_row_idx = idxs.ptrs[mode][i];
        double* out_row_ptr = result_ptr + (out_row_idx * col_count);

        for(int k = 0; k < col_count; k++) {
            out_row_ptr[k] += accum_ptr[k]; 
        }
    }
}

void compute_tensor_values(
        py::list factors_py,
        py::list idxs_py,
        py::array_t<double> result_py) {
    NumpyList<double> factors(factors_py);
    NumpyList<unsigned long long> idxs(idxs_py);
    NumpyArray<double> result(result_py);

    unsigned long long nnz = idxs.infos[0].shape[0];
    unsigned long long cols = factors.infos[0].shape[1];

    vector<double*> base_ptrs;
    for(int j = 0; j < factors.length; j++) {
        base_ptrs.push_back(nullptr);
    }

    for(unsigned long long i = 0; i < nnz; i++) {
        for(int j = 0; j < factors.length; j++) {
            base_ptrs[j] = factors.ptrs[j] + idxs.ptrs[j][i] * cols;
        } 
        double value = 0.0;
        for(unsigned long long k = 0; k < cols; k++) {
            double coord_buffer = 1.0;
            for(int j = 0; j < factors.length; j++) {
                coord_buffer *= base_ptrs[j][k]; 
            }
            value += coord_buffer;
        }
        result.ptr[i] = value;
    }
}

PYBIND11_MODULE(tensor_kernels, m) {
    m.def("sp_mttkrp", &sp_mttkrp);
    m.def("compute_tensor_values", &compute_tensor_values);
}

/*
<%
setup_pybind11(cfg)
%>
*/