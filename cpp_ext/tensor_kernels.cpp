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
            //out_row_ptr[k] = factors.ptrs[0][idxs.ptrs[0][i] * col_count + k];
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

void sampled_mttkrp(
        int mode,
        py::list factors_py,
        py::list krp_sample_idxs_py,
        COOSparse &sampled_rhs,
        py::array_t<double> weights_py
) {
    NumpyList<double> factors(factors_py);
    NumpyList<unsigned long long> krp_samples(krp_sample_idxs_py);
    NumpyArray<double> weights(weights_py);

    int dim = factors.length;
    int num_samples = krp_samples.infos[0].shape[0];
    int r = factors.infos[0].shape[1];
    double* result_ptr = factors.ptrs[mode];

    vector<double> lhs(num_samples * r);

    // Assemble the LHS using Hadamard products 
    for(int i = 0; i < num_samples; i++) {
        for(int j = 0; j < r; j++) {
            lhs[i * r + j] = weights.ptr[i];
        }

        for(int k = 0; k < dim; k++) {
            if(k < mode) {
                for(int j = 0; j < r; j++) {
                    lhs[i * r + j] *= factors.ptrs[k][krp_samples.ptrs[k][i] * r + j];
                }
            }
            if(k > mode) {
                for(int j = 0; j < r; j++) {
                    lhs[i * r + j] *= factors.ptrs[k][krp_samples.ptrs[k-1][i] * r + j];
                }
            }
        }
    }
    sampled_rhs.cpu_spmm(lhs.data(), result_ptr, r);
}

PYBIND11_MODULE(tensor_kernels, m) {
    m.def("sp_mttkrp", &sp_mttkrp);
    m.def("sampled_mttkrp", &sampled_mttkrp);
    m.def("compute_tensor_values", &compute_tensor_values);
}

/*
<%
setup_pybind11(cfg)
%>
*/