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
    NumpyList<uint64_t> idxs(idxs_py);
    NumpyArray<double> values(values_py);

    int dim = factors.length;
    uint64_t nnz = idxs.infos[0].shape[0];
    int col_count = factors.infos[0].shape[1];
    double* result_ptr = factors.ptrs[mode];

    // =======================================================
    // The code below actually implements the MTTKRP! 
    // =======================================================
    
    #pragma omp parallel
    {
        vector<double> accum_buffer(col_count, 1.0);
        double* accum_ptr = accum_buffer.data();

        #pragma omp for
        for(uint64_t i = 0; i < nnz; i++) {
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

            uint64_t out_row_idx = idxs.ptrs[mode][i];
            double* out_row_ptr = result_ptr + (out_row_idx * col_count);

            for(int k = 0; k < col_count; k++) {
                #pragma omp atomic update
                out_row_ptr[k] += accum_ptr[k]; 
            }
        }
    } 
}

void compute_tensor_values(
        py::list factors_py,
        py::array_t<double> singular_values_py,
        py::list idxs_py,
        py::array_t<double> result_py) {
    NumpyList<double> factors(factors_py);
    NumpyArray<double> singular_values(singular_values_py);
    NumpyList<uint64_t> idxs(idxs_py);
    NumpyArray<double> result(result_py);

    uint64_t nnz = idxs.infos[0].shape[0];
    uint64_t cols = factors.infos[0].shape[1];

    #pragma omp parallel
    {
        vector<double*> base_ptrs;
        for(int j = 0; j < factors.length; j++) {
            base_ptrs.push_back(nullptr);
        }
        
        #pragma omp for
        for(uint64_t i = 0; i < nnz; i++) {
            for(int j = 0; j < factors.length; j++) {
                base_ptrs[j] = factors.ptrs[j] + idxs.ptrs[j][i] * cols;
            } 
            double value = 0.0;
            for(uint64_t k = 0; k < cols; k++) {
                double coord_buffer = singular_values.ptr[k];
                for(int j = 0; j < factors.length; j++) {
                    coord_buffer *= base_ptrs[j][k]; 
                }
                value += coord_buffer;
            }
            result.ptr[i] = value;
        }
    }
}

void sampled_mttkrp(
        int mode,
        py::list factors_py,
        py::list krp_sample_idxs_py,
        py::array_t<double> sampled_lhs_py,
        COOSparse &sampled_rhs,
        py::array_t<double> weights_py
) {
    NumpyList<double> factors(factors_py);
    NumpyList<uint64_t> krp_samples(krp_sample_idxs_py);
    NumpyArray<double> lhs(sampled_lhs_py);
    NumpyArray<double> weights(weights_py);

    int dim = factors.length;
    int num_samples = krp_samples.infos[0].shape[0];
    int r = factors.infos[0].shape[1];
    double* result_ptr = factors.ptrs[mode];

    // Assemble the LHS using Hadamard products
    #pragma omp parallel for
    for(int i = 0; i < num_samples; i++) {
        for(int j = 0; j < r; j++) {
            lhs.ptr[i * r + j] = weights.ptr[i];
        }

        for(int k = 0; k < dim; k++) {
            if(k < mode) {
                for(int j = 0; j < r; j++) {
                    lhs.ptr[i * r + j] *= factors.ptrs[k][krp_samples.ptrs[k][i] * r + j];
                }
            }
            if(k > mode) {
                for(int j = 0; j < r; j++) {
                    lhs.ptr[i * r + j] *= factors.ptrs[k][krp_samples.ptrs[k-1][i] * r + j];
                }
            }
        }
    }
    sampled_rhs.cpu_spmm(lhs.ptr, result_ptr, r);
}

void sampled_mttkrp_with_lhs_assembled(
        py::array_t<double> lhs_py,
        py::array_t<uint64_t> rhs_rows_py,
        py::array_t<uint64_t> rhs_cols_py,
        py::array_t<double> rhs_values_py,
        py::array_t<double> result_py
        ) {

    NumpyArray<double> lhs(lhs_py);
    NumpyArray<uint64_t> rhs_rows(rhs_rows_py);
    NumpyArray<uint64_t> rhs_cols(rhs_cols_py);
    NumpyArray<double> rhs_values(rhs_values_py);
    NumpyArray<double> result(result_py);
    int r = result.info.shape[1];
    uint64_t nnz = rhs_rows.info.shape[0];

    COOSparse sampled_rhs_wrapped;

    // TODO: This next step could be way more efficient...
    // but I just want to finish this... 
    sampled_rhs_wrapped.rows.assign(rhs_rows.ptr, rhs_rows.ptr + nnz); 
    sampled_rhs_wrapped.cols.assign(rhs_cols.ptr, rhs_cols.ptr + nnz); 
    sampled_rhs_wrapped.values.assign(rhs_values.ptr, rhs_values.ptr + nnz);
    sampled_rhs_wrapped.print_contents();
    sampled_rhs_wrapped.cpu_spmm(lhs.ptr, result.ptr, r);
}


PYBIND11_MODULE(tensor_kernels, m) {
    m.def("sp_mttkrp", &sp_mttkrp);
    m.def("sampled_mttkrp", &sampled_mttkrp);
    m.def("sampled_mttkrp_with_lhs_assembled", &sampled_mttkrp_with_lhs_assembled);
    m.def("compute_tensor_values", &compute_tensor_values);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-fopenmp']
%>
*/