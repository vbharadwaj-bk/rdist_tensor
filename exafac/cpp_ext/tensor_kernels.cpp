//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include "common.h"

#include <shmem.h>
#include <shmemx.h>

using namespace std;
namespace py = pybind11;

/*
 * This is currently a single-threaded CPU MTTKRP.
 * We are also assuming that the factor matrices are stored in
 * row-major order. 
 */
template<typename IDX_T, typename VAL_T>
void sp_mttkrp(
        int mode,
        py::list factors_py,
        py::list idxs_py,
        py::array_t<double> values_py
        ) {

    NumpyList<double> factors(factors_py);
    NumpyList<IDX_T> idxs(idxs_py);
    NumpyArray<VAL_T> values(values_py);

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

        IDX_T out_row_idx = idxs.ptrs[mode][i];
        double* out_row_ptr = result_ptr + (out_row_idx * col_count);

        for(int k = 0; k < col_count; k++) {
            #pragma omp atomic
            out_row_ptr[k] += accum_ptr[k]; 
        }
    }
    }
}

template<typename IDX_T>
void compute_tensor_values(
        py::list factors_py,
        py::array_t<double> singular_values_py,
        py::list idxs_py,
        py::array_t<double> result_py) {
    NumpyList<double> factors(factors_py);
    NumpyArray<double> singular_values(singular_values_py);
    NumpyList<IDX_T> idxs(idxs_py);
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

template<typename IDX_T>
void inflate_samples_multiply(
    py::array_t<IDX_T> samples_py,
    py::array_t<int64_t> counts_py,
    py::array_t<double> probs_py,
    py::array_t<double> rows_py,
    py::array_t<IDX_T> inflated_samples_py,
    py::array_t<double> weight_prods_py,
    py::array_t<int64_t> permutation_py,
    py::array_t<int64_t> sample_ids_py
) {
    NumpyArray<IDX_T> samples(samples_py);
    NumpyArray<int64_t> counts(counts_py);
    NumpyArray<double> probs(probs_py);
    NumpyArray<double> rows(rows_py);
    NumpyArray<IDX_T> inflated_samples(inflated_samples_py);
    NumpyArray<double> weight_prods(weight_prods_py);
    NumpyArray<int64_t> permutation(permutation_py);
    NumpyArray<int64_t> sample_ids(sample_ids_py);

    uint64_t num_unique_samples = samples.info.shape[0];
    vector<int64_t> sample_offsets(num_unique_samples + 1, 0); 
    prefix_sum_ptr(counts.ptr, sample_offsets.data(), num_unique_samples); 

    sample_offsets[num_unique_samples] =  
        sample_offsets[num_unique_samples - 1]
        + counts.ptr[num_unique_samples - 1];

    assert(inflated_sample_count == sample_offsets[num_unique_samples]);
    int64_t* ptr = sample_offsets.data();

    for(uint64_t i = 0; i < num_unique_samples; i++) {
        for(int64_t j = ptr[i]; j < ptr[i+1]; j++) {
            int64_t perm_loc = permutation.ptr[j];
            sample_ids.ptr[perm_loc] = i;
            inflated_samples.ptr[perm_loc] = samples.ptr[i];
            weight_prods.ptr[perm_loc] -= 0.5 * log(probs.ptr[i]);
        }
    }
}

template<typename IDX_T, typename VAL_T>
void spmm(
        py::array_t<double> lhs_py,
        py::array_t<IDX_T> rhs_rows_py,
        py::array_t<IDX_T> rhs_cols_py,
        py::array_t<VAL_T> rhs_values_py,
        py::array_t<double> result_py
        ) {

    NumpyArray<double> lhs(lhs_py);
    NumpyArray<IDX_T> rhs_rows(rhs_rows_py);
    NumpyArray<IDX_T> rhs_cols(rhs_cols_py);
    NumpyArray<VAL_T> rhs_values(rhs_values_py);
    NumpyArray<double> result(result_py);
    int r = result.info.shape[1];
    uint64_t nnz = rhs_rows.info.shape[0];

    COOSparse<IDX_T, VAL_T> sampled_rhs_wrapped;

    // TODO: This next step could be way more efficient...
    // but I just want to finish this... 
    sampled_rhs_wrapped.rows.assign(rhs_rows.ptr, rhs_rows.ptr + nnz); 
    sampled_rhs_wrapped.cols.assign(rhs_cols.ptr, rhs_cols.ptr + nnz); 
    sampled_rhs_wrapped.values.assign(rhs_values.ptr, rhs_values.ptr + nnz);
    sampled_rhs_wrapped.cpu_spmm(lhs.ptr, result.ptr, r);
}

template<typename IDX_T, typename VAL_T>
void spmm_compressed(
        py::list inflated_sample_ids_py,
        py::list mode_rows_py,
        py::array_t<double> weights_py,
        py::array_t<IDX_T> rhs_rows_py,
        py::array_t<IDX_T> rhs_cols_py,
        py::array_t<VAL_T> rhs_values_py,
        py::array_t<double> result_py
        ) {
    NumpyList<int64_t> inflated_sample_ids(inflated_sample_ids_py);
    NumpyList<double> mode_rows(mode_rows_py);
    NumpyArray<double> weights(weights_py);
    NumpyArray<IDX_T> rhs_rows(rhs_rows_py);
    NumpyArray<IDX_T> rhs_cols(rhs_cols_py);
    NumpyArray<VAL_T> rhs_values(rhs_values_py);
    NumpyArray<double> result(result_py);
    int r = result.info.shape[1];
    uint64_t nnz = rhs_rows.info.shape[0];
    int dim_m1 = mode_rows.length; 

    IDX_T* row_ptr = rhs_rows.ptr;
    IDX_T* col_ptr = rhs_cols.ptr;
    VAL_T* val_ptr = rhs_values.ptr;

#pragma omp parallel 
{
    vector<double> accumulator_row(r, 1.0);
    double* accum_ptr = accumulator_row.data(); 

    #pragma omp for
    for(uint64_t i = 0; i < nnz; i++) {
        // We perform a transpose here
        IDX_T row = col_ptr[i];
        IDX_T col = row_ptr[i];
        VAL_T value = val_ptr[i];

        std::fill(accum_ptr, accum_ptr + r, value * weights.ptr[col]);
        for(int k = 0; k < dim_m1; k++) {
            double* row_ptr =
                mode_rows.ptrs[k] + (inflated_sample_ids.ptrs[k][col] * r);
            for(int j = 0; j < r; j++) {
                accum_ptr[j] *= row_ptr[j]; 
            } 
        }

        for(int j = 0; j < r; j++) {
            #pragma omp atomic 
            result.ptr[row * r + j] += accum_ptr[j];
        }        
    }
}

}

PYBIND11_MODULE(tensor_kernels, m) {
    //m.def("sampled_mttkrp", &sampled_mttkrp);
    m.def("spmm_u32_double", &spmm<uint32_t, double>);
    m.def("sp_mttkrp_u32_double", &sp_mttkrp<uint32_t, double>); 
    m.def("spmm_compressed_u32_double", &spmm_compressed<uint32_t, double>);
    m.def("compute_tensor_values_u32", &compute_tensor_values<uint32_t>);
    m.def("inflate_samples_multiply_u32", &inflate_samples_multiply<uint32_t>);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-fopenmp', '-O3']
cfg['extra_link_args'] = ['-openmp', '-O3']
cfg['dependencies'] = ['common.h']
%>
*/

