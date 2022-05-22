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
        py::list factors,
        py::list idxs,
        py::array_t<double> values,
        py::array_t<double> result
        ) {

    vector<double*> factor_ptrs;
    vector<unsigned long long*> idx_ptrs;
    double* value_ptr;
    double* result_ptr;

    int dim = py::len(factors);

    unsigned long long nnz;
    int col_count;
    bool first_element = true;

    for(int i = 0; i < dim; i++) {
        py::array_t<unsigned long long> arr1 = idxs[i].cast<py::array_t<unsigned long long>>();
        py::buffer_info info1 = arr1.request();
        idx_ptrs.push_back(static_cast<unsigned long long*>(info1.ptr));

        if(i != mode) {
            py::array_t<double> arr2 = factors[i].cast<py::array_t<double>>();
            py::buffer_info info2 = arr2.request();
            factor_ptrs.push_back(static_cast<double*>(info2.ptr));
            
            if(first_element) {
                first_element = false;
                nnz = info1.shape[0];
                col_count = info2.shape[1];
            }
        }
        else {
            factor_ptrs.push_back(nullptr);
        }
    }

    py::buffer_info info = values.request();
    value_ptr = static_cast<double*>(info.ptr);

    info = result.request();
    result_ptr = static_cast<double*>(info.ptr);

    // =======================================================
    // The code below actually implements the MTTKRP! 
    // =======================================================

    // TODO: Write a unit test for "sparsifying" a dense tensor.

    vector<double> accum_buffer(col_count, 1.0);
    double* accum_ptr = accum_buffer.data();

    // DEBUGGING!!
    //cout << "NNZ: " << nnz << endl;
    // END DEBUGGING!

    for(unsigned long long i = 0; i < nnz; i++) {
        for(int k = 0; k < col_count; k++) {
            accum_ptr[k] = value_ptr[i];
        }

        for(int j = 0; j < dim; j++) {
            if(j != mode) {
                double* row_ptr = factor_ptrs[j] + (idx_ptrs[j][i] * col_count);
                for(int k = 0; k < col_count; k++) {
                    accum_ptr[k] *= row_ptr[k]; 
                }
            }
        }

        unsigned long long out_row_idx = idx_ptrs[mode][i];
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