//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>

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
        py::buffer_info info = arr1.request();
        idx_ptrs.push_back(static_cast<unsigned long long*>(info.ptr));

        py::array_t<double> arr2 = factors[i].cast<py::array_t<double>>();
        info = arr2.request();
        factor_ptrs.push_back(static_cast<double*>(info.ptr));
        
        if(first_element){
            first_element = false;
            nnz = arr1.shape[0];
            col_count = arr2.shape[1];
        }
    }

    py::buffer_info info = values.request();
    value_ptr = static_cast<double*>(info.ptr);

    info = result.request();
    result_ptr = static_cast<double*>(info.ptr);

    // =======================================================
    // The code below actually implements the MTTKRP! 
    // =======================================================
    vector<double> accum_buffer(col_count, 1.0);
    double* accum_ptr = accum_buffer.data();

    for(unsigned long long i = 0; i < nnz; i++) {
        for(int k = 0; k < col_count; k++) {
            accum_ptr[k] = value_ptr[i];
        }

        for(int j = 0; j < dim; j++) {
            if(j != mode) {
                double* row_ptr = factor_ptrs[j][i * col_count];
                for(int k = 0; k < col_count; k++) {
                    accum_ptr[k] *= row_ptr[k]; 
                }
            }
        }

        unsigned long long out_row_idx = idx_ptrs[mode][i];
        double* out_row_ptr = result_ptr[out_row_idx * col_count];

        for(int k = 0; k < col_count; k++) {
            out_row_ptr[k] += accum_ptr[k]; 
        }
    }
}

PYBIND11_MODULE(tensor_kernels, m) {
    m.def("sp_mttkrp", &sp_mttkrp);
}

/*
<%
setup_pybind11(cfg)
%>
*/