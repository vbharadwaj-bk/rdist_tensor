//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
namespace py = pybind11;

typedef chrono::time_point<std::chrono::steady_clock> my_timer_t; 

my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}


unsigned long long sum_all_elements(py::list my_list) {
    unsigned long long total_sum = 0;

    for(py::handle obj : my_list) {
        py::array_t<unsigned long long> idxs = obj.cast<py::array_t<unsigned long long>>();

        py::buffer_info info = idxs.request();
        auto ptr = static_cast<unsigned long long*>(info.ptr);

        for(int i = 0; i < info.shape[0]; i++) {
            total_sum += ptr[i];
        }
    }

    return total_sum;
}

//auto t = start_clock();
//double time = stop_clock_get_elapsed(t);

/*
 * Count up the nonzeros in preparation to allocate receive buffers. 
 *
 *  
 */
vector<unsigned long long> redistribute_nonzeros(
        py::array_t<unsigned long long> intervals, 
        py::list coords, int proc_count, 
        py::array_t<int> prefix_mult) {
    // Count of nonzeros assigned to each processor
    vector<unsigned long long> proc_counts(proc_count, 0);

    // Unpack the list of coordinate buffers into pointers 
    vector<unsigned long long*> buffer_ptrs;

    unsigned long long nnz;
    bool first_element = true;
    for(py::handle obj : coords) { 

        py::array_t<unsigned long long> idxs = obj.cast<py::array_t<unsigned long long>>();

        py::buffer_info info = idxs.request();
        buffer_ptrs.push_back(static_cast<unsigned long long*>(info.ptr));
        
        if(first_element) {
            first_element = false;
            nnz = info.shape[0];
        }
    }

    py::buffer_info info = prefix_mult.request();
    int* prefixes = static_cast<int*>(info.ptr);
    int dim = info.shape[0];

    info = intervals.request();
    unsigned long long* interval_ptr = static_cast<unsigned long long*>(info.ptr);

    // TODO: Could parallelize using OpenMP if we want faster IO 
    for(unsigned long long i = 0; i < nnz; i++) {
        unsigned long long processor = 0;
        for(int j = 0; j < dim; j++) {
            processor += prefixes[j] * (buffer_ptrs[j][i] / interval_ptr[j]); 
        }
        proc_counts[processor]++;
    }

    return proc_counts;
}

PYBIND11_MODULE(redistribute_tensor, m) {
    m.def("sum_all_elements", &sum_all_elements);
    m.def("redistribute_nonzeros", &redistribute_nonzeros);
}

/*
<%
setup_pybind11(cfg)
%>
*/