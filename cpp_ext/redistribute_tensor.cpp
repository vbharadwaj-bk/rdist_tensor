//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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

/*
 * This is a bad prefix sum function.
 */
void prefix_sum(vector<unsigned long long> &values, vector<unsigned long long> &offsets) {
    unsigned long long sum = 0;
    for(unsigned long long i = 0; i < values.size(); i++) {
        offsets.push_back(sum);
        sum += values[i];
    }
}

/*
 * Count up the nonzeros in preparation to allocate receive buffers. 
 *
 *  
 */
vector<unsigned long long> redistribute_nonzeros(
        py::array_t<unsigned long long> intervals, 
        py::list coords,
        py::array_t<double> values,
        unsigned long long proc_count, 
        py::array_t<int> prefix_mult) {
    // Count of nonzeros assigned to each processor
    vector<unsigned long long> send_counts(proc_count, 0);
    vector<unsigned long long> recv_counts(proc_count, 0);

    vector<unsigned long long> send_offsets;
    vector<unsigned long long> recv_offsets;

    vector<unsigned long long> running_offsets;

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
    
    info = values.request();
    double* value_ptr = static_cast<double*>(info.ptr);   

    vector<int> processor_assignments(nnz, -1);

    // TODO: Could parallelize using OpenMP if we want faster IO 
    for(unsigned long long i = 0; i < nnz; i++) {
        unsigned long long processor = 0;
        for(int j = 0; j < dim; j++) {
            processor += prefixes[j] * (buffer_ptrs[j][i] / interval_ptr[j]); 
        }
        send_counts[processor]++;
        processor_assignments[i] = processor;
    }

    MPI_Alltoall(send_counts.data(), 
                1, MPI_UNSIGNED_LONG_LONG, 
                recv_counts.data(), 
                1, MPI_UNSIGNED_LONG_LONG, 
                MPI_COMM_WORLD);

    unsigned long long total_received_coords = 
				std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    prefix_sum(recv_counts, recv_offsets);

    vector<vector<unsigned long long>> send_buffers_idx;
    vector<double> send_buffer_values(nnz, 0.0);
    // We will use numpy to allocate the receive buffers for us

    for(int i = 0; i < dim; i++) {
        send_buffers_idx.emplace_back(nnz, 0);
    }

    // Pack the send buffers
    prefix_sum(send_counts, send_offsets);
    running_offsets = send_offsets;

    for(unsigned long long i = 0; i < nnz; i++) {
        int owner = processor_assignments[i];
        unsigned long long idx;

        // #pragma omp atomic capture 
        idx = running_offsets[owner]++;

        for(int j = 0; j < dim; j++) {
            send_buffers_idx[j][idx] = buffer_ptrs[j][i];
        }
        send_buffer_values[idx] = value_ptr[idx]; 
    }

    // Execute the AlltoAll operations

    for(int j = 0; j < dim; j++) {
        MPI_Alltoallv(send_buffer_idx[j].data(), 
                        send_counts.data(), 
                        send_offsets.data(), 
                        MPI_UNSIGNED_LONG_LONG, 
                        TODO RECV BUFFER HERE, recv_counts.data(), recv_offsets.data(), 
                        MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD 
                        );
    }

    MPI_Alltoallv(send_buffer_values.data(), 
                    send_counts.data(), 
                    send_offsets.data(), 
                    MPI_DOUBLE, 
                    TODO RECV BUFFER HERE, recv_counts.data(), recv_offsets.data(), 
                    MPI_DOUBLE, MPI_COMM_WORLD 
                    );
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