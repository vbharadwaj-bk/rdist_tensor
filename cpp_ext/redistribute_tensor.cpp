//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "common.h"

using namespace std;
namespace py = pybind11;

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
    vector<unsigned long long> send_counts(proc_count, 0);
    vector<unsigned long long> recv_counts(proc_count, 0);

    vector<unsigned long long> send_offsets;
    vector<unsigned long long> recv_offsets;
    vector<unsigned long long> running_offsets;

    vector<int> processor_assignments(nnz, -1);

    // TODO: Could parallelize using OpenMP if we want faster IO 
    for(unsigned long long i = 0; i < nnz; i++) {
        unsigned long long processor = 0;
        for(int j = 0; j < dim; j++) {
            processor += prefix_mult.ptr[j] * (coords.ptrs[j][i] / intervals.ptr[j]); 
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

    vector<vector<unsigned long long>> send_idx;
    vector<double> send_values(nnz, 0.0);

    allocate_recv_buffers(dim, total_received_coords, recv_idx_py, recv_values_py);
    NumpyList<unsigned long long> recv_idx(recv_idx_py);
    NumpyList<double> recv_values(recv_values_py);

    for(int i = 0; i < dim; i++) {
        send_idx.emplace_back(nnz, 0);
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
            send_idx[j][idx] = coords.ptrs[j][i];
        }
        send_values[idx] = values.ptr[i]; 
    }

    // Execute the AlltoAll operations

    // ===================================================
    // Explicit downcast: this errors if we have more than one
    // integer's worth of data to transfer. We should replace this 
    // with a more intelligent AlltoAll operation 

    unsigned long long total_send_coords = 
				std::accumulate(send_counts.begin(), send_counts.end(), 0);

    vector<int> send_counts_dcast;
    vector<int> send_offsets_dcast;
    vector<int> recv_counts_dcast;
    vector<int> recv_offsets_dcast;

    if(total_send_coords >= INT32_MAX || total_received_coords >= INT32_MAX) {
        cout << "ERROR, ALL_TO_ALL BUFFER_SIZE EXCEEDED!" << endl;
        exit(1);
    }
    else {
        for(uint i = 0; i < proc_count; i++) {
            send_counts_dcast.push_back((int) send_counts[i]);
            send_offsets_dcast.push_back((int) send_offsets[i]);

            recv_counts_dcast.push_back((int) recv_counts[i]);
            recv_offsets_dcast.push_back((int) recv_offsets[i]);
        } 
    }

    // ===================================================

    for(int j = 0; j < dim; j++) {
        MPI_Alltoallv(send_idx[j].data(), 
                        send_counts_dcast.data(), 
                        send_offsets_dcast.data(), 
                        MPI_UNSIGNED_LONG_LONG, 
                        recv_idx.ptrs[j], 
                        recv_counts_dcast.data(), 
                        recv_offsets_dcast.data(), 
                        MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD 
                        );
    }
    
    MPI_Alltoallv(send_values.data(), 
                    send_counts_dcast.data(), 
                    send_offsets_dcast.data(), 
                    MPI_DOUBLE, 
                    recv_values.ptrs[0], 
                    recv_counts_dcast.data(), 
                    recv_offsets_dcast.data(), 
                    MPI_DOUBLE, MPI_COMM_WORLD 
                    );
}

PYBIND11_MODULE(redistribute_tensor, m) {
    m.def("redistribute_nonzeros", &redistribute_nonzeros);
}

/*
<%
setup_pybind11(cfg)
%>
*/