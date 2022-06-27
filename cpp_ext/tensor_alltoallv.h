#pragma once

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
void prefix_sum(vector<uint64_t> &values, vector<uint64_t> &offsets) {
    uint64_t sum = 0;
    for(unsigned long long i = 0; i < values.size(); i++) {
        offsets.push_back(sum);
        sum += values[i];
    }
}

/*
 * Note: sendcounts could be computed from 
 * processor_assignments, but we require precomputation
 * for efficiency. 
 */
void tensor_alltoallv(
		int dim,
		uint64_t proc_count,
		uint64_t nnz,
        NumpyList<uint64_t> &coords,
        NumpyArray<double> &values,
		vector<int> &processor_assignments,
		vector<uint64_t> &send_counts,
        py::list &recv_idx_py,
        py::list &recv_values_py,
        py::function &allocate_recv_buffers 
        ) {
    vector<uint64_t> recv_counts(proc_count, 0);

    vector<uint64_t> send_offsets;
    vector<uint64_t> recv_offsets;
    vector<uint64_t> running_offsets;

    vector<vector<uint64_t>> send_idx;

    for(int i = 0; i < dim; i++) {
        send_idx.emplace_back(nnz, 0);
    }

    vector<double> send_values(nnz, 0.0);

    MPI_Alltoall(send_counts.data(), 
                1, MPI_UNSIGNED_LONG_LONG, 
                recv_counts.data(), 
                1, MPI_UNSIGNED_LONG_LONG, 
                MPI_COMM_WORLD);

    uint64_t total_received_coords = 
				std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    prefix_sum(recv_counts, recv_offsets);

    allocate_recv_buffers(dim, total_received_coords, recv_idx_py, recv_values_py);
    NumpyList<uint64_t> recv_idx(recv_idx_py);
    NumpyList<double> recv_values(recv_values_py);

    // Pack the send buffers
    prefix_sum(send_counts, send_offsets);
    running_offsets = send_offsets;

    for(uint64_t i = 0; i < nnz; i++) {
        int owner = processor_assignments[i];
        uint64_t idx;

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

    uint64_t total_send_coords = 
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