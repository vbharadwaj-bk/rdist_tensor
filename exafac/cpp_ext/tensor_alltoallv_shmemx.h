#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <string>
#include <shmem.h> 
#include <shmemx.h> 
#include "common.h"

using namespace std;
namespace py = pybind11;

template<typename T>
class SymArray {
public:
    T* ptr;
    uint64_t size;
    SymArray(uint64_t n_elements) { 
        size = n_elements;
        ptr = (T*) shmem_malloc(sizeof(T) * size);
        memset(ptr, 0x00, sizeof(T) * n_elements);
    }

    constexpr T& operator[](std::size_t idx) {
        return ptr[idx];
    }

    void fill(T value) {
        std::fill(ptr, ptr + size, value);
    }

    ~SymArray() {
        shmem_free(ptr);
    }
};

template<typename IDX_T, typename VAL_T>
void tensor_alltoallv_shmemx(
		int dim,
		uint64_t proc_count,
		uint64_t nnz,
        NumpyList<IDX_T> &coords,
        NumpyArray<VAL_T> &values,
		vector<int> &processor_assignments,
		vector<uint64_t> &send_counts,
        py::list &recv_idx_py,
        py::list &recv_values_py,
        py::function &allocate_recv_buffers 
        ) {

    auto start = start_clock();

    SymArray<int> send_counts_dcast(proc_count);
    for(uint i = 0; i < proc_count; i++) {
        send_counts_dcast.ptr[i] = (int) send_counts[i];
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    MPI_Datatype MPI_IDX_T, MPI_VAL_T;
    DEFINE_MPI_DATATYPES();

    string idx_t_str, val_t_str;
    if(MPI_IDX_T == MPI_UINT64_T) 
        idx_t_str = "u64";
    else 
        idx_t_str = "u32"; 

    if(MPI_VAL_T == MPI_DOUBLE) 
        val_t_str = "double";
    else 
        val_t_str = "float"; 

    SymArray<int> recv_counts(proc_count);
    SymArray<int> send_offsets(proc_count);
    SymArray<int> recv_offsets(proc_count);
    SymArray<int> running_offsets(proc_count);

    // Retrieve the maximum number of nonzeros owned
    // by any single processor

    uint64_t max_nnz_send;
    MPI_Allreduce(&nnz, 
            &max_nnz_send, 
            1, 
            MPI_UINT64_T, 
            MPI_MAX,
            MPI_COMM_WORLD 
            );

    SymArray<Triple<IDX_T, VAL_T>> send_buffer(max_nnz_send); 

    //SymArray<IDX_T> send_idx_rows(max_nnz_send); 
    //SymArray<IDX_T> send_idx_cols(max_nnz_send); 
    //SymArray<VAL_T> send_values(max_nnz_send); 

    MPI_Alltoall(send_counts_dcast.ptr, 
                1, MPI_INT, 
                recv_counts.ptr, 
                1, MPI_INT, 
                MPI_COMM_WORLD);

    uint64_t total_received_coords = 
				std::accumulate(recv_counts.ptr, recv_counts.ptr + proc_count, 0);

    prefix_sum_ptr(recv_counts.ptr, recv_offsets.ptr, proc_count);

    uint64_t max_nnz_recv;
    MPI_Allreduce(&total_received_coords, 
            &max_nnz_recv, 
            1, 
            MPI_UINT64_T, 
            MPI_MAX,
            MPI_COMM_WORLD 
            );

    allocate_recv_buffers(dim, 
            total_received_coords, 
            recv_idx_py, 
            recv_values_py,
            idx_t_str,
            val_t_str 
            );

    NumpyList<IDX_T> recv_idx(recv_idx_py);
    NumpyList<VAL_T> recv_values(recv_values_py);

    //SymArray<IDX_T> recv_idx_rows(max_nnz_recv);
    //SymArray<IDX_T> recv_idx_cols(max_nnz_recv);
    //SymArray<VAL_T> recv_idx_vals(max_nnz_recv);

    SymArray<Triple<IDX_T, VAL_T>> recv_buffer(max_nnz_recv); 

    // Pack the send buffers
    prefix_sum_ptr(send_counts_dcast.ptr, send_offsets.ptr, proc_count);
    memcpy(running_offsets.ptr, send_offsets.ptr, sizeof(int) * proc_count);
    //running_offsets = send_offsets;

    for(uint64_t i = 0; i < nnz; i++) {
        int owner = processor_assignments[i];
        uint64_t idx;

        // #pragma omp atomic capture 
        idx = running_offsets.ptr[owner]++;

        send_buffer[idx].row = coords.ptrs[0][i];
        send_buffer[idx].col = coords.ptrs[1][i];
        send_buffer[idx].val = values.ptr[i];

        //send_idx_rows.ptr[idx] = coords.ptrs[0][i];
        //send_idx_cols.ptr[idx] = coords.ptrs[1][i];
        //send_values.ptr[idx] = values.ptr[i]; 
    }

    // Execute the AlltoAll operations

    // ===================================================
    // Explicit downcast: this errors if we have more than one
    // integer's worth of data to transfer. We should replace this 
    // with a more intelligent AlltoAll operation 

    uint64_t total_send_coords = 
				std::accumulate(send_counts.begin(), send_counts.end(), 0);
    
    SymArray<int64_t> pSync(_SHMEM_ALLTOALL_SYNC_SIZE);
    pSync.fill(_SHMEM_SYNC_VALUE);

    SymArray<uint64_t> recv_counts_sym(proc_count);
    SymArray<uint64_t> send_offsets_sym(proc_count);
    SymArray<uint64_t> recv_offsets_sym(proc_count);
    SymArray<uint64_t> running_offsets_sym(proc_count);
    SymArray<uint64_t> send_counts_sym(proc_count);

    for(int i = 0; i < proc_count; i++) {
        size_t dt_size = sizeof(Triple<IDX_T, VAL_T>);
        recv_counts_sym.ptr[i] = recv_counts.ptr[i] * dt_size; 
        send_offsets_sym.ptr[i] = send_offsets.ptr[i] * dt_size; 
        recv_offsets_sym.ptr[i] = recv_offsets.ptr[i] * dt_size; 
        send_counts_sym.ptr[i] = send_counts_dcast.ptr[i] * dt_size; 
    }

    shmem_barrier_all();

    shmemx_alltoallv(   recv_buffer.ptr, 
                        recv_offsets_sym.ptr, 
                        recv_counts_sym.ptr,
                        send_buffer.ptr, 
                        send_offsets_sym.ptr, 
                        send_counts_sym.ptr,
                        0, 
                        0, 
                        shmem_n_pes(),
                        pSync.ptr);

    /*MPI_Alltoallv(send_idx_rows.ptr, 
                    send_counts_dcast.ptr, 
                    send_offsets.ptr, 
                    MPI_IDX_T, 
                    recv_idx_rows.ptr, 
                    recv_counts.ptr, 
                    recv_offsets.ptr, 
                    MPI_IDX_T, 
                    MPI_COMM_WORLD 
                    );*/

    shmem_barrier_all();

    /*
    MPI_Alltoallv(send_idx_cols.ptr, 
                    send_counts_dcast.ptr, 
                    send_offsets.ptr, 
                    MPI_IDX_T, 
                    recv_idx_cols.ptr, 
                    recv_counts.ptr, 
                    recv_offsets.ptr, 
                    MPI_IDX_T, 
                    MPI_COMM_WORLD 
                    );
    
    MPI_Alltoallv(send_values.ptr,
                    send_counts_dcast.ptr, 
                    send_offsets.ptr,
                    MPI_VAL_T, 
                    recv_idx_vals.ptr, 
                    recv_counts.ptr, 
                    recv_offsets.ptr, 
                    MPI_VAL_T, 
                    MPI_COMM_WORLD 
                    );
    */

    for(int i = 0; i < total_received_coords; i++) {
        recv_idx.ptrs[0][i] = recv_buffer[i].row;
        recv_idx.ptrs[1][i] = recv_buffer[i].col;
        recv_values.ptrs[0][i] = recv_buffer[i].val;
    }

    //memcpy(recv_idx.ptrs[0], recv_idx_rows.ptr, sizeof(IDX_T) * total_received_coords);
    //memcpy(recv_idx.ptrs[1], recv_idx_cols.ptr, sizeof(IDX_T) * total_received_coords);
    //memcpy(recv_values.ptrs[0], recv_idx_vals.ptr, sizeof(VAL_T) * total_received_coords);

    double elapsed = stop_clock_get_elapsed(start);

    if(rank == 0) {
        cout << elapsed << endl;
    }

    //recv_counts.free_memory();
    //send_offsets.free_memory();
    //recv_offsets.free_memory();
    //running_offsets.free_memory();
    //send_counts_dcast.free_memory();
    //send_values.free_memory();

    //send_idx_rows.free_memory(); 
    //send_idx_cols.free_memory();

}
