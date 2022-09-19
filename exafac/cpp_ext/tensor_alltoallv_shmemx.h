#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <string>
#include <memory>
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

    SymArray() {
        ptr = nullptr;
        size = 0;
    }

    SymArray(uint64_t n_elements) { 
        size = n_elements;
        ptr = (T*) shmem_malloc(sizeof(T) * size);
        memset(ptr, 0x00, sizeof(T) * n_elements);
    }

    void reallocate(uint64_t n_elements) {
        size = n_elements;
        if(ptr == nullptr) {
            ptr = (T*) shmem_malloc(sizeof(T) * size);
        }
        else {
            ptr = (T*) shmem_realloc(ptr, sizeof(T) * size);
        }
    }

    constexpr T& operator[](std::size_t idx) {
        return ptr[idx];
    }

    void fill(T value) {
        std::fill(ptr, ptr + size, value);
    }

    void destroy() {
        if(ptr != nullptr) {
            shmem_free(ptr);
            ptr = nullptr;
        }
    }

    /*~SymArray() {
        destroy();
    }*/
};

/* TODO: Should fill out this class 
class TensorAlltoallv {
    virtual void execute_alltoallv(
        py::list recv_idx_py,
        py::list recv_values_py
    ) = 0; 
};
*/

template<typename IDX_T, typename VAL_T>
class SHMEMX_Alltoallv : public TensorAlltoallv {
    uint64_t proc_count;

    SymArray<Triple<IDX_T, VAL_T>> recv_buffer;
    SymArray<uint64_t> recv_counts;
    SymArray<uint64_t> recv_offsets;
    SymArray<int64_t> pSync;

public:
    int rank;
    uint64_t total_received_coords;
    SymArray<Triple<IDX_T, VAL_T>> send_buffer;
    SymArray<uint64_t> send_counts;
    SymArray<uint64_t> send_offsets;
    SymArray<uint64_t> running_offsets;

    py::function allocate_recv_buffers;
    SHMEMX_Alltoallv(py::function allocate_recv_buffers) 
    {
        shmem_init();
        this->allocate_recv_buffers = allocate_recv_buffers;
        proc_count = shmem_n_pes();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

        send_counts.reallocate(proc_count);
        send_offsets.reallocate(proc_count);
        recv_counts.reallocate(proc_count);
        recv_offsets.reallocate(proc_count);
        running_offsets.reallocate(proc_count);
        pSync.reallocate(_SHMEM_ALLTOALL_SYNC_SIZE);

        pSync.fill(_SHMEM_SYNC_VALUE); 
        shmem_barrier_all();
    }

    void destroy() { 
        send_counts.destroy();
        send_offsets.destroy();
        recv_counts.destroy();
        recv_offsets.destroy();
        running_offsets.destroy();
        pSync.destroy();
        send_buffer.destroy();
        recv_buffer.destroy();

        shmem_finalize();
    }

    // This assumes the send buffer, send counts, and
    // send offsets have all been filled
    void execute_alltoallv(py::list recv_idx_py,
                           py::list recv_values_py) {

        MPI_Alltoall(send_counts.ptr, 
                    1, MPI_UINT64_T, 
                    recv_counts.ptr, 
                    1, MPI_UINT64_T, 
                    MPI_COMM_WORLD);

        total_received_coords = 
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

        if(max_nnz_recv > recv_buffer.size) {
            recv_buffer.reallocate(max_nnz_recv);
        }

        for(uint i = 0; i < proc_count; i++) {
            send_counts[i] *= sizeof(Triple<IDX_T, VAL_T>);
            send_offsets[i] *= sizeof(Triple<IDX_T, VAL_T>);
            recv_offsets[i] *= sizeof(Triple<IDX_T, VAL_T>);
        }

        shmemx_alltoallv(   recv_buffer.ptr, 
                            recv_offsets.ptr, 
                            recv_counts.ptr,
                            send_buffer.ptr, 
                            send_offsets.ptr, 
                            send_counts.ptr,
                            0, 
                            0, 
                            proc_count,
                            pSync.ptr);

        shmem_barrier_all(); 

        allocate_recv_buffers(2, 
                total_received_coords, 
                recv_idx_py, 
                recv_values_py,
                "u32",
                "double" 
                );

        NumpyList<IDX_T> recv_idx(recv_idx_py);
        NumpyList<VAL_T> recv_values(recv_values_py);

        #pragma omp parallel for
        for(uint i = 0; i < total_received_coords; i++) {
            recv_idx.ptrs[0][i] = recv_buffer[i].row;
            recv_idx.ptrs[1][i] = recv_buffer[i].col;
            recv_values.ptrs[0][i] = recv_buffer[i].val;
        }
    }
};
