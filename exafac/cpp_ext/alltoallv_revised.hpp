#pragma once

#include <iostream>
#include <memory>
#include <algorithm>
#include <omp.h>
#include "common.h"

template<typename VAL_T>
void alltoallv_matrix_rows(
        Buffer<VAL_T> &send_buffer,
		Buffer<int> &processor_assignments,
		Buffer<uint64_t> &send_counts_input,
        unique_ptr<Buffer<VAL_T>> &recv_buffer,
        MPI_Comm &world
        ) {

    int rank, world_size_i;
    uint64_t world_size;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &world_size_i);
    world_size = world_size_i;

    uint64_t rows = send_buffer.shape[0];
    uint64_t cols;
    if(send_buffer.shape.size() == 2) {
        cols = send_buffer.shape[1];
    }
    else {
        cols = 1;
    }

    Buffer<uint64_t> send_counts({world_size});
    Buffer<uint64_t> send_offsets({world_size});
    Buffer<uint64_t> recv_counts({world_size});
    Buffer<uint64_t> recv_offsets({world_size});
    Buffer<uint64_t> running_offsets({world_size});

    std::copy(send_counts_input(), 
            send_counts_input(world_size), 
            send_counts());

    MPI_Alltoall(send_counts(), 
                1, MPI_UINT64_T, 
                recv_counts(), 
                1, MPI_UINT64_T, 
                world);

    uint64_t total_send_rows = 
				std::accumulate(send_counts(), send_counts(world_size), 0);
    uint64_t total_received_rows = 
				std::accumulate(recv_counts(), recv_counts(world_size), 0);
    recv_buffer.reset(new Buffer<VAL_T>({total_received_rows, cols}));

    std::exclusive_scan(send_counts(), send_counts(world_size), send_offsets(), 0);
    std::exclusive_scan(recv_counts(), recv_counts(world_size), recv_offsets(), 0);
    std::copy(send_offsets(), send_offsets(world_size), running_offsets()); 

    Buffer<VAL_T> pack_buffer({rows, cols});

    uint64_t* r_offset_ptr = running_offsets();

    /*
    * TO-DO: This parallel packing does not work because the
    * packing of the send buffer may not occur in the
    * same order as the value buffer. Need to fix eventually. 
    */

    //#pragma omp parallel for 
    for(uint64_t i = 0; i < rows; i++) {
        int owner = processor_assignments[i];
        uint64_t idx;

        #pragma omp atomic capture 
        idx = r_offset_ptr[owner]++;

        for(uint64_t j = 0; j < cols; j++) {
            pack_buffer[idx * cols + j] = send_buffer[i * cols + j];
        }
    }

    for(uint64_t i = 0; i < world_size; i++) {
        send_counts[i] *= cols;
        send_offsets[i] *= cols;
        recv_counts[i] *= cols;
        recv_offsets[i] *= cols;
    } 

    // Execute the AlltoAll operations

    // ===================================================
    // Explicit downcast: this errors if we have more than one
    // integer's worth of data to transfer. We should replace this 
    // with a more intelligent AlltoAll operation 

    Buffer<int> send_counts_dcast({world_size});
    Buffer<int> send_offsets_dcast({world_size});
    Buffer<int> recv_counts_dcast({world_size});
    Buffer<int> recv_offsets_dcast({world_size});

    if(total_send_rows * cols >= INT32_MAX || total_received_rows * cols >= INT32_MAX) {
        cout << "ERROR, ALL_TO_ALL BUFFER_SIZE EXCEEDED!" << endl;
        exit(1);
    }
    else {
        for(uint64_t i = 0; i < world_size; i++) {
            send_counts_dcast[i] = (int) send_counts[i];
            send_offsets_dcast[i] = (int) send_offsets[i];
            recv_counts_dcast[i] = (int) recv_counts[i];
            recv_offsets_dcast[i] = (int) recv_offsets[i];
        } 
    }

    MPI_Datatype mpi_dtype = get_MPI_dtype<VAL_T>();
    MPI_Alltoallv(pack_buffer(), 
                    send_counts_dcast(), 
                    send_offsets_dcast(), 
                    mpi_dtype,
                    (*recv_buffer)(),
                    recv_counts_dcast(), 
                    recv_offsets_dcast(), 
                    mpi_dtype,
                    world
                    );
}
