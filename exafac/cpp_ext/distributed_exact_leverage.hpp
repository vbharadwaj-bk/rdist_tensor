#pragma once

#include <iostream>
#include <random>
#include <mpi.h>

#include "common.h"
#include "distmat.hpp"
#include "alltoallv_revised.hpp"

using namespace std;

class ExactLeverageSampler {
public:
    Buffer<double> h;
    MPI_Comm comm;
    int world_size;

    ExactLeverageSampler(MPI_Comm world) 
        : comm(world)
         {
        MPI_Comm_size(comm, &world_size);
    }

    void test_alltoallv_efficiency() {
        uint64_t global_row_count = (65000 / world_size) * world_size;
        uint64_t local_row_count = global_row_count / world_size;
        uint64_t col_count = 300;

        h.initialize_to_shape({local_row_count, col_count});
        Buffer<int> destinations({local_row_count});
        Buffer<uint64_t> send_counts({(uint64_t) world_size});

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, world_size - 1);

        for(uint64_t i = 0; i < local_row_count; i++) {
            destinations[i] = dis(gen);
        }

        std::fill(send_counts(), send_counts(world_size), 0);
        for(uint64_t i = 0; i < local_row_count; i++) {

            // #pragma omp atomic
            send_counts[destinations[i]]++;
        }

        Buffer<double> recv_buffer;

        MPI_Barrier(comm);
        auto start = MPI_Wtime();
        alltoallv_matrix_rows(
                h,
                destinations, 
                send_counts, 
                recv_buffer,
                comm 
                );
        double elapsed = MPI_Wtime() - start; 

        cout << "Elapsed time: " << elapsed << endl;
    }
};

void benchmark_distributed_communication() {
    MPI_Comm comm = MPI_COMM_WORLD;
    ExactLeverageSampler sampler(comm);
    sampler.test_alltoallv_efficiency();
}
