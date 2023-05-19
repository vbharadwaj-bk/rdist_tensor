#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) ExactALS {
public:
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;
    vector<Buffer<double>> gathered_factor_buffers;
    ExactALS(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ground_truth(ground_truth),
    low_rank_tensor(low_rank_tensor)
    {
        for(int i = 0; i < ground_truth.dim; i++) {
            low_rank_tensor.factors[i].initialize_deterministic();]    
        }

        for(int i = 0; i < ground_truth.dim; i++) {
            int world_size;
            MPI_Comm_size(ground_truth.grid.slices[i], &world_size);

            uint64_t row_count = ground_truth.tensor_grid.padded_row_counts[i] * world_size;

            factors.emplace_back(
                Buffer<double>({row_count, low_rank_tensor.rank})
            );
        }
    }

    void execute_ALS_step(uint64_t mode_to_leave) {
        for(int i = 0; i < ground_truth.dim; i++) {
            // Allgather the local data of low_rank_tensor.factors[i]
            // into the gathered factor buffers
            MPI_Allgather(
                low_rank_tensor.factors[i].data(),
                low_rank_tensor.factors[i].shape[0],
                MPI_DOUBLE,
                gathered_factor_buffers[i](),
                low_rank_tensor.factors[i].shape[0],
                MPI_DOUBLE,
                ground_truth.grid.slices[i]
            );
        }


    }

};