#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"
#include "json.hpp"

class __attribute__((visibility("hidden"))) AccumulatorStationaryOpt0 {
public:
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;
    TensorGrid &tensor_grid;
    Grid &grid;

    uint64_t dim;

    vector<Buffer<uint32_t>> indices;
    vector<Buffer<double>> values;

    AccumulatorStationaryOpt0(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ground_truth(ground_truth),
    low_rank_tensor(low_rank_tensor),
    tensor_grid(ground_truth.tensor_grid),
    grid(ground_truth.tensor_grid.grid),
    dim(ground_truth.dim)
    {
        uint64_t R = low_rank_tensor.rank; 
        for(uint64_t i = 0; i < dim; i++) {
            int world_size;
            MPI_Comm_size(grid.slices[i], &world_size);
            uint64_t row_count = tensor_grid.padded_row_counts[i] * world_size; 

            // Allgather factors into buffers and compute gram matrices
            DistMat1D &factor = low_rank_tensor.factors[i];
            factor.compute_leverage_scores();

            uint32_t row_start = factor.row_position * factor.padded_rows;
            uint32_t row_end = factor.row_position + 1 * factor.padded_rows;

            cout << "row_start: " << row_start << endl;
            cout << "row_end: " << row_end << endl;   
        }
    }

    void execute_ALS_rounds(uint64_t num_rounds) {
        // TODO: Implement!
    }

};
