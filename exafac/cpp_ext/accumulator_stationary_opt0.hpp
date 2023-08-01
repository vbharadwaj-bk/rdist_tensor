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
    vector<unique_ptr<SortIdxLookup<uint32_t, double>>> lookups;

    AccumulatorStationaryOpt0(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ground_truth(ground_truth),
    low_rank_tensor(low_rank_tensor),
    tensor_grid(ground_truth.tensor_grid),
    grid(ground_truth.tensor_grid.grid),
    dim(ground_truth.dim)
    {


        int world_size;
        MPI_Comm_size(grid.slices[i], &world_size);
        // uint64_t R = low_rank_tensor.rank;

        for(uint64_t i = 0; i < dim; i++) {
            uint64_t row_count = tensor_grid.padded_row_counts[i] * world_size; 

            indices.emplace_back();
            values.emplace_back();

            // Allgather factors into buffers and compute gram matrices
            DistMat1D &factor = low_rank_tensor.factors[i];
            factor.compute_leverage_scores();

            uint32_t row_start = factor.row_position * factor.padded_rows;
            uint32_t row_end = (factor.row_position + 1) * factor.padded_rows;

            uint64_t nnz = ground_truth.indices.shape[0];
            uint64_t proc_count = tensor_grid.grid.world_size;

            Buffer<int> prefix({(uint64_t) tensor_grid.dim}); 
            tensor_grid.grid.get_prefix_array(prefix);

            Buffer<uint64_t> send_counts({proc_count});
            std::fill(send_counts(), send_counts(proc_count), 0);
            Buffer<int> processor_assignments({nnz}); 

            #pragma omp parallel
    {
            vector<uint64_t> send_counts_local(proc_count, 0);

            #pragma omp for
            for(uint64_t j = 0; j < nnz; j++) {
                uint64_t target_proc = indices[j * dim + i] 
 
                send_counts_local[target_proc]++;
                processor_assignments[i] = target_proc; 
            }

            for(uint64_t i = 0; i < proc_count; i++) {
                #pragma omp atomic 
                send_counts[i] += send_counts_local[i];
            }
    }

            Buffer<uint32_t> recv_idxs;
            Buffer<double> recv_values;
            alltoallv_matrix_rows(
                indices,
                processor_assignments,
                send_counts,
                recv_idxs,
                tensor_grid.grid.world
            );
            alltoallv_matrix_rows(
                values,
                processor_assignments,
                send_counts,
                recv_values,
                tensor_grid.grid.world
            );

            indices.steal_resources(recv_idxs);
            values.steal_resources(recv_values);

        }
    }

    void execute_ALS_rounds(uint64_t num_rounds) {
        // TODO: Implement!
    }

};
