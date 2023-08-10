#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"
#include "alltoallv_revised.hpp"
#include "json.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) AccumulatorStationaryOpt0 : public ALS_Optimizer{
public:
    TensorGrid &tensor_grid;
    Grid &grid;

    uint64_t dim;

    vector<Buffer<uint32_t>> indices;
    vector<Buffer<double>> values;
    vector<unique_ptr<SortIdxLookup<uint32_t, double>>> lookups;

    AccumulatorStationaryOpt0(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ALS_Optimizer(ground_truth, low_rank_tensor),
    tensor_grid(ground_truth.tensor_grid),
    grid(ground_truth.tensor_grid.grid),
    dim(ground_truth.dim)
    {
    }

    void initialize_ground_truth_for_als() {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        // uint64_t R = low_rank_tensor.rank;

        for(uint64_t i = 0; i < dim; i++) {
            indices.emplace_back();
            values.emplace_back();

            // Allgather factors into buffers and compute gram matrices
            DistMat1D &factor = low_rank_tensor.factors[i];

            //uint64_t row_count = tensor_grid.padded_row_counts[i] * world_size; 
            uint32_t row_start = factor.row_position * factor.padded_rows;
            uint32_t row_end = (factor.row_position + 1) * factor.padded_rows;

            factor.compute_leverage_scores();

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
                uint64_t target_proc = ground_truth.indices[j * dim + i] / factor.padded_rows; 
                //uint64_t target_proc = 0;

                send_counts_local[target_proc]++;
                processor_assignments[j] = target_proc; 
            }

            for(uint64_t j = 0; j < proc_count; j++) {
                #pragma omp atomic 
                send_counts[j] += send_counts_local[j];
            }
    }

            alltoallv_matrix_rows(
                ground_truth.indices,
                processor_assignments,
                send_counts,
                indices[i],
                tensor_grid.grid.world
            );

            alltoallv_matrix_rows(
                ground_truth.values,
                processor_assignments,
                send_counts,
                values[i],
                tensor_grid.grid.world
            );

            #pragma omp parallel for
            for(uint64_t j = 0; j < indices[i].shape[0]; j++) {
                indices[i][j * dim + i] -= row_start; 
            }

            lookups.emplace_back(
                make_unique<SortIdxLookup<uint32_t, double>>(
                    dim, i, indices[i](), values[i](), indices[i].shape[0]
                ));
        }
    }

    void execute_ALS_rounds(uint64_t num_rounds, uint64_t J, uint32_t epoch_interval) {
        cout << "NOT IMPLEMENTED!" << endl;
        exit(1);
    }
};
