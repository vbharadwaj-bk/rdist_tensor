#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"
#include "als_optimizer.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) ExactALS : public ALS_Optimizer {
public:

    vector<Buffer<double>> gathered_factors;
    vector<Buffer<double>> gram_matrices; 

    ExactALS(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ALS_Optimizer(ground_truth, low_rank_tensor)
    {
        uint64_t R = low_rank_tensor.rank; 
        for(uint64_t i = 0; i < dim; i++) {
            int world_size;
            MPI_Comm_size(grid.slices[i], &world_size);
            uint64_t row_count = tensor_grid.padded_row_counts[i] * world_size;

            gathered_factors.emplace_back(
                Buffer<double>({row_count, low_rank_tensor.rank})
            );

            gram_matrices.emplace_back(
                Buffer<double>({low_rank_tensor.rank, low_rank_tensor.rank})
            );

            // Allgather factors into buffers and compute gram matrices
            DistMat1D &factor = low_rank_tensor.factors[i];
            Buffer<double> &factor_data = factor.data;

            MPI_Allgather(
                factor_data(),
                factor_data.shape[0] * R,
                MPI_DOUBLE,
                gathered_factors[i](),
                factor_data.shape[0] * R,
                MPI_DOUBLE,
                grid.slices[i]
            );

            factor.compute_gram_matrix(gram_matrices[i]);
        }
    }

    void initialize_ground_truth_for_als() {
        ground_truth.check_tensor_invariants();
        ground_truth.redistribute_to_grid(tensor_grid);
        ground_truth.check_tensor_invariants();

        uint64_t dim = ground_truth.dim;

        for(uint64_t i = 0; i < dim; i++) {
            ground_truth.offsets[i] = tensor_grid.start_coords[i][tensor_grid.grid.coords[i]];
        }

        #pragma omp parallel for
        for(uint64_t i = 0; i < ground_truth.indices.shape[0]; i++) {
            for(uint64_t j = 0; j < dim; j++) {
                ground_truth.indices[i * dim + j] -= ground_truth.offsets[j];
            }
        }

        for(uint64_t i = 0; i < dim; i++) {
            ground_truth.lookups.emplace_back(
                make_unique<SortIdxLookup<uint32_t, double>>(
                    dim, i, ground_truth.indices(), ground_truth.values(), ground_truth.indices.shape[0]
                )); 
        }
    }

    // The sample count J is ignored for exact ALS. 
    void execute_ALS_step(uint64_t mode_to_leave, uint64_t J) {
        uint64_t R = low_rank_tensor.rank; 

        Buffer<double> gram_product({R, R});
        Buffer<double> gram_product_inv({R, R});

        chain_had_prod(gram_matrices, gram_product, mode_to_leave);
        compute_pinv_square(gram_product, gram_product_inv, R);

        uint64_t output_buffer_rows = gathered_factors[mode_to_leave].shape[0];
        Buffer<double> mttkrp_res({output_buffer_rows, R});

        std::fill(mttkrp_res(), 
            mttkrp_res(output_buffer_rows * R), 
            0.0);

        ground_truth.lookups[mode_to_leave]->execute_exact_mttkrp( 
            gathered_factors,
            mttkrp_res 
        );

        DistMat1D &target_factor = low_rank_tensor.factors[mode_to_leave];
        Buffer<double> &target_factor_data = target_factor.data;
        uint64_t target_factor_rows = target_factor_data.shape[0];
        Buffer<double> temp_local({target_factor_rows, R}); 

        MPI_Reduce_scatter_block(
            mttkrp_res(),
            temp_local(),
            target_factor_rows * R,
            MPI_DOUBLE,
            MPI_SUM,
            grid.slices[mode_to_leave]
        );             

        #pragma omp parallel
        {
            parallel_dsymm(gram_product_inv, temp_local, target_factor_data); 
        }

        target_factor.renormalize_columns(&(low_rank_tensor.sigma));

        MPI_Allgather(
            target_factor_data(),
            target_factor_rows * R,
            MPI_DOUBLE,
            gathered_factors[mode_to_leave](),
            target_factor_rows * R,
            MPI_DOUBLE,
            grid.slices[mode_to_leave]
        );

        target_factor.compute_gram_matrix(gram_matrices[mode_to_leave]);
    }
    ~ExactALS() {
    }
};