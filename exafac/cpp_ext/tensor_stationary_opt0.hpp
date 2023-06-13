#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) TensorStationaryOpt0 {
public:
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;
    TensorGrid &tensor_grid;
    Grid &grid;
    uint64_t dim;

    vector<Buffer<double>> gathered_factors;
    vector<Buffer<double>> gram_matrices; 

    TensorStationaryOpt0(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
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

            gathered_factors.emplace_back(
                Buffer<double>({row_count, low_rank_tensor.rank})
            );

            gram_matrices.emplace_back(
                Buffer<double>({low_rank_tensor.rank, low_rank_tensor.rank})
            );


            // Allgather factors into buffers and compute gram matrices
            DistMat1D &factor = low_rank_tensor.factors[i];
            Buffer<double> &factor_data = *(factor.data);

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
            factor.compute_leverage_scores();
        }
    }

    void execute_ALS_step(uint64_t mode_to_leave, uint64_t J) {
        uint64_t R = low_rank_tensor.rank; 

        Buffer<double> gram_product({R, R});
        Buffer<double> gram_product_inv({R, R});

        chain_had_prod(gram_matrices, gram_product, mode_to_leave);
        compute_pinv_square(gram_product, gram_product_inv, R);

        Buffer<uint32_t> samples({J, ground_truth.dim});
        Buffer<double> weights({J});

        std::fill(weights(), weights(J), 1.0);
        Consistent_Multistream_RNG global_rng(MPI_COMM_WORLD);

        // Collect all samples and randomly permute along each mode 
        for(uint64_t i = 0; i < ground_truth.dim; i++) {
            if(i == mode_to_leave) {
                continue;
            }

            Buffer<uint32_t> sample_idxs;
            Buffer<double> sample_weights;
            low_rank_tensor.factors[0].draw_leverage_score_samples(J, sample_idxs, sample_weights);
        
            Buffer<uint32_t> rand_perm({J});
            std::iota(rand_perm(), rand_perm(J), 0);
            std::shuffle(rand_perm(), rand_perm(J), global_rng.par_gen[0]);

            apply_permutation(rand_perm, sample_idxs);
            apply_permutation(rand_perm, sample_weights);

            #pragma omp parallel for
            for(uint64_t j = 0; j < J; j++) {
                samples[j * ground_truth.dim + i] = sample_idxs[j];
                weights[j] *= sample_weights[j]; // TODO: Need to fix the weighting 
            }
        }

        // At some point, should deduplicate samples 

        // Now filter out all samples that don't belong to this processor
        Buffer<int> belongs_to_proc({J});
        std::fill(belongs_to_proc(), belongs_to_proc(J), 0);

        for(uint64_t j = 0; j < J; j++) {
            bool within_bounds = true;
            for(uint64_t i = 0; i < ground_truth.dim; i++) {
                if(i == mode_to_leave) {
                    continue;
                }

                int idx = (int) samples[j * ground_truth.dim + i];
                bool coord_within_bounds = idx >= tensor_grid.bound_starts[i] && idx < tensor_grid.bound_ends[i];

                within_bounds = within_bounds && coord_within_bounds;
            }
            belongs_to_proc[j] = within_bounds ? 1 : 0;
        }

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
        Buffer<double> &target_factor_data = *(target_factor.data);
        uint64_t target_factor_rows = target_factor_data.shape[0];
        Buffer<double> temp_local({target_factor_rows, R}); 

        // Reduce_scatter_block the mttkrp_res buffer into temp_local 
        // across grid.slices[mode_to_leave] 
        MPI_Reduce_scatter_block(
            mttkrp_res(),
            temp_local(),
            target_factor_rows * R,
            MPI_DOUBLE,
            MPI_SUM,
            grid.slices[mode_to_leave]
        );             

        cblas_dsymm(
            CblasRowMajor,
            CblasRight,
            CblasUpper,
            (uint32_t) target_factor_rows,
            (uint32_t) R,
            1.0,
            gram_product_inv(),
            R,
            temp_local(),
            R,
            0.0,
            target_factor_data(),
            R);

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
        target_factor.compute_leverage_scores();
    }

    void execute_ALS_rounds(uint64_t num_rounds, uint64_t J) {
        for(uint64_t round = 0; round < num_rounds; round++) {
            if(grid.rank == 0) {
                cout << "Starting ALS round " << (round + 1) << endl; 
            }

            for(int i = 0; i < grid.dim; i++) {
                execute_ALS_step(i, J);
            } 
        }
    }

    double compute_exact_fit() {
        return ground_truth.compute_exact_fit(low_rank_tensor);
    }
};