#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"
#include "als_optimizer.hpp"
#include "json.hpp"

using json = nlohmann::json;

using namespace std;

class __attribute__((visibility("hidden"))) TensorStationaryOpt0 : public ALS_Optimizer {
public:
    vector<Buffer<double>> gathered_factors;

    // Related to benchmarking...

    double leverage_computation_time, dense_gather_time, dense_reduce_time;
    double spmm_time, nonzeros_iterated;

    TensorStationaryOpt0(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
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

            factor.compute_leverage_scores();
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

    void execute_ALS_step(uint64_t mode_to_leave, uint64_t J) {
        uint64_t R = low_rank_tensor.rank;

        Buffer<uint32_t> samples({J, ground_truth.dim});
        Buffer<double> weights({J});

        gather_lk_samples_to_all(J, 
                mode_to_leave, 
                samples, 
                weights);

        Buffer<uint32_t> filtered_samples;
        Buffer<double> filtered_weights;

        // Now filter out all samples that don't belong to this processor
        Buffer<int> belongs_to_proc({J});
        Buffer<int> packed_offsets({J});
        std::fill(belongs_to_proc(), belongs_to_proc(J), 0);

        uint64_t local_sample_count = 0;

        #pragma omp parallel for reduction(+:local_sample_count)
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
            local_sample_count += belongs_to_proc[j]; 
        }

        // If necessary, change to parallel execution policy 
        std::exclusive_scan(belongs_to_proc(), belongs_to_proc(J), packed_offsets(), 0);

        filtered_samples.initialize_to_shape({local_sample_count, ground_truth.dim});
        filtered_weights.initialize_to_shape({local_sample_count});

        //#pragma omp parallel for
        for(uint64_t j = 0; j < J; j++) {
            if(belongs_to_proc[j] == 1) {
                for(uint64_t i = 0; i < ground_truth.dim; i++) {
                    uint32_t result = samples[j * ground_truth.dim + i];
                    result -= ground_truth.offsets[i];
                    filtered_samples[packed_offsets[j] * ground_truth.dim + i] = result;
                }
                filtered_weights[packed_offsets[j]] = exp(weights[j]);
            }
        }

        // END FILTERING -----------------------------------------

        Buffer<uint32_t> samples_dedup;
        Buffer<double> weights_dedup;

        deduplicate_design_matrix(
            filtered_samples,
            filtered_weights,
            mode_to_leave, 
            samples_dedup,
            weights_dedup);

        uint64_t sample_count_dedup = samples_dedup.shape[0];

        Buffer<double> design_matrix({sample_count_dedup, R});
        std::fill(design_matrix(), design_matrix(sample_count_dedup* R), 1.0);

        for(uint64_t i = 0; i < ground_truth.dim; i++) {
            if(i == mode_to_leave) {
                continue;
            }

            double *factor_data = gathered_factors[i]();

            #pragma omp parallel for
            for(uint64_t j = 0; j < sample_count_dedup; j++) {
                uint32_t idx = samples_dedup[j * ground_truth.dim + i];

                for(uint64_t r = 0; r < R; r++) {
                    design_matrix[j * R + r] *= factor_data[idx * R + r]; 
                }
            }
        }


        for(uint64_t j = 0; j < sample_count_dedup; j++) {
            for(uint64_t r = 0; r < R; r++) {
                design_matrix[j * R + r] *= sqrt(weights_dedup[j]);
            }
        }

        Buffer<double> design_gram({R, R});
        Buffer<double> design_gram_inv({R, R});
        compute_gram(design_matrix, design_gram);

        MPI_Allreduce(MPI_IN_PLACE, 
                    design_gram(), 
                    R * R, 
                    MPI_DOUBLE, 
                    MPI_SUM, 
                    grid.slices[mode_to_leave]);

        compute_pinv_square(design_gram, design_gram_inv, R);

        #pragma omp parallel for
        for(uint64_t j = 0; j < samples_dedup.shape[0]; j++) {
            for(uint64_t r = 0; r < R; r++) {
                design_matrix[j * R + r] *= sqrt(weights_dedup[j]);
            }
        }

        uint64_t output_buffer_rows = gathered_factors[mode_to_leave].shape[0];
        Buffer<double> mttkrp_res({output_buffer_rows, R});

        std::fill(mttkrp_res(), 
            mttkrp_res(output_buffer_rows * R), 
            0.0);

        auto t = start_clock();
        nonzeros_iterated += ground_truth.lookups[mode_to_leave]->execute_spmm(
            samples_dedup, 
            design_matrix,
            mttkrp_res
            );
        double elapsed = stop_clock_get_elapsed(t);
        MPI_Barrier(MPI_COMM_WORLD);

        spmm_time += elapsed; 

        DistMat1D &target_factor = low_rank_tensor.factors[mode_to_leave];
        Buffer<double> &target_factor_data = target_factor.data;
        uint64_t target_factor_rows = target_factor_data.shape[0];
        Buffer<double> temp_local({target_factor_rows, R}); 

        // Reduce_scatter_block the mttkrp_res buffer into temp_local 
        // across grid.slices[mode_to_leave]

        t = start_clock(); 
        MPI_Reduce_scatter_block(
            mttkrp_res(),
            temp_local(),
            target_factor_rows * R,
            MPI_DOUBLE,
            MPI_SUM,
            grid.slices[mode_to_leave]
        );
        dense_reduce_time += stop_clock_get_elapsed(t);

        cblas_dsymm(
            CblasRowMajor,
            CblasRight,
            CblasUpper,
            (uint32_t) target_factor_rows,
            (uint32_t) R,
            1.0,
            design_gram_inv(),
            R,
            temp_local(),
            R,
            0.0,
            target_factor_data(),
            R);

        target_factor.renormalize_columns(&(low_rank_tensor.sigma));

        t = start_clock();
        MPI_Allgather(
            target_factor_data(),
            target_factor_rows * R,
            MPI_DOUBLE,
            gathered_factors[mode_to_leave](),
            target_factor_rows * R,
            MPI_DOUBLE,
            grid.slices[mode_to_leave]
        );
        dense_gather_time += stop_clock_get_elapsed(t);

        target_factor.compute_leverage_scores();
    }
};