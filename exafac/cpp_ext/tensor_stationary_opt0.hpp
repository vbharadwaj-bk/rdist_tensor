#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"
#include "json.hpp"

using json = nlohmann::json;

using namespace std;

class __attribute__((visibility("hidden"))) TensorStationaryOpt0 {
public:
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;
    TensorGrid &tensor_grid;
    Grid &grid;

    json stats;

    uint64_t dim;
    vector<Buffer<double>> gathered_factors;

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

            factor.compute_leverage_scores();
        }
    }

    void execute_ALS_step(uint64_t mode_to_leave, uint64_t J) {
        uint64_t R = low_rank_tensor.rank; 

        Buffer<uint32_t> samples({J, ground_truth.dim});
        Buffer<double> weights({J});

        std::fill(samples(), samples(J * ground_truth.dim), 0);
        std::fill(weights(), weights(J), 0.0 - log((double) J));
        Consistent_Multistream_RNG global_rng(MPI_COMM_WORLD);

        // Collect all samples and randomly permute along each mode 
        for(uint64_t i = 0; i < ground_truth.dim; i++) {
            if(i == mode_to_leave) {
                continue;
            }

            Buffer<uint32_t> sample_idxs;
            Buffer<double> log_weights;
            low_rank_tensor.factors[i].draw_leverage_score_samples(J, sample_idxs, log_weights);

            Buffer<uint32_t> rand_perm({J});
            std::iota(rand_perm(), rand_perm(J), 0);
            std::shuffle(rand_perm(), rand_perm(J), global_rng.par_gen[0]);

            apply_permutation(rand_perm, sample_idxs);
            apply_permutation(rand_perm, log_weights);

            #pragma omp parallel for
            for(uint64_t j = 0; j < J; j++) {
                samples[j * ground_truth.dim + i] = sample_idxs[j];
                weights[j] += log_weights[j]; 
            }
        }

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

        Buffer<uint32_t> filtered_samples({local_sample_count, ground_truth.dim});
        Buffer<double> filtered_weights({local_sample_count});

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

        Buffer<double> design_matrix({local_sample_count, R});
        std::fill(design_matrix(), design_matrix(local_sample_count * R), 1.0);

        #pragma omp parallel
{
        for(uint64_t i = 0; i < ground_truth.dim; i++) {
            if(i == mode_to_leave) {
                continue;
            }

            double *factor_data = gathered_factors[i]();

            #pragma omp for
            for(uint64_t j = 0; j < local_sample_count; j++) {
                uint32_t idx = filtered_samples[j * ground_truth.dim + i];

                for(uint64_t r = 0; r < R; r++) {
                    design_matrix[j * R + r] *= factor_data[idx * R + r]; 
                }
            }
        }

        #pragma omp for 
        for(uint64_t j = 0; j < local_sample_count; j++) {
            for(uint64_t r = 0; r < R; r++) {
                design_matrix[j * R + r] *= sqrt(filtered_weights[j]); 
            }
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
        for(uint64_t j = 0; j < local_sample_count; j++) {
            for(uint64_t r = 0; r < R; r++) {
                design_matrix[j * R + r] *= sqrt(filtered_weights[j]); 
            }
        }

        uint64_t output_buffer_rows = gathered_factors[mode_to_leave].shape[0];
        Buffer<double> mttkrp_res({output_buffer_rows, R});

        std::fill(mttkrp_res(), 
            mttkrp_res(output_buffer_rows * R), 
            0.0);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t = start_clock();

        uint64_t nonzeros_iterated = ground_truth.lookups[mode_to_leave]->execute_spmm(
            filtered_samples, 
            design_matrix,
            mttkrp_res
            );

        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = stop_clock_get_elapsed(t);

        //cout << "Rank " << grid.rank << " nonzeros iterated: " 
        // << nonzeros_iterated << endl;
        //if(grid.rank == 0) {
        //    cout << "SpMM time: " << elapsed << endl;
        //}

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
            design_gram_inv(),
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

        target_factor.compute_leverage_scores();
    }

    void execute_ALS_rounds(uint64_t num_rounds, uint64_t J, uint32_t epoch_interval) {
        double als_total_time = 0.0;
        double fit_computation_time = 0.0;

        for(uint64_t round = 1; round <= num_rounds; round++) {
            if(grid.rank == 0) {
                cout << "Starting ALS round " << (round) << endl; 
            }

            auto start = start_clock();
            for(int i = 0; i < grid.dim; i++) {
                execute_ALS_step(i, J);
            }
            als_total_time += stop_clock_get_elapsed(start);

            if((round % epoch_interval) == 0) {
                start = start_clock();
                double exact_fit = compute_exact_fit();
                fit_computation_time += stop_clock_get_elapsed(start);

                if(grid.rank == 0) {
                    cout << "Exact fit after " << round << " rounds: " << exact_fit << endl;
                }
            }
        }

        stats["num_rounds"] = num_rounds;
        stats["als_total_time"] = als_total_time; 
        stats["fit_computation_time"] = fit_computation_time; 

        if(grid.rank == 0) {
            cout << stats << endl;
        }
    }

    double compute_exact_fit() {
        return ground_truth.compute_exact_fit(low_rank_tensor);
    }
};