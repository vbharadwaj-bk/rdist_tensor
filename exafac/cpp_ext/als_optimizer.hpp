#pragma once

#include "common.h"
#include <algorithm>
#include <execution>

using namespace std;

class ALS_Optimizer {
public:
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;

    TensorGrid &tensor_grid;
    Grid &grid;
    uint64_t dim;

    // Related to benchmarking 
    json stats;
    double leverage_computation_time, row_gather_time, dense_reduce_time, gram_mult_and_renorm_time;
    double spmm_time, nonzeros_iterated;
    double design_matrix_reindexing_time;

    ALS_Optimizer(SparseTensor &ground_truth, 
        LowRankTensor &low_rank_tensor) 
        :
        ground_truth(ground_truth),
        low_rank_tensor(low_rank_tensor),    
        tensor_grid(ground_truth.tensor_grid),
        grid(ground_truth.tensor_grid.grid),
        dim(ground_truth.dim)
        {
    }

    void execute_ALS_rounds(uint64_t num_rounds, uint64_t J, uint32_t epoch_interval) {
        double als_total_time = 0.0;
        double fit_computation_time = 0.0;

        // Benchmarking timers 
        spmm_time = 0.0;
        nonzeros_iterated = 0.0;
        leverage_computation_time = 0.0; 
        row_gather_time = 0.0; 
        dense_reduce_time = 0.0;
        gram_mult_and_renorm_time = 0.0;
        design_matrix_reindexing_time = 0.0; 

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
        stats["spmm_time"] = compute_dstat(spmm_time, MPI_COMM_WORLD);
        stats["nonzeros_iterated"] = compute_dstat(nonzeros_iterated, MPI_COMM_WORLD);
        stats["leverage_computation_time"] = compute_dstat(leverage_computation_time, MPI_COMM_WORLD);
        stats["row_gather_time"] = compute_dstat(row_gather_time, MPI_COMM_WORLD);
        stats["dense_reduce_time"] = compute_dstat(dense_reduce_time, MPI_COMM_WORLD);
        stats["gram_mult_and_renorm_time"] = compute_dstat(gram_mult_and_renorm_time, MPI_COMM_WORLD);
        stats["design_matrix_reindexing_time"] = compute_dstat(design_matrix_reindexing_time, MPI_COMM_WORLD);

        if(grid.rank == 0) {
            cout << stats.dump(4) << endl;
        }
    }

    void deduplicate_design_matrix(
            Buffer<uint32_t> &samples,
            Buffer<double> &weights,
            uint64_t j, 
            Buffer<uint32_t> &samples_dedup,
            Buffer<double> &weights_dedup) {

        uint64_t J = samples.shape[0];
        uint64_t N = samples.shape[1];

        Buffer<uint32_t*> sort_idxs({J});
        Buffer<uint32_t*> dedup_idxs({J});

        #pragma omp parallel for
        for(uint64_t i = 0; i < J; i++) {
            sort_idxs[i] = samples(i * N);
        }

        std::sort(std::execution::par_unseq, 
            sort_idxs(), 
            sort_idxs(J),
            [j, N](uint32_t* a, uint32_t* b) {
                for(uint32_t i = 0; i < N; i++) {
                    if(i != j && a[i] != b[i]) {
                        return a[i] < b[i];
                    }
                }
                return false;  
            });


        uint32_t** end_range = 
            std::unique_copy(std::execution::par_unseq,
                sort_idxs(),
                sort_idxs(J),
                dedup_idxs(),
                [j, N](uint32_t* a, uint32_t* b) {
                    for(uint32_t i = 0; i < N; i++) {
                        if(i != j && a[i] != b[i]) {
                            return false;
                        }
                    }
                    return true; 
                });

        uint64_t num_unique = end_range - dedup_idxs();

        samples_dedup.initialize_to_shape({num_unique, N});
        weights_dedup.initialize_to_shape({num_unique});

        #pragma omp parallel for
        for(uint64_t i = 0; i < num_unique; i++) {
            uint32_t* buf = dedup_idxs[i];
            uint32_t offset = (buf - samples()) / N;

            std::pair<uint32_t**, uint32_t**> bounds = std::equal_range(
                sort_idxs(),
                sort_idxs(J),
                buf, 
                [j, N](uint32_t* a, uint32_t* b) {
                    for(uint32_t i = 0; i < N; i++) {
                        if(i != j && a[i] != b[i]) {
                            return a[i] < b[i];
                        }
                    }
                    return false; 
                });

            uint64_t num_copies = bounds.second - bounds.first; 

            weights_dedup[i] = weights[offset] * num_copies; 

            for(uint64_t k = 0; k < N; k++) {
                samples_dedup[i * N + k] = buf[k];
            }
        }
    }

    void gather_lk_samples_to_all(
            uint64_t J, 
            uint64_t mode_to_leave,
            Buffer<uint32_t> &samples, 
            Buffer<double> &weights, 
            vector<Buffer<uint32_t>> &unique_row_indices) {

        std::fill(samples(), samples(J * ground_truth.dim), 0);
        std::fill(weights(), weights(J), 0.0 - log((double) J));
        Consistent_Multistream_RNG global_rng(MPI_COMM_WORLD);

        // Collect all samples and randomly permute along each mode 
        for(uint64_t i = 0; i < ground_truth.dim; i++) {
            unique_row_indices.emplace_back();

            if(i == mode_to_leave) {
                continue;
            }

            Buffer<uint32_t> sample_idxs;
            Buffer<double> log_weights;
             
            low_rank_tensor.factors[i].draw_leverage_score_samples(J, sample_idxs, log_weights, unique_row_indices[i]);

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
    }

    double compute_exact_fit() {
        return ground_truth.compute_exact_fit(low_rank_tensor);
    }

    virtual void execute_ALS_step(uint64_t mode, uint64_t J) = 0;
    virtual void initialize_ground_truth_for_als() = 0;
    virtual ~ALS_Optimizer() {};
};