#pragma once

#include "common.h"
#include <string>
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
    double leverage_sampling_time;        // Region 1
    double row_gather_time;               // Region 2
    double design_matrix_prep_time;       // Region 3
    double spmm_time, nonzeros_iterated;  // Region 4
    double dense_reduce_time;             // Region 5
    double postprocessing_time;           // Region 6
    double sampler_update_time;           // Region 7


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
        leverage_sampling_time = 0.0;        // Region 1
        row_gather_time = 0.0;               // Region 2
        design_matrix_prep_time = 0.0;       // Region 3
        spmm_time = 0.0;  
        nonzeros_iterated = 0.0              // Region 4
        dense_reduce_time = 0.0;             // Region 5
        postprocessing_time = 0.0;           // Region 6
        sampler_update_time = 0.0;           // Region 7

        vector<uint64_t> rounds;
        vector<double> fits, als_times, fit_computation_times;

        rounds.push_back(0);
        fits.push_back(0);
        als_times.push_back(0);
        fit_computation_times.push_back(0);

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
                rounds.push_back(round);
                fits.push_back(exact_fit);
                als_times.push_back(als_total_time);
                fit_computation_times.push_back(fit_computation_time);
            }
        }

        stats["num_rounds"] = num_rounds;
        stats["als_total_time"] = als_total_time; 
        stats["fit_computation_time"] = fit_computation_time; 

        stats["leverage_sampling_time"] = compute_dstat(leverage_sampling_time, MPI_COMM_WORLD);
        stats["row_gather_time"] = compute_dstat(row_gather_time, MPI_COMM_WORLD);

        stats["design_matrix_prep_time"] = compute_dstat(design_matrix_prep_time, MPI_COMM_WORLD);
        stats["spmm_time"] = compute_dstat(spmm_time, MPI_COMM_WORLD);
        stats["nonzeros_iterated"] = compute_dstat(nonzeros_iterated, MPI_COMM_WORLD);

        stats["dense_reduce_time"] = compute_dstat(dense_reduce_time, MPI_COMM_WORLD);
        stats["postprocessing_time"] = compute_dstat(gram_mult_and_renorm_time, MPI_COMM_WORLD);
        stats["sampler_update_time"] = compute_dstat(sampler_update_time, MPI_COMM_WORLD);

        json rounds_json(rounds);
        json fits_json(fits);
        json als_times_json(als_times);
        json fit_computation_times_json(fit_computation_times);

        stats["rounds"] = rounds_json;
        stats["fits"] = fits_json;
        stats["als_times"] = als_times_json;
        stats["fit_computation_times"] = fit_computation_times; 
    }

    std::string get_statistics_json() {
        return stats.dump(4);
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

    double compute_exact_fit() {
        return max(ground_truth.compute_exact_fit(low_rank_tensor), 0.0);
    }

    virtual void execute_ALS_step(uint64_t mode, uint64_t J) = 0;
    virtual void initialize_ground_truth_for_als() = 0;
    virtual ~ALS_Optimizer() {};
};