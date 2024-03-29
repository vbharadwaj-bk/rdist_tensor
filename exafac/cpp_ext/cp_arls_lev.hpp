#pragma once

#include "common.h"
#include "distmat.hpp"
#include "grid.hpp"
#include "sampler.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) CP_ARLS_LEV : public Sampler {
public:
    vector<Buffer<double>> leverage_scores;

    CP_ARLS_LEV(LowRankTensor &tensor)
    :       
            Sampler(tensor) 
    {
        for(uint64_t i = 0; i < N; i++) {
            leverage_scores.emplace_back();
            leverage_scores[i].initialize_to_shape({U[i].padded_rows});
            update_sampler(i);
        }
    }

    void update_sampler(uint64_t j) {
        DistMat1D &mat = U[j];
        Buffer<double> &scores = leverage_scores[j];
        Buffer<double> gram({mat.cols, mat.cols});
        Buffer<double> gram_pinv({mat.cols, mat.cols});

        mat.compute_gram_matrix(gram);
        compute_pinv_square(gram, gram_pinv, mat.cols);
        compute_DAGAT(mat.data(), gram_pinv(), scores(), mat.true_row_count, mat.cols);
    }

    void KRPDrawSamples(uint64_t J,
            uint32_t mode_to_leave,
            Buffer<uint32_t> &samples,
            Buffer<double> &weights,    
            vector<Buffer<uint32_t>> &unique_row_indices) {

        samples.initialize_to_shape({J, N});
        weights.initialize_to_shape({J});

        std::fill(samples(), samples(J * N), 0);
        std::fill(weights(), weights(J), 0.0 - log((double) J));

        // Collect all samples and randomly permute along each mode 
        for(uint64_t i = 0; i < N; i++) {
            unique_row_indices.emplace_back();

            if(i == mode_to_leave) {
                continue;
            }

            Buffer<uint32_t> sample_idxs;
            Buffer<double> log_weights;
             
            draw_leverage_score_samples(J, i, sample_idxs, log_weights, unique_row_indices[i]);

            Buffer<uint32_t> rand_perm({J});
            std::iota(rand_perm(), rand_perm(J), 0);
            std::shuffle(rand_perm(), rand_perm(J), global_rng.par_gen[0]);

            apply_permutation(rand_perm, sample_idxs);
            apply_permutation(rand_perm, log_weights);

            #pragma omp parallel for
            for(uint64_t j = 0; j < J; j++) {
                samples[j * N + i] = sample_idxs[j];
                weights[j] += log_weights[j]; 
            }
        }

        #pragma omp parallel for
        for(uint64_t j = 0; j < J; j++) { 
            weights[j] = exp(weights[j]); 
        }
    }

    void draw_leverage_score_samples(uint64_t J,
            uint64_t j, 
            Buffer<uint32_t> &sample_idxs, 
            Buffer<double> &log_weights,
            Buffer<uint32_t> &unique_local_samples) {

        DistMat1D &mat = U[j];
        Buffer<double> &scores = leverage_scores[j];
        Grid &grid = mat.grid;

        double leverage_sum = std::accumulate(scores(), scores(mat.true_row_count), 0.0);
        Buffer<double> leverage_sums({(uint64_t) grid.world_size});
        Buffer<uint64_t> samples_per_process({(uint64_t) grid.world_size}); 
        MPI_Allgather(&leverage_sum,
            1,
            MPI_DOUBLE,
            leverage_sums(),
            1,
            MPI_DOUBLE,
            grid.world
            );

        double total_leverage_weight = std::accumulate(leverage_sums(), leverage_sums(grid.world_size), 0.0);

        // Should cache the distributions 
        std::discrete_distribution<uint32_t> local_dist(scores(), scores(mat.true_row_count));
        std::discrete_distribution<uint32_t> global_dist(leverage_sums(), leverage_sums(grid.world_size));

        // Not multithreaded, can thread if this becomes the bottleneck. 
        std::fill(samples_per_process(), samples_per_process(grid.world_size), 0);

        #pragma omp parallel
        {
            Buffer<uint64_t> local_samples_per_process({(uint64_t) grid.world_size}); 
            std::fill(local_samples_per_process(), local_samples_per_process(grid.world_size), 0);

            int tid = omp_get_thread_num();
            #pragma omp for
            for(uint64_t j = 0; j < J; j++) {
                uint64_t sample = global_dist(global_rng.par_gen[tid]);

                local_samples_per_process[sample]++; 
            }

            for(uint64_t i = 0; i < (uint64_t) grid.world_size; i++) {
                #pragma omp atomic
                samples_per_process[i] += local_samples_per_process[i];
            }
        } 

        uint64_t local_samples = samples_per_process[grid.rank];

        Buffer<uint32_t> sample_idxs_local({local_samples});
        Buffer<double> sample_weights_local({local_samples});

        std::fill(sample_weights_local(), sample_weights_local(local_samples), 0.0);

        uint32_t offset = mat.row_position * mat.padded_rows;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for(uint64_t i = 0; i < local_samples; i++) {
                uint32_t sample = local_dist(local_rng.par_gen[tid]);
                sample_idxs_local[i] = offset + sample; 
                sample_weights_local[i] += log(total_leverage_weight) - log(scores[sample]);
            }
        }

        allgatherv_buffer(sample_idxs_local, sample_idxs, MPI_COMM_WORLD);
        allgatherv_buffer(sample_weights_local, log_weights, MPI_COMM_WORLD);

        // Get a deduplicated list of unique samples per process
        // Need to change the execution policy to parallel 
        std::sort(sample_idxs_local(), sample_idxs_local(local_samples));
        uint32_t* end_unique = std::unique(sample_idxs_local(), sample_idxs_local(local_samples));

        uint32_t num_unique = end_unique - sample_idxs_local();
        unique_local_samples.initialize_to_shape({num_unique});

        std::copy(sample_idxs_local(), sample_idxs_local(num_unique), unique_local_samples());
    }
};