#pragma once

#include "common.h"
#include <algorithm>
#include <execution>

using namespace std;

class ALS_Optimizer {
public:
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;

    ALS_Optimizer(SparseTensor &ground_truth, 
        LowRankTensor &low_rank_tensor) 
        :
        ground_truth(ground_truth),
        low_rank_tensor(low_rank_tensor) {
    }

    virtual void initialize_ground_truth_for_als() = 0;
    virtual void execute_ALS_rounds(uint64_t num_rounds, uint64_t J, uint32_t epoch_interval) = 0;

    void deduplicate_design_matrix(
            Buffer<uint32_t> &samples,
            Buffer<double> &weights,
            Buffer<double> &h,
            uint64_t j, 
            Buffer<uint32_t> &samples_dedup,
            Buffer<double> &weights_dedup,
            Buffer<double> &h_dedup) {

        uint64_t J = samples.shape[0];
        uint64_t N = samples.shape[1];
        uint64_t R = h.shape[1]; 

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
        h_dedup.initialize_to_shape({num_unique, R});

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
            for(uint64_t k = 0; k < R; k++) {
                h_dedup[i * R + k] = h[offset * R + k]; 
            }
        }
    }

    double compute_exact_fit() {
        return ground_truth.compute_exact_fit(low_rank_tensor);
    }

    virtual ~ALS_Optimizer() {};
};