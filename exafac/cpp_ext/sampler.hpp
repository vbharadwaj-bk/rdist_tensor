#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include "common.h"
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"

#include "distmat.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) Sampler {
public:
    vector<DistMat1D> &U;
    uint64_t N, R, R2;

    // Related to random number generation 
    std::random_device rd;  
    std::mt19937 gen;

    // Related to independent random number generation on multiple
    // streams
    int thread_count;
    vector<std::mt19937> par_gen; 

    Sampler(vector<DistMat1D> &U_matrices) : 
        U(U_matrices),
        R(U_matrices[0].cols),
        rd(),
        gen(rd())
        {
        this->N = U.size();
        R2 = R * R;

        // Set up independent random streams for different threads.
        // As written, might be more complicated than it needs to be. 
        #pragma omp parallel
        {
            #pragma omp single 
            {
                thread_count = omp_get_num_threads();
            }
        }

        vector<uint32_t> biased_seeds(thread_count, 0);
        vector<uint32_t> seeds(thread_count, 0);

        for(int i = 0; i < thread_count; i++) {
            biased_seeds[i] = rd();
        }
        std::seed_seq seq(biased_seeds.begin(), biased_seeds.end());
        seq.generate(seeds.begin(), seeds.end());

        for(int i = 0; i < thread_count; i++) {
            par_gen.emplace_back(seeds[i]);
        }
    }

    virtual void update_sampler(uint64_t j) = 0;
    virtual void KRPDrawSamples(uint64_t J,
            uint32_t j, 
            Buffer<uint32_t> &samples, 
            Buffer<double> &weights,
            vector<Buffer<uint32_t>> &unique_row_indices) = 0;

    virtual ~Sampler() {};
};
