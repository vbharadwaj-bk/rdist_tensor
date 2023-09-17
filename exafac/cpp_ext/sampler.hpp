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

    Consistent_Multistream_RNG global_rng;
    Multistream_RNG local_rng;

    Sampler(LowRankTensor &tensor) : 
        U(tensor.factors),
        N(U.size()),
        R(U[0].cols),
        R2(R * R),
        global_rng(MPI_COMM_WORLD)
    {
    }

    virtual void update_sampler(uint64_t j) = 0;
    virtual void KRPDrawSamples(uint64_t J,
            uint32_t j, 
            Buffer<uint32_t> &samples, 
            Buffer<double> &weights,
            vector<Buffer<uint32_t>> &unique_row_indices) = 0;

    virtual ~Sampler() {};
};
