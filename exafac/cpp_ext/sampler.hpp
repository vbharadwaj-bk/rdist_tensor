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

    Sampler(vector<DistMat1D> &U_matrices) : 
        U(U_matrices),
        R(U_matrices[0].cols),
        global_rng(MPI_COMM_WORLD),
        N(U.size()),
        R2(R * R)
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
