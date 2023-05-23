#pragma once

#include <iostream>
#include <cblas.h>
#include <lapacke.h>
#include "grid.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) LowRankTensor {
public:
    uint64_t rank;
    Buffer<double> sigma;
    vector<DistMat1D> factors;
    TensorGrid &tensor_grid;
    LowRankTensor(uint64_t rank, TensorGrid &tensor_grid) :
        rank(rank),
        sigma({rank}),
        tensor_grid(tensor_grid) {

        std::fill(sigma(), sigma(rank), 1.0);
        for(int i = 0; i < tensor_grid.dim; i++) {
            factors.emplace_back(rank, tensor_grid, i);
        } 
    }

    void initialize_factors_deterministic() {
        for(int i = 0; i < tensor_grid.dim; i++) {
            factors[i].initialize_deterministic();
        }
    }

    void test_gram_matrix_computation() {
        Buffer<double> gram_matrix({rank, rank});
        for(int i = 0; i < tensor_grid.dim; i++) {
            factors[i].initialize_deterministic();
            factors[i].compute_gram_matrix(gram_matrix);
            if(tensor_grid.grid.rank == 0) {
                cout << "Gram matrix for factor " << i << endl;
                gram_matrix.print();
            }
        }
    }
};