#pragma once

#include <iostream>
#include <cblas.h>
#include <lapacke.h>
#include "grid.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) LowRankTensor {
    uint64_t rank;
    vector<DistMat1D> factors;
    TensorGrid &tensor_grid;
public:
    LowRankTensor(uint64_t rank, TensorGrid &tensor_grid) :
        rank(rank),
        tensor_grid(tensor_grid) {

        for(int i = 0; i < tensor_grid.dim; i++) {
            factors.emplace_back(rank, tensor_grid, i);
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