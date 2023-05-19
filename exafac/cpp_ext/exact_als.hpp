#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"

using namespace std;

class Exact_ALS {
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;
public:
    TensorStationary(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ground_truth(ground_truth),
    low_rank_tensor(low_rank_tensor)
    {
        cout << "Exact ALS initialized!" << endl;
    }
};