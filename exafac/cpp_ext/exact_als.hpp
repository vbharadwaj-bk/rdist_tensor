#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) ExactALS {
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;
public:
    ExactALS(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ground_truth(ground_truth),
    low_rank_tensor(low_rank_tensor)
    {
        cout << "Exact ALS initialized!" << endl;
    }
};