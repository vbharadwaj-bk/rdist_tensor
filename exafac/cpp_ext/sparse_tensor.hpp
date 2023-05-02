#include <iostream>
#include <string>

#include "common.h"

using namespace std;

class SparseTensor {
public:
    TensorGrid &tensor_grid; 
    Buffer<uint32_t> indices;
    Buffer<double> values;
    std::string preprocessing;

    /*
    * To avoid a compile-time dependence on the HDF-5 library,
    * an instance of SparseTensor is tied to a Python file
    * that calls the HDF5 library. This allows a flexible
    *  

    */
    SparseTensor(
        TensorGrid &tensor_grid,
        py::array_t<uint32_t> indices,
        py::array_t<double> values, 
        std::string preprocessing) :
            tensor_grid(tensor_grid),
            indices(indices, true), 
            values(values, true), 
            preprocessing(preprocessing) { 
        if(preprocessing == "log_count") {
            #pragma omp parallel for
            for(uint64_t i = 0; i < this->values.shape[0]; i++) {
                this->values[i] = log(this->values[i] + 1);
            }
        }
    }

    /*void redistribute_to_grid(Grid &grid) {

    }*/
};