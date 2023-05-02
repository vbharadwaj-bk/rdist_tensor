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
    uint64_t dim;

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
            preprocessing(preprocessing),
            dim(tensor_grid.dim) 
            { 
        if(preprocessing == "log_count") {
            #pragma omp parallel for
            for(uint64_t i = 0; i < this->values.shape[0]; i++) {
                this->values[i] = log(this->values[i] + 1);
            }
        }

        check_tensor_invariants();
        redistribute_to_grid(tensor_grid);
    }

    void check_tensor_invariants() {
        // Sum up all the values
        double sum = 0;

        #pragma omp parallel for reduction(+:sum)
        for(uint64_t i = 0; i < this->values.shape[0]; i++) {
            sum += this->values[i];
        }

        // Allreduce sum across processors
        double global_sum = 0;
        MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if(tensor_grid.rank == 0) {
            cout << "Invariant: global sum of all values is " << global_sum << endl;
        }
    }

    void redistribute_to_grid(TensorGrid &tensor_grid) {
        Buffer<int> prefix({tensor_grid.dim}); 
        tensor_grid.grid.get_prefix_array(prefix);

        vector<uint64_t> send_counts(proc_count, 0);
        vector<int> processor_assignments(nnz, -1); 

        #pragma omp parallel
{
        vector<uint64_t> send_counts_local(proc_count, 0);

        #pragma omp for
        for(uint64_t i = 0; i < nnz; i++) {
            uint64_t processor = 0;
            for(int j = 0; j < dim; j++) {
                processor += prefix.[j] * (indices[i * dim + j] / tensor_grid.padded_row_counts[j]); 
            }

            send_counts_local[processor]++;
            processor_assignments[i] = processor;
        }

        for(uint64_t i = 0; i < proc_count; i++) {
            #pragma omp atomic 
            send_counts[i] += send_counts_local[i];
        }
}

        uint64_t total_send_counts = std::accumulate(send_counts.begin(), send_counts.end(), 0);
        cout << "Total send counts: " << total_send_counts << endl;
    }
};