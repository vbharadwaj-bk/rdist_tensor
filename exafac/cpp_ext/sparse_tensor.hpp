#include <iostream>
#include <string>

#include "common.h"
#include "alltoallv_revised.hpp"

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
        check_tensor_invariants();
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
        uint64_t nnz = indices.shape[0];
        uint64_t proc_count = tensor_grid.grid.world_size;

        Buffer<int> prefix({(uint64_t) tensor_grid.dim}); 
        tensor_grid.grid.get_prefix_array(prefix);

        Buffer<uint64_t> send_counts({proc_count});
        std::fill(send_counts(), send_counts(proc_count), 0);
        Buffer<int> processor_assignments({nnz}); 

        #pragma omp parallel
{
        vector<uint64_t> send_counts_local(proc_count, 0);

        #pragma omp for
        for(uint64_t i = 0; i < nnz; i++) {
            uint64_t target_proc = 0;
            for(uint64_t j = 0; j < dim; j++) {
                target_proc += prefix[j] * (indices[i * dim + j] / tensor_grid.padded_row_counts[j]); 
            }

            send_counts_local[target_proc]++;
            processor_assignments[i] = target_proc; 
        }

        for(uint64_t i = 0; i < proc_count; i++) {
            #pragma omp atomic 
            send_counts[i] += send_counts_local[i];
        }
}

        unique_ptr<Buffer<uint32_t>> recv_idxs;
        alltoallv_matrix_rows(
            indices,
            processor_assignments,
            send_counts,
            recv_idxs
        );
    }
};