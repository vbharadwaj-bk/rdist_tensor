#include <iostream>
#include <string>

#include "sort_lookup.hpp"
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
    Buffer<uint64_t> offsets;

    vector<SortIdxLookup<uint32_t, double>> sort_idx_lookups;

    /*
    * To avoid a compile-time dependence on the HDF-5 library,
    * an instance of SparseTensor is tied to a Python file
    * that calls the HDF5 library. 
    */
    SparseTensor(
        TensorGrid &tensor_grid,
        py::array_t<uint32_t> indices_py,
        py::array_t<double> values_py, 
        std::string preprocessing) :
            tensor_grid(tensor_grid),
            indices(indices_py, true), 
            values(values_py, true), 
            preprocessing(preprocessing),
            dim(tensor_grid.dim),
            offsets({dim}) 
            { 
        if(preprocessing == "log_count") {
            #pragma omp parallel for
            for(uint64_t i = 0; i < values.shape[0]; i++) {
                this->values[i] = log(values[i] + 1);
            }
        }

        check_tensor_invariants();
        redistribute_to_grid(tensor_grid);
        check_tensor_invariants();

        for(uint64_t i = 0; i < dim; i++) {
            offsets[i] = tensor_grid.start_coords[i][tensor_grid.grid.coords[i]];
        }


        //#pragma omp parallel for
        for(uint64_t i = 0; i < indices.shape[0]; i++) {
            for(uint64_t j = 0; j < dim; j++) {
                indices[i * dim + j] -= offsets[j];
            }
        }

        for(uint64_t i = 0; i < dim; i++) {
            sort_idx_lookups.emplace_back(dim, i, indices(), values(), indices.shape[0]);
        }
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

        //#pragma omp parallel
{
        vector<uint64_t> send_counts_local(proc_count, 0);

        //#pragma omp for
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
        unique_ptr<Buffer<double>> recv_values;
        alltoallv_matrix_rows(
            indices,
            processor_assignments,
            send_counts,
            recv_idxs
        );
        alltoallv_matrix_rows(
            values,
            processor_assignments,
            send_counts,
            recv_values
        );

        indices.steal_resources(*recv_idxs);
        values.steal_resources(*recv_values);
    }
};