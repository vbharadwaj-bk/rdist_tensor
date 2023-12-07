#pragma once

#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <numeric>

#include "sort_lookup.hpp"
#include "common.h"
#include "alltoallv_revised.hpp"
#include "low_rank_tensor.hpp"
#include "random_util.hpp"

using namespace std;

class SparseTensor {
public:
    TensorGrid &tensor_grid; 
    Buffer<uint32_t> indices;
    Buffer<double> values;
    std::string preprocessing;
    uint64_t dim;
    Buffer<uint64_t> offsets;
    vector<unique_ptr<SortIdxLookup<uint32_t, double>>> lookups;
    double tensor_norm;

    // Load balancing random permutations along each dimension
    vector<Buffer<uint32_t>> perms;

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
        tensor_norm = 0.0;
        if(preprocessing == "log_count") {
            #pragma omp parallel for reduction(+:tensor_norm)
            for(uint64_t i = 0; i < values.shape[0]; i++) {
                this->values[i] = log(values[i] + 1);
                tensor_norm += this->values[i] * this->values[i];
            }
        }
        else if(preprocessing == "ones") {
            #pragma omp parallel for reduction(+:tensor_norm)
            for(uint64_t i = 0; i < values.shape[0]; i++) {
                this->values[i] = 1.0; 
                tensor_norm += 1.0; 
            }
        }
        else {
            #pragma omp parallel for reduction(+:tensor_norm)
            for(uint64_t i = 0; i < values.shape[0]; i++) {
                tensor_norm += this->values[i] * this->values[i];
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &tensor_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tensor_norm = sqrt(tensor_norm);
        if(tensor_grid.rank == 0) {
            cout << "Tensor norm is " << tensor_norm << endl;
        }

        Consistent_Multistream_RNG gen(tensor_grid.grid.world);

        // Compute load-balancing permutations 
        for(uint64_t i = 0; i < dim; i++) {
            uint64_t mode_size = tensor_grid.tensor_dims[i];
            perms.emplace_back(
                Buffer<uint32_t>({mode_size})
            );

            std::iota(perms[i](), perms[i](mode_size), 0);
            std::shuffle(perms[i](), perms[i](mode_size), gen.par_gen[0]);
        }

        // Apply load-balancing permutations
        #pragma omp parallel for 
        for(uint64_t i = 0; i < values.shape[0]; i++) {
            for(uint64_t j = 0; j < dim; j++) {
                indices[i * dim + j] = perms[j][indices[i * dim + j]];
            }
        }
    }

    /*
    * Generates a random sparse tensor. Currently only works
    * for 3D synthetic tensors, but could be extended further. 
    */
    SparseTensor(
        TensorGrid &tensor_grid,
        uint64_t I,
        uint64_t N,
        uint64_t Q) :
            tensor_grid(tensor_grid),
            preprocessing("none"),
            dim(tensor_grid.dim),
            offsets({dim}) 
            {
        if(N != 3) {
            cout << "Error, dimension must be 3." << endl;
            exit(1);
        }

        // bl[i][j] is a list that tells you that factor matrix 
        // i has basis vector j in all rows.

        vector<vector<vector<int>>> bl; 
        for(uint64_t i = 0; i < N; i++) {
            bl.emplace_back();
            for(uint64_t j = 0; j < Q; j++) {
                bl[i].emplace_back();
            }
        }

        Consistent_Multistream_RNG seed_gen(tensor_grid.grid.world);
        std::uniform_int_distribution<int> dist(0, Q - 1);

        for(uint64_t i = 0; i < N; i++) {
            for(uint64_t k = 0; k < I; k++) {
                uint64_t j = dist(seed_gen.par_gen[0]);
                bl[i][j].emplace_back(k);
            }
        }

        uint64_t proc_count = tensor_grid.grid.world_size;
        uint64_t column_frac = (Q + proc_count - 1) / proc_count;

        int rank = tensor_grid.rank;
        uint64_t col_start = min(rank * column_frac, Q);
        uint64_t col_end = min((rank + 1) * column_frac, Q);

        uint64_t local_nnz = 0;   
        uint64_t total_nnz = 0;

        Buffer<uint32_t> nonzeros_per_col({col_end - col_start});
        Buffer<uint32_t> nonzeros_prefix_sum({col_end - col_start});
        for(uint64_t j = col_start; j < col_end; j++) {
            uint64_t prod = 1;
            for(uint64_t i = 0; i < N; i++) {
                prod *= bl[i][j].size();
            }
            local_nnz += prod;
            nonzeros_per_col[j - col_start] = prod;
        }

        std::exclusive_scan(nonzeros_per_col(), nonzeros_per_col(col_end - col_start), nonzeros_prefix_sum(), 0);
        MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

        indices.initialize_to_shape({local_nnz, N});
        values.initialize_to_shape({local_nnz});

        tensor_norm = 0.0;
        #pragma omp parallel for reduction(+: tensor_norm)
        for(uint64_t j = col_start; j < col_end; j++) {
            uint64_t pos = nonzeros_prefix_sum[j - col_start];
            for(uint64_t u = 0; u < bl[0][j].size(); u++) {
                for(uint64_t v = 0; v < bl[1][j].size(); v++) {
                    for(uint64_t w = 0; w < bl[2][j].size(); w++) {
                        indices[pos * N + 0] = bl[0][j][u];
                        indices[pos * N + 1] = bl[1][j][v];
                        indices[pos * N + 2] = bl[2][j][w];
                        values[pos] = 1;
                        tensor_norm += values[pos] * values[pos];
                        pos++;
                    }
                }
            }
        }

        if(rank == 0) {
            double expected_nonzeros = pow(I, N) / pow(Q, N-1);
            cout << "Expected Nonzero Count: " << expected_nonzeros << endl;
            cout << "True Nonzero Count: " << total_nnz << endl;
        }

        MPI_Allreduce(MPI_IN_PLACE, &tensor_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tensor_norm = sqrt(tensor_norm);
        if(tensor_grid.rank == 0) {
            cout << "Tensor norm is " << tensor_norm << endl;
        }

        Consistent_Multistream_RNG gen(tensor_grid.grid.world);

        // Compute load-balancing permutations 
        for(uint64_t i = 0; i < dim; i++) {
            uint64_t mode_size = tensor_grid.tensor_dims[i];
            perms.emplace_back(
                Buffer<uint32_t>({mode_size})
            );

            std::iota(perms[i](), perms[i](mode_size), 0);
            std::shuffle(perms[i](), perms[i](mode_size), gen.par_gen[0]);
        }

        // Apply load-balancing permutations
        #pragma omp parallel for 
        for(uint64_t i = 0; i < values.shape[0]; i++) {
            for(uint64_t j = 0; j < dim; j++) {
                indices[i * dim + j] = perms[j][indices[i * dim + j]];
            }
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

        #pragma omp parallel
{
        vector<uint64_t> send_counts_local(proc_count, 0);

        #pragma omp for
        for(uint64_t i = 0; i < nnz; i++) {
            uint64_t target_proc = 0;
            for(uint64_t j = 0; j < dim; j++) {
                target_proc += prefix[j] * (indices[i * dim + j] / tensor_grid.subblock_sizes_uniform[j]); 
            }
            
            send_counts_local[target_proc]++;
            processor_assignments[i] = target_proc; 
        }

        for(uint64_t i = 0; i < proc_count; i++) {
            #pragma omp atomic 
            send_counts[i] += send_counts_local[i];
        }
}

        Buffer<uint32_t> recv_idxs;
        Buffer<double> recv_values;
        alltoallv_matrix_rows(
            indices,
            processor_assignments,
            send_counts,
            recv_idxs,
            tensor_grid.grid.world
        );
        alltoallv_matrix_rows(
            values,
            processor_assignments,
            send_counts,
            recv_values,
            tensor_grid.grid.world
        );

        indices.steal_resources(recv_idxs);
        values.steal_resources(recv_values);
    }

    double compute_exact_fit(LowRankTensor &low_rank_tensor) {
        vector<Buffer<double>> gathered_factors;
        vector<Buffer<double>> gram_matrices;

        Grid &grid = tensor_grid.grid;
        uint64_t R = low_rank_tensor.rank; 
        for(int i = 0; i < grid.dim; i++) {
            DistMat1D &factor = low_rank_tensor.factors[i];
            Buffer<double> &factor_data = factor.data;

            int world_size;
            MPI_Comm_size(grid.slices[i], &world_size);
            uint64_t row_count = tensor_grid.padded_row_counts[i] * world_size;

            gathered_factors.emplace_back(
                Buffer<double>({row_count, low_rank_tensor.rank})
            );

            gram_matrices.emplace_back(
                Buffer<double>({R, R})
            );

            MPI_Allgather(
                factor_data(),
                factor_data.shape[0] * R,
                MPI_DOUBLE,
                gathered_factors[i](),
                factor_data.shape[0] * R,
                MPI_DOUBLE,
                grid.slices[i]
            );

            factor.compute_gram_matrix(gram_matrices[i]);
        }

        Buffer<double> gram_product({R, R});
        chain_had_prod(gram_matrices, gram_product, -1);

        double normsq_lowrank_tensor = 0;
        #pragma omp parallel for collapse(2) reduction(+:normsq_lowrank_tensor)
        for(uint64_t i = 0; i < R; i++) {
            for(uint64_t j = 0; j < R; j++) {
                normsq_lowrank_tensor += 
                    gram_product[i * R + j] 
                    * low_rank_tensor.sigma[i] * low_rank_tensor.sigma[j];
            }
        }

        double bmb = 0;
        bmb = lookups[0]->compute_2bmb(
            low_rank_tensor.sigma,
            gathered_factors
        );

        // Allreduce residual_normsq across processors 
        double global_bmb = 0;
        MPI_Allreduce(&bmb, 
            &global_bmb, 
            1, 
            MPI_DOUBLE, 
            MPI_SUM, 
            MPI_COMM_WORLD);

        double norm_residual = sqrt(normsq_lowrank_tensor + global_bmb);

        // Should floor with 0, but let's leave
        // it like this for now 
        return 1 - (norm_residual / tensor_norm); 
    }
};