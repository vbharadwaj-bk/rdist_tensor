#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) ExactALS {
public:
    SparseTensor &ground_truth;
    LowRankTensor &low_rank_tensor;
    TensorGrid &tensor_grid;
    Grid &grid;
    uint64_t dim;

    vector<Buffer<double>> gathered_factors;
    vector<Buffer<double>> gram_matrices; 

    ExactALS(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ground_truth(ground_truth),
    low_rank_tensor(low_rank_tensor),
    tensor_grid(ground_truth.tensor_grid),
    grid(ground_truth.tensor_grid.grid),
    dim(ground_truth.dim)
    {
        for(uint64_t i = 0; i < dim; i++) {
            int world_size;
            MPI_Comm_size(grid.slices[i], &world_size);
            uint64_t row_count = tensor_grid.padded_row_counts[i] * world_size;

            gathered_factors.emplace_back(
                Buffer<double>({row_count, low_rank_tensor.rank})
            );

            gram_matrices.emplace_back(
                Buffer<double>({low_rank_tensor.rank, low_rank_tensor.rank})
            );
        }
    }

    void execute_ALS_step(uint64_t mode_to_leave) {
        for(int i = 0; i < grid.dim; i++) {
            // Allgather the local data of low_rank_tensor.factors[i]
            // into the gathered factor buffers

            uint64_t R = low_rank_tensor.rank; 

            if(i != (int) mode_to_leave) {
                DistMat1D &factor = low_rank_tensor.factors[i];
                Buffer<double> &factor_data = *(factor.data);

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
                MPI_Barrier(grid.world);
            }

            Buffer<double> gram_product({R, R});
            Buffer<double> gram_product_inv({R, R});

            chain_had_prod(gram_matrices, gram_product, mode_to_leave);
            compute_pinv_square(gram_product, gram_product_inv, R);

            uint64_t output_buffer_rows = gathered_factors[mode_to_leave].shape[0];
            Buffer<double> temp_buf({output_buffer_rows, R});

            std::fill(temp_buf(), 
                temp_buf(output_buffer_rows * R), 
                0.0);

            ground_truth.lookups[i]->execute_exact_mttkrp( 
                gathered_factors,
                temp_buf 
            );
        }
    }

    // Warning: this doesn't compute the fit yet!
    double compute_exact_fit() {
        uint64_t R = low_rank_tensor.rank; 
        for(int i = 0; i < grid.dim; i++) {
            DistMat1D &factor = low_rank_tensor.factors[i];
            Buffer<double> &factor_data = *(factor.data);

            MPI_Allgather(
                factor_data(),
                factor_data.shape[0] * R,
                MPI_DOUBLE,
                gathered_factors[i](),
                factor_data.shape[0] * R,
                MPI_DOUBLE,
                grid.slices[i]
            );

            /*if(i == 0) {
                for(int k = 0; k < grid.world_size; k++) {
                    if(grid.rank == k) {
                        cout << "Factor data for rank " << grid.rank << endl;
                        gathered_factors[i].print(); 
                        cout << factor_data.shape[0] << " " << factor_data.shape[1] << endl;
                    }
                    MPI_Barrier(grid.world);
                }  
            }*/


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
        bmb = ground_truth.lookups[0]->compute_2bmb(
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

        return global_bmb;
    }

    void execute_ALS_round() {
        if(grid.rank == 0) {
            cout << "Executing ALS round" << endl;
        }
        execute_ALS_step(0);
    }
};