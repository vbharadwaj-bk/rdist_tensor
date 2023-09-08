#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include "common.h"
#include "sampler.hpp"
#include "partition_tree.hpp"
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"
#include "distmat.hpp"
#include "exact_leverage_tree.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) EfficientKRPSampler: public Sampler {
public:
    Buffer<double> M;
    Buffer<double> lambda;

    vector<Buffer<double>> scaled_eigenvecs;

    vector<ExactLeverageTree*> gram_trees;
    vector<PartitionTree*> eigen_trees;
    double eigenvalue_tolerance;

    // Related to random number generation 
    std::uniform_real_distribution<> dis;

    EfficientKRPSampler(
            vector<DistMat1D> &U_matrices)
    :       
            Sampler(U_matrices),
            M({U_matrices.size() + 2, R * R}),
            lambda({U_matrices.size() + 1, R}),
            dis(0.0, 1.0) 
    {   
        eigenvalue_tolerance = 1e-8; // Tolerance of eigenvalues for symmetric PINV 
    
        for(uint32_t i = 0; i < N; i++) {
            //uint64_t F = R < n ? R : n;
            gram_trees.push_back(new ExactLeverageTree(U[i], U[i].ordered_world));
            eigen_trees.push_back(new PartitionTree(R, 1, R));
        }

        // Should move the data structure initialization to another routine,
        // but this is fine for now.

        for(uint32_t i = 0; i < N; i++) {
            gram_trees[i]->build_tree(); 
        }

        for(uint32_t i = 0; i < N + 1; i++) {
            scaled_eigenvecs.emplace_back(initializer_list<uint64_t>{R, R}, M(i, 0));
        }
    }

    /*
    * Updates the j'th gram tree when the factor matrix is
    * updated. 
    */
    void update_sampler(uint64_t j) {
        gram_trees[j]->build_tree(); 
    }

    /*
     * Simple, unoptimized square-matrix in-place transpose.
    */
    void transpose_square_in_place(double* ptr, uint64_t n) {
        for(uint64_t i = 0; i < n - 1; i++) {
            for(uint64_t j = i + 1; j < n; j++) {
                double temp = ptr[i * n + j];
                ptr[i * n + j] = ptr[j * n + i];
                ptr[j * n + i] = temp;
            }
        }
    }

    void computeM(uint32_t j) {
        std::fill(M(N * R2), M((N + 1) * R2), 1.0);

        #pragma omp parallel
{
        uint32_t last_buffer = N;
        for(int k = N - 1; k >= 0; k--) {
            if((uint32_t) k != j) {
                Buffer<double> &G = (gram_trees[k]->ancestor_grams).back();

                #pragma omp for
                for(uint32_t i = 0; i < R2; i++) {
                    M[k * R2 + i] = G[i] * M[(last_buffer * R2) + i];   
                } 

                last_buffer = k;
            }
        }
}

        if(j == 0) {
            std::copy(M(1, 0), M(1, R2), M());
        }

        // Store the original matrix in slot N + 2 
        std::copy(M(), M(R2), M((N + 1) * R2));

        // Pseudo-inverse via eigendecomposition, stored in the N+1'th slot of
        // the 2D M array.

        LAPACKE_dsyev( CblasRowMajor, 
                        'V', 
                        'U', 
                        R,
                        M(), 
                        R, 
                        lambda() );

        #pragma omp parallel for
        for(uint32_t v = 0; v < R; v++) {
            if(lambda[v] > eigenvalue_tolerance) {
                for(uint32_t u = 0; u < R; u++) {
                        M[N * R2 + u * R + R - 1 - v] = M[u * R + v] / sqrt(lambda[v]); 
                }
            }
            else {
                for(uint32_t u = 0; u < R; u++) {
                        M[N * R2 + u * R + R - 1 - v] = 0.0; 
                }
            }
        }

        cblas_dsyrk(CblasRowMajor, 
                    CblasUpper, 
                    CblasNoTrans,
                    R,
                    R, 
                    1.0, 
                    (const double*) M(N, 0), 
                    R, 
                    0.0, 
                    M(), 
                    R);

        #pragma omp parallel
{
        for(uint32_t k = N - 1; k > 0; k--) {
            if(k != j) {
                #pragma omp for
                for(uint32_t i = 0; i < R2; i++) {
                    M[k * R2 + i] *= M[i];   
                }
            }
        }

        // Eigendecompose each of the gram matrices 
        #pragma omp for
        for(uint32_t k = N; k > 0; k--) {
            if(k != j) {
                if(k < N) {
                    LAPACKE_dsyev( CblasRowMajor, 
                                    'V', 
                                    'U', 
                                    R,
                                    M(k, 0), 
                                    R, 
                                    lambda(k, 0) );

                    for(uint32_t v = 0; v < R; v++) { 
                        for(uint32_t u = 0; u < R; u++) {
                            M[k * R2 + u * R + v] *= sqrt(lambda[k * R + v]); 
                        }
                    }
                }
                transpose_square_in_place(M(k, 0), R);
            }
        }
}

        for(int k = N-1; k >= 0; k--) {
            if((uint32_t) k != j) {
                Buffer<double> &G = (gram_trees[k]->ancestor_grams).back();

                int offset = (k + 1 == (int) j) ? k + 2 : k + 1;
                eigen_trees[k]->build_tree(scaled_eigenvecs[offset]);
                eigen_trees[k]->multiply_matrices_against_provided(G);
            }
        } 
    }

    void fill_buffer_random_draws(double* data, uint64_t len) {
        #pragma omp parallel
{
        int thread_id = omp_get_thread_num();
        auto &local_gen = par_gen[thread_id];

        #pragma omp for
        for(uint64_t i = 0; i < len; i++) {
            data[i] = dis(local_gen);
        }
}
    }

    void KRPDrawSamples(uint64_t J,
            uint32_t j,
            Buffer<uint32_t> &samples,
            Buffer<double> &weights,    
            vector<Buffer<uint32_t>> &unique_row_indices) {

        int world_size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        uint64_t work = (J + world_size - 1) / world_size; 
        uint64_t start = min(work * rank, J);
        uint64_t end = min(work * (rank + 1), J);

        Buffer<double> h({end - start, R});
        Buffer<double> scaled_h({end - start, R});
        samples.initialize_to_shape({end - start, N});

        // Samples is an array of size N x J 
        computeM(j);
        std::fill(h(), h({end - start}, 0), 1.0);

        for(uint32_t k = 0; k < N; k++) {
            unique_row_indices.emplace_back();

            if(k != j) {
                uint64_t row_count = scaled_h.shape[0];
                std::copy(h(), h(h.shape[0] * h.shape[1]), scaled_h());

                Buffer<double> random_draws({row_count});
                ScratchBuffer eigen_scratch(1, scaled_h.shape[0], R);

                fill_buffer_random_draws(random_draws(), row_count);
                int offset = (k + 1 == j) ? k + 2 : k + 1;
                eigen_trees[k]->PTSample(scaled_eigenvecs[offset], 
                        scaled_h,
                        h,
                        samples,
                        random_draws, 
                        eigen_scratch,
                        k);

                fill_buffer_random_draws(random_draws(), row_count);
                gram_trees[k]->execute_tree_computation(
                        h,
                        scaled_h, 
                        samples, 
                        random_draws, 
                        k);

                // Get a list of unique local samples
                row_count = h.shape[0];
                Buffer<uint32_t> sample_idxs_local({row_count});
                for(uint64_t i = 0; i < row_count; i++) {
                    sample_idxs_local[i] = samples[i * N + k];
                }

                std::sort(sample_idxs_local(), sample_idxs_local(row_count));
                uint32_t* end_unique = std::unique(sample_idxs_local(), sample_idxs_local(row_count));

                uint64_t num_unique = end_unique - sample_idxs_local();
                unique_row_indices.back().initialize_to_shape({num_unique});
                std::copy(sample_idxs_local(), sample_idxs_local(num_unique), unique_row_indices.back()() );
            }
        }


        weights.initialize_to_shape({h.shape[0]});

        // Compute the weights associated with the samples
        compute_DAGAT(
            h(),
            M(),
            weights(),
            h.shape[0],
            R);

        #pragma omp parallel for 
        for(uint32_t i = 0; i < h.shape[0]; i++) {
            weights[i] = (double) R / (weights[i] * J);
        }

        Buffer<uint32_t> gathered_samples;
        Buffer<double> gathered_weights;
        allgatherv_buffer(samples, gathered_samples, MPI_COMM_WORLD);
        allgatherv_buffer(weights, gathered_weights, MPI_COMM_WORLD);
        samples.steal_resources(gathered_samples);
        weights.steal_resources(gathered_weights);

    }

    ~EfficientKRPSampler() {
        for(uint32_t i = 0; i < N; i++) {
            delete gram_trees[i];
            delete eigen_trees[i];
        }
    }
};


void test_distributed_exact_leverage(LowRankTensor &ten) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint64_t J = 65000;
    EfficientKRPSampler sampler(ten.factors);

    if(rank == 0) {
        cout << "Constructed sampler..." << endl;
    }

    Buffer<uint32_t> samples;
    Buffer<double> weights;
    vector<Buffer<uint32_t>> unique_row_indices; 

    sampler.KRPDrawSamples(J, 0, samples, weights, unique_row_indices);
}