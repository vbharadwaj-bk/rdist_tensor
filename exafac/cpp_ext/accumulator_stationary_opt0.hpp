#pragma once

#include "common.h"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"
#include "alltoallv_revised.hpp"
#include "json.hpp"

#include <algorithm>

using namespace std;

class __attribute__((visibility("hidden"))) AccumulatorStationaryOpt0 : public ALS_Optimizer{
public:
    TensorGrid &tensor_grid;
    Grid &grid;

    uint64_t dim;

    vector<Buffer<uint32_t>> indices;
    vector<Buffer<double>> values;
    vector<unique_ptr<SortIdxLookup<uint32_t, double>>> lookups;

    AccumulatorStationaryOpt0(SparseTensor &ground_truth, LowRankTensor &low_rank_tensor) 
    :
    ALS_Optimizer(ground_truth, low_rank_tensor),
    tensor_grid(ground_truth.tensor_grid),
    grid(ground_truth.tensor_grid.grid),
    dim(ground_truth.dim)
    {
    }

    void initialize_ground_truth_for_als() {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        // uint64_t R = low_rank_tensor.rank;

        ground_truth.check_tensor_invariants();
        ground_truth.redistribute_to_grid(tensor_grid);
        ground_truth.check_tensor_invariants();

        for(uint64_t i = 0; i < dim; i++) {
            indices.emplace_back();
            values.emplace_back();

            // Allgather factors into buffers and compute gram matrices
            DistMat1D &factor = low_rank_tensor.factors[i];

            //uint64_t row_count = tensor_grid.padded_row_counts[i] * world_size; 
            uint32_t row_start = factor.row_position * factor.padded_rows;
            //uint32_t row_end = (factor.row_position + 1) * factor.padded_rows;

            factor.compute_leverage_scores();

            uint64_t nnz = ground_truth.indices.shape[0];
            uint64_t proc_count = tensor_grid.grid.world_size;

            Buffer<int> prefix({(uint64_t) tensor_grid.dim}); 
            tensor_grid.grid.get_prefix_array(prefix);

            Buffer<uint64_t> send_counts({proc_count});
            std::fill(send_counts(), send_counts(proc_count), 0);
            Buffer<int> processor_assignments({nnz}); 

            int* proc_map = grid.row_order_to_procs[i].data();

            #pragma omp parallel
    {
            vector<uint64_t> send_counts_local(proc_count, 0);

            #pragma omp for
            for(uint64_t j = 0; j < nnz; j++) {
                uint64_t target_proc = proc_map[ground_truth.indices[j * dim + i] / factor.padded_rows]; 

                send_counts_local[target_proc]++;
                processor_assignments[j] = target_proc; 
            }

            for(uint64_t j = 0; j < proc_count; j++) {
                #pragma omp atomic 
                send_counts[j] += send_counts_local[j];
            }
    }

            alltoallv_matrix_rows(
                ground_truth.indices,
                processor_assignments,
                send_counts,
                indices[i],
                tensor_grid.grid.world
            );

            alltoallv_matrix_rows(
                ground_truth.values,
                processor_assignments,
                send_counts,
                values[i],
                tensor_grid.grid.world
            );

            #pragma omp parallel for
            for(uint64_t j = 0; j < indices[i].shape[0]; j++) {
                indices[i][j * dim + i] -= row_start;
                if(indices[i][j * dim + i] > factor.padded_rows) {
                    cout << "Error: " << indices[i][j * dim + i] << " " << factor.padded_rows << endl;
                    exit(1);
                }
            }

            lookups.emplace_back(
                make_unique<SortIdxLookup<uint32_t, double>>(
                    dim, i, indices[i](), values[i](), indices[i].shape[0]
                ));
        }


        // Only do this after we have constructed the parallel
        // representations
        for(uint64_t i = 0; i < dim; i++) {
            ground_truth.offsets[i] = tensor_grid.start_coords[i][tensor_grid.grid.coords[i]];
        }

        #pragma omp parallel for
        for(uint64_t i = 0; i < ground_truth.indices.shape[0]; i++) {
            for(uint64_t j = 0; j < dim; j++) {
                ground_truth.indices[i * dim + j] -= ground_truth.offsets[j];
            }
        }

        ground_truth.lookups.emplace_back(
            make_unique<SortIdxLookup<uint32_t, double>>(
                dim, 0, ground_truth.indices(), ground_truth.values(), ground_truth.indices.shape[0]
            )); 
    }

    void execute_ALS_step(uint64_t mode_to_leave, uint64_t J) {
        // Step 1: Gather the factor matrices
        vector<Buffer<double>> gathered_factors;

        uint64_t R = low_rank_tensor.rank; 
        for(uint64_t i = 0; i < dim; i++) {
            uint64_t row_count = tensor_grid.padded_row_counts[i] * grid.world_size;

            gathered_factors.emplace_back(
                Buffer<double>({row_count, low_rank_tensor.rank})
            );

            // Allgather factors into buffers and compute gram matrices
            DistMat1D &factor = low_rank_tensor.factors[i];
            Buffer<double> &factor_data = factor.data;

            MPI_Allgather(
                factor_data(),
                factor_data.shape[0] * R,
                MPI_DOUBLE,
                gathered_factors[i](),
                factor_data.shape[0] * R,
                MPI_DOUBLE,
                factor.ordered_world 
            );

            factor.compute_leverage_scores();
        }

        // Step 2: Gather and sample factor matrices 

        Buffer<uint32_t> samples({J, ground_truth.dim});
        Buffer<double> weights({J});
        vector<Buffer<uint32_t>> unique_row_indices; 

        gather_lk_samples_to_all(J, 
                mode_to_leave, 
                samples, 
                weights,        
                unique_row_indices);

        vector<Buffer<uint32_t>> compressed_row_indices;
        vector<Buffer<double>> factors_compressed;

        for(uint64_t i = 0; i < dim; i++) {
            compressed_row_indices.emplace_back();
            factors_compressed.emplace_back();

            if(i != mode_to_leave) {
                low_rank_tensor.factors[i].gather_row_samples(
                    unique_row_indices[i],
                    compressed_row_indices[i],
                    factors_compressed[i]);
            }
        }

        #pragma omp parallel for
        for(uint64_t j = 0; j < J; j++) {
            weights[j] = exp(weights[j]);
        }

        Buffer<uint32_t> samples_dedup;
        Buffer<double> weights_dedup;

        deduplicate_design_matrix(
            samples,
            weights,
            mode_to_leave, 
            samples_dedup,
            weights_dedup);

        Buffer<uint32_t> sample_compressed_map({samples_dedup.shape[0], dim});


        #pragma omp parallel
{ 
        for(uint64_t i = 0; i < dim; i++) {
            if(i != mode_to_leave) {
                continue;
            }

            uint32_t* start_range = compressed_row_indices[i]();
            uint32_t* end_range = compressed_row_indices[i](compressed_row_indices[i].shape[0]);

            #pragma omp for
            for(uint64_t j = 0; j < samples_dedup.shape[0]; j++) {
                //sample_compressed_map[j * dim + i] = 
                
                uint32_t* lb = std::lower_bound(
                    start_range,
                    end_range
                    samples_dedup[j * dim + i]);

                if(lb == end_range) {
                    cout << "Error!" << endl;
                }
                sample_compressed_map[j * dim + i] = (uint32_t) (lb - start_range);
            }
        }
}

        uint64_t sample_count_dedup = samples_dedup.shape[0];

        Buffer<double> design_matrix({sample_count_dedup, R});
        std::fill(design_matrix(), design_matrix(sample_count_dedup* R), 1.0);

        for(uint64_t i = 0; i < ground_truth.dim; i++) {
            if(i == mode_to_leave) {
                continue;
            }

            double *factor_data = gathered_factors[i]();

            #pragma omp parallel for
            for(uint64_t j = 0; j < sample_count_dedup; j++) {
                uint32_t idx = samples_dedup[j * ground_truth.dim + i];

                for(uint64_t r = 0; r < R; r++) {
                    design_matrix[j * R + r] *= factor_data[idx * R + r]; 
                }
            }
        }

        for(uint64_t j = 0; j < samples_dedup.shape[0]; j++) {
            for(uint64_t r = 0; r < R; r++) {
                design_matrix[j * R + r] *= sqrt(weights_dedup[j]);
            }
        }

        Buffer<double> design_gram({R, R});
        Buffer<double> design_gram_inv({R, R});

        compute_gram(design_matrix, design_gram);
        compute_pinv_square(design_gram, design_gram_inv, R);

        #pragma omp parallel for
        for(uint64_t j = 0; j < samples_dedup.shape[0]; j++) {
            for(uint64_t r = 0; r < R; r++) {
                design_matrix[j * R + r] *= sqrt(weights_dedup[j]);
            }
        }

        DistMat1D &target_factor = low_rank_tensor.factors[mode_to_leave];

        // Step 3: Compute the MTTKRP 
        Buffer<double> mttkrp_res({target_factor.data.shape[0], R}); 

        std::fill(mttkrp_res(), 
            mttkrp_res(mttkrp_res.shape[0] * R), 
            0.0);

        auto t = start_clock();
        nonzeros_iterated += lookups[mode_to_leave]->execute_spmm(
            samples_dedup, 
            design_matrix,
            mttkrp_res 
            );
        double elapsed = stop_clock_get_elapsed(t);

        cblas_dsymm(
            CblasRowMajor,
            CblasRight,
            CblasUpper,
            (uint32_t) mttkrp_res.shape[0],
            (uint32_t) R,
            1.0,
            design_gram_inv(),
            R,
            mttkrp_res(),
            R,
            0.0,
            target_factor.data(),
            R);

        target_factor.renormalize_columns(&(low_rank_tensor.sigma));
    }
};
