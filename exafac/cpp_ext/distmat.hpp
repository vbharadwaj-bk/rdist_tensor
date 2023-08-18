#pragma once

#include <iostream>
#include <cblas.h>
#include <lapacke.h>
#include <cmath>
#include <algorithm>

#include "grid.hpp"
#include "random_util.hpp"
#include "common.h"

using namespace std;

class __attribute__((visibility("hidden"))) DistMat1D {
public:
    uint64_t cols;
    TensorGrid &tensor_grid;
    Grid &grid; 
    uint64_t slice_dim;
    uint64_t rows;

    uint64_t padded_rows;
    uint64_t row_position;

    uint64_t true_row_count;

    Buffer<double> data;

    // Related to leverage-score sampling
    Buffer<double> leverage_scores;

    // End leverage score-related variables

    MPI_Comm ordered_world; 

    DistMat1D(uint64_t cols, 
        TensorGrid &tensor_grid, uint64_t slice_dim) 
        : 
        tensor_grid(tensor_grid),
        grid(tensor_grid.grid)
        {

        this->slice_dim = slice_dim;
        this->cols = cols; 
        this->rows = tensor_grid.tensor_dims[slice_dim];

        padded_rows = tensor_grid.padded_row_counts[slice_dim]; 
        row_position = grid.row_positions[slice_dim][grid.rank];

        if(row_position * padded_rows > rows) {
            true_row_count = 0;
        } else {
            true_row_count = min(padded_rows, rows - row_position * padded_rows);
        }
        data.initialize_to_shape({padded_rows, cols});
        leverage_scores.initialize_to_shape({padded_rows});

        // Create a communicator that puts all slices in order 
        MPI_Group world_group; 
        MPI_Comm_group(grid.world, &world_group);

        MPI_Group ordered_group;
        MPI_Group_incl(world_group, 
                    grid.world_size, 
                    grid.row_order_to_procs[slice_dim].data(),
                    &ordered_group);

        MPI_Comm_create_group(grid.world, ordered_group, 0, &ordered_world);
    }

    void compute_gram_matrix(Buffer<double> &gram) {
        std::fill(gram(), gram(cols * cols), 0.0);
        if (true_row_count != 0) {
            Buffer<double> local_data_view({true_row_count, cols}, data());
            //compute_gram(local_data_view, gram);
            cblas_dgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                cols,
                cols,
                true_row_count,
                1.0,
                local_data_view(),
                cols,
                local_data_view(),
                cols,
                0.0,
                gram(),
                cols
                );
        }
        
        MPI_Allreduce(MPI_IN_PLACE,
            gram(),
            cols * cols,
            MPI_DOUBLE,
            MPI_SUM,
            grid.world
            );
    }

    void renormalize_columns(Buffer<double>* col_norms_out) {
        Buffer<double> col_norms({cols});

        std::fill(col_norms(), col_norms(cols), 0.0);

        #pragma omp parallel
{ 
        Buffer<double> col_norms_local({cols});
        std::fill(col_norms_local(), col_norms_local(cols), 0.0);

        #pragma omp for 
        for(uint64_t i = 0; i < true_row_count; i++) {
            for(uint64_t j = 0; j < cols; j++) {
                col_norms_local[j] += data[i * cols + j] * data[i * cols + j];
            }
        }

        for(uint64_t i = 0; i < cols; i++) {
            #pragma omp atomic
            col_norms[i] += col_norms_local[i];
        }

        #pragma omp barrier

        #pragma omp single
        {
            MPI_Allreduce(MPI_IN_PLACE,
                col_norms(),
                cols,
                MPI_DOUBLE,
                MPI_SUM,
                grid.world
                ); 
        }


        #pragma omp for
        for(uint64_t i = 0; i < cols; i++) {
            col_norms[i] = sqrt(col_norms[i]);
        }

        // TODO: Should handle division by zero around here 

        #pragma omp for 
        for(uint64_t i = 0; i < true_row_count; i++) {
            for(uint64_t j = 0; j < cols; j++) {
                data[i * cols + j] /= col_norms[j]; 
            }
        }
}
            
        if(col_norms_out != nullptr) {
            std::copy(col_norms(), col_norms(cols), (*col_norms_out)());
        }
    }

    void initialize_deterministic() {
        /*cout << "Rank " << grid.rank << " offset is " << row_position * padded_rows 
        << " with true row count " << true_row_count << endl;*/

        #pragma omp parallel for collapse(2)
        for(uint64_t i = 0; i < true_row_count; i++) {
            for(uint64_t j = 0; j < cols; j++) {
                data[i * cols + j] = (double) cos((i + row_position * padded_rows) * cols + j);
            }
        }
        //MPI_Barrier(grid.world);
    }

    void initialize_gaussian_random() {
        Multistream_RNG rng;
        std::normal_distribution<double> normal_dist(0.0, 1.0);

        #pragma omp parallel
{
        int thread_num = omp_get_thread_num();

        #pragma omp for collapse(2)
        for(uint64_t i = 0; i < true_row_count; i++) {
            for(uint64_t j = 0; j < cols; j++) {
                data[i * cols + j] = normal_dist(rng.par_gen[thread_num]); 
            }
        }
}
    }

    void compute_leverage_scores() {
        Buffer<double> gram({cols, cols});
        Buffer<double> gram_pinv({cols, cols});

        compute_gram_matrix(gram);
        compute_pinv_square(gram, gram_pinv, cols);
        compute_DAGAT(data(), gram_pinv(), leverage_scores(), true_row_count, cols);
    }

    void draw_leverage_score_samples(uint64_t J, 
            Buffer<uint32_t> &sample_idxs, 
            Buffer<double> &log_weights,
            Buffer<uint32_t> &unique_local_samples) {
        Consistent_Multistream_RNG global_rng(MPI_COMM_WORLD);
        Multistream_RNG local_rng;

        double leverage_sum = std::accumulate(leverage_scores(), leverage_scores(true_row_count), 0.0);
        Buffer<double> leverage_sums({(uint64_t) grid.world_size});
        Buffer<uint64_t> samples_per_process({(uint64_t) grid.world_size}); 
        MPI_Allgather(&leverage_sum,
            1,
            MPI_DOUBLE,
            leverage_sums(),
            1,
            MPI_DOUBLE,
            grid.world
            );

        double total_leverage_weight = std::accumulate(leverage_sums(), leverage_sums(grid.world_size), 0.0);

        // Should cache the distributions 
        std::discrete_distribution<uint32_t> local_dist(leverage_scores(), leverage_scores(true_row_count));
        std::discrete_distribution<uint32_t> global_dist(leverage_sums(), leverage_sums(grid.world_size));

        // Not multithreaded, can thread if this becomes the bottleneck. 
        std::fill(samples_per_process(), samples_per_process(grid.world_size), 0);

        for(uint64_t j = 0; j < J; j++) {
            uint64_t sample = global_dist(global_rng.par_gen[0]);
            samples_per_process[sample]++; 
        }

        uint64_t local_samples = samples_per_process[grid.rank];

        Buffer<uint32_t> sample_idxs_local({local_samples});
        Buffer<double> sample_weights_local({local_samples});

        std::fill(sample_weights_local(), sample_weights_local(local_samples), 0.0);

        uint32_t offset = row_position * padded_rows;

        for(uint64_t i = 0; i < local_samples; i++) {
            uint32_t sample = local_dist(local_rng.par_gen[0]);
            sample_idxs_local[i] = offset + sample; 

            // Need to do some more reweighting here, fine for now 
            sample_weights_local[i] += log(total_leverage_weight) - log(leverage_scores[sample]);
        }

        allgatherv_buffer(sample_idxs_local, sample_idxs, MPI_COMM_WORLD);
        allgatherv_buffer(sample_weights_local, log_weights, MPI_COMM_WORLD);

        // Get a deduplicated list of unique samples per process
        // Need to change the execution policy to parallel 
        std::sort(sample_idxs_local(), sample_idxs_local(local_samples));
        uint32_t* end_unique = std::unique(sample_idxs_local(), sample_idxs_local(local_samples));

        uint32_t num_unique = end_unique - sample_idxs_local();
        unique_local_samples.initialize_to_shape({num_unique});

        std::copy(sample_idxs_local(), sample_idxs_local(num_unique), unique_local_samples());
    }

    void gather_row_samples(
                Buffer<uint32_t> &indices_local,  
                Buffer<uint32_t> &indices_gathered,
                Buffer<double> &rows_gathered) {
            
        uint64_t local_samples = indices_local.shape[0];
        Buffer<double> rows_local({local_samples, cols});


        uint32_t offset = row_position * padded_rows;


        //#pragma omp parallel for
        for(uint64_t i = 0; i < local_samples; i++) {
            uint32_t sample = indices_local[i] - offset;

            for(uint64_t j = 0; j < cols; j++) {
                rows_local[i * cols + j] = data[sample * cols + j];
            }
        }

        allgatherv_buffer(indices_local, indices_gathered, ordered_world);
        allgatherv_buffer(rows_local, rows_gathered, ordered_world);
    }
};