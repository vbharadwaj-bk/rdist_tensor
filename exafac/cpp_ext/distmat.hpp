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

    void compute_gram_local_slice(Buffer<double> &gram) {
        std::fill(gram(), gram(cols * cols), 0.0);
        if (true_row_count != 0) {
            Buffer<double> local_data_view({true_row_count, cols}, data());
            compute_gram(local_data_view, gram);
        }
    }

    void compute_gram_matrix(Buffer<double> &gram) {
        compute_gram_local_slice(gram);

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