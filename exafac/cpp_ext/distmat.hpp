#pragma once

#include <iostream>
#include <cblas.h>
#include <lapacke.h>
#include <cmath>

#include "grid.hpp"
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

    unique_ptr<Buffer<double>> data;

    Buffer<uint64_t> proc_to_row_order;
    Buffer<uint64_t> row_order_to_proc;

    DistMat1D(uint64_t cols, 
        TensorGrid &tensor_grid, uint64_t slice_dim) 
        : 
        tensor_grid(tensor_grid),
        grid(tensor_grid.grid),
        proc_to_row_order({(uint64_t) grid.world_size}),
        row_order_to_proc({(uint64_t) grid.world_size}) 
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
        data.reset(new Buffer<double>({padded_rows, cols}));

        MPI_Allgather(&row_position,
            1,
            MPI_UINT64_T,
            proc_to_row_order(),
            1,
            MPI_UINT64_T,
            grid.world 
            );

        for(int i = 0; i < grid.world_size; i++) {
            row_order_to_proc[proc_to_row_order[i]] = i;
        }
    }

    void compute_gram_matrix(Buffer<double> &gram) {
        Buffer<double> &data = *(this->data);
        if (true_row_count == 0) {
            std::fill(gram(), gram(cols * cols), 0.0);
        } else {
            cblas_dgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                cols,
                cols,
                true_row_count,
                1.0,
                data(),
                cols,
                data(),
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
        Buffer<double> &data = *(this->data);

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
        Buffer<double> &data = *(this->data);
        /*cout << "Rank " << grid.rank << " offset is " << row_position * padded_rows 
        << " with true row count " << true_row_count << endl;*/

        for(uint64_t i = 0; i < true_row_count; i++) {
            for(uint64_t j = 0; j < cols; j++) {
                data[i * cols + j] = (double) cos((i + row_position * padded_rows) * cols + j);
            }
        }
        //MPI_Barrier(grid.world);
    }

    void initialize_gaussian_random() {
        // TODO: Implement!
    }
};