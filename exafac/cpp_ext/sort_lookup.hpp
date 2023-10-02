#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <random>
#include <execution>
#include <algorithm>
#include <numeric>

#include "random_util.hpp"
#include "common.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

typedef struct Triple {
    uint32_t r;
    uint32_t c;
    double value;
} Triple;

template<typename IDX_T, typename VAL_T>
class __attribute__((visibility("hidden"))) SortIdxLookup {
public:
  int N;
  int mode_to_leave;

  uint64_t nnz;
  IDX_T* idx_ptr;
  VAL_T* val_ptr;

  Buffer<IDX_T*> sort_idxs;
  Multistream_RNG ms_rng;

  SortIdxLookup(int N, 
                int mode_to_leave, 
                IDX_T* idx_ptr, 
                VAL_T* val_ptr, 
                uint64_t nnz,
                bool reorder
                ) 
  :
  sort_idxs({nnz}),
  ms_rng()
  {
    this->N = N;
    this->mode_to_leave = mode_to_leave;
    this->nnz = nnz;
    this->idx_ptr = idx_ptr;
    this->val_ptr = val_ptr;

    #pragma omp parallel for 
    for(uint64_t i = 0; i < nnz; i++) {
        sort_idxs[i] = idx_ptr + (i * N);
    }

    std::sort(std::execution::par_unseq, 
        sort_idxs(), 
        sort_idxs(nnz),
        [mode_to_leave, N](IDX_T* a, IDX_T* b) {
            for(int i = 0; i < N; i++) {
                if(i != mode_to_leave && a[i] != b[i]) {
                    return a[i] < b[i];
                }
            }
            return false;  
        });

    if(reorder) {
      Buffer<IDX_T> copy_idxs({nnz, (uint64_t) N});
      Buffer<VAL_T> copy_vals({nnz});

      #pragma omp parallel
{
      #pragma omp for
      for(uint64_t i = 0; i < nnz; i++) {
        for(uint64_t j = 0; j < (uint64_t) N; j++) {
          copy_idxs[i * N + j] = sort_idxs[i][j];
        }

        uint64_t diff = (uint64_t) (sort_idxs[i] - idx_ptr) / N;
        double value = val_ptr[diff];
        copy_vals[i] = value;
      }

      #pragma omp for
      for(uint64_t i = 0; i < nnz; i++) {
        for(uint64_t j = 0; j < (uint64_t) N; j++) {
          idx_ptr[i * N + j] = copy_idxs[i * N + j];
        }
        val_ptr[i] = copy_vals[i]; 
      }

      #pragma omp for 
      for(uint64_t i = 0; i < nnz; i++) {
          sort_idxs[i] = idx_ptr + (i * N);
      }
} 
    }
  }

  /*
  * Executes an SpMM with the tensor matricization tracked by this
  * lookup table.
  *
  * This function assumes that the output buffer has already been
  * initialized to zero. 
  */
  uint64_t execute_spmm(
      Buffer<IDX_T> &indices, 
      Buffer<double> &input,
      Buffer<double> &output
      ) {

    uint64_t J = indices.shape[0];
    uint64_t R = output.shape[1];

    int mode = this->mode_to_leave;
    int Nval = this->N;
    auto lambda_fcn = [mode, Nval](IDX_T* a, IDX_T* b) {
                for(int i = 0; i < Nval; i++) {
                    if(i != mode && a[i] != b[i]) {
                        return a[i] < b[i];
                    }
                }
                return false;  
            };

    uint64_t found_count = 0;

    #pragma omp parallel for reduction(+:found_count)
    for(uint64_t j = 0; j < J; j++) {
      uint64_t input_offset = j * R;
      IDX_T* buf = indices(j * N);

      std::pair<IDX_T**, IDX_T**> bounds = 
        std::equal_range(
            sort_idxs(), 
            sort_idxs(nnz),
            buf,
            lambda_fcn);

      bool found = false;
      if(bounds.first != sort_idxs(nnz)) {
        found = true;
        IDX_T* start = *(bounds.first);

        for(int i = 0; i < N; i++) {
            if(i != mode_to_leave && buf[i] != start[i]) {
                found = false;
            }
        }
      }

      if(found) {
        for(IDX_T** i = bounds.first; i < bounds.second; i++) {
          found_count++;
          IDX_T* nonzero = *i;
          uint64_t diff = (uint64_t) (nonzero - idx_ptr) / N;
          double value = val_ptr[diff];
          uint64_t output_offset = (nonzero[mode_to_leave]) * R;

          for(uint64_t k = 0; k < R; k++) {
            #pragma omp atomic 
            output[output_offset + k] += input[input_offset + k] * value; 
          }
        }
      }
    }
    return found_count;
  }

  /*
  * Here, workspace is a buffer of size 
  * (num_threads + 2) * output.shape[0] 
  * 
  */
  void parallel_sparse_transpose(
    Buffer<Triple> &in_buf,
    Buffer<Triple> &out_buf,
    Buffer<uint64_t> &workspace
  ) {
      Triple* in = in_buf.ptr;
      Triple* out = out_buf.ptr;

      int tid = omp_get_thread_num();
      int thread_count = (int) omp_get_num_threads();

      uint64_t output_size = workspace.shape[1];
      uint64_t* local_work = workspace(output_size * tid);
      std::fill(local_work, local_work + output_size, 0);

      uint64_t chunksize = (in_buf.shape[0] + thread_count - 1) / thread_count;
      uint64_t lb = min(chunksize * tid, in_buf.shape[0]);
      uint64_t ub = min(chunksize * (tid + 1), in_buf.shape[0]);

      for(uint64_t i = lb; i < ub; i++) {
        local_work[in[i].r]++;
      }
      #pragma omp barrier

      uint64_t* sum_workspace = workspace((uint64_t) thread_count * output_size);
      uint64_t* scan_output = workspace((uint64_t) (thread_count + 1) * output_size);

      #pragma omp for
      for(uint64_t i = 0; i < output_size; i++) {
        sum_workspace[i] = 0;
        for(uint64_t j = 0; j < (uint64_t) thread_count; j++) {
          sum_workspace[i] += workspace[j * output_size + i];
        }
      }

      // Can turn into a parallel prefix sum later 
      #pragma omp single
      {
        std::exclusive_scan(sum_workspace, sum_workspace + output_size, scan_output, 0);
      }

      #pragma omp for
      for(uint64_t i = 0; i < output_size; i++) {
        uint64_t accum = scan_output[i];
        for(uint64_t j = 0; j < (uint64_t) thread_count; j++) {
          uint64_t temp = workspace[j * output_size + i];
          workspace[j * output_size + i] = accum;
          accum += temp;
        }
      }

      for(uint64_t i = lb; i < ub; i++) {
        uint64_t pos = local_work[in[i].r]++;
        out[pos] = in[i];
      }
      #pragma omp barrier
  }

  uint64_t csr_based_spmm(
      Buffer<IDX_T> &indices, 
      Buffer<double> &input,
      Buffer<double> &output) {

    uint64_t J = indices.shape[0];
    uint64_t R = output.shape[1];

    int mode = this->mode_to_leave;
    int Nval = this->N;
    auto lambda_fcn = [mode, Nval](IDX_T* a, IDX_T* b) {
                for(int i = 0; i < Nval; i++) {
                    if(i != mode && a[i] != b[i]) {
                        return a[i] < b[i];
                    }
                }
                return false;  
            };

    uint64_t found_count = 0;

    Buffer<Triple> all_indices;
    Buffer<Triple> sorted_idxs;
    Buffer<uint64_t> thread_sample_counts;
    Buffer<uint64_t> offsets;
    Buffer<uint64_t> transpose_workspace; 

    #pragma omp parallel 
{
    int tid = omp_get_thread_num(); 
    uint64_t thread_count = (uint64_t) omp_get_num_threads();

    #pragma omp single
{
    thread_sample_counts.initialize_to_shape({thread_count});
    offsets.initialize_to_shape({thread_count+1});
    transpose_workspace.initialize_to_shape({thread_count + 2, output.shape[0]});
    std::fill(thread_sample_counts(), thread_sample_counts(thread_count), 0);
}
    vector<Triple> local_indices;

    #pragma omp for reduction(+:found_count)
    for(uint64_t j = 0; j < J; j++) {
      IDX_T* buf = indices(j * N);

      std::pair<IDX_T**, IDX_T**> bounds = 
        std::equal_range(
            sort_idxs(), 
            sort_idxs(nnz),
            buf,
            lambda_fcn);

      bool found = false;
      if(bounds.first != sort_idxs(nnz)) {
        found = true;
        IDX_T* start = *(bounds.first);

        for(int i = 0; i < N; i++) {
            if(i != mode_to_leave && buf[i] != start[i]) {
                found = false;
            }
        }
      }

      if(found) {
        for(IDX_T** i = bounds.first; i < bounds.second; i++) {
          found_count++;
          thread_sample_counts[tid]++;
          IDX_T* nonzero = *i;
          uint64_t diff = (uint64_t) (nonzero - idx_ptr) / N;
          double value = val_ptr[diff];
          uint32_t output_offset = (uint32_t) nonzero[mode_to_leave]; 

          local_indices.emplace_back();
          Triple &last = local_indices.back();
          last.r = output_offset;
          last.c = (uint32_t) j;
          last.value = value;
        }
      }
    }
    //cout << "Time: " << stop_clock_get_elapsed(t) << endl;

    #pragma omp single
{
    all_indices.initialize_to_shape({found_count});
    sorted_idxs.initialize_to_shape({found_count});

    // Prefix sum the thread counts to get the offsets
    std::exclusive_scan(
        thread_sample_counts(), 
        thread_sample_counts(thread_count), 
        offsets(), 0);
    offsets[thread_count] = found_count;
}

    uint64_t tid_offset = offsets[tid];
    for(uint64_t i = 0; i < local_indices.size(); i++) {
      all_indices[tid_offset + i] = local_indices[i];
    }

    #pragma omp barrier

    parallel_sparse_transpose(
      all_indices, 
      sorted_idxs,
      transpose_workspace
    );

    uint64_t chunksize = (found_count + thread_count - 1) / thread_count;
    uint64_t lb = min(chunksize * tid, found_count);
    uint64_t ub = min(chunksize * (tid + 1), found_count);

    while(lb != 0 &&  lb < found_count && sorted_idxs[lb].r == sorted_idxs[lb-1].r) {
      lb++;
    }
    while(ub < found_count && sorted_idxs[ub].r == sorted_idxs[ub-1].r) {
      ub++;
    }

    for(uint64_t i = lb; i < ub; i++) {
      Triple &triple = sorted_idxs[i];
      uint64_t output_offset = triple.r * R;
      uint64_t input_offset = triple.c * R;
      double value = triple.value;

      double* input_ptr = input(input_offset);
      double* output_ptr = output(output_offset);

      #pragma omp simd
      for(uint64_t k = 0; k < R; k++) {
        output_ptr[k] += input_ptr[k] * value; 
      }
    }
}

    return found_count;
  }

  double compute_2bmb(
      Buffer<double> &sigma, 
      vector<Buffer<double>> &U) {

      uint64_t R = U[0].shape[1];
      uint64_t j = mode_to_leave;

      double residual_normsq = 0.0;
      double value_sum = 0.0;
      #pragma omp parallel reduction(+: residual_normsq, value_sum)
{
      int thread_num = omp_get_thread_num();
      int total_threads = omp_get_num_threads();      

      uint64_t chunksize = (nnz + total_threads - 1) / total_threads;
      uint64_t lower_bound = min(chunksize * thread_num, nnz);
      uint64_t upper_bound = min(chunksize * (thread_num + 1), nnz);

      Buffer<double> partial_prod({R});

      for(uint64_t i = lower_bound; i < upper_bound; i++) {
        IDX_T* index = sort_idxs[i];

        uint64_t offset = (index - idx_ptr) / N; 
        bool recompute_partial_prod = false;
        if(i == lower_bound) {
          recompute_partial_prod = true;
        }
        else {
          IDX_T* prev_index = sort_idxs[i-1];
          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if((k != j) && (index[k] != prev_index[k])) {
              recompute_partial_prod = true;
            }
          }
        }

        if(recompute_partial_prod) {
          std::copy(sigma(), sigma(R), partial_prod()); 

          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if(k != j) {
              for(uint64_t u = 0; u < R; u++) {
                partial_prod[u] *= U[k][index[k] * R + u];
              }
            }
          }
        }

        double tensor_value = val_ptr[offset];

        double value = 0.0; 
        for(uint64_t u = 0; u < R; u++) {        
          value += partial_prod[u] * U[j][index[j] * R + u]; 
        }
        value_sum += value;
        residual_normsq += tensor_value * tensor_value - 2 * value * tensor_value; 
      }
}

      return residual_normsq;
  }

  void execute_exact_mttkrp(
      vector<Buffer<double>> &U, 
      Buffer<double> &mttkrp_res) {

      uint64_t R = U[0].shape[1];
      uint64_t j = mode_to_leave;

      #pragma omp parallel 
{
      int thread_num = omp_get_thread_num();
      int total_threads = omp_get_num_threads();      

      uint64_t chunksize = (nnz + total_threads - 1) / total_threads;
      uint64_t lower_bound = min(chunksize * thread_num, nnz);
      uint64_t upper_bound = min(chunksize * (thread_num + 1), nnz);

      Buffer<double> partial_prod({R});

      for(uint64_t i = lower_bound; i < upper_bound; i++) {
        IDX_T* index = sort_idxs[i];

        uint64_t offset = (index - idx_ptr) / N; 
        bool recompute_partial_prod = false;
        if(i == lower_bound) {
          recompute_partial_prod = true;
        }
        else {
          IDX_T* prev_index = sort_idxs[i-1];
          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if((k != j) && (index[k] != prev_index[k])) {
              recompute_partial_prod = true;
            }
          }
        }

        if(recompute_partial_prod) {
          std::fill(partial_prod(), partial_prod(R), 1.0); 

          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if(k != j) {
              for(uint64_t u = 0; u < R; u++) {
                partial_prod[u] *= U[k][index[k] * R + u];
              }
            }
          }
        }

        double tensor_value = val_ptr[offset];

        for(uint64_t u = 0; u < R; u++) {
          #pragma omp atomic 
          mttkrp_res[index[j] * R + u] += partial_prod[u] * tensor_value;
        }
      }
}
  }
};