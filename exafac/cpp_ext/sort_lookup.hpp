#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <random>
#include <execution>
#include <algorithm>
#include <numeric>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

template<typename IDX_T, typename VAL_T>
class SortIdxLookup {
public:
  int N;
  int mode_to_leave;

  uint64_t nnz;
  IDX_T* idx_ptr;
  VAL_T* val_ptr;

  Buffer<IDX_T*> sort_idxs;

  SortIdxLookup(int N, int mode_to_leave, IDX_T* idx_ptr, VAL_T* val_ptr, uint64_t nnz) 
  :
  sort_idxs({nnz})
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
  }

  void lookup_and_append(IDX_T r_idx, 
        double weight, 
        IDX_T* buf, 
        COOSparse<IDX_T, VAL_T> &res) {

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
            //found_count++;
            IDX_T* nonzero = *i;
            uint64_t diff = (uint64_t) (nonzero - idx_ptr) / N;
            VAL_T value = val_ptr[diff];

            res.rows.push_back(r_idx); 
            res.cols.push_back(nonzero[mode_to_leave]);
            res.values.push_back(weight * value);
        }
    }
  }
};