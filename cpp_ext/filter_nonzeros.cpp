//cppimport
#include <cassert>
#include <fcntl.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <bits/stdc++.h>

#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

#include "common.h"
#include "tensor_alltoallv.h"

using namespace std;
namespace py = pybind11;


template<typename IDX_T>
void compute_mode_hashes(
  py::array_t<IDX_T> &ranges_py,
  py::list &hashes_py) {

  NumpyArray<IDX_T> ranges(ranges_py);
  NumpyList<uint64_t> hashes(hashes_py);

  for(int j = 0; j < ranges.length; j++) {
    // TODO: Need to template this out!
    for(uint32_t i = 0; i < ranges.ptr[j]; i++) {
      hashes.ptrs[j][i] = murmurhash2(i, 0x9747b28c + j);
    } 
  }
}

/*
 * 
 * This function builds and returns a sparse matrix.
 * 
 */
template<typename IDX_T, typename VAL_T>
COOSparse<IDX_T, VAL_T> sample_nonzeros(py::list &idxs_py, 
      py::array_t<VAL_T> &values_py, 
      py::list &sample_idxs_py,
      py::array_t<double> &weights_py,
      int mode_to_leave) {

    COOSparse<IDX_T, VAL_T> gathered;
    NumpyList<IDX_T> idxs(idxs_py); 
    NumpyArray<VAL_T> values(values_py); 
    NumpyList<IDX_T> sample_idxs(sample_idxs_py);
    NumpyArray<double> weights(weights_py); 

    uint64_t nnz = idxs.infos[0].shape[0];

    // TODO: Add an assertion downcasting this!
    int64_t num_samples = (int64_t) sample_idxs.infos[0].shape[0];
    int dim = idxs.length;

    vector<int> counts(num_samples, 0);

    if(sample_idxs.length != dim - 1) {
      cout << "Error, incorrect sample dimensions" << endl;
      exit(1);
    }

    double load_factor = 0.10;

    // lightweight hashtable that we can easily port to a GPU 
    uint64_t hashtbl_size = (uint64_t) (num_samples / load_factor);
    vector<int64_t> hashtbl(hashtbl_size, -1);
    int64_t* hashtbl_ptr = hashtbl.data();

    // Insert all items into our hashtable; we will use simple linear probing 

    for(int64_t i = 0; i < num_samples; i++) {
      uint64_t hash = 0;
      for(int j = 0; j < dim - 1; j++) {
        hash += murmurhash2(sample_idxs.ptrs[j][i], 0x9747b28c + j);
      }
      hash %= hashtbl_size;

      bool found = false;

      // Should replace with atomics to make thread-safe 
      while(hashtbl_ptr[hash] != -1l) {
        int64_t val = hashtbl[hash];
        found = true;
        for(int j = 0; j < dim - 1; j++) {
          if(sample_idxs.ptrs[j][i] != sample_idxs.ptrs[j][val]) {
            found = false;
          }
        }
        if(found) {
          break;
        }
        hash = (hash + 1) % hashtbl_size;
      }
      if(! found) {
        hashtbl[hash] = i;
      }
      counts[hashtbl[hash]]++;
    }
 
    for(int64_t i = 0; i < num_samples; i++) {
      weights.ptr[i] *= sqrt(counts[i]);
    }

    // Check all items in the larger set against the hash table

    //#pragma omp parallel
    {

    //#pragma omp for
    for(uint64_t i = 0; i < nnz; i++) {
      // If we knew the dimension ahead of time, this loop could be compiled down. 
      uint64_t hash = 0;
      for(int j = 0; j < dim; j++) {
        if(j < mode_to_leave)
          hash += murmurhash2(idxs.ptrs[j][i], 0x9747b28c + j); 
        if(j > mode_to_leave)
          hash += murmurhash2(idxs.ptrs[j][i], 0x9747b28c + j-1);
      }
      hash %= hashtbl_size;

      int64_t val;

      // TODO: This loop is unsafe, need to fix it! 
      while(true) {
        val = hashtbl[hash];
        if(val == -1) {
          break;
        }
        else {
          bool eq = true;
          for(int j = 0; j < dim; j++) {
            if((j < mode_to_leave && idxs.ptrs[j][i] != sample_idxs.ptrs[j][val])
              || 
              (j > mode_to_leave && idxs.ptrs[j][i] != sample_idxs.ptrs[j-1][val]) 
            ) {
              eq = false;
            } 
          }
          if(eq) {
            break;
          }
        }
        hash = (hash + 1) % hashtbl_size;
      }

      if(val != -1) {
          //found_val[ctr] = (uint32_t) val;
          //found_i[ctr] = i;
          //ctr++;
          gathered.rows.push_back(val);
          gathered.cols.push_back(idxs.ptrs[mode_to_leave][i]);
          gathered.values.push_back(values.ptr[i] * weights.ptr[val]);
      }
    }

    } 

    return gathered;
}

template<typename IDX_T, typename VAL_T>
void sample_nonzeros_redistribute(
      py::list idxs_py, 
      py::array_t<VAL_T> values_py, 
      py::list sample_idxs_py,
      py::array_t<double> weights_py,
      int mode_to_leave,
      uint64_t row_divisor,
      py::array_t<int> row_order_to_proc_py,  
      py::list recv_idx_py,
      py::list recv_values_py,
      py::function allocate_recv_buffers 
      ) {

      COOSparse<IDX_T, VAL_T> gathered = 
        sample_nonzeros<IDX_T, VAL_T>(
          idxs_py, 
          values_py, 
          sample_idxs_py,
          weights_py,
          mode_to_leave);

      uint64_t nnz = gathered.rows.size(); 

      vector<IDX_T*> coords;
      coords.push_back(gathered.rows.data());
      coords.push_back(gathered.cols.data());
      IDX_T* col_ptr = gathered.cols.data(); 

      NumpyList<IDX_T> coords_wrapped(coords);
      NumpyArray<VAL_T> values_wrapped(gathered.values.data());
      NumpyArray<int> row_order_to_proc(row_order_to_proc_py);

      uint64_t proc_count = row_order_to_proc.info.shape[0]; 

      vector<int> processor_assignments(nnz, 0);
      int* assignment_ptr = processor_assignments.data();
      vector<uint64_t> send_counts(proc_count, 0);

      for(uint64_t i = 0; i < nnz; i++) {
          int processor = row_order_to_proc.ptr[col_ptr[i] / row_divisor];
          assignment_ptr[i] = processor; 
          send_counts[processor]++;
      }

      tensor_alltoallv(
          2, 
          proc_count, 
          nnz, 
          coords_wrapped, 
          values_wrapped, 
          processor_assignments,
          send_counts, 
          recv_idx_py, 
          recv_values_py, 
          allocate_recv_buffers 
          );
} 

PYBIND11_MODULE(filter_nonzeros, m) {
  py::class_<COOSparse<uint32_t, double>>(m, "COOSparse"); 
  //  .def("print_contents", &COOSparse::print_contents);

  //m.def("sample_nonzeros", &sample_nonzeros);

  m.def("sample_nonzeros_redistribute_u32_double", &sample_nonzeros_redistribute<uint32_t, double>);
  //m.def("sample_nonzeros_redistribute_u64_double", &sample_nonzeros_redistribute<uint64_t, double>);

  m.def("compute_mode_hashes_u32", &compute_mode_hashes<uint32_t>);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-fopenmp', '-O3']
cfg['extra_link_args'] = ['-openmp', '-O3']
cfg['dependencies'] = ['common.h', 'tensor_alltoallv.h']
%>
*/
