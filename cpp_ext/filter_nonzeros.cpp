//cppimport
#include <cassert>
#include <fcntl.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <bits/stdc++.h>
#include <unordered_map>

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
  py::array_t<IDX_T> &offsets_py,
  py::array_t<IDX_T> &ranges_py,
  py::list &hashes_py) {

  NumpyArray<IDX_T> offsets(offsets_py);
  NumpyArray<IDX_T> ranges(ranges_py);
  NumpyList<uint64_t> hashes(hashes_py);

  int dim = ranges.info.shape[0];
  for(int j = 0; j < dim; j++) {
    // TODO: Need to template this out!
    uint32_t offset = offsets.ptr[j];
    for(uint32_t i = 0; i < ranges.ptr[j]; i++) {
      hashes.ptrs[j][i] = murmurhash2(offset + i, 0x9747b28c + j);
      //hashes.ptrs[j][i] = offset + i; 
    } 
  }
}

/*
 * This function builds and returns a sparse matrix.
 */
template<typename IDX_T, typename VAL_T>
COOSparse<IDX_T, VAL_T> sample_nonzeros(
      py::array_t<IDX_T> &idxs_mat_py, 
      py::array_t<IDX_T> &offsets_py, 
      py::list &mode_hashes_py,
      py::array_t<VAL_T> &values_py, 
      py::array_t<IDX_T> &sample_mat_py,
      py::array_t<double> &weights_py,
      int mode_to_leave) {

    COOSparse<IDX_T, VAL_T> gathered;
    NumpyArray<IDX_T> idxs_mat(idxs_mat_py); 
    NumpyArray<IDX_T> offsets(offsets_py); 
    NumpyList<uint64_t> mode_hashes(mode_hashes_py); 
    NumpyArray<VAL_T> values(values_py); 
    NumpyArray<IDX_T> sample_mat(sample_mat_py);
    NumpyArray<double> weights(weights_py); 

    uint64_t nnz = values.info.shape[0];

    // TODO: Add an assertion downcasting this!
    uint32_t num_samples = (uint32_t) sample_mat.info.shape[0];
    int dim = idxs_mat.info.shape[1];

    vector<int> counts(num_samples, 0);

    auto hash_fcn = [dim, mode_to_leave](IDX_T* const &ptr) { 
      uint64_t hash = 0;
      for(int j = 0; j < dim - 1; j++) {
        uint64_t offset = j < mode_to_leave ? j : j + 1;
        hash += murmurhash2(ptr[offset], 0x9747b28c + offset); 
      }
      return hash;
    }; 

    auto cmp_fcn = [dim, mode_to_leave](
      IDX_T* const &ptr1, 
      IDX_T* const &ptr2) {
      int eq = 1;
      for(int j = 0; j < dim - 1; j++) { 
        uint64_t offset = j < mode_to_leave ? j : j + 1;
        if(ptr1[offset] != ptr2[offset]) {
          eq = 0;
        }
      }
      return eq;
    }; 

    std::allocator<IDX_T*> alloc;
    unordered_map<IDX_T*, 
        uint32_t, 
        decltype(hash_fcn),
        decltype(cmp_fcn),
        decltype(alloc)>
        sample_map(num_samples, hash_fcn, cmp_fcn, alloc); 

    // Insert all items into our hashtable; we will use simple linear probing 

    auto start = start_clock();
    double elapsed = 0.0;
    int64_t count;

    for(uint32_t i = 0; i < num_samples; i++) {
      IDX_T* nz_ptr = sample_mat.ptr + i * dim;

      auto res = sample_map.insert(std::make_pair(nz_ptr, i));
      if(res.second) 
        counts[i] = 1; 
      else
        counts[res.first->second]++; 
    }

    for(int64_t i = 0; i < num_samples; i++) {
      weights.ptr[i] *= sqrt(counts[i]);
    }

    // Check all items in the larger set against the hash table

    for(uint64_t i = 0; i < nnz; i++) {
      IDX_T* nz_ptr = idxs_mat.ptr + i * dim; 
      auto res = sample_map.find(nz_ptr);
      if(res != sample_map.end()) {
        uint32_t val = res->second;
        gathered.rows.push_back(val);
        gathered.cols.push_back(nz_ptr[mode_to_leave]);
        gathered.values.push_back(values.ptr[i] * weights.ptr[val]); 
        count += val;
      }
    }

    elapsed += stop_clock_get_elapsed(start);
    cout << elapsed << endl;
    exit(1);

    return gathered;
}

template<typename IDX_T, typename VAL_T>
void sample_nonzeros_redistribute(
      py::array_t<IDX_T> idxs_mat_py,
      py::array_t<IDX_T> offsets_py,
      py::array_t<VAL_T> values_py, 
      py::array_t<IDX_T> sample_mat_py,
      py::list mode_hashes_py,
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
          idxs_mat_py,
          offsets_py,
          mode_hashes_py,
          values_py, 
          sample_mat_py,
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
cfg['extra_compile_args'] = ['-O3']
cfg['extra_link_args'] = ['-O3']
cfg['dependencies'] = ['common.h', 'tensor_alltoallv.h']
%>
*/
