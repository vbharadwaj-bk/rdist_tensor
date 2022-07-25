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
    int64_t num_samples = (int64_t) sample_mat.info.shape[0];
    int dim = idxs_mat.info.shape[1];

    vector<int> counts(num_samples, 0);

    /*if(sample_idxs.length != dim - 1) {
      cout << "Error, incorrect sample dimensions" << endl;
      exit(1);
    }*/

    double load_factor = 0.33;

    // lightweight hashtable that we can easily port to a GPU 
    uint64_t hashtbl_size = (uint64_t) (num_samples / load_factor);
    vector<int64_t> hashtbl(hashtbl_size, -1);
    int64_t* hashtbl_ptr = hashtbl.data();

    // Insert all items into our hashtable; we will use simple linear probing 

    //auto start = start_clock();
    //double elapsed = 0.0;

    for(int64_t i = 0; i < num_samples; i++) {
      uint64_t hash = 0;
      IDX_T* nz_ptr = sample_mat.ptr + i * dim;
      for(int j = 0; j < dim - 1; j++) {
        uint64_t offset = j < mode_to_leave ? j : j + 1;
        hash += murmurhash2(nz_ptr[offset], 0x9747b28c + offset); 
      }
      hash %= hashtbl_size;

      bool found = false;
      while(hashtbl_ptr[hash] != -1l) {
        int64_t val = hashtbl[hash];
        IDX_T* ref_ptr = sample_mat.ptr + val * dim;
        found = true;
        for(int j = 0; j < dim; j++) {
          if(nz_ptr[j] != ref_ptr[j]) {
            found = false;
          }
        }
        if(found) {
          break;
        }
        hash++;
        if(hash > hashtbl_size)
          hash -= hashtbl_size;
      }
      if(! found) {
        hashtbl[hash] = i;
      }
      counts[hashtbl[hash]]++;
    }
 
    for(int64_t i = 0; i < num_samples; i++) {
      weights.ptr[i] *= sqrt(counts[i]);
    }

    uint64_t count = 0;

    /*
    unique_ptr<IDX_T*[]> idx_dptrs(new IDX_T*[dim-1]);
    for(int j = 0; j < dim; j++) {
      if(j < mode_to_leave)
        idx_dptrs[j] = idxs.ptrs[j];
      if(j > mode_to_leave) 
        idx_dptrs[j-1] = idxs.ptrs[j];
    }
    */

    // Check all items in the larger set against the hash table
    for(uint64_t i = 0; i < nnz; i++) {
      IDX_T* nz_ptr = idxs_mat.ptr + i * dim; 

      // If we knew the dimension ahead of time, this loop could be compiled down. 
      uint64_t hash = 0;
      for(int j = 0; j < dim - 1; j++) {
          uint64_t offset = j < mode_to_leave ? j : j + 1;
          uint64_t val = murmurhash2(nz_ptr[offset], 0x9747b28c + offset); 
          hash += val;
      }
      hash %= hashtbl_size;

      int64_t val;

      val = hash;
      // TODO: This loop is unsafe, need to fix it! 
      while(true) {
        val = hashtbl[hash];
        if(val == -1) {
          break;
        }
        else {
          IDX_T* ref_ptr = sample_mat.ptr + val * dim;
          int eq = 1;
          for(int j = 0; j < dim-1; j++) {
            uint64_t offset = j < mode_to_leave ? j : j + 1;
            eq = eq && (nz_ptr[offset] == ref_ptr[offset]);
          } 
          if(eq) 
            break;
        }
        hash++;
        if(hash > hashtbl_size)
          hash -= hashtbl_size;
        
        count++;
      }

      if(val != -1) {
        gathered.rows.push_back(val);
        gathered.cols.push_back(nz_ptr[mode_to_leave]);
        gathered.values.push_back(values.ptr[i] * weights.ptr[val]);
      }
    }

    //elapsed += stop_clock_get_elapsed(start);
    //cout << elapsed << endl;

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
