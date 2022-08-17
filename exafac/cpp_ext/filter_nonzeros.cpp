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
#include "hashing.h"
#include "sparsehash/dense_hash_map"
#include "cuckoofilter/src/cuckoofilter.h"
#include "tensor_alltoallv.h"
#include "fks_hash.hpp"

using namespace std;
using namespace cuckoofilter;
namespace py = pybind11;

template<typename IDX_T>
struct TupleHasher 
{
public:
  int mode_to_leave;
  int dim;

  TupleHasher(int mode_to_leave, int dim) {
    this->mode_to_leave = mode_to_leave;
    this->dim = dim;
  }

  uint32_t operator()(IDX_T* const &ptr) const
  {
    if(ptr != nullptr) {
      //uint32_t pre_hash = MurmurHash3_x86_32(ptr, mode_to_leave * sizeof(IDX_T), 0x9747b28c);

      uint32_t pre_hash = MurmurHash3_x86_32(ptr, 12, 0x9747b28c);

      uint32_t post_hash = MurmurHash3_x86_32(ptr + mode_to_leave + 1, 
          sizeof(IDX_T) * (dim - mode_to_leave - 1), 0x9747b28c);

      return pre_hash + post_hash;
    }
    else {
      return 0;
    }
  }
};

template<typename IDX_T>
struct TupleEqual 
{
public:
  int dim, mode_to_leave;
  TupleEqual(int mode_to_leave, int dim) {
    this->dim = dim;
    this->mode_to_leave = mode_to_leave;
  }

  bool operator()(IDX_T* const &s1, uint32_t* const &s2) const
  {
      if(s1 == nullptr || s2 == nullptr) {
        return s1 == s2;
      }
      else {
        auto res = (! memcmp(s1, s2, sizeof(IDX_T) * mode_to_leave)) 
            && (! memcmp(
            s1 + mode_to_leave + 1,
            s2 + mode_to_leave + 1,
            sizeof(IDX_T) * (dim - mode_to_leave - 1)
        ));

        return res;
      }
  }
};

template<typename IDX_T, typename VAL_T>
class HashIdxLookup {
public:
  int dim;
  int mode_to_leave;

  unique_ptr<google::dense_hash_map< IDX_T*, 
                          uint64_t,
                          TupleHasher<IDX_T>,
                          TupleEqual<IDX_T>>> lookup_table;

  vector<vector<pair<IDX_T, VAL_T>>> storage;
  uint64_t num_buckets;

  TupleHasher<IDX_T> hasher;
  TupleEqual<IDX_T> equality_checker;

  HashIdxLookup(int dim, int mode_to_leave, IDX_T* idx_ptr, VAL_T* val_ptr, uint64_t nnz) :
    hasher(mode_to_leave, dim), equality_checker(mode_to_leave, dim) {

    this->dim = dim;
    this->num_buckets = 0;

    lookup_table.reset(new google::dense_hash_map< IDX_T*, 
                          uint64_t,
                          TupleHasher<IDX_T>,
                          TupleEqual<IDX_T>>(nnz, hasher, equality_checker));

    lookup_table->set_empty_key(nullptr);

    for(uint64_t i = 0; i < nnz; i++) {
      IDX_T* idx = idx_ptr + i * dim;
      VAL_T val = val_ptr[i];

      auto res = lookup_table->find(idx);

      uint64_t bucket;
      if(res == lookup_table->end()) {
        bucket = num_buckets++;
        storage.emplace_back();
        lookup_table->insert(make_pair(idx, bucket));
      }
      else {
        bucket = res->second; 
      }
      storage[bucket].emplace_back(make_pair(idx[mode_to_leave], val));
    }
  }

  void lookup_and_append(IDX_T r_idx, IDX_T* buf, COOSparse<IDX_T, VAL_T> &res) {
    auto pair_loc = lookup_table->find(buf);
    if(res != lookup_table->end()) {
      vector<pair<IDX_T, VAL_T>> pairs = storage[pair_loc->second];
      for(auto it = pairs.start(); it != pairs.end(); it++) {
        res.rows.push_back(r_idx);
        res.cols.push_back(it->first);
        res.vals.push_back(it->second);
      }  
    }
  }
};

template<typename IDX_T, typename VAL_T>
class TensorSlicer {
public:
  vector<HashIdxLookup<IDX_T, VAL_T>> lookups;

  TensorSlicer(py::array_t<IDX_T> idxs_py, py::array_t<VAL_T> vals_py) {
    NumpyArray<IDX_T> idxs(idxs_py);
    NumpyArray<VAL_T> vals(vals_py);

    int dim = idxs.info.shape[1];
    uint64_t nnz = vals.info.shape[0];

    for(int i = 0; i < dim; i++) {
      lookups.emplace_back(dim, i, idxs.ptr, vals.ptr, nnz);
    }
  }

  void lookup_and_append(IDX_T r_idx, IDX_T* buf, int mode_to_leave, COOSparse<IDX_T, VAL_T> &res) {
    lookups[mode_to_leave].lookup_and_append(r_idx, buf, res); 
  }
};

template<typename IDX_T>
struct CuckooHash 
{
public:
  int num_bytes; 
  CuckooHash(int num_bytes) {
    //this->num_bytes = num_bytes;
    this->num_bytes = num_bytes;
  }

  uint64_t operator()(IDX_T* const &ptr) const
  {
      /*uint64_t buf[2];
      MurmurHash3_x64_128 ( ptr, num_bytes, 0x9747b28c, &buf);
      return buf[0];*/
      uint64_t h1 = MurmurHash3_x86_32 ( ptr, num_bytes, 0x9747b28c); 
      h1 = (h1 << 32) + MurmurHash3_x86_32 ( ptr, num_bytes, 0x9793b15c); 
      return h1;
  }
};

/*
 * This function builds and returns a sparse matrix.
 */
template<typename IDX_T, typename VAL_T>
COOSparse<IDX_T, VAL_T> sample_nonzeros(
      py::array_t<IDX_T> &idxs_mat_py, 
      py::array_t<IDX_T> &offsets_py, 
      py::array_t<VAL_T> &values_py, 
      py::array_t<IDX_T> &sample_mat_py,
      py::array_t<double> &weights_py,
      int mode_to_leave,
      int dim) {
    COOSparse<IDX_T, VAL_T> gathered;
    NumpyArray<IDX_T> idxs_mat(idxs_mat_py); 
    NumpyArray<IDX_T> offsets(offsets_py); 
    NumpyArray<VAL_T> values(values_py); 
    NumpyArray<IDX_T> sample_mat(sample_mat_py);
    NumpyArray<double> weights(weights_py); 

    uint64_t nnz = values.info.shape[0];

    // TODO: Add an assertion downcasting this!
    uint32_t num_samples = (uint32_t) sample_mat.info.shape[0];

    vector<int> counts(num_samples, 0);
    uint32_t num_bytes = dim * sizeof(IDX_T);

    FKSHash fastmap(sample_mat.ptr, 
                dim, 
                mode_to_leave, 
                num_samples, 
                45);

    CuckooFilter<IDX_T*, 
        12, 
        SingleTable, 
        CuckooHash<IDX_T>> filter(num_samples, num_bytes);

    // Insert all items into our hashtable; we will use simple linear probing 
    //auto start = start_clock();
    int64_t count = 0;

    for(uint32_t i = 0; i < num_samples; i++) {
      IDX_T* nz_ptr = sample_mat.ptr + i * dim;
      filter.Add(nz_ptr);
    }

    // Check all items in the larger set against the hash table
    double elapsed = 0.0;
    auto start = start_clock();

    for(uint64_t i = 0; i < nnz; i++) {
      IDX_T* nz_ptr = idxs_mat.ptr + i * dim; 
      IDX_T temp = nz_ptr[mode_to_leave];
      nz_ptr[mode_to_leave] = 0;

      //if(true) {
      if(filter.Contain(nz_ptr) == cuckoofilter::Ok) {
        count++;
        uint32_t val = fastmap.lookup_careful(nz_ptr, dim);

        if(val != num_samples) {
          gathered.rows.push_back(val);
          gathered.cols.push_back(temp);
          gathered.values.push_back(values.ptr[i] * weights.ptr[val]); 
        }
      }
      nz_ptr[mode_to_leave] = temp;
    }

    elapsed += stop_clock_get_elapsed(start);

    return gathered;
}

template<typename IDX_T, typename VAL_T>
void sample_nonzeros_redistribute(
      py::object sampler,
      py::array_t<IDX_T> sample_mat_py,
      py::array_t<double> weights_py,
      int mode_to_leave,
      uint64_t row_divisor,
      py::array_t<int> row_order_to_proc_py,  
      py::list recv_idx_py,
      py::list recv_values_py,
      py::function allocate_recv_buffers
      ) {

      py::array_t<IDX_T> idxs_mat_py = sampler.attr("idxs_mat").cast<py::array_t<IDX_T>>();
      py::array_t<IDX_T> offsets_py = sampler.attr("offsets").cast<py::array_t<IDX_T>>();
      py::array_t<VAL_T> values_py = sampler.attr("values").cast<py::array_t<VAL_T>>(); 

      NumpyArray<IDX_T> idxs_mat(idxs_mat_py); 
      int dim = idxs_mat.info.shape[1];

      COOSparse<IDX_T, VAL_T> gathered;

      gathered = sample_nonzeros<IDX_T, VAL_T>(
        idxs_mat_py,
        offsets_py,
        values_py, 
        sample_mat_py,
        weights_py,
        mode_to_leave,
        dim); 

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

  py::class_<TensorSlicer<uint32_t, double>>(m, "TensorSlicer")
    .def(py::init<py::array_t<uint32_t>, py::array_t<double>>());
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-O3', '-finline-limit=1000']
cfg['extra_link_args'] = ['-O3', '-L/global/cfs/projectdirs/m1982/vbharadw/rdist_tensor/exafac/cpp_ext/cuckoofilter']
cfg['dependencies'] = ['common.h', 'tensor_alltoallv.h', 'hashing.h', 'cuckoofilter/src/cuckoofilter.h', 'fks_hash.hpp', 'primality.hpp']
cfg['libraries'] = ['cuckoofilter']
%>
*/
