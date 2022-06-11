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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

#include "common.h"

using namespace std;
namespace py = pybind11;

/*
 * Assumptions that this function makes:
 * 
 * This function builds and returns a sparse matrix 
 */
COOSparse sample_nonzeros(py::list idxs_py, 
      py::array_t<double> values_py, 
      py::list sample_idxs_py,
      py::array_t<double> weights_py,
      int mode_to_leave) {
    COOSparse gathered;
    NumpyList<unsigned long long> idxs(idxs_py); 
    NumpyArray<double> values(values_py); 
    NumpyList<unsigned long long> sample_idxs(sample_idxs_py);
    NumpyArray<double> weights(weights_py); 

    unsigned long long nnz = idxs.infos[0].shape[0];

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
    int64_t hashtbl_size = (int64_t) (num_samples / load_factor);
    vector<int64_t> hashtbl(hashtbl_size, -1);
    int64_t* hashtbl_ptr = hashtbl.data();

    vector<unsigned long long> hbuf(dim - 1, 0);
    unsigned long long* hbuf_ptr = hbuf.data();
    int hbuf_len = 8 * (dim - 1);

    // Insert all items into our hashtable; we will use simple linear probing 

    for(int64_t i = 0; i < num_samples; i++) {
      for(int j = 0; j < dim - 1; j++) {
        hbuf_ptr[j] = sample_idxs.ptrs[j][i];
      }

      unsigned long long hash = murmurhash2(hbuf_ptr, hbuf_len, 0x9747b28c) % hashtbl_size;

      bool found = false;

      // Should replace with atomics to make thread-safe 
      while(hashtbl_ptr[hash] != -1l) {
        unsigned long long val = hashtbl[hash];
        found = true;
        for(int j = 0; j < dim - 1; j++) {
          if(hbuf_ptr[j] != sample_idxs.ptrs[j][val]) {
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

    for(unsigned long long i = 0; i < nnz; i++) {
      // If we knew the dimension ahead of time, this loop could be compiled down. 
      for(int j = 0; j < dim; j++) {
        if(j < mode_to_leave) {
          hbuf_ptr[j] = idxs.ptrs[j][i];
        }
        if(j > mode_to_leave) {
          hbuf_ptr[j - 1] = idxs.ptrs[j][i];
        }
      }

      unsigned long long hash = murmurhash2(hbuf_ptr, hbuf_len, 0x9747b28c) % hashtbl_size;
      int64_t val;

      // TODO: This loop is unsafe, need to fix it! 
      while(true) {
        val = hashtbl[hash];
        if(val == -1) {
          break;
        }
        else {
          bool eq = true;
          for(int j = 0; j < dim - 1; j++) {
            if(hbuf_ptr[j] != sample_idxs.ptrs[j][val]) {
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
        gathered.rows.push_back(val);
        gathered.cols.push_back(idxs.ptrs[mode_to_leave][i]);
        gathered.values.push_back(values.ptr[i] * weights.ptr[val]);
      }
    }
    return gathered;
}

PYBIND11_MODULE(filter_nonzeros, m) {
  py::class_<COOSparse>(m, "COOSparse") 
    .def("print_contents", &COOSparse::print_contents);

  m.def("sample_nonzeros", &sample_nonzeros);
}

/*
<%
setup_pybind11(cfg)
%>
*/