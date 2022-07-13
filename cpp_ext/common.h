#pragma once

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
#include <chrono>

#define BIG_CONSTANT(x) (x##LLU)

using namespace std;
namespace py = pybind11;

#pragma GCC visibility push(hidden)
template<typename T>
class NumpyArray {
public:
    py::buffer_info info;
    T* ptr;

    NumpyArray(py::array_t<T> arr_py) {
        info = arr_py.request();
        ptr = static_cast<T*>(info.ptr);
    }
    NumpyArray(T* input_ptr) {
        ptr = input_ptr;
    }
};

template<typename T>
class NumpyList {
public:
    vector<py::buffer_info> infos;
    vector<T*> ptrs;
    int length;

    NumpyList(py::list input_list) {
        length = py::len(input_list);
        for(int i = 0; i < length; i++) {
            py::array_t<T> casted = input_list[i].cast<py::array_t<T>>();
            infos.push_back(casted.request());
            ptrs.push_back(static_cast<T*>(infos[i].ptr));
        }
    }

    // Should refactor class name to something 
    // other than NumpyList, since this
    // constructor exists. This constructor 
    // does not perform any data copy 
    NumpyList(vector<T*> input_list) {
        length = input_list.size();
        ptrs = input_list;
    }
};

class COOSparse {
public:
    vector<uint64_t> rows;
    vector<uint64_t> cols;
    vector<double> values;

    void print_contents() {
      double normsq = 0.0;
      for(uint64_t i = 0; i < rows.size(); i++) {
        /*cout 
          << rows[i] 
          << " " 
          << cols[i] 
          << " "
          << values[i]
          << endl;*/
        normsq += values[i]; 
      }
      cout << "Norm Squared: " << normsq << endl;
    }

	/*
	 * Computes Y := S^T . X, where S is this
	 * sparse matrix.
	 * 
	 * This is currently a very inefficient single-threaded
	 * version of the code. 
	 */
	void cpu_spmm(double* X, double* Y, int r) {
		uint64_t* row_ptr = rows.data();
		uint64_t* col_ptr = cols.data();
		double* val_ptr = values.data();

    #pragma omp parallel for
		for(uint64_t i = 0; i < rows.size(); i++) {
			// We perform a transpose here
			uint64_t row = col_ptr[i];
			uint64_t col = row_ptr[i];
			double value = val_ptr[i];
			for(int j = 0; j < r; j++) {
        #pragma omp atomic update
				Y[row * r + j] += X[col * r + j] * value;
			}
		}
	}
};

//-----------------------------------------------------------------------------
// MurmurHash2, by Austin Appleby

// Note - This code makes a few assumptions about how your machine behaves -

// 1. We can read a 4-byte value from any address without crashing
// 2. sizeof(int) == 4

// And it has a few limitations -

// 1. It will not work incrementally.
// 2. It will not produce the same results on little-endian and big-endian
//    machines.

uint64_t murmurhash2( const void * key, int len, uint64_t seed )
{
  const uint64_t m = BIG_CONSTANT(0xc6a4a7935bd1e995);
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t * data = (const uint64_t *)key;
  const uint64_t * end = data + (len/8);

  while(data != end)
  {
    uint64_t k = *data++;

    k *= m; 
    k ^= k >> r; 
    k *= m; 
    
    h ^= k;
    h *= m; 
  }

  const unsigned char * data2 = (const unsigned char*)data;

  switch(len & 7)
  {
  case 7: h ^= uint64_t(data2[6]) << 48;
  case 6: h ^= uint64_t(data2[5]) << 40;
  case 5: h ^= uint64_t(data2[4]) << 32;
  case 4: h ^= uint64_t(data2[3]) << 24;
  case 3: h ^= uint64_t(data2[2]) << 16;
  case 2: h ^= uint64_t(data2[1]) << 8;
  case 1: h ^= uint64_t(data2[0]);
          h *= m;
  };
 
  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
} 


typedef chrono::time_point<std::chrono::steady_clock> my_timer_t; 

my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

/*
 * This is a bad prefix sum function.
 */
void prefix_sum(vector<uint64_t> &values, vector<uint64_t> &offsets) {
    uint64_t sum = 0;
    for(uint64_t i = 0; i < values.size(); i++) {
        offsets.push_back(sum);
        sum += values[i];
    }
}

template<typename T>
void prefix_sum_ptr(T * values, T * offsets, uint64_t size) {
    T sum = 0;
    for(uint64_t i = 0; i < size; i++) {
        T val = values[i];
        offsets[i] = sum;
        sum += val; 
    }
}