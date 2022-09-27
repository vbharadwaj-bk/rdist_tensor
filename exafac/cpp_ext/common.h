#pragma once

#include <cassert>
#include <fcntl.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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
#include <shmemx.h> 

#define BIG_CONSTANT(x) (x##LLU)

using namespace std;
namespace py = pybind11;

#define DEFINE_MPI_DATATYPES() ({\
    if(std::is_same<IDX_T, uint32_t>::value)\
        MPI_IDX_T = MPI_UINT32_T;\
    else\
        MPI_IDX_T = MPI_UINT64_T;\
    if(std::is_same<IDX_T, float>::value)\
        MPI_VAL_T = MPI_FLOAT;\
    else\
        MPI_VAL_T = MPI_DOUBLE;\
})

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

    NumpyArray(py::object obj, string attr_name) {
        py::array_t<T> arr_py = obj.attr(attr_name.c_str()).cast<py::array_t<T>>();
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

template<typename IDX_T, typename VAL_T>
class Triple {
public:
    IDX_T row;
    IDX_T col;
    VAL_T val;
};

template<typename IDX_T, typename VAL_T>
class COOSparse {
public:
    vector<IDX_T> rows;
    vector<IDX_T> cols;
    vector<VAL_T> values;

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
		IDX_T* row_ptr = rows.data();
		IDX_T* col_ptr = cols.data();
		VAL_T* val_ptr = values.data();

		for(uint64_t i = 0; i < rows.size(); i++) {
			// We perform a transpose here
		    IDX_T row = col_ptr[i];
			IDX_T col = row_ptr[i];
			VAL_T value = val_ptr[i];
			for(int j = 0; j < r; j++) {
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

// TODO: Need to add a hash function for other sizes as well! 

inline uint64_t murmurhash2_fast (uint32_t k1, uint64_t seed )
{
  const uint32_t m = 0x5bd1e995;
  const int r = 24;

  const int len = 4;

  uint32_t h1 = uint32_t(seed) ^ len;
  uint32_t h2 = uint32_t(seed >> 32);

  k1 *= m; k1 ^= k1 >> r; k1 *= m;
  h1 *= m; h1 ^= k1;

  h1 ^= h2 >> 18; h1 *= m;
  h2 ^= h1 >> 22; h2 *= m;
  h1 ^= h2 >> 17; h1 *= m;
  h2 ^= h1 >> 19; h2 *= m;

  uint64_t h = h1;

  h = (h << 32) | h2;

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

/*
enum Buffer_Type {
    SHMEM, REGULAR, UNINITIALIZED
};

template<typename T>
class Comm_Buffer {
public:
    T* ptr;
    uint64_t size;
    Buffer_Type mode;

    Comm_Buffer() {
        ptr = nullptr;
        size = 0;
        mode = UNINITIALIZED;
    }

    Comm_Buffer(uint64_t n_elements, Buffer_type mode) {
        this->mode = mode;
        size = n_elements;

        if(mode == SHMEM) {
            ptr = (T*) shmem_malloc(sizeof(T) * size);
        }
        else if(mode == REGULAR) {
            ptr = (T*) malloc(sizeof(T) * size);
        }

        memset(ptr, 0x00, sizeof(T) * n_elements);
    }

    void reallocate(uint64_t n_elements, Buffer_type mode) {
        this->mode = mode;
        size = n_elements;
        if(ptr == nullptr) {
            if(mode == SHMEM) {
                ptr = (T*) shmem_malloc(sizeof(T) * size);
            }
            else if(mode == REGULAR) {
                ptr = (T*) malloc(sizeof(T) * size);
            }
        }
        else {
            if(mode == SHMEM) {
                ptr = (T*) shmem_realloc(ptr, sizeof(T) * size);
            }
            else {
                ptr = (T*) realloc(ptr, sizeof(T) * size);
            }

        }
    }

    constexpr T& operator[](std::size_t idx) {
        return ptr[idx];
    }

    void fill(T value) {
        std::fill(ptr, ptr + size, value);
    }

    void destroy() {
        if(ptr != nullptr) {
            if(mode == SHMEM) {
                shmem_free(ptr);
            }
            else if(mode == REGULAR) {
                free(ptr);
            }

            ptr = nullptr;
        }
    }

    ~Comm_Buffer() {
        destroy();
    }
};
*/
