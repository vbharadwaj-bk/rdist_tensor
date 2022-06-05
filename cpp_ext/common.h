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

using namespace std;
namespace py = pybind11;

template<typename T>
class NumpyArray {
public:
    py::buffer_info info;
    T* ptr;

    NumpyArray(py::array_t<T> arr_py) {
        info = arr_py.request();
        ptr = static_cast<T*>(info.ptr);
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
};

class COOSparse {
public:
    vector<unsigned long long> rows;
    vector<unsigned long long> cols;
    vector<double> values;

    void print_contents() {
      for(unsigned long long i = 0; i < rows.size(); i++) {
        cout << rows[i] 
        << " " 
        << cols[i] 
        << " "
        << values[i]
        << endl;
      }
    }

	/*
	 * Computes Y := S^T . X, where S is this
	 * sparse matrix.
	 * 
	 * This is currently a very inefficient single-threaded
	 * version of the code. 
	 */
	void cpu_spmm(double* X, double* Y, int r) {
		unsigned long long* row_ptr = rows.data();
		unsigned long long* col_ptr = cols.data();
		double* val_ptr = values.data();

		for(unsigned long long i = 0; i < rows.size(); i++) {
			// We perform a transpose here
			unsigned long long row = col_ptr[i];
			unsigned long long col = row_ptr[i];
			double value = val_ptr[i];
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
unsigned int murmurhash2(const void * key, int len, const unsigned int seed)
{
	// 'm' and 'r' are mixing constants generated offline.
	// They're not really 'magic', they just happen to work well.

	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	// Initialize the hash to a 'random' value

	unsigned int h = seed ^ len;

	// Mix 4 bytes at a time into the hash

	const unsigned char * data = (const unsigned char *)key;

	while(len >= 4)
	{
		unsigned int k = *(unsigned int *)data;

		k *= m;
		k ^= k >> r;
		k *= m;

		h *= m;
		h ^= k;

		data += 4;
		len -= 4;
	}

	// Handle the last few bytes of the input array

	switch(len)
	{
	case 3: h ^= data[2] << 16;
	case 2: h ^= data[1] << 8;
	case 1: h ^= data[0];
	        h *= m;
	};

	// Do a few final mixes of the hash to ensure the last few
	// bytes are well-incorporated.

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

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