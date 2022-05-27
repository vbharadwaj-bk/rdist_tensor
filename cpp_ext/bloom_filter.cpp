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

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "common.h"

using namespace std;
namespace py = pybind11;

/*
 * These files were modified and glued together by
 * Vivek Bharadwaj, 2022, from the libbloom source.
 * A Python wrapper has been written. 
 */

/*
 *  Copyright (c) 2012-2019, Jyri J. Virkki
 *  All rights reserved.
 *
 *  This file is under BSD license. See LICENSE file.
 */

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

#define MAKESTRING(n) STRING(n)
#define STRING(n) #n

struct bloom
{
  // These fields are part of the public interface of this structure.
  // Client code may read these values if desired. Client code MUST NOT
  // modify any of these.
  int entries;
  double error;
  int bits;
  int bytes;
  int hashes;

  // Fields below are private to the implementation. These may go away or
  // change incompatibly at any moment. Client code MUST NOT access or rely
  // on these.
  double bpe;
  unsigned char * bf;
  int ready;
};

inline static int test_bit_set_bit(unsigned char * buf,
                                   unsigned int x, int set_bit)
{
  unsigned int byte = x >> 3;
  unsigned char c = buf[byte];        // expensive memory access
  unsigned int mask = 1 << (x % 8);

  if (c & mask) {
    return 1;
  } else {
    if (set_bit) {
      buf[byte] = c | mask;
    }
    return 0;
  }
}


static int bloom_check_add(struct bloom * bloom,
                           const void * buffer, int len, int add)
{
  if (bloom->ready == 0) {
    printf("bloom at %p not initialized!\n", (void *)bloom);
    return -1;
  }

  int hits = 0;
  register unsigned int a = murmurhash2(buffer, len, 0x9747b28c);
  register unsigned int b = murmurhash2(buffer, len, a);
  register unsigned int x;
  register unsigned int i;

  for (i = 0; i < bloom->hashes; i++) {
    x = (a + i*b) % bloom->bits;
    if (test_bit_set_bit(bloom->bf, x, add)) {
      hits++;
    } else if (!add) {
      // Don't care about the presence of all the bits. Just our own.
      return 0;
    }
  }

  if (hits == bloom->hashes) {
    return 1;                // 1 == element already in (or collision)
  }

  return 0;
}

int bloom_init(struct bloom * bloom, int entries, double error)
{
  bloom->ready = 0;

  if (entries < 1000 || error == 0) {
    return 1;
  }

  bloom->entries = entries;
  bloom->error = error;

  double num = log(bloom->error);
  double denom = 0.480453013918201; // ln(2)^2
  bloom->bpe = -(num / denom);

  double dentries = (double)entries;
  bloom->bits = (int)(dentries * bloom->bpe);

  if (bloom->bits % 8) {
    bloom->bytes = (bloom->bits / 8) + 1;
  } else {
    bloom->bytes = bloom->bits / 8;
  }

  bloom->hashes = (int)ceil(0.693147180559945 * bloom->bpe);  // ln(2)

  bloom->bf = (unsigned char *)calloc(bloom->bytes, sizeof(unsigned char));
  if (bloom->bf == NULL) {                                   // LCOV_EXCL_START
    return 1;
  }                                                          // LCOV_EXCL_STOP

  bloom->ready = 1;
  return 0;
}

int bloom_init_size(struct bloom * bloom, int entries, double error,
                    unsigned int cache_size)
{
  return bloom_init(bloom, entries, error);
}


int bloom_check(struct bloom * bloom, const void * buffer, int len)
{
  return bloom_check_add(bloom, buffer, len, 0);
}


int bloom_add(struct bloom * bloom, const void * buffer, int len)
{
  return bloom_check_add(bloom, buffer, len, 1);
}


void bloom_print(struct bloom * bloom)
{
  printf("bloom at %p\n", (void *)bloom);
  printf(" ->entries = %d\n", bloom->entries);
  printf(" ->error = %f\n", bloom->error);
  printf(" ->bits = %d\n", bloom->bits);
  printf(" ->bits per elem = %f\n", bloom->bpe);
  printf(" ->bytes = %d\n", bloom->bytes);
  printf(" ->hash functions = %d\n", bloom->hashes);
}


void bloom_free(struct bloom * bloom)
{
  if (bloom->ready) {
    free(bloom->bf);
  }
  bloom->ready = 0;
}


int bloom_reset(struct bloom * bloom)
{
  if (!bloom->ready) return 1;
  memset(bloom->bf, 0, bloom->bytes);
  return 0;
}

const char * bloom_version()
{
  return MAKESTRING(BLOOM_VERSION);
}

/*
 * Tests the nonzeros locally owned by each
 * processor. Efficient Python wrapper for the bloom filter. 
 */
class IndexFilter {
  struct bloom bf;

public:
  // False positive tolerance is a small double value 
  IndexFilter(py::list idxs_py, double fp_tol) {
    NumpyList<unsigned long long> idxs(idxs_py);
    int dim = idxs.length; 
    unsigned long long nnz = idxs.infos[0].shape[0];

    // TODO: This will fail if nnz is larger than
    // the signed integer maximum! We should modify the
    // bloom filter to fix this. 
    assert(nnz < INT32_MAX);
    int nnz_dcast = (int) nnz;
    int nnz_inflated = std::max(nnz_dcast, 1000);
    int status = bloom_init(&bf, nnz_inflated, fp_tol);

    assert(status == 0);


    vector<unsigned long long> buf(dim, 0);
    unsigned long long * buf_ptr = buf.data();
    int buffer_len = 8 * dim; // A single unsigned double for each dimension 

    for(unsigned long long i = 0; i < nnz; i++) {
      for(int j = 0; j < dim; j++) {
        buf_ptr[j] = idxs.ptrs[j][i];
      }
      bloom_add(&bf, buf_ptr, buffer_len);
    }
  }

  // We can afford to return a vector since we expect relatively
  // few collisions. May need to modify this for the GPU case to
  // fill a boolean array
  vector<unsigned long long> check_idxs(py::list idxs_py) {
    NumpyList<unsigned long long> idxs(idxs_py);
    int dim = idxs.length; 
    unsigned long long nnz = idxs.infos[0].shape[0];

    vector<unsigned long long> buf(dim, 0);
    unsigned long long * buf_ptr = buf.data();
    int buffer_len = 8 * dim; // A single unsigned double for each dimension 

    vector<unsigned long long> collisions;

    for(unsigned long long i = 0; i < nnz; i++) {
      for(int j = 0; j < dim; j++) {
        buf_ptr[j] = idxs.ptrs[j][i];
      }
      if(bloom_check(&bf, buf_ptr, buffer_len)) {
        collisions.push_back(i);
      }
    }
    return collisions;
  } 

  ~IndexFilter() {
    bloom_free(&bf);
  }
};


PYBIND11_MODULE(bloom_filter, m) {
  py::class_<IndexFilter>(m, "IndexFilter")
    .def(py::init<py::list, double>())
    .def("check_idxs", &IndexFilter::check_idxs);
}

/*
<%
setup_pybind11(cfg)
%>
*/