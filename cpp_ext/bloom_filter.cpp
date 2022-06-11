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

#define MAKESTRING(n) STRING(n)
#define STRING(n) #n

struct bloom
{
  // These fields are part of the public interface of this structure.
  // Client code may read these values if desired. Client code MUST NOT
  // modify any of these.
  uint64_t entries;
  double error;
  uint64_t bits;
  uint64_t bytes;
  uint hashes;

  // Fields below are private to the implementation. These may go away or
  // change incompatibly at any moment. Client code MUST NOT access or rely
  // on these.
  double bpe;
  unsigned char * bf;
  int ready;
};

inline static int test_bit_set_bit(unsigned char * buf,
                                   uint64_t x, int set_bit)
{
  uint64_t byte = x >> 3;
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
    exit(1);
  }

  int hits = 0;
  register uint64_t a = murmurhash2(buffer, len, 0x9747b28c);
  register uint64_t b = murmurhash2(buffer, len, a);
  register uint64_t x;
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

int bloom_init(struct bloom * bloom, uint64_t entries, double error)
{
  bloom->ready = 0;

  if (entries < 1000ul || error == 0) {
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

int bloom_init_size(struct bloom * bloom, uint64_t entries, double error,
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
  printf(" ->entries = %ld\n", bloom->entries);
  printf(" ->error = %f\n", bloom->error);
  printf(" ->bits = %ld\n", bloom->bits);
  printf(" ->bits per elem = %f\n", bloom->bpe);
  printf(" ->bytes = %ld\n", bloom->bytes);
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
    uint64_t nnz = idxs.infos[0].shape[0];

    uint64_t nnz_inflated = std::max(nnz, 1000ul);
    int status = bloom_init(&bf, nnz_inflated, fp_tol);
    assert(status == 0);

    vector<unsigned long long> buf(dim, 0);
    unsigned long long * buf_ptr = buf.data();
    int buffer_len = 8 * dim; // A single unsigned long for each dimension 

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