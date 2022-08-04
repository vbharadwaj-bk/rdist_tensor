#pragma once

#include <vector>
#include <iostream>
#include <vector>
#include <random>
#include "primality.hpp"
#include "hashing.h"

using namespace std;

struct fks_node_t {
	uint32_t count;
	uint32_t start_loc;
	uint32_t hash;
	vector<uint32_t> temp_storage;

public:
	fks_node_t() {
		count = 0;	
	}
};

inline uint32_t hash_moda_modb(
		uint32_t* ptr, 
		uint64_t seed, 
		int dim, 
		uint32_t a, 
		uint32_t b) {
	// https://stackoverflow.com/questions/33333363/
	// built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op
	int num_bytes = dim * 4;
	uint64_t hash = MurmurHash3_x86_32 ( ptr, num_bytes, seed); 
    uint32_t modp = hash >= a ? hash % a : hash;
    return modp >= b ? modp % b : modp;
}

class FKSHash {
public:
	vector<fks_node_t> table;
	vector<uint32_t> packed_storage;
	uint32_t base_seed;
	uint32_t dim;
	uint32_t mode_to_leave;
	uint32_t n;
	uint32_t prime;

	FKSHash(uint32_t* idx_mat, uint32_t dim, uint32_t mode_to_leave, uint32_t n, uint64_t seed) {
		this->dim = dim;
		this->mode_to_leave = mode_to_leave;
		this->n = n;

		table.resize(n);

		for(uint64_t i = n; i < 2 * n; i++) {
			if(MillerRabin(i)) {
				prime = (uint32_t) i;
				break;
			}
		}

		std::mt19937 gen(seed);
		std::uniform_int_distribution<> distrib(0, 1 << 31);

		base_seed = distrib(gen); 
		for(uint32_t i = 0; i < n; i++) {
			uint32_t* base_ptr = idx_mat + i * dim;
			uint32_t hash_loc = hash_moda_modb(
					base_ptr, 
					base_seed, 
					dim, 
					prime, 
					n);
			table[hash_loc].count++;
			table[hash_loc].temp_storage.push_back(i);
		}
	
		uint32_t rolling_sum = 0;
		for(uint32_t i = 0; i < n; i++) {
			uint32_t space_alloc = table[i].count * table[i].count;
			if(table[i].count > 0) {
				table[i].start_loc = rolling_sum;

				uint32_t* base_ptr = packed_storage.data() + table[i].start_loc;
				uint32_t count = table[i].count;

				if(count == 1) {
					*base_ptr = table[i].temp_storage[0];
				}
				else if(count > 0) {
					bool found_injection = false;

					while(! found_injection) {
						found_injection = true;
						std::fill(base_ptr, base_ptr + count, n);
						table[i].hash = distrib(gen);

						for(uint32_t j = 0; j < table[i].count; j++) {
							uint32_t hash_loc = hash_moda_modb(
									idx_mat + table[i].temp_storage[j] * dim, 
									table[i].hash, 
									dim, 
									prime, 
									space_alloc);
							
							if(base_ptr[hash_loc] == n) {
								base_ptr[hash_loc] = table[i].temp_storage[j];
							}
							else {
								found_injection = false;
								break;
							}
						}
					}	
				}
			}
			rolling_sum += space_alloc; 
		}
	}

	/*
	 * Look up an element that you are not certain is in the table. 
	 * Returns n if the element is not found. 
	 */
	uint32_t lookup_careful(uint32_t* buf) {
		uint32_t hash_loc1 = hash_moda_modb(
				buf, 
				base_seed, 
				dim, 
				prime, 
				n);

		if(table[hash_loc1].count == 0)
			return n;
		else {
			uint32_t* base_ptr = packed_storage.data() + table[hash_loc1].start_loc;

			if(table[hash_loc1].count == 1) {
				return *base_ptr;
			}
			else {
				uint32_t hash_loc2 = hash_moda_modb(
						buf, 
						table[hash_loc1].hash, 
						dim, 
						prime, 
						n);
				return base_ptr[hash_loc2];	
			}
		}
	}
};