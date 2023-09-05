#pragma once

#include <iostream>
#include <random>
#include <mpi.h>

#include "common.h"
#include "distmat.hpp"
#include "low_rank_tensor.hpp"
#include "alltoallv_revised.hpp"

using namespace std;

class ExactLeverageTree {
public:
    DistMat1D &mat;

    MPI_Comm comm;
    int world_size;
    int rank;

    uint64_t col_count;

    vector<Buffer<double>> ancestor_grams;
    vector<Buffer<double>> left_sibling_grams;
    vector<vector<int>> ancestor_node_ids;

    // Related to the full, complete binary tree 
    //uint32_t leaf_count, node_count;

    uint32_t leaf_count, node_count;
    uint32_t lfill_level, lfill_count;
    int total_levels;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;

    int node_id, tree_depth;

    ExactLeverageTree(DistMat1D &mat, MPI_Comm world)
        : 
        mat(mat), 
        comm(world) {

        col_count = mat.cols;

        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &rank);

        // Binary tree setup
        leaf_count = world_size; 
        node_count = 2 * leaf_count - 1;

        log2_round_down(leaf_count, lfill_level, lfill_count);

        nodes_upto_lfill = lfill_count * 2 - 1;
        nodes_before_lfill = lfill_count - 1;

        total_levels = node_count > nodes_upto_lfill ? lfill_level + 2 : lfill_level + 1;

        uint32_t nodes_at_partial_level_div2 = (node_count - nodes_upto_lfill) / 2;
        complete_level_offset = nodes_before_lfill - nodes_at_partial_level_div2;

        for(int i = 0; i < total_levels; i++) {
            ancestor_node_ids.emplace_back(world_size, -1);
        }

        vector<int> leaf_nodes(world_size, 0);
        for(int i = 0; i < world_size; i++) {
            leaf_nodes[leaf_idx(i + world_size - 1)] = i + world_size - 1;
        }

        for(int i = 0; i < world_size; i++) {  
            vector<int> ancestors;
            ancestors.push_back(leaf_nodes[i]);
            while(ancestors.back() != 0) {
                ancestors.push_back((ancestors.back() - 1) / 2);
            }
            // Reverse the ancestor list
            std::reverse(ancestors.begin(), ancestors.end());

            for(uint64_t j = 0; j < ancestors.size(); j++) {
                ancestor_node_ids[j][i] = ancestors[j];
            }
        }

        /*if(rank == 0) {
            cout << "Total levels in tree: " << total_levels << endl;
            cout << "-------------" << endl;
            // Print the ancestor node array
            for(uint32_t i = 0; i < total_levels; i++) {
                for(int j = 0; j < world_size; j++) {
                    cout << ancestor_node_ids[i][j] << " ";
                }
                cout << endl;
            }
            cout << "-------------" << endl;
        }*/
    }

    void get_exchange_assignments(int l_num, vector<int> &p1, vector<int> &p2) {
        std::fill(p1.begin(), p1.end(), -1);
        std::fill(p2.begin(), p2.end(), -1);

        vector<int> &level = ancestor_node_ids[l_num];

        int c_idx = 0;
        while(c_idx < world_size) {
            int c = level[c_idx];
            if(c == -1) {
                break; 
            }
            else {
                int first_segment_end = c_idx;
                while(level[first_segment_end] == c)
                    first_segment_end++;

                int second_segment_c = level[first_segment_end];
                int second_segment_end = first_segment_end;

                while((second_segment_end < world_size) &&
                    (level[second_segment_end] == second_segment_c))
                    second_segment_end++;

                int second_segment_size = second_segment_end - first_segment_end;

                for(int i = 0; i < first_segment_end - c_idx; i++) {
                    if(first_segment_end + i < second_segment_end) {
                        p1[c_idx + i] = first_segment_end + i;
                        p1[first_segment_end + i] = c_idx + i;
                    }
                    else {
                        int target_idx = first_segment_end + (i % second_segment_size); 
                        p1[c_idx + i] = target_idx;
                        p2[target_idx] = c_idx + i;
                    }
                }
                c_idx = second_segment_end;
            }
        }
    }

    void construct_gram_tree() {
        ancestor_grams.clear();
        left_sibling_grams.clear();

        ancestor_grams.emplace_back((initializer_list<uint64_t>){col_count, col_count});
        left_sibling_grams.emplace_back((initializer_list<uint64_t>){col_count, col_count});

        mat.compute_gram_local_slice(ancestor_grams[0]);

        vector<int> p1(world_size, -1);
        vector<int> p2(world_size, -1);

        MPI_Barrier(comm);
        auto start = MPI_Wtime();

        for(int level = total_levels - 1; level > 0; level--) {
            get_exchange_assignments(level, p1, p2);

            if(p1[rank] == -1) {
                continue;
            }

            ancestor_grams.emplace_back((initializer_list<uint64_t>){col_count, col_count});
            left_sibling_grams.emplace_back((initializer_list<uint64_t>){col_count, col_count});

            uint64_t n_ancestor_grams = ancestor_grams.size();
            uint64_t n_left_sibling_grams = left_sibling_grams.size();


            vector<int> &level_ancestors = ancestor_node_ids[level];
            int nid = level_ancestors[rank];

            // Print both p1 and p2

            Buffer<double> *target_mat;

            if(p1[rank] != -1) {
                if(nid % 2 == 1) { // Left child
                    std::copy(
                            ancestor_grams[n_ancestor_grams - 2](),
                            ancestor_grams[n_ancestor_grams - 2](col_count * col_count),
                            left_sibling_grams[n_left_sibling_grams - 1]()
                            );
                    target_mat = &(ancestor_grams[n_ancestor_grams - 1]);
                }
                else { // Right child
                    std::copy(
                            ancestor_grams[n_ancestor_grams - 2](),
                            ancestor_grams[n_ancestor_grams - 2](col_count * col_count),
                            ancestor_grams[n_ancestor_grams - 1]()
                            );
                    target_mat = &(left_sibling_grams[n_left_sibling_grams - 1]);
                }
            }


            if(p1[rank] != -1) {
                MPI_Sendrecv(
                        ancestor_grams[n_ancestor_grams - 2](),
                        col_count * col_count,
                        MPI_DOUBLE,
                        p1[rank],
                        level,
                        (*target_mat)(), 
                        col_count * col_count,
                        MPI_DOUBLE,
                        p1[rank],
                        level,
                        comm,
                        MPI_STATUS_IGNORE
                        );
            }
            if(p2[rank] != -1) {
                Buffer<double> temp({col_count, col_count});

                MPI_Sendrecv(
                        ancestor_grams[n_ancestor_grams - 2](),
                        col_count * col_count,
                        MPI_DOUBLE,
                        p2[rank],
                        level,
                        temp(), 
                        col_count * col_count,
                        MPI_DOUBLE,
                        p2[rank],
                        level,
                        comm,
                        MPI_STATUS_IGNORE
                        );
            }

            for(uint64_t i = 0; i < col_count * col_count; i++) {
                (ancestor_grams[n_ancestor_grams - 1])[i] += (left_sibling_grams[n_left_sibling_grams - 1])[i];
            }
        }

        MPI_Barrier(comm);
    } 

    bool is_leaf(int64_t c) {
        return 2 * c + 1 >= node_count; 
    }

    int64_t leaf_idx(int64_t c) {
        if(c >= nodes_upto_lfill) {
            return c - nodes_upto_lfill;
        }
        else {
            return c - complete_level_offset; 
        }
    }

    void batch_dot_product(
                Buffer<double> &A, 
                Buffer<double> &B, 
                Buffer<double> &result,
                uint64_t offset
                ) {
        uint64_t J = A.shape[0];
        uint64_t R = A.shape[1];
        uint64_t res_col_count = result.shape[1];

        //#pragma omp for
        for(uint64_t i = 0; i < J; i++) {
            double res = 0.0; 
            for(uint64_t j = 0; j < R; j++) {
                res += A[i * R + j] * B[i * R + j];
            }
            result[i * res_col_count + offset] = res;
        }
    }

    void dsymm(Buffer<double> &sym, Buffer<double> &mat, Buffer<double> &out) {
        // Should parallelize, but that can wait 
        cblas_dsymm(
            CblasRowMajor,
            CblasRight,
            CblasUpper,
            (uint32_t) mat.shape[0],
            (uint32_t) sym.shape[1],
            1.0,
            sym(),
            R,
            mat(),
            R,
            0.0,
            out(),
            R
        );
    }

    void execute_tree_computation(Buffer<double> &h, 
            Buffer<uint32_t> &indices, 
            Buffer<double> &draws) {
        uint64_t row_count = h.shape[0];
        Buffer<double> mData({row_count, 4});  // Elements are low, high, mass, and random draw
        Buffer<int> target_procs({row_count}); // Processor to send each element to
        Buffer<double> temp1({h.shape[0], h.shape[1]});

        dsymm(ancestor_grams.back(), h, temp1);
        batch_dot_product(h, temp1, mData);
    }
};

void test_distributed_exact_leverage(LowRankTensor &ten) {
    ExactLeverageTree tree(ten.factors[0], ten.factors[0].ordered_world);

    int world_size, rank;
    MPI_Comm_size(ten.factors[0].ordered_world, &world_size);
    MPI_Comm_rank(ten.factors[0].ordered_world, &rank);

    tree.construct_gram_tree();

    uint64_t n_samples = 20;

    uint64_t rounded_sample_count = (n_samples + world_size - 1) / world_size; 

    uint64_t start = min(work * rank, n_samples);
    uint64_t end = min(work * (rank + 1), n_samples);

    Buffer<double> h({end - start, ten.factors[0].cols});
    Buffer<uint32_t> indices({end - start, 3});
 
    std::fill(h(), h((end - start) * ten.factors[0].cols), 1.0);
    tree.execute_tree_computation(h, indices);

    Multistream_RNG local_rng;
    Buffer<double> draws({end - start});
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        for(uint64_t i = 0; i < end - start; i++) {
            draws[i] = dist(local_rng.par_gen[tid]);
        }
    } 

    execute_tree_computation(h, indices, draws);
}
