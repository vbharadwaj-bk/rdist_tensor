#pragma once

#include <iostream>
#include <random>
#include <mpi.h>

#include "common.h"
#include "distmat.hpp"
#include "alltoallv_revised.hpp"

using namespace std;

class ExactLeverageTree {
public:
    DistMat1D &mat; 
    MPI_Comm comm;
    int world_size;
    int rank;

    vector<Buffer<double>> ancestor_grams;
    vector<vector<int>> ancestor_node_ids;

    // Related to the full, complete binary tree 
    //uint32_t leaf_count, node_count;

    uint32_t leaf_count, node_count;
    uint32_t lfill_level, lfill_count;
    uint32_t total_levels;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;

    int node_id, tree_depth;

    ExactLeverageTree(DistMat1D *mat, MPI_Comm world)
        : 
        mat(*mat), 
        comm(world) {

        world_size = 5; 
        //MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &rank);

        // Binary tree setup

        leaf_count = world_size; 
        node_count = 2 * leaf_count - 1;

        log2_round_down(leaf_count, lfill_level, lfill_count);
        total_levels = node_count > lfill_count ? lfill_level + 2 : lfill_level + 1;

        nodes_upto_lfill = lfill_count * 2 - 1;
        nodes_before_lfill = lfill_count - 1;

        uint32_t nodes_at_partial_level_div2 = (node_count - nodes_upto_lfill) / 2;
        complete_level_offset = nodes_before_lfill - nodes_at_partial_level_div2;

        cout << total_levels << endl;
        for(uint32_t i = 0; i < total_levels; i++) {
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

            for(int j = 0; j < ancestors.size(); j++) {
                ancestor_node_ids[j][i] = ancestors[j];
            }
        }

        // Print the ancestor node array
        for(int i = 0; i < total_levels; i++) {
            for(int j = 0; j < world_size; j++) {
                cout << ancestor_node_ids[i][j] << " ";
            }
            cout << endl;
        }
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

    void test_alltoallv_efficiency() {
        uint64_t global_row_count = (65000 / world_size) * world_size;
        uint64_t local_row_count = global_row_count / world_size;
        uint64_t col_count = 25;

        //h.initialize_to_shape({local_row_count, col_count});
        Buffer<int> destinations({local_row_count});
        Buffer<uint64_t> send_counts({(uint64_t) world_size});

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, world_size - 1);

        for(uint64_t i = 0; i < local_row_count; i++) {
            destinations[i] = dis(gen);
        }

        std::fill(send_counts(), send_counts(world_size), 0);
        for(uint64_t i = 0; i < local_row_count; i++) {

            // #pragma omp atomic
            send_counts[destinations[i]]++;
        }

        Buffer<double> recv_buffer;

        MPI_Barrier(comm);
        auto start = MPI_Wtime();
        /*alltoallv_matrix_rows(
                h,
                destinations, 
                send_counts, 
                recv_buffer,
                comm 
                );*/
        double elapsed = MPI_Wtime() - start; 

        cout << "Elapsed time: " << elapsed << endl;
    }
};

void test_distributed_exact_leverage() {
    ExactLeverageTree tree(nullptr, MPI_COMM_WORLD);
}
