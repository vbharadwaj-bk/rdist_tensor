#pragma once

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>
#include "common.h"
#include "omp.h"
#include "cblas.h"

using namespace std;

// Create a struct with 5 double fields, q, m, mL, low, and high

typedef struct {
    int64_t original_idx;
    double draw;
    int64_t c;
    double m;
    double low;
    double high;
    double* a;
    double* x;
    double* y;
    
} mdata_t;

/*
* Collection of temporary buffers that can be reused by all tree samplers 
*/
class __attribute__((visibility("hidden"))) ScratchBuffer {
public:
    uint64_t J;
    Buffer<double> temp1;
    Buffer<mdata_t> mdata;
    Buffer<double> q;

    ScratchBuffer(uint32_t F, uint64_t J, uint64_t R) :
            J(J),
            temp1({J, R}),
            mdata({J}),
            q({J, F})
    {}
};

class __attribute__((visibility("hidden"))) PartitionTree {
public:
    int64_t n, F;
    uint32_t leaf_count, node_count;

    uint32_t lfill_level, lfill_count;
    uint32_t total_levels;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;

    int64_t R;
    int64_t R2;

    Buffer<double> G;

    unique_ptr<Buffer<double>> G_unmultiplied;

    void execute_mkl_dsymv_batch(ScratchBuffer &scratch) {
        #pragma omp for
        for(uint64_t i = 0; i < scratch.J; i++) {
            cblas_dsymv(CblasRowMajor, 
                    CblasUpper, 
                    R, 
                    1.0, 
                    (const double*) scratch.mdata[i].a,
                    R, 
                    (const double*) scratch.mdata[i].x, 
                    1, 
                    0.0, 
                    scratch.mdata[i].y, 
                    1);
        }
    }

    PartitionTree(uint32_t n, uint32_t F, uint64_t R)
        :   G({2 * divide_and_roundup(n, F) - 1, R * R})
        {
        this->n = n;
        this->F = F;
        this->R = R;
        R2 = R * R;

        leaf_count = divide_and_roundup(n, F);
        node_count = 2 * leaf_count - 1;

        log2_round_down(leaf_count, lfill_level, lfill_count);
        total_levels = node_count > lfill_count ? lfill_level + 1 : lfill_level;

        nodes_upto_lfill = lfill_count * 2 - 1;
        nodes_before_lfill = lfill_count - 1;

        uint32_t nodes_at_partial_level_div2 = (node_count - nodes_upto_lfill) / 2;
        complete_level_offset = nodes_before_lfill - nodes_at_partial_level_div2;
        G_unmultiplied.reset(nullptr);
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

    void build_tree(Buffer<double> &U) {
        G_unmultiplied.reset(nullptr);

        // First leaf must always be on the lowest filled level 
        int64_t first_leaf_idx = node_count - leaf_count; 

        Buffer<double*> a_array({leaf_count});
        Buffer<double*> c_array({leaf_count});

        #pragma omp parallel
{
        #pragma omp for
        for(int64_t i = 0; i < node_count * R2; i++) {
            G[i] = 0.0;
        }

        #pragma omp for
        for(int64_t i = 0; i < leaf_count; i++) {
            uint64_t idx = leaf_idx(first_leaf_idx + i);
            uint64_t row_ct = min((uint64_t) F, U.shape[0] - idx * F);
            a_array[i] = U(idx * F, 0);
            c_array[i] = G(first_leaf_idx + i, 0);

            cblas_dsyrk(CblasRowMajor, 
                        CblasUpper, 
                        CblasTrans,
                        R,
                        row_ct, 
                        1.0, 
                        (const double*) a_array[i], 
                        R, 
                        0.0, 
                        c_array[i], 
                        R);
        }

        int64_t start = nodes_before_lfill; 
        int64_t end = first_leaf_idx;

        for(int c_level = lfill_level; c_level >= 0; c_level--) {
            #pragma omp for 
            for(int c = start; c < end; c++) {
                for(int j = 0; j < R2; j++) {
                    G[c * R2 + j] += G[(2 * c + 1) * R2 + j] + G[(2 * c + 2) * R2 + j];
                } 
            }
            end = start;
            start = ((start + 1) / 2) - 1;
        }
}
    }

    void get_G0(py::array_t<double> M_buffer_py) {
        Buffer<double> M_buffer(M_buffer_py);
        for(int64_t i = 0; i < R2; i++) {
            M_buffer[i] = G[i];
        } 
    }

    void multiply_matrices_against_provided(Buffer<double> &mat) {
        if(! G_unmultiplied) {
            G_unmultiplied.reset(new Buffer<double>({node_count, static_cast<unsigned long>(R2)}));
            std::copy(G(), G(node_count * R2), (*G_unmultiplied)());
        }
        #pragma omp parallel for
        for(int64_t i = 0; i < node_count; i++) {
            for(int j = 0; j < R2; j++) {
                G[i * R2 + j] = (*G_unmultiplied)[i * R2 + j] * mat[j];
            }
        }
    }

    void batch_dot_product(
                double* A, 
                double* B, 
                double* result,
                int64_t J, int64_t R 
                ) {
        #pragma omp for
        for(int i = 0; i < J; i++) {
            result[i] = 0;
            for(int j = 0; j < R; j++) {
                result[i] += A[i * R + j] * B[i * R + j];
            }
        }
    }

    template<typename VAL_T>
    void apply_permutation_parallel(Buffer<VAL_T> &in, Buffer<uint32_t> &perm, Buffer<VAL_T> &out) {
        if(in.shape.size() == 1) {
            #pragma omp for
            for(uint64_t i = 0; i < in.shape[0]; i++) {
                out[i] = in[perm[i]];
            }
        }
        else if(in.shape.size() == 2) {
            uint64_t col_dim = in.shape[1];
            #pragma omp for
            for(uint64_t i = 0; i < in.shape[0]; i++) {
                for(uint64_t j = 0; j < in.shape[1]; j++) {
                    out[i * col_dim + j] = in[perm[i] * col_dim + j];
                }
            }
        }
    }

    void PTSample(Buffer<double> &U, 
            Buffer<double> &h,
            Buffer<double> &scaled_h,
            Buffer<uint32_t> &samples,
            Buffer<double> &random_draws,
            ScratchBuffer &scratch,
            int64_t sample_offset) {

        uint64_t sample_matrix_width = samples.shape[1];
        int64_t J = (int64_t) h.shape[0];

        Buffer<double> &temp1 = scratch.temp1;
        Buffer<mdata_t> &mdata = scratch.mdata;
        Buffer<double> &q = scratch.q;
        Buffer<mdata_t>* new_mdata;
        Buffer<double>* new_scaled_h;

        Buffer<double> symv_do({(uint64_t) J});
        Buffer<uint32_t> perm({(uint64_t) J});

        #pragma omp parallel
{

        #pragma omp single
        {
            std::iota(perm(), perm(J), 0);
            std::reverse(perm(), perm(J));
            new_mdata = new Buffer<mdata_t>({(uint64_t) J});
            new_scaled_h = new Buffer<double>({(uint64_t) J, (uint64_t) R});
        }
        apply_permutation_parallel(mdata, perm, *new_mdata);
        apply_permutation_parallel(scaled_h, perm, *new_scaled_h);
        #pragma omp barrier
        #pragma omp single
        {
            mdata.steal_resources(*new_mdata);
            scaled_h.steal_resources(*new_scaled_h);
            delete new_mdata;
            delete new_scaled_h;
        }

        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            mdata[i].original_idx = i;
            mdata[i].draw = random_draws[i];
            mdata[i].a = G(0);
            mdata[i].x = scaled_h(i, 0);
            mdata[i].y = temp1(i, 0); 

            mdata[i].c = 0;
            mdata[i].low = 0.0;
            mdata[i].high = 1.0;
        }

        execute_mkl_dsymv_batch(scratch);
        batch_dot_product(
            scaled_h(), 
            temp1(), 
            symv_do(),
            J, R 
            );

        #pragma omp for
        for(uint64_t i = 0; i < (uint64_t) J; i++) {
            mdata[i].m = symv_do[i];
        }

        for(uint32_t c_level = 0; c_level < lfill_level; c_level++) {
            // Prepare to compute m(L(v)) for all v

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                mdata[i].a = G((2 * mdata[i].c + 1) * R2); 
                mdata[i].x = scaled_h(i, 0);
                mdata[i].y = temp1(i, 0); 
            }

            execute_mkl_dsymv_batch(scratch);

            batch_dot_product(
                scaled_h(), 
                temp1(), 
                symv_do(),
                J, R 
                );

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                double cutoff = mdata[i].low + symv_do[i] / mdata[i].m;
                if(mdata[i].draw <= cutoff) {
                    mdata[i].c = 2 * mdata[i].c + 1;
                    mdata[i].high = cutoff;
                }
                else {
                    mdata[i].c = 2 * mdata[i].c + 2;
                    mdata[i].low = cutoff;
                }
            }
        }

        // Handle the tail case
        if(node_count > nodes_before_lfill) {
            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                mdata[i].a = is_leaf(mdata[i].c) ? mdata[i].a : G((2 * mdata[i].c + 1) * R2); 
                mdata[i].x = scaled_h(i, 0);
                mdata[i].y = temp1(i, 0); 
            }

            execute_mkl_dsymv_batch(scratch);

            batch_dot_product(
                scaled_h(), 
                temp1(), 
                symv_do(),
                J, R 
                );

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                double cutoff = mdata[i].low + symv_do[i] / mdata[i].m;
                if((! is_leaf(mdata[i].c)) && mdata[i].draw <= cutoff) {
                    mdata[i].c = 2 * mdata[i].c + 1;
                    mdata[i].high = cutoff;
                }
                else if((! is_leaf(mdata[i].c)) && mdata[i].draw > cutoff) {
                    mdata[i].c = 2 * mdata[i].c + 2;
                    mdata[i].low = cutoff;
                }
            }
        }

        #pragma omp single
        {
            new_mdata = new Buffer<mdata_t>({(uint64_t) J});
            new_scaled_h = new Buffer<double>({(uint64_t) J, (uint64_t) R});
        }
        apply_permutation_parallel(mdata, perm, *new_mdata);
        apply_permutation_parallel(scaled_h, perm, *new_scaled_h);
        #pragma omp barrier
        #pragma omp single
        {
            mdata.steal_resources(*new_mdata);
            scaled_h.steal_resources(*new_scaled_h);
            delete new_mdata;
            delete new_scaled_h;
        }


        // We will use the m array as a buffer 
        // for the draw fractions.
        if(F > 1) {
            #pragma omp for
            for(int i = 0; i < J; i++) {
                mdata[i].m = (mdata[i].draw - mdata[i].low) / (mdata[i].high - mdata[i].low);

                int64_t leaf_idx;
                if(mdata[i].c >= nodes_upto_lfill) {
                    leaf_idx = mdata[i].c - nodes_upto_lfill;
                }
                else {
                    leaf_idx = mdata[i].c - complete_level_offset; 
                }

                mdata[i].a = U(leaf_idx * F, 0);
                mdata[i].x = scaled_h(i, 0);
                mdata[i].y = q(i * F);
                
                std::fill(q(i * F), q((i + 1) * F), 0.0);
                uint64_t row_count = min((uint64_t) F, U.shape[0] - leaf_idx * F);

                cblas_dgemv(CblasRowMajor, 
                        CblasNoTrans,
                        row_count,
                        R, 
                        1.0, 
                        (const double*) mdata[i].a,
                        R, 
                        (const double*) mdata[i].x, 
                        1, 
                        0.0, 
                        mdata[i].y, 
                        1);
            }
        }

        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            int64_t res; 
            if(F > 1) {
                res = F - 1;
                double running_sum = 0.0;
                for(int64_t j = 0; j < F; j++) {
                    double temp = q[i * F + j] * q[i * F + j];
                    q[i * F + j] = running_sum;
                    running_sum += temp;
                }

                for(int64_t j = 0; j < F; j++) {
                    q[i * F + j] /= running_sum; 
                }

                for(int64_t j = 0; j < F - 1; j++) {
                    if(mdata[i].m < q[i * F + j + 1]) {
                        res = j; 
                        break;
                    }
                }
            }
            else {
                res = 0;
            }

            int64_t idx = leaf_idx(mdata[i].c);
            samples[sample_matrix_width * i + sample_offset] = (uint32_t) (res + idx * F);
            
            for(int64_t j = 0; j < R; j++) {
                h[i * R + j] *= U[(res + idx * F) * R + j];
            }  
        }
}
    }
};