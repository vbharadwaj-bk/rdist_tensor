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
#include "json.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#define BIG_CONSTANT(x) (x##LLU)

using namespace std;
namespace py = pybind11;

using json = nlohmann::json;

template<typename TYPE_T>
MPI_Datatype get_MPI_dtype() {
    if(std::is_same<TYPE_T, uint32_t>::value) {
        return MPI_UINT32_T;
    }
    else if(std::is_same<TYPE_T, uint64_t>::value) {
        return MPI_UINT64_T;
    }
    else if(std::is_same<TYPE_T, int32_t>::value) {
        return MPI_INT32_T;
    }
    else if(std::is_same<TYPE_T, int64_t>::value) {
        return MPI_INT64_T;
    }
    else if(std::is_same<TYPE_T, float>::value) {
        return MPI_FLOAT;
    }
    else if(std::is_same<TYPE_T, double>::value) {
        return MPI_DOUBLE;
    }
    else {
        cout << "Unknown MPI datatype" << endl;
        exit(1);
    }
}

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

uint64_t round_to_nearest_integer(uint64_t n, uint64_t m) {
    return (n + m) / m * m;
}

//#pragma GCC visibility push(hidden)
template<typename T>
class __attribute__((visibility("hidden"))) Buffer {
public:
    py::buffer_info info;
    unique_ptr<T[]> managed_ptr;
    T* ptr;
    uint64_t dim0;
    uint64_t dim1;

    bool initialized;
    vector<uint64_t> shape;

    Buffer(Buffer&& other)
        :   info(std::move(other.info)), 
            managed_ptr(std::move(other.managed_ptr)),
            ptr(std::move(other.ptr)),
            dim0(other.dim0),
            dim1(other.dim1),
            initialized(other.initialized),
            shape(std::move(other.shape))
    {}
    Buffer& operator=(const Buffer& other) = default;

    void steal_resources(Buffer& other) {
        info = std::move(other.info); 
        managed_ptr = std::move(other.managed_ptr);
        ptr = other.ptr;
        dim0 = other.dim0;
        dim1 = other.dim1;
        shape = other.shape;
        initialized = other.initialized;
    }

    Buffer(py::array_t<T> arr_py, bool copy) {
        info = arr_py.request();

        if(info.ndim == 2) {
            dim0 = info.shape[0];
            dim1 = info.shape[1];
        }
        else if(info.ndim == 1) {
            dim0 = info.shape[0];
            dim1 = 1;
        }

        uint64_t buffer_size = 1;
        for(int64_t i = 0; i < info.ndim; i++) {
            shape.push_back(info.shape[i]);
            buffer_size *= info.shape[i];
        }

        if(! copy) {
            ptr = static_cast<T*>(info.ptr);
        }
        else {
            managed_ptr.reset(new T[buffer_size]);
            ptr = managed_ptr.get();
            std::copy(static_cast<T*>(info.ptr), static_cast<T*>(info.ptr) + info.size, ptr);
        }
        initialized = true;
    }

    Buffer(py::array_t<T> arr_py) :
        Buffer(arr_py, false)
    {
        // Default behavior is a thin alias of the C++ array 
    }

    Buffer(initializer_list<uint64_t> args) {
        initialized = false;
        if(args.size() > 0) {
            initialize_to_shape(args);
        }
    }

    Buffer(initializer_list<uint64_t> args, T* ptr) {
        for(uint64_t i : args) {
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        this->ptr = ptr;
        initialized = true;
    }

    Buffer() {
        initialized = false;
    }

    void initialize_to_shape(initializer_list<uint64_t> args) {
        if(initialized) {
            throw std::runtime_error("Cannot initialize a buffer twice");
        }
        uint64_t buffer_size = 1;
        for(uint64_t i : args) {
            buffer_size *= i;
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        managed_ptr.reset(new T[buffer_size]);
        ptr = managed_ptr.get();
        initialized = true;
    }

    T* operator()() {
        return ptr;
    }

    T* operator()(uint64_t offset) {
        return ptr + offset;
    }

    // Assumes that this array is a row-major matrix 
    T* operator()(uint64_t off_x, uint64_t off_y) {
        return ptr + (dim1 * off_x) + off_y;
    }

    T& operator[](uint64_t offset) {
        return ptr[offset];
    }

    void print() {
        cout << "------------------------" << endl;
        if(shape.size() == 1) {
            cout << "[ " << " "; 
            for(uint64_t i = 0; i < shape[0]; i++) {
                cout << ptr[i] << " ";
            }
            cout << "]" << endl;
            return;
        }
        else if(shape.size() == 2) {
            for(uint64_t i = 0; i < shape[0]; i++) {
                cout << "[ ";
                for(uint64_t j = 0; j < shape[1]; j++) {
                    cout << ptr[i * shape[1] + j] << " ";
                }
                cout << "]" << endl; 
            }
        }
        else {
            cout << "Cannot print buffer with shape: ";
            for(uint64_t i : shape) {
                cout << i << " ";
            }
            cout << endl;
        }
        cout << "------------------------" << endl;
    }

    ~Buffer() {}
};

void compute_gram(Buffer<double> &in, Buffer<double> &M) {
    uint64_t R = in.shape[1];

    std::fill(M(), M(R * R), 0.0);

    uint64_t I = in.shape[0];
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        uint64_t work = (I + num_threads - 1) / num_threads;
        uint64_t start = min(work * thread_id, I);
        uint64_t end = min(work * (thread_id + 1), I);

        if(end - start > 0) {
            Buffer<double> local({R, R});
            cblas_dsyrk(CblasRowMajor, 
                        CblasUpper, 
                        CblasTrans,
                        R,
                        end-start, 
                        1.0, 
                        in(start * R), 
                        R, 
                        0.0, 
                        local(), 
                        R);

            for(uint64_t i = 0; i < R * R; i++) {
                #pragma omp atomic
                M[i] += local[i];
            }
        }

        for(uint64_t i = 0; i < R; i++) {
            for(uint64_t j = 0; j < i; j++) {
                M[i * R + j] = M[j * R + i];
            }
        } 
    }
}

void compute_pinv_square(Buffer<double> &M, Buffer<double> &out, uint64_t target_rank) {
    uint64_t R = M.shape[0];
    double eigenvalue_tolerance = 1e-11;
    Buffer<double> lambda({R});

    LAPACKE_dsyev( CblasRowMajor, 
                    'V', 
                    'U', 
                    R,
                    M(), 
                    R, 
                    lambda() );

    for(uint32_t v = 0; v < R; v++) {
        if(v >= R - target_rank && lambda[v] > eigenvalue_tolerance) {
            for(uint32_t u = 0; u < R; u++) {
                M[u * R + v] = M[u * R + v] / sqrt(lambda[v]); 
            }
        }
        else {
            for(uint32_t u = 0; u < R; u++) {
                M[u * R + v] = 0.0; 
            }
        }
    }

    cblas_dsyrk(CblasRowMajor, 
                CblasUpper, 
                CblasNoTrans,
                R,
                R, 
                1.0, 
                (const double*) M(), 
                R, 
                0.0, 
                out(), 
                R);

}

void ATB_chain_prod(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B,
        Buffer<double> &result,
        int exclude) {

        uint64_t N = A.size();
        uint64_t R_A = A[0].shape[1];
        uint64_t R_B = B[0].shape[1];

        vector<unique_ptr<Buffer<double>>> ATB;
        for(uint64_t i = 0; i < A.size(); i++) {
                ATB.emplace_back();
                ATB[i].reset(new Buffer<double>({R_A, R_B}));
        }

        for(uint64_t i = 0; i < R_A; i++) {
                for(uint64_t j = 0; j < R_B; j++) {
                        result[i * R_B + j] = sigma_A[i] * sigma_B[j];
                }

        }

        // Can replace with a batch DGEMM call
        for(uint64_t i = 0; i < N; i++) {
            if(((int) i) != exclude) {
                uint64_t K = A[i].shape[0];
                cblas_dgemm(
                        CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        R_A,
                        R_B,
                        K,
                        1.0,
                        A[i](),
                        R_A,
                        B[i](),
                        R_B,
                        0.0,
                        (*(ATB[i]))(),
                        R_B
                );
            }
        }

        #pragma omp parallel 
{
        for(uint64_t k = 0; k < N; k++) {
                if(((int) k) != exclude) {
                    #pragma omp for collapse(2)
                    for(uint64_t i = 0; i < R_A; i++) {
                            for(uint64_t j = 0; j < R_B; j++) {
                                    result[i * R_B + j] *= (*(ATB[k]))[i * R_B + j];
                            }
                    }
                }
        }
}
}

double ATB_chain_prod_sum(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B) {

    uint64_t R_A = A[0].shape[1];
    uint64_t R_B = B[0].shape[1];
    Buffer<double> result({R_A, R_B});
    ATB_chain_prod(A, B, sigma_A, sigma_B, result, -1);
    return std::accumulate(result(), result(R_A * R_B), 0.0); 
}

void chain_had_prod(
        vector<Buffer<double>> &A,
        Buffer<double> &result,
        int exclude) {

        uint64_t R_A = A[0].shape[0];
        uint64_t R_B = A[0].shape[1];

        std::fill(result(), result(R_A * R_B), 1.0);

        #pragma omp parallel 
{
        for(uint64_t k = 0; k < A.size(); k++) {
                if(((int) k) != exclude) {
                    #pragma omp for collapse(2)
                    for(uint64_t i = 0; i < R_A; i++) {
                            for(uint64_t j = 0; j < R_B; j++) {
                                    result[i * R_B + j] *= A[k][i * R_B + j];
                            }
                    }
                }
        }
}

}

template <typename VAL_T>
void allgatherv_buffer(Buffer<VAL_T> &input, Buffer<VAL_T> &output, MPI_Comm comm) {
    int row_count = (int) input.shape[0];
    int world_size;
    MPI_Comm_size(comm, &world_size);

    Buffer<int> recvcounts({(uint64_t) world_size});
    Buffer<int> displs({(uint64_t) world_size});

    MPI_Allgather(
        &row_count,
        1,
        MPI_INT32_T,
        recvcounts(),
        1,
        MPI_INT32_T,
        comm
    );

    int total_rows = std::accumulate(recvcounts(), recvcounts(world_size), 0);
    int send_size;

    if(input.shape.size() == 2) {
        output.initialize_to_shape({(uint64_t) total_rows, input.shape[1]});

        for(int i = 0; i < world_size; i++) {
            recvcounts[i] *= input.shape[1]; 
        }
        send_size = (int) (input.shape[0] * input.shape[1]);
    }
    else {
        output.initialize_to_shape({(uint64_t) total_rows});
        send_size = (int) input.shape[0];
    }

    MPI_Datatype dtype = get_MPI_dtype<VAL_T>();

    std::exclusive_scan(recvcounts(), recvcounts(world_size), displs(), 0);

    MPI_Allgatherv(
        input(),
        send_size,
        dtype,
        output(),
        recvcounts(),
        displs(),
        dtype,
        comm
    );
}

template<typename VAL_T>
void apply_permutation(Buffer<uint32_t> &perm, Buffer<VAL_T> &arr) {
    if(arr.shape.size() != 1) {
        throw std::runtime_error("apply_random_permutation only works on 1D arrays");
    }
    Buffer<VAL_T> temp({arr.shape[0]});

    #pragma omp parallel for
    for(uint64_t i = 0; i < arr.shape[0]; i++) {
        temp[i] = arr[perm[i]];
    }

    arr.steal_resources(temp);
}

json compute_dstat(double quantity, MPI_Comm world) {
    int world_size;
    MPI_Comm_size(world, &world_size);    

    json result;
    double mean;
    double std;
    double min;
    double max;

    MPI_Allreduce(
        &quantity,
        &mean,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        world
    );
    mean /= world_size;

    double diff_to_mean_sq = (quantity - mean) * (quantity - mean);

    MPI_Allreduce(
        &diff_to_mean_sq,
        &std,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        world
    );

    if(world_size > 0) {
        std = sqrt(std / (world_size - 1));
    }
    else {
        std = 0.0;
    }


    MPI_Allreduce(
        &quantity,
        &min,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        world
    );

    MPI_Allreduce(
        &quantity,
        &max,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        world
    );

    result["mean"] = mean;
    result["std"] = std;
    result["min"] = min;
    result["max"] = max;
    return result;
}

inline uint32_t divide_and_roundup(uint32_t n, uint32_t m) {
    return (n + m - 1) / m;
}

inline void log2_round_down(uint32_t m, 
        uint32_t& log2_res, 
        uint32_t& lowest_power_2) {
    
    assert(m > 0);
    log2_res = 0;
    lowest_power_2 = 1;

    while(lowest_power_2 * 2 <= m) {
        log2_res++; 
        lowest_power_2 *= 2;
    }
}


void parallel_dsymm(Buffer<double> &sym, Buffer<double> &mat, Buffer<double> &out) {
    uint32_t R = (uint32_t) mat.shape[1];

    uint64_t I = mat.shape[0];
    int num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    uint64_t work = (I + num_threads - 1) / num_threads;
    uint64_t start = min(work * thread_id, I);
    uint64_t end = min(work * (thread_id + 1), I);

    if(end - start > 0) {
        cblas_dsymm(
            CblasRowMajor,
            CblasRight,
            CblasUpper,
            (uint32_t) (end - start),
            (uint32_t) sym.shape[1],
            1.0,
            sym(),
            R,
            mat(start * R),
            R,
            0.0,
            out(start * R),
            R
        );
    }

    #pragma omp barrier
}

void compute_DAGAT(double* A, double* G, 
        double* res, uint64_t J, uint64_t R) {

    Buffer<double> A_buf({J, R}, A);
    Buffer<double> G_buf({R, R}, G);
    Buffer<double> temp({J, R});

    #pragma omp parallel
    {
        parallel_dsymm(G_buf, A_buf, temp);

        #pragma omp for 
        for(uint32_t i = 0; i < J; i++) {
            res[i] = 0.0;
            for(uint32_t j = 0; j < R; j++) {
                res[i] += A[i * R + j] * temp[i * R + j];
            }
        }
    }
}

template<typename T>
void parallel_exclusive_scan(T* start, T* end, T* out) {
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    uint64_t rangelen = (uint64_t) (end - start);

    uint64_t work = (rangelen + num_threads - 1) / num_threads;
    uint64_t s_chunk = min(work * tid, rangelen);
    uint64_t e_chunk = min(work * (tid + 1), rangelen);
 
    T* workspace;
    T* segment_sums; 
    #pragma omp single copyprivate(workspace, segment_sums)
    {
        workspace = new T[num_threads];
        segment_sums = new T[num_threads];
    }

    std::exclusive_scan(start + s_chunk, start + e_chunk, out + s_chunk, 0);
    if(e_chunk - s_chunk > 0) {
        workspace[tid] = out[e_chunk - 1] + start[e_chunk - 1]; 
    }

    #pragma omp barrier
    std::exclusive_scan(workspace, 
                        workspace + num_threads, 
                        segment_sums, (T) 0 
                        );

    for(uint64_t i = s_chunk; i < e_chunk; i++) {
        out[i] += segment_sums[tid];
    }

    #pragma omp barrier
    #pragma omp single
    {
        delete[] workspace;
        delete[] segment_sums;
    }
}


