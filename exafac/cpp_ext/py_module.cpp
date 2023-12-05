//cppimport
#include <iostream>
#include <mpi.h>

#include <algorithm>
#include <execution>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.h"
#include "grid.hpp"
#include "alltoallv_revised.hpp"

#include "distmat.hpp"
#include "low_rank_tensor.hpp"
#include "sparse_tensor.hpp"

#include "als_optimizer.hpp"
#include "exact_als.hpp"
#include "tensor_stationary.hpp"
#include "accumulator_stationary.hpp"

#include "sampler.hpp"
#include "cp_arls_lev.hpp"
#include "exact_leverage_tree.hpp"
#include "sts_cp.hpp"


using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(py_module, m) {
    py::class_<Grid>(m, "Grid")
        .def(py::init<py::array_t<int>>())
        .def("get_dimension", &Grid::get_dimension);
    py::class_<TensorGrid>(m, "TensorGrid")
        .def(py::init<py::array_t<int>, Grid&>());
    py::class_<DistMat1D>(m, "DistMat1D")
        .def(py::init<uint64_t, TensorGrid&, uint64_t>());
    py::class_<LowRankTensor>(m, "LowRankTensor")
        .def(py::init<uint64_t, TensorGrid&>())
        .def("test_gram_matrix_computation", &LowRankTensor::test_gram_matrix_computation)
        .def("initialize_factors_deterministic", &LowRankTensor::initialize_factors_deterministic)
        .def("initialize_factors_gaussian_random", &LowRankTensor::initialize_factors_gaussian_random);
    py::class_<SparseTensor>(m, "SparseTensor")
        .def(py::init<TensorGrid&, 
            py::array_t<uint32_t>, 
            py::array_t<double>, 
            std::string>())    
        .def(py::init<TensorGrid&, 
            uint64_t, 
            uint64_t, 
            uint64_t);
    py::class_<ALS_Optimizer>(m, "ALS_Optimizer")
        .def("initialize_ground_truth_for_als", &ALS_Optimizer::initialize_ground_truth_for_als)
        .def("execute_ALS_rounds", &ALS_Optimizer::execute_ALS_rounds)
        .def("get_statistics_json", &ALS_Optimizer::get_statistics_json)
        .def("compute_exact_fit", &ALS_Optimizer::compute_exact_fit)
        .def("deinitialize", &ALS_Optimizer::deinitialize);
    py::class_<ExactALS, ALS_Optimizer>(m, "ExactALS")
        .def(py::init<SparseTensor&, LowRankTensor&>());
    py::class_<TensorStationary, ALS_Optimizer>(m, "TensorStationary")
        .def(py::init<SparseTensor&, LowRankTensor&, Sampler&>());
    py::class_<AccumulatorStationary, ALS_Optimizer>(m, "AccumulatorStationary")
        .def(py::init<SparseTensor&, LowRankTensor&, Sampler&>());

    py::class_<Sampler>(m, "Sampler");
    py::class_<CP_ARLS_LEV, Sampler>(m, "CP_ARLS_LEV")
        .def(py::init<LowRankTensor&>());
    py::class_<STS_CP, Sampler>(m, "STS_CP")
        .def(py::init<LowRankTensor&>());
}

/*
<%
setup_pybind11(cfg)

import json
config = None
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
# Required compiler flags 
compile_args=config['required_compile_args']
link_args=config['required_link_args']

blas_include_path=[f'-I{config["blas_include_path"]}']
blas_link_path=[f'-L{config["blas_link_path"]}']

tbb_include_path=[f'-I{config["tbb_include_path"]}']
tbb_link_path=[f'-L{config["tbb_link_path"]}']
rpath_options=[f'-Wl,-rpath,{config["blas_link_path"]}:{config["tbb_link_path"]}']

for lst in [blas_include_path,
            tbb_include_path,
            config["extra_compile_args"]
            ]:
    compile_args.extend(lst)
for lst in [blas_link_path,
            tbb_link_path,
            config["blas_link_flags"], 
            config["tbb_link_flags"], 
            config["extra_link_args"],
            rpath_options 
            ]:
    link_args.extend(lst)

print(f"Compiling C++ extensions with {compile_args}")
print(f"Linking C++ extensions with {link_args}")
cfg['extra_compile_args'] = compile_args 
cfg['extra_link_args'] = link_args 
cfg['dependencies'] = [ 'common.h',
                        'grid.hpp',
                        'alltoallv_revised.hpp',
                        'distmat.hpp',
                        'low_rank_tensor.hpp',
                        'sparse_tensor.hpp',
                        'sort_lookup.hpp',
                        'als_optimizer.hpp',
                        'exact_als.hpp',
                        'accumulator_stationary.hpp',
                        'tensor_stationary.hpp',
                        'sampler.hpp',
                        'cp_arls_lev.hpp',
                        'partition_tree.hpp',
                        'exact_leverage_tree.hpp',
                        'sts_cp.hpp',
                        '../../config.json' 
                        ]
cfg['libraries'] = ['tbb']
%>
*/
