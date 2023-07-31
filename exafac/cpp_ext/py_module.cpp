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
#include "distmat.hpp"
#include "low_rank_tensor.hpp"
#include "sparse_tensor.hpp"
#include "exact_als.hpp"
#include "tensor_stationary_opt0.hpp"
#include "accumulator_stationary_opt0.hpp"

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
            std::string>());
    py::class_<ExactALS>(m, "ExactALS")
        .def(py::init<SparseTensor&, LowRankTensor&>())
        .def("execute_ALS_rounds", &ExactALS::execute_ALS_rounds)
        .def("compute_exact_fit", &ExactALS::compute_exact_fit);  
    py::class_<TensorStationaryOpt0>(m, "TensorStationaryOpt0")
        .def(py::init<SparseTensor&, LowRankTensor&>())
        .def("execute_ALS_rounds", &TensorStationaryOpt0::execute_ALS_rounds)
        .def("compute_exact_fit", &TensorStationaryOpt0::compute_exact_fit);
    py::class_<AccumulatorStationaryOpt0>(m, "AccumulatorStationaryOpt0")
        .def(py::init<SparseTensor&, LowRankTensor&>())
        .def("execute_ALS_rounds", &AccumulatorStationaryOpt0::execute_ALS_rounds);
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
                        'alltoallv_revised.hpp',
                        'grid.hpp',
                        'distmat.hpp',
                        'low_rank_tensor.hpp',
                        'sparse_tensor.hpp',
                        'sort_lookup.hpp',
                        'exact_als.hpp',
                        'tensor_stationary_opt0.hpp',
                        'accumulator_stationary_opt0.hpp',
                        '../../config.json' 
                        ]
cfg['libraries'] = ['tbb']
%>
*/
