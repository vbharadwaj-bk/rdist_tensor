//cppimport
#include <iostream>
#include <mpi.h>

#include <cblas.h>
#include <lapacke.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.h"
#include "grid.hpp"
#include "distmat.hpp"
#include "low_rank_tensor.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(py_module, m) {
    py::class_<Grid>(m, "Grid")
        .def(py::init<py::array_t<int>>());
    py::class_<TensorGrid>(m, "TensorGrid")
        .def(py::init<py::array_t<int>, Grid&>());
    py::class_<DistMat1D>(m, "DistMat1D")
        .def(py::init<uint64_t, TensorGrid&, uint64_t>());
    py::class_<LowRankTensor>(m, "LowRankTensor")
        .def(py::init<uint64_t, TensorGrid&>())
        .def("test_gram_matrix_computation", &LowRankTensor::test_gram_matrix_computation);
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
                        'distmat.hpp',
                        'low_rank_tensor.hpp', 
                        '../../config.json' 
                        ]
cfg['libraries'] = ['tbb']
%>
*/
