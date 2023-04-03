//cppimport
#include <iostream>
#include <mpi.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.h"
#include "grid.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(py_module, m) {
    py::class_<Grid>(m, "Grid")
        .def(py::init<py::array_t<int>>());
}

/*
<%
setup_pybind11(cfg)
# cfg['extra_compile_args'] = ['-fopenmp', '-O3', '-march=native']
# cfg['extra_link_args'] = ['-openmp', '-O3']

import json
config = None
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
# Required compiler flags 
compile_args=config['required_compile_args']
link_args=config['required_link_args']

# Add extra flags for the BLAS and LAPACK 
for lst in [config["blas_include_flags"], 
            config["tbb_include_flags"], 
            config["extra_compile_args"]]:
    compile_args.extend(lst)
for lst in [config["blas_link_flags"], 
            config["tbb_link_flags"], 
            config["extra_link_args"]]:
    link_args.extend(lst)

print(f"Compiling C++ extensions with {compile_args}")
print(f"Linking C++ extensions with {link_args}")
cfg['extra_compile_args'] = compile_args 
cfg['extra_link_args'] = link_args 
cfg['dependencies'] = [ 'common.h', 
                        'grid.hpp', 
                        '../../config.json' 
                        ]
cfg['libraries'] = ['tbb']
%>
*/
