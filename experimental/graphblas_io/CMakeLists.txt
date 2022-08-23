cmake_minimum_required(VERSION 3.16)

project(graphblas_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

configure_file(modules.sh modules.sh COPYONLY)

add_executable(grb_test prog.cpp)
target_include_directories(grb_test PUBLIC /global/project/projectdirs/m1982/vbharadw/rdist_tensor/GraphBLAS/Include)
target_compile_options(grb_test PRIVATE -O3)

target_link_options(grb_test PRIVATE -L/global/project/projectdirs/m1982/vbharadw/rdist_tensor/GraphBLAS/build/ -lgraphblas

)