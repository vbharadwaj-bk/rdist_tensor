cmake_minimum_required(VERSION 3.16)

project(one_sided_comm_bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

configure_file(modules.sh modules.sh COPYONLY)

add_executable(bench_accum bench_accum_get.cpp)