cmake_minimum_required(VERSION 3.16)

project(shmem_test )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

configure_file(modules.sh modules.sh COPYONLY)

add_executable(shmem_test shmemx_test.cpp)
target_compile_options(shmem_test PRIVATE -O3)
