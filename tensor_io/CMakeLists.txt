cmake_minimum_required(VERSION 3.16)

project(tensor_io)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_C_COMPILER "h5cc")
set(CMAKE_CXX_COMPILER "h5c++")

configure_file(modules.sh modules.sh COPYONLY)

add_executable(process_frostt_tensor process_frostt_tensor.cpp)
add_executable(process_konect_tensor process_konect_tensor.cpp)
add_executable(convert_konect_to_tns convert_konect_to_tns.cpp)