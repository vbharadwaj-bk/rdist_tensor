cmake_minimum_required(VERSION 3.16)

project(tensor_io)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

configure_file(modules.sh modules.sh COPYONLY)

add_executable(process_frostt_tensor process_frostt_tensor.cpp)

#target_compile_options(process_frostt_tensor PRIVATE -g) 

#foreach(TARGET bench_file bench_er bench_heatmap local_kernel_benchmark scratch)
#    target_compile_options(${TARGET} PRIVATE -O3 -march=knl -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
#    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
#    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib )
#    target_precompile_headers(${TARGET} PRIVATE json.hpp <Eigen/Dense>)
#endforeach()


#foreach(TARGET random_permute)
#    target_compile_options(${TARGET} PRIVATE -O3 -march=haswell -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
#    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
#    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib )
#    target_precompile_headers(${TARGET} PRIVATE json.hpp <Eigen/Dense>)
#endforeach()