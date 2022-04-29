/*
<%
cfg['compiler_args'] = ['-std=c++11']
setup_pybind11(cfg)
%>
*/

// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <mpi.h>

namespace py = pybind11;

double square(int x) {
    return x;
}

unsigned long long sum_all_elements(py::list my_list) {
    unsigned long long total_sum = 0;

    for(py::handle obj : my_list) {
        //std::cout << obj << std::endl;

        py::array_t<unsigned long long> idxs = obj.cast<py::array_t<unsigned long long>>();

        py::buffer_info info = idxs.request();
        auto ptr = static_cast<unsigned long long*>(info.ptr);

        for(int i = 0; i < info.shape[0]; i++) {
            total_sum += ptr[i];
        }
    }

    return total_sum;
}

/*
 * Count up the nonzeros and allocate the receive buffers 
 */
void redistribute_nonzeros_helper1() {

}

PYBIND11_MODULE(redistribute_tensor, m) {
    m.def("sum_all_elements", &sum_all_elements);
}