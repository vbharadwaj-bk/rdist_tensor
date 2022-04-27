/*
<%
setup_pybind11(cfg)
%>
*/

// cppimport
#include <pybind11/pybind11.h>

namespace py = pybind11;

int square(int x) {
    return x;
}

PYBIND11_MODULE(test, m) {
    m.def("square", &square);
}