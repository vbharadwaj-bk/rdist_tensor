import os, sys


from distutils import sysconfig
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

cpp_args = ['-std=c++11']

ext_modules = [
    Pybind11Extension(
    'redistribute_tensor',
        ['redistribute_tensor.cpp'],
        include_dirs=['pybind11/include'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(
    name='redistribute_tensor',
    version='0.0.1',
    author='Vivek Bharadwaj',
    author_email='vivek_bharadwaj@berkeley.edu',
    description='CPP Extensions for rdist tensor',
    ext_modules=ext_modules,
)