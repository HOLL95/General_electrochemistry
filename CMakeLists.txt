cmake_minimum_required(VERSION 3.1)

project(isolver_martin_brent)


set(include_dir "lib")
set(source_dir "src")
set (PYBIND11_PYTHON_VERSION "3.10")
# Boost
list(APPEND include_dirs ${include_dir}/boost)
# Pybind11
add_subdirectory(${include_dir}/pybind11 pybind11)
include_directories(${source_dir} ${include_dirs})
pybind11_add_module(isolver_martin_brent "${source_dir}/surface_process_brent.cpp")
pybind11_add_module(SWV_surface "${source_dir}/surface_process_SWV.cpp")
