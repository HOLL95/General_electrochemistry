cmake_minimum_required(VERSION 3.1)

project(isolver_martin_brent)


set(include_dir "lib")
set(source_dir "src")
#Pints is current incomaptible with python 3.11. If you are using python 3.11, you will need to run everything in =<3.10. Compiling the simulation code with an earlier python version can be achieved by uncommenting the line below, and setting the appropriate version number
#set (PYBIND11_PYTHON_VERSION "3.10")
# Boost
list(APPEND include_dirs ${include_dir}/boost)
# Pybind11
add_subdirectory(${include_dir}/pybind11 pybind11)
include_directories(${source_dir} ${include_dirs})
pybind11_add_module(isolver_martin_brent "src/surface_process_brent.cpp")
pybind11_add_module(SWV_surface "src/surface_process_SWV.cpp")
