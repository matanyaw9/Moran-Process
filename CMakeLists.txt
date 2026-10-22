cmake_minimum_required(VERSION 3.15...3.29)
project(moran_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# scikit-build-core provides Python; pybind11 is found via its installed config.
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_moran_cpp src/moran_process/simulations/_cpp/moran_core.cpp)
target_compile_options(_moran_cpp PRIVATE -O3 -funroll-loops)

# Land the compiled extension next to the Python package as
# moran_process/_moran_cpp.<abi>.so
install(TARGETS _moran_cpp LIBRARY DESTINATION moran_process)
