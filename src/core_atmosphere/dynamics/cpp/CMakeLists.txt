cmake_minimum_required(VERSION 3.21)
project(mpas_dycore_cpp
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(MPAS_DOUBLE_PRECISION "Use double precision for Scalar (RKIND)" ON)
option(MPAS_PHYSICS_ENABLED "Include physics halo groups (physics:blten, physics:cuten)" OFF)
option(MPAS_BUILD_TESTING "Build the Component_Test_Suite" ON)

# Dependencies
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

# Attempt to locate pre-installed HALO, fall back to submodule if needed
find_package(HALO QUIET)
if(NOT HALO_FOUND)
    message(STATUS "HALO not found via find_package. Adding helm-project/libs/halo subdirectory.")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../helm-project/libs/halo" "${CMAKE_CURRENT_BINARY_DIR}/halo")
endif()

# Configuration Header for Scalar Precision
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/mpas_dycore/scalar.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/mpas_dycore/scalar.hpp"
)

# Main Library
add_library(dycore_cpp STATIC
    src/mpas_dycore.cpp
    src/dycore_c_api.cpp
)
set_target_properties(dycore_cpp PROPERTIES OUTPUT_NAME mpas_dycore_cpp)
add_library(MPAS::dycore_cpp ALIAS dycore_cpp)

target_include_directories(dycore_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

target_link_libraries(dycore_cpp
    PUBLIC
        Kokkos::kokkos
        Kokkos::kokkoskernels
        halo
)

if(MPAS_PHYSICS_ENABLED)
    target_compile_definitions(dycore_cpp PUBLIC MPAS_PHYSICS_ENABLED)
endif()

if(MPAS_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
