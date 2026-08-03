cmake_minimum_required(VERSION 3.25)
project(mpas_dycore_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options propagated from MPAS Makefile
option(MPAS_USE_OPENMP "Enable OpenMP execution policy" OFF)
option(MPAS_USE_GPU "Enable GPU execution policy" OFF)
set(MPAS_GPU_BACKEND "" CACHE STRING "GPU backend: CUDA, HIP, or SYCL")
option(MPAS_BOUNDS_CHECK "Use checked_accessor (bounds checking)" ON)

# Find MPI
find_package(MPI REQUIRED COMPONENTS CXX)

# Library target
add_library(mpas_dycore_cpp STATIC
    src/placeholder.cpp
    src/mesh.cpp
    src/halo.cpp
    src/state.cpp
    src/api.cpp
    src/kernels/moist_coefficients.cpp
    src/kernels/vert_implicit.cpp
    src/kernels/diagnostics.cpp
    src/kernels/tend_u.cpp
    src/kernels/tend_theta.cpp
    src/kernels/tend_w.cpp
    src/kernels/rayleigh_damping.cpp
    src/kernels/dyn_tend.cpp
    src/kernels/divergence_damping.cpp
    src/kernels/acoustic.cpp
    src/kernels/scalars.cpp
    src/kernels/les_models.cpp
    src/kernels/dissipation.cpp
    src/kernels/boundaries.cpp
    src/kernels/iau.cpp
    src/kernels/velocity_reconstruct.cpp
)

target_include_directories(mpas_dycore_cpp PUBLIC include)
target_link_libraries(mpas_dycore_cpp PRIVATE MPI::MPI_CXX)

target_compile_definitions(mpas_dycore_cpp PUBLIC
    $<$<BOOL:${MPAS_BOUNDS_CHECK}>:MPAS_BOUNDS_CHECK>
)

if(MPAS_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(mpas_dycore_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(mpas_dycore_cpp PRIVATE MPAS_USE_OPENMP)
endif()

if(MPAS_USE_GPU)
    target_compile_definitions(mpas_dycore_cpp PRIVATE MPAS_USE_GPU)
    if(MPAS_GPU_BACKEND STREQUAL "CUDA")
        target_compile_definitions(mpas_dycore_cpp PRIVATE MPAS_GPU_BACKEND_CUDA)
    elseif(MPAS_GPU_BACKEND STREQUAL "HIP")
        target_compile_definitions(mpas_dycore_cpp PRIVATE MPAS_GPU_BACKEND_HIP)
    elseif(MPAS_GPU_BACKEND STREQUAL "SYCL")
        target_compile_definitions(mpas_dycore_cpp PRIVATE MPAS_GPU_BACKEND_SYCL)
    endif()
endif()

# Testing dependencies via FetchContent
include(FetchContent)

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

FetchContent_Declare(rapidcheck
    GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
    GIT_TAG master
)

# Enable RapidCheck's GTest integration extras
set(RC_ENABLE_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest rapidcheck)

# Enable testing and add test subdirectory
enable_testing()
add_subdirectory(tests)
