# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# MPAS-Atmosphere + C++ Dycore Fortran Integration Build
#
# Extends the mpas-dycore-build image (which already provides Ubuntu 24.04,
# GCC 13.3, MPICH 5.0.1, CMake 3.31, NetCDF-C/Fortran, HDF5, PIO 2.x,
# PnetCDF, Kokkos (Serial+OpenMP), and KokkosKernels (Graph component)) to
# compile the full MPAS-Atmosphere executable with USE_CPP_DYCORE=true.
#
# The resulting image contains a working `atmosphere_model` binary linked
# against the C++ dynamical core library.
#
# Prerequisites:
#   docker build --platform linux/amd64 \
#       -f docker/mpas-dycore-build/Dockerfile -t mpas-dycore-build .
#
# Build from the repository root with:
#   docker build --platform linux/amd64 \
#       -f docker/mpas-fortran-integration/Dockerfile \
#       -t mpas-fortran-integration .
#
# Requirements: 5.1, 5.2, 5.3
# ─────────────────────────────────────────────────────────────────────────────
FROM mpas-dycore-build

# Copy the entire MPAS source tree into the image.
COPY . /work/MPAS-Model

WORKDIR /work/MPAS-Model

# Build MPAS-Atmosphere with the C++ dycore enabled.
# The base image provides all required dependencies (gfortran, MPI, PIO,
# netCDF, PnetCDF, Kokkos, KokkosKernels) with environment variables already
# configured (CMAKE_PREFIX_PATH, NETCDF, PIO, PNETCDF, Kokkos_ROOT, etc.).
# Uses the MPAS built-in ESMF timekeeping library (no MPAS_EXTERNAL_ESMF_LIB).
RUN make clean CORE=atmosphere 2>/dev/null; \
    make gnu CORE=atmosphere USE_CPP_DYCORE=true AUTOCLEAN=true

# Verify the binary was produced.
RUN test -x atmosphere_model

# Default command: print version info to confirm the build is functional.
CMD ["./atmosphere_model", "--version"]
