# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# MPAS + C++/Kokkos dycore build image.
#
# Extends the NOAA-EMC Spack build-cache image (Ubuntu 24.04, GCC 13.3,
# MPICH 5.0.1, CMake 3.31, NetCDF-C/Fortran, HDF5, PIO 2.x, ESMF) with the
# dependencies MPAS-Atmosphere and the C++ dynamical-core port additionally
# require but that the base image ships without:
#
#   * PnetCDF          - MPAS Fortran I/O (find_package(PnetCDF) / Make PNETCDF)
#   * Kokkos           - C++ dycore parallel-execution foundation
#   * KokkosKernels    - C++ dycore graph coloring (KokkosGraph)
#
# Everything is installed into a single prefix (/opt/deps) that is layered on
# top of the base Spack view (/opt/views/view) via CMAKE_PREFIX_PATH, PATH and
# LD_LIBRARY_PATH so both build systems (CMake and the classic MPAS Makefile)
# discover the new libraries automatically.
#
# The base image is amd64-only; build with --platform linux/amd64.
# ─────────────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=ghcr.io/noaa-emc/ci-common-build-cache/ufs-utils-ubuntu-24.04-gcc-13-mpich-x:latest
FROM ${BASE_IMAGE}

# Dependency versions (overridable at build time).
ARG PNETCDF_VERSION=1.13.0
ARG KOKKOS_VERSION=4.5.01
ARG KOKKOSKERNELS_VERSION=4.5.01

# Common install prefix layered over the Spack view.
ENV DEPS_PREFIX=/opt/deps
ENV BUILD_JOBS=8

SHELL ["/bin/bash", "-c"]

# The base image ships MPICH via the Spack view; make sure the MPI wrappers are
# on PATH for every RUN below (the base sets these but be explicit/robust).
ENV PATH=/opt/deps/bin:/opt/views/view/bin:/opt/spack/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ─────────────────────────────────────────────────────────────────────────────
# 0) Install build utilities missing from the base image.
#    - m4: required by PnetCDF's configure script
#    - cpp: C preprocessor needed by MPAS Fortran build for .F files
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends m4 cpp && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# 1) PnetCDF - built against the image's MPICH, with the Fortran interface
#    (pnetcdf.mod) that MPAS's find_package(PnetCDF COMPONENTS Fortran) needs.
# ─────────────────────────────────────────────────────────────────────────────
RUN set -eux; \
    mkdir -p /tmp/build && cd /tmp/build; \
    curl -fsSL --retry 3 --retry-delay 5 "https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz" -o pnetcdf.tar.gz; \
    tar xzf pnetcdf.tar.gz; \
    cd "pnetcdf-${PNETCDF_VERSION}"; \
    ./configure --prefix="${DEPS_PREFIX}" \
        MPICC=mpicc MPICXX=mpicxx MPIF77=mpif77 MPIF90=mpif90 \
        CFLAGS="-fPIC" CXXFLAGS="-fPIC" FFLAGS="-fPIC" FCFLAGS="-fPIC" \
        --enable-shared --enable-static; \
    make -j"${BUILD_JOBS}"; \
    make install; \
    cd / && rm -rf /tmp/build; \
    "${DEPS_PREFIX}/bin/pnetcdf-config" --version

# ─────────────────────────────────────────────────────────────────────────────
# 2) Kokkos - SERIAL + OPENMP backends, C++20, no CUDA/HIP.
# ─────────────────────────────────────────────────────────────────────────────
RUN set -eux; \
    mkdir -p /tmp/build && cd /tmp/build; \
    curl -fsSL --retry 3 --retry-delay 5 "https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VERSION}.tar.gz" -o kokkos.tar.gz; \
    tar xzf kokkos.tar.gz; \
    cmake -S "kokkos-${KOKKOS_VERSION}" -B build-kokkos \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_STANDARD=20 \
        -DKokkos_ENABLE_SERIAL=ON \
        -DKokkos_ENABLE_OPENMP=ON \
        -DKokkos_ENABLE_CUDA=OFF \
        -DKokkos_ENABLE_HIP=OFF \
        -DBUILD_SHARED_LIBS=ON; \
    cmake --build build-kokkos -j"${BUILD_JOBS}"; \
    cmake --install build-kokkos; \
    cd / && rm -rf /tmp/build

# ─────────────────────────────────────────────────────────────────────────────
# 3) KokkosKernels - only the GRAPH component is needed by the dycore
#    (KokkosGraph_graph_color.hpp / graph coloring). Restricting to the GRAPH
#    component (ALL_COMPONENTS=OFF + COMPONENT_GRAPH=ON) avoids the very long
#    ETI compile of BLAS/SPARSE/LAPACK/BATCHED that the dycore never uses, which
#    matters a great deal under amd64 emulation. find_package(KokkosKernels)
#    still resolves the Kokkos::kokkoskernels target.
# ─────────────────────────────────────────────────────────────────────────────
RUN set -eux; \
    mkdir -p /tmp/build && cd /tmp/build; \
    curl -fsSL --retry 3 --retry-delay 5 "https://github.com/kokkos/kokkos-kernels/archive/refs/tags/${KOKKOSKERNELS_VERSION}.tar.gz" -o kk.tar.gz; \
    tar xzf kk.tar.gz; \
    cmake -S "kokkos-kernels-${KOKKOSKERNELS_VERSION}" -B build-kk \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_STANDARD=20 \
        -DKokkos_ROOT="${DEPS_PREFIX}" \
        -DKokkosKernels_ENABLE_ALL_COMPONENTS=OFF \
        -DKokkosKernels_ENABLE_COMPONENT_GRAPH=ON \
        -DKokkosKernels_ENABLE_TESTS=OFF \
        -DKokkosKernels_ENABLE_EXAMPLES=OFF \
        -DKokkosKernels_ENABLE_DOCS=OFF \
        -DBUILD_SHARED_LIBS=ON; \
    cmake --build build-kk -j"${BUILD_JOBS}"; \
    cmake --install build-kk; \
    cd / && rm -rf /tmp/build

# ─────────────────────────────────────────────────────────────────────────────
# Environment: layer /opt/deps over the base Spack view so both the CMake and
# the classic Makefile MPAS build systems find PnetCDF, Kokkos and KokkosKernels.
# ─────────────────────────────────────────────────────────────────────────────
ENV CMAKE_PREFIX_PATH=/opt/deps:/opt/views/view
ENV LD_LIBRARY_PATH=/opt/deps/lib:/opt/deps/lib64:/opt/views/view/lib64:/opt/views/view/lib
ENV PKG_CONFIG_PATH=/opt/deps/lib/pkgconfig:/opt/views/view/lib/pkgconfig:/opt/views/view/lib64/pkgconfig

# PnetCDF discovery: env var for the classic Makefile ($(PNETCDF)/lib/libpnetcdf.*)
# and for MPAS's FindPnetCDF.cmake (PnetCDF_ROOT / PNETCDF_ROOT -> pnetcdf-config).
ENV PNETCDF=/opt/deps
ENV PNETCDF_ROOT=/opt/deps
ENV PnetCDF_ROOT=/opt/deps

# NetCDF / PIO discovery for the classic Makefile (come from the base view).
ENV NETCDF=/opt/views/view
ENV NETCDFF=/opt/views/view
ENV PIO=/opt/views/view

# Kokkos / KokkosKernels discovery hints.
ENV Kokkos_ROOT=/opt/deps
ENV KokkosKernels_ROOT=/opt/deps

WORKDIR /work
