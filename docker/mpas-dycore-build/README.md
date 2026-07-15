# MPAS C++ Dycore Build Image

Docker image extending the NOAA-EMC CI base with dependencies needed by
MPAS-Atmosphere and the C++/Kokkos dynamical-core port.

## Image Contents

| Component | Version | Location |
|-----------|---------|----------|
| **Base** (GCC 13.3, MPICH 5.0.1, CMake 3.31, NetCDF-C/Fortran, HDF5, PIO 2.x, ESMF) | various | `/opt/views/view` |
| PnetCDF (with Fortran interface) | 1.13.0 | `/opt/deps` |
| Kokkos (Serial + OpenMP, C++20) | 4.5.1 | `/opt/deps` |
| KokkosKernels (Graph component) | 4.5.1 | `/opt/deps` |

### Environment Variables

The image sets discovery variables so both CMake and the classic MPAS Makefile
find all dependencies automatically:

- `CMAKE_PREFIX_PATH=/opt/deps:/opt/views/view`
- `PNETCDF=/opt/deps` (classic Makefile), `PnetCDF_ROOT=/opt/deps` (CMake)
- `Kokkos_ROOT=/opt/deps`, `KokkosKernels_ROOT=/opt/deps`
- `NETCDF=/opt/views/view`, `PIO=/opt/views/view`

## Building the Image

```bash
docker build --platform linux/amd64 \
    -t mpas-dycore-build:latest \
    docker/mpas-dycore-build/
```

> **Note:** The build compiles PnetCDF, Kokkos, and KokkosKernels from source.
> Expect 20-40 minutes on native amd64, longer under emulation (e.g., Apple Silicon).

### Build-time arguments

| ARG | Default | Description |
|-----|---------|-------------|
| `PNETCDF_VERSION` | 1.13.0 | PnetCDF release version |
| `KOKKOS_VERSION` | 4.5.01 | Kokkos release tag |
| `KOKKOSKERNELS_VERSION` | 4.5.01 | KokkosKernels release tag |
| `BUILD_JOBS` | 8 | Parallel make/cmake jobs |

## Standard Run Command

Mount the MPAS-Model repo at `/work` and run commands inside the container:

```bash
docker run --rm --platform linux/amd64 \
    -v /Users/barry/Documents/MPAS-Model:/work \
    -w /work \
    mpas-dycore-build:latest \
    <command>
```

For commands needing network access (e.g., FetchContent downloads):

```bash
docker run --rm --platform linux/amd64 --network=host \
    -v /Users/barry/Documents/MPAS-Model:/work \
    -w /work \
    mpas-dycore-build:latest \
    <command>
```

### Examples

**Configure the C++ dycore (standalone):**

```bash
docker run --rm --platform linux/amd64 --network=host \
    -v /Users/barry/Documents/MPAS-Model:/work \
    -w /work \
    mpas-dycore-build:latest \
    bash -c "mkdir -p build-cpp && cmake -S src/core_atmosphere/dynamics/cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8"
```

**Configure full MPAS-Atmosphere (CMake, no physics):**

```bash
docker run --rm --platform linux/amd64 --network=host \
    -v /Users/barry/Documents/MPAS-Model:/work \
    -w /work \
    mpas-dycore-build:latest \
    bash -c "cmake -S . -B build -DMPAS_CORES=atmosphere -DDO_PHYSICS=OFF -DMPAS_USE_PIO=OFF && cmake --build build -j8"
```

**Interactive shell:**

```bash
docker run --rm -it --platform linux/amd64 --network=host \
    -v /Users/barry/Documents/MPAS-Model:/work \
    -w /work \
    mpas-dycore-build:latest \
    bash
```
