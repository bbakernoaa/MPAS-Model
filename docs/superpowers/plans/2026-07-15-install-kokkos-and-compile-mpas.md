# Local Kokkos and KokkosKernels Build for MPAS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch, build, and install Kokkos 4.3.01 and KokkosKernels 4.3.01 locally, and compile the MPAS-Model with `MPAS_DYCORE_CPP=ON` under the `ufs_gaeac6.intelllvm` (Intel oneAPI LLVM) compiler environment on Gaea C6.

**Architecture:** Download Kokkos and KokkosKernels as Git repositories inside the `build/external` directory. Compile Kokkos first with serial and OpenMP backends enabled, install to a local subdirectory, then compile KokkosKernels using the local Kokkos installation as its root. Finally, build the MPAS project by pointing CMake to the local installation prefix.

**Tech Stack:** C++, Fortran, CMake, Git, Kokkos (v4.3.01), KokkosKernels (v4.3.01), `ufs_gaeac6.intelllvm` (Intel 2023.2.0 compilers `icpx`/`icx`/`ifx` and Cray compiler wrappers).

## Global Constraints
- Target compilers: `CC` (wraps `icpx`), `cc` (wraps `icx`), `ftn` (wraps `ifx`) provided by `ufs_gaeac6.intelllvm`
- Standard: C++20 (`-DCMAKE_CXX_STANDARD=20`)
- Built prefix: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install`
- Build type: `Release`

---

### Task 1: Initialize Local Source Tree and Clone Dependencies

**Files:**
- Create: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/.gitkeep` (or structure only)

**Interfaces:**
- Produces: Downloadeed Kokkos and Kokkos-kernels source repositories in `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src/kokkos` and `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src/kokkos-kernels`.

- [ ] **Step 1: Create external directory structure**

Run: `mkdir -p /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src`

- [ ] **Step 2: Clone Kokkos v4.3.01**

Run: `git clone --depth 1 --branch 4.3.01 https://github.com/kokkos/kokkos.git /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src/kokkos`
Expected output: Clone completes successfully.

- [ ] **Step 3: Clone KokkosKernels v4.3.01**

Run: `git clone --depth 1 --branch 4.3.01 https://github.com/kokkos/kokkos-kernels.git /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src/kokkos-kernels`
Expected output: Clone completes successfully.

---

### Task 2: Build and Install Kokkos Locally

**Files:**
- Create: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/build-kokkos/` (Temporary build directory)
- Install: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install/`

**Interfaces:**
- Consumes: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src/kokkos`
- Produces: Installed Kokkos headers and libraries in `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install`.

- [ ] **Step 1: Configure Kokkos with CMake**

Run:
```bash
module purge && module load ufs_gaeac6.intelllvm && \
cmake -S /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src/kokkos \
      -B /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/build-kokkos \
      -DCMAKE_INSTALL_PREFIX=/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_OPENMP=ON \
      -DCMAKE_CXX_COMPILER=CC \
      -DCMAKE_C_COMPILER=cc \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_BUILD_TYPE=Release
```
Expected output: CMake configures successfully.

- [ ] **Step 2: Compile Kokkos**

Run: `cmake --build /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/build-kokkos -j 8`
Expected output: Building of `kokkoscore` and companion targets completes without errors.

- [ ] **Step 3: Install Kokkos**

Run: `cmake --install /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/build-kokkos`
Expected output: Libraries, CMake targets, and headers are placed under `build/external/install/`.

- [ ] **Step 4: Verify Installation**

Run: `ls -la /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install/lib64/cmake/Kokkos`
Expected output: Directory exists and contains `KokkosConfig.cmake`.

---

### Task 3: Build and Install KokkosKernels Locally

**Files:**
- Create: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/build-kokkos-kernels/` (Temporary build directory)
- Install: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install/`

**Interfaces:**
- Consumes: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src/kokkos-kernels` and `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install`
- Produces: Installed KokkosKernels headers and libraries in `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install`.

- [ ] **Step 1: Configure KokkosKernels with CMake**

Run:
```bash
module purge && module load ufs_gaeac6.intelllvm && \
cmake -S /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/src/kokkos-kernels \
      -B /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/build-kokkos-kernels \
      -DCMAKE_INSTALL_PREFIX=/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install \
      -DKokkos_ROOT=/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install \
      -DKokkosKernels_ENABLE_COMPONENT_BLAS=ON \
      -DKokkosKernels_ENABLE_COMPONENT_SPARSE=ON \
      -DKokkosKernels_ENABLE_COMPONENT_GRAPH=ON \
      -DKokkosKernels_ENABLE_COMPONENT_BATCHED=ON \
      -DKokkosKernels_INST_DOUBLE=ON \
      -DCMAKE_CXX_COMPILER=CC \
      -DCMAKE_C_COMPILER=cc \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_BUILD_TYPE=Release
```
Expected output: CMake configures successfully finding the local Kokkos installation.

- [ ] **Step 2: Compile KokkosKernels**

Run: `cmake --build /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/build-kokkos-kernels -j 8`
Expected output: Compilation completes without errors.

- [ ] **Step 3: Install KokkosKernels**

Run: `cmake --install /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/build-kokkos-kernels`
Expected output: KokkosKernels is installed to the same `build/external/install/` prefix.

- [ ] **Step 4: Verify Installation**

Run: `ls -la /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install/lib64/cmake/KokkosKernels`
Expected output: Directory exists and contains `KokkosKernelsConfig.cmake`.

---

### Task 4: Configure and Compile MPAS-Model

**Files:**
- Modify: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/`

**Interfaces:**
- Consumes: `/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install`
- Produces: Full MPAS model compiled with `MPAS_DYCORE_CPP=ON`.

- [ ] **Step 1: Clean build directory of any stale configuration**

Run: `rm -f /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/CMakeCache.txt`

- [ ] **Step 2: Configure MPAS-Model with CMake**

Run:
```bash
module purge && module load ufs_gaeac6.intelllvm && \
cd /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build && \
cmake -DMPAS_DYCORE_CPP=ON \
      -DCMAKE_PREFIX_PATH=/gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build/external/install \
      ..
```
Expected output: CMake configures MPAS atmospheric cores and compiles the C++ dycore module without missing dependency errors.

- [ ] **Step 3: Compile MPAS-Model**

Run: `cd /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/build && make -j 8`
Expected output: Compiles the full executable `mpas_atmosphere` incorporating `libmpas_dycore_cpp.a` with no linker errors.
