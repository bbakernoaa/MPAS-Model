# Implementation Plan: Fortran-C++ Dycore Integration

## Overview

This plan implements the integration of the C++ MPAS dynamical core into the production Fortran MPAS-Atmosphere executable. The work progresses from the Fortran marshalling layer, through build system modifications, to the runtime switch and validation test. The C++ dycore library and its Fortran interface module (`dycore_fortran_shim.F90`) already exist; this plan focuses on the glue code, build wiring, and end-to-end validation.

**Build note:** Use `--cpus=4` when running Docker builds. The C++ dycore library and its tests are already built in the `mpas-dycore-build` Docker image.

## Tasks

- [x] 1. Create the Fortran marshalling module
  - [x] 1.1 Implement `mpas_dycore_marshalling.F` with pool extraction and index remapping
    - Create `src/core_atmosphere/dynamics/mpas_dycore_marshalling.F`
    - Module provides `mpas_dycore_call_init(domain)`, `mpas_dycore_call_timestep(dt, itimestep)`, and `mpas_dycore_call_finalize()`
    - In `mpas_dycore_call_init`: use `mpas_pool_get_array` to extract all mesh dimensions, connectivity arrays, geometry arrays, and prognostic state arrays from the domain's pool structures
    - Allocate module-level buffers for connectivity arrays (`cellsOnEdge`, `edgesOnCell`, `verticesOnEdge`, `nEdgesOnCell`) and perform 1-based to 0-based index remapping (`buffer(i) = original(i) - 1`)
    - Pass geometry and prognostic arrays directly (zero-copy) since they are already contiguous column-major
    - Extract config scalars from the `nhyd_model` config pool
    - Call `dycore_init(...)` from `mpas_dycore_interface` with all extracted data
    - Check return code and call `mpas_log_write(..., MPAS_LOG_CRIT)` on failure
    - `mpas_dycore_call_timestep` passes `dt` and `itimestep` through to `dycore_timestep`
    - `mpas_dycore_call_finalize` calls `dycore_finalize` and deallocates the connectivity buffers
    - Guard entire module with `#ifdef MPAS_CPP_DYCORE`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 1.2 Write property tests for index remapping (Property 2)
    - **Property 2: Index Remapping Correctness**
    - Create `tests/test_index_remapping.cpp` using GoogleTest
    - Generate random integer arrays of Fortran 1-based indices, apply the remapping logic, verify each output element equals input minus one
    - Tag: `// Feature: fortran-integration, Property 2: Index Remapping Correctness`
    - **Validates: Requirements 2.3, 2.4**

- [x] 2. Implement the runtime switch in the time integration driver
  - [x] 2.1 Add `config_use_cpp_dycore` namelist variable and runtime switch logic
    - Modify `src/core_atmosphere/dynamics/mpas_atm_time_integration.F`
    - Add `config_use_cpp_dycore` as a logical namelist variable in the `nhyd_model` config section (default `.false.`)
    - Add `#ifdef MPAS_CPP_DYCORE` guarded block in `atm_do_timestep` that:
      - On first call with `config_use_cpp_dycore = .true.`, calls `mpas_dycore_call_init(domain)`
      - On each timestep, calls `mpas_dycore_call_timestep(dt, itimestep)`
      - Falls through to existing `atm_srk3` when switch is `.false.`
    - Add `#ifndef MPAS_CPP_DYCORE` guard that aborts with `MPAS_LOG_CRIT` if `config_use_cpp_dycore` is `.true.` but the executable was not built with C++ dycore support
    - Add shutdown hook: call `mpas_dycore_call_finalize()` during model teardown when `config_use_cpp_dycore` is `.true.`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 8.1_

- [x] 3. Checkpoint - Verify marshalling and runtime switch compile
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Extend the C API with error handling
  - [x] 4.1 Add error return code to `dycore_init` and dimension validation
    - Modify `src/core_atmosphere/dynamics/cpp/include/mpas_dycore/dycore_c_api.h`: change `dycore_init` return type from `void` to `int`
    - Modify `src/core_atmosphere/dynamics/cpp/src/dycore_c_api.cpp`:
      - Add dimension validation at entry: return error code 2 if any of `nCells`, `nEdges`, `nVertLevels`, `maxEdges` is <= 0
      - Add null pointer checks for required arrays: return error code 3
      - Return error code 1 if Kokkos initialization fails (wrap in try/catch)
      - Return 0 on success
    - Update `dycore_fortran_shim.F90` interface block to declare `dycore_init` as a function returning `integer(c_int)`
    - Update `mpas_dycore_marshalling.F` to check the return code
    - _Requirements: 8.2, 8.3, 3.1_

  - [x] 4.2 Write property test for dimension validation (Property 8)
    - **Property 8: Dimension Validation Rejects Invalid Inputs**
    - Add test case in `tests/test_interop_boundary.cpp` or a new `tests/test_dimension_validation.cpp`
    - Generate random dimension tuples with at least one value <= 0, verify `dycore_init` returns non-zero error code without initializing Kokkos
    - Tag: `// Feature: fortran-integration, Property 8: Dimension Validation Rejects Invalid Inputs`
    - **Validates: Requirements 8.3**

  - [x] 4.3 Write property test for FFI value passthrough (Property 1)
    - **Property 1: FFI Value Passthrough**
    - Create `tests/test_ffi_passthrough.cpp` using GoogleTest
    - Call `dycore_init` with random valid dimension scalars and config parameters, instrument the C API to capture received values, verify identity
    - Tag: `// Feature: fortran-integration, Property 1: FFI Value Passthrough`
    - **Validates: Requirements 1.2, 1.6**

- [x] 5. Build system integration
  - [x] 5.1 Create `cpp_dycore.mk` Makefile include for the MPAS build system
    - Create `src/core_atmosphere/dynamics/cpp_dycore.mk`
    - When `USE_CPP_DYCORE=true`:
      - Set `CPP_DYCORE_DIR`, `CPP_DYCORE_LIB`, `CPP_DYCORE_INC`
      - Add recipe to build `libmpas_dycore_cpp.a` via CMake sub-build (`cmake -S ... -B ... -DCMAKE_BUILD_TYPE=Release -DMPAS_BUILD_TESTING=OFF`)
      - Override `CPPFLAGS += -DMPAS_CPP_DYCORE`
      - Override `LIBS += $(CPP_DYCORE_LIB) -lstdc++ -lkokkoscore -lkokkoskernels`
      - Override `FCINCLUDES += $(CPP_DYCORE_INC)`
      - Add `mpas_dycore_interface.o mpas_dycore_marshalling.o` to the dynamics object list
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 5.2 Integrate `cpp_dycore.mk` into the MPAS top-level Makefile
    - Modify the MPAS `Makefile` (or the atmosphere core Makefile) to include `cpp_dycore.mk` when `USE_CPP_DYCORE=true`
    - Ensure the C++ library is built before the Fortran link step
    - Verify that builds without `USE_CPP_DYCORE` require no C++ compiler or Kokkos
    - _Requirements: 5.4, 5.6, 7.2_

- [x] 6. Checkpoint - Verify full build with and without C++ dycore
  - Ensure all tests pass, ask the user if questions arise.
  - Verify: `make gnu CORE=atmosphere` (without `USE_CPP_DYCORE`) compiles cleanly
  - Verify: `make gnu CORE=atmosphere USE_CPP_DYCORE=true` compiles and links cleanly

- [x] 7. Create Dockerfile for Fortran integration testing
  - [x] 7.1 Create a Docker build stage that compiles MPAS with `USE_CPP_DYCORE=true`
    - Extend or create a Dockerfile (e.g., `docker/mpas-fortran-integration/Dockerfile`) based on `mpas-dycore-build`
    - The image should: build MPAS with `make gnu CORE=atmosphere USE_CPP_DYCORE=true` using the gfortran + MPI + PIO + netCDF environment
    - Use `--cpus=4` for Docker builds
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 8. Implement the validation test
  - [x] 8.1 Create the JW baroclinic wave comparison script
    - Create `testing/validate_cpp_dycore.sh`
    - Script should:
      - Run 120 timesteps (1 simulated day) at 240km/55 levels with Fortran dycore → `reference.nc`
      - Run same configuration with `config_use_cpp_dycore = .true.` → `cpp_run.nc`
      - For each prognostic field (u, w, theta_m, rho_zz, scalars) at each output interval, compute L-infinity relative norm: `max|cpp - ref| / max|ref|`
      - Report PASS if all values < 1.0e-13
      - Report FAIL with field name, timestep, and actual difference if any exceeds threshold
      - Report wall-clock time per timestep for both dycores
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 8.2 Create the Python/NCO comparison utility for field parity checking
    - Create `testing/compare_dycore_outputs.py`
    - Reads two NetCDF files, computes per-field L-infinity relative differences
    - Returns structured output (JSON or tabular) with field name, timestep, and difference
    - Exits with code 0 on PASS, non-zero on FAIL
    - _Requirements: 6.4, 6.5, 6.6_

  - [x] 8.3 Write property test for L-infinity norm computation (Property 6)
    - **Property 6: L-Infinity Relative Norm Computation**
    - Create `testing/test_linf_norm.py` or add to C++ test suite
    - Generate random array pairs, verify computed L-inf relative norm matches naive implementation
    - Tag: `// Feature: fortran-integration, Property 6: L-Infinity Relative Norm Computation`
    - **Validates: Requirements 6.4**

  - [x] 8.4 Write property test for parity threshold decision (Property 7)
    - **Property 7: Parity Threshold Decision**
    - Test that the comparison utility reports PASS iff all differences < 1.0e-13 and FAIL with correct details otherwise
    - Generate random difference sets around the 1e-13 boundary
    - Tag: `// Feature: fortran-integration, Property 7: Parity Threshold Decision`
    - **Validates: Requirements 6.5, 6.6**

- [x] 9. Wire backward compatibility safeguards
  - [x] 9.1 Add compile-time and runtime guards for backward compatibility
    - Verify existing `atm_srk3` calling signature is unchanged (no modifications to its interface)
    - Ensure `#ifndef MPAS_CPP_DYCORE` guard aborts if `config_use_cpp_dycore = .true.` but binary lacks C++ dycore support
    - Verify that MPAS pool data structures and memory layout are not modified
    - Add a comment block in `mpas_atm_time_integration.F` documenting the backward compatibility contract
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.1_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Run existing C++ dycore CTest suite to verify no regressions
  - Verify the interop boundary test (`tests/test_interop_boundary.cpp`) still passes

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The C++ dycore library and its tests are already built in the `mpas-dycore-build` Docker image — no need to rebuild them from scratch
- The existing `dycore_fortran_shim.F90` provides the `iso_c_binding` interface; the marshalling module is the new Fortran code
- Use `--cpus=4` when running Docker builds
- The Fortran MPAS model needs a different build environment (gfortran + MPI + PIO + netCDF) from the pure C++ dycore build

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "5.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "4.1", "5.2"] },
    { "id": 2, "tasks": ["4.2", "4.3", "9.1"] },
    { "id": 3, "tasks": ["7.1"] },
    { "id": 4, "tasks": ["8.1", "8.2"] },
    { "id": 5, "tasks": ["8.3", "8.4"] }
  ]
}
```
