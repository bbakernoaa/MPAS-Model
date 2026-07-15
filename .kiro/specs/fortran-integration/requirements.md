# Requirements Document

## Introduction

This feature integrates the existing C++ MPAS nonhydrostatic dynamical core (dycore) into the
production Fortran MPAS-Atmosphere executable. The C++ dycore is a complete header-only Kokkos
library located at `src/core_atmosphere/dynamics/cpp/` with 300+ passing unit tests. It exposes
a C API (`dycore_init`, `dycore_timestep`, `dycore_finalize`) declared in
`include/mpas_dycore/dycore_c_api.h`.

The integration comprises five work areas:

1. A Fortran `iso_c_binding` interface module that marshals MPAS pool data into the C API
2. C API implementation files that bridge Fortran arrays into Kokkos Views and invoke the
   time integrator
3. A runtime switch (`config_use_cpp_dycore`) in `mpas_atm_time_integration.F` that selects
   between the existing Fortran `atm_srk3` and the C++ dycore at runtime
4. Build system modifications to compile and link the C++ dycore library into the MPAS
   executable
5. A validation test case (Jablonowski-Williamson baroclinic wave) demonstrating parity
   between the Fortran and C++ dycores

The integration MUST preserve the ability to run with the original Fortran dycore unchanged
(opt-in behavior only). When the C++ dycore is enabled, prognostic fields at each output
interval MUST agree with the reference Fortran run within Parity_Tolerance.

## Glossary

- **MPAS_Executable**: The production Fortran MPAS-Atmosphere executable built by the existing
  MPAS Makefile system.
- **Cpp_Dycore**: The C++ dynamical core library (`libmpas_dycore_cpp.a`) built from
  `src/core_atmosphere/dynamics/cpp/`.
- **C_API**: The three C-linkage entry points (`dycore_init`, `dycore_timestep`,
  `dycore_finalize`) declared in `dycore_c_api.h` and implemented in `dycore_c_api.cpp`.
- **Fortran_Interface_Module**: The Fortran module (`mpas_dycore_interface`) using
  `iso_c_binding` that provides Fortran-callable wrappers to the C_API.
- **Marshalling_Layer**: The Fortran code that extracts fields from MPAS pool structures and
  passes them as contiguous arrays to the C_API.
- **Runtime_Switch**: The namelist configuration variable `config_use_cpp_dycore` that
  controls whether the C++ or Fortran dycore is invoked at runtime.
- **Parity_Tolerance**: The maximum acceptable difference between C++ and Fortran dycore
  outputs for any prognostic field, defined as the L-infinity norm of the difference normalized
  by the field magnitude, set to 1.0e-13 (near double-precision machine epsilon).
- **Reference_Run**: A simulation using the unmodified Fortran `atm_srk3` dycore that serves
  as the comparison baseline.
- **JW_Test**: The Jablonowski-Williamson baroclinic wave test case at 240km resolution used
  for validation.
- **Field_Store**: The Kokkos DualView-based storage system in the Cpp_Dycore that wraps
  Fortran-owned arrays as unmanaged views.
- **AdvanceDomain**: The aggregate struct in the Cpp_Dycore that bundles all module state and
  mesh data needed by the SRK3 orchestrator.
- **Kokkos_Backend**: The Kokkos execution space (OpenMP for CPU, CUDA/HIP for GPU) selected
  at compile time.

## Requirements

### Requirement 1: Fortran iso_c_binding Interface Module

**User Story:** As an MPAS developer, I want a Fortran interface module that bridges MPAS pool data structures to the C++ dycore C API, so that the C++ dycore can be called from within the existing Fortran time integration framework.

#### Acceptance Criteria

1. THE Fortran_Interface_Module SHALL declare `iso_c_binding` interface blocks matching the C_API signatures for `dycore_init`, `dycore_timestep`, and `dycore_finalize`
2. WHEN `dycore_init` is called from Fortran, THE Fortran_Interface_Module SHALL pass mesh dimensions (nCells, nEdges, nVertices, nVertLevels, maxEdges, num_scalars) as `c_int` values
3. WHEN `dycore_init` is called from Fortran, THE Fortran_Interface_Module SHALL pass mesh connectivity arrays (cellsOnEdge, edgesOnCell, verticesOnEdge, nEdgesOnCell) as contiguous `c_int` pointers
4. WHEN `dycore_init` is called from Fortran, THE Fortran_Interface_Module SHALL pass mesh geometry arrays (dvEdge, dcEdge, areaCell, zgrid, zz, fzm, fzp) as contiguous `c_double` pointers
5. WHEN `dycore_init` is called from Fortran, THE Fortran_Interface_Module SHALL pass prognostic state arrays (u, w, theta_m, rho_zz, scalars) for both time levels as contiguous `c_double` pointers
6. WHEN `dycore_init` is called from Fortran, THE Fortran_Interface_Module SHALL pass configuration parameters (time_integration_order, number_of_sub_steps, dynamics_split_steps, config_monotonic, config_scalar_advection, config_apply_lbcs, config_mix_full, config_iau, gpu_aware_comm) as `c_int` values

### Requirement 2: Marshalling Layer

**User Story:** As an MPAS developer, I want a marshalling layer that extracts fields from MPAS pool structures and presents them as contiguous C-compatible arrays, so that zero-copy interop is achieved where possible and data is correctly transferred when copies are needed.

#### Acceptance Criteria

1. WHEN extracting fields from MPAS pools, THE Marshalling_Layer SHALL use `mpas_pool_get_array` to obtain pointers to MPAS-managed field storage
2. WHEN a field is already stored contiguously in column-major (Fortran) order, THE Marshalling_Layer SHALL pass the pointer directly without copying
3. IF a field requires index remapping (e.g., 1-based to 0-based connectivity arrays), THEN THE Marshalling_Layer SHALL perform the remapping into a temporary contiguous buffer before passing to the C_API
4. WHEN passing connectivity arrays, THE Marshalling_Layer SHALL convert from Fortran 1-based to C 0-based indexing
5. THE Marshalling_Layer SHALL extract all required fields before calling `dycore_init` and pass them in a single call

### Requirement 3: C API Implementation for Time Integration

**User Story:** As an MPAS developer, I want the C API implementation to correctly invoke the C++ time integrator and return updated prognostic fields to Fortran, so that the C++ dycore produces physically correct results.

#### Acceptance Criteria

1. WHEN `dycore_init` is called, THE C_API implementation SHALL initialize Kokkos if not already initialized
2. WHEN `dycore_init` is called, THE C_API implementation SHALL wrap Fortran-owned arrays as unmanaged Kokkos Views using LayoutLeft (column-major) to match Fortran memory order
3. WHEN `dycore_init` is called, THE C_API implementation SHALL sync mesh geometry to the device execution space exactly once
4. WHEN `dycore_timestep` is called, THE C_API implementation SHALL sync prognostic state arrays from host to device before the time integration
5. WHEN `dycore_timestep` is called, THE C_API implementation SHALL populate the AdvanceDomain struct and invoke `Time_Integrator_Advance::advance()`
6. WHEN `dycore_timestep` completes, THE C_API implementation SHALL sync modified prognostic state arrays from device back to host
7. WHEN `dycore_finalize` is called, THE C_API implementation SHALL destroy all Kokkos Views before calling `Kokkos::finalize()`

### Requirement 4: Runtime Switch

**User Story:** As an MPAS user, I want a namelist option to select between the Fortran and C++ dycores at runtime, so that I can validate the C++ implementation against the reference without recompiling.

#### Acceptance Criteria

1. THE MPAS_Executable SHALL support a namelist variable `config_use_cpp_dycore` of type logical with default value `.false.`
2. WHILE `config_use_cpp_dycore` is `.false.`, THE MPAS_Executable SHALL invoke the existing Fortran `atm_srk3` subroutine with no behavior change
3. WHILE `config_use_cpp_dycore` is `.true.`, THE MPAS_Executable SHALL invoke the C++ dycore via the Fortran_Interface_Module for each dynamics timestep
4. WHEN `config_use_cpp_dycore` is `.true.`, THE MPAS_Executable SHALL call `dycore_init` during dynamics initialization with all required mesh and state data
5. WHEN `config_use_cpp_dycore` is `.true.`, THE MPAS_Executable SHALL call `dycore_timestep` with the current `dt` and `itimestep` for each timestep
6. WHEN model shutdown occurs with `config_use_cpp_dycore` set to `.true.`, THE MPAS_Executable SHALL call `dycore_finalize` to release C++ resources

### Requirement 5: Build System Integration

**User Story:** As a build engineer, I want the MPAS build system to optionally compile and link the C++ dycore library, so that the integrated executable can be built with or without C++ dycore support.

#### Acceptance Criteria

1. WHEN `USE_CPP_DYCORE=true` is specified at build time, THE build system SHALL compile the C++ dycore into a static library (`libmpas_dycore_cpp.a`)
2. WHEN `USE_CPP_DYCORE=true` is specified, THE build system SHALL link `libmpas_dycore_cpp.a` and its dependencies (Kokkos, KokkosKernels, halo library) into the MPAS_Executable
3. WHEN `USE_CPP_DYCORE=true` is specified, THE build system SHALL compile the Fortran_Interface_Module and the Marshalling_Layer
4. WHEN `USE_CPP_DYCORE` is not specified or set to `false`, THE build system SHALL build the MPAS_Executable without any C++ dycore code and without requiring Kokkos
5. WHEN `USE_CPP_DYCORE=true` is specified, THE build system SHALL define the preprocessor macro `MPAS_CPP_DYCORE` so that Fortran source can conditionally compile C++ dycore integration code
6. THE build system SHALL support finding Kokkos via CMake `find_package` or building it from a bundled source

### Requirement 6: Validation Test (Jablonowski-Williamson Baroclinic Wave)

**User Story:** As a model developer, I want an automated validation test that compares C++ and Fortran dycore outputs, so that I can verify numerical parity between the two implementations.

#### Acceptance Criteria

1. THE JW_Test SHALL configure a 240km quasi-uniform mesh with 55 vertical levels and a 720s timestep
2. THE JW_Test SHALL run for 1 simulated day (120 timesteps) using the Fortran dycore to produce the Reference_Run
3. THE JW_Test SHALL run the same configuration with `config_use_cpp_dycore = .true.` to produce the C++ run
4. WHEN comparing outputs, THE JW_Test SHALL compute the L-infinity norm of the difference for each prognostic field (u, w, theta_m, rho_zz, scalars) at each output interval
5. THE JW_Test SHALL report PASS when all prognostic field differences are within Parity_Tolerance (1.0e-13 relative)
6. THE JW_Test SHALL report FAIL with the field name, timestep, and actual difference when any field exceeds Parity_Tolerance
7. THE JW_Test SHALL report wall-clock time per timestep for both the Fortran and C++ dycores

### Requirement 7: Backward Compatibility

**User Story:** As an MPAS user, I want the integrated executable to behave identically to the current executable when the C++ dycore is disabled, so that existing workflows are not disrupted.

#### Acceptance Criteria

1. WHEN `config_use_cpp_dycore` is `.false.` (default), THE MPAS_Executable SHALL produce bit-for-bit identical results compared to the pre-integration codebase
2. WHEN built without `USE_CPP_DYCORE=true`, THE MPAS_Executable SHALL compile and link without any C++ compiler or Kokkos dependency
3. THE integration SHALL NOT modify the calling signature or behavior of the existing `atm_srk3` subroutine
4. THE integration SHALL NOT modify the existing MPAS pool data structures or their memory layout

### Requirement 8: Error Handling

**User Story:** As an MPAS developer, I want clear error reporting when the C++ dycore integration encounters problems, so that issues can be diagnosed quickly.

#### Acceptance Criteria

1. IF `config_use_cpp_dycore` is `.true.` but the executable was built without `USE_CPP_DYCORE=true`, THEN THE MPAS_Executable SHALL print an error message and abort with a non-zero exit code
2. IF `dycore_init` fails (e.g., Kokkos initialization failure), THEN THE C_API implementation SHALL return an error code and THE Fortran caller SHALL log the error and abort gracefully
3. IF array dimensions passed to `dycore_init` are inconsistent (e.g., nVertLevels <= 0), THEN THE C_API implementation SHALL detect the inconsistency and return an error code
