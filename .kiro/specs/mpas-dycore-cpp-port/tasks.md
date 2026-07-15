# Implementation Plan: MPAS Dycore C++ Port

## Overview

This plan ports the MPAS-Atmosphere nonhydrostatic dynamical core from Fortran to
performance-portable C++ using Kokkos and KokkosKernels, driven from the existing Fortran
atmosphere core through a thin C ABI. Work is ordered so the build/test foundation and the
C/C++ interop boundary come first (every component depends on them), then each numerical
component is implemented incrementally with its own parity unit tests, and finally everything
is wired together in the SRK3 `Time_Integrator` and validated by an end-to-end single-timestep
parity regression.

All tasks build and run inside the `helm-project-helm-dev` Dev_Container using its pinned
Kokkos, KokkosKernels, and Halo_Library. Numeric tests compare against Fortran Reference_Model
output within the unified Parity_Tolerance. Property-based tests use RapidCheck (>=100
iterations, shrinking). The implementation language is C++ (as specified by the design).

Sub-tasks marked with `*` are optional test tasks and are visually distinguished; core
implementation sub-tasks are never optional. Each property test is tagged in the form
`Feature: mpas-dycore-cpp-port, Property {n}: {property_text}`.

when building and testing use the mpas-dycore-build docker image

## Tasks

- [x] 1. Establish Dev_Container build and test foundation
  - [x] 1.1 Wire the C++ dycore into the MPAS-Atmosphere build
    - Add a C++ dycore build target under `src/core_atmosphere/dynamics` integrated with the
      existing MPAS-Atmosphere build system (Makefile/CMake), compiling inside the
      `helm-project-helm-dev` Dev_Container
    - Declare Kokkos, KokkosKernels, and the Halo_Library (`helm-project/libs/halo`) as
      dependencies resolved from the Dev_Container toolchain
    - Bind the C++ `Scalar` alias to RKIND (single/double) at build time
    - _Requirements: 13.3, 13.6, 1.6_

  - [x] 1.2 Wire the C++ test harness into the build
    - Integrate a C++ test runner and RapidCheck (property-based library with shrinking) into
      the MPAS-Atmosphere build so the Component_Test_Suite is runnable on host/CPU, on
      device/GPU where available, and inside the Dev_Container
    - Add a default property-test iteration count (>=100) configurable per run
    - _Requirements: 14.5, 14.7, 14.8, 14.11_

- [x] 2. Implement core data structures and the C/C++ interop boundary
  - [x] 2.1 Implement the Config snapshot
    - Define an immutable `Config` struct capturing namelist options (time_integration_order,
      number_of_sub_steps, dynamics_split_steps, config_monotonic, config_scalar_advection,
      config_apply_lbcs, config_mix_full, LES model/surface strings, config_iau,
      GPU-aware-comm flag, halo exchange method) sourced from existing MPAS data structures
    - _Requirements: 13.2, 13.4_

  - [x] 2.2 Implement MeshData connectivity/geometry views
    - Define a read-only `MeshData` struct wrapping MPAS pool arrays (cellsOnEdge, edgesOnCell,
      verticesOnEdge, nEdgesOnCell, dvEdge, dcEdge, areaCell, zgrid, zz, fzm, fzp) as
      DualViews via `Field_Store::wrap()`, preserving Reference_Model shapes and dimension
      order (LayoutLeft). Mesh geometry is synced to device once during `dycore_init` and
      treated as static thereafter.
    - _Requirements: 13.2, 13.4, 1.5, 1.8, 1.9_

  - [x] 2.3 Implement Field_Store
    - Implement `Field_Store` templated on `Scalar`/`ExecSpace` with:
      - `allocate(name, n_inner, n_elem)` → returns a C++-owned View in the active
        Execution_Space's memory space (device for GPU, host for CPU-only), RAII-freed on
        destruction; used for dycore-internal fields (tendencies, acoustic temporaries, scratch
        buffers, dissipation working arrays)
      - `wrap(fortran_ptr, n_inner, n_elem)` → creates a DualView whose host mirror IS the
        Fortran pointer (zero-copy on host) and whose device side is a Kokkos-allocated copy;
        used for prognostic state fields and mesh geometry
      - `sync_to_device(field)` → deep-copies the host mirror to the device copy of a
        DualView-wrapped field
      - `sync_to_host(field)` → deep-copies the device copy back to the host mirror (the
        Fortran pointer) of a DualView-wrapped field
      - `level(name, time_level)` → time-level access working uniformly for both owned and
        DualView-wrapped fields
      - `extents(name)` → dimension/extent queries
    - All Views/DualViews enforce LayoutLeft and use RKIND precision
    - Expose accessors usable inside a Kokkos parallel region on the active Execution_Space
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13_

  - [x] 2.4 Write unit tests for Field_Store
    - Test owned allocation/free, DualView wrapping (host mirror aliases Fortran pointer),
      sync_to_device/sync_to_host round-trip, time-level access for both owned and wrapped
      fields, extent preservation, and RKIND selection
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.9, 1.10, 1.11, 1.12, 1.13_

  - [x] 2.5 Write property test for Field_Store extents and layout
    - **Property 1: Field_Store preserves extents and column-major layout**
    - **Validates: Requirements 1.5, 1.8**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 1: Field_Store preserves extents and column-major layout`

  - [x] 2.6 Write property test for DualView zero-copy host aliasing and device sync round-trip
    - **Property 2: DualView zero-copy host aliasing and device sync round-trip**
    - **Validates: Requirements 1.9, 1.4, 1.11, 1.12**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 2: DualView zero-copy host aliasing and device sync round-trip`

  - [x] 2.7 Implement the dycore C API and Kokkos lifecycle
    - Implement `dycore_c_api` init/timestep/finalize with matching Fortran `iso_c_binding`
      shims callable from the atmosphere core driver, passing mesh/state raw pointers and dims
    - Call `Kokkos::initialize` in init before any kernel and `Kokkos::finalize` in finalize
    - In init: wrap incoming Fortran state/geometry pointers via `Field_Store::wrap()` (creating
      DualViews) and call `sync_to_device` on mesh geometry (static, synced once)
    - In timestep entry: call `sync_to_device` on all DualView-wrapped prognostic state fields
      before executing any dycore kernel
    - In timestep exit: call `sync_to_host` on all modified prognostic state fields before
      returning to Fortran, so physics/I/O see valid host data
    - Internal dycore temporaries are allocated via `Field_Store::allocate()` (never Fortran-visible)
    - _Requirements: 13.1, 13.2, 13.5, 13.7, 13.8, 13.9, 1.9, 1.10, 1.11, 1.12_

  - [x] 2.8 Write unit tests for the interop boundary
    - Test init/timestep/finalize round-trip, Kokkos lifecycle ordering, DualView wrapping of
      state pointers, sync_to_device at timestep entry, sync_to_host at timestep exit, and that
      Fortran host pointers reflect dycore-modified state after return
    - _Requirements: 13.1, 13.5, 13.7, 13.8, 13.9_

- [x] 3. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement the unstructured accumulation strategy
  - [x] 4.1 Implement Graph_Coloring / thread-team accumulation utility
    - Implement a reusable edge-to-cell (and general scatter/gather) accumulation utility using
      KokkosKernels `KokkosGraph::graph_color` to color elements so same-color elements never
      share a write target, iterating color sets sequentially and parallelizing within a color;
      provide a thread-team local-duplication fallback where no natural coloring exists
    - _Requirements: 2.6_

  - [x] 4.2 Write unit tests for the accumulation utility
    - Test that colored accumulation matches a serial reference sum for representative meshes
    - _Requirements: 2.6_

  - [x] 4.3 Write property test for deterministic unstructured accumulation
    - **Property 8: Deterministic unstructured accumulation**
    - **Validates: Requirements 2.6**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 8: Deterministic unstructured accumulation`

- [x] 5. Implement the Halo_Manager
  - [x] 5.1 Implement the Halo_Group registry
    - Build the table mapping each named group to its ordered `(field, time_level, {halo_layers})`
      tuples exactly as defined by the Reference_Model, including all dynamics/initialization
      groups in Requirement 11.8 and the physics groups `physics:blten`/`physics:cuten` when the
      physics build option is enabled
    - _Requirements: 11.2, 11.8, 11.9_

  - [x] 5.2 Implement the Halo_Manager over the Halo_Library
    - Construct all groups from the registry (RAII), implement `exchange(group_name)` via the
      Halo_Library updating every field's halo region, support GPU-aware exchange of
      device-resident data with no host copy and host-staged transfer otherwise, release
      buffers/group resources in the destructor, and implement `validate_method` reporting a
      configuration error naming an unrecognized exchange method
    - _Requirements: 11.1, 11.3, 11.4, 11.5, 11.6, 11.7, 11.10_

  - [x] 5.3 Write unit tests for halo group composition and method error
    - Verify group field/time-level/halo-layer composition matches the Reference_Model and that
      an invalid method is reported with the offending method name
    - _Requirements: 11.2, 11.8, 11.9, 11.10_

  - [x] 5.4 Write integration tests for halo exchange
    - On a small multi-rank partition, assert halo cells receive owner values and exercise both
      the GPU-aware and host-staged transfer paths
    - _Requirements: 11.3, 11.4, 11.5, 11.6_

- [x] 6. Implement the Diagnostics_Module
  - [x] 6.1 Implement solve diagnostics
    - Compute edge density, tangential velocity, relative vorticity, divergence, kinetic energy,
      and edge potential vorticity; apply APVM upstream bias for a positive coefficient and the
      Hollingsworth KE adjustment when enabled; use the accumulation utility for edge-to-cell
      gathers
    - _Requirements: 7.1, 7.2, 7.3, 2.1, 2.6_

  - [x] 6.2 Implement coupled diagnostics
    - Derive coupled potential temperature, density, mass fluxes, Exner function, and pressure
      from the prognostic state as defined by the Reference_Model
    - _Requirements: 7.4_

  - [x] 6.3 Write unit tests for the Diagnostics_Module
    - Compare solve and coupled diagnostics against Reference_Model output within
      Parity_Tolerance, failing and identifying any diverging field
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 14.2, 14.9_

- [x] 7. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement the Dissipation_Module
  - [x] 8.1 Implement eddy viscosity, Brunt-Väisälä frequency, and stability bounding
    - Implement 2-D Smagorinsky and fixed horizontal eddy viscosity, 3-D Smagorinsky and
      prognostic 1.5-order TKE LES models (with the TKE tendency), the moist Brunt-Väisälä
      frequency selecting dry/moist formulation by the cloud-water threshold, and stability
      bounding of every computed eddy viscosity
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 8.2 Implement fourth-order horizontal hyperdiffusion
    - Apply the fourth-order filter to horizontal momentum, vertical velocity, potential
      temperature, and, where scalar mixing is enabled, scalars
    - _Requirements: 8.7_

  - [x] 8.3 Implement vertical mixing and LES surface boundary condition
    - Apply the second-order vertical filter in physical height space on the full or perturbation
      state per the configured mixing option, preserving the sequential vertical dependency, and
      apply the configured LES surface flux lower boundary condition
    - _Requirements: 8.8, 8.9, 2.2_

  - [x] 8.4 Implement LES option-string mapping
    - Implement `les_model_from_string` and `les_surface_from_string`, returning the
      Reference_Model invalid-option value for unrecognized strings
    - _Requirements: 8.10, 8.11_

  - [x] 8.5 Write unit tests for the Dissipation_Module
    - Compare eddy-viscosity, hyperdiffusion, and vertical-mixing numerics against the
      Reference_Model within Parity_Tolerance and cover both option-string mappings including
      invalid inputs
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.7, 8.8, 8.9, 8.10, 8.11, 14.2, 14.9_

  - [x] 8.6 Write property test for the eddy-viscosity stability bound
    - **Property 9: Eddy-viscosity stability bound**
    - **Validates: Requirements 8.6**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 9: Eddy-viscosity stability bound`

- [x] 9. Implement the Dyn_Tend_Module
  - [x] 9.1 Implement coupled dynamic tendencies
    - Compute coupled tendencies for horizontal momentum, vertical momentum, coupled potential
      temperature, and density; compute mass-flux divergence from mesh connectivity/geometry;
      compute the vector-invariant nonlinear Coriolis term and spherical curvature terms when
      enabled; compute and cache horizontal/vertical mixing on stage 1 (via Dissipation_Module)
      for reuse; apply u Rayleigh damping over configured levels; add physics tendencies for u,
      theta_m, and rho; use the accumulation utility for edge-to-cell operations
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 2.1, 2.6_

  - [x] 9.2 Write unit tests for the Dyn_Tend_Module
    - Compare each computed tendency against the Reference_Model within Parity_Tolerance,
      including the stage-1 mixing cache reuse, failing and identifying any diverging field
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 14.2, 14.9_

- [x] 10. Implement the Acoustic_Solver
  - [x] 10.1 Implement the tridiagonal vertical sweep
    - Implement the Thomas-algorithm vertical sweep using the Reference_Model's precomputed
      implicit coefficients with implicit Rayleigh damping in the absorbing layer, parallelizing
      over horizontal elements while keeping the vertical recurrence sequential inside the kernel
    - _Requirements: 4.3, 4.5, 2.2_

  - [x] 10.2 Implement the acoustic substep update
    - Update perturbation horizontal/vertical momentum, density, and coupled potential
      temperature; accumulate time-averaged horizontal and vertical mass fluxes across substeps;
      zero accumulated fields when substep == 1; apply the regional specified-zone update path
      where applicable; apply 3-D divergence damping to horizontal momentum after each substep
    - _Requirements: 4.1, 4.2, 4.4, 4.6, 4.7_

  - [x] 10.3 Write property test for the tridiagonal solve
    - **Property 3: Tridiagonal solve satisfies its system**
    - **Validates: Requirements 2.2, 4.3**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 3: Tridiagonal solve satisfies its system`

  - [x] 10.4 Write unit tests for the Acoustic_Solver
    - Compare substep updates and accumulated fluxes against the Reference_Model within
      Parity_Tolerance, including the substep==1 zeroing and divergence-damping paths
    - _Requirements: 4.1, 4.2, 4.4, 4.5, 4.7, 14.2, 14.9_

- [x] 11. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement the Scalar_Transport
  - [x] 12.1 Implement standard scalar transport
    - Compute third-order upwind-biased horizontal fluxes with the configured coefficient and
      the Reference_Model vertical flux operator, re-integrate density over the transport
      timestep when transport is split, and use the standard (non-limited) scheme on non-final
      RK substeps; use the accumulation utility for edge-to-cell flux gathers
    - _Requirements: 5.1, 5.3, 5.4, 5.5, 2.6_

  - [x] 12.2 Implement the monotonic/positive-definite limiter and regional boundary flux
    - Apply the monotonic flux limiter on the final RK substep when enabled, keeping updated
      scalars within local min/max bounds; set negative water-species mixing ratios to zero;
      apply the regional boundary flux treatment in relaxation/specified zones
    - _Requirements: 5.2, 5.6, 5.7, 5.8_

  - [x] 12.3 Write property test for the monotonic limiter bounds
    - **Property 4: Monotonic limiter keeps scalars within local bounds**
    - **Validates: Requirements 5.6**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 4: Monotonic limiter keeps scalars within local bounds`

  - [x] 12.4 Write property test for water-species non-negativity
    - **Property 5: Positive-definite transport keeps water species non-negative**
    - **Validates: Requirements 5.7**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 5: Positive-definite transport keeps water species non-negative`

  - [x] 12.5 Write unit tests for the Scalar_Transport
    - Compare standard and monotonic transport output against the Reference_Model within
      Parity_Tolerance, including density re-integration and regional boundary-flux paths
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.8, 14.2, 14.9_

- [x] 13. Implement the Boundary_Module
  - [x] 13.1 Implement boundary tendency read/shift and state queries
    - Implement first vs subsequent `update_boundary_tendency` (read latest on/before time into
      TL2; then shift TL2->TL1, read earliest strictly after time, difference over the interval);
      implement `getTendency` and `getState` (extrapolated) for named fields at a future
      delta-time; derive coupled density, edge density, coupled moist theta, and mass flux from
      boundary fields
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 13.2 Implement boundary masks and relaxation-zone adjustments
    - Set up specified-zone masks for cells/edges/vertices and the nearest relaxation cell per
      specified-zone cell; apply Rayleigh relaxation and horizontal-filter adjustments to
      density, coupled potential temperature, momentum, and scalar tendencies in the relaxation
      zone
    - _Requirements: 9.6, 9.7_

  - [x] 13.3 Implement regional configuration validation
    - Implement `validate_regional_config` reporting configuration errors for: regional mode
      active with no boundary cells; regional mode inactive with boundary cells present; regional
      mode active with no valid boundary input interval
    - _Requirements: 9.8, 9.9, 9.10_

  - [x] 13.4 Write unit tests for the Boundary_Module
    - Compare read/shift/extrapolate/derived-value and mask/relaxation output against the
      Reference_Model within Parity_Tolerance and cover the three configuration-error cases
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 14.2, 14.9_

- [x] 14. Implement the IAU_Module
  - [x] 14.1 Implement IAU forcing
    - Apply no forcing when IAU is off; within the IAU window with IAU on, add forcing to
      momentum, density, coupled theta, and moisture tendencies using the constant IAU weight;
      return without modification when the weight is at or below the negligible threshold; convert
      the theta increment to a coupled moist theta tendency
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 14.2 Write unit tests for the IAU_Module
    - Compare gated forcing and theta coupling against the Reference_Model within
      Parity_Tolerance, including the off, in-window, and below-threshold cases
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 14.2, 14.9_

- [x] 15. Implement the Time_Integrator and wire the timestep together
  - [x] 15.1 Implement scheme validation and stage/substep tables
    - Load RK stage weights (order 2 and 3) and per-stage acoustic substep counts from
      Reference_Model tables; reject any scheme other than SRK3 with an error naming the scheme
    - _Requirements: 3.2, 3.3, 3.4, 3.6, 3.7_

  - [x] 15.2 Implement SRK3 orchestration wiring all modules
    - Implement `advance`/`rk_stage`: init coupled diagnostics, per stage compute solve
      diagnostics and dyn tendencies (mixing cached on stage 1), add LBC + IAU forcing, subcycle
      the Acoustic_Solver with halo exchanges between substeps, transport scalars (monotonic on
      the final stage) with halo exchange, subcycle dry dynamics when dynamics-transport splitting
      is enabled, and advance Time_Level 1 to Time_Level 2 updating time metadata on completion
    - _Requirements: 3.1, 3.5, 3.8, 11.3_

  - [x] 15.3 Implement post-timestep diagnostics, NaN guard, and mass conservation
    - Report global min/max for vertical velocity and horizontal momentum consistent with the
      Reference_Model; scan compared prognostic fields for NaN and report a critical error naming
      the offending field; verify total dry air mass conservation over the timestep within
      Parity_Tolerance
    - _Requirements: 12.3, 12.4, 12.5_

  - [x] 15.4 Write unit tests for the Time_Integrator
    - Test stage-weight/acoustic-substep tables, the unsupported-scheme error, global min/max
      diagnostics, and the NaN-guard critical error
    - _Requirements: 3.2, 3.3, 3.4, 3.6, 3.7, 12.4, 12.5, 14.2_

  - [x] 15.5 Write property test for dry-mass conservation
    - **Property 6: Conservation of total dry air mass**
    - **Validates: Requirements 12.3**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 6: Conservation of total dry air mass`

- [x] 16. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Capture reference snapshots and build the parity regression
  - [x] 17.1 Capture per-component Fortran reference snapshots
    - From the standardized MPAS idealized test cases (Jablonowski-Williamson baroclinic wave and
      mountain/gravity-wave) on a coarse mesh (e.g. 480 km/240 km), capture per-component
      Reference_Model input/output snapshots for the parity unit tests, pinned/cached and
      generated inside the Dev_Container for reproducibility
    - _Requirements: 12.6, 14.2, 14.11_

  - [x] 17.2 Write property test for cross-execution-space component parity
    - **Property 7: Component parity with the reference model across execution spaces**
    - **Validates: Requirements 2.4, 12.1, 12.2**
    - Tag: `Feature: mpas-dycore-cpp-port, Property 7: Component parity with the reference model across execution spaces`

  - [x] 17.3 Build the end-to-end single-timestep parity regression
    - Build an automated regression (modeled on NCAR MPAS-Model-CI) with a fixed build
      configuration and pinned idealized test cases that advances an identical initial state by
      one timestep on host/CPU (and device/GPU where available) and diffs prognostic outputs
      against the Reference_Model baselines within Parity_Tolerance, all inside the Dev_Container
    - _Requirements: 12.1, 12.2, 12.6_

- [x] 18. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; core
  implementation sub-tasks are never optional.
- Each task references the specific requirements (granular clauses) it implements for
  traceability; detailed numerics live in the design document.
- The build/test foundation (task 1) and the interop boundary + Field_Store (task 2) come first
  because every component depends on them; the Time_Integrator (task 15) wires all modules
  together so there is no orphaned code.
- Property tests use RapidCheck with >=100 iterations and shrinking, each tagged
  `Feature: mpas-dycore-cpp-port, Property {n}: ...` and wired to the component it validates.
- All build, test, snapshot-capture, and regression work runs inside the `helm-project-helm-dev`
  Dev_Container for reproducibility.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "2.2", "17.1"] },
    { "id": 2, "tasks": ["2.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "2.7", "4.1", "5.1", "8.4"] },
    { "id": 4, "tasks": ["2.8", "4.2", "4.3", "5.2", "6.1", "8.1", "13.1", "14.1"] },
    { "id": 5, "tasks": ["5.3", "5.4", "6.2", "8.2", "10.1", "12.1", "13.2", "14.2"] },
    { "id": 6, "tasks": ["6.3", "8.3", "9.1", "10.2", "12.2", "13.3"] },
    { "id": 7, "tasks": ["8.5", "8.6", "9.2", "10.3", "10.4", "12.3", "12.4", "12.5", "13.4", "15.1"] },
    { "id": 8, "tasks": ["15.2"] },
    { "id": 9, "tasks": ["15.3"] },
    { "id": 10, "tasks": ["15.4", "15.5", "17.2", "17.3"] }
  ]
}
```
