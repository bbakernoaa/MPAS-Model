# Requirements Document

## Introduction

This feature ports the MPAS-Atmosphere nonhydrostatic dynamical core ("dycore") and its
supporting halo-exchange setup from Fortran to C++. The port targets performance portability
across CPU and GPU execution using Kokkos and KokkosKernels, replacing the existing OpenACC
directive-based acceleration with Kokkos execution and memory spaces and Kokkos Views for
field storage.

The Fortran sources in scope are:

- `src/core_atmosphere/dynamics/mpas_atm_time_integration.F` — the split-explicit RK3 (SRK3)
  time integrator, acoustic substep solver, scalar transport (standard and monotonic),
  dynamic tendency computation, diagnostics, and regional-boundary adjustments.
- `src/core_atmosphere/dynamics/mpas_atm_dissipation_models.F` — 2-D Smagorinsky mixing,
  3-D LES models (3-D Smagorinsky and prognostic 1.5-order TKE), moist Brunt-Väisälä frequency,
  and horizontal/vertical dissipation of u, w, theta, and scalars.
- `src/core_atmosphere/dynamics/mpas_atm_boundaries.F` — limited-area (regional) lateral
  boundary condition (LBC) tendency/state management and boundary masks.
- `src/core_atmosphere/dynamics/mpas_atm_iau.F` — Incremental Analysis Update (IAU) forcing.
- `src/core_atmosphere/mpas_atm_halos.F` — construction and teardown of named halo-exchange
  groups used by initialization, dynamics, and physics.

Halo exchanges are to be ported to C++ so that they are GPU-ready and manage their resources
via RAII, using the halo-exchange library referenced by the user (from the `helm-project`
repository, `libs/halo`) as the underlying implementation.

The port MUST preserve the numerical results of the existing Fortran dynamical core. On both the
host/CPU Execution_Space and the device/GPU Execution_Space, the port MUST reproduce the
Reference_Model results to within a single documented tight tolerance that is as close to machine
precision as achievable given the accumulated arithmetic. The same Parity_Tolerance applies
uniformly across execution spaces, keeping ported and reference simulations scientifically
equivalent without requiring bit-for-bit reproduction on any execution space.

## Glossary

- **Dycore_Port**: The overall C++ implementation being produced by this feature, comprising
  the time integrator, acoustic solver, scalar transport, dissipation, boundary, IAU, and
  halo components ported from the Fortran sources listed in the Introduction.
- **Time_Integrator**: The Dycore_Port component that advances the model state forward by one
  model timestep using the split-explicit third-order Runge-Kutta (SRK3) scheme, equivalent to
  the Fortran `atm_timestep`/`atm_srk3` routines.
- **Acoustic_Solver**: The Dycore_Port component that performs the forward-backward vertically
  implicit acoustic substep update, equivalent to `atm_advance_acoustic_step` and its work routine.
- **Scalar_Transport**: The Dycore_Port component that advects scalar fields, including the
  standard transport (equivalent to `atm_advance_scalars`) and the monotonic/positive-definite
  transport (equivalent to `atm_advance_scalars_mono`).
- **Dyn_Tend_Module**: The Dycore_Port component that computes coupled tendencies for u, w,
  theta_m, and rho, equivalent to `atm_compute_dyn_tend`.
- **Diagnostics_Module**: The Dycore_Port component that computes solve diagnostics (v, vorticity,
  divergence, kinetic energy, potential vorticity) and coupled diagnostics, equivalent to
  `atm_compute_solve_diagnostics` and `atm_init_coupled_diagnostics`.
- **Dissipation_Module**: The Dycore_Port component that computes mixing and dissipation terms,
  equivalent to the routines in `mpas_atm_dissipation_models.F` (Smagorinsky 2-D, LES models,
  Brunt-Väisälä frequency, and u/w/theta/scalar dissipation).
- **Boundary_Module**: The Dycore_Port component that manages limited-area lateral boundary
  conditions, equivalent to the routines in `mpas_atm_boundaries.F`.
- **IAU_Module**: The Dycore_Port component that applies Incremental Analysis Update forcing,
  equivalent to the routines in `mpas_atm_iau.F`.
- **Halo_Manager**: The Dycore_Port component that creates, exchanges, and destroys named
  halo-exchange groups, equivalent to `mpas_atm_halos.F`.
- **Halo_Library**: The external halo-exchange library (referenced by the user from
  `helm-project/libs/halo`) that the Halo_Manager uses to perform GPU-ready halo exchanges.
- **Field_Store**: The C++ storage abstraction for mesh field data, implemented with Kokkos
  Views, that owns device/host memory and supports multiple time levels.
- **Kokkos_View**: A Kokkos::View, the performance-portable multidimensional array used by
  Field_Store to hold field data in a selected memory space.
- **LayoutLeft**: The `Kokkos::LayoutLeft` column-major memory layout, in which the leftmost
  index varies fastest in memory, matching Fortran array storage order.
- **Unmanaged_View**: A Kokkos_View created with `Kokkos::MemoryTraits<Kokkos::Unmanaged>` that
  wraps externally owned memory (for example a raw Fortran pointer) without allocating or
  releasing that memory.
- **DualView**: A Kokkos::DualView that maintains a host mirror and a device copy of a field.
  In the hybrid ownership model, the host mirror IS the Fortran pointer (zero-copy on host)
  and the device side is a Kokkos-allocated copy. Synchronization between host and device is
  explicit via `sync_to_device` and `sync_to_host` operations.
- **Graph_Coloring**: A partitioning of mesh elements into color sets in which elements sharing
  a write target are placed in different colors, provided by the KokkosKernels `KokkosGraph`
  facilities, so that same-color elements can be updated in parallel without memory races.
- **Execution_Space**: A Kokkos execution space (for example host or device) in which
  Dycore_Port computational kernels run.
- **Memory_Space**: A Kokkos memory space in which Field_Store data resides.
- **Halo_Group**: A named collection of fields, each associated with a time level and a set of
  halo layers, whose halo regions are exchanged together (for example
  `dynamics:theta_m,scalars,pressure_p,rtheta_p`).
- **Halo_Layer**: An integer identifying a ring of ghost cells/edges/vertices around the owned
  region of a mesh partition, taking values in {1, 2, 3} in the existing groups.
- **Time_Level**: An integer index selecting one of a field's stored temporal states (for
  example 1 or 2) within Field_Store.
- **Mesh**: The unstructured spherical centroidal Voronoi tessellation on which fields are
  defined, with cell, edge, and vertex elements and their connectivity.
- **Regional_Mode**: The configuration in which the model is run as a limited area, activated by
  the `config_apply_lbcs` namelist option being true.
- **Reference_Model**: The existing Fortran MPAS-Atmosphere dynamical core used as the baseline
  for numerical comparison.
- **RKIND**: The working real kind used by MPAS (single or double precision as selected at build).
- **Parity_Tolerance**: The single, unified numerical-agreement bound required between the
  Dycore_Port and the Reference_Model, applied uniformly on both the host/CPU and device/GPU
  Execution_Space. Parity_Tolerance is a tight relative tolerance on the order of a small multiple
  of machine epsilon, with documented ceilings of 1e-12 for double precision and 1e-6 for single
  precision. The goal is agreement as close to machine precision as the accumulated arithmetic
  allows; the ceilings are upper bounds that a compared field SHALL NOT exceed.
- **Bit_Reproducibility**: The property that two executions produce identical bit patterns for
  all compared output fields. In this feature, Bit_Reproducibility is an optional diagnostic
  concept only and is not a required outcome on any Execution_Space.
- **Unit_Test**: An automated test that exercises a single Dycore_Port component in isolation
  with fixed, developer-specified inputs and checks the component output against an expected
  result, such as the Reference_Model output for the same inputs.
- **Property_Based_Test**: An automated test that verifies a Correctness_Property holds across
  many randomly generated valid inputs, exercising a configurable number of generated cases and,
  on failure, reducing the failing input to a minimal counterexample (shrinking).
- **Correctness_Property**: An invariant or relationship that a Dycore_Port component is required
  to preserve for all valid inputs (for example the monotonic limiter keeping scalars within
  local minimum and maximum bounds, positive-definite transport keeping water-species mixing
  ratios non-negative, or conservation of total dry air mass).
- **Component_Test_Suite**: The collection of Unit_Tests and Property_Based_Tests delivered for
  the Dycore_Port components, runnable within the existing MPAS-Atmosphere build and test setup.
- **Dev_Container**: The containerized build and test environment based on the
  `helm-project-helm-dev` Docker image that provides the pinned toolchain and dependencies
  (compilers, MPI, NetCDF/PIO, Kokkos, KokkosKernels, and the Halo_Library from
  `helm-project/libs/halo`) used to build and test the Dycore_Port reproducibly.

## Requirements

### Requirement 1: Performance-portable field storage with Kokkos

**User Story:** As an HPC model developer, I want dycore field data stored in Kokkos Views with
RAII lifetime management, so that the same source code executes on CPU and GPU without manual
memory bookkeeping.

#### Acceptance Criteria

1. THE Field_Store SHALL store each ported mesh field in a Kokkos_View allocated in a configured Memory_Space.
2. WHERE a field has multiple time levels in the Reference_Model, THE Field_Store SHALL provide access to each Time_Level of that field.
3. WHEN a Field_Store object is constructed, THE Field_Store SHALL allocate the Kokkos_View storage for the field.
4. WHEN a Field_Store object is destroyed, THE Field_Store SHALL release the Kokkos_View storage owned by that object.
5. THE Field_Store SHALL preserve the logical dimension order and extents of each ported field as defined by the Reference_Model (for example vertical-level by horizontal-element ordering).
6. THE Field_Store SHALL represent real-valued fields using the RKIND precision selected at build time.
7. WHERE a field is accessed inside a computational kernel, THE Field_Store SHALL expose the field through an accessor usable within a Kokkos parallel region on the active Execution_Space.
8. THE Field_Store SHALL enforce LayoutLeft (column-major) memory layout for all multidimensional Kokkos_Views so that memory access is contiguous when interoperating with Fortran arrays.
9. WHERE field memory is allocated and owned by the Fortran driver (prognostic state fields and mesh geometry), THE Field_Store SHALL wrap the raw Fortran pointers as Kokkos DualViews whose host mirror IS the Fortran pointer (zero-copy on the host side) and whose device side is a Kokkos-allocated copy in the active device Memory_Space.
10. WHERE a field is internal to the dycore (intermediate tendencies, acoustic substep temporaries, scratch buffers, dissipation working arrays), THE Field_Store SHALL allocate the field as a fully C++-owned Kokkos_View in the active Execution_Space's Memory_Space, managed via RAII, and SHALL NOT expose that field to Fortran.
11. THE Field_Store SHALL provide a `sync_to_device(field)` operation that deep-copies the host mirror of a DualView-wrapped field to its device copy, ensuring the device side reflects the latest Fortran-written host data.
12. THE Field_Store SHALL provide a `sync_to_host(field)` operation that deep-copies the device side of a DualView-wrapped field back to its host mirror (the Fortran pointer), ensuring Fortran physics and I/O see valid updated data.
13. THE Field_Store SHALL support time-level access for both owned Views and DualView-wrapped fields through a uniform interface.

### Requirement 2: Execution-space-portable dycore kernels

**User Story:** As an HPC model developer, I want the dycore compute kernels written against
Kokkos execution spaces, so that a single implementation runs efficiently on host and device.

#### Acceptance Criteria

1. THE Dycore_Port SHALL express each ported computational loop as a Kokkos parallel construct that executes on the active Execution_Space.
2. WHERE a computation in the Reference_Model uses a tridiagonal vertical solve with a sequential vertical dependency, THE Dycore_Port SHALL preserve that sequential dependency within the ported kernel.
3. WHERE the Reference_Model uses linear algebra operations available in KokkosKernels, THE Dycore_Port SHALL use KokkosKernels for those operations.
4. WHEN the same input is executed on the host Execution_Space and on the device Execution_Space, THE Dycore_Port SHALL produce prognostic fields that agree to within the Parity_Tolerance, and THE Dycore_Port SHALL NOT require host-versus-device bit-for-bit equality.
5. THE Dycore_Port SHALL NOT require OpenACC directives to achieve GPU execution.
6. WHERE unstructured flux accumulations or edge-to-cell mappings risk memory race conditions, THE Dycore_Port SHALL use Graph_Coloring via KokkosKernels `KokkosGraph` or thread-team data duplication to avoid atomic operations.

### Requirement 3: SRK3 time integration

**User Story:** As a model user, I want the C++ time integrator to advance the state with the
SRK3 scheme, so that a ported timestep reproduces the dynamical evolution of the Fortran model.

#### Acceptance Criteria

1. WHEN the Time_Integrator is invoked for one model timestep, THE Time_Integrator SHALL advance the model state from Time_Level 1 to Time_Level 2 over the timestep interval.
2. THE Time_Integrator SHALL perform three Runge-Kutta substeps per dynamics step using the substep timestep weights defined by the Reference_Model for the configured time integration order.
3. WHERE the configured time integration order is 3, THE Time_Integrator SHALL use the three-stage weights defined by the Reference_Model.
4. WHERE the configured time integration order is 2, THE Time_Integrator SHALL use the two-stage weights defined by the Reference_Model.
5. WHERE dynamics-transport splitting is enabled, THE Time_Integrator SHALL subcycle the dry dynamics using the configured number of dynamics split steps and a correspondingly reduced dynamics timestep.
6. WHEN the number of acoustic substeps for a Runge-Kutta stage is determined, THE Time_Integrator SHALL use the per-stage acoustic substep counts defined by the Reference_Model.
7. IF the configured time integration scheme is not `SRK3`, THEN THE Time_Integrator SHALL report an error identifying the unsupported scheme.
8. WHEN a timestep completes, THE Time_Integrator SHALL update the model time metadata to the advanced time.

### Requirement 4: Acoustic substep solver

**User Story:** As a model developer, I want the acoustic substep solver ported to C++, so that
the fast-mode integration reproduces the Fortran forward-backward vertically implicit update.

#### Acceptance Criteria

1. WHEN the Acoustic_Solver executes an acoustic substep, THE Acoustic_Solver SHALL update the perturbation horizontal momentum, perturbation vertical momentum, perturbation density, and perturbation coupled potential temperature.
2. THE Acoustic_Solver SHALL accumulate the time-averaged horizontal mass flux and the time-averaged vertical mass flux across the acoustic substeps of a Runge-Kutta stage.
3. WHILE solving the vertically implicit system, THE Acoustic_Solver SHALL perform the tridiagonal sweep using the precomputed vertical implicit coefficients defined by the Reference_Model.
4. WHEN the acoustic substep index is 1, THE Acoustic_Solver SHALL initialize the accumulated flux and perturbation fields to zero.
5. THE Acoustic_Solver SHALL apply the implicit Rayleigh damping of vertical velocity in the gravity-wave absorbing layer defined by the Reference_Model.
6. WHERE a cell is inside the regional specified zone, THE Acoustic_Solver SHALL apply the specified-zone update path defined by the Reference_Model instead of the interior solve.
7. THE Acoustic_Solver SHALL apply the 3-D divergence damping to the horizontal momentum after each acoustic substep.

### Requirement 5: Scalar transport

**User Story:** As a model user, I want the scalar advection ported to C++, so that moisture and
other scalars are transported with the same schemes and constraints as the Fortran model.

#### Acceptance Criteria

1. WHEN Scalar_Transport advances scalars on a non-final Runge-Kutta substep, THE Scalar_Transport SHALL use the standard (non-limited) transport scheme.
2. WHERE monotonic or positive-definite transport is enabled and the Runge-Kutta substep is the final substep, THE Scalar_Transport SHALL apply the monotonic flux limiter defined by the Reference_Model.
3. THE Scalar_Transport SHALL compute horizontal fluxes using the third-order upwind-biased scheme of the Reference_Model with the configured third-order coefficient.
4. THE Scalar_Transport SHALL compute vertical fluxes using the flux operator defined by the Reference_Model.
5. WHERE scalar transport is split from the dry dynamics, THE Scalar_Transport SHALL re-integrate the density over the transport timestep before updating the scalars.
6. WHEN the monotonic limiter is applied, THE Scalar_Transport SHALL keep each updated scalar within the local minimum and maximum bounds derived from the surrounding cells, to within the Parity_Tolerance on both the host/CPU and device/GPU Execution_Space.
7. WHEN a scalar update completes, THE Scalar_Transport SHALL set any resulting negative mixing ratio to zero for water-species scalars.
8. WHERE Regional_Mode is active, THE Scalar_Transport SHALL apply the boundary flux treatment defined by the Reference_Model for edges in the relaxation and specified zones.

### Requirement 6: Coupled dynamic tendencies

**User Story:** As a model developer, I want the coupled tendency computation ported to C++, so
that momentum, vertical velocity, potential temperature, and density tendencies match the
Fortran model.

#### Acceptance Criteria

1. WHEN Dyn_Tend_Module is invoked, THE Dyn_Tend_Module SHALL compute the coupled tendencies for horizontal momentum, vertical momentum, coupled potential temperature, and density.
2. THE Dyn_Tend_Module SHALL compute the horizontal mass-flux divergence used in the density tendency using the mesh connectivity and geometry of the Reference_Model.
3. THE Dyn_Tend_Module SHALL compute the nonlinear Coriolis term following the vector-invariant formulation used by the Reference_Model.
4. WHERE the curvature terms are enabled in the Reference_Model, THE Dyn_Tend_Module SHALL include the spherical curvature terms in the momentum and vertical-velocity tendencies.
5. WHEN the Runge-Kutta substep is the first substep, THE Dyn_Tend_Module SHALL compute the horizontal and vertical mixing tendencies and retain them for reuse in later substeps.
6. WHERE the u Rayleigh damping option is enabled, THE Dyn_Tend_Module SHALL apply the Rayleigh damping of horizontal momentum over the configured number of damping levels.
7. THE Dyn_Tend_Module SHALL add the physics tendencies for horizontal momentum, coupled potential temperature, and density to the corresponding dynamic tendencies.

### Requirement 7: Solve and coupled diagnostics

**User Story:** As a model developer, I want the diagnostic computations ported to C++, so that
derived fields used by the tendencies match the Fortran model.

#### Acceptance Criteria

1. WHEN Diagnostics_Module computes solve diagnostics, THE Diagnostics_Module SHALL compute density at edges, tangential velocity, relative vorticity, divergence, kinetic energy, and potential vorticity at edges.
2. WHERE APVM upwinding is configured with a positive coefficient, THE Diagnostics_Module SHALL apply the anticipated-potential-vorticity-method upstream bias to the edge potential vorticity.
3. WHERE the Hollingsworth kinetic-energy construction is enabled in the Reference_Model, THE Diagnostics_Module SHALL apply that kinetic-energy adjustment.
4. WHEN Diagnostics_Module initializes coupled diagnostics, THE Diagnostics_Module SHALL derive the coupled potential temperature, density, mass fluxes, Exner function, and pressure from the prognostic state as defined by the Reference_Model.

### Requirement 8: Dissipation and mixing models

**User Story:** As a model user, I want the dissipation and LES models ported to C++, so that
sub-grid mixing reproduces the Fortran behavior across the supported options.

#### Acceptance Criteria

1. WHERE horizontal mixing is configured as 2-D Smagorinsky, THE Dissipation_Module SHALL compute the horizontal eddy viscosity using the Smagorinsky deformation formulation of the Reference_Model.
2. WHERE horizontal mixing is configured as 2-D fixed, THE Dissipation_Module SHALL set the horizontal eddy viscosity to the configured fixed second-order value.
3. WHERE the LES model is configured as 3-D Smagorinsky, THE Dissipation_Module SHALL compute horizontal and vertical eddy viscosities from the three-dimensional deformation and the moist Brunt-Väisälä frequency.
4. WHERE the LES model is configured as prognostic 1.5-order TKE, THE Dissipation_Module SHALL compute eddy viscosities from the prognostic turbulent kinetic energy and produce the TKE tendency defined by the Reference_Model.
5. THE Dissipation_Module SHALL compute the moist Brunt-Väisälä frequency using the dry formulation where cloud water is below the critical threshold and the moist formulation otherwise.
6. THE Dissipation_Module SHALL bound each computed eddy viscosity by the stability limit defined by the Reference_Model.
7. WHERE fourth-order horizontal hyperdiffusion is active, THE Dissipation_Module SHALL apply the fourth-order filter to horizontal momentum, vertical velocity, potential temperature, and, where scalar mixing is enabled, scalars.
8. WHERE vertical mixing is active, THE Dissipation_Module SHALL apply the second-order vertical filter in physical height space, mixing either the full state or the perturbation from the initial one-dimensional state according to the configured mixing option.
9. WHERE an LES surface option is configured, THE Dissipation_Module SHALL apply the corresponding surface flux lower boundary condition defined by the Reference_Model.
10. THE Dissipation_Module SHALL convert an LES model option string to the corresponding option value, and IF the string is not a recognized LES model option, THEN THE Dissipation_Module SHALL return the invalid-option value defined by the Reference_Model.
11. THE Dissipation_Module SHALL convert an LES surface option string to the corresponding option value, and IF the string is not a recognized LES surface option, THEN THE Dissipation_Module SHALL return the invalid-option value defined by the Reference_Model.

### Requirement 9: Limited-area lateral boundary conditions

**User Story:** As a regional-model user, I want the LBC handling ported to C++, so that
regional simulations reproduce the Fortran boundary treatment.

#### Acceptance Criteria

1. WHEN Boundary_Module updates boundary tendencies for the first time, THE Boundary_Module SHALL read the latest boundary data valid on or before the current time into the second Time_Level of the boundary fields.
2. WHEN Boundary_Module updates boundary tendencies after the first time, THE Boundary_Module SHALL shift the second Time_Level to the first Time_Level, read the earliest boundary data strictly after the current time, and compute the boundary tendencies from the difference over the boundary interval.
3. WHEN Boundary_Module is asked for a boundary tendency for a named field at a future delta-time, THE Boundary_Module SHALL return the tendency array for that field.
4. WHEN Boundary_Module is asked for a boundary state for a named field at a future delta-time, THE Boundary_Module SHALL return the state extrapolated from the stored state and tendency for that field.
5. THE Boundary_Module SHALL derive boundary values of coupled density, edge density, coupled moist potential temperature, and mass flux from the read boundary fields as defined by the Reference_Model.
6. WHEN Boundary_Module sets up boundary masks, THE Boundary_Module SHALL mark cells, edges, and vertices in the specified zone and SHALL identify the nearest relaxation cell for each specified-zone cell.
7. WHILE a cell is in the relaxation zone, THE Boundary_Module SHALL apply the Rayleigh relaxation and horizontal-filter adjustments to the density, coupled potential temperature, momentum, and scalar tendencies defined by the Reference_Model.
8. IF Regional_Mode is active and the boundary-mask field contains no boundary cells, THEN THE Boundary_Module SHALL report a configuration error.
9. IF Regional_Mode is inactive and the boundary-mask field contains boundary cells, THEN THE Boundary_Module SHALL report a configuration error.
10. IF Regional_Mode is active and the boundary input stream has no valid input interval, THEN THE Boundary_Module SHALL report a configuration error.

### Requirement 10: Incremental Analysis Update forcing

**User Story:** As a data-assimilation user, I want the IAU forcing ported to C++, so that
analysis increments are applied identically to the Fortran model.

#### Acceptance Criteria

1. WHEN the IAU option is off, THE IAU_Module SHALL apply no analysis-increment forcing to the tendencies.
2. WHILE the current timestep index is within the IAU time window and the IAU option is on, THE IAU_Module SHALL add the analysis-increment forcing to the momentum, density, coupled potential temperature, and moisture tendencies using the constant IAU weight defined by the Reference_Model.
3. WHEN the computed IAU weight is at or below the negligible threshold defined by the Reference_Model, THE IAU_Module SHALL return without modifying the tendencies.
4. THE IAU_Module SHALL convert the potential-temperature increment tendency to a coupled moist potential-temperature tendency as defined by the Reference_Model.

### Requirement 11: GPU-ready halo exchange with RAII resource management

**User Story:** As an HPC model developer, I want halo exchanges ported to C++ with RAII and
GPU-ready buffers, so that halo communication works on device without directive-based data
management and without resource leaks.

#### Acceptance Criteria

1. WHEN the Halo_Manager is initialized, THE Halo_Manager SHALL create every Halo_Group used by initialization, dynamics, and physics defined by the Reference_Model.
2. THE Halo_Manager SHALL add each field to its Halo_Group with the Time_Level and the set of Halo_Layer values defined by the Reference_Model for that field.
3. WHEN the Halo_Manager exchanges a named Halo_Group, THE Halo_Manager SHALL update the halo regions of every field in that group.
4. THE Halo_Manager SHALL perform halo exchanges using the Halo_Library.
5. WHERE GPU-aware communication is enabled, THE Halo_Manager SHALL exchange field data resident in device Memory_Space without an intervening host copy.
6. WHERE GPU-aware communication is disabled, THE Halo_Manager SHALL transfer the affected field data between device Memory_Space and host Memory_Space around the exchange.
7. WHEN a Halo_Manager resource object is destroyed, THE Halo_Manager SHALL release the communication buffers and group resources it owns.
8. THE Halo_Manager SHALL preserve the field membership, Time_Level, and Halo_Layer composition of each named Halo_Group listed in the Reference_Model, including the groups `dynamics:theta_m,scalars,pressure_p,rtheta_p`, `dynamics:rw_p,ru_p,rho_pp,rtheta_pp`, `dynamics:w,pv_edge,rho_edge`, `dynamics:w,pv_edge,rho_edge,scalars`, `dynamics:theta_m,pressure_p,rtheta_p`, `dynamics:exner`, `dynamics:tend_u`, `dynamics:rho_pp`, `dynamics:rtheta_pp`, `dynamics:u_123`, `dynamics:u_3`, `dynamics:scalars`, `dynamics:scalars_old`, `dynamics:w`, `dynamics:scale`, `initialization:u`, and `initialization:pv_edge,ru,rw`.
9. WHERE the physics build option is enabled, THE Halo_Manager SHALL create the physics Halo_Groups `physics:blten` and `physics:cuten` with the field composition defined by the Reference_Model.
10. IF the configured halo exchange method is neither the direct method nor the grouped method recognized by the Reference_Model, THEN THE Halo_Manager SHALL report a configuration error identifying the invalid method.

### Requirement 12: Numerical equivalence with the reference model

**User Story:** As a model scientist, I want the C++ dycore to reproduce the Fortran results to
within a single tight tolerance near machine precision on both CPU and GPU, so that I can trust
the port for scientific use.

#### Acceptance Criteria

1. WHERE Dycore_Port runs on the host/CPU Execution_Space, WHEN Dycore_Port advances an identical initial state by one timestep, THE Dycore_Port SHALL produce prognostic fields that match the Reference_Model output to within the Parity_Tolerance.
2. WHERE Dycore_Port runs on the device/GPU Execution_Space, WHEN Dycore_Port advances an identical initial state by one timestep, THE Dycore_Port SHALL produce prognostic fields that match the Reference_Model output to within the Parity_Tolerance.
3. THE Dycore_Port SHALL conserve the total dry air mass over a timestep to within the Parity_Tolerance on both the host/CPU and device/GPU Execution_Space, matching the conservation behavior of the Reference_Model.
4. WHEN Dycore_Port completes a timestep, THE Dycore_Port SHALL report a global minimum and maximum for the vertical velocity and horizontal momentum fields consistent with the Reference_Model diagnostic output, to within the Parity_Tolerance on both the host/CPU and device/GPU Execution_Space.
5. IF any compared prognostic field contains a value that is not a number, THEN THE Dycore_Port SHALL report a critical error identifying the field.
6. THE Dycore_Port SHALL perform numerical parity comparisons against the Reference_Model within the Dev_Container so that the compared results are reproducible.

### Requirement 13: Interoperability with the existing MPAS build

**User Story:** As a model integrator, I want the C++ dycore to interoperate with the existing
MPAS-Atmosphere infrastructure, so that the port can be built and driven within the current model.

#### Acceptance Criteria

1. THE Dycore_Port SHALL be invocable from the existing atmosphere core driver for the timestep-advance, dynamics-initialize, and dynamics-finalize entry points defined by the Reference_Model.
2. THE Dycore_Port SHALL obtain mesh geometry, connectivity, dimensions, and configuration values from the existing MPAS data structures.
3. THE Dycore_Port SHALL build within the existing MPAS-Atmosphere build system with Kokkos and KokkosKernels as dependencies.
4. WHERE the model is configured with a per-build fixed inner dimension for vertical levels, maximum edges, or number of scalars, THE Dycore_Port SHALL use those fixed dimensions consistently with the Reference_Model.
5. THE Dycore_Port SHALL initialize the Kokkos runtime before executing any Kokkos kernel and SHALL finalize the Kokkos runtime during model shutdown.
6. THE Dycore_Port SHALL build within the Dev_Container using the toolchain and dependencies, including Kokkos, KokkosKernels, and the Halo_Library, that the Dev_Container provides.
7. WHEN the C API timestep entry point is invoked, THE Dycore_Port SHALL call `sync_to_device` on all DualView-wrapped prognostic state fields before executing any dycore kernel, ensuring device-side data reflects the latest Fortran host state.
8. WHEN the C API timestep entry point returns, THE Dycore_Port SHALL call `sync_to_host` on all DualView-wrapped prognostic state fields that were modified by the dycore, ensuring Fortran physics and I/O continue to operate on valid host-side data unchanged from their perspective.
9. THE Dycore_Port interop boundary SHALL NOT require changes to the Fortran physics or I/O subsystems; those subsystems SHALL continue to access prognostic state through their existing host-side Fortran pool pointers.

### Requirement 14: Component-level verification with unit and property-based tests

**User Story:** As a model developer, I want unit and property-based tests developed alongside
each ported component, so that individual schemes and code are verified to reproduce the
Reference_Model at the component level as the port progresses.

#### Acceptance Criteria

1. WHEN a Dycore_Port component (Time_Integrator, Acoustic_Solver, Scalar_Transport, Dyn_Tend_Module, Diagnostics_Module, Dissipation_Module, Boundary_Module, IAU_Module, Halo_Manager, or Field_Store) is delivered, THE Component_Test_Suite SHALL include a Unit_Test for that component delivered together with the component.
2. THE Component_Test_Suite SHALL provide, for each ported Dycore_Port component, a Unit_Test that compares the component output against the Reference_Model output for the same inputs to within the Parity_Tolerance.
3. THE Component_Test_Suite SHALL provide a Property_Based_Test for each Correctness_Property preserved by a Dycore_Port component, including the monotonic limiter keeping updated scalars within the local minimum and maximum bounds, positive-definite transport keeping water-species mixing ratios non-negative, and conservation of total dry air mass over a timestep.
4. WHEN a Property_Based_Test executes, THE Property_Based_Test SHALL evaluate its Correctness_Property over generated valid inputs on the active Execution_Space.
5. THE Component_Test_Suite SHALL execute each Property_Based_Test over a configurable number of generated inputs.
6. IF a Property_Based_Test detects an input that violates its Correctness_Property, THEN THE Property_Based_Test SHALL report a minimal failing case obtained by shrinking the generated input.
7. THE Component_Test_Suite SHALL be runnable within the existing MPAS-Atmosphere build and test setup on the host/CPU Execution_Space.
8. WHERE the device/GPU Execution_Space is available, THE Component_Test_Suite SHALL be runnable on the device/GPU Execution_Space.
9. IF a component output diverges from the Reference_Model output beyond the Parity_Tolerance for the same inputs, THEN the corresponding Unit_Test SHALL fail and identify the diverging field.
10. IF a Dycore_Port component violates a specified Correctness_Property for a generated input, THEN the corresponding Property_Based_Test SHALL fail and identify the violated Correctness_Property.
11. THE Component_Test_Suite SHALL be runnable within the Dev_Container using the toolchain and dependencies that the Dev_Container provides.
