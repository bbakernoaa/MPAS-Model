# Requirements Document

## Introduction

The MPAS-Atmosphere dynamical core is being ported from Fortran to a C++/Kokkos
implementation with the goal of numerical parity (target 1e-13 L-infinity
relative difference) against the legacy Fortran on the Jablonowski-Williamson
(JW) baroclinic-wave test. The Fortran implementation is the source of truth and
the C++ implementation must mimic its behavior.

The production C++ dycore currently performs no halo (ghost-cell) exchange: in
`dycore_c_api.cpp` the `AdvanceDomain` is constructed with
`.halo_manager = nullptr`, so every guarded halo-exchange call site in
`time_integrator_advance.hpp` silently no-ops. On a multi-rank decomposition
(the validation runs on 48 MPI ranks) owned cells adjacent to partition
boundaries read stale or uninitialized halo neighbors through their stencils,
producing large parity errors (u ~0.23, w ~88, theta_m ~1e-3, rho_zz ~1e-2) and
tens of thousands of non-finite tendency values in non-owned cells. On a single
rank the run produces NaN because the ghost/padding slot is read uninitialized.

The chosen solution adds a general, reusable indexed (gather/scatter)
halo-exchange capability to the HALO library at `helm-project/libs/halo/`, then
wires the existing MPAS decomposition into it. The existing HALO contiguous
flat-buffer API (`Halo_Plan` + `exchange_blocking`) assumes a
`[send | owned | recv]` layout and does not fit the MPAS `[owned | halo]` array
layout with arbitrary per-neighbor index lists. This feature introduces an
indexed plan and exchange that gathers owned elements by index, communicates via
MPI, and scatters received values into halo elements by index. The MPAS
send/recv exchange lists that already exist after MPAS initialization are
marshalled through the dycore C API and used to build the indexed plans, without
altering MPAS pool memory or the `atm_srk3` interface.

## Glossary

- **HALO_Library**: The domain-agnostic C++ halo-exchange library located at
  `helm-project/libs/halo/`.
- **Indexed_Halo_Plan**: A new precomputed HALO plan type that stores, per
  neighbor, a list of local send indices (owned elements to gather) and a list
  of local recv indices (halo elements to scatter into), for a single element
  kind. Distinct from the existing contiguous `Halo_Plan`.
- **Indexed_Exchange**: A new HALO API (`exchange_indexed`) that performs a
  blocking gather/communicate/scatter halo exchange using an `Indexed_Halo_Plan`.
- **Element_Kind**: The mesh element type a field is defined on; one of cell,
  edge, or vertex.
- **Halo_Layer**: A ring of ghost elements at increasing topological distance
  from a partition's owned region. MPAS organizes exchange index lists per halo
  layer.
- **Layer_Subset**: The specific set of halo layers a given field or group must
  exchange (for example `{1,2}`, `{3}`, or `{1}`).
- **Field_View**: A Kokkos view holding a field's values in the C++ dycore, in
  `[owned | halo]` element order, either rank-1 (nElements) or rank-2
  (nVertLevels x nElements, LayoutLeft).
- **MPAS_Exchange_List**: The Fortran `mpas_exchange_list` linked-list node
  (`endPointID`, `nlist`, `srcList`, `destList`, `next`) describing one neighbor.
- **MPAS_Multihalo_Exchange_List**: The Fortran `mpas_multihalo_exchange_list`
  (`halos(:) % exchList`) that indexes MPAS_Exchange_Lists by halo layer.
- **MPAS_Marshaller**: The component that extracts existing MPAS send/recv
  exchange lists (per Element_Kind, per Halo_Layer) and passes them across the
  dycore C API using 0-based local indices.
- **Dycore_C_API**: The C entry points (`dycore_init`, `dycore_timestep`,
  `dycore_finalize`) that the Fortran driver calls via iso_c_binding.
- **Halo_Manager**: The C++ dycore component
  (`mpas_dycore/halo_manager.hpp`) that owns the HALO communicator, builds
  indexed plans per Element_Kind, and performs named-group exchanges.
- **Halo_Group_Registry**: The compile-time table
  (`halo_group_registry.hpp`) mapping a group name to its ordered field entries
  (field name, time level, Layer_Subset).
- **AdvanceDomain**: The aggregate in `time_integrator_advance.hpp` that carries
  a `Halo_Manager*` and drives the SRK3 time integration.
- **JW_Validation**: The 48-rank Jablonowski-Williamson baroclinic-wave
  comparison against the Fortran reference output, at the first output step, for
  fields u, w, theta_m, and rho_zz.
- **Parity_Tolerance**: The acceptance threshold of 1e-13 L-infinity relative
  difference between C++ and Fortran output fields.

## Requirements

### Requirement 1: Indexed halo-exchange plan type in the HALO library

**User Story:** As a developer of an unstructured-mesh application, I want an
indexed halo-exchange plan that stores per-neighbor send and receive index
lists, so that I can exchange fields stored in an `[owned | halo]` layout without
reorganizing memory into a contiguous send/recv buffer.

#### Acceptance Criteria

1. THE HALO_Library SHALL provide an Indexed_Halo_Plan type that stores, for
   each send neighbor, the neighbor rank and an ordered list of local send
   indices, and for each receive neighbor, the neighbor rank and an ordered list
   of local receive indices.
2. WHEN an Indexed_Halo_Plan is constructed with a neighbor rank that is negative
   or greater than or equal to the communicator size, THEN THE HALO_Library SHALL
   reject construction with an error identifying the invalid rank.
3. WHEN an Indexed_Halo_Plan is constructed, THE HALO_Library SHALL compute and
   expose the total send index count and total receive index count.
4. WHERE a neighbor's send or receive index list is empty, THE HALO_Library SHALL
   accept the Indexed_Halo_Plan as valid with zero exchanged elements for that
   neighbor.
5. THE Indexed_Halo_Plan SHALL record the Element_Kind it applies to so that a
   field can be matched to the correct plan.

### Requirement 2: Indexed gather/scatter blocking exchange API

**User Story:** As a developer, I want a blocking `exchange_indexed` operation
that gathers owned elements by index, communicates them via MPI, and scatters
received values into halo elements by index, so that halo cells hold the correct
neighbor-owned values after the call.

#### Acceptance Criteria

1. WHEN Indexed_Exchange is invoked with an Indexed_Halo_Plan and a Field_View,
   THE HALO_Library SHALL gather the field values at each send neighbor's send
   indices into per-neighbor send buffers.
2. WHEN Indexed_Exchange is invoked, THE HALO_Library SHALL post all receive
   operations before any send operation and SHALL complete all operations before
   returning.
3. WHEN Indexed_Exchange completes, THE HALO_Library SHALL scatter each received
   buffer into the Field_View at the corresponding receive neighbor's receive
   indices.
4. WHEN Indexed_Exchange is invoked with an Indexed_Halo_Plan that has no send
   neighbors and no receive neighbors, THE HALO_Library SHALL return without
   performing communication and SHALL leave the Field_View unchanged.
5. IF an MPI operation invoked by Indexed_Exchange returns a failure code, THEN
   THE HALO_Library SHALL raise an error identifying the failing operation and
   the neighbor rank involved.
6. THE HALO_Library SHALL compute message tags for Indexed_Exchange using the
   same deterministic sender-receiver tag scheme used by the existing contiguous
   exchange.

### Requirement 3: Field rank and memory-space support

**User Story:** As a developer, I want indexed exchange to support both host and
device Kokkos views and both rank-1 and rank-2 fields, so that all dycore
prognostic and tendency fields can be exchanged with one API.

#### Acceptance Criteria

1. WHERE a Field_View is a rank-1 view indexed by element, THE HALO_Library SHALL
   gather and scatter one value per index.
2. WHERE a Field_View is a rank-2 view of shape (nVertLevels x nElements) in
   LayoutLeft, THE HALO_Library SHALL gather and scatter all nVertLevels values
   for each index as a contiguous per-element block.
3. WHERE a Field_View resides in a device memory space and GPU-aware MPI is not
   available, THE HALO_Library SHALL stage gathered data through host memory for
   communication and copy scattered results back to device memory.
4. WHERE a Field_View resides in a device memory space and GPU-aware MPI is
   available, THE HALO_Library SHALL communicate device-resident buffers directly
   without a host copy.
5. WHERE a Field_View resides in a host memory space, THE HALO_Library SHALL
   communicate directly from host buffers.

### Requirement 4: Reuse of existing HALO infrastructure

**User Story:** As a maintainer of the HALO library, I want the indexed exchange
to reuse the existing staging, GPU-aware detection, diagnostics, and
thread-safety facilities, so that behavior stays consistent and no parallel
infrastructure is duplicated.

#### Acceptance Criteria

1. THE HALO_Library SHALL perform Indexed_Exchange gather and scatter using
   Kokkos kernels that execute in the Field_View's native execution space.
2. WHILE the active MPI thread level is below MPI_THREAD_MULTIPLE, THE
   HALO_Library SHALL serialize Indexed_Exchange MPI operations using the
   existing serialization guard.
3. WHERE diagnostics are active, THE HALO_Library SHALL emit begin and end
   exchange events for Indexed_Exchange recording neighbor count and byte volume.
4. THE HALO_Library SHALL select the host-staged path or the GPU-aware path for
   Indexed_Exchange using the same runtime GPU-aware detection used by the
   existing contiguous exchange.

### Requirement 5: Halo-layer-subset exchange

**User Story:** As a developer porting MPAS halo groups, I want to exchange a
specified subset of halo layers for a given field, so that the C++ exchange
matches the exact per-field layer sets that the Fortran reference exchanges.

#### Acceptance Criteria

1. THE Indexed_Halo_Plan SHALL store send and receive index lists organized by
   Halo_Layer.
2. WHEN Indexed_Exchange is invoked with a Layer_Subset, THE HALO_Library SHALL
   gather and scatter only the send and receive indices belonging to the layers
   in that Layer_Subset.
3. WHERE a field's Layer_Subset is a proper subset of the available halo layers,
   THE HALO_Library SHALL leave halo elements in the excluded layers unchanged.

### Requirement 6: Domain-agnostic, independently testable HALO feature

**User Story:** As a HALO library user outside MPAS, I want the indexed exchange
to depend only on ranks and index lists, so that any unstructured-mesh client can
use it and it can be unit-tested without MPAS.

#### Acceptance Criteria

1. THE Indexed_Halo_Plan and Indexed_Exchange SHALL depend only on the HALO
   communicator, neighbor ranks, and index lists, and SHALL NOT depend on any
   MPAS-specific type.
2. THE HALO_Library SHALL provide unit tests for Indexed_Exchange that construct
   index lists directly and verify gather/scatter correctness independently of
   MPAS.
3. WHEN Indexed_Exchange is run on a single-rank communicator with a plan that
   has no neighbors, THE HALO_Library SHALL leave all owned and halo elements at
   their pre-call values.

### Requirement 7: Marshalling MPAS exchange lists across the C API

**User Story:** As an MPAS integrator, I want the existing MPAS send/recv
exchange lists to be extracted and passed into the C++ dycore, so that the
indexed plans use the real partition decomposition rather than a rebuilt one.

#### Acceptance Criteria

1. THE MPAS_Marshaller SHALL extract, for each Element_Kind (cell, edge, vertex),
   the existing MPAS send and receive exchange lists, reading each
   MPAS_Exchange_List's neighbor rank, count, and index arrays.
2. THE MPAS_Marshaller SHALL extract exchange index lists per Halo_Layer from the
   MPAS_Multihalo_Exchange_List halos array.
3. WHEN the MPAS_Marshaller passes exchange index values across the Dycore_C_API,
   THE MPAS_Marshaller SHALL convert MPAS 1-based local indices to 0-based local
   indices for use in C++ Field_Views.
4. THE MPAS_Marshaller SHALL read the existing MPAS exchange lists without
   rebuilding or recomputing the partition decomposition.
5. THE Dycore_C_API SHALL accept the marshalled per-Element_Kind, per-Halo_Layer
   send and receive index lists together with neighbor ranks as parameters to the
   dycore initialization entry point.

### Requirement 8: Preservation of MPAS memory layout and interface

**User Story:** As an MPAS maintainer, I want the halo integration to leave MPAS
pool memory and the atm_srk3 interface unchanged, so that the Fortran side is not
disturbed by the C++ halo work.

#### Acceptance Criteria

1. THE MPAS_Marshaller SHALL NOT modify the memory layout of any MPAS pool field.
2. THE feature SHALL NOT change the argument list or calling convention of the
   MPAS atm_srk3 interface.

### Requirement 9: Halo_Manager construction of indexed plans and group exchange

**User Story:** As a dycore developer, I want the Halo_Manager to build indexed
plans from the real topology and honor each field's time level and layer subset,
so that a named-group exchange updates exactly the fields, time levels, and halo
layers that the Fortran reference updates.

#### Acceptance Criteria

1. WHEN the Halo_Manager is constructed with the marshalled topology, THE
   Halo_Manager SHALL build one Indexed_Halo_Plan per Element_Kind from the
   per-neighbor, per-Halo_Layer send and receive index lists.
2. WHEN the Halo_Manager exchanges a named group, THE Halo_Manager SHALL select
   the Indexed_Halo_Plan matching each field's Element_Kind.
3. WHEN the Halo_Manager exchanges a named group, THE Halo_Manager SHALL exchange
   each field at the time level recorded in the Halo_Group_Registry entry.
4. WHEN the Halo_Manager exchanges a named group, THE Halo_Manager SHALL exchange
   only the halo layers in each field entry's Layer_Subset.
5. IF the Halo_Manager is asked to exchange a group name that is not in the
   Halo_Group_Registry, THEN THE Halo_Manager SHALL raise an error naming the
   unknown group.
6. WHERE a field named in a group is not present in the field store, THE
   Halo_Manager SHALL skip that field and continue exchanging the remaining
   fields in the group.

### Requirement 10: Wiring the Halo_Manager into the production advance

**User Story:** As a dycore developer, I want the Halo_Manager wired into the
production AdvanceDomain, so that the halo-exchange call sites in the SRK3
integrator perform real exchanges instead of no-ops.

#### Acceptance Criteria

1. WHEN the Dycore_C_API initializes the dycore, THE Dycore_C_API SHALL construct
   a Halo_Manager from the marshalled topology and the field store.
2. WHEN the Dycore_C_API constructs the AdvanceDomain for a timestep, THE
   Dycore_C_API SHALL set the AdvanceDomain halo_manager to the constructed
   Halo_Manager.
3. WHEN the SRK3 integrator reaches a halo-exchange call site, THE C++_Dycore
   SHALL invoke the corresponding named-group exchange on the Halo_Manager.

### Requirement 11: Numerical parity and finiteness acceptance

**User Story:** As a model validator, I want the halo-enabled C++ dycore to match
the Fortran reference and produce only finite values, so that I can confirm the
port is numerically correct.

#### Acceptance Criteria

1. WHEN JW_Validation is run with halo exchange enabled, THE C++_Dycore SHALL
   produce output fields u, w, theta_m, and rho_zz that match the Fortran
   reference within the Parity_Tolerance at the first output step.
2. WHEN JW_Validation is run with halo exchange enabled, THE C++_Dycore SHALL
   produce only finite values in owned elements and in halo elements for every
   tendency and prognostic field.
3. WHEN the C++_Dycore is run on a single MPI rank, THE C++_Dycore SHALL produce
   only finite values and SHALL NOT produce NaN.

### Requirement 12: Deterministic, order-independent results

**User Story:** As a model validator, I want halo-exchange results to be
independent of neighbor processing order, so that runs are reproducible.

#### Acceptance Criteria

1. FOR ALL orderings of the neighbors within an Indexed_Halo_Plan, Indexed_Exchange
   SHALL produce identical Field_View values after completion.
2. WHEN the same Indexed_Exchange is applied twice in succession with no
   intervening change to owned elements, THE HALO_Library SHALL leave the halo
   elements at the same values after the second exchange as after the first
   (idempotence for a fixed owned state).
