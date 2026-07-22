# Implementation Plan: cpp-dycore-halo-exchange

## Overview

This plan implements a general indexed (gather/scatter) halo-exchange capability
in the HALO library and wires the existing MPAS partition decomposition into the
C++/Kokkos dycore. Work proceeds in four independently verifiable layers, then a
validation layer:

1. **HALO indexed exchange** (Requirements 1-6, 12) - buildable and testable
   without MPAS.
2. **MPAS Fortran->C marshalling** (Requirements 7-8).
3. **Halo_Manager integration** (Requirement 9).
4. **Production wiring** (Requirements 10) and **end-to-end parity**
   (Requirement 11).

Each task builds on the previous ones and ends with integration into a working
whole. Implementation language is C++ (with a Fortran shim for marshalling), per
the design.

**Build note:** Use `--cpus=4` when running Docker builds. The C++ dycore library and its tests are already built in the `mpas-dycore-build` Docker image.

## Tasks

- [ ] 1. HALO indexed plan type (domain-agnostic)
  - [x] 1.1 Implement `Indexed_Halo_Plan`, `Element_Kind`, and `Indexed_Neighbor`
    - Create `helm-project/libs/halo/include/halo/indexed_halo_plan.hpp`
    - Define `enum class Element_Kind { cell, edge, vertex, generic }`
    - Define `Indexed_Neighbor { rank; layers[][]; total_indices(); indices_for(layer_subset) }`
    - Implement `Indexed_Halo_Plan` with per-neighbor, per-layer send/recv index
      lists, storing the element kind, exposing send/recv info accessors,
      neighbor counts, total send/recv index counts, layer count, and communicator
    - Validate ranks in `[0, comm.size())`, reject duplicate ranks in a list,
      accept empty index lists as valid; throw `std::invalid_argument` naming the
      offending rank
    - Depend only on `Communicator`, ranks, and index lists (no MPAS types)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 6.1_

  - [-] 1.2 Write property test for plan topology preservation
    - **Property 1: Plan preserves neighbor topology**
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 1`
    - _Validates: Requirements 1.1, 1.5, 5.1_

  - [-] 1.3 Write property test for plan totals
    - **Property 2: Plan totals equal the sum of index lists**
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 2`
    - _Validates: Requirements 1.3_

  - [-] 1.4 Write unit tests for plan construction and accessors
    - Invalid-rank construction throws naming the rank (Req 1.2)
    - Empty send/recv index lists accepted as valid (Req 1.4)
    - Element-kind accessor returns the supplied kind (Req 1.5)
    - _Requirements: 1.2, 1.4, 1.5_

- [ ] 2. HALO indexed gather/scatter exchange
  - [-] 2.1 Implement `exchange_indexed`
    - Create `helm-project/libs/halo/include/halo/exchange_indexed.hpp`
    - Early-return unchanged when the plan has no send and no recv neighbors
    - Compute per-element block size: rank-1 -> 1; rank-2 (nVertLevels x nElements,
      LayoutLeft) -> `extent(0)`
    - Gather owned values at each send neighbor's `layer_subset` indices into dense
      per-neighbor buffers using Kokkos kernels in the view's execution space
    - Post all `MPI_Irecv`, then all `MPI_Isend`, then `MPI_Waitall`; scatter recv
      buffers into halo indices for the selected layers only
    - Reuse `detail::compute_tag`, `detail::Serialized_MPI_Guard`,
      `detail::staging`, `detail::gpu_aware_probe` /
      `Environment::is_gpu_aware_mpi()`, and `Diagnostics` (begin/end events with
      neighbor count and byte volume)
    - Route MPI failures through `detail::handle_mpi_error` naming the operation and
      neighbor rank
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 5.2, 5.3, 6.1_

  - [~] 2.2 Write property test for indexed exchange round-trip
    - **Property 3: Indexed exchange round-trip fills halo with source-owned values**
    - In-process paired send/recv index model; rank-1 and rank-2 fields
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 3`
    - _Validates: Requirements 2.1, 2.3, 3.1, 3.2, 3.5_

  - [~] 2.3 Write property test for empty/neighborless no-op
    - **Property 4: Empty or neighborless plan is a no-op**
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 4`
    - _Validates: Requirements 2.4, 6.3_

  - [~] 2.4 Write property test for layer-subset exactness
    - **Property 5: Layer-subset exactness**
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 5`
    - _Validates: Requirements 5.2, 5.3_

  - [~] 2.5 Write property test for neighbor-order independence
    - **Property 8: Neighbor-order independence**
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 8`
    - _Validates: Requirements 12.1_

  - [~] 2.6 Write property test for idempotence with fixed owned state
    - **Property 9: Idempotence for a fixed owned state**
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 9`
    - _Validates: Requirements 12.2_

  - [~] 2.7 Write unit/example tests for exchange infrastructure paths
    - Diagnostics begin/end emission with correct neighbor count and byte volume (Req 4.3)
    - Tag equivalence with the contiguous exchange for matching rank pairs (Req 2.6)
    - MPI failure raises an error naming operation and neighbor rank (Req 2.5)
    - Single-rank no-neighbor plan leaves all elements unchanged (Req 6.3)
    - _Requirements: 2.5, 2.6, 4.3, 6.3_

- [~] 3. Checkpoint - HALO indexed exchange
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. HALO multi-rank and device integration tests
  - [~] 4.1 Write multi-rank MPI integration test
    - 2-4 rank exchange on a small synthetic partition; seed owned values as a
      function of (rank, index) and verify each halo cell against the analytic
      source value (also exercises Property 3 over the real MPI path)
    - _Requirements: 2.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4_

  - [~] 4.2 Write device-view round-trip integration test
    - Host-staged path for a device-resident view, and the GPU-aware direct path
      when a GPU-aware MPI environment is available
    - _Requirements: 3.3, 3.4, 4.1_

- [ ] 5. MPAS marshalling across the dycore C API
  - [x] 5.1 Extend the `dycore_init` C API signature with marshalled topology
    - Add per-element-kind, per-direction CSR parameters to the `dycore_init` entry
      point in `dycore_c_api` (`n_neighbors`, `n_layers`, `neighbor_ranks[]`,
      `layer_counts[n_neighbors*n_layers]`, concatenated `indices[]`) for cell,
      edge, and vertex send and recv lists
    - _Requirements: 7.5_

  - [-] 5.2 Implement `dycore_fortran_shim.F90` extraction and marshalling
    - Walk each field's `mpas_multihalo_exchange_list % halos(:) % exchList` per
      element kind, collecting per-neighbor `(endPointID, per-layer srcList/destList)`
    - Flatten into CSR arrays and pass via iso_c_binding to `dycore_init`
    - Convert MPAS 1-based local indices to 0-based at the boundary
    - Read existing lists only; do not rebuild the decomposition; do not modify
      pool memory or the `atm_srk3` interface
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.2_

  - [~] 5.3 Write property test for 1-based to 0-based conversion
    - **Property 6: One-based to zero-based index conversion**
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 6`
    - _Validates: Requirements 7.3_

  - [~] 5.4 Write unit tests for the marshalling conversion helper
    - Boundary cases for 1-based to 0-based conversion (index 1 -> 0, non-negative)
    - _Requirements: 7.3_

  - [~] 5.5 Write integration test for marshalling against a small MPAS partition
    - Extracted per-element-kind, per-layer neighbor ranks and index counts match
      the MPAS `sendList`/`recvList` multihalo lists
    - _Requirements: 7.1, 7.2, 7.4_

- [ ] 6. Halo_Manager indexed-plan construction and group exchange
  - [x] 6.1 Extend `HaloTopology` and `HaloNeighborInfo` with per-layer index lists
    - Replace `{rank, count}` with `{rank, layers[][]}` in `HaloNeighborInfo`
    - Add cell/edge/vertex send and recv neighbor lists to `HaloTopology`
    - _Requirements: 9.1_

  - [~] 6.2 Build one `Indexed_Halo_Plan` per element kind and add element-kind mapping
    - Construct cell, edge, and vertex plans from the topology, leaving a plan
      `nullptr` when its lists are empty (single-rank)
    - Add a static field-name -> `Element_Kind` table replacing the hard-coded
      name-based edge/cell split
    - _Requirements: 9.1, 9.2_

  - [~] 6.3 Make `exchange(group_name)` honor element kind, time level, and layer subset
    - For each registry field entry, resolve element kind -> plan, time level ->
      view, and pass `halo_layers` (converted to 0-based) as the `layer_subset` to
      `exchange_indexed`
    - Throw naming the group for an unknown group; skip missing fields and continue
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.6_

  - [~] 6.4 Write property test for Halo_Manager plan construction
    - **Property 7: Halo_Manager builds plans faithfully from topology**
    - Minimum 100 iterations; tag with `Feature: cpp-dycore-halo-exchange, Property 7`
    - _Validates: Requirements 9.1_

  - [~] 6.5 Write unit tests for group exchange behavior
    - Element-kind mapping per representative field (Req 9.2)
    - Time level forwarded from registry (Req 9.3)
    - Layer subset forwarded 0-based (Req 9.4)
    - Unknown-group throw (Req 9.5); missing-field skip (Req 9.6)
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.6_

- [~] 7. Checkpoint - marshalling and Halo_Manager
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Production wiring in the dycore C API
  - [~] 8.1 Reconstruct `HaloTopology` from marshalled arrays in `dycore_init`
    - Decode the CSR per-element-kind, per-layer arrays into `HaloTopology`
    - _Requirements: 10.1_

  - [~] 8.2 Construct the `Halo_Manager` and attach it to `AdvanceDomain`
    - Build the `Halo_Manager` from the reconstructed topology and field store,
      store it in the persistent dycore context, and set
      `AdvanceDomain.halo_manager = &context->halo_manager` (replacing `nullptr`)
    - _Requirements: 10.1, 10.2, 10.3_

  - [~] 8.3 Write smoke test for wiring
    - `dycore_init` succeeds with marshalled arrays and produces a non-null
      `Halo_Manager`; `AdvanceDomain.halo_manager` is non-null when built for a
      timestep
    - _Requirements: 7.5, 10.1, 10.2_

- [ ] 9. Build and end-to-end parity verification
  - [~] 9.1 Build the dycore incrementally and fix compile errors
    - Run `bash -l /gpfs/f6/bil-fire3/scratch/Barry.Baker/models/MPAS-Model/testing/_kiro_build.sh`
    - _Requirements: 10.1, 10.2, 10.3_

  - [~] 9.2 Run the 48-rank JW validation and compare against the Fortran reference
    - `cd testing && sbatch submit_validate_gaea.sh`
    - `compare_dycore_outputs.py <ref>/output.nc <cpp>/output.nc --tolerance 1e-13
      --fields u w theta_m rho_zz`
    - _Requirements: 11.1_

  - [~] 9.3 Confirm finiteness and single-rank stability
    - Scan output fields for NaN/Inf in owned and halo cells
    - Run the single-rank stability check and confirm no NaN
    - _Requirements: 11.2, 11.3_

  - [~] 9.4 Confirm interface stability
    - Diff the `atm_srk3` interface to confirm no signature change; confirm the
      marshaller only reads MPAS pool pointers
    - _Requirements: 8.1, 8.2_

- [~] 10. Final checkpoint - full parity
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster
  MVP, but they validate the correctness properties and error paths.
- Each task references specific requirements (granular sub-requirements) for
  traceability.
- Property tests (Properties 1-9) target the pure indexed-exchange and
  plan-construction logic; MPI-path, marshalling, wiring, and 48-rank parity
  criteria are covered by integration, smoke, and end-to-end tests.
- The HALO library layer (Tasks 1-4) is buildable and testable without MPAS.
- Checkpoints ensure incremental validation before moving to the next layer.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "5.1", "6.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4", "2.1", "5.2", "6.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "4.1", "4.2", "5.3", "5.4", "5.5", "6.3"] },
    { "id": 3, "tasks": ["6.4", "6.5", "8.1"] },
    { "id": 4, "tasks": ["8.2", "8.3"] },
    { "id": 5, "tasks": ["9.1"] },
    { "id": 6, "tasks": ["9.2", "9.4"] },
    { "id": 7, "tasks": ["9.3"] }
  ]
}
```
