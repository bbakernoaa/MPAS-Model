/**
 * @file dycore_c_api.h
 * @brief C API for the MPAS nonhydrostatic dycore (C++ / Kokkos implementation).
 *
 * This header declares the three entry points that the Fortran atmosphere core
 * driver calls via iso_c_binding.  All parameters are passed as plain C types
 * (pointers, int, double) so no C++ headers or Kokkos awareness is required on
 * the Fortran side.
 *
 * Lifecycle:
 *   1. dycore_init   -- called once during dynamics-initialize (Req 13.1, 13.5)
 *   2. dycore_timestep -- called once per model timestep     (Req 13.7, 13.8)
 *   3. dycore_finalize -- called once during model shutdown  (Req 13.5)
 *
 * Requirements: 13.1, 13.2, 13.5, 13.7, 13.8, 13.9, 1.9, 1.10, 1.11, 1.12
 */

#ifndef MPAS_DYCORE_C_API_H
#define MPAS_DYCORE_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the C++ dycore.
 *
 * Must be called exactly once before the first dycore_timestep.  Performs:
 *  - Kokkos::initialize()
 *  - Wraps mesh geometry pointers as DualViews via Field_Store::wrap()
 *  - sync_to_device on mesh geometry (static, one-time sync)
 *  - Wraps prognostic state pointers as DualViews
 *  - Builds Config from the provided scalar parameters
 *
 * @param nCells        Number of cells (owned + halo).
 * @param nEdges        Number of edges (owned + halo).
 * @param nVertices     Number of vertices (owned + halo).
 * @param nVertLevels   Number of vertical layers (mid-levels).
 * @param maxEdges      Maximum number of edges on any cell.
 * @param num_scalars   Number of scalar tracers.
 *
 * Mesh geometry pointers (Fortran-owned, LayoutLeft):
 * @param cellsOnEdge     (2, nEdges)             - int*
 * @param edgesOnCell     (maxEdges, nCells)      - int*
 * @param verticesOnEdge  (2, nEdges)             - int*
 * @param nEdgesOnCell_ptr (nCells)               - int*
 * @param dvEdge          (nEdges)                - double*
 * @param dcEdge          (nEdges)                - double*
 * @param areaCell        (nCells)                - double*
 * @param zgrid           (nVertLevels+1, nCells) - double*
 * @param zz              (nVertLevels, nCells)   - double*
 * @param fzm             (nVertLevels, nCells)   - double*
 * @param fzp             (nVertLevels, nCells)   - double*
 *
 * Prognostic state pointers (Fortran-owned, LayoutLeft, time-level 1 and 2):
 * @param u_tl1      (nVertLevels, nEdges) time level 1
 * @param u_tl2      (nVertLevels, nEdges) time level 2
 * @param w_tl1      (nVertLevels+1, nCells) time level 1
 * @param w_tl2      (nVertLevels+1, nCells) time level 2
 * @param theta_m_tl1 (nVertLevels, nCells) time level 1
 * @param theta_m_tl2 (nVertLevels, nCells) time level 2
 * @param rho_zz_tl1  (nVertLevels, nCells) time level 1
 * @param rho_zz_tl2  (nVertLevels, nCells) time level 2
 * @param scalars_tl1  (num_scalars * nVertLevels, nCells) time level 1
 * @param scalars_tl2  (num_scalars * nVertLevels, nCells) time level 2
 *
 * Configuration scalars:
 * @param time_integration_order  RK order (2 or 3).
 * @param number_of_sub_steps     Acoustic substeps per RK stage.
 * @param dynamics_split_steps    Dynamics-transport splitting count.
 * @param config_monotonic        Monotonic transport flag (0/1).
 * @param config_scalar_advection Scalar advection enable (0/1).
 * @param config_apply_lbcs       Regional mode flag (0/1).
 * @param config_mix_full         Full-state mixing flag (0/1).
 * @param config_iau              IAU enable (0/1).
 * @param gpu_aware_comm          GPU-aware MPI flag (0/1).
 * @param mpi_comm_fortran  Fortran MPI communicator handle (converted via MPI_Comm_f2c).
 *
 * Marshalled halo topology (Req 7.5).  For each element kind (cell, edge,
 * vertex) and direction (send, recv), the MPAS exchange lists are flattened by
 * the Fortran shim into a CSR-style description that this entry point accepts
 * as plain C arrays.  For a given (kind, direction) group:
 *   - <kind>_<dir>_n_neighbors : number of neighbor ranks in this direction.
 *   - <kind>_<dir>_n_layers    : number of halo layers.
 *   - <kind>_<dir>_neighbor_ranks : int[n_neighbors] neighbor MPI ranks.
 *   - <kind>_<dir>_layer_counts   : int[n_neighbors * n_layers] index count for
 *                                   each (neighbor, layer), row-major by neighbor
 *                                   then layer.
 *   - <kind>_<dir>_indices        : int[sum(layer_counts)] concatenated 0-based
 *                                   local indices; slices are delimited by the
 *                                   prefix sum of layer_counts.
 * An empty direction is signalled by n_neighbors == 0 (the pointer arguments may
 * then be null).  Indices are already converted to 0-based by the marshaller.
 * NOTE: This entry point currently only accepts these arrays; reconstruction of
 * the HaloTopology from them is performed in a later step.
 *
 * @return 0 on success, non-zero error code on failure:
 *   - 1: Kokkos initialization failed
 *   - 2: Invalid dimensions (nVertLevels <= 0, nCells <= 0, etc.)
 *   - 3: Null pointer for a required array
 */
int dycore_init(
    /* Mesh dimensions */
    int nCells, int nEdges, int nVertices, int nVertLevels, int maxEdges,
    int num_scalars, int nCellsSolve, int nEdgesSolve,
    /* Mesh geometry / connectivity pointers */
    int* cellsOnEdge, int* edgesOnCell, int* verticesOnEdge,
    int* nEdgesOnCell_ptr,
    double* dvEdge, double* dcEdge, double* areaCell,
    double* zgrid, double* zz, double* fzm, double* fzp,
    /* Prognostic state pointers (two time levels each) */
    double* u_tl1, double* u_tl2,
    double* w_tl1, double* w_tl2,
    double* theta_m_tl1, double* theta_m_tl2,
    double* rho_zz_tl1, double* rho_zz_tl2,
    double* scalars_tl1, double* scalars_tl2,
    /* Configuration */
    int time_integration_order, int number_of_sub_steps,
    int dynamics_split_steps,
    int config_monotonic, int config_scalar_advection,
    int config_apply_lbcs, int config_mix_full,
    int config_iau, int gpu_aware_comm,
    double config_smdiv, double config_len_disp,
    double config_apvm_upwinding, int config_hollingsworth,
    /* MPI */
    int mpi_comm_fortran,
    /* Marshalled halo topology: cell send */
    int cell_send_n_neighbors, int cell_send_n_layers,
    int* cell_send_neighbor_ranks, int* cell_send_layer_counts,
    int* cell_send_indices,
    /* Marshalled halo topology: cell recv */
    int cell_recv_n_neighbors, int cell_recv_n_layers,
    int* cell_recv_neighbor_ranks, int* cell_recv_layer_counts,
    int* cell_recv_indices,
    /* Marshalled halo topology: edge send */
    int edge_send_n_neighbors, int edge_send_n_layers,
    int* edge_send_neighbor_ranks, int* edge_send_layer_counts,
    int* edge_send_indices,
    /* Marshalled halo topology: edge recv */
    int edge_recv_n_neighbors, int edge_recv_n_layers,
    int* edge_recv_neighbor_ranks, int* edge_recv_layer_counts,
    int* edge_recv_indices,
    /* Marshalled halo topology: vertex send */
    int vertex_send_n_neighbors, int vertex_send_n_layers,
    int* vertex_send_neighbor_ranks, int* vertex_send_layer_counts,
    int* vertex_send_indices,
    /* Marshalled halo topology: vertex recv */
    int vertex_recv_n_neighbors, int vertex_recv_n_layers,
    int* vertex_recv_neighbor_ranks, int* vertex_recv_layer_counts,
    int* vertex_recv_indices);

/**
 * @brief Advance the dycore by one timestep.
 *
 * On entry: sync_to_device on all prognostic state DualView fields (Req 13.7).
 * On exit:  sync_to_host on all modified prognostic state fields (Req 13.8).
 *
 * @param dt         Timestep size in seconds.
 * @param itimestep  Integer timestep index.
 */
void dycore_timestep(double dt, int itimestep);

/**
 * @brief Finalize the C++ dycore and shut down Kokkos.
 *
 * Must be called exactly once during model shutdown after the last timestep.
 * Calls Kokkos::finalize() (Req 13.5).
 */
void dycore_finalize(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* MPAS_DYCORE_C_API_H */
