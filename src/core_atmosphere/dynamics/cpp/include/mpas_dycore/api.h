#ifndef MPAS_DYCORE_API_H
#define MPAS_DYCORE_API_H

/**
 * @file api.h
 * @brief C-linkage API for the MPAS dynamical core C++ implementation.
 *
 * Callable from Fortran via ISO_C_BINDING. Each entry point returns an
 * integer error code: 0 on success, 1-255 on error with a diagnostic
 * message written to the caller-provided errmsg buffer.
 *
 * Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.8
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the C++ dynamical core.
 *
 * Validates extents, copies connectivity into owned storage, builds mdspan
 * views, validates connectivity ranges, stores halo descriptors, converts
 * the Fortran MPI communicator, and sets the dycore status to Ready.
 *
 * @param nCells            Number of cells in the local partition.
 * @param nEdges            Number of edges in the local partition.
 * @param nVertices         Number of vertices in the local partition.
 * @param maxEdges          Maximum number of edges per cell.
 * @param nVertLevels       Number of vertical levels.
 * @param cellsOnEdge       Edge-to-cell connectivity (nEdges x 2), zero-based.
 * @param cellsOnEdge_d0    First extent of cellsOnEdge array.
 * @param cellsOnEdge_d1    Second extent of cellsOnEdge array.
 * @param verticesOnEdge    Edge-to-vertex connectivity (nEdges x 2), zero-based.
 * @param verticesOnEdge_d0 First extent.
 * @param verticesOnEdge_d1 Second extent.
 * @param edgesOnCell       Cell-to-edge connectivity (nCells x maxEdges), zero-based.
 * @param edgesOnCell_d0    First extent.
 * @param edgesOnCell_d1    Second extent.
 * @param cellsOnCell       Cell-to-cell connectivity (nCells x maxEdges), zero-based.
 * @param cellsOnCell_d0    First extent.
 * @param cellsOnCell_d1    Second extent.
 * @param nEdgesOnCell_arr  Per-cell neighbor count array (nCells).
 * @param nEdgesOnCell_len  Length of nEdgesOnCell_arr.
 * @param halo_ranks        Ranks of halo neighbors.
 * @param halo_n_neighbors  Number of halo neighbors.
 * @param halo_counts       Per-neighbor per-layer send/recv counts.
 * @param halo_counts_len   Length of halo_counts array.
 * @param halo_send_indices Flattened CSR send index list.
 * @param halo_send_len     Length of halo_send_indices.
 * @param halo_recv_indices Flattened CSR recv index list.
 * @param halo_recv_len     Length of halo_recv_indices.
 * @param halo_n_layers     Number of halo layers (1-3).
 * @param dt                Timestep size in seconds.
 * @param number_of_sub_steps Number of acoustic sub-steps (valid: 2-24).
 * @param config_apply_lbcs Whether to apply lateral boundary conditions (0/1).
 * @param config_horiz_mixing Horizontal mixing mode string.
 * @param config_h_mom_eddy_visc2 Del-2 momentum eddy viscosity.
 * @param config_h_mom_eddy_visc4 Del-4 momentum eddy viscosity.
 * @param config_h_theta_eddy_visc2 Del-2 theta eddy viscosity.
 * @param mpi_comm_fortran  Fortran MPI communicator handle.
 * @param errmsg            Buffer for error message output.
 * @param errmsg_len        Maximum length of error message buffer.
 * @return 0 on success, 1-255 on error.
 */
int mpas_dycore_cpp_init(
    /* Mesh dimensions */
    int nCells, int nEdges, int nVertices, int maxEdges, int nVertLevels,
    /* Connectivity arrays (zero-based from marshaller) */
    const int* cellsOnEdge,      int cellsOnEdge_d0, int cellsOnEdge_d1,
    const int* verticesOnEdge,   int verticesOnEdge_d0, int verticesOnEdge_d1,
    const int* edgesOnCell,      int edgesOnCell_d0, int edgesOnCell_d1,
    const int* cellsOnCell,      int cellsOnCell_d0, int cellsOnCell_d1,
    const int* nEdgesOnCell_arr, int nEdgesOnCell_len,
    /* Halo CSR descriptors */
    const int* halo_ranks, int halo_n_neighbors,
    const int* halo_counts, int halo_counts_len,
    const int* halo_send_indices, int halo_send_len,
    const int* halo_recv_indices, int halo_recv_len,
    int halo_n_layers,
    /* Configuration */
    double dt, int number_of_sub_steps,
    int config_apply_lbcs,
    const char* config_horiz_mixing,
    double config_h_mom_eddy_visc2,
    double config_h_mom_eddy_visc4,
    double config_h_theta_eddy_visc2,
    /* MPI communicator (Fortran handle) */
    int mpi_comm_fortran,
    /* Error output */
    char* errmsg, int errmsg_len
);

/**
 * @brief Execute one large timestep of the dynamical core.
 *
 * Must be called after a successful mpas_dycore_cpp_init(). Returns error
 * if the dycore is not in the Ready state.
 *
 * @param u           Horizontal velocity field (nVertLevels x nEdges), read-write.
 * @param u_d0        First extent of u array.
 * @param u_d1        Second extent of u array.
 * @param theta_m     Potential temperature (nVertLevels x nCells), read-write.
 * @param theta_m_d0  First extent.
 * @param theta_m_d1  Second extent.
 * @param rho_zz      Dry air density (nVertLevels x nCells), read-write.
 * @param rho_zz_d0   First extent.
 * @param rho_zz_d1   Second extent.
 * @param w           Vertical velocity (nVertLevels+1 x nCells), read-write.
 * @param w_d0        First extent.
 * @param w_d1        Second extent.
 * @param scalars     Scalar tracers (nScalars x nVertLevels x nCells), read-write.
 * @param scalars_d0  First extent (nScalars).
 * @param scalars_d1  Second extent (nVertLevels).
 * @param scalars_d2  Third extent (nCells).
 * @param errmsg      Buffer for error message output.
 * @param errmsg_len  Maximum length of error message buffer.
 * @return 0 on success, 1-255 on error.
 */
int mpas_dycore_cpp_timestep(
    /* Prognostic fields (read-write) */
    double* u,       int u_d0, int u_d1,
    double* theta_m, int theta_m_d0, int theta_m_d1,
    double* rho_zz,  int rho_zz_d0, int rho_zz_d1,
    double* w,       int w_d0, int w_d1,
    /* Scalars (read-write, nScalars x nVertLevels x nCells) */
    double* scalars, int scalars_d0, int scalars_d1, int scalars_d2,
    /* Error output */
    char* errmsg, int errmsg_len
);

/**
 * @brief Release all internally-allocated resources and reset to uninitialized.
 *
 * After this call, mpas_dycore_cpp_init() may be called again to reinitialize.
 *
 * @param errmsg     Buffer for error message output.
 * @param errmsg_len Maximum length of error message buffer.
 * @return 0 on success, 1-255 on error.
 */
int mpas_dycore_cpp_finalize(char* errmsg, int errmsg_len);

#ifdef __cplusplus
}
#endif

#endif /* MPAS_DYCORE_API_H */
