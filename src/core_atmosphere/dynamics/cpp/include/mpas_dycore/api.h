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

/**
 * @brief Set mesh geometry, metric, base-state, and stencil data.
 *
 * Must be called after a successful mpas_dycore_cpp_init() and before timestep
 * execution. Copies all geometry arrays into owned storage inside DycoreState.
 *
 * @param areaCell           Cell areas (nCells).
 * @param invAreaCell        Inverse cell areas (nCells).
 * @param dvEdge             Edge lengths dvEdge (nEdges).
 * @param dcEdge             Edge lengths dcEdge (nEdges).
 * @param invDcEdge          Inverse dcEdge (nEdges).
 * @param rdzw               Vertical metric rdzw (nVertLevels).
 * @param rdzu               Vertical metric rdzu (nVertLevels).
 * @param fzm                Vertical interpolation weight fzm (nVertLevels).
 * @param fzp                Vertical interpolation weight fzp (nVertLevels).
 * @param etp                Vertical extrapolation weight etp (nVertLevels).
 * @param etm                Vertical extrapolation weight etm (nVertLevels).
 * @param ewp                Vertical weight ewp (nVertLevels+1).
 * @param ewm                Vertical weight ewm (nVertLevels+1).
 * @param zz                 Terrain metric dz/dzeta (nVertLevels x nCells).
 * @param rb                 Base-state density (nVertLevels x nCells).
 * @param rtb                Base-state rho*theta (nVertLevels x nCells).
 * @param pb                 Base-state pressure (nVertLevels x nCells).
 * @param edgesOnCell_sign   Edge orientation signs (maxEdges x nCells).
 * @param specZoneMaskEdge   Specified zone edge mask (nEdges).
 * @param specZoneMaskCell   Specified zone cell mask (nCells).
 * @param weightsOnEdge      TRiSK reconstruction weights (nEdges x maxEdges2).
 * @param nEdgesOnEdge       Edge neighbor counts (nEdges).
 * @param edgesOnEdge        Edge-on-edge connectivity (nEdges x maxEdges2).
 * @param advCellsForEdge    Advection cell stencils (nEdges x maxAdvCells).
 * @param nAdvCellsForEdge   Advection cell counts (nEdges).
 * @param adv_coefs          Advection coefficients (maxAdvCells x nEdges).
 * @param adv_coefs_3rd      3rd-order advection coefficients (maxAdvCells x nEdges).
 * @param fVertex            Coriolis parameter at vertices (nVertices).
 * @param areaTriangle       Triangle areas at vertices (nVertices).
 * @param zb_cell_ptr        Terrain slope metric ((nVertLevels+1) x maxEdges x nCells), or NULL.
 * @param zb3_cell_ptr       3rd-order terrain correction ((nVertLevels+1) x maxEdges x nCells), or NULL.
 * @param rho_base_ptr       Base-state density for perturbation formulation (nVertLevels x nCells), or NULL.
 * @param rtheta_base_ptr    Base-state rho*theta for perturbation formulation (nVertLevels x nCells), or NULL.
 * @param exner_base_ptr     Base-state Exner function (nVertLevels x nCells), or NULL.
 * @param cf1                Vertical extrapolation coefficient 1.
 * @param cf2                Vertical extrapolation coefficient 2.
 * @param cf3                Vertical extrapolation coefficient 3.
 * @param nEdgesOnCell_ptr   Per-cell edge counts for terrain correction (nCells), or NULL.
 * @param nCells             Number of cells in local partition.
 * @param nEdges             Number of edges in local partition.
 * @param nVertices          Number of vertices in local partition.
 * @param nVertLevels        Number of vertical levels.
 * @param maxEdges           Maximum edges per cell (for edgesOnCell_sign).
 * @param maxEdges2          Maximum edges for TRiSK reconstruction.
 * @param maxAdvCells        Maximum advection stencil cells per edge.
 * @param errmsg             Buffer for error message output.
 * @param errmsg_len         Maximum length of error message buffer.
 * @return 0 on success, 1-255 on error.
 */
int mpas_dycore_cpp_set_geometry(
    /* Cell geometry */
    const double* areaCell, const double* invAreaCell,
    /* Edge geometry */
    const double* dvEdge, const double* dcEdge, const double* invDcEdge,
    /* Vertical metrics */
    const double* rdzw, const double* rdzu,
    const double* fzm, const double* fzp,
    const double* etp, const double* etm,
    const double* ewp, const double* ewm,
    /* Terrain metric */
    const double* zz,
    /* Base-state profiles */
    const double* rb, const double* rtb, const double* pb,
    /* Edge orientation signs */
    const double* edgesOnCell_sign,
    /* Specified zone masks */
    const double* specZoneMaskEdge, const double* specZoneMaskCell,
    /* Edge reconstruction data (TRiSK) */
    const double* weightsOnEdge,
    const int* nEdgesOnEdge,
    const int* edgesOnEdge,
    /* Advection stencils */
    const int* advCellsForEdge,
    const int* nAdvCellsForEdge,
    const double* adv_coefs,
    const double* adv_coefs_3rd,
    /* Vertex geometry */
    const double* fVertex, const double* areaTriangle,
    /* Terrain correction arrays for w recovery */
    const double* zb_cell_ptr, const double* zb3_cell_ptr,
    /* Base-state profiles for perturbation formulation */
    const double* rho_base_ptr, const double* rtheta_base_ptr, const double* exner_base_ptr,
    /* Vertical extrapolation coefficients */
    double cf1, double cf2, double cf3,
    /* Per-cell edge count for terrain correction */
    const int* nEdgesOnCell_ptr,
    /* Dimension metadata */
    int nCells, int nEdges, int nVertices, int nVertLevels,
    int maxEdges, int maxEdges2, int maxAdvCells,
    /* Error output */
    char* errmsg, int errmsg_len
);

/* ======================================================================
 * Callback type definitions for hybrid Fortran/C++ execution mode.
 *
 * These function pointer types define the signatures for Fortran callback
 * routines that the C++ SRK3 integrator can invoke for kernels configured
 * for Fortran delegation. Each callback receives workspace pointers directly
 * (no copy) and operates on the same memory as the C++ dycore.
 *
 * Requirements: 8.1, 8.3, 8.6
 * ====================================================================== */

/**
 * @brief Callback for computing dynamic tendencies.
 *
 * Invoked at each Runge-Kutta stage to compute the tendencies for horizontal
 * velocity (u), potential temperature (theta_m), dry density (rho_zz), and
 * vertical velocity (w).
 *
 * @param u            Horizontal velocity field (nVertLevels x nEdges), read.
 * @param theta_m      Potential temperature (nVertLevels x nCells), read.
 * @param rho_zz       Dry air density (nVertLevels x nCells), read.
 * @param w            Vertical velocity ((nVertLevels+1) x nCells), read.
 * @param tend_u       Tendency for u (nVertLevels x nEdges), write.
 * @param tend_theta   Tendency for theta (nVertLevels x nCells), write.
 * @param tend_rho     Tendency for rho_zz (nVertLevels x nCells), write.
 * @param tend_w       Tendency for w ((nVertLevels+1) x nCells), write.
 * @param nVertLevels  Number of vertical levels.
 * @param nCells       Number of cells.
 * @param nEdges       Number of edges.
 * @param rk_step      Current Runge-Kutta step (0, 1, or 2).
 * @param dt_rk        Timestep for this RK stage (seconds).
 */
typedef void (*mpas_compute_dyn_tend_cb_t)(
    double* u, double* theta_m, double* rho_zz, double* w,
    double* tend_u, double* tend_theta, double* tend_rho, double* tend_w,
    int nVertLevels, int nCells, int nEdges, int rk_step, double dt_rk);

/**
 * @brief Callback for advancing one acoustic sub-step.
 *
 * Invoked within each Runge-Kutta stage for the split-explicit acoustic
 * sub-stepping loop.
 *
 * @param ru_p         Perturbation horizontal momentum flux, read-write.
 * @param rw_p         Perturbation vertical momentum flux, read-write.
 * @param rtheta_pp    Double-perturbation rho*theta, read-write.
 * @param rho_pp       Double-perturbation density, read-write.
 * @param nVertLevels  Number of vertical levels.
 * @param nCells       Number of cells.
 * @param nEdges       Number of edges.
 * @param substep      Current acoustic sub-step index (0-based).
 * @param dts          Acoustic sub-step timestep (seconds).
 */
typedef void (*mpas_advance_acoustic_step_cb_t)(
    double* ru_p, double* rw_p, double* rtheta_pp, double* rho_pp,
    int nVertLevels, int nCells, int nEdges, int substep, double dts);

/**
 * @brief Callback for monotonic scalar transport.
 *
 * Invoked to advance scalar tracers using flux-corrected transport with
 * monotonicity enforcement.
 *
 * @param scalars      Scalar tracer array (nScalars x nVertLevels x nCells), read-write.
 * @param ruAvg        Time-averaged horizontal mass flux (nVertLevels x nEdges), read.
 * @param wwAvg        Time-averaged vertical mass flux ((nVertLevels+1) x nCells), read.
 * @param nScalars     Number of scalar species.
 * @param nVertLevels  Number of vertical levels.
 * @param nCells       Number of cells.
 * @param nEdges       Number of edges.
 * @param dt           Full dynamics timestep (seconds).
 */
typedef void (*mpas_advance_scalars_mono_cb_t)(
    double* scalars, double* ruAvg, double* wwAvg,
    int nScalars, int nVertLevels, int nCells, int nEdges, double dt);

/**
 * @brief Callback for halo exchange of a single field.
 *
 * Invoked at synchronization points where ghost-cell data must be updated
 * across MPI partitions. The field pointer includes the full halo region
 * (owned + halo cells/edges) so that the Fortran side can fill halo values
 * in place.
 *
 * @param field        Pointer to the field data (full extent including halo), read-write.
 * @param field_id     Integer identifier for the field (used by Fortran to select the exchange).
 * @param nVertLevels  Number of vertical levels (or vertical extent of the field).
 * @param nElements    Number of elements in the horizontal dimension (cells or edges, including halo).
 */
typedef void (*mpas_halo_exchange_cb_t)(
    double* field, int field_id, int nVertLevels, int nElements);

/**
 * @brief Register Fortran callback functions for hybrid execution mode.
 *
 * Must be called after a successful mpas_dycore_cpp_init() and before the
 * first mpas_dycore_cpp_timestep(). Stores callback function pointers in
 * the internal DycoreState and marks each kernel as "Fortran-delegated"
 * when a non-NULL pointer is provided.
 *
 * Individual callbacks may be set to NULL to indicate that the C++
 * implementation should be used for that kernel. This allows incremental
 * migration — initially all kernels delegate to Fortran, then one by one
 * the C++ implementations replace them.
 *
 * The internal DycoreState maintains per-kernel flags (use_cpp vs.
 * use_callback) that control dispatch during timestep execution.
 *
 * @param compute_dyn_tend_cb       Callback for tendency computation, or NULL for C++.
 * @param advance_acoustic_step_cb  Callback for acoustic sub-stepping, or NULL for C++.
 * @param advance_scalars_mono_cb   Callback for scalar transport, or NULL for C++.
 * @param halo_exchange_cb          Callback for halo exchange, or NULL for C++.
 * @return 0 on success, 1 if dycore is not initialized.
 *
 * Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
 */
int mpas_dycore_cpp_set_callbacks(
    mpas_compute_dyn_tend_cb_t compute_dyn_tend_cb,
    mpas_advance_acoustic_step_cb_t advance_acoustic_step_cb,
    mpas_advance_scalars_mono_cb_t advance_scalars_mono_cb,
    mpas_halo_exchange_cb_t halo_exchange_cb
);

#ifdef __cplusplus
}
#endif

#endif /* MPAS_DYCORE_API_H */
