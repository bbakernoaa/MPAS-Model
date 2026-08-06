#pragma once

/// @file workspace.hpp
/// @brief SRK3 workspace allocation for the MPAS dynamical core.
///
/// Defines the SRK3Workspace struct that holds all intermediate storage
/// vectors required during the split-explicit Runge-Kutta 3rd-order (SRK3)
/// time integration loop. Allocated once during initialization and reused
/// across timesteps.

#include <mpas_dycore/types.hpp>

#include <vector>

namespace mpas::dycore {

/// @brief Workspace holding all intermediate arrays for SRK3 integration.
///
/// Contains storage vectors for perturbation fields, tendencies,
/// accumulated fluxes, diagnostics, vertical implicit coefficients,
/// moist coefficients, and state save arrays. Allocated once during
/// initialization, sized to mesh dimensions.
struct SRK3Workspace {

    // --- Perturbation fields (acoustic sub-stepping) ---
    std::vector<real_type> ru_p_storage;          ///< (nVertLevels, nEdges)
    std::vector<real_type> rw_p_storage;          ///< (nVertLevels+1, nCells)
    std::vector<real_type> rtheta_pp_storage;     ///< (nVertLevels, nCells)
    std::vector<real_type> rho_pp_storage;        ///< (nVertLevels, nCells)
    std::vector<real_type> rtheta_pp_old_storage; ///< (nVertLevels, nCells)

    // --- Accumulated flux averages ---
    std::vector<real_type> ruAvg_storage;         ///< (nVertLevels, nEdges)
    std::vector<real_type> wwAvg_storage;         ///< (nVertLevels+1, nCells)

    // --- Tendency arrays ---
    std::vector<real_type> tend_u_storage;        ///< (nVertLevels, nEdges)
    std::vector<real_type> tend_theta_storage;    ///< (nVertLevels, nCells)
    std::vector<real_type> tend_w_storage;        ///< (nVertLevels+1, nCells)
    std::vector<real_type> tend_rho_storage;      ///< (nVertLevels, nCells)

    // --- Diagnostic workspaces ---
    std::vector<real_type> ke_storage;            ///< (nVertLevels, nCells)
    std::vector<real_type> vorticity_storage;     ///< (nVertLevels, nVertices)
    std::vector<real_type> divergence_storage;    ///< (nVertLevels, nCells)
    std::vector<real_type> pv_edge_storage;       ///< (nVertLevels, nEdges)
    std::vector<real_type> rho_edge_storage;      ///< (nVertLevels, nEdges)

    // --- Vertical implicit coefficients ---
    std::vector<real_type> cofwr_storage;         ///< (nVertLevels, nCells)
    std::vector<real_type> cofwz_storage;         ///< (nVertLevels, nCells)
    std::vector<real_type> coftz_storage;         ///< (nVertLevels+1, nCells)
    std::vector<real_type> cofwt_storage;         ///< (nVertLevels, nCells)
    std::vector<real_type> a_tri_storage;         ///< (nVertLevels, nCells)
    std::vector<real_type> alpha_tri_storage;     ///< (nVertLevels, nCells)
    std::vector<real_type> gamma_tri_storage;     ///< (nVertLevels, nCells)
    std::vector<real_type> cofrz_storage;         ///< (nVertLevels)

    // --- Moist coefficients ---
    std::vector<real_type> cqw_storage;           ///< (nVertLevels+1, nCells)
    std::vector<real_type> cqu_storage;           ///< (nVertLevels, nEdges)

    // --- Additional workspace ---
    std::vector<real_type> exner_storage;         ///< (nVertLevels, nCells) Exner function
    std::vector<real_type> pressure_storage;      ///< (nVertLevels, nCells)
    std::vector<real_type> rtheta_flux_storage;   ///< (nVertLevels, nEdges)
    std::vector<real_type> pp_storage;            ///< (nVertLevels, nCells) perturbation pressure
    std::vector<real_type> dpdz_storage;          ///< (nVertLevels, nCells) buoyancy
    std::vector<real_type> zxu_storage;           ///< (nVertLevels, nEdges) terrain metric on edges
    std::vector<real_type> dss_storage;           ///< (nVertLevels, nCells) Rayleigh damping coef
    std::vector<real_type> qtot_storage;          ///< (nVertLevels, nCells) total moisture

    // --- State save arrays for recovery ---
    std::vector<real_type> u_save_storage;        ///< (nVertLevels, nEdges)
    std::vector<real_type> rho_zz_save_storage;   ///< (nVertLevels, nCells)
    std::vector<real_type> theta_m_save_storage;  ///< (nVertLevels, nCells)
    std::vector<real_type> w_save_storage;        ///< (nVertLevels+1, nCells)
    std::vector<real_type> rw_save_storage;       ///< (nVertLevels+1, nCells)
    std::vector<real_type> rho_zz_old_storage;    ///< (nVertLevels, nCells) for scalar transport

    // --- Perturbation state variables (Fortran formulation) ---
    std::vector<real_type> rho_p_storage;           ///< (nVertLevels, nCells) density perturbation
    std::vector<real_type> rtheta_p_storage;        ///< (nVertLevels, nCells) rho*theta perturbation
    std::vector<real_type> ru_storage;              ///< (nVertLevels, nEdges) density-weighted horizontal momentum
    std::vector<real_type> rw_storage;              ///< (nVertLevels+1, nCells) density-weighted vertical momentum

    // --- Perturbation state saves (saved ONCE before RK loop) ---
    std::vector<real_type> rho_p_save_storage;      ///< (nVertLevels, nCells)
    std::vector<real_type> rtheta_p_save_storage;   ///< (nVertLevels, nCells)
    std::vector<real_type> ru_save_storage;         ///< (nVertLevels, nEdges)
    // Note: rw_save_storage already exists above

    // --- Diabatic tendency ---
    std::vector<real_type> rt_diabatic_tend_storage; ///< (nVertLevels, nCells)

    // --- Exner function base state ---
    std::vector<real_type> exner_base_storage;      ///< (nVertLevels, nCells)

    // --- Connectivity conversion cache (real_type copies for kernel interface) ---
    std::vector<real_type> cellsOnEdge_real_storage;    ///< (2 * nEdges) cellsOnEdge as real_type
    std::vector<real_type> edgesOnCell_real_storage;    ///< (maxEdges * nCells) edgesOnCell as real_type
    std::vector<real_type> nEdgesOnCell_real_storage;   ///< (nCells) nEdgesOnCell as real_type

    // --- Mesh dimensions (stored for validation) ---
    index_type nCells_ = 0;      ///< Number of cells in the local partition.
    index_type nEdges_ = 0;      ///< Number of edges in the local partition.
    index_type nVertices_ = 0;   ///< Number of vertices in the local partition.
    index_type nVertLevels_ = 0; ///< Number of vertical levels.

    /// @brief Whether workspace has been allocated.
    bool allocated_ = false;

    /// @brief Allocate all workspace arrays given mesh dimensions.
    ///
    /// Sizes each storage vector according to the field dimensions specified
    /// in the design. All storage is zero-initialized. This function is
    /// idempotent — calling it again re-allocates with the new dimensions.
    ///
    /// @param nCells Number of cells in local partition (must be > 0).
    /// @param nEdges Number of edges in local partition (must be > 0).
    /// @param nVertices Number of vertices in local partition (must be > 0).
    /// @param nVertLevels Number of vertical levels (must be > 0).
    void allocate(index_type nCells, index_type nEdges,
                  index_type nVertices, index_type nVertLevels);

    /// @brief Query whether workspace memory has been allocated.
    /// @return true if allocate() has been called successfully.
    [[nodiscard]] bool allocated() const noexcept { return allocated_; }
};

} // namespace mpas::dycore
