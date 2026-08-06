/// @file workspace.cpp
/// @brief Implementation of SRK3Workspace allocation for the MPAS dynamical core.

#include <mpas_dycore/workspace.hpp>

#include <stdexcept>
#include <string>

namespace mpas::dycore {

void SRK3Workspace::allocate(index_type nCells, index_type nEdges,
                             index_type nVertices, index_type nVertLevels) {
    // Validate all dimension arguments are > 0.
    if (nCells <= 0) {
        throw std::invalid_argument(
            "SRK3Workspace::allocate: nCells must be > 0, got " +
            std::to_string(nCells));
    }
    if (nEdges <= 0) {
        throw std::invalid_argument(
            "SRK3Workspace::allocate: nEdges must be > 0, got " +
            std::to_string(nEdges));
    }
    if (nVertices <= 0) {
        throw std::invalid_argument(
            "SRK3Workspace::allocate: nVertices must be > 0, got " +
            std::to_string(nVertices));
    }
    if (nVertLevels <= 0) {
        throw std::invalid_argument(
            "SRK3Workspace::allocate: nVertLevels must be > 0, got " +
            std::to_string(nVertLevels));
    }

    // Store dimensions.
    nCells_ = nCells;
    nEdges_ = nEdges;
    nVertices_ = nVertices;
    nVertLevels_ = nVertLevels;

    // Compute common size products.
    const auto nVL_nC =
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells);
    const auto nVL_nE =
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges);
    const auto nVLp1_nC =
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells);
    const auto nVL_nV =
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nVertices);
    const auto nVL =
        static_cast<std::size_t>(nVertLevels);

    // --- Perturbation fields (acoustic sub-stepping) ---
    ru_p_storage.assign(nVL_nE, 0.0);
    rw_p_storage.assign(nVLp1_nC, 0.0);
    rtheta_pp_storage.assign(nVL_nC, 0.0);
    rho_pp_storage.assign(nVL_nC, 0.0);
    rtheta_pp_old_storage.assign(nVL_nC, 0.0);

    // --- Accumulated flux averages ---
    ruAvg_storage.assign(nVL_nE, 0.0);
    wwAvg_storage.assign(nVLp1_nC, 0.0);

    // --- Tendency arrays ---
    tend_u_storage.assign(nVL_nE, 0.0);
    tend_theta_storage.assign(nVL_nC, 0.0);
    tend_w_storage.assign(nVLp1_nC, 0.0);
    tend_rho_storage.assign(nVL_nC, 0.0);

    // --- State save arrays for recovery ---
    u_save_storage.assign(nVL_nE, 0.0);
    rho_zz_save_storage.assign(nVL_nC, 0.0);
    theta_m_save_storage.assign(nVL_nC, 0.0);
    w_save_storage.assign(nVLp1_nC, 0.0);
    rw_save_storage.assign(nVLp1_nC, 0.0);
    rho_zz_old_storage.assign(nVL_nC, 0.0);

    // --- Perturbation state variables (Fortran formulation) ---
    rho_p_storage.assign(nVL_nC, 0.0);
    rtheta_p_storage.assign(nVL_nC, 0.0);
    ru_storage.assign(nVL_nE, 0.0);
    rw_storage.assign(nVLp1_nC, 0.0);

    // --- Perturbation state saves (saved ONCE before RK loop) ---
    rho_p_save_storage.assign(nVL_nC, 0.0);
    rtheta_p_save_storage.assign(nVL_nC, 0.0);
    ru_save_storage.assign(nVL_nE, 0.0);
    // Note: rw_save_storage already allocated above

    // --- Diabatic tendency ---
    rt_diabatic_tend_storage.assign(nVL_nC, 0.0);

    // --- Exner function base state ---
    exner_base_storage.assign(nVL_nC, 0.0);

    // --- Diagnostic workspaces ---
    ke_storage.assign(nVL_nC, 0.0);
    vorticity_storage.assign(nVL_nV, 0.0);
    divergence_storage.assign(nVL_nC, 0.0);
    pv_edge_storage.assign(nVL_nE, 0.0);
    rho_edge_storage.assign(nVL_nE, 0.0);

    // --- Vertical implicit coefficients ---
    cofwr_storage.assign(nVL_nC, 0.0);
    cofwz_storage.assign(nVL_nC, 0.0);
    coftz_storage.assign(nVLp1_nC, 0.0);
    cofwt_storage.assign(nVL_nC, 0.0);
    a_tri_storage.assign(nVL_nC, 0.0);
    alpha_tri_storage.assign(nVL_nC, 0.0);
    gamma_tri_storage.assign(nVL_nC, 0.0);
    cofrz_storage.assign(nVL, 0.0);

    // --- Moist coefficients ---
    cqw_storage.assign(nVLp1_nC, 0.0);
    cqu_storage.assign(nVL_nE, 0.0);

    // --- Additional workspace ---
    exner_storage.assign(nVL_nC, 0.0);
    pressure_storage.assign(nVL_nC, 0.0);
    rtheta_flux_storage.assign(nVL_nE, 0.0);
    pp_storage.assign(nVL_nC, 0.0);
    dpdz_storage.assign(nVL_nC, 0.0);
    zxu_storage.assign(nVL_nE, 0.0);
    dss_storage.assign(nVL_nC, 0.0);
    qtot_storage.assign(nVL_nC, 0.0);

    allocated_ = true;
}

} // namespace mpas::dycore
