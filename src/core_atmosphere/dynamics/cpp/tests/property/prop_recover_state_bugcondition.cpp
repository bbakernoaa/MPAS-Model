/// @file prop_recover_state_bugcondition.cpp
/// @brief Bug condition exploration test: Perturbation Recovery Matches Fortran
///
/// **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.7**
///
/// Property 1: Expected Behavior - Perturbation Recovery Matches Fortran
///
/// This test validates that the fixed recover_state_perturbation kernel
/// produces results matching the Fortran reference algorithm.
/// The 4 cases demonstrate:
///   Case 1: Density formula (perturbation formulation)
///   Case 2: Velocity formula (rho_zz cell-average)
///   Case 3: Terrain correction for w
///   Case 4: Diabatic tendency on rk_step==3

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/recover_state.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;

namespace {

/// Absolute tolerance for comparing C++ result to Fortran reference.
constexpr real_type kTol = 1.0e-12;

/// Check if two values match within tolerance.
bool matches(real_type actual, real_type expected) {
    return std::abs(actual - expected) <= kTol;
}

} // anonymous namespace

// ============================================================================
// Case 1: Density Formula — Perturbation Formulation
//
// Fortran: rho_zz = rho_p_save + rho_pp + rho_base
// With rho_p_save = 0 (initial state = base state):
//   rho_zz = 0 + 0.01 + 1.0 = 1.01
// ============================================================================

TEST(RecoverStateBugCondition, Case1_DensityFormulaDivergence) {
    constexpr index_type nCells = 1;
    constexpr index_type nEdges = 1;
    constexpr index_type nVertLevels = 3;
    constexpr index_type maxEdges = 1;

    // --- Fortran reference parameters ---
    const real_type rho_base_val = 1.0;
    const real_type rho_pp_val = 0.01;
    const index_type ns = 6;
    const index_type rk_step = 1; // non-final step
    const real_type dt = 10.0;

    // Perturbation save: rho_p_save = rho_zz_initial - rho_base = 0
    const real_type rho_p_save_val = 0.0;
    // Fortran result: rho_zz = rho_p_save + rho_pp + rho_base = 1.01
    const real_type fortran_rho_zz = rho_p_save_val + rho_pp_val + rho_base_val;

    // --- Allocate arrays for recover_state_perturbation ---
    // Output fields
    std::vector<real_type> u_out(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rho_zz_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> theta_m_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> w_out((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> exner_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> pressure_p_out(nVertLevels * nCells, 0.0);

    // Perturbation saves
    std::vector<real_type> rho_p_save_data(nVertLevels * nCells, rho_p_save_val);
    std::vector<real_type> rtheta_p_save_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> ru_save_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rw_save_data((nVertLevels + 1) * nCells, 0.0);

    // Acoustic perturbations
    std::vector<real_type> rho_pp_data(nVertLevels * nCells, rho_pp_val);
    std::vector<real_type> rtheta_pp_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> ru_p_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rw_p_data((nVertLevels + 1) * nCells, 0.0);

    // Accumulated flux averages (in/out)
    std::vector<real_type> ruAvg_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> wwAvg_data((nVertLevels + 1) * nCells, 0.0);

    // Base-state profiles
    std::vector<real_type> rho_base_data(nVertLevels * nCells, rho_base_val);
    std::vector<real_type> rtheta_base_data(nVertLevels * nCells, 300.0);
    std::vector<real_type> exner_base_data(nVertLevels * nCells, 1.0);

    // Diabatic tendency (unused for rk_step != 3)
    std::vector<real_type> rt_diabatic_tend_data(nVertLevels * nCells, 0.0);

    // Terrain arrays (flat terrain for this case)
    std::vector<real_type> zb_cell_data((nVertLevels + 1) * maxEdges * nCells, 0.0);
    std::vector<real_type> zb3_cell_data((nVertLevels + 1) * maxEdges * nCells, 0.0);

    // Connectivity
    std::vector<real_type> cellsOnEdge_data(2 * nEdges, 0.0); // both cells = 0
    std::vector<real_type> edgesOnCell_data(maxEdges * nCells, 0.0); // edge 0
    std::vector<real_type> nEdgesOnCell_data(nCells, static_cast<real_type>(maxEdges));
    std::vector<real_type> edgesOnCell_sign_data(maxEdges * nCells, 1.0);

    // Vertical metrics
    std::vector<real_type> zz_data(nVertLevels * nCells, 1.0);
    std::vector<real_type> fzm_data(nVertLevels, 0.5);
    std::vector<real_type> fzp_data(nVertLevels, 0.5);

    // Vertical extrapolation coefficients
    const real_type cf1 = 1.0, cf2 = 0.0, cf3 = 0.0;

    // --- Create mdspan views ---
    Field2D<default_layout, unchecked_accessor> u(u_out.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> rho_zz(rho_zz_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> theta_m(theta_m_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> w(w_out.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> exner(exner_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> pressure_p(pressure_p_out.data(), nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> rho_p_save(rho_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_p_save(rtheta_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_save(ru_save_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_save(rw_save_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_pp(rho_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_pp(rtheta_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_p(ru_p_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_p(rw_p_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> ruAvg(ruAvg_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> wwAvg(wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_base(rho_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_base(rtheta_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> exner_base(exner_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rt_diabatic_tend(rt_diabatic_tend_data.data(), nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> zb_cell(zb_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> zb3_cell(zb3_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> cellsOnEdge(cellsOnEdge_data.data(), 2, nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell(edgesOnCell_data.data(), maxEdges, nCells);
    ConstSpan1D nEdgesOnCell_span(nEdgesOnCell_data.data(), nCells);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign(edgesOnCell_sign_data.data(), maxEdges, nCells);
    ConstField2D<default_layout, unchecked_accessor> zz(zz_data.data(), nVertLevels, nCells);
    ConstSpan1D fzm(fzm_data.data(), nVertLevels);
    ConstSpan1D fzp(fzp_data.data(), nVertLevels);

    // --- Call recover_state_perturbation ---
    recover_state_perturbation<default_layout>(
        SerialPolicy{},
        u, rho_zz, theta_m, w, exner, pressure_p,
        rho_p_save, rtheta_p_save, ru_save, rw_save,
        rho_pp, rtheta_pp, ru_p, rw_p,
        ruAvg, wwAvg,
        rho_base, rtheta_base, exner_base,
        rt_diabatic_tend,
        zb_cell, zb3_cell,
        cellsOnEdge, edgesOnCell, nEdgesOnCell_span, edgesOnCell_sign,
        zz, fzm, fzp,
        cf1, cf2, cf3,
        rk_step, dt, ns,
        nCells, nEdges, nVertLevels, maxEdges);

    // Verify: rho_zz = rho_p_save + rho_pp + rho_base = 0 + 0.01 + 1.0 = 1.01
    for (index_type k = 0; k < nVertLevels; ++k) {
        EXPECT_TRUE(matches(rho_zz_out[k], fortran_rho_zz))
            << "Case 1 (Density): k=" << k
            << " C++ rho_zz=" << rho_zz_out[k]
            << " Fortran rho_zz=" << fortran_rho_zz
            << " diff=" << std::abs(rho_zz_out[k] - fortran_rho_zz);
    }
}

// ============================================================================
// Case 2: Velocity Formula — Cell-Average Density
//
// Fortran: u = 2 * ru / (rho_zz[cell1] + rho_zz[cell2])
//          where ru = ru_save + ru_p
// ============================================================================

TEST(RecoverStateBugCondition, Case2_VelocityFormulaDivergence) {
    constexpr index_type nCells = 2;
    constexpr index_type nEdges = 1;
    constexpr index_type nVertLevels = 3;
    constexpr index_type maxEdges = 1;

    const index_type ns = 6;
    const index_type rk_step = 1;
    const real_type dt = 10.0;

    // Two cells with different base densities
    const real_type rho_base1 = 1.0;
    const real_type rho_base2 = 1.1;
    const real_type rho_pp_val = 0.01;

    // Fortran rho_zz (with rho_p_save = 0):
    const real_type fortran_rho_zz1 = 0.0 + rho_pp_val + rho_base1; // 1.01
    const real_type fortran_rho_zz2 = 0.0 + rho_pp_val + rho_base2; // 1.11

    // ru_save = 0.5 * u_initial * (rho_base1 + rho_base2) for this edge
    const real_type u_initial = 5.0;
    const real_type ru_save_val = 0.5 * u_initial * (rho_base1 + rho_base2); // 5.25
    const real_type ru_p_val = 0.5; // acoustic velocity perturbation

    // Fortran: u = 2 * (ru_save + ru_p) / (rho_zz[cell1] + rho_zz[cell2])
    const real_type fortran_u = 2.0 * (ru_save_val + ru_p_val)
                              / (fortran_rho_zz1 + fortran_rho_zz2);

    // --- Allocate arrays ---
    std::vector<real_type> u_out(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rho_zz_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> theta_m_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> w_out((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> exner_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> pressure_p_out(nVertLevels * nCells, 0.0);

    // Perturbation saves: rho_p_save = 0 for both cells
    std::vector<real_type> rho_p_save_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> rtheta_p_save_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> ru_save_data(nVertLevels * nEdges, ru_save_val);
    std::vector<real_type> rw_save_data((nVertLevels + 1) * nCells, 0.0);

    // Acoustic perturbations
    std::vector<real_type> rho_pp_data(nVertLevels * nCells, rho_pp_val);
    std::vector<real_type> rtheta_pp_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> ru_p_data(nVertLevels * nEdges, ru_p_val);
    std::vector<real_type> rw_p_data((nVertLevels + 1) * nCells, 0.0);

    // Flux averages
    std::vector<real_type> ruAvg_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> wwAvg_data((nVertLevels + 1) * nCells, 0.0);

    // Base-state: cell 0 = rho_base1, cell 1 = rho_base2 (column-major)
    std::vector<real_type> rho_base_data(nVertLevels * nCells);
    for (index_type k = 0; k < nVertLevels; ++k) {
        rho_base_data[k + 0 * nVertLevels] = rho_base1;
        rho_base_data[k + 1 * nVertLevels] = rho_base2;
    }
    std::vector<real_type> rtheta_base_data(nVertLevels * nCells, 300.0);
    std::vector<real_type> exner_base_data(nVertLevels * nCells, 1.0);
    std::vector<real_type> rt_diabatic_tend_data(nVertLevels * nCells, 0.0);

    // Terrain (flat)
    std::vector<real_type> zb_cell_data((nVertLevels + 1) * maxEdges * nCells, 0.0);
    std::vector<real_type> zb3_cell_data((nVertLevels + 1) * maxEdges * nCells, 0.0);

    // Connectivity: cellsOnEdge = {0, 1} (edge connects cell 0 and cell 1)
    std::vector<real_type> cellsOnEdge_data = {0.0, 1.0}; // (2, nEdges=1) col-major
    // edgesOnCell: each cell has 1 edge (edge 0)
    std::vector<real_type> edgesOnCell_data(maxEdges * nCells, 0.0);
    std::vector<real_type> nEdgesOnCell_data(nCells, static_cast<real_type>(maxEdges));
    std::vector<real_type> edgesOnCell_sign_data(maxEdges * nCells, 1.0);

    // Vertical metrics
    std::vector<real_type> zz_data(nVertLevels * nCells, 1.0);
    std::vector<real_type> fzm_data(nVertLevels, 0.5);
    std::vector<real_type> fzp_data(nVertLevels, 0.5);
    const real_type cf1 = 1.0, cf2 = 0.0, cf3 = 0.0;

    // --- Create mdspan views ---
    Field2D<default_layout, unchecked_accessor> u_o(u_out.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> rho_zz_o(rho_zz_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> theta_m_o(theta_m_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> w_o(w_out.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> exner_o(exner_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> pressure_p_o(pressure_p_out.data(), nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> rho_p_save_v(rho_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_p_save_v(rtheta_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_save_v(ru_save_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_save_v(rw_save_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_pp_v(rho_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_pp_v(rtheta_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_p_v(ru_p_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_p_v(rw_p_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> ruAvg_v(ruAvg_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> wwAvg_v(wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_base_v(rho_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_base_v(rtheta_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> exner_base_v(exner_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rt_diabatic_tend_v(rt_diabatic_tend_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> zb_cell_v(zb_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> zb3_cell_v(zb3_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> cellsOnEdge_v(cellsOnEdge_data.data(), 2, nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_v(edgesOnCell_data.data(), maxEdges, nCells);
    ConstSpan1D nEdgesOnCell_span(nEdgesOnCell_data.data(), nCells);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign_v(edgesOnCell_sign_data.data(), maxEdges, nCells);
    ConstField2D<default_layout, unchecked_accessor> zz_v(zz_data.data(), nVertLevels, nCells);
    ConstSpan1D fzm_s(fzm_data.data(), nVertLevels);
    ConstSpan1D fzp_s(fzp_data.data(), nVertLevels);

    // --- Call recover_state_perturbation ---
    recover_state_perturbation<default_layout>(
        SerialPolicy{},
        u_o, rho_zz_o, theta_m_o, w_o, exner_o, pressure_p_o,
        rho_p_save_v, rtheta_p_save_v, ru_save_v, rw_save_v,
        rho_pp_v, rtheta_pp_v, ru_p_v, rw_p_v,
        ruAvg_v, wwAvg_v,
        rho_base_v, rtheta_base_v, exner_base_v,
        rt_diabatic_tend_v,
        zb_cell_v, zb3_cell_v,
        cellsOnEdge_v, edgesOnCell_v, nEdgesOnCell_span, edgesOnCell_sign_v,
        zz_v, fzm_s, fzp_s,
        cf1, cf2, cf3,
        rk_step, dt, ns,
        nCells, nEdges, nVertLevels, maxEdges);

    // Verify: u = 2 * (ru_save + ru_p) / (rho_zz[cell1] + rho_zz[cell2])
    for (index_type k = 0; k < nVertLevels; ++k) {
        EXPECT_TRUE(matches(u_out[k], fortran_u))
            << "Case 2 (Velocity): k=" << k
            << " C++ u=" << u_out[k]
            << " Fortran u=" << fortran_u
            << " diff=" << std::abs(u_out[k] - fortran_u);
    }
}

// ============================================================================
// Case 3: Terrain w Correction
//
// Fortran: After Phase 1 w recovery (rw/metric), Phase 4 adds a terrain
//          flux-divergence correction looping over cell edges.
// With non-zero zb_cell and non-zero ru, the terrain correction is non-zero.
// ============================================================================

TEST(RecoverStateBugCondition, Case3_TerrainWCorrectionMissing) {
    constexpr index_type nCells = 1;
    constexpr index_type nEdges = 1;
    constexpr index_type nVertLevels = 4;
    constexpr index_type maxEdges = 1;

    const index_type ns = 6;
    const index_type rk_step = 1;
    const real_type dt = 10.0;

    // Terrain parameters
    const real_type zb_cell_val = 0.05;
    const real_type zb3_cell_val = 0.0;
    const real_type rho_base_val = 1.0;
    const real_type ru_p_val = 5.0; // non-zero momentum for terrain flux

    // fzm/fzp for interpolation
    const real_type fzm_val = 0.5;
    const real_type fzp_val = 0.5;

    // Phase 1: w = rw / (fzm*rho_zz*zz + fzp*rho_zz_below*zz_below)
    // With rw_save=0, rw_p=0 → rw=0, so Phase 1 w = 0 for all interior levels
    // Phase 4 terrain correction at k=1:
    //   flux = cf1*ru[0] + cf2*ru[1] + cf3*ru[2] (cf1=1,cf2=0,cf3=0 → flux = ru[0] = 5.0)
    //   w[1] += sign * (zb_cell + copysign(1,flux)*zb3) * flux
    //         / (fzm*rho_zz[1]*zz[1] + fzp*rho_zz[0]*zz[0])
    //   = 1.0 * (0.05 + 0.0) * 5.0 / (0.5*1.01*1.0 + 0.5*1.01*1.0)
    //   = 0.25 / 1.01
    // With rho_pp=0.01, rho_base=1.0 → rho_zz = 1.01
    const real_type rho_pp_val = 0.01;
    const real_type rho_zz_val = rho_base_val + rho_pp_val; // 1.01

    // Compute expected Fortran w at k=1 after terrain correction
    // Phase 4: flux = cf1*ru[0] = 1.0 * 5.0 = 5.0 (since ru = ru_save + ru_p = 0 + 5.0)
    // terrain_contrib = sign_edge * (zb_cell + copysign(1,flux)*zb3) * flux
    //                 = 1.0 * (0.05 + 0.0) * 5.0 = 0.25
    // Divide by (fzm*rho_zz[k]*zz[k] + fzp*rho_zz[k-1]*zz[k-1])
    //         = (0.5*1.01*1.0 + 0.5*1.01*1.0) = 1.01
    // Final terrain correction for w[1] = 0.25 / 1.01
    const real_type cf1 = 1.0, cf2 = 0.0, cf3 = 0.0;
    const real_type flux_k1 = cf1 * ru_p_val; // ru_save=0, ru_p=5.0, so ru[0]=5.0
    const real_type terrain_contrib = 1.0 * (zb_cell_val + std::copysign(1.0, flux_k1) * zb3_cell_val) * flux_k1;
    const real_type rho_denom = fzm_val * rho_zz_val * 1.0 + fzp_val * rho_zz_val * 1.0; // = 1.01
    // Phase 1 w[1] = 0 (rw=0), so final w[1] = 0 + terrain_contrib / rho_denom
    const real_type fortran_w_k1 = terrain_contrib / rho_denom;

    // --- Allocate arrays ---
    std::vector<real_type> u_out(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rho_zz_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> theta_m_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> w_out((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> exner_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> pressure_p_out(nVertLevels * nCells, 0.0);

    std::vector<real_type> rho_p_save_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> rtheta_p_save_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> ru_save_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rw_save_data((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> rho_pp_data(nVertLevels * nCells, rho_pp_val);
    std::vector<real_type> rtheta_pp_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> ru_p_data(nVertLevels * nEdges, ru_p_val);
    std::vector<real_type> rw_p_data((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> ruAvg_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> wwAvg_data((nVertLevels + 1) * nCells, 0.0);

    std::vector<real_type> rho_base_data(nVertLevels * nCells, rho_base_val);
    std::vector<real_type> rtheta_base_data(nVertLevels * nCells, 300.0);
    std::vector<real_type> exner_base_data(nVertLevels * nCells, 1.0);
    std::vector<real_type> rt_diabatic_tend_data(nVertLevels * nCells, 0.0);

    // Terrain: zb_cell non-zero at k=1 (the level we test)
    // zb_cell layout: (nVertLevels+1, maxEdges*nCells)
    // For 1 cell, 1 edge: col index = i*nCells + iCell = 0*1 + 0 = 0
    // We need zb_cell[k=1, col=0] = zb_cell_val
    std::vector<real_type> zb_cell_data((nVertLevels + 1) * maxEdges * nCells, 0.0);
    zb_cell_data[1] = zb_cell_val; // k=1, col=0 (column-major: index = k + col*(nVertLevels+1) = 1 + 0 = 1)
    std::vector<real_type> zb3_cell_data((nVertLevels + 1) * maxEdges * nCells, 0.0);

    // Connectivity: single cell, single edge
    std::vector<real_type> cellsOnEdge_data = {0.0, 0.0}; // both cells = 0 (self-edge for minimal mesh)
    std::vector<real_type> edgesOnCell_data(maxEdges * nCells, 0.0);
    std::vector<real_type> nEdgesOnCell_data(nCells, static_cast<real_type>(1));
    std::vector<real_type> edgesOnCell_sign_data(maxEdges * nCells, 1.0);

    std::vector<real_type> zz_data(nVertLevels * nCells, 1.0);
    std::vector<real_type> fzm_data(nVertLevels, fzm_val);
    std::vector<real_type> fzp_data(nVertLevels, fzp_val);

    // --- Create mdspan views ---
    Field2D<default_layout, unchecked_accessor> u_o(u_out.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> rho_zz_o(rho_zz_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> theta_m_o(theta_m_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> w_o(w_out.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> exner_o(exner_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> pressure_p_o(pressure_p_out.data(), nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> rho_p_save_v(rho_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_p_save_v(rtheta_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_save_v(ru_save_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_save_v(rw_save_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_pp_v(rho_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_pp_v(rtheta_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_p_v(ru_p_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_p_v(rw_p_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> ruAvg_v(ruAvg_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> wwAvg_v(wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_base_v(rho_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_base_v(rtheta_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> exner_base_v(exner_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rt_diabatic_tend_v(rt_diabatic_tend_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> zb_cell_v(zb_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> zb3_cell_v(zb3_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> cellsOnEdge_v(cellsOnEdge_data.data(), 2, nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_v(edgesOnCell_data.data(), maxEdges, nCells);
    ConstSpan1D nEdgesOnCell_span(nEdgesOnCell_data.data(), nCells);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign_v(edgesOnCell_sign_data.data(), maxEdges, nCells);
    ConstField2D<default_layout, unchecked_accessor> zz_v(zz_data.data(), nVertLevels, nCells);
    ConstSpan1D fzm_s(fzm_data.data(), nVertLevels);
    ConstSpan1D fzp_s(fzp_data.data(), nVertLevels);

    // --- Call recover_state_perturbation ---
    recover_state_perturbation<default_layout>(
        SerialPolicy{},
        u_o, rho_zz_o, theta_m_o, w_o, exner_o, pressure_p_o,
        rho_p_save_v, rtheta_p_save_v, ru_save_v, rw_save_v,
        rho_pp_v, rtheta_pp_v, ru_p_v, rw_p_v,
        ruAvg_v, wwAvg_v,
        rho_base_v, rtheta_base_v, exner_base_v,
        rt_diabatic_tend_v,
        zb_cell_v, zb3_cell_v,
        cellsOnEdge_v, edgesOnCell_v, nEdgesOnCell_span, edgesOnCell_sign_v,
        zz_v, fzm_s, fzp_s,
        cf1, cf2, cf3,
        rk_step, dt, ns,
        nCells, nEdges, nVertLevels, maxEdges);

    // Verify w at k=1 includes terrain correction
    // w_out layout: (nVertLevels+1, nCells) column-major → w[k=1, iCell=0] = w_out[1]
    const real_type cpp_w_k1 = w_out[1];

    EXPECT_TRUE(matches(cpp_w_k1, fortran_w_k1))
        << "Case 3 (Terrain w): k=1"
        << " C++ w=" << cpp_w_k1
        << " Fortran w=" << fortran_w_k1
        << " diff=" << std::abs(cpp_w_k1 - fortran_w_k1)
        << " (terrain correction zb_cell=" << zb_cell_val << ")";
}

// ============================================================================
// Case 4: Diabatic Tendency on rk_step==3
//
// Fortran (rk_step==3):
//   rtheta_p = rtheta_p_save + rtheta_pp - dt * rho_zz * rt_diabatic_tend
//   theta_m = (rtheta_p + rtheta_base) / rho_zz
// ============================================================================

TEST(RecoverStateBugCondition, Case4_DiabaticTendencyMissing) {
    constexpr index_type nCells = 1;
    constexpr index_type nEdges = 1;
    constexpr index_type nVertLevels = 3;
    constexpr index_type maxEdges = 1;

    const real_type dt = 720.0;
    const index_type ns = 6;
    const index_type rk_step = 3; // final RK step

    const real_type rho_base_val = 1.0;
    const real_type rtheta_base_val = 300.0;
    const real_type rho_pp_val = 0.0;
    const real_type rtheta_pp_val = 1.0;
    const real_type rt_diabatic_tend_val = 0.001;

    // Fortran density: rho_zz = rho_p_save + rho_pp + rho_base = 0 + 0 + 1.0 = 1.0
    const real_type fortran_rho_zz = 0.0 + rho_pp_val + rho_base_val;

    // Fortran theta on rk_step==3:
    // rtheta_p = rtheta_p_save + rtheta_pp - dt * rho_zz * rt_diabatic_tend
    //          = 0.0 + 1.0 - 720.0 * 1.0 * 0.001 = 0.28
    const real_type rtheta_p_save_val = 0.0;
    const real_type fortran_rtheta_p = rtheta_p_save_val + rtheta_pp_val
                                     - dt * fortran_rho_zz * rt_diabatic_tend_val;
    // theta_m = (rtheta_p + rtheta_base) / rho_zz = (0.28 + 300.0) / 1.0 = 300.28
    const real_type fortran_theta_m = (fortran_rtheta_p + rtheta_base_val) / fortran_rho_zz;

    // --- Allocate arrays ---
    std::vector<real_type> u_out(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rho_zz_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> theta_m_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> w_out((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> exner_out(nVertLevels * nCells, 0.0);
    std::vector<real_type> pressure_p_out(nVertLevels * nCells, 0.0);

    std::vector<real_type> rho_p_save_data(nVertLevels * nCells, 0.0);
    std::vector<real_type> rtheta_p_save_data(nVertLevels * nCells, rtheta_p_save_val);
    std::vector<real_type> ru_save_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rw_save_data((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> rho_pp_data(nVertLevels * nCells, rho_pp_val);
    std::vector<real_type> rtheta_pp_data(nVertLevels * nCells, rtheta_pp_val);
    std::vector<real_type> ru_p_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> rw_p_data((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> ruAvg_data(nVertLevels * nEdges, 0.0);
    std::vector<real_type> wwAvg_data((nVertLevels + 1) * nCells, 0.0);
    std::vector<real_type> rho_base_data(nVertLevels * nCells, rho_base_val);
    std::vector<real_type> rtheta_base_data(nVertLevels * nCells, rtheta_base_val);
    std::vector<real_type> exner_base_data(nVertLevels * nCells, 1.0);
    std::vector<real_type> rt_diabatic_tend_data(nVertLevels * nCells, rt_diabatic_tend_val);

    // Flat terrain
    std::vector<real_type> zb_cell_data((nVertLevels + 1) * maxEdges * nCells, 0.0);
    std::vector<real_type> zb3_cell_data((nVertLevels + 1) * maxEdges * nCells, 0.0);
    std::vector<real_type> cellsOnEdge_data = {0.0, 0.0};
    std::vector<real_type> edgesOnCell_data(maxEdges * nCells, 0.0);
    std::vector<real_type> nEdgesOnCell_data(nCells, static_cast<real_type>(maxEdges));
    std::vector<real_type> edgesOnCell_sign_data(maxEdges * nCells, 1.0);
    std::vector<real_type> zz_data(nVertLevels * nCells, 1.0);
    std::vector<real_type> fzm_data(nVertLevels, 0.5);
    std::vector<real_type> fzp_data(nVertLevels, 0.5);
    const real_type cf1 = 1.0, cf2 = 0.0, cf3 = 0.0;

    // --- Create mdspan views ---
    Field2D<default_layout, unchecked_accessor> u_o(u_out.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> rho_zz_o(rho_zz_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> theta_m_o(theta_m_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> w_o(w_out.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> exner_o(exner_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> pressure_p_o(pressure_p_out.data(), nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> rho_p_save_v(rho_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_p_save_v(rtheta_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_save_v(ru_save_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_save_v(rw_save_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_pp_v(rho_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_pp_v(rtheta_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_p_v(ru_p_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_p_v(rw_p_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> ruAvg_v(ruAvg_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> wwAvg_v(wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_base_v(rho_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_base_v(rtheta_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> exner_base_v(exner_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rt_diabatic_tend_v(rt_diabatic_tend_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> zb_cell_v(zb_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> zb3_cell_v(zb3_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> cellsOnEdge_v(cellsOnEdge_data.data(), 2, nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_v(edgesOnCell_data.data(), maxEdges, nCells);
    ConstSpan1D nEdgesOnCell_span(nEdgesOnCell_data.data(), nCells);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign_v(edgesOnCell_sign_data.data(), maxEdges, nCells);
    ConstField2D<default_layout, unchecked_accessor> zz_v(zz_data.data(), nVertLevels, nCells);
    ConstSpan1D fzm_s(fzm_data.data(), nVertLevels);
    ConstSpan1D fzp_s(fzp_data.data(), nVertLevels);

    // --- Call recover_state_perturbation ---
    recover_state_perturbation<default_layout>(
        SerialPolicy{},
        u_o, rho_zz_o, theta_m_o, w_o, exner_o, pressure_p_o,
        rho_p_save_v, rtheta_p_save_v, ru_save_v, rw_save_v,
        rho_pp_v, rtheta_pp_v, ru_p_v, rw_p_v,
        ruAvg_v, wwAvg_v,
        rho_base_v, rtheta_base_v, exner_base_v,
        rt_diabatic_tend_v,
        zb_cell_v, zb3_cell_v,
        cellsOnEdge_v, edgesOnCell_v, nEdgesOnCell_span, edgesOnCell_sign_v,
        zz_v, fzm_s, fzp_s,
        cf1, cf2, cf3,
        rk_step, dt, ns,
        nCells, nEdges, nVertLevels, maxEdges);

    // Verify theta_m includes diabatic subtraction
    for (index_type k = 0; k < nVertLevels; ++k) {
        EXPECT_TRUE(matches(theta_m_out[k], fortran_theta_m))
            << "Case 4 (Diabatic): k=" << k
            << " C++ theta_m=" << theta_m_out[k]
            << " Fortran theta_m=" << fortran_theta_m
            << " diff=" << std::abs(theta_m_out[k] - fortran_theta_m)
            << " (expected diabatic term = " << dt * rt_diabatic_tend_val << ")";
    }
}

// ============================================================================
// Property-based test: Perturbation Recovery Matches Fortran Reference
//
// This RC_GTEST_PROP generates random inputs and verifies the new
// recover_state_perturbation kernel produces the expected Fortran result
// for density recovery. The test PASSES when the kernel is correct.
// ============================================================================

RC_GTEST_PROP(RecoverStateBugCondition, Property1_PerturbationDivergence, ()) {
    constexpr index_type nCells = 1;
    constexpr index_type nEdges = 1;
    constexpr index_type maxEdges = 1;
    const auto nVertLevels = *rc::gen::inRange(3, 8);
    const auto nLev = static_cast<std::size_t>(nVertLevels);

    // Generate random parameters
    const real_type rho_p_save_val =
        (*rc::gen::inRange(-100, 101)) / 10000.0;
    const real_type rho_pp_val =
        (*rc::gen::inRange(-100, 101)) / 10000.0;
    const real_type rho_base_val = 0.5 + (*rc::gen::inRange(1, 1000)) / 1000.0;
    const index_type ns = *rc::gen::inRange(1, 11);
    const index_type rk_step = 1;
    const real_type dt = 1.0 + (*rc::gen::inRange(1, 100)) / 10.0;

    // Fortran reference: rho_zz = rho_p_save + rho_pp + rho_base
    const real_type fortran_rho_zz = rho_p_save_val + rho_pp_val + rho_base_val;

    // Allocate arrays
    std::vector<real_type> u_out(nLev, 0.0);
    std::vector<real_type> rho_zz_out(nLev, 0.0);
    std::vector<real_type> theta_m_out(nLev, 0.0);
    std::vector<real_type> w_out(nLev + 1, 0.0);
    std::vector<real_type> exner_out(nLev, 0.0);
    std::vector<real_type> pressure_p_out(nLev, 0.0);

    std::vector<real_type> rho_p_save_data(nLev, rho_p_save_val);
    std::vector<real_type> rtheta_p_save_data(nLev, 0.0);
    std::vector<real_type> ru_save_data(nLev, 0.0);
    std::vector<real_type> rw_save_data(nLev + 1, 0.0);
    std::vector<real_type> rho_pp_data(nLev, rho_pp_val);
    std::vector<real_type> rtheta_pp_data(nLev, 0.0);
    std::vector<real_type> ru_p_data(nLev, 0.0);
    std::vector<real_type> rw_p_data(nLev + 1, 0.0);
    std::vector<real_type> ruAvg_data(nLev, 0.0);
    std::vector<real_type> wwAvg_data(nLev + 1, 0.0);
    std::vector<real_type> rho_base_data(nLev, rho_base_val);
    std::vector<real_type> rtheta_base_data(nLev, 300.0);
    std::vector<real_type> exner_base_data(nLev, 1.0);
    std::vector<real_type> rt_diabatic_tend_data(nLev, 0.0);
    std::vector<real_type> zb_cell_data((nLev + 1) * maxEdges * nCells, 0.0);
    std::vector<real_type> zb3_cell_data((nLev + 1) * maxEdges * nCells, 0.0);
    std::vector<real_type> cellsOnEdge_data = {0.0, 0.0};
    std::vector<real_type> edgesOnCell_data(maxEdges * nCells, 0.0);
    std::vector<real_type> nEdgesOnCell_data(nCells, static_cast<real_type>(maxEdges));
    std::vector<real_type> edgesOnCell_sign_data(maxEdges * nCells, 1.0);
    std::vector<real_type> zz_data(nLev, 1.0);
    std::vector<real_type> fzm_data(nLev, 0.5);
    std::vector<real_type> fzp_data(nLev, 0.5);
    const real_type cf1 = 1.0, cf2 = 0.0, cf3 = 0.0;

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> u_o(u_out.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> rho_zz_o(rho_zz_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> theta_m_o(theta_m_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> w_o(w_out.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> exner_o(exner_out.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> pressure_p_o(pressure_p_out.data(), nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> rho_p_save_v(rho_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_p_save_v(rtheta_p_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_save_v(ru_save_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_save_v(rw_save_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_pp_v(rho_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_pp_v(rtheta_pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ru_p_v(ru_p_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rw_p_v(rw_p_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> ruAvg_v(ruAvg_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> wwAvg_v(wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_base_v(rho_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_base_v(rtheta_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> exner_base_v(exner_base_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rt_diabatic_tend_v(rt_diabatic_tend_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> zb_cell_v(zb_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> zb3_cell_v(zb3_cell_data.data(), nVertLevels + 1, maxEdges * nCells);
    ConstField2D<default_layout, unchecked_accessor> cellsOnEdge_v(cellsOnEdge_data.data(), 2, nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_v(edgesOnCell_data.data(), maxEdges, nCells);
    ConstSpan1D nEdgesOnCell_span(nEdgesOnCell_data.data(), nCells);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign_v(edgesOnCell_sign_data.data(), maxEdges, nCells);
    ConstField2D<default_layout, unchecked_accessor> zz_v(zz_data.data(), nVertLevels, nCells);
    ConstSpan1D fzm_s(fzm_data.data(), nVertLevels);
    ConstSpan1D fzp_s(fzp_data.data(), nVertLevels);

    recover_state_perturbation<default_layout>(
        SerialPolicy{},
        u_o, rho_zz_o, theta_m_o, w_o, exner_o, pressure_p_o,
        rho_p_save_v, rtheta_p_save_v, ru_save_v, rw_save_v,
        rho_pp_v, rtheta_pp_v, ru_p_v, rw_p_v,
        ruAvg_v, wwAvg_v,
        rho_base_v, rtheta_base_v, exner_base_v,
        rt_diabatic_tend_v,
        zb_cell_v, zb3_cell_v,
        cellsOnEdge_v, edgesOnCell_v, nEdgesOnCell_span, edgesOnCell_sign_v,
        zz_v, fzm_s, fzp_s,
        cf1, cf2, cf3,
        rk_step, dt, ns,
        nCells, nEdges, nVertLevels, maxEdges);

    // Verify C++ matches Fortran reference for density
    for (index_type k = 0; k < nVertLevels; ++k) {
        RC_ASSERT(matches(rho_zz_out[k], fortran_rho_zz));
    }
}
