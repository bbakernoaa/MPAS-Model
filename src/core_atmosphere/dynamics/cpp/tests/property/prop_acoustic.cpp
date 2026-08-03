/// @file prop_acoustic.cpp
/// @brief Property-based tests for acoustic stepping and divergence damping.
///
/// **Validates: Requirements 3.2, 3.5, 20.2**
///
/// Property 22: Acoustic Step Mass Conservation
///   After one acoustic step with zero tendencies and zero initial perturbations,
///   all perturbation fields remain zero.
///
/// Property 23: Divergence Damping Monotonicity
///   After divergence damping, the L2 norm of divergence does not increase.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/acoustic.hpp>
#include <mpas_dycore/kernels/divergence_damping.hpp>
#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include "../synthetic_mesh.hpp"
#include <vector>
#include <cmath>
#include <numeric>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;
using namespace mpas::dycore::testing;

constexpr index_type SENTINEL = mpas::dycore::INVALID_INDEX;

// ============================================================================
// Helper: Build MeshConnectivity from a SyntheticMesh
// ============================================================================

static MeshConnectivity buildConnectivity(const SyntheticMesh& sm) {
    MeshConnectivity mc;
    mc.cellsOnEdge    = sm.cellsOnEdge_view();
    mc.verticesOnEdge = sm.verticesOnEdge_view();
    mc.edgesOnCell    = sm.edgesOnCell_view();
    mc.cellsOnCell    = sm.cellsOnCell_view();
    mc.verticesOnCell = sm.verticesOnCell_view();
    mc.cellsOnVertex  = sm.cellsOnVertex_view();
    mc.nEdgesOnCell   = sm.nEdgesOnCell_view();
    return mc;
}

// ============================================================================
// Helper: Compute L2 norm of divergence from ru_p
// ============================================================================

static double computeDivergenceL2(
    const std::vector<real_type>& ru_p_data,
    const SyntheticMesh& sm,
    const MeshConnectivity& mc,
    index_type nVertLevels)
{
    const auto nCells = sm.nCells;
    const auto nEdges = sm.nEdges;

    // Compute invAreaCell
    std::vector<real_type> invAreaCell(static_cast<std::size_t>(nCells));
    for (index_type i = 0; i < nCells; ++i) {
        invAreaCell[static_cast<std::size_t>(i)] =
            1.0 / sm.areaCell[static_cast<std::size_t>(i)];
    }

    // Build ru_p view
    ConstField2D<default_layout, unchecked_accessor> ru_p(
        ru_p_data.data(), nVertLevels, nEdges);

    // Compute divergence at each cell
    double l2_sum = 0.0;
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const index_type n_edges = mc.nEdgesOnCell[iCell];
        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type div_sum = 0.0;
            for (index_type j = 0; j < n_edges; ++j) {
                const index_type iEdge = mc.edgesOnCell[iCell, j];
                if (iEdge == SENTINEL) break;
                const real_type sign =
                    (mc.cellsOnEdge[iEdge, 0] == iCell) ? 1.0 : -1.0;
                div_sum += sign * sm.dvEdge[static_cast<std::size_t>(iEdge)]
                           * ru_p[k, iEdge];
            }
            double div_val = div_sum * invAreaCell[static_cast<std::size_t>(iCell)];
            l2_sum += div_val * div_val;
        }
    }
    return std::sqrt(l2_sum);
}

// ============================================================================
// Test 1: Zero perturbation preservation (Property 22)
//
// Call advance_acoustic_step with small_step=1, all tendencies=0. After the
// call, verify that all perturbation fields (ru_p, rw_p, rho_pp, rtheta_pp)
// remain zero (since there's no forcing).
// ============================================================================

RC_GTEST_PROP(AcousticStepMassConservation,
              ZeroPerturbationPreservation, ()) {
    const auto nCells = *rc::gen::inRange(4, 10);
    const auto nVertLevels = *rc::gen::inRange(3, 8);

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;
    const auto maxEdges = sm.maxEdges;
    const auto nCellsAll = nCells; // no halo cells in synthetic mesh
    const index_type small_step = 1;
    const real_type dts = 0.5; // any positive timestep

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Prognostic perturbation fields — all zero
    std::vector<real_type> ru_p_data(nLev * nE, 0.0);
    std::vector<real_type> rw_p_data((nLev + 1) * nC, 0.0);
    std::vector<real_type> rtheta_pp_data(nLev * nC, 0.0);
    std::vector<real_type> rho_pp_data(nLev * nC, 0.0);
    std::vector<real_type> rtheta_pp_old_data(nLev * nC, 0.0);

    // Accumulated flux fields — will be initialized by the kernel
    std::vector<real_type> ruAvg_data(nLev * nE, 0.0);
    std::vector<real_type> wwAvg_data((nLev + 1) * nC, 0.0);

    // Base-state fields — reasonable constants
    std::vector<real_type> rho_zz_data(nLev * nC, 1.0);
    std::vector<real_type> theta_m_data(nLev * nC, 300.0);
    std::vector<real_type> zz_data(nLev * nC, 1.0);
    std::vector<real_type> exner_data(nLev * nC, 1.0);
    std::vector<real_type> cqu_data(nLev * nE, 1.0);
    std::vector<real_type> zxu_data(nLev * nE, 0.0); // flat terrain

    // Implicit solve coefficients — all zero (no coupling)
    std::vector<real_type> cofwt_data(nLev * nC, 0.0);
    std::vector<real_type> coftz_data((nLev + 1) * nC, 0.0);
    std::vector<real_type> cofwr_data(nLev * nC, 0.0);
    std::vector<real_type> cofwz_data(nLev * nC, 0.0);
    std::vector<real_type> a_tri_data(nLev * nC, 0.0);
    std::vector<real_type> alpha_tri_data(nLev * nC, 1.0); // identity
    std::vector<real_type> gamma_tri_data(nLev * nC, 0.0);

    // Rayleigh damping — disabled (zero coefficient)
    std::vector<real_type> dss_data(nLev * nC, 0.0);

    // Tendencies — all zero (no forcing)
    std::vector<real_type> tend_ru_data(nLev * nE, 0.0);
    std::vector<real_type> tend_rho_data(nLev * nC, 0.0);
    std::vector<real_type> tend_rt_data(nLev * nC, 0.0);
    std::vector<real_type> tend_rw_data((nLev + 1) * nC, 0.0);

    // Velocity fields
    std::vector<real_type> w_data((nLev + 1) * nC, 0.0);
    std::vector<real_type> rw_data((nLev + 1) * nC, 0.0);
    std::vector<real_type> rw_save_data((nLev + 1) * nC, 0.0);

    // Edge orientation signs: +1 or -1 (maxEdges, nCells)
    std::vector<real_type> edgesOnCell_sign_data(
        static_cast<std::size_t>(maxEdges) * nC, 1.0);
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const index_type n_edges_on_cell = mc.nEdgesOnCell[iCell];
        for (index_type j = 0; j < n_edges_on_cell; ++j) {
            const index_type iEdge = mc.edgesOnCell[iCell, j];
            if (iEdge == SENTINEL) break;
            const real_type sign =
                (mc.cellsOnEdge[iEdge, 0] == iCell) ? 1.0 : -1.0;
            edgesOnCell_sign_data[static_cast<std::size_t>(j) * nC
                + static_cast<std::size_t>(iCell)] = sign;
        }
    }

    // 1D geometry arrays
    std::vector<real_type> invDcEdge_data(nE);
    for (index_type e = 0; e < nEdges; ++e) {
        invDcEdge_data[static_cast<std::size_t>(e)] =
            1.0 / sm.dcEdge[static_cast<std::size_t>(e)];
    }
    std::vector<real_type> invAreaCell_data(nC);
    for (index_type c = 0; c < nCells; ++c) {
        invAreaCell_data[static_cast<std::size_t>(c)] =
            1.0 / sm.areaCell[static_cast<std::size_t>(c)];
    }

    // Vertical 1D arrays
    std::vector<real_type> cofrz_data(nLev, 0.0);
    std::vector<real_type> rdzw_data(nLev, 1.0);
    std::vector<real_type> fzm_data(nLev, 0.5);
    std::vector<real_type> fzp_data(nLev, 0.5);
    std::vector<real_type> etp_data(nLev, 0.5);
    std::vector<real_type> etm_data(nLev, 0.5);
    std::vector<real_type> ewp_data(nLev + 1, 0.5);
    std::vector<real_type> ewm_data(nLev + 1, 0.5);

    // Specified zone masks — all zero (no specified zone)
    std::vector<real_type> specZoneMaskEdge_data(nE, 0.0);
    std::vector<real_type> specZoneMaskCell_data(nC, 0.0);

    // Create mdspan views for all 2D fields
    Field2D<default_layout, unchecked_accessor> ru_p(
        ru_p_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> rw_p(
        rw_p_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> rtheta_pp(
        rtheta_pp_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> rho_pp(
        rho_pp_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> rtheta_pp_old(
        rtheta_pp_old_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> ruAvg(
        ruAvg_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> wwAvg(
        wwAvg_data.data(), nVertLevels + 1, nCells);

    ConstField2D<default_layout, unchecked_accessor> rho_zz(
        rho_zz_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> theta_m(
        theta_m_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> zz(
        zz_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> exner(
        exner_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> cqu(
        cqu_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> zxu(
        zxu_data.data(), nVertLevels, nEdges);

    ConstField2D<default_layout, unchecked_accessor> cofwt(
        cofwt_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> coftz(
        coftz_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> cofwr(
        cofwr_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> cofwz(
        cofwz_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> a_tri(
        a_tri_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> alpha_tri(
        alpha_tri_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> gamma_tri(
        gamma_tri_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> dss(
        dss_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> tend_ru(
        tend_ru_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> tend_rho(
        tend_rho_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> tend_rt(
        tend_rt_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> tend_rw(
        tend_rw_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> w(
        w_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rw(
        rw_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rw_save(
        rw_save_data.data(), nVertLevels + 1, nCells);

    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign(
        edgesOnCell_sign_data.data(), maxEdges, nCells);

    // 1D spans
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invDcEdge(invDcEdge_data.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invAreaCell(invAreaCell_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        dvEdge(sm.dvEdge.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        cofrz(cofrz_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        rdzw(rdzw_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzm(fzm_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzp(fzp_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        etp(etp_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        etm(etm_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        ewp(ewp_data.data(), nVertLevels + 1);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        ewm(ewm_data.data(), nVertLevels + 1);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        specZoneMaskEdge(specZoneMaskEdge_data.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        specZoneMaskCell(specZoneMaskCell_data.data(), nCells);

    // Call advance_acoustic_step with small_step=1 and zero tendencies
    advance_acoustic_step<default_layout>(
        SerialPolicy{},
        ru_p, rw_p, rtheta_pp, rho_pp, rtheta_pp_old,
        ruAvg, wwAvg,
        rho_zz, theta_m, zz, exner, cqu, zxu,
        cofwt, coftz, cofwr, cofwz,
        a_tri, alpha_tri, gamma_tri,
        dss,
        tend_ru, tend_rho, tend_rt, tend_rw,
        w, rw, rw_save,
        mc, edgesOnCell_sign,
        invDcEdge, invAreaCell, dvEdge,
        cofrz, rdzw, fzm, fzp, etp, etm, ewp, ewm,
        specZoneMaskEdge, specZoneMaskCell,
        dts, small_step,
        nCells, nCellsAll, nEdges, nVertLevels, maxEdges);

    // Verify all perturbation fields remain zero
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            auto val = ru_p[k, iEdge];
            RC_ASSERT(val == 0.0);
        }
    }
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            auto rho_val = rho_pp[k, iCell];
            auto rt_val = rtheta_pp[k, iCell];
            RC_ASSERT(rho_val == 0.0);
            RC_ASSERT(rt_val == 0.0);
        }
        for (index_type k = 0; k <= nVertLevels; ++k) {
            auto rw_val = rw_p[k, iCell];
            RC_ASSERT(rw_val == 0.0);
        }
    }
}

// ============================================================================
// Test 2: Divergence damping reduces divergence norm (Property 23)
//
// Create a field ru_p with non-zero divergence. Apply divergence damping.
// Verify that the L2 norm of the resulting divergence is <= the L2 norm
// before damping (for positive damping coefficient).
// ============================================================================

RC_GTEST_PROP(DivergenceDampingMonotonicity,
              DampingReducesDivergenceNorm, ()) {
    const auto nCells = *rc::gen::inRange(4, 10);
    const auto nVertLevels = *rc::gen::inRange(3, 8);

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;
    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Generate ru_p with non-zero divergence
    std::vector<real_type> ru_p_data(nLev * nE);
    for (auto& v : ru_p_data) {
        int raw = *rc::gen::inRange(-1000, 1001);
        v = static_cast<real_type>(raw) / 100.0;
    }

    // Divergence workspace
    std::vector<real_type> divergence_data(nLev * nC, 0.0);

    // Geometry arrays
    std::vector<real_type> invAreaCell_data(nC);
    for (index_type c = 0; c < nCells; ++c) {
        invAreaCell_data[static_cast<std::size_t>(c)] =
            1.0 / sm.areaCell[static_cast<std::size_t>(c)];
    }
    std::vector<real_type> invDcEdge_data(nE);
    for (index_type e = 0; e < nEdges; ++e) {
        invDcEdge_data[static_cast<std::size_t>(e)] =
            1.0 / sm.dcEdge[static_cast<std::size_t>(e)];
    }

    // Positive damping coefficient and timestep — use moderate values
    // to ensure damping is effective but not excessive
    const real_type dts = 0.5;
    // Generate small positive coefficient: divdamp_coef in [0.001, 0.1]
    const int coef_raw = *rc::gen::inRange(1, 100);
    const real_type divdamp_coef = static_cast<real_type>(coef_raw) / 1000.0;

    // Compute L2 norm of divergence before damping
    double l2_before = computeDivergenceL2(ru_p_data, sm, mc, nVertLevels);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> ru_p(
        ru_p_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> divergence(
        divergence_data.data(), nVertLevels, nCells);

    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        dvEdge(sm.dvEdge.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invAreaCell(invAreaCell_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invDcEdge(invDcEdge_data.data(), nEdges);

    // Apply divergence damping
    apply_divergence_damping<default_layout>(
        SerialPolicy{},
        ru_p, divergence, mc,
        dvEdge, invAreaCell, invDcEdge,
        dts, divdamp_coef,
        nCells, nEdges, nVertLevels);

    // Compute L2 norm of divergence after damping
    double l2_after = computeDivergenceL2(ru_p_data, sm, mc, nVertLevels);

    // Divergence norm should not increase after damping
    // Allow small floating-point tolerance
    RC_ASSERT(l2_after <= l2_before + 1.0e-10);
}

// ============================================================================
// Test 3: Divergence damping with zero coefficient is no-op
//
// Call apply_divergence_damping with divdamp_coef=0. Verify ru_p unchanged.
// ============================================================================

RC_GTEST_PROP(DivergenceDampingMonotonicity,
              ZeroCoefficientIsNoOp, ()) {
    const auto nCells = *rc::gen::inRange(4, 10);
    const auto nVertLevels = *rc::gen::inRange(3, 8);

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;
    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Generate random ru_p
    std::vector<real_type> ru_p_data(nLev * nE);
    for (auto& v : ru_p_data) {
        int raw = *rc::gen::inRange(-10000, 10001);
        v = static_cast<real_type>(raw) / 100.0;
    }

    // Save original values
    std::vector<real_type> ru_p_original(ru_p_data);

    // Workspace
    std::vector<real_type> divergence_data(nLev * nC, 0.0);

    // Geometry arrays
    std::vector<real_type> invAreaCell_data(nC);
    for (index_type c = 0; c < nCells; ++c) {
        invAreaCell_data[static_cast<std::size_t>(c)] =
            1.0 / sm.areaCell[static_cast<std::size_t>(c)];
    }
    std::vector<real_type> invDcEdge_data(nE);
    for (index_type e = 0; e < nEdges; ++e) {
        invDcEdge_data[static_cast<std::size_t>(e)] =
            1.0 / sm.dcEdge[static_cast<std::size_t>(e)];
    }

    const real_type dts = 1.0;
    const real_type divdamp_coef = 0.0; // zero coefficient

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> ru_p(
        ru_p_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> divergence(
        divergence_data.data(), nVertLevels, nCells);

    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        dvEdge(sm.dvEdge.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invAreaCell(invAreaCell_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invDcEdge(invDcEdge_data.data(), nEdges);

    // Apply divergence damping with zero coefficient
    apply_divergence_damping<default_layout>(
        SerialPolicy{},
        ru_p, divergence, mc,
        dvEdge, invAreaCell, invDcEdge,
        dts, divdamp_coef,
        nCells, nEdges, nVertLevels);

    // Verify ru_p is bitwise unchanged
    for (std::size_t i = 0; i < ru_p_data.size(); ++i) {
        RC_ASSERT(ru_p_data[i] == ru_p_original[i]);
    }
}
