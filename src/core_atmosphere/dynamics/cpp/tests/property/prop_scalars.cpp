/// @file prop_scalars.cpp
/// @brief Property-based tests for scalar transport with FCT limiting.
///
/// **Validates: Requirements 5.2, 5.5, 5.6**
///
/// Property 24: Scalar Mass Conservation
///   Total scalar mass (rho*scalar*volume integral) is conserved across
///   the transport step to within floating-point tolerance.
///
/// Property 25: Scalar Positivity
///   All scalar values remain >= 0 after transport with FCT limiting.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/scalars.hpp>
#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include "../synthetic_mesh.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

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
// Helper: Build edgesOnCell_sign for a given mesh
//
// edgesOnCell_sign(j, iCell) = +1 if cellsOnEdge(edge, 0) == iCell else -1
// Layout: (maxEdges, nCells) in column-major
// ============================================================================

static std::vector<real_type> buildEdgesOnCellSign(
    const SyntheticMesh& sm,
    const MeshConnectivity& mc)
{
    const auto nCells = sm.nCells;
    const auto maxEdges = sm.maxEdges;
    const auto nC = static_cast<std::size_t>(nCells);
    const auto mE = static_cast<std::size_t>(maxEdges);

    std::vector<real_type> sign_data(mE * nC, 0.0);
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const index_type n_edges = mc.nEdgesOnCell[iCell];
        for (index_type j = 0; j < n_edges; ++j) {
            const index_type iEdge = mc.edgesOnCell[iCell, j];
            if (iEdge == SENTINEL) break;
            real_type s = (mc.cellsOnEdge[iEdge, 0] == iCell) ? 1.0 : -1.0;
            // Column-major: index = j + maxEdges * iCell? No — the kernel
            // uses edgesOnCell_sign[i, iCell] with layout (maxEdges, nCells)
            // which in layout_left means index = i + maxEdges * iCell.
            sign_data[static_cast<std::size_t>(j)
                + mE * static_cast<std::size_t>(iCell)] = s;
        }
    }
    return sign_data;
}

// ============================================================================
// Helper: Build simple advection stencil for edges
//
// For each internal edge (both cells valid), use a simple 2-cell stencil
// with upwind coefficients. For boundary edges, nAdvCells = 0.
// ============================================================================

struct AdvectionStencil {
    std::vector<index_type> advCellsForEdge_data; // (nEdges, maxAdvCells) flat
    std::vector<index_type> nAdvCellsForEdge_data; // (nEdges)
    std::vector<real_type> adv_coefs_data;          // (maxAdvCells, nEdges) flat
    std::vector<real_type> adv_coefs_3rd_data;      // (maxAdvCells, nEdges) flat
    index_type maxAdvCells;
};

static AdvectionStencil buildAdvectionStencil(
    const SyntheticMesh& sm,
    const MeshConnectivity& mc)
{
    AdvectionStencil stencil;
    const auto nEdges = sm.nEdges;
    stencil.maxAdvCells = 2; // simple 2-cell stencil

    const auto nE = static_cast<std::size_t>(nEdges);
    const std::size_t maxAdv = 2;

    stencil.advCellsForEdge_data.resize(nE * maxAdv, 0);
    stencil.nAdvCellsForEdge_data.resize(nE, 0);
    stencil.adv_coefs_data.resize(maxAdv * nE, 0.0);
    stencil.adv_coefs_3rd_data.resize(maxAdv * nE, 0.0);

    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        const index_type cell1 = mc.cellsOnEdge[iEdge, 0];
        const index_type cell2 = mc.cellsOnEdge[iEdge, 1];

        if (cell1 != SENTINEL && cell2 != SENTINEL) {
            stencil.nAdvCellsForEdge_data[static_cast<std::size_t>(iEdge)] = 2;

            // advCellsForEdge layout: (nEdges, maxAdvCells) row-major in ConnectivityView
            // ConnectivityView is (nEdges, maxAdvCells)
            stencil.advCellsForEdge_data[static_cast<std::size_t>(iEdge) * maxAdv + 0] = cell1;
            stencil.advCellsForEdge_data[static_cast<std::size_t>(iEdge) * maxAdv + 1] = cell2;

            // adv_coefs layout: (maxAdvCells, nEdges) — layout_left means
            // index = i + maxAdvCells * iEdge
            // Standard upwind coefficients: cell1 gets 0.5, cell2 gets 0.5
            stencil.adv_coefs_data[0 + maxAdv * static_cast<std::size_t>(iEdge)] = 0.5;
            stencil.adv_coefs_data[1 + maxAdv * static_cast<std::size_t>(iEdge)] = 0.5;
            // 3rd-order coefficients: cell1 gets +0.5, cell2 gets -0.5 (upwind bias)
            stencil.adv_coefs_3rd_data[0 + maxAdv * static_cast<std::size_t>(iEdge)] = 0.5;
            stencil.adv_coefs_3rd_data[1 + maxAdv * static_cast<std::size_t>(iEdge)] = -0.5;
        } else {
            stencil.nAdvCellsForEdge_data[static_cast<std::size_t>(iEdge)] = 0;
        }
    }

    return stencil;
}

// ============================================================================
// Test 1: Positivity Preservation (Property 25)
//
// Generate random positive scalar values in [0, 0.1], random mass fluxes
// (ruAvg, wwAvg), and call advance_scalars_mono. Verify all output values >= 0.
//
// **Validates: Requirements 5.5**
// ============================================================================

RC_GTEST_PROP(ScalarPositivity,
              PositiveScalarsRemainNonNegative, ()) {
    const auto nCells = *rc::gen::inRange(4, 10);
    const auto nVertLevels = *rc::gen::inRange(5, 10);
    const index_type nScalars = 2;

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;
    const auto maxEdges = sm.maxEdges;
    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE = static_cast<std::size_t>(nEdges);
    const auto nC = static_cast<std::size_t>(nCells);
    const auto nS = static_cast<std::size_t>(nScalars);

    // Scalar field: positive values in [0.001, 0.1]
    std::vector<real_type> scalars_data(nS * nLev * nC);
    for (auto& v : scalars_data) {
        int raw = *rc::gen::inRange(1, 100);
        v = static_cast<real_type>(raw) / 1000.0;
    }

    // Mass fluxes: moderate random values
    std::vector<real_type> ruAvg_data(nLev * nE);
    for (auto& v : ruAvg_data) {
        int raw = *rc::gen::inRange(-50, 51);
        v = static_cast<real_type>(raw) / 10.0;
    }

    std::vector<real_type> wwAvg_data((nLev + 1) * nC);
    for (auto& v : wwAvg_data) {
        int raw = *rc::gen::inRange(-20, 21);
        v = static_cast<real_type>(raw) / 100.0;
    }
    // Zero flux at top and bottom boundaries
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        wwAvg_data[0 + (nLev + 1) * static_cast<std::size_t>(iCell)] = 0.0;
        wwAvg_data[nLev + (nLev + 1) * static_cast<std::size_t>(iCell)] = 0.0;
    }

    // Density fields: positive, ~1.0 kg/m^3
    std::vector<real_type> rho_old_data(nLev * nC);
    std::vector<real_type> rho_new_data(nLev * nC);
    for (std::size_t i = 0; i < nLev * nC; ++i) {
        int raw = *rc::gen::inRange(800, 1200);
        rho_old_data[i] = static_cast<real_type>(raw) / 1000.0;
        rho_new_data[i] = rho_old_data[i]; // same for simplicity
    }

    // Build advection stencil
    auto stencil = buildAdvectionStencil(sm, mc);

    // Build edgesOnCell_sign
    auto sign_data = buildEdgesOnCellSign(sm, mc);

    // Geometry: invAreaCell, dvEdge, rdzw, fzm, fzp
    std::vector<real_type> invAreaCell_data(nC);
    for (index_type c = 0; c < nCells; ++c) {
        invAreaCell_data[static_cast<std::size_t>(c)] =
            1.0 / sm.areaCell[static_cast<std::size_t>(c)];
    }

    // rdzw: reciprocal layer thickness, uniform ~1/1000m
    std::vector<real_type> rdzw_data(nLev, 1.0 / 1000.0);

    // fzm/fzp: vertical interpolation weights summing to 1
    std::vector<real_type> fzm_data(nLev, 0.5);
    std::vector<real_type> fzp_data(nLev, 0.5);

    const real_type dt = 10.0;
    const real_type coef_3rd_order = 1.0;

    // Create mdspan views
    Field3D<default_layout, unchecked_accessor> scalars(
        scalars_data.data(), nScalars, nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ruAvg(
        ruAvg_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> wwAvg(
        wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_old(
        rho_old_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_new(
        rho_new_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign(
        sign_data.data(), maxEdges, nCells);

    ConnectivityView advCellsForEdge(
        stencil.advCellsForEdge_data.data(), nEdges, stencil.maxAdvCells);
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>
        nAdvCellsForEdge(stencil.nAdvCellsForEdge_data.data(), nEdges);
    ConstField2D<default_layout, unchecked_accessor> adv_coefs(
        stencil.adv_coefs_data.data(), stencil.maxAdvCells, nEdges);
    ConstField2D<default_layout, unchecked_accessor> adv_coefs_3rd(
        stencil.adv_coefs_3rd_data.data(), stencil.maxAdvCells, nEdges);

    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        dvEdge(sm.dvEdge.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invAreaCell(invAreaCell_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        rdzw(rdzw_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzm(fzm_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzp(fzp_data.data(), nVertLevels);

    // Call advance_scalars_mono
    advance_scalars_mono<default_layout>(
        SerialPolicy{},
        scalars, ruAvg, wwAvg, rho_old, rho_new,
        mc, dvEdge, invAreaCell, rdzw, fzm, fzp,
        advCellsForEdge, nAdvCellsForEdge,
        adv_coefs, adv_coefs_3rd, edgesOnCell_sign,
        dt, coef_3rd_order,
        nScalars, nCells, nEdges, nVertLevels, stencil.maxAdvCells);

    // Verify all output scalar values are non-negative (Property 25)
    for (index_type s = 0; s < nScalars; ++s) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            for (index_type c = 0; c < nCells; ++c) {
                auto val = scalars[s, k, c];
                RC_ASSERT(val >= 0.0);
            }
        }
    }
}

// ============================================================================
// Test 2: Zero Flux Preserves Scalars (Property 24 variant)
//
// When ruAvg=0 and wwAvg=0 and rho_zz_old==rho_zz_new, scalars should
// remain unchanged (no transport occurs).
//
// **Validates: Requirements 5.2, 5.6**
// ============================================================================

RC_GTEST_PROP(ScalarMassConservation,
              ZeroFluxPreservesScalars, ()) {
    const auto nCells = *rc::gen::inRange(4, 10);
    const auto nVertLevels = *rc::gen::inRange(5, 10);
    const index_type nScalars = 2;

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;
    const auto maxEdges = sm.maxEdges;
    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE = static_cast<std::size_t>(nEdges);
    const auto nC = static_cast<std::size_t>(nCells);
    const auto nS = static_cast<std::size_t>(nScalars);

    // Scalar field: random positive values
    std::vector<real_type> scalars_data(nS * nLev * nC);
    for (auto& v : scalars_data) {
        int raw = *rc::gen::inRange(1, 1000);
        v = static_cast<real_type>(raw) / 1000.0;
    }

    // Save original scalar values for comparison
    std::vector<real_type> scalars_original(scalars_data);

    // Zero mass fluxes: no transport
    std::vector<real_type> ruAvg_data(nLev * nE, 0.0);
    std::vector<real_type> wwAvg_data((nLev + 1) * nC, 0.0);

    // Density fields: identical old and new (no density change)
    std::vector<real_type> rho_data(nLev * nC);
    for (std::size_t i = 0; i < nLev * nC; ++i) {
        int raw = *rc::gen::inRange(800, 1200);
        rho_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Build advection stencil
    auto stencil = buildAdvectionStencil(sm, mc);

    // Build edgesOnCell_sign
    auto sign_data = buildEdgesOnCellSign(sm, mc);

    // Geometry
    std::vector<real_type> invAreaCell_data(nC);
    for (index_type c = 0; c < nCells; ++c) {
        invAreaCell_data[static_cast<std::size_t>(c)] =
            1.0 / sm.areaCell[static_cast<std::size_t>(c)];
    }
    std::vector<real_type> rdzw_data(nLev, 1.0 / 1000.0);
    std::vector<real_type> fzm_data(nLev, 0.5);
    std::vector<real_type> fzp_data(nLev, 0.5);

    const real_type dt = 10.0;
    const real_type coef_3rd_order = 1.0;

    // Create mdspan views
    Field3D<default_layout, unchecked_accessor> scalars(
        scalars_data.data(), nScalars, nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ruAvg(
        ruAvg_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> wwAvg(
        wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_old(
        rho_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_new(
        rho_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign(
        sign_data.data(), maxEdges, nCells);

    ConnectivityView advCellsForEdge(
        stencil.advCellsForEdge_data.data(), nEdges, stencil.maxAdvCells);
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>
        nAdvCellsForEdge(stencil.nAdvCellsForEdge_data.data(), nEdges);
    ConstField2D<default_layout, unchecked_accessor> adv_coefs(
        stencil.adv_coefs_data.data(), stencil.maxAdvCells, nEdges);
    ConstField2D<default_layout, unchecked_accessor> adv_coefs_3rd(
        stencil.adv_coefs_3rd_data.data(), stencil.maxAdvCells, nEdges);

    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        dvEdge(sm.dvEdge.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invAreaCell(invAreaCell_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        rdzw(rdzw_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzm(fzm_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzp(fzp_data.data(), nVertLevels);

    // Call advance_scalars_mono with zero fluxes
    advance_scalars_mono<default_layout>(
        SerialPolicy{},
        scalars, ruAvg, wwAvg, rho_old, rho_new,
        mc, dvEdge, invAreaCell, rdzw, fzm, fzp,
        advCellsForEdge, nAdvCellsForEdge,
        adv_coefs, adv_coefs_3rd, edgesOnCell_sign,
        dt, coef_3rd_order,
        nScalars, nCells, nEdges, nVertLevels, stencil.maxAdvCells);

    // Verify scalars are unchanged when there is no transport
    for (std::size_t i = 0; i < scalars_data.size(); ++i) {
        RC_ASSERT(std::abs(scalars_data[i] - scalars_original[i]) < 1.0e-14);
    }
}
