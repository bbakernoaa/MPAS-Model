/// @file prop_scalars_disabled.cpp
/// @brief Property-based test for scalar advection disabled behavior.
///
/// **Validates: Requirements 23.1, 23.7**
///
/// Property 39: Scalar Advection Disabled Leaves Scalars Unchanged
///   When nScalars=0, advance_scalars_mono returns immediately without
///   modifying any data. Also verifies graceful return with nCells=0
///   and nVertLevels=0.

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
// Test 1: nScalars=0 returns immediately without modifying data (Property 39)
//
// Call advance_scalars_mono with nScalars=0 on a valid mesh with random
// scalar data pre-filled. Verify all arrays remain bitwise unchanged.
// ============================================================================

RC_GTEST_PROP(ScalarAdvectionDisabled,
              ZeroScalarsLeavesDataUnchanged, ()) {
    // **Validates: Requirements 23.1, 23.7**

    const auto nCells = *rc::gen::inRange(4, 10);
    const auto nVertLevels = *rc::gen::inRange(3, 8);
    const auto maxAdvCells = 6;

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;
    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Use nScalars > 0 for the array sizing, but pass nScalars=0 to the kernel
    const index_type arrayScalars = 3;  // actual array has some scalars
    const index_type passedScalars = 0; // but we tell kernel there are 0

    // Scalar field with random non-zero data
    std::vector<real_type> scalars_data(
        static_cast<std::size_t>(arrayScalars) * nLev * nC);
    for (auto& v : scalars_data) {
        int raw = *rc::gen::inRange(1, 1000);
        v = static_cast<real_type>(raw) / 100.0;
    }
    // Save original data for comparison
    std::vector<real_type> scalars_original(scalars_data);

    // Mass flux fields (random non-zero to ensure kernel would modify if it ran)
    std::vector<real_type> ruAvg_data(nLev * nE);
    for (auto& v : ruAvg_data) {
        int raw = *rc::gen::inRange(-500, 501);
        v = static_cast<real_type>(raw) / 100.0;
    }
    std::vector<real_type> wwAvg_data((nLev + 1) * nC);
    for (auto& v : wwAvg_data) {
        int raw = *rc::gen::inRange(-500, 501);
        v = static_cast<real_type>(raw) / 100.0;
    }

    // Density fields
    std::vector<real_type> rho_zz_old_data(nLev * nC, 1.0);
    std::vector<real_type> rho_zz_new_data(nLev * nC, 1.0);

    // Geometry arrays
    std::vector<real_type> invAreaCell_data(nC);
    for (index_type c = 0; c < nCells; ++c) {
        invAreaCell_data[static_cast<std::size_t>(c)] =
            1.0 / sm.areaCell[static_cast<std::size_t>(c)];
    }

    // Vertical 1D arrays
    std::vector<real_type> rdzw_data(nLev, 1.0);
    std::vector<real_type> fzm_data(nLev, 0.5);
    std::vector<real_type> fzp_data(nLev, 0.5);

    // Advection stencil arrays (minimal but valid)
    std::vector<index_type> advCellsForEdge_data(
        nE * static_cast<std::size_t>(maxAdvCells), 0);
    std::vector<index_type> nAdvCellsForEdge_data(nE, 0);
    std::vector<real_type> adv_coefs_data(
        static_cast<std::size_t>(maxAdvCells) * nE, 0.0);
    std::vector<real_type> adv_coefs_3rd_data(
        static_cast<std::size_t>(maxAdvCells) * nE, 0.0);

    // Edge orientation signs
    std::vector<real_type> edgesOnCell_sign_data(
        static_cast<std::size_t>(sm.maxEdges) * nC, 1.0);

    // Create mdspan views
    Field3D<default_layout, unchecked_accessor> scalars(
        scalars_data.data(), arrayScalars, nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> ruAvg(
        ruAvg_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> wwAvg(
        wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_zz_old(
        rho_zz_old_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_zz_new(
        rho_zz_new_data.data(), nVertLevels, nCells);

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

    ConnectivityView advCellsForEdge(
        advCellsForEdge_data.data(), nEdges, maxAdvCells);
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>
        nAdvCellsForEdge(nAdvCellsForEdge_data.data(), nEdges);

    ConstField2D<default_layout, unchecked_accessor> adv_coefs(
        adv_coefs_data.data(), maxAdvCells, nEdges);
    ConstField2D<default_layout, unchecked_accessor> adv_coefs_3rd(
        adv_coefs_3rd_data.data(), maxAdvCells, nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign(
        edgesOnCell_sign_data.data(), sm.maxEdges, nCells);

    real_type dt = 10.0;
    real_type coef_3rd_order = 1.0;

    // Call advance_scalars_mono with nScalars=0
    advance_scalars_mono<default_layout>(
        SerialPolicy{},
        scalars, ruAvg, wwAvg, rho_zz_old, rho_zz_new,
        mc, dvEdge, invAreaCell, rdzw, fzm, fzp,
        advCellsForEdge, nAdvCellsForEdge,
        adv_coefs, adv_coefs_3rd, edgesOnCell_sign,
        dt, coef_3rd_order,
        passedScalars,  // nScalars = 0
        nCells, nEdges, nVertLevels, maxAdvCells);

    // Verify scalar data is bitwise unchanged
    for (std::size_t i = 0; i < scalars_data.size(); ++i) {
        RC_ASSERT(scalars_data[i] == scalars_original[i]);
    }
}

// ============================================================================
// Test 2: nCells=0 returns immediately without modifying data (Property 39)
//
// Call advance_scalars_mono with nCells=0 to verify zero-extent graceful return.
// ============================================================================

RC_GTEST_PROP(ScalarAdvectionDisabled,
              ZeroCellsLeavesDataUnchanged, ()) {
    // **Validates: Requirements 23.1, 23.7**

    const index_type nCells = 0;
    const auto nVertLevels = *rc::gen::inRange(3, 8);
    const auto nScalars = *rc::gen::inRange(1, 5);
    const index_type nEdges = 0;
    const index_type maxAdvCells = 6;

    const auto nLev = static_cast<std::size_t>(nVertLevels);

    // Minimal data (empty arrays since dimensions are zero)
    std::vector<real_type> scalars_data;
    std::vector<real_type> ruAvg_data;
    std::vector<real_type> wwAvg_data;
    std::vector<real_type> rho_zz_old_data;
    std::vector<real_type> rho_zz_new_data;
    std::vector<real_type> dvEdge_data;
    std::vector<real_type> invAreaCell_data;
    std::vector<real_type> rdzw_data(nLev, 1.0);
    std::vector<real_type> fzm_data(nLev, 0.5);
    std::vector<real_type> fzp_data(nLev, 0.5);
    std::vector<index_type> advCellsForEdge_data;
    std::vector<index_type> nAdvCellsForEdge_data;
    std::vector<real_type> adv_coefs_data;
    std::vector<real_type> adv_coefs_3rd_data;
    std::vector<real_type> edgesOnCell_sign_data;

    // Create mdspan views with zero extents where applicable
    Field3D<default_layout, unchecked_accessor> scalars(
        scalars_data.data(), nScalars, nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ruAvg(
        ruAvg_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> wwAvg(
        wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_zz_old(
        rho_zz_old_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_zz_new(
        rho_zz_new_data.data(), nVertLevels, nCells);

    // Construct a minimal empty MeshConnectivity
    std::vector<index_type> empty_idx;
    MeshConnectivity mc;
    mc.cellsOnEdge    = ConnectivityView(empty_idx.data(), nEdges, 2);
    mc.verticesOnEdge = ConnectivityView(empty_idx.data(), nEdges, 2);
    mc.edgesOnCell    = ConnectivityView(empty_idx.data(), nCells, index_type(6));
    mc.cellsOnCell    = ConnectivityView(empty_idx.data(), nCells, index_type(6));
    mc.verticesOnCell = ConnectivityView(empty_idx.data(), nCells, index_type(6));
    mc.cellsOnVertex  = ConnectivityView(empty_idx.data(), index_type(0), 3);
    mc.nEdgesOnCell   = std::mdspan<const index_type,
        std::extents<index_type, std::dynamic_extent>>(empty_idx.data(), nCells);

    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        dvEdge(dvEdge_data.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invAreaCell(invAreaCell_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        rdzw(rdzw_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzm(fzm_data.data(), nVertLevels);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzp(fzp_data.data(), nVertLevels);

    ConnectivityView advCellsForEdge(
        advCellsForEdge_data.data(), nEdges, maxAdvCells);
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>
        nAdvCellsForEdge(nAdvCellsForEdge_data.data(), nEdges);

    ConstField2D<default_layout, unchecked_accessor> adv_coefs(
        adv_coefs_data.data(), maxAdvCells, nEdges);
    ConstField2D<default_layout, unchecked_accessor> adv_coefs_3rd(
        adv_coefs_3rd_data.data(), maxAdvCells, nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign(
        edgesOnCell_sign_data.data(), index_type(6), nCells);

    real_type dt = 10.0;
    real_type coef_3rd_order = 1.0;

    // Call should return immediately without error
    advance_scalars_mono<default_layout>(
        SerialPolicy{},
        scalars, ruAvg, wwAvg, rho_zz_old, rho_zz_new,
        mc, dvEdge, invAreaCell, rdzw, fzm, fzp,
        advCellsForEdge, nAdvCellsForEdge,
        adv_coefs, adv_coefs_3rd, edgesOnCell_sign,
        dt, coef_3rd_order,
        nScalars,
        nCells, nEdges, nVertLevels, maxAdvCells);

    // If we reach here without crash/abort, the test passes
    RC_SUCCEED("advance_scalars_mono returned gracefully with nCells=0");
}

// ============================================================================
// Test 3: nVertLevels=0 returns immediately without modifying data (Property 39)
//
// Call advance_scalars_mono with nVertLevels=0 to verify zero-extent graceful return.
// ============================================================================

RC_GTEST_PROP(ScalarAdvectionDisabled,
              ZeroVertLevelsLeavesDataUnchanged, ()) {
    // **Validates: Requirements 23.1, 23.7**

    const auto nCells = *rc::gen::inRange(4, 10);
    const index_type nVertLevels = 0;
    const auto nScalars = *rc::gen::inRange(1, 5);
    const index_type maxAdvCells = 6;

    // Use a valid mesh for edges/connectivity but pass nVertLevels=0
    auto sm = genValidMesh(nCells, 5);  // need valid mesh for structure
    auto mc = buildConnectivity(sm);
    const auto nEdges = sm.nEdges;

    const auto nE = static_cast<std::size_t>(nEdges);
    const auto nC = static_cast<std::size_t>(nCells);

    // Empty data arrays (zero vert levels means zero product in most arrays)
    std::vector<real_type> scalars_data;
    std::vector<real_type> ruAvg_data;
    std::vector<real_type> wwAvg_data(nC, 0.0);  // (0+1)*nCells = nCells for level 0
    std::vector<real_type> rho_zz_old_data;
    std::vector<real_type> rho_zz_new_data;
    std::vector<real_type> invAreaCell_data(nC);
    for (index_type c = 0; c < nCells; ++c) {
        invAreaCell_data[static_cast<std::size_t>(c)] =
            1.0 / sm.areaCell[static_cast<std::size_t>(c)];
    }
    std::vector<real_type> rdzw_data;
    std::vector<real_type> fzm_data;
    std::vector<real_type> fzp_data;
    std::vector<index_type> advCellsForEdge_data(
        nE * static_cast<std::size_t>(maxAdvCells), 0);
    std::vector<index_type> nAdvCellsForEdge_data(nE, 0);
    std::vector<real_type> adv_coefs_data(
        static_cast<std::size_t>(maxAdvCells) * nE, 0.0);
    std::vector<real_type> adv_coefs_3rd_data(
        static_cast<std::size_t>(maxAdvCells) * nE, 0.0);
    std::vector<real_type> edgesOnCell_sign_data(
        static_cast<std::size_t>(sm.maxEdges) * nC, 1.0);

    // Create mdspan views
    Field3D<default_layout, unchecked_accessor> scalars(
        scalars_data.data(), nScalars, nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> ruAvg(
        ruAvg_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> wwAvg(
        wwAvg_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_zz_old(
        rho_zz_old_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_zz_new(
        rho_zz_new_data.data(), nVertLevels, nCells);

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

    ConnectivityView advCellsForEdge(
        advCellsForEdge_data.data(), nEdges, maxAdvCells);
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>
        nAdvCellsForEdge(nAdvCellsForEdge_data.data(), nEdges);

    ConstField2D<default_layout, unchecked_accessor> adv_coefs(
        adv_coefs_data.data(), maxAdvCells, nEdges);
    ConstField2D<default_layout, unchecked_accessor> adv_coefs_3rd(
        adv_coefs_3rd_data.data(), maxAdvCells, nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign(
        edgesOnCell_sign_data.data(), sm.maxEdges, nCells);

    real_type dt = 10.0;
    real_type coef_3rd_order = 1.0;

    // Call should return immediately without error
    advance_scalars_mono<default_layout>(
        SerialPolicy{},
        scalars, ruAvg, wwAvg, rho_zz_old, rho_zz_new,
        mc, dvEdge, invAreaCell, rdzw, fzm, fzp,
        advCellsForEdge, nAdvCellsForEdge,
        adv_coefs, adv_coefs_3rd, edgesOnCell_sign,
        dt, coef_3rd_order,
        nScalars,
        nCells, nEdges, nVertLevels, maxAdvCells);

    // If we reach here without crash/abort, the test passes
    RC_SUCCEED("advance_scalars_mono returned gracefully with nVertLevels=0");
}
