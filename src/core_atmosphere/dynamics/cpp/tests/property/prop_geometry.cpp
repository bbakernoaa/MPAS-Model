/// @file prop_geometry.cpp
/// @brief Property-based tests for MeshGeometry storage sizing consistency.
///
/// **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**
///
/// Property 8: For any valid mesh dimensions, all storage vectors in MeshGeometry
/// have sizes consistent with the documented dimension products after populate().

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/geometry.hpp>

#include <vector>
#include <cstddef>
#include <stdexcept>

using namespace mpas::dycore;

// ============================================================================
// Helper: populate a MeshGeometry with dummy arrays of the expected sizes
// ============================================================================

/// Populate a MeshGeometry struct with constant-filled dummy arrays of the
/// correct sizes for the given dimensions.
static void populate_with_dummy(
    MeshGeometry& geom,
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels,
    index_type maxEdges,
    index_type maxEdges2,
    index_type maxAdvCells)
{
    const auto nc  = static_cast<std::size_t>(nCells);
    const auto ne  = static_cast<std::size_t>(nEdges);
    const auto nv  = static_cast<std::size_t>(nVertices);
    const auto nvl = static_cast<std::size_t>(nVertLevels);
    const auto me  = static_cast<std::size_t>(maxEdges);
    const auto me2 = static_cast<std::size_t>(maxEdges2);
    const auto mac = static_cast<std::size_t>(maxAdvCells);

    // Fill real arrays with 1.0, index arrays with 0
    const real_type rv = 1.0;
    const index_type iv = 0;

    std::vector<real_type> areaCell(nc, rv);
    std::vector<real_type> invAreaCell(nc, rv);
    std::vector<real_type> dvEdge(ne, rv);
    std::vector<real_type> dcEdge(ne, rv);
    std::vector<real_type> invDcEdge(ne, rv);
    std::vector<real_type> rdzw(nvl, rv);
    std::vector<real_type> rdzu(nvl, rv);
    std::vector<real_type> fzm(nvl, rv);
    std::vector<real_type> fzp(nvl, rv);
    std::vector<real_type> etp(nvl, rv);
    std::vector<real_type> etm(nvl, rv);
    std::vector<real_type> ewp(nvl + 1, rv);
    std::vector<real_type> ewm(nvl + 1, rv);
    std::vector<real_type> zz(nvl * nc, rv);
    std::vector<real_type> rb(nvl * nc, rv);
    std::vector<real_type> rtb(nvl * nc, rv);
    std::vector<real_type> pb(nvl * nc, rv);
    std::vector<real_type> edgesOnCell_sign(me * nc, rv);
    std::vector<real_type> specZoneMaskEdge(ne, rv);
    std::vector<real_type> specZoneMaskCell(nc, rv);
    std::vector<real_type> weightsOnEdge(ne * me2, rv);
    std::vector<index_type> nEdgesOnEdge(ne, iv);
    std::vector<index_type> edgesOnEdge(ne * me2, iv);
    std::vector<index_type> advCellsForEdge(ne * mac, iv);
    std::vector<index_type> nAdvCellsForEdge(ne, iv);
    std::vector<real_type> adv_coefs(mac * ne, rv);
    std::vector<real_type> adv_coefs_3rd(mac * ne, rv);
    std::vector<real_type> fVertex(nv, rv);
    std::vector<real_type> areaTriangle(nv, rv);

    // New terrain correction and base-state arrays
    const auto nvlp1 = nvl + 1;
    std::vector<real_type> zb_cell(nvlp1 * me * nc, rv);
    std::vector<real_type> zb3_cell(nvlp1 * me * nc, rv);
    std::vector<real_type> rho_base(nvl * nc, rv);
    std::vector<real_type> rtheta_base(nvl * nc, rv);
    std::vector<real_type> exner_base(nvl * nc, rv);
    std::vector<index_type> nEdgesOnCell_vec(nc, iv);

    geom.populate(
        areaCell.data(), invAreaCell.data(),
        dvEdge.data(), dcEdge.data(), invDcEdge.data(),
        rdzw.data(), rdzu.data(), fzm.data(), fzp.data(),
        etp.data(), etm.data(), ewp.data(), ewm.data(),
        zz.data(), rb.data(), rtb.data(), pb.data(),
        edgesOnCell_sign.data(),
        specZoneMaskEdge.data(), specZoneMaskCell.data(),
        weightsOnEdge.data(),
        nEdgesOnEdge.data(), edgesOnEdge.data(),
        advCellsForEdge.data(), nAdvCellsForEdge.data(),
        adv_coefs.data(), adv_coefs_3rd.data(),
        fVertex.data(), areaTriangle.data(),
        zb_cell.data(), zb3_cell.data(),
        rho_base.data(), rtheta_base.data(), exner_base.data(),
        nEdgesOnCell_vec.data(),
        rv, rv, rv,  // cf1, cf2, cf3
        nCells, nEdges, nVertices, nVertLevels,
        maxEdges, maxEdges2, maxAdvCells);
}

// ============================================================================
// Property 8: MeshGeometry storage sizing consistency
// ============================================================================

RC_GTEST_PROP(MeshGeometrySizing, AllVectorSizesMatchDocumentedProducts,
              ()) {
    // Generate random valid dimensions
    const auto nCells      = *rc::gen::inRange(1, 101);
    const auto nEdges      = *rc::gen::inRange(1, 301);
    const auto nVertices   = *rc::gen::inRange(1, 201);
    const auto nVertLevels = *rc::gen::inRange(1, 51);
    const auto maxEdges    = *rc::gen::inRange(3, 9);
    const auto maxEdges2   = *rc::gen::inRange(3, 21);
    const auto maxAdvCells = *rc::gen::inRange(1, 16);

    const auto nc  = static_cast<std::size_t>(nCells);
    const auto ne  = static_cast<std::size_t>(nEdges);
    const auto nv  = static_cast<std::size_t>(nVertices);
    const auto nvl = static_cast<std::size_t>(nVertLevels);
    const auto me  = static_cast<std::size_t>(maxEdges);
    const auto me2 = static_cast<std::size_t>(maxEdges2);
    const auto mac = static_cast<std::size_t>(maxAdvCells);

    MeshGeometry geom;
    populate_with_dummy(geom, nCells, nEdges, nVertices, nVertLevels,
                        maxEdges, maxEdges2, maxAdvCells);

    // Verify populated flag
    RC_ASSERT(geom.populated());

    // Cell geometry (Requirement 6.1)
    RC_ASSERT(geom.areaCell.size() == nc);
    RC_ASSERT(geom.invAreaCell.size() == nc);

    // Edge geometry (Requirement 6.1)
    RC_ASSERT(geom.dvEdge.size() == ne);
    RC_ASSERT(geom.dcEdge.size() == ne);
    RC_ASSERT(geom.invDcEdge.size() == ne);

    // Vertical metrics (Requirement 6.2)
    RC_ASSERT(geom.rdzw.size() == nvl);
    RC_ASSERT(geom.rdzu.size() == nvl);
    RC_ASSERT(geom.fzm.size() == nvl);
    RC_ASSERT(geom.fzp.size() == nvl);
    RC_ASSERT(geom.etp.size() == nvl);
    RC_ASSERT(geom.etm.size() == nvl);
    RC_ASSERT(geom.ewp.size() == nvl + 1);
    RC_ASSERT(geom.ewm.size() == nvl + 1);

    // Terrain metric (Requirement 6.4)
    RC_ASSERT(geom.zz.size() == nvl * nc);

    // Base-state profiles (Requirement 6.3)
    RC_ASSERT(geom.rb.size() == nvl * nc);
    RC_ASSERT(geom.rtb.size() == nvl * nc);
    RC_ASSERT(geom.pb.size() == nvl * nc);

    // Edge orientation signs (Requirement 6.7)
    RC_ASSERT(geom.edgesOnCell_sign.size() == me * nc);

    // Specified zone masks (Requirement 6.7)
    RC_ASSERT(geom.specZoneMaskEdge.size() == ne);
    RC_ASSERT(geom.specZoneMaskCell.size() == nc);

    // Edge reconstruction data - TRiSK (Requirement 6.5)
    RC_ASSERT(geom.weightsOnEdge.size() == ne * me2);
    RC_ASSERT(geom.nEdgesOnEdge.size() == ne);
    RC_ASSERT(geom.edgesOnEdge_storage.size() == ne * me2);

    // Advection stencils (Requirement 6.6)
    RC_ASSERT(geom.advCellsForEdge_storage.size() == ne * mac);
    RC_ASSERT(geom.nAdvCellsForEdge.size() == ne);
    RC_ASSERT(geom.adv_coefs.size() == mac * ne);
    RC_ASSERT(geom.adv_coefs_3rd.size() == mac * ne);

    // Vertex geometry (Requirement 6.1)
    RC_ASSERT(geom.fVertex.size() == nv);
    RC_ASSERT(geom.areaTriangle.size() == nv);

    // Dimension members
    RC_ASSERT(geom.maxEdges2 == maxEdges2);
    RC_ASSERT(geom.maxAdvCells == maxAdvCells);

    // validate_sizes() should return true for matching dimensions
    RC_ASSERT(geom.validate_sizes(nCells, nEdges, nVertices, nVertLevels,
                                  maxEdges2, maxAdvCells));
}

// ============================================================================
// Property: validate_sizes() returns false for mismatched dimensions
// ============================================================================

RC_GTEST_PROP(MeshGeometrySizing, ValidateSizesReturnsFalseForMismatch,
              ()) {
    // Generate random valid dimensions
    const auto nCells      = *rc::gen::inRange(1, 101);
    const auto nEdges      = *rc::gen::inRange(1, 301);
    const auto nVertices   = *rc::gen::inRange(1, 201);
    const auto nVertLevels = *rc::gen::inRange(1, 51);
    const auto maxEdges    = *rc::gen::inRange(3, 9);
    const auto maxEdges2   = *rc::gen::inRange(3, 21);
    const auto maxAdvCells = *rc::gen::inRange(1, 16);

    MeshGeometry geom;
    populate_with_dummy(geom, nCells, nEdges, nVertices, nVertLevels,
                        maxEdges, maxEdges2, maxAdvCells);

    // Perturb one dimension — validate_sizes() should return false
    // Pick a random dimension to perturb
    const auto which = *rc::gen::inRange(0, 6);
    index_type nc2 = nCells, ne2 = nEdges, nv2 = nVertices;
    index_type nvl2 = nVertLevels, me2_2 = maxEdges2, mac2 = maxAdvCells;

    switch (which) {
        case 0: nc2 += 1; break;
        case 1: ne2 += 1; break;
        case 2: nv2 += 1; break;
        case 3: nvl2 += 1; break;
        case 4: me2_2 += 1; break;
        case 5: mac2 += 1; break;
    }

    RC_ASSERT(!geom.validate_sizes(nc2, ne2, nv2, nvl2, me2_2, mac2));
}

// ============================================================================
// Property: Invalid dimensions (any <= 0) throw std::invalid_argument
// ============================================================================

RC_GTEST_PROP(MeshGeometrySizing, InvalidDimensionsThrow,
              ()) {
    // Generate valid dimensions as baseline
    const auto nCells      = *rc::gen::inRange(1, 101);
    const auto nEdges      = *rc::gen::inRange(1, 301);
    const auto nVertices   = *rc::gen::inRange(1, 201);
    const auto nVertLevels = *rc::gen::inRange(1, 51);
    const auto maxEdges    = *rc::gen::inRange(3, 9);
    const auto maxEdges2   = *rc::gen::inRange(3, 21);
    const auto maxAdvCells = *rc::gen::inRange(1, 16);

    // Pick a random dimension to make invalid (<= 0)
    const auto which = *rc::gen::inRange(0, 7);
    index_type nc = nCells, ne = nEdges, nv = nVertices;
    index_type nvl = nVertLevels, me = maxEdges, me2 = maxEdges2, mac = maxAdvCells;

    const auto bad_val = *rc::gen::inRange(-10, 1); // values in [-10, 0]

    switch (which) {
        case 0: nc = bad_val; break;
        case 1: ne = bad_val; break;
        case 2: nv = bad_val; break;
        case 3: nvl = bad_val; break;
        case 4: me = bad_val; break;
        case 5: me2 = bad_val; break;
        case 6: mac = bad_val; break;
    }

    MeshGeometry geom;

    // We need to create dummy arrays large enough to not segfault if any
    // dimension happens to be valid. Use absolute values + 1 for safety.
    const auto safe_nc  = static_cast<std::size_t>(std::abs(nc) + 1);
    const auto safe_ne  = static_cast<std::size_t>(std::abs(ne) + 1);
    const auto safe_nv  = static_cast<std::size_t>(std::abs(nv) + 1);
    const auto safe_nvl = static_cast<std::size_t>(std::abs(nvl) + 1);
    const auto safe_me  = static_cast<std::size_t>(std::abs(me) + 1);
    const auto safe_me2 = static_cast<std::size_t>(std::abs(me2) + 1);
    const auto safe_mac = static_cast<std::size_t>(std::abs(mac) + 1);

    const real_type rv = 1.0;
    const index_type iv = 0;

    std::vector<real_type> areaCell(safe_nc, rv);
    std::vector<real_type> invAreaCell(safe_nc, rv);
    std::vector<real_type> dvEdge(safe_ne, rv);
    std::vector<real_type> dcEdge(safe_ne, rv);
    std::vector<real_type> invDcEdge(safe_ne, rv);
    std::vector<real_type> rdzw(safe_nvl, rv);
    std::vector<real_type> rdzu(safe_nvl, rv);
    std::vector<real_type> fzm(safe_nvl, rv);
    std::vector<real_type> fzp(safe_nvl, rv);
    std::vector<real_type> etp(safe_nvl, rv);
    std::vector<real_type> etm(safe_nvl, rv);
    std::vector<real_type> ewp(safe_nvl + 1, rv);
    std::vector<real_type> ewm(safe_nvl + 1, rv);
    std::vector<real_type> zz(safe_nvl * safe_nc, rv);
    std::vector<real_type> rb(safe_nvl * safe_nc, rv);
    std::vector<real_type> rtb(safe_nvl * safe_nc, rv);
    std::vector<real_type> pb(safe_nvl * safe_nc, rv);
    std::vector<real_type> edgesOnCell_sign(safe_me * safe_nc, rv);
    std::vector<real_type> specZoneMaskEdge(safe_ne, rv);
    std::vector<real_type> specZoneMaskCell(safe_nc, rv);
    std::vector<real_type> weightsOnEdge(safe_ne * safe_me2, rv);
    std::vector<index_type> nEdgesOnEdge(safe_ne, iv);
    std::vector<index_type> edgesOnEdge(safe_ne * safe_me2, iv);
    std::vector<index_type> advCellsForEdge(safe_ne * safe_mac, iv);
    std::vector<index_type> nAdvCellsForEdge(safe_ne, iv);
    std::vector<real_type> adv_coefs(safe_mac * safe_ne, rv);
    std::vector<real_type> adv_coefs_3rd(safe_mac * safe_ne, rv);
    std::vector<real_type> fVertex(safe_nv, rv);
    std::vector<real_type> areaTriangle(safe_nv, rv);

    RC_ASSERT_THROWS_AS(
        geom.populate(
            areaCell.data(), invAreaCell.data(),
            dvEdge.data(), dcEdge.data(), invDcEdge.data(),
            rdzw.data(), rdzu.data(), fzm.data(), fzp.data(),
            etp.data(), etm.data(), ewp.data(), ewm.data(),
            zz.data(), rb.data(), rtb.data(), pb.data(),
            edgesOnCell_sign.data(),
            specZoneMaskEdge.data(), specZoneMaskCell.data(),
            weightsOnEdge.data(),
            nEdgesOnEdge.data(), edgesOnEdge.data(),
            advCellsForEdge.data(), nAdvCellsForEdge.data(),
            adv_coefs.data(), adv_coefs_3rd.data(),
            fVertex.data(), areaTriangle.data(),
            nullptr, nullptr,  // zb_cell, zb3_cell (optional)
            nullptr, nullptr, nullptr,  // rho_base, rtheta_base, exner_base (optional)
            nullptr,  // nEdgesOnCell (optional)
            0.0, 0.0, 0.0,  // cf1, cf2, cf3
            nc, ne, nv, nvl, me, me2, mac),
        std::invalid_argument);
}

// ============================================================================
// Property: Null pointer throws std::invalid_argument
// ============================================================================

TEST(MeshGeometrySizing, NullPointerThrows) {
    MeshGeometry geom;

    const index_type nCells = 10;
    const index_type nEdges = 30;
    const index_type nVertices = 20;
    const index_type nVertLevels = 5;
    const index_type maxEdges = 6;
    const index_type maxEdges2 = 12;
    const index_type maxAdvCells = 10;

    // Passing nullptr for the first argument should throw
    EXPECT_THROW(
        geom.populate(
            nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr,  // zb_cell, zb3_cell (optional)
            nullptr, nullptr, nullptr,  // rho_base, rtheta_base, exner_base (optional)
            nullptr,  // nEdgesOnCell (optional)
            0.0, 0.0, 0.0,  // cf1, cf2, cf3
            nCells, nEdges, nVertices, nVertLevels,
            maxEdges, maxEdges2, maxAdvCells),
        std::invalid_argument);
}

// ============================================================================
// Deterministic test: populated() is false before populate() is called
// ============================================================================

TEST(MeshGeometrySizing, NotPopulatedByDefault) {
    MeshGeometry geom;
    EXPECT_FALSE(geom.populated());
}

// ============================================================================
// Deterministic test: populated() returns true after successful populate()
// ============================================================================

TEST(MeshGeometrySizing, PopulatedAfterPopulate) {
    MeshGeometry geom;

    const index_type nCells = 5;
    const index_type nEdges = 15;
    const index_type nVertices = 10;
    const index_type nVertLevels = 3;
    const index_type maxEdges = 6;
    const index_type maxEdges2 = 12;
    const index_type maxAdvCells = 8;

    populate_with_dummy(geom, nCells, nEdges, nVertices, nVertLevels,
                        maxEdges, maxEdges2, maxAdvCells);

    EXPECT_TRUE(geom.populated());
}
