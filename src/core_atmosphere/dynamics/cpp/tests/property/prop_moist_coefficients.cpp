/// @file prop_moist_coefficients.cpp
/// @brief Property-based tests for the moist coefficient computation kernel.
///
/// **Validates: Requirements 15.1, 15.2, 15.3, 15.5**
///
/// Property 31: Moist Coefficients with No Moisture
///   When moist_end < moist_start (no moisture species), all cqw and cqu values
///   are exactly 1.0.
///
/// Property 32: Moist Coefficient Formula Verification
///   For known scalar values, the computed cqw and cqu match the expected
///   formulae: cq_cell = 1/(1+sum(q_s)), cqw = vertical average, cqu = horizontal average.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/moist_coefficients.hpp>
#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include "../synthetic_mesh.hpp"
#include <vector>
#include <cmath>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;
using namespace mpas::dycore::testing;

// Use the sentinel from the main dycore namespace explicitly
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
// Property 31: Moist Coefficients with No Moisture
// ============================================================================
// When moist_end < moist_start, all cqw and cqu are exactly 1.0.

RC_GTEST_PROP(MoistCoefficientsNoMoisture, AllValuesAreOneWhenNoMoistureSpecies,
              ()) {
    // Generate random mesh dimensions
    const auto nCells = *rc::gen::inRange(2, 15);
    const auto nEdges = *rc::gen::inRange(3, 40);
    const auto nVertLevels = *rc::gen::inRange(2, 20);

    // Allocate output arrays
    // cqw: (nVertLevels+1, nCells) — interfaces
    std::vector<real_type> cqw_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), 0.0);
    // cqu: (nVertLevels, nEdges)
    std::vector<real_type> cqu_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 0.0);

    // Create mdspan views (layout_left for column-major)
    Field2D<default_layout, unchecked_accessor> cqw(
        cqw_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> cqu(
        cqu_data.data(), nVertLevels, nEdges);

    // Create dummy scalars (won't be accessed since no moisture species)
    std::vector<real_type> scalar_data(1, 0.0);
    ConstField3D<default_layout, unchecked_accessor> scalars(
        scalar_data.data(), 1, 1, 1);

    // Dummy mesh connectivity (not accessed in no-moisture path)
    std::vector<index_type> cellsOnEdge_data(
        static_cast<std::size_t>(nEdges) * 2, 0);
    MeshConnectivity mesh;
    mesh.cellsOnEdge = ConnectivityView(cellsOnEdge_data.data(), nEdges, 2);

    // Call with moist_end=0, moist_start=1 (end < start: no moisture)
    index_type moist_start = 1;
    index_type moist_end = 0;

    compute_moist_coefficients<default_layout>(
        SerialPolicy{}, cqw, cqu, scalars, mesh,
        moist_start, moist_end,
        static_cast<index_type>(nCells),
        static_cast<index_type>(nEdges),
        static_cast<index_type>(nVertLevels));

    // Verify ALL cqw values == 1.0 exactly
    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        for (index_type k = 0; k <= static_cast<index_type>(nVertLevels); ++k) {
            auto val = cqw[k, iCell];
            RC_ASSERT(val == 1.0);
        }
    }

    // Verify ALL cqu values == 1.0 exactly
    for (index_type iEdge = 0; iEdge < static_cast<index_type>(nEdges); ++iEdge) {
        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            auto val = cqu[k, iEdge];
            RC_ASSERT(val == 1.0);
        }
    }
}

// ============================================================================
// Property 32: Moist Coefficient Formula Verification
// ============================================================================
// For known scalar values, verify computed coefficients match the expected
// formula: cq_cell = 1/(1 + sum(q_s for s in [moist_start, moist_end]))
// cqw at interior interfaces = 0.5*(cq_cell[k-1] + cq_cell[k])
// cqw at top boundary = cq_cell[0], cqw at bottom = cq_cell[nVertLevels-1]
// cqu at interior edges = 0.5*(cq_cell of cell0 + cq_cell of cell1)

RC_GTEST_PROP(MoistCoefficientFormula, ComputedCoefficientsMatchExpectedFormula,
              ()) {
    // Use a small synthetic mesh
    const int meshCells = *rc::gen::inRange(4, 10);
    const int nVertLevels = *rc::gen::inRange(3, 12);

    auto synMesh = genValidMesh(meshCells, nVertLevels);
    auto mc = buildConnectivity(synMesh);

    const auto nCells = synMesh.nCells;
    const auto nEdges = synMesh.nEdges;

    // Set up scalars with random moisture mixing ratios
    // Use 2-4 moisture species in the range [0, 0.05]
    const int nMoistSpecies = *rc::gen::inRange(2, 5);
    const index_type moist_start = 0;
    const index_type moist_end = static_cast<index_type>(nMoistSpecies - 1);
    const index_type nScalars = static_cast<index_type>(nMoistSpecies);

    // Allocate scalars: (nScalars, nVertLevels, nCells) in layout_left
    std::vector<real_type> scalar_data(
        static_cast<std::size_t>(nScalars) *
        static_cast<std::size_t>(nVertLevels) *
        static_cast<std::size_t>(nCells));

    // Fill with random values in [0, 0.05]
    for (std::size_t i = 0; i < scalar_data.size(); ++i) {
        // Generate integer in [0, 5000] then scale to [0.0, 0.05]
        int raw = *rc::gen::inRange(0, 5001);
        scalar_data[i] = static_cast<real_type>(raw) / 100000.0;
    }

    ConstField3D<default_layout, unchecked_accessor> scalars(
        scalar_data.data(), nScalars, nVertLevels, nCells);

    // Allocate output arrays
    std::vector<real_type> cqw_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), 0.0);
    std::vector<real_type> cqu_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 0.0);

    Field2D<default_layout, unchecked_accessor> cqw(
        cqw_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> cqu(
        cqu_data.data(), nVertLevels, nEdges);

    // Call the kernel
    compute_moist_coefficients<default_layout>(
        SerialPolicy{}, cqw, cqu, scalars, mc,
        moist_start, moist_end,
        nCells, nEdges,
        static_cast<index_type>(nVertLevels));

    // Compute expected cq_cell values manually
    std::vector<real_type> cq_cell(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells));

    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            real_type sum = 0.0;
            for (index_type s = moist_start; s <= moist_end; ++s) {
                sum += scalars[s, k, iCell];
            }
            real_type expected = 1.0 / (1.0 + sum);
            cq_cell[static_cast<std::size_t>(k) * static_cast<std::size_t>(nCells)
                    + static_cast<std::size_t>(iCell)] = expected;
        }
    }

    // Helper to access cq_cell
    auto getCqCell = [&](index_type k, index_type iCell) -> real_type {
        return cq_cell[static_cast<std::size_t>(k) * static_cast<std::size_t>(nCells)
                       + static_cast<std::size_t>(iCell)];
    };

    const real_type tol = 1.0e-14;

    // Verify cqw at interfaces
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        // Top boundary: cqw[0, iCell] == cq_cell[0, iCell]
        {
            auto val = cqw[0, iCell];
            RC_ASSERT(std::abs(val - getCqCell(0, iCell)) < tol);
        }

        // Interior interfaces: cqw[k, iCell] == 0.5*(cq_cell[k-1] + cq_cell[k])
        for (index_type k = 1; k < static_cast<index_type>(nVertLevels); ++k) {
            real_type expected_cqw = 0.5 * (getCqCell(k - 1, iCell) + getCqCell(k, iCell));
            auto val = cqw[k, iCell];
            RC_ASSERT(std::abs(val - expected_cqw) < tol);
        }

        // Bottom boundary: cqw[nVertLevels, iCell] == cq_cell[nVertLevels-1, iCell]
        {
            auto val = cqw[static_cast<index_type>(nVertLevels), iCell];
            RC_ASSERT(std::abs(val - getCqCell(static_cast<index_type>(nVertLevels) - 1, iCell)) < tol);
        }
    }

    // Verify cqu at edges
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        // Read cellsOnEdge from the mesh connectivity
        index_type cell0 = mc.cellsOnEdge[iEdge, 0];
        index_type cell1 = mc.cellsOnEdge[iEdge, 1];

        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            real_type expected_cqu;
            if (cell1 == SENTINEL) {
                // Boundary edge: use single adjacent cell
                expected_cqu = getCqCell(k, cell0);
            } else {
                // Interior edge: average of two cells
                expected_cqu = 0.5 * (getCqCell(k, cell0) + getCqCell(k, cell1));
            }
            auto val = cqu[k, iEdge];
            RC_ASSERT(std::abs(val - expected_cqu) < tol);
        }
    }
}
