/// @file prop_velocity_reconstruct.cpp
/// @brief Property-based tests for velocity reconstruction kernel.
///
/// **Validates: Requirements 17.1, 17.2**
///
/// Property 33: Zero Velocity Reconstructs to Zero
///   When u=0 on all edges, both u_cell and v_cell are zero.
///
/// Property 34: Linearity
///   Scaling input u by a factor scales output by the same factor.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/velocity_reconstruct.hpp>
#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include "../synthetic_mesh.hpp"
#include <vector>
#include <cmath>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;
using namespace mpas::dycore::testing;

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
// Property 33: Zero Velocity Reconstructs to Zero
// ============================================================================
// When u=0 on all edges, both u_cell and v_cell must be identically zero
// regardless of the reconstruction coefficients, because the reconstruction
// is a weighted sum of edge velocities.

RC_GTEST_PROP(VelocityReconstruction, ZeroVelocityReconstructsToZero,
              ()) {
    // Generate random mesh dimensions
    const auto nCells = *rc::gen::inRange(2, 12);
    const auto nVertLevels = *rc::gen::inRange(2, 15);

    SyntheticMesh sm = genValidMesh(nCells, nVertLevels);
    MeshConnectivity mc = buildConnectivity(sm);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(sm.nCells);
    const auto nE   = static_cast<std::size_t>(sm.nEdges);
    const auto maxE = static_cast<std::size_t>(sm.maxEdges);

    // u = 0 on all edges and levels
    std::vector<real_type> u_data(nLev * nE, 0.0);

    // Random reconstruction coefficients (arbitrary, nonzero values)
    std::vector<real_type> coeffs_u_data(maxE * nC);
    std::vector<real_type> coeffs_v_data(maxE * nC);
    for (std::size_t i = 0; i < maxE * nC; ++i) {
        int raw = *rc::gen::inRange(-1000, 1001);
        coeffs_u_data[i] = static_cast<real_type>(raw) / 1000.0;
        raw = *rc::gen::inRange(-1000, 1001);
        coeffs_v_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Output arrays initialized to sentinel values
    std::vector<real_type> u_cell_data(nLev * nC, -999.0);
    std::vector<real_type> v_cell_data(nLev * nC, -999.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> u_cell(
        u_cell_data.data(), nVertLevels, sm.nCells);
    Field2D<default_layout, unchecked_accessor> v_cell(
        v_cell_data.data(), nVertLevels, sm.nCells);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, sm.nEdges);
    ConstField2D<default_layout, unchecked_accessor> coeffs_reconstruct_u(
        coeffs_u_data.data(), sm.maxEdges, sm.nCells);
    ConstField2D<default_layout, unchecked_accessor> coeffs_reconstruct_v(
        coeffs_v_data.data(), sm.maxEdges, sm.nCells);

    // Call the kernel
    reconstruct_velocity<default_layout>(
        SerialPolicy{},
        u_cell,
        v_cell,
        u,
        mc,
        coeffs_reconstruct_u,
        coeffs_reconstruct_v,
        sm.nCells,
        static_cast<index_type>(nVertLevels));

    // Verify both u_cell and v_cell are exactly zero everywhere
    for (std::size_t i = 0; i < u_cell_data.size(); ++i) {
        RC_ASSERT(u_cell_data[i] == 0.0);
    }
    for (std::size_t i = 0; i < v_cell_data.size(); ++i) {
        RC_ASSERT(v_cell_data[i] == 0.0);
    }
}

// ============================================================================
// Property 34: Linearity
// ============================================================================
// Scaling the input velocity field u by a constant factor alpha must scale
// both u_cell and v_cell by the same factor. This follows from the
// reconstruction being a linear weighted sum:
//   reconstruct(alpha * u) = alpha * reconstruct(u)

RC_GTEST_PROP(VelocityReconstruction, LinearityScaling,
              ()) {
    // Generate random mesh dimensions
    const auto nCells = *rc::gen::inRange(2, 12);
    const auto nVertLevels = *rc::gen::inRange(2, 15);

    SyntheticMesh sm = genValidMesh(nCells, nVertLevels);
    MeshConnectivity mc = buildConnectivity(sm);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(sm.nCells);
    const auto nE   = static_cast<std::size_t>(sm.nEdges);
    const auto maxE = static_cast<std::size_t>(sm.maxEdges);

    // Generate random edge-normal velocities
    std::vector<real_type> u_data(nLev * nE);
    for (std::size_t i = 0; i < u_data.size(); ++i) {
        int raw = *rc::gen::inRange(-10000, 10001);
        u_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Generate a nonzero scaling factor alpha
    const int alpha_raw = *rc::gen::suchThat(
        rc::gen::inRange(-1000, 1001),
        [](int v) { return v != 0; });
    const real_type alpha = static_cast<real_type>(alpha_raw) / 100.0;

    // Scaled velocity: alpha * u
    std::vector<real_type> u_scaled_data(nLev * nE);
    for (std::size_t i = 0; i < u_data.size(); ++i) {
        u_scaled_data[i] = alpha * u_data[i];
    }

    // Random reconstruction coefficients
    std::vector<real_type> coeffs_u_data(maxE * nC);
    std::vector<real_type> coeffs_v_data(maxE * nC);
    for (std::size_t i = 0; i < maxE * nC; ++i) {
        int raw = *rc::gen::inRange(-1000, 1001);
        coeffs_u_data[i] = static_cast<real_type>(raw) / 1000.0;
        raw = *rc::gen::inRange(-1000, 1001);
        coeffs_v_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Output arrays for reconstruct(u)
    std::vector<real_type> u_cell_data(nLev * nC, 0.0);
    std::vector<real_type> v_cell_data(nLev * nC, 0.0);

    // Output arrays for reconstruct(alpha * u)
    std::vector<real_type> u_cell_scaled_data(nLev * nC, 0.0);
    std::vector<real_type> v_cell_scaled_data(nLev * nC, 0.0);

    // Create mdspan views
    ConstField2D<default_layout, unchecked_accessor> u_view(
        u_data.data(), nVertLevels, sm.nEdges);
    ConstField2D<default_layout, unchecked_accessor> u_scaled_view(
        u_scaled_data.data(), nVertLevels, sm.nEdges);
    ConstField2D<default_layout, unchecked_accessor> coeffs_reconstruct_u(
        coeffs_u_data.data(), sm.maxEdges, sm.nCells);
    ConstField2D<default_layout, unchecked_accessor> coeffs_reconstruct_v(
        coeffs_v_data.data(), sm.maxEdges, sm.nCells);

    Field2D<default_layout, unchecked_accessor> u_cell(
        u_cell_data.data(), nVertLevels, sm.nCells);
    Field2D<default_layout, unchecked_accessor> v_cell(
        v_cell_data.data(), nVertLevels, sm.nCells);
    Field2D<default_layout, unchecked_accessor> u_cell_scaled(
        u_cell_scaled_data.data(), nVertLevels, sm.nCells);
    Field2D<default_layout, unchecked_accessor> v_cell_scaled(
        v_cell_scaled_data.data(), nVertLevels, sm.nCells);

    // Call reconstruct(u)
    reconstruct_velocity<default_layout>(
        SerialPolicy{},
        u_cell,
        v_cell,
        u_view,
        mc,
        coeffs_reconstruct_u,
        coeffs_reconstruct_v,
        sm.nCells,
        static_cast<index_type>(nVertLevels));

    // Call reconstruct(alpha * u)
    reconstruct_velocity<default_layout>(
        SerialPolicy{},
        u_cell_scaled,
        v_cell_scaled,
        u_scaled_view,
        mc,
        coeffs_reconstruct_u,
        coeffs_reconstruct_v,
        sm.nCells,
        static_cast<index_type>(nVertLevels));

    // Verify: reconstruct(alpha*u) == alpha * reconstruct(u)
    // Use relative tolerance for floating-point comparison
    const real_type tol = 1.0e-12;
    for (std::size_t i = 0; i < u_cell_data.size(); ++i) {
        const real_type expected = alpha * u_cell_data[i];
        const real_type actual = u_cell_scaled_data[i];
        const real_type diff = std::abs(actual - expected);
        const real_type scale = std::max(1.0, std::abs(expected));
        RC_ASSERT(diff / scale < tol);
    }
    for (std::size_t i = 0; i < v_cell_data.size(); ++i) {
        const real_type expected = alpha * v_cell_data[i];
        const real_type actual = v_cell_scaled_data[i];
        const real_type diff = std::abs(actual - expected);
        const real_type scale = std::max(1.0, std::abs(expected));
        RC_ASSERT(diff / scale < tol);
    }
}
