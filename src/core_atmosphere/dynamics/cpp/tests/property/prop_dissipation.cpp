/// @file prop_dissipation.cpp
/// @brief Property-based tests for dissipation model kernels.
///
/// **Validates: Requirements 19.2, 19.6**
///
/// Property 27: Dissipation Monotonicity
///   Del-2 diffusion reduces the L2 norm of the diffused quantity.
///
/// Property 28: Fixed Viscosity Uniformity
///   compute_2d_fixed_viscosity produces uniform values.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/dissipation.hpp>
#include <mpas_dycore/kernels/les_models.hpp>
#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include "../synthetic_mesh.hpp"
#include <vector>
#include <cmath>
#include <numeric>

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
// Property 28: Fixed Viscosity Uniformity
// ============================================================================
// Call compute_2d_fixed_viscosity with a known constant. Verify ALL output
// values equal that constant.

RC_GTEST_PROP(FixedViscosityUniformity, AllOutputsEqualConstant,
              ()) {
    // Generate random dimensions
    const auto nCells = *rc::gen::inRange(1, 20);
    const auto nVertLevels = *rc::gen::inRange(1, 30);

    // Generate a random viscosity constant (positive, physically reasonable)
    const int visc_raw = *rc::gen::inRange(1, 100000);
    const real_type config_h_theta_eddy_visc2 = static_cast<real_type>(visc_raw) / 100.0;

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Allocate output array initialized to an obviously wrong value
    std::vector<real_type> eddy_visc_data(nLev * nC, -999.0);

    // Create mdspan view
    Field2D<default_layout, unchecked_accessor> eddy_visc_horz(
        eddy_visc_data.data(), nVertLevels, nCells);

    // Call the kernel
    compute_2d_fixed_viscosity<default_layout>(
        SerialPolicy{},
        eddy_visc_horz,
        config_h_theta_eddy_visc2,
        static_cast<index_type>(nCells),
        static_cast<index_type>(nVertLevels));

    // Verify ALL output values equal the constant
    for (std::size_t i = 0; i < eddy_visc_data.size(); ++i) {
        RC_ASSERT(eddy_visc_data[i] == config_h_theta_eddy_visc2);
    }
}

// ============================================================================
// Property 27 (partial): Smagorinsky with zero velocity gives zero viscosity
// ============================================================================
// When u=0 and v=0 everywhere, compute_2d_smagorinsky should produce
// eddy_visc_horz=0 everywhere (the strain rate is zero, so |S|=0 and
// viscosity = (c_s * len)^2 * 0 = 0).

RC_GTEST_PROP(SmagorinskyZeroVelocity, ProducesZeroViscosity,
              ()) {
    // Generate a random valid mesh
    const auto nCells = *rc::gen::inRange(2, 12);
    const auto nVertLevels = *rc::gen::inRange(2, 15);

    SyntheticMesh sm = genValidMesh(nCells, nVertLevels);
    MeshConnectivity mc = buildConnectivity(sm);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(sm.nCells);
    const auto nE   = static_cast<std::size_t>(sm.nEdges);
    const auto maxE = static_cast<std::size_t>(sm.maxEdges);

    // u = 0, v = 0 everywhere
    std::vector<real_type> u_data(nLev * nE, 0.0);
    std::vector<real_type> v_data(nLev * nE, 0.0);

    // Deformation coefficients: generate arbitrary nonzero values
    // (doesn't matter because u=v=0 means all gradient contributions are zero)
    std::vector<real_type> coef_c2_data(maxE * nC, 0.0);
    std::vector<real_type> coef_s2_data(maxE * nC, 0.0);
    std::vector<real_type> coef_cs_data(maxE * nC, 0.0);
    for (std::size_t i = 0; i < maxE * nC; ++i) {
        int raw = *rc::gen::inRange(-1000, 1001);
        coef_c2_data[i] = static_cast<real_type>(raw) / 1000.0;
        raw = *rc::gen::inRange(-1000, 1001);
        coef_s2_data[i] = static_cast<real_type>(raw) / 1000.0;
        raw = *rc::gen::inRange(-1000, 1001);
        coef_cs_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Random positive Smagorinsky parameters
    const real_type c_s = 0.1 + (*rc::gen::inRange(0, 900)) / 1000.0;
    const real_type config_len_disp = 1000.0 + (*rc::gen::inRange(0, 9000));
    const real_type invDt = 1.0 / (10.0 + (*rc::gen::inRange(0, 100)));

    // Output: initialize to nonzero to detect that the kernel actually wrote
    std::vector<real_type> eddy_visc_data(nLev * nC, -999.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> eddy_visc_horz(
        eddy_visc_data.data(), nVertLevels, sm.nCells);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, sm.nEdges);
    ConstField2D<default_layout, unchecked_accessor> v(
        v_data.data(), nVertLevels, sm.nEdges);

    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_c2(
        coef_c2_data.data(), sm.maxEdges, sm.nCells);
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_s2(
        coef_s2_data.data(), sm.maxEdges, sm.nCells);
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_cs(
        coef_cs_data.data(), sm.maxEdges, sm.nCells);

    // Call the Smagorinsky kernel
    compute_2d_smagorinsky<default_layout>(
        SerialPolicy{},
        eddy_visc_horz,
        u, v,
        c_s,
        config_len_disp,
        invDt,
        mc,
        deformation_coef_c2,
        deformation_coef_s2,
        deformation_coef_cs,
        sm.nCells,
        sm.nEdges,
        static_cast<index_type>(nVertLevels));

    // Verify all output values are exactly zero
    for (std::size_t i = 0; i < eddy_visc_data.size(); ++i) {
        RC_ASSERT(eddy_visc_data[i] == 0.0);
    }
}

// ============================================================================
// Property 27 (partial): Del-2 of uniform field is zero
// ============================================================================
// When the input scalar field is constant everywhere, compute_scalar_del2
// should produce exactly zero output. This is because the Laplacian of a
// constant is zero: for all edges, (scalar[cell2] - scalar[cell1]) = 0.

RC_GTEST_PROP(Del2UniformFieldIsZero, UniformScalarProducesZeroDel2,
              ()) {
    // Generate a random valid mesh
    const auto nCells = *rc::gen::inRange(2, 12);
    const auto nVertLevels = *rc::gen::inRange(2, 15);

    SyntheticMesh sm = genValidMesh(nCells, nVertLevels);
    MeshConnectivity mc = buildConnectivity(sm);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(sm.nCells);
    const auto nE   = static_cast<std::size_t>(sm.nEdges);
    const auto maxE = static_cast<std::size_t>(sm.maxEdges);

    // Uniform scalar field: all cells have the same value
    const int scalar_raw = *rc::gen::inRange(-10000, 10001);
    const real_type scalar_val = static_cast<real_type>(scalar_raw) / 100.0;
    std::vector<real_type> scalar_data(nLev * nC, scalar_val);

    // rho_edge: arbitrary positive values (nonzero so we aren't trivially zero)
    std::vector<real_type> rho_edge_data(nLev * nE);
    for (std::size_t i = 0; i < rho_edge_data.size(); ++i) {
        int raw = *rc::gen::inRange(100, 2000);
        rho_edge_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // edgesOnCell_sign: use +1.0 or -1.0 (normal sign convention)
    // For the property to hold, we just need consistent signs.
    // Use +1.0 for all (signs cancel out when scalar difference is zero).
    std::vector<real_type> edgesOnCell_sign_data(maxE * nC, 1.0);

    // dvEdge and invDcEdge from the mesh geometry
    std::vector<real_type> dvEdge_data(nE);
    std::vector<real_type> invDcEdge_data(nE);
    std::vector<real_type> invAreaCell_data(nC);
    for (std::size_t i = 0; i < nE; ++i) {
        dvEdge_data[i] = sm.dvEdge[i];
        invDcEdge_data[i] = 1.0 / sm.dcEdge[i];
    }
    for (std::size_t i = 0; i < nC; ++i) {
        invAreaCell_data[i] = 1.0 / sm.areaCell[i];
    }

    // Output: initialize to nonzero sentinel
    std::vector<real_type> delsq_scalar_data(nLev * nC, -999.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> delsq_scalar(
        delsq_scalar_data.data(), nVertLevels, sm.nCells);
    ConstField2D<default_layout, unchecked_accessor> scalar(
        scalar_data.data(), nVertLevels, sm.nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(
        rho_edge_data.data(), nVertLevels, sm.nEdges);
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign(
        edgesOnCell_sign_data.data(), sm.maxEdges, sm.nCells);
    Field1D dvEdge(dvEdge_data.data(), sm.nEdges);
    Field1D invDcEdge(invDcEdge_data.data(), sm.nEdges);
    Field1D invAreaCell(invAreaCell_data.data(), sm.nCells);

    // Call the del-2 scalar kernel
    compute_scalar_del2<default_layout>(
        SerialPolicy{},
        delsq_scalar,
        scalar,
        rho_edge,
        mc,
        edgesOnCell_sign,
        dvEdge,
        invDcEdge,
        invAreaCell,
        sm.nCells,
        static_cast<index_type>(nVertLevels));

    // Verify all del-2 output values are exactly zero
    // The Laplacian of a constant field is zero because
    // scalar[cell2] - scalar[cell1] = 0 for all edges.
    for (std::size_t i = 0; i < delsq_scalar_data.size(); ++i) {
        RC_ASSERT(delsq_scalar_data[i] == 0.0);
    }
}
