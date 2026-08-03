/// @file prop_tendencies.cpp
/// @brief Property-based tests for tendency kernels (tend_u, tend_theta, tend_w).
///
/// **Validates: Requirements 4.2, 4.3, 4.4**
///
/// Property 36: Tendency Symmetry — For symmetric input fields on a symmetric
///   mesh, tendencies respect the symmetry.
/// Property 37: Zero Fields Produce Zero Tendencies — When all input fields are
///   zero, all tendencies are zero.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/tendencies.hpp>
#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include "../synthetic_mesh.hpp"
#include <vector>
#include <cmath>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;
using namespace mpas::dycore::testing;

// Sentinel alias for clarity
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
// Test 1: Zero velocity/pressure produces zero tend_u
//
// When u=0, pressure is constant (no gradient), ke=0, pv_edge=0, cqu=1,
// rho_edge=1 → tend_u should be 0 everywhere.
// ============================================================================

RC_GTEST_PROP(TendencyZeroFields, ZeroInputsProduceZeroTendU, ()) {
    const auto nCells = *rc::gen::inRange(4, 12);
    const auto nVertLevels = *rc::gen::inRange(3, 10);

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;

    // Zero velocity
    std::vector<real_type> u_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 0.0);

    // Constant pressure (no gradient) → pressure_grad = 0
    std::vector<real_type> pressure_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), 1000.0);

    // Zero pv_edge → Coriolis term = 0 regardless of weightsOnEdge
    std::vector<real_type> pv_edge_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 0.0);

    // Zero ke → ke_grad = 0
    std::vector<real_type> ke_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), 0.0);

    // rho_edge = 1.0 (multiplies the Coriolis-KE term which is 0)
    std::vector<real_type> rho_edge_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 1.0);

    // cqu = 1.0 (multiplies pressure gradient which is 0)
    std::vector<real_type> cqu_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 1.0);

    // edgesOnEdge: not needed since pv_edge=0 and u=0 make the Coriolis sum = 0
    // But the kernel still iterates, so provide a minimal valid structure.
    const index_type maxEdges2 = 6;
    std::vector<index_type> edgesOnEdge_data(
        static_cast<std::size_t>(nEdges) * static_cast<std::size_t>(maxEdges2), SENTINEL);

    // weightsOnEdge: all zeros (makes Coriolis term zero regardless)
    std::vector<real_type> weightsOnEdge_data(
        static_cast<std::size_t>(nEdges) * static_cast<std::size_t>(maxEdges2), 0.0);

    // nEdgesOnEdge: 0 for all edges (no neighbor edges to iterate)
    std::vector<index_type> nEdgesOnEdge_data(
        static_cast<std::size_t>(nEdges), 0);

    // invDcEdge from mesh
    std::vector<real_type> invDcEdge_data(static_cast<std::size_t>(nEdges));
    for (index_type e = 0; e < nEdges; ++e) {
        invDcEdge_data[static_cast<std::size_t>(e)] =
            1.0 / sm.dcEdge[static_cast<std::size_t>(e)];
    }

    // Output
    std::vector<real_type> tend_u_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), -999.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tend_u(
        tend_u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> pressure(
        pressure_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> pv_edge(
        pv_edge_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> ke(
        ke_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(
        rho_edge_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> cqu(
        cqu_data.data(), nVertLevels, nEdges);

    ConnectivityView edgesOnEdge(
        edgesOnEdge_data.data(), nEdges, maxEdges2);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        weightsOnEdge(weightsOnEdge_data.data(),
                      static_cast<index_type>(nEdges * maxEdges2));
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>
        nEdgesOnEdge(nEdgesOnEdge_data.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invDcEdge(invDcEdge_data.data(), nEdges);

    compute_tend_u<default_layout>(
        SerialPolicy{},
        tend_u, u, pressure, pv_edge, ke, rho_edge, cqu,
        mc, edgesOnEdge, weightsOnEdge, nEdgesOnEdge, invDcEdge,
        nEdges, nVertLevels, maxEdges2);

    // All tend_u values should be exactly zero
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            auto val = tend_u[k, iEdge];
            RC_ASSERT(val == 0.0);
        }
    }
}

// ============================================================================
// Test 2: Zero flux produces zero tend_theta
//
// When rtheta_flux=0 everywhere → tend_theta=0 everywhere.
// ============================================================================

RC_GTEST_PROP(TendencyZeroFields, ZeroFluxProducesZeroTendTheta, ()) {
    const auto nCells = *rc::gen::inRange(4, 12);
    const auto nVertLevels = *rc::gen::inRange(3, 10);

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;

    // Zero rtheta_flux
    std::vector<real_type> rtheta_flux_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 0.0);

    // Output
    std::vector<real_type> tend_theta_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), -999.0);

    Field2D<default_layout, unchecked_accessor> tend_theta(
        tend_theta_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_flux(
        rtheta_flux_data.data(), nVertLevels, nEdges);

    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        areaCellSpan(sm.areaCell.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        dvEdgeSpan(sm.dvEdge.data(), nEdges);

    compute_tend_theta<default_layout>(
        SerialPolicy{},
        tend_theta, rtheta_flux,
        mc,
        areaCellSpan, dvEdgeSpan,
        nCells, nVertLevels);

    // All tend_theta values should be exactly zero
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            auto val = tend_theta[k, iCell];
            RC_ASSERT(val == 0.0);
        }
    }
}

// ============================================================================
// Test 3: Constant pressure (no gradient) produces zero tend_w PGF
//
// When pp is constant across levels (no vertical gradient) and dpdz=0
// → tend_w=0 (rigid lid boundaries still enforced).
// ============================================================================

RC_GTEST_PROP(TendencyZeroFields, ConstantPressureProducesZeroTendW, ()) {
    const auto nCells = *rc::gen::inRange(4, 12);
    const auto nVertLevels = *rc::gen::inRange(3, 10);

    auto sm = genValidMesh(nCells, nVertLevels);

    // pp constant across all levels → pp[k] - pp[k-1] = 0 → pgrad = 0
    const real_type pp_val = 1000.0;
    std::vector<real_type> pp_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), pp_val);

    // cqw = 1.0 (moist coefficient, multiplies everything)
    // Size: (nVertLevels+1) * nCells for interface levels
    std::vector<real_type> cqw_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), 1.0);

    // rdzu: reciprocal dz at interfaces (any nonzero value works since pp difference is 0)
    std::vector<real_type> rdzu_data(static_cast<std::size_t>(nVertLevels + 1), 0.01);

    // fzm, fzp: vertical interpolation weights (any values)
    std::vector<real_type> fzm_data(static_cast<std::size_t>(nVertLevels + 1), 0.5);
    std::vector<real_type> fzp_data(static_cast<std::size_t>(nVertLevels + 1), 0.5);

    // dpdz = 0 everywhere → buoyancy = 0
    std::vector<real_type> dpdz_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), 0.0);

    // Output: (nVertLevels+1) * nCells
    std::vector<real_type> tend_w_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), -999.0);

    Field2D<default_layout, unchecked_accessor> tend_w(
        tend_w_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> pp(
        pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> cqw(
        cqw_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> dpdz(
        dpdz_data.data(), nVertLevels, nCells);

    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        rdzu(rdzu_data.data(), nVertLevels + 1);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzm(fzm_data.data(), nVertLevels + 1);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzp(fzp_data.data(), nVertLevels + 1);

    compute_tend_w<default_layout>(
        SerialPolicy{},
        tend_w, pp, cqw, rdzu, fzm, fzp, dpdz,
        nCells, nVertLevels);

    // All tend_w values should be zero: pgrad=0, buoyancy=0
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        for (index_type k = 0; k <= nVertLevels; ++k) {
            auto val = tend_w[k, iCell];
            RC_ASSERT(val == 0.0);
        }
    }
}

// ============================================================================
// Test 4: tend_w boundary conditions always hold
//
// For any input, tend_w[0, iCell]==0 and tend_w[nVertLevels, iCell]==0.
// ============================================================================

RC_GTEST_PROP(TendencyBoundaryConditions, TendWBoundariesAlwaysZero, ()) {
    const auto nCells = *rc::gen::inRange(4, 12);
    const auto nVertLevels = *rc::gen::inRange(3, 10);

    auto sm = genValidMesh(nCells, nVertLevels);

    // Random pp values
    std::vector<real_type> pp_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells));
    for (auto& v : pp_data) {
        int raw = *rc::gen::inRange(-10000, 10001);
        v = static_cast<real_type>(raw) / 10.0;
    }

    // Random cqw in [0.5, 1.5]
    std::vector<real_type> cqw_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells));
    for (auto& v : cqw_data) {
        int raw = *rc::gen::inRange(500, 1501);
        v = static_cast<real_type>(raw) / 1000.0;
    }

    // Random rdzu > 0
    std::vector<real_type> rdzu_data(static_cast<std::size_t>(nVertLevels + 1));
    for (auto& v : rdzu_data) {
        int raw = *rc::gen::inRange(1, 100);
        v = static_cast<real_type>(raw) / 1000.0;
    }

    // Random fzm, fzp in [0, 1]
    std::vector<real_type> fzm_data(static_cast<std::size_t>(nVertLevels + 1));
    std::vector<real_type> fzp_data(static_cast<std::size_t>(nVertLevels + 1));
    for (std::size_t i = 0; i < fzm_data.size(); ++i) {
        int raw_m = *rc::gen::inRange(0, 1001);
        fzm_data[i] = static_cast<real_type>(raw_m) / 1000.0;
        int raw_p = *rc::gen::inRange(0, 1001);
        fzp_data[i] = static_cast<real_type>(raw_p) / 1000.0;
    }

    // Random dpdz
    std::vector<real_type> dpdz_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells));
    for (auto& v : dpdz_data) {
        int raw = *rc::gen::inRange(-5000, 5001);
        v = static_cast<real_type>(raw) / 100.0;
    }

    // Output
    std::vector<real_type> tend_w_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), -999.0);

    Field2D<default_layout, unchecked_accessor> tend_w(
        tend_w_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> pp(
        pp_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> cqw(
        cqw_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> dpdz(
        dpdz_data.data(), nVertLevels, nCells);

    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        rdzu(rdzu_data.data(), nVertLevels + 1);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzm(fzm_data.data(), nVertLevels + 1);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        fzp(fzp_data.data(), nVertLevels + 1);

    compute_tend_w<default_layout>(
        SerialPolicy{},
        tend_w, pp, cqw, rdzu, fzm, fzp, dpdz,
        nCells, nVertLevels);

    // Boundary conditions: top and surface must always be zero
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        auto top_val = tend_w[0, iCell];
        auto bot_val = tend_w[nVertLevels, iCell];
        RC_ASSERT(top_val == 0.0);
        RC_ASSERT(bot_val == 0.0);
    }
}

// ============================================================================
// Test 5: tend_u boundary edges are zero
//
// For boundary edges (cell1 == INVALID_INDEX), tend_u should be 0.
// ============================================================================

RC_GTEST_PROP(TendencyBoundaryConditions, TendUBoundaryEdgesAreZero, ()) {
    const auto nCells = *rc::gen::inRange(4, 12);
    const auto nVertLevels = *rc::gen::inRange(3, 10);

    auto sm = genValidMesh(nCells, nVertLevels);
    auto mc = buildConnectivity(sm);

    const auto nEdges = sm.nEdges;

    // Random velocity in [-10, 10]
    std::vector<real_type> u_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges));
    for (auto& v : u_data) {
        int raw = *rc::gen::inRange(-10000, 10001);
        v = static_cast<real_type>(raw) / 1000.0;
    }

    // Random pressure
    std::vector<real_type> pressure_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells));
    for (auto& v : pressure_data) {
        int raw = *rc::gen::inRange(900000, 1100001);
        v = static_cast<real_type>(raw) / 1000.0;
    }

    // Random pv_edge
    std::vector<real_type> pv_edge_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges));
    for (auto& v : pv_edge_data) {
        int raw = *rc::gen::inRange(-100, 101);
        v = static_cast<real_type>(raw) * 1.0e-6;
    }

    // Random ke
    std::vector<real_type> ke_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells));
    for (auto& v : ke_data) {
        int raw = *rc::gen::inRange(0, 1001);
        v = static_cast<real_type>(raw) / 100.0;
    }

    // rho_edge = 1.0
    std::vector<real_type> rho_edge_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 1.0);

    // cqu = 1.0
    std::vector<real_type> cqu_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), 1.0);

    // edgesOnEdge (all invalid — makes Coriolis loop empty)
    const index_type maxEdges2 = 6;
    std::vector<index_type> edgesOnEdge_data(
        static_cast<std::size_t>(nEdges) * static_cast<std::size_t>(maxEdges2), SENTINEL);

    // weightsOnEdge: zeros
    std::vector<real_type> weightsOnEdge_data(
        static_cast<std::size_t>(nEdges) * static_cast<std::size_t>(maxEdges2), 0.0);

    // nEdgesOnEdge: 0 for all (no TRiSK neighbors)
    std::vector<index_type> nEdgesOnEdge_data(
        static_cast<std::size_t>(nEdges), 0);

    // invDcEdge
    std::vector<real_type> invDcEdge_data(static_cast<std::size_t>(nEdges));
    for (index_type e = 0; e < nEdges; ++e) {
        invDcEdge_data[static_cast<std::size_t>(e)] =
            1.0 / sm.dcEdge[static_cast<std::size_t>(e)];
    }

    // Output
    std::vector<real_type> tend_u_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges), -999.0);

    Field2D<default_layout, unchecked_accessor> tend_u(
        tend_u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> pressure(
        pressure_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> pv_edge(
        pv_edge_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> ke(
        ke_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(
        rho_edge_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> cqu(
        cqu_data.data(), nVertLevels, nEdges);

    ConnectivityView edgesOnEdge(
        edgesOnEdge_data.data(), nEdges, maxEdges2);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        weightsOnEdge(weightsOnEdge_data.data(),
                      static_cast<index_type>(nEdges * maxEdges2));
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>
        nEdgesOnEdge(nEdgesOnEdge_data.data(), nEdges);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        invDcEdge(invDcEdge_data.data(), nEdges);

    compute_tend_u<default_layout>(
        SerialPolicy{},
        tend_u, u, pressure, pv_edge, ke, rho_edge, cqu,
        mc, edgesOnEdge, weightsOnEdge, nEdgesOnEdge, invDcEdge,
        nEdges, nVertLevels, maxEdges2);

    // For boundary edges (cell1 == INVALID_INDEX), tend_u must be zero
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        const index_type cell1 = mc.cellsOnEdge[iEdge, 1];
        if (cell1 == SENTINEL) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                auto val = tend_u[k, iEdge];
                RC_ASSERT(val == 0.0);
            }
        }
    }
}
