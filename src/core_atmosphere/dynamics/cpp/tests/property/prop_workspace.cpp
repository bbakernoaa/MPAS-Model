/// @file prop_workspace.cpp
/// @brief Property-based tests for SRK3Workspace allocation.
///
/// **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8**
///
/// Property 1: Workspace allocation sizes are consistent with mesh dimensions.
/// Property 2: Workspace zero-initialization.
/// Additional deterministic tests for re-allocation and invalid dimensions.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/workspace.hpp>
#include <algorithm>
#include <stdexcept>

using namespace mpas::dycore;

// ============================================================================
// Property 1: Workspace allocation sizes are consistent with mesh dimensions
// ============================================================================

RC_GTEST_PROP(WorkspaceAllocation, SizesConsistentWithMeshDimensions,
              ()) {
    // Generate random valid mesh dimensions
    const auto nCells = *rc::gen::inRange(1, 501);
    const auto nEdges = *rc::gen::inRange(1, 1501);
    const auto nVertices = *rc::gen::inRange(1, 1001);
    const auto nVertLevels = *rc::gen::inRange(1, 101);

    SRK3Workspace ws;
    ws.allocate(nCells, nEdges, nVertices, nVertLevels);

    // Compute expected sizes
    const auto nVL_nC = static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells);
    const auto nVL_nE = static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges);
    const auto nVLp1_nC = static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells);
    const auto nVL_nV = static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nVertices);
    const auto nVL = static_cast<std::size_t>(nVertLevels);

    // --- Perturbation fields ---
    RC_ASSERT(ws.ru_p_storage.size() == nVL_nE);
    RC_ASSERT(ws.rw_p_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.rtheta_pp_storage.size() == nVL_nC);
    RC_ASSERT(ws.rho_pp_storage.size() == nVL_nC);
    RC_ASSERT(ws.rtheta_pp_old_storage.size() == nVL_nC);

    // --- Accumulated flux averages ---
    RC_ASSERT(ws.ruAvg_storage.size() == nVL_nE);
    RC_ASSERT(ws.wwAvg_storage.size() == nVLp1_nC);

    // --- Tendency arrays ---
    RC_ASSERT(ws.tend_u_storage.size() == nVL_nE);
    RC_ASSERT(ws.tend_theta_storage.size() == nVL_nC);
    RC_ASSERT(ws.tend_w_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.tend_rho_storage.size() == nVL_nC);

    // --- State save arrays ---
    RC_ASSERT(ws.u_save_storage.size() == nVL_nE);
    RC_ASSERT(ws.rho_zz_save_storage.size() == nVL_nC);
    RC_ASSERT(ws.theta_m_save_storage.size() == nVL_nC);
    RC_ASSERT(ws.w_save_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.rw_save_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.rho_zz_old_storage.size() == nVL_nC);

    // --- Diagnostic workspaces ---
    RC_ASSERT(ws.ke_storage.size() == nVL_nC);
    RC_ASSERT(ws.vorticity_storage.size() == nVL_nV);
    RC_ASSERT(ws.divergence_storage.size() == nVL_nC);
    RC_ASSERT(ws.pv_edge_storage.size() == nVL_nE);
    RC_ASSERT(ws.rho_edge_storage.size() == nVL_nE);

    // --- Vertical implicit coefficients ---
    RC_ASSERT(ws.cofwr_storage.size() == nVL_nC);
    RC_ASSERT(ws.cofwz_storage.size() == nVL_nC);
    RC_ASSERT(ws.coftz_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.cofwt_storage.size() == nVL_nC);
    RC_ASSERT(ws.a_tri_storage.size() == nVL_nC);
    RC_ASSERT(ws.alpha_tri_storage.size() == nVL_nC);
    RC_ASSERT(ws.gamma_tri_storage.size() == nVL_nC);
    RC_ASSERT(ws.cofrz_storage.size() == nVL);

    // --- Moist coefficients ---
    RC_ASSERT(ws.cqw_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.cqu_storage.size() == nVL_nE);

    // --- Additional workspace ---
    RC_ASSERT(ws.exner_storage.size() == nVL_nC);
    RC_ASSERT(ws.pressure_storage.size() == nVL_nC);
    RC_ASSERT(ws.rtheta_flux_storage.size() == nVL_nE);
    RC_ASSERT(ws.pp_storage.size() == nVL_nC);
    RC_ASSERT(ws.dpdz_storage.size() == nVL_nC);
    RC_ASSERT(ws.zxu_storage.size() == nVL_nE);
    RC_ASSERT(ws.dss_storage.size() == nVL_nC);
    RC_ASSERT(ws.qtot_storage.size() == nVL_nC);

    // --- Perturbation state variables (Fortran formulation) ---
    RC_ASSERT(ws.rho_p_storage.size() == nVL_nC);
    RC_ASSERT(ws.rtheta_p_storage.size() == nVL_nC);
    RC_ASSERT(ws.ru_storage.size() == nVL_nE);
    RC_ASSERT(ws.rw_storage.size() == nVLp1_nC);

    // --- Perturbation state saves ---
    RC_ASSERT(ws.rho_p_save_storage.size() == nVL_nC);
    RC_ASSERT(ws.rtheta_p_save_storage.size() == nVL_nC);
    RC_ASSERT(ws.ru_save_storage.size() == nVL_nE);

    // --- Diabatic tendency ---
    RC_ASSERT(ws.rt_diabatic_tend_storage.size() == nVL_nC);

    // --- Exner function base state ---
    RC_ASSERT(ws.exner_base_storage.size() == nVL_nC);

    // --- Verify stored dimensions ---
    RC_ASSERT(ws.nCells_ == nCells);
    RC_ASSERT(ws.nEdges_ == nEdges);
    RC_ASSERT(ws.nVertices_ == nVertices);
    RC_ASSERT(ws.nVertLevels_ == nVertLevels);
    RC_ASSERT(ws.allocated() == true);
}

// ============================================================================
// Property 2: Workspace zero-initialization
// ============================================================================

namespace {
/// Helper to check all elements of a vector are exactly 0.0
bool all_zero(const std::vector<real_type>& v) {
    return std::all_of(v.begin(), v.end(),
                       [](real_type x) { return x == 0.0; });
}
} // anonymous namespace

RC_GTEST_PROP(WorkspaceAllocation, ZeroInitialization,
              ()) {
    const auto nCells = *rc::gen::inRange(1, 501);
    const auto nEdges = *rc::gen::inRange(1, 1501);
    const auto nVertices = *rc::gen::inRange(1, 1001);
    const auto nVertLevels = *rc::gen::inRange(1, 101);

    SRK3Workspace ws;
    ws.allocate(nCells, nEdges, nVertices, nVertLevels);

    // Verify every element of every storage vector is exactly 0.0
    RC_ASSERT(all_zero(ws.ru_p_storage));
    RC_ASSERT(all_zero(ws.rw_p_storage));
    RC_ASSERT(all_zero(ws.rtheta_pp_storage));
    RC_ASSERT(all_zero(ws.rho_pp_storage));
    RC_ASSERT(all_zero(ws.rtheta_pp_old_storage));
    RC_ASSERT(all_zero(ws.ruAvg_storage));
    RC_ASSERT(all_zero(ws.wwAvg_storage));
    RC_ASSERT(all_zero(ws.tend_u_storage));
    RC_ASSERT(all_zero(ws.tend_theta_storage));
    RC_ASSERT(all_zero(ws.tend_w_storage));
    RC_ASSERT(all_zero(ws.tend_rho_storage));
    RC_ASSERT(all_zero(ws.u_save_storage));
    RC_ASSERT(all_zero(ws.rho_zz_save_storage));
    RC_ASSERT(all_zero(ws.theta_m_save_storage));
    RC_ASSERT(all_zero(ws.w_save_storage));
    RC_ASSERT(all_zero(ws.rw_save_storage));
    RC_ASSERT(all_zero(ws.rho_zz_old_storage));
    RC_ASSERT(all_zero(ws.ke_storage));
    RC_ASSERT(all_zero(ws.vorticity_storage));
    RC_ASSERT(all_zero(ws.divergence_storage));
    RC_ASSERT(all_zero(ws.pv_edge_storage));
    RC_ASSERT(all_zero(ws.rho_edge_storage));
    RC_ASSERT(all_zero(ws.cofwr_storage));
    RC_ASSERT(all_zero(ws.cofwz_storage));
    RC_ASSERT(all_zero(ws.coftz_storage));
    RC_ASSERT(all_zero(ws.cofwt_storage));
    RC_ASSERT(all_zero(ws.a_tri_storage));
    RC_ASSERT(all_zero(ws.alpha_tri_storage));
    RC_ASSERT(all_zero(ws.gamma_tri_storage));
    RC_ASSERT(all_zero(ws.cofrz_storage));
    RC_ASSERT(all_zero(ws.cqw_storage));
    RC_ASSERT(all_zero(ws.cqu_storage));
    RC_ASSERT(all_zero(ws.exner_storage));
    RC_ASSERT(all_zero(ws.pressure_storage));
    RC_ASSERT(all_zero(ws.rtheta_flux_storage));
    RC_ASSERT(all_zero(ws.pp_storage));
    RC_ASSERT(all_zero(ws.dpdz_storage));
    RC_ASSERT(all_zero(ws.zxu_storage));
    RC_ASSERT(all_zero(ws.dss_storage));
    RC_ASSERT(all_zero(ws.qtot_storage));

    // --- Perturbation state variables (Fortran formulation) ---
    RC_ASSERT(all_zero(ws.rho_p_storage));
    RC_ASSERT(all_zero(ws.rtheta_p_storage));
    RC_ASSERT(all_zero(ws.ru_storage));
    RC_ASSERT(all_zero(ws.rw_storage));

    // --- Perturbation state saves ---
    RC_ASSERT(all_zero(ws.rho_p_save_storage));
    RC_ASSERT(all_zero(ws.rtheta_p_save_storage));
    RC_ASSERT(all_zero(ws.ru_save_storage));

    // --- Diabatic tendency ---
    RC_ASSERT(all_zero(ws.rt_diabatic_tend_storage));

    // --- Exner function base state ---
    RC_ASSERT(all_zero(ws.exner_base_storage));
}

// ============================================================================
// Re-allocation test: allocate with dims1, then allocate again with dims2.
// Verify sizes match dims2 (idempotent behavior).
// ============================================================================

RC_GTEST_PROP(WorkspaceAllocation, ReallocationUpdatesSize,
              ()) {
    const auto nCells1 = *rc::gen::inRange(1, 501);
    const auto nEdges1 = *rc::gen::inRange(1, 1501);
    const auto nVertices1 = *rc::gen::inRange(1, 1001);
    const auto nVertLevels1 = *rc::gen::inRange(1, 101);

    const auto nCells2 = *rc::gen::inRange(1, 501);
    const auto nEdges2 = *rc::gen::inRange(1, 1501);
    const auto nVertices2 = *rc::gen::inRange(1, 1001);
    const auto nVertLevels2 = *rc::gen::inRange(1, 101);

    SRK3Workspace ws;

    // First allocation
    ws.allocate(nCells1, nEdges1, nVertices1, nVertLevels1);
    RC_ASSERT(ws.allocated() == true);

    // Second allocation with different dimensions
    ws.allocate(nCells2, nEdges2, nVertices2, nVertLevels2);
    RC_ASSERT(ws.allocated() == true);

    // Verify sizes match the second set of dimensions
    const auto nVL_nC = static_cast<std::size_t>(nVertLevels2) * static_cast<std::size_t>(nCells2);
    const auto nVL_nE = static_cast<std::size_t>(nVertLevels2) * static_cast<std::size_t>(nEdges2);
    const auto nVLp1_nC = static_cast<std::size_t>(nVertLevels2 + 1) * static_cast<std::size_t>(nCells2);
    const auto nVL_nV = static_cast<std::size_t>(nVertLevels2) * static_cast<std::size_t>(nVertices2);
    const auto nVL = static_cast<std::size_t>(nVertLevels2);

    RC_ASSERT(ws.ru_p_storage.size() == nVL_nE);
    RC_ASSERT(ws.rw_p_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.rtheta_pp_storage.size() == nVL_nC);
    RC_ASSERT(ws.rho_pp_storage.size() == nVL_nC);
    RC_ASSERT(ws.rtheta_pp_old_storage.size() == nVL_nC);
    RC_ASSERT(ws.ruAvg_storage.size() == nVL_nE);
    RC_ASSERT(ws.wwAvg_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.tend_u_storage.size() == nVL_nE);
    RC_ASSERT(ws.tend_theta_storage.size() == nVL_nC);
    RC_ASSERT(ws.tend_w_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.tend_rho_storage.size() == nVL_nC);
    RC_ASSERT(ws.u_save_storage.size() == nVL_nE);
    RC_ASSERT(ws.rho_zz_save_storage.size() == nVL_nC);
    RC_ASSERT(ws.theta_m_save_storage.size() == nVL_nC);
    RC_ASSERT(ws.w_save_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.rw_save_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.rho_zz_old_storage.size() == nVL_nC);
    RC_ASSERT(ws.ke_storage.size() == nVL_nC);
    RC_ASSERT(ws.vorticity_storage.size() == nVL_nV);
    RC_ASSERT(ws.divergence_storage.size() == nVL_nC);
    RC_ASSERT(ws.pv_edge_storage.size() == nVL_nE);
    RC_ASSERT(ws.rho_edge_storage.size() == nVL_nE);
    RC_ASSERT(ws.cofwr_storage.size() == nVL_nC);
    RC_ASSERT(ws.cofwz_storage.size() == nVL_nC);
    RC_ASSERT(ws.coftz_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.cofwt_storage.size() == nVL_nC);
    RC_ASSERT(ws.a_tri_storage.size() == nVL_nC);
    RC_ASSERT(ws.alpha_tri_storage.size() == nVL_nC);
    RC_ASSERT(ws.gamma_tri_storage.size() == nVL_nC);
    RC_ASSERT(ws.cofrz_storage.size() == nVL);
    RC_ASSERT(ws.cqw_storage.size() == nVLp1_nC);
    RC_ASSERT(ws.cqu_storage.size() == nVL_nE);
    RC_ASSERT(ws.exner_storage.size() == nVL_nC);
    RC_ASSERT(ws.pressure_storage.size() == nVL_nC);
    RC_ASSERT(ws.rtheta_flux_storage.size() == nVL_nE);
    RC_ASSERT(ws.pp_storage.size() == nVL_nC);
    RC_ASSERT(ws.dpdz_storage.size() == nVL_nC);
    RC_ASSERT(ws.zxu_storage.size() == nVL_nE);
    RC_ASSERT(ws.dss_storage.size() == nVL_nC);
    RC_ASSERT(ws.qtot_storage.size() == nVL_nC);

    // Verify dimensions updated
    RC_ASSERT(ws.nCells_ == nCells2);
    RC_ASSERT(ws.nEdges_ == nEdges2);
    RC_ASSERT(ws.nVertices_ == nVertices2);
    RC_ASSERT(ws.nVertLevels_ == nVertLevels2);

    // Verify zero-initialization after re-allocation
    RC_ASSERT(all_zero(ws.ru_p_storage));
    RC_ASSERT(all_zero(ws.rw_p_storage));
    RC_ASSERT(all_zero(ws.rtheta_pp_storage));
    RC_ASSERT(all_zero(ws.tend_u_storage));
    RC_ASSERT(all_zero(ws.cofwr_storage));
    RC_ASSERT(all_zero(ws.cofrz_storage));
    RC_ASSERT(all_zero(ws.exner_storage));
}

// ============================================================================
// Invalid dimensions: verify std::invalid_argument is thrown when any dim <= 0
// ============================================================================

TEST(WorkspaceValidation, ThrowsOnZeroCells) {
    SRK3Workspace ws;
    EXPECT_THROW(ws.allocate(0, 10, 5, 26), std::invalid_argument);
}

TEST(WorkspaceValidation, ThrowsOnNegativeCells) {
    SRK3Workspace ws;
    EXPECT_THROW(ws.allocate(-1, 10, 5, 26), std::invalid_argument);
}

TEST(WorkspaceValidation, ThrowsOnZeroEdges) {
    SRK3Workspace ws;
    EXPECT_THROW(ws.allocate(10, 0, 5, 26), std::invalid_argument);
}

TEST(WorkspaceValidation, ThrowsOnNegativeEdges) {
    SRK3Workspace ws;
    EXPECT_THROW(ws.allocate(10, -5, 5, 26), std::invalid_argument);
}

TEST(WorkspaceValidation, ThrowsOnZeroVertices) {
    SRK3Workspace ws;
    EXPECT_THROW(ws.allocate(10, 30, 0, 26), std::invalid_argument);
}

TEST(WorkspaceValidation, ThrowsOnNegativeVertices) {
    SRK3Workspace ws;
    EXPECT_THROW(ws.allocate(10, 30, -2, 26), std::invalid_argument);
}

TEST(WorkspaceValidation, ThrowsOnZeroVertLevels) {
    SRK3Workspace ws;
    EXPECT_THROW(ws.allocate(10, 30, 5, 0), std::invalid_argument);
}

TEST(WorkspaceValidation, ThrowsOnNegativeVertLevels) {
    SRK3Workspace ws;
    EXPECT_THROW(ws.allocate(10, 30, 5, -10), std::invalid_argument);
}

TEST(WorkspaceValidation, DoesNotAllocateOnInvalidDims) {
    SRK3Workspace ws;
    // Attempt invalid allocation
    EXPECT_THROW(ws.allocate(0, 10, 5, 26), std::invalid_argument);
    // Workspace should remain in unallocated state
    EXPECT_FALSE(ws.allocated());
    EXPECT_EQ(ws.ru_p_storage.size(), 0u);
}

// Property test for invalid dimensions with random negative/zero values
RC_GTEST_PROP(WorkspaceValidation, InvalidDimsThrow,
              ()) {
    // Generate at least one invalid dimension (<=0)
    const auto choice = *rc::gen::inRange(0, 4);
    int nCells = *rc::gen::inRange(1, 501);
    int nEdges = *rc::gen::inRange(1, 1501);
    int nVertices = *rc::gen::inRange(1, 1001);
    int nVertLevels = *rc::gen::inRange(1, 101);

    // Make one dimension invalid
    const auto badVal = *rc::gen::inRange(-100, 1); // range is [-100, 0]
    switch (choice) {
        case 0: nCells = badVal; break;
        case 1: nEdges = badVal; break;
        case 2: nVertices = badVal; break;
        case 3: nVertLevels = badVal; break;
    }

    SRK3Workspace ws;
    RC_ASSERT_THROWS_AS(ws.allocate(nCells, nEdges, nVertices, nVertLevels),
                        std::invalid_argument);
}
