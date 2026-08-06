/// @file prop_recover_state_validation.cpp
/// @brief Property-based tests for recover_state parameter validation.
///
/// **Validates: Requirements 2.5, 2.6**
///
/// Property 4: State recovery rejects invalid parameters
///   For any input where n_acoustic <= 0 or dt_rk <= 0, the recover_state
///   kernel throws std::invalid_argument without modifying any output field.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/recover_state.hpp>
#include <mpas_dycore/types.hpp>
#include <mpas_dycore/execution_policy.hpp>
#include <vector>
#include <stdexcept>
#include <algorithm>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;

// ============================================================================
// Helper: Allocate and fill a vector with a constant value
// ============================================================================

static std::vector<real_type> make_filled(std::size_t n, real_type val) {
    return std::vector<real_type>(n, val);
}

// ============================================================================
// Test fixture holding mesh dimensions and pre-allocated arrays
// ============================================================================

struct RecoverStateValidationData {
    index_type nCells;
    index_type nEdges;
    index_type nVertLevels;

    // Output fields (these should NOT be modified on invalid input)
    std::vector<real_type> u_data;
    std::vector<real_type> rho_zz_data;
    std::vector<real_type> theta_m_data;
    std::vector<real_type> w_data;

    // Copies to verify outputs remain unchanged
    std::vector<real_type> u_original;
    std::vector<real_type> rho_zz_original;
    std::vector<real_type> theta_m_original;
    std::vector<real_type> w_original;

    // Input fields
    std::vector<real_type> u_save_data;
    std::vector<real_type> rho_zz_save_data;
    std::vector<real_type> theta_m_save_data;
    std::vector<real_type> w_save_data;
    std::vector<real_type> ruAvg_data;
    std::vector<real_type> wwAvg_data;
    std::vector<real_type> rho_pp_data;
    std::vector<real_type> rtheta_pp_data;
    std::vector<real_type> tend_u_data;
    std::vector<real_type> tend_rho_data;
    std::vector<real_type> tend_theta_data;
    std::vector<real_type> tend_w_data;
    std::vector<real_type> rho_edge_data;
    std::vector<real_type> zz_data;
    std::vector<real_type> fzm_data;
    std::vector<real_type> fzp_data;

    RecoverStateValidationData(index_type nc, index_type ne, index_type nv)
        : nCells(nc), nEdges(ne), nVertLevels(nv)
    {
        const auto edge_size = static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nEdges);
        const auto cell_size = static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells);
        const auto w_size = static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells);
        const auto vert_size = static_cast<std::size_t>(nVertLevels);

        // Initialize outputs with sentinel values to detect modification
        u_data = make_filled(edge_size, 42.0);
        rho_zz_data = make_filled(cell_size, 43.0);
        theta_m_data = make_filled(cell_size, 44.0);
        w_data = make_filled(w_size, 45.0);

        // Save copies
        u_original = u_data;
        rho_zz_original = rho_zz_data;
        theta_m_original = theta_m_data;
        w_original = w_data;

        // Initialize input fields with reasonable values
        u_save_data = make_filled(edge_size, 1.0);
        rho_zz_save_data = make_filled(cell_size, 1.2);
        theta_m_save_data = make_filled(cell_size, 300.0);
        w_save_data = make_filled(w_size, 0.0);
        ruAvg_data = make_filled(edge_size, 0.5);
        wwAvg_data = make_filled(w_size, 0.1);
        rho_pp_data = make_filled(cell_size, 0.01);
        rtheta_pp_data = make_filled(cell_size, 0.02);
        tend_u_data = make_filled(edge_size, 0.001);
        tend_rho_data = make_filled(cell_size, 0.0001);
        tend_theta_data = make_filled(cell_size, 0.0002);
        tend_w_data = make_filled(w_size, 0.0003);
        rho_edge_data = make_filled(edge_size, 1.1);
        zz_data = make_filled(cell_size, 1.0);

        // Vertical interpolation weights (fzm + fzp = 1 at each level)
        fzm_data.resize(vert_size, 0.5);
        fzp_data.resize(vert_size, 0.5);
    }

    void callRecoverState(real_type dt_rk, index_type n_acoustic) {
        Field2D<default_layout, unchecked_accessor> u(
            u_data.data(), nVertLevels, nEdges);
        Field2D<default_layout, unchecked_accessor> rho_zz(
            rho_zz_data.data(), nVertLevels, nCells);
        Field2D<default_layout, unchecked_accessor> theta_m(
            theta_m_data.data(), nVertLevels, nCells);
        Field2D<default_layout, unchecked_accessor> w(
            w_data.data(), nVertLevels + 1, nCells);

        ConstField2D<default_layout, unchecked_accessor> u_save(
            u_save_data.data(), nVertLevels, nEdges);
        ConstField2D<default_layout, unchecked_accessor> rho_zz_save(
            rho_zz_save_data.data(), nVertLevels, nCells);
        ConstField2D<default_layout, unchecked_accessor> theta_m_save(
            theta_m_save_data.data(), nVertLevels, nCells);
        ConstField2D<default_layout, unchecked_accessor> w_save(
            w_save_data.data(), nVertLevels + 1, nCells);
        ConstField2D<default_layout, unchecked_accessor> ruAvg(
            ruAvg_data.data(), nVertLevels, nEdges);
        ConstField2D<default_layout, unchecked_accessor> wwAvg(
            wwAvg_data.data(), nVertLevels + 1, nCells);
        ConstField2D<default_layout, unchecked_accessor> rho_pp(
            rho_pp_data.data(), nVertLevels, nCells);
        ConstField2D<default_layout, unchecked_accessor> rtheta_pp(
            rtheta_pp_data.data(), nVertLevels, nCells);
        ConstField2D<default_layout, unchecked_accessor> tend_u(
            tend_u_data.data(), nVertLevels, nEdges);
        ConstField2D<default_layout, unchecked_accessor> tend_rho(
            tend_rho_data.data(), nVertLevels, nCells);
        ConstField2D<default_layout, unchecked_accessor> tend_theta(
            tend_theta_data.data(), nVertLevels, nCells);
        ConstField2D<default_layout, unchecked_accessor> tend_w(
            tend_w_data.data(), nVertLevels + 1, nCells);
        ConstField2D<default_layout, unchecked_accessor> rho_edge(
            rho_edge_data.data(), nVertLevels, nEdges);
        ConstField2D<default_layout, unchecked_accessor> zz(
            zz_data.data(), nVertLevels, nCells);

        ConstSpan1D fzm(fzm_data.data(), nVertLevels);
        ConstSpan1D fzp(fzp_data.data(), nVertLevels);

        recover_state<default_layout, SerialPolicy>(
            SerialPolicy{},
            u, rho_zz, theta_m, w,
            u_save, rho_zz_save, theta_m_save, w_save,
            ruAvg, wwAvg, rho_pp, rtheta_pp,
            tend_u, tend_rho, tend_theta, tend_w,
            rho_edge, zz, fzm, fzp,
            dt_rk, n_acoustic,
            nCells, nEdges, nVertLevels);
    }

    bool outputsUnchanged() const {
        return u_data == u_original
            && rho_zz_data == rho_zz_original
            && theta_m_data == theta_m_original
            && w_data == w_original;
    }
};

// ============================================================================
// Property 4: State recovery rejects invalid parameters
// ============================================================================

/// Test case A: n_acoustic <= 0 throws std::invalid_argument without modifying outputs
RC_GTEST_PROP(RecoverStateValidation,
              InvalidNAcousticThrowsWithoutModification,
              ())
{
    // Generate small mesh dimensions
    const auto nCells = *rc::gen::inRange<index_type>(1, 6);
    const auto nEdges = *rc::gen::inRange<index_type>(1, 16);
    const auto nVertLevels = *rc::gen::inRange<index_type>(3, 11);

    // Generate invalid n_acoustic (value in [-10, 0])
    const auto n_acoustic = *rc::gen::inRange<index_type>(-10, 1);  // [-10, 0]

    // Generate valid dt_rk > 0
    const auto dt_rk_int = *rc::gen::inRange<int>(1, 100);
    const real_type dt_rk = static_cast<real_type>(dt_rk_int) * 0.1;

    RecoverStateValidationData data(nCells, nEdges, nVertLevels);

    // Verify that std::invalid_argument is thrown
    RC_ASSERT_THROWS_AS(data.callRecoverState(dt_rk, n_acoustic), std::invalid_argument);

    // Verify output arrays are unchanged
    RC_ASSERT(data.outputsUnchanged());
}

/// Test case B: dt_rk <= 0 throws std::invalid_argument without modifying outputs
RC_GTEST_PROP(RecoverStateValidation,
              InvalidDtRkThrowsWithoutModification,
              ())
{
    // Generate small mesh dimensions
    const auto nCells = *rc::gen::inRange<index_type>(1, 6);
    const auto nEdges = *rc::gen::inRange<index_type>(1, 16);
    const auto nVertLevels = *rc::gen::inRange<index_type>(3, 11);

    // Generate valid n_acoustic > 0
    const auto n_acoustic = *rc::gen::inRange<index_type>(1, 20);

    // Generate invalid dt_rk <= 0
    const auto dt_rk_int = *rc::gen::inRange<int>(-100, 1);  // [-100, 0]
    const real_type dt_rk = static_cast<real_type>(dt_rk_int) * 0.1;

    RecoverStateValidationData data(nCells, nEdges, nVertLevels);

    // Verify that std::invalid_argument is thrown
    RC_ASSERT_THROWS_AS(data.callRecoverState(dt_rk, n_acoustic), std::invalid_argument);

    // Verify output arrays are unchanged
    RC_ASSERT(data.outputsUnchanged());
}
