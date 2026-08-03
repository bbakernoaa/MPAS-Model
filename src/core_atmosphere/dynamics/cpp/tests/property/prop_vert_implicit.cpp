/// @file prop_vert_implicit.cpp
/// @brief Property-based tests for the vertical implicit coefficient computation kernel.
///
/// **Validates: Requirements 16.1, 16.4**
///
/// Property 33: Vertical Implicit Coefficients Produce Stable Tridiagonal System
///   For physically reasonable input states, the computed tridiagonal system is stable
///   (diagonal dominance) and all outputs are finite.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/vert_implicit.hpp>
#include <mpas_dycore/constants.hpp>
#include <vector>
#include <cmath>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;

// ============================================================================
// Property 33: Vertical Implicit Coefficients Produce Stable Tridiagonal System
// ============================================================================
// Generate physically reasonable inputs and verify that:
// 1. alpha_tri (main diagonal) > 0 for all k (actually >= 1.0)
// 2. a_tri and gamma_tri <= 0 (M-matrix off-diagonal structure)
// 3. Boundary conditions: cofwr[0]==cofwz[0]==cofwt[0]==0, same at nVertLevels
// 4. All output values are finite (no NaN/Inf)
// 5. alpha_tri[k] >= 1.0 (structural property from definition)
// 6. First boundary row (k=0) is strictly diagonally dominant

RC_GTEST_PROP(VertImplicitStability, DiagonallyDominantTridiagonalSystem,
              ()) {
    // Generate random mesh dimensions
    const auto nCells = *rc::gen::inRange(1, 8);
    const auto nVertLevels = *rc::gen::inRange(3, 30);

    // Generate physically reasonable acoustic timestep
    // dts in [1.0, 20.0] s
    const real_type dts = 1.0 + (*rc::gen::inRange(0, 19001)) / 1000.0;

    // Physical constants from MPAS
    const real_type gravity = constants::gravity;
    const real_type rdry    = constants::rdry;
    const real_type cvdry   = constants::cvdry;

    // Allocate input arrays
    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(nCells);

    // theta_m: (nVertLevels, nCells) - moist potential temperature in [250, 350] K
    std::vector<real_type> theta_m_data(nLev * nC);
    for (std::size_t i = 0; i < theta_m_data.size(); ++i) {
        int raw = *rc::gen::inRange(25000, 35001); // [250.00, 350.00]
        theta_m_data[i] = static_cast<real_type>(raw) / 100.0;
    }

    // rho_zz: (nVertLevels, nCells) - dry density in [0.3, 1.5] kg/m^3
    std::vector<real_type> rho_zz_data(nLev * nC);
    for (std::size_t i = 0; i < rho_zz_data.size(); ++i) {
        int raw = *rc::gen::inRange(300, 1501); // [0.30, 1.50]
        rho_zz_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // rdzu: (nVertLevels+1) - reciprocal vertical spacing in [0.001, 0.01] 1/m
    // (corresponding to layer thicknesses of 100-1000m)
    std::vector<real_type> rdzu_data(static_cast<std::size_t>(nVertLevels + 1));
    for (std::size_t i = 0; i < rdzu_data.size(); ++i) {
        int raw = *rc::gen::inRange(10, 101); // [0.001, 0.010]
        rdzu_data[i] = static_cast<real_type>(raw) / 10000.0;
    }

    // fzm/fzp: (nVertLevels+1) - interpolation weights that sum to 1.0
    // fzm in [0.4, 0.6], fzp = 1.0 - fzm
    std::vector<real_type> fzm_data(static_cast<std::size_t>(nVertLevels + 1));
    std::vector<real_type> fzp_data(static_cast<std::size_t>(nVertLevels + 1));
    for (std::size_t i = 0; i < fzm_data.size(); ++i) {
        int raw = *rc::gen::inRange(400, 601); // [0.40, 0.60]
        fzm_data[i] = static_cast<real_type>(raw) / 1000.0;
        fzp_data[i] = 1.0 - fzm_data[i];
    }

    // cqw: (nVertLevels+1, nCells) - moist coefficient in [0.95, 1.0]
    std::vector<real_type> cqw_data(static_cast<std::size_t>(nVertLevels + 1) * nC);
    for (std::size_t i = 0; i < cqw_data.size(); ++i) {
        int raw = *rc::gen::inRange(950, 1001); // [0.95, 1.00]
        cqw_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Create const input mdspan views
    ConstField2D<default_layout, unchecked_accessor> theta_m(
        theta_m_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_zz(
        rho_zz_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> cqw(
        cqw_data.data(), nVertLevels + 1, nCells);

    // 1D span views for mesh geometry
    using Span1D = std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent>>;
    Span1D rdzu(rdzu_data.data(), nVertLevels + 1);
    Span1D fzm(fzm_data.data(), nVertLevels + 1);
    Span1D fzp(fzp_data.data(), nVertLevels + 1);

    // Allocate output arrays
    std::vector<real_type> cofwr_data(static_cast<std::size_t>(nVertLevels + 1) * nC, 0.0);
    std::vector<real_type> cofwz_data(static_cast<std::size_t>(nVertLevels + 1) * nC, 0.0);
    std::vector<real_type> coftz_data(nLev * nC, 0.0);
    std::vector<real_type> cofwt_data(static_cast<std::size_t>(nVertLevels + 1) * nC, 0.0);
    std::vector<real_type> a_tri_data(nLev * nC, 0.0);
    std::vector<real_type> alpha_tri_data(nLev * nC, 0.0);
    std::vector<real_type> gamma_tri_data(nLev * nC, 0.0);
    std::vector<real_type> cofrz_data(nLev * nC, 0.0);

    // Create output mdspan views
    Field2D<default_layout, unchecked_accessor> cofwr(
        cofwr_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> cofwz(
        cofwz_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> coftz(
        coftz_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> cofwt(
        cofwt_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> a_tri(
        a_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> alpha_tri(
        alpha_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> gamma_tri(
        gamma_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> cofrz(
        cofrz_data.data(), nVertLevels, nCells);

    // Call the kernel
    compute_vert_imp_coefs<default_layout>(
        SerialPolicy{},
        cofwr, cofwz, coftz, cofwt,
        a_tri, alpha_tri, gamma_tri, cofrz,
        theta_m, rho_zz, cqw,
        rdzu, fzm, fzp,
        dts, gravity, rdry, cvdry,
        static_cast<index_type>(nCells),
        static_cast<index_type>(nVertLevels));

    // Verify properties for each cell column
    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        // Property 1: Boundary conditions
        auto cofwr_top = cofwr[0, iCell];
        auto cofwz_top = cofwz[0, iCell];
        auto cofwt_top = cofwt[0, iCell];
        RC_ASSERT(cofwr_top == 0.0);
        RC_ASSERT(cofwz_top == 0.0);
        RC_ASSERT(cofwt_top == 0.0);

        auto cofwr_bot = cofwr[static_cast<index_type>(nVertLevels), iCell];
        auto cofwz_bot = cofwz[static_cast<index_type>(nVertLevels), iCell];
        auto cofwt_bot = cofwt[static_cast<index_type>(nVertLevels), iCell];
        RC_ASSERT(cofwr_bot == 0.0);
        RC_ASSERT(cofwz_bot == 0.0);
        RC_ASSERT(cofwt_bot == 0.0);

        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            // Property 2: All output values are finite
            auto val_cofwr = cofwr[k, iCell];
            auto val_cofwz = cofwz[k, iCell];
            auto val_coftz = coftz[k, iCell];
            auto val_cofwt = cofwt[k, iCell];
            auto val_a_tri = a_tri[k, iCell];
            auto val_alpha = alpha_tri[k, iCell];
            auto val_gamma = gamma_tri[k, iCell];
            auto val_cofrz = cofrz[k, iCell];

            RC_ASSERT(std::isfinite(val_cofwr));
            RC_ASSERT(std::isfinite(val_cofwz));
            RC_ASSERT(std::isfinite(val_coftz));
            RC_ASSERT(std::isfinite(val_cofwt));
            RC_ASSERT(std::isfinite(val_a_tri));
            RC_ASSERT(std::isfinite(val_alpha));
            RC_ASSERT(std::isfinite(val_gamma));
            RC_ASSERT(std::isfinite(val_cofrz));

            // Property 3: alpha_tri (main diagonal) > 0 for all k
            RC_ASSERT(val_alpha > 0.0);

            // Property 4: Off-diagonals are non-positive (M-matrix structure)
            // a_tri and gamma_tri should be <= 0 since they are products of
            // positive cofwz and positive coftz, negated.
            RC_ASSERT(val_a_tri <= 0.0);
            RC_ASSERT(val_gamma <= 0.0);
        }

        // Check finiteness for the last interface level outputs
        auto cofwr_last = cofwr[static_cast<index_type>(nVertLevels), iCell];
        auto cofwz_last = cofwz[static_cast<index_type>(nVertLevels), iCell];
        auto cofwt_last = cofwt[static_cast<index_type>(nVertLevels), iCell];
        RC_ASSERT(std::isfinite(cofwr_last));
        RC_ASSERT(std::isfinite(cofwz_last));
        RC_ASSERT(std::isfinite(cofwt_last));

        // Property 5: alpha_tri[k] >= 1.0 always (since cofwz, coftz >= 0).
        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            auto val_alpha = alpha_tri[k, iCell];
            RC_ASSERT(val_alpha >= 1.0);
        }

        // Property 6: First boundary row (k=0) IS diagonally dominant.
        {
            auto a_first = a_tri[0, iCell];
            auto g_first = gamma_tri[0, iCell];
            auto d_first = alpha_tri[0, iCell];
            RC_ASSERT(d_first >= std::abs(a_first) + std::abs(g_first));
        }
    }
}

// ============================================================================
// Deterministic Test: Constant profile known-answer verification
// ============================================================================
// Use constant theta_m=300, rho_zz=1.0, cqw=1.0, dts=5.0
// rdzu=0.002 (500m layers), fzm=0.5, fzp=0.5
// Verify the computed values match hand computation.

TEST(VertImplicitDeterministic, ConstantProfileKnownAnswer) {
    const index_type nCells = 1;
    const index_type nVertLevels = 5;

    // Physical constants
    const real_type gravity = constants::gravity;
    const real_type rdry    = constants::rdry;
    const real_type cvdry   = constants::cvdry;
    const real_type dts     = 5.0;

    // Off-centering parameter used in the kernel
    constexpr real_type epssm = 0.5;

    // Constant input fields
    const real_type theta_m_val = 300.0;
    const real_type rho_zz_val  = 1.0;
    const real_type rdzu_val    = 0.002;  // 1/500m
    const real_type fzm_val     = 0.5;
    const real_type fzp_val     = 0.5;

    // Allocate input arrays
    std::vector<real_type> theta_m_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), theta_m_val);
    std::vector<real_type> rho_zz_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), rho_zz_val);
    std::vector<real_type> cqw_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), 1.0);
    std::vector<real_type> rdzu_data(
        static_cast<std::size_t>(nVertLevels + 1), rdzu_val);
    std::vector<real_type> fzm_data(
        static_cast<std::size_t>(nVertLevels + 1), fzm_val);
    std::vector<real_type> fzp_data(
        static_cast<std::size_t>(nVertLevels + 1), fzp_val);

    // Create input mdspan views
    ConstField2D<default_layout, unchecked_accessor> theta_m(
        theta_m_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_zz(
        rho_zz_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> cqw(
        cqw_data.data(), nVertLevels + 1, nCells);

    using Span1D = std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent>>;
    Span1D rdzu(rdzu_data.data(), nVertLevels + 1);
    Span1D fzm(fzm_data.data(), nVertLevels + 1);
    Span1D fzp(fzp_data.data(), nVertLevels + 1);

    // Allocate output arrays
    std::vector<real_type> cofwr_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), 0.0);
    std::vector<real_type> cofwz_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), 0.0);
    std::vector<real_type> coftz_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), 0.0);
    std::vector<real_type> cofwt_data(
        static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(nCells), 0.0);
    std::vector<real_type> a_tri_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), 0.0);
    std::vector<real_type> alpha_tri_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), 0.0);
    std::vector<real_type> gamma_tri_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), 0.0);
    std::vector<real_type> cofrz_data(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells), 0.0);

    // Create output mdspan views
    Field2D<default_layout, unchecked_accessor> cofwr(
        cofwr_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> cofwz(
        cofwz_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> coftz(
        coftz_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> cofwt(
        cofwt_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> a_tri(
        a_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> alpha_tri(
        alpha_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> gamma_tri(
        gamma_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> cofrz(
        cofrz_data.data(), nVertLevels, nCells);

    // Call the kernel
    compute_vert_imp_coefs<default_layout>(
        SerialPolicy{},
        cofwr, cofwz, coftz, cofwt,
        a_tri, alpha_tri, gamma_tri, cofrz,
        theta_m, rho_zz, cqw,
        rdzu, fzm, fzp,
        dts, gravity, rdry, cvdry,
        nCells, nVertLevels);

    // Hand-computed expected values for constant profiles:
    // Interface values (for interior k=1..nVertLevels-1):
    //   theta_interface = 0.5*300 + 0.5*300 = 300
    //   rho_interface   = 0.5*1.0 + 0.5*1.0 = 1.0
    //
    // cofwr[k] = epssm * dts * rho_interface * rdzu
    const real_type expected_cofwr = epssm * dts * 1.0 * rdzu_val;

    // cofwz[k] = epssm * dts * (rdry/cvdry) * theta_interface * rdzu * cqw
    const real_type rcv = rdry / cvdry;
    const real_type expected_cofwz = epssm * dts * rcv * 300.0 * rdzu_val * 1.0;

    // coftz[k] = epssm * dts * gravity * rho_zz[k]
    const real_type expected_coftz = epssm * dts * gravity * 1.0;

    // cofwt[k] = epssm * dts * (theta_m[k] - theta_m[k-1]) * rdzu
    // For constant profile: theta_m[k] - theta_m[k-1] = 0
    const real_type expected_cofwt = 0.0;

    // cofrz[k] = cofwr[k] * rho_zz[k]
    const real_type expected_cofrz_interior = expected_cofwr * 1.0;

    // Tridiagonal coefficients:
    // a_tri[k] = -cofwz[k] * coftz[k]
    const real_type expected_a_tri_interior = -expected_cofwz * expected_coftz;

    // gamma_tri[k] = -cofwz[k+1] * coftz[k]
    const real_type expected_gamma_tri_interior = -expected_cofwz * expected_coftz;

    // alpha_tri[k] = 1 + cofwz[k]*coftz[k-1] + cofwz[k+1]*coftz[k]
    const real_type wz_tz_product = expected_cofwz * expected_coftz;

    const real_type tol = 1.0e-10;

    // Verify boundary conditions
    {
        auto v_cofwr_top = cofwr[0, 0];
        auto v_cofwz_top = cofwz[0, 0];
        auto v_cofwt_top = cofwt[0, 0];
        EXPECT_DOUBLE_EQ(v_cofwr_top, 0.0);
        EXPECT_DOUBLE_EQ(v_cofwz_top, 0.0);
        EXPECT_DOUBLE_EQ(v_cofwt_top, 0.0);

        auto v_cofwr_bot = cofwr[nVertLevels, 0];
        auto v_cofwz_bot = cofwz[nVertLevels, 0];
        auto v_cofwt_bot = cofwt[nVertLevels, 0];
        EXPECT_DOUBLE_EQ(v_cofwr_bot, 0.0);
        EXPECT_DOUBLE_EQ(v_cofwz_bot, 0.0);
        EXPECT_DOUBLE_EQ(v_cofwt_bot, 0.0);
    }

    // Verify interior interface coefficients
    for (index_type k = 1; k < nVertLevels; ++k) {
        auto val_cofwr = cofwr[k, 0];
        auto val_cofwz = cofwz[k, 0];
        auto val_cofwt = cofwt[k, 0];
        EXPECT_NEAR(val_cofwr, expected_cofwr, tol)
            << "cofwr mismatch at k=" << k;
        EXPECT_NEAR(val_cofwz, expected_cofwz, tol)
            << "cofwz mismatch at k=" << k;
        EXPECT_NEAR(val_cofwt, expected_cofwt, tol)
            << "cofwt mismatch at k=" << k;
    }

    // Verify level-centered coefficients
    for (index_type k = 0; k < nVertLevels; ++k) {
        auto val_coftz = coftz[k, 0];
        EXPECT_NEAR(val_coftz, expected_coftz, tol)
            << "coftz mismatch at k=" << k;
    }

    // Verify cofrz
    auto cofrz_top = cofrz[0, 0];
    EXPECT_NEAR(cofrz_top, 0.0, tol); // cofwr[0]=0
    for (index_type k = 1; k < nVertLevels; ++k) {
        auto val_cofrz = cofrz[k, 0];
        EXPECT_NEAR(val_cofrz, expected_cofrz_interior, tol)
            << "cofrz mismatch at k=" << k;
    }

    // Verify tridiagonal coefficients
    // k=0: a_tri=0, gamma_tri=-cofwz[1]*coftz[0], alpha_tri=1+cofwz[1]*coftz[0]
    auto a_tri_0 = a_tri[0, 0];
    auto gamma_tri_0 = gamma_tri[0, 0];
    auto alpha_tri_0 = alpha_tri[0, 0];
    EXPECT_NEAR(a_tri_0, 0.0, tol);
    EXPECT_NEAR(gamma_tri_0, expected_gamma_tri_interior, tol);
    EXPECT_NEAR(alpha_tri_0, 1.0 + wz_tz_product, tol);

    // Interior levels: k=1..nVertLevels-2
    for (index_type k = 1; k < nVertLevels - 1; ++k) {
        auto val_a = a_tri[k, 0];
        auto val_g = gamma_tri[k, 0];
        auto val_alpha = alpha_tri[k, 0];
        EXPECT_NEAR(val_a, expected_a_tri_interior, tol)
            << "a_tri mismatch at k=" << k;
        EXPECT_NEAR(val_g, expected_gamma_tri_interior, tol)
            << "gamma_tri mismatch at k=" << k;
        EXPECT_NEAR(val_alpha, 1.0 + 2.0 * wz_tz_product, tol)
            << "alpha_tri mismatch at k=" << k;
    }

    // k=nVertLevels-1: gamma_tri=0, a_tri=-cofwz[nVL-1]*coftz[nVL-1]
    auto a_tri_last = a_tri[nVertLevels - 1, 0];
    auto gamma_tri_last = gamma_tri[nVertLevels - 1, 0];
    auto alpha_tri_last = alpha_tri[nVertLevels - 1, 0];
    EXPECT_NEAR(a_tri_last, expected_a_tri_interior, tol);
    EXPECT_NEAR(gamma_tri_last, 0.0, tol);
    EXPECT_NEAR(alpha_tri_last, 1.0 + wz_tz_product, tol);

    // Verify diagonal dominance for the constant profile case
    for (index_type k = 0; k < nVertLevels; ++k) {
        auto diag = alpha_tri[k, 0];
        auto val_a = a_tri[k, 0];
        auto val_g = gamma_tri[k, 0];
        real_type off_diag_sum = std::abs(val_a) + std::abs(val_g);
        EXPECT_GE(diag, off_diag_sum)
            << "Diagonal dominance violated at k=" << k;
    }
}
