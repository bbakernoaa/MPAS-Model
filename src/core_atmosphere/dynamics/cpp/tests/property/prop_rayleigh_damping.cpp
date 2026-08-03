/// @file prop_rayleigh_damping.cpp
/// @brief Property-based tests for the Rayleigh damping kernel.
///
/// **Validates: Requirements 21.1, 21.2, 21.3, 21.4**
///
/// Property 38: Rayleigh Damping Disabled No-Op
///   When enabled=false, tend_u is unchanged bit-for-bit.
///
/// Property 39: Rayleigh Damping Linear Ramp
///   The damping coefficient increases linearly from the bottom of the
///   sponge layer to the top (k=0).

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/rayleigh_damping.hpp>
#include <mpas_dycore/constants.hpp>
#include <vector>
#include <cmath>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;

// ============================================================================
// Property 38: Rayleigh Damping Disabled No-Op
// ============================================================================
// When enabled=false, tend_u is unchanged bit-for-bit regardless of input values.

RC_GTEST_PROP(RayleighDampingDisabled, TendUUnchangedWhenDisabled,
              ()) {
    // Generate random mesh dimensions
    const auto nEdges = *rc::gen::inRange(1, 10);
    const auto nVertLevels = *rc::gen::inRange(2, 20);
    const auto n_damp_levels = *rc::gen::inRange(1, nVertLevels + 1);

    // Generate timescale in days [0.1, 10.0]
    const real_type timescale_days = 0.1 + (*rc::gen::inRange(0, 9900)) / 1000.0;

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);

    // Fill tend_u with known values
    std::vector<real_type> tend_u_data(nLev * nE);
    for (std::size_t i = 0; i < tend_u_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        tend_u_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Save a copy before calling the kernel
    std::vector<real_type> tend_u_original(tend_u_data);

    // Generate arbitrary u and rho_edge values
    std::vector<real_type> u_data(nLev * nE);
    for (std::size_t i = 0; i < u_data.size(); ++i) {
        int raw = *rc::gen::inRange(-50000, 50001);
        u_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    std::vector<real_type> rho_edge_data(nLev * nE);
    for (std::size_t i = 0; i < rho_edge_data.size(); ++i) {
        int raw = *rc::gen::inRange(100, 2001); // [0.1, 2.0]
        rho_edge_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tend_u(
        tend_u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(
        rho_edge_data.data(), nVertLevels, nEdges);

    // Call with enabled=false
    rayleigh_damp_u<default_layout>(
        SerialPolicy{},
        tend_u, u, rho_edge,
        false,  // enabled = false
        static_cast<index_type>(n_damp_levels),
        timescale_days,
        static_cast<index_type>(nEdges),
        static_cast<index_type>(nVertLevels));

    // Verify tend_u is unchanged bit-for-bit
    for (std::size_t i = 0; i < tend_u_data.size(); ++i) {
        RC_ASSERT(tend_u_data[i] == tend_u_original[i]);
    }
}

// ============================================================================
// Property 39: Rayleigh Damping Linear Ramp Verification
// ============================================================================
// With known u=1.0, rho_edge=1.0, verify the damping coefficient ramps linearly.

RC_GTEST_PROP(RayleighDampingLinearRamp, CoefficientIncreasesLinearlyToTop,
              ()) {
    // Generate random mesh dimensions
    const auto nEdges = *rc::gen::inRange(1, 8);
    const auto nVertLevels = *rc::gen::inRange(5, 20);
    const auto n_damp_levels = *rc::gen::inRange(2, nVertLevels);

    // Generate timescale in days [0.5, 5.0]
    const real_type timescale_days = 0.5 + (*rc::gen::inRange(0, 4500)) / 1000.0;

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);

    // Set u=1.0 and rho_edge=1.0 for clean coefficient verification
    std::vector<real_type> u_data(nLev * nE, 1.0);
    std::vector<real_type> rho_edge_data(nLev * nE, 1.0);

    // Initialize tend_u to zero
    std::vector<real_type> tend_u_data(nLev * nE, 0.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tend_u(
        tend_u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(
        rho_edge_data.data(), nVertLevels, nEdges);

    // Call with enabled=true
    rayleigh_damp_u<default_layout>(
        SerialPolicy{},
        tend_u, u, rho_edge,
        true,
        static_cast<index_type>(n_damp_levels),
        timescale_days,
        static_cast<index_type>(nEdges),
        static_cast<index_type>(nVertLevels));

    // Verify the linear ramp: coef(k) = (n_damp_levels - k) / (n_damp_levels * timescale_days * 86400)
    // Since tend_u starts at 0 and u=1, rho_edge=1:
    //   tend_u[k, iEdge] = -coef(k)
    const real_type seconds_per_day = constants::seconds_per_day;
    const real_type rayleigh_coef_inverse =
        1.0 / (static_cast<real_type>(n_damp_levels) * timescale_days * seconds_per_day);

    for (index_type iEdge = 0; iEdge < static_cast<index_type>(nEdges); ++iEdge) {
        for (index_type k = 0; k < static_cast<index_type>(n_damp_levels); ++k) {
            const real_type expected_coef =
                static_cast<real_type>(n_damp_levels - k) * rayleigh_coef_inverse;
            const real_type expected_tend = -expected_coef;
            const real_type actual_tend = tend_u[k, iEdge];
            // Check relative tolerance
            RC_ASSERT(std::abs(actual_tend - expected_tend) < 1.0e-12);
        }

        // Verify linear ordering: damping at k is stronger than at k+1
        for (index_type k = 0; k < static_cast<index_type>(n_damp_levels) - 1; ++k) {
            // tend_u[k] should be more negative than tend_u[k+1]
            auto val_k = tend_u[k, iEdge];
            auto val_k1 = tend_u[k + 1, iEdge];
            RC_ASSERT(val_k < val_k1);
        }
    }
}

// ============================================================================
// Deterministic Test: Zero velocity produces zero damping
// ============================================================================
// When u=0 everywhere, damping adds nothing regardless of rho_edge values.

TEST(RayleighDampingDeterministic, ZeroVelocityProducesZeroDamping) {
    const index_type nEdges = 4;
    const index_type nVertLevels = 8;
    const index_type n_damp_levels = 5;
    const real_type timescale_days = 1.0;

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);

    // u = 0 everywhere
    std::vector<real_type> u_data(nLev * nE, 0.0);

    // rho_edge = arbitrary nonzero values
    std::vector<real_type> rho_edge_data(nLev * nE);
    for (std::size_t i = 0; i < rho_edge_data.size(); ++i) {
        rho_edge_data[i] = 0.5 + static_cast<real_type>(i % 10) * 0.1;
    }

    // Initialize tend_u to some known nonzero values
    std::vector<real_type> tend_u_data(nLev * nE);
    for (std::size_t i = 0; i < tend_u_data.size(); ++i) {
        tend_u_data[i] = 1.0 + static_cast<real_type>(i) * 0.01;
    }

    // Save original
    std::vector<real_type> tend_u_original(tend_u_data);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tend_u(
        tend_u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(
        rho_edge_data.data(), nVertLevels, nEdges);

    // Call with enabled=true
    rayleigh_damp_u<default_layout>(
        SerialPolicy{},
        tend_u, u, rho_edge,
        true,
        n_damp_levels,
        timescale_days,
        nEdges,
        nVertLevels);

    // Since u=0, damping contribution is zero: tend_u should be unchanged
    for (std::size_t i = 0; i < tend_u_data.size(); ++i) {
        EXPECT_DOUBLE_EQ(tend_u_data[i], tend_u_original[i])
            << "tend_u changed at index " << i << " despite u=0";
    }
}

// ============================================================================
// Deterministic Test: Levels beyond sponge are untouched
// ============================================================================
// For k >= n_damp_levels, verify tend_u is not modified.

TEST(RayleighDampingDeterministic, LevelsBeyondSpongeUntouched) {
    const index_type nEdges = 3;
    const index_type nVertLevels = 10;
    const index_type n_damp_levels = 4;
    const real_type timescale_days = 2.0;

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);

    // Non-zero u and rho_edge
    std::vector<real_type> u_data(nLev * nE, 5.0);
    std::vector<real_type> rho_edge_data(nLev * nE, 1.2);

    // Initialize tend_u to a sentinel pattern
    std::vector<real_type> tend_u_data(nLev * nE);
    for (std::size_t i = 0; i < tend_u_data.size(); ++i) {
        tend_u_data[i] = 42.0 + static_cast<real_type>(i);
    }

    // Save original
    std::vector<real_type> tend_u_original(tend_u_data);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tend_u(
        tend_u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(
        rho_edge_data.data(), nVertLevels, nEdges);

    // Call with enabled=true
    rayleigh_damp_u<default_layout>(
        SerialPolicy{},
        tend_u, u, rho_edge,
        true,
        n_damp_levels,
        timescale_days,
        nEdges,
        nVertLevels);

    // Verify levels k >= n_damp_levels are unchanged
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        for (index_type k = n_damp_levels; k < nVertLevels; ++k) {
            // Compute flat index for default_layout (column-major: k + nVertLevels * iEdge)
            std::size_t idx = static_cast<std::size_t>(k) +
                              static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(iEdge);
            EXPECT_DOUBLE_EQ(tend_u_data[idx], tend_u_original[idx])
                << "tend_u modified beyond sponge at k=" << k << ", iEdge=" << iEdge;
        }
    }

    // Verify levels k < n_damp_levels ARE modified (damping applied)
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        for (index_type k = 0; k < n_damp_levels; ++k) {
            std::size_t idx = static_cast<std::size_t>(k) +
                              static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(iEdge);
            EXPECT_LT(tend_u_data[idx], tend_u_original[idx])
                << "tend_u not damped at k=" << k << ", iEdge=" << iEdge;
        }
    }
}

// ============================================================================
// Deterministic Test: Known-answer linear ramp with n_damp_levels=5
// ============================================================================
// Verify exact coefficient values for a specific configuration.

TEST(RayleighDampingDeterministic, KnownAnswerLinearRamp) {
    const index_type nEdges = 2;
    const index_type nVertLevels = 8;
    const index_type n_damp_levels = 5;
    const real_type timescale_days = 1.0;

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEdges);

    // u=1.0, rho_edge=1.0 to isolate the coefficient
    std::vector<real_type> u_data(nLev * nE, 1.0);
    std::vector<real_type> rho_edge_data(nLev * nE, 1.0);

    // Initialize tend_u to zero
    std::vector<real_type> tend_u_data(nLev * nE, 0.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tend_u(
        tend_u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> u(
        u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(
        rho_edge_data.data(), nVertLevels, nEdges);

    rayleigh_damp_u<default_layout>(
        SerialPolicy{},
        tend_u, u, rho_edge,
        true,
        n_damp_levels,
        timescale_days,
        nEdges,
        nVertLevels);

    // Expected coefficients:
    // coef(k) = (5 - k) / (5 * 1.0 * 86400)
    // k=0: 5/(5*86400) = 1/86400
    // k=1: 4/(5*86400)
    // k=2: 3/(5*86400)
    // k=3: 2/(5*86400)
    // k=4: 1/(5*86400)
    const real_type base = 5.0 * 1.0 * constants::seconds_per_day;
    const double tol = 1.0e-14;

    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        // k=0: expected = -5/base = -1/86400
        auto val_k0 = tend_u[0, iEdge];
        EXPECT_NEAR(val_k0, -5.0 / base, tol);
        // k=1: expected = -4/base
        auto val_k1 = tend_u[1, iEdge];
        EXPECT_NEAR(val_k1, -4.0 / base, tol);
        // k=2: expected = -3/base
        auto val_k2 = tend_u[2, iEdge];
        EXPECT_NEAR(val_k2, -3.0 / base, tol);
        // k=3: expected = -2/base
        auto val_k3 = tend_u[3, iEdge];
        EXPECT_NEAR(val_k3, -2.0 / base, tol);
        // k=4: expected = -1/base
        auto val_k4 = tend_u[4, iEdge];
        EXPECT_NEAR(val_k4, -1.0 / base, tol);

        // k >= 5: unchanged (still zero)
        for (index_type k = n_damp_levels; k < nVertLevels; ++k) {
            auto val_k_above = tend_u[k, iEdge];
            EXPECT_DOUBLE_EQ(val_k_above, 0.0)
                << "Level k=" << k << " should be unchanged";
        }
    }
}
