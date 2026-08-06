/// @file prop_vert_implicit.cpp
/// @brief Property-based tests for the vertical implicit coefficient computation kernel.
///
/// **Validates: Requirements 16.1, 16.4**
///
/// Tests the faithful port of atm_compute_vert_imp_coefs_work.
/// Property 33: All outputs are finite and the LU factorization is well-defined.
/// Deterministic test: Constant-profile known-answer verification of alpha_tri/gamma_tri.

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
// Property 33: Vertical Implicit Coefficients — Outputs Finite & LU Well-Defined
// ============================================================================

RC_GTEST_PROP(VertImplicitStability, OutputsFiniteAndLUWellDefined,
              ()) {
    const auto nCells = *rc::gen::inRange(1, 6);
    const auto nVertLevels = *rc::gen::inRange(3, 20);
    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(nCells);
    const real_type dts = 1.0 + (*rc::gen::inRange(0, 19001)) / 1000.0;

    // Input arrays
    std::vector<real_type> zz_data(nLev * nC);
    for (auto& v : zz_data) v = 0.9 + (*rc::gen::inRange(0, 2001)) / 10000.0;

    std::vector<real_type> p_data(nLev * nC);
    for (auto& v : p_data) v = 80000.0 + (*rc::gen::inRange(0, 25001));

    std::vector<real_type> t_data(nLev * nC);
    for (auto& v : t_data) v = 250.0 + (*rc::gen::inRange(0, 10001)) / 100.0;

    std::vector<real_type> rb_data(nLev * nC);
    for (auto& v : rb_data) v = 0.3 + (*rc::gen::inRange(0, 12001)) / 10000.0;

    std::vector<real_type> rtb_data(nLev * nC);
    for (auto& v : rtb_data) v = 250000.0 + (*rc::gen::inRange(0, 150001));

    std::vector<real_type> pb_data(nLev * nC);
    for (auto& v : pb_data) v = 80000.0 + (*rc::gen::inRange(0, 25001));

    std::vector<real_type> rt_data(nLev * nC);
    for (auto& v : rt_data) v = (*rc::gen::inRange(-100, 101));

    std::vector<real_type> cqw_data(nLev * nC);
    for (auto& v : cqw_data) v = 0.95 + (*rc::gen::inRange(0, 501)) / 10000.0;

    std::vector<real_type> qtot_data(nLev * nC);
    for (auto& v : qtot_data) v = (*rc::gen::inRange(0, 301)) / 10000.0;

    std::vector<real_type> rdzw_data(nLev);
    for (auto& v : rdzw_data) v = 0.001 + (*rc::gen::inRange(0, 9001)) / 1000000.0;

    std::vector<real_type> fzm_data(nLev);
    std::vector<real_type> fzp_data(nLev);
    for (std::size_t i = 0; i < nLev; ++i) {
        fzm_data[i] = 0.4 + (*rc::gen::inRange(0, 2001)) / 10000.0;
        fzp_data[i] = 1.0 - fzm_data[i];
    }

    std::vector<real_type> rdzu_data(nLev);
    for (auto& v : rdzu_data) v = 0.001 + (*rc::gen::inRange(0, 9001)) / 1000000.0;

    std::vector<real_type> etp_data(nLev);
    for (auto& v : etp_data) v = 0.3 + (*rc::gen::inRange(0, 4001)) / 10000.0;

    std::vector<real_type> ewp_data(nLev + 1);
    for (auto& v : ewp_data) v = 0.3 + (*rc::gen::inRange(0, 4001)) / 10000.0;

    // Create input mdspan views
    ConstField2D<default_layout, unchecked_accessor> zz(zz_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> p_in(p_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> t_in(t_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rb(rb_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtb(rtb_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> pb(pb_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rt(rt_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> cqw(cqw_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> qtot(qtot_data.data(), nVertLevels, nCells);

    ConstSpan1D rdzw(rdzw_data.data(), nVertLevels);
    ConstSpan1D fzm(fzm_data.data(), nVertLevels);
    ConstSpan1D fzp(fzp_data.data(), nVertLevels);
    ConstSpan1D rdzu(rdzu_data.data(), nVertLevels);
    ConstSpan1D etp(etp_data.data(), nVertLevels);
    ConstSpan1D ewp(ewp_data.data(), nVertLevels + 1);

    // Allocate output arrays
    std::vector<real_type> cofwr_data(nLev * nC, 0.0);
    std::vector<real_type> cofwz_data(nLev * nC, 0.0);
    std::vector<real_type> coftz_data((nLev + 1) * nC, 0.0);
    std::vector<real_type> cofwt_data(nLev * nC, 0.0);
    std::vector<real_type> a_tri_data(nLev * nC, 0.0);
    std::vector<real_type> alpha_tri_data(nLev * nC, 0.0);
    std::vector<real_type> gamma_tri_data(nLev * nC, 0.0);
    std::vector<real_type> cofrz_data(nLev, 0.0);

    // Create output mdspan views
    Field2D<default_layout, unchecked_accessor> cofwr_out(cofwr_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> cofwz_out(cofwz_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> coftz_out(coftz_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> cofwt_out(cofwt_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> a_tri_out(a_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> alpha_out(alpha_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> gamma_out(gamma_tri_data.data(), nVertLevels, nCells);
    Span1D cofrz_out(cofrz_data.data(), nVertLevels);

    // Call the kernel
    compute_vert_imp_coefs<default_layout>(
        SerialPolicy{},
        cofwr_out, cofwz_out, coftz_out, cofwt_out,
        a_tri_out, alpha_out, gamma_out, cofrz_out,
        zz, p_in, t_in, rb, rtb, pb, rt, cqw, qtot,
        rdzw, fzm, fzp, rdzu, etp, ewp,
        dts,
        static_cast<index_type>(nCells),
        static_cast<index_type>(nVertLevels));

    // Property 1: cofrz[k] == rdzw[k]
    for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
        RC_ASSERT(cofrz_out[k] == rdzw_data[static_cast<std::size_t>(k)]);
    }

    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        // Property 2: coftz boundary conditions
        auto coftz_top = coftz_out[0, iCell];
        auto coftz_bot = coftz_out[static_cast<index_type>(nVertLevels), iCell];
        RC_ASSERT(coftz_top == 0.0);
        RC_ASSERT(coftz_bot == 0.0);

        // Property 3: Initialization at k=0
        auto a0 = a_tri_out[0, iCell];
        auto g0 = gamma_out[0, iCell];
        auto al0 = alpha_out[0, iCell];
        RC_ASSERT(a0 == 0.0);
        RC_ASSERT(g0 == 0.0);
        RC_ASSERT(al0 == 0.0);

        // Property 4: All outputs finite
        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            auto v1 = cofwr_out[k, iCell];
            auto v2 = cofwz_out[k, iCell];
            auto v3 = cofwt_out[k, iCell];
            auto v4 = a_tri_out[k, iCell];
            auto v5 = alpha_out[k, iCell];
            auto v6 = gamma_out[k, iCell];
            RC_ASSERT(std::isfinite(v1));
            RC_ASSERT(std::isfinite(v2));
            RC_ASSERT(std::isfinite(v3));
            RC_ASSERT(std::isfinite(v4));
            RC_ASSERT(std::isfinite(v5));
            RC_ASSERT(std::isfinite(v6));
        }
        for (index_type k = 0; k <= static_cast<index_type>(nVertLevels); ++k) {
            auto v = coftz_out[k, iCell];
            RC_ASSERT(std::isfinite(v));
        }

        // Property 5: alpha_tri[k] != 0 for k>=1 (LU factorization valid)
        for (index_type k = 1; k < static_cast<index_type>(nVertLevels); ++k) {
            auto val = alpha_out[k, iCell];
            RC_ASSERT(val != 0.0);
        }
    }
}

// ============================================================================
// Deterministic Test: Constant profile known-answer verification
// ============================================================================
// Verify the LU factorization (alpha_tri, gamma_tri) matches hand computation.

TEST(VertImplicitDeterministic, ConstantProfileLUFactorization) {
    const index_type nCells = 1;
    const index_type nVertLevels = 5;
    const real_type dts = 5.0;

    const real_type rgas = constants::rdry;
    const real_type cp   = constants::cpdry;
    const real_type grav = constants::gravity;
    const real_type rcv  = rgas / (cp - rgas);
    const real_type c2   = cp * rcv;

    // Constant input values
    const real_type zz_val   = 1.0;
    const real_type p_val    = 100000.0;
    const real_type t_val    = 300.0;
    const real_type rb_val   = 1.0;
    const real_type rtb_val  = 300000.0;
    const real_type pb_val   = 100000.0;
    const real_type rt_val   = 0.0;
    const real_type cqw_val  = 1.0;
    const real_type qtot_val = 0.0;
    const real_type rdzw_val = 0.002;
    const real_type fzm_val  = 0.5;
    const real_type fzp_val  = 0.5;
    const real_type rdzu_val = 0.002;
    const real_type etp_val  = 0.5;
    const real_type ewp_val  = 0.5;

    const std::size_t nLev = static_cast<std::size_t>(nVertLevels);

    // Allocate input arrays
    std::vector<real_type> zz_data(nLev, zz_val);
    std::vector<real_type> p_data(nLev, p_val);
    std::vector<real_type> t_data(nLev, t_val);
    std::vector<real_type> rb_data(nLev, rb_val);
    std::vector<real_type> rtb_data(nLev, rtb_val);
    std::vector<real_type> pb_data(nLev, pb_val);
    std::vector<real_type> rt_data(nLev, rt_val);
    std::vector<real_type> cqw_data(nLev, cqw_val);
    std::vector<real_type> qtot_data(nLev, qtot_val);
    std::vector<real_type> rdzw_data(nLev, rdzw_val);
    std::vector<real_type> fzm_data(nLev, fzm_val);
    std::vector<real_type> fzp_data(nLev, fzp_val);
    std::vector<real_type> rdzu_data(nLev, rdzu_val);
    std::vector<real_type> etp_data(nLev, etp_val);
    std::vector<real_type> ewp_data(nLev + 1, ewp_val);

    // Create input mdspan views
    ConstField2D<default_layout, unchecked_accessor> zz(zz_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> p_in(p_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> t_in(t_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rb(rb_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtb(rtb_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> pb(pb_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rt(rt_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> cqw(cqw_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> qtot(qtot_data.data(), nVertLevels, nCells);

    ConstSpan1D rdzw(rdzw_data.data(), nVertLevels);
    ConstSpan1D fzm(fzm_data.data(), nVertLevels);
    ConstSpan1D fzp(fzp_data.data(), nVertLevels);
    ConstSpan1D rdzu(rdzu_data.data(), nVertLevels);
    ConstSpan1D etp(etp_data.data(), nVertLevels);
    ConstSpan1D ewp(ewp_data.data(), nVertLevels + 1);

    // Allocate output arrays
    std::vector<real_type> cofwr_data(nLev, 0.0);
    std::vector<real_type> cofwz_data(nLev, 0.0);
    std::vector<real_type> coftz_data(nLev + 1, 0.0);
    std::vector<real_type> cofwt_data(nLev, 0.0);
    std::vector<real_type> a_tri_data(nLev, 0.0);
    std::vector<real_type> alpha_tri_data(nLev, 0.0);
    std::vector<real_type> gamma_tri_data(nLev, 0.0);
    std::vector<real_type> cofrz_data(nLev, 0.0);

    // Create output mdspan views
    Field2D<default_layout, unchecked_accessor> cofwr_out(cofwr_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> cofwz_out(cofwz_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> coftz_out(coftz_data.data(), nVertLevels + 1, nCells);
    Field2D<default_layout, unchecked_accessor> cofwt_out(cofwt_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> a_tri_out(a_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> alpha_out(alpha_tri_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> gamma_out(gamma_tri_data.data(), nVertLevels, nCells);
    Span1D cofrz_out(cofrz_data.data(), nVertLevels);

    // Call the kernel
    compute_vert_imp_coefs<default_layout>(
        SerialPolicy{},
        cofwr_out, cofwz_out, coftz_out, cofwt_out,
        a_tri_out, alpha_out, gamma_out, cofrz_out,
        zz, p_in, t_in, rb, rtb, pb, rt, cqw, qtot,
        rdzw, fzm, fzp, rdzu, etp, ewp,
        dts, nCells, nVertLevels);

    const real_type tol = 1.0e-10;

    // Verify cofrz
    for (index_type k = 0; k < nVertLevels; ++k) {
        EXPECT_DOUBLE_EQ(cofrz_out[k], rdzw_val);
    }

    // Hand-computed expected intermediate values for constant profile:
    // cofwr[k] = 0.5*gravity*(fzm*zz + fzp*zz) = 0.5*grav*1.0 = 0.5*gravity
    const real_type exp_cofwr = 0.5 * grav * (fzm_val * zz_val + fzp_val * zz_val);

    // coftz[k] = fzm*t + fzp*t = 300.0 for interior, 0 at boundaries
    const real_type exp_coftz = fzm_val * t_val + fzp_val * t_val;

    // cofwz[k] = c2*(fzm*zz+fzp*zz)*rdzu*cqw*(fzm*p+fzp*p)
    const real_type zz_iface = fzm_val * zz_val + fzp_val * zz_val;
    const real_type p_iface  = fzm_val * p_val + fzp_val * p_val;
    const real_type exp_cofwz = c2 * zz_iface * rdzu_val * cqw_val * p_iface;

    // cofwt[k] = 0.5*rcv*zz*gravity*rb/(1+qtot) * p/((rtb+rt)*pb)
    const real_type exp_cofwt = 0.5 * rcv * zz_val * grav * rb_val / (1.0 + qtot_val)
                                * p_val / ((rtb_val + rt_val) * pb_val);

    // Verify intermediate coefficients
    for (index_type k = 1; k < nVertLevels; ++k) {
        auto v_cofwr = cofwr_out[k, 0];
        auto v_cofwz = cofwz_out[k, 0];
        EXPECT_NEAR(v_cofwr, exp_cofwr, tol) << "cofwr at k=" << k;
        EXPECT_NEAR(v_cofwz, exp_cofwz, tol) << "cofwz at k=" << k;
    }

    auto coftz_0 = coftz_out[0, 0];
    EXPECT_DOUBLE_EQ(coftz_0, 0.0);
    for (index_type k = 1; k < nVertLevels; ++k) {
        auto v = coftz_out[k, 0];
        EXPECT_NEAR(v, exp_coftz, tol) << "coftz at k=" << k;
    }
    auto coftz_top = coftz_out[nVertLevels, 0];
    EXPECT_DOUBLE_EQ(coftz_top, 0.0);

    for (index_type k = 0; k < nVertLevels; ++k) {
        auto v = cofwt_out[k, 0];
        EXPECT_NEAR(v, exp_cofwt, tol) << "cofwt at k=" << k;
    }

    // ---- Verify LU factorization by recomputing the forward sweep ----
    // Recompute a_tri, b_tri, c_tri from the expected constant values,
    // then run the forward sweep and compare.
    const real_type dts2 = dts * dts;

    // Helper to get coftz for a given index
    auto get_coftz = [&](index_type k) -> real_type {
        if (k == 0 || k == nVertLevels) return 0.0;
        return exp_coftz;
    };

    std::vector<real_type> a_hand(nLev, 0.0);
    std::vector<real_type> b_hand(nLev, 0.0);
    std::vector<real_type> c_hand(nLev, 0.0);

    for (index_type k = 1; k < nVertLevels; ++k) {
        auto ki = static_cast<std::size_t>(k);
        // a_tri(k) = (-cofwz*coftz(k-1)*rdzw*zz + cofwr*cofrz(k-1) - cofwt*coftz(k-1)*rdzw)
        //            * etp(k-1)*ewp(k-1)
        a_hand[ki] = (-exp_cofwz * get_coftz(k - 1) * rdzw_val * zz_val
                      + exp_cofwr * rdzw_val
                      - exp_cofwt * get_coftz(k - 1) * rdzw_val)
                     * etp_val * ewp_val;

        // b_tri(k) = (cofwz*coftz(k)*(etp*rdzw*zz + etp*rdzw*zz)
        //            - coftz(k)*(etp*cofwt*rdzw - etp*cofwt*rdzw)
        //            + cofwr*(etp*cofrz(k) - etp*cofrz(k-1))) * ewp
        b_hand[ki] = (exp_cofwz * get_coftz(k)
                          * (etp_val * rdzw_val * zz_val + etp_val * rdzw_val * zz_val)
                      - get_coftz(k)
                          * (etp_val * exp_cofwt * rdzw_val - etp_val * exp_cofwt * rdzw_val)
                      + exp_cofwr
                          * (etp_val * rdzw_val - etp_val * rdzw_val))
                     * ewp_val;

        // c_tri(k) = (-cofwz*coftz(k+1)*rdzw*zz - cofwr*cofrz(k) + cofwt*coftz(k+1)*rdzw)
        //            * etp(k)*ewp(k+1)
        c_hand[ki] = (-exp_cofwz * get_coftz(k + 1) * rdzw_val * zz_val
                      - exp_cofwr * rdzw_val
                      + exp_cofwt * get_coftz(k + 1) * rdzw_val)
                     * etp_val * ewp_val;
    }
    c_hand[nLev - 1] = 0.0;

    // Forward sweep
    std::vector<real_type> alpha_hand(nLev, 0.0);
    std::vector<real_type> gamma_hand(nLev, 0.0);

    for (index_type k = 1; k < nVertLevels; ++k) {
        auto ki = static_cast<std::size_t>(k);
        alpha_hand[ki] = 1.0 / (1.0 + dts2 * (b_hand[ki] - a_hand[ki] * gamma_hand[ki - 1]));
        gamma_hand[ki] = dts2 * c_hand[ki] * alpha_hand[ki];
    }

    // Verify alpha_tri and gamma_tri match
    auto al0 = alpha_out[0, 0];
    auto gm0 = gamma_out[0, 0];
    EXPECT_DOUBLE_EQ(al0, 0.0);
    EXPECT_DOUBLE_EQ(gm0, 0.0);

    for (index_type k = 1; k < nVertLevels; ++k) {
        auto ki = static_cast<std::size_t>(k);
        auto val_alpha = alpha_out[k, 0];
        auto val_gamma = gamma_out[k, 0];
        EXPECT_NEAR(val_alpha, alpha_hand[ki], tol)
            << "alpha_tri mismatch at k=" << k;
        EXPECT_NEAR(val_gamma, gamma_hand[ki], tol)
            << "gamma_tri mismatch at k=" << k;
    }

    // Also verify a_tri matches
    auto a0 = a_tri_out[0, 0];
    EXPECT_DOUBLE_EQ(a0, 0.0);
    for (index_type k = 1; k < nVertLevels; ++k) {
        auto ki = static_cast<std::size_t>(k);
        auto val_a = a_tri_out[k, 0];
        EXPECT_NEAR(val_a, a_hand[ki], tol) << "a_tri mismatch at k=" << k;
    }
}
