/// @file prop_recover_state.cpp
/// @brief Property-based tests for the state recovery kernel.
///
/// **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
///
/// Property 3: State recovery algebraic correctness
/// After calling recover_state, every output element matches the documented
/// algebraic formula to within floating-point tolerance (relative error < 1e-14).

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/recover_state.hpp>
#include <vector>
#include <cmath>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;

namespace {

/// Relative error check. For values near zero, falls back to absolute comparison.
bool rel_eq(real_type actual, real_type expected, real_type tol = 1.0e-14) {
    if (expected == 0.0) {
        return std::abs(actual) <= tol;
    }
    return std::abs((actual - expected) / expected) <= tol;
}

} // anonymous namespace

// ============================================================================
// Property 3: State recovery algebraic correctness
// ============================================================================

RC_GTEST_PROP(RecoverStateAlgebraic, OutputsMatchFormula, ()) {
    // Generate small mesh dimensions
    const auto nCells = *rc::gen::inRange(1, 11);
    const auto nEdges = *rc::gen::inRange(1, 31);
    const auto nVertLevels = *rc::gen::inRange(3, 21);

    const auto nC = static_cast<std::size_t>(nCells);
    const auto nE = static_cast<std::size_t>(nEdges);
    const auto nLev = static_cast<std::size_t>(nVertLevels);

    // Generate timing parameters
    const real_type dt_rk = 1.0 + (*rc::gen::inRange(0, 10001)) / 10000.0;
    const index_type n_acoustic = *rc::gen::inRange(1, 11);

    // --- Generate input arrays ---

    // u_save: random finite values
    std::vector<real_type> u_save_data(nLev * nE);
    for (auto& v : u_save_data) v = (*rc::gen::inRange(-10000, 10001)) / 100.0;

    // tend_u: random finite values
    std::vector<real_type> tend_u_data(nLev * nE);
    for (auto& v : tend_u_data) v = (*rc::gen::inRange(-5000, 5001)) / 1000.0;

    // ruAvg: random finite values
    std::vector<real_type> ruAvg_data(nLev * nE);
    for (auto& v : ruAvg_data) v = (*rc::gen::inRange(-10000, 10001)) / 100.0;

    // rho_edge: strictly positive
    std::vector<real_type> rho_edge_data(nLev * nE);
    for (auto& v : rho_edge_data) v = 0.5 + (*rc::gen::inRange(0, 10001)) / 10000.0;

    // rho_zz_save: strictly positive
    std::vector<real_type> rho_zz_save_data(nLev * nC);
    for (auto& v : rho_zz_save_data) v = 0.5 + (*rc::gen::inRange(0, 10001)) / 10000.0;

    // tend_rho: random finite values
    std::vector<real_type> tend_rho_data(nLev * nC);
    for (auto& v : tend_rho_data) v = (*rc::gen::inRange(-1000, 1001)) / 10000.0;

    // rho_pp: values that keep rho_zz positive after recovery
    // rho_zz = rho_zz_save + dt_rk * tend_rho + rho_pp must be > 0
    // We'll use small rho_pp to keep things safe
    std::vector<real_type> rho_pp_data(nLev * nC);
    for (auto& v : rho_pp_data) v = (*rc::gen::inRange(-1000, 1001)) / 100000.0;

    // theta_m_save: random positive
    std::vector<real_type> theta_m_save_data(nLev * nC);
    for (auto& v : theta_m_save_data) v = 200.0 + (*rc::gen::inRange(0, 10001)) / 100.0;

    // tend_theta: random
    std::vector<real_type> tend_theta_data(nLev * nC);
    for (auto& v : tend_theta_data) v = (*rc::gen::inRange(-5000, 5001)) / 100.0;

    // rtheta_pp: random
    std::vector<real_type> rtheta_pp_data(nLev * nC);
    for (auto& v : rtheta_pp_data) v = (*rc::gen::inRange(-5000, 5001)) / 100.0;

    // w_save: (nVertLevels+1) x nCells
    std::vector<real_type> w_save_data((nLev + 1) * nC);
    for (auto& v : w_save_data) v = (*rc::gen::inRange(-5000, 5001)) / 1000.0;

    // tend_w: (nVertLevels+1) x nCells
    std::vector<real_type> tend_w_data((nLev + 1) * nC);
    for (auto& v : tend_w_data) v = (*rc::gen::inRange(-1000, 1001)) / 1000.0;

    // wwAvg: (nVertLevels+1) x nCells
    std::vector<real_type> wwAvg_data((nLev + 1) * nC);
    for (auto& v : wwAvg_data) v = (*rc::gen::inRange(-5000, 5001)) / 100.0;

    // zz: vertical metric (nVertLevels x nCells), positive
    std::vector<real_type> zz_data(nLev * nC);
    for (auto& v : zz_data) v = 0.8 + (*rc::gen::inRange(0, 4001)) / 10000.0;

    // fzm, fzp: vertical interpolation weights such that fzm + fzp approx 1
    std::vector<real_type> fzm_data(nLev);
    std::vector<real_type> fzp_data(nLev);
    for (std::size_t i = 0; i < nLev; ++i) {
        fzm_data[i] = 0.4 + (*rc::gen::inRange(0, 2001)) / 10000.0;
        fzp_data[i] = 1.0 - fzm_data[i];
    }

    // --- Create mdspan views for inputs ---
    ConstField2D<default_layout, unchecked_accessor> u_save(u_save_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> tend_u(tend_u_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> ruAvg(ruAvg_data.data(), nVertLevels, nEdges);
    ConstField2D<default_layout, unchecked_accessor> rho_edge(rho_edge_data.data(), nVertLevels, nEdges);

    ConstField2D<default_layout, unchecked_accessor> rho_zz_save(rho_zz_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> tend_rho(tend_rho_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rho_pp(rho_pp_data.data(), nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> theta_m_save(theta_m_save_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> tend_theta(tend_theta_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> rtheta_pp(rtheta_pp_data.data(), nVertLevels, nCells);

    ConstField2D<default_layout, unchecked_accessor> w_save(w_save_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> tend_w(tend_w_data.data(), nVertLevels + 1, nCells);
    ConstField2D<default_layout, unchecked_accessor> wwAvg(wwAvg_data.data(), nVertLevels + 1, nCells);

    ConstField2D<default_layout, unchecked_accessor> zz(zz_data.data(), nVertLevels, nCells);

    ConstSpan1D fzm(fzm_data.data(), nVertLevels);
    ConstSpan1D fzp(fzp_data.data(), nVertLevels);

    // --- Allocate output arrays ---
    std::vector<real_type> u_data(nLev * nE, 0.0);
    std::vector<real_type> rho_zz_data(nLev * nC, 0.0);
    std::vector<real_type> theta_m_data(nLev * nC, 0.0);
    std::vector<real_type> w_data((nLev + 1) * nC, 0.0);

    Field2D<default_layout, unchecked_accessor> u_out(u_data.data(), nVertLevels, nEdges);
    Field2D<default_layout, unchecked_accessor> rho_zz_out(rho_zz_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> theta_m_out(theta_m_data.data(), nVertLevels, nCells);
    Field2D<default_layout, unchecked_accessor> w_out(w_data.data(), nVertLevels + 1, nCells);

    // --- Call the kernel ---
    recover_state<default_layout>(
        SerialPolicy{},
        u_out, rho_zz_out, theta_m_out, w_out,
        u_save, rho_zz_save, theta_m_save, w_save,
        ruAvg, wwAvg, rho_pp, rtheta_pp,
        tend_u, tend_rho, tend_theta, tend_w,
        rho_edge, zz, fzm, fzp,
        dt_rk, n_acoustic,
        static_cast<index_type>(nCells),
        static_cast<index_type>(nEdges),
        static_cast<index_type>(nVertLevels));

    const real_type inv_n_acoustic = 1.0 / static_cast<real_type>(n_acoustic);

    // --- Verify u recovery ---
    // u[k, iEdge] = u_save[k, iEdge] + dt_rk * tend_u[k, iEdge]
    //             + ruAvg[k, iEdge] / (n_acoustic * rho_edge[k, iEdge])
    for (index_type iEdge = 0; iEdge < static_cast<index_type>(nEdges); ++iEdge) {
        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            auto u_s = u_save[k, iEdge];
            auto t_u = tend_u[k, iEdge];
            auto ru  = ruAvg[k, iEdge];
            auto rho_e = rho_edge[k, iEdge];
            const real_type expected = u_s + dt_rk * t_u + ru * inv_n_acoustic / rho_e;
            const real_type actual = u_out[k, iEdge];
            RC_ASSERT(rel_eq(actual, expected));
        }
    }

    // --- Verify rho_zz recovery ---
    // rho_zz[k, iCell] = rho_zz_save[k, iCell] + dt_rk * tend_rho[k, iCell] + rho_pp[k, iCell]
    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            auto rho_s = rho_zz_save[k, iCell];
            auto t_rho = tend_rho[k, iCell];
            auto rpp   = rho_pp[k, iCell];
            const real_type expected = rho_s + dt_rk * t_rho + rpp;
            const real_type actual = rho_zz_out[k, iCell];
            RC_ASSERT(rel_eq(actual, expected));
        }
    }

    // --- Verify theta_m recovery ---
    // theta_m[k, iCell] = (theta_m_save[k, iCell] * rho_zz_save[k, iCell]
    //                     + dt_rk * tend_theta[k, iCell]
    //                     + rtheta_pp[k, iCell]) / rho_zz[k, iCell]
    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            const real_type rho_new = rho_zz_out[k, iCell];
            // Skip if rho_zz is zero (would be division by zero)
            if (std::abs(rho_new) < 1.0e-300) continue;
            auto th_s = theta_m_save[k, iCell];
            auto rho_s = rho_zz_save[k, iCell];
            auto t_th = tend_theta[k, iCell];
            auto rth_pp = rtheta_pp[k, iCell];
            const real_type expected = (th_s * rho_s + dt_rk * t_th + rth_pp) / rho_new;
            const real_type actual = theta_m_out[k, iCell];
            RC_ASSERT(rel_eq(actual, expected));
        }
    }

    // --- Verify w boundary conditions ---
    // w[0, iCell] == 0.0 and w[nVertLevels, iCell] == 0.0
    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        const real_type w_bot = w_out[0, iCell];
        const real_type w_top = w_out[static_cast<index_type>(nVertLevels), iCell];
        RC_ASSERT(w_bot == 0.0);
        RC_ASSERT(w_top == 0.0);
    }

    // --- Verify w interior recovery ---
    // w[k, iCell] = (w_save[k, iCell] * rho_zz_at_w_save
    //              + dt_rk * tend_w[k, iCell]
    //              + wwAvg[k, iCell] / n_acoustic) / rho_zz_at_w
    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        for (index_type k = 1; k < static_cast<index_type>(nVertLevels); ++k) {
            auto rho_k   = rho_zz_out[k, iCell];
            auto rho_km1 = rho_zz_out[k - 1, iCell];
            auto rho_s_k   = rho_zz_save[k, iCell];
            auto rho_s_km1 = rho_zz_save[k - 1, iCell];
            const real_type rho_zz_at_w =
                fzm[k] * rho_k + fzp[k] * rho_km1;
            const real_type rho_zz_at_w_save =
                fzm[k] * rho_s_k + fzp[k] * rho_s_km1;

            // Skip if denominator is near zero
            if (std::abs(rho_zz_at_w) < 1.0e-300) continue;

            auto w_s = w_save[k, iCell];
            auto t_w = tend_w[k, iCell];
            auto ww  = wwAvg[k, iCell];
            const real_type expected =
                (w_s * rho_zz_at_w_save + dt_rk * t_w + ww * inv_n_acoustic) / rho_zz_at_w;
            const real_type actual = w_out[k, iCell];
            RC_ASSERT(rel_eq(actual, expected));
        }
    }
}
