/// @file recover_state.cpp
/// @brief Implementation of the state recovery kernel for the MPAS dynamical core.
///
/// After acoustic sub-stepping completes within a Runge-Kutta stage, the
/// prognostic fields (u, rho_zz, theta_m, w) are recovered by combining:
///   - The saved state from the start of the RK stage
///   - The slow-physics tendencies scaled by the RK stage timestep
///   - The accumulated perturbations from acoustic sub-stepping
///
/// @section u_recovery Normal Velocity Recovery
/// u[k, iEdge] = u_save[k, iEdge]
///             + dt_rk * tend_u[k, iEdge]
///             + ruAvg[k, iEdge] / (n_acoustic * rho_edge[k, iEdge])
///
/// @section rho_recovery Dry Density Recovery
/// rho_zz[k, iCell] = rho_zz_save[k, iCell]
///                   + dt_rk * tend_rho[k, iCell]
///                   + rho_pp[k, iCell]
///
/// @section theta_recovery Moist Potential Temperature Recovery
/// theta_m[k, iCell] = (theta_m_save[k, iCell] * rho_zz_save[k, iCell]
///                     + dt_rk * tend_theta[k, iCell]
///                     + rtheta_pp[k, iCell]) / rho_zz[k, iCell]
///
/// @section w_recovery Vertical Velocity Recovery
/// w[k, iCell] = (w_save[k, iCell] * rho_zz_at_w_save[k]
///              + dt_rk * tend_w[k, iCell]
///              + wwAvg[k, iCell] / n_acoustic) / rho_zz_at_w[k]
/// with rigid-lid boundary conditions: w[0, iCell] = 0, w[nVertLevels, iCell] = 0.
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.

#include <mpas_dycore/kernels/recover_state.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mpas::dycore::kernels {

// Explicit instantiation for default layout and serial policy.
template void recover_state<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> u,
    Field2D<default_layout, unchecked_accessor> rho_zz,
    Field2D<default_layout, unchecked_accessor> theta_m,
    Field2D<default_layout, unchecked_accessor> w,
    ConstField2D<default_layout, unchecked_accessor> u_save,
    ConstField2D<default_layout, unchecked_accessor> rho_zz_save,
    ConstField2D<default_layout, unchecked_accessor> theta_m_save,
    ConstField2D<default_layout, unchecked_accessor> w_save,
    ConstField2D<default_layout, unchecked_accessor> ruAvg,
    ConstField2D<default_layout, unchecked_accessor> wwAvg,
    ConstField2D<default_layout, unchecked_accessor> rho_pp,
    ConstField2D<default_layout, unchecked_accessor> rtheta_pp,
    ConstField2D<default_layout, unchecked_accessor> tend_u,
    ConstField2D<default_layout, unchecked_accessor> tend_rho,
    ConstField2D<default_layout, unchecked_accessor> tend_theta,
    ConstField2D<default_layout, unchecked_accessor> tend_w,
    ConstField2D<default_layout, unchecked_accessor> rho_edge,
    ConstField2D<default_layout, unchecked_accessor> zz,
    ConstSpan1D fzm,
    ConstSpan1D fzp,
    real_type dt_rk,
    index_type n_acoustic,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels);

/// @brief Implementation of recover_state.
///
/// Recovers all prognostic fields after acoustic sub-stepping. The recovery
/// is performed in order: velocity, density, theta, vertical velocity.
/// Density must be recovered before theta and w because those formulas
/// depend on the newly-recovered rho_zz values.
template <typename Layout, ExecutionPolicy Policy>
void recover_state(
    Policy /*policy*/,
    // Updated prognostic fields (output)
    Field2D<Layout, unchecked_accessor> u,
    Field2D<Layout, unchecked_accessor> rho_zz,
    Field2D<Layout, unchecked_accessor> theta_m,
    Field2D<Layout, unchecked_accessor> w,
    // Saved prognostic fields from start of RK stage (input)
    ConstField2D<Layout, unchecked_accessor> u_save,
    ConstField2D<Layout, unchecked_accessor> rho_zz_save,
    ConstField2D<Layout, unchecked_accessor> theta_m_save,
    ConstField2D<Layout, unchecked_accessor> w_save,
    // Accumulated perturbations from acoustic sub-stepping (input)
    ConstField2D<Layout, unchecked_accessor> ruAvg,
    ConstField2D<Layout, unchecked_accessor> wwAvg,
    ConstField2D<Layout, unchecked_accessor> rho_pp,
    ConstField2D<Layout, unchecked_accessor> rtheta_pp,
    // Tendencies from slow dynamics (input)
    ConstField2D<Layout, unchecked_accessor> tend_u,
    ConstField2D<Layout, unchecked_accessor> tend_rho,
    ConstField2D<Layout, unchecked_accessor> tend_theta,
    ConstField2D<Layout, unchecked_accessor> tend_w,
    // Edge density for velocity recovery
    ConstField2D<Layout, unchecked_accessor> rho_edge,
    // Vertical metric
    ConstField2D<Layout, unchecked_accessor> zz,
    // Vertical interpolation weights (1D spans)
    ConstSpan1D fzm,
    ConstSpan1D fzp,
    // Timing parameters
    real_type dt_rk,
    index_type n_acoustic,
    // Dimensions
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels)
{
    // ========================================================================
    // Precondition validation
    // ========================================================================
    if (n_acoustic <= 0) {
        throw std::invalid_argument(
            "recover_state: n_acoustic must be > 0, got " +
            std::to_string(n_acoustic));
    }
    if (dt_rk <= 0.0) {
        throw std::invalid_argument(
            "recover_state: dt_rk must be > 0, got " +
            std::to_string(dt_rk));
    }

    const real_type inv_n_acoustic = 1.0 / static_cast<real_type>(n_acoustic);

    // ========================================================================
    // Step 1: Velocity recovery (u on edges)
    //
    // u[k, iEdge] = u_save[k, iEdge]
    //             + dt_rk * tend_u[k, iEdge]
    //             + ruAvg[k, iEdge] / (n_acoustic * rho_edge[k, iEdge])
    // ========================================================================
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            u[k, iEdge] = u_save[k, iEdge]
                        + dt_rk * tend_u[k, iEdge]
                        + ruAvg[k, iEdge] * inv_n_acoustic / rho_edge[k, iEdge];
        }
    }

    // ========================================================================
    // Step 2: Density recovery (rho_zz at cells)
    //
    // rho_zz[k, iCell] = rho_zz_save[k, iCell]
    //                   + dt_rk * tend_rho[k, iCell]
    //                   + rho_pp[k, iCell]
    // ========================================================================
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            rho_zz[k, iCell] = rho_zz_save[k, iCell]
                             + dt_rk * tend_rho[k, iCell]
                             + rho_pp[k, iCell];
        }
    }

    // ========================================================================
    // Step 3: Theta recovery (theta_m at cells, depends on recovered rho_zz)
    //
    // theta_m[k, iCell] = (theta_m_save[k, iCell] * rho_zz_save[k, iCell]
    //                     + dt_rk * tend_theta[k, iCell]
    //                     + rtheta_pp[k, iCell]) / rho_zz[k, iCell]
    // ========================================================================
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            theta_m[k, iCell] = (theta_m_save[k, iCell] * rho_zz_save[k, iCell]
                               + dt_rk * tend_theta[k, iCell]
                               + rtheta_pp[k, iCell]) / rho_zz[k, iCell];
        }
    }

    // ========================================================================
    // Step 4: Vertical velocity recovery (w at cells, w-levels)
    //
    // w has nVertLevels+1 levels. Boundaries use rigid-lid condition (w=0).
    // Interior levels use fzm/fzp interpolation of rho_zz to w-levels.
    //
    // For interior k in [1, nVertLevels):
    //   rho_zz_at_w      = fzm[k] * rho_zz[k, iCell]      + fzp[k] * rho_zz[k-1, iCell]
    //   rho_zz_at_w_save = fzm[k] * rho_zz_save[k, iCell]  + fzp[k] * rho_zz_save[k-1, iCell]
    //   w[k, iCell] = (w_save[k, iCell] * rho_zz_at_w_save
    //                 + dt_rk * tend_w[k, iCell]
    //                 + wwAvg[k, iCell] * inv_n_acoustic) / rho_zz_at_w
    // ========================================================================
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        // Bottom boundary (k=0): rigid lid
        w[0, iCell] = 0.0;

        // Interior w-levels
        for (index_type k = 1; k < nVertLevels; ++k) {
            const real_type rho_zz_at_w =
                fzm[k] * rho_zz[k, iCell] + fzp[k] * rho_zz[k - 1, iCell];
            const real_type rho_zz_at_w_save =
                fzm[k] * rho_zz_save[k, iCell] + fzp[k] * rho_zz_save[k - 1, iCell];

            w[k, iCell] = (w_save[k, iCell] * rho_zz_at_w_save
                         + dt_rk * tend_w[k, iCell]
                         + wwAvg[k, iCell] * inv_n_acoustic) / rho_zz_at_w;
        }

        // Top boundary (k=nVertLevels): rigid lid
        w[nVertLevels, iCell] = 0.0;
    }
}

/// @brief Implementation of recover_state_perturbation.
///
/// Recovers all prognostic fields using the Fortran perturbation-variable
/// formulation (`atm_recover_large_step_variables_work`). The algorithm proceeds
/// in four phases:
///   Phase 1: Density + initial w/rw + wwAvg (cell loop)
///   Phase 2: Theta recovery with optional diabatic term (cell loop)
///   Phase 3: Edge recovery — ru, ruAvg, u (edge loop)
///   Phase 4: Terrain correction for w (cell loop over edges)
///
/// Rigid-lid boundary conditions enforced: w[0] = 0, w[nVertLevels] = 0.
template <typename Layout, ExecutionPolicy Policy>
void recover_state_perturbation(
    Policy /*policy*/,
    // Output prognostic fields
    Field2D<Layout, unchecked_accessor> u,
    Field2D<Layout, unchecked_accessor> rho_zz,
    Field2D<Layout, unchecked_accessor> theta_m,
    Field2D<Layout, unchecked_accessor> w,
    // Output diagnostics (rk_step==3 only)
    Field2D<Layout, unchecked_accessor> exner,
    Field2D<Layout, unchecked_accessor> pressure_p,
    // Perturbation state saves (from before RK loop)
    ConstField2D<Layout, unchecked_accessor> rho_p_save,
    ConstField2D<Layout, unchecked_accessor> rtheta_p_save,
    ConstField2D<Layout, unchecked_accessor> ru_save,
    ConstField2D<Layout, unchecked_accessor> rw_save,
    // Acoustic sub-step perturbations
    ConstField2D<Layout, unchecked_accessor> rho_pp,
    ConstField2D<Layout, unchecked_accessor> rtheta_pp,
    ConstField2D<Layout, unchecked_accessor> ru_p,
    ConstField2D<Layout, unchecked_accessor> rw_p,
    // Accumulated flux averages
    Field2D<Layout, unchecked_accessor> ruAvg,
    Field2D<Layout, unchecked_accessor> wwAvg,
    // Base-state profiles
    ConstField2D<Layout, unchecked_accessor> rho_base,
    ConstField2D<Layout, unchecked_accessor> rtheta_base,
    ConstField2D<Layout, unchecked_accessor> exner_base,
    // Diabatic tendency (used only when rk_step==3)
    ConstField2D<Layout, unchecked_accessor> rt_diabatic_tend,
    // Terrain correction
    ConstField2D<Layout, unchecked_accessor> zb_cell,
    ConstField2D<Layout, unchecked_accessor> zb3_cell,
    // Edge connectivity for terrain correction
    ConstField2D<Layout, unchecked_accessor> cellsOnEdge,
    ConstField2D<Layout, unchecked_accessor> edgesOnCell,
    ConstSpan1D nEdgesOnCell_arr,
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign,
    // Vertical metrics
    ConstField2D<Layout, unchecked_accessor> zz,
    ConstSpan1D fzm,
    ConstSpan1D fzp,
    // Vertical extrapolation coefficients
    real_type cf1, real_type cf2, real_type cf3,
    // Timing/step parameters
    index_type rk_step,
    real_type dt,
    index_type ns,
    // Dimensions
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges)
{
    // Physical constants matching MPAS Fortran
    constexpr real_type rgas = 287.0;
    constexpr real_type cp   = 1003.5;
    constexpr real_type rcv  = rgas / (cp - rgas);
    constexpr real_type p0   = 1.0e5;

    const real_type invNs = 1.0 / static_cast<real_type>(ns);

    // ========================================================================
    // PHASE 1: Density recovery, initial w/rw, wwAvg (cell loop)
    // ========================================================================
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        // --- Density: perturbation accumulation ---
        for (index_type k = 0; k < nVertLevels; ++k) {
            const real_type rho_p = rho_p_save[k, iCell] + rho_pp[k, iCell];
            rho_zz[k, iCell] = rho_p + rho_base[k, iCell];
        }

        // --- Vertical momentum and initial w (interior levels) ---
        for (index_type k = 1; k < nVertLevels; ++k) {
            wwAvg[k, iCell] = rw_save[k, iCell] + wwAvg[k, iCell] * invNs;
            const real_type rw = rw_save[k, iCell] + rw_p[k, iCell];
            // Phase 1 w: divide by terrain metric only (not density yet — that's in Phase 4 via terrain correction)
            // From the Fortran: w = rw / (fzm*rho_zz*zz + fzp*rho_zz_below*zz_below)
            // But Phase 4 adds terrain correction then divides by the same density factor.
            // Actually per the design pseudocode Phase 1:
            //   w[k,iCell] = rw / (fzm[k]*rho_zz[k,iCell]*zz[k,iCell] + fzp[k]*rho_zz[k-1,iCell]*zz[k-1,iCell])
            w[k, iCell] = rw / (fzm[k] * rho_zz[k, iCell] * zz[k, iCell]
                               + fzp[k] * rho_zz[k - 1, iCell] * zz[k - 1, iCell]);
        }

        // --- Rigid-lid BCs ---
        w[0, iCell] = 0.0;
        w[nVertLevels, iCell] = 0.0;
    }

    // ========================================================================
    // PHASE 2: Theta recovery (rk_step-dependent)
    // ========================================================================
    if (rk_step == 3) {
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                const real_type rtheta_p = rtheta_p_save[k, iCell]
                                         + rtheta_pp[k, iCell]
                                         - dt * rho_zz[k, iCell] * rt_diabatic_tend[k, iCell];
                theta_m[k, iCell] = (rtheta_p + rtheta_base[k, iCell]) / rho_zz[k, iCell];

                // Exner function recomputation
                const real_type rtheta_total = rtheta_p + rtheta_base[k, iCell];
                exner[k, iCell] = std::pow(
                    zz[k, iCell] * (rgas / p0) * rtheta_total, rcv);

                // Perturbation pressure
                pressure_p[k, iCell] = zz[k, iCell] * rgas
                    * (exner[k, iCell] * rtheta_p
                     + rtheta_base[k, iCell] * (exner[k, iCell] - exner_base[k, iCell]));
            }
        }
    } else {
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                const real_type rtheta_p = rtheta_p_save[k, iCell] + rtheta_pp[k, iCell];
                theta_m[k, iCell] = (rtheta_p + rtheta_base[k, iCell]) / rho_zz[k, iCell];
            }
        }
    }

    // ========================================================================
    // PHASE 3: Edge recovery (ru, ruAvg, u)
    // ========================================================================
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        const auto cell1 = static_cast<index_type>(cellsOnEdge[0, iEdge]);
        const auto cell2 = static_cast<index_type>(cellsOnEdge[1, iEdge]);

        for (index_type k = 0; k < nVertLevels; ++k) {
            ruAvg[k, iEdge] = ru_save[k, iEdge] + ruAvg[k, iEdge] * invNs;
            const real_type ru = ru_save[k, iEdge] + ru_p[k, iEdge];
            u[k, iEdge] = 2.0 * ru / (rho_zz[k, cell1] + rho_zz[k, cell2]);
        }
    }

    // ========================================================================
    // PHASE 4: Terrain correction for w (cell loop over edges)
    //
    // The zb_cell and zb3_cell arrays are linearized as:
    //   (nVertLevels+1, maxEdges*nCells) with index [k, i*nCells + iCell]
    // where i is the edge-slot index (0..nEdgesOnCell-1).
    // ========================================================================
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const auto ne = static_cast<index_type>(nEdgesOnCell_arr[iCell]);

        for (index_type i = 0; i < ne; ++i) {
            const auto iEdge = static_cast<index_type>(edgesOnCell[i, iCell]);
            const real_type sign_edge = edgesOnCell_sign[i, iCell];

            // Linearized column index for zb_cell/zb3_cell
            const index_type zb_col = i * nCells + iCell;

            // Level k=1 (first interior w-level that gets terrain correction):
            // Use cf1/cf2/cf3 extrapolation for the flux at the lowest w-level
            {
                const index_type k = 1;
                const real_type flux = cf1 * (ru_save[0, iEdge] + ru_p[0, iEdge])
                                     + cf2 * (ru_save[1, iEdge] + ru_p[1, iEdge])
                                     + cf3 * (ru_save[2, iEdge] + ru_p[2, iEdge]);
                const real_type zb_val  = zb_cell[k, zb_col];
                const real_type zb3_val = zb3_cell[k, zb_col];
                w[k, iCell] += sign_edge
                             * (zb_val + std::copysign(1.0, flux) * zb3_val)
                             * flux
                             / (fzm[k] * rho_zz[k, iCell] * zz[k, iCell]
                              + fzp[k] * rho_zz[k - 1, iCell] * zz[k - 1, iCell]);
            }

            // Interior levels k=2..nVertLevels-1: use fzm/fzp interpolation for flux
            for (index_type k = 2; k < nVertLevels; ++k) {
                const real_type ru_k   = ru_save[k, iEdge] + ru_p[k, iEdge];
                const real_type ru_km1 = ru_save[k - 1, iEdge] + ru_p[k - 1, iEdge];
                const real_type flux = fzm[k] * ru_k + fzp[k] * ru_km1;
                const real_type zb_val  = zb_cell[k, zb_col];
                const real_type zb3_val = zb3_cell[k, zb_col];
                w[k, iCell] += sign_edge
                             * (zb_val + std::copysign(1.0, flux) * zb3_val)
                             * flux
                             / (fzm[k] * rho_zz[k, iCell] * zz[k, iCell]
                              + fzp[k] * rho_zz[k - 1, iCell] * zz[k - 1, iCell]);
            }
        }
    }

    // ========================================================================
    // Enforce rigid-lid BCs (final, after terrain correction)
    // ========================================================================
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        w[0, iCell] = 0.0;
        w[nVertLevels, iCell] = 0.0;
    }
}

// Explicit instantiation for the new perturbation-variable recover_state function.
template void recover_state_perturbation<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> u,
    Field2D<default_layout, unchecked_accessor> rho_zz,
    Field2D<default_layout, unchecked_accessor> theta_m,
    Field2D<default_layout, unchecked_accessor> w,
    Field2D<default_layout, unchecked_accessor> exner,
    Field2D<default_layout, unchecked_accessor> pressure_p,
    ConstField2D<default_layout, unchecked_accessor> rho_p_save,
    ConstField2D<default_layout, unchecked_accessor> rtheta_p_save,
    ConstField2D<default_layout, unchecked_accessor> ru_save,
    ConstField2D<default_layout, unchecked_accessor> rw_save,
    ConstField2D<default_layout, unchecked_accessor> rho_pp,
    ConstField2D<default_layout, unchecked_accessor> rtheta_pp,
    ConstField2D<default_layout, unchecked_accessor> ru_p,
    ConstField2D<default_layout, unchecked_accessor> rw_p,
    Field2D<default_layout, unchecked_accessor> ruAvg,
    Field2D<default_layout, unchecked_accessor> wwAvg,
    ConstField2D<default_layout, unchecked_accessor> rho_base,
    ConstField2D<default_layout, unchecked_accessor> rtheta_base,
    ConstField2D<default_layout, unchecked_accessor> exner_base,
    ConstField2D<default_layout, unchecked_accessor> rt_diabatic_tend,
    ConstField2D<default_layout, unchecked_accessor> zb_cell,
    ConstField2D<default_layout, unchecked_accessor> zb3_cell,
    ConstField2D<default_layout, unchecked_accessor> cellsOnEdge,
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell,
    ConstSpan1D nEdgesOnCell_arr,
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign,
    ConstField2D<default_layout, unchecked_accessor> zz,
    ConstSpan1D fzm,
    ConstSpan1D fzp,
    real_type cf1, real_type cf2, real_type cf3,
    index_type rk_step,
    real_type dt,
    index_type ns,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges);

} // namespace mpas::dycore::kernels
