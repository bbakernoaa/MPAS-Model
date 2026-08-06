#pragma once

/// @file recover_state.hpp
/// @brief State recovery kernel for the MPAS dynamical core.
///
/// After acoustic sub-stepping completes within a Runge-Kutta stage, the
/// prognostic fields (u, rho_zz, theta_m, w) must be recovered by combining:
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
/// where rho_zz_at_w is interpolated to w-levels via fzm/fzp weighting.
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief 1D const mdspan (dynamic extent) for vertical metric arrays.
using ConstSpan1D = std::mdspan<const real_type,
    std::extents<index_type, std::dynamic_extent>>;

/// @brief Recover prognostic fields after acoustic sub-stepping.
///
/// Combines saved RK-stage state, slow-physics tendencies, and accumulated
/// acoustic perturbations to produce updated prognostic fields (u, rho_zz,
/// theta_m, w) at the end of a Runge-Kutta stage.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy for parallelization.
///
/// @param[in]  policy       Execution policy (accepted for API consistency).
/// @param[out] u            Updated normal velocity on edges (nVertLevels, nEdges).
/// @param[out] rho_zz       Updated dry air density at cells (nVertLevels, nCells).
/// @param[out] theta_m      Updated moist potential temperature at cells (nVertLevels, nCells).
/// @param[out] w            Updated vertical velocity at cells (nVertLevels+1, nCells).
/// @param[in]  u_save       Saved velocity from start of RK stage (nVertLevels, nEdges).
/// @param[in]  rho_zz_save  Saved density from start of RK stage (nVertLevels, nCells).
/// @param[in]  theta_m_save Saved theta_m from start of RK stage (nVertLevels, nCells).
/// @param[in]  w_save       Saved vertical velocity from start of RK stage (nVertLevels+1, nCells).
/// @param[in]  ruAvg        Accumulated velocity perturbation (nVertLevels, nEdges).
/// @param[in]  wwAvg        Accumulated vertical velocity perturbation (nVertLevels+1, nCells).
/// @param[in]  rho_pp       Density perturbation from acoustic steps (nVertLevels, nCells).
/// @param[in]  rtheta_pp    Rho*theta perturbation from acoustic steps (nVertLevels, nCells).
/// @param[in]  tend_u       Slow-physics velocity tendency (nVertLevels, nEdges).
/// @param[in]  tend_rho     Slow-physics density tendency (nVertLevels, nCells).
/// @param[in]  tend_theta   Slow-physics theta tendency (nVertLevels, nCells).
/// @param[in]  tend_w       Slow-physics vertical velocity tendency (nVertLevels+1, nCells).
/// @param[in]  rho_edge     Density interpolated to edges (nVertLevels, nEdges).
/// @param[in]  zz           Terrain height metric dz/dzeta (nVertLevels, nCells).
/// @param[in]  fzm          Vertical interpolation weight from level k (nVertLevels).
/// @param[in]  fzp          Vertical interpolation weight from level k-1 (nVertLevels).
/// @param[in]  dt_rk        RK stage timestep fraction.
/// @param[in]  n_acoustic   Number of acoustic sub-steps executed.
/// @param[in]  nCells       Number of cells.
/// @param[in]  nEdges       Number of edges.
/// @param[in]  nVertLevels  Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void recover_state(
    Policy policy,
    // Updated prognostic fields (output)
    Field2D<Layout, unchecked_accessor> u,          // (nVertLevels, nEdges)
    Field2D<Layout, unchecked_accessor> rho_zz,     // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> theta_m,    // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> w,          // (nVertLevels+1, nCells)
    // Saved prognostic fields from start of RK stage (input)
    ConstField2D<Layout, unchecked_accessor> u_save,
    ConstField2D<Layout, unchecked_accessor> rho_zz_save,
    ConstField2D<Layout, unchecked_accessor> theta_m_save,
    ConstField2D<Layout, unchecked_accessor> w_save,
    // Accumulated perturbations from acoustic sub-stepping (input)
    ConstField2D<Layout, unchecked_accessor> ruAvg,      // (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> wwAvg,      // (nVertLevels+1, nCells)
    ConstField2D<Layout, unchecked_accessor> rho_pp,     // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rtheta_pp,  // (nVertLevels, nCells)
    // Tendencies from slow dynamics (input)
    ConstField2D<Layout, unchecked_accessor> tend_u,
    ConstField2D<Layout, unchecked_accessor> tend_rho,
    ConstField2D<Layout, unchecked_accessor> tend_theta,
    ConstField2D<Layout, unchecked_accessor> tend_w,
    // Edge density for velocity recovery
    ConstField2D<Layout, unchecked_accessor> rho_edge,   // (nVertLevels, nEdges)
    // Vertical metric
    ConstField2D<Layout, unchecked_accessor> zz,         // (nVertLevels, nCells)
    // Vertical interpolation weights (1D spans)
    ConstSpan1D fzm,  // (nVertLevels)
    ConstSpan1D fzp,  // (nVertLevels)
    // Timing parameters
    real_type dt_rk,           // RK stage timestep fraction
    index_type n_acoustic,     // number of acoustic sub-steps executed
    // Dimensions
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels);

/// @brief Recover prognostic fields using the Fortran perturbation-variable formulation.
///
/// Implements the `atm_recover_large_step_variables_work` algorithm from the
/// Fortran reference using perturbation variables relative to time-invariant
/// base states, terrain corrections for vertical velocity, and conditional
/// diabatic/Exner updates on the final RK step.
///
/// The algorithm proceeds in four phases:
///
/// **Phase 1 — Density + initial w/rw (cell loop):**
/// @code
/// rho_p   = rho_p_save + rho_pp
/// rho_zz  = rho_p + rho_base
/// wwAvg   = rw_save + wwAvg * invNs
/// rw      = rw_save + rw_p
/// w       = rw / (fzm*zz + fzp*zz)  (pre-terrain-correction)
/// @endcode
///
/// **Phase 2 — Theta recovery (rk_step-dependent):**
/// @code
/// rtheta_p = rtheta_p_save + rtheta_pp [- dt*rho_zz*rt_diabatic_tend if rk_step==3]
/// theta_m  = (rtheta_p + rtheta_base) / rho_zz
/// // On rk_step==3: recompute exner and pressure_p
/// @endcode
///
/// **Phase 3 — Edge recovery (ru, ruAvg, u):**
/// @code
/// ruAvg = ru_save + ruAvg * invNs
/// ru    = ru_save + ru_p
/// u     = 2 * ru / (rho_zz[cell1] + rho_zz[cell2])
/// @endcode
///
/// **Phase 4 — Terrain correction for w:**
/// @code
/// // Loop over cell edges accumulating zb_cell/zb3_cell flux contributions
/// // Use cf1/cf2/cf3 extrapolation at k=0, fzm/fzp interpolation at k>0
/// // Divide by interpolated density for final w
/// @endcode
///
/// Rigid-lid boundary conditions enforced: w[0] = 0, w[nVertLevels] = 0.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy for parallelization.
///
/// @param[in]  policy            Execution policy (accepted for API consistency).
/// @param[out] u                 Recovered normal velocity on edges (nVertLevels, nEdges).
/// @param[out] rho_zz            Recovered dry air density at cells (nVertLevels, nCells).
/// @param[out] theta_m           Recovered moist potential temperature at cells (nVertLevels, nCells).
/// @param[out] w                 Recovered vertical velocity at cells (nVertLevels+1, nCells).
/// @param[out] exner             Exner function (nVertLevels, nCells) — updated on rk_step==3 only.
/// @param[out] pressure_p        Perturbation pressure (nVertLevels, nCells) — updated on rk_step==3 only.
/// @param[in]  rho_p_save        Saved perturbation density from before RK loop (nVertLevels, nCells).
/// @param[in]  rtheta_p_save     Saved perturbation rho*theta from before RK loop (nVertLevels, nCells).
/// @param[in]  ru_save           Saved perturbation ru from before RK loop (nVertLevels, nEdges).
/// @param[in]  rw_save           Saved perturbation rw from before RK loop (nVertLevels+1, nCells).
/// @param[in]  rho_pp            Density perturbation from acoustic steps (nVertLevels, nCells).
/// @param[in]  rtheta_pp         Rho*theta perturbation from acoustic steps (nVertLevels, nCells).
/// @param[in]  ru_p              Acoustic velocity perturbation (nVertLevels, nEdges).
/// @param[in]  rw_p              Acoustic w perturbation (nVertLevels+1, nCells).
/// @param[in,out] ruAvg          Accumulated velocity flux average (nVertLevels, nEdges).
/// @param[in,out] wwAvg          Accumulated vertical velocity flux average (nVertLevels+1, nCells).
/// @param[in]  rho_base          Base-state density profile (nVertLevels, nCells).
/// @param[in]  rtheta_base       Base-state rho*theta profile (nVertLevels, nCells).
/// @param[in]  exner_base        Base-state Exner function (nVertLevels, nCells).
/// @param[in]  rt_diabatic_tend  Diabatic heating tendency (nVertLevels, nCells) — used on rk_step==3.
/// @param[in]  zb_cell           Terrain slope metric (nVertLevels+1, maxEdges*nCells linearized).
/// @param[in]  zb3_cell          3rd-order terrain correction (nVertLevels+1, maxEdges*nCells linearized).
/// @param[in]  cellsOnEdge       Edge-to-cell connectivity (2, nEdges).
/// @param[in]  edgesOnCell       Cell-to-edge connectivity (maxEdges, nCells).
/// @param[in]  nEdgesOnCell_arr  Number of edges per cell (nCells).
/// @param[in]  edgesOnCell_sign  Edge orientation signs per cell (maxEdges, nCells).
/// @param[in]  zz                Terrain height metric dz/dzeta (nVertLevels, nCells).
/// @param[in]  fzm               Vertical interpolation weight from level k (nVertLevels).
/// @param[in]  fzp               Vertical interpolation weight from level k-1 (nVertLevels).
/// @param[in]  cf1               Vertical extrapolation coefficient for level 1.
/// @param[in]  cf2               Vertical extrapolation coefficient for level 2.
/// @param[in]  cf3               Vertical extrapolation coefficient for level 3.
/// @param[in]  rk_step           Current RK step (1, 2, or 3 — 1-based, matching Fortran).
/// @param[in]  dt                Full timestep (for diabatic term).
/// @param[in]  ns                Number of acoustic sub-steps executed.
/// @param[in]  nCells            Number of cells.
/// @param[in]  nEdges            Number of edges.
/// @param[in]  nVertLevels       Number of vertical levels.
/// @param[in]  maxEdges          Maximum number of edges per cell.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void recover_state_perturbation(
    Policy policy,
    // Output prognostic fields
    Field2D<Layout, unchecked_accessor> u,          // (nVertLevels, nEdges)
    Field2D<Layout, unchecked_accessor> rho_zz,     // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> theta_m,    // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> w,          // (nVertLevels+1, nCells)
    // Output diagnostics (rk_step==3 only)
    Field2D<Layout, unchecked_accessor> exner,      // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> pressure_p, // (nVertLevels, nCells)
    // Perturbation state saves (from before RK loop)
    ConstField2D<Layout, unchecked_accessor> rho_p_save,    // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rtheta_p_save, // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> ru_save,       // (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> rw_save,       // (nVertLevels+1, nCells)
    // Acoustic sub-step perturbations
    ConstField2D<Layout, unchecked_accessor> rho_pp,     // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rtheta_pp,  // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> ru_p,       // (nVertLevels, nEdges) acoustic velocity perturbation
    ConstField2D<Layout, unchecked_accessor> rw_p,       // (nVertLevels+1, nCells) acoustic w perturbation
    // Accumulated flux averages
    Field2D<Layout, unchecked_accessor> ruAvg,      // (nVertLevels, nEdges) IN/OUT
    Field2D<Layout, unchecked_accessor> wwAvg,      // (nVertLevels+1, nCells) IN/OUT
    // Base-state profiles
    ConstField2D<Layout, unchecked_accessor> rho_base,    // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rtheta_base, // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> exner_base,  // (nVertLevels, nCells)
    // Diabatic tendency (used only when rk_step==3)
    ConstField2D<Layout, unchecked_accessor> rt_diabatic_tend, // (nVertLevels, nCells)
    // Terrain correction
    ConstField2D<Layout, unchecked_accessor> zb_cell,       // (nVertLevels+1, maxEdges*nCells) linearized
    ConstField2D<Layout, unchecked_accessor> zb3_cell,      // (nVertLevels+1, maxEdges*nCells) linearized
    // Edge connectivity for terrain correction
    ConstField2D<Layout, unchecked_accessor> cellsOnEdge,   // (2, nEdges)
    ConstField2D<Layout, unchecked_accessor> edgesOnCell,   // (maxEdges, nCells)
    ConstSpan1D nEdgesOnCell_arr,                           // (nCells)
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign, // (maxEdges, nCells)
    // Vertical metrics
    ConstField2D<Layout, unchecked_accessor> zz,  // (nVertLevels, nCells)
    ConstSpan1D fzm,  // (nVertLevels)
    ConstSpan1D fzp,  // (nVertLevels)
    // Vertical extrapolation coefficients
    real_type cf1, real_type cf2, real_type cf3,
    // Timing/step parameters
    index_type rk_step,       // 1, 2, or 3
    real_type dt,             // full timestep (for diabatic term)
    index_type ns,            // number of acoustic sub-steps
    // Dimensions
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges);

} // namespace mpas::dycore::kernels
