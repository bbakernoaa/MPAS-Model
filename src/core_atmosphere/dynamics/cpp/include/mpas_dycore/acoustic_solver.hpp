#ifndef MPAS_DYCORE_ACOUSTIC_SOLVER_HPP
#define MPAS_DYCORE_ACOUSTIC_SOLVER_HPP

/// @file acoustic_solver.hpp
/// @brief Acoustic substep update for the C++ dycore.
///
/// Implements the forward-backward vertically implicit acoustic substep update
/// equivalent to the Reference_Model `atm_advance_acoustic_step_work` and
/// `atm_divergence_damping_3d` subroutines.
///
/// The acoustic substep:
/// - Updates perturbation horizontal momentum (ru_p) from pressure gradient
/// - Accumulates time-averaged horizontal mass flux (ruAvg)
/// - Computes cell-level divergence and theta flux from updated ru_p
/// - Updates perturbation density (rho_pp) and coupled potential temperature
///   (rtheta_pp) via the vertically implicit system
/// - Accumulates time-averaged vertical mass flux (wwAvg)
/// - Applies the tridiagonal solve for rw_p (vertical momentum)
/// - Applies implicit Rayleigh damping in the absorbing layer
/// - Applies the specified-zone update path for regional mode
/// - Applies 3-D divergence damping to horizontal momentum
///
/// Requirements: 4.1, 4.2, 4.4, 4.6, 4.7

#include "mpas_dycore/scalar.hpp"
#include "mpas_dycore/tridiagonal_solve.hpp"

#include <Kokkos_Core.hpp>

namespace mpas {
namespace dycore {

/// Physical constants matching the Reference_Model (mpas_constants.F).
namespace acoustic_constants {
inline constexpr Scalar rgas = 287.0;
inline constexpr Scalar cp = 7.0 * rgas / 2.0;
inline constexpr Scalar gravity = 9.80616;
}  // namespace acoustic_constants

/// Parameters for the acoustic substep update kernel.
struct AcousticStepParams {
  int nVertLevels = 0;   ///< Number of vertical full levels
  int nCells = 0;        ///< Number of cells (total in partition)
  int nEdges = 0;        ///< Number of edges (total in partition)
  int nCellsSolve = 0;   ///< Number of owned cells to solve
  int maxEdges = 0;      ///< Maximum edges per cell
  Scalar dts = 0.0;      ///< Acoustic substep timestep [s]
  int small_step = 0;    ///< Current acoustic substep index (1-based)
  Scalar cf1 = 0.0;      ///< Vertical interpolation coefficient
  Scalar cf2 = 0.0;      ///< Vertical interpolation coefficient
  Scalar cf3 = 0.0;      ///< Vertical interpolation coefficient
};

/// Parameters for the 3-D divergence damping kernel.
struct DivergenceDampingParams {
  int nVertLevels = 0;   ///< Number of vertical full levels
  int nEdges = 0;        ///< Number of edges
  int nCellsSolve = 0;   ///< Number of owned cells (for edge filtering)
  Scalar dts = 0.0;      ///< Acoustic substep timestep [s]
  Scalar smdiv = 0.0;    ///< Divergence damping coefficient (config_smdiv)
  Scalar config_len_disp = 0.0;  ///< Horizontal filter length scale [m]
};

/// @brief Update perturbation horizontal momentum and accumulate ruAvg.
///
/// Ports the edge loop from `atm_advance_acoustic_step_work`.
/// - On substep 1: ru_p = dts * tend_ru; ruAvg = ru_p
/// - On substeps > 1: ru_p += dts * (tend_ru - pgrad); ruAvg += ru_p
///
/// The pressure gradient includes:
///   pgrad = cqu * 0.5 * c2 * (exner(cell1)+exner(cell2)) *
///           (rtheta_pp(cell2)-rtheta_pp(cell1)) / (dcEdge * 0.5*(zz(cell2)+zz(cell1)))
///         + 0.5 * zxu * gravity * (rho_pp(cell1)+rho_pp(cell2))
///
/// The specZoneMaskEdge factor zeros the pressure gradient inside the
/// specified zone (Requirement 4.6).
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void acoustic_step_update_edges(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ru_p,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ruAvg,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rtheta_pp,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& exner,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cqu,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_pp,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zxu,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_ru,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invDcEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& specZoneMaskEdge,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
    const AcousticStepParams& params) {

  namespace ac = acoustic_constants;
  const int nVertLevels = params.nVertLevels;
  const int nEdges = params.nEdges;
  const int nCells = params.nCells;
  const int nCellsSolve = params.nCellsSolve;
  const Scalar dts = params.dts;
  const int small_step = params.small_step;

  const Scalar rcv = ac::rgas / (ac::cp - ac::rgas);
  const Scalar c2 = ac::cp * rcv;

  if (small_step != 1) {
    // Substeps > 1: update ru_p with pressure gradient and accumulate
    Kokkos::parallel_for(
        "acoustic_step_update_edges_gt1",
        Kokkos::RangePolicy<ExecSpace>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          // cellsOnEdge is 1-based Fortran index
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;

          // Only update edges touching at least one owned cell, and only when
          // both neighbour cells are in range. The second guard skips edges
          // whose far neighbour is the boundary "sentinel" cell (index nCells,
          // outside the wrapped view), matching the guarded reads in
          // compute_dyn_tend / recover.
          if ((cell1 < nCellsSolve || cell2 < nCellsSolve) &&
              cell1 >= 0 && cell1 < nCells && cell2 >= 0 && cell2 < nCells) {
            const Scalar mask = Scalar(1.0) - specZoneMaskEdge(iEdge);

            for (int k = 0; k < nVertLevels; ++k) {
              // Pressure gradient on the edge
              Scalar pgrad = (rtheta_pp(k, cell2) - rtheta_pp(k, cell1))
                             * invDcEdge(iEdge)
                             / (Scalar(0.5) * (zz(k, cell2) + zz(k, cell1)));
              pgrad = cqu(k, iEdge) * Scalar(0.5) * c2
                      * (exner(k, cell1) + exner(k, cell2)) * pgrad;
              pgrad = pgrad + Scalar(0.5) * zxu(k, iEdge) * ac::gravity
                      * (rho_pp(k, cell1) + rho_pp(k, cell2));

              ru_p(k, iEdge) = ru_p(k, iEdge)
                               + dts * (tend_ru(k, iEdge) - mask * pgrad);

              ruAvg(k, iEdge) = ruAvg(k, iEdge) + ru_p(k, iEdge);
            }
          }
        });
  } else {
    // Substep 1: simple initialization from tendency
    Kokkos::parallel_for(
        "acoustic_step_update_edges_eq1",
        Kokkos::RangePolicy<ExecSpace>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;

          if (cell1 < nCellsSolve || cell2 < nCellsSolve) {
            for (int k = 0; k < nVertLevels; ++k) {
              ru_p(k, iEdge) = dts * tend_ru(k, iEdge);
              ruAvg(k, iEdge) = ru_p(k, iEdge);
            }
          }
        });
  }
}

/// @brief Update cell-level fields: rho_pp, rtheta_pp, rw_p, wwAvg.
///
/// Ports the cell loop from `atm_advance_acoustic_step_work`.
/// This kernel performs:
/// 1. Zero accumulated fields when substep == 1 (Req 4.4)
/// 2. Save rtheta_pp_old for divergence damping
/// 3. For interior cells (not in specified zone):
///    - Compute horizontal flux divergence (rs, ts) from updated ru_p
///    - Build the RHS for the tridiagonal system
///    - Accumulate pre-solve wwAvg contribution
///    - Compute the explicit rw_p update
///    - Run the tridiagonal solve and Rayleigh damping (via tridiagonal_vertical_solve)
///    - Accumulate post-solve wwAvg contribution
///    - Update rho_pp and rtheta_pp from solved rw_p
/// 4. For specified-zone cells:
///    - Simple tendency update (Req 4.6)
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void acoustic_step_update_cells(
    // In/out perturbation fields
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw_p,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_pp,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rtheta_pp,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& wwAvg,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rtheta_pp_old,
    // Input state fields
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ru_p,
    // Tendencies
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_rho,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_rt,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_rw,
    // Implicit coefficients
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cofwt,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& coftz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cofwr,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cofwz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& a_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& alpha_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& gamma_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dss,
    // Full-state fields for Rayleigh damping
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw_save,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw,
    // 1-D vertical coefficients
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzm,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzp,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzw,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cofrz,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& etp,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& etm,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ewp,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ewm,
    // Mesh connectivity/geometry
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dvEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invAreaCell,
    const Kokkos::View<const int*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& nEdgesOnCell,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell_sign,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& specZoneMaskCell,
    const AcousticStepParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const int nCellsSolve = params.nCellsSolve;
  const Scalar dts = params.dts;
  const Scalar dts2 = dts * dts;
  const int small_step = params.small_step;

  // Step 1: Save rtheta_pp_old for divergence damping (Req 4.7)
  // On substep 1, rtheta_pp_old = 0; otherwise, rtheta_pp_old = rtheta_pp
  if (small_step == 1) {
    Kokkos::parallel_for(
        "acoustic_step_zero_rtheta_pp_old",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          for (int k = 0; k < nVertLevels; ++k) {
            rtheta_pp_old(k, iCell) = Scalar(0.0);
          }
        });
  } else {
    Kokkos::parallel_for(
        "acoustic_step_save_rtheta_pp_old",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          for (int k = 0; k < nVertLevels; ++k) {
            rtheta_pp_old(k, iCell) = rtheta_pp(k, iCell);
          }
        });
  }
  Kokkos::fence("acoustic_step_rtheta_pp_old_fence");

  // Step 2: Main cell update loop using Kokkos TeamPolicy for scratch memory.
  // Each cell needs a per-column scratch array for ts[] and rs[].
  // We use TeamPolicy with thread-team scratch to store them.
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using team_member = typename team_policy::member_type;

  const int scratch_size = 2 * nVertLevels * sizeof(Scalar);
  team_policy policy(nCellsSolve, Kokkos::AUTO);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

  Kokkos::parallel_for(
      "acoustic_step_update_cells",
      policy,
      KOKKOS_LAMBDA(const team_member& team) {
        const int iCell = team.league_rank();

        // Thread-team scratch for ts and rs arrays
        Scalar* scratch = reinterpret_cast<Scalar*>(
            team.team_shmem().get_shmem(scratch_size));
        Scalar* ts = scratch;
        Scalar* rs = scratch + nVertLevels;

        // Substep 1: zero accumulated fields (Req 4.4)
        if (small_step == 1) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            for (int k = 0; k < nVertLevels; ++k) {
              wwAvg(k, iCell) = Scalar(0.0);
              rho_pp(k, iCell) = Scalar(0.0);
              rtheta_pp(k, iCell) = Scalar(0.0);
              rw_p(k, iCell) = Scalar(0.0);
            }
            wwAvg(nVertLevels, iCell) = Scalar(0.0);
            rw_p(nVertLevels, iCell) = Scalar(0.0);
          });
          team.team_barrier();
        }

        // Check if this cell is in the specified zone (Req 4.6)
        const bool is_spec_zone = (specZoneMaskCell(iCell) != Scalar(0.0));

        if (!is_spec_zone) {
          // ─── Interior cell solve ───────────────────────────────────────

          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            // Initialize ts and rs to zero
            for (int k = 0; k < nVertLevels; ++k) {
              ts[k] = Scalar(0.0);
              rs[k] = Scalar(0.0);
            }

            // Compute horizontal flux divergence from ru_p
            const int nEdges_i = nEdgesOnCell(iCell);
            for (int i = 0; i < nEdges_i; ++i) {
              const int iEdge = edgesOnCell(i, iCell) - 1;  // 0-based
              const int cell1 = cellsOnEdge(0, iEdge) - 1;
              const int cell2 = cellsOnEdge(1, iEdge) - 1;

              for (int k = 0; k < nVertLevels; ++k) {
                const Scalar flux = edgesOnCell_sign(i, iCell) * dts
                                    * dvEdge(iEdge) * ru_p(k, iEdge)
                                    * invAreaCell(iCell);
                rs[k] -= flux;
                ts[k] -= flux * Scalar(0.5)
                         * (theta_m(k, cell2) + theta_m(k, cell1));
              }
            }

            // Vertically implicit integration: build intermediate rs and ts
            // Fortran: rs(k) = rho_pp(k) + dts*tend_rho(k) + rs(k)
            //                 - dts*cofrz(k)*(ewm(k+1)*rw_p(k+1) - ewm(k)*rw_p(k))
            // Fortran: ts(k) = rtheta_pp(k) + dts*tend_rt(k) + ts(k)
            //                 - dts*rdzw(k)*(ewm(k+1)*coftz(k+1)*rw_p(k+1)
            //                               - ewm(k)*coftz(k)*rw_p(k))
            for (int k = 0; k < nVertLevels; ++k) {
              rs[k] = rho_pp(k, iCell) + dts * tend_rho(k, iCell) + rs[k]
                      - dts * cofrz(k) * (ewm(k + 1) * rw_p(k + 1, iCell)
                                          - ewm(k) * rw_p(k, iCell));
              ts[k] = rtheta_pp(k, iCell) + dts * tend_rt(k, iCell) + ts[k]
                      - dts * rdzw(k) * (ewm(k + 1) * coftz(k + 1, iCell)
                                                     * rw_p(k + 1, iCell)
                                         - ewm(k) * coftz(k, iCell)
                                                   * rw_p(k, iCell));
            }

            // Accumulate pre-solve wwAvg (ewm part)
            // Fortran: wwAvg(k) += ewm(k)*rw_p(k) for k=2..nVertLevels
            for (int k = 1; k < nVertLevels; ++k) {
              wwAvg(k, iCell) += ewm(k) * rw_p(k, iCell);
            }

            // Compute explicit rw_p update (building RHS for tridiag)
            // Fortran: rw_p(k) = rw_p(k) + dts*(tend_rw(k)
            //     - cofwz(k)*((etp(k)*zz(k)*ts(k) - etp(k-1)*zz(k-1)*ts(k-1))
            //                + (etm(k)*zz(k)*rtheta_pp(k) - etm(k-1)*zz(k-1)*rtheta_pp(k-1)))
            //     - cofwr(k)*((etp(k)*rs(k) + etp(k-1)*rs(k-1))
            //                + (etm(k)*rho_pp(k) + etm(k-1)*rho_pp(k-1)))
            //     + cofwt(k)*(etp(k)*ts(k) + etm(k)*rtheta_pp(k))
            //     + cofwt(k-1)*(etp(k-1)*ts(k-1) + etm(k-1)*rtheta_pp(k-1)))
            // for k=2..nVertLevels (0-based: k=1..nVertLevels-1)
            for (int k = 1; k < nVertLevels; ++k) {
              rw_p(k, iCell) = rw_p(k, iCell) + dts * (tend_rw(k, iCell)
                  - cofwz(k, iCell) * (
                      (etp(k) * zz(k, iCell) * ts[k]
                       - etp(k - 1) * zz(k - 1, iCell) * ts[k - 1])
                    + (etm(k) * zz(k, iCell) * rtheta_pp(k, iCell)
                       - etm(k - 1) * zz(k - 1, iCell) * rtheta_pp(k - 1, iCell)))
                  - cofwr(k, iCell) * (
                      (etp(k) * rs[k] + etp(k - 1) * rs[k - 1])
                    + (etm(k) * rho_pp(k, iCell) + etm(k - 1) * rho_pp(k - 1, iCell)))
                  + cofwt(k, iCell) * (etp(k) * ts[k]
                                       + etm(k) * rtheta_pp(k, iCell))
                  + cofwt(k - 1, iCell) * (etp(k - 1) * ts[k - 1]
                                           + etm(k - 1) * rtheta_pp(k - 1, iCell)));
            }

            // Tridiagonal solve: forward sweep (Req 4.3)
            for (int k = 1; k < nVertLevels; ++k) {
              rw_p(k, iCell) = (rw_p(k, iCell)
                                - dts2 * a_tri(k, iCell) * rw_p(k - 1, iCell))
                               * alpha_tri(k, iCell);
            }

            // Tridiagonal solve: backward sweep
            for (int k = nVertLevels - 1; k >= 0; --k) {
              rw_p(k, iCell) = rw_p(k, iCell)
                               - gamma_tri(k, iCell) * rw_p(k + 1, iCell);
            }

            // Implicit Rayleigh damping in absorbing layer (Req 4.5)
            for (int k = 1; k < nVertLevels; ++k) {
              const Scalar dss_k = dss(k, iCell);
              if (dss_k != Scalar(0.0)) {
                const Scalar rw_diff = rw_save(k, iCell) - rw(k, iCell);
                const Scalar zz_interp = fzm(k) * zz(k, iCell)
                                         + fzp(k) * zz(k - 1, iCell);
                const Scalar rho_interp = fzm(k) * rho_zz(k, iCell)
                                          + fzp(k) * rho_zz(k - 1, iCell);
                const Scalar damp_term = dts * dss_k * zz_interp
                                         * rho_interp * w(k, iCell);
                rw_p(k, iCell) = (rw_p(k, iCell) + rw_diff - damp_term)
                                 / (Scalar(1.0) + dts * dss_k)
                                 - rw_diff;
              }
            }

            // Accumulate post-solve wwAvg (ewp part)
            // Fortran: wwAvg(k) += ewp(k)*rw_p(k) for k=2..nVertLevels
            for (int k = 1; k < nVertLevels; ++k) {
              wwAvg(k, iCell) += ewp(k) * rw_p(k, iCell);
            }

            // Update rho_pp and rtheta_pp given solved rw_p
            // Fortran: rho_pp(k) = rs(k) - dts*cofrz(k)*(ewp(k+1)*rw_p(k+1) - ewp(k)*rw_p(k))
            //          rtheta_pp(k) = ts(k) - dts*rdzw(k)*(ewp(k+1)*coftz(k+1)*rw_p(k+1)
            //                                              - ewp(k)*coftz(k)*rw_p(k))
            for (int k = 0; k < nVertLevels; ++k) {
              rho_pp(k, iCell) = rs[k]
                  - dts * cofrz(k) * (ewp(k + 1) * rw_p(k + 1, iCell)
                                      - ewp(k) * rw_p(k, iCell));
              rtheta_pp(k, iCell) = ts[k]
                  - dts * rdzw(k) * (ewp(k + 1) * coftz(k + 1, iCell)
                                                 * rw_p(k + 1, iCell)
                                     - ewp(k) * coftz(k, iCell)
                                               * rw_p(k, iCell));
            }
          });  // end Kokkos::single

        } else {
          // ─── Specified zone update (Req 4.6) ─────────────────────────────
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            for (int k = 0; k < nVertLevels; ++k) {
              rho_pp(k, iCell) = rho_pp(k, iCell) + dts * tend_rho(k, iCell);
              rtheta_pp(k, iCell) = rtheta_pp(k, iCell) + dts * tend_rt(k, iCell);
              rw_p(k, iCell) = rw_p(k, iCell) + dts * tend_rw(k, iCell);
              wwAvg(k, iCell) += ewp(k) * rw_p(k, iCell);
            }
          });
        }
      });  // end parallel_for over cells
}

/// @brief Apply 3-D divergence damping to horizontal momentum (Req 4.7).
///
/// Ports `atm_divergence_damping_3d` from the Reference_Model.
/// After each acoustic substep, applies scaled 3-D divergence damping:
///
///   coef_divdamp = 2 * smdiv * config_len_disp * rdts
///   divCell1 = -(rtheta_pp(k,cell1) - rtheta_pp_old(k,cell1))
///   divCell2 = -(rtheta_pp(k,cell2) - rtheta_pp_old(k,cell2))
///   ru_p(k,iEdge) += coef_divdamp * (divCell2 - divCell1)
///                  * (1 - specZoneMaskEdge) / (theta_m(k,cell1)+theta_m(k,cell2))
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void divergence_damping_3d(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ru_p,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rtheta_pp,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rtheta_pp_old,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& specZoneMaskEdge,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
    const DivergenceDampingParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nEdges = params.nEdges;
  const int nCellsSolve = params.nCellsSolve;
  const Scalar dts = params.dts;
  const Scalar smdiv = params.smdiv;
  const Scalar config_len_disp = params.config_len_disp;

  const Scalar rdts = Scalar(1.0) / dts;
  const Scalar coef_divdamp = Scalar(2.0) * smdiv * config_len_disp * rdts;

  Kokkos::parallel_for(
      "divergence_damping_3d",
      Kokkos::RangePolicy<ExecSpace>(0, nEdges),
      KOKKOS_LAMBDA(const int iEdge) {
        const int cell1 = cellsOnEdge(0, iEdge) - 1;
        const int cell2 = cellsOnEdge(1, iEdge) - 1;

        // Only update edges touching at least one owned cell
        if (cell1 < nCellsSolve || cell2 < nCellsSolve) {
          const Scalar mask = Scalar(1.0) - specZoneMaskEdge(iEdge);

          for (int k = 0; k < nVertLevels; ++k) {
            // Scaled 3-D divergence damping (matches Reference_Model)
            const Scalar divCell1 = -(rtheta_pp(k, cell1)
                                      - rtheta_pp_old(k, cell1));
            const Scalar divCell2 = -(rtheta_pp(k, cell2)
                                      - rtheta_pp_old(k, cell2));
            ru_p(k, iEdge) += coef_divdamp * (divCell2 - divCell1) * mask
                              / (theta_m(k, cell1) + theta_m(k, cell2));
          }
        }
      });
}

/// @brief Full acoustic substep: orchestrates edge update, cell update,
///        and divergence damping.
///
/// This is the top-level entry point equivalent to calling
/// `atm_advance_acoustic_step_work` followed by `atm_divergence_damping_3d`
/// in the Reference_Model.
///
/// The execution order is:
/// 1. Update edges: compute ru_p and accumulate ruAvg
/// 2. Update cells: zero on substep 1, horizontal divergence, vertical
///    implicit solve, Rayleigh damping, accumulate wwAvg, update rho_pp/rtheta_pp
/// 3. Divergence damping: apply 3-D damping to ru_p
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void advance_acoustic_step(
    // In/out perturbation fields
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ru_p,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw_p,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_pp,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rtheta_pp,
    // Accumulated fields
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ruAvg,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& wwAvg,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rtheta_pp_old,
    // Input state fields
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& exner,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cqu,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zxu,
    // Tendencies
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_ru,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_rho,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_rt,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_rw,
    // Implicit coefficients
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cofwt,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& coftz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cofwr,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cofwz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& a_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& alpha_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& gamma_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dss,
    // Full-state fields for Rayleigh damping
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw_save,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw_base,
    // 1-D vertical coefficients
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzm,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzp,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzw,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cofrz,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& etp,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& etm,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ewp,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ewm,
    // Mesh connectivity/geometry
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dvEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invDcEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invAreaCell,
    const Kokkos::View<const int*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& nEdgesOnCell_v,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell_sign,
    // Regional masks
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& specZoneMaskEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& specZoneMaskCell,
    // Parameters
    const AcousticStepParams& step_params,
    const DivergenceDampingParams& damp_params) {

  // Step 1: Update edges (ru_p and ruAvg)
  acoustic_step_update_edges<ExecSpace>(
      ru_p, ruAvg, rtheta_pp, zz, exner, cqu, rho_pp, zxu, tend_ru,
      invDcEdge, specZoneMaskEdge, cellsOnEdge, step_params);
  Kokkos::fence("acoustic_step_edges_fence");

  // Step 2: Update cells (rho_pp, rtheta_pp, rw_p, wwAvg)
  acoustic_step_update_cells<ExecSpace>(
      rw_p, rho_pp, rtheta_pp, wwAvg, rtheta_pp_old,
      rho_zz, theta_m, ru_p,
      tend_rho, tend_rt, tend_rw,
      cofwt, coftz, cofwr, cofwz, zz,
      a_tri, alpha_tri, gamma_tri, dss,
      w, rw_save, rw_base,
      fzm, fzp, rdzw, cofrz, etp, etm, ewp, ewm,
      dvEdge, invAreaCell, nEdgesOnCell_v, cellsOnEdge, edgesOnCell,
      edgesOnCell_sign, specZoneMaskCell, step_params);
  Kokkos::fence("acoustic_step_cells_fence");

  // Step 3: 3-D divergence damping (Req 4.7)
  if (damp_params.smdiv > Scalar(0.0)) {
    // theta_m is non-const in signature but read-only here
    divergence_damping_3d<ExecSpace>(
        ru_p, rtheta_pp, rtheta_pp_old, theta_m,
        specZoneMaskEdge, cellsOnEdge, damp_params);
  }
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_ACOUSTIC_SOLVER_HPP
