#ifndef MPAS_DYCORE_TRIDIAGONAL_SOLVE_HPP
#define MPAS_DYCORE_TRIDIAGONAL_SOLVE_HPP

/// @file tridiagonal_solve.hpp
/// @brief Thomas-algorithm tridiagonal vertical sweep for the acoustic solver.
///
/// Implements the vertically-implicit tridiagonal solve from the Reference_Model
/// `atm_advance_acoustic_step_work` subroutine. The Thomas algorithm uses the
/// precomputed implicit coefficients (a_tri, alpha_tri, gamma_tri) and applies
/// implicit Rayleigh damping in the gravity-wave absorbing layer after the
/// back-substitution.
///
/// The outer parallel_for ranges over horizontal cells while the vertical
/// recurrence (sequential dependency) is kept sequential inside each thread
/// (Requirement 2.2). This matches the Fortran+OpenACC structure where
/// `!$acc loop seq` is used for the forward and backward sweeps.
///
/// Requirements: 4.3, 4.5, 2.2

#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>

namespace mpas {
namespace dycore {

/// Parameters for the tridiagonal vertical sweep kernel.
struct TridiagonalSolveParams {
  int nVertLevels = 0;   ///< Number of vertical full levels
  int nCells = 0;        ///< Number of cells to solve over (cellSolveStart..cellSolveEnd)
  Scalar dts = 0.0;      ///< Acoustic substep timestep [s]
};

/// @brief Perform the Thomas-algorithm tridiagonal vertical sweep on rw_p.
///
/// This kernel is the C++ port of the tridiagonal solve from the Reference_Model
/// `atm_advance_acoustic_step_work` routine. It operates on the perturbation
/// vertical momentum (rw_p) field, using precomputed implicit coefficients
/// (a_tri, alpha_tri, gamma_tri) that encode the vertically-implicit coupling.
///
/// After the Thomas algorithm forward and backward sweeps, implicit Rayleigh
/// damping is applied in the gravity-wave absorbing layer using the dss damping
/// profile, the full-state vertical velocity w, the rw_save field (perturbation
/// from previous substep), and mesh geometry (fzm, fzp, zz, rho_zz).
///
/// The parallelization strategy:
/// - Outer: Kokkos::RangePolicy over cells (horizontal parallelism)
/// - Inner: Sequential loop over vertical levels (Thomas algorithm recurrence)
///
/// This preserves the sequential vertical dependency (Req 2.2) while
/// parallelizing over horizontal elements.
///
/// @tparam ExecSpace  Kokkos execution space.
///
/// @param rw_p        [in/out] Perturbation vertical momentum (nVertLevels+1, nCells).
///                    On entry: contains the RHS of the implicit system after
///                    explicit tendency accumulation. On exit: the solved vertical
///                    momentum perturbation with Rayleigh damping applied.
/// @param a_tri       [in] Lower diagonal coefficient (nVertLevels, nCells). Precomputed.
/// @param alpha_tri   [in] Forward-sweep multiplier (nVertLevels, nCells). Precomputed.
/// @param gamma_tri   [in] Back-substitution coefficient (nVertLevels, nCells). Precomputed.
/// @param dss         [in] Rayleigh damping profile (nVertLevels, nCells). Zero where
///                    no absorbing layer is present.
/// @param w           [in] Full-state vertical velocity (nVertLevels+1, nCells).
/// @param rw_save     [in] Saved rw perturbation from beginning of acoustic step
///                    (nVertLevels+1, nCells).
/// @param rw          [in] Base-state rw at current time level (nVertLevels+1, nCells).
/// @param fzm         [in] Vertical interpolation weight (nVertLevels). 1-D.
/// @param fzp         [in] Vertical interpolation weight (nVertLevels). 1-D.
/// @param zz          [in] d(zeta)/dz metric (nVertLevels, nCells).
/// @param rho_zz      [in] Dry density (nVertLevels, nCells).
/// @param params      Solve parameters (nVertLevels, nCells, dts).
///
/// Reference: Klemp et al., MWR 2007; Klemp et al., MWR 2008 (Rayleigh absorbing layer).
template <class ExecSpace>
void tridiagonal_vertical_solve(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw_p,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& a_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& alpha_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& gamma_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dss,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw_save,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzm,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzp,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const TridiagonalSolveParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const Scalar dts = params.dts;
  const Scalar dts2 = dts * dts;

  // Parallelize over horizontal cells; vertical recurrence is sequential
  // within each thread (Requirement 2.2).
  Kokkos::parallel_for(
      "tridiagonal_vertical_solve",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // ──────────────────────────────────────────────────────────────────
        // Forward sweep of the Thomas algorithm (Requirement 4.3).
        // Sweeps from level k=1 (0-based, which is Fortran k=2) up to
        // k=nVertLevels-1 (0-based, Fortran nVertLevels).
        //
        // Fortran: do k=2,nVertLevels
        //   rw_p(k,iCell) = (rw_p(k,iCell) - dts^2*a_tri(k,iCell)*rw_p(k-1,iCell))*alpha_tri(k,iCell)
        //
        // Note: In Fortran, rw_p is (nVertLevels+1, nCells) with indices 1..nVertLevels+1
        //       a_tri, alpha_tri, gamma_tri are (nVertLevels, nCells) with indices 1..nVertLevels
        //       The tridiagonal solve operates on levels 2..nVertLevels (Fortran)
        //       which is levels 1..nVertLevels-1 in 0-based C++.
        //
        // In the C++ LayoutLeft views:
        //   rw_p(k, iCell) with k in [0, nVertLevels] (size nVertLevels+1)
        //   a_tri(k, iCell), alpha_tri(k, iCell), gamma_tri(k, iCell) with k in [0, nVertLevels-1]
        // ──────────────────────────────────────────────────────────────────

        for (int k = 1; k < nVertLevels; ++k) {
          rw_p(k, iCell) = (rw_p(k, iCell) - dts2 * a_tri(k, iCell) * rw_p(k - 1, iCell))
                           * alpha_tri(k, iCell);
        }

        // ──────────────────────────────────────────────────────────────────
        // Backward sweep (back-substitution).
        // Fortran: do k=nVertLevels,1,-1 (but effectively k=nVertLevels-1,1,-1
        //          since gamma_tri(nVertLevels,iCell) may be nonzero only for
        //          k < nVertLevels, and rw_p(nVertLevels+1) = 0 boundary)
        //   rw_p(k,iCell) = rw_p(k,iCell) - gamma_tri(k,iCell)*rw_p(k+1,iCell)
        //
        // In 0-based: k goes from nVertLevels-1 down to 0.
        // Note: rw_p(nVertLevels, iCell) is the top boundary (k=nVertLevels+1
        //       in Fortran) which should be 0, so the k=nVertLevels-1 iteration
        //       uses rw_p(nVertLevels) which is the boundary value.
        // ──────────────────────────────────────────────────────────────────

        for (int k = nVertLevels - 1; k >= 0; --k) {
          rw_p(k, iCell) = rw_p(k, iCell) - gamma_tri(k, iCell) * rw_p(k + 1, iCell);
        }

        // ──────────────────────────────────────────────────────────────────
        // Implicit Rayleigh damping in the absorbing layer (Requirement 4.5).
        //
        // Fortran (k=2..nVertLevels, i.e. 0-based k=1..nVertLevels-1):
        //   rw_p(k,iCell) = (rw_p(k,iCell)
        //       + (rw_save(k,iCell) - rw(k,iCell))
        //       - dts*dss(k,iCell)
        //         * (fzm(k)*zz(k,iCell) + fzp(k)*zz(k-1,iCell))
        //         * (fzm(k)*rho_zz(k,iCell) + fzp(k)*rho_zz(k-1,iCell))
        //         * w(k,iCell)
        //     ) / (1.0 + dts*dss(k,iCell))
        //     - (rw_save(k,iCell) - rw(k,iCell))
        //
        // Where dss is zero outside the absorbing layer, the expression
        // reduces to an identity (division by 1, subtraction cancels addition).
        // The formulation handles both cases uniformly.
        // ──────────────────────────────────────────────────────────────────

        for (int k = 1; k < nVertLevels; ++k) {
          const Scalar dss_k = dss(k, iCell);
          // Skip Rayleigh damping where dss is zero (optimization for
          // the common case outside the absorbing layer)
          if (dss_k != Scalar(0.0)) {
            const Scalar rw_diff = rw_save(k, iCell) - rw(k, iCell);
            const Scalar zz_interp = fzm(k) * zz(k, iCell) + fzp(k) * zz(k - 1, iCell);
            const Scalar rho_interp = fzm(k) * rho_zz(k, iCell) + fzp(k) * rho_zz(k - 1, iCell);
            const Scalar damp_term = dts * dss_k * zz_interp * rho_interp * w(k, iCell);

            rw_p(k, iCell) = (rw_p(k, iCell) + rw_diff - damp_term)
                             / (Scalar(1.0) + dts * dss_k)
                             - rw_diff;
          }
        }
      });
}

/// @brief Convenience overload for the tridiagonal solve without Rayleigh damping.
///
/// This is useful for testing the pure Thomas algorithm independent of the
/// damping layer. It performs only the forward and backward sweeps.
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void tridiagonal_solve_no_damping(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rw_p,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& a_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& alpha_tri,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& gamma_tri,
    const TridiagonalSolveParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const Scalar dts2 = params.dts * params.dts;

  Kokkos::parallel_for(
      "tridiagonal_solve_no_damping",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // Forward sweep
        for (int k = 1; k < nVertLevels; ++k) {
          rw_p(k, iCell) = (rw_p(k, iCell) - dts2 * a_tri(k, iCell) * rw_p(k - 1, iCell))
                           * alpha_tri(k, iCell);
        }

        // Backward sweep
        for (int k = nVertLevels - 1; k >= 0; --k) {
          rw_p(k, iCell) = rw_p(k, iCell) - gamma_tri(k, iCell) * rw_p(k + 1, iCell);
        }
      });
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_TRIDIAGONAL_SOLVE_HPP
