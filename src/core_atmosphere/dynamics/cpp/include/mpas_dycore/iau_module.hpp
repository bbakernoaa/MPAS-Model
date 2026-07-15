#ifndef MPAS_DYCORE_IAU_MODULE_HPP
#define MPAS_DYCORE_IAU_MODULE_HPP

/// @file iau_module.hpp
/// @brief Incremental Analysis Update (IAU) forcing module.
///
/// Ports `mpas_atm_iau.F` to C++. Within the IAU window the module adds a
/// constant-weight forcing (= increment / window_length) to momentum, density,
/// coupled theta, and moisture tendencies. Outside the window or when IAU is
/// off, no modification is applied.
///
/// Requirements: 10.1, 10.2, 10.3, 10.4

#include "mpas_dycore/config.hpp"
#include "mpas_dycore/field_store.hpp"
#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>

namespace mpas {
namespace dycore {

/// Physical constant: ratio of water vapor gas constant to dry air gas
/// constant (rv / rgas = 461.6 / 287.0), matching the Reference_Model.
inline constexpr Scalar rvord = Scalar(461.6) / Scalar(287.0);

/// The negligible-weight threshold below which IAU returns without modifying
/// tendencies (Req 10.3). Matches the Reference_Model's `1.0e-12_RKIND`
/// comparison for double precision, and uses `1.0e-6` for single precision
/// (matching the Parity_Tolerance ceiling for float).
inline constexpr Scalar iau_negligible_threshold =
    std::is_same_v<Scalar, double> ? Scalar(1.0e-12) : Scalar(1.0e-6);

/// @brief Incremental Analysis Update (IAU) forcing module.
///
/// Template parameters:
///   - ExecSpace: A Kokkos execution space (e.g. DefaultExecutionSpace).
///
/// The module is stateless; all data is accessed through the Field_Store and
/// Config references passed to `add_iau_tendency`.
template <class ExecSpace>
class IAU_Module {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using view2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, memory_space>;

  /// @brief Apply IAU forcing to tendency fields.
  ///
  /// Behavior:
  ///   - When config_iau is false: returns immediately (Req 10.1).
  ///   - Computes the IAU weight = 1 / iau_window_length_s. If the weight is
  ///     at or below the negligible threshold, returns without modification
  ///     (Req 10.3).
  ///   - When within the IAU window (itimestep <= nsteps_iau): adds forcing to
  ///     momentum (tend_ru), density (tend_rho), coupled theta (tend_rtheta),
  ///     and moisture (tend_scalars) tendencies (Req 10.2).
  ///   - Converts theta increment to coupled moist theta tendency (Req 10.4).
  ///
  /// @param config         Immutable configuration snapshot.
  /// @param fields         Field store containing all required fields.
  /// @param itimestep      Current timestep index (1-based).
  /// @param dt             Model timestep length [seconds].
  /// @param iau_window_length_s  IAU window length [seconds].
  /// @param nEdgesSolve    Number of solve-domain edges.
  /// @param nCellsSolve    Number of solve-domain cells.
  /// @param nVertLevels    Number of vertical levels.
  /// @param moist_start    Start index of moist scalars (0-based in C++).
  /// @param moist_end      End index of moist scalars (exclusive, 0-based).
  /// @param index_qv       Index of water vapor in the scalars array (0-based).
  /// @param tend_ru        Momentum tendency [nVertLevels x nEdges].
  /// @param tend_rho       Density tendency [nVertLevels x nCells].
  /// @param tend_rtheta    Coupled theta tendency [nVertLevels x nCells].
  /// @param tend_scalars   Scalar tendencies [nScalars x nVertLevels x nCells] — passed as
  ///                       a 2-D slice [nVertLevels x nCells] per scalar, or a rank-3 view.
  /// @param rho_edge       Edge density [nVertLevels x nEdges].
  /// @param rho_zz         Coupled density [nVertLevels x nCells].
  /// @param theta_m        Modified potential temperature [nVertLevels x nCells].
  /// @param scalars_qv     Water vapor mixing ratio [nVertLevels x nCells].
  /// @param zz             Height metric [nVertLevels x nCells].
  /// @param u_amb          IAU increment for u [nVertLevels x nEdges].
  /// @param rho_amb        IAU increment for rho [nVertLevels x nCells].
  /// @param theta_amb      IAU increment for theta [nVertLevels x nCells].
  /// @param scalars_amb    IAU increment for moist scalars [nScalars x nVertLevels x nCells].
  static void add_iau_tendency(
      const Config& config,
      int itimestep,
      Scalar dt,
      Scalar iau_window_length_s,
      int nEdgesSolve,
      int nCellsSolve,
      int nVertLevels,
      int moist_start,
      int moist_end,
      int index_qv,
      const view2d& tend_ru,
      const view2d& tend_rho,
      const view2d& tend_rtheta,
      const view2d& rho_edge,
      const view2d& rho_zz,
      const view2d& theta_m,
      const view2d& scalars_qv,
      const view2d& zz,
      const view2d& u_amb,
      const view2d& rho_amb,
      const view2d& theta_amb,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, memory_space>& tend_scalars_slice,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, memory_space>& scalars_amb_slice,
      const view2d& scalars_slice) {
    // Req 10.1: No forcing when IAU is off.
    if (!config.config_iau) {
      return;
    }

    // Compute IAU weight: constant within the window.
    const int nsteps_iau = static_cast<int>(
        std::round(iau_window_length_s / dt));

    // Outside the IAU window: weight is zero, return.
    if (itimestep > nsteps_iau) {
      return;
    }

    const Scalar wgt_iau = Scalar(1.0) / iau_window_length_s;

    // Req 10.3: Return without modification when weight is negligible.
    if (wgt_iau <= iau_negligible_threshold) {
      return;
    }

    // Req 10.2: Add IAU forcing to momentum tendency on edges.
    Kokkos::parallel_for(
        "IAU_tend_ru",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nEdgesSolve}),
        KOKKOS_LAMBDA(const int k, const int iEdge) {
          tend_ru(k, iEdge) += wgt_iau * rho_edge(k, iEdge) * u_amb(k, iEdge);
        });

    // Req 10.2: Add IAU forcing to density tendency (rho_zz = rho/zz).
    Kokkos::parallel_for(
        "IAU_tend_rho",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nCellsSolve}),
        KOKKOS_LAMBDA(const int k, const int iCell) {
          tend_rho(k, iCell) += wgt_iau * rho_amb(k, iCell) / zz(k, iCell);
        });

    // Req 10.2 & 10.4: Compute theta tendency and convert to coupled moist theta.
    // Also add moisture scalar tendencies.
    // Note: moist scalars are packed into tend_scalars_slice as
    //   [nMoist * nVertLevels x nCells] where the first dimension packs
    //   (scalar_index * nVertLevels + k). We use a flat approach here.
    //
    // The Fortran does this in two passes:
    //   1) Compute tend_th (theta tendency) and add moisture scalar tendencies.
    //   2) Convert tend_th to coupled moist theta tendency.
    //
    // We fuse these into a single kernel over cells.
    const int n_moist = moist_end - moist_start;

    Kokkos::parallel_for(
        "IAU_tend_theta_scalars",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nCellsSolve}),
        KOKKOS_LAMBDA(const int k, const int iCell) {
          // Get current theta from theta_m: theta = theta_m / (1 + rvord * qv)
          const Scalar qv = scalars_qv(k, iCell);
          const Scalar theta = theta_m(k, iCell) / (Scalar(1.0) + rvord * qv);
          const Scalar rho_zz_val = rho_zz(k, iCell);
          const Scalar zz_val = zz(k, iCell);
          const Scalar rho_amb_val = rho_amb(k, iCell);

          // Theta tendency: wgt * (theta_amb * rho_zz + theta * rho_amb / zz)
          Scalar tend_th = wgt_iau * (theta_amb(k, iCell) * rho_zz_val
                                      + theta * rho_amb_val / zz_val);

          // Add moisture scalar tendencies (Req 10.2).
          // tend_scalars for each moist species:
          //   tend_scalars += wgt * (scalars_amb * rho_zz + scalars * rho_amb / zz)
          // The qv tendency is needed for the coupled theta conversion.
          Scalar tend_qv = Scalar(0.0);
          for (int m = 0; m < n_moist; ++m) {
            // Packed index: m * nVertLevels + k rows, iCell columns
            const int row = m * nVertLevels + k;
            const Scalar scalar_val = scalars_slice(m * nVertLevels + k, iCell);
            const Scalar scalar_amb_val = scalars_amb_slice(row, iCell);
            const Scalar tend_s = wgt_iau * (scalar_amb_val * rho_zz_val
                                             + scalar_val * rho_amb_val / zz_val);
            tend_scalars_slice(row, iCell) += tend_s;
            if (m == (index_qv - moist_start)) {
              tend_qv = tend_s;
            }
          }

          // Req 10.4: Convert theta tendency to coupled moist theta tendency.
          // tend_rtheta += (1 + rvord * qv) * tend_th + rvord * theta * tend_qv
          tend_rtheta(k, iCell) += (Scalar(1.0) + rvord * qv) * tend_th
                                   + rvord * theta * tend_qv;
        });
  }
};

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_IAU_MODULE_HPP
