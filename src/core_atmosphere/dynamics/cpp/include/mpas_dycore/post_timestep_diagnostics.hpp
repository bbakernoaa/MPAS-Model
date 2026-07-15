#ifndef MPAS_DYCORE_POST_TIMESTEP_DIAGNOSTICS_HPP
#define MPAS_DYCORE_POST_TIMESTEP_DIAGNOSTICS_HPP

/// @file post_timestep_diagnostics.hpp
/// @brief Post-timestep diagnostics: coupled diagnostics recompute, NaN guard,
///        global min/max reporting, and dry-air mass conservation diagnostic.
///
/// After the SRK3 timestep advance completes, this module performs:
/// 1. Recompute coupled diagnostics (init_coupled_diagnostics) on the
///    advanced state (Requirement 3.12 / 7.4).
/// 2. Scan all prognostic fields for NaN, aborting with the field name on
///    detection (Requirement 3.13 / 12.5).
/// 3. Report global min/max for vertical velocity (w) and horizontal
///    momentum (u), consistent with the Reference_Model diagnostic output
///    (Requirement 12.4).
/// 4. Compute the global dry-air mass for the conservation diagnostic
///    (Requirement 3.14 / 12.3).
///
/// Requirements: 12.3, 12.4, 12.5

#include "mpas_dycore/scalar.hpp"
#include "mpas_dycore/diagnostics_module.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mpas {
namespace dycore {

// ============================================================================
// Exception: NaN detected in a prognostic field (Requirement 12.5)
// ============================================================================

/// Exception thrown when a prognostic field contains a NaN value.
/// The error message identifies the field name per Requirement 12.5.
class NaNDetectedError : public std::runtime_error {
 public:
  explicit NaNDetectedError(const std::string& field_name)
      : std::runtime_error(
            "CRITICAL: NaN detected in prognostic field '" + field_name + "'"),
        field_name_(field_name) {}

  /// Returns the name of the field that contains NaN.
  const std::string& field_name() const noexcept { return field_name_; }

 private:
  std::string field_name_;
};

// ============================================================================
// MinMax result struct
// ============================================================================

/// Holds the global minimum and maximum values of a field.
struct FieldMinMax {
  Scalar min_val = std::numeric_limits<Scalar>::max();
  Scalar max_val = std::numeric_limits<Scalar>::lowest();
};

// ============================================================================
// Post-timestep diagnostics functions
// ============================================================================

/// @brief Check a 2D field (nVertLevels, nElements) for NaN values.
///
/// Scans every element of the field. If any NaN is found, throws a
/// NaNDetectedError identifying the field by name (Requirement 12.5).
///
/// @tparam ExecSpace  Kokkos execution space.
/// @param field       The 2D View (nVertLevels, nElements) to check.
/// @param field_name  Human-readable field name for error reporting.
/// @param nLevels     Number of vertical levels to scan.
/// @param nElements   Number of horizontal elements to scan.
/// @throws NaNDetectedError if any element is NaN.
template <class ExecSpace>
void check_field_for_nan(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& field,
    const std::string& field_name,
    int nLevels,
    int nElements) {

  int nan_count = 0;

  Kokkos::parallel_reduce(
      "post_timestep::nan_check_" + field_name,
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
          {0, 0}, {nLevels, nElements}),
      KOKKOS_LAMBDA(const int k, const int i, int& local_count) {
        if (Kokkos::isnan(field(k, i))) {
          local_count += 1;
        }
      },
      nan_count);

  if (nan_count > 0) {
    throw NaNDetectedError(field_name);
  }
}

/// @brief Compute the global minimum and maximum of a 2D field.
///
/// Reports global min/max for vertical velocity and horizontal momentum
/// consistent with the Reference_Model diagnostic output (Requirement 12.4).
///
/// @tparam ExecSpace  Kokkos execution space.
/// @param field       The 2D View (nVertLevels, nElements) to analyze.
/// @param nLevels     Number of vertical levels.
/// @param nElements   Number of horizontal elements.
/// @return FieldMinMax with the global minimum and maximum values.
template <class ExecSpace>
FieldMinMax compute_field_min_max(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& field,
    int nLevels,
    int nElements) {

  Scalar global_min = std::numeric_limits<Scalar>::max();
  Scalar global_max = std::numeric_limits<Scalar>::lowest();

  Kokkos::parallel_reduce(
      "post_timestep::min_max",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
          {0, 0}, {nLevels, nElements}),
      KOKKOS_LAMBDA(const int k, const int i, Scalar& lmin, Scalar& lmax) {
        const Scalar val = field(k, i);
        if (val < lmin) lmin = val;
        if (val > lmax) lmax = val;
      },
      Kokkos::Min<Scalar>(global_min),
      Kokkos::Max<Scalar>(global_max));

  return FieldMinMax{global_min, global_max};
}

/// @brief Compute the total dry-air mass over the domain.
///
/// Computes sum over all cells: rho_zz(k,iCell) * zz(k,iCell) * areaCell(iCell)
/// * dz(k,iCell) where dz is the layer thickness derived from zgrid.
///
/// This implements the conservation diagnostic: the total dry-air mass before
/// and after a timestep should agree within Parity_Tolerance (Requirement 12.3).
///
/// @tparam ExecSpace  Kokkos execution space.
/// @param rho_zz      Dry air density (nVertLevels, nCells).
/// @param zz          Jacobian dz/dzeta (nVertLevels, nCells).
/// @param areaCell    Cell areas (nCells).
/// @param zgrid       Vertical grid interface heights (nVertLevels+1, nCells).
/// @param nVertLevels Number of vertical levels.
/// @param nCells      Number of cells.
/// @return Total dry-air mass (scalar sum over all cells and levels).
template <class ExecSpace>
Scalar compute_total_dry_air_mass(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& zz,
    const Kokkos::View<Scalar*, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& areaCell,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& zgrid,
    int nVertLevels,
    int nCells) {

  Scalar total_mass = 0.0;

  Kokkos::parallel_reduce(
      "post_timestep::dry_air_mass",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
          {0, 0}, {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell, Scalar& lsum) {
        // Layer thickness: zgrid(k+1, iCell) - zgrid(k, iCell)
        const Scalar dz = zgrid(k + 1, iCell) - zgrid(k, iCell);
        // Mass contribution: rho_zz * zz * area * dz
        // Note: rho_zz already includes the metric term division, so the
        // mass is rho_zz * zz * dz * area = rho * dz * area
        lsum += rho_zz(k, iCell) * zz(k, iCell) * dz * areaCell(iCell);
      },
      total_mass);

  return total_mass;
}

// ============================================================================
// Aggregate post-timestep diagnostics result
// ============================================================================

/// Results from the post-timestep diagnostics pass.
struct PostTimestepDiagnostics {
  FieldMinMax w_min_max;    ///< Global min/max of vertical velocity
  FieldMinMax u_min_max;    ///< Global min/max of horizontal momentum
  Scalar total_dry_air_mass; ///< Total dry-air mass for conservation check
};

/// @brief Run all post-timestep diagnostics after advance() completes.
///
/// This is the top-level entry point called after Time_Integrator_Advance::advance().
/// It performs:
/// 1. Recomputes coupled diagnostics (caller responsibility — this function
///    focuses on the NaN guard, min/max, and conservation).
/// 2. NaN guard: scans prognostic fields, aborting on detection (Req 12.5).
/// 3. Global min/max for w and u (Req 12.4).
/// 4. Total dry-air mass for conservation diagnostic (Req 12.3).
///
/// @tparam ExecSpace  Kokkos execution space.
/// @param u           Horizontal momentum (nVertLevels, nEdges) at advanced time.
/// @param w           Vertical velocity (nVertLevels+1, nCells) at advanced time.
/// @param theta_m     Coupled potential temperature (nVertLevels, nCells).
/// @param rho_zz      Dry density (nVertLevels, nCells).
/// @param scalars     Scalar fields (num_scalars*nVertLevels, nCells) or
///                    individual scalar slices — for NaN checking.
/// @param zz          Jacobian (nVertLevels, nCells).
/// @param areaCell    Cell areas (nCells).
/// @param zgrid       Vertical grid (nVertLevels+1, nCells).
/// @param nVertLevels Number of vertical levels.
/// @param nCells      Number of cells (solve region).
/// @param nEdges      Number of edges (solve region).
/// @return PostTimestepDiagnostics with min/max and mass results.
/// @throws NaNDetectedError if any prognostic field contains NaN.
template <class ExecSpace>
PostTimestepDiagnostics run_post_timestep_diagnostics(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& u,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& w,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& theta_m,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& zz,
    const Kokkos::View<Scalar*, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& areaCell,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft,
                        typename ExecSpace::memory_space>& zgrid,
    int nVertLevels,
    int nCells,
    int nEdges) {

  PostTimestepDiagnostics result{};

  // ── Step 1: NaN guard (Requirement 12.5) ────────────────────────────────
  // Check all prognostic fields for NaN values. On detection, throw a
  // critical error identifying the field name. The check order matches
  // the Reference_Model diagnostic scan.
  check_field_for_nan<ExecSpace>(u, "u", nVertLevels, nEdges);
  check_field_for_nan<ExecSpace>(w, "w", nVertLevels + 1, nCells);
  check_field_for_nan<ExecSpace>(theta_m, "theta_m", nVertLevels, nCells);
  check_field_for_nan<ExecSpace>(rho_zz, "rho_zz", nVertLevels, nCells);

  // ── Step 2: Global min/max for w and u (Requirement 12.4) ───────────────
  result.w_min_max = compute_field_min_max<ExecSpace>(w, nVertLevels + 1, nCells);
  result.u_min_max = compute_field_min_max<ExecSpace>(u, nVertLevels, nEdges);

  // ── Step 3: Total dry-air mass (Requirement 12.3) ───────────────────────
  result.total_dry_air_mass = compute_total_dry_air_mass<ExecSpace>(
      rho_zz, zz, areaCell, zgrid, nVertLevels, nCells);

  return result;
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_POST_TIMESTEP_DIAGNOSTICS_HPP
