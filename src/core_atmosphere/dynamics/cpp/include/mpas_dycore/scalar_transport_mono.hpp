#ifndef MPAS_DYCORE_SCALAR_TRANSPORT_MONO_HPP
#define MPAS_DYCORE_SCALAR_TRANSPORT_MONO_HPP

/// @file scalar_transport_mono.hpp
/// @brief Monotonic/positive-definite scalar transport module for the C++ dycore.
///
/// Ports `atm_advance_scalars_mono` / `atm_advance_scalars_mono_work` from the
/// Fortran Reference_Model to C++ using Kokkos.
///
/// Requirements: 5.2, 5.6, 5.7, 5.8

#include "mpas_dycore/scalar.hpp"
#include "mpas_dycore/scalar_transport.hpp"
#include "mpas_dycore/config.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

namespace mpas {
namespace dycore {

/// Specified zone constant matching the Reference_Model (same as boundary_module).
inline constexpr int mono_nSpecZone = 2;

/// Additional mesh data needed by the monotonic transport (beyond ScalarTransportMeshData).
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct MonoTransportMeshData {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  template <class T> using View1D = Kokkos::View<T*, layout, memory_space>;
  template <class T> using View2D = Kokkos::View<T**, layout, memory_space>;

  int nCellsSolve = 0;        ///< Number of owned cells (excludes halo)
  int maxEdges = 0;           ///< Max edges per cell (for cellsOnCell)

  View2D<int> cellsOnCell;    ///< (maxEdges, nCells) - 1-based neighbor indices
};

/// Input/output state fields for monotonic scalar transport.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct MonoTransportState {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  template <class T> using View2D = Kokkos::View<T**, layout, memory_space>;
  template <class T> using View3D = Kokkos::View<T***, layout, memory_space>;

  View3D<Scalar> scalars_old;  ///< (num_scalars, nVertLevels, nCells) - TL1
  View3D<Scalar> scalars_new;  ///< (num_scalars, nVertLevels, nCells) - TL2
  View3D<Scalar> scalar_tend;  ///< (num_scalars, nVertLevels, nCells)
  View2D<Scalar> rho_zz_old;   ///< (nVertLevels, nCells)
  View2D<Scalar> rho_zz_new;   ///< (nVertLevels, nCells)
  View2D<Scalar> uhAvg;        ///< (nVertLevels, nEdges)
  View2D<Scalar> wwAvg;        ///< (nVertLevels+1, nCells)
};

/// @brief Monotonic/positive-definite scalar transport module.
///
/// Applies the FCT (flux-corrected transport) monotonic limiter on the final
/// RK substep, keeping scalars within local min/max bounds. Sets negative
/// water-species mixing ratios to zero. Applies regional boundary flux
/// treatment in relaxation/specified zones.
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
class Scalar_Transport_Mono {
 public:
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  using view2d = Kokkos::View<Scalar**, layout, memory_space>;
  using view3d = Kokkos::View<Scalar***, layout, memory_space>;

  Scalar_Transport_Mono() = default;

  /// @brief Advance scalars using the monotonic flux-corrected transport.
  ///
  /// Equivalent to `atm_advance_scalars_mono_work` in the Reference_Model.
  /// Algorithm:
  /// 1. Apply source tendencies to scalars_old (pre-transport update)
  /// 2. For each scalar:
  ///    a. Compute local min/max bounds from old scalars + neighbors
  ///    b. Compute vertical fluxes (same as standard transport)
  ///    c. Compute horizontal fluxes (high-order)
  ///    d. Compute upwind fluxes (first-order)
  ///    e. Separate perturbation flux = high-order - upwind
  ///    f. Update scalars with upwind fluxes
  ///    g. Compute FCT scale factors to keep within bounds
  ///    h. Apply limited perturbation fluxes
  ///    i. Divide by density to get mixing ratio
  ///    j. Set negatives to zero for water species (Req 5.7)
  ///    k. Apply spec-zone update for regional (Req 5.8)
  void advance_scalars_mono(
      const ScalarTransportMeshData<ExecSpace>& mesh,
      const MonoTransportMeshData<ExecSpace>& mono_mesh,
      MonoTransportState<ExecSpace>& state,
      Scalar dt,
      Scalar coef_3rd_order,
      bool advance_density,
      bool config_apply_lbcs,
      int moist_start,      ///< First water species index (0-based)
      int moist_end) const; ///< One past last water species index (0-based)
};

// ============================================================================
// Implementation
// ============================================================================

template <class ExecSpace>
void Scalar_Transport_Mono<ExecSpace>::advance_scalars_mono(
    const ScalarTransportMeshData<ExecSpace>& mesh,
    const MonoTransportMeshData<ExecSpace>& mono_mesh,
    MonoTransportState<ExecSpace>& state,
    Scalar dt,
    Scalar coef_3rd_order,
    bool advance_density,
    bool config_apply_lbcs,
    int moist_start,
    int moist_end) const {

  const int nCells = mesh.nCells;
  const int nEdges = mesh.nEdges;
  const int nVertLevels = mesh.nVertLevels;
  const int num_scalars = mesh.num_scalars;
  const int nCellsSolve = mono_mesh.nCellsSolve;

  // Mesh connectivity and geometry
  const auto cellsOnEdge = mesh.cellsOnEdge;
  const auto edgesOnCell = mesh.edgesOnCell;
  const auto nEdgesOnCell_v = mesh.nEdgesOnCell;
  const auto advCellsForEdge = mesh.advCellsForEdge;
  const auto nAdvCellsForEdge_v = mesh.nAdvCellsForEdge;
  const auto adv_coefs = mesh.adv_coefs;
  const auto adv_coefs_3rd = mesh.adv_coefs_3rd;
  const auto edgesOnCell_sign = mesh.edgesOnCell_sign;
  const auto dvEdge = mesh.dvEdge;
  const auto invAreaCell = mesh.invAreaCell;
  const auto fnm = mesh.fnm;
  const auto fnp = mesh.fnp;
  const auto rdnw = mesh.rdnw;
  const auto bdyMaskCell = mesh.bdyMaskCell;
  const auto bdyMaskEdge = mesh.bdyMaskEdge;
  const auto cellsOnCell = mono_mesh.cellsOnCell;

  // State fields
  auto scalars_old = state.scalars_old;
  auto scalars_new = state.scalars_new;
  auto scalar_tend = state.scalar_tend;
  const auto rho_zz_old = state.rho_zz_old;
  const auto rho_zz_new = state.rho_zz_new;
  const auto uhAvg = state.uhAvg;
  const auto wwAvg = state.wwAvg;

  // Small epsilon to avoid division by zero in the limiter (matches Fortran)
  constexpr Scalar eps = Scalar(1.0e-20);

  // ── Step 0: Re-integrate density when advance_density=true (Req 5.5) ──
  // Compute rho_zz_int = rho_zz_old + dt * (horiz_div + vert_div of mass flux)
  view2d rho_zz_int("rho_zz_int", nVertLevels, nCells);

  if (advance_density) {
    Kokkos::parallel_for(
        "mono::density_reintegrate",
        Kokkos::RangePolicy<exec_space>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          // Zero accumulator
          for (int k = 0; k < nVertLevels; ++k) {
            rho_zz_int(k, iCell) = Scalar(0.0);
          }
          // Horizontal mass flux divergence
          const int ne = nEdgesOnCell_v(iCell);
          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            if (iEdge < 0 || iEdge >= nEdges) continue;
            for (int k = 0; k < nVertLevels; ++k) {
              rho_zz_int(k, iCell) -= edgesOnCell_sign(i, iCell) *
                  uhAvg(k, iEdge) * dvEdge(iEdge) * invAreaCell(iCell);
            }
          }
          // Add vertical divergence and time integrate
          for (int k = 0; k < nVertLevels; ++k) {
            rho_zz_int(k, iCell) = rho_zz_old(k, iCell) +
                dt * (rho_zz_int(k, iCell) -
                      rdnw(k) * (wwAvg(k + 1, iCell) - wwAvg(k, iCell)));
          }
        });
    Kokkos::fence("mono::density_reintegrate_fence");
  }

  // ── Step 1: Apply source tendencies to scalars_old (pre-transport) ──
  // This is the positive-definite pre-update from the Reference_Model:
  //   scalars_old += dt * scalar_tend / rho_zz_old
  //   scalar_tend = 0 (consumed)
  Kokkos::parallel_for(
      "mono::apply_source_tend",
      Kokkos::RangePolicy<exec_space>(0, nCellsSolve),
      KOKKOS_LAMBDA(const int iCell) {
        for (int k = 0; k < nVertLevels; ++k) {
          for (int s = 0; s < num_scalars; ++s) {
            scalars_old(s, k, iCell) += dt * scalar_tend(s, k, iCell) /
                rho_zz_old(k, iCell);
            scalar_tend(s, k, iCell) = Scalar(0.0);
          }
        }
      });
  Kokkos::fence("mono::apply_source_tend_fence");

  // NOTE: In the full model, a halo exchange on scalars_old would happen here.
  // For the component-level implementation, we assume the caller handles it.

  // ── Per-scalar loop: FCT monotonic limiter (Req 5.2, 5.6) ──
  // Allocate working buffers (reused per scalar)
  view2d scalar_old_2d("scalar_old_2d", nVertLevels, nCells);
  view2d scalar_new_2d("scalar_new_2d", nVertLevels, nCells);
  view2d s_max("s_max", nVertLevels, nCells);
  view2d s_min("s_min", nVertLevels, nCells);
  view2d wdtn("wdtn", nVertLevels + 1, nCells);
  view2d flux_arr("flux_arr", nVertLevels, nEdges);
  view2d flux_upwind_tmp("flux_upwind_tmp", nVertLevels, nEdges);
  view2d flux_tmp("flux_tmp", nVertLevels, nEdges);
  // scale_arr: (nVertLevels, 2, nCells) -> flatten as (nVertLevels*2, nCells)
  // SCALE_IN = index 0..nVertLevels-1, SCALE_OUT = index nVertLevels..2*nVertLevels-1
  view2d scale_in("scale_in", nVertLevels, nCells);
  view2d scale_out("scale_out", nVertLevels, nCells);

  for (int iScalar = 0; iScalar < num_scalars; ++iScalar) {
    // ── Copy scalar data to 2D working arrays ──
    Kokkos::parallel_for(
        "mono::copy_to_2d",
        Kokkos::RangePolicy<exec_space>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          for (int k = 0; k < nVertLevels; ++k) {
            scalar_old_2d(k, iCell) = scalars_old(iScalar, k, iCell);
            scalar_new_2d(k, iCell) = scalars_new(iScalar, k, iCell);
          }
        });
    Kokkos::fence("mono::copy_to_2d_fence");

    // ── Compute vertical fluxes and local min/max bounds ──
    Kokkos::parallel_for(
        "mono::vert_flux_and_bounds",
        Kokkos::RangePolicy<exec_space>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          // Zero boundary fluxes
          wdtn(0, iCell) = Scalar(0.0);
          wdtn(nVertLevels, iCell) = Scalar(0.0);

          // k=0: bounds from k=0,1
          s_max(0, iCell) = Kokkos::fmax(
              scalar_old_2d(0, iCell), scalar_old_2d(1, iCell));
          s_min(0, iCell) = Kokkos::fmin(
              scalar_old_2d(0, iCell), scalar_old_2d(1, iCell));

          // k=1: linear vert flux, bounds from k=0,1,2
          if (nVertLevels > 1) {
            wdtn(1, iCell) = wwAvg(1, iCell) *
                (fnm(1) * scalar_new_2d(1, iCell) +
                 fnp(1) * scalar_new_2d(0, iCell));
            if (nVertLevels > 2) {
              s_max(1, iCell) = Kokkos::fmax(
                  Kokkos::fmax(scalar_old_2d(0, iCell),
                               scalar_old_2d(1, iCell)),
                  scalar_old_2d(2, iCell));
              s_min(1, iCell) = Kokkos::fmin(
                  Kokkos::fmin(scalar_old_2d(0, iCell),
                               scalar_old_2d(1, iCell)),
                  scalar_old_2d(2, iCell));
            } else {
              s_max(1, iCell) = Kokkos::fmax(
                  scalar_old_2d(0, iCell), scalar_old_2d(1, iCell));
              s_min(1, iCell) = Kokkos::fmin(
                  scalar_old_2d(0, iCell), scalar_old_2d(1, iCell));
            }
          }

          // Interior interfaces k=2..nVertLevels-2: flux3
          for (int k = 2; k <= nVertLevels - 2; ++k) {
            const Scalar q_im2 = scalar_new_2d(k - 2, iCell);
            const Scalar q_im1 = scalar_new_2d(k - 1, iCell);
            const Scalar q_i   = scalar_new_2d(k, iCell);
            const Scalar q_ip1 = scalar_new_2d(k + 1, iCell);
            const Scalar w = wwAvg(k, iCell);
            const Scalar flux4 = w *
                (Scalar(7.0) * (q_i + q_im1) - (q_ip1 + q_im2)) /
                Scalar(12.0);
            wdtn(k, iCell) = flux4 +
                coef_3rd_order * Kokkos::fabs(w) *
                ((q_ip1 - q_im2) - Scalar(3.0) * (q_i - q_im1)) /
                Scalar(12.0);

            // Bounds for k (interior)
            s_max(k, iCell) = Kokkos::fmax(
                Kokkos::fmax(scalar_old_2d(k - 1, iCell),
                             scalar_old_2d(k, iCell)),
                scalar_old_2d(k + 1, iCell));
            s_min(k, iCell) = Kokkos::fmin(
                Kokkos::fmin(scalar_old_2d(k - 1, iCell),
                             scalar_old_2d(k, iCell)),
                scalar_old_2d(k + 1, iCell));
          }

          // k=nVertLevels-1: linear vert flux, bounds from k-1,k
          if (nVertLevels > 1) {
            const int kk = nVertLevels - 1;
            wdtn(kk, iCell) = wwAvg(kk, iCell) *
                (fnm(kk) * scalar_new_2d(kk, iCell) +
                 fnp(kk) * scalar_new_2d(kk - 1, iCell));
            s_max(kk, iCell) = Kokkos::fmax(
                scalar_old_2d(kk, iCell), scalar_old_2d(kk - 1, iCell));
            s_min(kk, iCell) = Kokkos::fmin(
                scalar_old_2d(kk, iCell), scalar_old_2d(kk - 1, iCell));
          }

          // Pull min/max from horizontal neighbors (cellsOnCell)
          const int ne = nEdgesOnCell_v(iCell);
          for (int i = 0; i < ne; ++i) {
            const int iNeighbor = cellsOnCell(i, iCell) - 1; // 1-based to 0-based
            if (iNeighbor >= 0 && iNeighbor < nCells) {
              for (int k = 0; k < nVertLevels; ++k) {
                s_max(k, iCell) = Kokkos::fmax(
                    s_max(k, iCell), scalar_old_2d(k, iNeighbor));
                s_min(k, iCell) = Kokkos::fmin(
                    s_min(k, iCell), scalar_old_2d(k, iNeighbor));
              }
            }
          }
        });
    Kokkos::fence("mono::vert_flux_and_bounds_fence");

    // ── Compute high-order horizontal fluxes ──
    Kokkos::parallel_for(
        "mono::horiz_flux",
        Kokkos::RangePolicy<exec_space>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;

          if (cell1 < nCellsSolve || cell2 < nCellsSolve) {
            const int nAdv = nAdvCellsForEdge_v(iEdge);
            for (int k = 0; k < nVertLevels; ++k) {
              const Scalar sign_u = (uhAvg(k, iEdge) >= Scalar(0.0))
                  ? Scalar(1.0) : Scalar(-1.0);
              Scalar flux_val = Scalar(0.0);
              for (int j = 0; j < nAdv; ++j) {
                const int iAdvCell = advCellsForEdge(j, iEdge) - 1;
                if (iAdvCell < 0 || iAdvCell >= nCells) continue;
                const Scalar sw = adv_coefs(j, iEdge) +
                    sign_u * adv_coefs_3rd(j, iEdge);
                flux_val += sw * scalar_new_2d(k, iAdvCell);
              }
              flux_arr(k, iEdge) = uhAvg(k, iEdge) * flux_val;
            }
          } else {
            for (int k = 0; k < nVertLevels; ++k) {
              flux_arr(k, iEdge) = Scalar(0.0);
            }
          }
        });
    Kokkos::fence("mono::horiz_flux_fence");

    // ── Upwind update + compute scale factors (SCALE_IN/SCALE_OUT) ──
    // First: vertical upwind update and vertical scale contributions
    Kokkos::parallel_for(
        "mono::upwind_vert",
        Kokkos::RangePolicy<exec_space>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          // Initialize scalar_new_2d with rho*scalar (upwind base)
          for (int k = 0; k < nVertLevels; ++k) {
            scalar_new_2d(k, iCell) =
                scalar_old_2d(k, iCell) * rho_zz_old(k, iCell);
          }

          // Vertical upwind flux and its application
          // flux_upwind(k) = dt*(max(0,w)*q(k-1) + min(0,w)*q(k))
          // Apply to scalar_new_2d and compute perturbation wdtn
          for (int k = 1; k < nVertLevels; ++k) {
            const Scalar flux_up = dt * (
                Kokkos::fmax(Scalar(0.0), wwAvg(k, iCell)) *
                    scalar_old_2d(k - 1, iCell) +
                Kokkos::fmin(Scalar(0.0), wwAvg(k, iCell)) *
                    scalar_old_2d(k, iCell));
            // Remove upwind from above cell (k-1)
            scalar_new_2d(k - 1, iCell) -= flux_up * rdnw(k - 1);
            // Add upwind to below cell (k)
            scalar_new_2d(k, iCell) += flux_up * rdnw(k);
            // Perturbation vert flux = high-order - upwind
            wdtn(k, iCell) = dt * wdtn(k, iCell) - flux_up;
          }

          // Vertical contribution to scale_in/scale_out
          for (int k = 0; k < nVertLevels; ++k) {
            scale_in(k, iCell) = -rdnw(k) *
                (Kokkos::fmin(Scalar(0.0), wdtn(k + 1, iCell)) -
                 Kokkos::fmax(Scalar(0.0), wdtn(k, iCell)));
            scale_out(k, iCell) = -rdnw(k) *
                (Kokkos::fmax(Scalar(0.0), wdtn(k + 1, iCell)) -
                 Kokkos::fmin(Scalar(0.0), wdtn(k, iCell)));
          }
        });
    Kokkos::fence("mono::upwind_vert_fence");

    // ── Horizontal upwind flux and perturbation computation (Req 5.8) ──
    Kokkos::parallel_for(
        "mono::upwind_horiz_edge",
        Kokkos::RangePolicy<exec_space>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;

          for (int k = 0; k < nVertLevels; ++k) {
            // First-order upwind flux
            flux_upwind_tmp(k, iEdge) = dvEdge(iEdge) * dt * (
                Kokkos::fmax(Scalar(0.0), uhAvg(k, iEdge)) *
                    scalar_old_2d(k, cell1) +
                Kokkos::fmin(Scalar(0.0), uhAvg(k, iEdge)) *
                    scalar_old_2d(k, cell2));
            // Perturbation flux = high-order*dt - upwind
            flux_tmp(k, iEdge) = dt * flux_arr(k, iEdge) -
                flux_upwind_tmp(k, iEdge);
          }

          // Regional boundary treatment (Req 5.8):
          // In relaxation boundary zone, use pure upwind (zero perturbation)
          if (config_apply_lbcs &&
              (bdyMaskEdge(iEdge) == transport_nRelaxZone ||
               bdyMaskEdge(iEdge) == transport_nRelaxZone - 1)) {
            for (int k = 0; k < nVertLevels; ++k) {
              flux_tmp(k, iEdge) = Scalar(0.0);
            }
          }
        });
    Kokkos::fence("mono::upwind_horiz_edge_fence");

    // ── Accumulate horizontal upwind flux and scale contributions ──
    Kokkos::parallel_for(
        "mono::upwind_horiz_cell",
        Kokkos::RangePolicy<exec_space>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          const int ne = nEdgesOnCell_v(iCell);
          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            if (iEdge < 0 || iEdge >= nEdges) continue;
            for (int k = 0; k < nVertLevels; ++k) {
              // Apply upwind horizontal flux
              scalar_new_2d(k, iCell) -= edgesOnCell_sign(i, iCell) *
                  flux_upwind_tmp(k, iEdge) * invAreaCell(iCell);
              // Accumulate perturbation contributions for limiter
              scale_out(k, iCell) -= Kokkos::fmax(Scalar(0.0),
                  edgesOnCell_sign(i, iCell) * flux_tmp(k, iEdge)) *
                  invAreaCell(iCell);
              scale_in(k, iCell) -= Kokkos::fmin(Scalar(0.0),
                  edgesOnCell_sign(i, iCell) * flux_tmp(k, iEdge)) *
                  invAreaCell(iCell);
            }
          }
        });
    Kokkos::fence("mono::upwind_horiz_cell_fence");

    // ── Compute FCT scale factors (Req 5.6) ──
    // scale_in = min(1, max(0, (s_max*rho - scalar_new) / (scale_in + eps)))
    // scale_out = min(1, max(0, (s_min*rho - scalar_new) / (scale_out - eps)))
    Kokkos::parallel_for(
        "mono::compute_scale_factors",
        Kokkos::RangePolicy<exec_space>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          for (int k = 0; k < nVertLevels; ++k) {
            const Scalar rho_target = advance_density
                ? rho_zz_int(k, iCell)
                : rho_zz_new(k, iCell);

            Scalar sf_in = (s_max(k, iCell) * rho_target -
                            scalar_new_2d(k, iCell)) /
                (scale_in(k, iCell) + eps);
            scale_in(k, iCell) = Kokkos::fmin(Scalar(1.0),
                Kokkos::fmax(Scalar(0.0), sf_in));

            Scalar sf_out = (s_min(k, iCell) * rho_target -
                             scalar_new_2d(k, iCell)) /
                (scale_out(k, iCell) - eps);
            scale_out(k, iCell) = Kokkos::fmin(Scalar(1.0),
                Kokkos::fmax(Scalar(0.0), sf_out));
          }
        });
    Kokkos::fence("mono::compute_scale_factors_fence");

    // NOTE: In the full model, a halo exchange on scale_in/scale_out happens
    // here (dynamics:scale). For component-level, we assume the caller handles.

    // ── Rescale horizontal perturbation fluxes ──
    Kokkos::parallel_for(
        "mono::rescale_horiz_flux",
        Kokkos::RangePolicy<exec_space>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;

          if (cell1 < nCellsSolve || cell2 < nCellsSolve) {
            // Recompute perturbation flux (high-order - upwind) for rescaling
            for (int k = 0; k < nVertLevels; ++k) {
              const Scalar flux_up = dvEdge(iEdge) * dt * (
                  Kokkos::fmax(Scalar(0.0), uhAvg(k, iEdge)) *
                      scalar_old_2d(k, cell1) +
                  Kokkos::fmin(Scalar(0.0), uhAvg(k, iEdge)) *
                      scalar_old_2d(k, cell2));
              Scalar flux_pert = dt * flux_arr(k, iEdge) - flux_up;

              // Regional: zero perturbation at boundary (Req 5.8)
              if (config_apply_lbcs &&
                  (bdyMaskEdge(iEdge) == transport_nRelaxZone ||
                   bdyMaskEdge(iEdge) == transport_nRelaxZone - 1)) {
                flux_pert = Scalar(0.0);
              }

              // Apply FCT limiter to perturbation flux
              flux_arr(k, iEdge) =
                  Kokkos::fmax(Scalar(0.0), flux_pert) *
                      Kokkos::fmin(scale_out(k, cell1), scale_in(k, cell2)) +
                  Kokkos::fmin(Scalar(0.0), flux_pert) *
                      Kokkos::fmin(scale_in(k, cell1), scale_out(k, cell2));
            }
          }
        });
    Kokkos::fence("mono::rescale_horiz_flux_fence");

    // ── Rescale vertical perturbation fluxes ──
    Kokkos::parallel_for(
        "mono::rescale_vert_flux",
        Kokkos::RangePolicy<exec_space>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          for (int k = 1; k < nVertLevels; ++k) {
            const Scalar flux_pert = wdtn(k, iCell);
            wdtn(k, iCell) =
                Kokkos::fmax(Scalar(0.0), flux_pert) *
                    Kokkos::fmin(scale_out(k - 1, iCell),
                                 scale_in(k, iCell)) +
                Kokkos::fmin(Scalar(0.0), flux_pert) *
                    Kokkos::fmin(scale_out(k, iCell),
                                 scale_in(k - 1, iCell));
          }
        });
    Kokkos::fence("mono::rescale_vert_flux_fence");

    // ── Final scalar update: apply limited fluxes and divide by density ──
    Kokkos::parallel_for(
        "mono::final_update",
        Kokkos::RangePolicy<exec_space>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          // Apply limited horizontal perturbation flux
          const int ne = nEdgesOnCell_v(iCell);
          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            if (iEdge < 0 || iEdge >= nEdges) continue;
            for (int k = 0; k < nVertLevels; ++k) {
              scalar_new_2d(k, iCell) -= edgesOnCell_sign(i, iCell) *
                  flux_arr(k, iEdge) * invAreaCell(iCell);
            }
          }

          // Apply limited vertical perturbation flux and divide by density
          const Scalar rho_denom_advance = advance_density
              ? Scalar(1.0) : Scalar(0.0);
          for (int k = 0; k < nVertLevels; ++k) {
            scalar_new_2d(k, iCell) +=
                -rdnw(k) * (wdtn(k + 1, iCell) - wdtn(k, iCell));
            // Divide by appropriate density
            if (advance_density) {
              scalar_new_2d(k, iCell) /= rho_zz_int(k, iCell);
            } else {
              scalar_new_2d(k, iCell) /= rho_zz_new(k, iCell);
            }
          }
        });
    Kokkos::fence("mono::final_update_fence");

    // ── Copy back to 3D state with positive-definite enforcement (Req 5.7, 5.8) ──
    // For water species: set negatives to zero
    // For regional: specified zone gets the scalar_new result (bdyMask <= nSpecZone)
    const bool is_water = (iScalar >= moist_start && iScalar < moist_end);

    Kokkos::parallel_for(
        "mono::copy_back_3d",
        Kokkos::RangePolicy<exec_space>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          // Specified zone cells (bdyMaskCell <= mono_nSpecZone) get the update
          // Matches Fortran: "if(bdyMaskCell(iCell) <= nSpecZone)"
          if (!config_apply_lbcs || bdyMaskCell(iCell) <= mono_nSpecZone) {
            for (int k = 0; k < nVertLevels; ++k) {
              Scalar val = scalar_new_2d(k, iCell);
              // Positive-definite enforcement (Req 5.7)
              if (is_water) {
                val = Kokkos::fmax(Scalar(0.0), val);
              }
              scalars_new(iScalar, k, iCell) = val;
            }
          }
        });
    Kokkos::fence("mono::copy_back_3d_fence");

  }  // end per-scalar loop
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_SCALAR_TRANSPORT_MONO_HPP
