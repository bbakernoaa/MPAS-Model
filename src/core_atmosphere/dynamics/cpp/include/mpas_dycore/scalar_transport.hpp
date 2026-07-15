#ifndef MPAS_DYCORE_SCALAR_TRANSPORT_HPP
#define MPAS_DYCORE_SCALAR_TRANSPORT_HPP

/// @file scalar_transport.hpp
/// @brief Scalar transport module for the C++ dycore.
///
/// Ports `atm_advance_scalars` / `atm_advance_scalars_work` from the Fortran
/// Reference_Model to C++ using Kokkos.
///
/// Requirements: 5.1, 5.3, 5.4, 5.5, 2.6

#include "mpas_dycore/scalar.hpp"
#include "mpas_dycore/config.hpp"
#include "mpas_dycore/accumulation.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

namespace mpas {
namespace dycore {

/// Boundary zone constant matching the Reference_Model.
inline constexpr int transport_nRelaxZone = 5;

/// Mesh data required by the Scalar_Transport module.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct ScalarTransportMeshData {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  template <class T> using View1D = Kokkos::View<T*, layout, memory_space>;
  template <class T> using View2D = Kokkos::View<T**, layout, memory_space>;

  int nCells = 0;
  int nEdges = 0;
  int nVertLevels = 0;
  int num_scalars = 0;
  int maxEdges = 0;
  int maxAdvCellsForEdge = 0;

  View2D<int> cellsOnEdge;         ///< (2, nEdges) - 1-based
  View2D<int> edgesOnCell;         ///< (maxEdges, nCells) - 1-based
  View1D<int> nEdgesOnCell;        ///< (nCells)
  View2D<int> advCellsForEdge;     ///< (maxAdvCellsForEdge, nEdges) - 1-based
  View1D<int> nAdvCellsForEdge;    ///< (nEdges)

  View2D<Scalar> adv_coefs;        ///< (maxAdvCellsForEdge, nEdges)
  View2D<Scalar> adv_coefs_3rd;    ///< (maxAdvCellsForEdge, nEdges)
  View2D<Scalar> edgesOnCell_sign; ///< (maxEdges, nCells)

  View1D<Scalar> dvEdge;           ///< (nEdges)
  View1D<Scalar> invAreaCell;      ///< (nCells)

  View1D<Scalar> fnm;              ///< (nVertLevels+1) vert interp weight
  View1D<Scalar> fnp;              ///< (nVertLevels+1) vert interp weight
  View1D<Scalar> rdnw;             ///< (nVertLevels) 1/dz

  View1D<int> bdyMaskCell;         ///< (nCells)
  View1D<int> bdyMaskEdge;         ///< (nEdges)
};

/// Input/output state fields for scalar transport.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct ScalarTransportState {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  template <class T> using View2D = Kokkos::View<T**, layout, memory_space>;
  template <class T> using View3D = Kokkos::View<T***, layout, memory_space>;

  View3D<Scalar> scalar_old;   ///< (num_scalars, nVertLevels, nCells)
  View3D<Scalar> scalar_new;   ///< (num_scalars, nVertLevels, nCells)
  View2D<Scalar> rho_zz_old;   ///< (nVertLevels, nCells)
  View2D<Scalar> rho_zz_new;   ///< (nVertLevels, nCells)
  View2D<Scalar> uhAvg;        ///< (nVertLevels, nEdges)
  View2D<Scalar> wwAvg;        ///< (nVertLevels+1, nCells)
  View3D<Scalar> scalar_tend;  ///< (num_scalars, nVertLevels, nCells)
};

/// @brief Scalar_Transport: standard (non-limited) transport module.
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
class Scalar_Transport {
 public:
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  using view3d = Kokkos::View<Scalar***, layout, memory_space>;

  Scalar_Transport() = default;

  /// @brief Advance scalars using the standard (non-limited) transport.
  ///
  /// Equivalent to `atm_advance_scalars_work` in the Reference_Model.
  void advance_scalars(
      const ScalarTransportMeshData<ExecSpace>& mesh,
      ScalarTransportState<ExecSpace>& state,
      Scalar dt,
      Scalar coef_3rd_order,
      int rk_step,
      int config_time_integration_order,
      bool advance_density,
      bool config_apply_lbcs) const;
};

// ============================================================================
// Implementation
// ============================================================================

template <class ExecSpace>
void Scalar_Transport<ExecSpace>::advance_scalars(
    const ScalarTransportMeshData<ExecSpace>& mesh,
    ScalarTransportState<ExecSpace>& state,
    Scalar dt,
    Scalar coef_3rd_order,
    int rk_step,
    int config_time_integration_order,
    bool advance_density,
    bool config_apply_lbcs) const {

  const int nCells = mesh.nCells;
  const int nEdges = mesh.nEdges;
  const int nVertLevels = mesh.nVertLevels;
  const int num_scalars = mesh.num_scalars;

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

  const auto scalar_old = state.scalar_old;
  auto scalar_new = state.scalar_new;
  const auto rho_zz_old = state.rho_zz_old;
  const auto rho_zz_new = state.rho_zz_new;
  const auto uhAvg = state.uhAvg;
  const auto wwAvg = state.wwAvg;
  const auto scalar_tend = state.scalar_tend;

  // ── Density re-integration weights (Req 5.5) ──
  Scalar weight_time_new;
  if (!advance_density) {
    weight_time_new = Scalar(1.0);
  } else {
    if (rk_step == 1 && config_time_integration_order == 3)
      weight_time_new = Scalar(1.0) / Scalar(3.0);
    else if (rk_step == 1 && config_time_integration_order == 2)
      weight_time_new = Scalar(1.0) / Scalar(2.0);
    else if (rk_step == 2)
      weight_time_new = Scalar(1.0) / Scalar(2.0);
    else
      weight_time_new = Scalar(1.0);
  }
  const Scalar wt_old = Scalar(1.0) - weight_time_new;
  const Scalar wt_new = weight_time_new;

  // ── Step 1: Compute horizontal edge scalar values (Req 5.3) ──
  // horiz_flux stores the scalar value at each edge, computed from the
  // third-order upwind-biased advection stencil.
  view3d horiz_flux("horiz_flux", num_scalars, nVertLevels, nEdges);

  Kokkos::parallel_for(
      "scalar_transport::compute_horiz_flux",
      Kokkos::RangePolicy<exec_space>(0, nEdges),
      KOKKOS_LAMBDA(const int iEdge) {
        const bool use_upwind = config_apply_lbcs &&
            (bdyMaskEdge(iEdge) >= transport_nRelaxZone - 1) &&
            (bdyMaskEdge(iEdge) <= transport_nRelaxZone);
        const bool skip_edge = config_apply_lbcs &&
            (bdyMaskEdge(iEdge) > transport_nRelaxZone);

        if (skip_edge) {
          for (int k = 0; k < nVertLevels; ++k)
            for (int s = 0; s < num_scalars; ++s)
              horiz_flux(s, k, iEdge) = Scalar(0.0);
          return;
        }

        if (use_upwind) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;
          for (int k = 0; k < nVertLevels; ++k) {
            const Scalar u_dir = (uhAvg(k, iEdge) >= Scalar(0.0))
                ? Scalar(0.5) : Scalar(-0.5);
            const Scalar u_pos = dvEdge(iEdge) *
                Kokkos::fabs(u_dir + Scalar(0.5));
            const Scalar u_neg = dvEdge(iEdge) *
                Kokkos::fabs(u_dir - Scalar(0.5));
            for (int s = 0; s < num_scalars; ++s) {
              horiz_flux(s, k, iEdge) =
                  u_pos * scalar_new(s, k, cell1) +
                  u_neg * scalar_new(s, k, cell2);
            }
          }
          return;
        }

        // Full high-order flux: third-order upwind-biased
        const int nAdv = nAdvCellsForEdge_v(iEdge);
        for (int k = 0; k < nVertLevels; ++k) {
          const Scalar sign_u = (uhAvg(k, iEdge) >= Scalar(0.0))
              ? Scalar(1.0) : Scalar(-1.0);
          for (int s = 0; s < num_scalars; ++s) {
            Scalar flux_val = Scalar(0.0);
            for (int j = 0; j < nAdv; ++j) {
              const int iAdvCell = advCellsForEdge(j, iEdge) - 1;
              const Scalar sw = adv_coefs(j, iEdge) +
                  sign_u * adv_coefs_3rd(j, iEdge);
              flux_val += sw * scalar_new(s, k, iAdvCell);
            }
            horiz_flux(s, k, iEdge) = flux_val;
          }
        }
      });
  Kokkos::fence("scalar_transport::horiz_flux_fence");

  // ── Step 2: Compute vertical fluxes (Req 5.4) ──
  // wdtn(s, k, iCell) = vertical flux at interface k for scalar s.
  // This must be computed BEFORE the scalar update because it reads the
  // input scalar_new values. Allocate a scratch buffer for the fluxes.
  //
  // Interface indexing (0-based):
  //   k=0: top boundary (flux=0)
  //   k=1: linear interpolation
  //   k=2..nVertLevels-2: third-order flux3
  //   k=nVertLevels-1: linear interpolation
  //   k=nVertLevels: bottom boundary (flux=0)
  view3d wdtn("wdtn", num_scalars, nVertLevels + 1, nCells);

  Kokkos::parallel_for(
      "scalar_transport::compute_vert_flux",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        if (config_apply_lbcs && bdyMaskCell(iCell) > transport_nRelaxZone)
          return;

        for (int s = 0; s < num_scalars; ++s) {
          // Boundary fluxes = 0
          wdtn(s, 0, iCell) = Scalar(0.0);
          wdtn(s, nVertLevels, iCell) = Scalar(0.0);

          // Interface k=1: linear interpolation
          if (nVertLevels > 1) {
            wdtn(s, 1, iCell) = wwAvg(1, iCell) *
                (fnm(1) * scalar_new(s, 1, iCell) +
                 fnp(1) * scalar_new(s, 0, iCell));
          }

          // Interface k=nVertLevels-1: linear interpolation
          if (nVertLevels > 2) {
            const int kk = nVertLevels - 1;
            wdtn(s, kk, iCell) = wwAvg(kk, iCell) *
                (fnm(kk) * scalar_new(s, kk, iCell) +
                 fnp(kk) * scalar_new(s, kk - 1, iCell));
          }

          // Interior interfaces k=2..nVertLevels-2: flux3
          for (int k = 2; k <= nVertLevels - 2; ++k) {
            const Scalar q_im2 = scalar_new(s, k - 2, iCell);
            const Scalar q_im1 = scalar_new(s, k - 1, iCell);
            const Scalar q_i   = scalar_new(s, k, iCell);
            const Scalar q_ip1 = scalar_new(s, k + 1, iCell);
            const Scalar w = wwAvg(k, iCell);

            // flux4 = w*(7*(q_i + q_im1) - (q_ip1 + q_im2))/12
            const Scalar flux4 = w *
                (Scalar(7.0) * (q_i + q_im1) - (q_ip1 + q_im2)) /
                Scalar(12.0);
            // flux3 = flux4 + coef3*|w|*((q_ip1-q_im2)-3*(q_i-q_im1))/12
            wdtn(s, k, iCell) = flux4 +
                coef_3rd_order * Kokkos::fabs(w) *
                ((q_ip1 - q_im2) - Scalar(3.0) * (q_i - q_im1)) /
                Scalar(12.0);
          }
        }
      });
  Kokkos::fence("scalar_transport::vert_flux_fence");

  // ── Step 3: Cell update (Req 2.6, 5.5) ──
  // Gather horizontal flux divergence, combine with physics tendency and
  // vertical flux divergence, and write the final scalar update.
  //
  // Cell-parallel decomposition is inherently race-free for the write to
  // scalar_new (each thread writes only its own cell). The edge gather
  // within a cell has a fixed order determined by mesh connectivity,
  // ensuring deterministic accumulation (Req 2.6).
  Kokkos::parallel_for(
      "scalar_transport::cell_update",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        if (config_apply_lbcs && bdyMaskCell(iCell) > transport_nRelaxZone)
          return;

        const int ne = nEdgesOnCell_v(iCell);

        for (int k = 0; k < nVertLevels; ++k) {
          for (int s = 0; s < num_scalars; ++s) {
            // Horizontal flux divergence
            Scalar h_tend = Scalar(0.0);
            for (int i = 0; i < ne; ++i) {
              const int iEdge = edgesOnCell(i, iCell) - 1;
              h_tend -= edgesOnCell_sign(i, iCell) *
                  uhAvg(k, iEdge) * horiz_flux(s, k, iEdge);
            }
            h_tend *= invAreaCell(iCell);

            // Add physics/source tendency
            h_tend += scalar_tend(s, k, iCell);

            // Vertical flux divergence
            const Scalar v_div = rdnw(k) *
                (wdtn(s, k + 1, iCell) - wdtn(s, k, iCell));

            // Final scalar update (Req 5.5):
            // scalar_new = (scalar_old*rho_old + dt*(h_tend - v_div))
            //            / (wt_old*rho_old + wt_new*rho_new)
            const Scalar rho_inv = Scalar(1.0) /
                (wt_old * rho_zz_old(k, iCell) +
                 wt_new * rho_zz_new(k, iCell));
            scalar_new(s, k, iCell) =
                (scalar_old(s, k, iCell) * rho_zz_old(k, iCell) +
                 dt * (h_tend - v_div)) * rho_inv;
          }
        }
      });
  Kokkos::fence("scalar_transport::cell_update_fence");
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_SCALAR_TRANSPORT_HPP
