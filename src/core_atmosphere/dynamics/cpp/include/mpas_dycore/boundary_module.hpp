#ifndef MPAS_DYCORE_BOUNDARY_MODULE_HPP
#define MPAS_DYCORE_BOUNDARY_MODULE_HPP

/// @file boundary_module.hpp
/// @brief Limited-area lateral boundary condition (LBC) management module.
///
/// Ports `mpas_atm_boundaries.F` to C++. Manages two time levels of boundary
/// data, computes boundary tendencies, returns extrapolated states, derives
/// coupled fields (density, edge density, coupled moist theta, mass flux),
/// sets up boundary masks, and applies relaxation-zone adjustments.
///
/// Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7

#include "mpas_dycore/config.hpp"
#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mpas {
namespace dycore {

/// Physical constant: ratio of water vapor to dry air gas constants.
/// rv / rd = 461.6 / 287.0, matching the Reference_Model `rvord`.
inline constexpr Scalar boundary_rvord = Scalar(461.6) / Scalar(287.0);

/// Boundary zone parameters matching the Reference_Model (mpas_atm_boundaries.F).
inline constexpr int nSpecZone = 2;     ///< Number of specified-zone layers
inline constexpr int nRelaxZone = 5;    ///< Number of relaxation-zone layers
inline constexpr int nBdyZone = nSpecZone + nRelaxZone;  ///< Total boundary zone layers

/// @brief Limited-area lateral boundary condition module.
///
/// Stores boundary fields at two time levels (TL1 = old, TL2 = new) and
/// the boundary interval. On first call, reads into TL2. On subsequent
/// calls, shifts TL2->TL1, reads new data into TL2, and computes
/// tendencies = (TL2 - TL1) / interval.
///
/// @tparam ExecSpace  A Kokkos execution space.
template <class ExecSpace>
class Boundary_Module {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using view2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, memory_space>;
  using int_view1d = Kokkos::View<int*, memory_space>;
  using scalar_view1d = Kokkos::View<Scalar*, memory_space>;

  /// Information about a stored boundary field.
  struct FieldInfo {
    view2d tl1;   ///< Time level 1 (old data / tendency after differencing)
    view2d tl2;   ///< Time level 2 (new/latest data)
    int n_inner;  ///< Inner (vertical) dimension
    int n_outer;  ///< Outer (horizontal) dimension
  };

  // ── Construction ──────────────────────────────────────────────────────────

  Boundary_Module() = default;

  /// @brief Register a named boundary field with its dimensions.
  ///
  /// Allocates storage for both time levels. Must be called for each
  /// boundary field before update_boundary_tendency.
  ///
  /// @param name      Field name (e.g. "u", "w", "theta", "rho", etc.)
  /// @param n_inner   Inner (vertical) dimension size
  /// @param n_outer   Outer (horizontal) dimension size
  void register_field(const std::string& name, int n_inner, int n_outer) {
    view2d tl1(std::string("lbc_") + name + "_tl1", n_inner, n_outer);
    view2d tl2(std::string("lbc_") + name + "_tl2", n_inner, n_outer);
    Kokkos::deep_copy(tl1, Scalar{0});
    Kokkos::deep_copy(tl2, Scalar{0});
    fields_[name] = FieldInfo{tl1, tl2, n_inner, n_outer};
  }

  /// @brief Check whether a field is registered.
  bool has_field(const std::string& name) const {
    return fields_.find(name) != fields_.end();
  }

  /// @brief Access time level 2 (latest read data) for loading boundary data.
  ///
  /// Callers use this to fill boundary data from an external source (stream).
  view2d get_tl2(const std::string& name) {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
      throw std::runtime_error(
          "Boundary_Module::get_tl2: unknown field '" + name + "'");
    }
    return it->second.tl2;
  }

  /// @brief Access time level 1 data.
  view2d get_tl1(const std::string& name) {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
      throw std::runtime_error(
          "Boundary_Module::get_tl1: unknown field '" + name + "'");
    }
    return it->second.tl1;
  }

  /// @brief Set the boundary interval in seconds.
  void set_boundary_interval(Scalar interval_seconds) {
    boundary_interval_ = interval_seconds;
  }

  /// @brief Get the boundary interval in seconds.
  Scalar boundary_interval() const { return boundary_interval_; }

  /// @brief Whether the first update has been performed.
  bool first_update_done() const { return first_update_done_; }

  /// @brief Update boundary tendencies (Req 9.1, 9.2, 9.5).
  ///
  /// On first call (first_call=true):
  ///   - Assumes TL2 has been filled with the latest boundary data on/before
  ///     the current time (caller responsible for the read).
  ///   - Derives coupled fields in TL2.
  ///   - Records that the first update is done.
  ///
  /// On subsequent calls (first_call=false):
  ///   - Shifts TL2 -> TL1 for all fields.
  ///   - Assumes TL2 has been filled with the earliest data strictly after
  ///     the current time (caller responsible for the read).
  ///   - Derives coupled fields in TL2.
  ///   - Computes tendencies: TL1 = (TL2 - TL1) / boundary_interval.
  ///
  /// @param cellsOnEdge   Mesh connectivity (2, nEdges), 1-based indices.
  /// @param zz            Height metric (nVertLevels, nCells+1).
  /// @param nCells        Number of cells (includes +1 garbage element).
  /// @param nEdges        Number of edges (includes +1 garbage element).
  /// @param nVertLevels   Number of vertical levels.
  /// @param index_qv      Index of qv in the scalars (0-based in C++).
  /// @param nScalars      Number of scalar species.
  /// @param first_call    True for the first boundary update.
  /// @param interval_seconds  Boundary interval in seconds (used on subsequent calls).
  template <class IntView2D>
  void update_boundary_tendency(
      const IntView2D& cellsOnEdge,
      const view2d& zz,
      int nCells,
      int nEdges,
      int nVertLevels,
      int index_qv,
      int nScalars,
      bool first_call,
      Scalar interval_seconds) {
    if (!first_call) {
      // Shift TL2 -> TL1 for all registered fields (Req 9.2)
      for (auto& [name, info] : fields_) {
        Kokkos::deep_copy(info.tl1, info.tl2);
      }
    }

    // At this point, TL2 should contain the newly-read raw boundary data.
    // Derive coupled fields from the raw fields in TL2 (Req 9.5):
    //   rho_zz = rho / zz
    //   rho_edge = 0.5 * (rho_zz(cell1) + rho_zz(cell2))
    //   rtheta_m = theta * rho_zz * (1 + rvord * qv)
    //   ru = u * rho_edge
    derive_coupled_fields(cellsOnEdge, zz, nCells, nEdges,
                          nVertLevels, index_qv, nScalars);

    if (!first_call) {
      // Compute tendencies: TL1 = (TL2 - TL1) / interval (Req 9.2)
      boundary_interval_ = interval_seconds;
      const Scalar inv_dt = Scalar(1.0) / interval_seconds;

      for (auto& [name, info] : fields_) {
        auto tl1 = info.tl1;
        auto tl2 = info.tl2;
        const int ni = info.n_inner;
        const int no = info.n_outer;

        Kokkos::parallel_for(
            std::string("BdyTend_") + name,
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                {0, 0}, {ni, no}),
            KOKKOS_LAMBDA(const int k, const int j) {
              tl1(k, j) = (tl2(k, j) - tl1(k, j)) * inv_dt;
            });
      }
      Kokkos::fence();
    }

    first_update_done_ = true;
  }

  /// @brief Return the tendency array for a named field (Req 9.3).
  ///
  /// After a non-first update, TL1 contains the tendency (TL2-TL1)/interval.
  /// The tendency is constant over the boundary interval, so delta_t is
  /// accepted for interface compatibility but does not modify the result
  /// (matching the Fortran `mpas_atm_get_bdy_tend` which simply returns
  /// the stored tendency regardless of delta_t).
  ///
  /// @param field    Field name (e.g. "u", "w", "theta", "rho_zz", etc.)
  /// @param delta_t  Future time offset [seconds] (unused for tendency).
  /// @return         View containing the tendency [n_inner x n_outer].
  view2d getTendency(const std::string& field, Scalar /*delta_t*/) const {
    auto it = fields_.find(field);
    if (it == fields_.end()) {
      throw std::runtime_error(
          "Boundary_Module::getTendency: unknown field '" + field + "'");
    }
    // TL1 holds the tendency after update_boundary_tendency(first_call=false)
    return it->second.tl1;
  }

  /// @brief Return the extrapolated state for a named field (Req 9.4).
  ///
  /// Computes: state = TL2 - (remaining_time - delta_t) * tendency
  /// where remaining_time = boundary_interval (time from LBC_intv_start to
  /// LBC_intv_end minus elapsed). This matches the Fortran formula:
  ///   return_state(j,i) = state(j,i) - dt * tend(j,i)
  /// where dt = (LBC_intv_end - currTime) - delta_t.
  ///
  /// For simplicity and consistency with the Fortran, the caller provides
  /// `remaining_time` which is the time from the current clock to the end
  /// of the LBC interval. Then:
  ///   result = TL2 - (remaining_time - delta_t) * TL1
  ///
  /// @param field            Field name.
  /// @param delta_t          Future time offset from current clock [seconds].
  /// @param remaining_time   Time remaining in interval (LBC_end - now) [s].
  /// @param result           Output view [n_inner x n_outer] to fill.
  void getState(const std::string& field,
                Scalar delta_t,
                Scalar remaining_time,
                const view2d& result) const {
    auto it = fields_.find(field);
    if (it == fields_.end()) {
      throw std::runtime_error(
          "Boundary_Module::getState: unknown field '" + field + "'");
    }
    const auto& info = it->second;
    auto state = info.tl2;   // TL2 = state at end of interval
    auto tend = info.tl1;    // TL1 = tendency (after non-first update)
    const int ni = info.n_inner;
    const int no = info.n_outer;

    // dt_offset = remaining_time - delta_t
    // Fortran: return_state = state - dt * tend
    const Scalar dt = remaining_time - delta_t;

    Kokkos::parallel_for(
        std::string("BdyState_") + field,
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {ni, no}),
        KOKKOS_LAMBDA(const int k, const int j) {
          result(k, j) = state(k, j) - dt * tend(k, j);
        });
    Kokkos::fence();
  }

  // ── Boundary Masks (Req 9.6) ─────────────────────────────────────────────

  /// @brief Set up specified-zone masks and nearest relaxation cells.
  ///
  /// Matches `mpas_atm_setup_bdy_masks` in the Reference_Model. For each cell,
  /// edge, and vertex with bdyMask > nRelaxZone, the specZoneMask is set to 1.
  /// For each specified-zone cell, finds the nearest relaxation-zone cell via
  /// the two-hop neighbor search.
  ///
  /// @param bdyMaskCell         Integer mask for cells (nCells).
  /// @param bdyMaskEdge         Integer mask for edges (nEdges).
  /// @param bdyMaskVertex       Integer mask for vertices (nVertices).
  /// @param specZoneMaskCell    Output: 1.0 for specified zone, 0.0 otherwise.
  /// @param specZoneMaskEdge    Output: same for edges.
  /// @param specZoneMaskVertex  Output: same for vertices.
  /// @param nearestRelaxationCell  Output: index of nearest relaxation cell
  ///                               for each cell in the specified zone.
  /// @param nEdgesOnCell        Number of edges/neighbors per cell (nCells).
  /// @param cellsOnCell         Cell-cell connectivity (maxEdges, nCells), 0-based.
  /// @param xCell, yCell, zCell Cell coordinates (nCells).
  /// @param nCells              Number of cells.
  void setup_boundary_masks(
      const int_view1d& bdyMaskCell,
      const int_view1d& bdyMaskEdge,
      const int_view1d& bdyMaskVertex,
      const scalar_view1d& specZoneMaskCell,
      const scalar_view1d& specZoneMaskEdge,
      const scalar_view1d& specZoneMaskVertex,
      const Kokkos::View<int*, memory_space>& nearestRelaxationCell,
      const Kokkos::View<int*, memory_space>& nEdgesOnCell,
      const Kokkos::View<int**, Kokkos::LayoutLeft, memory_space>& cellsOnCell,
      const scalar_view1d& xCell,
      const scalar_view1d& yCell,
      const scalar_view1d& zCell,
      int nCells) {
    // Set specZoneMask: 1.0 where bdyMask > nRelaxZone
    Kokkos::parallel_for(
        "Bdy_specZoneMaskCell",
        Kokkos::RangePolicy<ExecSpace>(0, bdyMaskCell.extent(0)),
        KOKKOS_LAMBDA(const int iCell) {
          specZoneMaskCell(iCell) =
              (bdyMaskCell(iCell) > nRelaxZone) ? Scalar(1.0) : Scalar(0.0);
        });
    Kokkos::parallel_for(
        "Bdy_specZoneMaskEdge",
        Kokkos::RangePolicy<ExecSpace>(0, bdyMaskEdge.extent(0)),
        KOKKOS_LAMBDA(const int iEdge) {
          specZoneMaskEdge(iEdge) =
              (bdyMaskEdge(iEdge) > nRelaxZone) ? Scalar(1.0) : Scalar(0.0);
        });
    Kokkos::parallel_for(
        "Bdy_specZoneMaskVertex",
        Kokkos::RangePolicy<ExecSpace>(0, bdyMaskVertex.extent(0)),
        KOKKOS_LAMBDA(const int iVertex) {
          specZoneMaskVertex(iVertex) =
              (bdyMaskVertex(iVertex) > nRelaxZone) ? Scalar(1.0) : Scalar(0.0);
        });

    Kokkos::fence();

    // Initialize nearestRelaxationCell to nCells (garbage index, matching Fortran nCells+1)
    Kokkos::deep_copy(nearestRelaxationCell, nCells);

    // For inner specified zone cells (bdyMaskCell == nRelaxZone+1):
    // search cellsOnCell neighbors with bdyMaskCell == nRelaxZone
    Kokkos::parallel_for(
        "Bdy_nearestRelax_inner",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          if (bdyMaskCell(iCell) == (nRelaxZone + 1)) {
            Scalar dmin = Scalar(1.0e36);
            int best = nCells;
            const int nNeighbors = nEdgesOnCell(iCell);
            for (int j = 0; j < nNeighbors; ++j) {
              const int i = cellsOnCell(j, iCell);
              if (i >= 0 && i < nCells && bdyMaskCell(i) == nRelaxZone) {
                const Scalar dx = xCell(i) - xCell(iCell);
                const Scalar dy = yCell(i) - yCell(iCell);
                const Scalar dz = zCell(i) - zCell(iCell);
                const Scalar d = dx * dx + dy * dy + dz * dz;
                if (d < dmin) {
                  dmin = d;
                  best = i;
                }
              }
            }
            nearestRelaxationCell(iCell) = best;
          }
        });

    Kokkos::fence();

    // For outer specified zone cells (bdyMaskCell == nRelaxZone+2):
    // search cellsOnCell of cellsOnCell (two-hop) to find nRelaxZone cells
    Kokkos::parallel_for(
        "Bdy_nearestRelax_outer",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          if (bdyMaskCell(iCell) == (nRelaxZone + 2)) {
            Scalar dmin = Scalar(1.0e36);
            int best = nCells;
            const int nNeighbors = nEdgesOnCell(iCell);
            for (int j = 0; j < nNeighbors; ++j) {
              const int i = cellsOnCell(j, iCell);
              if (i >= 0 && i < nCells &&
                  bdyMaskCell(i) == (nRelaxZone + 1)) {
                const int nNeighbors2 = nEdgesOnCell(i);
                for (int jj = 0; jj < nNeighbors2; ++jj) {
                  const int ii = cellsOnCell(jj, i);
                  if (ii >= 0 && ii < nCells &&
                      bdyMaskCell(ii) == nRelaxZone) {
                    const Scalar dx = xCell(ii) - xCell(iCell);
                    const Scalar dy = yCell(ii) - yCell(iCell);
                    const Scalar dz = zCell(ii) - zCell(iCell);
                    const Scalar d = dx * dx + dy * dy + dz * dz;
                    if (d < dmin) {
                      dmin = d;
                      best = ii;
                    }
                  }
                }
              }
            }
            nearestRelaxationCell(iCell) = best;
          }
        });

    Kokkos::fence();
  }

  // ── Relaxation Zone Adjustments (Req 9.7) ────────────────────────────────

  /// @brief Apply Rayleigh relaxation and horizontal-filter adjustments to
  /// density, coupled potential temperature, and momentum tendencies.
  ///
  /// Matches `atm_bdy_adjust_dynamics_relaxzone_tend` in the Reference_Model.
  ///
  /// Rayleigh damping formula (for cells in relaxation zone, bdyMask in [2..nRelaxZone]):
  ///   rayleigh_coef = (bdyMask - 1) / nRelaxZone / (50 * dt * meshScaling)
  ///   tend_rho  -= rayleigh_coef * (rho_zz - rho_driving)
  ///   tend_rt   -= rayleigh_coef * (rho_zz*theta_m - rt_driving)
  ///   tend_ru   -= rayleigh_coef * (ru - ru_driving)
  ///
  /// Horizontal filter formula (dimensionless Laplacian):
  ///   filter_coef = (bdyMask - 1) / nRelaxZone / (10 * dt * meshScaling)
  ///   For each edge of cell: edge_sign * filter_coef * (difference)
  ///
  /// @param tend_rho            Density tendency (nVertLevels, nCells), in/out.
  /// @param tend_rt             Coupled pot. temp. tendency (nVertLevels, nCells), in/out.
  /// @param tend_ru             Momentum tendency (nVertLevels, nEdges), in/out.
  /// @param rho_zz              Current density field (nVertLevels, nCells).
  /// @param theta_m             Current coupled pot. temp. (nVertLevels, nCells).
  /// @param ru                  Current momentum (nVertLevels, nEdges).
  /// @param rho_driving         Driving density values (nVertLevels, nCells).
  /// @param rt_driving          Driving coupled theta values (nVertLevels, nCells).
  /// @param ru_driving          Driving momentum values (nVertLevels, nEdges).
  /// @param bdyMaskCell         Boundary mask for cells.
  /// @param bdyMaskEdge         Boundary mask for edges.
  /// @param meshScalingCell     Regional mesh scaling for cells.
  /// @param meshScalingEdge     Regional mesh scaling for edges.
  /// @param nEdgesOnCell        Number of edges per cell.
  /// @param edgesOnCell         Edge indices for each cell (maxEdges, nCells), 0-based.
  /// @param cellsOnEdge         Cell connectivity per edge (2, nEdges), 0-based.
  /// @param edgesOnCell_sign    Edge sign convention (maxEdges, nCells).
  /// @param dvEdge              Edge dual lengths.
  /// @param invDcEdge           Inverse edge primary lengths.
  /// @param dt                  Model timestep [seconds].
  /// @param nVertLevels         Number of vertical levels.
  /// @param nCellsSolve         Number of cells to process.
  /// @param nEdges              Number of edges to process.
  template <class IntView1D>
  void apply_relaxation_dynamics(
      const view2d& tend_rho,
      const view2d& tend_rt,
      const view2d& tend_ru,
      const view2d& rho_zz,
      const view2d& theta_m,
      const view2d& ru,
      const view2d& rho_driving,
      const view2d& rt_driving,
      const view2d& ru_driving,
      const IntView1D& bdyMaskCell,
      const IntView1D& bdyMaskEdge,
      const scalar_view1d& meshScalingCell,
      const scalar_view1d& meshScalingEdge,
      const IntView1D& nEdgesOnCell_view,
      const Kokkos::View<int**, Kokkos::LayoutLeft, memory_space>& edgesOnCell,
      const Kokkos::View<int**, Kokkos::LayoutLeft, memory_space>& cellsOnEdge,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, memory_space>& edgesOnCell_sign,
      const scalar_view1d& dvEdge,
      const scalar_view1d& invDcEdge,
      Scalar dt,
      int nVertLevels,
      int nCellsSolve,
      int nEdges) {
    const Scalar inv_nRelax = Scalar(1.0) / Scalar(nRelaxZone);
    const Scalar inv_50 = Scalar(1.0) / Scalar(50.0);
    const Scalar inv_10 = Scalar(1.0) / Scalar(10.0);

    // Rayleigh damping for rho_zz and rtheta_m tendencies
    Kokkos::parallel_for(
        "Bdy_relax_rayleigh_cell",
        Kokkos::RangePolicy<ExecSpace>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          const int mask = bdyMaskCell(iCell);
          if (mask > 1 && mask <= nRelaxZone) {
            const Scalar rayleigh_coef =
                (Scalar(mask) - Scalar(1.0)) * inv_nRelax * inv_50 /
                (dt * meshScalingCell(iCell));
            for (int k = 0; k < nVertLevels; ++k) {
              tend_rho(k, iCell) -= rayleigh_coef *
                  (rho_zz(k, iCell) - rho_driving(k, iCell));
              tend_rt(k, iCell) -= rayleigh_coef *
                  (rho_zz(k, iCell) * theta_m(k, iCell) - rt_driving(k, iCell));
            }
          }
        });

    // Rayleigh damping for ru tendency
    Kokkos::parallel_for(
        "Bdy_relax_rayleigh_edge",
        Kokkos::RangePolicy<ExecSpace>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          const int mask = bdyMaskEdge(iEdge);
          if (mask > 1 && mask <= nRelaxZone) {
            const Scalar rayleigh_coef =
                (Scalar(mask) - Scalar(1.0)) * inv_nRelax * inv_50 /
                (dt * meshScalingEdge(iEdge));
            for (int k = 0; k < nVertLevels; ++k) {
              tend_ru(k, iEdge) -= rayleigh_coef *
                  (ru(k, iEdge) - ru_driving(k, iEdge));
            }
          }
        });

    Kokkos::fence();

    // Horizontal filter for rtheta_m and rho_zz tendencies
    Kokkos::parallel_for(
        "Bdy_relax_hfilter_cell",
        Kokkos::RangePolicy<ExecSpace>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          const int mask = bdyMaskCell(iCell);
          if (mask > 1 && mask <= nRelaxZone) {
            const Scalar laplacian_coef =
                (Scalar(mask) - Scalar(1.0)) * inv_nRelax * inv_10 /
                (dt * meshScalingCell(iCell));
            const int nEdg = nEdgesOnCell_view(iCell);
            for (int i = 0; i < nEdg; ++i) {
              const int iEdge = edgesOnCell(i, iCell);
              const Scalar edge_sign =
                  edgesOnCell_sign(i, iCell) * dvEdge(iEdge) *
                  invDcEdge(iEdge) * laplacian_coef;
              const int cell1 = cellsOnEdge(0, iEdge);
              const int cell2 = cellsOnEdge(1, iEdge);
              for (int k = 0; k < nVertLevels; ++k) {
                const Scalar diff_rt =
                    (rho_zz(k, cell2) * theta_m(k, cell2) -
                     rt_driving(k, cell2)) -
                    (rho_zz(k, cell1) * theta_m(k, cell1) -
                     rt_driving(k, cell1));
                const Scalar diff_rho =
                    (rho_zz(k, cell2) - rho_driving(k, cell2)) -
                    (rho_zz(k, cell1) - rho_driving(k, cell1));
                tend_rt(k, iCell) += edge_sign * diff_rt;
                tend_rho(k, iCell) += edge_sign * diff_rho;
              }
            }
          }
        });

    Kokkos::fence();
  }

  /// @brief Apply Rayleigh relaxation and horizontal-filter adjustments to
  /// scalar fields directly (Req 9.7).
  ///
  /// Matches `atm_bdy_adjust_scalars_work` in the Reference_Model.
  /// Scalars are adjusted in-place (state is modified, not tendency):
  ///   laplacian_coef = dt_rk * (bdyMask - 1) / nRelaxZone / (10*dt*meshScaling)
  ///   rayleigh_coef = laplacian_coef / 5.0
  ///
  /// For specified zone cells (bdyMask > nRelaxZone): set to driving values.
  ///
  /// @param scalars             Scalar field (nScalars*nVertLevels, nCells), in/out.
  /// @param scalars_driving     Driving scalar values.
  /// @param bdyMaskCell         Boundary mask for cells.
  /// @param meshScalingCell     Regional mesh scaling for cells.
  /// @param nEdgesOnCell_view   Number of edges per cell.
  /// @param edgesOnCell         Edge indices per cell (maxEdges, nCells), 0-based.
  /// @param cellsOnEdge         Cell connectivity per edge (2, nEdges), 0-based.
  /// @param edgesOnCell_sign    Edge sign convention (maxEdges, nCells).
  /// @param dvEdge              Edge dual lengths.
  /// @param invDcEdge           Inverse edge primary lengths.
  /// @param dt                  Model timestep [seconds].
  /// @param dt_rk               Runge-Kutta substep timestep [seconds].
  /// @param nVertLevels         Number of vertical levels.
  /// @param nScalars            Number of scalar species.
  /// @param nCellsSolve         Number of cells to process.
  template <class IntView1D>
  void apply_relaxation_scalars(
      const view2d& scalars,
      const view2d& scalars_driving,
      const IntView1D& bdyMaskCell,
      const scalar_view1d& meshScalingCell,
      const IntView1D& nEdgesOnCell_view,
      const Kokkos::View<int**, Kokkos::LayoutLeft, memory_space>& edgesOnCell,
      const Kokkos::View<int**, Kokkos::LayoutLeft, memory_space>& cellsOnEdge,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, memory_space>& edgesOnCell_sign,
      const scalar_view1d& dvEdge,
      const scalar_view1d& invDcEdge,
      Scalar dt,
      Scalar dt_rk,
      int nVertLevels,
      int nScalars,
      int nCellsSolve) {
    // Create temporary storage for the adjusted scalars
    // scalars layout: (nScalars*nVertLevels, nCells)
    const int nInner = nScalars * nVertLevels;
    view2d scalars_tmp("scalars_tmp", nInner, nCellsSolve);

    const Scalar inv_nRelax = Scalar(1.0) / Scalar(nRelaxZone);
    const Scalar inv_10 = Scalar(1.0) / Scalar(10.0);
    const Scalar inv_5 = Scalar(1.0) / Scalar(5.0);

    // First pass: compute adjusted values into temporary
    Kokkos::parallel_for(
        "Bdy_relax_scalars_compute",
        Kokkos::RangePolicy<ExecSpace>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          const int mask = bdyMaskCell(iCell);
          if (mask > 1 && mask <= nRelaxZone) {
            // Relaxation zone
            const Scalar laplacian_coef =
                dt_rk * (Scalar(mask) - Scalar(1.0)) * inv_nRelax * inv_10 /
                (dt * meshScalingCell(iCell));
            const Scalar rayleigh_coef = laplacian_coef * inv_5;

            // Copy current values
            for (int idx = 0; idx < nInner; ++idx) {
              scalars_tmp(idx, iCell) = scalars(idx, iCell);
            }

            // Horizontal filter
            const int nEdg = nEdgesOnCell_view(iCell);
            for (int i = 0; i < nEdg; ++i) {
              const int iEdge = edgesOnCell(i, iCell);
              const Scalar edge_sign =
                  edgesOnCell_sign(i, iCell) * dvEdge(iEdge) *
                  invDcEdge(iEdge) * laplacian_coef;
              const int cell1 = cellsOnEdge(0, iEdge);
              const int cell2 = cellsOnEdge(1, iEdge);
              for (int idx = 0; idx < nInner; ++idx) {
                const Scalar filter_flux =
                    edge_sign *
                    ((scalars(idx, cell2) - scalars_driving(idx, cell2)) -
                     (scalars(idx, cell1) - scalars_driving(idx, cell1)));
                scalars_tmp(idx, iCell) += filter_flux;
              }
            }

            // Rayleigh damping
            for (int idx = 0; idx < nInner; ++idx) {
              scalars_tmp(idx, iCell) -= rayleigh_coef *
                  (scalars(idx, iCell) - scalars_driving(idx, iCell));
            }
          } else if (mask > nRelaxZone) {
            // Specified zone: set to driving values
            for (int idx = 0; idx < nInner; ++idx) {
              scalars_tmp(idx, iCell) = scalars_driving(idx, iCell);
            }
          } else {
            // Interior: copy unchanged
            for (int idx = 0; idx < nInner; ++idx) {
              scalars_tmp(idx, iCell) = scalars(idx, iCell);
            }
          }
        });

    Kokkos::fence();

    // Second pass: copy adjusted values back for boundary cells
    Kokkos::parallel_for(
        "Bdy_relax_scalars_writeback",
        Kokkos::RangePolicy<ExecSpace>(0, nCellsSolve),
        KOKKOS_LAMBDA(const int iCell) {
          if (bdyMaskCell(iCell) > 1) {
            for (int idx = 0; idx < nInner; ++idx) {
              scalars(idx, iCell) = scalars_tmp(idx, iCell);
            }
          }
        });

    Kokkos::fence();
  }

  // ── Regional Configuration Validation (Req 9.8, 9.9, 9.10) ────────────────

  /// @brief Validate regional configuration consistency.
  ///
  /// Checks for configuration errors:
  /// - Req 9.8: Regional mode active (config_apply_lbcs) but boundary-mask
  ///   field contains no boundary cells (all values are zero).
  /// - Req 9.9: Regional mode inactive (!config_apply_lbcs) but boundary-mask
  ///   field contains boundary cells (some values > 0).
  /// - Req 9.10: Regional mode active (config_apply_lbcs) but the boundary
  ///   input interval is not valid (<= 0).
  ///
  /// @param config              The configuration snapshot.
  /// @param bdyMaskCell         Boundary mask for cells.
  /// @param nCells              Number of cells to inspect.
  /// @param boundary_input_interval  Boundary input interval in seconds.
  ///
  /// @throws std::runtime_error with a descriptive message on configuration error.
  void validate_regional_config(
      const Config& config,
      const int_view1d& bdyMaskCell,
      int nCells,
      Scalar boundary_input_interval) const {
    // Check whether any boundary cells exist in the mask.
    // A boundary cell has bdyMaskCell > 0.
    int boundary_cell_count = 0;
    Kokkos::parallel_reduce(
        "Bdy_validate_has_boundary_cells",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell, int& lsum) {
          if (bdyMaskCell(iCell) > 0) {
            lsum += 1;
          }
        },
        boundary_cell_count);
    Kokkos::fence();

    const bool has_boundary_cells = (boundary_cell_count > 0);

    // Req 9.8: Regional mode active but no boundary cells
    if (config.config_apply_lbcs && !has_boundary_cells) {
      throw std::runtime_error(
          "validate_regional_config: Regional mode is active "
          "(config_apply_lbcs=true) but the boundary-mask field contains "
          "no boundary cells");
    }

    // Req 9.9: Regional mode inactive but boundary cells present
    if (!config.config_apply_lbcs && has_boundary_cells) {
      throw std::runtime_error(
          "validate_regional_config: Regional mode is inactive "
          "(config_apply_lbcs=false) but the boundary-mask field contains "
          "boundary cells");
    }

    // Req 9.10: Regional mode active but no valid boundary input interval
    if (config.config_apply_lbcs && boundary_input_interval <= Scalar{0}) {
      throw std::runtime_error(
          "validate_regional_config: Regional mode is active "
          "(config_apply_lbcs=true) but the boundary input interval is "
          "not valid (<=0)");
    }
  }

 private:
  std::unordered_map<std::string, FieldInfo> fields_;
  Scalar boundary_interval_ = Scalar{0};
  bool first_update_done_ = false;

  /// @brief Derive coupled fields in TL2 from raw boundary data (Req 9.5).
  ///
  /// Matches the Fortran derivation in mpas_atm_update_bdy_tend:
  ///   rho_zz(k,iCell) = rho(k,iCell) / zz(k,iCell)
  ///   rho_edge(k,iEdge) = 0.5 * (rho_zz(k,cell1) + rho_zz(k,cell2))
  ///   rtheta_m(k,iCell) = theta(k,iCell) * rho_zz(k,iCell) *
  ///                        (1 + rvord * scalars(index_qv, k, iCell))
  ///   ru(k,iEdge) = u(k,iEdge) * rho_edge(k,iEdge)
  ///
  /// Assumes "rho", "theta", "u", "scalars" are registered and TL2 is filled.
  /// Derives into registered fields "rho_zz", "rho_edge", "rtheta_m", "ru".
  template <class IntView2D>
  void derive_coupled_fields(
      const IntView2D& cellsOnEdge,
      const view2d& zz,
      int nCells,
      int nEdges,
      int nVertLevels,
      int index_qv,
      int /*nScalars*/) {
    // Get the TL2 views for raw and derived fields
    auto rho = fields_.at("rho").tl2;
    auto theta = fields_.at("theta").tl2;
    auto u_field = fields_.at("u").tl2;
    auto scalars = fields_.at("scalars").tl2;
    auto rho_zz = fields_.at("rho_zz").tl2;
    auto rho_edge = fields_.at("rho_edge").tl2;
    auto rtheta_m = fields_.at("rtheta_m").tl2;
    auto ru = fields_.at("ru").tl2;

    const int nCellsP1 = nCells + 1;  // includes garbage element

    // Compute rho_zz = rho / zz (over nCells+1 including garbage)
    Kokkos::parallel_for(
        "Bdy_derive_rho_zz",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nCellsP1}),
        KOKKOS_LAMBDA(const int k, const int iCell) {
          const Scalar zz_val = zz(k, iCell);
          // Avoid division by zero on garbage cell (Fortran sets zz=1 there)
          const Scalar safe_zz = (zz_val == Scalar{0}) ? Scalar{1} : zz_val;
          rho_zz(k, iCell) = rho(k, iCell) / safe_zz;
        });

    Kokkos::fence();

    // Compute rho_edge = 0.5 * (rho_zz(cell1) + rho_zz(cell2))
    // cellsOnEdge uses 1-based Fortran indices; convert to 0-based.
    Kokkos::parallel_for(
        "Bdy_derive_rho_edge",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nEdges}),
        KOKKOS_LAMBDA(const int k, const int iEdge) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;  // 1-based to 0-based
          const int cell2 = cellsOnEdge(1, iEdge) - 1;
          if (cell1 >= 0 && cell2 >= 0) {
            rho_edge(k, iEdge) =
                Scalar(0.5) * (rho_zz(k, cell1) + rho_zz(k, cell2));
          }
        });

    Kokkos::fence();

    // Compute ru = u * rho_edge (over nEdges+1 including garbage)
    const int nEdgesP1 = nEdges + 1;
    Kokkos::parallel_for(
        "Bdy_derive_ru",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nEdgesP1}),
        KOKKOS_LAMBDA(const int k, const int iEdge) {
          ru(k, iEdge) = u_field(k, iEdge) * rho_edge(k, iEdge);
        });

    // Compute rtheta_m = theta * rho_zz * (1 + rvord * scalars(qv))
    // scalars is stored as (nScalars*nVertLevels, nCells) where the index
    // for species s at level k is (s*nVertLevels + k). For a single-species
    // approach, or when scalars is (nVertLevels, nCells) for qv only,
    // we use the index_qv offset.
    const Scalar rvord_val = boundary_rvord;
    const int qv_offset = index_qv * nVertLevels;
    Kokkos::parallel_for(
        "Bdy_derive_rtheta_m",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nCellsP1}),
        KOKKOS_LAMBDA(const int k, const int iCell) {
          const Scalar qv = scalars(qv_offset + k, iCell);
          rtheta_m(k, iCell) = theta(k, iCell) * rho_zz(k, iCell) *
                               (Scalar(1.0) + rvord_val * qv);
        });

    Kokkos::fence();
  }
};

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_BOUNDARY_MODULE_HPP
