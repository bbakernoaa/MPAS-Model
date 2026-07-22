#ifndef MPAS_DYCORE_DYN_TEND_MODULE_HPP
#define MPAS_DYCORE_DYN_TEND_MODULE_HPP

/// @file dyn_tend_module.hpp
/// @brief Coupled dynamic tendency computation for the C++ dycore.
///
/// Implements `compute_dyn_tend` equivalent to `atm_compute_dyn_tend_work`
/// in the Reference_Model (Requirements 6.1-6.7, 2.1, 2.6).

#include "mpas_dycore/config.hpp"
#include "mpas_dycore/dissipation_module.hpp"
#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <mpi.h>

// Debug tracing hook, disabled for production (previously appended to a
// hard-coded per-rank log file on every call). Kept as a no-op so existing
// call sites compile unchanged.
inline void dyn_debug_log(const char* /*format*/, ...) {}

namespace mpas {
namespace dycore {

/// Parameters controlling the dynamic tendency computation.
struct DynTendParams {
  int nCells = 0;
  int nEdges = 0;
  int nVertices = 0;
  int nVertLevels = 0;
  int maxEdges = 0;
  int maxEdges2 = 0;
  int vertexDegree = 3;

  Scalar dt = 0.0;
  Scalar coef_3rd_order = 1.0;
  Scalar r_earth = 6371229.0;
  Scalar inv_r_earth = 0.0;

  // Rayleigh damping
  bool config_rayleigh_damp_u = false;
  Scalar config_rayleigh_damp_u_timescale_days = 10.0;
  int config_number_rayleigh_damp_u_levels = 0;

  // Curvature terms
  bool curvature_enabled = false;
  Scalar omega = 7.29212e-5;  // Earth's angular velocity

  // Horizontal mixing / dissipation (2d Smagorinsky del2 + del4 background filter)
  bool mixing_enabled = false;         // whether to compute horizontal mixing (rk_step==1)
  Scalar c_s = 0.125;                  // config_smagorinsky_coef
  Scalar config_len_disp = 0.0;        // horizontal filter length scale (=nominalMinDc) [m]
  Scalar config_visc4_2dsmag = 0.05;   // del4 background coefficient
  Scalar config_del4u_div_factor = 10.0; // scaling of divergent part of del4 u filter
  Scalar prandtl_inv = 1.0;            // 1/Prandtl for theta/scalar filtering
};

/// Mesh connectivity and geometry required by Dyn_Tend_Module.
/// Extends the base MeshData with additional arrays needed for
/// tendency computations.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct DynTendMeshData {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;

  template <class T>
  using View1D = Kokkos::View<T*, layout, memory_space>;
  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;
  template <class T>
  using View3D = Kokkos::View<T***, layout, memory_space>;

  // Connectivity (integer arrays, 1-based Fortran indices)
  View2D<int> cellsOnEdge;       // (2, nEdges)
  View2D<int> verticesOnEdge;    // (2, nEdges)
  View2D<int> edgesOnCell;       // (maxEdges, nCells)
  View2D<int> edgesOnEdge;       // (maxEdges2, nEdges)
  View2D<int> edgesOnVertex;     // (vertexDegree, nVertices)
  View1D<int> nEdgesOnCell;      // (nCells)
  View1D<int> nEdgesOnEdge;      // (nEdges)
  View2D<int> advCellsForEdge;   // (15, nEdges)
  View1D<int> nAdvCellsForEdge;  // (nEdges)

  // Geometry (real-valued)
  View1D<Scalar> dvEdge;          // (nEdges)
  View1D<Scalar> dcEdge;          // (nEdges)
  View1D<Scalar> invDcEdge;       // (nEdges)
  View1D<Scalar> invDvEdge;       // (nEdges)
  View1D<Scalar> invAreaCell;     // (nCells)
  View1D<Scalar> fEdge;           // (nEdges)
  View2D<Scalar> weightsOnEdge;   // (maxEdges2, nEdges)
  View2D<Scalar> edgesOnCell_sign;  // (maxEdges, nCells)
  View2D<Scalar> zgrid;           // (nVertLevels+1, nCells)
  View2D<Scalar> zz;              // (nVertLevels, nCells)
  View2D<Scalar> zxu;             // (nVertLevels, nEdges)
  View2D<Scalar> cqu;             // (nVertLevels, nEdges)
  View2D<Scalar> cqw;             // (nVertLevels, nCells)
  View1D<Scalar> rdzu;            // (nVertLevels)
  View1D<Scalar> rdzw;            // (nVertLevels)
  View1D<Scalar> fzm;             // (nVertLevels)
  View1D<Scalar> fzp;             // (nVertLevels)
  View2D<Scalar> adv_coefs;       // (15, nEdges)
  View2D<Scalar> adv_coefs_3rd;   // (15, nEdges)

  // Curvature-related
  View1D<Scalar> latCell;         // (nCells)
  View1D<Scalar> latEdge;         // (nEdges)
  View1D<Scalar> angleEdge;       // (nEdges)
  View1D<Scalar> u_init;          // (nVertLevels)
  View1D<Scalar> v_init;          // (nVertLevels)

  // Dissipation / horizontal-mixing related (2d Smagorinsky del2 + del4)
  // (verticesOnEdge is already declared above in the connectivity section)
  View2D<Scalar> edgesOnVertex_sign; // (vertexDegree, nVertices)
  View1D<Scalar> invAreaTriangle;   // (nVertices)
  View1D<Scalar> meshScalingDel2;   // (nEdges)
  View1D<Scalar> meshScalingDel4;   // (nEdges)
  View2D<Scalar> deformation_coef_c2; // (maxEdges, nCells)
  View2D<Scalar> deformation_coef_s2; // (maxEdges, nCells)
  View2D<Scalar> deformation_coef_cs; // (maxEdges, nCells)
};

/// Input state fields required by Dyn_Tend_Module.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct DynTendState {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;

  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;

  View2D<Scalar> u;             // (nVertLevels, nEdges)
  View2D<Scalar> v;             // (nVertLevels, nEdges) - tangential velocity
  View2D<Scalar> w;             // (nVertLevels+1, nCells)
  View2D<Scalar> theta_m;       // (nVertLevels, nCells)
  View2D<Scalar> rho_zz;        // (nVertLevels, nCells)
  View2D<Scalar> rho_edge;      // (nVertLevels, nEdges)
  View2D<Scalar> ru;            // (nVertLevels, nEdges) - horizontal mass flux
  View2D<Scalar> rw;            // (nVertLevels+1, nCells) - vertical mass flux
  View2D<Scalar> ke;            // (nVertLevels, nCells) - kinetic energy
  View2D<Scalar> pv_edge;       // (nVertLevels, nEdges) - potential vorticity
  View2D<Scalar> divergence;    // (nVertLevels, nCells)
  View2D<Scalar> vorticity;     // (nVertLevels, nVertices) - relative vorticity (for u mixing)
  View2D<Scalar> pp;            // (nVertLevels, nCells) - pressure perturbation
  View2D<Scalar> rb;            // (nVertLevels, nCells) - base-state density
  View2D<Scalar> rr;            // (nVertLevels, nCells) - density perturbation
  View2D<Scalar> rr_save;       // (nVertLevels, nCells) - saved density pert
  View2D<Scalar> qtot;          // (nVertLevels, nCells) - total water mixing ratio
  View2D<Scalar> exner;         // (nVertLevels, nCells) - Exner function
  View2D<Scalar> pressure_b;    // (nVertLevels, nCells) - base pressure
  View2D<Scalar> rt_diabatic_tend; // (nVertLevels, nCells)
  View2D<Scalar> theta_m_save;  // (nVertLevels, nCells) - saved theta_m
  View2D<Scalar> ru_save;       // (nVertLevels, nEdges) - saved ru
  View2D<Scalar> rw_save;       // (nVertLevels+1, nCells) - saved rw

  // Cell-centered reconstructed velocities (for curvature terms)
  View2D<Scalar> ur_cell;       // (nVertLevels, nCells)
  View2D<Scalar> vr_cell;       // (nVertLevels, nCells)
};

/// Output tendency fields produced by Dyn_Tend_Module.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct DynTendOutput {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;

  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;

  View2D<Scalar> tend_u;         // (nVertLevels, nEdges)
  View2D<Scalar> tend_w;         // (nVertLevels+1, nCells)
  View2D<Scalar> tend_theta;     // (nVertLevels, nCells)
  View2D<Scalar> tend_rho;       // (nVertLevels, nCells)
  View2D<Scalar> h_divergence;   // (nVertLevels, nCells)

  // Euler (mixing) tendencies cached on stage 1 for reuse
  View2D<Scalar> tend_u_euler;   // (nVertLevels, nEdges)
  View2D<Scalar> tend_w_euler;   // (nVertLevels+1, nCells)
  View2D<Scalar> tend_theta_euler; // (nVertLevels, nCells)
};

/// Physics tendencies to be added to dynamic tendencies (Requirement 6.7).
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct PhysTendencies {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;

  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;

  View2D<Scalar> tend_ru_physics;     // (nVertLevels, nEdges)
  View2D<Scalar> tend_rho_physics;    // (nVertLevels, nCells)
  View2D<Scalar> tend_rtheta_physics; // (nVertLevels, nCells)
};

/// @brief Dyn_Tend_Module: coupled dynamic tendency computation.
///
/// Computes coupled tendencies for horizontal momentum (u), vertical
/// momentum (w), coupled potential temperature (theta_m), and density (rho).
/// Equivalent to `atm_compute_dyn_tend` in the Reference_Model.
///
/// @tparam ExecSpace Kokkos execution space for all parallel kernels.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
class Dyn_Tend_Module {
 public:
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  using view2d = Kokkos::View<Scalar**, layout, memory_space>;

  Dyn_Tend_Module() = default;

  /// @brief Compute coupled dynamic tendencies (Requirement 6.1-6.7).
  ///
  /// On rk_step==1: computes and caches mixing tendencies (via
  /// Dissipation_Module) for reuse in subsequent RK stages (Req 6.5).
  /// Computes mass-flux divergence (Req 6.2), nonlinear Coriolis (Req 6.3),
  /// curvature terms when enabled (Req 6.4), Rayleigh damping (Req 6.6),
  /// and adds physics tendencies (Req 6.7).
  ///
  /// @param mesh    Extended mesh connectivity/geometry.
  /// @param state   Input prognostic/diagnostic state fields.
  /// @param output  Output tendency fields (modified in place).
  /// @param phys    Physics tendencies to add (Req 6.7).
  /// @param params  Dynamic tendency parameters.
  /// @param rk_step Current RK step (1, 2, or 3).
  void compute_dyn_tend(
      const DynTendMeshData<ExecSpace>& mesh,
      const DynTendState<ExecSpace>& state,
      DynTendOutput<ExecSpace>& output,
      const PhysTendencies<ExecSpace>& phys,
      const DynTendParams& params,
      int rk_step) const;
};

// ============================================================================
// Implementation (header-only, required for templates)
// ============================================================================

namespace detail {

/// Third-order upwind-biased flux (matching Reference_Model flux3 statement fn)
KOKKOS_INLINE_FUNCTION
Scalar flux4(Scalar q_im2, Scalar q_im1, Scalar q_i, Scalar q_ip1, Scalar ua) {
  return ua * (Scalar(7.0) * (q_i + q_im1) - (q_ip1 + q_im2)) / Scalar(12.0);
}

KOKKOS_INLINE_FUNCTION
Scalar flux3(Scalar q_im2, Scalar q_im1, Scalar q_i, Scalar q_ip1,
             Scalar ua, Scalar coef3) {
  return flux4(q_im2, q_im1, q_i, q_ip1, ua) +
         coef3 * Kokkos::fabs(ua) *
             ((q_ip1 - q_im2) - Scalar(3.0) * (q_i - q_im1)) / Scalar(12.0);
}

/// Seconds per day for Rayleigh damping timescale conversion.
inline constexpr Scalar seconds_per_day = 86400.0;

}  // namespace detail

template <class ExecSpace>
void Dyn_Tend_Module<ExecSpace>::compute_dyn_tend(
    const DynTendMeshData<ExecSpace>& mesh,
    const DynTendState<ExecSpace>& state,
    DynTendOutput<ExecSpace>& output,
    const PhysTendencies<ExecSpace>& phys,
    const DynTendParams& params,
    int rk_step) const {

  const int nCells = params.nCells;
  const int nEdges = params.nEdges;
  const int nVertLevels = params.nVertLevels;
  const Scalar coef_3rd_order = params.coef_3rd_order;

  // Capture mesh views for lambda capture
  const auto cellsOnEdge = mesh.cellsOnEdge;
  const auto edgesOnCell = mesh.edgesOnCell;
  const auto edgesOnEdge = mesh.edgesOnEdge;
  const auto nEdgesOnCell_v = mesh.nEdgesOnCell;
  const auto nEdgesOnEdge_v = mesh.nEdgesOnEdge;
  const auto nAdvCellsForEdge = mesh.nAdvCellsForEdge;
  const auto advCellsForEdge = mesh.advCellsForEdge;

  const auto dvEdge = mesh.dvEdge;
  const auto invDcEdge = mesh.invDcEdge;
  const auto invAreaCell = mesh.invAreaCell;
  const auto fEdge = mesh.fEdge;
  const auto weightsOnEdge = mesh.weightsOnEdge;
  const auto edgesOnCell_sign = mesh.edgesOnCell_sign;
  const auto zz = mesh.zz;
  const auto zxu = mesh.zxu;
  const auto cqu = mesh.cqu;
  const auto cqw = mesh.cqw;
  const auto rdzu = mesh.rdzu;
  const auto rdzw = mesh.rdzw;
  const auto fzm = mesh.fzm;
  const auto fzp = mesh.fzp;
  const auto adv_coefs = mesh.adv_coefs;
  const auto adv_coefs_3rd = mesh.adv_coefs_3rd;
  const auto angleEdge = mesh.angleEdge;
  const auto u_init = mesh.u_init;
  const auto v_init = mesh.v_init;
  const auto latEdge = mesh.latEdge;
  const auto latCell = mesh.latCell;

  // Capture state views
  const auto u = state.u;
  const auto w = state.w;
  const auto theta_m = state.theta_m;
  const auto rho_zz = state.rho_zz;
  const auto rho_edge = state.rho_edge;
  const auto ru = state.ru;
  const auto rw = state.rw;
  const auto ke = state.ke;
  const auto pv_edge = state.pv_edge;
  const auto divergence = state.divergence;
  const auto pp = state.pp;
  const auto rb = state.rb;
  const auto rr_save = state.rr_save;
  const auto qtot = state.qtot;
  const auto rt_diabatic_tend = state.rt_diabatic_tend;
  const auto theta_m_save = state.theta_m_save;
  const auto ru_save = state.ru_save;
  const auto rw_save = state.rw_save;
  const auto ur_cell = state.ur_cell;
  const auto vr_cell = state.vr_cell;

  // Capture output views
  auto tend_u = output.tend_u;
  auto tend_w = output.tend_w;
  auto tend_theta = output.tend_theta;
  auto tend_rho = output.tend_rho;
  auto h_divergence = output.h_divergence;
  auto tend_u_euler = output.tend_u_euler;
  auto tend_w_euler = output.tend_w_euler;
  auto tend_theta_euler = output.tend_theta_euler;

  // Physics tendencies
  const auto tend_ru_physics = phys.tend_ru_physics;
  const auto tend_rho_physics = phys.tend_rho_physics;
  const auto tend_rtheta_physics = phys.tend_rtheta_physics;

  const Scalar inv_r_earth = Scalar(1.0) / params.r_earth;
  const Scalar omega = params.omega;
  const bool curvature_enabled = params.curvature_enabled;

  // ════════════════════════════════════════════════════════════════════════════
  // On stage 1: zero the Euler (mixing) tendency cache (Req 6.5)
  // ════════════════════════════════════════════════════════════════════════════
  if (rk_step == 1) {
    dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: zeroing tend_u_euler. data=%p shape=(%d, %d)\n",
                  tend_u_euler.data(), (int)tend_u_euler.extent(0), (int)tend_u_euler.extent(1));
    dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: tend_w_euler. data=%p shape=(%d, %d)\n",
                  tend_w_euler.data(), (int)tend_w_euler.extent(0), (int)tend_w_euler.extent(1));
    dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: tend_theta_euler. data=%p shape=(%d, %d)\n",
                  tend_theta_euler.data(), (int)tend_theta_euler.extent(0), (int)tend_theta_euler.extent(1));
    Kokkos::parallel_for(
        "dyn_tend::zero_tend_u_euler",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nEdges}),
        KOKKOS_LAMBDA(const int k, const int iEdge) {
          tend_u_euler(k, iEdge) = Scalar(0.0);
        });
    Kokkos::fence("dyn_tend::zero_tend_u_euler_fence");
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Compute horizontal mass-flux divergence (Requirement 6.2)
  // h_divergence(k,iCell) = sum_edges[sign * dvEdge * ru(k,e)] * invAreaCell
  // ════════════════════════════════════════════════════════════════════════════
  dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: starting h_divergence parallel_for...\n");
  Kokkos::parallel_for(
      "dyn_tend::h_divergence",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // Zero h_divergence
        for (int k = 0; k < nVertLevels; ++k) {
          h_divergence(k, iCell) = Scalar(0.0);
        }
        // Accumulate horizontal mass flux
        const int ne = nEdgesOnCell_v(iCell);
        for (int i = 0; i < ne; ++i) {
          const int iEdge = edgesOnCell(i, iCell) - 1;
          if (iEdge < 0 || iEdge >= nEdges) continue;
          const Scalar edge_sign = edgesOnCell_sign(i, iCell) * dvEdge(iEdge);
          for (int k = 0; k < nVertLevels; ++k) {
            h_divergence(k, iCell) += edge_sign * ru(k, iEdge);
          }
        }
        // Divide by cell area
        const Scalar r = invAreaCell(iCell);
        for (int k = 0; k < nVertLevels; ++k) {
          h_divergence(k, iCell) *= r;
        }
      });
  Kokkos::fence("dyn_tend::h_divergence_fence");

  // ════════════════════════════════════════════════════════════════════════════
  // Density tendency (Requirement 6.1 - rho part)
  // tend_rho = -h_divergence - rdzw*(rw(k+1)-rw(k)) + tend_rho_physics
  // Pressure gradient (dpdz) only computed on rk_step == 1
  // ════════════════════════════════════════════════════════════════════════════
  // Temporary for pressure gradient needed by w tendency
  view2d dpdz("dpdz", nVertLevels, nCells);

  if (rk_step == 1) {
    dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: starting tend_rho_dpdz parallel_for...\n");
    const Scalar rgas = constants::rgas;
    const Scalar cp_val = constants::cp;
    const Scalar cv = cp_val - rgas;

    Kokkos::parallel_for(
        "dyn_tend::tend_rho_dpdz",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nCells}),
        KOKKOS_LAMBDA(const int k, const int iCell) {
          tend_rho(k, iCell) = -h_divergence(k, iCell)
              - rdzw(k) * (rw(k + 1, iCell) - rw(k, iCell))
              + tend_rho_physics(k, iCell);
          // Buoyancy term for the w equation, exactly as the Reference_Model
          // (atm_compute_dyn_tend): dpdz = -g*(rb*qtot + rr_save*(1+qtot)).
          // qtot is the total water mixing ratio computed once per timestep in
          // compute_moist_coefficients (matches Fortran's shared qtot array).
          // (The previous code used -g*(rb + rr_save), i.e. it added the full
          // base density as buoyancy, producing a spurious O(g*rho_base) term
          // that drove the w equation unstable.)
          dpdz(k, iCell) = -constants::gravity *
              (rb(k, iCell) * qtot(k, iCell)
               + rr_save(k, iCell) * (Scalar(1.0) + qtot(k, iCell)));
        });
    Kokkos::fence("dyn_tend::tend_rho_dpdz_fence");
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Horizontal momentum tendency (Requirement 6.1, 6.3, 6.4)
  // Includes: horizontal pressure gradient (stage 1 only),
  //           vertical transport of u,
  //           nonlinear Coriolis (vector-invariant form),
  //           KE gradient, mass-flux-divergence drag,
  //           and curvature terms when enabled.
  // ════════════════════════════════════════════════════════════════════════════
  dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: starting tend_u parallel_for...\n");
  Kokkos::parallel_for(
      "dyn_tend::tend_u",
      Kokkos::RangePolicy<exec_space>(0, nEdges),
      KOKKOS_LAMBDA(const int iEdge) {
        const int cell1 = cellsOnEdge(0, iEdge) - 1;
        const int cell2 = cellsOnEdge(1, iEdge) - 1;
        if (cell1 < 0 || cell1 >= nCells || cell2 < 0 || cell2 >= nCells) return;

        // ── Horizontal pressure gradient (stage 1, cached in tend_u_euler) ──
        if (rk_step == 1) {
          for (int k = 0; k < nVertLevels; ++k) {
            tend_u_euler(k, iEdge) = -cqu(k, iEdge) * (
                (pp(k, cell2) - pp(k, cell1)) * invDcEdge(iEdge) /
                    (Scalar(0.5) * (zz(k, cell2) + zz(k, cell1)))
                - Scalar(0.5) * zxu(k, iEdge) *
                    (dpdz(k, cell1) + dpdz(k, cell2)));
          }
        }

        // ── Vertical transport of u ──────────────────────────────────────────
        // wduz(k) computed inline; flux3 for interior, linear for boundary
        // wduz(0) = 0, wduz(nVertLevels) = 0

        // k=0 boundary
        {
          // wduz at k=1 (interface between level 0 and 1)
          const Scalar wduz_1 = Scalar(0.5) * (rw(1, cell1) + rw(1, cell2)) *
              (fzm(1) * u(1, iEdge) + fzp(1) * u(0, iEdge));
          // tend_u(0) = -rdzw(0) * (wduz_1 - 0)
          tend_u(0, iEdge) = -rdzw(0) * wduz_1;
        }

        // Interior levels: compute wduz(k) and wduz(k+1) for each level k
        for (int k = 1; k < nVertLevels - 1; ++k) {
          Scalar wduz_k, wduz_kp1;
          // wduz at interface k
          if (k == 1) {
            wduz_k = Scalar(0.5) * (rw(1, cell1) + rw(1, cell2)) *
                (fzm(1) * u(1, iEdge) + fzp(1) * u(0, iEdge));
          } else {
            wduz_k = detail::flux3(
                u(k - 2, iEdge), u(k - 1, iEdge),
                u(k, iEdge), u(k + 1, iEdge),
                Scalar(0.5) * (rw(k, cell1) + rw(k, cell2)),
                Scalar(1.0));
          }
          // wduz at interface k+1
          if (k + 1 == nVertLevels - 1) {
            wduz_kp1 = Scalar(0.5) * (rw(k + 1, cell1) + rw(k + 1, cell2)) *
                (fzm(k + 1) * u(k + 1, iEdge) + fzp(k + 1) * u(k, iEdge));
          } else {
            wduz_kp1 = detail::flux3(
                u(k - 1, iEdge), u(k, iEdge),
                u(k + 1, iEdge), u(k + 2, iEdge),
                Scalar(0.5) * (rw(k + 1, cell1) + rw(k + 1, cell2)),
                Scalar(1.0));
          }
          tend_u(k, iEdge) = -rdzw(k) * (wduz_kp1 - wduz_k);
        }

        // Top level
        {
          const int k = nVertLevels - 1;
          Scalar wduz_k;
          if (nVertLevels >= 3) {
            wduz_k = Scalar(0.5) * (rw(k, cell1) + rw(k, cell2)) *
                (fzm(k) * u(k, iEdge) + fzp(k) * u(k - 1, iEdge));
          } else {
            wduz_k = Scalar(0.5) * (rw(k, cell1) + rw(k, cell2)) *
                (fzm(k) * u(k, iEdge) + fzp(k) * u(k - 1, iEdge));
          }
          // wduz(nVertLevels) = 0
          tend_u(k, iEdge) = -rdzw(k) * (Scalar(0.0) - wduz_k);
        }

        // ── Nonlinear Coriolis term (Requirement 6.3) ─────────────────────
        // Vector-invariant formulation following Ringler et al JCP 2009:
        // q(k) = sum_eoe weightsOnEdge(j,iEdge) * u(k,eoe) *
        //        0.5*(pv_edge(k,iEdge) + pv_edge(k,eoe))
        // With perturbation Coriolis correction.
        const int ne_on_edge = nEdgesOnEdge_v(iEdge);
        for (int k = 0; k < nVertLevels; ++k) {
          Scalar q_k = Scalar(0.0);
          for (int j = 0; j < ne_on_edge; ++j) {
            const int eoe = edgesOnEdge(j, iEdge) - 1;
            if (eoe < 0 || eoe >= nEdges) continue;
            const Scalar workpv = Scalar(0.5) *
                (pv_edge(k, iEdge) + pv_edge(k, eoe));
            q_k += weightsOnEdge(j, iEdge) * u(k, eoe) * workpv;
          }

          // Perturbation Coriolis correction (constant-f approximation)
          for (int j = 0; j < ne_on_edge; ++j) {
            const int eoe = edgesOnEdge(j, iEdge) - 1;
            if (eoe < 0 || eoe >= nEdges) continue;
            const Scalar reference_u = u_init(k) * Kokkos::cos(angleEdge(eoe))
                - v_init(k) * Kokkos::sin(angleEdge(eoe));
            q_k -= weightsOnEdge(j, iEdge) * reference_u * fEdge(iEdge);
          }

          // KE gradient + vorticity terms in vector-invariant form
          tend_u(k, iEdge) += rho_edge(k, iEdge) *
              (q_k - (ke(k, cell2) - ke(k, cell1)) * invDcEdge(iEdge))
              - u(k, iEdge) * Scalar(0.5) *
                  (h_divergence(k, cell1) + h_divergence(k, cell2));

          // ── Curvature terms for the sphere (Requirement 6.4) ────────────
          if (curvature_enabled) {
            const Scalar w_avg = Scalar(0.25) *
                (w(k, cell1) + w(k + 1, cell1) +
                 w(k, cell2) + w(k + 1, cell2));
            tend_u(k, iEdge) +=
                -Scalar(2.0) * omega * Kokkos::cos(angleEdge(iEdge)) *
                    Kokkos::cos(latEdge(iEdge)) *
                    rho_edge(k, iEdge) * w_avg
                - u(k, iEdge) * w_avg *
                    rho_edge(k, iEdge) * inv_r_earth;
          }
        }  // end k loop for Coriolis + curvature
      });  // end parallel_for over edges
  Kokkos::fence("dyn_tend::tend_u_fence");

  // ════════════════════════════════════════════════════════════════════════════
  // Rayleigh damping of u (Requirement 6.6)
  // Applied over configured number of levels at the model top.
  // ════════════════════════════════════════════════════════════════════════════
  if (params.config_rayleigh_damp_u &&
      params.config_number_rayleigh_damp_u_levels > 0) {
    const int n_damp_levels = params.config_number_rayleigh_damp_u_levels;
    const Scalar rayleigh_coef_inverse = Scalar(1.0) /
        (Scalar(n_damp_levels) *
         (params.config_rayleigh_damp_u_timescale_days *
          detail::seconds_per_day));
    const int k_start = nVertLevels - n_damp_levels;

    Kokkos::parallel_for(
        "dyn_tend::rayleigh_damp_u",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {k_start, 0}, {nVertLevels, nEdges}),
        KOKKOS_LAMBDA(const int k, const int iEdge) {
          const Scalar damp_coef =
              Scalar(k - k_start + 1) * rayleigh_coef_inverse;
          // Note: Fortran uses k - (nVertLevels - n_damp_levels)
          // which in 0-based is k - k_start + 1 (already +1 for 1-based Fortran)
          // Correction: Fortran k is 1-based, so k_fortran - offset = level index.
          // In 0-based: (k+1) - (nVertLevels - n_damp_levels) = k - k_start + 1
          tend_u(k, iEdge) -= rho_edge(k, iEdge) * u(k, iEdge) * damp_coef;
        });
    Kokkos::fence("dyn_tend::rayleigh_damp_fence");
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Horizontal mixing / dissipation (Reference_Model u_dissipation_3d,
  // w_dissipation_3d, scalar_dissipation_3d_les with les_model_opt == NONE and
  // config_horiz_mixing == "2d_smagorinsky"): 2-D Smagorinsky del^2 mixing plus
  // del^4 background hyperdiffusion for u, w, and theta_m.  Mixing terms are
  // forward-Euler integrated, so they are computed only on the first RK substep
  // and cached in tend_*_euler for reuse in substeps 2 and 3.  This block runs
  // before tend_u_euler/tend_w_euler/tend_theta_euler are folded into the total
  // tendencies (add_euler_phys_u below, add_euler_w and add_euler_phys_theta
  // later), matching the Reference_Model ordering.
  // ════════════════════════════════════════════════════════════════════════════
  if (rk_step == 1 && params.mixing_enabled) {
    const int nVertices = params.nVertices;
    const int vertexDegree = params.vertexDegree;
    const auto v = state.v;
    const auto vorticity = state.vorticity;
    const auto verticesOnEdge = mesh.verticesOnEdge;
    const auto edgesOnVertex = mesh.edgesOnVertex;
    const auto edgesOnVertex_sign = mesh.edgesOnVertex_sign;
    const auto invAreaTriangle = mesh.invAreaTriangle;
    const auto invDvEdge = mesh.invDvEdge;
    const auto dcEdge = mesh.dcEdge;
    const auto meshScalingDel2 = mesh.meshScalingDel2;
    const auto meshScalingDel4 = mesh.meshScalingDel4;
    const auto def_c2 = mesh.deformation_coef_c2;
    const auto def_s2 = mesh.deformation_coef_s2;
    const auto def_cs = mesh.deformation_coef_cs;

    const Scalar c_s = params.c_s;
    const Scalar len_disp = params.config_len_disp;
    const Scalar invDt = Scalar(1.0) / params.dt;
    const Scalar smag_coeff = (c_s * len_disp) * (c_s * len_disp);
    const Scalar stability_limit = Scalar(0.01) * len_disp * len_disp * invDt;
    const Scalar h_mom_eddy_visc4 =
        params.config_visc4_2dsmag * len_disp * len_disp * len_disp;
    const Scalar h_theta_eddy_visc4 = h_mom_eddy_visc4;
    const Scalar del4u_div_factor = params.config_del4u_div_factor;
    const Scalar prandtl_inv = params.prandtl_inv;

    // ── 2-D Smagorinsky horizontal eddy viscosity (kdiff = eddy_visc_horz) ──
    view2d eddy_visc_horz("eddy_visc_horz", nVertLevels, nCells);
    Kokkos::parallel_for(
        "dyn_tend::smagorinsky_2d",
        Kokkos::RangePolicy<exec_space>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          const int ne = nEdgesOnCell_v(iCell);
          for (int k = 0; k < nVertLevels; ++k) {
            Scalar dudx = Scalar(0.0), dudy = Scalar(0.0);
            Scalar dvdx = Scalar(0.0), dvdy = Scalar(0.0);
            for (int i = 0; i < ne; ++i) {
              const int ie = edgesOnCell(i, iCell) - 1;
              const Scalar c2 = def_c2(i, iCell);
              const Scalar s2 = def_s2(i, iCell);
              const Scalar cs = def_cs(i, iCell);
              const Scalar uu = u(k, ie);
              const Scalar vv = v(k, ie);
              dudx += c2 * uu - cs * vv;
              dudy += cs * uu - s2 * vv;
              dvdx += cs * uu + c2 * vv;
              dvdy += s2 * uu + cs * vv;
            }
            const Scalar d_11 = Scalar(2.0) * dudx;
            const Scalar d_22 = Scalar(2.0) * dvdy;
            const Scalar d_12 = dudy + dvdx;
            Scalar visc = smag_coeff * Kokkos::sqrt(
                Scalar(0.25) * (d_11 - d_22) * (d_11 - d_22) + d_12 * d_12);
            visc = Kokkos::fmin(visc, stability_limit);
            eddy_visc_horz(k, iCell) = visc;
          }
        });
    Kokkos::fence("dyn_tend::smagorinsky_2d_fence");

    // ── u dissipation: del^2 (Smagorinsky) into tend_u_euler, and cache
    //    delsq_u for the del^4 filter (u_dissipation_3d) ──
    view2d delsq_u("delsq_u", nVertLevels, nEdges);
    Kokkos::parallel_for(
        "dyn_tend::u_diss_del2",
        Kokkos::RangePolicy<exec_space>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;
          const int vertex1 = verticesOnEdge(0, iEdge) - 1;
          const int vertex2 = verticesOnEdge(1, iEdge) - 1;
          const Scalar r_dc = invDcEdge(iEdge);
          const Scalar r_dv =
              Kokkos::fmin(invDvEdge(iEdge), Scalar(4.0) * invDcEdge(iEdge));
          const Scalar mSd2 = meshScalingDel2(iEdge);
          for (int k = 0; k < nVertLevels; ++k) {
            const Scalar u_diffusion =
                (divergence(k, cell2) - divergence(k, cell1)) * r_dc
              - (vorticity(k, vertex2) - vorticity(k, vertex1)) * r_dv;
            delsq_u(k, iEdge) = u_diffusion;
            const Scalar kdiffu = Scalar(0.5) *
                (eddy_visc_horz(k, cell1) + eddy_visc_horz(k, cell2));
            tend_u_euler(k, iEdge) +=
                rho_edge(k, iEdge) * kdiffu * u_diffusion * mSd2;
          }
        });
    Kokkos::fence("dyn_tend::u_diss_del2_fence");

    if (h_mom_eddy_visc4 > Scalar(0.0)) {  // del^4 filter for u
      view2d delsq_vort("delsq_vort", nVertLevels, nVertices);
      view2d delsq_div("delsq_div", nVertLevels, nCells);
      Kokkos::parallel_for(
          "dyn_tend::u_diss_delsq_vort",
          Kokkos::RangePolicy<exec_space>(0, nVertices),
          KOKKOS_LAMBDA(const int iVertex) {
            for (int k = 0; k < nVertLevels; ++k) {
              Scalar acc = Scalar(0.0);
              for (int i = 0; i < vertexDegree; ++i) {
                const int iEdge = edgesOnVertex(i, iVertex) - 1;
                const Scalar edge_sign = invAreaTriangle(iVertex) *
                    dcEdge(iEdge) * edgesOnVertex_sign(i, iVertex);
                acc += edge_sign * delsq_u(k, iEdge);
              }
              delsq_vort(k, iVertex) = acc;
            }
          });
      Kokkos::parallel_for(
          "dyn_tend::u_diss_delsq_div",
          Kokkos::RangePolicy<exec_space>(0, nCells),
          KOKKOS_LAMBDA(const int iCell) {
            const Scalar r = invAreaCell(iCell);
            const int ne = nEdgesOnCell_v(iCell);
            for (int k = 0; k < nVertLevels; ++k) {
              Scalar acc = Scalar(0.0);
              for (int i = 0; i < ne; ++i) {
                const int iEdge = edgesOnCell(i, iCell) - 1;
                const Scalar edge_sign =
                    r * dvEdge(iEdge) * edgesOnCell_sign(i, iCell);
                acc += edge_sign * delsq_u(k, iEdge);
              }
              delsq_div(k, iCell) = acc;
            }
          });
      Kokkos::fence("dyn_tend::u_diss_delsq_fence");
      Kokkos::parallel_for(
          "dyn_tend::u_diss_del4",
          Kokkos::RangePolicy<exec_space>(0, nEdges),
          KOKKOS_LAMBDA(const int iEdge) {
            const int cell1 = cellsOnEdge(0, iEdge) - 1;
            const int cell2 = cellsOnEdge(1, iEdge) - 1;
            const int vertex1 = verticesOnEdge(0, iEdge) - 1;
            const int vertex2 = verticesOnEdge(1, iEdge) - 1;
            const Scalar u_mix_scale = meshScalingDel4(iEdge) * h_mom_eddy_visc4;
            const Scalar r_dc = u_mix_scale * del4u_div_factor * invDcEdge(iEdge);
            const Scalar r_dv = u_mix_scale *
                Kokkos::fmin(invDvEdge(iEdge), Scalar(4.0) * invDcEdge(iEdge));
            for (int k = 0; k < nVertLevels; ++k) {
              const Scalar u_diffusion = rho_edge(k, iEdge) * (
                  (delsq_div(k, cell2) - delsq_div(k, cell1)) * r_dc
                - (delsq_vort(k, vertex2) - delsq_vort(k, vertex1)) * r_dv);
              tend_u_euler(k, iEdge) -= u_diffusion;
            }
          });
      Kokkos::fence("dyn_tend::u_diss_del4_fence");
    }

    // ── w dissipation (w_dissipation_3d): zero tend_w_euler, then del^2 +
    //    del^4.  Zeroing here (before the pressure-gradient/buoyancy term is
    //    added in tend_w_vadv_pgf) mirrors the Reference_Model. ──
    view2d delsq_w("delsq_w", nVertLevels, nCells);
    Kokkos::parallel_for(
        "dyn_tend::w_diss_del2",
        Kokkos::RangePolicy<exec_space>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          for (int k = 0; k < nVertLevels; ++k) delsq_w(k, iCell) = Scalar(0.0);
          for (int k = 0; k <= nVertLevels; ++k) tend_w_euler(k, iCell) = Scalar(0.0);
          const Scalar r_areaCell = invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);
          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            const Scalar edge_sign = Scalar(0.5) * r_areaCell *
                edgesOnCell_sign(i, iCell) * dvEdge(iEdge) * invDcEdge(iEdge);
            const int cell1 = cellsOnEdge(0, iEdge) - 1;
            const int cell2 = cellsOnEdge(1, iEdge) - 1;
            const Scalar mSd2 = meshScalingDel2(iEdge);
            for (int k = 1; k < nVertLevels; ++k) {
              Scalar wtf = edge_sign *
                  (rho_edge(k, iEdge) + rho_edge(k - 1, iEdge)) *
                  (w(k, cell2) - w(k, cell1));
              delsq_w(k, iCell) += wtf;
              wtf = wtf * mSd2 * Scalar(0.25) * (
                  eddy_visc_horz(k, cell1) + eddy_visc_horz(k, cell2)
                + eddy_visc_horz(k - 1, cell1) + eddy_visc_horz(k - 1, cell2));
              tend_w_euler(k, iCell) += wtf;
            }
          }
        });
    Kokkos::fence("dyn_tend::w_diss_del2_fence");

    if (h_mom_eddy_visc4 > Scalar(0.0)) {  // del^4 filter for w
      Kokkos::parallel_for(
          "dyn_tend::w_diss_del4",
          Kokkos::RangePolicy<exec_space>(0, nCells),
          KOKKOS_LAMBDA(const int iCell) {
            const Scalar r_areaCell = h_mom_eddy_visc4 * invAreaCell(iCell);
            const int ne = nEdgesOnCell_v(iCell);
            for (int i = 0; i < ne; ++i) {
              const int iEdge = edgesOnCell(i, iCell) - 1;
              const int cell1 = cellsOnEdge(0, iEdge) - 1;
              const int cell2 = cellsOnEdge(1, iEdge) - 1;
              const Scalar edge_sign = meshScalingDel4(iEdge) * r_areaCell *
                  dvEdge(iEdge) * edgesOnCell_sign(i, iCell) * invDcEdge(iEdge);
              for (int k = 1; k < nVertLevels; ++k) {
                tend_w_euler(k, iCell) -=
                    edge_sign * (delsq_w(k, cell2) - delsq_w(k, cell1));
              }
            }
          });
      Kokkos::fence("dyn_tend::w_diss_del4_fence");
    }

    // ── theta_m dissipation (scalar_dissipation_3d_les): zero tend_theta_euler,
    //    then del^2 + del^4. ──
    view2d delsq_theta("delsq_theta", nVertLevels, nCells);
    Kokkos::parallel_for(
        "dyn_tend::theta_diss_del2",
        Kokkos::RangePolicy<exec_space>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          for (int k = 0; k < nVertLevels; ++k) {
            delsq_theta(k, iCell) = Scalar(0.0);
            tend_theta_euler(k, iCell) = Scalar(0.0);
          }
          const Scalar r_areaCell = invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);
          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            const Scalar edge_sign = r_areaCell *
                edgesOnCell_sign(i, iCell) * dvEdge(iEdge) * invDcEdge(iEdge);
            const Scalar pr_scale = prandtl_inv * meshScalingDel2(iEdge);
            const int cell1 = cellsOnEdge(0, iEdge) - 1;
            const int cell2 = cellsOnEdge(1, iEdge) - 1;
            for (int k = 0; k < nVertLevels; ++k) {
              Scalar ttf = edge_sign *
                  (theta_m(k, cell2) - theta_m(k, cell1)) * rho_edge(k, iEdge);
              delsq_theta(k, iCell) += ttf;
              ttf = ttf * Scalar(0.5) *
                  (eddy_visc_horz(k, cell1) + eddy_visc_horz(k, cell2)) * pr_scale;
              tend_theta_euler(k, iCell) += ttf;
            }
          }
        });
    Kokkos::fence("dyn_tend::theta_diss_del2_fence");

    if (h_theta_eddy_visc4 > Scalar(0.0)) {  // del^4 filter for theta_m
      Kokkos::parallel_for(
          "dyn_tend::theta_diss_del4",
          Kokkos::RangePolicy<exec_space>(0, nCells),
          KOKKOS_LAMBDA(const int iCell) {
            const Scalar r_areaCell =
                h_theta_eddy_visc4 * prandtl_inv * invAreaCell(iCell);
            const int ne = nEdgesOnCell_v(iCell);
            for (int i = 0; i < ne; ++i) {
              const int iEdge = edgesOnCell(i, iCell) - 1;
              const int cell1 = cellsOnEdge(0, iEdge) - 1;
              const int cell2 = cellsOnEdge(1, iEdge) - 1;
              const Scalar edge_sign = meshScalingDel4(iEdge) * r_areaCell *
                  dvEdge(iEdge) * edgesOnCell_sign(i, iCell) * invDcEdge(iEdge);
              for (int k = 0; k < nVertLevels; ++k) {
                tend_theta_euler(k, iCell) -=
                    edge_sign * (delsq_theta(k, cell2) - delsq_theta(k, cell1));
              }
            }
          });
      Kokkos::fence("dyn_tend::theta_diss_del4_fence");
    }
  }  // end horizontal mixing (rk_step == 1)

  // ════════════════════════════════════════════════════════════════════════════
  // Add mixing (Euler) tendencies and physics tendency for u (Req 6.5, 6.7)
  // tend_u += tend_u_euler + tend_ru_physics
  // ════════════════════════════════════════════════════════════════════════════
  Kokkos::parallel_for(
      "dyn_tend::add_euler_phys_u",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
          {0, 0}, {nVertLevels, nEdges}),
      KOKKOS_LAMBDA(const int k, const int iEdge) {
        tend_u(k, iEdge) += tend_u_euler(k, iEdge) +
            tend_ru_physics(k, iEdge);
      });
  Kokkos::fence("dyn_tend::add_euler_phys_u_fence");

  // ════════════════════════════════════════════════════════════════════════════
  // Vertical momentum (w) tendency (Requirement 6.1 - w part)
  // Includes: horizontal advection of w,
  //           curvature terms when enabled,
  //           vertical advection of w,
  //           pressure gradient and buoyancy (stage 1 in tend_w_euler),
  //           and mixing tendency from tend_w_euler.
  // ════════════════════════════════════════════════════════════════════════════

  // Step 1: Horizontal advection of w
  dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: starting tend_w_hadv parallel_for...\n");
  Kokkos::parallel_for(
      "dyn_tend::tend_w_hadv",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // Zero tend_w
        for (int k = 0; k <= nVertLevels; ++k) {
          tend_w(k, iCell) = Scalar(0.0);
        }

        const int ne = nEdgesOnCell_v(iCell);
        for (int i = 0; i < ne; ++i) {
          const int iEdge = edgesOnCell(i, iCell) - 1;
          if (iEdge < 0 || iEdge >= nEdges) continue;
          const Scalar edge_sign_dv =
              edgesOnCell_sign(i, iCell) * dvEdge(iEdge) * Scalar(0.5);

          // Horizontal transport of w at interior interfaces (k=1..nVertLevels-1)
          // Uses third-order advection coefficients
          const int nAdv = nAdvCellsForEdge(iEdge);
          for (int k = 1; k < nVertLevels; ++k) {
            // ru_edge_w at interface k
            const Scalar ru_edge_w_k =
                fzm(k) * ru(k, iEdge) + fzp(k) * ru(k - 1, iEdge);

            // Compute flux of w at this edge using advection coefficients
            Scalar flux_arr_k = Scalar(0.0);
            for (int j = 0; j < nAdv; ++j) {
              const int iAdvCell = advCellsForEdge(j, iEdge) - 1;
              if (iAdvCell < 0 || iAdvCell >= nCells) continue;
              const Scalar scalar_weight = adv_coefs(j, iEdge) +
                  Kokkos::copysign(Scalar(1.0), ru_edge_w_k) *
                  adv_coefs_3rd(j, iEdge);
              flux_arr_k += scalar_weight * w(k, iAdvCell);
            }

            tend_w(k, iCell) -= edgesOnCell_sign(i, iCell) *
                ru_edge_w_k * flux_arr_k;
          }
        }
      });
  Kokkos::fence("dyn_tend::tend_w_hadv_fence");

  // Step 2: Curvature terms for w (Requirement 6.4)
  if (curvature_enabled) {
    dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: starting tend_w_curvature parallel_for...\n");
    Kokkos::parallel_for(
        "dyn_tend::tend_w_curvature",
        Kokkos::RangePolicy<exec_space>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          for (int k = 1; k < nVertLevels; ++k) {
            const Scalar rho_w = rho_zz(k, iCell) * fzm(k) +
                rho_zz(k - 1, iCell) * fzp(k);
            const Scalar ur_w = fzm(k) * ur_cell(k, iCell) +
                fzp(k) * ur_cell(k - 1, iCell);
            const Scalar vr_w = fzm(k) * vr_cell(k, iCell) +
                fzp(k) * vr_cell(k - 1, iCell);
            tend_w(k, iCell) += rho_w *
                (ur_w * ur_w + vr_w * vr_w) / params.r_earth
                + Scalar(2.0) * omega * Kokkos::cos(latCell(iCell)) *
                    ur_w * rho_w;
          }
        });
    Kokkos::fence("dyn_tend::tend_w_curvature_fence");
  }

  // Step 3: Vertical advection of w + pressure gradient/buoyancy + area division
  dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: starting tend_w_vadv_pgf parallel_for...\n");
  Kokkos::parallel_for(
      "dyn_tend::tend_w_vadv_pgf",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // Vertical advection: wdwz using flux3 for interior
        // wdwz(0) = 0, wdwz(nVertLevels) = 0
        // For k=1 and k=nVertLevels-1: use linear interpolation
        // For interior: use flux3

        for (int k = 1; k < nVertLevels; ++k) {
          Scalar wdwz_k, wdwz_kp1;

          // wdwz at interface k (between w levels k-1 and k)
          if (k == 1) {
            wdwz_k = Scalar(0.25) * (rw(1, iCell) + rw(0, iCell)) *
                (w(1, iCell) + w(0, iCell));
          } else if (k == 2) {
            wdwz_k = Scalar(0.25) * (rw(k, iCell) + rw(k - 1, iCell)) *
                (w(k, iCell) + w(k - 1, iCell));
          } else {
            wdwz_k = detail::flux3(
                w(k - 2, iCell), w(k - 1, iCell),
                w(k, iCell), w(k + 1, iCell),
                Scalar(0.5) * (rw(k, iCell) + rw(k - 1, iCell)),
                Scalar(1.0));
          }

          // wdwz at interface k+1
          if (k + 1 > nVertLevels) {
            wdwz_kp1 = Scalar(0.0);
          } else if (k + 1 == nVertLevels) {
            wdwz_kp1 = Scalar(0.25) * (rw(k + 1, iCell) + rw(k, iCell)) *
                (w(k + 1, iCell) + w(k, iCell));
          } else if (k + 1 <= 2) {
            wdwz_kp1 = Scalar(0.25) * (rw(k + 1, iCell) + rw(k, iCell)) *
                (w(k + 1, iCell) + w(k, iCell));
          } else if (k + 2 >= nVertLevels) {
            wdwz_kp1 = Scalar(0.25) * (rw(k + 1, iCell) + rw(k, iCell)) *
                (w(k + 1, iCell) + w(k, iCell));
          } else {
            wdwz_kp1 = detail::flux3(
                w(k - 1, iCell), w(k, iCell),
                w(k + 1, iCell), w(k + 2, iCell),
                Scalar(0.5) * (rw(k + 1, iCell) + rw(k, iCell)),
                Scalar(1.0));
          }

          // Divide horizontal adv by cell area and subtract vertical adv
          tend_w(k, iCell) = tend_w(k, iCell) * invAreaCell(iCell)
              - rdzu(k) * (wdwz_kp1 - wdwz_k);

          // Pressure gradient + buoyancy (stage 1 only, cached in tend_w_euler)
          if (rk_step == 1) {
            tend_w_euler(k, iCell) += -cqw(k, iCell) * (
                rdzu(k) * (pp(k, iCell) - pp(k - 1, iCell))
                - (fzm(k) * dpdz(k, iCell) + fzp(k) * dpdz(k - 1, iCell)));
          }
        }
      });
  Kokkos::fence("dyn_tend::tend_w_vadv_pgf_fence");

  // Step 4: Add mixing tendency for w (Req 6.5)
  Kokkos::parallel_for(
      "dyn_tend::add_euler_w",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        for (int k = 1; k < nVertLevels; ++k) {
          tend_w(k, iCell) += tend_w_euler(k, iCell);
        }
      });
  Kokkos::fence("dyn_tend::add_euler_w_fence");

  // ════════════════════════════════════════════════════════════════════════════
  // Coupled potential temperature (theta_m) tendency (Requirement 6.1)
  // Includes: horizontal advection of theta_m,
  //           perturbation flux correction (rk_step > 1),
  //           vertical advection + diabatic tendency,
  //           mixing tendency + physics tendency.
  // ════════════════════════════════════════════════════════════════════════════

  // Step 1: Horizontal advection of theta_m
  dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: starting tend_theta_hadv parallel_for...\n");
  Kokkos::parallel_for(
      "dyn_tend::tend_theta_hadv",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // Zero tend_theta
        for (int k = 0; k < nVertLevels; ++k) {
          tend_theta(k, iCell) = Scalar(0.0);
        }

        const int ne = nEdgesOnCell_v(iCell);
        for (int i = 0; i < ne; ++i) {
          const int iEdge = edgesOnCell(i, iCell) - 1;
          if (iEdge < 0 || iEdge >= nEdges) continue;
          const int nAdv = nAdvCellsForEdge(iEdge);

          for (int k = 0; k < nVertLevels; ++k) {
            Scalar flux_arr_k = Scalar(0.0);
            for (int j = 0; j < nAdv; ++j) {
              const int iAdvCell = advCellsForEdge(j, iEdge) - 1;
              if (iAdvCell < 0 || iAdvCell >= nCells) continue;
              const Scalar scalar_weight = adv_coefs(j, iEdge) +
                  Kokkos::copysign(Scalar(1.0), ru(k, iEdge)) *
                  adv_coefs_3rd(j, iEdge);
              flux_arr_k += scalar_weight * theta_m(k, iAdvCell);
            }
            tend_theta(k, iCell) -= edgesOnCell_sign(i, iCell) *
                ru(k, iEdge) * flux_arr_k;
          }
        }

        // Perturbation flux correction (rk_step > 1)
        if (rk_step > 1) {
          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            if (iEdge < 0 || iEdge >= nEdges) continue;
            const int c1 = cellsOnEdge(0, iEdge) - 1;
            const int c2 = cellsOnEdge(1, iEdge) - 1;
            if (c1 < 0 || c1 >= nCells || c2 < 0 || c2 >= nCells) continue;
            for (int k = 0; k < nVertLevels; ++k) {
              const Scalar flux = edgesOnCell_sign(i, iCell) *
                  dvEdge(iEdge) *
                  (ru_save(k, iEdge) - ru(k, iEdge)) *
                  Scalar(0.5) * (theta_m_save(k, c2) + theta_m_save(k, c1));
              tend_theta(k, iCell) -= flux;
            }
          }
        }
      });
  Kokkos::fence("dyn_tend::tend_theta_hadv_fence");

  // Step 2: Vertical advection + diabatic tendency + area division
  dyn_debug_log("[CPP DEBUG] Dyn_Tend_Module: starting tend_theta_vadv parallel_for...\n");
  Kokkos::parallel_for(
      "dyn_tend::tend_theta_vadv",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        for (int k = 0; k < nVertLevels; ++k) {
          // Compute wdtz(k) and wdtz(k+1) at interfaces surrounding level k
          Scalar wdtz_k, wdtz_kp1;

          // wdtz at interface k (bottom of level k)
          if (k == 0) {
            wdtz_k = Scalar(0.0);
          } else if (k == 1) {
            wdtz_k = rw(1, iCell) *
                (fzm(1) * theta_m(1, iCell) + fzp(1) * theta_m(0, iCell));
            // Add perturbation flux from rw_save
            wdtz_k += (rw_save(1, iCell) - rw(1, iCell)) *
                (fzm(1) * theta_m_save(1, iCell) +
                 fzp(1) * theta_m_save(0, iCell));
          } else if (k == nVertLevels - 1) {
            // Top interface uses rw_save directly
            wdtz_k = rw_save(k, iCell) *
                (fzm(k) * theta_m(k, iCell) +
                 fzp(k) * theta_m(k - 1, iCell));
          } else {
            wdtz_k = detail::flux3(
                theta_m(k - 2, iCell), theta_m(k - 1, iCell),
                theta_m(k, iCell), theta_m(k + 1, iCell),
                rw(k, iCell), coef_3rd_order);
            wdtz_k += (rw_save(k, iCell) - rw(k, iCell)) *
                (fzm(k) * theta_m_save(k, iCell) +
                 fzp(k) * theta_m_save(k - 1, iCell));
          }

          // wdtz at interface k+1 (top of level k)
          if (k + 1 >= nVertLevels) {
            wdtz_kp1 = Scalar(0.0);
          } else if (k + 1 == 1) {
            wdtz_kp1 = rw(1, iCell) *
                (fzm(1) * theta_m(1, iCell) + fzp(1) * theta_m(0, iCell));
            wdtz_kp1 += (rw_save(1, iCell) - rw(1, iCell)) *
                (fzm(1) * theta_m_save(1, iCell) +
                 fzp(1) * theta_m_save(0, iCell));
          } else if (k + 1 == nVertLevels - 1) {
            wdtz_kp1 = rw_save(k + 1, iCell) *
                (fzm(k + 1) * theta_m(k + 1, iCell) +
                 fzp(k + 1) * theta_m(k, iCell));
          } else if (k + 1 <= 1 || k + 2 >= nVertLevels) {
            wdtz_kp1 = rw(k + 1, iCell) *
                (fzm(k + 1) * theta_m(k + 1, iCell) +
                 fzp(k + 1) * theta_m(k, iCell));
            wdtz_kp1 += (rw_save(k + 1, iCell) - rw(k + 1, iCell)) *
                (fzm(k + 1) * theta_m_save(k + 1, iCell) +
                 fzp(k + 1) * theta_m_save(k, iCell));
          } else {
            wdtz_kp1 = detail::flux3(
                theta_m(k - 1, iCell), theta_m(k, iCell),
                theta_m(k + 1, iCell), theta_m(k + 2, iCell),
                rw(k + 1, iCell), coef_3rd_order);
            wdtz_kp1 += (rw_save(k + 1, iCell) - rw(k + 1, iCell)) *
                (fzm(k + 1) * theta_m_save(k + 1, iCell) +
                 fzp(k + 1) * theta_m_save(k, iCell));
          }

          // Combine: area division + vertical advection + diabatic
          tend_theta(k, iCell) = tend_theta(k, iCell) * invAreaCell(iCell)
              - rdzw(k) * (wdtz_kp1 - wdtz_k)
              + rho_zz(k, iCell) * rt_diabatic_tend(k, iCell);
        }
      });
  Kokkos::fence("dyn_tend::tend_theta_vadv_fence");

  // Step 3: Add mixing tendency + physics tendency for theta (Req 6.5, 6.7)
  Kokkos::parallel_for(
      "dyn_tend::add_euler_phys_theta",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
          {0, 0}, {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        tend_theta(k, iCell) += tend_theta_euler(k, iCell) +
            tend_rtheta_physics(k, iCell);
      });
  Kokkos::fence("dyn_tend::add_euler_phys_theta_fence");

}  // end compute_dyn_tend

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_DYN_TEND_MODULE_HPP
