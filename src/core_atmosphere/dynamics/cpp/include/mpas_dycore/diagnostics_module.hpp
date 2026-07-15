#ifndef MPAS_DYCORE_DIAGNOSTICS_MODULE_HPP
#define MPAS_DYCORE_DIAGNOSTICS_MODULE_HPP

/// @file diagnostics_module.hpp
/// @brief Solve and coupled diagnostics for the C++ dycore.
///
/// Implements `compute_solve_diagnostics` equivalent to
/// `atm_compute_solve_diagnostics_work` in the Reference_Model.
/// Computes edge density, tangential velocity, relative vorticity,
/// divergence, kinetic energy, edge potential vorticity, APVM upstream
/// bias, and the Hollingsworth KE adjustment (Requirements 7.1-7.3).
///
/// Also implements `init_coupled_diagnostics` equivalent to
/// `atm_init_coupled_diagnostics` in the Reference_Model.
/// Derives coupled potential temperature, density, mass fluxes, Exner
/// function, and pressure from the prognostic state (Requirement 7.4).

#include "mpas_dycore/scalar.hpp"
#include "mpas_dycore/config.hpp"
#include "mpas_dycore/accumulation.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <cmath>

namespace mpas {
namespace dycore {

/// Extended mesh data for the Diagnostics_Module.
/// Contains additional connectivity/geometry arrays beyond the base MeshData
/// that are needed for vorticity, KE, and PV computations.
/// All arrays follow LayoutLeft with Fortran 1-based connectivity indices.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct DiagMeshData {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;

  template <class T>
  using View1D = Kokkos::View<T*, layout, memory_space>;
  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;

  // Dimensions
  int nCells = 0;
  int nEdges = 0;
  int nVertices = 0;
  int nVertLevels = 0;
  int maxEdges = 0;
  int maxEdges2 = 0;   // max edges on edge (for weightsOnEdge, edgesOnEdge)
  int vertexDegree = 0; // typically 3 for triangular dual

  // Connectivity (integer arrays, 1-based Fortran indices)
  View2D<int> cellsOnEdge;       // (2, nEdges)
  View2D<int> verticesOnEdge;    // (2, nEdges)
  View2D<int> edgesOnCell;       // (maxEdges, nCells)
  View2D<int> edgesOnEdge;       // (maxEdges2, nEdges)
  View2D<int> edgesOnVertex;     // (vertexDegree, nVertices)
  View2D<int> verticesOnCell;    // (maxEdges, nCells)
  View2D<int> kiteForCell;       // (maxEdges, nCells)
  View1D<int> nEdgesOnCell;      // (nCells)
  View1D<int> nEdgesOnEdge;      // (nEdges)

  // Geometry (real-valued)
  View1D<Scalar> dvEdge;          // (nEdges)
  View1D<Scalar> dcEdge;          // (nEdges)
  View1D<Scalar> invDvEdge;       // (nEdges)
  View1D<Scalar> invDcEdge;       // (nEdges)
  View1D<Scalar> invAreaCell;     // (nCells)
  View1D<Scalar> invAreaTriangle; // (nVertices)
  View1D<Scalar> fVertex;         // (nVertices)
  View1D<Scalar> fEdge;           // (nEdges)
  View2D<Scalar> weightsOnEdge;   // (maxEdges2, nEdges)
  View2D<Scalar> kiteAreasOnVertex; // (vertexDegree, nVertices)
  View2D<Scalar> edgesOnVertex_sign; // (vertexDegree, nVertices)
  View2D<Scalar> edgesOnCell_sign;   // (maxEdges, nCells)
};

/// Output fields produced by compute_solve_diagnostics.
/// All Views are (nVertLevels, nElements) in LayoutLeft.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct DiagFields {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;

  View2D<Scalar> h_edge;      // (nVertLevels, nEdges) - edge density
  View2D<Scalar> v;           // (nVertLevels, nEdges) - tangential velocity
  View2D<Scalar> vorticity;   // (nVertLevels, nVertices) - relative vorticity
  View2D<Scalar> divergence;  // (nVertLevels, nCells) - divergence
  View2D<Scalar> ke;          // (nVertLevels, nCells) - kinetic energy
  View2D<Scalar> pv_edge;     // (nVertLevels, nEdges) - potential vorticity at edges
  View2D<Scalar> pv_vertex;   // (nVertLevels, nVertices) - PV at vertices
  View2D<Scalar> pv_cell;     // (nVertLevels, nCells) - PV at cells
  View2D<Scalar> gradPVn;     // (nVertLevels, nEdges) - PV gradient normal
  View2D<Scalar> gradPVt;     // (nVertLevels, nEdges) - PV gradient tangent
};

/// Physical constants for coupled diagnostics, matching the Reference_Model
/// (mpas_constants.F).
namespace diag_constants {
inline constexpr Scalar rgas = 287.0;
inline constexpr Scalar rv = 461.6;
inline constexpr Scalar cp = 7.0 * rgas / 2.0;
inline constexpr Scalar cv = cp - rgas;
inline constexpr Scalar rcv = rgas / cv;  // rgas / (cp - rgas)
inline constexpr Scalar rvord = rv / rgas;
inline constexpr Scalar p0 = 1.0e5;
}  // namespace diag_constants

/// Extended mesh data required by init_coupled_diagnostics.
/// Contains connectivity/geometry beyond DiagMeshData needed for mass fluxes
/// (zb_cell, zb3_cell, fzm, fzp).
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct CoupledDiagMeshData {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;

  template <class T>
  using View1D = Kokkos::View<T*, layout, memory_space>;
  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;
  template <class T>
  using View3D = Kokkos::View<T***, layout, memory_space>;

  // Dimensions
  int nCells = 0;
  int nEdges = 0;
  int nVertLevels = 0;
  int maxEdges = 0;

  // Connectivity (1-based Fortran indices)
  View2D<int> cellsOnEdge;       // (2, nEdges)
  View2D<int> edgesOnCell;       // (maxEdges, nCells)
  View1D<int> nEdgesOnCell;      // (nCells)

  // Geometry
  View2D<Scalar> edgesOnCell_sign;  // (maxEdges, nCells)
  View2D<Scalar> zz;               // (nVertLevels, nCells) - Jacobian dz/dzeta
  View1D<Scalar> fzm;              // (nVertLevels+1) - weight for upper level
  View1D<Scalar> fzp;              // (nVertLevels+1) - weight for lower level
  View3D<Scalar> zb_cell;          // (nVertLevels+1, maxEdges, nCells)
  View3D<Scalar> zb3_cell;         // (nVertLevels+1, maxEdges, nCells)
};

/// Input prognostic state for init_coupled_diagnostics.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct CoupledDiagState {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;

  View2D<Scalar> theta;       // (nVertLevels, nCells) - dry potential temperature
  View2D<Scalar> rho;         // (nVertLevels, nCells) - density (= rho_zz * zz)
  View2D<Scalar> u;           // (nVertLevels, nEdges) - horizontal velocity
  View2D<Scalar> w;           // (nVertLevels+1, nCells) - vertical velocity
  View2D<Scalar> qv;          // (nVertLevels, nCells) - water vapor mixing ratio
  View2D<Scalar> rho_base;    // (nVertLevels, nCells) - base state density
  View2D<Scalar> theta_base;  // (nVertLevels, nCells) - base state theta
};

/// Output fields produced by init_coupled_diagnostics.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct CoupledDiagFields {
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  template <class T>
  using View2D = Kokkos::View<T**, layout, memory_space>;

  View2D<Scalar> theta_m;        // (nVertLevels, nCells) - moist potential temp
  View2D<Scalar> rho_zz;         // (nVertLevels, nCells) - coupled density
  View2D<Scalar> rho_p;          // (nVertLevels, nCells) - density perturbation
  View2D<Scalar> rtheta_base;    // (nVertLevels, nCells) - base rho*theta
  View2D<Scalar> rtheta_p;       // (nVertLevels, nCells) - perturbation rho*theta
  View2D<Scalar> ru;             // (nVertLevels, nEdges) - horizontal mass flux
  View2D<Scalar> rw;             // (nVertLevels+1, nCells) - vertical mass flux
  View2D<Scalar> exner;          // (nVertLevels, nCells) - Exner function
  View2D<Scalar> exner_base;     // (nVertLevels, nCells) - base Exner function
  View2D<Scalar> pressure_p;     // (nVertLevels, nCells) - pressure perturbation
  View2D<Scalar> pressure_base;  // (nVertLevels, nCells) - base pressure
};

/// @brief Diagnostics_Module: solve and coupled diagnostics.
///
/// Implements the diagnostic computations equivalent to
/// `atm_compute_solve_diagnostics` in the Reference_Model (Requirement 7).
///
/// @tparam ExecSpace Kokkos execution space for all parallel kernels.
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
class Diagnostics_Module {
 public:
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;
  using view2d = Kokkos::View<Scalar**, layout, memory_space>;

  Diagnostics_Module() = default;

  /// @brief Compute solve diagnostics (Requirement 7.1, 7.2, 7.3).
  ///
  /// Computes: edge density, tangential velocity (on rk_step==3),
  /// relative vorticity, divergence, kinetic energy (with Hollingsworth
  /// adjustment when enabled), potential vorticity at vertices/edges,
  /// and APVM upstream bias when config_apvm_upwinding > 0.
  ///
  /// @param mesh       Extended mesh connectivity/geometry.
  /// @param u          Horizontal velocity (nVertLevels, nEdges).
  /// @param h          Cell density rho_zz (nVertLevels, nCells).
  /// @param diag       Output diagnostic fields (modified in place).
  /// @param dt         Timestep (used for APVM).
  /// @param config_apvm_upwinding  APVM coefficient (>0 enables upwinding).
  /// @param hollingsworth  Whether to apply Hollingsworth KE adjustment.
  /// @param rk_step    Current RK step (tangential velocity only on step 3).
  ///                   Pass -1 if not in an RK context (reconstructs v).
  void compute_solve_diagnostics(
      const DiagMeshData<ExecSpace>& mesh,
      const view2d& u,
      const view2d& h,
      DiagFields<ExecSpace>& diag,
      Scalar dt,
      Scalar config_apvm_upwinding,
      bool hollingsworth,
      int rk_step) const;

  /// @brief Initialize coupled diagnostics (Requirement 7.4).
  ///
  /// Derives coupled potential temperature, density, mass fluxes, Exner
  /// function, and pressure from the prognostic state, replicating
  /// `atm_init_coupled_diagnostics` in the Reference_Model.
  ///
  /// @param cmesh  Mesh connectivity/geometry for coupled diagnostics.
  /// @param state  Input prognostic state fields.
  /// @param out    Output coupled diagnostic fields (modified in place).
  void init_coupled_diagnostics(
      const CoupledDiagMeshData<ExecSpace>& cmesh,
      const CoupledDiagState<ExecSpace>& state,
      CoupledDiagFields<ExecSpace>& out) const;
};

// ============================================================================
// Implementation (header-only, required for templates)
// ============================================================================

template <class ExecSpace>
void Diagnostics_Module<ExecSpace>::compute_solve_diagnostics(
    const DiagMeshData<ExecSpace>& mesh,
    const view2d& u,
    const view2d& h,
    DiagFields<ExecSpace>& diag,
    Scalar dt,
    Scalar config_apvm_upwinding,
    bool hollingsworth,
    int rk_step) const {

  const int nCells = mesh.nCells;
  const int nEdges = mesh.nEdges;
  const int nVertices = mesh.nVertices;
  const int nVertLevels = mesh.nVertLevels;
  const int vertexDegree = mesh.vertexDegree;

  // Capture mesh views for lambda capture
  const auto cellsOnEdge = mesh.cellsOnEdge;
  const auto verticesOnEdge = mesh.verticesOnEdge;
  const auto edgesOnCell = mesh.edgesOnCell;
  const auto edgesOnEdge = mesh.edgesOnEdge;
  const auto edgesOnVertex = mesh.edgesOnVertex;
  const auto verticesOnCell = mesh.verticesOnCell;
  const auto kiteForCell = mesh.kiteForCell;
  const auto nEdgesOnCell_v = mesh.nEdgesOnCell;
  const auto nEdgesOnEdge_v = mesh.nEdgesOnEdge;

  const auto dvEdge = mesh.dvEdge;
  const auto dcEdge = mesh.dcEdge;
  const auto invDvEdge = mesh.invDvEdge;
  const auto invDcEdge = mesh.invDcEdge;
  const auto invAreaCell = mesh.invAreaCell;
  const auto invAreaTriangle = mesh.invAreaTriangle;
  const auto fVertex = mesh.fVertex;
  const auto weightsOnEdge = mesh.weightsOnEdge;
  const auto kiteAreasOnVertex = mesh.kiteAreasOnVertex;
  const auto edgesOnVertex_sign = mesh.edgesOnVertex_sign;
  const auto edgesOnCell_sign = mesh.edgesOnCell_sign;

  // Output views
  auto h_edge = diag.h_edge;
  auto v_out = diag.v;
  auto vorticity = diag.vorticity;
  auto divergence = diag.divergence;
  auto ke = diag.ke;
  auto pv_edge = diag.pv_edge;
  auto pv_vertex = diag.pv_vertex;
  auto pv_cell = diag.pv_cell;
  auto gradPVn = diag.gradPVn;
  auto gradPVt = diag.gradPVt;

  // Temporary: kinetic energy per edge (dcEdge * dvEdge * u^2)
  view2d ke_edge("ke_edge", nVertLevels, nEdges);

  // ──────────────────────────────────────────────────────────────────────────
  // Step 1: Compute edge density (h_edge) and ke_edge
  // h_edge(k,iEdge) = 0.5 * (h(k,cell1) + h(k,cell2))
  // ke_edge(k,iEdge) = dcEdge(iEdge) * dvEdge(iEdge) * u(k,iEdge)^2
  // Note: Fortran uses 1-based indices; our connectivity arrays store them
  //       as-is. We subtract 1 for 0-based C++ indexing.
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "diag::edge_density_and_ke_edge",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nEdges}),
      KOKKOS_LAMBDA(const int k, const int iEdge) {
        const int cell1 = cellsOnEdge(0, iEdge) - 1;  // 1-based to 0-based
        const int cell2 = cellsOnEdge(1, iEdge) - 1;
        h_edge(k, iEdge) =
            Scalar(0.5) * (h(k, cell1) + h(k, cell2));

        const Scalar efac = dcEdge(iEdge) * dvEdge(iEdge);
        ke_edge(k, iEdge) = efac * u(k, iEdge) * u(k, iEdge);
      });
  Kokkos::fence("diag::edge_density_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 2: Compute relative vorticity at each vertex
  // vorticity(k,iVertex) = sum_i(edgesOnVertex_sign(i,iV) * dcEdge(e) * u(k,e))
  //                        * invAreaTriangle(iVertex)
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "diag::vorticity",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nVertices}),
      KOKKOS_LAMBDA(const int k, const int iVertex) {
        Scalar circ = Scalar(0);
        for (int i = 0; i < vertexDegree; ++i) {
          const int iEdge = edgesOnVertex(i, iVertex) - 1;  // 1-based to 0-based
          const Scalar s = edgesOnVertex_sign(i, iVertex) * dcEdge(iEdge);
          circ += s * u(k, iEdge);
        }
        vorticity(k, iVertex) = circ * invAreaTriangle(iVertex);
      });
  Kokkos::fence("diag::vorticity_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 3: Compute divergence at each cell center
  // divergence(k,iCell) = sum_i(edgesOnCell_sign(i,iC) * dvEdge(e) * u(k,e))
  //                       * invAreaCell(iCell)
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "diag::divergence",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        Scalar div_sum = Scalar(0);
        const int ne = nEdgesOnCell_v(iCell);
        for (int i = 0; i < ne; ++i) {
          const int iEdge = edgesOnCell(i, iCell) - 1;
          const Scalar s = edgesOnCell_sign(i, iCell) * dvEdge(iEdge);
          div_sum += s * u(k, iEdge);
        }
        divergence(k, iCell) = div_sum * invAreaCell(iCell);
      });
  Kokkos::fence("diag::divergence_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 4: Compute kinetic energy at cells (Ringler et al JCP 2009)
  // ke(k,iCell) = sum_i(0.25 * ke_edge(k,edgesOnCell(i,iCell)))
  //              * invAreaCell(iCell)
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "diag::kinetic_energy",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        Scalar ke_sum = Scalar(0);
        const int ne = nEdgesOnCell_v(iCell);
        for (int i = 0; i < ne; ++i) {
          const int iEdge = edgesOnCell(i, iCell) - 1;
          ke_sum += Scalar(0.25) * ke_edge(k, iEdge);
        }
        ke(k, iCell) = ke_sum * invAreaCell(iCell);
      });
  Kokkos::fence("diag::kinetic_energy_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 5: Hollingsworth KE adjustment (Requirement 7.3)
  // Part 1: Compute ke at vertices from surrounding edge ke_edge values
  // Part 2: Blend cell KE with vertex KE via kite areas
  // ──────────────────────────────────────────────────────────────────────────
  if (hollingsworth) {
    // ke_vertex temporary (nVertLevels, nVertices)
    view2d ke_vertex("ke_vertex", nVertLevels, nVertices);

    // Part 1: ke_vertex(k,iVertex) = 0.25 * invAreaTriangle(iV)
    //           * sum(ke_edge(k, edgesOnVertex(i,iV)), i=1..vertexDegree)
    Kokkos::parallel_for(
        "diag::ke_vertex",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nVertices}),
        KOKKOS_LAMBDA(const int k, const int iVertex) {
          const Scalar r = Scalar(0.25) * invAreaTriangle(iVertex);
          Scalar ke_v = Scalar(0);
          for (int i = 0; i < vertexDegree; ++i) {
            const int iEdge = edgesOnVertex(i, iVertex) - 1;
            ke_v += ke_edge(k, iEdge);
          }
          ke_vertex(k, iVertex) = ke_v * r;
        });
    Kokkos::fence("diag::ke_vertex_fence");

    // Part 2: Blend ke with ke_vertex
    // ke_fact = 1.0 - 0.375 = 0.625
    // ke(k,iCell) = ke_fact * ke(k,iCell)
    //            + sum_i((1-ke_fact)*kiteAreasOnVertex(j,iV)*ke_vertex(k,iV)*invAreaCell(iC))
    const Scalar ke_fact = Scalar(1.0) - Scalar(0.375);

    Kokkos::parallel_for(
        "diag::hollingsworth_blend",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nCells}),
        KOKKOS_LAMBDA(const int k, const int iCell) {
          ke(k, iCell) = ke_fact * ke(k, iCell);
          const Scalar r = invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);
          for (int i = 0; i < ne; ++i) {
            const int iVertex = verticesOnCell(i, iCell) - 1;
            const int j = kiteForCell(i, iCell) - 1;  // 1-based to 0-based
            ke(k, iCell) += (Scalar(1.0) - ke_fact) *
                            kiteAreasOnVertex(j, iVertex) *
                            ke_vertex(k, iVertex) * r;
          }
        });
    Kokkos::fence("diag::hollingsworth_fence");
  }  // hollingsworth

  // ──────────────────────────────────────────────────────────────────────────
  // Step 6: Tangential velocity reconstruction (Thuburn et al JCP 2009)
  // Only computed when rk_step == 3 (or if rk_step == -1, meaning no RK context)
  // v(k,iEdge) = sum_i(weightsOnEdge(i,iEdge) * u(k, edgesOnEdge(i,iEdge)))
  // ──────────────────────────────────────────────────────────────────────────
  const bool reconstruct_v = (rk_step == 3) || (rk_step < 0);

  if (reconstruct_v) {
    Kokkos::parallel_for(
        "diag::tangential_velocity",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nEdges}),
        KOKKOS_LAMBDA(const int k, const int iEdge) {
          Scalar v_sum = Scalar(0);
          const int ne = nEdgesOnEdge_v(iEdge);
          for (int i = 0; i < ne; ++i) {
            const int eoe = edgesOnEdge(i, iEdge) - 1;
            v_sum += weightsOnEdge(i, iEdge) * u(k, eoe);
          }
          v_out(k, iEdge) = v_sum;
        });
    Kokkos::fence("diag::tangential_velocity_fence");
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Step 7: Potential vorticity at vertices and edges
  // pv_vertex(k,iVertex) = fVertex(iVertex) + vorticity(k,iVertex)
  // pv_edge(k,iEdge) = 0.5 * (pv_vertex(k,v1) + pv_vertex(k,v2))
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "diag::pv_vertex",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
          {0, 0}, {nVertLevels, nVertices}),
      KOKKOS_LAMBDA(const int k, const int iVertex) {
        pv_vertex(k, iVertex) = fVertex(iVertex) + vorticity(k, iVertex);
      });
  Kokkos::fence("diag::pv_vertex_fence");

  Kokkos::parallel_for(
      "diag::pv_edge",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
          {0, 0}, {nVertLevels, nEdges}),
      KOKKOS_LAMBDA(const int k, const int iEdge) {
        const int v1 = verticesOnEdge(0, iEdge) - 1;
        const int v2 = verticesOnEdge(1, iEdge) - 1;
        pv_edge(k, iEdge) =
            Scalar(0.5) * (pv_vertex(k, v1) + pv_vertex(k, v2));
      });
  Kokkos::fence("diag::pv_edge_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 8: APVM upstream bias (Requirement 7.2)
  // Only applied when config_apvm_upwinding > 0
  // Computes pv_cell, gradPVt, gradPVn, then adjusts pv_edge
  // ──────────────────────────────────────────────────────────────────────────
  if (config_apvm_upwinding > Scalar(0)) {
    // Compute pv_cell: area-weighted average of surrounding pv_vertex
    Kokkos::parallel_for(
        "diag::pv_cell",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nCells}),
        KOKKOS_LAMBDA(const int k, const int iCell) {
          Scalar pvc = Scalar(0);
          const Scalar r = invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);
          for (int i = 0; i < ne; ++i) {
            const int iVertex = verticesOnCell(i, iCell) - 1;
            const int j = kiteForCell(i, iCell) - 1;
            pvc += kiteAreasOnVertex(j, iVertex) * pv_vertex(k, iVertex) * r;
          }
          pv_cell(k, iCell) = pvc;
        });
    Kokkos::fence("diag::pv_cell_fence");

    // Compute gradPVt, gradPVn, and apply APVM bias to pv_edge
    // gradPVt(k,iEdge) = (pv_vertex(k,v2) - pv_vertex(k,v1)) * invDvEdge
    // gradPVn(k,iEdge) = (pv_cell(k,c2) - pv_cell(k,c1)) * invDcEdge
    // pv_edge -= apvm_upwinding * dt * (v * gradPVt + u * gradPVn)
    const Scalar apvm_factor = config_apvm_upwinding * dt;

    Kokkos::parallel_for(
        "diag::apvm_bias",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
            {0, 0}, {nVertLevels, nEdges}),
        KOKKOS_LAMBDA(const int k, const int iEdge) {
          const int v1 = verticesOnEdge(0, iEdge) - 1;
          const int v2 = verticesOnEdge(1, iEdge) - 1;
          const int c1 = cellsOnEdge(0, iEdge) - 1;
          const int c2 = cellsOnEdge(1, iEdge) - 1;

          const Scalar r1 = Scalar(1.0) * invDvEdge(iEdge);
          const Scalar r2 = Scalar(1.0) * invDcEdge(iEdge);

          gradPVt(k, iEdge) = (pv_vertex(k, v2) - pv_vertex(k, v1)) * r1;
          gradPVn(k, iEdge) = (pv_cell(k, c2) - pv_cell(k, c1)) * r2;

          pv_edge(k, iEdge) -= apvm_factor *
              (v_out(k, iEdge) * gradPVt(k, iEdge) +
               u(k, iEdge) * gradPVn(k, iEdge));
        });
    Kokkos::fence("diag::apvm_fence");
  }  // config_apvm_upwinding > 0
}

// ============================================================================
// init_coupled_diagnostics implementation (Requirement 7.4)
// ============================================================================

template <class ExecSpace>
void Diagnostics_Module<ExecSpace>::init_coupled_diagnostics(
    const CoupledDiagMeshData<ExecSpace>& cmesh,
    const CoupledDiagState<ExecSpace>& state,
    CoupledDiagFields<ExecSpace>& out) const {

  const int nCells = cmesh.nCells;
  const int nEdges = cmesh.nEdges;
  const int nVertLevels = cmesh.nVertLevels;

  // Physical constants
  const Scalar rgas = diag_constants::rgas;
  const Scalar rcv = diag_constants::rcv;
  const Scalar p0 = diag_constants::p0;
  const Scalar rvord = diag_constants::rvord;

  // Capture mesh/state views for lambda capture
  const auto zz = cmesh.zz;
  const auto fzm = cmesh.fzm;
  const auto fzp = cmesh.fzp;
  const auto cellsOnEdge = cmesh.cellsOnEdge;
  const auto edgesOnCell = cmesh.edgesOnCell;
  const auto nEdgesOnCell_v = cmesh.nEdgesOnCell;
  const auto edgesOnCell_sign = cmesh.edgesOnCell_sign;
  const auto zb_cell = cmesh.zb_cell;
  const auto zb3_cell = cmesh.zb3_cell;

  const auto theta = state.theta;
  const auto rho = state.rho;
  const auto u_in = state.u;
  const auto w = state.w;
  const auto qv = state.qv;
  const auto rho_base = state.rho_base;
  const auto theta_base = state.theta_base;

  auto theta_m = out.theta_m;
  auto rho_zz = out.rho_zz;
  auto rho_p = out.rho_p;
  auto rtheta_base = out.rtheta_base;
  auto rtheta_p = out.rtheta_p;
  auto ru = out.ru;
  auto rw = out.rw;
  auto exner_out = out.exner;
  auto exner_base = out.exner_base;
  auto pressure_p = out.pressure_p;
  auto pressure_base = out.pressure_base;

  // ──────────────────────────────────────────────────────────────────────────
  // Step 1: Compute moist potential temperature and coupled density
  // theta_m(k,iCell) = theta(k,iCell) * (1 + rvord * qv(k,iCell))
  // rho_zz(k,iCell) = rho(k,iCell) / zz(k,iCell)
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "coupled_diag::theta_m_rho_zz",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        theta_m(k, iCell) = theta(k, iCell) *
            (Scalar(1.0) + rvord * qv(k, iCell));
        rho_zz(k, iCell) = rho(k, iCell) / zz(k, iCell);
      });
  Kokkos::fence("coupled_diag::theta_m_rho_zz_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 2: Compute horizontal mass flux at edges
  // ru(k,iEdge) = 0.5 * u(k,iEdge) * (rho_zz(k,cell1) + rho_zz(k,cell2))
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "coupled_diag::ru",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nEdges}),
      KOKKOS_LAMBDA(const int k, const int iEdge) {
        const int cell1 = cellsOnEdge(0, iEdge) - 1;
        const int cell2 = cellsOnEdge(1, iEdge) - 1;
        ru(k, iEdge) = Scalar(0.5) * u_in(k, iEdge) *
            (rho_zz(k, cell1) + rho_zz(k, cell2));
      });
  Kokkos::fence("coupled_diag::ru_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 3: Compute vertical mass flux rw
  // rw(0,iCell) = 0; rw(nVertLevels,iCell) = 0 (boundaries)
  // rw(k,iCell) = w(k,iCell) * (fzp(k)*rho_zz(k-1,iCell) + fzm(k)*rho_zz(k,iCell))
  //             * (fzp(k)*zz(k-1,iCell) + fzm(k)*zz(k,iCell))
  // Then subtract horizontal flux divergence contribution
  // ──────────────────────────────────────────────────────────────────────────
  // Part 3a: w-dependent piece + boundary zeros
  Kokkos::parallel_for(
      "coupled_diag::rw_w_part",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        rw(0, iCell) = Scalar(0.0);
        rw(nVertLevels, iCell) = Scalar(0.0);
        for (int k = 1; k < nVertLevels; ++k) {
          rw(k, iCell) = w(k, iCell)
              * (fzp(k) * rho_zz(k - 1, iCell) + fzm(k) * rho_zz(k, iCell))
              * (fzp(k) * zz(k - 1, iCell) + fzm(k) * zz(k, iCell));
        }
      });
  Kokkos::fence("coupled_diag::rw_w_fence");

  // Part 3b: subtract flux-divergence contribution from ru
  Kokkos::parallel_for(
      "coupled_diag::rw_flux_correction",
      Kokkos::RangePolicy<exec_space>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        const int ne = nEdgesOnCell_v(iCell);
        for (int i = 0; i < ne; ++i) {
          const int iEdge = edgesOnCell(i, iCell) - 1;
          for (int k = 1; k < nVertLevels; ++k) {
            const Scalar flux = fzm(k) * ru(k, iEdge) + fzp(k) * ru(k - 1, iEdge);
            const Scalar zz_face = fzp(k) * zz(k - 1, iCell) + fzm(k) * zz(k, iCell);
            const Scalar sign_flux = (flux >= Scalar(0.0)) ? Scalar(1.0) : Scalar(-1.0);
            rw(k, iCell) -= edgesOnCell_sign(i, iCell) *
                (zb_cell(k, i, iCell) + sign_flux * zb3_cell(k, i, iCell)) *
                flux * zz_face;
          }
        }
      });
  Kokkos::fence("coupled_diag::rw_flux_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 4: Compute rho_p = rho_zz - rho_base
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "coupled_diag::rho_p",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        rho_p(k, iCell) = rho_zz(k, iCell) - rho_base(k, iCell);
      });
  Kokkos::fence("coupled_diag::rho_p_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 5: Compute rtheta_base = theta_base * rho_base
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "coupled_diag::rtheta_base",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        rtheta_base(k, iCell) = theta_base(k, iCell) * rho_base(k, iCell);
      });
  Kokkos::fence("coupled_diag::rtheta_base_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 6: Compute rtheta_p (perturbation coupled theta)
  // rtheta_p = theta_m * rho_p + rho_base * (theta_m - theta_base)
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "coupled_diag::rtheta_p",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        rtheta_p(k, iCell) = theta_m(k, iCell) * rho_p(k, iCell)
            + rho_base(k, iCell) * (theta_m(k, iCell) - theta_base(k, iCell));
      });
  Kokkos::fence("coupled_diag::rtheta_p_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 7: Compute Exner function and base Exner function
  // exner = (zz * (rgas/p0) * (rtheta_p + rtheta_base))^rcv
  // exner_base = (zz * (rgas/p0) * rtheta_base)^rcv
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "coupled_diag::exner",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        const Scalar rtheta_m = rtheta_p(k, iCell) + rtheta_base(k, iCell);
        exner_out(k, iCell) = Kokkos::pow(
            zz(k, iCell) * (rgas / p0) * rtheta_m, rcv);
        exner_base(k, iCell) = Kokkos::pow(
            zz(k, iCell) * (rgas / p0) * rtheta_base(k, iCell), rcv);
      });
  Kokkos::fence("coupled_diag::exner_fence");

  // ──────────────────────────────────────────────────────────────────────────
  // Step 8: Compute pressure perturbation and base pressure
  // pressure_p = zz * rgas * (exner * rtheta_p + rtheta_base * (exner - exner_base))
  // pressure_base = zz * rgas * exner_base * rtheta_base
  // ──────────────────────────────────────────────────────────────────────────
  Kokkos::parallel_for(
      "coupled_diag::pressure",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0},
                                                          {nVertLevels, nCells}),
      KOKKOS_LAMBDA(const int k, const int iCell) {
        pressure_p(k, iCell) = zz(k, iCell) * rgas *
            (exner_out(k, iCell) * rtheta_p(k, iCell) +
             rtheta_base(k, iCell) *
             (exner_out(k, iCell) - exner_base(k, iCell)));
        pressure_base(k, iCell) = zz(k, iCell) * rgas *
            exner_base(k, iCell) * rtheta_base(k, iCell);
      });
  Kokkos::fence("coupled_diag::pressure_fence");
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_DIAGNOSTICS_MODULE_HPP
