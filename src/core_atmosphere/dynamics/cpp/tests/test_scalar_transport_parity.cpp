#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/scalar_transport.hpp"
#include "mpas_dycore/scalar_transport_mono.hpp"
#include "mpas_dycore/scalar.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace {

using Scalar = mpas::dycore::Scalar;
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using layout = Kokkos::LayoutLeft;

/// Parity_Tolerance: tight tolerance near machine precision (Req 14.2).
constexpr Scalar kTol = std::is_same_v<Scalar, double> ? 1e-12 : 1e-6f;

template <class T>
using View1D = Kokkos::View<T*, layout, MemSpace>;
template <class T>
using View2D = Kokkos::View<T**, layout, MemSpace>;
template <class T>
using View3D = Kokkos::View<T***, layout, MemSpace>;

/// @brief Parity comparison helper: compares a 3D field (num_scalars, nVertLevels,
/// nCells) against expected values. On divergence, fails identifying the specific
/// scalar index, level, and cell (Req 14.9).
/// @return true if all values match within Parity_Tolerance.
bool compare_scalar_field_3d(
    const View3D<Scalar>& actual,
    const View3D<Scalar>& expected,
    int num_scalars, int nVertLevels, int nCells,
    const std::string& field_name,
    std::string& diverging_info) {
  auto act_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, actual);
  auto exp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);

  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k) {
      for (int s = 0; s < num_scalars; ++s) {
        const Scalar a = act_h(s, k, c);
        const Scalar e = exp_h(s, k, c);
        const Scalar diff = std::abs(a - e);
        const Scalar scale = std::max(Scalar(1.0), std::abs(e));
        if (diff > kTol * scale + kTol) {
          diverging_info = field_name + " at (scalar=" +
              std::to_string(s) + ", k=" + std::to_string(k) +
              ", cell=" + std::to_string(c) + "): got " +
              std::to_string(a) + " expected " + std::to_string(e) +
              " diff=" + std::to_string(diff);
          return false;
        }
      }
    }
  }
  return true;
}

/// @brief Build a test mesh for parity testing.
/// 7 cells forming a hexagonal pattern, 10 edges, 4 vertical levels, 3 scalars.
/// Provides enough structure for meaningful advection and boundary tests.
struct ParityTestMesh {
  static constexpr int nCells = 7;
  static constexpr int nEdges = 10;
  static constexpr int nVertLevels = 4;
  static constexpr int num_scalars = 3;
  static constexpr int maxEdges = 4;
  static constexpr int maxAdvCellsForEdge = 4;
  static constexpr int nCellsSolve = 7;

  mpas::dycore::ScalarTransportMeshData<ExecSpace> mesh;
  mpas::dycore::MonoTransportMeshData<ExecSpace> mono_mesh;

  ParityTestMesh() {
    mesh.nCells = nCells;
    mesh.nEdges = nEdges;
    mesh.nVertLevels = nVertLevels;
    mesh.num_scalars = num_scalars;
    mesh.maxEdges = maxEdges;
    mesh.maxAdvCellsForEdge = maxAdvCellsForEdge;

    mono_mesh.nCellsSolve = nCellsSolve;
    mono_mesh.maxEdges = maxEdges;

    // Allocate all connectivity and geometry views
    mesh.cellsOnEdge = View2D<int>("cellsOnEdge", 2, nEdges);
    mesh.edgesOnCell = View2D<int>("edgesOnCell", maxEdges, nCells);
    mesh.nEdgesOnCell = View1D<int>("nEdgesOnCell", nCells);
    mesh.advCellsForEdge = View2D<int>("advCellsForEdge", maxAdvCellsForEdge, nEdges);
    mesh.nAdvCellsForEdge = View1D<int>("nAdvCellsForEdge", nEdges);
    mesh.adv_coefs = View2D<Scalar>("adv_coefs", maxAdvCellsForEdge, nEdges);
    mesh.adv_coefs_3rd = View2D<Scalar>("adv_coefs_3rd", maxAdvCellsForEdge, nEdges);
    mesh.edgesOnCell_sign = View2D<Scalar>("edgesOnCell_sign", maxEdges, nCells);
    mesh.dvEdge = View1D<Scalar>("dvEdge", nEdges);
    mesh.invAreaCell = View1D<Scalar>("invAreaCell", nCells);
    mesh.fnm = View1D<Scalar>("fnm", nVertLevels + 1);
    mesh.fnp = View1D<Scalar>("fnp", nVertLevels + 1);
    mesh.rdnw = View1D<Scalar>("rdnw", nVertLevels);
    mesh.bdyMaskCell = View1D<int>("bdyMaskCell", nCells);
    mesh.bdyMaskEdge = View1D<int>("bdyMaskEdge", nEdges);
    mono_mesh.cellsOnCell = View2D<int>("cellsOnCell", maxEdges, nCells);

    fill_mesh();
  }

  void fill_mesh() {
    // Ring topology: center cell 0 connected to ring cells 1-6 by edges 0-5.
    // Additional ring-to-ring edges: 6-9 connecting cells 1-2, 2-3, 3-4, 4-5.
    auto coe_h = Kokkos::create_mirror_view(mesh.cellsOnEdge);
    // Radial edges (center to ring): 0-5
    for (int e = 0; e < 6; ++e) {
      coe_h(0, e) = 1;       // center cell (1-based)
      coe_h(1, e) = e + 2;   // ring cell (1-based)
    }
    // Ring-to-ring edges: 6-9
    coe_h(0, 6) = 2; coe_h(1, 6) = 3;
    coe_h(0, 7) = 3; coe_h(1, 7) = 4;
    coe_h(0, 8) = 4; coe_h(1, 8) = 5;
    coe_h(0, 9) = 5; coe_h(1, 9) = 6;
    Kokkos::deep_copy(mesh.cellsOnEdge, coe_h);

    // edgesOnCell: cell 0 has 4 edges (0-3), ring cells have 2-3 edges each
    auto eoc_h = Kokkos::create_mirror_view(mesh.edgesOnCell);
    auto nec_h = Kokkos::create_mirror_view(mesh.nEdgesOnCell);
    // Cell 0: radial edges 0,1,2,3
    eoc_h(0, 0) = 1; eoc_h(1, 0) = 2; eoc_h(2, 0) = 3; eoc_h(3, 0) = 4;
    nec_h(0) = 4;
    // Cell 1: edges 0(radial), 6(ring) => 2 edges
    eoc_h(0, 1) = 1; eoc_h(1, 1) = 7; nec_h(1) = 2;
    // Cell 2: edges 1(radial), 6, 7 => 3 edges
    eoc_h(0, 2) = 2; eoc_h(1, 2) = 7; eoc_h(2, 2) = 8; nec_h(2) = 3;
    // Cell 3: edges 2(radial), 7, 8 => 3 edges
    eoc_h(0, 3) = 3; eoc_h(1, 3) = 8; eoc_h(2, 3) = 9; nec_h(3) = 3;
    // Cell 4: edges 3(radial), 8, 9 => 3 edges
    eoc_h(0, 4) = 4; eoc_h(1, 4) = 9; eoc_h(2, 4) = 10; nec_h(4) = 3;
    // Cell 5: edges 4(radial), 9 => 2 edges
    eoc_h(0, 5) = 5; eoc_h(1, 5) = 10; nec_h(5) = 2;
    // Cell 6: edge 5(radial) => 1 edge
    eoc_h(0, 6) = 6; nec_h(6) = 1;
    Kokkos::deep_copy(mesh.edgesOnCell, eoc_h);
    Kokkos::deep_copy(mesh.nEdgesOnCell, nec_h);

    // Edge signs: convention - outward from cell1 is positive
    auto ecs_h = Kokkos::create_mirror_view(mesh.edgesOnCell_sign);
    // Cell 0: all outward (positive)
    for (int i = 0; i < 4; ++i) ecs_h(i, 0) = Scalar(1.0);
    // Cell 1: edge 0 inward (-1), edge 6 outward (+1)
    ecs_h(0, 1) = Scalar(-1.0); ecs_h(1, 1) = Scalar(1.0);
    // Cell 2: edge 1 inward, edge 6 inward, edge 7 outward
    ecs_h(0, 2) = Scalar(-1.0); ecs_h(1, 2) = Scalar(-1.0);
    ecs_h(2, 2) = Scalar(1.0);
    // Cell 3: edge 2 inward, edge 7 inward, edge 8 outward
    ecs_h(0, 3) = Scalar(-1.0); ecs_h(1, 3) = Scalar(-1.0);
    ecs_h(2, 3) = Scalar(1.0);
    // Cell 4: edge 3 inward, edge 8 inward, edge 9 outward
    ecs_h(0, 4) = Scalar(-1.0); ecs_h(1, 4) = Scalar(-1.0);
    ecs_h(2, 4) = Scalar(1.0);
    // Cell 5: edge 4 inward, edge 9 inward
    ecs_h(0, 5) = Scalar(-1.0); ecs_h(1, 5) = Scalar(-1.0);
    // Cell 6: edge 5 inward
    ecs_h(0, 6) = Scalar(-1.0);
    Kokkos::deep_copy(mesh.edgesOnCell_sign, ecs_h);

    // cellsOnCell for monotonic limiter (neighbor connectivity, 1-based)
    auto coc_h = Kokkos::create_mirror_view(mono_mesh.cellsOnCell);
    // Cell 0 neighbors: cells 1-4
    coc_h(0, 0) = 2; coc_h(1, 0) = 3; coc_h(2, 0) = 4; coc_h(3, 0) = 5;
    // Cell 1: center(1) and cell 2
    coc_h(0, 1) = 1; coc_h(1, 1) = 3;
    // Cell 2: center(1), cell 1, cell 3
    coc_h(0, 2) = 1; coc_h(1, 2) = 2; coc_h(2, 2) = 4;
    // Cell 3: center(1), cell 2, cell 4
    coc_h(0, 3) = 1; coc_h(1, 3) = 3; coc_h(2, 3) = 5;
    // Cell 4: center(1), cell 3, cell 5
    coc_h(0, 4) = 1; coc_h(1, 4) = 4; coc_h(2, 4) = 6;
    // Cell 5: center(1), cell 4
    coc_h(0, 5) = 1; coc_h(1, 5) = 5;
    // Cell 6: center(1)
    coc_h(0, 6) = 1;
    Kokkos::deep_copy(mono_mesh.cellsOnCell, coc_h);

    // Advection stencil: 2-point centered (cell1, cell2)
    auto acfe_h = Kokkos::create_mirror_view(mesh.advCellsForEdge);
    auto nacfe_h = Kokkos::create_mirror_view(mesh.nAdvCellsForEdge);
    auto ac_h = Kokkos::create_mirror_view(mesh.adv_coefs);
    auto ac3_h = Kokkos::create_mirror_view(mesh.adv_coefs_3rd);
    for (int e = 0; e < nEdges; ++e) {
      nacfe_h(e) = 2;
      acfe_h(0, e) = coe_h(0, e); // cell1 (1-based)
      acfe_h(1, e) = coe_h(1, e); // cell2 (1-based)
      ac_h(0, e) = Scalar(0.5);
      ac_h(1, e) = Scalar(0.5);
      ac3_h(0, e) = Scalar(0.5);
      ac3_h(1, e) = Scalar(-0.5);
    }
    Kokkos::deep_copy(mesh.advCellsForEdge, acfe_h);
    Kokkos::deep_copy(mesh.nAdvCellsForEdge, nacfe_h);
    Kokkos::deep_copy(mesh.adv_coefs, ac_h);
    Kokkos::deep_copy(mesh.adv_coefs_3rd, ac3_h);

    // Geometry: non-uniform to exercise re-integration
    auto dv_h = Kokkos::create_mirror_view(mesh.dvEdge);
    auto ia_h = Kokkos::create_mirror_view(mesh.invAreaCell);
    for (int e = 0; e < nEdges; ++e)
      dv_h(e) = Scalar(1.0) + Scalar(0.1) * (e % 3);
    for (int c = 0; c < nCells; ++c)
      ia_h(c) = Scalar(1.0) / (Scalar(1.0) + Scalar(0.05) * c);
    Kokkos::deep_copy(mesh.dvEdge, dv_h);
    Kokkos::deep_copy(mesh.invAreaCell, ia_h);

    // Vertical weights: realistic non-uniform
    auto fnm_h = Kokkos::create_mirror_view(mesh.fnm);
    auto fnp_h = Kokkos::create_mirror_view(mesh.fnp);
    auto rdnw_h = Kokkos::create_mirror_view(mesh.rdnw);
    for (int k = 0; k <= nVertLevels; ++k) {
      fnm_h(k) = Scalar(0.5) + Scalar(0.02) * k;
      fnp_h(k) = Scalar(1.0) - fnm_h(k);
    }
    for (int k = 0; k < nVertLevels; ++k) {
      rdnw_h(k) = Scalar(1.0) / (Scalar(500.0) + Scalar(100.0) * k);
    }
    Kokkos::deep_copy(mesh.fnm, fnm_h);
    Kokkos::deep_copy(mesh.fnp, fnp_h);
    Kokkos::deep_copy(mesh.rdnw, rdnw_h);

    // No boundaries by default
    Kokkos::deep_copy(mesh.bdyMaskCell, 0);
    Kokkos::deep_copy(mesh.bdyMaskEdge, 0);
  }
};

/// @brief Compute the Reference_Model standard scalar transport output
/// analytically. Replicates the algorithm in scalar_transport.hpp step by step
/// on the host to produce the expected result for parity comparison.
///
/// This serves as the "Reference_Model" output for the component-level test.
View3D<Scalar> compute_reference_standard_transport(
    const ParityTestMesh& tm,
    const View3D<Scalar>& scalar_old_in,
    const View3D<Scalar>& scalar_new_in,
    const View2D<Scalar>& rho_zz_old_in,
    const View2D<Scalar>& rho_zz_new_in,
    const View2D<Scalar>& uhAvg_in,
    const View2D<Scalar>& wwAvg_in,
    const View3D<Scalar>& scalar_tend_in,
    Scalar dt, Scalar coef_3rd_order,
    int rk_step, int config_time_integration_order,
    bool advance_density) {

  const int nC = ParityTestMesh::nCells;
  const int nE = ParityTestMesh::nEdges;
  const int nK = ParityTestMesh::nVertLevels;
  const int nS = ParityTestMesh::num_scalars;

  // Host mirrors
  auto so = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scalar_old_in);
  auto sn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scalar_new_in);
  auto rho_old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rho_zz_old_in);
  auto rho_new = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rho_zz_new_in);
  auto uh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, uhAvg_in);
  auto ww = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, wwAvg_in);
  auto tend = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scalar_tend_in);

  // Mesh data (host)
  auto coe = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.cellsOnEdge);
  auto eoc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.edgesOnCell);
  auto nec = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.nEdgesOnCell);
  auto acfe = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.advCellsForEdge);
  auto nacfe = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.nAdvCellsForEdge);
  auto ac = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.adv_coefs);
  auto ac3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.adv_coefs_3rd);
  auto ecs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.edgesOnCell_sign);
  auto dv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.dvEdge);
  auto ia = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.invAreaCell);
  auto fnm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.fnm);
  auto fnp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.fnp);
  auto rdnw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.mesh.rdnw);

  // Density re-integration weights (Req 5.5)
  Scalar wt_new;
  if (!advance_density) {
    wt_new = Scalar(1.0);
  } else {
    if (rk_step == 1 && config_time_integration_order == 3)
      wt_new = Scalar(1.0) / Scalar(3.0);
    else if (rk_step == 1 && config_time_integration_order == 2)
      wt_new = Scalar(1.0) / Scalar(2.0);
    else if (rk_step == 2)
      wt_new = Scalar(1.0) / Scalar(2.0);
    else
      wt_new = Scalar(1.0);
  }
  const Scalar wt_old = Scalar(1.0) - wt_new;

  // Step 1: Horizontal edge scalar values (third-order upwind-biased, Req 5.3)
  std::vector<std::vector<std::vector<Scalar>>> horiz_flux(
      nS, std::vector<std::vector<Scalar>>(nK, std::vector<Scalar>(nE, Scalar(0.0))));
  for (int e = 0; e < nE; ++e) {
    const int nAdv = nacfe(e);
    for (int k = 0; k < nK; ++k) {
      const Scalar sign_u = (uh(k, e) >= Scalar(0.0)) ? Scalar(1.0) : Scalar(-1.0);
      for (int s = 0; s < nS; ++s) {
        Scalar flux_val = Scalar(0.0);
        for (int j = 0; j < nAdv; ++j) {
          const int iAdvCell = acfe(j, e) - 1;
          const Scalar sw = ac(j, e) + sign_u * ac3(j, e);
          flux_val += sw * sn(s, k, iAdvCell);
        }
        horiz_flux[s][k][e] = flux_val;
      }
    }
  }

  // Step 2: Vertical fluxes (Req 5.4)
  std::vector<std::vector<std::vector<Scalar>>> wdtn(
      nS, std::vector<std::vector<Scalar>>(nK + 1, std::vector<Scalar>(nC, Scalar(0.0))));
  for (int c = 0; c < nC; ++c) {
    for (int s = 0; s < nS; ++s) {
      wdtn[s][0][c] = Scalar(0.0);
      wdtn[s][nK][c] = Scalar(0.0);
      if (nK > 1) {
        wdtn[s][1][c] = ww(1, c) * (fnm(1) * sn(s, 1, c) + fnp(1) * sn(s, 0, c));
      }
      if (nK > 2) {
        int kk = nK - 1;
        wdtn[s][kk][c] = ww(kk, c) * (fnm(kk) * sn(s, kk, c) + fnp(kk) * sn(s, kk - 1, c));
      }
      for (int k = 2; k <= nK - 2; ++k) {
        Scalar qm2 = sn(s, k - 2, c), qm1 = sn(s, k - 1, c);
        Scalar qi = sn(s, k, c), qp1 = sn(s, k + 1, c);
        Scalar w = ww(k, c);
        Scalar flux4 = w * (Scalar(7.0) * (qi + qm1) - (qp1 + qm2)) / Scalar(12.0);
        wdtn[s][k][c] = flux4 + coef_3rd_order * std::abs(w) *
            ((qp1 - qm2) - Scalar(3.0) * (qi - qm1)) / Scalar(12.0);
      }
    }
  }

  // Step 3: Cell update (Req 5.5)
  View3D<Scalar> result("reference_result", nS, nK, nC);
  auto res_h = Kokkos::create_mirror_view(result);
  for (int c = 0; c < nC; ++c) {
    const int ne = nec(c);
    for (int k = 0; k < nK; ++k) {
      for (int s = 0; s < nS; ++s) {
        // Horizontal flux divergence
        Scalar h_tend_val = Scalar(0.0);
        for (int i = 0; i < ne; ++i) {
          const int iEdge = eoc(i, c) - 1;
          h_tend_val -= ecs(i, c) * uh(k, iEdge) * horiz_flux[s][k][iEdge];
        }
        h_tend_val *= ia(c);
        // Add physics tendency
        h_tend_val += tend(s, k, c);
        // Vertical flux divergence
        Scalar v_div = rdnw(k) * (wdtn[s][k + 1][c] - wdtn[s][k][c]);
        // Final update
        Scalar rho_inv = Scalar(1.0) /
            (wt_old * rho_old(k, c) + wt_new * rho_new(k, c));
        res_h(s, k, c) = (so(s, k, c) * rho_old(k, c) +
            dt * (h_tend_val - v_div)) * rho_inv;
      }
    }
  }
  Kokkos::deep_copy(result, res_h);
  return result;
}

// ===========================================================================
// Req 14.2: Standard scalar transport output matches Reference_Model within
//           Parity_Tolerance.
// Req 14.9: On divergence, the test SHALL fail and identify the diverging field.
// Validates: Req 5.1, 5.3, 5.4, 5.5
// ===========================================================================
TEST(ScalarTransportParity, StandardTransportMatchesReferenceModel) {
  ParityTestMesh tm;

  // Create state with spatially-varying fields
  mpas::dycore::ScalarTransportState<ExecSpace> state;
  state.scalar_old = View3D<Scalar>("scalar_old", tm.num_scalars, tm.nVertLevels, tm.nCells);
  state.scalar_new = View3D<Scalar>("scalar_new", tm.num_scalars, tm.nVertLevels, tm.nCells);
  state.rho_zz_old = View2D<Scalar>("rho_zz_old", tm.nVertLevels, tm.nCells);
  state.rho_zz_new = View2D<Scalar>("rho_zz_new", tm.nVertLevels, tm.nCells);
  state.uhAvg = View2D<Scalar>("uhAvg", tm.nVertLevels, tm.nEdges);
  state.wwAvg = View2D<Scalar>("wwAvg", tm.nVertLevels + 1, tm.nCells);
  state.scalar_tend = View3D<Scalar>("scalar_tend", tm.num_scalars, tm.nVertLevels, tm.nCells);

  // Fill with realistic spatially-varying data
  auto so_h = Kokkos::create_mirror_view(state.scalar_old);
  auto sn_h = Kokkos::create_mirror_view(state.scalar_new);
  auto rho_old_h = Kokkos::create_mirror_view(state.rho_zz_old);
  auto rho_new_h = Kokkos::create_mirror_view(state.rho_zz_new);
  auto uh_h = Kokkos::create_mirror_view(state.uhAvg);
  auto ww_h = Kokkos::create_mirror_view(state.wwAvg);
  auto tend_h = Kokkos::create_mirror_view(state.scalar_tend);

  for (int c = 0; c < tm.nCells; ++c) {
    for (int k = 0; k < tm.nVertLevels; ++k) {
      rho_old_h(k, c) = Scalar(1.0) + Scalar(0.1) * k + Scalar(0.02) * c;
      rho_new_h(k, c) = Scalar(1.05) + Scalar(0.12) * k + Scalar(0.025) * c;
      for (int s = 0; s < tm.num_scalars; ++s) {
        Scalar val = Scalar(1.0) + Scalar(0.5) * s + Scalar(0.1) * k +
            Scalar(0.05) * c + Scalar(0.01) * s * k;
        so_h(s, k, c) = val;
        sn_h(s, k, c) = val;
        tend_h(s, k, c) = Scalar(0.01) * (s + 1) * (k + 1);
      }
    }
  }
  for (int e = 0; e < tm.nEdges; ++e) {
    for (int k = 0; k < tm.nVertLevels; ++k) {
      uh_h(k, e) = Scalar(0.5) * std::sin(Scalar(e) * Scalar(0.7) + Scalar(k) * Scalar(0.3));
    }
  }
  for (int c = 0; c < tm.nCells; ++c) {
    ww_h(0, c) = Scalar(0.0);
    for (int k = 1; k < tm.nVertLevels; ++k) {
      ww_h(k, c) = Scalar(0.02) * std::cos(Scalar(c) * Scalar(0.5) + Scalar(k) * Scalar(0.4));
    }
    ww_h(tm.nVertLevels, c) = Scalar(0.0);
  }

  Kokkos::deep_copy(state.scalar_old, so_h);
  Kokkos::deep_copy(state.scalar_new, sn_h);
  Kokkos::deep_copy(state.rho_zz_old, rho_old_h);
  Kokkos::deep_copy(state.rho_zz_new, rho_new_h);
  Kokkos::deep_copy(state.uhAvg, uh_h);
  Kokkos::deep_copy(state.wwAvg, ww_h);
  Kokkos::deep_copy(state.scalar_tend, tend_h);
