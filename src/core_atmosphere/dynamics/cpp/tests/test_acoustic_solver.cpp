#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cmath>

#include "mpas_dycore/acoustic_solver.hpp"
#include "mpas_dycore/scalar.hpp"

namespace mpas {
namespace dycore {
namespace test {

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using view1d = Kokkos::View<Scalar*, Kokkos::LayoutLeft, MemSpace>;
using view2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace>;
using const_view1d = Kokkos::View<const Scalar*, Kokkos::LayoutLeft, MemSpace>;
using const_view2d = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace>;
using int_view1d = Kokkos::View<int*, Kokkos::LayoutLeft, MemSpace>;
using int_view2d = Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace>;
using const_int_view1d = Kokkos::View<const int*, Kokkos::LayoutLeft, MemSpace>;
using const_int_view2d = Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace>;

/// Helper to set up a minimal mesh with known connectivity for testing.
/// Creates a simple 1-D linear mesh: nCells cells connected by nEdges edges.
/// Each cell has exactly 2 edges (left and right).
struct MiniMesh {
  int nCells;
  int nEdges;
  int nVertLevels;
  int maxEdges;

  int_view2d cellsOnEdge;      // (2, nEdges) — 1-based
  int_view2d edgesOnCell;      // (maxEdges, nCells) — 1-based
  view2d edgesOnCell_sign;     // (maxEdges, nCells)
  int_view1d nEdgesOnCell_v;   // (nCells)
  view1d dvEdge;               // (nEdges)
  view1d invDcEdge;            // (nEdges)
  view1d invAreaCell;          // (nCells)

  MiniMesh(int nc, int nv) : nCells(nc), nEdges(nc + 1), nVertLevels(nv), maxEdges(2) {
    cellsOnEdge = int_view2d("cellsOnEdge", 2, nEdges);
    edgesOnCell = int_view2d("edgesOnCell", maxEdges, nCells);
    edgesOnCell_sign = view2d("edgesOnCell_sign", maxEdges, nCells);
    nEdgesOnCell_v = int_view1d("nEdgesOnCell", nCells);
    dvEdge = view1d("dvEdge", nEdges);
    invDcEdge = view1d("invDcEdge", nEdges);
    invAreaCell = view1d("invAreaCell", nCells);

    // Set up linear 1-D mesh connectivity
    // Edge i connects cell i-1 and cell i (1-based)
    for (int ie = 0; ie < nEdges; ++ie) {
      // Boundary edges: leftmost/rightmost connect to ghost
      cellsOnEdge(0, ie) = (ie == 0) ? 1 : ie;           // 1-based
      cellsOnEdge(1, ie) = (ie == nEdges - 1) ? nCells : ie + 1;  // 1-based
    }

    for (int ic = 0; ic < nCells; ++ic) {
      nEdgesOnCell_v(ic) = 2;
      edgesOnCell(0, ic) = ic + 1;      // left edge, 1-based
      edgesOnCell(1, ic) = ic + 2;      // right edge, 1-based
      edgesOnCell_sign(0, ic) = -1.0;   // inflow
      edgesOnCell_sign(1, ic) = 1.0;    // outflow
    }

    Kokkos::deep_copy(dvEdge, Scalar(1.0));
    Kokkos::deep_copy(invDcEdge, Scalar(1.0));
    Kokkos::deep_copy(invAreaCell, Scalar(1.0));
  }
};

/// Test: On substep 1, ru_p = dts * tend_ru and ruAvg = ru_p (Req 4.4).
TEST(AcousticSolver, EdgeUpdateSubstep1) {
  const int nCells = 4;
  const int nEdges = 5;
  const int nVertLevels = 3;
  const Scalar dts = 2.0;

  MiniMesh mesh(nCells, nVertLevels);

  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d ruAvg("ruAvg", nVertLevels, nEdges);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d exner("exner", nVertLevels, nCells);
  view2d cqu("cqu", nVertLevels, nEdges);
  view2d rho_pp("rho_pp", nVertLevels, nCells);
  view2d zxu("zxu", nVertLevels, nEdges);
  view2d tend_ru("tend_ru", nVertLevels, nEdges);
  view1d specMaskEdge("specMaskEdge", nEdges);

  // Initialize tendencies to known values
  for (int ie = 0; ie < nEdges; ++ie) {
    for (int k = 0; k < nVertLevels; ++k) {
      tend_ru(k, ie) = Scalar(k + 1) * Scalar(ie + 1);
    }
  }

  // Old ru_p and ruAvg should be overwritten on substep 1
  Kokkos::deep_copy(ru_p, Scalar(999.0));
  Kokkos::deep_copy(ruAvg, Scalar(888.0));
  Kokkos::deep_copy(specMaskEdge, Scalar(0.0));
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(exner, Scalar(1.0));

  AcousticStepParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.nCellsSolve = nCells;
  params.dts = dts;
  params.small_step = 1;

  const_view2d rtheta_c = rtheta_pp;
  const_view2d zz_c = zz;
  const_view2d exner_c = exner;
  const_view2d cqu_c = cqu;
  const_view2d rho_pp_c = rho_pp;
  const_view2d zxu_c = zxu;
  const_view2d tend_ru_c = tend_ru;
  const_view1d invDcEdge_c = mesh.invDcEdge;
  const_view1d specMaskEdge_c = specMaskEdge;
  const_int_view2d cellsOnEdge_c = mesh.cellsOnEdge;

  acoustic_step_update_edges<ExecSpace>(
      ru_p, ruAvg, rtheta_c, zz_c, exner_c, cqu_c, rho_pp_c, zxu_c,
      tend_ru_c, invDcEdge_c, specMaskEdge_c, cellsOnEdge_c, params);

  // Verify: ru_p = dts * tend_ru, ruAvg = ru_p
  const Scalar tol = 1.0e-12;
  for (int ie = 0; ie < nEdges; ++ie) {
    for (int k = 0; k < nVertLevels; ++k) {
      Scalar expected = dts * Scalar(k + 1) * Scalar(ie + 1);
      EXPECT_NEAR(ru_p(k, ie), expected, tol)
          << "ru_p mismatch at edge=" << ie << " k=" << k;
      EXPECT_NEAR(ruAvg(k, ie), expected, tol)
          << "ruAvg mismatch at edge=" << ie << " k=" << k;
    }
  }
}

/// Test: On substep > 1, ru_p accumulates tendency and pressure gradient (Req 4.1).
TEST(AcousticSolver, EdgeUpdateSubstepGt1) {
  const int nCells = 4;
  const int nEdges = 5;
  const int nVertLevels = 3;
  const Scalar dts = 1.0;

  MiniMesh mesh(nCells, nVertLevels);

  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d ruAvg("ruAvg", nVertLevels, nEdges);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d exner("exner", nVertLevels, nCells);
  view2d cqu("cqu", nVertLevels, nEdges);
  view2d rho_pp("rho_pp", nVertLevels, nCells);
  view2d zxu("zxu", nVertLevels, nEdges);
  view2d tend_ru("tend_ru", nVertLevels, nEdges);
  view1d specMaskEdge("specMaskEdge", nEdges);

  // Set initial ru_p to 1.0 everywhere
  Kokkos::deep_copy(ru_p, Scalar(1.0));
  // ruAvg starts at 5.0
  Kokkos::deep_copy(ruAvg, Scalar(5.0));
  // tend_ru = 2.0 everywhere
  Kokkos::deep_copy(tend_ru, Scalar(2.0));
  // Set rtheta_pp, rho_pp to zero (no pressure gradient)
  Kokkos::deep_copy(rtheta_pp, Scalar(0.0));
  Kokkos::deep_copy(rho_pp, Scalar(0.0));
  Kokkos::deep_copy(zxu, Scalar(0.0));
  // zz and exner are 1.0
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(exner, Scalar(1.0));
  Kokkos::deep_copy(cqu, Scalar(1.0));
  Kokkos::deep_copy(specMaskEdge, Scalar(0.0));

  AcousticStepParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.nCellsSolve = nCells;
  params.dts = dts;
  params.small_step = 2;  // Not first substep

  const_view2d rtheta_c = rtheta_pp;
  const_view2d zz_c = zz;
  const_view2d exner_c = exner;
  const_view2d cqu_c = cqu;
  const_view2d rho_pp_c = rho_pp;
  const_view2d zxu_c = zxu;
  const_view2d tend_ru_c = tend_ru;
  const_view1d invDcEdge_c = mesh.invDcEdge;
  const_view1d specMaskEdge_c = specMaskEdge;
  const_int_view2d cellsOnEdge_c = mesh.cellsOnEdge;

  acoustic_step_update_edges<ExecSpace>(
      ru_p, ruAvg, rtheta_c, zz_c, exner_c, cqu_c, rho_pp_c, zxu_c,
      tend_ru_c, invDcEdge_c, specMaskEdge_c, cellsOnEdge_c, params);

  // With zero rtheta_pp and rho_pp, pgrad = 0.
  // So: ru_p = 1.0 + dts*(2.0 - 0) = 3.0
  //     ruAvg = 5.0 + 3.0 = 8.0
  const Scalar tol = 1.0e-12;
  for (int ie = 0; ie < nEdges; ++ie) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(ru_p(k, ie), 3.0, tol)
          << "ru_p mismatch at edge=" << ie << " k=" << k;
      EXPECT_NEAR(ruAvg(k, ie), 8.0, tol)
          << "ruAvg mismatch at edge=" << ie << " k=" << k;
    }
  }
}

/// Test: Cell update zeros accumulated fields on substep 1 (Req 4.4).
TEST(AcousticSolver, CellUpdateZerosOnSubstep1) {
  const int nCells = 2;
  const int nEdges = 3;
  const int nVertLevels = 4;

  MiniMesh mesh(nCells, nVertLevels);

  // Allocate all fields
  view2d rw_p("rw_p", nVertLevels + 1, nCells);
  view2d rho_pp("rho_pp", nVertLevels, nCells);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d wwAvg("wwAvg", nVertLevels + 1, nCells);
  view2d rtheta_pp_old("rtheta_pp_old", nVertLevels, nCells);
  view2d rho_zz("rho_zz", nVertLevels, nCells);
  view2d theta_m("theta_m", nVertLevels, nCells);
  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d tend_rho("tend_rho", nVertLevels, nCells);
  view2d tend_rt("tend_rt", nVertLevels, nCells);
  view2d tend_rw("tend_rw", nVertLevels + 1, nCells);
  view2d cofwt("cofwt", nVertLevels, nCells);
  view2d coftz("coftz", nVertLevels + 1, nCells);
  view2d cofwr("cofwr", nVertLevels, nCells);
  view2d cofwz("cofwz", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d a_tri("a_tri", nVertLevels, nCells);
  view2d alpha_tri("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri("gamma_tri", nVertLevels, nCells);
  view2d dss("dss", nVertLevels, nCells);
  view2d w_full("w", nVertLevels + 1, nCells);
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d cofrz_v("cofrz", nVertLevels);
  view1d etp_v("etp", nVertLevels);
  view1d etm_v("etm", nVertLevels);
  view1d ewp_v("ewp", nVertLevels + 1);
  view1d ewm_v("ewm", nVertLevels + 1);
  view1d specMaskCell("specMaskCell", nCells);

  // Initialize perturbation fields to non-zero
  Kokkos::deep_copy(rw_p, Scalar(99.0));
  Kokkos::deep_copy(rho_pp, Scalar(88.0));
  Kokkos::deep_copy(rtheta_pp, Scalar(77.0));
  Kokkos::deep_copy(wwAvg, Scalar(66.0));

  // Zero out tendencies and all other fields (no solve, just verify zeroing)
  Kokkos::deep_copy(tend_rho, Scalar(0.0));
  Kokkos::deep_copy(tend_rt, Scalar(0.0));
  Kokkos::deep_copy(tend_rw, Scalar(0.0));
  Kokkos::deep_copy(ru_p, Scalar(0.0));
  Kokkos::deep_copy(cofwt, Scalar(0.0));
  Kokkos::deep_copy(coftz, Scalar(0.0));
  Kokkos::deep_copy(cofwr, Scalar(0.0));
  Kokkos::deep_copy(cofwz, Scalar(0.0));
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(a_tri, Scalar(0.0));
  Kokkos::deep_copy(alpha_tri, Scalar(1.0));
  Kokkos::deep_copy(gamma_tri, Scalar(0.0));
  Kokkos::deep_copy(dss, Scalar(0.0));
  Kokkos::deep_copy(w_full, Scalar(0.0));
  Kokkos::deep_copy(rw_save, Scalar(0.0));
  Kokkos::deep_copy(rw_base, Scalar(0.0));
  Kokkos::deep_copy(fzm_v, Scalar(0.5));
  Kokkos::deep_copy(fzp_v, Scalar(0.5));
  Kokkos::deep_copy(rdzw_v, Scalar(1.0));
  Kokkos::deep_copy(cofrz_v, Scalar(0.0));
  Kokkos::deep_copy(etp_v, Scalar(0.5));
  Kokkos::deep_copy(etm_v, Scalar(0.5));
  Kokkos::deep_copy(ewp_v, Scalar(0.5));
  Kokkos::deep_copy(ewm_v, Scalar(0.5));
  Kokkos::deep_copy(rho_zz, Scalar(1.0));
  Kokkos::deep_copy(theta_m, Scalar(300.0));
  Kokkos::deep_copy(specMaskCell, Scalar(0.0));

  AcousticStepParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.nCellsSolve = nCells;
  params.maxEdges = mesh.maxEdges;
  params.dts = 1.0;
  params.small_step = 1;

  acoustic_step_update_cells<ExecSpace>(
      rw_p, rho_pp, rtheta_pp, wwAvg, rtheta_pp_old,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(ru_p),
      const_view2d(tend_rho), const_view2d(tend_rt), const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign), const_view1d(specMaskCell),
      params);

  // After substep 1 with zero tendencies and zero ru_p,
  // all perturbation fields and accumulators should be zero
  const Scalar tol = 1.0e-12;
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(rho_pp(k, ic), 0.0, tol)
          << "rho_pp not zeroed at cell=" << ic << " k=" << k;
      EXPECT_NEAR(rtheta_pp(k, ic), 0.0, tol)
          << "rtheta_pp not zeroed at cell=" << ic << " k=" << k;
      EXPECT_NEAR(rw_p(k, ic), 0.0, tol)
          << "rw_p not zeroed at cell=" << ic << " k=" << k;
      EXPECT_NEAR(wwAvg(k, ic), 0.0, tol)
          << "wwAvg not zeroed at cell=" << ic << " k=" << k;
    }
    EXPECT_NEAR(rw_p(nVertLevels, ic), 0.0, tol)
        << "rw_p top boundary not zeroed at cell=" << ic;
    EXPECT_NEAR(wwAvg(nVertLevels, ic), 0.0, tol)
        << "wwAvg top boundary not zeroed at cell=" << ic;
  }
}

/// Test: Specified-zone cells use the simple tendency update path (Req 4.6).
TEST(AcousticSolver, SpecifiedZoneTendencyUpdate) {
  const int nCells = 2;
  const int nEdges = 3;
  const int nVertLevels = 3;

  MiniMesh mesh(nCells, nVertLevels);

  view2d rw_p("rw_p", nVertLevels + 1, nCells);
  view2d rho_pp("rho_pp", nVertLevels, nCells);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d wwAvg("wwAvg", nVertLevels + 1, nCells);
  view2d rtheta_pp_old("rtheta_pp_old", nVertLevels, nCells);
  view2d rho_zz("rho_zz", nVertLevels, nCells);
  view2d theta_m("theta_m", nVertLevels, nCells);
  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d tend_rho("tend_rho", nVertLevels, nCells);
  view2d tend_rt("tend_rt", nVertLevels, nCells);
  view2d tend_rw("tend_rw", nVertLevels + 1, nCells);
  view2d cofwt("cofwt", nVertLevels, nCells);
  view2d coftz("coftz", nVertLevels + 1, nCells);
  view2d cofwr("cofwr", nVertLevels, nCells);
  view2d cofwz("cofwz", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d a_tri("a_tri", nVertLevels, nCells);
  view2d alpha_tri("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri("gamma_tri", nVertLevels, nCells);
  view2d dss("dss", nVertLevels, nCells);
  view2d w_full("w", nVertLevels + 1, nCells);
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d cofrz_v("cofrz", nVertLevels);
  view1d etp_v("etp", nVertLevels);
  view1d etm_v("etm", nVertLevels);
  view1d ewp_v("ewp", nVertLevels + 1);
  view1d ewm_v("ewm", nVertLevels + 1);
  view1d specMaskCell("specMaskCell", nCells);

  // Mark both cells as specified zone
  Kokkos::deep_copy(specMaskCell, Scalar(1.0));

  // Set initial state: rho_pp=1, rtheta_pp=2, rw_p=3
  Kokkos::deep_copy(rho_pp, Scalar(1.0));
  Kokkos::deep_copy(rtheta_pp, Scalar(2.0));
  Kokkos::deep_copy(rw_p, Scalar(3.0));
  Kokkos::deep_copy(wwAvg, Scalar(0.0));

  // Set tendencies
  Kokkos::deep_copy(tend_rho, Scalar(0.5));
  Kokkos::deep_copy(tend_rt, Scalar(0.25));
  Kokkos::deep_copy(tend_rw, Scalar(0.1));

  // Other fields: initialize to prevent issues
  Kokkos::deep_copy(ru_p, Scalar(0.0));
  Kokkos::deep_copy(cofwt, Scalar(0.0));
  Kokkos::deep_copy(coftz, Scalar(0.0));
  Kokkos::deep_copy(cofwr, Scalar(0.0));
  Kokkos::deep_copy(cofwz, Scalar(0.0));
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(a_tri, Scalar(0.0));
  Kokkos::deep_copy(alpha_tri, Scalar(1.0));
  Kokkos::deep_copy(gamma_tri, Scalar(0.0));
  Kokkos::deep_copy(dss, Scalar(0.0));
  Kokkos::deep_copy(w_full, Scalar(0.0));
  Kokkos::deep_copy(rw_save, Scalar(0.0));
  Kokkos::deep_copy(rw_base, Scalar(0.0));
  Kokkos::deep_copy(fzm_v, Scalar(0.5));
  Kokkos::deep_copy(fzp_v, Scalar(0.5));
  Kokkos::deep_copy(rdzw_v, Scalar(1.0));
  Kokkos::deep_copy(cofrz_v, Scalar(0.0));
  Kokkos::deep_copy(etp_v, Scalar(0.5));
  Kokkos::deep_copy(etm_v, Scalar(0.5));
  Kokkos::deep_copy(ewp_v, Scalar(0.5));
  Kokkos::deep_copy(ewm_v, Scalar(0.5));
  Kokkos::deep_copy(rho_zz, Scalar(1.0));
  Kokkos::deep_copy(theta_m, Scalar(300.0));

  const Scalar dts = 2.0;
  AcousticStepParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.nCellsSolve = nCells;
  params.maxEdges = mesh.maxEdges;
  params.dts = dts;
  params.small_step = 2;  // Not first substep (so no zeroing)

  acoustic_step_update_cells<ExecSpace>(
      rw_p, rho_pp, rtheta_pp, wwAvg, rtheta_pp_old,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(ru_p),
      const_view2d(tend_rho), const_view2d(tend_rt), const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign), const_view1d(specMaskCell),
      params);

  // In specified zone: rho_pp += dts*tend_rho, rtheta_pp += dts*tend_rt,
  // rw_p += dts*tend_rw, wwAvg += ewp*rw_p
  const Scalar tol = 1.0e-12;
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(rho_pp(k, ic), 1.0 + dts * 0.5, tol)
          << "Spec zone rho_pp at cell=" << ic << " k=" << k;
      EXPECT_NEAR(rtheta_pp(k, ic), 2.0 + dts * 0.25, tol)
          << "Spec zone rtheta_pp at cell=" << ic << " k=" << k;
      // rw_p(k) = 3.0 + dts * 0.1 = 3.2
      EXPECT_NEAR(rw_p(k, ic), 3.0 + dts * 0.1, tol)
          << "Spec zone rw_p at cell=" << ic << " k=" << k;
      // wwAvg(k) += ewp(k) * rw_p(k) = 0 + 0.5 * 3.2 = 1.6
      EXPECT_NEAR(wwAvg(k, ic), 0.5 * (3.0 + dts * 0.1), tol)
          << "Spec zone wwAvg at cell=" << ic << " k=" << k;
    }
  }
}

/// Test: 3-D divergence damping modifies ru_p based on rtheta_pp change (Req 4.7).
TEST(AcousticSolver, DivergenceDamping3D) {
  const int nCells = 4;
  const int nEdges = 5;
  const int nVertLevels = 3;

  MiniMesh mesh(nCells, nVertLevels);

  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d rtheta_pp_old("rtheta_pp_old", nVertLevels, nCells);
  view2d theta_m("theta_m", nVertLevels, nCells);
  view1d specMaskEdge("specMaskEdge", nEdges);

  // Initial ru_p = 0
  Kokkos::deep_copy(ru_p, Scalar(0.0));
  // theta_m = 300 everywhere
  Kokkos::deep_copy(theta_m, Scalar(300.0));
  // No spec zone masking
  Kokkos::deep_copy(specMaskEdge, Scalar(0.0));

  // Set up a rtheta_pp pattern with spatial gradient:
  // rtheta_pp(k,iCell) = iCell * 1.0
  // rtheta_pp_old = 0 everywhere
  Kokkos::deep_copy(rtheta_pp_old, Scalar(0.0));
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k < nVertLevels; ++k) {
      rtheta_pp(k, ic) = Scalar(ic + 1);
    }
  }

  DivergenceDampingParams dparams;
  dparams.nVertLevels = nVertLevels;
  dparams.nEdges = nEdges;
  dparams.nCellsSolve = nCells;
  dparams.dts = 1.0;
  dparams.smdiv = 0.1;
  dparams.config_len_disp = 1000.0;

  divergence_damping_3d<ExecSpace>(
      ru_p, const_view2d(rtheta_pp), const_view2d(rtheta_pp_old),
      const_view2d(theta_m), const_view1d(specMaskEdge),
      const_int_view2d(mesh.cellsOnEdge), dparams);

  // coef_divdamp = 2 * 0.1 * 1000 * (1/1.0) = 200
  // For an edge connecting cell1 and cell2:
  //   divCell1 = -(rtheta_pp(cell1) - rtheta_pp_old(cell1)) = -(cell1+1)
  //   divCell2 = -(rtheta_pp(cell2) - rtheta_pp_old(cell2)) = -(cell2+1)
  //   ru_p += 200 * (divCell2 - divCell1) / (300 + 300)
  //         = 200 * (-(cell2+1) + (cell1+1)) / 600
  //         = 200 * (cell1 - cell2) / 600
  //
  // For interior edges: cell2 - cell1 = 1 (our linear mesh)
  //   ru_p = 200 * (-1) / 600 = -1/3

  const Scalar tol = 1.0e-12;
  const Scalar coef = 200.0;
  // Check an interior edge (e.g., edge 1 connects cell 0 and cell 1, 0-based)
  for (int ie = 1; ie < nEdges - 1; ++ie) {
    const int c1 = mesh.cellsOnEdge(0, ie) - 1;
    const int c2 = mesh.cellsOnEdge(1, ie) - 1;
    Scalar divC1 = -(rtheta_pp(0, c1) - rtheta_pp_old(0, c1));
    Scalar divC2 = -(rtheta_pp(0, c2) - rtheta_pp_old(0, c2));
    Scalar expected = coef * (divC2 - divC1) / (Scalar(300.0) + Scalar(300.0));
    EXPECT_NEAR(ru_p(0, ie), expected, tol)
        << "Divergence damping mismatch at edge=" << ie;
  }
}

/// Test: Specified zone mask suppresses divergence damping (Req 4.6).
TEST(AcousticSolver, DivergenceDampingSpecZoneMask) {
  const int nCells = 4;
  const int nEdges = 5;
  const int nVertLevels = 3;

  MiniMesh mesh(nCells, nVertLevels);

  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d rtheta_pp_old("rtheta_pp_old", nVertLevels, nCells);
  view2d theta_m("theta_m", nVertLevels, nCells);
  view1d specMaskEdge("specMaskEdge", nEdges);

  Kokkos::deep_copy(ru_p, Scalar(0.0));
  Kokkos::deep_copy(theta_m, Scalar(300.0));
  // Set spec zone mask = 1 on all edges (should zero out the damping)
  Kokkos::deep_copy(specMaskEdge, Scalar(1.0));

  // Non-zero gradient in rtheta_pp
  Kokkos::deep_copy(rtheta_pp_old, Scalar(0.0));
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k < nVertLevels; ++k) {
      rtheta_pp(k, ic) = Scalar(ic + 1) * 10.0;
    }
  }

  DivergenceDampingParams dparams;
  dparams.nVertLevels = nVertLevels;
  dparams.nEdges = nEdges;
  dparams.nCellsSolve = nCells;
  dparams.dts = 1.0;
  dparams.smdiv = 0.1;
  dparams.config_len_disp = 1000.0;

  divergence_damping_3d<ExecSpace>(
      ru_p, const_view2d(rtheta_pp), const_view2d(rtheta_pp_old),
      const_view2d(theta_m), const_view1d(specMaskEdge),
      const_int_view2d(mesh.cellsOnEdge), dparams);

  // With specZoneMaskEdge = 1, mask = 1 - 1 = 0, so ru_p should remain 0
  const Scalar tol = 1.0e-14;
  for (int ie = 0; ie < nEdges; ++ie) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(ru_p(k, ie), 0.0, tol)
          << "Spec zone mask should zero damping at edge=" << ie << " k=" << k;
    }
  }
}

/// Test: ruAvg accumulates across multiple substeps (Req 4.2).
TEST(AcousticSolver, RuAvgAccumulatesAcrossSubsteps) {
  const int nCells = 2;
  const int nEdges = 3;
  const int nVertLevels = 2;
  const Scalar dts = 1.0;

  MiniMesh mesh(nCells, nVertLevels);

  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d ruAvg("ruAvg", nVertLevels, nEdges);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d exner("exner", nVertLevels, nCells);
  view2d cqu("cqu", nVertLevels, nEdges);
  view2d rho_pp("rho_pp", nVertLevels, nCells);
  view2d zxu("zxu", nVertLevels, nEdges);
  view2d tend_ru("tend_ru", nVertLevels, nEdges);
  view1d specMaskEdge("specMaskEdge", nEdges);

  // Zero pressure gradient fields
  Kokkos::deep_copy(rtheta_pp, Scalar(0.0));
  Kokkos::deep_copy(rho_pp, Scalar(0.0));
  Kokkos::deep_copy(zxu, Scalar(0.0));
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(exner, Scalar(1.0));
  Kokkos::deep_copy(cqu, Scalar(1.0));
  Kokkos::deep_copy(specMaskEdge, Scalar(0.0));

  // Constant tendency = 1.0
  Kokkos::deep_copy(tend_ru, Scalar(1.0));

  AcousticStepParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.nCellsSolve = nCells;
  params.dts = dts;

  // Substep 1: ru_p = 1, ruAvg = 1
  params.small_step = 1;
  acoustic_step_update_edges<ExecSpace>(
      ru_p, ruAvg, const_view2d(rtheta_pp), const_view2d(zz),
      const_view2d(exner), const_view2d(cqu), const_view2d(rho_pp),
      const_view2d(zxu), const_view2d(tend_ru), const_view1d(mesh.invDcEdge),
      const_view1d(specMaskEdge), const_int_view2d(mesh.cellsOnEdge), params);

  // Substep 2: ru_p = 1 + 1 = 2, ruAvg = 1 + 2 = 3
  params.small_step = 2;
  acoustic_step_update_edges<ExecSpace>(
      ru_p, ruAvg, const_view2d(rtheta_pp), const_view2d(zz),
      const_view2d(exner), const_view2d(cqu), const_view2d(rho_pp),
      const_view2d(zxu), const_view2d(tend_ru), const_view1d(mesh.invDcEdge),
      const_view1d(specMaskEdge), const_int_view2d(mesh.cellsOnEdge), params);

  // Substep 3: ru_p = 2 + 1 = 3, ruAvg = 3 + 3 = 6
  params.small_step = 3;
  acoustic_step_update_edges<ExecSpace>(
      ru_p, ruAvg, const_view2d(rtheta_pp), const_view2d(zz),
      const_view2d(exner), const_view2d(cqu), const_view2d(rho_pp),
      const_view2d(zxu), const_view2d(tend_ru), const_view1d(mesh.invDcEdge),
      const_view1d(specMaskEdge), const_int_view2d(mesh.cellsOnEdge), params);

  const Scalar tol = 1.0e-12;
  for (int ie = 0; ie < nEdges; ++ie) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(ru_p(k, ie), 3.0, tol)
          << "ru_p after 3 substeps at edge=" << ie << " k=" << k;
      EXPECT_NEAR(ruAvg(k, ie), 6.0, tol)
          << "ruAvg after 3 substeps at edge=" << ie << " k=" << k;
    }
  }
}

/// Test: Interior cell update with non-zero tendencies on substep > 1 (Req 4.1).
/// Verifies rho_pp, rtheta_pp, and rw_p update via the vertically-implicit system.
TEST(AcousticSolver, InteriorCellUpdateSubstepGt1) {
  const int nCells = 1;
  const int nEdges = 2;
  const int nVertLevels = 3;

  MiniMesh mesh(nCells, nVertLevels);

  view2d rw_p("rw_p", nVertLevels + 1, nCells);
  view2d rho_pp("rho_pp", nVertLevels, nCells);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d wwAvg("wwAvg", nVertLevels + 1, nCells);
  view2d rtheta_pp_old("rtheta_pp_old", nVertLevels, nCells);
  view2d rho_zz("rho_zz", nVertLevels, nCells);
  view2d theta_m("theta_m", nVertLevels, nCells);
  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d tend_rho("tend_rho", nVertLevels, nCells);
  view2d tend_rt("tend_rt", nVertLevels, nCells);
  view2d tend_rw("tend_rw", nVertLevels + 1, nCells);
  view2d cofwt("cofwt", nVertLevels, nCells);
  view2d coftz("coftz", nVertLevels + 1, nCells);
  view2d cofwr("cofwr", nVertLevels, nCells);
  view2d cofwz("cofwz", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d a_tri("a_tri", nVertLevels, nCells);
  view2d alpha_tri("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri("gamma_tri", nVertLevels, nCells);
  view2d dss("dss", nVertLevels, nCells);
  view2d w_full("w", nVertLevels + 1, nCells);
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d cofrz_v("cofrz", nVertLevels);
  view1d etp_v("etp", nVertLevels);
  view1d etm_v("etm", nVertLevels);
  view1d ewp_v("ewp", nVertLevels + 1);
  view1d ewm_v("ewm", nVertLevels + 1);
  view1d specMaskCell("specMaskCell", nCells);

  // Interior cell (not specified zone)
  Kokkos::deep_copy(specMaskCell, Scalar(0.0));

  // Initial perturbation state
  Kokkos::deep_copy(rho_pp, Scalar(0.5));
  Kokkos::deep_copy(rtheta_pp, Scalar(1.0));
  Kokkos::deep_copy(rw_p, Scalar(0.0));
  Kokkos::deep_copy(wwAvg, Scalar(0.0));

  // Nonzero tendencies
  Kokkos::deep_copy(tend_rho, Scalar(0.1));
  Kokkos::deep_copy(tend_rt, Scalar(0.2));
  Kokkos::deep_copy(tend_rw, Scalar(0.0));

  // Zero ru_p means no horizontal flux divergence
  Kokkos::deep_copy(ru_p, Scalar(0.0));

  // Simple vertical coefficients: identity-like tridiagonal
  Kokkos::deep_copy(cofwt, Scalar(0.0));
  Kokkos::deep_copy(coftz, Scalar(0.0));
  Kokkos::deep_copy(cofwr, Scalar(0.0));
  Kokkos::deep_copy(cofwz, Scalar(0.0));
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(a_tri, Scalar(0.0));
  Kokkos::deep_copy(alpha_tri, Scalar(1.0));
  Kokkos::deep_copy(gamma_tri, Scalar(0.0));
  Kokkos::deep_copy(dss, Scalar(0.0));
  Kokkos::deep_copy(w_full, Scalar(0.0));
  Kokkos::deep_copy(rw_save, Scalar(0.0));
  Kokkos::deep_copy(rw_base, Scalar(0.0));
  Kokkos::deep_copy(fzm_v, Scalar(0.5));
  Kokkos::deep_copy(fzp_v, Scalar(0.5));
  Kokkos::deep_copy(rdzw_v, Scalar(0.0));
  Kokkos::deep_copy(cofrz_v, Scalar(0.0));
  // ewp and ewm = 0.5
  Kokkos::deep_copy(etp_v, Scalar(0.5));
  Kokkos::deep_copy(etm_v, Scalar(0.5));
  Kokkos::deep_copy(ewp_v, Scalar(0.5));
  Kokkos::deep_copy(ewm_v, Scalar(0.5));
  Kokkos::deep_copy(rho_zz, Scalar(1.0));
  Kokkos::deep_copy(theta_m, Scalar(300.0));

  const Scalar dts = 1.0;
  AcousticStepParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.nCellsSolve = nCells;
  params.maxEdges = mesh.maxEdges;
  params.dts = dts;
  params.small_step = 2;

  acoustic_step_update_cells<ExecSpace>(
      rw_p, rho_pp, rtheta_pp, wwAvg, rtheta_pp_old,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(ru_p),
      const_view2d(tend_rho), const_view2d(tend_rt), const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign), const_view1d(specMaskCell),
      params);

  // With zero ru_p, cofwt=coftz=cofwr=cofwz=0, cofrz=0, rdzw=0:
  //   rs[k] = rho_pp(k) + dts*tend_rho(k) + 0 - 0 = 0.5 + 1.0*0.1 = 0.6
  //   ts[k] = rtheta_pp(k) + dts*tend_rt(k) + 0 - 0 = 1.0 + 1.0*0.2 = 1.2
  //   rw_p(k) stays 0 (tend_rw=0, all coefs=0)
  //   Tridiag with identity (alpha=1, gamma=0, a=0) leaves rw_p=0
  //   No damping (dss=0)
  //   rho_pp(k) = rs[k] - dts*cofrz(k)*(ewp(k+1)*rw_p(k+1)-ewp(k)*rw_p(k)) = 0.6
  //   rtheta_pp(k) = ts[k] - dts*rdzw(k)*(ewp(k+1)*coftz(k+1)*rw_p(k+1)
  //                          -ewp(k)*coftz(k)*rw_p(k)) = 1.2
  const Scalar tol = 1.0e-12;
  for (int k = 0; k < nVertLevels; ++k) {
    EXPECT_NEAR(rho_pp(k, 0), 0.6, tol)
        << "rho_pp diverges from reference at k=" << k;
    EXPECT_NEAR(rtheta_pp(k, 0), 1.2, tol)
        << "rtheta_pp diverges from reference at k=" << k;
    EXPECT_NEAR(rw_p(k, 0), 0.0, tol)
        << "rw_p diverges from reference at k=" << k;
  }
}

/// Test: wwAvg accumulates vertical mass flux across substeps (Req 4.2).
/// Uses the specified-zone path for simplicity (predictable wwAvg formula).
TEST(AcousticSolver, WwAvgAccumulatesAcrossSubsteps) {
  const int nCells = 1;
  const int nEdges = 2;
  const int nVertLevels = 3;

  MiniMesh mesh(nCells, nVertLevels);

  view2d rw_p("rw_p", nVertLevels + 1, nCells);
  view2d rho_pp("rho_pp", nVertLevels, nCells);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d wwAvg("wwAvg", nVertLevels + 1, nCells);
  view2d rtheta_pp_old("rtheta_pp_old", nVertLevels, nCells);
  view2d rho_zz("rho_zz", nVertLevels, nCells);
  view2d theta_m("theta_m", nVertLevels, nCells);
  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d tend_rho("tend_rho", nVertLevels, nCells);
  view2d tend_rt("tend_rt", nVertLevels, nCells);
  view2d tend_rw("tend_rw", nVertLevels + 1, nCells);
  view2d cofwt("cofwt", nVertLevels, nCells);
  view2d coftz("coftz", nVertLevels + 1, nCells);
  view2d cofwr("cofwr", nVertLevels, nCells);
  view2d cofwz("cofwz", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d a_tri("a_tri", nVertLevels, nCells);
  view2d alpha_tri("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri("gamma_tri", nVertLevels, nCells);
  view2d dss("dss", nVertLevels, nCells);
  view2d w_full("w", nVertLevels + 1, nCells);
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d cofrz_v("cofrz", nVertLevels);
  view1d etp_v("etp", nVertLevels);
  view1d etm_v("etm", nVertLevels);
  view1d ewp_v("ewp", nVertLevels + 1);
  view1d ewm_v("ewm", nVertLevels + 1);
  view1d specMaskCell("specMaskCell", nCells);

  // Mark as specified zone for predictable wwAvg accumulation
  Kokkos::deep_copy(specMaskCell, Scalar(1.0));

  // Initial state
  Kokkos::deep_copy(rho_pp, Scalar(0.0));
  Kokkos::deep_copy(rtheta_pp, Scalar(0.0));
  Kokkos::deep_copy(rw_p, Scalar(0.0));
  Kokkos::deep_copy(wwAvg, Scalar(0.0));

  // Set tend_rw = 1.0 so rw_p increments by dts each substep
  Kokkos::deep_copy(tend_rho, Scalar(0.0));
  Kokkos::deep_copy(tend_rt, Scalar(0.0));
  Kokkos::deep_copy(tend_rw, Scalar(1.0));
  Kokkos::deep_copy(ru_p, Scalar(0.0));

  // ewp = 0.5 for all levels
  Kokkos::deep_copy(ewp_v, Scalar(0.5));
  Kokkos::deep_copy(ewm_v, Scalar(0.5));
  Kokkos::deep_copy(etp_v, Scalar(0.5));
  Kokkos::deep_copy(etm_v, Scalar(0.5));
  Kokkos::deep_copy(cofwt, Scalar(0.0));
  Kokkos::deep_copy(coftz, Scalar(0.0));
  Kokkos::deep_copy(cofwr, Scalar(0.0));
  Kokkos::deep_copy(cofwz, Scalar(0.0));
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(a_tri, Scalar(0.0));
  Kokkos::deep_copy(alpha_tri, Scalar(1.0));
  Kokkos::deep_copy(gamma_tri, Scalar(0.0));
  Kokkos::deep_copy(dss, Scalar(0.0));
  Kokkos::deep_copy(w_full, Scalar(0.0));
  Kokkos::deep_copy(rw_save, Scalar(0.0));
  Kokkos::deep_copy(rw_base, Scalar(0.0));
  Kokkos::deep_copy(fzm_v, Scalar(0.5));
  Kokkos::deep_copy(fzp_v, Scalar(0.5));
  Kokkos::deep_copy(rdzw_v, Scalar(0.0));
  Kokkos::deep_copy(cofrz_v, Scalar(0.0));
  Kokkos::deep_copy(rho_zz, Scalar(1.0));
  Kokkos::deep_copy(theta_m, Scalar(300.0));

  const Scalar dts = 1.0;
  AcousticStepParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.nCellsSolve = nCells;
  params.maxEdges = mesh.maxEdges;
  params.dts = dts;

  // Substep 1: zeros then updates.
  // rw_p zeroed to 0, then rw_p += dts*tend_rw = 1.0
  // wwAvg zeroed to 0, then wwAvg += ewp*rw_p = 0.5*1.0 = 0.5
  params.small_step = 1;
  acoustic_step_update_cells<ExecSpace>(
      rw_p, rho_pp, rtheta_pp, wwAvg, rtheta_pp_old,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(ru_p),
      const_view2d(tend_rho), const_view2d(tend_rt), const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign), const_view1d(specMaskCell),
      params);

  // Substep 2: rw_p = 1.0 + 1.0 = 2.0; wwAvg = 0.5 + 0.5*2.0 = 1.5
  params.small_step = 2;
  acoustic_step_update_cells<ExecSpace>(
      rw_p, rho_pp, rtheta_pp, wwAvg, rtheta_pp_old,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(ru_p),
      const_view2d(tend_rho), const_view2d(tend_rt), const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign), const_view1d(specMaskCell),
      params);

  // Substep 3: rw_p = 2.0 + 1.0 = 3.0; wwAvg = 1.5 + 0.5*3.0 = 3.0
  params.small_step = 3;
  acoustic_step_update_cells<ExecSpace>(
      rw_p, rho_pp, rtheta_pp, wwAvg, rtheta_pp_old,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(ru_p),
      const_view2d(tend_rho), const_view2d(tend_rt), const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign), const_view1d(specMaskCell),
      params);

  const Scalar tol = 1.0e-12;
  for (int k = 0; k < nVertLevels; ++k) {
    EXPECT_NEAR(rw_p(k, 0), 3.0, tol)
        << "rw_p after 3 substeps at k=" << k;
    EXPECT_NEAR(wwAvg(k, 0), 3.0, tol)
        << "wwAvg after 3 substeps at k=" << k;
  }
}

/// Test: Rayleigh damping is applied through the cell update kernel (Req 4.5).
/// Verifies that the cell solve path with non-zero dss modifies rw_p relative to
/// the no-damping baseline, confirming Rayleigh damping is integrated.
TEST(AcousticSolver, CellUpdateAppliesRayleighDamping) {
  const int nCells = 1;
  const int nEdges = 2;
  const int nVertLevels = 4;

  MiniMesh mesh(nCells, nVertLevels);

  // Two copies of state: one with damping, one without
  view2d rw_p_d("rw_p_damped", nVertLevels + 1, nCells);
  view2d rw_p_u("rw_p_undamped", nVertLevels + 1, nCells);
  view2d rho_pp_d("rho_pp_d", nVertLevels, nCells);
  view2d rho_pp_u("rho_pp_u", nVertLevels, nCells);
  view2d rtheta_pp_d("rtheta_pp_d", nVertLevels, nCells);
  view2d rtheta_pp_u("rtheta_pp_u", nVertLevels, nCells);
  view2d wwAvg_d("wwAvg_d", nVertLevels + 1, nCells);
  view2d wwAvg_u("wwAvg_u", nVertLevels + 1, nCells);
  view2d rtheta_pp_old_d("rtheta_pp_old_d", nVertLevels, nCells);
  view2d rtheta_pp_old_u("rtheta_pp_old_u", nVertLevels, nCells);

  view2d rho_zz("rho_zz", nVertLevels, nCells);
  view2d theta_m("theta_m", nVertLevels, nCells);
  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d tend_rho("tend_rho", nVertLevels, nCells);
  view2d tend_rt("tend_rt", nVertLevels, nCells);
  view2d tend_rw("tend_rw", nVertLevels + 1, nCells);
  view2d cofwt("cofwt", nVertLevels, nCells);
  view2d coftz("coftz", nVertLevels + 1, nCells);
  view2d cofwr("cofwr", nVertLevels, nCells);
  view2d cofwz("cofwz", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d a_tri("a_tri", nVertLevels, nCells);
  view2d alpha_tri("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri("gamma_tri", nVertLevels, nCells);
  view2d dss_damped("dss_damped", nVertLevels, nCells);
  view2d dss_zero("dss_zero", nVertLevels, nCells);
  view2d w_full("w", nVertLevels + 1, nCells);
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d cofrz_v("cofrz", nVertLevels);
  view1d etp_v("etp", nVertLevels);
  view1d etm_v("etm", nVertLevels);
  view1d ewp_v("ewp", nVertLevels + 1);
  view1d ewm_v("ewm", nVertLevels + 1);
  view1d specMaskCell("specMaskCell", nCells);

  Kokkos::deep_copy(specMaskCell, Scalar(0.0));

  // Initial state
  Kokkos::deep_copy(rho_pp_d, Scalar(0.0));
  Kokkos::deep_copy(rho_pp_u, Scalar(0.0));
  Kokkos::deep_copy(rtheta_pp_d, Scalar(0.0));
  Kokkos::deep_copy(rtheta_pp_u, Scalar(0.0));
  Kokkos::deep_copy(rw_p_d, Scalar(0.0));
  Kokkos::deep_copy(rw_p_u, Scalar(0.0));
  Kokkos::deep_copy(wwAvg_d, Scalar(0.0));
  Kokkos::deep_copy(wwAvg_u, Scalar(0.0));

  // Set up tend_rw so rw_p gets non-trivial values after the explicit update
  Kokkos::deep_copy(tend_rho, Scalar(0.0));
  Kokkos::deep_copy(tend_rt, Scalar(0.0));
  Kokkos::deep_copy(tend_rw, Scalar(5.0));
  Kokkos::deep_copy(ru_p, Scalar(0.0));

  // Identity-like tridiagonal (no coupling)
  Kokkos::deep_copy(cofwt, Scalar(0.0));
  Kokkos::deep_copy(coftz, Scalar(0.0));
  Kokkos::deep_copy(cofwr, Scalar(0.0));
  Kokkos::deep_copy(cofwz, Scalar(0.0));
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(a_tri, Scalar(0.0));
  Kokkos::deep_copy(alpha_tri, Scalar(1.0));
  Kokkos::deep_copy(gamma_tri, Scalar(0.0));

  // Damping: dss non-zero at top levels
  Kokkos::deep_copy(dss_zero, Scalar(0.0));
  Kokkos::deep_copy(dss_damped, Scalar(0.0));
  dss_damped(2, 0) = Scalar(0.3);
  dss_damped(3, 0) = Scalar(0.5);

  // Non-zero w for damping to act on
  Kokkos::deep_copy(w_full, Scalar(2.0));
  Kokkos::deep_copy(rw_save, Scalar(0.5));
  Kokkos::deep_copy(rw_base, Scalar(0.5));
  Kokkos::deep_copy(fzm_v, Scalar(0.5));
  Kokkos::deep_copy(fzp_v, Scalar(0.5));
  Kokkos::deep_copy(rdzw_v, Scalar(0.0));
  Kokkos::deep_copy(cofrz_v, Scalar(0.0));
  Kokkos::deep_copy(etp_v, Scalar(0.5));
  Kokkos::deep_copy(etm_v, Scalar(0.5));
  Kokkos::deep_copy(ewp_v, Scalar(0.5));
  Kokkos::deep_copy(ewm_v, Scalar(0.5));
  Kokkos::deep_copy(rho_zz, Scalar(1.0));
  Kokkos::deep_copy(theta_m, Scalar(300.0));

  const Scalar dts = 1.0;
  AcousticStepParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.nCellsSolve = nCells;
  params.maxEdges = mesh.maxEdges;
  params.dts = dts;
  params.small_step = 2;

  // Run with zero dss (no damping)
  acoustic_step_update_cells<ExecSpace>(
      rw_p_u, rho_pp_u, rtheta_pp_u, wwAvg_u, rtheta_pp_old_u,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(ru_p),
      const_view2d(tend_rho), const_view2d(tend_rt), const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss_zero),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign), const_view1d(specMaskCell),
      params);

  // Run with non-zero dss (damping active)
  acoustic_step_update_cells<ExecSpace>(
      rw_p_d, rho_pp_d, rtheta_pp_d, wwAvg_d, rtheta_pp_old_d,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(ru_p),
      const_view2d(tend_rho), const_view2d(tend_rt), const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss_damped),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign), const_view1d(specMaskCell),
      params);

  // Levels without damping (k=1) should match
  const Scalar tol = 1.0e-12;
  EXPECT_NEAR(rw_p_d(1, 0), rw_p_u(1, 0), tol)
      << "Level 1 (no dss) should match undamped";

  // Levels with damping (k=2,3) should differ
  EXPECT_NE(rw_p_d(2, 0), rw_p_u(2, 0))
      << "Level 2 (dss=0.3) rw_p should differ from undamped";
  EXPECT_NE(rw_p_d(3, 0), rw_p_u(3, 0))
      << "Level 3 (dss=0.5) rw_p should differ from undamped";
}

/// Test: Full advance_acoustic_step orchestration end-to-end (Req 4.1, 4.2, 4.7, 14.2).
/// Verifies that calling the top-level entry point produces output fields that match
/// hand-computed reference values within Parity_Tolerance. On failure, identifies
/// the diverging field (Req 14.9).
TEST(AcousticSolver, FullAcousticStepParity) {
  const int nCells = 2;
  const int nEdges = 3;
  const int nVertLevels = 2;

  MiniMesh mesh(nCells, nVertLevels);

  // Allocate all fields for advance_acoustic_step
  view2d ru_p("ru_p", nVertLevels, nEdges);
  view2d rw_p("rw_p", nVertLevels + 1, nCells);
  view2d rho_pp("rho_pp", nVertLevels, nCells);
  view2d rtheta_pp("rtheta_pp", nVertLevels, nCells);
  view2d ruAvg("ruAvg", nVertLevels, nEdges);
  view2d wwAvg("wwAvg", nVertLevels + 1, nCells);
  view2d rtheta_pp_old("rtheta_pp_old", nVertLevels, nCells);
  view2d rho_zz("rho_zz", nVertLevels, nCells);
  view2d theta_m("theta_m", nVertLevels, nCells);
  view2d exner("exner", nVertLevels, nCells);
  view2d cqu("cqu", nVertLevels, nEdges);
  view2d zxu("zxu", nVertLevels, nEdges);
  view2d tend_ru("tend_ru", nVertLevels, nEdges);
  view2d tend_rho("tend_rho", nVertLevels, nCells);
  view2d tend_rt("tend_rt", nVertLevels, nCells);
  view2d tend_rw("tend_rw", nVertLevels + 1, nCells);
  view2d cofwt("cofwt", nVertLevels, nCells);
  view2d coftz("coftz", nVertLevels + 1, nCells);
  view2d cofwr("cofwr", nVertLevels, nCells);
  view2d cofwz("cofwz", nVertLevels, nCells);
  view2d zz("zz", nVertLevels, nCells);
  view2d a_tri("a_tri", nVertLevels, nCells);
  view2d alpha_tri("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri("gamma_tri", nVertLevels, nCells);
  view2d dss("dss", nVertLevels, nCells);
  view2d w_full("w", nVertLevels + 1, nCells);
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d cofrz_v("cofrz", nVertLevels);
  view1d etp_v("etp", nVertLevels);
  view1d etm_v("etm", nVertLevels);
  view1d ewp_v("ewp", nVertLevels + 1);
  view1d ewm_v("ewm", nVertLevels + 1);
  view1d specMaskEdge("specMaskEdge", nEdges);
  view1d specMaskCell("specMaskCell", nCells);

  // Setup: substep 1, no pressure gradient, no spec zone, no damping,
  // identity tridiag, non-zero tend_ru and tend_rho/tend_rt.
  Kokkos::deep_copy(ru_p, Scalar(999.0));
  Kokkos::deep_copy(rw_p, Scalar(888.0));
  Kokkos::deep_copy(rho_pp, Scalar(777.0));
  Kokkos::deep_copy(rtheta_pp, Scalar(666.0));
  Kokkos::deep_copy(ruAvg, Scalar(555.0));
  Kokkos::deep_copy(wwAvg, Scalar(444.0));

  Kokkos::deep_copy(specMaskEdge, Scalar(0.0));
  Kokkos::deep_copy(specMaskCell, Scalar(0.0));
  Kokkos::deep_copy(rho_zz, Scalar(1.0));
  Kokkos::deep_copy(theta_m, Scalar(300.0));
  Kokkos::deep_copy(exner, Scalar(1.0));
  Kokkos::deep_copy(cqu, Scalar(1.0));
  Kokkos::deep_copy(zxu, Scalar(0.0));
  Kokkos::deep_copy(zz, Scalar(1.0));
  Kokkos::deep_copy(tend_ru, Scalar(2.0));
  Kokkos::deep_copy(tend_rho, Scalar(0.1));
  Kokkos::deep_copy(tend_rt, Scalar(0.2));
  Kokkos::deep_copy(tend_rw, Scalar(0.0));
  Kokkos::deep_copy(cofwt, Scalar(0.0));
  Kokkos::deep_copy(coftz, Scalar(0.0));
  Kokkos::deep_copy(cofwr, Scalar(0.0));
  Kokkos::deep_copy(cofwz, Scalar(0.0));
  Kokkos::deep_copy(a_tri, Scalar(0.0));
  Kokkos::deep_copy(alpha_tri, Scalar(1.0));
  Kokkos::deep_copy(gamma_tri, Scalar(0.0));
  Kokkos::deep_copy(dss, Scalar(0.0));
  Kokkos::deep_copy(w_full, Scalar(0.0));
  Kokkos::deep_copy(rw_save, Scalar(0.0));
  Kokkos::deep_copy(rw_base, Scalar(0.0));
  Kokkos::deep_copy(fzm_v, Scalar(0.5));
  Kokkos::deep_copy(fzp_v, Scalar(0.5));
  Kokkos::deep_copy(rdzw_v, Scalar(0.0));
  Kokkos::deep_copy(cofrz_v, Scalar(0.0));
  Kokkos::deep_copy(etp_v, Scalar(0.5));
  Kokkos::deep_copy(etm_v, Scalar(0.5));
  Kokkos::deep_copy(ewp_v, Scalar(0.5));
  Kokkos::deep_copy(ewm_v, Scalar(0.5));

  const Scalar dts = 1.0;
  AcousticStepParams step_params;
  step_params.nVertLevels = nVertLevels;
  step_params.nCells = nCells;
  step_params.nEdges = nEdges;
  step_params.nCellsSolve = nCells;
  step_params.maxEdges = mesh.maxEdges;
  step_params.dts = dts;
  step_params.small_step = 1;

  DivergenceDampingParams damp_params;
  damp_params.nVertLevels = nVertLevels;
  damp_params.nEdges = nEdges;
  damp_params.nCellsSolve = nCells;
  damp_params.dts = dts;
  damp_params.smdiv = 0.0;  // No divergence damping for simplicity
  damp_params.config_len_disp = 1000.0;

  advance_acoustic_step<ExecSpace>(
      ru_p, rw_p, rho_pp, rtheta_pp, ruAvg, wwAvg, rtheta_pp_old,
      const_view2d(rho_zz), const_view2d(theta_m), const_view2d(exner),
      const_view2d(cqu), const_view2d(zxu),
      const_view2d(tend_ru), const_view2d(tend_rho), const_view2d(tend_rt),
      const_view2d(tend_rw),
      const_view2d(cofwt), const_view2d(coftz), const_view2d(cofwr),
      const_view2d(cofwz), const_view2d(zz),
      const_view2d(a_tri), const_view2d(alpha_tri), const_view2d(gamma_tri),
      const_view2d(dss),
      const_view2d(w_full), const_view2d(rw_save), const_view2d(rw_base),
      const_view1d(fzm_v), const_view1d(fzp_v), const_view1d(rdzw_v),
      const_view1d(cofrz_v), const_view1d(etp_v), const_view1d(etm_v),
      const_view1d(ewp_v), const_view1d(ewm_v),
      const_view1d(mesh.dvEdge), const_view1d(mesh.invDcEdge),
      const_view1d(mesh.invAreaCell),
      const_int_view1d(mesh.nEdgesOnCell_v), const_int_view2d(mesh.cellsOnEdge),
      const_int_view2d(mesh.edgesOnCell),
      const_view2d(mesh.edgesOnCell_sign),
      const_view1d(specMaskEdge), const_view1d(specMaskCell),
      step_params, damp_params);

  // Reference values for substep 1 with these inputs:
  // Edge update: ru_p = dts * tend_ru = 1.0 * 2.0 = 2.0; ruAvg = ru_p = 2.0
  // Cell update (interior): zeroing then:
  //   With ru_p=2.0 on all edges, for each cell with 2 edges (signs -1, +1):
  //     rs[k] accumulates flux divergence
  //   Then with zero vertical coupling coefficients and identity tridiag:
  //     rho_pp and rtheta_pp get the updated values from rs/ts
  //
  // For the linear mesh: each cell has 2 edges. The flux contribution is:
  //   For edge i with sign s: flux = s * dts * dvEdge * ru_p * invAreaCell
  //   dvEdge=1, invAreaCell=1, dts=1, ru_p=2.0
  //   edge0 (sign=-1): flux0 = -1 * 1 * 1 * 2 * 1 = -2
  //   edge1 (sign=+1): flux1 = +1 * 1 * 1 * 2 * 1 = +2
  //   rs[k] -= flux0 + flux1 = -(-2 + 2) = 0
  //   ts[k] similarly: theta_m at both cells = 300, so
  //     ts[k] -= flux * 0.5*(theta_m(cell1)+theta_m(cell2))
  //            Each flux's theta contribution also cancels symmetrically for
  //            uniform theta_m, giving ts contribution = 0
  //
  // After zeroing (substep 1):
  //   rs[k] = 0 + dts*tend_rho + 0 - 0 = 0.1
  //   ts[k] = 0 + dts*tend_rt + 0 - 0 = 0.2
  //   rw_p stays 0 (tend_rw=0, all coupling=0)
  //   rho_pp = rs = 0.1 (with zero cofrz and zero rw_p)
  //   rtheta_pp = ts = 0.2 (with zero rdzw and zero rw_p)

  const Scalar tol = 1.0e-12;

  // Verify ruAvg (Req 4.2)
  for (int ie = 0; ie < nEdges; ++ie) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(ruAvg(k, ie), 2.0, tol)
          << "ruAvg diverges from reference at edge=" << ie << " k=" << k;
    }
  }

  // Verify ru_p (Req 4.1)
  for (int ie = 0; ie < nEdges; ++ie) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(ru_p(k, ie), 2.0, tol)
          << "ru_p diverges from reference at edge=" << ie << " k=" << k;
    }
  }

  // Verify rho_pp (Req 4.1)
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(rho_pp(k, ic), 0.1, tol)
          << "rho_pp diverges from reference at cell=" << ic << " k=" << k;
    }
  }

  // Verify rtheta_pp (Req 4.1)
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k < nVertLevels; ++k) {
      EXPECT_NEAR(rtheta_pp(k, ic), 0.2, tol)
          << "rtheta_pp diverges from reference at cell=" << ic << " k=" << k;
    }
  }

  // Verify rw_p (Req 4.1)
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k <= nVertLevels; ++k) {
      EXPECT_NEAR(rw_p(k, ic), 0.0, tol)
          << "rw_p diverges from reference at cell=" << ic << " k=" << k;
    }
  }

  // Verify wwAvg (Req 4.2): with rw_p=0, wwAvg = ewp*rw_p = 0
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k <= nVertLevels; ++k) {
      EXPECT_NEAR(wwAvg(k, ic), 0.0, tol)
          << "wwAvg diverges from reference at cell=" << ic << " k=" << k;
    }
  }
}

}  // namespace test
}  // namespace dycore
}  // namespace mpas
