#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/scalar_transport.hpp"
#include "mpas_dycore/scalar.hpp"

#include <cmath>
#include <vector>

namespace {

using Scalar = mpas::dycore::Scalar;
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using layout = Kokkos::LayoutLeft;

constexpr Scalar kTol = std::is_same_v<Scalar, double> ? 1e-12 : 1e-6f;

template <class T>
using View1D = Kokkos::View<T*, layout, MemSpace>;
template <class T>
using View2D = Kokkos::View<T**, layout, MemSpace>;
template <class T>
using View3D = Kokkos::View<T***, layout, MemSpace>;

/// Build a small linear mesh for scalar transport testing.
/// 5 cells in a line, 4 interior edges, with a simple advection stencil.
struct LinearMesh {
  static constexpr int nCells = 5;
  static constexpr int nEdges = 4;
  static constexpr int nVertLevels = 4;
  static constexpr int num_scalars = 2;
  static constexpr int maxEdges = 2;
  static constexpr int maxAdvCellsForEdge = 4; // 4-point stencil

  mpas::dycore::ScalarTransportMeshData<ExecSpace> mesh;
  mpas::dycore::ScalarTransportState<ExecSpace> state;

  LinearMesh() {
    mesh.nCells = nCells;
    mesh.nEdges = nEdges;
    mesh.nVertLevels = nVertLevels;
    mesh.num_scalars = num_scalars;
    mesh.maxEdges = maxEdges;
    mesh.maxAdvCellsForEdge = maxAdvCellsForEdge;

    // Allocate connectivity
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

    // Set up a linear mesh: cells 0-1-2-3-4 connected by edges 0-1-2-3
    // Edge i connects cells i and i+1 (1-based: i+1 and i+2)
    auto coe_h = Kokkos::create_mirror_view(mesh.cellsOnEdge);
    for (int e = 0; e < nEdges; ++e) {
      coe_h(0, e) = e + 1;     // cell1 (1-based)
      coe_h(1, e) = e + 2;     // cell2 (1-based)
    }
    Kokkos::deep_copy(mesh.cellsOnEdge, coe_h);

    // Each interior cell has 2 edges; boundary cells have 1
    auto eoc_h = Kokkos::create_mirror_view(mesh.edgesOnCell);
    auto nec_h = Kokkos::create_mirror_view(mesh.nEdgesOnCell);
    // Cell 0: edge 0 only
    eoc_h(0, 0) = 1; nec_h(0) = 1;
    // Cells 1-3: two edges
    for (int c = 1; c < nCells - 1; ++c) {
      eoc_h(0, c) = c;       // left edge (1-based)
      eoc_h(1, c) = c + 1;   // right edge (1-based)
      nec_h(c) = 2;
    }
    // Cell 4: edge 3 only
    eoc_h(0, nCells - 1) = nEdges; nec_h(nCells - 1) = 1;
    Kokkos::deep_copy(mesh.edgesOnCell, eoc_h);
    Kokkos::deep_copy(mesh.nEdgesOnCell, nec_h);

    // Edge sign: +1 for outflow from cell, -1 for inflow
    // Convention: edge goes from cell1 to cell2 (positive direction)
    auto ecs_h = Kokkos::create_mirror_view(mesh.edgesOnCell_sign);
    // Cell 0: edge 0 exits to the right -> sign = +1
    ecs_h(0, 0) = Scalar(1.0);
    // Cells 1-3: left edge enters (-1), right edge exits (+1)
    for (int c = 1; c < nCells - 1; ++c) {
      ecs_h(0, c) = Scalar(-1.0); // left edge: inflow
      ecs_h(1, c) = Scalar(1.0);  // right edge: outflow
    }
    // Cell 4: left edge enters
    ecs_h(0, nCells - 1) = Scalar(-1.0);
    Kokkos::deep_copy(mesh.edgesOnCell_sign, ecs_h);

    // Simple advection stencil: 2-point (cell1, cell2) for each edge
    // with centered coefficients (adv_coefs = 0.5 each, no 3rd-order bias)
    auto acfe_h = Kokkos::create_mirror_view(mesh.advCellsForEdge);
    auto nacfe_h = Kokkos::create_mirror_view(mesh.nAdvCellsForEdge);
    auto ac_h = Kokkos::create_mirror_view(mesh.adv_coefs);
    auto ac3_h = Kokkos::create_mirror_view(mesh.adv_coefs_3rd);
    for (int e = 0; e < nEdges; ++e) {
      nacfe_h(e) = 2;
      acfe_h(0, e) = e + 1;  // cell1 (1-based)
      acfe_h(1, e) = e + 2;  // cell2 (1-based)
      ac_h(0, e) = Scalar(0.5);
      ac_h(1, e) = Scalar(0.5);
      ac3_h(0, e) = Scalar(0.5);   // upwind bias
      ac3_h(1, e) = Scalar(-0.5);  // downwind correction
    }
    Kokkos::deep_copy(mesh.advCellsForEdge, acfe_h);
    Kokkos::deep_copy(mesh.nAdvCellsForEdge, nacfe_h);
    Kokkos::deep_copy(mesh.adv_coefs, ac_h);
    Kokkos::deep_copy(mesh.adv_coefs_3rd, ac3_h);

    // Geometry: uniform cell area = 1.0, dvEdge = 1.0
    auto dv_h = Kokkos::create_mirror_view(mesh.dvEdge);
    auto ia_h = Kokkos::create_mirror_view(mesh.invAreaCell);
    for (int e = 0; e < nEdges; ++e) dv_h(e) = Scalar(1.0);
    for (int c = 0; c < nCells; ++c) ia_h(c) = Scalar(1.0);
    Kokkos::deep_copy(mesh.dvEdge, dv_h);
    Kokkos::deep_copy(mesh.invAreaCell, ia_h);

    // Vertical weights: simple linear interpolation (fnm=0.5, fnp=0.5)
    auto fnm_h = Kokkos::create_mirror_view(mesh.fnm);
    auto fnp_h = Kokkos::create_mirror_view(mesh.fnp);
    auto rdnw_h = Kokkos::create_mirror_view(mesh.rdnw);
    for (int k = 0; k <= nVertLevels; ++k) {
      fnm_h(k) = Scalar(0.5);
      fnp_h(k) = Scalar(0.5);
    }
    for (int k = 0; k < nVertLevels; ++k) {
      rdnw_h(k) = Scalar(1.0);  // uniform vertical spacing
    }
    Kokkos::deep_copy(mesh.fnm, fnm_h);
    Kokkos::deep_copy(mesh.fnp, fnp_h);
    Kokkos::deep_copy(mesh.rdnw, rdnw_h);

    // No regional boundaries (all interior)
    Kokkos::deep_copy(mesh.bdyMaskCell, 0);
    Kokkos::deep_copy(mesh.bdyMaskEdge, 0);

    // State fields
    state.scalar_old = View3D<Scalar>("scalar_old", num_scalars, nVertLevels, nCells);
    state.scalar_new = View3D<Scalar>("scalar_new", num_scalars, nVertLevels, nCells);
    state.rho_zz_old = View2D<Scalar>("rho_zz_old", nVertLevels, nCells);
    state.rho_zz_new = View2D<Scalar>("rho_zz_new", nVertLevels, nCells);
    state.uhAvg = View2D<Scalar>("uhAvg", nVertLevels, nEdges);
    state.wwAvg = View2D<Scalar>("wwAvg", nVertLevels + 1, nCells);
    state.scalar_tend = View3D<Scalar>("scalar_tend", num_scalars, nVertLevels, nCells);

    // Initialize with uniform density = 1.0
    Kokkos::deep_copy(state.rho_zz_old, Scalar(1.0));
    Kokkos::deep_copy(state.rho_zz_new, Scalar(1.0));

    // Zero mass fluxes and tendencies
    Kokkos::deep_copy(state.uhAvg, Scalar(0.0));
    Kokkos::deep_copy(state.wwAvg, Scalar(0.0));
    Kokkos::deep_copy(state.scalar_tend, Scalar(0.0));
  }
};

// ============================================================================
// Test: Zero flux preserves scalar field (no advection, no sources)
// ============================================================================
TEST(ScalarTransport, ZeroFluxPreservesField) {
  LinearMesh lm;

  // Set initial scalars to a linear profile
  auto so_h = Kokkos::create_mirror_view(lm.state.scalar_old);
  auto sn_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        Scalar val = Scalar(1.0) + Scalar(c) * Scalar(0.1) + Scalar(k) * Scalar(0.01);
        so_h(s, k, c) = val;
        sn_h(s, k, c) = val;
      }
    }
  }
  Kokkos::deep_copy(lm.state.scalar_old, so_h);
  Kokkos::deep_copy(lm.state.scalar_new, sn_h);

  // Run transport with zero fluxes
  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state,
      /*dt=*/Scalar(1.0),
      /*coef_3rd_order=*/Scalar(1.0),
      /*rk_step=*/1,
      /*config_time_integration_order=*/3,
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false);

  // With zero mass fluxes (uhAvg=0, wwAvg=0) and zero tendency,
  // the update should give: scalar_new = scalar_old * rho_old / rho_interp
  // With uniform rho_old = rho_new = 1.0 and wt_old=2/3, wt_new=1/3 for
  // rk_step=1/order=3: rho_interp = 2/3*1 + 1/3*1 = 1.0
  // So scalar_new should equal scalar_old
  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        EXPECT_NEAR(result_h(s, k, c), so_h(s, k, c), kTol)
            << "Mismatch at scalar=" << s << " k=" << k << " cell=" << c;
      }
    }
  }
}

// ============================================================================
// Test: Uniform horizontal advection moves scalar
// ============================================================================
TEST(ScalarTransport, UniformHorizAdvection) {
  LinearMesh lm;

  // Set uniform scalar_old = 1.0, scalar_new = 1.0 (for flux computation)
  Kokkos::deep_copy(lm.state.scalar_old, Scalar(1.0));
  Kokkos::deep_copy(lm.state.scalar_new, Scalar(1.0));

  // Set uniform positive horizontal mass flux uhAvg = 1.0
  Kokkos::deep_copy(lm.state.uhAvg, Scalar(1.0));

  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state,
      /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0),
      /*rk_step=*/1,
      /*config_time_integration_order=*/3,
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false);

  // With uniform scalar = 1.0, the horizontal flux is 1.0 at every edge.
  // For an interior cell (e.g. cell 2): net flux divergence = 0 because
  // inflow = outflow for uniform scalar. So scalar_new should remain ~1.0
  // (modulo vertical flux which is also 0 with wwAvg=0).
  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  // Interior cells should stay at 1.0 since scalar is uniform
  for (int c = 1; c < LinearMesh::nCells - 1; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        EXPECT_NEAR(result_h(s, k, c), Scalar(1.0), Scalar(1e-10))
            << "Interior cell=" << c << " should stay 1.0 with uniform field";
      }
    }
  }
}

// ============================================================================
// Test: Density re-integration weights are correct (Req 5.5)
// ============================================================================
TEST(ScalarTransport, DensityWeightsRK1Order3) {
  LinearMesh lm;

  // Set scalar_old = 2.0, scalar_new = 2.0
  Kokkos::deep_copy(lm.state.scalar_old, Scalar(2.0));
  Kokkos::deep_copy(lm.state.scalar_new, Scalar(2.0));

  // Set different rho_zz_old and rho_zz_new to test the weight
  Kokkos::deep_copy(lm.state.rho_zz_old, Scalar(1.0));
  Kokkos::deep_copy(lm.state.rho_zz_new, Scalar(2.0));

  // No fluxes or tendencies
  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state,
      /*dt=*/Scalar(1.0),
      /*coef_3rd_order=*/Scalar(1.0),
      /*rk_step=*/1,
      /*config_time_integration_order=*/3,
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false);

  // rk_step=1, order=3 -> weight_time_new = 1/3
  // rho_interp = (2/3)*1.0 + (1/3)*2.0 = 4/3
  // scalar_new = (2.0 * 1.0 + dt*0) / (4/3) = 2.0 * 3/4 = 1.5
  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  const Scalar expected = Scalar(2.0) * Scalar(1.0) / (Scalar(2.0)/Scalar(3.0) + Scalar(2.0)/Scalar(3.0));
  // Actually: rho_interp = (1-1/3)*1.0 + (1/3)*2.0 = 2/3 + 2/3 = 4/3
  // scalar_new = 2.0 * 1.0 / (4/3) = 2.0 * 3/4 = 1.5
  const Scalar expected_val = Scalar(2.0) * Scalar(3.0) / Scalar(4.0);

  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      EXPECT_NEAR(result_h(0, k, c), expected_val, kTol)
          << "Density weight incorrect at k=" << k << " cell=" << c;
    }
  }
}

// ============================================================================
// Test: No density re-integration when advance_density = false
// ============================================================================
TEST(ScalarTransport, NoDensityReintegration) {
  LinearMesh lm;

  Kokkos::deep_copy(lm.state.scalar_old, Scalar(2.0));
  Kokkos::deep_copy(lm.state.scalar_new, Scalar(2.0));
  Kokkos::deep_copy(lm.state.rho_zz_old, Scalar(1.0));
  Kokkos::deep_copy(lm.state.rho_zz_new, Scalar(3.0));

  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state,
      /*dt=*/Scalar(1.0),
      /*coef_3rd_order=*/Scalar(1.0),
      /*rk_step=*/1,
      /*config_time_integration_order=*/3,
      /*advance_density=*/false,  // <-- no re-integration
      /*config_apply_lbcs=*/false);

  // advance_density=false -> weight_time_new = 1.0
  // rho_interp = 0*rho_old + 1*rho_new = 3.0
  // scalar_new = (2.0 * 1.0) / 3.0 = 2/3
  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  const Scalar expected = Scalar(2.0) / Scalar(3.0);
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      EXPECT_NEAR(result_h(0, k, c), expected, kTol)
          << "No-reintegration incorrect at k=" << k << " cell=" << c;
    }
  }
}

// ============================================================================
// Test: Physics tendency is correctly added to scalar update
// ============================================================================
TEST(ScalarTransport, PhysTendencyAdded) {
  LinearMesh lm;

  Kokkos::deep_copy(lm.state.scalar_old, Scalar(1.0));
  Kokkos::deep_copy(lm.state.scalar_new, Scalar(1.0));
  Kokkos::deep_copy(lm.state.rho_zz_old, Scalar(1.0));
  Kokkos::deep_copy(lm.state.rho_zz_new, Scalar(1.0));

  // Set physics tendency = 0.5 for all cells/levels/scalars
  Kokkos::deep_copy(lm.state.scalar_tend, Scalar(0.5));

  Scalar dt = Scalar(0.1);
  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state,
      dt,
      /*coef_3rd_order=*/Scalar(1.0),
      /*rk_step=*/3,
      /*config_time_integration_order=*/3,
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false);

  // rk_step=3 -> weight_time_new=1.0, so rho_interp = rho_new = 1.0
  // With zero fluxes (uhAvg=0, wwAvg=0):
  // scalar_new = (1.0*1.0 + 0.1*0.5) / 1.0 = 1.05
  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  const Scalar expected = Scalar(1.0) + dt * Scalar(0.5);
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        EXPECT_NEAR(result_h(s, k, c), expected, kTol)
            << "Physics tendency not applied at s=" << s
            << " k=" << k << " c=" << c;
      }
    }
  }
}

// ============================================================================
// Test: Vertical flux divergence with uniform vertical advection
// ============================================================================
TEST(ScalarTransport, VerticalFluxDivergence) {
  LinearMesh lm;

  // Set a linear vertical profile: scalar = k+1 at level k
  auto so_h = Kokkos::create_mirror_view(lm.state.scalar_old);
  auto sn_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        Scalar val = Scalar(k + 1);
        so_h(s, k, c) = val;
        sn_h(s, k, c) = val;
      }
    }
  }
  Kokkos::deep_copy(lm.state.scalar_old, so_h);
  Kokkos::deep_copy(lm.state.scalar_new, sn_h);

  // Set uniform vertical mass flux wwAvg = 1.0 at all interior interfaces
  auto ww_h = Kokkos::create_mirror_view(lm.state.wwAvg);
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    ww_h(0, c) = Scalar(0.0); // top boundary
    for (int k = 1; k < LinearMesh::nVertLevels; ++k) {
      ww_h(k, c) = Scalar(1.0);
    }
    ww_h(LinearMesh::nVertLevels, c) = Scalar(0.0); // bottom boundary
  }
  Kokkos::deep_copy(lm.state.wwAvg, ww_h);

  Kokkos::deep_copy(lm.state.rho_zz_old, Scalar(1.0));
  Kokkos::deep_copy(lm.state.rho_zz_new, Scalar(1.0));

  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state,
      /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0),
      /*rk_step=*/3,
      /*config_time_integration_order=*/3,
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false);

  // Verify finite values and correct vertical flux computation.
  // With linear profile q=[1,2,3,4], fnm=fnp=0.5, w=1, rdnw=1, dt=0.1:
  // Interface k=1: linear interp = 1*(0.5*2+0.5*1) = 1.5
  // Interface k=2: only 4 levels so nVertLevels-2=2 -> no interior flux3
  //                (k must be in [2..nVertLevels-2]=[2..2] -> k=2)
  //   q_im2=q(0)=1, q_im1=q(1)=2, q_i=q(2)=3, q_ip1=q(3)=4
  //   flux4 = 1*(7*(3+2)-(4+1))/12 = (35-5)/12 = 30/12 = 2.5
  //   correction = |1|*((4-1)-3*(3-2))/12 = (3-3)/12 = 0
  //   flux3 = 2.5
  // Interface k=3 (=nVertLevels-1): linear = 1*(0.5*4+0.5*3) = 3.5
  //
  // scalar_new(k) = (q_old(k)*1 + dt*(0 - rdnw*(wdtn(k+1)-wdtn(k))))/1
  //   k=0: 1 + 0.1*(0 - (1.5-0)) = 1 - 0.15 = 0.85
  //   k=1: 2 + 0.1*(0 - (2.5-1.5)) = 2 - 0.10 = 1.90
  //   k=2: 3 + 0.1*(0 - (3.5-2.5)) = 3 - 0.10 = 2.90
  //   k=3: 4 + 0.1*(0 - (0-3.5)) = 4 + 0.35 = 4.35
  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  const Scalar exp_k0 = Scalar(0.85);
  const Scalar exp_k1 = Scalar(1.90);
  const Scalar exp_k2 = Scalar(2.90);
  const Scalar exp_k3 = Scalar(4.35);

  for (int c = 0; c < LinearMesh::nCells; ++c) {
    EXPECT_NEAR(result_h(0, 0, c), exp_k0, kTol)
        << "DIVERGING FIELD: scalar_new — vert flux at k=0, cell=" << c;
    EXPECT_NEAR(result_h(0, 1, c), exp_k1, kTol)
        << "DIVERGING FIELD: scalar_new — vert flux at k=1, cell=" << c;
    EXPECT_NEAR(result_h(0, 2, c), exp_k2, kTol)
        << "DIVERGING FIELD: scalar_new — vert flux at k=2, cell=" << c;
    EXPECT_NEAR(result_h(0, 3, c), exp_k3, kTol)
        << "DIVERGING FIELD: scalar_new — vert flux at k=3, cell=" << c;
  }
}

// ============================================================================
// Test: Third-order coefficient affects vertical flux (Req 5.3, 5.4)
// Validates that coef_3rd_order modulates the vertical flux3 correction term.
// ============================================================================
TEST(ScalarTransport, ThirdOrderCoefficientAffectsVertFlux) {
  LinearMesh lm;

  // Use a profile where the 3rd-order correction is nonzero.
  // q=[1, 3, 5, 2]: at k=2 interior interface:
  //   q_im2=1, q_im1=3, q_i=5, q_ip1=2
  //   correction = |w|*((2-1)-3*(5-3))/12 = 1*(1-6)/12 = -5/12 != 0
  auto so_h = Kokkos::create_mirror_view(lm.state.scalar_old);
  auto sn_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Scalar vals[4] = {Scalar(1), Scalar(3), Scalar(5), Scalar(2)};
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        so_h(s, k, c) = vals[k];
        sn_h(s, k, c) = vals[k];
      }
    }
  }
  Kokkos::deep_copy(lm.state.scalar_old, so_h);
  Kokkos::deep_copy(lm.state.scalar_new, sn_h);

  // Set vertical flux at interior interfaces
  auto ww_h = Kokkos::create_mirror_view(lm.state.wwAvg);
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    ww_h(0, c) = Scalar(0.0);
    for (int k = 1; k < LinearMesh::nVertLevels; ++k)
      ww_h(k, c) = Scalar(1.0);
    ww_h(LinearMesh::nVertLevels, c) = Scalar(0.0);
  }
  Kokkos::deep_copy(lm.state.wwAvg, ww_h);
  Kokkos::deep_copy(lm.state.uhAvg, Scalar(0.0));

  // Run with coef_3rd_order = 0 (pure 4th-order, no correction)
  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  auto state0 = lm.state;
  state0.scalar_new = View3D<Scalar>("sn0", LinearMesh::num_scalars,
      LinearMesh::nVertLevels, LinearMesh::nCells);
  Kokkos::deep_copy(state0.scalar_new, sn_h);
  transport.advance_scalars(
      lm.mesh, state0, /*dt=*/Scalar(0.1), /*coef_3rd_order=*/Scalar(0.0),
      /*rk_step=*/3, /*config_time_integration_order=*/3,
      /*advance_density=*/true, /*config_apply_lbcs=*/false);
  auto r0_h = Kokkos::create_mirror_view(state0.scalar_new);
  Kokkos::deep_copy(r0_h, state0.scalar_new);

  // Run with coef_3rd_order = 1.0 (full 3rd-order upwind bias)
  auto state1 = lm.state;
  state1.scalar_new = View3D<Scalar>("sn1", LinearMesh::num_scalars,
      LinearMesh::nVertLevels, LinearMesh::nCells);
  Kokkos::deep_copy(state1.scalar_new, sn_h);
  transport.advance_scalars(
      lm.mesh, state1, /*dt=*/Scalar(0.1), /*coef_3rd_order=*/Scalar(1.0),
      /*rk_step=*/3, /*config_time_integration_order=*/3,
      /*advance_density=*/true, /*config_apply_lbcs=*/false);
  auto r1_h = Kokkos::create_mirror_view(state1.scalar_new);
  Kokkos::deep_copy(r1_h, state1.scalar_new);

  // With q=[1,3,5,2] the correction at k=2 is -5/12, so the two runs
  // must produce different results.
  bool found_difference = false;
  for (int c = 0; c < LinearMesh::nCells && !found_difference; ++c)
    for (int k = 0; k < LinearMesh::nVertLevels && !found_difference; ++k)
      if (std::abs(r0_h(0, k, c) - r1_h(0, k, c)) > kTol)
        found_difference = true;

  EXPECT_TRUE(found_difference)
      << "DIVERGING FIELD: scalar_new — coef_3rd_order has no effect "
         "on the vertical flux3 computation (Req 5.3, 5.4).";
}

// ============================================================================
// Test: Standard transport on non-final RK substep (Req 5.1)
// ============================================================================
TEST(ScalarTransport, StandardTransportNonFinalRK) {
  LinearMesh lm;

  // Sharp spike: cell 2 = 100, rest = 1
  auto so_h = Kokkos::create_mirror_view(lm.state.scalar_old);
  auto sn_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        so_h(s, k, c) = (c == 2) ? Scalar(100.0) : Scalar(1.0);
        sn_h(s, k, c) = (c == 2) ? Scalar(100.0) : Scalar(1.0);
      }
    }
  }
  Kokkos::deep_copy(lm.state.scalar_old, so_h);
  Kokkos::deep_copy(lm.state.scalar_new, sn_h);
  Kokkos::deep_copy(lm.state.uhAvg, Scalar(2.0));

  // rk_step=1 with order=3 -> non-final substep -> standard transport (Req 5.1)
  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state, /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0), /*rk_step=*/1,
      /*config_time_integration_order=*/3,
      /*advance_density=*/true, /*config_apply_lbcs=*/false);

  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  // Verify finite results (standard scheme has no monotonic bound enforcement)
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        EXPECT_TRUE(std::isfinite(result_h(s, k, c)))
            << "DIVERGING FIELD: scalar_new — non-finite on non-final "
               "RK step at s=" << s << " k=" << k << " c=" << c;
      }
    }
  }
}

// ============================================================================
// Test: Regional boundary — specified zone is skipped, relaxation
//       zone uses upwind (Req 5.8, standard transport)
// ============================================================================
TEST(ScalarTransport, RegionalBoundarySkipsSpecifiedZone) {
  LinearMesh lm;

  // Non-uniform scalar
  auto so_h = Kokkos::create_mirror_view(lm.state.scalar_old);
  auto sn_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  for (int c = 0; c < LinearMesh::nCells; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        so_h(s, k, c) = Scalar(c + 1);
        sn_h(s, k, c) = Scalar(c + 1);
      }
    }
  }
  Kokkos::deep_copy(lm.state.scalar_old, so_h);
  Kokkos::deep_copy(lm.state.scalar_new, sn_h);
  Kokkos::deep_copy(lm.state.uhAvg, Scalar(0.5));

  // Mark cell 4 in specified zone (bdyMask > nRelaxZone)
  auto bmc_h = Kokkos::create_mirror_view(lm.mesh.bdyMaskCell);
  bmc_h(0) = 0; bmc_h(1) = 0; bmc_h(2) = 0; bmc_h(3) = 0;
  bmc_h(4) = mpas::dycore::transport_nRelaxZone + 1;
  Kokkos::deep_copy(lm.mesh.bdyMaskCell, bmc_h);

  // Mark edge 3 beyond relaxation (skip)
  auto bme_h = Kokkos::create_mirror_view(lm.mesh.bdyMaskEdge);
  bme_h(0) = 0; bme_h(1) = 0; bme_h(2) = 0;
  bme_h(3) = mpas::dycore::transport_nRelaxZone + 1;
  Kokkos::deep_copy(lm.mesh.bdyMaskEdge, bme_h);

  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state, /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0), /*rk_step=*/3,
      /*config_time_integration_order=*/3,
      /*advance_density=*/true, /*config_apply_lbcs=*/true);

  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  // Cell 4 (specified zone) should remain unchanged
  for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
    for (int s = 0; s < LinearMesh::num_scalars; ++s) {
      EXPECT_NEAR(result_h(s, k, 4), sn_h(s, k, 4), kTol)
          << "DIVERGING FIELD: scalar_new — specified-zone cell 4 was "
             "modified at s=" << s << " k=" << k;
    }
  }
}

// ============================================================================
// Test: Reference_Model parity — uniform field with horizontal advection
// (Req 14.2, 14.9)
// ============================================================================
TEST(ScalarTransport, ReferenceModelParityUniform) {
  LinearMesh lm;

  // Uniform scalar=2, uniform rho=1, uniform uhAvg=0.5, no vertical flux
  Kokkos::deep_copy(lm.state.scalar_old, Scalar(2.0));
  Kokkos::deep_copy(lm.state.scalar_new, Scalar(2.0));
  Kokkos::deep_copy(lm.state.rho_zz_old, Scalar(1.0));
  Kokkos::deep_copy(lm.state.rho_zz_new, Scalar(1.0));
  Kokkos::deep_copy(lm.state.uhAvg, Scalar(0.5));
  Kokkos::deep_copy(lm.state.wwAvg, Scalar(0.0));
  Kokkos::deep_copy(lm.state.scalar_tend, Scalar(0.0));

  mpas::dycore::Scalar_Transport<ExecSpace> transport;
  transport.advance_scalars(
      lm.mesh, lm.state, /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0), /*rk_step=*/3,
      /*config_time_integration_order=*/3,
      /*advance_density=*/true, /*config_apply_lbcs=*/false);

  auto result_h = Kokkos::create_mirror_view(lm.state.scalar_new);
  Kokkos::deep_copy(result_h, lm.state.scalar_new);

  // Reference: uniform scalar -> flux=scalar at every edge -> divergence=0
  // for interior cells (inflow=outflow). scalar_new = scalar_old = 2.0.
  for (int c = 1; c < LinearMesh::nCells - 1; ++c) {
    for (int k = 0; k < LinearMesh::nVertLevels; ++k) {
      for (int s = 0; s < LinearMesh::num_scalars; ++s) {
        EXPECT_NEAR(result_h(s, k, c), Scalar(2.0), kTol)
            << "DIVERGING FIELD: scalar_new — Reference_Model parity "
               "violation (expected 2.0) at scalar=" << s
            << " k=" << k << " cell=" << c;
      }
    }
  }
}

}  // namespace
