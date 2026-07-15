#include "mpas_dycore/dissipation_module.hpp"
#include "mpas_dycore/vertical_mixing.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>

namespace {

using ES = Kokkos::DefaultHostExecutionSpace;
using MS = ES::memory_space;
using Scalar = mpas::dycore::Scalar;
using view2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, MS>;
using cview2d = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;
using view3d = Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS>;
using cview3d = Kokkos::View<const Scalar***, Kokkos::LayoutLeft, MS>;
using iview1d = Kokkos::View<int*, Kokkos::LayoutLeft, MS>;
using ciview1d = Kokkos::View<const int*, Kokkos::LayoutLeft, MS>;
using iview2d = Kokkos::View<int**, Kokkos::LayoutLeft, MS>;
using ciview2d = Kokkos::View<const int**, Kokkos::LayoutLeft, MS>;

// Helper: small mesh for testing
struct SmallMesh {
  static constexpr int nVertLevels = 4;
  static constexpr int nCells = 3;
  static constexpr int nEdges = 6;
  static constexpr int maxEdges = 4;
  static constexpr int num_scalars = 3;

  iview1d nEdgesOnCell_v;
  iview2d edgesOnCell_v;
  iview2d cellsOnEdge_v;

  SmallMesh() {
    nEdgesOnCell_v = iview1d("nEdgesOnCell", nCells);
    edgesOnCell_v = iview2d("edgesOnCell", maxEdges, nCells);
    cellsOnEdge_v = iview2d("cellsOnEdge", 2, nEdges);

    // Each cell has 3 edges (simplest unstructured mesh)
    for (int c = 0; c < nCells; ++c) {
      nEdgesOnCell_v(c) = 3;
    }

    // Simple connectivity: edges 1-3 for cell 0, 2-4 for cell 1, etc (1-based)
    for (int c = 0; c < nCells; ++c) {
      for (int e = 0; e < 3; ++e) {
        edgesOnCell_v(e, c) = (c * 2 + e) % nEdges + 1;  // 1-based
      }
    }

    // cellsOnEdge: each edge connects two cells (1-based)
    for (int e = 0; e < nEdges; ++e) {
      cellsOnEdge_v(0, e) = (e % nCells) + 1;        // 1-based
      cellsOnEdge_v(1, e) = ((e + 1) % nCells) + 1;  // 1-based
    }
  }
};

// ─── Test: 2-D Smagorinsky eddy viscosity (Req 8.1, 8.6) ─────────────────────

TEST(DissipationModule, Smagorinsky2D_ZeroDeformation_ZeroViscosity) {
  // With zero velocity, deformation is zero → eddy viscosity should be zero
  SmallMesh mesh;

  view2d kdiff("kdiff", SmallMesh::nVertLevels, SmallMesh::nCells);
  view2d u("u", SmallMesh::nVertLevels, SmallMesh::nEdges);
  view2d v("v", SmallMesh::nVertLevels, SmallMesh::nEdges);
  view2d deformation_coef_c2("c2", SmallMesh::maxEdges, SmallMesh::nCells);
  view2d deformation_coef_s2("s2", SmallMesh::maxEdges, SmallMesh::nCells);
  view2d deformation_coef_cs("cs", SmallMesh::maxEdges, SmallMesh::nCells);

  // Fill deformation coefficients with non-zero values
  Kokkos::deep_copy(deformation_coef_c2, 1.0);
  Kokkos::deep_copy(deformation_coef_s2, 0.5);
  Kokkos::deep_copy(deformation_coef_cs, 0.3);

  // u, v stay zero → deformation is zero

  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = SmallMesh::nVertLevels;
  params.nCells = SmallMesh::nCells;
  params.c_s = 0.18;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;
  params.config_visc4_2dsmag = 0.05;

  ciview1d nEdgesOnCell_c = mesh.nEdgesOnCell_v;
  ciview2d edgesOnCell_c = mesh.edgesOnCell_v;
  cview2d u_c = u;
  cview2d v_c = v;
  cview2d c2_c = deformation_coef_c2;
  cview2d s2_c = deformation_coef_s2;
  cview2d cs_c = deformation_coef_cs;

  auto result = mpas::dycore::smagorinsky_2d<ES>(
      kdiff, u_c, v_c, c2_c, s2_c, cs_c, nEdgesOnCell_c, edgesOnCell_c, params);

  // All viscosities should be zero
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      EXPECT_DOUBLE_EQ(kdiff(k, c), 0.0)
          << "kdiff(" << k << "," << c << ") should be 0 for zero deformation";
    }
  }

  // Check 4th-order filter coefficients
  const Scalar expected_visc4 = 0.05 * 3000.0 * 3000.0 * 3000.0;
  EXPECT_DOUBLE_EQ(result.h_mom_eddy_visc4, expected_visc4);
  EXPECT_DOUBLE_EQ(result.h_theta_eddy_visc4, expected_visc4);
}

TEST(DissipationModule, Smagorinsky2D_StabilityBound) {
  // With large deformation, viscosity should be bounded by stability limit
  SmallMesh mesh;

  view2d kdiff("kdiff", SmallMesh::nVertLevels, SmallMesh::nCells);
  view2d u("u", SmallMesh::nVertLevels, SmallMesh::nEdges);
  view2d v("v", SmallMesh::nVertLevels, SmallMesh::nEdges);
  view2d deformation_coef_c2("c2", SmallMesh::maxEdges, SmallMesh::nCells);
  view2d deformation_coef_s2("s2", SmallMesh::maxEdges, SmallMesh::nCells);
  view2d deformation_coef_cs("cs", SmallMesh::maxEdges, SmallMesh::nCells);

  // Large deformation coefficients and velocities
  Kokkos::deep_copy(deformation_coef_c2, 1.0);
  Kokkos::deep_copy(deformation_coef_s2, 1.0);
  Kokkos::deep_copy(deformation_coef_cs, 0.0);
  Kokkos::deep_copy(u, 1000.0);  // Very large velocity
  Kokkos::deep_copy(v, 1000.0);

  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = SmallMesh::nVertLevels;
  params.nCells = SmallMesh::nCells;
  params.c_s = 0.18;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;
  params.config_visc4_2dsmag = 0.05;

  const Scalar stability_limit = 0.01 * 3000.0 * 3000.0 / 60.0;

  ciview1d nEdgesOnCell_c = mesh.nEdgesOnCell_v;
  ciview2d edgesOnCell_c = mesh.edgesOnCell_v;
  cview2d u_c = u;
  cview2d v_c = v;
  cview2d c2_c = deformation_coef_c2;
  cview2d s2_c = deformation_coef_s2;
  cview2d cs_c = deformation_coef_cs;

  mpas::dycore::smagorinsky_2d<ES>(
      kdiff, u_c, v_c, c2_c, s2_c, cs_c, nEdgesOnCell_c, edgesOnCell_c, params);

  // All viscosities should be at the stability limit (Req 8.6)
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      EXPECT_LE(kdiff(k, c), stability_limit + 1.0e-10)
          << "kdiff(" << k << "," << c << ") exceeds stability limit";
    }
  }
}

// ─── Test: Fixed horizontal eddy viscosity (Req 8.2, 8.6) ───────────────────

TEST(DissipationModule, FixedHorizontalEddyViscosity_ValueBelowStability) {
  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = 4;
  params.nCells = 3;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;

  const Scalar fixed_visc = 100.0;  // Well below stability limit
  const Scalar stability_limit = 0.01 * 3000.0 * 3000.0 / 60.0;  // 1500

  view2d kdiff("kdiff", 4, 3);
  mpas::dycore::fixed_horizontal_eddy_viscosity<ES>(kdiff, fixed_visc, params);

  for (int k = 0; k < 4; ++k) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_DOUBLE_EQ(kdiff(k, c), fixed_visc);
    }
  }
}

TEST(DissipationModule, FixedHorizontalEddyViscosity_BoundedByStability) {
  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = 4;
  params.nCells = 3;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;

  const Scalar stability_limit = 0.01 * 3000.0 * 3000.0 / 60.0;
  const Scalar fixed_visc = stability_limit * 10.0;  // Way above limit

  view2d kdiff("kdiff", 4, 3);
  mpas::dycore::fixed_horizontal_eddy_viscosity<ES>(kdiff, fixed_visc, params);

  for (int k = 0; k < 4; ++k) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_DOUBLE_EQ(kdiff(k, c), stability_limit);
    }
  }
}

// ─── Test: Brunt-Väisälä frequency (Req 8.5) ─────────────────────────────────

TEST(DissipationModule, BruntVaisala_DryFormulation_NoCloudWater) {
  // With no cloud water (index_qc <= 0), should use dry formulation always
  constexpr int nVertLevels = 6;
  constexpr int nCells = 2;
  constexpr int num_scalars = 2;  // qv only, no qc

  view2d bn2("bn2", nVertLevels, nCells);
  view2d theta_m_v("theta_m", nVertLevels, nCells);
  view2d exner_v("exner", nVertLevels, nCells);
  view2d pressure_b_v("pressure_b", nVertLevels, nCells);
  view2d pp_v("pp", nVertLevels, nCells);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  view3d scalars_v("scalars", num_scalars, nVertLevels, nCells);
  view2d qtot_v("qtot", nVertLevels, nCells);

  // Set up a simple stable atmosphere
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k) {
      // theta increases with height (stable)
      theta_m_v(k, c) = 300.0 + 5.0 * k;
      exner_v(k, c) = 1.0 - 0.01 * k;
      pressure_b_v(k, c) = 100000.0 - 10000.0 * k;
      pp_v(k, c) = 0.0;
      scalars_v(0, k, c) = 0.01;  // qv = 0.01 kg/kg
      qtot_v(k, c) = 0.01;
    }
    for (int k = 0; k <= nVertLevels; ++k) {
      zgrid_v(k, c) = 500.0 * k;  // uniform 500m layers
    }
  }

  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.num_scalars = num_scalars;
  params.index_qv = 1;   // 1-based
  params.index_qc = 0;   // <=0 means no cloud water → always dry

  cview2d theta_m_c = theta_m_v;
  cview2d exner_c = exner_v;
  cview2d pressure_b_c = pressure_b_v;
  cview2d pp_c = pp_v;
  cview2d zgrid_c = zgrid_v;
  cview3d scalars_c = scalars_v;
  cview2d qtot_c = qtot_v;

  mpas::dycore::calculate_brunt_vaisala<ES>(
      bn2, theta_m_c, exner_c, pressure_b_c, pp_c, zgrid_c, scalars_c, qtot_c, params);

  // With stable atmosphere (theta increasing with height), N^2 should be > 0
  // for interior levels
  for (int c = 0; c < nCells; ++c) {
    for (int k = 1; k < nVertLevels - 1; ++k) {
      EXPECT_GT(bn2(k, c), 0.0)
          << "bn2(" << k << "," << c << ") should be positive for stable atmosphere";
    }
    // Boundary levels should be copies of adjacent interior
    EXPECT_DOUBLE_EQ(bn2(0, c), bn2(1, c));
    EXPECT_DOUBLE_EQ(bn2(nVertLevels - 1, c), bn2(nVertLevels - 2, c));
  }
}

TEST(DissipationModule, BruntVaisala_MoistFormulation_WithCloudWater) {
  // With cloud water above threshold, should use moist formulation
  constexpr int nVertLevels = 6;
  constexpr int nCells = 2;
  constexpr int num_scalars = 3;  // qv, qc, tke

  view2d bn2("bn2", nVertLevels, nCells);
  view2d theta_m_v("theta_m", nVertLevels, nCells);
  view2d exner_v("exner", nVertLevels, nCells);
  view2d pressure_b_v("pressure_b", nVertLevels, nCells);
  view2d pp_v("pp", nVertLevels, nCells);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  view3d scalars_v("scalars", num_scalars, nVertLevels, nCells);
  view2d qtot_v("qtot", nVertLevels, nCells);

  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k) {
      theta_m_v(k, c) = 300.0 + 3.0 * k;
      exner_v(k, c) = 1.0 - 0.01 * k;
      pressure_b_v(k, c) = 100000.0 - 10000.0 * k;
      pp_v(k, c) = 0.0;
      scalars_v(0, k, c) = 0.01;       // qv
      scalars_v(1, k, c) = 0.001;      // qc above threshold (0.00001)
      scalars_v(2, k, c) = 0.0;        // tke
      qtot_v(k, c) = 0.011;
    }
    for (int k = 0; k <= nVertLevels; ++k) {
      zgrid_v(k, c) = 500.0 * k;
    }
  }

  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.num_scalars = num_scalars;
  params.index_qv = 1;   // 1-based
  params.index_qc = 2;   // 1-based, qc present

  cview2d theta_m_c = theta_m_v;
  cview2d exner_c = exner_v;
  cview2d pressure_b_c = pressure_b_v;
  cview2d pp_c = pp_v;
  cview2d zgrid_c = zgrid_v;
  cview3d scalars_c = scalars_v;
  cview2d qtot_c = qtot_v;

  mpas::dycore::calculate_brunt_vaisala<ES>(
      bn2, theta_m_c, exner_c, pressure_b_c, pp_c, zgrid_c, scalars_c, qtot_c, params);

  // With cloud water above threshold, the moist formulation is used.
  // It should still produce finite non-NaN values for a stable atmosphere.
  for (int c = 0; c < nCells; ++c) {
    for (int k = 1; k < nVertLevels - 1; ++k) {
      EXPECT_FALSE(std::isnan(bn2(k, c)))
          << "bn2(" << k << "," << c << ") is NaN";
      EXPECT_FALSE(std::isinf(bn2(k, c)))
          << "bn2(" << k << "," << c << ") is Inf";
    }
  }
}

// ─── Test: Fourth-order horizontal hyperdiffusion (Req 8.7) ──────────────────

// Helper: Extended mesh for hyperdiffusion tests (needs vertices, signs, etc.)
struct HyperdiffMesh {
  static constexpr int nVertLevels = 3;
  static constexpr int nCells = 4;
  static constexpr int nEdges = 6;
  static constexpr int nVertices = 4;
  static constexpr int maxEdges = 3;
  static constexpr int vertexDegree = 3;
  static constexpr int num_scalars = 2;

  // Connectivity
  iview2d cellsOnEdge_v;
  iview2d verticesOnEdge_v;
  iview2d edgesOnCell_v;
  iview2d edgesOnVertex_v;
  iview1d nEdgesOnCell_v;

  // Geometry and signs
  using sview1d = Kokkos::View<Scalar*, Kokkos::LayoutLeft, MS>;
  using sview2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, MS>;

  sview2d edgesOnCell_sign_v;
  sview2d edgesOnVertex_sign_v;
  sview1d invAreaCell_v;
  sview1d invAreaTriangle_v;
  sview1d invDcEdge_v;
  sview1d invDvEdge_v;
  sview1d dvEdge_v;
  sview1d dcEdge_v;
  sview1d meshScalingDel4_v;

  HyperdiffMesh() {
    cellsOnEdge_v = iview2d("cellsOnEdge", 2, nEdges);
    verticesOnEdge_v = iview2d("verticesOnEdge", 2, nEdges);
    edgesOnCell_v = iview2d("edgesOnCell", maxEdges, nCells);
    edgesOnVertex_v = iview2d("edgesOnVertex", vertexDegree, nVertices);
    nEdgesOnCell_v = iview1d("nEdgesOnCell", nCells);
    edgesOnCell_sign_v = sview2d("edgesOnCell_sign", maxEdges, nCells);
    edgesOnVertex_sign_v = sview2d("edgesOnVertex_sign", vertexDegree, nVertices);
    invAreaCell_v = sview1d("invAreaCell", nCells);
    invAreaTriangle_v = sview1d("invAreaTriangle", nVertices);
    invDcEdge_v = sview1d("invDcEdge", nEdges);
    invDvEdge_v = sview1d("invDvEdge", nEdges);
    dvEdge_v = sview1d("dvEdge", nEdges);
    dcEdge_v = sview1d("dcEdge", nEdges);
    meshScalingDel4_v = sview1d("meshScalingDel4", nEdges);

    // Set up a simple mesh connectivity (1-based indices)
    for (int c = 0; c < nCells; ++c) {
      nEdgesOnCell_v(c) = 3;
      for (int e = 0; e < 3; ++e) {
        edgesOnCell_v(e, c) = (c + e) % nEdges + 1;
        edgesOnCell_sign_v(e, c) = (e % 2 == 0) ? 1.0 : -1.0;
      }
    }

    for (int e = 0; e < nEdges; ++e) {
      cellsOnEdge_v(0, e) = (e % nCells) + 1;
      cellsOnEdge_v(1, e) = ((e + 1) % nCells) + 1;
      verticesOnEdge_v(0, e) = (e % nVertices) + 1;
      verticesOnEdge_v(1, e) = ((e + 1) % nVertices) + 1;
    }

    for (int v = 0; v < nVertices; ++v) {
      for (int i = 0; i < vertexDegree; ++i) {
        edgesOnVertex_v(i, v) = (v + i) % nEdges + 1;
        edgesOnVertex_sign_v(i, v) = (i % 2 == 0) ? 1.0 : -1.0;
      }
    }

    // Geometry: uniform mesh with dcEdge=1000, dvEdge=1000, area=1e6
    for (int e = 0; e < nEdges; ++e) {
      dcEdge_v(e) = 1000.0;
      dvEdge_v(e) = 1000.0;
      invDcEdge_v(e) = 1.0 / 1000.0;
      invDvEdge_v(e) = 1.0 / 1000.0;
      meshScalingDel4_v(e) = 1.0;
    }
    for (int c = 0; c < nCells; ++c) {
      invAreaCell_v(c) = 1.0 / 1.0e6;
    }
    for (int v = 0; v < nVertices; ++v) {
      invAreaTriangle_v(v) = 1.0 / 5.0e5;
    }
  }
};

TEST(DissipationModule, Hyperdiffusion_ZeroField_ZeroTendency) {
  // With zero input fields, tendency should remain zero
  HyperdiffMesh mesh;
  constexpr int nk = HyperdiffMesh::nVertLevels;
  constexpr int nc = HyperdiffMesh::nCells;
  constexpr int ne = HyperdiffMesh::nEdges;
  constexpr int nv = HyperdiffMesh::nVertices;
  constexpr int ns = HyperdiffMesh::num_scalars;

  view2d tend_u("tend_u", nk, ne);
  view2d tend_w("tend_w", nk + 1, nc);
  view2d tend_theta("tend_theta", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> tend_scalars("tend_s", ns, nk, nc);
  view2d u_v("u", nk, ne);
  view2d w_v("w", nk + 1, nc);
  view2d theta_m_v("theta_m", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> scalars_v("scalars", ns, nk, nc);
  view2d div_v("div", nk, nc);
  view2d vort_v("vort", nk, nv);
  view2d rho_edge_v("rho_edge", nk, ne);

  Kokkos::deep_copy(rho_edge_v, 1.0);  // unit density

  mpas::dycore::HyperdiffusionParams params;
  params.h_mom_eddy_visc4 = 1.0e10;
  params.h_theta_eddy_visc4 = 1.0e10;
  params.config_del4u_div_factor = 1.0;
  params.prandtl_inv = 1.0;
  params.config_mix_scalars = true;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.nVertices = nv;
  params.vertexDegree = HyperdiffMesh::vertexDegree;
  params.num_scalars = ns;

  using cview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;
  using cview3d_t = Kokkos::View<const Scalar***, Kokkos::LayoutLeft, MS>;
  using ciview1d_t = Kokkos::View<const int*, Kokkos::LayoutLeft, MS>;
  using ciview2d_t = Kokkos::View<const int**, Kokkos::LayoutLeft, MS>;
  using csview1d_t = Kokkos::View<const Scalar*, Kokkos::LayoutLeft, MS>;
  using csview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;

  mpas::dycore::apply_hyperdiffusion<ES>(
      tend_u, tend_w, tend_theta, tend_scalars,
      cview2d_t(u_v), cview2d_t(w_v), cview2d_t(theta_m_v), cview3d_t(scalars_v),
      cview2d_t(div_v), cview2d_t(vort_v), cview2d_t(rho_edge_v),
      ciview2d_t(mesh.cellsOnEdge_v), ciview2d_t(mesh.verticesOnEdge_v),
      ciview2d_t(mesh.edgesOnCell_v), ciview2d_t(mesh.edgesOnVertex_v),
      ciview1d_t(mesh.nEdgesOnCell_v),
      csview2d_t(mesh.edgesOnCell_sign_v), csview2d_t(mesh.edgesOnVertex_sign_v),
      csview1d_t(mesh.invAreaCell_v), csview1d_t(mesh.invAreaTriangle_v),
      csview1d_t(mesh.invDcEdge_v), csview1d_t(mesh.invDvEdge_v),
      csview1d_t(mesh.dvEdge_v), csview1d_t(mesh.dcEdge_v),
      csview1d_t(mesh.meshScalingDel4_v),
      params);

  // All tendencies should remain zero when fields are uniform (zero)
  for (int k = 0; k < nk; ++k) {
    for (int e = 0; e < ne; ++e) {
      EXPECT_DOUBLE_EQ(tend_u(k, e), 0.0)
          << "tend_u(" << k << "," << e << ") should be 0 for zero fields";
    }
  }
  for (int k = 1; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      EXPECT_DOUBLE_EQ(tend_w(k, c), 0.0)
          << "tend_w(" << k << "," << c << ") should be 0 for zero fields";
    }
  }
  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      EXPECT_DOUBLE_EQ(tend_theta(k, c), 0.0)
          << "tend_theta(" << k << "," << c << ") should be 0 for zero fields";
    }
  }
}

TEST(DissipationModule, Hyperdiffusion_UniformField_ZeroTendency) {
  // With uniform (non-zero) fields, gradients are zero → tendency stays zero
  HyperdiffMesh mesh;
  constexpr int nk = HyperdiffMesh::nVertLevels;
  constexpr int nc = HyperdiffMesh::nCells;
  constexpr int ne = HyperdiffMesh::nEdges;
  constexpr int nv = HyperdiffMesh::nVertices;
  constexpr int ns = HyperdiffMesh::num_scalars;

  view2d tend_u("tend_u", nk, ne);
  view2d tend_w("tend_w", nk + 1, nc);
  view2d tend_theta("tend_theta", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> tend_scalars("tend_s", ns, nk, nc);
  view2d u_v("u", nk, ne);
  view2d w_v("w", nk + 1, nc);
  view2d theta_m_v("theta_m", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> scalars_v("scalars", ns, nk, nc);
  view2d div_v("div", nk, nc);
  view2d vort_v("vort", nk, nv);
  view2d rho_edge_v("rho_edge", nk, ne);

  // Uniform fields: gradients should be zero
  Kokkos::deep_copy(u_v, 10.0);
  Kokkos::deep_copy(w_v, 5.0);
  Kokkos::deep_copy(theta_m_v, 300.0);
  Kokkos::deep_copy(scalars_v, 0.01);
  Kokkos::deep_copy(div_v, 2.0);     // uniform divergence
  Kokkos::deep_copy(vort_v, 1.5);    // uniform vorticity
  Kokkos::deep_copy(rho_edge_v, 1.2);

  mpas::dycore::HyperdiffusionParams params;
  params.h_mom_eddy_visc4 = 1.0e10;
  params.h_theta_eddy_visc4 = 1.0e10;
  params.config_del4u_div_factor = 1.0;
  params.prandtl_inv = 1.0;
  params.config_mix_scalars = true;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.nVertices = nv;
  params.vertexDegree = HyperdiffMesh::vertexDegree;
  params.num_scalars = ns;

  using cview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;
  using cview3d_t = Kokkos::View<const Scalar***, Kokkos::LayoutLeft, MS>;
  using ciview1d_t = Kokkos::View<const int*, Kokkos::LayoutLeft, MS>;
  using ciview2d_t = Kokkos::View<const int**, Kokkos::LayoutLeft, MS>;
  using csview1d_t = Kokkos::View<const Scalar*, Kokkos::LayoutLeft, MS>;
  using csview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;

  mpas::dycore::apply_hyperdiffusion<ES>(
      tend_u, tend_w, tend_theta, tend_scalars,
      cview2d_t(u_v), cview2d_t(w_v), cview2d_t(theta_m_v), cview3d_t(scalars_v),
      cview2d_t(div_v), cview2d_t(vort_v), cview2d_t(rho_edge_v),
      ciview2d_t(mesh.cellsOnEdge_v), ciview2d_t(mesh.verticesOnEdge_v),
      ciview2d_t(mesh.edgesOnCell_v), ciview2d_t(mesh.edgesOnVertex_v),
      ciview1d_t(mesh.nEdgesOnCell_v),
      csview2d_t(mesh.edgesOnCell_sign_v), csview2d_t(mesh.edgesOnVertex_sign_v),
      csview1d_t(mesh.invAreaCell_v), csview1d_t(mesh.invAreaTriangle_v),
      csview1d_t(mesh.invDcEdge_v), csview1d_t(mesh.invDvEdge_v),
      csview1d_t(mesh.dvEdge_v), csview1d_t(mesh.dcEdge_v),
      csview1d_t(mesh.meshScalingDel4_v),
      params);

  // For u: delsq_u uses (div(c2)-div(c1)) and (vort(v2)-vort(v1))
  // With uniform fields these gradients are zero → delsq_u=0 → tend_u=0
  for (int k = 0; k < nk; ++k) {
    for (int e = 0; e < ne; ++e) {
      EXPECT_DOUBLE_EQ(tend_u(k, e), 0.0)
          << "tend_u(" << k << "," << e << ") should be 0 for uniform fields";
    }
  }

  // For w and theta: (f(cell2)-f(cell1)) = 0 for uniform fields → delsq=0
  for (int k = 1; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      EXPECT_DOUBLE_EQ(tend_w(k, c), 0.0)
          << "tend_w(" << k << "," << c << ") should be 0 for uniform fields";
    }
  }
  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      EXPECT_DOUBLE_EQ(tend_theta(k, c), 0.0)
          << "tend_theta(" << k << "," << c << ") should be 0 for uniform fields";
    }
  }
}

TEST(DissipationModule, Hyperdiffusion_ZeroCoefficients_NoEffect) {
  // With zero viscosity coefficients, no filtering should be applied
  HyperdiffMesh mesh;
  constexpr int nk = HyperdiffMesh::nVertLevels;
  constexpr int nc = HyperdiffMesh::nCells;
  constexpr int ne = HyperdiffMesh::nEdges;
  constexpr int nv = HyperdiffMesh::nVertices;
  constexpr int ns = HyperdiffMesh::num_scalars;

  view2d tend_u("tend_u", nk, ne);
  view2d tend_w("tend_w", nk + 1, nc);
  view2d tend_theta("tend_theta", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> tend_scalars("tend_s", ns, nk, nc);
  view2d u_v("u", nk, ne);
  view2d w_v("w", nk + 1, nc);
  view2d theta_m_v("theta_m", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> scalars_v("scalars", ns, nk, nc);
  view2d div_v("div", nk, nc);
  view2d vort_v("vort", nk, nv);
  view2d rho_edge_v("rho_edge", nk, ne);

  // Non-uniform fields to produce gradients
  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      theta_m_v(k, c) = 300.0 + 10.0 * c;
      div_v(k, c) = 0.001 * c;
    }
  }
  Kokkos::deep_copy(rho_edge_v, 1.0);

  // Pre-fill tendencies to check they aren't modified
  Kokkos::deep_copy(tend_u, 42.0);
  Kokkos::deep_copy(tend_w, 42.0);
  Kokkos::deep_copy(tend_theta, 42.0);

  mpas::dycore::HyperdiffusionParams params;
  params.h_mom_eddy_visc4 = 0.0;    // disabled!
  params.h_theta_eddy_visc4 = 0.0;  // disabled!
  params.config_del4u_div_factor = 1.0;
  params.prandtl_inv = 1.0;
  params.config_mix_scalars = true;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.nVertices = nv;
  params.vertexDegree = HyperdiffMesh::vertexDegree;
  params.num_scalars = ns;

  using cview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;
  using cview3d_t = Kokkos::View<const Scalar***, Kokkos::LayoutLeft, MS>;
  using ciview1d_t = Kokkos::View<const int*, Kokkos::LayoutLeft, MS>;
  using ciview2d_t = Kokkos::View<const int**, Kokkos::LayoutLeft, MS>;
  using csview1d_t = Kokkos::View<const Scalar*, Kokkos::LayoutLeft, MS>;
  using csview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;

  mpas::dycore::apply_hyperdiffusion<ES>(
      tend_u, tend_w, tend_theta, tend_scalars,
      cview2d_t(u_v), cview2d_t(w_v), cview2d_t(theta_m_v), cview3d_t(scalars_v),
      cview2d_t(div_v), cview2d_t(vort_v), cview2d_t(rho_edge_v),
      ciview2d_t(mesh.cellsOnEdge_v), ciview2d_t(mesh.verticesOnEdge_v),
      ciview2d_t(mesh.edgesOnCell_v), ciview2d_t(mesh.edgesOnVertex_v),
      ciview1d_t(mesh.nEdgesOnCell_v),
      csview2d_t(mesh.edgesOnCell_sign_v), csview2d_t(mesh.edgesOnVertex_sign_v),
      csview1d_t(mesh.invAreaCell_v), csview1d_t(mesh.invAreaTriangle_v),
      csview1d_t(mesh.invDcEdge_v), csview1d_t(mesh.invDvEdge_v),
      csview1d_t(mesh.dvEdge_v), csview1d_t(mesh.dcEdge_v),
      csview1d_t(mesh.meshScalingDel4_v),
      params);

  // Tendencies should remain at their initial values (42.0)
  for (int k = 0; k < nk; ++k) {
    for (int e = 0; e < ne; ++e) {
      EXPECT_DOUBLE_EQ(tend_u(k, e), 42.0)
          << "tend_u should be unmodified when visc4=0";
    }
    for (int c = 0; c < nc; ++c) {
      EXPECT_DOUBLE_EQ(tend_theta(k, c), 42.0)
          << "tend_theta should be unmodified when visc4=0";
    }
  }
  for (int k = 0; k <= nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      EXPECT_DOUBLE_EQ(tend_w(k, c), 42.0)
          << "tend_w should be unmodified when visc4=0";
    }
  }
}

TEST(DissipationModule, Hyperdiffusion_ScalarMixingDisabled_NoScalarEffect) {
  // When config_mix_scalars is false, scalars should not be affected
  HyperdiffMesh mesh;
  constexpr int nk = HyperdiffMesh::nVertLevels;
  constexpr int nc = HyperdiffMesh::nCells;
  constexpr int ne = HyperdiffMesh::nEdges;
  constexpr int nv = HyperdiffMesh::nVertices;
  constexpr int ns = HyperdiffMesh::num_scalars;

  view2d tend_u("tend_u", nk, ne);
  view2d tend_w("tend_w", nk + 1, nc);
  view2d tend_theta("tend_theta", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> tend_scalars("tend_s", ns, nk, nc);
  view2d u_v("u", nk, ne);
  view2d w_v("w", nk + 1, nc);
  view2d theta_m_v("theta_m", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> scalars_v("scalars", ns, nk, nc);
  view2d div_v("div", nk, nc);
  view2d vort_v("vort", nk, nv);
  view2d rho_edge_v("rho_edge", nk, ne);

  // Non-uniform scalars
  for (int s = 0; s < ns; ++s)
    for (int k = 0; k < nk; ++k)
      for (int c = 0; c < nc; ++c)
        scalars_v(s, k, c) = 0.01 * (c + 1);

  Kokkos::deep_copy(rho_edge_v, 1.0);
  Kokkos::deep_copy(tend_scalars, 99.0);

  mpas::dycore::HyperdiffusionParams params;
  params.h_mom_eddy_visc4 = 0.0;
  params.h_theta_eddy_visc4 = 1.0e10;
  params.config_del4u_div_factor = 1.0;
  params.prandtl_inv = 1.0;
  params.config_mix_scalars = false;  // disabled!
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.nVertices = nv;
  params.vertexDegree = HyperdiffMesh::vertexDegree;
  params.num_scalars = ns;

  using cview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;
  using cview3d_t = Kokkos::View<const Scalar***, Kokkos::LayoutLeft, MS>;
  using ciview1d_t = Kokkos::View<const int*, Kokkos::LayoutLeft, MS>;
  using ciview2d_t = Kokkos::View<const int**, Kokkos::LayoutLeft, MS>;
  using csview1d_t = Kokkos::View<const Scalar*, Kokkos::LayoutLeft, MS>;
  using csview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;

  mpas::dycore::apply_hyperdiffusion<ES>(
      tend_u, tend_w, tend_theta, tend_scalars,
      cview2d_t(u_v), cview2d_t(w_v), cview2d_t(theta_m_v), cview3d_t(scalars_v),
      cview2d_t(div_v), cview2d_t(vort_v), cview2d_t(rho_edge_v),
      ciview2d_t(mesh.cellsOnEdge_v), ciview2d_t(mesh.verticesOnEdge_v),
      ciview2d_t(mesh.edgesOnCell_v), ciview2d_t(mesh.edgesOnVertex_v),
      ciview1d_t(mesh.nEdgesOnCell_v),
      csview2d_t(mesh.edgesOnCell_sign_v), csview2d_t(mesh.edgesOnVertex_sign_v),
      csview1d_t(mesh.invAreaCell_v), csview1d_t(mesh.invAreaTriangle_v),
      csview1d_t(mesh.invDcEdge_v), csview1d_t(mesh.invDvEdge_v),
      csview1d_t(mesh.dvEdge_v), csview1d_t(mesh.dcEdge_v),
      csview1d_t(mesh.meshScalingDel4_v),
      params);

  // Scalar tendencies should remain at 99.0 (unmodified)
  for (int s = 0; s < ns; ++s)
    for (int k = 0; k < nk; ++k)
      for (int c = 0; c < nc; ++c)
        EXPECT_DOUBLE_EQ(tend_scalars(s, k, c), 99.0)
            << "tend_scalars should be unmodified when mix_scalars is false";
}

// ─── Test: LES option-string mapping re-exported from Dissipation_Module ─────

TEST(DissipationModule, LesModelFromString) {
  using DM = mpas::dycore::Dissipation_Module;
  EXPECT_EQ(DM::les_model_opt_from_string("none"), mpas::dycore::LES_MODEL_NONE);
  EXPECT_EQ(DM::les_model_opt_from_string("3d_smagorinsky"),
            mpas::dycore::LES_MODEL_3D_SMAGORINSKY);
  EXPECT_EQ(DM::les_model_opt_from_string("prognostic_1.5_order"),
            mpas::dycore::LES_MODEL_PROGNOSTIC_15_ORDER);
  EXPECT_EQ(DM::les_model_opt_from_string("invalid"), mpas::dycore::LES_INVALID_OPT);
}

TEST(DissipationModule, LesSurfaceFromString) {
  using DM = mpas::dycore::Dissipation_Module;
  EXPECT_EQ(DM::les_surface_opt_from_string("none"), mpas::dycore::LES_SURFACE_NONE);
  EXPECT_EQ(DM::les_surface_opt_from_string("specified"),
            mpas::dycore::LES_SURFACE_SPECIFIED);
  EXPECT_EQ(DM::les_surface_opt_from_string("varying"),
            mpas::dycore::LES_SURFACE_VARYING);
  EXPECT_EQ(DM::les_surface_opt_from_string("bad"), mpas::dycore::LES_INVALID_OPT);
}

// ─── Test: 3-D Smagorinsky LES model (Req 8.3, 8.6) ─────────────────────────

TEST(DissipationModule, LesModels_3DSmagorinsky_StabilityBound) {
  SmallMesh mesh;
  constexpr int nk = SmallMesh::nVertLevels;
  constexpr int nc = SmallMesh::nCells;
  constexpr int ne = SmallMesh::nEdges;
  constexpr int ns = SmallMesh::num_scalars;

  view2d eddy_visc_horz("ev_h", nk, nc);
  view2d eddy_visc_vert("ev_v", nk, nc);
  view2d prandtl_3d_inv("pr3d", nk, nc);
  view2d u_v("u", nk, ne);
  view2d v_v("v", nk, ne);
  view2d uCell_v("uCell", nk, nc);
  view2d vCell_v("vCell", nk, nc);
  view2d w_v("w", nk + 1, nc);
  view2d bv_freq2_v("bv_freq2", nk, nc);
  view2d zgrid_v("zgrid", nk + 1, nc);
  view2d rho_zz_v("rho_zz", nk, nc);
  view3d scalars_v("scalars", ns, nk, nc);
  view3d tend_scalars_v("tend_s", ns, nk, nc);
  view2d deform_c2("c2", SmallMesh::maxEdges, nc);
  view2d deform_s2("s2", SmallMesh::maxEdges, nc);
  view2d deform_cs("cs", SmallMesh::maxEdges, nc);
  view2d deform_c("c", SmallMesh::maxEdges, nc);
  view2d deform_s("s", SmallMesh::maxEdges, nc);

  // Large velocities to exceed stability limit
  Kokkos::deep_copy(u_v, 500.0);
  Kokkos::deep_copy(v_v, 500.0);
  Kokkos::deep_copy(uCell_v, 500.0);
  Kokkos::deep_copy(vCell_v, 500.0);
  Kokkos::deep_copy(w_v, 100.0);
  Kokkos::deep_copy(bv_freq2_v, 0.0001);
  Kokkos::deep_copy(rho_zz_v, 1.0);
  Kokkos::deep_copy(deform_c2, 1.0);
  Kokkos::deep_copy(deform_s2, 0.5);
  Kokkos::deep_copy(deform_cs, 0.3);
  Kokkos::deep_copy(deform_c, 0.8);
  Kokkos::deep_copy(deform_s, 0.6);

  for (int c = 0; c < nc; ++c) {
    for (int k = 0; k <= nk; ++k) {
      zgrid_v(k, c) = 500.0 * k;
    }
  }

  mpas::dycore::EddyViscosityParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_3D_SMAGORINSKY;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.maxEdges = SmallMesh::maxEdges;
  params.c_s = 0.18;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;
  params.config_visc4_2dsmag = 0.05;
  params.dynamics_substep = 1;
  params.index_tke = 1;
  params.num_scalars = ns;

  const Scalar h_limit = 0.01 * 3000.0 * 3000.0 / 60.0;

  ciview1d nEdgesOnCell_c = mesh.nEdgesOnCell_v;
  ciview2d edgesOnCell_c = mesh.edgesOnCell_v;
  ciview2d cellsOnEdge_c = mesh.cellsOnEdge_v;
  cview2d u_c = u_v, v_c = v_v, uCell_c = uCell_v;
  cview2d vCell_c = vCell_v, w_c = w_v;
  cview2d bv_c = bv_freq2_v, zgrid_c = zgrid_v, rho_c = rho_zz_v;
  cview2d c2_c = deform_c2, s2_c = deform_s2;
  cview2d cs_c = deform_cs, dc_c = deform_c, ds_c = deform_s;

  auto result = mpas::dycore::les_models<ES>(
      eddy_visc_horz, eddy_visc_vert, prandtl_3d_inv,
      u_c, v_c, uCell_c, vCell_c, w_c, bv_c, zgrid_c,
      rho_c, scalars_v, tend_scalars_v,
      c2_c, s2_c, cs_c, dc_c, ds_c,
      nEdgesOnCell_c, edgesOnCell_c, cellsOnEdge_c, params);

  // Stability bound check (Req 8.6)
  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      EXPECT_LE(eddy_visc_horz(k, c), h_limit + 1.0e-10)
          << "3D Smag h-visc exceeds stability at k=" << k;
      const Scalar dz = 500.0;
      const Scalar v_limit = 0.01 * dz * dz / 60.0;
      EXPECT_LE(eddy_visc_vert(k, c), v_limit + 1.0e-10)
          << "3D Smag v-visc exceeds stability at k=" << k;
      EXPECT_GE(eddy_visc_horz(k, c), 0.0);
      EXPECT_GE(eddy_visc_vert(k, c), 0.0);
    }
  }

  EXPECT_GT(result.h_mom_eddy_visc4, 0.0);
}

// ─── Test: Prognostic 1.5-order TKE (Req 8.4, 8.6) ─────────────────────────

TEST(DissipationModule, LesModels_PrognosticTKE_TendencyAndBound) {
  SmallMesh mesh;
  constexpr int nk = SmallMesh::nVertLevels;
  constexpr int nc = SmallMesh::nCells;
  constexpr int ne = SmallMesh::nEdges;
  constexpr int ns = SmallMesh::num_scalars;
  constexpr int idx_tke = 1;  // 1-based

  view2d eddy_visc_horz("ev_h", nk, nc);
  view2d eddy_visc_vert("ev_v", nk, nc);
  view2d prandtl_3d_inv("pr3d", nk, nc);
  view2d u_v("u", nk, ne);
  view2d v_v("v", nk, ne);
  view2d uCell_v("uCell", nk, nc);
  view2d vCell_v("vCell", nk, nc);
  view2d w_v("w", nk + 1, nc);
  view2d bv_freq2_v("bv_freq2", nk, nc);
  view2d zgrid_v("zgrid", nk + 1, nc);
  view2d rho_zz_v("rho_zz", nk, nc);
  view3d scalars_v("scalars", ns, nk, nc);
  view3d tend_scalars_v("tend_s", ns, nk, nc);
  view2d deform_c2("c2", SmallMesh::maxEdges, nc);
  view2d deform_s2("s2", SmallMesh::maxEdges, nc);
  view2d deform_cs("cs", SmallMesh::maxEdges, nc);
  view2d deform_c("c", SmallMesh::maxEdges, nc);
  view2d deform_s("s", SmallMesh::maxEdges, nc);

  Kokkos::deep_copy(u_v, 10.0);
  Kokkos::deep_copy(v_v, 10.0);
  Kokkos::deep_copy(uCell_v, 10.0);
  Kokkos::deep_copy(vCell_v, 10.0);
  Kokkos::deep_copy(w_v, 1.0);
  Kokkos::deep_copy(bv_freq2_v, 0.001);
  Kokkos::deep_copy(rho_zz_v, 1.2);
  Kokkos::deep_copy(deform_c2, 1.0);
  Kokkos::deep_copy(deform_s2, 0.5);
  Kokkos::deep_copy(deform_cs, 0.3);
  Kokkos::deep_copy(deform_c, 0.8);
  Kokkos::deep_copy(deform_s, 0.6);

  // Set TKE to a positive value
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c)
      scalars_v(idx_tke - 1, k, c) = 2.0;  // 0-based

  for (int c = 0; c < nc; ++c)
    for (int k = 0; k <= nk; ++k)
      zgrid_v(k, c) = 500.0 * k;

  Kokkos::deep_copy(tend_scalars_v, 0.0);

  mpas::dycore::EddyViscosityParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_PROGNOSTIC_15_ORDER;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.maxEdges = SmallMesh::maxEdges;
  params.c_s = 0.18;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;
  params.config_visc4_2dsmag = 0.05;
  params.dynamics_substep = 1;  // TKE tendency computed on substep 1
  params.index_tke = idx_tke;
  params.num_scalars = ns;

  ciview1d nEdgesOnCell_c = mesh.nEdgesOnCell_v;
  ciview2d edgesOnCell_c = mesh.edgesOnCell_v;
  ciview2d cellsOnEdge_c = mesh.cellsOnEdge_v;
  cview2d u_c = u_v, v_c = v_v, uCell_c = uCell_v;
  cview2d vCell_c = vCell_v, w_c = w_v;
  cview2d bv_c = bv_freq2_v, zgrid_c = zgrid_v, rho_c = rho_zz_v;
  cview2d c2_c = deform_c2, s2_c = deform_s2;
  cview2d cs_c = deform_cs, dc_c = deform_c, ds_c = deform_s;

  mpas::dycore::les_models<ES>(
      eddy_visc_horz, eddy_visc_vert, prandtl_3d_inv,
      u_c, v_c, uCell_c, vCell_c, w_c, bv_c, zgrid_c,
      rho_c, scalars_v, tend_scalars_v,
      c2_c, s2_c, cs_c, dc_c, ds_c,
      nEdgesOnCell_c, edgesOnCell_c, cellsOnEdge_c, params);

  const Scalar h_limit = 0.01 * 3000.0 * 3000.0 / 60.0;

  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      // Eddy viscosities should be positive and bounded
      EXPECT_GE(eddy_visc_horz(k, c), 0.0);
      EXPECT_LE(eddy_visc_horz(k, c), h_limit + 1.0e-10);
      EXPECT_GE(eddy_visc_vert(k, c), 0.0);
      const Scalar dz = 500.0;
      const Scalar v_limit = 0.01 * dz * dz / 60.0;
      EXPECT_LE(eddy_visc_vert(k, c), v_limit + 1.0e-10);

      // TKE tendency should be non-zero (has shear production)
      EXPECT_FALSE(std::isnan(tend_scalars_v(idx_tke - 1, k, c)));

      // Prandtl_3d_inv should be >= 1
      EXPECT_GE(prandtl_3d_inv(k, c), 1.0);
    }
  }

  // TKE tendency on substep 1 should have been computed
  bool has_nonzero_tend = false;
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c)
      if (tend_scalars_v(idx_tke - 1, k, c) != 0.0)
        has_nonzero_tend = true;
  EXPECT_TRUE(has_nonzero_tend)
      << "TKE tendency should be non-zero on dynamics_substep==1";
}

TEST(DissipationModule, LesModels_PrognosticTKE_NoTendencyOnSubstep2) {
  SmallMesh mesh;
  constexpr int nk = SmallMesh::nVertLevels;
  constexpr int nc = SmallMesh::nCells;
  constexpr int ne = SmallMesh::nEdges;
  constexpr int ns = SmallMesh::num_scalars;
  constexpr int idx_tke = 1;

  view2d eddy_visc_horz("ev_h", nk, nc);
  view2d eddy_visc_vert("ev_v", nk, nc);
  view2d prandtl_3d_inv("pr3d", nk, nc);
  view2d u_v("u", nk, ne);
  view2d v_v("v", nk, ne);
  view2d uCell_v("uCell", nk, nc);
  view2d vCell_v("vCell", nk, nc);
  view2d w_v("w", nk + 1, nc);
  view2d bv_freq2_v("bv_freq2", nk, nc);
  view2d zgrid_v("zgrid", nk + 1, nc);
  view2d rho_zz_v("rho_zz", nk, nc);
  view3d scalars_v("scalars", ns, nk, nc);
  view3d tend_scalars_v("tend_s", ns, nk, nc);
  view2d deform_c2("c2", SmallMesh::maxEdges, nc);
  view2d deform_s2("s2", SmallMesh::maxEdges, nc);
  view2d deform_cs("cs", SmallMesh::maxEdges, nc);
  view2d deform_c("c", SmallMesh::maxEdges, nc);
  view2d deform_s("s", SmallMesh::maxEdges, nc);

  Kokkos::deep_copy(u_v, 10.0);
  Kokkos::deep_copy(v_v, 10.0);
  Kokkos::deep_copy(uCell_v, 10.0);
  Kokkos::deep_copy(vCell_v, 10.0);
  Kokkos::deep_copy(w_v, 1.0);
  Kokkos::deep_copy(bv_freq2_v, 0.001);
  Kokkos::deep_copy(rho_zz_v, 1.2);
  Kokkos::deep_copy(deform_c2, 1.0);
  Kokkos::deep_copy(deform_s2, 0.5);
  Kokkos::deep_copy(deform_cs, 0.3);
  Kokkos::deep_copy(deform_c, 0.8);
  Kokkos::deep_copy(deform_s, 0.6);

  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c)
      scalars_v(idx_tke - 1, k, c) = 2.0;

  for (int c = 0; c < nc; ++c)
    for (int k = 0; k <= nk; ++k)
      zgrid_v(k, c) = 500.0 * k;

  // Initialize tendency to sentinel value
  Kokkos::deep_copy(tend_scalars_v, 77.0);

  mpas::dycore::EddyViscosityParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_PROGNOSTIC_15_ORDER;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.maxEdges = SmallMesh::maxEdges;
  params.c_s = 0.18;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;
  params.config_visc4_2dsmag = 0.05;
  params.dynamics_substep = 2;  // NOT substep 1
  params.index_tke = idx_tke;
  params.num_scalars = ns;

  ciview1d nEdgesOnCell_c = mesh.nEdgesOnCell_v;
  ciview2d edgesOnCell_c = mesh.edgesOnCell_v;
  ciview2d cellsOnEdge_c = mesh.cellsOnEdge_v;
  cview2d u_c = u_v, v_c = v_v, uCell_c = uCell_v;
  cview2d vCell_c = vCell_v, w_c = w_v;
  cview2d bv_c = bv_freq2_v, zgrid_c = zgrid_v, rho_c = rho_zz_v;
  cview2d c2_c = deform_c2, s2_c = deform_s2;
  cview2d cs_c = deform_cs, dc_c = deform_c, ds_c = deform_s;

  mpas::dycore::les_models<ES>(
      eddy_visc_horz, eddy_visc_vert, prandtl_3d_inv,
      u_c, v_c, uCell_c, vCell_c, w_c, bv_c, zgrid_c,
      rho_c, scalars_v, tend_scalars_v,
      c2_c, s2_c, cs_c, dc_c, ds_c,
      nEdgesOnCell_c, edgesOnCell_c, cellsOnEdge_c, params);

  // On substep != 1, TKE tendency should NOT be overwritten
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c)
      EXPECT_DOUBLE_EQ(tend_scalars_v(idx_tke - 1, k, c), 77.0)
          << "TKE tendency should not be modified on substep 2";
}

// ─── Test: Vertical mixing (Req 8.8, 8.9) ────────────────────────────────────

using view1d = Kokkos::View<Scalar*, Kokkos::LayoutLeft, MS>;
using cview1d = Kokkos::View<const Scalar*, Kokkos::LayoutLeft, MS>;

TEST(DissipationModule, VerticalMixingU_FullState_UniformField_ZeroTendency) {
  // Uniform u field → second-order finite difference is zero → no tendency
  constexpr int nVertLevels = 6;
  constexpr int nEdges = 3;
  constexpr int nCells = 2;

  view2d tend_u("tend_u", nVertLevels, nEdges);
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d rho_edge_v("rho_edge", nVertLevels, nEdges);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);
  view1d u_init_v("u_init", nVertLevels);
  view1d v_init_v("v_init", nVertLevels);
  view1d angleEdge_v("angleEdge", nEdges);

  Kokkos::deep_copy(u_v, 10.0);  // uniform
  Kokkos::deep_copy(rho_edge_v, 1.2);

  for (int c = 0; c < nCells; ++c)
    for (int k = 0; k <= nVertLevels; ++k)
      zgrid_v(k, c) = 500.0 * k;

  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = 1;  // 1-based
    cellsOnEdge_v(1, e) = 2;
  }

  mpas::dycore::VerticalMixingParams params;
  params.v_mom_eddy_visc2 = 100.0;
  params.config_mix_full = true;
  params.nVertLevels = nVertLevels;
  params.nEdges = nEdges;

  cview2d u_c = u_v, v_c = v_v, rho_c = rho_edge_v, zgrid_c = zgrid_v;
  Kokkos::View<const int**, Kokkos::LayoutLeft, MS> cellsOnEdge_c = cellsOnEdge_v;
  cview1d u_init_c = u_init_v, v_init_c = v_init_v, angle_c = angleEdge_v;

  mpas::dycore::apply_vertical_mixing_u<ES>(
      tend_u, u_c, v_c, rho_c, zgrid_c, cellsOnEdge_c,
      u_init_c, v_init_c, angle_c, params);

  // Uniform field: d2u/dz2 = 0 → tendency should be 0
  for (int k = 1; k < nVertLevels - 1; ++k)
    for (int e = 0; e < nEdges; ++e)
      EXPECT_NEAR(tend_u(k, e), 0.0, 1.0e-12)
          << "tend_u(" << k << "," << e << ") should be 0 for uniform u";
}

TEST(DissipationModule, VerticalMixingU_FullState_QuadraticProfile) {
  // Quadratic u profile → constant second derivative → nonzero tendency
  constexpr int nVertLevels = 6;
  constexpr int nEdges = 2;
  constexpr int nCells = 2;

  view2d tend_u("tend_u", nVertLevels, nEdges);
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d rho_edge_v("rho_edge", nVertLevels, nEdges);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);
  view1d u_init_v("u_init", nVertLevels);
  view1d v_init_v("v_init", nVertLevels);
  view1d angleEdge_v("angleEdge", nEdges);

  // Quadratic profile: u(k) = k^2, dz=500m uniform
  for (int k = 0; k < nVertLevels; ++k)
    for (int e = 0; e < nEdges; ++e)
      u_v(k, e) = Scalar(k) * Scalar(k);

  Kokkos::deep_copy(rho_edge_v, 1.0);

  for (int c = 0; c < nCells; ++c)
    for (int k = 0; k <= nVertLevels; ++k)
      zgrid_v(k, c) = 500.0 * k;

  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = 1;
    cellsOnEdge_v(1, e) = 2;
  }

  mpas::dycore::VerticalMixingParams params;
  params.v_mom_eddy_visc2 = 1.0;
  params.config_mix_full = true;
  params.nVertLevels = nVertLevels;
  params.nEdges = nEdges;

  cview2d u_c = u_v, v_c = v_v, rho_c = rho_edge_v, zgrid_c = zgrid_v;
  Kokkos::View<const int**, Kokkos::LayoutLeft, MS> cellsOnEdge_c = cellsOnEdge_v;
  cview1d u_init_c = u_init_v, v_init_c = v_init_v, angle_c = angleEdge_v;

  mpas::dycore::apply_vertical_mixing_u<ES>(
      tend_u, u_c, v_c, rho_c, zgrid_c, cellsOnEdge_c,
      u_init_c, v_init_c, angle_c, params);

  // For uniform dz=500, z_centers at 250, 750, 1250, ...
  // The second derivative of k^2 in index space is 2
  // In physical space with uniform dz: d2u/dz2 = 2/(dz^2) where dz=500m between centers
  // But the stencil is in physical height space using zgrid interfaces.
  // For k=1..4 (interior): should be nonzero and positive (upward curvature)
  for (int k = 1; k < nVertLevels - 1; ++k)
    for (int e = 0; e < nEdges; ++e)
      EXPECT_GT(tend_u(k, e), 0.0)
          << "tend_u(" << k << "," << e << ") should be > 0 for u=k^2";
}

TEST(DissipationModule, VerticalMixingW_UniformField_ZeroTendency) {
  constexpr int nVertLevels = 5;
  constexpr int nCells = 2;

  view2d tend_w("tend_w", nVertLevels + 1, nCells);
  view2d w_v("w", nVertLevels + 1, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d rdzu_v("rdzu", nVertLevels);

  Kokkos::deep_copy(w_v, 5.0);  // uniform
  Kokkos::deep_copy(rho_zz_v, 1.0);

  const Scalar dz = 500.0;
  for (int k = 0; k < nVertLevels; ++k) {
    rdzw_v(k) = 1.0 / dz;
    rdzu_v(k) = 1.0 / dz;
  }

  mpas::dycore::VerticalMixingParams params;
  params.v_mom_eddy_visc2 = 50.0;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;

  cview2d w_c = w_v, rho_c = rho_zz_v;
  cview1d rdzw_c = rdzw_v, rdzu_c = rdzu_v;

  mpas::dycore::apply_vertical_mixing_w<ES>(
      tend_w, w_c, rho_c, rdzw_c, rdzu_c, params);

  for (int k = 1; k < nVertLevels; ++k)
    for (int c = 0; c < nCells; ++c)
      EXPECT_NEAR(tend_w(k, c), 0.0, 1.0e-12)
          << "tend_w(" << k << "," << c << ") should be 0 for uniform w";
}

TEST(DissipationModule, VerticalMixingTheta_FullState_LinearProfile_ZeroTendency) {
  // Linear theta profile: d2theta/dz2 = 0 → zero tendency
  constexpr int nVertLevels = 6;
  constexpr int nCells = 2;

  view2d tend_theta("tend_theta", nVertLevels, nCells);
  view2d theta_m_v("theta_m", nVertLevels, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  view1d t_init_v("t_init", nVertLevels);

  // Linear theta profile
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k)
      theta_m_v(k, c) = 300.0 + 5.0 * k;
    for (int k = 0; k <= nVertLevels; ++k)
      zgrid_v(k, c) = 500.0 * k;
  }
  Kokkos::deep_copy(rho_zz_v, 1.0);

  mpas::dycore::VerticalMixingParams params;
  params.v_theta_eddy_visc2 = 100.0;
  params.prandtl_inv = 1.0;
  params.config_mix_full = true;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;

  cview2d theta_c = theta_m_v, rho_c = rho_zz_v, zgrid_c = zgrid_v;
  cview1d t_init_c = t_init_v;

  mpas::dycore::apply_vertical_mixing_theta<ES>(
      tend_theta, theta_c, rho_c, zgrid_c, t_init_c, params);

  // Linear profile → zero second derivative → zero tendency
  for (int k = 1; k < nVertLevels - 1; ++k)
    for (int c = 0; c < nCells; ++c)
      EXPECT_NEAR(tend_theta(k, c), 0.0, 1.0e-10)
          << "tend_theta(" << k << "," << c << ") should be 0 for linear theta";
}

TEST(DissipationModule, VerticalMixingTheta_PerturbationState) {
  // When config_mix_full is false, should mix on perturbation from t_init
  constexpr int nVertLevels = 6;
  constexpr int nCells = 2;

  view2d tend_theta("tend_theta", nVertLevels, nCells);
  view2d theta_m_v("theta_m", nVertLevels, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  view1d t_init_v("t_init", nVertLevels);

  // theta_m = t_init + perturbation; if perturbation is linear → zero tendency
  for (int k = 0; k < nVertLevels; ++k) {
    t_init_v(k) = 300.0 + 5.0 * k;
  }
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k)
      theta_m_v(k, c) = t_init_v(k) + 2.0 * k;  // linear perturbation
    for (int k = 0; k <= nVertLevels; ++k)
      zgrid_v(k, c) = 500.0 * k;
  }
  Kokkos::deep_copy(rho_zz_v, 1.0);

  mpas::dycore::VerticalMixingParams params;
  params.v_theta_eddy_visc2 = 100.0;
  params.prandtl_inv = 1.0;
  params.config_mix_full = false;  // perturbation mode
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;

  cview2d theta_c = theta_m_v, rho_c = rho_zz_v, zgrid_c = zgrid_v;
  cview1d t_init_c = t_init_v;

  mpas::dycore::apply_vertical_mixing_theta<ES>(
      tend_theta, theta_c, rho_c, zgrid_c, t_init_c, params);

  // Linear perturbation → zero second derivative → zero tendency
  for (int k = 1; k < nVertLevels - 1; ++k)
    for (int c = 0; c < nCells; ++c)
      EXPECT_NEAR(tend_theta(k, c), 0.0, 1.0e-10)
          << "tend_theta(" << k << "," << c << ") should be 0 for linear perturbation";
}

TEST(DissipationModule, VerticalMixing_DisabledWhenViscZero) {
  // When v_mom_eddy_visc2 = 0, tendency should remain unmodified
  constexpr int nVertLevels = 5;
  constexpr int nEdges = 2;
  constexpr int nCells = 2;

  view2d tend_u("tend_u", nVertLevels, nEdges);
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d rho_edge_v("rho_edge", nVertLevels, nEdges);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);
  view1d u_init_v("u_init", nVertLevels);
  view1d v_init_v("v_init", nVertLevels);
  view1d angleEdge_v("angleEdge", nEdges);

  Kokkos::deep_copy(tend_u, 42.0);

  for (int k = 0; k < nVertLevels; ++k)
    for (int e = 0; e < nEdges; ++e)
      u_v(k, e) = Scalar(k) * Scalar(k);

  for (int c = 0; c < nCells; ++c)
    for (int k = 0; k <= nVertLevels; ++k)
      zgrid_v(k, c) = 500.0 * k;

  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = 1;
    cellsOnEdge_v(1, e) = 2;
  }

  mpas::dycore::VerticalMixingParams params;
  params.v_mom_eddy_visc2 = 0.0;  // disabled!
  params.config_mix_full = true;
  params.nVertLevels = nVertLevels;
  params.nEdges = nEdges;

  cview2d u_c = u_v, v_c = v_v, rho_c = rho_edge_v, zgrid_c = zgrid_v;
  Kokkos::View<const int**, Kokkos::LayoutLeft, MS> cellsOnEdge_c = cellsOnEdge_v;
  cview1d u_init_c = u_init_v, v_init_c = v_init_v, angle_c = angleEdge_v;

  mpas::dycore::apply_vertical_mixing_u<ES>(
      tend_u, u_c, v_c, rho_c, zgrid_c, cellsOnEdge_c,
      u_init_c, v_init_c, angle_c, params);

  for (int k = 0; k < nVertLevels; ++k)
    for (int e = 0; e < nEdges; ++e)
      EXPECT_DOUBLE_EQ(tend_u(k, e), 42.0)
          << "tend_u should remain unmodified when v_mom_eddy_visc2=0";
}

TEST(DissipationModule, LesVerticalMixingU_SpecifiedSurfaceBC) {
  // Test that LES surface BC produces non-zero bottom flux for u
  constexpr int nVertLevels = 4;
  constexpr int nEdges = 2;
  constexpr int nCells = 2;

  view2d tend_u("tend_u", nVertLevels, nEdges);
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d rho_edge_v("rho_edge", nVertLevels, nEdges);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view2d eddy_visc_vert_v("ev_v", nVertLevels, nCells);
  view2d zz_v("zz", nVertLevels, nCells);
  view1d rdzu_v("rdzu", nVertLevels);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);
  view1d ustm_v("ustm", nCells);

  // Non-zero velocity at bottom
  for (int k = 0; k < nVertLevels; ++k)
    for (int e = 0; e < nEdges; ++e) {
      u_v(k, e) = 10.0;
      v_v(k, e) = 5.0;
    }

  Kokkos::deep_copy(rho_edge_v, 1.2);
  Kokkos::deep_copy(rho_zz_v, 1.2);
  Kokkos::deep_copy(eddy_visc_vert_v, 10.0);
  Kokkos::deep_copy(zz_v, 1.0);

  const Scalar dz = 500.0;
  for (int k = 0; k < nVertLevels; ++k) {
    rdzu_v(k) = 1.0 / dz;
    rdzw_v(k) = 1.0 / dz;
    fzm_v(k) = 0.5;
    fzp_v(k) = 0.5;
  }

  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = 1;
    cellsOnEdge_v(1, e) = 2;
  }

  mpas::dycore::VerticalMixingParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_3D_SMAGORINSKY;
  params.les_surface_opt = mpas::dycore::LES_SURFACE_SPECIFIED;
  params.config_surface_drag_coefficient = 0.01;
  params.nVertLevels = nVertLevels;
  params.nEdges = nEdges;
  params.nCells = nCells;

  cview2d u_c = u_v, v_c = v_v, rho_e_c = rho_edge_v, rho_c = rho_zz_v;
  cview2d ev_c = eddy_visc_vert_v, zz_c = zz_v;
  cview1d rdzu_c = rdzu_v, rdzw_c = rdzw_v, fzm_c = fzm_v, fzp_c = fzp_v;
  Kokkos::View<const int**, Kokkos::LayoutLeft, MS> cellsOnEdge_c = cellsOnEdge_v;
  cview1d ustm_c = ustm_v;

  mpas::dycore::apply_les_vertical_mixing_u<ES>(
      tend_u, u_c, v_c, rho_e_c, rho_c, ev_c, zz_c,
      rdzu_c, rdzw_c, fzm_c, fzp_c, cellsOnEdge_c, ustm_c, params);

  // With specified surface BC and non-zero velocity, tendency at bottom
  // should be non-zero
  bool has_nonzero = false;
  for (int e = 0; e < nEdges; ++e) {
    if (tend_u(0, e) != 0.0) has_nonzero = true;
    EXPECT_FALSE(std::isnan(tend_u(0, e)));
  }
  EXPECT_TRUE(has_nonzero) << "Surface BC should produce non-zero tendency at bottom";
}

TEST(DissipationModule, LesModels_3DSmagorinsky_ZeroDeformation) {
  // With zero velocities, 3-D Smagorinsky should produce zero viscosity
  SmallMesh mesh;
  constexpr int nk = SmallMesh::nVertLevels;
  constexpr int nc = SmallMesh::nCells;
  constexpr int ne = SmallMesh::nEdges;
  constexpr int ns = SmallMesh::num_scalars;

  view2d eddy_visc_horz("ev_h", nk, nc);
  view2d eddy_visc_vert("ev_v", nk, nc);
  view2d prandtl_3d_inv("pr3d", nk, nc);
  view2d u_v("u", nk, ne);
  view2d v_v("v", nk, ne);
  view2d uCell_v("uCell", nk, nc);
  view2d vCell_v("vCell", nk, nc);
  view2d w_v("w", nk + 1, nc);
  view2d bv_freq2_v("bv_freq2", nk, nc);
  view2d zgrid_v("zgrid", nk + 1, nc);
  view2d rho_zz_v("rho_zz", nk, nc);
  view3d scalars_v("scalars", ns, nk, nc);
  view3d tend_scalars_v("tend_s", ns, nk, nc);
  view2d deform_c2("c2", SmallMesh::maxEdges, nc);
  view2d deform_s2("s2", SmallMesh::maxEdges, nc);
  view2d deform_cs("cs", SmallMesh::maxEdges, nc);
  view2d deform_c("c", SmallMesh::maxEdges, nc);
  view2d deform_s("s", SmallMesh::maxEdges, nc);

  // All velocities zero
  Kokkos::deep_copy(bv_freq2_v, 0.0001);
  Kokkos::deep_copy(deform_c2, 1.0);
  Kokkos::deep_copy(deform_s2, 0.5);
  Kokkos::deep_copy(deform_cs, 0.3);
  Kokkos::deep_copy(deform_c, 0.8);
  Kokkos::deep_copy(deform_s, 0.6);

  for (int c = 0; c < nc; ++c)
    for (int k = 0; k <= nk; ++k)
      zgrid_v(k, c) = 500.0 * k;

  mpas::dycore::EddyViscosityParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_3D_SMAGORINSKY;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.maxEdges = SmallMesh::maxEdges;
  params.c_s = 0.18;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;
  params.config_visc4_2dsmag = 0.05;
  params.dynamics_substep = 1;
  params.index_tke = 1;
  params.num_scalars = ns;

  ciview1d nEdgesOnCell_c = mesh.nEdgesOnCell_v;
  ciview2d edgesOnCell_c = mesh.edgesOnCell_v;
  ciview2d cellsOnEdge_c = mesh.cellsOnEdge_v;
  cview2d u_c = u_v, v_c = v_v, uCell_c = uCell_v;
  cview2d vCell_c = vCell_v, w_c = w_v;
  cview2d bv_c = bv_freq2_v, zgrid_c = zgrid_v, rho_c = rho_zz_v;
  cview2d c2_c = deform_c2, s2_c = deform_s2;
  cview2d cs_c = deform_cs, dc_c = deform_c, ds_c = deform_s;

  mpas::dycore::les_models<ES>(
      eddy_visc_horz, eddy_visc_vert, prandtl_3d_inv,
      u_c, v_c, uCell_c, vCell_c, w_c, bv_c, zgrid_c,
      rho_c, scalars_v, tend_scalars_v,
      c2_c, s2_c, cs_c, dc_c, ds_c,
      nEdgesOnCell_c, edgesOnCell_c, cellsOnEdge_c, params);

  // Zero deformation with positive bv_freq2 → def2 - pr_inv*bv < 0
  // → max(0, ...) = 0 → viscosity = 0
  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      EXPECT_DOUBLE_EQ(eddy_visc_horz(k, c), 0.0);
      EXPECT_DOUBLE_EQ(eddy_visc_vert(k, c), 0.0);
    }
  }
}

// ─── Test: LES vertical mixing w (Req 8.8, 8.9) ──────────────────────────────

TEST(DissipationModule, LesVerticalMixingW_UniformField_ZeroTendency) {
  // Uniform w field → no vertical gradient → zero tendency
  constexpr int nVertLevels = 5;
  constexpr int nCells = 2;

  view2d tend_w("tend_w", nVertLevels + 1, nCells);
  view2d w_v("w", nVertLevels + 1, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view2d eddy_visc_vert_v("ev_v", nVertLevels, nCells);
  view2d zz_v("zz", nVertLevels, nCells);
  view2d divergence_v("div", nVertLevels, nCells);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d rdzu_v("rdzu", nVertLevels);

  Kokkos::deep_copy(w_v, 5.0);  // uniform
  Kokkos::deep_copy(rho_zz_v, 1.2);
  Kokkos::deep_copy(eddy_visc_vert_v, 10.0);
  Kokkos::deep_copy(zz_v, 1.0);
  Kokkos::deep_copy(divergence_v, 0.0);

  const Scalar dz = 500.0;
  for (int k = 0; k < nVertLevels; ++k) {
    rdzw_v(k) = 1.0 / dz;
    rdzu_v(k) = 1.0 / dz;
  }

  mpas::dycore::VerticalMixingParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;

  cview2d w_c = w_v, rho_c = rho_zz_v, ev_c = eddy_visc_vert_v;
  cview2d zz_c = zz_v, div_c = divergence_v;
  cview1d rdzw_c = rdzw_v, rdzu_c = rdzu_v;

  mpas::dycore::apply_les_vertical_mixing_w<ES>(
      tend_w, w_c, rho_c, ev_c, zz_c, div_c, rdzw_c, rdzu_c, params);

  // Uniform w: (w(k+1)-w(k)) = 0, divergence=0 → flux = 0 → tendency = 0
  for (int k = 1; k < nVertLevels; ++k)
    for (int c = 0; c < nCells; ++c)
      EXPECT_NEAR(tend_w(k, c), 0.0, 1.0e-12)
          << "Diverging field: tend_w(" << k << "," << c
          << ") expected 0 for uniform w";
}

// ─── Test: LES vertical mixing theta/scalars with varying surface BC (Req 8.9) ─

TEST(DissipationModule, LesVerticalMixingThetaScalars_VaryingSurfaceBC) {
  // Test that LES_SURFACE_VARYING uses hfx/qfx per-cell fields
  constexpr int nVertLevels = 4;
  constexpr int nCells = 2;
  constexpr int num_scalars = 2;  // qv + one other

  view2d tend_theta("tend_theta", nVertLevels, nCells);
  view3d tend_scalars_v("tend_s", num_scalars, nVertLevels, nCells);
  view2d theta_m_v("theta_m", nVertLevels, nCells);
  view3d scalars_v("scalars", num_scalars, nVertLevels, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view2d eddy_visc_vert_v("ev_v", nVertLevels, nCells);
  view2d zz_v("zz", nVertLevels, nCells);
  view2d prandtl_3d_inv_v("pr3d", nVertLevels, nCells);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d rdzu_v("rdzu", nVertLevels);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view1d hfx_v("hfx", nCells);
  view1d qfx_v("qfx", nCells);

  // Uniform theta_m → zero vertical gradient → tendency from surface BC only
  Kokkos::deep_copy(theta_m_v, 300.0);
  Kokkos::deep_copy(rho_zz_v, 1.2);
  Kokkos::deep_copy(eddy_visc_vert_v, 10.0);
  Kokkos::deep_copy(zz_v, 1.0);
  Kokkos::deep_copy(prandtl_3d_inv_v, 1.0);

  for (int s = 0; s < num_scalars; ++s)
    for (int k = 0; k < nVertLevels; ++k)
      for (int c = 0; c < nCells; ++c)
        scalars_v(s, k, c) = 0.01;

  const Scalar dz = 500.0;
  for (int k = 0; k < nVertLevels; ++k) {
    rdzw_v(k) = 1.0 / dz;
    rdzu_v(k) = 1.0 / dz;
    fzm_v(k) = 0.5;
    fzp_v(k) = 0.5;
  }

  // Non-zero spatially-varying heat/moisture fluxes
  hfx_v(0) = 100.0;  // W/m2 → will be divided by rho*cp
  hfx_v(1) = 200.0;
  qfx_v(0) = 0.001;  // kg/m2/s → will be divided by rho
  qfx_v(1) = 0.002;

  mpas::dycore::VerticalMixingParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_3D_SMAGORINSKY;
  params.les_surface_opt = mpas::dycore::LES_SURFACE_VARYING;
  params.config_mix_scalars = true;
  params.index_qv = 1;  // 1-based
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.num_scalars = num_scalars;
  params.prandtl_inv = 1.0;

  using cview3d_t = Kokkos::View<const Scalar***, Kokkos::LayoutLeft, MS>;

  mpas::dycore::apply_les_vertical_mixing_theta_scalars<ES>(
      tend_theta, tend_scalars_v,
      cview2d(theta_m_v), cview3d_t(scalars_v),
      cview2d(rho_zz_v), cview2d(eddy_visc_vert_v),
      cview2d(zz_v), cview2d(prandtl_3d_inv_v),
      cview1d(rdzw_v), cview1d(rdzu_v),
      cview1d(fzm_v), cview1d(fzp_v),
      cview1d(hfx_v), cview1d(qfx_v), params);

  // With non-zero hfx, the bottom-level theta tendency should be non-zero
  // due to surface heat flux BC (Req 8.9 "varying" path)
  bool has_nonzero_theta = false;
  for (int c = 0; c < nCells; ++c) {
    if (tend_theta(0, c) != 0.0) has_nonzero_theta = true;
    EXPECT_FALSE(std::isnan(tend_theta(0, c)))
        << "Diverging field: tend_theta(0," << c << ") is NaN";
  }
  EXPECT_TRUE(has_nonzero_theta)
      << "Diverging field: tend_theta - varying surface BC should produce "
         "non-zero bottom tendency";

  // Different hfx at each cell should produce different tendencies
  if (nCells >= 2) {
    // The two cells have different hfx, so their tendencies should differ
    EXPECT_NE(tend_theta(0, 0), tend_theta(0, 1))
        << "Diverging field: tend_theta - varying surface BC should "
           "produce per-cell-varying tendencies";
  }
}

// ─── Test: Hyperdiffusion with non-uniform field produces diffusive tendency ──
//          (Req 8.7, 14.2, 14.9)

TEST(DissipationModule, Hyperdiffusion_NonUniformTheta_ProducesTendency) {
  // With varying theta across cells, the del4 filter should produce non-zero
  // tendency that is finite and smooth
  HyperdiffMesh mesh;
  constexpr int nk = HyperdiffMesh::nVertLevels;
  constexpr int nc = HyperdiffMesh::nCells;
  constexpr int ne = HyperdiffMesh::nEdges;
  constexpr int nv = HyperdiffMesh::nVertices;
  constexpr int ns = HyperdiffMesh::num_scalars;

  view2d tend_u("tend_u", nk, ne);
  view2d tend_w("tend_w", nk + 1, nc);
  view2d tend_theta("tend_theta", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> tend_scalars("tend_s", ns, nk, nc);
  view2d u_v("u", nk, ne);
  view2d w_v("w", nk + 1, nc);
  view2d theta_m_v("theta_m", nk, nc);
  Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS> scalars_v("scalars", ns, nk, nc);
  view2d div_v("div", nk, nc);
  view2d vort_v("vort", nk, nv);
  view2d rho_edge_v("rho_edge", nk, ne);

  // Non-uniform theta: large gradient across cells
  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      theta_m_v(k, c) = 300.0 + 50.0 * c;  // strong cell-to-cell gradient
    }
  }
  Kokkos::deep_copy(rho_edge_v, 1.0);

  mpas::dycore::HyperdiffusionParams params;
  params.h_mom_eddy_visc4 = 0.0;       // only theta active
  params.h_theta_eddy_visc4 = 1.0e10;  // large to produce measurable tendency
  params.config_del4u_div_factor = 1.0;
  params.prandtl_inv = 1.0;
  params.config_mix_scalars = false;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.nEdges = ne;
  params.nVertices = nv;
  params.vertexDegree = HyperdiffMesh::vertexDegree;
  params.num_scalars = ns;

  using cview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;
  using cview3d_t = Kokkos::View<const Scalar***, Kokkos::LayoutLeft, MS>;
  using ciview1d_t = Kokkos::View<const int*, Kokkos::LayoutLeft, MS>;
  using ciview2d_t = Kokkos::View<const int**, Kokkos::LayoutLeft, MS>;
  using csview1d_t = Kokkos::View<const Scalar*, Kokkos::LayoutLeft, MS>;
  using csview2d_t = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;

  mpas::dycore::apply_hyperdiffusion<ES>(
      tend_u, tend_w, tend_theta, tend_scalars,
      cview2d_t(u_v), cview2d_t(w_v), cview2d_t(theta_m_v), cview3d_t(scalars_v),
      cview2d_t(div_v), cview2d_t(vort_v), cview2d_t(rho_edge_v),
      ciview2d_t(mesh.cellsOnEdge_v), ciview2d_t(mesh.verticesOnEdge_v),
      ciview2d_t(mesh.edgesOnCell_v), ciview2d_t(mesh.edgesOnVertex_v),
      ciview1d_t(mesh.nEdgesOnCell_v),
      csview2d_t(mesh.edgesOnCell_sign_v), csview2d_t(mesh.edgesOnVertex_sign_v),
      csview1d_t(mesh.invAreaCell_v), csview1d_t(mesh.invAreaTriangle_v),
      csview1d_t(mesh.invDcEdge_v), csview1d_t(mesh.invDvEdge_v),
      csview1d_t(mesh.dvEdge_v), csview1d_t(mesh.dcEdge_v),
      csview1d_t(mesh.meshScalingDel4_v),
      params);

  // With non-uniform theta, the 4th-order filter should produce non-zero
  // tendency (confirms Req 8.7 filtering is active for theta)
  bool has_nonzero = false;
  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      EXPECT_FALSE(std::isnan(tend_theta(k, c)))
          << "Diverging field: tend_theta(" << k << "," << c << ") is NaN";
      EXPECT_FALSE(std::isinf(tend_theta(k, c)))
          << "Diverging field: tend_theta(" << k << "," << c << ") is Inf";
      if (tend_theta(k, c) != 0.0) has_nonzero = true;
    }
  }
  EXPECT_TRUE(has_nonzero)
      << "Diverging field: tend_theta - hyperdiffusion should produce "
         "non-zero tendency for non-uniform theta";

  // u tendency should remain zero since h_mom_eddy_visc4 = 0
  for (int k = 0; k < nk; ++k)
    for (int e = 0; e < ne; ++e)
      EXPECT_DOUBLE_EQ(tend_u(k, e), 0.0)
          << "Diverging field: tend_u(" << k << "," << e
          << ") should be 0 when h_mom_eddy_visc4=0";
}

// ─── Test: Parity_Tolerance verification pattern (Req 14.2, 14.9) ────────────
// This test demonstrates the pattern for identifying diverging fields
// by name when results exceed tolerance.

TEST(DissipationModule, ParityTolerance_EddyViscosityFieldIdentification) {
  // Verify eddy viscosity values are within Parity_Tolerance of expected
  // reference values, identifying specific diverging fields on failure.
  SmallMesh mesh;
  constexpr int nk = SmallMesh::nVertLevels;
  constexpr int nc = SmallMesh::nCells;

  view2d kdiff("kdiff", nk, nc);

  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = nk;
  params.nCells = nc;
  params.config_len_disp = 3000.0;
  params.invDt = 1.0 / 60.0;

  const Scalar fixed_visc = 500.0;
  mpas::dycore::fixed_horizontal_eddy_viscosity<ES>(kdiff, fixed_visc, params);

  // Reference value: min(fixed_visc, stability_limit)
  const Scalar stability_limit = 0.01 * 3000.0 * 3000.0 / 60.0;
  const Scalar reference_value = std::min(fixed_visc, stability_limit);

  // Parity_Tolerance: 1e-12 for double precision
  constexpr Scalar parity_tol = 1.0e-12;

  for (int k = 0; k < nk; ++k) {
    for (int c = 0; c < nc; ++c) {
      const Scalar rel_err = (reference_value != 0.0)
          ? std::abs(kdiff(k, c) - reference_value) / std::abs(reference_value)
          : std::abs(kdiff(k, c));
      EXPECT_LE(rel_err, parity_tol)
          << "Diverging field: kdiff(" << k << "," << c
          << ") value=" << kdiff(k, c)
          << " reference=" << reference_value
          << " relative_error=" << rel_err
          << " exceeds Parity_Tolerance=" << parity_tol;
    }
  }
}

TEST(DissipationModule, ParityTolerance_VerticalMixing_FieldIdentification) {
  // Verify vertical mixing tendency within Parity_Tolerance of expected
  // analytical reference. A linear profile should produce zero tendency;
  // any deviation identifies a diverging field (Req 14.9).
  constexpr int nVertLevels = 6;
  constexpr int nEdges = 2;
  constexpr int nCells = 2;
  constexpr Scalar parity_tol = 1.0e-12;

  view2d tend_u("tend_u", nVertLevels, nEdges);
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d rho_edge_v("rho_edge", nVertLevels, nEdges);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);
  view1d u_init_v("u_init", nVertLevels);
  view1d v_init_v("v_init", nVertLevels);
  view1d angleEdge_v("angleEdge", nEdges);

  // Linear u profile: u(k) = 10 + 5*k → d2u/dz2 = 0
  for (int k = 0; k < nVertLevels; ++k)
    for (int e = 0; e < nEdges; ++e)
      u_v(k, e) = 10.0 + 5.0 * k;

  Kokkos::deep_copy(rho_edge_v, 1.0);

  for (int c = 0; c < nCells; ++c)
    for (int k = 0; k <= nVertLevels; ++k)
      zgrid_v(k, c) = 500.0 * k;

  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = 1;
    cellsOnEdge_v(1, e) = 2;
  }

  mpas::dycore::VerticalMixingParams params;
  params.v_mom_eddy_visc2 = 100.0;
  params.config_mix_full = true;
  params.nVertLevels = nVertLevels;
  params.nEdges = nEdges;

  cview2d u_c = u_v, v_c = v_v, rho_c = rho_edge_v, zgrid_c = zgrid_v;
  Kokkos::View<const int**, Kokkos::LayoutLeft, MS> cellsOnEdge_c = cellsOnEdge_v;
  cview1d u_init_c = u_init_v, v_init_c = v_init_v, angle_c = angleEdge_v;

  mpas::dycore::apply_vertical_mixing_u<ES>(
      tend_u, u_c, v_c, rho_c, zgrid_c, cellsOnEdge_c,
      u_init_c, v_init_c, angle_c, params);

  // Reference: zero for linear profile at interior levels
  for (int k = 1; k < nVertLevels - 1; ++k)
    for (int e = 0; e < nEdges; ++e)
      EXPECT_LE(std::abs(tend_u(k, e)), parity_tol)
          << "Diverging field: tend_u(" << k << "," << e
          << ") value=" << tend_u(k, e)
          << " expected 0 for linear u profile"
          << " exceeds Parity_Tolerance=" << parity_tol;
}

// ─── Test: LES option-string mapping edge cases (Req 8.10, 8.11) ─────────────

TEST(DissipationModule, LesModelFromString_EmptyReturnsInvalid) {
  using DM = mpas::dycore::Dissipation_Module;
  EXPECT_EQ(DM::les_model_opt_from_string(""), mpas::dycore::LES_INVALID_OPT);
}

TEST(DissipationModule, LesSurfaceFromString_EmptyReturnsInvalid) {
  using DM = mpas::dycore::Dissipation_Module;
  EXPECT_EQ(DM::les_surface_opt_from_string(""), mpas::dycore::LES_INVALID_OPT);
}

TEST(DissipationModule, LesModelFromString_CaseSensitive) {
  using DM = mpas::dycore::Dissipation_Module;
  // Strings must match exactly; wrong case returns invalid
  EXPECT_EQ(DM::les_model_opt_from_string("None"), mpas::dycore::LES_INVALID_OPT);
  EXPECT_EQ(DM::les_model_opt_from_string("3D_SMAGORINSKY"), mpas::dycore::LES_INVALID_OPT);
}

// ─── Test: Vertical mixing u perturbation mode (Req 8.8) ────────────────────

TEST(DissipationModule, VerticalMixingU_PerturbationMode_LinearPerturbation) {
  // When config_mix_full is false, should mix perturbation from initial 1-D
  // profile. A linear perturbation → d2/dz2 = 0 → zero tendency.
  constexpr int nVertLevels = 6;
  constexpr int nEdges = 2;
  constexpr int nCells = 2;

  view2d tend_u("tend_u", nVertLevels, nEdges);
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d rho_edge_v("rho_edge", nVertLevels, nEdges);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);
  view1d u_init_v("u_init", nVertLevels);
  view1d v_init_v("v_init", nVertLevels);
  view1d angleEdge_v("angleEdge", nEdges);

  // u(k) = u_init(k) + linear_perturbation(k)
  // where perturbation = 3.0 * k → linear → d2/dz2 = 0
  for (int k = 0; k < nVertLevels; ++k) {
    u_init_v(k) = 10.0 + 2.0 * k;  // some base state
    v_init_v(k) = 0.0;
    for (int e = 0; e < nEdges; ++e)
      u_v(k, e) = u_init_v(k) + 3.0 * k;  // linear perturbation
  }

  Kokkos::deep_copy(rho_edge_v, 1.0);

  for (int c = 0; c < nCells; ++c)
    for (int k = 0; k <= nVertLevels; ++k)
      zgrid_v(k, c) = 500.0 * k;

  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = 1;
    cellsOnEdge_v(1, e) = 2;
    angleEdge_v(e) = 0.0;  // cos(0)=1, sin(0)=0 → u_mix = u - u_init
  }

  mpas::dycore::VerticalMixingParams params;
  params.v_mom_eddy_visc2 = 100.0;
  params.config_mix_full = false;  // perturbation mode
  params.nVertLevels = nVertLevels;
  params.nEdges = nEdges;

  cview2d u_c = u_v, v_c = v_v, rho_c = rho_edge_v, zgrid_c = zgrid_v;
  Kokkos::View<const int**, Kokkos::LayoutLeft, MS> cellsOnEdge_c = cellsOnEdge_v;
  cview1d u_init_c = u_init_v, v_init_c = v_init_v, angle_c = angleEdge_v;

  mpas::dycore::apply_vertical_mixing_u<ES>(
      tend_u, u_c, v_c, rho_c, zgrid_c, cellsOnEdge_c,
      u_init_c, v_init_c, angle_c, params);

  // Linear perturbation: d2(pert)/dz2 = 0 → tendency should be zero
  constexpr Scalar parity_tol = 1.0e-10;
  for (int k = 1; k < nVertLevels - 1; ++k)
    for (int e = 0; e < nEdges; ++e)
      EXPECT_LE(std::abs(tend_u(k, e)), parity_tol)
          << "Diverging field: tend_u(" << k << "," << e
          << ") should be 0 for linear perturbation in perturbation mode";
}

// ─── Test: Vertical mixing w with quadratic profile (Req 8.8) ────────────────

TEST(DissipationModule, VerticalMixingW_QuadraticProfile_NonZeroTendency) {
  // Quadratic w profile → constant second derivative → nonzero tendency
  constexpr int nVertLevels = 5;
  constexpr int nCells = 2;

  view2d tend_w("tend_w", nVertLevels + 1, nCells);
  view2d w_v("w", nVertLevels + 1, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d rdzu_v("rdzu", nVertLevels);

  // Quadratic w profile: w(k) = k^2
  for (int c = 0; c < nCells; ++c)
    for (int k = 0; k <= nVertLevels; ++k)
      w_v(k, c) = Scalar(k) * Scalar(k);

  Kokkos::deep_copy(rho_zz_v, 1.0);

  const Scalar dz = 500.0;
  for (int k = 0; k < nVertLevels; ++k) {
    rdzw_v(k) = 1.0 / dz;
    rdzu_v(k) = 1.0 / dz;
  }

  mpas::dycore::VerticalMixingParams params;
  params.v_mom_eddy_visc2 = 1.0;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;

  cview2d w_c = w_v, rho_c = rho_zz_v;
  cview1d rdzw_c = rdzw_v, rdzu_c = rdzu_v;

  mpas::dycore::apply_vertical_mixing_w<ES>(
      tend_w, w_c, rho_c, rdzw_c, rdzu_c, params);

  // For quadratic w(k) = k^2, the stencil:
  // tend_w(k) += v_mom_eddy_visc2 * 0.5*(rho(k)+rho(k-1)) *
  //   ((w(k+1)-w(k))*rdzw(k) - (w(k)-w(k-1))*rdzw(k-1)) * rdzu(k)
  // With uniform dz and rho=1: d2w/dz2 is constant → non-zero tendency
  for (int k = 1; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      EXPECT_NE(tend_w(k, c), 0.0)
          << "Diverging field: tend_w(" << k << "," << c
          << ") should be non-zero for quadratic w";
      EXPECT_FALSE(std::isnan(tend_w(k, c)))
          << "Diverging field: tend_w(" << k << "," << c << ") is NaN";
    }
  }
}

// ─── Test: LES vertical mixing theta/scalars with specified surface BC ───────
//          (Req 8.9)

TEST(DissipationModule, LesVerticalMixingThetaScalars_SpecifiedSurfaceBC) {
  // Test that LES_SURFACE_SPECIFIED uses the configured heat/moisture flux
  constexpr int nVertLevels = 4;
  constexpr int nCells = 2;
  constexpr int num_scalars = 2;

  view2d tend_theta("tend_theta", nVertLevels, nCells);
  view3d tend_scalars_v("tend_s", num_scalars, nVertLevels, nCells);
  view2d theta_m_v("theta_m", nVertLevels, nCells);
  view3d scalars_v("scalars", num_scalars, nVertLevels, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view2d eddy_visc_vert_v("ev_v", nVertLevels, nCells);
  view2d zz_v("zz", nVertLevels, nCells);
  view2d prandtl_3d_inv_v("pr3d", nVertLevels, nCells);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d rdzu_v("rdzu", nVertLevels);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view1d hfx_v("hfx", nCells);
  view1d qfx_v("qfx", nCells);

  // Uniform theta → zero vertical gradient → tendency only from surface BC
  Kokkos::deep_copy(theta_m_v, 300.0);
  Kokkos::deep_copy(rho_zz_v, 1.2);
  Kokkos::deep_copy(eddy_visc_vert_v, 10.0);
  Kokkos::deep_copy(zz_v, 1.0);
  Kokkos::deep_copy(prandtl_3d_inv_v, 1.0);

  for (int s = 0; s < num_scalars; ++s)
    for (int k = 0; k < nVertLevels; ++k)
      for (int c = 0; c < nCells; ++c)
        scalars_v(s, k, c) = 0.01;

  const Scalar dz = 500.0;
  for (int k = 0; k < nVertLevels; ++k) {
    rdzw_v(k) = 1.0 / dz;
    rdzu_v(k) = 1.0 / dz;
    fzm_v(k) = 0.5;
    fzp_v(k) = 0.5;
  }

  mpas::dycore::VerticalMixingParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_3D_SMAGORINSKY;
  params.les_surface_opt = mpas::dycore::LES_SURFACE_SPECIFIED;
  params.config_surface_heat_flux = 0.05;     // specified heat flux [K m/s]
  params.config_surface_moisture_flux = 0.0001; // specified moisture flux
  params.config_mix_scalars = true;
  params.index_qv = 1;  // 1-based
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.num_scalars = num_scalars;
  params.prandtl_inv = 1.0;

  using cview3d_t = Kokkos::View<const Scalar***, Kokkos::LayoutLeft, MS>;

  mpas::dycore::apply_les_vertical_mixing_theta_scalars<ES>(
      tend_theta, tend_scalars_v,
      cview2d(theta_m_v), cview3d_t(scalars_v),
      cview2d(rho_zz_v), cview2d(eddy_visc_vert_v),
      cview2d(zz_v), cview2d(prandtl_3d_inv_v),
      cview1d(rdzw_v), cview1d(rdzu_v),
      cview1d(fzm_v), cview1d(fzp_v),
      cview1d(hfx_v), cview1d(qfx_v), params);

  // With non-zero config_surface_heat_flux, the bottom-level theta tendency
  // should be non-zero due to surface heat flux BC (Req 8.9 "specified" path)
  bool has_nonzero_theta = false;
  for (int c = 0; c < nCells; ++c) {
    if (tend_theta(0, c) != 0.0) has_nonzero_theta = true;
    EXPECT_FALSE(std::isnan(tend_theta(0, c)))
        << "Diverging field: tend_theta(0," << c << ") is NaN";
  }
  EXPECT_TRUE(has_nonzero_theta)
      << "Diverging field: tend_theta - specified surface BC should produce "
         "non-zero bottom tendency";

  // Both cells have the same specified flux → same tendency at bottom
  if (nCells >= 2) {
    EXPECT_DOUBLE_EQ(tend_theta(0, 0), tend_theta(0, 1))
        << "Diverging field: tend_theta - specified (constant) surface flux "
           "should produce uniform tendencies across cells";
  }

  // With config_surface_moisture_flux > 0, qv scalar at bottom should have
  // non-zero tendency (Req 8.9)
  bool has_nonzero_qv_tend = false;
  const int idx_qv = 0;  // 0-based
  for (int c = 0; c < nCells; ++c) {
    if (tend_scalars_v(idx_qv, 0, c) != 0.0) has_nonzero_qv_tend = true;
    EXPECT_FALSE(std::isnan(tend_scalars_v(idx_qv, 0, c)))
        << "Diverging field: tend_scalars[qv](0," << c << ") is NaN";
  }
  EXPECT_TRUE(has_nonzero_qv_tend)
      << "Diverging field: tend_scalars[qv] - specified surface moisture flux "
         "should produce non-zero bottom tendency for qv";
}

// ─── Test: LES vertical mixing u with varying surface BC (Req 8.9) ──────────

TEST(DissipationModule, LesVerticalMixingU_VaryingSurfaceBC) {
  // Test that LES_SURFACE_VARYING uses ustm per-cell fields for u
  constexpr int nVertLevels = 4;
  constexpr int nEdges = 2;
  constexpr int nCells = 2;

  view2d tend_u("tend_u", nVertLevels, nEdges);
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d rho_edge_v("rho_edge", nVertLevels, nEdges);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view2d eddy_visc_vert_v("ev_v", nVertLevels, nCells);
  view2d zz_v("zz", nVertLevels, nCells);
  view1d rdzu_v("rdzu", nVertLevels);
  view1d rdzw_v("rdzw", nVertLevels);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);
  view1d ustm_v("ustm", nCells);

  // Non-zero velocity at bottom
  for (int k = 0; k < nVertLevels; ++k)
    for (int e = 0; e < nEdges; ++e) {
      u_v(k, e) = 10.0;
      v_v(k, e) = 5.0;
    }

  Kokkos::deep_copy(rho_edge_v, 1.2);
  Kokkos::deep_copy(rho_zz_v, 1.2);
  Kokkos::deep_copy(eddy_visc_vert_v, 10.0);
  Kokkos::deep_copy(zz_v, 1.0);

  const Scalar dz = 500.0;
  for (int k = 0; k < nVertLevels; ++k) {
    rdzu_v(k) = 1.0 / dz;
    rdzw_v(k) = 1.0 / dz;
    fzm_v(k) = 0.5;
    fzp_v(k) = 0.5;
  }

  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = 1;
    cellsOnEdge_v(1, e) = 2;
  }

  // Non-zero friction velocity (different per cell for varying check)
  ustm_v(0) = 0.5;
  ustm_v(1) = 1.0;

  mpas::dycore::VerticalMixingParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_3D_SMAGORINSKY;
  params.les_surface_opt = mpas::dycore::LES_SURFACE_VARYING;
  params.nVertLevels = nVertLevels;
  params.nEdges = nEdges;
  params.nCells = nCells;

  cview2d u_c = u_v, v_c = v_v, rho_e_c = rho_edge_v, rho_c = rho_zz_v;
  cview2d ev_c = eddy_visc_vert_v, zz_c = zz_v;
  cview1d rdzu_c = rdzu_v, rdzw_c = rdzw_v, fzm_c = fzm_v, fzp_c = fzp_v;
  Kokkos::View<const int**, Kokkos::LayoutLeft, MS> cellsOnEdge_c = cellsOnEdge_v;
  cview1d ustm_c = ustm_v;

  mpas::dycore::apply_les_vertical_mixing_u<ES>(
      tend_u, u_c, v_c, rho_e_c, rho_c, ev_c, zz_c,
      rdzu_c, rdzw_c, fzm_c, fzp_c, cellsOnEdge_c, ustm_c, params);

  // With varying surface BC and non-zero ustm, tendency at bottom should be
  // non-zero (Req 8.9)
  bool has_nonzero = false;
  for (int e = 0; e < nEdges; ++e) {
    if (tend_u(0, e) != 0.0) has_nonzero = true;
    EXPECT_FALSE(std::isnan(tend_u(0, e)))
        << "Diverging field: tend_u(0," << e << ") is NaN";
  }
  EXPECT_TRUE(has_nonzero)
      << "Diverging field: tend_u - varying surface BC should produce "
         "non-zero tendency at bottom";
}

// ─── Test: Vertical mixing theta quadratic profile (Req 8.8, 14.2, 14.9) ────

TEST(DissipationModule, VerticalMixingTheta_FullState_QuadraticProfile) {
  // Quadratic theta profile → constant second derivative → nonzero tendency
  constexpr int nVertLevels = 6;
  constexpr int nCells = 2;

  view2d tend_theta("tend_theta", nVertLevels, nCells);
  view2d theta_m_v("theta_m", nVertLevels, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  view1d t_init_v("t_init", nVertLevels);

  // Quadratic theta profile: theta(k) = 300 + k^2
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k)
      theta_m_v(k, c) = 300.0 + Scalar(k) * Scalar(k);
    for (int k = 0; k <= nVertLevels; ++k)
      zgrid_v(k, c) = 500.0 * k;
  }
  Kokkos::deep_copy(rho_zz_v, 1.0);

  mpas::dycore::VerticalMixingParams params;
  params.v_theta_eddy_visc2 = 1.0;
  params.prandtl_inv = 1.0;
  params.config_mix_full = true;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;

  cview2d theta_c = theta_m_v, rho_c = rho_zz_v, zgrid_c = zgrid_v;
  cview1d t_init_c = t_init_v;

  mpas::dycore::apply_vertical_mixing_theta<ES>(
      tend_theta, theta_c, rho_c, zgrid_c, t_init_c, params);

  // Quadratic profile → positive second derivative → non-zero tendency
  for (int k = 1; k < nVertLevels - 1; ++k)
    for (int c = 0; c < nCells; ++c) {
      EXPECT_GT(tend_theta(k, c), 0.0)
          << "Diverging field: tend_theta(" << k << "," << c
          << ") should be > 0 for quadratic theta";
      EXPECT_FALSE(std::isnan(tend_theta(k, c)))
          << "Diverging field: tend_theta(" << k << "," << c << ") is NaN";
    }
}

}  // namespace
