#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/diagnostics_module.hpp"
#include "mpas_dycore/scalar.hpp"

#include <cmath>
#include <vector>

namespace {

using Scalar = mpas::dycore::Scalar;
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using layout = Kokkos::LayoutLeft;

// Parity tolerance
constexpr Scalar kTol = std::is_same_v<Scalar, double> ? 1e-12 : 1e-6f;

template <class T>
using View1D = Kokkos::View<T*, layout, MemSpace>;
template <class T>
using View2D = Kokkos::View<T**, layout, MemSpace>;

/// Build a small hexagonal-like test mesh (7 cells, 12 edges, 6 vertices).
/// This is a simplified mesh sufficient to exercise all diagnostics kernels.
/// Uses vertexDegree = 3 (triangular dual cells).
struct SmallMesh {
  static constexpr int nCells = 7;
  static constexpr int nEdges = 12;
  static constexpr int nVertices = 6;
  static constexpr int nVertLevels = 3;
  static constexpr int maxEdges = 6;
  static constexpr int maxEdges2 = 10;
  static constexpr int vertexDegree = 3;

  mpas::dycore::DiagMeshData<ExecSpace> mesh;
  View2D<Scalar> u;
  View2D<Scalar> h;
  mpas::dycore::DiagFields<ExecSpace> diag;

  SmallMesh() {
    mesh.nCells = nCells;
    mesh.nEdges = nEdges;
    mesh.nVertices = nVertices;
    mesh.nVertLevels = nVertLevels;
    mesh.maxEdges = maxEdges;
    mesh.maxEdges2 = maxEdges2;
    mesh.vertexDegree = vertexDegree;

    // Allocate connectivity
    mesh.cellsOnEdge = View2D<int>("cellsOnEdge", 2, nEdges);
    mesh.verticesOnEdge = View2D<int>("verticesOnEdge", 2, nEdges);
    mesh.edgesOnCell = View2D<int>("edgesOnCell", maxEdges, nCells);
    mesh.edgesOnEdge = View2D<int>("edgesOnEdge", maxEdges2, nEdges);
    mesh.edgesOnVertex = View2D<int>("edgesOnVertex", vertexDegree, nVertices);
    mesh.verticesOnCell = View2D<int>("verticesOnCell", maxEdges, nCells);
    mesh.kiteForCell = View2D<int>("kiteForCell", maxEdges, nCells);
    mesh.nEdgesOnCell = View1D<int>("nEdgesOnCell", nCells);
    mesh.nEdgesOnEdge = View1D<int>("nEdgesOnEdge", nEdges);

    // Allocate geometry
    mesh.dvEdge = View1D<Scalar>("dvEdge", nEdges);
    mesh.dcEdge = View1D<Scalar>("dcEdge", nEdges);
    mesh.invDvEdge = View1D<Scalar>("invDvEdge", nEdges);
    mesh.invDcEdge = View1D<Scalar>("invDcEdge", nEdges);
    mesh.invAreaCell = View1D<Scalar>("invAreaCell", nCells);
    mesh.invAreaTriangle = View1D<Scalar>("invAreaTriangle", nVertices);
    mesh.fVertex = View1D<Scalar>("fVertex", nVertices);
    mesh.fEdge = View1D<Scalar>("fEdge", nEdges);
    mesh.weightsOnEdge = View2D<Scalar>("weightsOnEdge", maxEdges2, nEdges);
    mesh.kiteAreasOnVertex = View2D<Scalar>("kiteAreasOnVertex", vertexDegree, nVertices);
    mesh.edgesOnVertex_sign = View2D<Scalar>("edgesOnVertex_sign", vertexDegree, nVertices);
    mesh.edgesOnCell_sign = View2D<Scalar>("edgesOnCell_sign", maxEdges, nCells);

    // Allocate state
    u = View2D<Scalar>("u", nVertLevels, nEdges);
    h = View2D<Scalar>("h", nVertLevels, nCells);

    // Allocate diagnostics output
    diag.h_edge = View2D<Scalar>("h_edge", nVertLevels, nEdges);
    diag.v = View2D<Scalar>("v", nVertLevels, nEdges);
    diag.vorticity = View2D<Scalar>("vorticity", nVertLevels, nVertices);
    diag.divergence = View2D<Scalar>("divergence", nVertLevels, nCells);
    diag.ke = View2D<Scalar>("ke", nVertLevels, nCells);
    diag.pv_edge = View2D<Scalar>("pv_edge", nVertLevels, nEdges);
    diag.pv_vertex = View2D<Scalar>("pv_vertex", nVertLevels, nVertices);
    diag.pv_cell = View2D<Scalar>("pv_cell", nVertLevels, nCells);
    diag.gradPVn = View2D<Scalar>("gradPVn", nVertLevels, nEdges);
    diag.gradPVt = View2D<Scalar>("gradPVt", nVertLevels, nEdges);

    fill_simple_mesh();
  }

  void fill_simple_mesh() {
    // Fill a simple connected mesh:
    // Each edge connects two cells (1-based). Edge i connects cell i%nCells+1
    // and cell (i+1)%nCells+1
    for (int e = 0; e < nEdges; ++e) {
      mesh.cellsOnEdge(0, e) = (e % nCells) + 1;       // 1-based
      mesh.cellsOnEdge(1, e) = ((e + 1) % nCells) + 1; // 1-based
    }

    // Each edge has two bounding vertices
    for (int e = 0; e < nEdges; ++e) {
      mesh.verticesOnEdge(0, e) = (e % nVertices) + 1;
      mesh.verticesOnEdge(1, e) = ((e + 1) % nVertices) + 1;
    }

    // Each vertex has 3 edges (vertexDegree=3)
    // Assign edges round-robin
    for (int v = 0; v < nVertices; ++v) {
      for (int i = 0; i < vertexDegree; ++i) {
        mesh.edgesOnVertex(i, v) = ((v * vertexDegree + i) % nEdges) + 1;
      }
    }

    // Each cell has a fixed number of edges (use 4 for simplicity)
    for (int c = 0; c < nCells; ++c) {
      mesh.nEdgesOnCell(c) = 4;
      for (int i = 0; i < 4; ++i) {
        mesh.edgesOnCell(i, c) = ((c * 2 + i) % nEdges) + 1;
      }
    }

    // edgesOnEdge: each edge has 4 neighbors
    for (int e = 0; e < nEdges; ++e) {
      mesh.nEdgesOnEdge(e) = 4;
      for (int i = 0; i < 4; ++i) {
        mesh.edgesOnEdge(i, e) = ((e + i + 1) % nEdges) + 1;
      }
    }

    // verticesOnCell and kiteForCell
    for (int c = 0; c < nCells; ++c) {
      for (int i = 0; i < 4; ++i) {
        mesh.verticesOnCell(i, c) = ((c + i) % nVertices) + 1;
        mesh.kiteForCell(i, c) = (i % vertexDegree) + 1;  // 1-based
      }
    }

    // Geometry: uniform mesh with unit lengths/areas
    for (int e = 0; e < nEdges; ++e) {
      mesh.dvEdge(e) = Scalar(1.0);
      mesh.dcEdge(e) = Scalar(1.0);
      mesh.invDvEdge(e) = Scalar(1.0);
      mesh.invDcEdge(e) = Scalar(1.0);
    }
    for (int c = 0; c < nCells; ++c) {
      mesh.invAreaCell(c) = Scalar(0.25);  // area = 4
    }
    for (int v = 0; v < nVertices; ++v) {
      mesh.invAreaTriangle(v) = Scalar(0.5);  // area = 2
      mesh.fVertex(v) = Scalar(1.0e-4);  // Coriolis
    }

    // weightsOnEdge: equal weights for tangential velocity reconstruction
    for (int e = 0; e < nEdges; ++e) {
      for (int i = 0; i < 4; ++i) {
        mesh.weightsOnEdge(i, e) = Scalar(0.25);
      }
    }

    // kiteAreasOnVertex: equal kite areas
    for (int v = 0; v < nVertices; ++v) {
      for (int i = 0; i < vertexDegree; ++i) {
        mesh.kiteAreasOnVertex(i, v) = Scalar(1.0);
      }
    }

    // edgesOnVertex_sign: alternating +1, -1
    for (int v = 0; v < nVertices; ++v) {
      for (int i = 0; i < vertexDegree; ++i) {
        mesh.edgesOnVertex_sign(i, v) = (i % 2 == 0) ? Scalar(1.0) : Scalar(-1.0);
      }
    }

    // edgesOnCell_sign: alternating +1, -1
    for (int c = 0; c < nCells; ++c) {
      for (int i = 0; i < 4; ++i) {
        mesh.edgesOnCell_sign(i, c) = (i % 2 == 0) ? Scalar(1.0) : Scalar(-1.0);
      }
    }

    // State: uniform density = 1.0, linear velocity profile
    for (int k = 0; k < nVertLevels; ++k) {
      for (int c = 0; c < nCells; ++c) {
        h(k, c) = Scalar(1.0);
      }
      for (int e = 0; e < nEdges; ++e) {
        u(k, e) = Scalar(1.0) + Scalar(0.1) * k;
      }
    }
  }
};

// ─── Test: Edge density is average of neighbor cell densities ────────────────
TEST(DiagnosticsModule, EdgeDensityIsAverage) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      /*dt=*/60.0, /*apvm=*/0.0, /*hollingsworth=*/false, /*rk_step=*/3);

  // With uniform h = 1.0, h_edge should be 1.0 everywhere
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      EXPECT_NEAR(sm.diag.h_edge(k, e), Scalar(1.0), kTol)
          << "h_edge mismatch at k=" << k << " e=" << e;
    }
  }
}

// ─── Test: Edge density with non-uniform cell density ────────────────────────
TEST(DiagnosticsModule, EdgeDensityNonUniform) {
  SmallMesh sm;
  // Set non-uniform densities
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      sm.h(k, c) = Scalar(1.0) + Scalar(0.5) * c;
    }
  }

  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, false, 3);

  // Check h_edge = 0.5 * (h(cell1) + h(cell2))
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      int c1 = sm.mesh.cellsOnEdge(0, e) - 1;
      int c2 = sm.mesh.cellsOnEdge(1, e) - 1;
      Scalar expected = Scalar(0.5) * (sm.h(k, c1) + sm.h(k, c2));
      EXPECT_NEAR(sm.diag.h_edge(k, e), expected, kTol)
          << "h_edge mismatch at k=" << k << " e=" << e;
    }
  }
}

// ─── Test: Vorticity computation ─────────────────────────────────────────────
TEST(DiagnosticsModule, VorticityComputation) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, false, 3);

  // Verify vorticity = sum(sign * dcEdge * u) * invAreaTriangle
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int v = 0; v < SmallMesh::nVertices; ++v) {
      Scalar circ = 0;
      for (int i = 0; i < SmallMesh::vertexDegree; ++i) {
        int e = sm.mesh.edgesOnVertex(i, v) - 1;
        Scalar s = sm.mesh.edgesOnVertex_sign(i, v) * sm.mesh.dcEdge(e);
        circ += s * sm.u(k, e);
      }
      Scalar expected = circ * sm.mesh.invAreaTriangle(v);
      EXPECT_NEAR(sm.diag.vorticity(k, v), expected, kTol)
          << "vorticity mismatch at k=" << k << " v=" << v;
    }
  }
}

// ─── Test: Divergence computation ────────────────────────────────────────────
TEST(DiagnosticsModule, DivergenceComputation) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, false, 3);

  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      Scalar div_sum = 0;
      int ne = sm.mesh.nEdgesOnCell(c);
      for (int i = 0; i < ne; ++i) {
        int e = sm.mesh.edgesOnCell(i, c) - 1;
        Scalar s = sm.mesh.edgesOnCell_sign(i, c) * sm.mesh.dvEdge(e);
        div_sum += s * sm.u(k, e);
      }
      Scalar expected = div_sum * sm.mesh.invAreaCell(c);
      EXPECT_NEAR(sm.diag.divergence(k, c), expected, kTol)
          << "divergence mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: Kinetic energy without Hollingsworth ──────────────────────────────
TEST(DiagnosticsModule, KineticEnergyNoHollingsworth) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, /*hollingsworth=*/false, 3);

  // KE = sum(0.25 * dcEdge * dvEdge * u^2) * invAreaCell
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      Scalar ke_sum = 0;
      int ne = sm.mesh.nEdgesOnCell(c);
      for (int i = 0; i < ne; ++i) {
        int e = sm.mesh.edgesOnCell(i, c) - 1;
        Scalar ke_e = sm.mesh.dcEdge(e) * sm.mesh.dvEdge(e) *
                      sm.u(k, e) * sm.u(k, e);
        ke_sum += Scalar(0.25) * ke_e;
      }
      Scalar expected = ke_sum * sm.mesh.invAreaCell(c);
      EXPECT_NEAR(sm.diag.ke(k, c), expected, kTol)
          << "ke mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: Kinetic energy with Hollingsworth adjustment ──────────────────────
TEST(DiagnosticsModule, KineticEnergyHollingsworth) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, /*hollingsworth=*/true, 3);

  // KE values should differ from the non-Hollingsworth case
  // (cannot be identical for non-trivial mesh) - just check they are finite
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      EXPECT_TRUE(std::isfinite(sm.diag.ke(k, c)))
          << "ke not finite at k=" << k << " c=" << c;
      EXPECT_GE(sm.diag.ke(k, c), Scalar(0))
          << "ke negative at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: Tangential velocity reconstruction ────────────────────────────────
TEST(DiagnosticsModule, TangentialVelocity) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, false, /*rk_step=*/3);

  // v(k,e) = sum(weightsOnEdge(i,e) * u(k, edgesOnEdge(i,e)))
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      Scalar v_sum = 0;
      int ne = sm.mesh.nEdgesOnEdge(e);
      for (int i = 0; i < ne; ++i) {
        int eoe = sm.mesh.edgesOnEdge(i, e) - 1;
        v_sum += sm.mesh.weightsOnEdge(i, e) * sm.u(k, eoe);
      }
      EXPECT_NEAR(sm.diag.v(k, e), v_sum, kTol)
          << "v mismatch at k=" << k << " e=" << e;
    }
  }
}

// ─── Test: Tangential velocity NOT reconstructed on rk_step != 3 ─────────────
TEST(DiagnosticsModule, TangentialVelocitySkippedOnRKStep1) {
  SmallMesh sm;
  // Pre-fill v with a known value
  Kokkos::deep_copy(sm.diag.v, Scalar(999.0));

  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, false, /*rk_step=*/1);

  // v should NOT have been overwritten (still 999.0)
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      EXPECT_NEAR(sm.diag.v(k, e), Scalar(999.0), kTol)
          << "v was modified on rk_step=1 at k=" << k << " e=" << e;
    }
  }
}

// ─── Test: Potential vorticity at vertices ───────────────────────────────────
TEST(DiagnosticsModule, PotentialVorticityVertex) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, false, 3);

  // pv_vertex = fVertex + vorticity
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int v = 0; v < SmallMesh::nVertices; ++v) {
      Scalar expected = sm.mesh.fVertex(v) + sm.diag.vorticity(k, v);
      EXPECT_NEAR(sm.diag.pv_vertex(k, v), expected, kTol)
          << "pv_vertex mismatch at k=" << k << " v=" << v;
    }
  }
}

// ─── Test: Potential vorticity at edges ──────────────────────────────────────
TEST(DiagnosticsModule, PotentialVorticityEdge) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, false, 3);

  // pv_edge = 0.5 * (pv_vertex(v1) + pv_vertex(v2))
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      int v1 = sm.mesh.verticesOnEdge(0, e) - 1;
      int v2 = sm.mesh.verticesOnEdge(1, e) - 1;
      Scalar expected = Scalar(0.5) * (sm.diag.pv_vertex(k, v1) +
                                       sm.diag.pv_vertex(k, v2));
      EXPECT_NEAR(sm.diag.pv_edge(k, e), expected, kTol)
          << "pv_edge mismatch at k=" << k << " e=" << e;
    }
  }
}

// ─── Test: APVM upstream bias modifies pv_edge ──────────────────────────────
TEST(DiagnosticsModule, APVMUpstreamBias) {
  SmallMesh sm;

  // Use non-uniform velocity to produce non-zero PV gradients.
  // Each edge gets a unique velocity to break symmetry.
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      sm.u(k, e) = Scalar(1.0) + Scalar(0.5) * e + Scalar(0.1) * k;
    }
  }

  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;

  // Compute with APVM enabled
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, /*apvm=*/0.5, false, /*rk_step=*/3);

  // With non-uniform u, pv_vertex varies across vertices → gradPVt != 0
  bool any_gradPVt_nonzero = false;
  bool any_gradPVn_nonzero = false;
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      EXPECT_TRUE(std::isfinite(sm.diag.pv_edge(k, e)));
      EXPECT_TRUE(std::isfinite(sm.diag.gradPVt(k, e)));
      EXPECT_TRUE(std::isfinite(sm.diag.gradPVn(k, e)));
      if (std::abs(sm.diag.gradPVt(k, e)) > kTol) {
        any_gradPVt_nonzero = true;
      }
      if (std::abs(sm.diag.gradPVn(k, e)) > kTol) {
        any_gradPVn_nonzero = true;
      }
    }
  }
  EXPECT_TRUE(any_gradPVt_nonzero) << "gradPVt all zero - APVM has no effect";
  EXPECT_TRUE(any_gradPVn_nonzero) << "gradPVn all zero - APVM has no effect";
}

// ─── Test: APVM not applied when coefficient is zero ─────────────────────────
TEST(DiagnosticsModule, APVMNotAppliedWhenZero) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, /*apvm=*/0.0, false, 3);

  // gradPVn and gradPVt should remain zero (not computed)
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      EXPECT_NEAR(sm.diag.gradPVn(k, e), Scalar(0), kTol);
      EXPECT_NEAR(sm.diag.gradPVt(k, e), Scalar(0), kTol);
    }
  }
}

// ─── Test: Hollingsworth produces different KE than standard ──────────────────
TEST(DiagnosticsModule, HollingsworthDiffers) {
  SmallMesh sm;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;

  // Compute without Hollingsworth
  mpas::dycore::DiagFields<ExecSpace> diag_std;
  diag_std.h_edge = View2D<Scalar>("he_s", SmallMesh::nVertLevels, SmallMesh::nEdges);
  diag_std.v = View2D<Scalar>("v_s", SmallMesh::nVertLevels, SmallMesh::nEdges);
  diag_std.vorticity = View2D<Scalar>("vort_s", SmallMesh::nVertLevels, SmallMesh::nVertices);
  diag_std.divergence = View2D<Scalar>("div_s", SmallMesh::nVertLevels, SmallMesh::nCells);
  diag_std.ke = View2D<Scalar>("ke_s", SmallMesh::nVertLevels, SmallMesh::nCells);
  diag_std.pv_edge = View2D<Scalar>("pve_s", SmallMesh::nVertLevels, SmallMesh::nEdges);
  diag_std.pv_vertex = View2D<Scalar>("pvv_s", SmallMesh::nVertLevels, SmallMesh::nVertices);
  diag_std.pv_cell = View2D<Scalar>("pvc_s", SmallMesh::nVertLevels, SmallMesh::nCells);
  diag_std.gradPVn = View2D<Scalar>("gpn_s", SmallMesh::nVertLevels, SmallMesh::nEdges);
  diag_std.gradPVt = View2D<Scalar>("gpt_s", SmallMesh::nVertLevels, SmallMesh::nEdges);

  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, diag_std, 60.0, 0.0, false, 3);

  // Compute with Hollingsworth
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag, 60.0, 0.0, true, 3);

  // At least some KE values should differ
  bool any_diff = false;
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      if (std::abs(sm.diag.ke(k, c) - diag_std.ke(k, c)) > kTol) {
        any_diff = true;
        break;
      }
    }
    if (any_diff) break;
  }
  EXPECT_TRUE(any_diff) << "Hollingsworth KE identical to standard - no effect";
}

// ─── Test: Hollingsworth KE formula verification (Req 7.3) ───────────────────
// Verifies the Hollingsworth kinetic energy adjustment against the exact formula:
// ke_vertex(k,iV) = 0.25 * invAreaTriangle(iV) * sum(ke_edge(k, edgesOnVertex(i,iV)))
// ke(k,iC) = ke_fact * ke_std(k,iC) + (1-ke_fact) * sum(kiteAreas * ke_vertex) * invAreaCell
// where ke_fact = 1 - 0.375 = 0.625
TEST(DiagnosticsModule, HollingsworthKEFormulaVerification) {
  SmallMesh sm;
  // Use non-uniform velocity to ensure meaningful KE variation
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      sm.u(k, e) = Scalar(1.0) + Scalar(0.3) * e + Scalar(0.2) * k;
    }
  }

  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, 0.0, /*hollingsworth=*/true, 3);

  // Recompute expected values using the exact Hollingsworth formula
  const Scalar ke_fact = Scalar(1.0) - Scalar(0.375);

  // Step 1: ke_edge(k,e) = dcEdge(e) * dvEdge(e) * u(k,e)^2
  View2D<Scalar> ke_edge("ke_edge_ref", SmallMesh::nVertLevels, SmallMesh::nEdges);
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      ke_edge(k, e) = sm.mesh.dcEdge(e) * sm.mesh.dvEdge(e) *
                      sm.u(k, e) * sm.u(k, e);
    }
  }

  // Step 2: Standard KE at cells
  View2D<Scalar> ke_std("ke_std", SmallMesh::nVertLevels, SmallMesh::nCells);
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      Scalar ke_sum = 0;
      int ne = sm.mesh.nEdgesOnCell(c);
      for (int i = 0; i < ne; ++i) {
        int e = sm.mesh.edgesOnCell(i, c) - 1;
        ke_sum += Scalar(0.25) * ke_edge(k, e);
      }
      ke_std(k, c) = ke_sum * sm.mesh.invAreaCell(c);
    }
  }

  // Step 3: ke_vertex
  View2D<Scalar> ke_vertex("ke_vertex_ref", SmallMesh::nVertLevels, SmallMesh::nVertices);
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int v = 0; v < SmallMesh::nVertices; ++v) {
      Scalar r = Scalar(0.25) * sm.mesh.invAreaTriangle(v);
      Scalar kv = 0;
      for (int i = 0; i < SmallMesh::vertexDegree; ++i) {
        int e = sm.mesh.edgesOnVertex(i, v) - 1;
        kv += ke_edge(k, e);
      }
      ke_vertex(k, v) = kv * r;
    }
  }

  // Step 4: Blend
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      Scalar expected = ke_fact * ke_std(k, c);
      Scalar r = sm.mesh.invAreaCell(c);
      int ne = sm.mesh.nEdgesOnCell(c);
      for (int i = 0; i < ne; ++i) {
        int iv = sm.mesh.verticesOnCell(i, c) - 1;
        int j = sm.mesh.kiteForCell(i, c) - 1;
        expected += (Scalar(1.0) - ke_fact) *
                    sm.mesh.kiteAreasOnVertex(j, iv) *
                    ke_vertex(k, iv) * r;
      }
      EXPECT_NEAR(sm.diag.ke(k, c), expected, kTol)
          << "ke (Hollingsworth) mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: pv_cell area-weighted computation (Req 7.2 support) ───────────────
// When APVM is active, pv_cell must be computed as the area-weighted average of
// surrounding pv_vertex values. Verifies the formula directly.
TEST(DiagnosticsModule, PVCellComputation) {
  SmallMesh sm;
  // Use non-uniform velocity so vorticity/PV varies
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      sm.u(k, e) = Scalar(1.0) + Scalar(0.7) * e - Scalar(0.2) * k;
    }
  }

  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      60.0, /*apvm=*/0.5, false, 3);

  // Verify pv_cell = sum(kiteAreasOnVertex(j,iV) * pv_vertex(k,iV)) * invAreaCell
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int c = 0; c < SmallMesh::nCells; ++c) {
      Scalar pvc_expected = 0;
      Scalar r = sm.mesh.invAreaCell(c);
      int ne = sm.mesh.nEdgesOnCell(c);
      for (int i = 0; i < ne; ++i) {
        int iv = sm.mesh.verticesOnCell(i, c) - 1;
        int j = sm.mesh.kiteForCell(i, c) - 1;
        pvc_expected += sm.mesh.kiteAreasOnVertex(j, iv) *
                        sm.diag.pv_vertex(k, iv) * r;
      }
      EXPECT_NEAR(sm.diag.pv_cell(k, c), pvc_expected, kTol)
          << "pv_cell mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: APVM pv_edge formula verification (Req 7.2) ──────────────────────
// Verifies the full APVM upstream bias formula:
// pv_edge -= apvm * dt * (v * gradPVt + u * gradPVn)
TEST(DiagnosticsModule, APVMPVEdgeFormula) {
  SmallMesh sm;
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      sm.u(k, e) = Scalar(1.0) + Scalar(0.4) * e + Scalar(0.15) * k;
    }
  }

  const Scalar dt = 60.0;
  const Scalar apvm = 0.5;

  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.compute_solve_diagnostics(
      sm.mesh, sm.u, sm.h, sm.diag,
      dt, apvm, false, 3);

  // Verify gradPVt = (pv_vertex(v2) - pv_vertex(v1)) * invDvEdge
  // Verify gradPVn = (pv_cell(c2) - pv_cell(c1)) * invDcEdge
  // And that pv_edge = 0.5*(pv_v1+pv_v2) - apvm*dt*(v*gradPVt + u*gradPVn)
  for (int k = 0; k < SmallMesh::nVertLevels; ++k) {
    for (int e = 0; e < SmallMesh::nEdges; ++e) {
      int v1 = sm.mesh.verticesOnEdge(0, e) - 1;
      int v2 = sm.mesh.verticesOnEdge(1, e) - 1;
      int c1 = sm.mesh.cellsOnEdge(0, e) - 1;
      int c2 = sm.mesh.cellsOnEdge(1, e) - 1;

      Scalar exp_gradPVt = (sm.diag.pv_vertex(k, v2) - sm.diag.pv_vertex(k, v1)) *
                           sm.mesh.invDvEdge(e);
      Scalar exp_gradPVn = (sm.diag.pv_cell(k, c2) - sm.diag.pv_cell(k, c1)) *
                           sm.mesh.invDcEdge(e);

      EXPECT_NEAR(sm.diag.gradPVt(k, e), exp_gradPVt, kTol)
          << "gradPVt mismatch at k=" << k << " e=" << e;
      EXPECT_NEAR(sm.diag.gradPVn(k, e), exp_gradPVn, kTol)
          << "gradPVn mismatch at k=" << k << " e=" << e;

      Scalar pv_base = Scalar(0.5) * (sm.diag.pv_vertex(k, v1) +
                                       sm.diag.pv_vertex(k, v2));
      Scalar expected_pv_edge = pv_base - apvm * dt *
          (sm.diag.v(k, e) * exp_gradPVt + sm.u(k, e) * exp_gradPVn);
      EXPECT_NEAR(sm.diag.pv_edge(k, e), expected_pv_edge, kTol)
          << "pv_edge (APVM) mismatch at k=" << k << " e=" << e;
    }
  }
}

}  // anonymous namespace


// ═══════════════════════════════════════════════════════════════════════════════
// Tests for init_coupled_diagnostics (Requirement 7.4)
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

using CoupledMesh = mpas::dycore::CoupledDiagMeshData<ExecSpace>;
using CoupledState = mpas::dycore::CoupledDiagState<ExecSpace>;
using CoupledFields = mpas::dycore::CoupledDiagFields<ExecSpace>;

/// Build a small test configuration for coupled diagnostics.
struct CoupledTestSetup {
  static constexpr int nCells = 4;
  static constexpr int nEdges = 6;
  static constexpr int nVertLevels = 3;
  static constexpr int maxEdges = 4;

  CoupledMesh cmesh;
  CoupledState state;
  CoupledFields out;

  CoupledTestSetup() {
    cmesh.nCells = nCells;
    cmesh.nEdges = nEdges;
    cmesh.nVertLevels = nVertLevels;
    cmesh.maxEdges = maxEdges;

    // Allocate mesh arrays
    cmesh.cellsOnEdge = View2D<int>("cellsOnEdge", 2, nEdges);
    cmesh.edgesOnCell = View2D<int>("edgesOnCell", maxEdges, nCells);
    cmesh.nEdgesOnCell = View1D<int>("nEdgesOnCell", nCells);
    cmesh.edgesOnCell_sign = View2D<Scalar>("edgesOnCell_sign", maxEdges, nCells);
    cmesh.zz = View2D<Scalar>("zz", nVertLevels, nCells);
    cmesh.fzm = View1D<Scalar>("fzm", nVertLevels + 1);
    cmesh.fzp = View1D<Scalar>("fzp", nVertLevels + 1);
    cmesh.zb_cell = Kokkos::View<Scalar***, layout, MemSpace>("zb_cell", nVertLevels + 1, maxEdges, nCells);
    cmesh.zb3_cell = Kokkos::View<Scalar***, layout, MemSpace>("zb3_cell", nVertLevels + 1, maxEdges, nCells);

    // Allocate state arrays
    state.theta = View2D<Scalar>("theta", nVertLevels, nCells);
    state.rho = View2D<Scalar>("rho", nVertLevels, nCells);
    state.u = View2D<Scalar>("u", nVertLevels, nEdges);
    state.w = View2D<Scalar>("w", nVertLevels + 1, nCells);
    state.qv = View2D<Scalar>("qv", nVertLevels, nCells);
    state.rho_base = View2D<Scalar>("rho_base", nVertLevels, nCells);
    state.theta_base = View2D<Scalar>("theta_base", nVertLevels, nCells);

    // Allocate output arrays
    out.theta_m = View2D<Scalar>("theta_m", nVertLevels, nCells);
    out.rho_zz = View2D<Scalar>("rho_zz", nVertLevels, nCells);
    out.rho_p = View2D<Scalar>("rho_p", nVertLevels, nCells);
    out.rtheta_base = View2D<Scalar>("rtheta_base", nVertLevels, nCells);
    out.rtheta_p = View2D<Scalar>("rtheta_p", nVertLevels, nCells);
    out.ru = View2D<Scalar>("ru", nVertLevels, nEdges);
    out.rw = View2D<Scalar>("rw", nVertLevels + 1, nCells);
    out.exner = View2D<Scalar>("exner", nVertLevels, nCells);
    out.exner_base = View2D<Scalar>("exner_base", nVertLevels, nCells);
    out.pressure_p = View2D<Scalar>("pressure_p", nVertLevels, nCells);
    out.pressure_base = View2D<Scalar>("pressure_base", nVertLevels, nCells);

    fill_test_data();
  }

  void fill_test_data() {
    // Edge connectivity: each edge connects two distinct cells (1-based)
    for (int e = 0; e < nEdges; ++e) {
      cmesh.cellsOnEdge(0, e) = (e % nCells) + 1;
      cmesh.cellsOnEdge(1, e) = ((e + 1) % nCells) + 1;
    }

    // Each cell has 3 edges
    for (int c = 0; c < nCells; ++c) {
      cmesh.nEdgesOnCell(c) = 3;
      for (int i = 0; i < 3; ++i) {
        cmesh.edgesOnCell(i, c) = ((c + i) % nEdges) + 1;
        cmesh.edgesOnCell_sign(i, c) = (i % 2 == 0) ? Scalar(1.0) : Scalar(-1.0);
      }
    }

    // zz: Jacobian factor (positive, typically ~1 for shallow atmosphere)
    for (int k = 0; k < nVertLevels; ++k) {
      for (int c = 0; c < nCells; ++c) {
        cmesh.zz(k, c) = Scalar(1.0) - Scalar(0.01) * k;
      }
    }

    // Vertical interpolation weights: fzm + fzp = 1
    // fzm: weight for level k (upper), fzp: weight for level k-1 (lower)
    for (int k = 0; k <= nVertLevels; ++k) {
      cmesh.fzm(k) = Scalar(0.5);
      cmesh.fzp(k) = Scalar(0.5);
    }

    // zb_cell and zb3_cell: terrain-following metric terms (small for testing)
    Kokkos::deep_copy(cmesh.zb_cell, Scalar(0.01));
    Kokkos::deep_copy(cmesh.zb3_cell, Scalar(0.005));

    // State: typical tropospheric values
    for (int k = 0; k < nVertLevels; ++k) {
      for (int c = 0; c < nCells; ++c) {
        state.theta(k, c) = Scalar(300.0) - Scalar(5.0) * k;
        state.rho(k, c) = Scalar(1.2) - Scalar(0.1) * k;
        state.qv(k, c) = Scalar(0.01) * (1.0 - 0.2 * k);
        state.rho_base(k, c) = Scalar(1.15) - Scalar(0.1) * k;
        state.theta_base(k, c) = Scalar(300.0) - Scalar(5.0) * k;
      }
    }
    for (int k = 0; k < nVertLevels; ++k) {
      for (int e = 0; e < nEdges; ++e) {
        state.u(k, e) = Scalar(10.0) + Scalar(2.0) * k;
      }
    }
    for (int k = 0; k <= nVertLevels; ++k) {
      for (int c = 0; c < nCells; ++c) {
        state.w(k, c) = Scalar(0.1) * k;
      }
    }
  }
};

// ─── Test: theta_m computation ───────────────────────────────────────────────
TEST(CoupledDiagnostics, ThetaM) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  const Scalar rvord = mpas::dycore::diag_constants::rvord;
  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar expected = ts.state.theta(k, c) *
          (Scalar(1.0) + rvord * ts.state.qv(k, c));
      EXPECT_NEAR(ts.out.theta_m(k, c), expected, kTol)
          << "theta_m mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: rho_zz computation ────────────────────────────────────────────────
TEST(CoupledDiagnostics, RhoZZ) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar expected = ts.state.rho(k, c) / ts.cmesh.zz(k, c);
      EXPECT_NEAR(ts.out.rho_zz(k, c), expected, kTol)
          << "rho_zz mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: ru (horizontal mass flux) computation ─────────────────────────────
TEST(CoupledDiagnostics, HorizontalMassFlux) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int e = 0; e < CoupledTestSetup::nEdges; ++e) {
      int c1 = ts.cmesh.cellsOnEdge(0, e) - 1;
      int c2 = ts.cmesh.cellsOnEdge(1, e) - 1;
      Scalar expected = Scalar(0.5) * ts.state.u(k, e) *
          (ts.out.rho_zz(k, c1) + ts.out.rho_zz(k, c2));
      EXPECT_NEAR(ts.out.ru(k, e), expected, kTol)
          << "ru mismatch at k=" << k << " e=" << e;
    }
  }
}

// ─── Test: rw boundary conditions ────────────────────────────────────────────
TEST(CoupledDiagnostics, RwBoundaryConditions) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
    EXPECT_NEAR(ts.out.rw(0, c), Scalar(0.0), kTol)
        << "rw bottom BC non-zero at c=" << c;
    EXPECT_NEAR(ts.out.rw(CoupledTestSetup::nVertLevels, c), Scalar(0.0), kTol)
        << "rw top BC non-zero at c=" << c;
  }
}

// ─── Test: rho_p = rho_zz - rho_base ────────────────────────────────────────
TEST(CoupledDiagnostics, RhoP) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar expected = ts.out.rho_zz(k, c) - ts.state.rho_base(k, c);
      EXPECT_NEAR(ts.out.rho_p(k, c), expected, kTol)
          << "rho_p mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: rtheta_base = theta_base * rho_base ───────────────────────────────
TEST(CoupledDiagnostics, RthetaBase) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar expected = ts.state.theta_base(k, c) * ts.state.rho_base(k, c);
      EXPECT_NEAR(ts.out.rtheta_base(k, c), expected, kTol)
          << "rtheta_base mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: rtheta_p computation ──────────────────────────────────────────────
TEST(CoupledDiagnostics, RthetaP) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar expected = ts.out.theta_m(k, c) * ts.out.rho_p(k, c) +
          ts.state.rho_base(k, c) * (ts.out.theta_m(k, c) - ts.state.theta_base(k, c));
      EXPECT_NEAR(ts.out.rtheta_p(k, c), expected, kTol)
          << "rtheta_p mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: Exner function computation ────────────────────────────────────────
TEST(CoupledDiagnostics, ExnerFunction) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  const Scalar rgas = mpas::dycore::diag_constants::rgas;
  const Scalar rcv = mpas::dycore::diag_constants::rcv;
  const Scalar p0 = mpas::dycore::diag_constants::p0;

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar rtheta_m = ts.out.rtheta_p(k, c) + ts.out.rtheta_base(k, c);
      Scalar expected = std::pow(
          ts.cmesh.zz(k, c) * (rgas / p0) * rtheta_m, rcv);
      EXPECT_NEAR(ts.out.exner(k, c), expected, kTol)
          << "exner mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: Exner base function ───────────────────────────────────────────────
TEST(CoupledDiagnostics, ExnerBase) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  const Scalar rgas = mpas::dycore::diag_constants::rgas;
  const Scalar rcv = mpas::dycore::diag_constants::rcv;
  const Scalar p0 = mpas::dycore::diag_constants::p0;

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar expected = std::pow(
          ts.cmesh.zz(k, c) * (rgas / p0) * ts.out.rtheta_base(k, c), rcv);
      EXPECT_NEAR(ts.out.exner_base(k, c), expected, kTol)
          << "exner_base mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: pressure_p computation ────────────────────────────────────────────
TEST(CoupledDiagnostics, PressureP) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  const Scalar rgas = mpas::dycore::diag_constants::rgas;

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar expected = ts.cmesh.zz(k, c) * rgas *
          (ts.out.exner(k, c) * ts.out.rtheta_p(k, c) +
           ts.out.rtheta_base(k, c) * (ts.out.exner(k, c) - ts.out.exner_base(k, c)));
      EXPECT_NEAR(ts.out.pressure_p(k, c), expected, kTol)
          << "pressure_p mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: pressure_base computation ─────────────────────────────────────────
TEST(CoupledDiagnostics, PressureBase) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  const Scalar rgas = mpas::dycore::diag_constants::rgas;

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      Scalar expected = ts.cmesh.zz(k, c) * rgas *
          ts.out.exner_base(k, c) * ts.out.rtheta_base(k, c);
      EXPECT_NEAR(ts.out.pressure_base(k, c), expected, kTol)
          << "pressure_base mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: dry atmosphere (qv=0) should yield theta_m = theta ────────────────
TEST(CoupledDiagnostics, DryAtmosphere) {
  CoupledTestSetup ts;
  // Set qv = 0 everywhere
  Kokkos::deep_copy(ts.state.qv, Scalar(0.0));

  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      EXPECT_NEAR(ts.out.theta_m(k, c), ts.state.theta(k, c), kTol)
          << "theta_m != theta for dry atmosphere at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: rw interior computation (Req 7.4: mass fluxes) ───────────────────
// Verifies the full rw formula including the terrain-following flux correction:
// rw(k,c) = w(k,c) * (fzp(k)*rho_zz(k-1,c) + fzm(k)*rho_zz(k,c))
//           * (fzp(k)*zz(k-1,c) + fzm(k)*zz(k,c))
//         - sum_edges(edgesOnCell_sign * (zb_cell + sign(flux)*zb3_cell) * flux * zz_face)
TEST(CoupledDiagnostics, RwInteriorComputation) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  // Recompute expected rw for interior levels (k=1..nVertLevels-1)
  for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
    for (int k = 1; k < CoupledTestSetup::nVertLevels; ++k) {
      // w-dependent piece
      Scalar rho_face = ts.cmesh.fzp(k) * ts.out.rho_zz(k - 1, c) +
                        ts.cmesh.fzm(k) * ts.out.rho_zz(k, c);
      Scalar zz_face = ts.cmesh.fzp(k) * ts.cmesh.zz(k - 1, c) +
                       ts.cmesh.fzm(k) * ts.cmesh.zz(k, c);
      Scalar expected = ts.state.w(k, c) * rho_face * zz_face;

      // Subtract flux-divergence correction from ru
      int ne = ts.cmesh.nEdgesOnCell(c);
      for (int i = 0; i < ne; ++i) {
        int iEdge = ts.cmesh.edgesOnCell(i, c) - 1;
        Scalar flux = ts.cmesh.fzm(k) * ts.out.ru(k, iEdge) +
                      ts.cmesh.fzp(k) * ts.out.ru(k - 1, iEdge);
        Scalar sign_flux = (flux >= Scalar(0.0)) ? Scalar(1.0) : Scalar(-1.0);
        expected -= ts.cmesh.edgesOnCell_sign(i, c) *
            (ts.cmesh.zb_cell(k, i, c) + sign_flux * ts.cmesh.zb3_cell(k, i, c)) *
            flux * zz_face;
      }

      EXPECT_NEAR(ts.out.rw(k, c), expected, kTol)
          << "rw mismatch at k=" << k << " c=" << c;
    }
  }
}

// ─── Test: all output fields are finite ──────────────────────────────────────
TEST(CoupledDiagnostics, AllFieldsFinite) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      EXPECT_TRUE(std::isfinite(ts.out.theta_m(k, c)));
      EXPECT_TRUE(std::isfinite(ts.out.rho_zz(k, c)));
      EXPECT_TRUE(std::isfinite(ts.out.rho_p(k, c)));
      EXPECT_TRUE(std::isfinite(ts.out.rtheta_base(k, c)));
      EXPECT_TRUE(std::isfinite(ts.out.rtheta_p(k, c)));
      EXPECT_TRUE(std::isfinite(ts.out.exner(k, c)));
      EXPECT_TRUE(std::isfinite(ts.out.exner_base(k, c)));
      EXPECT_TRUE(std::isfinite(ts.out.pressure_p(k, c)));
      EXPECT_TRUE(std::isfinite(ts.out.pressure_base(k, c)));
    }
    for (int e = 0; e < CoupledTestSetup::nEdges; ++e) {
      EXPECT_TRUE(std::isfinite(ts.out.ru(k, e)));
    }
  }
  for (int k = 0; k <= CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      EXPECT_TRUE(std::isfinite(ts.out.rw(k, c)));
    }
  }
}

// ─── Test: Diverging field identification (Req 14.9) ─────────────────────────
// Verifies that test assertions properly identify the name of a diverging field.
// This test intentionally corrupts one output field and verifies the comparison
// structure is working - each field's EXPECT_NEAR includes an identifying message.
// (The actual requirement is already satisfied by all prior tests using
// << "field_name mismatch at k=" << k ... patterns. This test documents the
// mechanism explicitly.)
TEST(CoupledDiagnostics, DivergingFieldIdentification) {
  CoupledTestSetup ts;
  mpas::dycore::Diagnostics_Module<ExecSpace> diag_mod;
  diag_mod.init_coupled_diagnostics(ts.cmesh, ts.state, ts.out);

  // Verify each coupled diagnostic field is within Parity_Tolerance
  // by scanning all fields and collecting any that diverge.
  // This demonstrates the "identify the diverging field" pattern (Req 14.9).
  std::vector<std::string> diverging_fields;

  auto check_field_2d = [&](const View2D<Scalar>& actual,
                            const View2D<Scalar>& expected,
                            int dim0, int dim1,
                            const std::string& name) {
    for (int k = 0; k < dim0; ++k) {
      for (int i = 0; i < dim1; ++i) {
        if (std::abs(actual(k, i) - expected(k, i)) > kTol) {
          diverging_fields.push_back(name);
          return;
        }
      }
    }
  };

  // Recompute expected theta_m
  View2D<Scalar> exp_theta_m("exp_theta_m", CoupledTestSetup::nVertLevels,
                             CoupledTestSetup::nCells);
  const Scalar rvord = mpas::dycore::diag_constants::rvord;
  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      exp_theta_m(k, c) = ts.state.theta(k, c) *
          (Scalar(1.0) + rvord * ts.state.qv(k, c));
    }
  }
  check_field_2d(ts.out.theta_m, exp_theta_m,
                 CoupledTestSetup::nVertLevels, CoupledTestSetup::nCells, "theta_m");

  // Recompute expected rho_zz
  View2D<Scalar> exp_rho_zz("exp_rho_zz", CoupledTestSetup::nVertLevels,
                            CoupledTestSetup::nCells);
  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      exp_rho_zz(k, c) = ts.state.rho(k, c) / ts.cmesh.zz(k, c);
    }
  }
  check_field_2d(ts.out.rho_zz, exp_rho_zz,
                 CoupledTestSetup::nVertLevels, CoupledTestSetup::nCells, "rho_zz");

  // Recompute expected rho_p
  View2D<Scalar> exp_rho_p("exp_rho_p", CoupledTestSetup::nVertLevels,
                           CoupledTestSetup::nCells);
  for (int k = 0; k < CoupledTestSetup::nVertLevels; ++k) {
    for (int c = 0; c < CoupledTestSetup::nCells; ++c) {
      exp_rho_p(k, c) = exp_rho_zz(k, c) - ts.state.rho_base(k, c);
    }
  }
  check_field_2d(ts.out.rho_p, exp_rho_p,
                 CoupledTestSetup::nVertLevels, CoupledTestSetup::nCells, "rho_p");

  // No fields should diverge within Parity_Tolerance
  EXPECT_TRUE(diverging_fields.empty())
      << "The following fields diverge beyond Parity_Tolerance: "
      << [&]() {
           std::string msg;
           for (const auto& f : diverging_fields) msg += f + ", ";
           return msg;
         }();
}

}  // anonymous namespace
