#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/scalar_transport_mono.hpp"
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

/// Build a small mesh for monotonic scalar transport testing.
/// 5 cells in a line, 4 edges, 4 vertical levels, 2 scalars.
struct MonoTestMesh {
  static constexpr int nCells = 5;
  static constexpr int nEdges = 4;
  static constexpr int nVertLevels = 4;
  static constexpr int num_scalars = 2;
  static constexpr int maxEdges = 2;
  static constexpr int maxAdvCellsForEdge = 4;
  static constexpr int nCellsSolve = 5;

  mpas::dycore::ScalarTransportMeshData<ExecSpace> mesh;
  mpas::dycore::MonoTransportMeshData<ExecSpace> mono_mesh;
  mpas::dycore::MonoTransportState<ExecSpace> state;

  MonoTestMesh() {
    mesh.nCells = nCells;
    mesh.nEdges = nEdges;
    mesh.nVertLevels = nVertLevels;
    mesh.num_scalars = num_scalars;
    mesh.maxEdges = maxEdges;
    mesh.maxAdvCellsForEdge = maxAdvCellsForEdge;

    mono_mesh.nCellsSolve = nCellsSolve;
    mono_mesh.maxEdges = maxEdges;

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
    mono_mesh.cellsOnCell = View2D<int>("cellsOnCell", maxEdges, nCells);

    // Linear mesh: cells 0-1-2-3-4 connected by edges 0-1-2-3
    auto coe_h = Kokkos::create_mirror_view(mesh.cellsOnEdge);
    for (int e = 0; e < nEdges; ++e) {
      coe_h(0, e) = e + 1;     // cell1 (1-based)
      coe_h(1, e) = e + 2;     // cell2 (1-based)
    }
    Kokkos::deep_copy(mesh.cellsOnEdge, coe_h);

    // edgesOnCell and nEdgesOnCell
    auto eoc_h = Kokkos::create_mirror_view(mesh.edgesOnCell);
    auto nec_h = Kokkos::create_mirror_view(mesh.nEdgesOnCell);
    eoc_h(0, 0) = 1; nec_h(0) = 1;
    for (int c = 1; c < nCells - 1; ++c) {
      eoc_h(0, c) = c;       // left edge (1-based)
      eoc_h(1, c) = c + 1;   // right edge (1-based)
      nec_h(c) = 2;
    }
    eoc_h(0, nCells - 1) = nEdges; nec_h(nCells - 1) = 1;
    Kokkos::deep_copy(mesh.edgesOnCell, eoc_h);
    Kokkos::deep_copy(mesh.nEdgesOnCell, nec_h);

    // Edge signs
    auto ecs_h = Kokkos::create_mirror_view(mesh.edgesOnCell_sign);
    ecs_h(0, 0) = Scalar(1.0);
    for (int c = 1; c < nCells - 1; ++c) {
      ecs_h(0, c) = Scalar(-1.0);
      ecs_h(1, c) = Scalar(1.0);
    }
    ecs_h(0, nCells - 1) = Scalar(-1.0);
    Kokkos::deep_copy(mesh.edgesOnCell_sign, ecs_h);

    // cellsOnCell: neighbor connectivity (1-based)
    auto coc_h = Kokkos::create_mirror_view(mono_mesh.cellsOnCell);
    // Cell 0 neighbors: cell 1
    coc_h(0, 0) = 2;
    // Cells 1-3: two neighbors
    for (int c = 1; c < nCells - 1; ++c) {
      coc_h(0, c) = c;      // left neighbor (1-based)
      coc_h(1, c) = c + 2;  // right neighbor (1-based)
    }
    // Cell 4: cell 3
    coc_h(0, nCells - 1) = nCells - 1;
    Kokkos::deep_copy(mono_mesh.cellsOnCell, coc_h);

    // Advection stencil: simple 2-point upwind
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
      ac3_h(0, e) = Scalar(0.5);
      ac3_h(1, e) = Scalar(-0.5);
    }
    Kokkos::deep_copy(mesh.advCellsForEdge, acfe_h);
    Kokkos::deep_copy(mesh.nAdvCellsForEdge, nacfe_h);
    Kokkos::deep_copy(mesh.adv_coefs, ac_h);
    Kokkos::deep_copy(mesh.adv_coefs_3rd, ac3_h);

    // Geometry: uniform
    auto dv_h = Kokkos::create_mirror_view(mesh.dvEdge);
    auto ia_h = Kokkos::create_mirror_view(mesh.invAreaCell);
    for (int e = 0; e < nEdges; ++e) dv_h(e) = Scalar(1.0);
    for (int c = 0; c < nCells; ++c) ia_h(c) = Scalar(1.0);
    Kokkos::deep_copy(mesh.dvEdge, dv_h);
    Kokkos::deep_copy(mesh.invAreaCell, ia_h);

    // Vertical weights
    auto fnm_h = Kokkos::create_mirror_view(mesh.fnm);
    auto fnp_h = Kokkos::create_mirror_view(mesh.fnp);
    auto rdnw_h = Kokkos::create_mirror_view(mesh.rdnw);
    for (int k = 0; k <= nVertLevels; ++k) {
      fnm_h(k) = Scalar(0.5);
      fnp_h(k) = Scalar(0.5);
    }
    for (int k = 0; k < nVertLevels; ++k) {
      rdnw_h(k) = Scalar(1.0);
    }
    Kokkos::deep_copy(mesh.fnm, fnm_h);
    Kokkos::deep_copy(mesh.fnp, fnp_h);
    Kokkos::deep_copy(mesh.rdnw, rdnw_h);

    // No regional boundaries
    Kokkos::deep_copy(mesh.bdyMaskCell, 0);
    Kokkos::deep_copy(mesh.bdyMaskEdge, 0);

    // State fields
    state.scalars_old = View3D<Scalar>("scalars_old", num_scalars, nVertLevels, nCells);
    state.scalars_new = View3D<Scalar>("scalars_new", num_scalars, nVertLevels, nCells);
    state.scalar_tend = View3D<Scalar>("scalar_tend", num_scalars, nVertLevels, nCells);
    state.rho_zz_old = View2D<Scalar>("rho_zz_old", nVertLevels, nCells);
    state.rho_zz_new = View2D<Scalar>("rho_zz_new", nVertLevels, nCells);
    state.uhAvg = View2D<Scalar>("uhAvg", nVertLevels, nEdges);
    state.wwAvg = View2D<Scalar>("wwAvg", nVertLevels + 1, nCells);

    // Initialize: uniform density = 1.0, zero fluxes and tendencies
    Kokkos::deep_copy(state.rho_zz_old, Scalar(1.0));
    Kokkos::deep_copy(state.rho_zz_new, Scalar(1.0));
    Kokkos::deep_copy(state.uhAvg, Scalar(0.0));
    Kokkos::deep_copy(state.wwAvg, Scalar(0.0));
    Kokkos::deep_copy(state.scalar_tend, Scalar(0.0));
  }
};

// ============================================================================
// Test: Zero flux preserves uniform scalar (monotonic transport)
// ============================================================================
TEST(ScalarTransportMono, ZeroFluxPreservesUniform) {
  MonoTestMesh tm;

  // Uniform scalar = 2.0
  Kokkos::deep_copy(tm.state.scalars_old, Scalar(2.0));
  Kokkos::deep_copy(tm.state.scalars_new, Scalar(2.0));

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      tm.mesh, tm.mono_mesh, tm.state,
      /*dt=*/Scalar(1.0),
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,
      /*moist_end=*/2);

  auto result_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  Kokkos::deep_copy(result_h, tm.state.scalars_new);

  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        EXPECT_NEAR(result_h(s, k, c), Scalar(2.0), kTol)
            << "Mismatch at s=" << s << " k=" << k << " c=" << c;
      }
    }
  }
}

// ============================================================================
// Test: Monotonic limiter keeps scalars within local min/max (Req 5.6)
// ============================================================================
TEST(ScalarTransportMono, LimiterKeepsWithinBounds) {
  MonoTestMesh tm;

  // Set a step-function profile that will generate overshoots without limiter
  auto so_h = Kokkos::create_mirror_view(tm.state.scalars_old);
  auto sn_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        // Step: cells 0-1 = 1.0, cells 2-4 = 10.0
        Scalar val = (c < 2) ? Scalar(1.0) : Scalar(10.0);
        so_h(s, k, c) = val;
        sn_h(s, k, c) = val;
      }
    }
  }
  Kokkos::deep_copy(tm.state.scalars_old, so_h);
  Kokkos::deep_copy(tm.state.scalars_new, sn_h);

  // Set uniform positive horizontal mass flux to drive advection
  Kokkos::deep_copy(tm.state.uhAvg, Scalar(0.5));

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      tm.mesh, tm.mono_mesh, tm.state,
      /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,
      /*moist_end=*/2);

  auto result_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  Kokkos::deep_copy(result_h, tm.state.scalars_new);

  // The global min is 1.0 and max is 10.0.
  // The monotonic limiter should keep all results within [local_min, local_max].
  // With our mesh, the local bounds for each cell include its neighbors.
  // The result should never go below 1.0 or above 10.0.
  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        EXPECT_GE(result_h(s, k, c), Scalar(1.0) - kTol)
            << "Below min at s=" << s << " k=" << k << " c=" << c;
        EXPECT_LE(result_h(s, k, c), Scalar(10.0) + kTol)
            << "Above max at s=" << s << " k=" << k << " c=" << c;
      }
    }
  }
}

// ============================================================================
// Test: Positive-definite enforcement for water species (Req 5.7)
// ============================================================================
TEST(ScalarTransportMono, WaterSpeciesNonNegative) {
  MonoTestMesh tm;

  // Set a scenario that would produce negatives without PD enforcement:
  // Large negative tendency applied to a small positive scalar
  auto so_h = Kokkos::create_mirror_view(tm.state.scalars_old);
  auto sn_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  auto tend_h = Kokkos::create_mirror_view(tm.state.scalar_tend);

  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        so_h(s, k, c) = Scalar(0.001);  // Small positive value
        sn_h(s, k, c) = Scalar(0.001);
        // Large negative tendency that would push scalar negative
        tend_h(s, k, c) = Scalar(-0.1);
      }
    }
  }
  Kokkos::deep_copy(tm.state.scalars_old, so_h);
  Kokkos::deep_copy(tm.state.scalars_new, sn_h);
  Kokkos::deep_copy(tm.state.scalar_tend, tend_h);

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      tm.mesh, tm.mono_mesh, tm.state,
      /*dt=*/Scalar(1.0),
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,    // both scalars are water species
      /*moist_end=*/2);

  auto result_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  Kokkos::deep_copy(result_h, tm.state.scalars_new);

  // All water species should be non-negative (Req 5.7)
  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        EXPECT_GE(result_h(s, k, c), Scalar(0.0))
            << "Negative water species at s=" << s
            << " k=" << k << " c=" << c
            << " value=" << result_h(s, k, c);
      }
    }
  }
}

// ============================================================================
// Test: Non-water species can go negative (not enforced)
// ============================================================================
TEST(ScalarTransportMono, NonWaterCanGoNegative) {
  MonoTestMesh tm;

  // Same scenario but scalars are NOT water species (moist_start=moist_end=0)
  auto so_h = Kokkos::create_mirror_view(tm.state.scalars_old);
  auto sn_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  auto tend_h = Kokkos::create_mirror_view(tm.state.scalar_tend);

  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        so_h(s, k, c) = Scalar(0.001);
        sn_h(s, k, c) = Scalar(0.001);
        tend_h(s, k, c) = Scalar(-0.1);
      }
    }
  }
  Kokkos::deep_copy(tm.state.scalars_old, so_h);
  Kokkos::deep_copy(tm.state.scalars_new, sn_h);
  Kokkos::deep_copy(tm.state.scalar_tend, tend_h);

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      tm.mesh, tm.mono_mesh, tm.state,
      /*dt=*/Scalar(1.0),
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/2,    // No scalars are water species
      /*moist_end=*/2);

  auto result_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  Kokkos::deep_copy(result_h, tm.state.scalars_new);

  // Non-water species can be negative (tendency pushes them there)
  bool any_negative = false;
  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        if (result_h(s, k, c) < Scalar(0.0)) any_negative = true;
      }
    }
  }
  EXPECT_TRUE(any_negative) << "Expected some negative values for non-water species";
}

// ============================================================================
// Test: Regional boundary flux treatment (Req 5.8)
// ============================================================================
TEST(ScalarTransportMono, RegionalBoundaryFluxTreatment) {
  MonoTestMesh tm;

  // Set up boundary masks: cells 3,4 and edges 2,3 are in the relaxation zone
  auto bmc_h = Kokkos::create_mirror_view(tm.mesh.bdyMaskCell);
  auto bme_h = Kokkos::create_mirror_view(tm.mesh.bdyMaskEdge);
  bmc_h(0) = 0; bmc_h(1) = 0; bmc_h(2) = 0;
  bmc_h(3) = 1; bmc_h(4) = 2;  // specified zone
  bme_h(0) = 0; bme_h(1) = 0;
  bme_h(2) = mpas::dycore::transport_nRelaxZone - 1;  // In relaxation boundary
  bme_h(3) = mpas::dycore::transport_nRelaxZone;      // In relaxation boundary
  Kokkos::deep_copy(tm.mesh.bdyMaskCell, bmc_h);
  Kokkos::deep_copy(tm.mesh.bdyMaskEdge, bme_h);

  // Step function: cell 0,1 = 1.0, cell 2,3,4 = 5.0
  auto so_h = Kokkos::create_mirror_view(tm.state.scalars_old);
  auto sn_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        Scalar val = (c < 2) ? Scalar(1.0) : Scalar(5.0);
        so_h(s, k, c) = val;
        sn_h(s, k, c) = val;
      }
    }
  }
  Kokkos::deep_copy(tm.state.scalars_old, so_h);
  Kokkos::deep_copy(tm.state.scalars_new, sn_h);

  // Positive horizontal flow
  Kokkos::deep_copy(tm.state.uhAvg, Scalar(0.5));

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      tm.mesh, tm.mono_mesh, tm.state,
      /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/true,
      /*moist_start=*/0,
      /*moist_end=*/2);

  // Verify it runs without error. Regional cells (bdyMask <= nSpecZone)
  // should be updated. The key property: results should be finite.
  auto result_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  Kokkos::deep_copy(result_h, tm.state.scalars_new);

  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        EXPECT_TRUE(std::isfinite(result_h(s, k, c)))
            << "Non-finite at s=" << s << " k=" << k << " c=" << c;
      }
    }
  }
}

// ============================================================================
// Test: Density re-integration is applied correctly
// ============================================================================
TEST(ScalarTransportMono, DensityReintegration) {
  MonoTestMesh tm;

  // Uniform scalar, non-uniform density to test density re-integration
  Kokkos::deep_copy(tm.state.scalars_old, Scalar(1.0));
  Kokkos::deep_copy(tm.state.scalars_new, Scalar(1.0));
  Kokkos::deep_copy(tm.state.rho_zz_old, Scalar(1.0));
  Kokkos::deep_copy(tm.state.rho_zz_new, Scalar(1.5));

  // With zero fluxes, the final scalar = old * rho_old / rho_int
  // where rho_int = rho_old + dt * (horiz_div + vert_div)
  // With zero mass fluxes: rho_int = rho_old = 1.0
  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      tm.mesh, tm.mono_mesh, tm.state,
      /*dt=*/Scalar(1.0),
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,
      /*moist_end=*/2);

  auto result_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  Kokkos::deep_copy(result_h, tm.state.scalars_new);

  // With zero mass fluxes: rho_int = rho_old = 1.0
  // scalar_new = 1.0 * 1.0 / 1.0 = 1.0
  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        EXPECT_NEAR(result_h(s, k, c), Scalar(1.0), kTol)
            << "DIVERGING FIELD: scalars_new — density reintegration "
               "incorrect at s=" << s << " k=" << k << " c=" << c;
      }
    }
  }
}

// ============================================================================
// Test: Reference_Model parity — monotonic transport (Req 14.2, 14.9)
// Verifies monotonic transport of a uniform field with horizontal flux
// produces the expected output, failing and identifying any diverging field.
// ============================================================================
TEST(ScalarTransportMono, ReferenceModelParity) {
  MonoTestMesh tm;

  // Uniform scalar = 3.0, uniform density = 1.0
  Kokkos::deep_copy(tm.state.scalars_old, Scalar(3.0));
  Kokkos::deep_copy(tm.state.scalars_new, Scalar(3.0));
  Kokkos::deep_copy(tm.state.rho_zz_old, Scalar(1.0));
  Kokkos::deep_copy(tm.state.rho_zz_new, Scalar(1.0));

  // Uniform horizontal flux: with uniform scalar, divergence = 0
  Kokkos::deep_copy(tm.state.uhAvg, Scalar(0.5));
  Kokkos::deep_copy(tm.state.wwAvg, Scalar(0.0));
  Kokkos::deep_copy(tm.state.scalar_tend, Scalar(0.0));

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      tm.mesh, tm.mono_mesh, tm.state,
      /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,
      /*moist_end=*/2);

  auto result_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  Kokkos::deep_copy(result_h, tm.state.scalars_new);

  // Reference: uniform scalar -> all fluxes reconstruct the uniform value.
  // Net divergence=0 for interior cells. With rho_int = rho_old = 1.0
  // (zero mass flux divergence), scalar_new = scalar_old = 3.0.
  // Source tendency already applied (3.0 + dt*0/1.0 = 3.0).
  for (int c = 1; c < MonoTestMesh::nCells - 1; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        EXPECT_NEAR(result_h(s, k, c), Scalar(3.0), kTol)
            << "DIVERGING FIELD: scalars_new — Reference_Model parity "
               "violation at scalar=" << s << " k=" << k << " cell=" << c
            << " (expected 3.0, got " << result_h(s, k, c) << ")";
      }
    }
  }
}

// ============================================================================
// Test: Monotonic limiter identifies diverging field (Req 14.9)
// Validates that after step-function advection, all outputs are within
// bounds and finite. Failures identify the specific scalar, level, and cell.
// ============================================================================
TEST(ScalarTransportMono, LimiterParityDivergenceCheck) {
  MonoTestMesh tm;

  // Set varying profile across cells
  auto so_h = Kokkos::create_mirror_view(tm.state.scalars_old);
  auto sn_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        // Smooth profile: scalar = 2 + sin(c)
        Scalar val = Scalar(2.0) + std::sin(Scalar(c));
        so_h(s, k, c) = val;
        sn_h(s, k, c) = val;
      }
    }
  }
  Kokkos::deep_copy(tm.state.scalars_old, so_h);
  Kokkos::deep_copy(tm.state.scalars_new, sn_h);
  Kokkos::deep_copy(tm.state.uhAvg, Scalar(0.3));

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      tm.mesh, tm.mono_mesh, tm.state,
      /*dt=*/Scalar(0.1),
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,
      /*moist_end=*/2);

  auto result_h = Kokkos::create_mirror_view(tm.state.scalars_new);
  Kokkos::deep_copy(result_h, tm.state.scalars_new);

  // Check all results are finite and positive (water species)
  for (int c = 0; c < MonoTestMesh::nCells; ++c) {
    for (int k = 0; k < MonoTestMesh::nVertLevels; ++k) {
      for (int s = 0; s < MonoTestMesh::num_scalars; ++s) {
        EXPECT_TRUE(std::isfinite(result_h(s, k, c)))
            << "DIVERGING FIELD: scalars_new — non-finite at scalar="
            << s << " k=" << k << " cell=" << c;
        EXPECT_GE(result_h(s, k, c), Scalar(0.0))
            << "DIVERGING FIELD: scalars_new — negative water species "
               "at scalar=" << s << " k=" << k << " cell=" << c
            << " value=" << result_h(s, k, c);
      }
    }
  }
}

}  // namespace
