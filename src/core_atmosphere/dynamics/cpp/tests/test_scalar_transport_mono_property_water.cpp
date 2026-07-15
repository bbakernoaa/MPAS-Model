/// @file test_scalar_transport_mono_property_water.cpp
/// @brief Property-based test: Water-species non-negativity after monotonic transport.
///
/// Feature: mpas-dycore-cpp-port, Property 12: Water-species non-negativity
///
/// **Validates: Requirements 5.7**
///
/// For randomly generated initial water-species scalar fields (positive values) and
/// random mass fluxes/tendencies, the monotonic transport NEVER produces negative
/// water-species values after the update.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

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

template <class T>
using View1D = Kokkos::View<T*, layout, MemSpace>;
template <class T>
using View2D = Kokkos::View<T**, layout, MemSpace>;
template <class T>
using View3D = Kokkos::View<T***, layout, MemSpace>;

/// Generate a random mesh size within reasonable bounds for property testing.
struct MeshParams {
  int nCells;
  int nEdges;
  int nVertLevels;
  int num_scalars;
};

/// Generate mesh parameters for a linear mesh.
/// We keep sizes small enough for fast execution but large enough to exercise the algorithm.
MeshParams generateMeshParams() {
  MeshParams p;
  p.nCells = *rc::gen::inRange(3, 12);       // [3, 11] cells
  p.nVertLevels = *rc::gen::inRange(3, 10);  // [3, 9] levels (need >=3 for interior flux3)
  p.num_scalars = *rc::gen::inRange(1, 5);   // [1, 4] scalars (all water species)
  // For a linear mesh: nEdges = nCells - 1
  p.nEdges = p.nCells - 1;
  return p;
}

/// Generate a positive scalar value (water species are non-negative initially).
rc::Gen<Scalar> genPositiveScalar() {
  return rc::gen::map(rc::gen::inRange(1, 10001), [](int v) {
    return static_cast<Scalar>(v) * Scalar(0.001);  // [0.001, 10.0]
  });
}

/// Generate a scalar tendency value that can be negative (drives scalar toward negative).
rc::Gen<Scalar> genTendency() {
  return rc::gen::map(rc::gen::inRange(-5000, 5001), [](int v) {
    return static_cast<Scalar>(v) * Scalar(0.001);  // [-5.0, 5.0]
  });
}

/// Generate a mass flux value (horizontal).
rc::Gen<Scalar> genMassFlux() {
  return rc::gen::map(rc::gen::inRange(-2000, 2001), [](int v) {
    return static_cast<Scalar>(v) * Scalar(0.0001);  // [-0.2, 0.2]
  });
}

/// Generate a vertical mass flux value (smaller magnitude).
rc::Gen<Scalar> genVertFlux() {
  return rc::gen::map(rc::gen::inRange(-500, 501), [](int v) {
    return static_cast<Scalar>(v) * Scalar(0.0001);  // [-0.05, 0.05]
  });
}

/// Generate a positive density value.
rc::Gen<Scalar> genDensity() {
  return rc::gen::map(rc::gen::inRange(500, 2001), [](int v) {
    return static_cast<Scalar>(v) * Scalar(0.001);  // [0.5, 2.0]
  });
}

/// Build a complete test scenario with randomized inputs for the monotonic transport.
/// Uses a linear mesh topology (cells connected in a line).
struct MonoPropertyTestData {
  MeshParams params;
  mpas::dycore::ScalarTransportMeshData<ExecSpace> mesh;
  mpas::dycore::MonoTransportMeshData<ExecSpace> mono_mesh;
  mpas::dycore::MonoTransportState<ExecSpace> state;

  static constexpr int maxEdges = 2;
  static constexpr int maxAdvCellsForEdge = 4;

  void build(const MeshParams& p) {
    params = p;
    const int nCells = p.nCells;
    const int nEdges = p.nEdges;
    const int nVertLevels = p.nVertLevels;
    const int num_scalars = p.num_scalars;

    mesh.nCells = nCells;
    mesh.nEdges = nEdges;
    mesh.nVertLevels = nVertLevels;
    mesh.num_scalars = num_scalars;
    mesh.maxEdges = maxEdges;
    mesh.maxAdvCellsForEdge = maxAdvCellsForEdge;

    mono_mesh.nCellsSolve = nCells;
    mono_mesh.maxEdges = maxEdges;

    // Allocate connectivity arrays
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

    // ── Set up linear mesh connectivity ──
    // cellsOnEdge: edge e connects cell e and cell e+1 (1-based)
    auto coe_h = Kokkos::create_mirror_view(mesh.cellsOnEdge);
    for (int e = 0; e < nEdges; ++e) {
      coe_h(0, e) = e + 1;     // cell1 (1-based)
      coe_h(1, e) = e + 2;     // cell2 (1-based)
    }
    Kokkos::deep_copy(mesh.cellsOnEdge, coe_h);

    // edgesOnCell and nEdgesOnCell for linear mesh
    auto eoc_h = Kokkos::create_mirror_view(mesh.edgesOnCell);
    auto nec_h = Kokkos::create_mirror_view(mesh.nEdgesOnCell);
    eoc_h(0, 0) = 1; nec_h(0) = 1;  // cell 0: edge 0 only
    for (int c = 1; c < nCells - 1; ++c) {
      eoc_h(0, c) = c;       // left edge (1-based)
      eoc_h(1, c) = c + 1;   // right edge (1-based)
      nec_h(c) = 2;
    }
    eoc_h(0, nCells - 1) = nEdges; nec_h(nCells - 1) = 1;  // last cell: last edge
    Kokkos::deep_copy(mesh.edgesOnCell, eoc_h);
    Kokkos::deep_copy(mesh.nEdgesOnCell, nec_h);

    // Edge signs: outward-positive convention for the linear mesh
    auto ecs_h = Kokkos::create_mirror_view(mesh.edgesOnCell_sign);
    ecs_h(0, 0) = Scalar(1.0);
    for (int c = 1; c < nCells - 1; ++c) {
      ecs_h(0, c) = Scalar(-1.0);   // left edge (inward)
      ecs_h(1, c) = Scalar(1.0);    // right edge (outward)
    }
    ecs_h(0, nCells - 1) = Scalar(-1.0);
    Kokkos::deep_copy(mesh.edgesOnCell_sign, ecs_h);

    // cellsOnCell: neighbor connectivity (1-based)
    auto coc_h = Kokkos::create_mirror_view(mono_mesh.cellsOnCell);
    coc_h(0, 0) = 2;  // cell 0's neighbor is cell 1
    for (int c = 1; c < nCells - 1; ++c) {
      coc_h(0, c) = c;      // left neighbor (1-based)
      coc_h(1, c) = c + 2;  // right neighbor (1-based)
    }
    coc_h(0, nCells - 1) = nCells - 1;  // last cell's neighbor
    Kokkos::deep_copy(mono_mesh.cellsOnCell, coc_h);

    // Advection stencil: 2-point for each edge
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

    // Geometry: uniform mesh spacing
    auto dv_h = Kokkos::create_mirror_view(mesh.dvEdge);
    auto ia_h = Kokkos::create_mirror_view(mesh.invAreaCell);
    for (int e = 0; e < nEdges; ++e) dv_h(e) = Scalar(1.0);
    for (int c = 0; c < nCells; ++c) ia_h(c) = Scalar(1.0);
    Kokkos::deep_copy(mesh.dvEdge, dv_h);
    Kokkos::deep_copy(mesh.invAreaCell, ia_h);

    // Vertical interpolation weights
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

    // ── Allocate state fields ──
    state.scalars_old = View3D<Scalar>("scalars_old", num_scalars, nVertLevels, nCells);
    state.scalars_new = View3D<Scalar>("scalars_new", num_scalars, nVertLevels, nCells);
    state.scalar_tend = View3D<Scalar>("scalar_tend", num_scalars, nVertLevels, nCells);
    state.rho_zz_old = View2D<Scalar>("rho_zz_old", nVertLevels, nCells);
    state.rho_zz_new = View2D<Scalar>("rho_zz_new", nVertLevels, nCells);
    state.uhAvg = View2D<Scalar>("uhAvg", nVertLevels, nEdges);
    state.wwAvg = View2D<Scalar>("wwAvg", nVertLevels + 1, nCells);
  }

  /// Fill state with randomly generated values.
  void fillRandom() {
    const int nCells = params.nCells;
    const int nEdges = params.nEdges;
    const int nVertLevels = params.nVertLevels;
    const int num_scalars = params.num_scalars;

    // Fill scalars with positive random values (water species start non-negative)
    auto so_h = Kokkos::create_mirror_view(state.scalars_old);
    auto sn_h = Kokkos::create_mirror_view(state.scalars_new);
    for (int c = 0; c < nCells; ++c) {
      for (int k = 0; k < nVertLevels; ++k) {
        for (int s = 0; s < num_scalars; ++s) {
          Scalar val = *genPositiveScalar();
          so_h(s, k, c) = val;
          sn_h(s, k, c) = val;
        }
      }
    }
    Kokkos::deep_copy(state.scalars_old, so_h);
    Kokkos::deep_copy(state.scalars_new, sn_h);

    // Fill tendencies with random values (can be negative, drives toward negative)
    auto tend_h = Kokkos::create_mirror_view(state.scalar_tend);
    for (int c = 0; c < nCells; ++c) {
      for (int k = 0; k < nVertLevels; ++k) {
        for (int s = 0; s < num_scalars; ++s) {
          tend_h(s, k, c) = *genTendency();
        }
      }
    }
    Kokkos::deep_copy(state.scalar_tend, tend_h);

    // Fill density with positive values
    auto rho_old_h = Kokkos::create_mirror_view(state.rho_zz_old);
    auto rho_new_h = Kokkos::create_mirror_view(state.rho_zz_new);
    for (int c = 0; c < nCells; ++c) {
      for (int k = 0; k < nVertLevels; ++k) {
        rho_old_h(k, c) = *genDensity();
        rho_new_h(k, c) = *genDensity();
      }
    }
    Kokkos::deep_copy(state.rho_zz_old, rho_old_h);
    Kokkos::deep_copy(state.rho_zz_new, rho_new_h);

    // Fill horizontal mass fluxes
    auto uh_h = Kokkos::create_mirror_view(state.uhAvg);
    for (int e = 0; e < nEdges; ++e) {
      for (int k = 0; k < nVertLevels; ++k) {
        uh_h(k, e) = *genMassFlux();
      }
    }
    Kokkos::deep_copy(state.uhAvg, uh_h);

    // Fill vertical mass fluxes (boundaries must be zero)
    auto ww_h = Kokkos::create_mirror_view(state.wwAvg);
    for (int c = 0; c < nCells; ++c) {
      ww_h(0, c) = Scalar(0.0);  // top boundary
      ww_h(nVertLevels, c) = Scalar(0.0);  // bottom boundary
      for (int k = 1; k < nVertLevels; ++k) {
        ww_h(k, c) = *genVertFlux();
      }
    }
    Kokkos::deep_copy(state.wwAvg, ww_h);
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Property 12: Water-species non-negativity
// Feature: mpas-dycore-cpp-port, Property 12: Water-species non-negativity
// ---------------------------------------------------------------------------

/// For any randomly generated positive water-species scalar fields and random
/// mass fluxes/tendencies, after monotonic transport ALL water-species mixing
/// ratios must be >= 0 (Req 5.7: negative water-species set to zero).
RC_GTEST_PROP(ScalarTransportMonoPropertyWater,
              WaterSpeciesNeverNegative,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 12: Water-species non-negativity
  auto params = generateMeshParams();

  MonoPropertyTestData testData;
  testData.build(params);
  testData.fillRandom();

  // Use a small timestep to keep the transport stable
  const Scalar dt = Scalar(0.01);

  // All scalars are water species (moist_start=0, moist_end=num_scalars)
  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      testData.mesh, testData.mono_mesh, testData.state,
      dt,
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,
      /*moist_end=*/params.num_scalars);

  // Verify: ALL water-species values must be non-negative after transport
  auto result_h = Kokkos::create_mirror_view(testData.state.scalars_new);
  Kokkos::deep_copy(result_h, testData.state.scalars_new);

  for (int c = 0; c < params.nCells; ++c) {
    for (int k = 0; k < params.nVertLevels; ++k) {
      for (int s = 0; s < params.num_scalars; ++s) {
        RC_ASSERT(result_h(s, k, c) >= Scalar(0.0));
      }
    }
  }
}

/// Variant: Test with larger tendencies that aggressively push scalars negative.
/// The positive-definite enforcement (Req 5.7) must still ensure non-negativity.
RC_GTEST_PROP(ScalarTransportMonoPropertyWater,
              AggressiveNegativeTendenciesStillNonNegative,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 12: Water-species non-negativity
  auto params = generateMeshParams();

  MonoPropertyTestData testData;
  testData.build(params);

  const int nCells = params.nCells;
  const int nEdges = params.nEdges;
  const int nVertLevels = params.nVertLevels;
  const int num_scalars = params.num_scalars;

  // Start with very small positive scalar values
  auto so_h = Kokkos::create_mirror_view(testData.state.scalars_old);
  auto sn_h = Kokkos::create_mirror_view(testData.state.scalars_new);
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k) {
      for (int s = 0; s < num_scalars; ++s) {
        // Very small positive initial values
        int raw = *rc::gen::inRange(1, 100);
        Scalar val = static_cast<Scalar>(raw) * Scalar(0.0001);  // [0.0001, 0.01]
        so_h(s, k, c) = val;
        sn_h(s, k, c) = val;
      }
    }
  }
  Kokkos::deep_copy(testData.state.scalars_old, so_h);
  Kokkos::deep_copy(testData.state.scalars_new, sn_h);

  // Use large negative tendencies to drive scalar values below zero
  auto tend_h = Kokkos::create_mirror_view(testData.state.scalar_tend);
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k) {
      for (int s = 0; s < num_scalars; ++s) {
        int raw = *rc::gen::inRange(-10000, -1000);
        tend_h(s, k, c) = static_cast<Scalar>(raw) * Scalar(0.001);  // [-10.0, -1.0]
      }
    }
  }
  Kokkos::deep_copy(testData.state.scalar_tend, tend_h);

  // Density: positive
  auto rho_old_h = Kokkos::create_mirror_view(testData.state.rho_zz_old);
  auto rho_new_h = Kokkos::create_mirror_view(testData.state.rho_zz_new);
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k) {
      rho_old_h(k, c) = *genDensity();
      rho_new_h(k, c) = *genDensity();
    }
  }
  Kokkos::deep_copy(testData.state.rho_zz_old, rho_old_h);
  Kokkos::deep_copy(testData.state.rho_zz_new, rho_new_h);

  // Random mass fluxes
  auto uh_h = Kokkos::create_mirror_view(testData.state.uhAvg);
  for (int e = 0; e < nEdges; ++e) {
    for (int k = 0; k < nVertLevels; ++k) {
      uh_h(k, e) = *genMassFlux();
    }
  }
  Kokkos::deep_copy(testData.state.uhAvg, uh_h);

  // Vertical fluxes (boundaries zero)
  auto ww_h = Kokkos::create_mirror_view(testData.state.wwAvg);
  for (int c = 0; c < nCells; ++c) {
    ww_h(0, c) = Scalar(0.0);
    ww_h(nVertLevels, c) = Scalar(0.0);
    for (int k = 1; k < nVertLevels; ++k) {
      ww_h(k, c) = *genVertFlux();
    }
  }
  Kokkos::deep_copy(testData.state.wwAvg, ww_h);

  // Run monotonic transport with a larger dt to stress the enforcement
  const Scalar dt = Scalar(0.1);

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      testData.mesh, testData.mono_mesh, testData.state,
      dt,
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,
      /*moist_end=*/num_scalars);

  // Verify: ALL water-species values must be non-negative
  auto result_h = Kokkos::create_mirror_view(testData.state.scalars_new);
  Kokkos::deep_copy(result_h, testData.state.scalars_new);

  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k) {
      for (int s = 0; s < num_scalars; ++s) {
        RC_ASSERT(result_h(s, k, c) >= Scalar(0.0));
      }
    }
  }
}

/// Variant: Mixed water/non-water scalars. Only water species must be non-negative.
/// Non-water species have no such constraint.
RC_GTEST_PROP(ScalarTransportMonoPropertyWater,
              MixedScalarsOnlyWaterNonNegative,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 12: Water-species non-negativity
  // Generate mesh with at least 2 scalars so we can partition into water/non-water
  MeshParams params;
  params.nCells = *rc::gen::inRange(3, 10);
  params.nVertLevels = *rc::gen::inRange(3, 8);
  params.num_scalars = *rc::gen::inRange(2, 5);  // at least 2
  params.nEdges = params.nCells - 1;

  // Partition: first scalar is non-water, rest are water species
  const int moist_start = 1;
  const int moist_end = params.num_scalars;

  MonoPropertyTestData testData;
  testData.build(params);
  testData.fillRandom();

  const Scalar dt = Scalar(0.05);

  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      testData.mesh, testData.mono_mesh, testData.state,
      dt,
      /*coef_3rd_order=*/Scalar(1.0),
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/moist_start,
      /*moist_end=*/moist_end);

  // Verify: water-species (indices [moist_start, moist_end)) must be non-negative
  auto result_h = Kokkos::create_mirror_view(testData.state.scalars_new);
  Kokkos::deep_copy(result_h, testData.state.scalars_new);

  for (int c = 0; c < params.nCells; ++c) {
    for (int k = 0; k < params.nVertLevels; ++k) {
      for (int s = moist_start; s < moist_end; ++s) {
        RC_ASSERT(result_h(s, k, c) >= Scalar(0.0));
      }
    }
  }
}
