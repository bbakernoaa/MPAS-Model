/// @file test_scalar_transport_mono_property.cpp
/// @brief Property-based test: Monotonic limiter keeps scalars within local bounds.
///
/// Feature: mpas-dycore-cpp-port, Property 11: Monotonic limiter bounds
///
/// **Validates: Requirements 5.6**
///
/// For randomly generated scalar fields and mass fluxes, the monotonic limiter keeps
/// all updated scalars within the local min/max bounds (derived from surrounding cells),
/// to within the Parity_Tolerance on the host/CPU Execution_Space.
///
/// The property verifies that for any generated input:
/// 1. After monotonic transport, every cell's scalar value lies within
///    [local_min - tol, local_max + tol], where local_min/max are computed
///    from the cell and its neighbors in the pre-transport state.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include "mpas_dycore/scalar_transport_mono.hpp"
#include "mpas_dycore/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using Scalar = mpas::dycore::Scalar;
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using layout = Kokkos::LayoutLeft;

/// Parity_Tolerance: tight relative tolerance near machine precision.
constexpr Scalar kTol = std::is_same_v<Scalar, double> ? 1e-12 : 1e-6f;

template <class T>
using View1D = Kokkos::View<T*, layout, MemSpace>;
template <class T>
using View2D = Kokkos::View<T**, layout, MemSpace>;
template <class T>
using View3D = Kokkos::View<T***, layout, MemSpace>;

/// Parameters for a randomly generated monotonic transport test case.
struct MonoTestCase {
  int nCells;
  int nEdges;
  int nVertLevels;
  int num_scalars;

  // Connectivity: linear chain of cells connected by edges
  // cellsOnEdge(2, nEdges): for edge e, cell1=e, cell2=e+1 (0-based internally, stored 1-based)
  // cellsOnCell(maxEdges, nCells): neighbor list (1-based)
  // edgesOnCell(maxEdges, nCells): edges around each cell (1-based)
  // nEdgesOnCell(nCells): number of edges per cell

  // Scalar field values (pre-transport)
  std::vector<Scalar> scalar_values; // (num_scalars * nVertLevels * nCells)

  // Horizontal mass flux values
  std::vector<Scalar> uhAvg_values; // (nVertLevels * nEdges)

  // Vertical mass flux values
  std::vector<Scalar> wwAvg_values; // ((nVertLevels+1) * nCells)

  // Density
  std::vector<Scalar> rho_values; // (nVertLevels * nCells)

  // Timestep
  Scalar dt;

  // 3rd-order coefficient
  Scalar coef_3rd;
};

/// RapidCheck generator for MonoTestCase.
/// Creates a small linear mesh with random scalar values and fluxes.
MonoTestCase generateMonoTestCase() {
  MonoTestCase tc;

  // Small mesh dimensions to keep property tests fast
  tc.nCells = *rc::gen::inRange(3, 12);        // [3, 11]
  tc.nVertLevels = *rc::gen::inRange(2, 8);    // [2, 7]
  tc.num_scalars = *rc::gen::inRange(1, 4);    // [1, 3]
  tc.nEdges = tc.nCells - 1;  // Linear chain: nCells-1 internal edges

  // Random scalar values in a positive range [0.1, 10.0]
  // (positive values ensure meaningful min/max bounds)
  const int total_scalars = tc.num_scalars * tc.nVertLevels * tc.nCells;
  tc.scalar_values.resize(total_scalars);
  for (int i = 0; i < total_scalars; ++i) {
    int raw = *rc::gen::inRange(1, 1001); // [1, 1000]
    tc.scalar_values[i] = static_cast<Scalar>(raw) * Scalar(0.01); // [0.01, 10.0]
  }

  // Random horizontal mass fluxes in [-0.5, 0.5] (small to keep CFL < 1)
  const int total_uh = tc.nVertLevels * tc.nEdges;
  tc.uhAvg_values.resize(total_uh);
  for (int i = 0; i < total_uh; ++i) {
    int raw = *rc::gen::inRange(-50, 51); // [-50, 50]
    tc.uhAvg_values[i] = static_cast<Scalar>(raw) * Scalar(0.01); // [-0.5, 0.5]
  }

  // Random vertical mass fluxes in [-0.3, 0.3] (small)
  const int total_ww = (tc.nVertLevels + 1) * tc.nCells;
  tc.wwAvg_values.resize(total_ww);
  for (int i = 0; i < total_ww; ++i) {
    int raw = *rc::gen::inRange(-30, 31); // [-30, 30]
    tc.wwAvg_values[i] = static_cast<Scalar>(raw) * Scalar(0.01); // [-0.3, 0.3]
  }
  // Zero vertical flux at top and bottom boundaries
  for (int c = 0; c < tc.nCells; ++c) {
    tc.wwAvg_values[0 + (tc.nVertLevels + 1) * c] = Scalar(0.0);       // k=0
    tc.wwAvg_values[tc.nVertLevels + (tc.nVertLevels + 1) * c] = Scalar(0.0); // k=nVertLevels
  }

  // Random positive density in [0.5, 2.0]
  const int total_rho = tc.nVertLevels * tc.nCells;
  tc.rho_values.resize(total_rho);
  for (int i = 0; i < total_rho; ++i) {
    int raw = *rc::gen::inRange(50, 201); // [50, 200]
    tc.rho_values[i] = static_cast<Scalar>(raw) * Scalar(0.01); // [0.5, 2.0]
  }

  // Small timestep to maintain stability (CFL < 1)
  tc.dt = Scalar(0.05);

  // 3rd-order coefficient: standard value
  tc.coef_3rd = Scalar(1.0);

  return tc;
}

/// Compute local min/max bounds for each cell from the pre-transport scalar field.
/// For each cell, the local bounds include the cell itself, its vertical neighbors,
/// and its horizontal neighbors (via cellsOnCell connectivity).
/// This mirrors the bounds computation in the FCT limiter.
struct LocalBounds {
  std::vector<Scalar> s_min; // (nVertLevels * nCells)
  std::vector<Scalar> s_max; // (nVertLevels * nCells)
};

LocalBounds computeLocalBounds(
    const MonoTestCase& tc,
    int iScalar,
    const std::vector<Scalar>& scalar_field) {
  // scalar_field is in LayoutLeft order: (num_scalars, nVertLevels, nCells)
  // Access: scalar_field[s + num_scalars * (k + nVertLevels * c)]
  const int nv = tc.nVertLevels;
  const int nc = tc.nCells;
  const int ns = tc.num_scalars;

  auto get_scalar = [&](int s, int k, int c) -> Scalar {
    return scalar_field[s + ns * (k + nv * c)];
  };

  LocalBounds bounds;
  bounds.s_min.resize(nv * nc);
  bounds.s_max.resize(nv * nc);

  for (int c = 0; c < nc; ++c) {
    for (int k = 0; k < nv; ++k) {
      Scalar vmin = get_scalar(iScalar, k, c);
      Scalar vmax = vmin;

      // Vertical neighbors
      if (k > 0) {
        Scalar v = get_scalar(iScalar, k - 1, c);
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
      }
      if (k < nv - 1) {
        Scalar v = get_scalar(iScalar, k + 1, c);
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
      }

      // Horizontal neighbors (linear chain: cell c-1 and c+1)
      if (c > 0) {
        Scalar v = get_scalar(iScalar, k, c - 1);
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
      }
      if (c < nc - 1) {
        Scalar v = get_scalar(iScalar, k, c + 1);
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
      }

      bounds.s_min[k + nv * c] = vmin;
      bounds.s_max[k + nv * c] = vmax;
    }
  }

  return bounds;
}

/// Build Kokkos views and run the monotonic transport on a generated test case.
/// Returns the updated scalar values as a flat vector in LayoutLeft order.
std::vector<Scalar> runMonoTransport(const MonoTestCase& tc) {
  const int nCells = tc.nCells;
  const int nEdges = tc.nEdges;
  const int nVertLevels = tc.nVertLevels;
  const int num_scalars = tc.num_scalars;
  constexpr int maxEdges = 2;  // Linear chain: max 2 edges per cell
  constexpr int maxAdvCellsForEdge = 2;  // Simple 2-point stencil

  // Build mesh data
  mpas::dycore::ScalarTransportMeshData<ExecSpace> mesh;
  mesh.nCells = nCells;
  mesh.nEdges = nEdges;
  mesh.nVertLevels = nVertLevels;
  mesh.num_scalars = num_scalars;
  mesh.maxEdges = maxEdges;
  mesh.maxAdvCellsForEdge = maxAdvCellsForEdge;

  mpas::dycore::MonoTransportMeshData<ExecSpace> mono_mesh;
  mono_mesh.nCellsSolve = nCells;
  mono_mesh.maxEdges = maxEdges;

  // Allocate connectivity views
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

  // Linear chain connectivity: edge e connects cell e to cell e+1
  auto coe_h = Kokkos::create_mirror_view(mesh.cellsOnEdge);
  for (int e = 0; e < nEdges; ++e) {
    coe_h(0, e) = e + 1;     // cell1 (1-based)
    coe_h(1, e) = e + 2;     // cell2 (1-based)
  }
  Kokkos::deep_copy(mesh.cellsOnEdge, coe_h);

  // edgesOnCell and nEdgesOnCell for a linear chain
  auto eoc_h = Kokkos::create_mirror_view(mesh.edgesOnCell);
  auto nec_h = Kokkos::create_mirror_view(mesh.nEdgesOnCell);
  // Cell 0: 1 edge (edge 0 on the right)
  eoc_h(0, 0) = 1; nec_h(0) = 1;
  // Interior cells: 2 edges
  for (int c = 1; c < nCells - 1; ++c) {
    eoc_h(0, c) = c;       // left edge (1-based)
    eoc_h(1, c) = c + 1;   // right edge (1-based)
    nec_h(c) = 2;
  }
  // Cell nCells-1: 1 edge (last edge on the left)
  eoc_h(0, nCells - 1) = nEdges; nec_h(nCells - 1) = 1;
  Kokkos::deep_copy(mesh.edgesOnCell, eoc_h);
  Kokkos::deep_copy(mesh.nEdgesOnCell, nec_h);

  // Edge signs: sign = +1 for cell2 (outward), -1 for cell1 (inward)
  auto ecs_h = Kokkos::create_mirror_view(mesh.edgesOnCell_sign);
  // Cell 0: edge 0 points outward (+1)
  ecs_h(0, 0) = Scalar(1.0);
  // Interior cells: left edge inward (-1), right edge outward (+1)
  for (int c = 1; c < nCells - 1; ++c) {
    ecs_h(0, c) = Scalar(-1.0);  // left edge
    ecs_h(1, c) = Scalar(1.0);   // right edge
  }
  // Cell nCells-1: last edge inward (-1)
  ecs_h(0, nCells - 1) = Scalar(-1.0);
  Kokkos::deep_copy(mesh.edgesOnCell_sign, ecs_h);

  // cellsOnCell: horizontal neighbor connectivity (1-based)
  auto coc_h = Kokkos::create_mirror_view(mono_mesh.cellsOnCell);
  // Cell 0: neighbor is cell 1
  coc_h(0, 0) = 2;
  // Interior cells: two neighbors
  for (int c = 1; c < nCells - 1; ++c) {
    coc_h(0, c) = c;       // left neighbor (1-based)
    coc_h(1, c) = c + 2;   // right neighbor (1-based)
  }
  // Cell nCells-1: neighbor is cell nCells-2
  coc_h(0, nCells - 1) = nCells - 1;
  Kokkos::deep_copy(mono_mesh.cellsOnCell, coc_h);

  // Advection stencil: simple 2-point average (upwind-biased with coef_3rd)
  auto acfe_h = Kokkos::create_mirror_view(mesh.advCellsForEdge);
  auto nacfe_h = Kokkos::create_mirror_view(mesh.nAdvCellsForEdge);
  auto ac_h = Kokkos::create_mirror_view(mesh.adv_coefs);
  auto ac3_h = Kokkos::create_mirror_view(mesh.adv_coefs_3rd);
  for (int e = 0; e < nEdges; ++e) {
    nacfe_h(e) = 2;
    acfe_h(0, e) = e + 1;     // cell1 (1-based)
    acfe_h(1, e) = e + 2;     // cell2 (1-based)
    ac_h(0, e) = Scalar(0.5);
    ac_h(1, e) = Scalar(0.5);
    ac3_h(0, e) = Scalar(0.5);
    ac3_h(1, e) = Scalar(-0.5);
  }
  Kokkos::deep_copy(mesh.advCellsForEdge, acfe_h);
  Kokkos::deep_copy(mesh.nAdvCellsForEdge, nacfe_h);
  Kokkos::deep_copy(mesh.adv_coefs, ac_h);
  Kokkos::deep_copy(mesh.adv_coefs_3rd, ac3_h);

  // Uniform geometry
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

  // State fields
  mpas::dycore::MonoTransportState<ExecSpace> state;
  state.scalars_old = View3D<Scalar>("scalars_old", num_scalars, nVertLevels, nCells);
  state.scalars_new = View3D<Scalar>("scalars_new", num_scalars, nVertLevels, nCells);
  state.scalar_tend = View3D<Scalar>("scalar_tend", num_scalars, nVertLevels, nCells);
  state.rho_zz_old = View2D<Scalar>("rho_zz_old", nVertLevels, nCells);
  state.rho_zz_new = View2D<Scalar>("rho_zz_new", nVertLevels, nCells);
  state.uhAvg = View2D<Scalar>("uhAvg", nVertLevels, nEdges);
  state.wwAvg = View2D<Scalar>("wwAvg", nVertLevels + 1, nCells);

  // Fill scalar values (LayoutLeft: scalar_old(s, k, c))
  {
    auto so_h = Kokkos::create_mirror_view(state.scalars_old);
    auto sn_h = Kokkos::create_mirror_view(state.scalars_new);
    for (int c = 0; c < nCells; ++c) {
      for (int k = 0; k < nVertLevels; ++k) {
        for (int s = 0; s < num_scalars; ++s) {
          Scalar val = tc.scalar_values[s + num_scalars * (k + nVertLevels * c)];
          so_h(s, k, c) = val;
          sn_h(s, k, c) = val;
        }
      }
    }
    Kokkos::deep_copy(state.scalars_old, so_h);
    Kokkos::deep_copy(state.scalars_new, sn_h);
  }

  // Fill density (LayoutLeft: rho(k, c))
  {
    auto rho_h = Kokkos::create_mirror_view(state.rho_zz_old);
    auto rho_new_h = Kokkos::create_mirror_view(state.rho_zz_new);
    for (int c = 0; c < nCells; ++c) {
      for (int k = 0; k < nVertLevels; ++k) {
        Scalar val = tc.rho_values[k + nVertLevels * c];
        rho_h(k, c) = val;
        rho_new_h(k, c) = val;
      }
    }
    Kokkos::deep_copy(state.rho_zz_old, rho_h);
    Kokkos::deep_copy(state.rho_zz_new, rho_new_h);
  }

  // Fill horizontal mass flux (LayoutLeft: uhAvg(k, e))
  {
    auto uh_h = Kokkos::create_mirror_view(state.uhAvg);
    for (int e = 0; e < nEdges; ++e) {
      for (int k = 0; k < nVertLevels; ++k) {
        uh_h(k, e) = tc.uhAvg_values[k + nVertLevels * e];
      }
    }
    Kokkos::deep_copy(state.uhAvg, uh_h);
  }

  // Fill vertical mass flux (LayoutLeft: wwAvg(k, c))
  {
    auto ww_h = Kokkos::create_mirror_view(state.wwAvg);
    for (int c = 0; c < nCells; ++c) {
      for (int k = 0; k <= nVertLevels; ++k) {
        ww_h(k, c) = tc.wwAvg_values[k + (nVertLevels + 1) * c];
      }
    }
    Kokkos::deep_copy(state.wwAvg, ww_h);
  }

  // Zero tendencies
  Kokkos::deep_copy(state.scalar_tend, Scalar(0.0));

  // Run monotonic transport
  mpas::dycore::Scalar_Transport_Mono<ExecSpace> mono;
  mono.advance_scalars_mono(
      mesh, mono_mesh, state,
      tc.dt,
      tc.coef_3rd,
      /*advance_density=*/true,
      /*config_apply_lbcs=*/false,
      /*moist_start=*/0,
      /*moist_end=*/num_scalars);

  // Extract result
  auto result_h = Kokkos::create_mirror_view(state.scalars_new);
  Kokkos::deep_copy(result_h, state.scalars_new);

  std::vector<Scalar> result(num_scalars * nVertLevels * nCells);
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nVertLevels; ++k) {
      for (int s = 0; s < num_scalars; ++s) {
        result[s + num_scalars * (k + nVertLevels * c)] = result_h(s, k, c);
      }
    }
  }

  return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Property 11: Monotonic limiter bounds
// Feature: mpas-dycore-cpp-port, Property 11: Monotonic limiter bounds
// ---------------------------------------------------------------------------

/// For randomly generated scalar fields and mass fluxes on a linear mesh,
/// the monotonic limiter keeps all updated scalars within local min/max bounds
/// (within Parity_Tolerance).
RC_GTEST_PROP(MonotonicLimiterProperty,
              LimiterKeepsScalarsWithinLocalBounds,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 11: Monotonic limiter bounds
  auto tc = generateMonoTestCase();

  // Compute local bounds from the pre-transport scalar field
  // (these are the bounds the limiter should enforce)

  // Run the monotonic transport
  auto result = runMonoTransport(tc);

  const int nCells = tc.nCells;
  const int nVertLevels = tc.nVertLevels;
  const int num_scalars = tc.num_scalars;

  // Verify each scalar, level, cell is within local bounds
  for (int s = 0; s < num_scalars; ++s) {
    auto bounds = computeLocalBounds(tc, s, tc.scalar_values);

    for (int c = 0; c < nCells; ++c) {
      for (int k = 0; k < nVertLevels; ++k) {
        Scalar updated = result[s + num_scalars * (k + nVertLevels * c)];
        Scalar local_min = bounds.s_min[k + nVertLevels * c];
        Scalar local_max = bounds.s_max[k + nVertLevels * c];

        // The monotonic limiter guarantees updated value is within
        // [local_min, local_max] to within tolerance.
        // Note: positive-definite enforcement (clamp to 0) means value could
        // be 0 if local_min > 0 won't happen (our inputs are positive).
        RC_ASSERT(updated >= local_min - kTol);
        RC_ASSERT(updated <= local_max + kTol);
      }
    }
  }
}

/// Sub-property: For a uniform scalar field with zero fluxes,
/// the monotonic transport preserves the scalar exactly.
RC_GTEST_PROP(MonotonicLimiterProperty,
              UniformFieldPreservedExactly,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 11: Monotonic limiter bounds
  MonoTestCase tc;

  tc.nCells = *rc::gen::inRange(3, 12);
  tc.nVertLevels = *rc::gen::inRange(2, 8);
  tc.num_scalars = *rc::gen::inRange(1, 4);
  tc.nEdges = tc.nCells - 1;

  // Generate a single uniform scalar value
  int raw_val = *rc::gen::inRange(1, 1001);
  Scalar uniform_val = static_cast<Scalar>(raw_val) * Scalar(0.01);

  const int total = tc.num_scalars * tc.nVertLevels * tc.nCells;
  tc.scalar_values.resize(total, uniform_val);

  // Zero fluxes
  tc.uhAvg_values.resize(tc.nVertLevels * tc.nEdges, Scalar(0.0));
  tc.wwAvg_values.resize((tc.nVertLevels + 1) * tc.nCells, Scalar(0.0));

  // Uniform density
  tc.rho_values.resize(tc.nVertLevels * tc.nCells, Scalar(1.0));

  tc.dt = Scalar(0.05);
  tc.coef_3rd = Scalar(1.0);

  auto result = runMonoTransport(tc);

  // With zero fluxes, scalar should remain uniform
  for (int i = 0; i < total; ++i) {
    RC_ASSERT(std::abs(result[i] - uniform_val) <= kTol);
  }
}

