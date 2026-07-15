/// @file test_dry_mass_conservation_property.cpp
/// @brief Property-based test: Dry-mass conservation.
///
/// Feature: mpas-dycore-cpp-port, Property 13: Dry-mass conservation
///
/// **Validates: Requirements 3.14**
///
/// For randomly generated mesh configurations with valid density fields, the
/// compute_total_dry_air_mass function produces non-negative results that scale
/// linearly with density (i.e., mass(2*rho) == 2*mass(rho)).
///
/// This verifies that the mass computation is a proper linear integral over the
/// density field, which is the fundamental prerequisite for the conservation
/// diagnostic to detect mass non-conservation after a timestep.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include "mpas_dycore/post_timestep_diagnostics.hpp"
#include "mpas_dycore/scalar.hpp"

#include <cmath>

namespace {

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using Scalar = mpas::dycore::Scalar;
using layout = Kokkos::LayoutLeft;
using View1D = Kokkos::View<Scalar*, layout, MemSpace>;
using View2D = Kokkos::View<Scalar**, layout, MemSpace>;

/// Parity tolerance for floating-point comparisons.
constexpr Scalar parity_tolerance = std::is_same_v<Scalar, double> ? 1e-12 : 1e-6f;

}  // namespace

// ---------------------------------------------------------------------------
// Property 13: Dry-mass conservation — Non-negativity
// Feature: mpas-dycore-cpp-port, Property 13: Dry-mass conservation
// ---------------------------------------------------------------------------

/// For any valid (non-negative) density field and positive-volume mesh, the
/// computed total dry air mass is non-negative.
RC_GTEST_PROP(DryMassConservationProperty,
              TotalMassIsNonNegativeForValidDensity,
              ()) {
  // Generate random mesh dimensions
  const int nVertLevels = *rc::gen::inRange(1, 21);  // [1, 20]
  const int nCells = *rc::gen::inRange(1, 31);       // [1, 30]

  // Allocate views
  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  auto host_rho = Kokkos::create_mirror_view(rho_zz);
  auto host_zz = Kokkos::create_mirror_view(zz);
  auto host_area = Kokkos::create_mirror_view(areaCell);
  auto host_zgrid = Kokkos::create_mirror_view(zgrid);

  // Generate positive density values (physically valid: 0.1 to 2.0 kg/m^3)
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      int raw = *rc::gen::inRange(1, 2001);  // [1, 2000]
      host_rho(k, c) = static_cast<Scalar>(raw) * 0.001;  // [0.001, 2.0]
    }
  }

  // Generate positive Jacobian values (physical range: 0.5 to 2.0)
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      int raw = *rc::gen::inRange(500, 2001);  // [500, 2000]
      host_zz(k, c) = static_cast<Scalar>(raw) * 0.001;  // [0.5, 2.0]
    }
  }

  // Generate positive cell areas (100 to 100000 m^2)
  for (int c = 0; c < nCells; ++c) {
    int raw = *rc::gen::inRange(100, 100001);
    host_area(c) = static_cast<Scalar>(raw);
  }

  // Generate monotonically increasing vertical grid (positive layer thickness)
  for (int c = 0; c < nCells; ++c) {
    host_zgrid(0, c) = 0.0;
    for (int k = 1; k <= nVertLevels; ++k) {
      int dz_raw = *rc::gen::inRange(100, 3001);  // dz in [100, 3000] meters
      host_zgrid(k, c) = host_zgrid(k - 1, c) + static_cast<Scalar>(dz_raw);
    }
  }

  Kokkos::deep_copy(rho_zz, host_rho);
  Kokkos::deep_copy(zz, host_zz);
  Kokkos::deep_copy(areaCell, host_area);
  Kokkos::deep_copy(zgrid, host_zgrid);

  Scalar mass = mpas::dycore::compute_total_dry_air_mass<ExecSpace>(
      rho_zz, zz, areaCell, zgrid, nVertLevels, nCells);

  // Total mass must be non-negative for valid (positive) density fields
  RC_ASSERT(mass >= Scalar(0.0));
  // Actually, with strictly positive density, Jacobian, area, and thickness,
  // mass must be strictly positive
  RC_ASSERT(mass > Scalar(0.0));
}

// ---------------------------------------------------------------------------
// Property 13: Dry-mass conservation — Linear scaling with density
// Feature: mpas-dycore-cpp-port, Property 13: Dry-mass conservation
// ---------------------------------------------------------------------------

/// For any valid mesh configuration and density field, scaling density by a
/// factor alpha scales the computed total dry air mass by the same factor:
///   mass(alpha * rho) == alpha * mass(rho)
///
/// This linearity property is essential for the conservation diagnostic:
/// if mass(rho) is linear in rho, then mass equality before and after a
/// timestep verifies that no spurious mass was created or destroyed.
RC_GTEST_PROP(DryMassConservationProperty,
              MassScalesLinearlyWithDensity,
              ()) {
  // Generate random mesh dimensions
  const int nVertLevels = *rc::gen::inRange(1, 16);  // [1, 15]
  const int nCells = *rc::gen::inRange(1, 21);       // [1, 20]

  // Generate a positive scale factor (avoid 0 to keep mass non-trivial)
  int alpha_raw = *rc::gen::inRange(1, 1001);  // [1, 1000]
  const Scalar alpha = static_cast<Scalar>(alpha_raw) * 0.01;  // [0.01, 10.0]

  // Allocate views for the base density
  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D rho_zz_scaled("rho_zz_scaled", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  auto host_rho = Kokkos::create_mirror_view(rho_zz);
  auto host_rho_scaled = Kokkos::create_mirror_view(rho_zz_scaled);
  auto host_zz = Kokkos::create_mirror_view(zz);
  auto host_area = Kokkos::create_mirror_view(areaCell);
  auto host_zgrid = Kokkos::create_mirror_view(zgrid);

  // Generate positive density values
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      int raw = *rc::gen::inRange(1, 2001);
      Scalar rho_val = static_cast<Scalar>(raw) * 0.001;
      host_rho(k, c) = rho_val;
      host_rho_scaled(k, c) = alpha * rho_val;
    }
  }

  // Generate positive Jacobian values
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      int raw = *rc::gen::inRange(500, 2001);
      host_zz(k, c) = static_cast<Scalar>(raw) * 0.001;
    }
  }

  // Generate positive cell areas
  for (int c = 0; c < nCells; ++c) {
    int raw = *rc::gen::inRange(100, 100001);
    host_area(c) = static_cast<Scalar>(raw);
  }

  // Generate monotonically increasing vertical grid
  for (int c = 0; c < nCells; ++c) {
    host_zgrid(0, c) = 0.0;
    for (int k = 1; k <= nVertLevels; ++k) {
      int dz_raw = *rc::gen::inRange(100, 3001);
      host_zgrid(k, c) = host_zgrid(k - 1, c) + static_cast<Scalar>(dz_raw);
    }
  }

  Kokkos::deep_copy(rho_zz, host_rho);
  Kokkos::deep_copy(rho_zz_scaled, host_rho_scaled);
  Kokkos::deep_copy(zz, host_zz);
  Kokkos::deep_copy(areaCell, host_area);
  Kokkos::deep_copy(zgrid, host_zgrid);

  // Compute mass for base density
  Scalar mass_base = mpas::dycore::compute_total_dry_air_mass<ExecSpace>(
      rho_zz, zz, areaCell, zgrid, nVertLevels, nCells);

  // Compute mass for scaled density
  Scalar mass_scaled = mpas::dycore::compute_total_dry_air_mass<ExecSpace>(
      rho_zz_scaled, zz, areaCell, zgrid, nVertLevels, nCells);

  // Assert linearity: mass(alpha * rho) == alpha * mass(rho)
  Scalar expected = alpha * mass_base;
  Scalar rel_err = std::abs(mass_scaled - expected) /
                   std::max(Scalar(1.0), std::abs(expected));
  RC_ASSERT(rel_err <= parity_tolerance);
}

// ---------------------------------------------------------------------------
// Property 13: Dry-mass conservation — Additivity (superposition)
// Feature: mpas-dycore-cpp-port, Property 13: Dry-mass conservation
// ---------------------------------------------------------------------------

/// For any two valid density fields rho_a and rho_b on the same mesh:
///   mass(rho_a + rho_b) == mass(rho_a) + mass(rho_b)
///
/// This additivity property, together with linearity, confirms that
/// compute_total_dry_air_mass is a true linear functional of the density.
RC_GTEST_PROP(DryMassConservationProperty,
              MassIsAdditiveOverDensity,
              ()) {
  const int nVertLevels = *rc::gen::inRange(1, 12);
  const int nCells = *rc::gen::inRange(1, 16);

  View2D rho_a("rho_a", nVertLevels, nCells);
  View2D rho_b("rho_b", nVertLevels, nCells);
  View2D rho_sum("rho_sum", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  auto host_rho_a = Kokkos::create_mirror_view(rho_a);
  auto host_rho_b = Kokkos::create_mirror_view(rho_b);
  auto host_rho_sum = Kokkos::create_mirror_view(rho_sum);
  auto host_zz = Kokkos::create_mirror_view(zz);
  auto host_area = Kokkos::create_mirror_view(areaCell);
  auto host_zgrid = Kokkos::create_mirror_view(zgrid);

  // Generate two independent positive density fields
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      int raw_a = *rc::gen::inRange(1, 1001);
      int raw_b = *rc::gen::inRange(1, 1001);
      host_rho_a(k, c) = static_cast<Scalar>(raw_a) * 0.001;
      host_rho_b(k, c) = static_cast<Scalar>(raw_b) * 0.001;
      host_rho_sum(k, c) = host_rho_a(k, c) + host_rho_b(k, c);
    }
  }

  // Generate mesh (same as other tests)
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      int raw = *rc::gen::inRange(500, 2001);
      host_zz(k, c) = static_cast<Scalar>(raw) * 0.001;
    }
  }
  for (int c = 0; c < nCells; ++c) {
    int raw = *rc::gen::inRange(100, 50001);
    host_area(c) = static_cast<Scalar>(raw);
  }
  for (int c = 0; c < nCells; ++c) {
    host_zgrid(0, c) = 0.0;
    for (int k = 1; k <= nVertLevels; ++k) {
      int dz_raw = *rc::gen::inRange(100, 2001);
      host_zgrid(k, c) = host_zgrid(k - 1, c) + static_cast<Scalar>(dz_raw);
    }
  }

  Kokkos::deep_copy(rho_a, host_rho_a);
  Kokkos::deep_copy(rho_b, host_rho_b);
  Kokkos::deep_copy(rho_sum, host_rho_sum);
  Kokkos::deep_copy(zz, host_zz);
  Kokkos::deep_copy(areaCell, host_area);
  Kokkos::deep_copy(zgrid, host_zgrid);

  Scalar mass_a = mpas::dycore::compute_total_dry_air_mass<ExecSpace>(
      rho_a, zz, areaCell, zgrid, nVertLevels, nCells);
  Scalar mass_b = mpas::dycore::compute_total_dry_air_mass<ExecSpace>(
      rho_b, zz, areaCell, zgrid, nVertLevels, nCells);
  Scalar mass_sum = mpas::dycore::compute_total_dry_air_mass<ExecSpace>(
      rho_sum, zz, areaCell, zgrid, nVertLevels, nCells);

  // Assert: mass(rho_a + rho_b) == mass(rho_a) + mass(rho_b)
  Scalar expected = mass_a + mass_b;
  Scalar rel_err = std::abs(mass_sum - expected) /
                   std::max(Scalar(1.0), std::abs(expected));
  RC_ASSERT(rel_err <= parity_tolerance);
}
