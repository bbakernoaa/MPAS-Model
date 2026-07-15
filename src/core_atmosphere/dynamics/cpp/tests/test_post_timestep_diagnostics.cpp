#include <gtest/gtest.h>

#include "mpas_dycore/post_timestep_diagnostics.hpp"
#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <limits>

namespace mpas {
namespace dycore {
namespace {

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using layout = Kokkos::LayoutLeft;
using View1D = Kokkos::View<Scalar*, layout, MemSpace>;
using View2D = Kokkos::View<Scalar**, layout, MemSpace>;

// ════════════════════════════════════════════════════════════════════════════
// NaN Guard Tests (Requirement 12.5)
// ════════════════════════════════════════════════════════════════════════════

TEST(PostTimestepDiagnosticsTest, NaNGuardPassesCleanField) {
  const int nLevels = 5;
  const int nElements = 10;

  View2D field("clean_field", nLevels, nElements);
  Kokkos::deep_copy(field, 1.0);

  // Should not throw for a clean field
  EXPECT_NO_THROW(
      check_field_for_nan<ExecSpace>(field, "test_field", nLevels, nElements));
}

TEST(PostTimestepDiagnosticsTest, NaNGuardDetectsNaN) {
  const int nLevels = 5;
  const int nElements = 10;

  View2D field("nan_field", nLevels, nElements);
  Kokkos::deep_copy(field, 1.0);

  // Inject a NaN at a specific location
  auto host_field = Kokkos::create_mirror_view(field);
  Kokkos::deep_copy(host_field, field);
  host_field(2, 5) = std::numeric_limits<Scalar>::quiet_NaN();
  Kokkos::deep_copy(field, host_field);

  // Should throw NaNDetectedError identifying the field
  EXPECT_THROW(
      check_field_for_nan<ExecSpace>(field, "u", nLevels, nElements),
      NaNDetectedError);

  try {
    check_field_for_nan<ExecSpace>(field, "u", nLevels, nElements);
  } catch (const NaNDetectedError& e) {
    EXPECT_EQ(e.field_name(), "u");
    std::string msg = e.what();
    EXPECT_NE(msg.find("u"), std::string::npos);
  }
}

TEST(PostTimestepDiagnosticsTest, NaNGuardDetectsNaNInDifferentFields) {
  const int nLevels = 3;
  const int nElements = 4;

  View2D field("theta_nan", nLevels, nElements);
  Kokkos::deep_copy(field, 300.0);

  auto host_field = Kokkos::create_mirror_view(field);
  Kokkos::deep_copy(host_field, field);
  host_field(0, 0) = std::numeric_limits<Scalar>::quiet_NaN();
  Kokkos::deep_copy(field, host_field);

  try {
    check_field_for_nan<ExecSpace>(field, "theta_m", nLevels, nElements);
    FAIL() << "Expected NaNDetectedError";
  } catch (const NaNDetectedError& e) {
    EXPECT_EQ(e.field_name(), "theta_m");
  }
}

TEST(PostTimestepDiagnosticsTest, NaNGuardHandlesEmptyField) {
  // Zero-size field should not trigger NaN detection
  View2D field("empty_field", 0, 0);
  EXPECT_NO_THROW(
      check_field_for_nan<ExecSpace>(field, "empty", 0, 0));
}

// ════════════════════════════════════════════════════════════════════════════
// Global Min/Max Tests (Requirement 12.4)
// ════════════════════════════════════════════════════════════════════════════

TEST(PostTimestepDiagnosticsTest, MinMaxConstantField) {
  const int nLevels = 5;
  const int nElements = 10;

  View2D field("const_field", nLevels, nElements);
  Kokkos::deep_copy(field, 42.0);

  auto result = compute_field_min_max<ExecSpace>(field, nLevels, nElements);
  EXPECT_NEAR(result.min_val, 42.0, 1e-14);
  EXPECT_NEAR(result.max_val, 42.0, 1e-14);
}

TEST(PostTimestepDiagnosticsTest, MinMaxVaryingField) {
  const int nLevels = 4;
  const int nElements = 6;

  View2D field("vary_field", nLevels, nElements);
  auto host_field = Kokkos::create_mirror_view(field);

  // Fill with varying values
  for (int k = 0; k < nLevels; ++k) {
    for (int i = 0; i < nElements; ++i) {
      host_field(k, i) = static_cast<Scalar>(k * nElements + i) - 10.0;
    }
  }
  Kokkos::deep_copy(field, host_field);

  auto result = compute_field_min_max<ExecSpace>(field, nLevels, nElements);

  // Min is at (0,0) = 0*6 + 0 - 10 = -10
  // Max is at (3,5) = 3*6 + 5 - 10 = 13
  EXPECT_NEAR(result.min_val, -10.0, 1e-14);
  EXPECT_NEAR(result.max_val, 13.0, 1e-14);
}

TEST(PostTimestepDiagnosticsTest, MinMaxWithNegativeValues) {
  const int nLevels = 3;
  const int nElements = 3;

  View2D field("neg_field", nLevels, nElements);
  auto host_field = Kokkos::create_mirror_view(field);

  // Put extremes in specific locations
  Kokkos::deep_copy(host_field, 0.0);
  host_field(1, 1) = -999.5;
  host_field(2, 0) = 500.25;
  Kokkos::deep_copy(field, host_field);

  auto result = compute_field_min_max<ExecSpace>(field, nLevels, nElements);
  EXPECT_NEAR(result.min_val, -999.5, 1e-14);
  EXPECT_NEAR(result.max_val, 500.25, 1e-14);
}

// ════════════════════════════════════════════════════════════════════════════
// Dry-Air Mass Conservation Tests (Requirement 12.3)
// ════════════════════════════════════════════════════════════════════════════

TEST(PostTimestepDiagnosticsTest, DryAirMassUniformDensity) {
  // For uniform rho_zz=1, zz=1, uniform area, and uniform layer thickness,
  // total mass = rho_zz * zz * dz * area * nLevels * nCells
  const int nVertLevels = 3;
  const int nCells = 4;

  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  Kokkos::deep_copy(rho_zz, 1.0);
  Kokkos::deep_copy(zz, 1.0);
  Kokkos::deep_copy(areaCell, 100.0);

  // Set up zgrid with uniform layer thickness = 1000 m
  auto host_zgrid = Kokkos::create_mirror_view(zgrid);
  for (int iCell = 0; iCell < nCells; ++iCell) {
    for (int k = 0; k <= nVertLevels; ++k) {
      host_zgrid(k, iCell) = static_cast<Scalar>(k) * 1000.0;
    }
  }
  Kokkos::deep_copy(zgrid, host_zgrid);

  Scalar mass = compute_total_dry_air_mass<ExecSpace>(
      rho_zz, zz, areaCell, zgrid, nVertLevels, nCells);

  // Expected: 1.0 * 1.0 * 1000.0 * 100.0 * 3 levels * 4 cells = 1,200,000
  EXPECT_NEAR(mass, 1200000.0, 1e-8);
}

TEST(PostTimestepDiagnosticsTest, DryAirMassNonUniformGrid) {
  // Non-uniform vertical grid: test that layer thickness is correctly derived
  const int nVertLevels = 2;
  const int nCells = 2;

  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  auto host_rho = Kokkos::create_mirror_view(rho_zz);
  auto host_zz = Kokkos::create_mirror_view(zz);
  auto host_area = Kokkos::create_mirror_view(areaCell);
  auto host_zgrid = Kokkos::create_mirror_view(zgrid);

  // rho_zz = 1.2 everywhere
  for (int k = 0; k < nVertLevels; ++k)
    for (int i = 0; i < nCells; ++i)
      host_rho(k, i) = 1.2;

  // zz = 1.0 (identity Jacobian)
  for (int k = 0; k < nVertLevels; ++k)
    for (int i = 0; i < nCells; ++i)
      host_zz(k, i) = 1.0;

  // areaCell: cell 0 = 50, cell 1 = 100
  host_area(0) = 50.0;
  host_area(1) = 100.0;

  // zgrid: layer 0 = [0,500], layer 1 = [500, 1500]
  // dz(0) = 500, dz(1) = 1000
  for (int i = 0; i < nCells; ++i) {
    host_zgrid(0, i) = 0.0;
    host_zgrid(1, i) = 500.0;
    host_zgrid(2, i) = 1500.0;
  }

  Kokkos::deep_copy(rho_zz, host_rho);
  Kokkos::deep_copy(zz, host_zz);
  Kokkos::deep_copy(areaCell, host_area);
  Kokkos::deep_copy(zgrid, host_zgrid);

  Scalar mass = compute_total_dry_air_mass<ExecSpace>(
      rho_zz, zz, areaCell, zgrid, nVertLevels, nCells);

  // Cell 0: 1.2 * 1.0 * (500*50 + 1000*50) = 1.2 * 75000 = 90000
  // Cell 1: 1.2 * 1.0 * (500*100 + 1000*100) = 1.2 * 150000 = 180000
  // Total: 270000
  EXPECT_NEAR(mass, 270000.0, 1e-8);
}

TEST(PostTimestepDiagnosticsTest, DryAirMassWithJacobian) {
  // Test that the Jacobian (zz) is properly accounted for
  const int nVertLevels = 1;
  const int nCells = 1;

  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  auto host_rho = Kokkos::create_mirror_view(rho_zz);
  auto host_zz = Kokkos::create_mirror_view(zz);
  auto host_area = Kokkos::create_mirror_view(areaCell);
  auto host_zgrid = Kokkos::create_mirror_view(zgrid);

  host_rho(0, 0) = 2.0;
  host_zz(0, 0) = 0.5;  // Jacobian = 0.5
  host_area(0) = 200.0;
  host_zgrid(0, 0) = 0.0;
  host_zgrid(1, 0) = 1000.0;  // dz = 1000

  Kokkos::deep_copy(rho_zz, host_rho);
  Kokkos::deep_copy(zz, host_zz);
  Kokkos::deep_copy(areaCell, host_area);
  Kokkos::deep_copy(zgrid, host_zgrid);

  Scalar mass = compute_total_dry_air_mass<ExecSpace>(
      rho_zz, zz, areaCell, zgrid, nVertLevels, nCells);

  // Expected: 2.0 * 0.5 * 1000.0 * 200.0 = 200000
  EXPECT_NEAR(mass, 200000.0, 1e-8);
}

// ════════════════════════════════════════════════════════════════════════════
// Integration: run_post_timestep_diagnostics (all three in one call)
// ════════════════════════════════════════════════════════════════════════════

TEST(PostTimestepDiagnosticsTest, RunAllDiagnosticsCleanState) {
  const int nVertLevels = 4;
  const int nCells = 6;
  const int nEdges = 10;

  View2D u("u", nVertLevels, nEdges);
  View2D w("w", nVertLevels + 1, nCells);
  View2D theta_m("theta_m", nVertLevels, nCells);
  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  // Fill with representative atmospheric values
  Kokkos::deep_copy(u, 10.0);       // 10 m/s
  Kokkos::deep_copy(w, 0.1);        // 0.1 m/s vertical
  Kokkos::deep_copy(theta_m, 300.0); // 300 K
  Kokkos::deep_copy(rho_zz, 1.1);   // ~1.1 kg/m^3
  Kokkos::deep_copy(zz, 1.0);
  Kokkos::deep_copy(areaCell, 1000.0);

  auto host_zgrid = Kokkos::create_mirror_view(zgrid);
  for (int iCell = 0; iCell < nCells; ++iCell) {
    for (int k = 0; k <= nVertLevels; ++k) {
      host_zgrid(k, iCell) = static_cast<Scalar>(k) * 2500.0;
    }
  }
  Kokkos::deep_copy(zgrid, host_zgrid);

  PostTimestepDiagnostics result;
  EXPECT_NO_THROW(
      result = run_post_timestep_diagnostics<ExecSpace>(
          u, w, theta_m, rho_zz, zz, areaCell, zgrid,
          nVertLevels, nCells, nEdges));

  // Check min/max for w (all values are 0.1)
  EXPECT_NEAR(result.w_min_max.min_val, 0.1, 1e-14);
  EXPECT_NEAR(result.w_min_max.max_val, 0.1, 1e-14);

  // Check min/max for u (all values are 10.0)
  EXPECT_NEAR(result.u_min_max.min_val, 10.0, 1e-14);
  EXPECT_NEAR(result.u_min_max.max_val, 10.0, 1e-14);

  // Check mass: 1.1 * 1.0 * 2500.0 * 1000.0 * 4 * 6 = 66,000,000
  EXPECT_NEAR(result.total_dry_air_mass, 66000000.0, 1e-4);
}

TEST(PostTimestepDiagnosticsTest, RunAllDiagnosticsNaNInU) {
  const int nVertLevels = 3;
  const int nCells = 4;
  const int nEdges = 8;

  View2D u("u", nVertLevels, nEdges);
  View2D w("w", nVertLevels + 1, nCells);
  View2D theta_m("theta_m", nVertLevels, nCells);
  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  Kokkos::deep_copy(u, 5.0);
  Kokkos::deep_copy(w, 0.01);
  Kokkos::deep_copy(theta_m, 290.0);
  Kokkos::deep_copy(rho_zz, 1.0);
  Kokkos::deep_copy(zz, 1.0);
  Kokkos::deep_copy(areaCell, 500.0);

  auto host_zgrid = Kokkos::create_mirror_view(zgrid);
  for (int i = 0; i < nCells; ++i)
    for (int k = 0; k <= nVertLevels; ++k)
      host_zgrid(k, i) = static_cast<Scalar>(k) * 1000.0;
  Kokkos::deep_copy(zgrid, host_zgrid);

  // Inject NaN into u
  auto host_u = Kokkos::create_mirror_view(u);
  Kokkos::deep_copy(host_u, u);
  host_u(1, 3) = std::numeric_limits<Scalar>::quiet_NaN();
  Kokkos::deep_copy(u, host_u);

  EXPECT_THROW(
      run_post_timestep_diagnostics<ExecSpace>(
          u, w, theta_m, rho_zz, zz, areaCell, zgrid,
          nVertLevels, nCells, nEdges),
      NaNDetectedError);
}

TEST(PostTimestepDiagnosticsTest, RunAllDiagnosticsNaNInW) {
  const int nVertLevels = 3;
  const int nCells = 4;
  const int nEdges = 8;

  View2D u("u", nVertLevels, nEdges);
  View2D w("w", nVertLevels + 1, nCells);
  View2D theta_m("theta_m", nVertLevels, nCells);
  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  Kokkos::deep_copy(u, 5.0);
  Kokkos::deep_copy(w, 0.01);
  Kokkos::deep_copy(theta_m, 290.0);
  Kokkos::deep_copy(rho_zz, 1.0);
  Kokkos::deep_copy(zz, 1.0);
  Kokkos::deep_copy(areaCell, 500.0);

  auto host_zgrid = Kokkos::create_mirror_view(zgrid);
  for (int i = 0; i < nCells; ++i)
    for (int k = 0; k <= nVertLevels; ++k)
      host_zgrid(k, i) = static_cast<Scalar>(k) * 1000.0;
  Kokkos::deep_copy(zgrid, host_zgrid);

  // Inject NaN into w (u is clean, so u passes; w should fail)
  auto host_w = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(host_w, w);
  host_w(2, 1) = std::numeric_limits<Scalar>::quiet_NaN();
  Kokkos::deep_copy(w, host_w);

  try {
    run_post_timestep_diagnostics<ExecSpace>(
        u, w, theta_m, rho_zz, zz, areaCell, zgrid,
        nVertLevels, nCells, nEdges);
    FAIL() << "Expected NaNDetectedError for w field";
  } catch (const NaNDetectedError& e) {
    EXPECT_EQ(e.field_name(), "w");
  }
}

TEST(PostTimestepDiagnosticsTest, RunAllDiagnosticsNaNInRhoZZ) {
  const int nVertLevels = 3;
  const int nCells = 4;
  const int nEdges = 8;

  View2D u("u", nVertLevels, nEdges);
  View2D w("w", nVertLevels + 1, nCells);
  View2D theta_m("theta_m", nVertLevels, nCells);
  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  Kokkos::deep_copy(u, 5.0);
  Kokkos::deep_copy(w, 0.01);
  Kokkos::deep_copy(theta_m, 290.0);
  Kokkos::deep_copy(rho_zz, 1.0);
  Kokkos::deep_copy(zz, 1.0);
  Kokkos::deep_copy(areaCell, 500.0);

  auto host_zgrid = Kokkos::create_mirror_view(zgrid);
  for (int i = 0; i < nCells; ++i)
    for (int k = 0; k <= nVertLevels; ++k)
      host_zgrid(k, i) = static_cast<Scalar>(k) * 1000.0;
  Kokkos::deep_copy(zgrid, host_zgrid);

  // Inject NaN into rho_zz (u and w are clean; theta_m passes before rho_zz)
  auto host_rho = Kokkos::create_mirror_view(rho_zz);
  Kokkos::deep_copy(host_rho, rho_zz);
  host_rho(0, 2) = std::numeric_limits<Scalar>::quiet_NaN();
  Kokkos::deep_copy(rho_zz, host_rho);

  try {
    run_post_timestep_diagnostics<ExecSpace>(
        u, w, theta_m, rho_zz, zz, areaCell, zgrid,
        nVertLevels, nCells, nEdges);
    FAIL() << "Expected NaNDetectedError for rho_zz field";
  } catch (const NaNDetectedError& e) {
    EXPECT_EQ(e.field_name(), "rho_zz");
  }
}

}  // namespace
}  // namespace dycore
}  // namespace mpas
