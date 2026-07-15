#include <gtest/gtest.h>

#include "mpas_dycore/time_integrator.hpp"
#include "mpas_dycore/post_timestep_diagnostics.hpp"
#include "mpas_dycore/config.hpp"
#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace mpas {
namespace dycore {
namespace {

/// Parity_Tolerance: tight relative tolerance near machine epsilon.
/// Documented ceiling: 1e-12 for double, 1e-6 for single (Requirement 12).
constexpr Scalar kParityTolerance =
    std::is_same_v<Scalar, double> ? 1.0e-12 : 1.0e-6f;

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using layout = Kokkos::LayoutLeft;
using View1D = Kokkos::View<Scalar*, layout, MemSpace>;
using View2D = Kokkos::View<Scalar**, layout, MemSpace>;

// ---------------------------------------------------------------------------
// Req 3.7: Reject non-SRK3 schemes with an error naming the scheme
// ---------------------------------------------------------------------------

TEST(TimeIntegratorTest, ValidateSchemeSRK3Passes) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .build();

  // Should not throw
  EXPECT_NO_THROW(validate_scheme(config));
}

TEST(TimeIntegratorTest, ValidateSchemeRejectsNonSRK3) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("Euler")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .build();

  EXPECT_THROW(validate_scheme(config), UnsupportedSchemeError);
}

TEST(TimeIntegratorTest, ValidateSchemeErrorNamesTheScheme) {
  const std::string bad_scheme = "RK4_Fancy";
  auto config = ConfigBuilder{}
      .time_integration_scheme(bad_scheme)
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .build();

  try {
    validate_scheme(config);
    FAIL() << "Expected UnsupportedSchemeError was not thrown";
  } catch (const UnsupportedSchemeError& e) {
    // Error must name the scheme (Requirement 3.7)
    EXPECT_EQ(e.scheme(), bad_scheme);
    std::string msg = e.what();
    EXPECT_NE(msg.find(bad_scheme), std::string::npos)
        << "Error message does not name the unsupported scheme: " << msg;
  }
}

TEST(TimeIntegratorTest, ValidateSchemeRejectsEmptyString) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .build();

  EXPECT_THROW(validate_scheme(config), UnsupportedSchemeError);
}

// ---------------------------------------------------------------------------
// Req 3.3: Order 3 stage weights from Reference_Model
// ---------------------------------------------------------------------------

TEST(TimeIntegratorTest, Order3StageWeights) {
  const int ns = 6;
  const Scalar dt = 720.0;  // typical dynamics timestep

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(ns)
      .build();

  auto tables = compute_rk_tables(config, dt);

  // rk_timestep: {dt/3, dt/2, dt}
  EXPECT_DOUBLE_EQ(tables.rk_timestep[0], dt / 3.0);
  EXPECT_DOUBLE_EQ(tables.rk_timestep[1], dt / 2.0);
  EXPECT_DOUBLE_EQ(tables.rk_timestep[2], dt);

  // rk_sub_timestep: {dt/3, dt/ns, dt/ns}
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[0], dt / 3.0);
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[1], dt / static_cast<Scalar>(ns));
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[2], dt / static_cast<Scalar>(ns));

  // number_sub_steps: {1, max(1, ns/2), ns} = {1, 3, 6}
  EXPECT_EQ(tables.number_sub_steps[0], 1);
  EXPECT_EQ(tables.number_sub_steps[1], 3);
  EXPECT_EQ(tables.number_sub_steps[2], 6);
}

// ---------------------------------------------------------------------------
// Req 3.4: Order 2 stage weights from Reference_Model
// ---------------------------------------------------------------------------

TEST(TimeIntegratorTest, Order2StageWeights) {
  const int ns = 6;
  const Scalar dt = 720.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(2)
      .number_of_sub_steps(ns)
      .build();

  auto tables = compute_rk_tables(config, dt);

  // rk_timestep: {dt/2, dt/2, dt}
  EXPECT_DOUBLE_EQ(tables.rk_timestep[0], dt / 2.0);
  EXPECT_DOUBLE_EQ(tables.rk_timestep[1], dt / 2.0);
  EXPECT_DOUBLE_EQ(tables.rk_timestep[2], dt);

  // rk_sub_timestep: {dt/ns, dt/ns, dt/ns}
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[0], dt / static_cast<Scalar>(ns));
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[1], dt / static_cast<Scalar>(ns));
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[2], dt / static_cast<Scalar>(ns));

  // number_sub_steps: {max(1, ns/2), max(1, ns/2), ns} = {3, 3, 6}
  EXPECT_EQ(tables.number_sub_steps[0], 3);
  EXPECT_EQ(tables.number_sub_steps[1], 3);
  EXPECT_EQ(tables.number_sub_steps[2], 6);
}

// ---------------------------------------------------------------------------
// Req 3.6: Per-stage acoustic substep counts
// ---------------------------------------------------------------------------

TEST(TimeIntegratorTest, AcousticSubstepsOrder3) {
  const int ns = 10;
  const Scalar dt = 600.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(ns)
      .build();

  auto tables = compute_rk_tables(config, dt);

  // Stage 1: 1 substep; Stage 2: max(1, 10/2)=5; Stage 3: 10
  EXPECT_EQ(acoustic_substeps(tables, 1), 1);
  EXPECT_EQ(acoustic_substeps(tables, 2), 5);
  EXPECT_EQ(acoustic_substeps(tables, 3), 10);
}

TEST(TimeIntegratorTest, AcousticSubstepsOrder2) {
  const int ns = 8;
  const Scalar dt = 600.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(2)
      .number_of_sub_steps(ns)
      .build();

  auto tables = compute_rk_tables(config, dt);

  // All stages: {max(1, 8/2), max(1, 8/2), 8} = {4, 4, 8}
  EXPECT_EQ(acoustic_substeps(tables, 1), 4);
  EXPECT_EQ(acoustic_substeps(tables, 2), 4);
  EXPECT_EQ(acoustic_substeps(tables, 3), 8);
}

// ---------------------------------------------------------------------------
// Edge case: number_of_sub_steps == 1 (minimum)
// ---------------------------------------------------------------------------

TEST(TimeIntegratorTest, Order3WithOneSubStep) {
  const int ns = 1;
  const Scalar dt = 300.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(ns)
      .build();

  auto tables = compute_rk_tables(config, dt);

  // number_sub_steps: {1, max(1, 1/2)=max(1,0)=1, 1}
  EXPECT_EQ(tables.number_sub_steps[0], 1);
  EXPECT_EQ(tables.number_sub_steps[1], 1);
  EXPECT_EQ(tables.number_sub_steps[2], 1);

  // rk_sub_timestep: {dt/3, dt/1, dt/1}
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[0], dt / 3.0);
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[1], dt);
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[2], dt);
}

TEST(TimeIntegratorTest, Order2WithOneSubStep) {
  const int ns = 1;
  const Scalar dt = 300.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(2)
      .number_of_sub_steps(ns)
      .build();

  auto tables = compute_rk_tables(config, dt);

  // number_sub_steps: {max(1, 0)=1, max(1, 0)=1, 1}
  EXPECT_EQ(tables.number_sub_steps[0], 1);
  EXPECT_EQ(tables.number_sub_steps[1], 1);
  EXPECT_EQ(tables.number_sub_steps[2], 1);

  // rk_sub_timestep: all dt/1 = dt
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[0], dt);
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[1], dt);
  EXPECT_DOUBLE_EQ(tables.rk_sub_timestep[2], dt);
}

// ---------------------------------------------------------------------------
// Invalid order throws
// ---------------------------------------------------------------------------

TEST(TimeIntegratorTest, InvalidOrderThrows) {
  const Scalar dt = 720.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(4)
      .number_of_sub_steps(6)
      .build();

  EXPECT_THROW(compute_rk_tables(config, dt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// compute_rk_tables also validates scheme
// ---------------------------------------------------------------------------

TEST(TimeIntegratorTest, ComputeRKTablesRejectsNonSRK3) {
  const Scalar dt = 720.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("Leapfrog")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .build();

  EXPECT_THROW(compute_rk_tables(config, dt), UnsupportedSchemeError);
}

// ===========================================================================
// Req 14.2: RK table values within Parity_Tolerance of Reference_Model
// Req 14.9: Identify diverging field on failure
// ===========================================================================

/// @brief Verify Order-3 RK stage weights match the Reference_Model to within
/// Parity_Tolerance, identifying any diverging table entry by name.
TEST(TimeIntegratorTest, ParityTolerance_Order3TablesMatchReference) {
  const int ns = 6;
  const Scalar dt = 720.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(ns)
      .build();

  auto tables = compute_rk_tables(config, dt);

  // Reference_Model expected values for order 3, ns=6, dt=720:
  // rk_timestep = {dt/3, dt/2, dt} = {240, 360, 720}
  // rk_sub_timestep = {dt/3, dt/ns, dt/ns} = {240, 120, 120}
  // number_sub_steps = {1, max(1,ns/2), ns} = {1, 3, 6}
  struct RefEntry {
    std::string name;
    Scalar actual;
    Scalar expected;
  };

  std::vector<RefEntry> entries = {
      {"rk_timestep[0]", tables.rk_timestep[0], dt / 3.0},
      {"rk_timestep[1]", tables.rk_timestep[1], dt / 2.0},
      {"rk_timestep[2]", tables.rk_timestep[2], dt},
      {"rk_sub_timestep[0]", tables.rk_sub_timestep[0], dt / 3.0},
      {"rk_sub_timestep[1]", tables.rk_sub_timestep[1], dt / static_cast<Scalar>(ns)},
      {"rk_sub_timestep[2]", tables.rk_sub_timestep[2], dt / static_cast<Scalar>(ns)},
      {"number_sub_steps[0]", static_cast<Scalar>(tables.number_sub_steps[0]), 1.0},
      {"number_sub_steps[1]", static_cast<Scalar>(tables.number_sub_steps[1]), 3.0},
      {"number_sub_steps[2]", static_cast<Scalar>(tables.number_sub_steps[2]), 6.0},
  };

  std::vector<std::string> diverging_fields;
  for (const auto& e : entries) {
    Scalar diff = std::abs(e.actual - e.expected);
    Scalar scale = std::max(Scalar{1.0}, std::abs(e.expected));
    if (diff / scale > kParityTolerance) {
      diverging_fields.push_back(e.name);
      ADD_FAILURE() << "RK table entry '" << e.name << "' diverges from Reference_Model:"
                    << " actual=" << e.actual << " expected=" << e.expected
                    << " rel_err=" << diff / scale
                    << " > Parity_Tolerance=" << kParityTolerance;
    }
  }

  EXPECT_TRUE(diverging_fields.empty())
      << "Time_Integrator RK tables diverge beyond Parity_Tolerance";
}

/// @brief Verify Order-2 RK stage weights match the Reference_Model to within
/// Parity_Tolerance, identifying any diverging table entry by name.
TEST(TimeIntegratorTest, ParityTolerance_Order2TablesMatchReference) {
  const int ns = 8;
  const Scalar dt = 600.0;

  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(2)
      .number_of_sub_steps(ns)
      .build();

  auto tables = compute_rk_tables(config, dt);

  // Reference_Model expected values for order 2, ns=8, dt=600:
  // rk_timestep = {dt/2, dt/2, dt} = {300, 300, 600}
  // rk_sub_timestep = {dt/ns, dt/ns, dt/ns} = {75, 75, 75}
  // number_sub_steps = {max(1,ns/2), max(1,ns/2), ns} = {4, 4, 8}
  struct RefEntry {
    std::string name;
    Scalar actual;
    Scalar expected;
  };

  std::vector<RefEntry> entries = {
      {"rk_timestep[0]", tables.rk_timestep[0], dt / 2.0},
      {"rk_timestep[1]", tables.rk_timestep[1], dt / 2.0},
      {"rk_timestep[2]", tables.rk_timestep[2], dt},
      {"rk_sub_timestep[0]", tables.rk_sub_timestep[0], dt / static_cast<Scalar>(ns)},
      {"rk_sub_timestep[1]", tables.rk_sub_timestep[1], dt / static_cast<Scalar>(ns)},
      {"rk_sub_timestep[2]", tables.rk_sub_timestep[2], dt / static_cast<Scalar>(ns)},
      {"number_sub_steps[0]", static_cast<Scalar>(tables.number_sub_steps[0]), 4.0},
      {"number_sub_steps[1]", static_cast<Scalar>(tables.number_sub_steps[1]), 4.0},
      {"number_sub_steps[2]", static_cast<Scalar>(tables.number_sub_steps[2]), 8.0},
  };

  std::vector<std::string> diverging_fields;
  for (const auto& e : entries) {
    Scalar diff = std::abs(e.actual - e.expected);
    Scalar scale = std::max(Scalar{1.0}, std::abs(e.expected));
    if (diff / scale > kParityTolerance) {
      diverging_fields.push_back(e.name);
      ADD_FAILURE() << "RK table entry '" << e.name << "' diverges from Reference_Model:"
                    << " actual=" << e.actual << " expected=" << e.expected
                    << " rel_err=" << diff / scale
                    << " > Parity_Tolerance=" << kParityTolerance;
    }
  }

  EXPECT_TRUE(diverging_fields.empty())
      << "Time_Integrator RK tables (order 2) diverge beyond Parity_Tolerance";
}

// ===========================================================================
// Req 12.4: Global min/max diagnostics within Parity_Tolerance of expected
// Req 14.2, 14.9: Component parity comparison with diverging-field reporting
// ===========================================================================

/// @brief Verify global min/max diagnostics against a known synthetic state
/// within Parity_Tolerance, identifying diverging fields by name.
TEST(TimeIntegratorTest, ParityTolerance_MinMaxDiagnostics) {
  const int nVertLevels = 5;
  const int nCells = 8;
  const int nEdges = 12;

  View2D u("u", nVertLevels, nEdges);
  View2D w("w", nVertLevels + 1, nCells);

  // Fill with a known pattern: u(k,i) = k*nEdges + i - 30
  // w(k,i) = -0.5 + 0.01 * (k * nCells + i)
  auto host_u = Kokkos::create_mirror_view(u);
  auto host_w = Kokkos::create_mirror_view(w);

  for (int k = 0; k < nVertLevels; ++k)
    for (int i = 0; i < nEdges; ++i)
      host_u(k, i) = static_cast<Scalar>(k * nEdges + i) - 30.0;

  for (int k = 0; k <= nVertLevels; ++k)
    for (int i = 0; i < nCells; ++i)
      host_w(k, i) = -0.5 + 0.01 * static_cast<Scalar>(k * nCells + i);

  Kokkos::deep_copy(u, host_u);
  Kokkos::deep_copy(w, host_w);

  // Reference_Model expected:
  // u: min = 0*12 + 0 - 30 = -30, max = 4*12 + 11 - 30 = 29
  // w: min = -0.5 + 0.01*0 = -0.5, max = -0.5 + 0.01*(5*8+7) = -0.5 + 0.47 = -0.03
  const Scalar ref_u_min = -30.0;
  const Scalar ref_u_max = 29.0;
  const Scalar ref_w_min = -0.5;
  const Scalar ref_w_max = -0.5 + 0.01 * static_cast<Scalar>(5 * 8 + 7);

  auto u_mm = compute_field_min_max<ExecSpace>(u, nVertLevels, nEdges);
  auto w_mm = compute_field_min_max<ExecSpace>(w, nVertLevels + 1, nCells);

  struct RefEntry {
    std::string name;
    Scalar actual;
    Scalar expected;
  };

  std::vector<RefEntry> entries = {
      {"u_min", u_mm.min_val, ref_u_min},
      {"u_max", u_mm.max_val, ref_u_max},
      {"w_min", w_mm.min_val, ref_w_min},
      {"w_max", w_mm.max_val, ref_w_max},
  };

  std::vector<std::string> diverging_fields;
  for (const auto& e : entries) {
    Scalar diff = std::abs(e.actual - e.expected);
    Scalar scale = std::max(Scalar{1.0}, std::abs(e.expected));
    if (diff / scale > kParityTolerance) {
      diverging_fields.push_back(e.name);
      ADD_FAILURE() << "Diagnostic '" << e.name << "' diverges from Reference_Model:"
                    << " actual=" << e.actual << " expected=" << e.expected
                    << " rel_err=" << diff / scale
                    << " > Parity_Tolerance=" << kParityTolerance;
    }
  }

  EXPECT_TRUE(diverging_fields.empty())
      << "Post-timestep min/max diagnostics diverge beyond Parity_Tolerance";
}

// ===========================================================================
// Req 12.3: Dry-air mass conservation diagnostic within Parity_Tolerance
// Req 12.5: NaN guard identifies the offending field by name
// ===========================================================================

/// @brief Verify that mass conservation diagnostic computes correctly for
/// a known state and the result matches the analytical reference within
/// Parity_Tolerance.
TEST(TimeIntegratorTest, ParityTolerance_DryAirMassConservation) {
  // Setup a synthetic state where the expected mass is analytically computable.
  // rho_zz = 1.225 (sea-level air density), zz = 1.0 (flat terrain),
  // uniform area = 1e6 m^2, uniform layer thickness = 2000 m.
  const int nVertLevels = 4;
  const int nCells = 6;

  View2D rho_zz("rho_zz", nVertLevels, nCells);
  View2D zz("zz", nVertLevels, nCells);
  View1D areaCell("areaCell", nCells);
  View2D zgrid("zgrid", nVertLevels + 1, nCells);

  Kokkos::deep_copy(rho_zz, 1.225);
  Kokkos::deep_copy(zz, 1.0);
  Kokkos::deep_copy(areaCell, 1.0e6);

  auto host_zgrid = Kokkos::create_mirror_view(zgrid);
  for (int iCell = 0; iCell < nCells; ++iCell) {
    for (int k = 0; k <= nVertLevels; ++k) {
      host_zgrid(k, iCell) = static_cast<Scalar>(k) * 2000.0;
    }
  }
  Kokkos::deep_copy(zgrid, host_zgrid);

  Scalar mass = compute_total_dry_air_mass<ExecSpace>(
      rho_zz, zz, areaCell, zgrid, nVertLevels, nCells);

  // Reference: 1.225 * 1.0 * 2000 * 1e6 * 4 levels * 6 cells = 5.88e10
  const Scalar ref_mass = 1.225 * 1.0 * 2000.0 * 1.0e6 * 4.0 * 6.0;

  Scalar diff = std::abs(mass - ref_mass);
  Scalar scale = std::max(Scalar{1.0}, std::abs(ref_mass));
  EXPECT_LE(diff / scale, kParityTolerance)
      << "Dry-air mass diverges from Reference_Model:"
      << " computed=" << mass << " expected=" << ref_mass
      << " rel_err=" << diff / scale
      << " > Parity_Tolerance=" << kParityTolerance;
}

/// @brief Verify NaN guard identifies the correct field when multiple fields
/// are checked (integration test exercising the ordering in
/// run_post_timestep_diagnostics). This validates Req 12.5 + 14.9: the
/// critical error must name the offending field.
TEST(TimeIntegratorTest, NaNGuard_IdentifiesThetaMField) {
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

  // All fields clean except theta_m
  Kokkos::deep_copy(u, 10.0);
  Kokkos::deep_copy(w, 0.05);
  Kokkos::deep_copy(theta_m, 300.0);
  Kokkos::deep_copy(rho_zz, 1.1);
  Kokkos::deep_copy(zz, 1.0);
  Kokkos::deep_copy(areaCell, 1000.0);

  auto host_zgrid = Kokkos::create_mirror_view(zgrid);
  for (int i = 0; i < nCells; ++i)
    for (int k = 0; k <= nVertLevels; ++k)
      host_zgrid(k, i) = static_cast<Scalar>(k) * 1000.0;
  Kokkos::deep_copy(zgrid, host_zgrid);

  // Inject NaN only into theta_m (u and w are clean)
  auto host_theta = Kokkos::create_mirror_view(theta_m);
  Kokkos::deep_copy(host_theta, theta_m);
  host_theta(1, 2) = std::numeric_limits<Scalar>::quiet_NaN();
  Kokkos::deep_copy(theta_m, host_theta);

  try {
    run_post_timestep_diagnostics<ExecSpace>(
        u, w, theta_m, rho_zz, zz, areaCell, zgrid,
        nVertLevels, nCells, nEdges);
    FAIL() << "Expected NaNDetectedError for theta_m";
  } catch (const NaNDetectedError& e) {
    // Req 14.9: The test identifies the diverging field by name
    EXPECT_EQ(e.field_name(), "theta_m")
        << "NaN guard should identify 'theta_m' as the offending field";
  }
}

}  // namespace
}  // namespace dycore
}  // namespace mpas
