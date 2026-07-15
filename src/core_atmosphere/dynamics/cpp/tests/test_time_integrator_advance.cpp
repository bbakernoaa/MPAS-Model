#include <gtest/gtest.h>

#include "mpas_dycore/time_integrator_advance.hpp"
#include "mpas_dycore/config.hpp"
#include "mpas_dycore/scalar.hpp"

#include <string>
#include <vector>

namespace mpas {
namespace dycore {
namespace {

// ─── Test: advance() rejects non-SRK3 scheme ─────────────────────────────

TEST(TimeIntegratorAdvanceTest, AdvanceRejectsNonSRK3) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("Euler")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(1)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 720.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = false,
  };

  Time_Integrator_Advance integrator;
  EXPECT_THROW(integrator.advance(domain), UnsupportedSchemeError);
}

// ─── Test: advance() runs without halo manager (single process) ───────────

TEST(TimeIntegratorAdvanceTest, AdvanceRunsWithoutHaloManager) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(1)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 720.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = false,
  };

  Time_Integrator_Advance integrator;
  // Should not throw or crash with nullptr halo_manager
  EXPECT_NO_THROW(integrator.advance(domain));
}

// ─── Test: advance() with dynamics_split_steps > 1 (Req 3.5) ─────────────

TEST(TimeIntegratorAdvanceTest, AdvanceWithDynamicsSplit) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(3)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 720.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = false,
  };

  Time_Integrator_Advance integrator;
  // Should handle multiple dynamics substeps without error
  EXPECT_NO_THROW(integrator.advance(domain));
}

// ─── Test: advance() with split dynamics-transport (Req 3.5) ──────────────

TEST(TimeIntegratorAdvanceTest, AdvanceWithSplitTransport) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(2)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 720.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = true,
  };

  Time_Integrator_Advance integrator;
  EXPECT_NO_THROW(integrator.advance(domain));
}

// ─── Test: advance() with order 2 (Req 3.4) ──────────────────────────────

TEST(TimeIntegratorAdvanceTest, AdvanceOrder2) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(2)
      .number_of_sub_steps(4)
      .dynamics_split_steps(1)
      .config_monotonic(false)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 600.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = false,
  };

  Time_Integrator_Advance integrator;
  EXPECT_NO_THROW(integrator.advance(domain));
}

// ─── Test: advance() with regional mode (Req 9.7 boundary adjustments) ───

TEST(TimeIntegratorAdvanceTest, AdvanceWithRegionalMode) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(1)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(true)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 720.0,
      .boundary_remaining_time = 3600.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = false,
  };

  Time_Integrator_Advance integrator;
  // Should handle regional mode branch without error
  EXPECT_NO_THROW(integrator.advance(domain));
}

// ─── Test: advance() with scalar advection disabled ───────────────────────

TEST(TimeIntegratorAdvanceTest, AdvanceScalarAdvectionDisabled) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(1)
      .config_monotonic(true)
      .config_scalar_advection(false)
      .config_apply_lbcs(false)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 720.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = false,
      .split_dynamics_transport = false,
  };

  Time_Integrator_Advance integrator;
  EXPECT_NO_THROW(integrator.advance(domain));
}

// ─── Test: advance() with IAU enabled (Req 10.2 injection point) ──────────

TEST(TimeIntegratorAdvanceTest, AdvanceWithIAUEnabled) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(1)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .config_iau(true)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 720.0,
      .iau_window_length_s = 3600.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = false,
  };

  Time_Integrator_Advance integrator;
  // IAU injection point should not affect orchestration structure
  EXPECT_NO_THROW(integrator.advance(domain));
}

// ─── Test: advance() validates table consistency ──────────────────────────

TEST(TimeIntegratorAdvanceTest, RKTablesUsedCorrectly) {
  // Verify that the RK tables produced by advance match the expected values
  // for order 3 with 6 substeps (standard MPAS configuration).
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(1)
      .build();

  const Scalar dt = 720.0;
  const Scalar dt_dynamics = dt / 1.0;  // dynamics_split_steps = 1
  auto tables = compute_rk_tables(config, dt_dynamics);

  // Order 3: substep counts are {1, 3, 6}
  EXPECT_EQ(tables.number_sub_steps[0], 1);
  EXPECT_EQ(tables.number_sub_steps[1], 3);
  EXPECT_EQ(tables.number_sub_steps[2], 6);

  // Order 3: rk_timestep = {dt/3, dt/2, dt}
  EXPECT_NEAR(tables.rk_timestep[0], dt_dynamics / 3.0, 1e-12);
  EXPECT_NEAR(tables.rk_timestep[1], dt_dynamics / 2.0, 1e-12);
  EXPECT_NEAR(tables.rk_timestep[2], dt_dynamics, 1e-12);
}

// ─── Test: advance() with both regional + split transport ─────────────────

TEST(TimeIntegratorAdvanceTest, AdvanceRegionalSplitTransport) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(2)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(true)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 10,
      .nEdges = 20,
      .nVertices = 5,
      .nVertLevels = 10,
      .nCellsSolve = 10,
      .nEdgesSolve = 20,
      .num_scalars = 3,
      .maxEdges = 6,
      .itimestep = 1,
      .dt = 720.0,
      .boundary_remaining_time = 3600.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = true,
  };

  Time_Integrator_Advance integrator;
  // Combines regional mode with split transport — tests both branches
  EXPECT_NO_THROW(integrator.advance(domain));
}

// ─── Test: advance() with single acoustic substep (order 3 stage 1) ───────

TEST(TimeIntegratorAdvanceTest, AdvanceSingleAcousticSubstep) {
  auto config = ConfigBuilder{}
      .time_integration_scheme("SRK3")
      .time_integration_order(3)
      .number_of_sub_steps(2)   // ns=2 → stage 1 gets max(1,2/2)=1 substep
      .dynamics_split_steps(1)
      .config_monotonic(false)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .build();

  AdvanceDomain domain{
      .config = config,
      .nCells = 5,
      .nEdges = 10,
      .nVertices = 3,
      .nVertLevels = 5,
      .nCellsSolve = 5,
      .nEdgesSolve = 10,
      .num_scalars = 2,
      .maxEdges = 4,
      .itimestep = 1,
      .dt = 300.0,
      .halo_manager = nullptr,
      .scalar_advection_enabled = true,
      .split_dynamics_transport = false,
  };

  Time_Integrator_Advance integrator;
  EXPECT_NO_THROW(integrator.advance(domain));
}

}  // namespace
}  // namespace dycore
}  // namespace mpas
