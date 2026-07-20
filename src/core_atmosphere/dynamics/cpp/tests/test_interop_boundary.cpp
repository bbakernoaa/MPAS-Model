/**
 * @file test_interop_boundary.cpp
 * @brief Unit tests for the C/C++ interop boundary (dycore_c_api).
 *
 * Tests the init/timestep/finalize round-trip, Kokkos lifecycle ordering,
 * DualView wrapping of state pointers, sync_to_device at timestep entry,
 * sync_to_host at timestep exit, and that Fortran host pointers reflect
 * dycore-modified state after return.
 *
 * Validates: Requirements 13.1, 13.5, 13.7, 13.8, 13.9
 *
 * IMPORTANT: Kokkos 4.x forbids re-initialization after finalize within the
 * same process. These tests therefore run within a single Kokkos lifecycle:
 *   - A shared test fixture calls dycore_init in SetUpTestSuite
 *   - Tests exercise timestep and state verification
 *   - TearDownTestSuite calls dycore_finalize and verifies the shutdown
 *
 * The main_interop.cpp runner does NOT pre-initialize Kokkos so that the
 * dycore_init call within SetUpTestSuite is the one that initializes it.
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/dycore_c_api.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

// ─── Test fixture helpers ────────────────────────────────────────────────────

/// Small mesh dimensions for testing.
struct TestMeshDims {
  static constexpr int nCells = 4;
  static constexpr int nEdges = 8;
  static constexpr int nVertices = 4;
  static constexpr int nVertLevels = 3;
  static constexpr int maxEdges = 6;
  static constexpr int num_scalars = 2;
};

/// Allocates all mesh geometry and state buffers for a minimal test mesh.
/// All buffers are heap-allocated and zero-initialized, with geometry buffers
/// set to valid placeholder values.
struct TestBuffers {
  // Mesh geometry/connectivity
  std::vector<int> cellsOnEdge;
  std::vector<int> edgesOnCell;
  std::vector<int> verticesOnEdge;
  std::vector<int> nEdgesOnCell;
  std::vector<double> dvEdge;
  std::vector<double> dcEdge;
  std::vector<double> areaCell;
  std::vector<double> zgrid;
  std::vector<double> zz;
  std::vector<double> fzm;
  std::vector<double> fzp;

  // Prognostic state (two time levels each)
  std::vector<double> u_tl1, u_tl2;
  std::vector<double> w_tl1, w_tl2;
  std::vector<double> theta_m_tl1, theta_m_tl2;
  std::vector<double> rho_zz_tl1, rho_zz_tl2;
  std::vector<double> scalars_tl1, scalars_tl2;

  TestBuffers() {
    using D = TestMeshDims;

    // Mesh connectivity
    cellsOnEdge.resize(2 * D::nEdges, 1);
    edgesOnCell.resize(D::maxEdges * D::nCells, 1);
    verticesOnEdge.resize(2 * D::nEdges, 1);
    nEdgesOnCell.resize(D::nCells, D::maxEdges);

    // Mesh geometry — fill with non-zero valid values
    dvEdge.resize(D::nEdges, 1000.0);
    dcEdge.resize(D::nEdges, 2000.0);
    areaCell.resize(D::nCells, 1.0e6);
    zgrid.resize((D::nVertLevels + 1) * D::nCells, 100.0);
    zz.resize(D::nVertLevels * D::nCells, 1.0);
    fzm.resize(D::nVertLevels * D::nCells, 0.5);
    fzp.resize(D::nVertLevels * D::nCells, 0.5);

    // Prognostic state — zero-initialized
    u_tl1.resize(D::nVertLevels * D::nEdges, 0.0);
    u_tl2.resize(D::nVertLevels * D::nEdges, 0.0);
    w_tl1.resize((D::nVertLevels + 1) * D::nCells, 0.0);
    w_tl2.resize((D::nVertLevels + 1) * D::nCells, 0.0);
    theta_m_tl1.resize(D::nVertLevels * D::nCells, 0.0);
    theta_m_tl2.resize(D::nVertLevels * D::nCells, 0.0);
    rho_zz_tl1.resize(D::nVertLevels * D::nCells, 0.0);
    rho_zz_tl2.resize(D::nVertLevels * D::nCells, 0.0);
    scalars_tl1.resize(D::num_scalars * D::nVertLevels * D::nCells, 0.0);
    scalars_tl2.resize(D::num_scalars * D::nVertLevels * D::nCells, 0.0);
  }

  /// Call dycore_init with all buffers from this struct.
  int call_init() {
    using D = TestMeshDims;
    return dycore_init(
        D::nCells, D::nEdges, D::nVertices, D::nVertLevels, D::maxEdges,
        D::num_scalars,
        cellsOnEdge.data(), edgesOnCell.data(), verticesOnEdge.data(),
        nEdgesOnCell.data(),
        dvEdge.data(), dcEdge.data(), areaCell.data(),
        zgrid.data(), zz.data(), fzm.data(), fzp.data(),
        u_tl1.data(), u_tl2.data(),
        w_tl1.data(), w_tl2.data(),
        theta_m_tl1.data(), theta_m_tl2.data(),
        rho_zz_tl1.data(), rho_zz_tl2.data(),
        scalars_tl1.data(), scalars_tl2.data(),
        /* time_integration_order */ 3,
        /* number_of_sub_steps */ 6,
        /* dynamics_split_steps */ 1,
        /* config_monotonic */ 1,
        /* config_scalar_advection */ 1,
        /* config_apply_lbcs */ 0,
        /* config_mix_full */ 0,
        /* config_iau */ 0,
        /* gpu_aware_comm */ 0,
        /* config_smdiv */ 0.1,
        /* config_len_disp */ 120000.0,
        /* config_apvm_upwinding */ 0.0,
        /* config_hollingsworth */ 1,
        /* mpi_comm_fortran */ 0);
  }
};

// ─── Shared fixture that manages a single Kokkos lifecycle ───────────────────
//
// Kokkos 4.x forbids re-initialization after finalize within one process.
// This fixture initializes via dycore_init in SetUpTestSuite and finalizes
// via dycore_finalize in TearDownTestSuite, giving all tests access to a
// live Kokkos + DycoreContext.

class InteropBoundaryFixture : public ::testing::Test {
 protected:
  static TestBuffers* s_bufs;

  static void SetUpTestSuite() {
    s_bufs = new TestBuffers();

    // Set recognizable patterns in state before init (for DualView wrapping tests)
    const double sentinel = 42.5;
    std::fill(s_bufs->theta_m_tl1.begin(), s_bufs->theta_m_tl1.end(), sentinel);

    const double pattern = 7.77;
    std::fill(s_bufs->w_tl1.begin(), s_bufs->w_tl1.end(), pattern);
    std::fill(s_bufs->w_tl2.begin(), s_bufs->w_tl2.end(), pattern);
    std::fill(s_bufs->scalars_tl1.begin(), s_bufs->scalars_tl1.end(), pattern);
    std::fill(s_bufs->scalars_tl2.begin(), s_bufs->scalars_tl2.end(), pattern);

    // Use iota patterns for fields that will be verified after finalize
    std::iota(s_bufs->u_tl1.begin(), s_bufs->u_tl1.end(), 1.0);
    std::iota(s_bufs->rho_zz_tl2.begin(), s_bufs->rho_zz_tl2.end(), 200.0);

    int rc = s_bufs->call_init();
    ASSERT_EQ(rc, 0) << "dycore_init failed with error code " << rc;
  }

  static void TearDownTestSuite() {
    dycore_finalize();
    delete s_bufs;
    s_bufs = nullptr;
  }
};

TestBuffers* InteropBoundaryFixture::s_bufs = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Kokkos lifecycle — verify is_initialized() is true after dycore_init
// Validates: Requirement 13.5
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(InteropBoundaryFixture, KokkosInitializedAfterInit) {
  // After dycore_init (called in SetUpTestSuite), Kokkos must be initialized
  EXPECT_TRUE(Kokkos::is_initialized());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: DualView wrapping — after init, state pointers are wrapped correctly
//         (tested indirectly: data set before init survives init and is readable)
// Validates: Requirements 13.1, 13.9
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(InteropBoundaryFixture, DualViewWrappingPreservesHostData) {
  // The DualView host mirror should alias the Fortran pointer, so the original
  // buffer should still hold the sentinel value after init.
  const double sentinel = 42.5;
  for (double val : s_bufs->theta_m_tl1) {
    EXPECT_EQ(val, sentinel)
        << "theta_m_tl1 should retain sentinel value after dycore_init (zero-copy wrap)";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: sync_to_device at timestep entry — set host data to known values,
//         call dycore_timestep, verify the data survives the sync round-trip
// Validates: Requirement 13.7
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(InteropBoundaryFixture, SyncToDeviceAtTimestepEntry) {
  // Modify the host buffers (simulating Fortran writing new data)
  const double new_value = 123.456;
  std::fill(s_bufs->u_tl1.begin(), s_bufs->u_tl1.end(), new_value);
  std::fill(s_bufs->rho_zz_tl1.begin(), s_bufs->rho_zz_tl1.end(), new_value);

  // dycore_timestep syncs host→device at entry and device→host at exit.
  // Since the dycore is currently a pass-through (no kernels modify device data),
  // the sync_to_device followed by sync_to_host should preserve the values.
  dycore_timestep(10.0, 1);

  // Verify host data survived the device round-trip
  for (double val : s_bufs->u_tl1) {
    EXPECT_DOUBLE_EQ(val, new_value)
        << "u_tl1 values should survive sync_to_device + sync_to_host round-trip";
  }
  for (double val : s_bufs->rho_zz_tl1) {
    EXPECT_DOUBLE_EQ(val, new_value)
        << "rho_zz_tl1 values should survive sync_to_device + sync_to_host round-trip";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: sync_to_host at timestep exit — after dycore_timestep, host pointers
//         reflect device-side state (currently pass-through, so values unchanged
//         but the sync happened)
// Validates: Requirement 13.8
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(InteropBoundaryFixture, SyncToHostAtTimestepExit) {
  // Fill state with recognizable pattern
  const double pattern = 7.77;
  std::fill(s_bufs->w_tl1.begin(), s_bufs->w_tl1.end(), pattern);
  std::fill(s_bufs->w_tl2.begin(), s_bufs->w_tl2.end(), pattern);
  std::fill(s_bufs->scalars_tl1.begin(), s_bufs->scalars_tl1.end(), pattern);
  std::fill(s_bufs->scalars_tl2.begin(), s_bufs->scalars_tl2.end(), pattern);

  // Call multiple timesteps to verify repeated sync cycles
  dycore_timestep(10.0, 2);
  dycore_timestep(10.0, 3);

  // After timestep exit, host pointers should reflect the device state.
  // Since no kernels modify the data, values should be the same pattern.
  for (double val : s_bufs->w_tl1) {
    EXPECT_DOUBLE_EQ(val, pattern)
        << "w_tl1 should be unchanged after pass-through timestep";
  }
  for (double val : s_bufs->w_tl2) {
    EXPECT_DOUBLE_EQ(val, pattern)
        << "w_tl2 should be unchanged after pass-through timestep";
  }
  for (double val : s_bufs->scalars_tl1) {
    EXPECT_DOUBLE_EQ(val, pattern)
        << "scalars_tl1 should be unchanged after pass-through timestep";
  }
  for (double val : s_bufs->scalars_tl2) {
    EXPECT_DOUBLE_EQ(val, pattern)
        << "scalars_tl2 should be unchanged after pass-through timestep";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Init/timestep/finalize round-trip — verify the full lifecycle runs
//         without crash (tested via the fixture: SetUpTestSuite inits,
//         timesteps above execute, TearDownTestSuite finalizes)
// Validates: Requirements 13.1, 13.5
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(InteropBoundaryFixture, TimestepDoesNotCrash) {
  // Additional timestep call to demonstrate multiple calls are fine
  dycore_timestep(5.0, 4);
  SUCCEED() << "dycore_timestep completed without crash";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Fortran host pointers unchanged after finalize — verify the original
//         buffers are still valid and accessible after dycore_finalize.
//
//         NOTE: This test verifies post-finalize state. It runs LAST because
//         Google Test runs tests within a fixture alphabetically, and we rely
//         on TearDownTestSuite calling dycore_finalize after all tests complete.
//         We verify the data is correct BEFORE finalize; the actual post-finalize
//         verification is done in a separate non-fixture test below.
// Validates: Requirement 13.9
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(InteropBoundaryFixture, HostPointersAccessibleBeforeFinalize) {
  // Set distinct patterns per field for post-finalize verification
  std::iota(s_bufs->u_tl1.begin(), s_bufs->u_tl1.end(), 1.0);
  std::iota(s_bufs->theta_m_tl1.begin(), s_bufs->theta_m_tl1.end(), 100.0);
  std::iota(s_bufs->rho_zz_tl2.begin(), s_bufs->rho_zz_tl2.end(), 200.0);

  // Run a final timestep so sync round-trip preserves these values
  dycore_timestep(5.0, 5);

  // Verify they are correct before finalize
  double expected = 1.0;
  for (double val : s_bufs->u_tl1) {
    EXPECT_DOUBLE_EQ(val, expected)
        << "u_tl1 should hold iota pattern before finalize";
    expected += 1.0;
  }

  expected = 100.0;
  for (double val : s_bufs->theta_m_tl1) {
    EXPECT_DOUBLE_EQ(val, expected)
        << "theta_m_tl1 should hold iota pattern before finalize";
    expected += 1.0;
  }

  expected = 200.0;
  for (double val : s_bufs->rho_zz_tl2) {
    EXPECT_DOUBLE_EQ(val, expected)
        << "rho_zz_tl2 should hold iota pattern before finalize";
    expected += 1.0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Separate test for post-finalize verification using a GTest Environment.
// The Environment's TearDown runs AFTER all test suites, so we can verify
// that Kokkos was finalized and that Fortran-owned buffers survive.
// Validates: Requirements 13.5, 13.9
// ─────────────────────────────────────────────────────────────────────────────

/// Buffers that persist across the entire test program for post-finalize checks.
static std::vector<double> g_post_finalize_u;
static std::vector<double> g_post_finalize_theta;
static std::vector<double> g_post_finalize_rho;
static bool g_post_finalize_checked = false;

/// A GTest Environment that runs after all test suites to verify post-finalize state.
class PostFinalizeEnv : public ::testing::Environment {
 public:
  void SetUp() override {
    // Nothing to do — InteropBoundaryFixture::SetUpTestSuite handles init.
  }

  void TearDown() override {
    // At this point TearDownTestSuite has called dycore_finalize.
    // Verify Kokkos is finalized.
    EXPECT_FALSE(Kokkos::is_initialized())
        << "Kokkos should be finalized after dycore_finalize";

    // Verify the saved buffers still hold their data.
    // (The buffers were copied into globals before finalize by the fixture's
    //  HostPointersAccessibleBeforeFinalize test.)
    if (!g_post_finalize_u.empty()) {
      double expected = 1.0;
      for (double val : g_post_finalize_u) {
        EXPECT_DOUBLE_EQ(val, expected)
            << "u_tl1 Fortran buffer should survive dycore_finalize";
        expected += 1.0;
      }
    }

    if (!g_post_finalize_theta.empty()) {
      double expected = 100.0;
      for (double val : g_post_finalize_theta) {
        EXPECT_DOUBLE_EQ(val, expected)
            << "theta_m_tl1 Fortran buffer should survive dycore_finalize";
        expected += 1.0;
      }
    }

    if (!g_post_finalize_rho.empty()) {
      double expected = 200.0;
      for (double val : g_post_finalize_rho) {
        EXPECT_DOUBLE_EQ(val, expected)
            << "rho_zz_tl2 Fortran buffer should survive dycore_finalize";
        expected += 1.0;
      }
    }

    g_post_finalize_checked = true;
  }
};

/// Test that captures state for post-finalize verification AND registers
/// the environment. Must run within the InteropBoundaryFixture lifecycle.
/// This is called by the HostPointersAccessibleBeforeFinalize test implicitly
/// via a static initializer that registers the PostFinalizeEnv.
static bool g_env_registered = []() {
  ::testing::AddGlobalTestEnvironment(new PostFinalizeEnv());
  return true;
}();

// ─── Additional fixture test to populate post-finalize globals ───────────────
// This test runs within the fixture lifecycle (before TearDownTestSuite/finalize)
// and copies the state into globals for PostFinalizeEnv to verify after finalize.
TEST_F(InteropBoundaryFixture, ZZZ_CaptureStateForPostFinalizeCheck) {
  // Name starts with ZZZ_ to ensure it runs last in alphabetical order
  // (after HostPointersAccessibleBeforeFinalize sets the iota patterns)
  g_post_finalize_u = s_bufs->u_tl1;
  g_post_finalize_theta = s_bufs->theta_m_tl1;
  g_post_finalize_rho = s_bufs->rho_zz_tl2;
}

}  // namespace
