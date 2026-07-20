/**
 * @file test_dimension_validation.cpp
 * @brief Property test for dimension validation in dycore_init.
 *
 * Feature: fortran-integration, Property 8: Dimension Validation Rejects Invalid Inputs
 *
 * Validates: Requirements 8.3
 *
 * For any dimension tuple passed to dycore_init where at least one of
 * (nCells, nEdges, nVertLevels, maxEdges) is <= 0, the C API SHALL return
 * error code 2 without initializing Kokkos or wrapping any Views.
 *
 * The test generates random dimension tuples with at least one invalid
 * (non-positive) value and verifies that dycore_init returns 2 in all cases.
 * Valid (non-null) dummy pointers are passed for array arguments to isolate
 * the dimension validation check.
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/dycore_c_api.h"

#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

namespace {

// Number of random iterations to run
static constexpr int kNumIterations = 100;

/// Helper to generate a random int in [lo, hi] inclusive.
int rand_int(std::mt19937& rng, int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(rng);
}

/// Generate an invalid dimension value (<= 0).
int rand_invalid_dim(std::mt19937& rng) {
  // Generate values in [-1000, 0] to cover negative and zero
  return rand_int(rng, -1000, 0);
}

/// Generate a valid dimension value (> 0).
int rand_valid_dim(std::mt19937& rng) {
  return rand_int(rng, 1, 10000);
}

/// Dummy buffer large enough for any array parameter at maximum test dimensions.
/// We only need non-null pointers since dimension validation happens first.
struct DummyBuffers {
  std::vector<int> int_buf;
  std::vector<double> double_buf;

  DummyBuffers() : int_buf(100000, 1), double_buf(100000, 1.0) {}

  int* iptr() { return int_buf.data(); }
  double* dptr() { return double_buf.data(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Property 8: Dimension Validation Rejects Invalid Inputs
//
// For any dimension tuple where at least one of (nCells, nEdges, nVertLevels,
// maxEdges) is <= 0, dycore_init SHALL return error code 2 without initializing
// Kokkos.
// ─────────────────────────────────────────────────────────────────────────────

TEST(DimensionValidationProperty, InvalidDimensionsReturnErrorCode2) {
  // Feature: fortran-integration, Property 8: Dimension Validation Rejects Invalid Inputs
  // **Validates: Requirements 8.3**

  std::mt19937 rng(42);  // Fixed seed for reproducibility
  DummyBuffers bufs;

  for (int iter = 0; iter < kNumIterations; ++iter) {
    // Generate 4 dimension values, each independently valid or invalid
    int dims[4];
    for (int i = 0; i < 4; ++i) {
      dims[i] = rand_valid_dim(rng);
    }

    // Randomly choose which dimensions to make invalid (at least one must be invalid)
    // Strategy: pick a random non-empty subset of {0,1,2,3} to invalidate
    int invalid_mask = rand_int(rng, 1, 15);  // 1..15 (at least one bit set)
    for (int i = 0; i < 4; ++i) {
      if (invalid_mask & (1 << i)) {
        dims[i] = rand_invalid_dim(rng);
      }
    }

    int nCells = dims[0];
    int nEdges = dims[1];
    int nVertLevels = dims[2];
    int maxEdges = dims[3];

    // nVertices and num_scalars can be valid — they're not part of dimension check
    int nVertices = rand_valid_dim(rng);
    int num_scalars = rand_int(rng, 1, 10);

    int rc = dycore_init(
        nCells, nEdges, nVertices, nVertLevels, maxEdges, num_scalars,
        /* connectivity */ bufs.iptr(), bufs.iptr(), bufs.iptr(), bufs.iptr(),
        /* geometry */ bufs.dptr(), bufs.dptr(), bufs.dptr(),
        bufs.dptr(), bufs.dptr(), bufs.dptr(), bufs.dptr(),
        /* state TL1 */ bufs.dptr(), bufs.dptr(), bufs.dptr(),
        bufs.dptr(), bufs.dptr(),
        /* state TL2 */ bufs.dptr(), bufs.dptr(), bufs.dptr(),
        bufs.dptr(), bufs.dptr(),
        /* config */ 3, 6, 1, 1, 1, 0, 0, 0, 0,
        0.1, 120000.0, 0.0, 1,
        /* mpi_comm_fortran */ 0);

    ASSERT_EQ(rc, 2)
        << "Iteration " << iter << ": dycore_init should return error code 2 "
        << "for invalid dimensions (nCells=" << nCells << ", nEdges=" << nEdges
        << ", nVertLevels=" << nVertLevels << ", maxEdges=" << maxEdges << ")";
  }

  // Verify Kokkos was NOT initialized (dimension validation rejects before init)
  EXPECT_FALSE(Kokkos::is_initialized())
      << "Kokkos should NOT be initialized when all calls used invalid dimensions";
}

// Additional test: verify each individual dimension being invalid triggers error 2
TEST(DimensionValidationProperty, EachDimensionIndependentlyValidated) {
  // Feature: fortran-integration, Property 8: Dimension Validation Rejects Invalid Inputs
  // **Validates: Requirements 8.3**

  DummyBuffers bufs;

  // Test each dimension individually being zero
  struct TestCase {
    int nCells, nEdges, nVertLevels, maxEdges;
    const char* description;
  };

  TestCase cases[] = {
      {0, 10, 10, 10, "nCells=0"},
      {10, 0, 10, 10, "nEdges=0"},
      {10, 10, 0, 10, "nVertLevels=0"},
      {10, 10, 10, 0, "maxEdges=0"},
      {-1, 10, 10, 10, "nCells=-1"},
      {10, -5, 10, 10, "nEdges=-5"},
      {10, 10, -100, 10, "nVertLevels=-100"},
      {10, 10, 10, -1, "maxEdges=-1"},
  };

  for (const auto& tc : cases) {
    int rc = dycore_init(
        tc.nCells, tc.nEdges, 4 /* nVertices */, tc.nVertLevels, tc.maxEdges,
        2 /* num_scalars */,
        bufs.iptr(), bufs.iptr(), bufs.iptr(), bufs.iptr(),
        bufs.dptr(), bufs.dptr(), bufs.dptr(),
        bufs.dptr(), bufs.dptr(), bufs.dptr(), bufs.dptr(),
        bufs.dptr(), bufs.dptr(), bufs.dptr(),
        bufs.dptr(), bufs.dptr(),
        bufs.dptr(), bufs.dptr(), bufs.dptr(),
        bufs.dptr(), bufs.dptr(),
        3, 6, 1, 1, 1, 0, 0, 0, 0,
        0.1, 120000.0, 0.0, 1,
        /* mpi_comm_fortran */ 0);

    EXPECT_EQ(rc, 2)
        << "dycore_init should return 2 for invalid dimension: " << tc.description;
  }

  EXPECT_FALSE(Kokkos::is_initialized())
      << "Kokkos should NOT be initialized after invalid dimension calls";
}

}  // namespace
