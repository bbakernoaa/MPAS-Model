#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/field_store.hpp"
#include "mpas_dycore/scalar.hpp"

namespace {

using Scalar = mpas::dycore::Scalar;
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using FieldStore = mpas::dycore::Field_Store<Scalar, ExecSpace>;

// ---------------------------------------------------------------------------
// Test 1: Allocate a field, verify extents match, verify the View is valid
// Validates: Requirements 1.3, 1.5
// ---------------------------------------------------------------------------
TEST(FieldStoreTest, AllocateVerifiesExtentsAndValidity) {
  FieldStore store;
  const int n_inner = 40;
  const int n_elem = 100;

  auto view = store.allocate("theta_tend", n_inner, n_elem);

  // Verify extents via the View directly
  EXPECT_EQ(static_cast<int>(view.extent(0)), n_inner);
  EXPECT_EQ(static_cast<int>(view.extent(1)), n_elem);

  // Verify the view has a valid (non-null) data pointer
  EXPECT_NE(view.data(), nullptr);

  // Verify the store reports the same extents
  auto ext = store.extents("theta_tend");
  EXPECT_EQ(ext.n_inner, n_inner);
  EXPECT_EQ(ext.n_elem, n_elem);
}

// ---------------------------------------------------------------------------
// Test 2: Verify owned field is freed on Field_Store destruction (RAII)
// Validates: Requirements 1.4, 1.10
// ---------------------------------------------------------------------------
TEST(FieldStoreTest, OwnedFieldFreedOnDestruction) {
  // This test verifies that allocating a field and then destroying the store
  // runs without error (RAII cleanup). We cannot easily verify the memory is
  // freed in a unit test, but we verify the construction/destruction lifecycle
  // compiles and runs without crashes or sanitizer errors.
  {
    FieldStore store;
    auto view = store.allocate("scratch", 10, 20);
    EXPECT_NE(view.data(), nullptr);
    // store goes out of scope here — RAII frees the owned View
  }
  // If we reach here without a crash/leak, RAII works.
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Test 3: Wrap a Fortran pointer, verify host mirror aliases the original pointer
// Validates: Requirements 1.9
// ---------------------------------------------------------------------------
TEST(FieldStoreTest, WrapFortranPointerAliasesHostMirror) {
  const int n_inner = 5;
  const int n_elem = 3;
  std::vector<Scalar> fortran_buffer(n_inner * n_elem, 0.0);

  // Fill the buffer with known values
  for (int j = 0; j < n_elem; ++j) {
    for (int i = 0; i < n_inner; ++i) {
      fortran_buffer[i + n_inner * j] = static_cast<Scalar>(i * 100 + j);
    }
  }

  FieldStore store;
  auto dv = store.wrap("theta_m", fortran_buffer.data(), n_inner, n_elem);

  // The host mirror of the DualView should alias the original Fortran pointer.
  // i.e. the host view's data pointer should be exactly the fortran_buffer pointer.
  auto host_view = dv.view_host();
  EXPECT_EQ(host_view.data(), fortran_buffer.data());

  // Verify data is accessible through the host mirror
  for (int j = 0; j < n_elem; ++j) {
    for (int i = 0; i < n_inner; ++i) {
      EXPECT_EQ(host_view(i, j), static_cast<Scalar>(i * 100 + j));
    }
  }

  // Mutate through the host mirror and verify original buffer changes
  host_view(0, 0) = static_cast<Scalar>(999.0);
  EXPECT_EQ(fortran_buffer[0], static_cast<Scalar>(999.0));

  // Mutate the original buffer and verify host mirror sees it
  fortran_buffer[1] = static_cast<Scalar>(888.0);
  EXPECT_EQ(host_view(1, 0), static_cast<Scalar>(888.0));
}

// ---------------------------------------------------------------------------
// Test 4: sync_to_device / sync_to_host round-trip
// Validates: Requirements 1.11, 1.12
// ---------------------------------------------------------------------------
TEST(FieldStoreTest, SyncToDeviceSyncToHostRoundTrip) {
  const int n_inner = 4;
  const int n_elem = 3;
  std::vector<Scalar> fortran_buffer(n_inner * n_elem, 0.0);

  // Step 1: Initialize the Fortran buffer with host data
  for (int j = 0; j < n_elem; ++j) {
    for (int i = 0; i < n_inner; ++i) {
      fortran_buffer[i + n_inner * j] = static_cast<Scalar>((i + 1) * (j + 1));
    }
  }

  FieldStore store;
  auto dv = store.wrap("rho_zz", fortran_buffer.data(), n_inner, n_elem);

  // Step 2: Modify host data through the Fortran pointer
  fortran_buffer[0] = static_cast<Scalar>(42.0);

  // Step 3: sync_to_device — device copy should now reflect the modified host data
  store.sync_to_device("rho_zz");

  // Step 4: Modify device data — set all device values to a known pattern
  auto device_view = dv.view_device();
  Kokkos::parallel_for(
      "ModifyDevice",
      Kokkos::RangePolicy<ExecSpace>(0, n_elem),
      KOKKOS_LAMBDA(const int j) {
        for (int i = 0; i < n_inner; ++i) {
          device_view(i, j) = static_cast<Scalar>(i + j * 10 + 1000);
        }
      });
  Kokkos::fence();

  // Step 5: sync_to_host — host/Fortran buffer should now reflect device changes
  store.sync_to_host("rho_zz");

  // Step 6: Verify host (Fortran buffer) reflects the device-written values
  for (int j = 0; j < n_elem; ++j) {
    for (int i = 0; i < n_inner; ++i) {
      Scalar expected = static_cast<Scalar>(i + j * 10 + 1000);
      EXPECT_EQ(fortran_buffer[i + n_inner * j], expected)
          << "Mismatch at (" << i << ", " << j << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Test 5: Time-level access for owned fields
// Validates: Requirements 1.2, 1.13
// ---------------------------------------------------------------------------
TEST(FieldStoreTest, TimeLevelAccessOwnedFields) {
  FieldStore store;
  const int n_inner = 8;
  const int n_elem = 5;
  const int n_levels = 2;

  store.allocate("u", n_inner, n_elem, n_levels);

  // Access levels 1 and 2
  auto level1 = store.level("u", 1);
  auto level2 = store.level("u", 2);

  // Verify both views are valid
  EXPECT_NE(level1.data(), nullptr);
  EXPECT_NE(level2.data(), nullptr);

  // Verify they are distinct views (different data pointers)
  EXPECT_NE(level1.data(), level2.data());

  // Verify extents for both levels
  EXPECT_EQ(static_cast<int>(level1.extent(0)), n_inner);
  EXPECT_EQ(static_cast<int>(level1.extent(1)), n_elem);
  EXPECT_EQ(static_cast<int>(level2.extent(0)), n_inner);
  EXPECT_EQ(static_cast<int>(level2.extent(1)), n_elem);

  // Write distinct data to each level and verify independence
  Kokkos::deep_copy(level1, static_cast<Scalar>(1.0));
  Kokkos::deep_copy(level2, static_cast<Scalar>(2.0));

  // Read back and confirm they hold different values
  auto h_level1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, level1);
  auto h_level2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, level2);
  EXPECT_EQ(h_level1(0, 0), static_cast<Scalar>(1.0));
  EXPECT_EQ(h_level2(0, 0), static_cast<Scalar>(2.0));
}

// ---------------------------------------------------------------------------
// Test 6: Time-level access for wrapped fields
// Validates: Requirements 1.2, 1.9, 1.13
// ---------------------------------------------------------------------------
TEST(FieldStoreTest, TimeLevelAccessWrappedFields) {
  const int n_inner = 6;
  const int n_elem = 4;

  std::vector<Scalar> buf1(n_inner * n_elem, static_cast<Scalar>(10.0));
  std::vector<Scalar> buf2(n_inner * n_elem, static_cast<Scalar>(20.0));

  FieldStore store;
  std::vector<Scalar*> ptrs = {buf1.data(), buf2.data()};
  store.wrap("w", ptrs, n_inner, n_elem);

  // Access levels 1 and 2
  auto level1 = store.level("w", 1);
  auto level2 = store.level("w", 2);

  // Verify both are valid and distinct
  EXPECT_NE(level1.data(), nullptr);
  EXPECT_NE(level2.data(), nullptr);
  EXPECT_NE(level1.data(), level2.data());

  // Verify extents
  EXPECT_EQ(static_cast<int>(level1.extent(0)), n_inner);
  EXPECT_EQ(static_cast<int>(level1.extent(1)), n_elem);
  EXPECT_EQ(static_cast<int>(level2.extent(0)), n_inner);
  EXPECT_EQ(static_cast<int>(level2.extent(1)), n_elem);
}

// ---------------------------------------------------------------------------
// Test 7: Extent preservation
// Validates: Requirements 1.5
// ---------------------------------------------------------------------------
TEST(FieldStoreTest, ExtentPreservationAllocated) {
  FieldStore store;
  const int n_inner = 55;
  const int n_elem = 123;

  store.allocate("tendency_u", n_inner, n_elem);

  auto ext = store.extents("tendency_u");
  EXPECT_EQ(ext.n_inner, n_inner);
  EXPECT_EQ(ext.n_elem, n_elem);
}

TEST(FieldStoreTest, ExtentPreservationWrapped) {
  const int n_inner = 30;
  const int n_elem = 77;
  std::vector<Scalar> buf(n_inner * n_elem, 0.0);

  FieldStore store;
  store.wrap("geom", buf.data(), n_inner, n_elem);

  auto ext = store.extents("geom");
  EXPECT_EQ(ext.n_inner, n_inner);
  EXPECT_EQ(ext.n_elem, n_elem);
}

// ---------------------------------------------------------------------------
// Test 8: RKIND selection — verify sizeof(Scalar) matches expected precision
// Validates: Requirements 1.6
// ---------------------------------------------------------------------------
TEST(FieldStoreTest, RKINDSelection) {
  // The build is configured with MPAS_DOUBLE_PRECISION ON by default (from CMakeLists.txt).
  // So Scalar should be double (8 bytes). If the build is configured for single,
  // it should be float (4 bytes). We just verify sizeof is consistent.
#if MPAS_DOUBLE_PRECISION
  EXPECT_EQ(sizeof(Scalar), sizeof(double));
  static_assert(std::is_same_v<Scalar, double>,
                "Scalar must be double when MPAS_DOUBLE_PRECISION is ON");
#else
  EXPECT_EQ(sizeof(Scalar), sizeof(float));
  static_assert(std::is_same_v<Scalar, float>,
                "Scalar must be float when MPAS_DOUBLE_PRECISION is OFF");
#endif
  // Either way, the scalar type must be 4 or 8 bytes (float or double)
  EXPECT_TRUE(sizeof(Scalar) == 4 || sizeof(Scalar) == 8);
}

}  // namespace
