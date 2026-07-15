/// @file test_field_store_property_extents.cpp
/// @brief Property-based test: Field_Store preserves extents and column-major layout.
///
/// Feature: mpas-dycore-cpp-port, Property 1: Field_Store preserves extents and column-major layout
///
/// **Validates: Requirements 1.5, 1.8**
///
/// For randomly generated dimensions (n_inner in [1, 200], n_elem in [1, 500]):
/// 1. After allocate(name, n_inner, n_elem), extents() returns {n_inner, n_elem}
/// 2. After wrap(name, ptr, n_inner, n_elem), extents() returns {n_inner, n_elem}
/// 3. The returned View has LayoutLeft (stride(0)==1 for the inner dimension)
/// 4. The View's extent(0) == n_inner and extent(1) == n_elem

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include "mpas_dycore/field_store.hpp"

#include <string>
#include <vector>

namespace {

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using Store = mpas::dycore::Field_Store<mpas::dycore::Scalar, ExecSpace>;

/// Generate a valid n_inner dimension in [1, 200].
rc::Gen<int> genInner() {
  return rc::gen::inRange(1, 201);
}

/// Generate a valid n_elem dimension in [1, 500].
rc::Gen<int> genElem() {
  return rc::gen::inRange(1, 501);
}

}  // namespace

// ---------------------------------------------------------------------------
// Property 1a: allocate() preserves extents
// ---------------------------------------------------------------------------
RC_GTEST_PROP(FieldStorePropertyExtents,
              AllocatePreservesExtents,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 1: Field_Store preserves extents and column-major layout
  const int n_inner = *genInner();
  const int n_elem = *genElem();

  Store store;
  auto view = store.allocate("test_field", n_inner, n_elem);

  // Check extents() returns the requested dimensions
  auto ext = store.extents("test_field");
  RC_ASSERT(ext.n_inner == n_inner);
  RC_ASSERT(ext.n_elem == n_elem);

  // Check View's extent matches
  RC_ASSERT(static_cast<int>(view.extent(0)) == n_inner);
  RC_ASSERT(static_cast<int>(view.extent(1)) == n_elem);
}

// ---------------------------------------------------------------------------
// Property 1b: wrap() preserves extents
// ---------------------------------------------------------------------------
RC_GTEST_PROP(FieldStorePropertyExtents,
              WrapPreservesExtents,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 1: Field_Store preserves extents and column-major layout
  const int n_inner = *genInner();
  const int n_elem = *genElem();

  // Allocate a host buffer to simulate Fortran-owned memory
  std::vector<mpas::dycore::Scalar> buffer(
      static_cast<std::size_t>(n_inner) * static_cast<std::size_t>(n_elem), 0.0);

  Store store;
  auto dv = store.wrap("wrapped_field", buffer.data(), n_inner, n_elem);

  // Check extents() returns the requested dimensions
  auto ext = store.extents("wrapped_field");
  RC_ASSERT(ext.n_inner == n_inner);
  RC_ASSERT(ext.n_elem == n_elem);

  // Check the host-side View's extents match
  auto host_view = dv.view_host();
  RC_ASSERT(static_cast<int>(host_view.extent(0)) == n_inner);
  RC_ASSERT(static_cast<int>(host_view.extent(1)) == n_elem);

  // Check the device-side View's extents match
  auto dev_view = dv.view_device();
  RC_ASSERT(static_cast<int>(dev_view.extent(0)) == n_inner);
  RC_ASSERT(static_cast<int>(dev_view.extent(1)) == n_elem);
}

// ---------------------------------------------------------------------------
// Property 1c: allocate() produces LayoutLeft (stride(0) == 1)
// ---------------------------------------------------------------------------
RC_GTEST_PROP(FieldStorePropertyExtents,
              AllocateProducesLayoutLeft,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 1: Field_Store preserves extents and column-major layout
  const int n_inner = *genInner();
  const int n_elem = *genElem();

  Store store;
  auto view = store.allocate("layout_test", n_inner, n_elem);

  // LayoutLeft: stride of dimension 0 (inner/fastest) must be 1
  RC_ASSERT(view.stride(0) == 1);
  // LayoutLeft: stride of dimension 1 must equal n_inner
  RC_ASSERT(static_cast<int>(view.stride(1)) == n_inner);
}

// ---------------------------------------------------------------------------
// Property 1d: wrap() produces LayoutLeft (stride(0) == 1) on both sides
// ---------------------------------------------------------------------------
RC_GTEST_PROP(FieldStorePropertyExtents,
              WrapProducesLayoutLeft,
              ()) {
  // Feature: mpas-dycore-cpp-port, Property 1: Field_Store preserves extents and column-major layout
  const int n_inner = *genInner();
  const int n_elem = *genElem();

  std::vector<mpas::dycore::Scalar> buffer(
      static_cast<std::size_t>(n_inner) * static_cast<std::size_t>(n_elem), 0.0);

  Store store;
  auto dv = store.wrap("layout_wrap_test", buffer.data(), n_inner, n_elem);

  // Host side: LayoutLeft stride checks
  auto host_view = dv.view_host();
  RC_ASSERT(host_view.stride(0) == 1);
  RC_ASSERT(static_cast<int>(host_view.stride(1)) == n_inner);

  // Device side: LayoutLeft stride checks
  auto dev_view = dv.view_device();
  RC_ASSERT(dev_view.stride(0) == 1);
  RC_ASSERT(static_cast<int>(dev_view.stride(1)) == n_inner);
}
