/// @file test_field_store_property_dualview.cpp
/// @brief Property-based test for DualView zero-copy host aliasing and device sync round-trip.
///
/// Feature: mpas-dycore-cpp-port, Property 2: DualView zero-copy host aliasing and device sync
/// round-trip
///
/// **Validates: Requirements 1.9, 1.4, 1.11, 1.12**
///
/// For randomly generated dimensions and random data:
/// 1. After wrap(name, fortran_ptr, n_inner, n_elem), the DualView's host mirror data pointer
///    IS the original fortran_ptr (zero-copy aliasing).
/// 2. After writing values to the Fortran pointer directly and calling sync_to_device, the
///    device side reflects those values.
/// 3. After modifying device data (via a Kokkos kernel) and calling sync_to_host, the original
///    Fortran pointer array reflects the device modifications.
/// 4. On Field_Store destruction, the Fortran pointer's memory is NOT freed (caller retains
///    ownership).

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include "mpas_dycore/field_store.hpp"

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using Store = mpas::dycore::Field_Store<mpas::dycore::Scalar, ExecSpace>;
using Scalar = mpas::dycore::Scalar;

/// Parity tolerance used for floating-point comparisons.
constexpr Scalar parity_tolerance = std::is_same_v<Scalar, double> ? 1e-12 : 1e-6f;

}  // namespace

// ---------------------------------------------------------------------------
// Property 2: DualView zero-copy host aliasing and device sync round-trip
// Feature: mpas-dycore-cpp-port, Property 2: DualView zero-copy host aliasing and device sync
// round-trip
// ---------------------------------------------------------------------------

RC_GTEST_PROP(FieldStorePropertyDualView,
              ZeroCopyHostAliasingAndSyncRoundTrip,
              ()) {
    // Generate random dimensions: n_inner in [1, 100], n_elem in [1, 200]
    const int n_inner = *rc::gen::inRange(1, 101);
    const int n_elem = *rc::gen::inRange(1, 201);
    const int total = n_inner * n_elem;

    // Allocate a Fortran-like buffer (caller-owned, simulating Fortran allocation)
    std::vector<Scalar> fortran_buffer(total);

    // Fill with random initial values
    for (int idx = 0; idx < total; ++idx) {
        fortran_buffer[idx] = *rc::gen::arbitrary<Scalar>();
    }

    Scalar* fortran_ptr = fortran_buffer.data();

    // --- Sub-property 1: Zero-copy host aliasing ---
    // After wrap(), the DualView's host mirror data pointer IS the original fortran_ptr.
    {
        Store store;
        auto dv = store.wrap("test_field", fortran_ptr, n_inner, n_elem);

        // The host mirror's data() must be the same pointer as fortran_ptr.
        Scalar* host_data = dv.view_host().data();
        RC_ASSERT(host_data == fortran_ptr);

        // Verify that reading through the DualView host mirror gives the same values
        // as reading through the original Fortran pointer.
        for (int j = 0; j < n_elem; ++j) {
            for (int i = 0; i < n_inner; ++i) {
                // LayoutLeft: linear offset = i + n_inner * j
                Scalar expected = fortran_ptr[i + n_inner * j];
                Scalar actual = dv.view_host()(i, j);
                RC_ASSERT(actual == expected);
            }
        }
    }

    // --- Sub-property 2: sync_to_device propagates host writes to device ---
    // After writing values to the Fortran pointer directly and calling sync_to_device,
    // the device side reflects those values.
    {
        Store store;
        auto dv = store.wrap("test_field", fortran_ptr, n_inner, n_elem);

        // Write new random values directly to the Fortran pointer
        std::vector<Scalar> expected_values(total);
        for (int idx = 0; idx < total; ++idx) {
            Scalar val = *rc::gen::arbitrary<Scalar>();
            fortran_ptr[idx] = val;
            expected_values[idx] = val;
        }

        // Sync host -> device
        store.sync_to_device("test_field");
        Kokkos::fence("sync_to_device fence");

        // Read back from device and verify
        auto device_view = dv.view_device();
        auto host_check = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, device_view);

        for (int j = 0; j < n_elem; ++j) {
            for (int i = 0; i < n_inner; ++i) {
                Scalar expected = expected_values[i + n_inner * j];
                Scalar actual = host_check(i, j);
                RC_ASSERT(actual == expected);
            }
        }
    }

    // --- Sub-property 3: sync_to_host propagates device modifications back ---
    // After modifying device data (via a Kokkos kernel) and calling sync_to_host,
    // the original Fortran pointer array reflects the device modifications.
    {
        Store store;

        // Reset fortran buffer to known values
        for (int idx = 0; idx < total; ++idx) {
            fortran_ptr[idx] = static_cast<Scalar>(idx);
        }

        auto dv = store.wrap("test_field", fortran_ptr, n_inner, n_elem);

        // Modify device data via a Kokkos kernel: multiply each element by 2.0 + 1.0
        auto device_view = dv.view_device();
        Kokkos::parallel_for(
            "ModifyDeviceData",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(
                {0, 0}, {n_inner, n_elem}),
            KOKKOS_LAMBDA(int i, int j) {
                device_view(i, j) = device_view(i, j) * Scalar{2.0} + Scalar{1.0};
            });
        Kokkos::fence("ModifyDeviceData fence");

        // Sync device -> host
        store.sync_to_host("test_field");

        // Verify the Fortran pointer now reflects the device modifications
        for (int j = 0; j < n_elem; ++j) {
            for (int i = 0; i < n_inner; ++i) {
                Scalar original = static_cast<Scalar>(i + n_inner * j);
                Scalar expected = original * Scalar{2.0} + Scalar{1.0};
                Scalar actual = fortran_ptr[i + n_inner * j];
                // Use tolerance for floating-point comparison
                Scalar diff = std::abs(actual - expected);
                Scalar scale = std::max(Scalar{1.0}, std::abs(expected));
                RC_ASSERT(diff / scale <= parity_tolerance);
            }
        }
    }

    // --- Sub-property 4: Field_Store destruction does NOT free the Fortran pointer ---
    // On Field_Store destruction, the Fortran pointer's memory is NOT freed
    // (caller retains ownership).
    {
        // Write a sentinel pattern into the buffer
        for (int idx = 0; idx < total; ++idx) {
            fortran_ptr[idx] = static_cast<Scalar>(42.0 + idx);
        }

        {
            Store store;
            store.wrap("test_field", fortran_ptr, n_inner, n_elem);
            // store goes out of scope here — destructor runs
        }

        // After Field_Store destruction, the Fortran buffer must still be valid
        // and contain the sentinel values (memory not freed).
        for (int idx = 0; idx < total; ++idx) {
            Scalar expected = static_cast<Scalar>(42.0 + idx);
            RC_ASSERT(fortran_ptr[idx] == expected);
        }
    }
}
