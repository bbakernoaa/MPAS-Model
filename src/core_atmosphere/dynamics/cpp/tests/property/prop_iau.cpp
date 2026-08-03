/// @file prop_iau.cpp
/// @brief Property-based tests for the Incremental Analysis Update (IAU) kernel.
///
/// **Validates: Requirements 23.2, 23.3**
///
/// Property 31: IAU Disabled No-Op
///   When iau_active=false, tendency is unchanged bit-for-bit.
///
/// Property 32: IAU Active Adds Increment
///   When active, tendency increases by increment / window_seconds exactly.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/iau.hpp>
#include <vector>
#include <cmath>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;

// ============================================================================
// Property 31: IAU Disabled No-Op
// ============================================================================
// When iau_active=false, tendency is unchanged bit-for-bit regardless of input values.

RC_GTEST_PROP(IAUDisabledNoOp, TendencyUnchangedWhenInactive,
              ()) {
    // Generate random mesh dimensions
    const auto nEntities = *rc::gen::inRange(1, 10);
    const auto nVertLevels = *rc::gen::inRange(1, 20);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEntities);

    // Generate a positive IAU window duration [1.0, 10000.0] seconds
    const real_type iau_window_seconds =
        1.0 + (*rc::gen::inRange(0, 9999)) * 1.0;

    // Fill tendency with known values
    std::vector<real_type> tendency_data(nLev * nE);
    for (std::size_t i = 0; i < tendency_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        tendency_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Save a copy before calling the kernel
    std::vector<real_type> tendency_original(tendency_data);

    // Generate arbitrary IAU increment values
    std::vector<real_type> increment_data(nLev * nE);
    for (std::size_t i = 0; i < increment_data.size(); ++i) {
        int raw = *rc::gen::inRange(-50000, 50001);
        increment_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tendency(
        tendency_data.data(), nVertLevels, nEntities);
    ConstField2D<default_layout, unchecked_accessor> iau_increment(
        increment_data.data(), nVertLevels, nEntities);

    // Call with iau_active=false
    apply_iau_tendency<default_layout>(
        SerialPolicy{},
        tendency, iau_increment,
        iau_window_seconds,
        false,  // iau_active = false
        static_cast<index_type>(nEntities),
        static_cast<index_type>(nVertLevels));

    // Verify tendency is unchanged bit-for-bit
    for (std::size_t i = 0; i < tendency_data.size(); ++i) {
        RC_ASSERT(tendency_data[i] == tendency_original[i]);
    }
}

// ============================================================================
// Property 32: IAU Active Adds Increment
// ============================================================================
// When active, tendency increases by increment / window_seconds exactly.

RC_GTEST_PROP(IAUActiveAddsIncrement, TendencyIncreasedByIncrementOverWindow,
              ()) {
    // Generate random mesh dimensions
    const auto nEntities = *rc::gen::inRange(1, 10);
    const auto nVertLevels = *rc::gen::inRange(1, 20);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEntities);

    // Generate a positive IAU window duration [10.0, 10000.0] seconds
    // Avoid very small values to keep floating-point well-conditioned
    const real_type iau_window_seconds =
        10.0 + (*rc::gen::inRange(0, 9990)) * 1.0;

    // Fill tendency with known values
    std::vector<real_type> tendency_data(nLev * nE);
    for (std::size_t i = 0; i < tendency_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        tendency_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Save a copy before calling the kernel
    std::vector<real_type> tendency_original(tendency_data);

    // Generate IAU increment values
    std::vector<real_type> increment_data(nLev * nE);
    for (std::size_t i = 0; i < increment_data.size(); ++i) {
        int raw = *rc::gen::inRange(-50000, 50001);
        increment_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tendency(
        tendency_data.data(), nVertLevels, nEntities);
    ConstField2D<default_layout, unchecked_accessor> iau_increment(
        increment_data.data(), nVertLevels, nEntities);

    // Call with iau_active=true
    apply_iau_tendency<default_layout>(
        SerialPolicy{},
        tendency, iau_increment,
        iau_window_seconds,
        true,  // iau_active = true
        static_cast<index_type>(nEntities),
        static_cast<index_type>(nVertLevels));

    // Verify tendency += increment / window_seconds for each element
    const real_type inv_window = 1.0 / iau_window_seconds;
    for (std::size_t i = 0; i < tendency_data.size(); ++i) {
        const real_type expected = tendency_original[i] + increment_data[i] * inv_window;
        RC_ASSERT(tendency_data[i] == expected);
    }
}

// ============================================================================
// Deterministic Test: Zero increment produces no change when active
// ============================================================================
// When iau_active=true but increment is all zeros, tendency remains unchanged.

TEST(IAUDeterministic, ZeroIncrementProducesNoChange) {
    const index_type nEntities = 5;
    const index_type nVertLevels = 8;
    const real_type iau_window_seconds = 3600.0;

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nE   = static_cast<std::size_t>(nEntities);

    // Non-zero tendency values
    std::vector<real_type> tendency_data(nLev * nE);
    for (std::size_t i = 0; i < tendency_data.size(); ++i) {
        tendency_data[i] = 1.0 + static_cast<real_type>(i) * 0.01;
    }

    // Save original
    std::vector<real_type> tendency_original(tendency_data);

    // Zero increment
    std::vector<real_type> increment_data(nLev * nE, 0.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> tendency(
        tendency_data.data(), nVertLevels, nEntities);
    ConstField2D<default_layout, unchecked_accessor> iau_increment(
        increment_data.data(), nVertLevels, nEntities);

    // Call with iau_active=true
    apply_iau_tendency<default_layout>(
        SerialPolicy{},
        tendency, iau_increment,
        iau_window_seconds,
        true,
        nEntities,
        nVertLevels);

    // Tendency should be unchanged since increment is zero
    for (std::size_t i = 0; i < tendency_data.size(); ++i) {
        EXPECT_DOUBLE_EQ(tendency_data[i], tendency_original[i])
            << "tendency changed at index " << i << " despite zero increment";
    }
}
