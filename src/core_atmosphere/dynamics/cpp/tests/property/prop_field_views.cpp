/// @file prop_field_views.cpp
/// @brief Property-based tests for field view construction and accessor policies.
///
/// **Validates: Requirements 1.1, 1.2, 1.3, 1.5, 1.6, 1.7**

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/types.hpp>

#include <vector>
#include <numeric>
#include <cstddef>

using namespace mpas::dycore;

// ============================================================================
// Property 1: Field View Construction and Access
// ============================================================================
// For any valid dimensions and data, constructing a Field2D/Field3D and
// accessing elements returns the correct values.

RC_GTEST_PROP(FieldViewConstruction, Field2DAccessReturnsCorrectValues,
              ()) {
    // Generate random dimensions within task-specified ranges
    const auto nVertLevels = *rc::gen::inRange(1, 51);
    const auto nEntities = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(nVertLevels) * nEntities;

    // Fill data with index-based pattern
    std::vector<real_type> data(total);
    std::iota(data.begin(), data.end(), 0.0);

    // Construct Field2D with layout_left (default) and unchecked_accessor
    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;
    using field_t = Field2D<layout_left, unchecked_accessor>;

    field_t field(data.data(), extents_t{nVertLevels, nEntities});

    // Verify all elements via coordinate access
    // layout_left: element [k, i] maps to offset k + i * nVertLevels
    for (index_type i = 0; i < nEntities; ++i) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            const std::size_t expected_offset = static_cast<std::size_t>(k)
                + static_cast<std::size_t>(i) * nVertLevels;
            auto val = field[k, i];
            RC_ASSERT(val == static_cast<real_type>(expected_offset));
        }
    }
}

RC_GTEST_PROP(FieldViewConstruction, Field3DAccessReturnsCorrectValues,
              ()) {
    // Generate random dimensions
    const auto nScalars = *rc::gen::inRange(1, 11);
    const auto nVertLevels = *rc::gen::inRange(1, 51);
    const auto nEntities = *rc::gen::inRange(1, 51);
    const std::size_t total = static_cast<std::size_t>(nScalars) * nVertLevels * nEntities;

    // Fill data with index-based pattern
    std::vector<real_type> data(total);
    std::iota(data.begin(), data.end(), 0.0);

    // Construct Field3D with layout_left and unchecked_accessor
    using extents_t = std::extents<index_type, std::dynamic_extent,
                                   std::dynamic_extent, std::dynamic_extent>;
    using field_t = Field3D<layout_left, unchecked_accessor>;

    field_t field(data.data(), extents_t{nScalars, nVertLevels, nEntities});

    // Verify all elements via coordinate access
    // layout_left (column-major): [s, k, i] -> offset = s + k*nScalars + i*nScalars*nVertLevels
    for (index_type i = 0; i < nEntities; ++i) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            for (index_type s = 0; s < nScalars; ++s) {
                const std::size_t expected_offset =
                    static_cast<std::size_t>(s)
                    + static_cast<std::size_t>(k) * nScalars
                    + static_cast<std::size_t>(i) * nScalars * nVertLevels;
                auto val = field[s, k, i];
                RC_ASSERT(val == static_cast<real_type>(expected_offset));
            }
        }
    }
}

// ============================================================================
// Property 2: Layout and Accessor Transparency
// ============================================================================
// The same data accessed through different layouts produces consistent results
// (same element via coordinate mapping).

RC_GTEST_PROP(LayoutTransparency, LayoutLeftAndRightReturnCorrectElements,
              ()) {
    const auto nVertLevels = *rc::gen::inRange(1, 51);
    const auto nEntities = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(nVertLevels) * nEntities;

    // Generate random field data
    auto data_values = *rc::gen::container<std::vector<real_type>>(
        total, rc::gen::arbitrary<real_type>());

    // We'll store data in two separate buffers with different layouts
    // and verify that logical coordinate access returns the correct element.
    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;

    // For layout_left: physical offset of [k, i] = k + i * nVertLevels
    std::vector<real_type> data_left(total);
    // For layout_right: physical offset of [k, i] = k * nEntities + i
    std::vector<real_type> data_right(total);

    // Fill each buffer so that logical[k, i] holds the same semantic value
    for (index_type i = 0; i < nEntities; ++i) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            const std::size_t idx = static_cast<std::size_t>(k) * nEntities + i;
            // Use a deterministic value based on logical coordinates
            const real_type value = static_cast<real_type>(k * 1000 + i);
            // Place in layout_left buffer at physical position
            data_left[static_cast<std::size_t>(k) + static_cast<std::size_t>(i) * nVertLevels] = value;
            // Place in layout_right buffer at physical position
            data_right[static_cast<std::size_t>(k) * nEntities + i] = value;
        }
    }

    using field_left_t = Field2D<layout_left, unchecked_accessor>;
    using field_right_t = Field2D<layout_right, unchecked_accessor>;

    field_left_t field_left(data_left.data(), extents_t{nVertLevels, nEntities});
    field_right_t field_right(data_right.data(), extents_t{nVertLevels, nEntities});

    // Both layouts must return the same logical value for each coordinate pair
    for (index_type i = 0; i < nEntities; ++i) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            auto val_left = field_left[k, i];
            auto val_right = field_right[k, i];
            RC_ASSERT(val_left == val_right);
        }
    }
}

// ============================================================================
// Property 3: Bounds Checking Detection
// ============================================================================
// checked_accessor with known size stores correct total_size and valid indices
// pass without issue.

RC_GTEST_PROP(BoundsChecking, CheckedAccessorStoresCorrectTotalSize,
              ()) {
    const auto total_size = *rc::gen::inRange<std::size_t>(1, 10001);
    checked_accessor<real_type> acc(total_size);
    RC_ASSERT(acc.total_size_ == total_size);
}

RC_GTEST_PROP(BoundsChecking, ValidIndicesPassCheckedAccessor,
              ()) {
    const auto total_size = *rc::gen::inRange<std::size_t>(1, 1001);
    const auto index = *rc::gen::inRange<std::size_t>(0, total_size);

    std::vector<real_type> data(total_size);
    std::iota(data.begin(), data.end(), 0.0);

    checked_accessor<real_type> acc(total_size);
    // Valid index access should succeed and return correct value
    RC_ASSERT(acc.access(data.data(), index) == static_cast<real_type>(index));
}

RC_GTEST_PROP(BoundsChecking, Field2DWithCheckedAccessorValidAccess,
              ()) {
    const auto nVertLevels = *rc::gen::inRange(1, 51);
    const auto nEntities = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(nVertLevels) * nEntities;

    std::vector<real_type> data(total);
    std::iota(data.begin(), data.end(), 0.0);

    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;
    using mapping_t = layout_left::mapping<extents_t>;

    checked_accessor<real_type> acc(total);
    mapping_t mapping(extents_t{nVertLevels, nEntities});
    std::mdspan<real_type, extents_t, layout_left, checked_accessor<real_type>>
        field(data.data(), mapping, acc);

    // Pick a random valid coordinate and verify access succeeds
    const auto k = *rc::gen::inRange(0, nVertLevels);
    const auto i = *rc::gen::inRange(0, nEntities);

    // layout_left: [k, i] -> k + i * nVertLevels
    const std::size_t expected_offset = static_cast<std::size_t>(k)
        + static_cast<std::size_t>(i) * nVertLevels;
    auto val = field[k, i];
    RC_ASSERT(val == static_cast<real_type>(expected_offset));
}

// Death test: out-of-bounds access with checked_accessor aborts
TEST(BoundsCheckingDeath, OutOfBoundsAccessAborts) {
    constexpr std::size_t total_size = 20;
    std::vector<real_type> data(total_size, 1.0);
    checked_accessor<real_type> acc(total_size);

    // Access at index == total_size should trigger abort
    EXPECT_DEATH(acc.access(data.data(), total_size),
        "MPAS Dycore bounds violation");
}

TEST(BoundsCheckingDeath, LargeOutOfBoundsIndexAborts) {
    constexpr std::size_t total_size = 10;
    std::vector<real_type> data(total_size, 1.0);
    checked_accessor<real_type> acc(total_size);

    // Access at index much larger than total_size should trigger abort
    EXPECT_DEATH(acc.access(data.data(), 999),
        "MPAS Dycore bounds violation");
}
