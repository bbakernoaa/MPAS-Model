/// @file test_accessors.cpp
/// @brief Unit tests for checked_accessor and unchecked_accessor policies.

#include <gtest/gtest.h>
#include <mpas_dycore/types.hpp>
#include <array>
#include <csignal>
#include <cstdlib>

using namespace mpas::dycore;

// ============================================================================
// unchecked_accessor tests
// ============================================================================

TEST(UncheckedAccessor, DirectAccess) {
    std::array<double, 6> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    unchecked_accessor<double> acc;

    EXPECT_EQ(acc.access(data.data(), 0), 1.0);
    EXPECT_EQ(acc.access(data.data(), 3), 4.0);
    EXPECT_EQ(acc.access(data.data(), 5), 6.0);
}

TEST(UncheckedAccessor, OffsetReturnsCorrectPointer) {
    std::array<double, 6> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    unchecked_accessor<double> acc;

    double* offset_ptr = acc.offset(data.data(), 3);
    EXPECT_EQ(offset_ptr, data.data() + 3);
    EXPECT_EQ(*offset_ptr, 4.0);
}

TEST(UncheckedAccessor, DefaultConstructible) {
    unchecked_accessor<double> acc;
    (void)acc; // Should compile and be valid
    SUCCEED();
}

TEST(UncheckedAccessor, ConstructWithTotalSize) {
    // unchecked_accessor accepts total size for API compat but ignores it
    unchecked_accessor<double> acc(100);
    std::array<double, 4> data = {10.0, 20.0, 30.0, 40.0};
    EXPECT_EQ(acc.access(data.data(), 2), 30.0);
}

TEST(UncheckedAccessor, ConvertingConstructor) {
    unchecked_accessor<double> mutable_acc;
    unchecked_accessor<const double> const_acc(mutable_acc);
    (void)const_acc;
    SUCCEED();
}

// ============================================================================
// checked_accessor tests
// ============================================================================

TEST(CheckedAccessor, ValidAccessWithinBounds) {
    std::array<double, 6> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    checked_accessor<double> acc(6);

    EXPECT_EQ(acc.access(data.data(), 0), 1.0);
    EXPECT_EQ(acc.access(data.data(), 3), 4.0);
    EXPECT_EQ(acc.access(data.data(), 5), 6.0);
}

TEST(CheckedAccessor, OffsetReturnsCorrectPointer) {
    std::array<double, 6> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    checked_accessor<double> acc(6);

    double* offset_ptr = acc.offset(data.data(), 3);
    EXPECT_EQ(offset_ptr, data.data() + 3);
    EXPECT_EQ(*offset_ptr, 4.0);
}

TEST(CheckedAccessor, DefaultConstructedAllowsAccess) {
    // Default constructed (total_size_ == 0) means "no checking"
    std::array<double, 4> data = {10.0, 20.0, 30.0, 40.0};
    checked_accessor<double> acc;

    EXPECT_EQ(acc.access(data.data(), 2), 30.0);
}

TEST(CheckedAccessor, ConvertingConstructorPreservesTotalSize) {
    checked_accessor<double> mutable_acc(42);
    checked_accessor<const double> const_acc(mutable_acc);
    EXPECT_EQ(const_acc.total_size_, 42u);
}

TEST(CheckedAccessor, BoundsViolationAborts) {
    std::array<double, 4> data = {1.0, 2.0, 3.0, 4.0};
    checked_accessor<double> acc(4);

    // Accessing index 4 (== total_size_) should abort
    EXPECT_DEATH(acc.access(data.data(), 4),
        "MPAS Dycore bounds violation");
}

TEST(CheckedAccessor, BoundsViolationDiagnosticShowsValues) {
    std::array<double, 4> data = {1.0, 2.0, 3.0, 4.0};
    checked_accessor<double> acc(4);

    // Check that the diagnostic includes the specific index and extent values
    EXPECT_DEATH(acc.access(data.data(), 10),
        "linearized index 10 >= total extent 4");
}

TEST(CheckedAccessor, LastValidIndexSucceeds) {
    std::array<double, 8> data = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    checked_accessor<double> acc(8);

    // Index 7 is the last valid (total_size_ - 1)
    EXPECT_EQ(acc.access(data.data(), 7), 7.0);
}

// ============================================================================
// Compile-time accessor selection tests
// ============================================================================

TEST(DefaultAccessorPolicy, SelectsCorrectAccessor) {
    // This test verifies the compile-time selection.
    // When MPAS_BOUNDS_CHECK is defined, default_accessor_policy == checked_accessor.
    // When not defined, default_accessor_policy == unchecked_accessor.
#ifdef MPAS_BOUNDS_CHECK
    static_assert(std::is_same_v<default_accessor_policy<double>, checked_accessor<double>>,
        "With MPAS_BOUNDS_CHECK defined, default should be checked_accessor");
#else
    static_assert(std::is_same_v<default_accessor_policy<double>, unchecked_accessor<double>>,
        "Without MPAS_BOUNDS_CHECK, default should be unchecked_accessor");
#endif
    SUCCEED();
}

// ============================================================================
// mdspan integration tests
// ============================================================================

TEST(MdspanIntegration, Field2DWithCheckedAccessor) {
    // Create a 3x4 Field2D with checked_accessor
    std::array<double, 12> data{};
    for (int i = 0; i < 12; ++i) data[i] = static_cast<double>(i);

    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;
    using mapping_t = layout_left::mapping<extents_t>;
    using field_t = std::mdspan<real_type, extents_t, layout_left, checked_accessor<real_type>>;

    checked_accessor<real_type> acc(12);
    mapping_t mapping(extents_t{3, 4});
    field_t field(data.data(), mapping, acc);

    // With layout_left (column-major): element at [i,j] is at offset i + j*3
    EXPECT_EQ((field[0, 0]), 0.0);
    EXPECT_EQ((field[1, 0]), 1.0);
    EXPECT_EQ((field[2, 0]), 2.0);
    EXPECT_EQ((field[0, 1]), 3.0);
}

TEST(MdspanIntegration, Field2DWithUncheckedAccessor) {
    std::array<double, 12> data{};
    for (int i = 0; i < 12; ++i) data[i] = static_cast<double>(i * 10);

    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;
    using field_t = std::mdspan<real_type, extents_t, layout_left, unchecked_accessor<real_type>>;

    field_t field(data.data(), extents_t{3, 4});

    EXPECT_EQ((field[0, 0]), 0.0);
    EXPECT_EQ((field[1, 0]), 10.0);
    EXPECT_EQ((field[2, 3]), 110.0); // offset = 2 + 3*3 = 11
}

TEST(MdspanIntegration, Field2DDefaultAccessorCompiles) {
    // Verify the Field2D type alias works with the default accessor policy
    std::array<double, 6> data{};
    for (int i = 0; i < 6; ++i) data[i] = static_cast<double>(i);

    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;

    // Use the Field2D alias (which uses default_accessor_policy)
    Field2D<> field(data.data(), extents_t{2, 3});

    EXPECT_EQ((field[0, 0]), 0.0);
    EXPECT_EQ((field[1, 0]), 1.0);
}
