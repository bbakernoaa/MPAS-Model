/// @file prop_constants.cpp
/// @brief Property test: Named constants match MPAS physical constants.
///
/// **Validates: Requirements 24.2**
///
/// Verifies that all named constants in mpas::dycore::constants match the
/// expected MPAS Fortran values exactly (bit-for-bit double precision).
/// Also checks type correctness, positivity, and Mayer's relation consistency.

#include <gtest/gtest.h>
#include <mpas_dycore/constants.hpp>
#include <mpas_dycore/types.hpp>
#include <cmath>
#include <type_traits>

using namespace mpas::dycore;
using namespace mpas::dycore::constants;

// ============================================================================
// Property: Exact value verification (bit-for-bit match with Fortran MPAS)
// ============================================================================

TEST(PropConstants, GravityMatchesMPAS) {
    EXPECT_EQ(gravity, 9.80616);
}

TEST(PropConstants, RdryMatchesMPAS) {
    EXPECT_EQ(rdry, 287.0);
}

TEST(PropConstants, RvaporMatchesMPAS) {
    EXPECT_EQ(rvapor, 461.6);
}

TEST(PropConstants, CpdryMatchesMPAS) {
    EXPECT_EQ(cpdry, 7.0 * 287.0 / 2.0);
    EXPECT_EQ(cpdry, 1004.5);
}

TEST(PropConstants, CvdryMatchesMPAS) {
    EXPECT_EQ(cvdry, cpdry - rdry);
    EXPECT_EQ(cvdry, 717.5);
}

TEST(PropConstants, PrefMatchesMPAS) {
    EXPECT_EQ(pref, 1.0e5);
}

TEST(PropConstants, RearthMatchesMPAS) {
    EXPECT_EQ(rearth, 6371229.0);
}

TEST(PropConstants, PiMatchesMPAS) {
    EXPECT_EQ(pi, 3.141592653589793);
}

TEST(PropConstants, OmegaMatchesMPAS) {
    EXPECT_EQ(omega, 7.29212e-5);
}

TEST(PropConstants, SecondsPerDayMatchesMPAS) {
    EXPECT_EQ(seconds_per_day, 86400.0);
}

// ============================================================================
// Property: All constants are of type real_type (double)
// ============================================================================

TEST(PropConstants, AllConstantsAreRealType) {
    static_assert(std::is_same_v<decltype(gravity), const real_type>);
    static_assert(std::is_same_v<decltype(rdry), const real_type>);
    static_assert(std::is_same_v<decltype(rvapor), const real_type>);
    static_assert(std::is_same_v<decltype(cpdry), const real_type>);
    static_assert(std::is_same_v<decltype(cvdry), const real_type>);
    static_assert(std::is_same_v<decltype(pref), const real_type>);
    static_assert(std::is_same_v<decltype(rearth), const real_type>);
    static_assert(std::is_same_v<decltype(pi), const real_type>);
    static_assert(std::is_same_v<decltype(omega), const real_type>);
    static_assert(std::is_same_v<decltype(seconds_per_day), const real_type>);
    SUCCEED(); // static_asserts verified at compile time
}

// ============================================================================
// Property: Mayer's relation consistency (cpdry == cvdry + rdry)
// ============================================================================

TEST(PropConstants, MayersRelationConsistency) {
    EXPECT_EQ(cpdry, cvdry + rdry);
}

// ============================================================================
// Property: All constants are positive
// ============================================================================

TEST(PropConstants, AllConstantsArePositive) {
    EXPECT_GT(gravity, 0.0);
    EXPECT_GT(rdry, 0.0);
    EXPECT_GT(rvapor, 0.0);
    EXPECT_GT(cpdry, 0.0);
    EXPECT_GT(cvdry, 0.0);
    EXPECT_GT(pref, 0.0);
    EXPECT_GT(rearth, 0.0);
    EXPECT_GT(pi, 0.0);
    EXPECT_GT(omega, 0.0);
    EXPECT_GT(seconds_per_day, 0.0);
}
