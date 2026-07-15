#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

// Sanity Unit Test
TEST(PlaceholderTest, SanityCheck) {
    EXPECT_EQ(1 + 1, 2);
}

// RapidCheck Property-Based Test
RC_GTEST_PROP(PlaceholderPropertyTest, AdditionIsCommutative, (int a, int b)) {
    RC_ASSERT(a + b == b + a);
}
