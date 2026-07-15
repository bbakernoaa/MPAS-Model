/**
 * @file main_interop.cpp
 * @brief Test runner for interop boundary tests.
 *
 * Unlike the main component_test_suite runner, this one does NOT pre-initialize
 * Kokkos, because the interop boundary tests exercise the full Kokkos lifecycle
 * through dycore_init / dycore_finalize.
 */

#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
