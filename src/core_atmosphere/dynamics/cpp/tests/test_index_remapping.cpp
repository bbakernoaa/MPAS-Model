/// @file test_index_remapping.cpp
/// @brief Property-based test for index remapping correctness.
///
/// Feature: fortran-integration, Property 2: Index Remapping Correctness
///
/// **Validates: Requirements 2.3, 2.4**
///
/// For any integer array of Fortran 1-based connectivity indices, after the
/// marshalling layer applies the 0-based remapping, every element in the output
/// buffer SHALL equal the corresponding input element minus one.
///
/// This test exercises the pure remapping logic independently of the full dycore.
/// It uses std::mt19937 with a fixed seed for reproducible random generation.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

/// Pure index remapping function: converts Fortran 1-based indices to C 0-based.
/// This replicates the logic from the marshalling layer:
///   buffer(i) = original(i) - 1
void remap_indices(const std::vector<int>& input, std::vector<int>& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] - 1;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Feature: fortran-integration, Property 2: Index Remapping Correctness
// ---------------------------------------------------------------------------

/// Property test: For random arrays of Fortran 1-based indices, the remapping
/// produces output[i] == input[i] - 1 for all elements.
/// Uses 200 random iterations with varying array sizes and values.
TEST(IndexRemappingProperty, OutputEqualsInputMinusOne) {
    constexpr int num_iterations = 200;
    constexpr int min_array_size = 1;
    constexpr int max_array_size = 10000;
    constexpr int min_index_value = 1;      // Valid Fortran 1-based index (minimum)
    constexpr int max_index_value = 100000; // Valid Fortran 1-based index (maximum)
    constexpr unsigned int seed = 42;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> size_dist(min_array_size, max_array_size);
    std::uniform_int_distribution<int> value_dist(min_index_value, max_index_value);

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Generate a random array size
        const int array_size = size_dist(rng);

        // Generate random Fortran 1-based indices
        std::vector<int> input(array_size);
        for (int i = 0; i < array_size; ++i) {
            input[i] = value_dist(rng);
        }

        // Apply the remapping
        std::vector<int> output;
        remap_indices(input, output);

        // Verify: output size matches input size
        ASSERT_EQ(output.size(), input.size())
            << "Iteration " << iter << ": output size mismatch";

        // Verify: every element in output equals corresponding input minus one
        for (int i = 0; i < array_size; ++i) {
            ASSERT_EQ(output[i], input[i] - 1)
                << "Iteration " << iter << ", index " << i
                << ": expected " << (input[i] - 1)
                << " but got " << output[i];
        }
    }
}

/// Property test: Remapping preserves array length (no elements added or lost).
TEST(IndexRemappingProperty, PreservesArrayLength) {
    constexpr int num_iterations = 100;
    constexpr unsigned int seed = 123;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> size_dist(1, 10000);
    std::uniform_int_distribution<int> value_dist(1, 100000);

    for (int iter = 0; iter < num_iterations; ++iter) {
        const int array_size = size_dist(rng);

        std::vector<int> input(array_size);
        for (int i = 0; i < array_size; ++i) {
            input[i] = value_dist(rng);
        }

        std::vector<int> output;
        remap_indices(input, output);

        ASSERT_EQ(static_cast<int>(output.size()), array_size)
            << "Iteration " << iter << ": array length not preserved";
    }
}

/// Property test: Remapping produces valid 0-based indices (all elements >= 0)
/// when input contains valid Fortran 1-based indices (all elements >= 1).
TEST(IndexRemappingProperty, ProducesValidZeroBasedIndices) {
    constexpr int num_iterations = 100;
    constexpr unsigned int seed = 7;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> size_dist(1, 10000);
    std::uniform_int_distribution<int> value_dist(1, 100000);

    for (int iter = 0; iter < num_iterations; ++iter) {
        const int array_size = size_dist(rng);

        std::vector<int> input(array_size);
        for (int i = 0; i < array_size; ++i) {
            input[i] = value_dist(rng);
        }

        std::vector<int> output;
        remap_indices(input, output);

        for (int i = 0; i < array_size; ++i) {
            ASSERT_GE(output[i], 0)
                << "Iteration " << iter << ", index " << i
                << ": output value " << output[i]
                << " is negative (invalid 0-based index)";
        }
    }
}

/// Property test: Remapping is the inverse of adding one — applying remap then
/// adding 1 recovers the original array (round-trip identity).
TEST(IndexRemappingProperty, RoundTripWithAddOne) {
    constexpr int num_iterations = 100;
    constexpr unsigned int seed = 99;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> size_dist(1, 10000);
    std::uniform_int_distribution<int> value_dist(1, 100000);

    for (int iter = 0; iter < num_iterations; ++iter) {
        const int array_size = size_dist(rng);

        std::vector<int> input(array_size);
        for (int i = 0; i < array_size; ++i) {
            input[i] = value_dist(rng);
        }

        std::vector<int> output;
        remap_indices(input, output);

        // Adding 1 to every output element should recover the input
        for (int i = 0; i < array_size; ++i) {
            ASSERT_EQ(output[i] + 1, input[i])
                << "Iteration " << iter << ", index " << i
                << ": round-trip failed";
        }
    }
}
