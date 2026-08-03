/// @file prop_halo.cpp
/// @brief Property-based tests for halo exchange logic.
///
/// **Validates: Requirements 6.2, 6.3, 6.5, 6.6, 6.7**
///
/// These tests exercise the halo exchange without requiring multiple MPI ranks.
/// Property 15 uses MPI_COMM_SELF (single rank loopback) to test the full
/// pack→MPI→unpack pipeline. Property 16 tests the no-op behavior when
/// n_neighbors == 0.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/halo.hpp>
#include <mpas_dycore/types.hpp>
#include <mpi.h>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace mpas::dycore;

// ============================================================================
// MPI Environment for halo tests
// ============================================================================

class HaloTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override { MPI_Init(nullptr, nullptr); }
    void TearDown() override { MPI_Finalize(); }
};

auto* const halo_env = ::testing::AddGlobalTestEnvironment(new HaloTestEnvironment);

// ============================================================================
// Property 15: Halo Exchange Round-Trip
// ============================================================================
// Packing data from a field, performing exchange via MPI_COMM_SELF (loopback),
// and unpacking produces the correct values at destination indices.
//
// With MPI_COMM_SELF, n_neighbors=1, rank=0, and the process sends/receives
// to itself. This tests the full pack→MPI→unpack pipeline.

RC_GTEST_PROP(HaloExchangeRoundTrip, Field2DLayoutLeft, ()) {
    // Generate field dimensions
    const auto nVertLevels = *rc::gen::inRange(1, 21);
    const auto nEntities = *rc::gen::inRange(4, 31);
    const auto nLayers = *rc::gen::inRange(1, 4);

    // Generate number of entities to exchange per layer.
    // Total across all layers must not exceed nEntities (for unique recv indices).
    const auto maxTotal = std::max(1, nEntities / 2);
    std::vector<int> counts_vec(static_cast<std::size_t>(nLayers));
    int remaining = maxTotal;
    for (int j = 0; j < nLayers; ++j) {
        int maxForLayer = std::max(1, remaining / (nLayers - j));
        counts_vec[static_cast<std::size_t>(j)] = *rc::gen::inRange(1, maxForLayer + 1);
        remaining -= counts_vec[static_cast<std::size_t>(j)];
        if (remaining <= 0) {
            // Fill rest with zero-count layers (skip in processing)
            for (int jj = j + 1; jj < nLayers; ++jj)
                counts_vec[static_cast<std::size_t>(jj)] = 0;
            break;
        }
    }

    // Total entities exchanged across all layers
    int total_indices = 0;
    for (int c : counts_vec) total_indices += c;
    RC_PRE(total_indices > 0);
    RC_PRE(total_indices <= nEntities);

    // Generate send indices (may have duplicates — packing the same source twice is valid)
    std::vector<int> send_indices_vec;
    send_indices_vec.reserve(static_cast<std::size_t>(total_indices));
    for (int c = 0; c < total_indices; ++c) {
        send_indices_vec.push_back(*rc::gen::inRange(0, nEntities));
    }

    // Generate UNIQUE recv indices to avoid last-write-wins ambiguity.
    // Use rc::gen::unique to generate a set of distinct indices, then convert to vector.
    auto recv_set = *rc::gen::unique<std::vector<int>>(
        static_cast<std::size_t>(total_indices),
        rc::gen::inRange(0, nEntities));
    std::vector<int> recv_indices_vec(recv_set.begin(), recv_set.end());
    RC_PRE(static_cast<int>(recv_indices_vec.size()) == total_indices);

    // Set up CSRHaloDescriptor: 1 neighbor (self), rank 0
    int rank = 0;
    CSRHaloDescriptor desc{};
    desc.n_neighbors = 1;
    desc.n_layers = static_cast<index_type>(nLayers);
    desc.ranks = std::span<const int>(&rank, 1);
    desc.counts = std::span<const int>(counts_vec.data(), counts_vec.size());
    desc.send_indices = std::span<const int>(send_indices_vec.data(), send_indices_vec.size());
    desc.recv_indices = std::span<const int>(recv_indices_vec.data(), recv_indices_vec.size());

    // Create field with known data pattern: field[k, i] = k * 1000 + i
    const std::size_t total = static_cast<std::size_t>(nVertLevels) * nEntities;
    std::vector<real_type> data(total);
    for (index_type i = 0; i < nEntities; ++i) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            // layout_left: [k, i] -> k + i * nVertLevels
            data[static_cast<std::size_t>(k) + static_cast<std::size_t>(i) * nVertLevels] =
                static_cast<real_type>(k * 1000 + i);
        }
    }

    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;
    using field_t = Field2D<layout_left, unchecked_accessor>;
    field_t field(data.data(), extents_t{nVertLevels, nEntities});

    // Perform halo exchange with all layers
    std::vector<int> layers_vec(static_cast<std::size_t>(nLayers));
    std::iota(layers_vec.begin(), layers_vec.end(), 0);

    HaloExchange halo;
    halo.start(field, desc, std::span<const int>(layers_vec), MPI_COMM_SELF);
    halo.wait();

    // Verify: recv_indices should now hold values from send_indices
    int idx = 0;
    for (int j = 0; j < nLayers; ++j) {
        int count = counts_vec[static_cast<std::size_t>(j)];
        for (int c = 0; c < count; ++c) {
            int src_entity = send_indices_vec[static_cast<std::size_t>(idx)];
            int dst_entity = recv_indices_vec[static_cast<std::size_t>(idx)];
            for (index_type k = 0; k < nVertLevels; ++k) {
                real_type expected = static_cast<real_type>(k * 1000 + src_entity);
                real_type actual = field[k, static_cast<index_type>(dst_entity)];
                RC_ASSERT(actual == expected);
            }
            ++idx;
        }
    }
}

RC_GTEST_PROP(HaloExchangeRoundTrip, Field2DLayoutRight, ()) {
    // Generate field dimensions
    const auto nVertLevels = *rc::gen::inRange(1, 21);
    const auto nEntities = *rc::gen::inRange(4, 31);
    const auto nLayers = *rc::gen::inRange(1, 4);

    // Generate counts per layer with total not exceeding nEntities
    const auto maxTotal = std::max(1, nEntities / 2);
    std::vector<int> counts_vec(static_cast<std::size_t>(nLayers));
    int remaining = maxTotal;
    for (int j = 0; j < nLayers; ++j) {
        int maxForLayer = std::max(1, remaining / (nLayers - j));
        counts_vec[static_cast<std::size_t>(j)] = *rc::gen::inRange(1, maxForLayer + 1);
        remaining -= counts_vec[static_cast<std::size_t>(j)];
        if (remaining <= 0) {
            for (int jj = j + 1; jj < nLayers; ++jj)
                counts_vec[static_cast<std::size_t>(jj)] = 0;
            break;
        }
    }

    int total_indices = 0;
    for (int c : counts_vec) total_indices += c;
    RC_PRE(total_indices > 0);
    RC_PRE(total_indices <= nEntities);

    // Send indices (may have duplicates)
    std::vector<int> send_indices_vec;
    send_indices_vec.reserve(static_cast<std::size_t>(total_indices));
    for (int c = 0; c < total_indices; ++c) {
        send_indices_vec.push_back(*rc::gen::inRange(0, nEntities));
    }

    // Unique recv indices
    auto recv_set = *rc::gen::unique<std::vector<int>>(
        static_cast<std::size_t>(total_indices),
        rc::gen::inRange(0, nEntities));
    std::vector<int> recv_indices_vec(recv_set.begin(), recv_set.end());
    RC_PRE(static_cast<int>(recv_indices_vec.size()) == total_indices);

    int rank = 0;
    CSRHaloDescriptor desc{};
    desc.n_neighbors = 1;
    desc.n_layers = static_cast<index_type>(nLayers);
    desc.ranks = std::span<const int>(&rank, 1);
    desc.counts = std::span<const int>(counts_vec.data(), counts_vec.size());
    desc.send_indices = std::span<const int>(send_indices_vec.data(), send_indices_vec.size());
    desc.recv_indices = std::span<const int>(recv_indices_vec.data(), recv_indices_vec.size());

    // Create field with layout_right: [k, i] -> i + k * nEntities
    const std::size_t total = static_cast<std::size_t>(nVertLevels) * nEntities;
    std::vector<real_type> data(total);
    for (index_type i = 0; i < nEntities; ++i) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            data[static_cast<std::size_t>(i) + static_cast<std::size_t>(k) * nEntities] =
                static_cast<real_type>(k * 1000 + i);
        }
    }

    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;
    using field_t = Field2D<layout_right, unchecked_accessor>;
    field_t field(data.data(), extents_t{nVertLevels, nEntities});

    std::vector<int> layers_vec(static_cast<std::size_t>(nLayers));
    std::iota(layers_vec.begin(), layers_vec.end(), 0);

    HaloExchange halo;
    halo.start(field, desc, std::span<const int>(layers_vec), MPI_COMM_SELF);
    halo.wait();

    // Verify: recv positions hold values from send positions
    int idx = 0;
    for (int j = 0; j < nLayers; ++j) {
        int count = counts_vec[static_cast<std::size_t>(j)];
        for (int c = 0; c < count; ++c) {
            int src_entity = send_indices_vec[static_cast<std::size_t>(idx)];
            int dst_entity = recv_indices_vec[static_cast<std::size_t>(idx)];
            for (index_type k = 0; k < nVertLevels; ++k) {
                real_type expected = static_cast<real_type>(k * 1000 + src_entity);
                real_type actual = field[k, static_cast<index_type>(dst_entity)];
                RC_ASSERT(actual == expected);
            }
            ++idx;
        }
    }
}

RC_GTEST_PROP(HaloExchangeRoundTrip, Field3DLayoutLeft, ()) {
    // Generate dimensions
    const auto nScalars = *rc::gen::inRange(1, 6);
    const auto nVertLevels = *rc::gen::inRange(1, 11);
    const auto nEntities = *rc::gen::inRange(4, 21);
    const auto nLayers = *rc::gen::inRange(1, 4);

    // Generate counts with total not exceeding nEntities
    const auto maxTotal = std::max(1, nEntities / 2);
    std::vector<int> counts_vec(static_cast<std::size_t>(nLayers));
    int remaining = maxTotal;
    for (int j = 0; j < nLayers; ++j) {
        int maxForLayer = std::max(1, remaining / (nLayers - j));
        counts_vec[static_cast<std::size_t>(j)] = *rc::gen::inRange(1, maxForLayer + 1);
        remaining -= counts_vec[static_cast<std::size_t>(j)];
        if (remaining <= 0) {
            for (int jj = j + 1; jj < nLayers; ++jj)
                counts_vec[static_cast<std::size_t>(jj)] = 0;
            break;
        }
    }

    int total_indices = 0;
    for (int c : counts_vec) total_indices += c;
    RC_PRE(total_indices > 0);
    RC_PRE(total_indices <= nEntities);

    // Send indices (may have duplicates)
    std::vector<int> send_indices_vec;
    send_indices_vec.reserve(static_cast<std::size_t>(total_indices));
    for (int c = 0; c < total_indices; ++c) {
        send_indices_vec.push_back(*rc::gen::inRange(0, nEntities));
    }

    // Unique recv indices
    auto recv_set = *rc::gen::unique<std::vector<int>>(
        static_cast<std::size_t>(total_indices),
        rc::gen::inRange(0, nEntities));
    std::vector<int> recv_indices_vec(recv_set.begin(), recv_set.end());
    RC_PRE(static_cast<int>(recv_indices_vec.size()) == total_indices);

    int rank = 0;
    CSRHaloDescriptor desc{};
    desc.n_neighbors = 1;
    desc.n_layers = static_cast<index_type>(nLayers);
    desc.ranks = std::span<const int>(&rank, 1);
    desc.counts = std::span<const int>(counts_vec.data(), counts_vec.size());
    desc.send_indices = std::span<const int>(send_indices_vec.data(), send_indices_vec.size());
    desc.recv_indices = std::span<const int>(recv_indices_vec.data(), recv_indices_vec.size());

    // Create 3D field with layout_left: [s, k, i] -> s + k*nScalars + i*nScalars*nVertLevels
    const std::size_t total = static_cast<std::size_t>(nScalars) * nVertLevels * nEntities;
    std::vector<real_type> data(total);
    for (index_type i = 0; i < nEntities; ++i) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            for (index_type s = 0; s < nScalars; ++s) {
                std::size_t offset = static_cast<std::size_t>(s)
                    + static_cast<std::size_t>(k) * nScalars
                    + static_cast<std::size_t>(i) * nScalars * nVertLevels;
                data[offset] = static_cast<real_type>(s * 100000 + k * 1000 + i);
            }
        }
    }

    using extents_t = std::extents<index_type, std::dynamic_extent,
                                   std::dynamic_extent, std::dynamic_extent>;
    using field_t = Field3D<layout_left, unchecked_accessor>;
    field_t field(data.data(), extents_t{nScalars, nVertLevels, nEntities});

    std::vector<int> layers_vec(static_cast<std::size_t>(nLayers));
    std::iota(layers_vec.begin(), layers_vec.end(), 0);

    HaloExchange halo;
    halo.start(field, desc, std::span<const int>(layers_vec), MPI_COMM_SELF);
    halo.wait();

    // Verify: recv positions hold values from send positions
    int idx = 0;
    for (int j = 0; j < nLayers; ++j) {
        int count = counts_vec[static_cast<std::size_t>(j)];
        for (int c = 0; c < count; ++c) {
            int src_entity = send_indices_vec[static_cast<std::size_t>(idx)];
            int dst_entity = recv_indices_vec[static_cast<std::size_t>(idx)];
            for (index_type s = 0; s < nScalars; ++s) {
                for (index_type k = 0; k < nVertLevels; ++k) {
                    real_type expected = static_cast<real_type>(s * 100000 + k * 1000 + src_entity);
                    real_type actual = field[s, k, static_cast<index_type>(dst_entity)];
                    RC_ASSERT(actual == expected);
                }
            }
            ++idx;
        }
    }
}

// ============================================================================
// Property 16: Single-Process Halo No-Op
// ============================================================================
// When n_neighbors == 0, start()/wait() are no-ops that don't modify the field,
// and is_noop() returns true.

RC_GTEST_PROP(HaloNoOp, Field2DUnchangedWhenNoNeighbors, ()) {
    // Generate field dimensions
    const auto nVertLevels = *rc::gen::inRange(1, 51);
    const auto nEntities = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(nVertLevels) * nEntities;

    // Generate random field data
    std::vector<real_type> data(total);
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = *rc::gen::arbitrary<real_type>();
    }

    // Save original data for comparison
    std::vector<real_type> original(data);

    // Create descriptor with zero neighbors (single-process)
    CSRHaloDescriptor desc{};
    desc.n_neighbors = 0;
    desc.n_layers = 1;
    desc.ranks = std::span<const int>{};
    desc.counts = std::span<const int>{};
    desc.send_indices = std::span<const int>{};
    desc.recv_indices = std::span<const int>{};

    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;
    using field_t = Field2D<layout_left, unchecked_accessor>;
    field_t field(data.data(), extents_t{nVertLevels, nEntities});

    std::vector<int> layers_vec = {0};

    HaloExchange halo;
    halo.start(field, desc, std::span<const int>(layers_vec), MPI_COMM_SELF);
    halo.wait();

    // Verify: field is completely unchanged (bit-for-bit)
    for (std::size_t i = 0; i < total; ++i) {
        RC_ASSERT(data[i] == original[i]);
    }

    // Verify: is_noop() returns true
    RC_ASSERT(halo.is_noop());
}

RC_GTEST_PROP(HaloNoOp, Field3DUnchangedWhenNoNeighbors, ()) {
    // Generate field dimensions
    const auto nScalars = *rc::gen::inRange(1, 6);
    const auto nVertLevels = *rc::gen::inRange(1, 21);
    const auto nEntities = *rc::gen::inRange(1, 31);
    const std::size_t total = static_cast<std::size_t>(nScalars) * nVertLevels * nEntities;

    // Generate random field data
    std::vector<real_type> data(total);
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = *rc::gen::arbitrary<real_type>();
    }

    // Save original data
    std::vector<real_type> original(data);

    // Zero neighbors descriptor
    CSRHaloDescriptor desc{};
    desc.n_neighbors = 0;
    desc.n_layers = 1;
    desc.ranks = std::span<const int>{};
    desc.counts = std::span<const int>{};
    desc.send_indices = std::span<const int>{};
    desc.recv_indices = std::span<const int>{};

    using extents_t = std::extents<index_type, std::dynamic_extent,
                                   std::dynamic_extent, std::dynamic_extent>;
    using field_t = Field3D<layout_left, unchecked_accessor>;
    field_t field(data.data(), extents_t{nScalars, nVertLevels, nEntities});

    std::vector<int> layers_vec = {0};

    HaloExchange halo;
    halo.start(field, desc, std::span<const int>(layers_vec), MPI_COMM_SELF);
    halo.wait();

    // Verify: field is completely unchanged (bit-for-bit)
    for (std::size_t i = 0; i < total; ++i) {
        RC_ASSERT(data[i] == original[i]);
    }

    // Verify: is_noop() returns true
    RC_ASSERT(halo.is_noop());
}

RC_GTEST_PROP(HaloNoOp, IsNoopReturnsTrueForZeroNeighbors, ()) {
    const auto nVertLevels = *rc::gen::inRange(1, 21);
    const auto nEntities = *rc::gen::inRange(1, 51);
    const std::size_t total = static_cast<std::size_t>(nVertLevels) * nEntities;

    std::vector<real_type> data(total, 1.0);

    CSRHaloDescriptor desc{};
    desc.n_neighbors = 0;
    desc.n_layers = *rc::gen::inRange<index_type>(1, 4);
    desc.ranks = std::span<const int>{};
    desc.counts = std::span<const int>{};
    desc.send_indices = std::span<const int>{};
    desc.recv_indices = std::span<const int>{};

    using extents_t = std::extents<index_type, std::dynamic_extent, std::dynamic_extent>;
    using field_t = Field2D<layout_left, unchecked_accessor>;
    field_t field(data.data(), extents_t{nVertLevels, nEntities});

    std::vector<int> layers_vec = {0};

    HaloExchange halo;
    halo.start(field, desc, std::span<const int>(layers_vec), MPI_COMM_SELF);

    // is_noop() should be true immediately after start() with zero neighbors
    RC_ASSERT(halo.is_noop());

    halo.wait();

    // Still true after wait()
    RC_ASSERT(halo.is_noop());
}
