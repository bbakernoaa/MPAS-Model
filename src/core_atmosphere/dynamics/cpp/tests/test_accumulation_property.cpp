/// @file test_accumulation_property.cpp
/// @brief Property-based test for deterministic unstructured accumulation.
///
/// Feature: mpas-dycore-cpp-port, Property 8: Deterministic unstructured accumulation
///
/// **Validates: Requirements 2.6**
///
/// For randomly generated unstructured meshes (random connectivity):
/// 1. Determinism: Running the ColoredAccumulator twice with the same inputs produces
///    bit-for-bit identical results.
/// 2. Correctness: The ColoredAccumulator result matches a serial reference sum (within
///    tolerance for floating point).
/// 3. TeamDuplicationAccumulator determinism: Same property for the fallback strategy.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include "mpas_dycore/accumulation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using Scalar = mpas::dycore::Scalar;

/// Tolerance for floating-point comparison in correctness checks.
/// We use a relative tolerance since accumulation of random values can magnify errors.
constexpr Scalar tolerance = std::is_same_v<Scalar, double> ? 1e-10 : 1e-4f;

/// Generate a random unstructured mesh connectivity (cellsOnEdge style).
/// Returns the cellsOnEdge host data in column-major order (LayoutLeft: (2, n_edges)).
struct RandomMesh {
    int n_cells;
    int n_edges;
    // cellsOnEdge: for each edge, up to 2 cell neighbors. -1 means boundary (no target).
    // Stored as LayoutLeft: index (slot, edge) -> slot + 2 * edge
    std::vector<int> cells_on_edge;
    // Random edge values to accumulate
    std::vector<Scalar> edge_values;
};

/// RapidCheck generator for RandomMesh.
RandomMesh generateRandomMesh() {
    RandomMesh mesh;
    mesh.n_cells = *rc::gen::inRange(2, 51);  // [2, 50]
    mesh.n_edges = *rc::gen::inRange(1, 101); // [1, 100]

    mesh.cells_on_edge.resize(2 * mesh.n_edges);
    for (int e = 0; e < mesh.n_edges; ++e) {
        // Each edge connects to 1 or 2 valid cells
        int n_neighbors = *rc::gen::inRange(1, 3); // 1 or 2
        // First neighbor: always valid
        mesh.cells_on_edge[0 + 2 * e] = *rc::gen::inRange(0, mesh.n_cells);
        if (n_neighbors == 2) {
            mesh.cells_on_edge[1 + 2 * e] = *rc::gen::inRange(0, mesh.n_cells);
        } else {
            mesh.cells_on_edge[1 + 2 * e] = -1; // boundary
        }
    }

    // Random edge values
    mesh.edge_values.resize(mesh.n_edges);
    for (int e = 0; e < mesh.n_edges; ++e) {
        // Use values in a reasonable range to avoid overflow
        double raw = *rc::gen::inRange(-1000, 1001);
        mesh.edge_values[e] = static_cast<Scalar>(raw * 0.01);
    }

    return mesh;
}

/// Compute a serial reference accumulation on the host.
/// For each edge, adds edge_values[e] to each valid cell neighbor.
std::vector<Scalar> serialReferenceAccumulation(const RandomMesh& mesh) {
    std::vector<Scalar> result(mesh.n_cells, Scalar{0});
    for (int e = 0; e < mesh.n_edges; ++e) {
        for (int slot = 0; slot < 2; ++slot) {
            int cell = mesh.cells_on_edge[slot + 2 * e];
            if (cell >= 0 && cell < mesh.n_cells) {
                result[cell] += mesh.edge_values[e];
            }
        }
    }
    return result;
}

/// Helper struct that holds device views and a ColoredAccumulator built from a RandomMesh.
/// The accumulator (and its coloring) is constructed once so that repeated calls to
/// `run()` exercise the property: "for a given coloring, repeated accumulations are
/// bit-for-bit identical."
struct ColoredAccumulatorFixture {
    int n_cells;
    int n_edges;
    Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_dev;
    Kokkos::View<Scalar*, MemSpace> edge_values_dev;
    mpas::dycore::ColoredAccumulator<ExecSpace, Scalar>* accum_ptr = nullptr;

    ColoredAccumulatorFixture(const RandomMesh& mesh)
        : n_cells(mesh.n_cells), n_edges(mesh.n_edges) {
        // Create cellsOnEdge view on device: shape (2, n_edges) LayoutLeft
        cells_on_edge_dev = Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace>(
            "cells_on_edge", 2, n_edges);
        {
            auto host_mirror = Kokkos::create_mirror_view(cells_on_edge_dev);
            for (int e = 0; e < n_edges; ++e) {
                host_mirror(0, e) = mesh.cells_on_edge[0 + 2 * e];
                host_mirror(1, e) = mesh.cells_on_edge[1 + 2 * e];
            }
            Kokkos::deep_copy(cells_on_edge_dev, host_mirror);
        }

        // Create edge_values view on device
        edge_values_dev = Kokkos::View<Scalar*, MemSpace>("edge_values", n_edges);
        {
            auto host_mirror = Kokkos::create_mirror_view(edge_values_dev);
            for (int e = 0; e < n_edges; ++e) {
                host_mirror(e) = mesh.edge_values[e];
            }
            Kokkos::deep_copy(edge_values_dev, host_mirror);
        }

        // Build the ColoredAccumulator ONCE
        Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_dev;
        accum_ptr = new mpas::dycore::ColoredAccumulator<ExecSpace, Scalar>(
            n_edges, n_cells, cells_on_edge_const);
    }

    ~ColoredAccumulatorFixture() { delete accum_ptr; }

    /// Run the accumulation and return the result as a host vector.
    std::vector<Scalar> run() const {
        Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
        Kokkos::deep_copy(output, Scalar{0});

        auto ev = edge_values_dev;
        accum_ptr->accumulate(output, KOKKOS_LAMBDA(int e, int /*slot*/) -> Scalar {
            return ev(e);
        });
        Kokkos::fence("accumulate fence");

        auto host_output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
        std::vector<Scalar> result(n_cells);
        for (int c = 0; c < n_cells; ++c) {
            result[c] = host_output(c);
        }
        return result;
    }
};

/// Helper struct that holds device views and a TeamDuplicationAccumulator.
struct TeamDuplicationFixture {
    int n_cells;
    int n_edges;
    Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_dev;
    Kokkos::View<Scalar*, MemSpace> edge_values_dev;
    mpas::dycore::TeamDuplicationAccumulator<ExecSpace, Scalar>* accum_ptr = nullptr;

    TeamDuplicationFixture(const RandomMesh& mesh)
        : n_cells(mesh.n_cells), n_edges(mesh.n_edges) {
        cells_on_edge_dev = Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace>(
            "cells_on_edge", 2, n_edges);
        {
            auto host_mirror = Kokkos::create_mirror_view(cells_on_edge_dev);
            for (int e = 0; e < n_edges; ++e) {
                host_mirror(0, e) = mesh.cells_on_edge[0 + 2 * e];
                host_mirror(1, e) = mesh.cells_on_edge[1 + 2 * e];
            }
            Kokkos::deep_copy(cells_on_edge_dev, host_mirror);
        }

        edge_values_dev = Kokkos::View<Scalar*, MemSpace>("edge_values", n_edges);
        {
            auto host_mirror = Kokkos::create_mirror_view(edge_values_dev);
            for (int e = 0; e < n_edges; ++e) {
                host_mirror(e) = mesh.edge_values[e];
            }
            Kokkos::deep_copy(edge_values_dev, host_mirror);
        }

        // Use a fixed number of duplicates for testing
        accum_ptr = new mpas::dycore::TeamDuplicationAccumulator<ExecSpace, Scalar>(n_cells, 4);
    }

    ~TeamDuplicationFixture() { delete accum_ptr; }

    /// Run the accumulation and return the result as a host vector.
    std::vector<Scalar> run() const {
        Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
        Kokkos::deep_copy(output, Scalar{0});

        auto ev = edge_values_dev;
        auto coe = cells_on_edge_dev;
        const int nc = n_cells;
        accum_ptr->accumulate(output, n_edges,
            KOKKOS_LAMBDA(int e, Kokkos::View<Scalar*, MemSpace> local_output) {
                for (int slot = 0; slot < 2; ++slot) {
                    int cell = coe(slot, e);
                    if (cell >= 0 && cell < nc) {
                        local_output(cell) += ev(e);
                    }
                }
            });
        Kokkos::fence("team duplication accumulate fence");

        auto host_output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
        std::vector<Scalar> result(n_cells);
        for (int c = 0; c < n_cells; ++c) {
            result[c] = host_output(c);
        }
        return result;
    }
};

/// Helper for correctness checking - creates a fresh accumulator for each call.
std::vector<Scalar> runColoredAccumulatorForCorrectness(const RandomMesh& mesh) {
    ColoredAccumulatorFixture fixture(mesh);
    return fixture.run();
}

}  // namespace

// ---------------------------------------------------------------------------
// Property 8: Deterministic unstructured accumulation
// Feature: mpas-dycore-cpp-port, Property 8: Deterministic unstructured accumulation
// ---------------------------------------------------------------------------

/// Sub-property 1: ColoredAccumulator determinism — running accumulate() twice
/// on the same accumulator instance produces bit-for-bit identical results.
/// This validates that the accumulation order is fixed by the coloring.
RC_GTEST_PROP(AccumulationProperty,
              ColoredAccumulatorDeterminism,
              ()) {
    auto mesh = generateRandomMesh();

    // Build ONE accumulator (one coloring) and run accumulate() twice
    ColoredAccumulatorFixture fixture(mesh);
    auto result1 = fixture.run();
    auto result2 = fixture.run();

    // Bit-for-bit equality — determinism means identical results
    RC_ASSERT(result1.size() == result2.size());
    for (size_t c = 0; c < result1.size(); ++c) {
        RC_ASSERT(result1[c] == result2[c]);
    }
}

/// Sub-property 2: ColoredAccumulator correctness — result matches a serial
/// reference sum within floating-point tolerance.
RC_GTEST_PROP(AccumulationProperty,
              ColoredAccumulatorCorrectness,
              ()) {
    auto mesh = generateRandomMesh();

    auto parallel_result = runColoredAccumulatorForCorrectness(mesh);
    auto serial_result = serialReferenceAccumulation(mesh);

    RC_ASSERT(parallel_result.size() == serial_result.size());
    for (size_t c = 0; c < parallel_result.size(); ++c) {
        Scalar diff = std::abs(parallel_result[c] - serial_result[c]);
        Scalar scale = std::max(Scalar{1.0}, std::abs(serial_result[c]));
        RC_ASSERT(diff / scale <= tolerance);
    }
}

/// Sub-property 3: TeamDuplicationAccumulator determinism — running accumulate()
/// twice on the same accumulator instance produces bit-for-bit identical results.
RC_GTEST_PROP(AccumulationProperty,
              TeamDuplicationAccumulatorDeterminism,
              ()) {
    auto mesh = generateRandomMesh();

    // Build ONE accumulator and run accumulate() twice
    TeamDuplicationFixture fixture(mesh);
    auto result1 = fixture.run();
    auto result2 = fixture.run();

    // Bit-for-bit equality — determinism means identical results
    RC_ASSERT(result1.size() == result2.size());
    for (size_t c = 0; c < result1.size(); ++c) {
        RC_ASSERT(result1[c] == result2[c]);
    }
}
