#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/accumulation.hpp"
#include "mpas_dycore/scalar.hpp"

#include <cmath>
#include <numeric>
#include <vector>

namespace {

using Scalar = mpas::dycore::Scalar;
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;

// Tolerance for floating-point comparisons
constexpr Scalar kTol = std::is_same_v<Scalar, double> ? 1e-12 : 1e-6f;

// Helper: compute a serial reference accumulation for cellsOnEdge-style connectivity.
// For each edge e and target slot t: output[target_of_source(t, e)] += op(e, t)
// where op(e, t) = edge_values[e] (simple case: same value to both targets)
std::vector<Scalar> serial_accumulate(
    int n_edges, int n_cells,
    const std::vector<int>& cells_on_edge_flat,  // (2, nEdges) in LayoutLeft
    const std::vector<Scalar>& edge_values) {
  std::vector<Scalar> result(n_cells, 0.0);
  for (int e = 0; e < n_edges; ++e) {
    for (int t = 0; t < 2; ++t) {
      int cell = cells_on_edge_flat[t + 2 * e];
      if (cell >= 0 && cell < n_cells) {
        result[cell] += edge_values[e];
      }
    }
  }
  return result;
}

// Helper: compute a serial reference 2D accumulation (n_levels, n_cells)
std::vector<Scalar> serial_accumulate_2d(
    int n_edges, int n_cells, int n_levels,
    const std::vector<int>& cells_on_edge_flat,
    const std::vector<Scalar>& edge_level_values) {  // (n_levels, n_edges) LayoutLeft
  std::vector<Scalar> result(n_levels * n_cells, 0.0);
  for (int e = 0; e < n_edges; ++e) {
    for (int t = 0; t < 2; ++t) {
      int cell = cells_on_edge_flat[t + 2 * e];
      if (cell >= 0 && cell < n_cells) {
        for (int k = 0; k < n_levels; ++k) {
          result[k + n_levels * cell] += edge_level_values[k + n_levels * e];
        }
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Test 1: ColoredAccumulator with a simple 4-cell, 4-edge quad mesh
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
// Mesh topology (a simple quad):
//   Cell 0 -- Edge 0 -- Cell 1
//     |                    |
//   Edge 3              Edge 1
//     |                    |
//   Cell 3 -- Edge 2 -- Cell 2
//
// cellsOnEdge:
//   Edge 0: cells (0, 1)
//   Edge 1: cells (1, 2)
//   Edge 2: cells (2, 3)
//   Edge 3: cells (3, 0)
TEST(AccumulationTest, ColoredAccumulatorQuadMesh) {
  const int n_edges = 4;
  const int n_cells = 4;

  // cellsOnEdge in LayoutLeft: shape (2, nEdges)
  // Linear memory: [cell0_e0, cell1_e0, cell0_e1, cell1_e1, ...]
  // Actually LayoutLeft means (t, e) -> index = t + 2*e
  std::vector<int> cells_on_edge_flat = {
      0, 1,   // edge 0: cells (0, 1)
      1, 2,   // edge 1: cells (1, 2)
      2, 3,   // edge 2: cells (2, 3)
      3, 0    // edge 3: cells (3, 0)
  };

  // Edge values: edge e contributes value (e+1) to both its cells
  std::vector<Scalar> edge_values = {1.0, 2.0, 3.0, 4.0};

  // Compute serial reference
  auto serial_result = serial_accumulate(n_edges, n_cells, cells_on_edge_flat, edge_values);

  // Set up Kokkos views — allocate a mutable view, then create a const alias
  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge_rw", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }

  // Create the const view from the non-const view
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  // Create edge values on device
  Kokkos::View<Scalar*, MemSpace> edge_vals_dev("edge_values", n_edges);
  auto edge_vals_h = Kokkos::create_mirror_view(edge_vals_dev);
  for (int e = 0; e < n_edges; ++e) {
    edge_vals_h(e) = edge_values[e];
  }
  Kokkos::deep_copy(edge_vals_dev, edge_vals_h);

  // Build the accumulator
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> accum(n_edges, n_cells, cells_on_edge_const);

  // Run accumulation
  Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
  Kokkos::deep_copy(output, Scalar{0});

  const auto ev = edge_vals_dev;
  accum.accumulate(output, KOKKOS_LAMBDA(int e, int /*t*/) -> Scalar {
    return ev(e);
  });

  // Compare with serial reference
  auto output_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  for (int c = 0; c < n_cells; ++c) {
    EXPECT_NEAR(output_h(c), serial_result[c], kTol)
        << "Mismatch at cell " << c;
  }

  // Verify the expected values:
  // Cell 0: edge 0 (1.0) + edge 3 (4.0) = 5.0
  // Cell 1: edge 0 (1.0) + edge 1 (2.0) = 3.0
  // Cell 2: edge 1 (2.0) + edge 2 (3.0) = 5.0
  // Cell 3: edge 2 (3.0) + edge 3 (4.0) = 7.0
  EXPECT_NEAR(output_h(0), 5.0, kTol);
  EXPECT_NEAR(output_h(1), 3.0, kTol);
  EXPECT_NEAR(output_h(2), 5.0, kTol);
  EXPECT_NEAR(output_h(3), 7.0, kTol);
}

// ---------------------------------------------------------------------------
// Test 2: ColoredAccumulator with a hexagonal mesh (7 cells, 12 edges)
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
// Hexagonal pattern: 1 center cell (0) surrounded by 6 cells (1-6)
// 6 interior edges connect center to neighbors, 6 exterior edges connect neighbors
//   Interior edges (0-5): center(0) <-> cell(i+1)
//   Exterior edges (6-11): cell(i+1) <-> cell(((i+1)%6)+1)
TEST(AccumulationTest, ColoredAccumulatorHexMesh) {
  const int n_edges = 12;
  const int n_cells = 7;

  // Build cellsOnEdge: LayoutLeft (2, nEdges)
  std::vector<int> cells_on_edge_flat(2 * n_edges);
  // Interior edges: edge i connects cell 0 to cell (i+1)
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * i] = 0;       // center
    cells_on_edge_flat[1 + 2 * i] = i + 1;   // neighbor
  }
  // Exterior edges: edge (6+i) connects cell (i+1) to cell (((i+1)%6)+1)
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * (6 + i)] = i + 1;
    cells_on_edge_flat[1 + 2 * (6 + i)] = ((i + 1) % 6) + 1;
  }

  // Edge values: edge e contributes value (e + 1.0)
  std::vector<Scalar> edge_values(n_edges);
  for (int e = 0; e < n_edges; ++e) {
    edge_values[e] = static_cast<Scalar>(e + 1.0);
  }

  // Serial reference
  auto serial_result = serial_accumulate(n_edges, n_cells, cells_on_edge_flat, edge_values);

  // Set up Kokkos views
  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  Kokkos::View<Scalar*, MemSpace> edge_vals_dev("edge_values", n_edges);
  auto edge_vals_h = Kokkos::create_mirror_view(edge_vals_dev);
  for (int e = 0; e < n_edges; ++e) {
    edge_vals_h(e) = edge_values[e];
  }
  Kokkos::deep_copy(edge_vals_dev, edge_vals_h);

  // Build accumulator
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> accum(n_edges, n_cells, cells_on_edge_const);

  // Run accumulation
  Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
  Kokkos::deep_copy(output, Scalar{0});

  const auto ev = edge_vals_dev;
  accum.accumulate(output, KOKKOS_LAMBDA(int e, int /*t*/) -> Scalar {
    return ev(e);
  });

  // Compare
  auto output_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  for (int c = 0; c < n_cells; ++c) {
    EXPECT_NEAR(output_h(c), serial_result[c], kTol)
        << "Mismatch at cell " << c;
  }

  // Verify center cell (0): receives from interior edges 0..5 (values 1..6) = sum(1..6) = 21
  EXPECT_NEAR(output_h(0), 21.0, kTol);
}

// ---------------------------------------------------------------------------
// Test 3: ColoredAccumulator 2D accumulation: multi-level results match serial
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
TEST(AccumulationTest, ColoredAccumulator2DMultiLevel) {
  const int n_edges = 4;
  const int n_cells = 4;
  const int n_levels = 3;

  // Same quad mesh as test 1
  std::vector<int> cells_on_edge_flat = {
      0, 1, 1, 2, 2, 3, 3, 0
  };

  // Edge-level values: value at (level k, edge e) = (k+1) * (e+1)
  std::vector<Scalar> edge_level_values(n_levels * n_edges);
  for (int e = 0; e < n_edges; ++e) {
    for (int k = 0; k < n_levels; ++k) {
      edge_level_values[k + n_levels * e] = static_cast<Scalar>((k + 1) * (e + 1));
    }
  }

  // Serial reference 2D
  auto serial_result = serial_accumulate_2d(n_edges, n_cells, n_levels,
                                             cells_on_edge_flat, edge_level_values);

  // Set up Kokkos views
  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  // Edge-level values on device: (n_levels, n_edges) LayoutLeft
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> edge_level_dev("edge_level_vals", n_levels, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    for (int k = 0; k < n_levels; ++k) {
      edge_level_dev(k, e) = static_cast<Scalar>((k + 1) * (e + 1));
    }
  }

  // Build accumulator
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> accum(n_edges, n_cells, cells_on_edge_const);

  // Run 2D accumulation
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> output("output_2d", n_levels, n_cells);
  Kokkos::deep_copy(output, Scalar{0});

  const auto elv = edge_level_dev;
  accum.accumulate_2d(output, n_levels,
      KOKKOS_LAMBDA(int e, int k, int /*t*/) -> Scalar {
        return elv(k, e);
      });

  // Compare
  for (int c = 0; c < n_cells; ++c) {
    for (int k = 0; k < n_levels; ++k) {
      Scalar expected = serial_result[k + n_levels * c];
      EXPECT_NEAR(output(k, c), expected, kTol)
          << "Mismatch at (level=" << k << ", cell=" << c << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Test 4: TeamDuplicationAccumulator matches serial reference
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
TEST(AccumulationTest, TeamDuplicationAccumulatorMatchesSerial) {
  const int n_edges = 4;
  const int n_cells = 4;

  // Same quad mesh topology
  std::vector<int> cells_on_edge_flat = {
      0, 1, 1, 2, 2, 3, 3, 0
  };
  std::vector<Scalar> edge_values = {1.0, 2.0, 3.0, 4.0};

  // Serial reference
  auto serial_result = serial_accumulate(n_edges, n_cells, cells_on_edge_flat, edge_values);

  // Set up views
  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }

  Kokkos::View<Scalar*, MemSpace> edge_vals_dev("edge_values", n_edges);
  for (int e = 0; e < n_edges; ++e) {
    edge_vals_dev(e) = edge_values[e];
  }

  // Build TeamDuplicationAccumulator with a small number of duplicates (e.g. 2)
  mpas::dycore::TeamDuplicationAccumulator<ExecSpace, Scalar> accum(n_cells, 2);

  // Run accumulation
  Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
  Kokkos::deep_copy(output, Scalar{0});

  const auto ev = edge_vals_dev;
  const auto coe = cells_on_edge_rw;
  const int nc = n_cells;

  accum.accumulate(output, n_edges,
      KOKKOS_LAMBDA(int e, Kokkos::View<Scalar*, MemSpace> local_out) {
        for (int t = 0; t < 2; ++t) {
          int cell = coe(t, e);
          if (cell >= 0 && cell < nc) {
            local_out(cell) += ev(e);
          }
        }
      });

  // Compare with serial reference
  auto output_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  for (int c = 0; c < n_cells; ++c) {
    EXPECT_NEAR(output_h(c), serial_result[c], kTol)
        << "Mismatch at cell " << c;
  }
}

// ---------------------------------------------------------------------------
// Test 5: Both accumulator strategies produce the same result (deterministic)
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
TEST(AccumulationTest, BothStrategiesProduceSameResult) {
  const int n_edges = 12;
  const int n_cells = 7;

  // Hexagonal mesh from test 2
  std::vector<int> cells_on_edge_flat(2 * n_edges);
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * i] = 0;
    cells_on_edge_flat[1 + 2 * i] = i + 1;
  }
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * (6 + i)] = i + 1;
    cells_on_edge_flat[1 + 2 * (6 + i)] = ((i + 1) % 6) + 1;
  }

  // Edge values with non-trivial floating point values
  std::vector<Scalar> edge_values(n_edges);
  for (int e = 0; e < n_edges; ++e) {
    edge_values[e] = static_cast<Scalar>(1.0 / (e + 1));
  }

  // Set up views
  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  Kokkos::View<Scalar*, MemSpace> edge_vals_dev("edge_values", n_edges);
  for (int e = 0; e < n_edges; ++e) {
    edge_vals_dev(e) = edge_values[e];
  }

  // Strategy 1: ColoredAccumulator
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> colored_accum(n_edges, n_cells, cells_on_edge_const);
  Kokkos::View<Scalar*, MemSpace> output_colored("output_colored", n_cells);
  Kokkos::deep_copy(output_colored, Scalar{0});

  const auto ev = edge_vals_dev;
  colored_accum.accumulate(output_colored, KOKKOS_LAMBDA(int e, int /*t*/) -> Scalar {
    return ev(e);
  });

  // Strategy 2: TeamDuplicationAccumulator
  mpas::dycore::TeamDuplicationAccumulator<ExecSpace, Scalar> team_accum(n_cells, 4);
  Kokkos::View<Scalar*, MemSpace> output_team("output_team", n_cells);
  Kokkos::deep_copy(output_team, Scalar{0});

  const auto coe = cells_on_edge_rw;
  const int nc = n_cells;

  team_accum.accumulate(output_team, n_edges,
      KOKKOS_LAMBDA(int e, Kokkos::View<Scalar*, MemSpace> local_out) {
        for (int t = 0; t < 2; ++t) {
          int cell = coe(t, e);
          if (cell >= 0 && cell < nc) {
            local_out(cell) += ev(e);
          }
        }
      });

  // Both strategies should produce the same result
  auto colored_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output_colored);
  auto team_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output_team);
  for (int c = 0; c < n_cells; ++c) {
    EXPECT_NEAR(colored_h(c), team_h(c), kTol)
        << "Strategies disagree at cell " << c;
  }

  // Also verify both match the serial reference
  auto serial_result = serial_accumulate(n_edges, n_cells, cells_on_edge_flat, edge_values);
  for (int c = 0; c < n_cells; ++c) {
    EXPECT_NEAR(colored_h(c), serial_result[c], kTol)
        << "ColoredAccumulator differs from serial at cell " << c;
    EXPECT_NEAR(team_h(c), serial_result[c], kTol)
        << "TeamDuplicationAccumulator differs from serial at cell " << c;
  }
}

// ---------------------------------------------------------------------------
// Test 6: Edge case — single cell, single edge
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
TEST(AccumulationTest, SingleCellSingleEdge) {
  const int n_edges = 1;
  const int n_cells = 1;

  // One edge connects cell 0 to itself (boundary-like: one side valid, other -1)
  std::vector<int> cells_on_edge_flat = {0, -1};
  std::vector<Scalar> edge_values = {7.5};

  auto serial_result = serial_accumulate(n_edges, n_cells, cells_on_edge_flat, edge_values);

  // Set up views
  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  cells_on_edge_rw(0, 0) = 0;
  cells_on_edge_rw(1, 0) = -1;  // boundary: no second cell
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  Kokkos::View<Scalar*, MemSpace> edge_vals_dev("edge_values", n_edges);
  edge_vals_dev(0) = 7.5;

  // Test ColoredAccumulator
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> colored_accum(n_edges, n_cells, cells_on_edge_const);
  Kokkos::View<Scalar*, MemSpace> output_colored("output_colored", n_cells);
  Kokkos::deep_copy(output_colored, Scalar{0});

  const auto ev = edge_vals_dev;
  colored_accum.accumulate(output_colored, KOKKOS_LAMBDA(int e, int /*t*/) -> Scalar {
    return ev(e);
  });

  EXPECT_NEAR(output_colored(0), serial_result[0], kTol);
  EXPECT_NEAR(output_colored(0), 7.5, kTol);  // only one valid target

  // Test TeamDuplicationAccumulator
  mpas::dycore::TeamDuplicationAccumulator<ExecSpace, Scalar> team_accum(n_cells, 1);
  Kokkos::View<Scalar*, MemSpace> output_team("output_team", n_cells);
  Kokkos::deep_copy(output_team, Scalar{0});

  const auto coe = cells_on_edge_rw;
  const int nc = n_cells;
  team_accum.accumulate(output_team, n_edges,
      KOKKOS_LAMBDA(int e, Kokkos::View<Scalar*, MemSpace> local_out) {
        for (int t = 0; t < 2; ++t) {
          int cell = coe(t, e);
          if (cell >= 0 && cell < nc) {
            local_out(cell) += ev(e);
          }
        }
      });

  EXPECT_NEAR(output_team(0), 7.5, kTol);
}

// ---------------------------------------------------------------------------
// Test 7: ColoredAccumulator with icosahedral-like mesh (12 pentagonal +
//         variable hexagonal cells) — representative of MPAS Voronoi meshes.
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
// Builds a small icosahedral mesh fragment: 1 center cell (hexagonal, 6 edges),
// 6 ring-1 cells (pentagonal, 5 edges each), 6 ring-2 cells (hexagonal, 6 edges).
// Total: 13 cells, 30 edges.
TEST(AccumulationTest, ColoredAccumulatorIcosahedralFragment) {
  // Build a mesh fragment resembling an icosahedral patch:
  // - Cell 0 (center): connected to cells 1-6 via edges 0-5
  // - Cells 1-6 (ring 1): connected to center and two ring-1 neighbors each,
  //   plus connections to ring-2 cells 7-12
  // - Cells 7-12 (ring 2): connected to two ring-1 cells each
  const int n_cells = 13;
  const int n_edges = 30;

  std::vector<int> cells_on_edge_flat(2 * n_edges);

  // Ring-0 to ring-1 edges (0-5): cell 0 <-> cell (i+1)
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * i] = 0;
    cells_on_edge_flat[1 + 2 * i] = i + 1;
  }
  // Ring-1 to ring-1 edges (6-11): cell (i+1) <-> cell ((i+1)%6 + 1)
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * (6 + i)] = i + 1;
    cells_on_edge_flat[1 + 2 * (6 + i)] = (i + 1) % 6 + 1;
  }
  // Ring-1 to ring-2 edges (12-23): each ring-1 cell connects to 2 ring-2 cells
  // cell (i+1) <-> cell (7 + i) and cell (i+1) <-> cell (7 + (i+5)%6)
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * (12 + 2 * i)] = i + 1;
    cells_on_edge_flat[1 + 2 * (12 + 2 * i)] = 7 + i;
    cells_on_edge_flat[0 + 2 * (12 + 2 * i + 1)] = i + 1;
    cells_on_edge_flat[1 + 2 * (12 + 2 * i + 1)] = 7 + (i + 5) % 6;
  }
  // Ring-2 boundary edges (24-29): cell (7+i) <-> -1 (boundary)
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * (24 + i)] = 7 + i;
    cells_on_edge_flat[1 + 2 * (24 + i)] = -1;  // boundary
  }

  // Edge values: use non-trivial values v(e) = sin(e * 0.7 + 0.3) to get
  // mixed positive/negative contributions
  std::vector<Scalar> edge_values(n_edges);
  for (int e = 0; e < n_edges; ++e) {
    edge_values[e] = static_cast<Scalar>(std::sin(e * 0.7 + 0.3));
  }

  // Serial reference
  auto serial_result = serial_accumulate(n_edges, n_cells, cells_on_edge_flat, edge_values);

  // Set up Kokkos views
  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  Kokkos::View<Scalar*, MemSpace> edge_vals_dev("edge_values", n_edges);
  for (int e = 0; e < n_edges; ++e) {
    edge_vals_dev(e) = edge_values[e];
  }

  // Build accumulator
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> accum(n_edges, n_cells, cells_on_edge_const);

  // Verify coloring is valid: num_colors should be > 0 and reasonable
  EXPECT_GT(accum.num_colors(), 0);
  // For a mesh this dense, expect at least 2 colors
  EXPECT_GE(accum.num_colors(), 2);

  // Run accumulation
  Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
  Kokkos::deep_copy(output, Scalar{0});

  const auto ev = edge_vals_dev;
  accum.accumulate(output, KOKKOS_LAMBDA(int e, int /*t*/) -> Scalar {
    return ev(e);
  });

  // Compare with serial reference
  auto output_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  for (int c = 0; c < n_cells; ++c) {
    EXPECT_NEAR(output_h(c), serial_result[c], kTol)
        << "Mismatch at cell " << c;
  }
}

// ---------------------------------------------------------------------------
// Test 8: ColoredAccumulator with asymmetric functor (different contribution
//         per target slot) — validates that the slot index is correctly passed.
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
TEST(AccumulationTest, ColoredAccumulatorAsymmetricContributions) {
  const int n_edges = 4;
  const int n_cells = 4;

  // Same quad mesh
  std::vector<int> cells_on_edge_flat = {
      0, 1, 1, 2, 2, 3, 3, 0
  };

  // Asymmetric: to slot 0 contribute (e+1), to slot 1 contribute -(e+1)
  // This mimics flux accumulation where the sign depends on the edge normal direction

  // Serial reference with asymmetric contributions
  std::vector<Scalar> serial_result(n_cells, 0.0);
  for (int e = 0; e < n_edges; ++e) {
    for (int t = 0; t < 2; ++t) {
      int cell = cells_on_edge_flat[t + 2 * e];
      if (cell >= 0 && cell < n_cells) {
        Scalar val = (t == 0) ? static_cast<Scalar>(e + 1) : static_cast<Scalar>(-(e + 1));
        serial_result[cell] += val;
      }
    }
  }

  // Set up views
  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  // Build accumulator
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> accum(n_edges, n_cells, cells_on_edge_const);

  // Run with asymmetric functor
  Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
  Kokkos::deep_copy(output, Scalar{0});

  accum.accumulate(output, KOKKOS_LAMBDA(int e, int t) -> Scalar {
    return (t == 0) ? static_cast<Scalar>(e + 1) : static_cast<Scalar>(-(e + 1));
  });

  auto output_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  for (int c = 0; c < n_cells; ++c) {
    EXPECT_NEAR(output_h(c), serial_result[c], kTol)
        << "Mismatch at cell " << c;
  }

  // Expected:
  // Cell 0: slot 0 of edge 0 (+1) + slot 1 of edge 3 (-4) = -3
  // Cell 1: slot 1 of edge 0 (-1) + slot 0 of edge 1 (+2) = +1
  // Cell 2: slot 1 of edge 1 (-2) + slot 0 of edge 2 (+3) = +1
  // Cell 3: slot 1 of edge 2 (-3) + slot 0 of edge 3 (+4) = +1
  EXPECT_NEAR(output_h(0), -3.0, kTol);
  EXPECT_NEAR(output_h(1),  1.0, kTol);
  EXPECT_NEAR(output_h(2),  1.0, kTol);
  EXPECT_NEAR(output_h(3),  1.0, kTol);
}

// ---------------------------------------------------------------------------
// Test 9: Coloring validity — verify no two same-color edges share a target cell
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
TEST(AccumulationTest, ColoringValidityNoConflicts) {
  const int n_edges = 12;
  const int n_cells = 7;

  // Hexagonal mesh
  std::vector<int> cells_on_edge_flat(2 * n_edges);
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * i] = 0;
    cells_on_edge_flat[1 + 2 * i] = i + 1;
  }
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * (6 + i)] = i + 1;
    cells_on_edge_flat[1 + 2 * (6 + i)] = ((i + 1) % 6) + 1;
  }

  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> accum(n_edges, n_cells, cells_on_edge_const);

  // Extract the colors
  auto colors_dev = accum.get_colors();
  auto colors_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, colors_dev);

  int num_colors = accum.num_colors();
  ASSERT_GT(num_colors, 0);

  // For each color, collect the set of target cells written by edges of that color.
  // No target should appear more than once within a single color.
  for (int color = 1; color <= num_colors; ++color) {
    std::vector<int> targets_in_color;
    for (int e = 0; e < n_edges; ++e) {
      if (colors_h(e) != color) continue;
      for (int t = 0; t < 2; ++t) {
        int cell = cells_on_edge_flat[t + 2 * e];
        if (cell >= 0 && cell < n_cells) {
          targets_in_color.push_back(cell);
        }
      }
    }
    // Sort and check for duplicates
    std::sort(targets_in_color.begin(), targets_in_color.end());
    for (size_t i = 1; i < targets_in_color.size(); ++i) {
      EXPECT_NE(targets_in_color[i], targets_in_color[i - 1])
          << "Color " << color << " has conflicting writes to cell "
          << targets_in_color[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Test 10: Repeated accumulation produces identical results (determinism check)
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
TEST(AccumulationTest, RepeatedAccumulationDeterministic) {
  const int n_edges = 12;
  const int n_cells = 7;

  // Hexagonal mesh
  std::vector<int> cells_on_edge_flat(2 * n_edges);
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * i] = 0;
    cells_on_edge_flat[1 + 2 * i] = i + 1;
  }
  for (int i = 0; i < 6; ++i) {
    cells_on_edge_flat[0 + 2 * (6 + i)] = i + 1;
    cells_on_edge_flat[1 + 2 * (6 + i)] = ((i + 1) % 6) + 1;
  }

  // Use values that would be sensitive to ordering: 1/e to get varied magnitudes
  std::vector<Scalar> edge_values(n_edges);
  for (int e = 0; e < n_edges; ++e) {
    edge_values[e] = static_cast<Scalar>(1.0 / (e + 1));
  }

  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = cells_on_edge_flat[0 + 2 * e];
    cells_on_edge_rw(1, e) = cells_on_edge_flat[1 + 2 * e];
  }
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  Kokkos::View<Scalar*, MemSpace> edge_vals_dev("edge_values", n_edges);
  for (int e = 0; e < n_edges; ++e) {
    edge_vals_dev(e) = edge_values[e];
  }

  // Build ONE accumulator (establishes a fixed coloring)
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> accum(n_edges, n_cells, cells_on_edge_const);

  // Run accumulation 5 times and verify bit-for-bit identical results each time
  std::vector<Scalar> first_result(n_cells);
  for (int run = 0; run < 5; ++run) {
    Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
    Kokkos::deep_copy(output, Scalar{0});

    const auto ev = edge_vals_dev;
    accum.accumulate(output, KOKKOS_LAMBDA(int e, int /*t*/) -> Scalar {
      return ev(e);
    });

    auto output_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
    if (run == 0) {
      for (int c = 0; c < n_cells; ++c) {
        first_result[c] = output_h(c);
      }
    } else {
      for (int c = 0; c < n_cells; ++c) {
        // Bit-for-bit identical — not EXPECT_NEAR but exact equality
        EXPECT_EQ(output_h(c), first_result[c])
            << "Non-deterministic result at cell " << c << " on run " << run;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Test 11: All-boundary edges (all -1 targets) — accumulation produces zero
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
TEST(AccumulationTest, AllBoundaryEdgesProduceZeroOutput) {
  const int n_edges = 5;
  const int n_cells = 3;

  // All edges have -1 targets (fully boundary)
  std::vector<int> cells_on_edge_flat(2 * n_edges, -1);

  Kokkos::View<int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_rw("cellsOnEdge", 2, n_edges);
  for (int e = 0; e < n_edges; ++e) {
    cells_on_edge_rw(0, e) = -1;
    cells_on_edge_rw(1, e) = -1;
  }
  Kokkos::View<const int**, Kokkos::LayoutLeft, MemSpace> cells_on_edge_const = cells_on_edge_rw;

  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar> accum(n_edges, n_cells, cells_on_edge_const);

  Kokkos::View<Scalar*, MemSpace> output("output", n_cells);
  Kokkos::deep_copy(output, Scalar{0});

  accum.accumulate(output, KOKKOS_LAMBDA(int /*e*/, int /*t*/) -> Scalar {
    return static_cast<Scalar>(99.9);  // large value — should never land anywhere
  });

  auto output_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  for (int c = 0; c < n_cells; ++c) {
    EXPECT_EQ(output_h(c), Scalar{0})
        << "Cell " << c << " should remain zero with all-boundary edges";
  }
}

}  // namespace
