/// @file test_execution_space_parity_property.cpp
/// @brief Property-based test: Cross-execution-space component parity.
///
/// Feature: mpas-dycore-cpp-port, Property 14: Cross-execution-space component parity
///
/// **Validates: Requirements 14.1 (execution-space portability)**
///
/// Property 7 from the design document:
/// For any valid input state to a ported component, executing that component on the
/// host/CPU Execution_Space and on the device/GPU Execution_Space SHALL each produce
/// output that matches within Parity_Tolerance, without requiring host-versus-device
/// bit-for-bit equality.
///
/// On CPU-only builds (DefaultHostExecutionSpace == DefaultExecutionSpace), this test
/// verifies determinism: running the same component kernel twice on the default execution
/// space produces bit-for-bit identical results. Determinism is a necessary condition for
/// cross-space parity — if the same computation isn't deterministic on a single space, it
/// cannot be expected to agree across spaces.
///
/// On GPU builds (DefaultExecutionSpace != DefaultHostExecutionSpace), the test compares
/// results from host vs device execution within Parity_Tolerance.
///
/// Representative components tested:
/// 1. ColoredAccumulator (edge-to-cell flux gather) — exercises graph coloring and
///    parallel_for with unstructured connectivity.
/// 2. Tridiagonal solve (Thomas algorithm) — exercises parallel-over-columns with
///    sequential vertical dependency.
/// Both are core dycore kernels that exercise different parallelization patterns.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include "mpas_dycore/accumulation.hpp"
#include "mpas_dycore/tridiagonal_solve.hpp"
#include "mpas_dycore/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace {

using Scalar = mpas::dycore::Scalar;
using DefaultExec = Kokkos::DefaultExecutionSpace;
using HostExec = Kokkos::DefaultHostExecutionSpace;
using DefaultMem = DefaultExec::memory_space;
using HostMem = HostExec::memory_space;

/// Whether we have distinct host and device execution spaces.
/// On CPU-only builds (OpenMP), these are the same.
constexpr bool kHasDistinctDeviceSpace =
    !std::is_same_v<DefaultExec, HostExec>;

/// Parity_Tolerance for cross-space comparison (Req 12.1, 12.2).
/// On same-space determinism checks we require bit-for-bit equality;
/// on cross-space checks we allow the unified Parity_Tolerance.
constexpr Scalar kParityTolerance =
    std::is_same_v<Scalar, double> ? Scalar(1e-12) : Scalar(1e-6);

// ============================================================================
// Helper: Random mesh generation for accumulation tests
// ============================================================================

struct RandomAccumMesh {
  int n_cells;
  int n_edges;
  std::vector<int> cells_on_edge;  // LayoutLeft: (2, n_edges)
  std::vector<Scalar> edge_values; // per-level per-edge: (n_levels * n_edges)
  int n_levels;
};

RandomAccumMesh generateRandomAccumMesh() {
  RandomAccumMesh mesh;
  mesh.n_cells = *rc::gen::inRange(4, 31);   // [4, 30]
  mesh.n_edges = *rc::gen::inRange(4, 51);   // [4, 50]
  mesh.n_levels = *rc::gen::inRange(2, 11);  // [2, 10]

  mesh.cells_on_edge.resize(2 * mesh.n_edges);
  for (int e = 0; e < mesh.n_edges; ++e) {
    mesh.cells_on_edge[0 + 2 * e] = *rc::gen::inRange(0, mesh.n_cells);
    int n_neighbors = *rc::gen::inRange(1, 3);
    if (n_neighbors == 2) {
      mesh.cells_on_edge[1 + 2 * e] = *rc::gen::inRange(0, mesh.n_cells);
    } else {
      mesh.cells_on_edge[1 + 2 * e] = -1;
    }
  }

  mesh.edge_values.resize(mesh.n_levels * mesh.n_edges);
  for (int i = 0; i < mesh.n_levels * mesh.n_edges; ++i) {
    double raw = *rc::gen::inRange(-1000, 1001);
    mesh.edge_values[i] = static_cast<Scalar>(raw * 0.01);
  }

  return mesh;
}

/// A fixture that holds a built ColoredAccumulator on a given execution space,
/// allowing repeated calls to run() to exercise determinism on the same coloring.
template <class ExecSpace>
struct AccumFixture2D {
  using mem_space = typename ExecSpace::memory_space;

  int n_cells;
  int n_edges;
  int n_levels;
  Kokkos::View<int**, Kokkos::LayoutLeft, mem_space> cells_on_edge_dev;
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, mem_space> edge_values_dev;
  mpas::dycore::ColoredAccumulator<ExecSpace, Scalar>* accum_ptr = nullptr;

  explicit AccumFixture2D(const RandomAccumMesh& mesh)
      : n_cells(mesh.n_cells), n_edges(mesh.n_edges), n_levels(mesh.n_levels) {
    cells_on_edge_dev = Kokkos::View<int**, Kokkos::LayoutLeft, mem_space>(
        "cells_on_edge", 2, n_edges);
    {
      auto host_m = Kokkos::create_mirror_view(cells_on_edge_dev);
      for (int e = 0; e < n_edges; ++e) {
        host_m(0, e) = mesh.cells_on_edge[0 + 2 * e];
        host_m(1, e) = mesh.cells_on_edge[1 + 2 * e];
      }
      Kokkos::deep_copy(cells_on_edge_dev, host_m);
    }

    edge_values_dev = Kokkos::View<Scalar**, Kokkos::LayoutLeft, mem_space>(
        "edge_values", n_levels, n_edges);
    {
      auto host_m = Kokkos::create_mirror_view(edge_values_dev);
      for (int e = 0; e < n_edges; ++e) {
        for (int k = 0; k < n_levels; ++k) {
          host_m(k, e) = mesh.edge_values[k + n_levels * e];
        }
      }
      Kokkos::deep_copy(edge_values_dev, host_m);
    }

    Kokkos::View<const int**, Kokkos::LayoutLeft, mem_space> coe_const =
        cells_on_edge_dev;
    accum_ptr = new mpas::dycore::ColoredAccumulator<ExecSpace, Scalar>(
        n_edges, n_cells, coe_const);
  }

  ~AccumFixture2D() { delete accum_ptr; }

  /// Run the 2D accumulation and return as a host vector.
  std::vector<Scalar> run() const {
    Kokkos::View<Scalar**, Kokkos::LayoutLeft, mem_space> output(
        "output", n_levels, n_cells);
    Kokkos::deep_copy(output, Scalar{0});

    auto ev = edge_values_dev;
    const int nl = n_levels;
    accum_ptr->accumulate_2d(output, nl,
        KOKKOS_LAMBDA(int e, int k, int /*slot*/) -> Scalar {
          return ev(k, e);
        });
    Kokkos::fence("accumulate_2d fence");

    auto host_output = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, output);
    std::vector<Scalar> result(n_levels * n_cells);
    for (int c = 0; c < n_cells; ++c) {
      for (int k = 0; k < nl; ++k) {
        result[k + nl * c] = host_output(k, c);
      }
    }
    return result;
  }
};

/// Compute a serial reference accumulation for the 2D case.
std::vector<Scalar> serialReference2D(const RandomAccumMesh& mesh) {
  std::vector<Scalar> result(mesh.n_levels * mesh.n_cells, Scalar{0});
  for (int e = 0; e < mesh.n_edges; ++e) {
    for (int slot = 0; slot < 2; ++slot) {
      int cell = mesh.cells_on_edge[slot + 2 * e];
      if (cell >= 0 && cell < mesh.n_cells) {
        for (int k = 0; k < mesh.n_levels; ++k) {
          result[k + mesh.n_levels * cell] +=
              mesh.edge_values[k + mesh.n_levels * e];
        }
      }
    }
  }
  return result;
}

// ============================================================================
// Helper: Random tridiagonal system for solve tests
// ============================================================================

struct RandomTridiagSystem {
  int n;       // nVertLevels
  int nCells;  // number of columns
  std::vector<Scalar> lower;  // size n, lower[0] unused
  std::vector<Scalar> diag;
  std::vector<Scalar> upper;  // size n, upper[n-1] unused
  // Per-cell RHS: (nCells * n)
  std::vector<Scalar> rhs;
};

RandomTridiagSystem generateRandomTridiagSystem() {
  RandomTridiagSystem sys;
  sys.n = *rc::gen::inRange(3, 31);      // [3, 30]
  sys.nCells = *rc::gen::inRange(1, 9);  // [1, 8]

  sys.lower.resize(sys.n, Scalar{0});
  sys.diag.resize(sys.n, Scalar{0});
  sys.upper.resize(sys.n, Scalar{0});
  sys.rhs.resize(sys.n * sys.nCells, Scalar{0});

  for (int k = 0; k < sys.n; ++k) {
    Scalar l_val = Scalar{0};
    Scalar u_val = Scalar{0};
    if (k > 0) {
      double raw = *rc::gen::inRange(-100, 101);
      l_val = static_cast<Scalar>(raw * 0.01);
    }
    if (k < sys.n - 1) {
      double raw = *rc::gen::inRange(-100, 101);
      u_val = static_cast<Scalar>(raw * 0.01);
    }
    sys.lower[k] = l_val;
    sys.upper[k] = u_val;

    Scalar off_sum = std::abs(l_val) + std::abs(u_val);
    double margin_raw = *rc::gen::inRange(10, 201);
    Scalar margin = static_cast<Scalar>(margin_raw * 0.01);
    sys.diag[k] = off_sum + margin;
  }

  for (int c = 0; c < sys.nCells; ++c) {
    for (int k = 0; k < sys.n; ++k) {
      double raw = *rc::gen::inRange(-1000, 1001);
      sys.rhs[k + sys.n * c] = static_cast<Scalar>(raw * 0.01);
    }
  }

  return sys;
}

/// Precompute Thomas algorithm coefficients from a tridiagonal system.
struct TriCoeffs {
  std::vector<Scalar> a_tri;
  std::vector<Scalar> alpha_tri;
  std::vector<Scalar> gamma_tri;
};

TriCoeffs precomputeCoeffs(const RandomTridiagSystem& sys) {
  const int n = sys.n;
  TriCoeffs coeffs;
  coeffs.a_tri.resize(n, Scalar{0});
  coeffs.alpha_tri.resize(n, Scalar{0});
  coeffs.gamma_tri.resize(n, Scalar{0});

  for (int k = 0; k < n; ++k) {
    coeffs.a_tri[k] = sys.lower[k];
  }

  Scalar d_prime = sys.diag[0];
  coeffs.alpha_tri[0] = Scalar{1} / d_prime;
  coeffs.gamma_tri[0] = sys.upper[0] * coeffs.alpha_tri[0];

  for (int k = 1; k < n; ++k) {
    d_prime = sys.diag[k] - sys.lower[k] * coeffs.gamma_tri[k - 1];
    coeffs.alpha_tri[k] = Scalar{1} / d_prime;
    if (k < n - 1) {
      coeffs.gamma_tri[k] = sys.upper[k] * coeffs.alpha_tri[k];
    } else {
      coeffs.gamma_tri[k] = Scalar{0};
    }
  }

  return coeffs;
}

/// Run the tridiagonal solve on a given execution space and return the
/// solution as a host vector (size n * nCells, column-major).
template <class ExecSpace>
std::vector<Scalar> runTridiagSolve(const RandomTridiagSystem& sys) {
  using mem_space = typename ExecSpace::memory_space;
  const int n = sys.n;
  const int nCells = sys.nCells;

  auto coeffs = precomputeCoeffs(sys);

  // Create views: (nVertLevels[+1], nCells) LayoutLeft
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, mem_space> rw_p("rw_p", n + 1, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, mem_space> a_tri_dev("a_tri", n, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, mem_space> alpha_tri_dev("alpha_tri", n, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, mem_space> gamma_tri_dev("gamma_tri", n, nCells);

  auto rw_p_host = Kokkos::create_mirror_view(rw_p);
  auto a_tri_host = Kokkos::create_mirror_view(a_tri_dev);
  auto alpha_tri_host = Kokkos::create_mirror_view(alpha_tri_dev);
  auto gamma_tri_host = Kokkos::create_mirror_view(gamma_tri_dev);

  for (int iCell = 0; iCell < nCells; ++iCell) {
    // Level 0: pre-scaled by alpha_tri(0) (kernel expects this)
    rw_p_host(0, iCell) = sys.rhs[0 + n * iCell] * coeffs.alpha_tri[0];
    for (int k = 1; k < n; ++k) {
      rw_p_host(k, iCell) = sys.rhs[k + n * iCell];
    }
    rw_p_host(n, iCell) = Scalar{0}; // boundary

    for (int k = 0; k < n; ++k) {
      a_tri_host(k, iCell) = coeffs.a_tri[k];
      alpha_tri_host(k, iCell) = coeffs.alpha_tri[k];
      gamma_tri_host(k, iCell) = coeffs.gamma_tri[k];
    }
  }

  Kokkos::deep_copy(rw_p, rw_p_host);
  Kokkos::deep_copy(a_tri_dev, a_tri_host);
  Kokkos::deep_copy(alpha_tri_dev, alpha_tri_host);
  Kokkos::deep_copy(gamma_tri_dev, gamma_tri_host);

  // Run the solve
  mpas::dycore::TridiagonalSolveParams params;
  params.nVertLevels = n;
  params.nCells = nCells;
  params.dts = Scalar{1.0};

  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, mem_space> a_const = a_tri_dev;
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, mem_space> alpha_const = alpha_tri_dev;
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, mem_space> gamma_const = gamma_tri_dev;

  mpas::dycore::tridiagonal_solve_no_damping<ExecSpace>(
      rw_p, a_const, alpha_const, gamma_const, params);
  Kokkos::fence("tridiag_solve parity fence");

  // Copy to host
  Kokkos::deep_copy(rw_p_host, rw_p);
  std::vector<Scalar> result(n * nCells);
  for (int iCell = 0; iCell < nCells; ++iCell) {
    for (int k = 0; k < n; ++k) {
      result[k + n * iCell] = rw_p_host(k, iCell);
    }
  }
  return result;
}

}  // namespace

// ===========================================================================
// Property 14: Cross-execution-space component parity
// Feature: mpas-dycore-cpp-port, Property 14: Cross-execution-space component parity
// ===========================================================================

/// Sub-property 1: ColoredAccumulator determinism across execution spaces.
///
/// On CPU-only builds: running the accumulation twice on the same accumulator
/// instance (same coloring) produces bit-for-bit identical results.
/// On GPU builds: host vs device results each agree with a serial reference
/// within Parity_Tolerance (since colorings may differ between spaces).
RC_GTEST_PROP(ExecutionSpaceParityProperty,
              ColoredAccumulatorCrossSpaceParity,
              ()) {
  auto mesh = generateRandomAccumMesh();
  auto serial_ref = serialReference2D(mesh);

  if constexpr (kHasDistinctDeviceSpace) {
    // GPU build: both host and device results must match serial reference
    // within Parity_Tolerance (colorings may differ, but correctness holds)
    AccumFixture2D<HostExec> host_fixture(mesh);
    auto host_result = host_fixture.run();
    AccumFixture2D<DefaultExec> dev_fixture(mesh);
    auto device_result = dev_fixture.run();

    RC_ASSERT(host_result.size() == serial_ref.size());
    RC_ASSERT(device_result.size() == serial_ref.size());
    for (size_t i = 0; i < serial_ref.size(); ++i) {
      Scalar scale = std::max(Scalar{1.0}, std::abs(serial_ref[i]));
      Scalar h_diff = std::abs(host_result[i] - serial_ref[i]);
      Scalar d_diff = std::abs(device_result[i] - serial_ref[i]);
      RC_ASSERT(h_diff / scale <= kParityTolerance);
      RC_ASSERT(d_diff / scale <= kParityTolerance);
    }
  } else {
    // CPU-only build: determinism check — same accumulator, two runs must
    // produce bit-for-bit identical results (fixed coloring fixes FP order)
    AccumFixture2D<DefaultExec> fixture(mesh);
    auto result1 = fixture.run();
    auto result2 = fixture.run();

    RC_ASSERT(result1.size() == result2.size());
    for (size_t i = 0; i < result1.size(); ++i) {
      RC_ASSERT(result1[i] == result2[i]);
    }

    // Also verify correctness within tolerance against serial reference
    for (size_t i = 0; i < serial_ref.size(); ++i) {
      Scalar scale = std::max(Scalar{1.0}, std::abs(serial_ref[i]));
      Scalar diff = std::abs(result1[i] - serial_ref[i]);
      RC_ASSERT(diff / scale <= kParityTolerance);
    }
  }
}

/// Sub-property 2: Tridiagonal solve determinism across execution spaces.
///
/// On CPU-only builds: running the solve twice on DefaultExecutionSpace produces
/// bit-for-bit identical results.
/// On GPU builds: host vs device solutions agree within Parity_Tolerance.
RC_GTEST_PROP(ExecutionSpaceParityProperty,
              TridiagonalSolveCrossSpaceParity,
              ()) {
  auto sys = generateRandomTridiagSystem();

  if constexpr (kHasDistinctDeviceSpace) {
    // GPU build: compare host vs device solutions
    auto host_result = runTridiagSolve<HostExec>(sys);
    auto device_result = runTridiagSolve<DefaultExec>(sys);

    RC_ASSERT(host_result.size() == device_result.size());
    for (size_t i = 0; i < host_result.size(); ++i) {
      Scalar diff = std::abs(host_result[i] - device_result[i]);
      Scalar scale = std::max(Scalar{1.0}, std::abs(host_result[i]));
      RC_ASSERT(diff / scale <= kParityTolerance);
    }
  } else {
    // CPU-only build: determinism check
    auto result1 = runTridiagSolve<DefaultExec>(sys);
    auto result2 = runTridiagSolve<DefaultExec>(sys);

    RC_ASSERT(result1.size() == result2.size());
    for (size_t i = 0; i < result1.size(); ++i) {
      RC_ASSERT(result1[i] == result2[i]);
    }
  }
}

/// Sub-property 3: Combined accumulation + solve pipeline determinism.
///
/// Exercises a representative dycore compute pattern: accumulate edge fluxes into
/// cells (mimicking tendency computation), then solve a tridiagonal system (mimicking
/// the acoustic solver). Verifies the combined pipeline is deterministic/portable.
RC_GTEST_PROP(ExecutionSpaceParityProperty,
              CombinedPipelineCrossSpaceParity,
              ()) {
  // Generate a small mesh for accumulation
  auto mesh = generateRandomAccumMesh();
  // Generate a tridiagonal system whose column count matches n_cells
  RandomTridiagSystem sys;
  sys.n = mesh.n_levels;
  sys.nCells = mesh.n_cells;
  sys.lower.resize(sys.n, Scalar{0});
  sys.diag.resize(sys.n, Scalar{0});
  sys.upper.resize(sys.n, Scalar{0});
  sys.rhs.resize(sys.n * sys.nCells, Scalar{0});

  // Fill tridiagonal system with random diagonally-dominant data
  for (int k = 0; k < sys.n; ++k) {
    Scalar l_val = Scalar{0};
    Scalar u_val = Scalar{0};
    if (k > 0) {
      double raw = *rc::gen::inRange(-100, 101);
      l_val = static_cast<Scalar>(raw * 0.01);
    }
    if (k < sys.n - 1) {
      double raw = *rc::gen::inRange(-100, 101);
      u_val = static_cast<Scalar>(raw * 0.01);
    }
    sys.lower[k] = l_val;
    sys.upper[k] = u_val;
    Scalar off_sum = std::abs(l_val) + std::abs(u_val);
    double margin_raw = *rc::gen::inRange(10, 201);
    sys.diag[k] = off_sum + static_cast<Scalar>(margin_raw * 0.01);
  }

  for (int c = 0; c < sys.nCells; ++c) {
    for (int k = 0; k < sys.n; ++k) {
      double raw = *rc::gen::inRange(-1000, 1001);
      sys.rhs[k + sys.n * c] = static_cast<Scalar>(raw * 0.01);
    }
  }

  auto serial_accum_ref = serialReference2D(mesh);

  if constexpr (kHasDistinctDeviceSpace) {
    // GPU: compare host vs device pipeline within tolerance
    AccumFixture2D<HostExec> host_accum(mesh);
    auto accum_host = host_accum.run();
    auto solve_host = runTridiagSolve<HostExec>(sys);

    AccumFixture2D<DefaultExec> dev_accum(mesh);
    auto accum_dev = dev_accum.run();
    auto solve_dev = runTridiagSolve<DefaultExec>(sys);

    // Both must match serial reference within tolerance
    RC_ASSERT(accum_host.size() == serial_accum_ref.size());
    for (size_t i = 0; i < serial_accum_ref.size(); ++i) {
      Scalar scale = std::max(Scalar{1.0}, std::abs(serial_accum_ref[i]));
      RC_ASSERT(std::abs(accum_host[i] - serial_accum_ref[i]) / scale <= kParityTolerance);
      RC_ASSERT(std::abs(accum_dev[i] - serial_accum_ref[i]) / scale <= kParityTolerance);
    }

    // Solve must agree across spaces within tolerance
    RC_ASSERT(solve_host.size() == solve_dev.size());
    for (size_t i = 0; i < solve_host.size(); ++i) {
      Scalar diff = std::abs(solve_host[i] - solve_dev[i]);
      Scalar scale = std::max(Scalar{1.0}, std::abs(solve_host[i]));
      RC_ASSERT(diff / scale <= kParityTolerance);
    }
  } else {
    // CPU-only: determinism — same fixture run twice must be bit-for-bit identical
    AccumFixture2D<DefaultExec> fixture(mesh);
    auto accum1 = fixture.run();
    auto accum2 = fixture.run();

    RC_ASSERT(accum1.size() == accum2.size());
    for (size_t i = 0; i < accum1.size(); ++i) {
      RC_ASSERT(accum1[i] == accum2[i]);
    }

    // Tridiagonal solve determinism
    auto solve1 = runTridiagSolve<DefaultExec>(sys);
    auto solve2 = runTridiagSolve<DefaultExec>(sys);

    RC_ASSERT(solve1.size() == solve2.size());
    for (size_t i = 0; i < solve1.size(); ++i) {
      RC_ASSERT(solve1[i] == solve2[i]);
    }

    // Correctness: accumulation matches serial reference within tolerance
    for (size_t i = 0; i < serial_accum_ref.size(); ++i) {
      Scalar scale = std::max(Scalar{1.0}, std::abs(serial_accum_ref[i]));
      RC_ASSERT(std::abs(accum1[i] - serial_accum_ref[i]) / scale <= kParityTolerance);
    }
  }
}
