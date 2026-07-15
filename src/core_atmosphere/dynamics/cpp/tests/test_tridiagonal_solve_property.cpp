/// @file test_tridiagonal_solve_property.cpp
/// @brief Property-based test: Tridiagonal solve satisfies its system.
///
/// Feature: mpas-dycore-cpp-port, Property 10: Tridiagonal solve
///
/// **Validates: Requirements 4.3**
///
/// For randomly generated diagonally-dominant tridiagonal systems with random RHS
/// vectors, the Thomas algorithm produces a solution x such that A*x = b within
/// floating-point tolerance.
///
/// The tridiagonal_solve_no_damping kernel operates with precomputed coefficients
/// (a_tri, alpha_tri, gamma_tri) that encode the Thomas algorithm forward/backward
/// sweeps. This property test:
/// 1. Generates random diagonally-dominant tridiagonal matrices
/// 2. Generates random RHS vectors
/// 3. Precomputes the Thomas algorithm coefficients
/// 4. Runs the kernel
/// 5. Verifies A * x = b within Parity_Tolerance

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include "mpas_dycore/tridiagonal_solve.hpp"

#include <cmath>
#include <vector>

namespace {

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using Scalar = mpas::dycore::Scalar;

/// Parity tolerance for floating-point comparisons.
constexpr Scalar parity_tolerance = std::is_same_v<Scalar, double> ? 1e-12 : 1e-6f;

/// A tridiagonal system for a single column: Ax = b
/// with n levels (the solve operates on levels 0..n-1 of the rw_p view which has n+1 entries).
struct TridiagonalSystem {
  int n;                     ///< System size (nVertLevels)
  std::vector<Scalar> lower; ///< Lower diagonal (size n, lower[0] unused)
  std::vector<Scalar> diag;  ///< Main diagonal (size n)
  std::vector<Scalar> upper; ///< Upper diagonal (size n, upper[n-1] unused)
  std::vector<Scalar> rhs;   ///< Right-hand side (size n)
};

/// Generate a random diagonally-dominant tridiagonal system.
///
/// Diagonal dominance ensures the Thomas algorithm is stable and the system
/// has a unique solution. We generate |d(k)| > |l(k)| + |u(k)| for each row.
TridiagonalSystem generateDiagonallyDominantSystem() {
  TridiagonalSystem sys;
  // nVertLevels in [3, 60] — reasonable column depth for property testing
  sys.n = *rc::gen::inRange(3, 61);

  sys.lower.resize(sys.n, Scalar{0});
  sys.diag.resize(sys.n, Scalar{0});
  sys.upper.resize(sys.n, Scalar{0});
  sys.rhs.resize(sys.n, Scalar{0});

  for (int k = 0; k < sys.n; ++k) {
    // Generate off-diagonal entries in a bounded range to avoid extreme values
    Scalar l_val = Scalar{0};
    Scalar u_val = Scalar{0};

    if (k > 0) {
      // Lower diagonal entry (non-zero for k > 0)
      double raw_l = *rc::gen::inRange(-100, 101);
      l_val = static_cast<Scalar>(raw_l * 0.01);
    }
    if (k < sys.n - 1) {
      // Upper diagonal entry (non-zero for k < n-1)
      double raw_u = *rc::gen::inRange(-100, 101);
      u_val = static_cast<Scalar>(raw_u * 0.01);
    }

    sys.lower[k] = l_val;
    sys.upper[k] = u_val;

    // Main diagonal: ensure strict diagonal dominance
    // |d(k)| > |l(k)| + |u(k)|, so set d = sign * (|l| + |u| + margin)
    Scalar off_sum = std::abs(l_val) + std::abs(u_val);
    // Add a margin in [0.1, 2.0] to ensure strict dominance
    double margin_raw = *rc::gen::inRange(10, 201);
    Scalar margin = static_cast<Scalar>(margin_raw * 0.01);
    // Choose sign for diagonal (always positive to keep things simple and stable)
    sys.diag[k] = off_sum + margin;

    // Random RHS in [-10, 10]
    double rhs_raw = *rc::gen::inRange(-1000, 1001);
    sys.rhs[k] = static_cast<Scalar>(rhs_raw * 0.01);
  }

  return sys;
}

/// Precompute the Thomas algorithm coefficients (a_tri, alpha_tri, gamma_tri) from
/// a standard tridiagonal system (lower, diag, upper).
///
/// The kernel's forward sweep is (for k=1..n-1):
///   rw_p(k) = (rw_p(k) - dts^2 * a_tri(k) * rw_p(k-1)) * alpha_tri(k)
///
/// The kernel's backward sweep is (for k=n-1..0):
///   rw_p(k) = rw_p(k) - gamma_tri(k) * rw_p(k+1)
///
/// IMPORTANT: The kernel does NOT apply alpha_tri to level 0 during the forward
/// sweep. The kernel expects rw_p(0) to enter already pre-divided by the level-0
/// diagonal, i.e. rw_p(0) = b(0) * alpha_tri(0). This matches the Reference_Model
/// where the explicit tendency at level 1 (Fortran) is already normalized.
///
/// We set dts = 1.0 so that dts^2 = 1.0, making:
///   dts^2 * a_tri(k) = lower(k)  =>  a_tri(k) = lower(k)
///
/// For the Thomas algorithm:
///   d'(0) = diag(0)
///   alpha_tri(0) = 1 / d'(0)
///   gamma_tri(0) = upper(0) / d'(0) = upper(0) * alpha_tri(0)
///
///   For k = 1..n-1:
///     d'(k) = diag(k) - lower(k) * gamma_tri(k-1)
///     alpha_tri(k) = 1 / d'(k)
///     gamma_tri(k) = upper(k) / d'(k) = upper(k) * alpha_tri(k)
struct PrecomputedCoeffs {
  std::vector<Scalar> a_tri;      ///< Lower diagonal (= lower[k] when dts=1)
  std::vector<Scalar> alpha_tri;  ///< Reciprocal of modified diagonal
  std::vector<Scalar> gamma_tri;  ///< Back-substitution ratio
};

PrecomputedCoeffs precomputeCoefficients(const TridiagonalSystem& sys) {
  const int n = sys.n;
  PrecomputedCoeffs coeffs;
  coeffs.a_tri.resize(n, Scalar{0});
  coeffs.alpha_tri.resize(n, Scalar{0});
  coeffs.gamma_tri.resize(n, Scalar{0});

  // a_tri(k) = lower(k) (since dts^2 = 1)
  for (int k = 0; k < n; ++k) {
    coeffs.a_tri[k] = sys.lower[k];
  }

  // Forward elimination to get alpha_tri and gamma_tri
  // d'(0) = diag(0)
  Scalar d_prime = sys.diag[0];
  coeffs.alpha_tri[0] = Scalar{1} / d_prime;
  coeffs.gamma_tri[0] = sys.upper[0] * coeffs.alpha_tri[0];

  for (int k = 1; k < n; ++k) {
    // d'(k) = diag(k) - lower(k) * gamma_tri(k-1)
    d_prime = sys.diag[k] - sys.lower[k] * coeffs.gamma_tri[k - 1];
    coeffs.alpha_tri[k] = Scalar{1} / d_prime;
    if (k < n - 1) {
      coeffs.gamma_tri[k] = sys.upper[k] * coeffs.alpha_tri[k];
    } else {
      coeffs.gamma_tri[k] = Scalar{0}; // No upper entry for the last row
    }
  }

  return coeffs;
}

/// Compute the matrix-vector product A*x for a tridiagonal system.
/// Returns A*x as a vector.
std::vector<Scalar> matVecProduct(const TridiagonalSystem& sys,
                                  const std::vector<Scalar>& x) {
  const int n = sys.n;
  std::vector<Scalar> result(n, Scalar{0});

  for (int k = 0; k < n; ++k) {
    result[k] = sys.diag[k] * x[k];
    if (k > 0) {
      result[k] += sys.lower[k] * x[k - 1];
    }
    if (k < n - 1) {
      result[k] += sys.upper[k] * x[k + 1];
    }
  }
  return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Property 10: Tridiagonal solve satisfies its system
// Feature: mpas-dycore-cpp-port, Property 10: Tridiagonal solve
// ---------------------------------------------------------------------------

/// For any diagonally-dominant tridiagonal system A and right-hand side b,
/// the Thomas algorithm (tridiagonal_solve_no_damping) produces a solution x
/// such that A*x == b to within the Parity_Tolerance.
RC_GTEST_PROP(TridiagonalSolveProperty,
              SolveSatisfiesSystem,
              ()) {
  // Generate a random diagonally-dominant tridiagonal system
  auto sys = generateDiagonallyDominantSystem();
  const int n = sys.n;
  const int nCells = *rc::gen::inRange(1, 11); // Test with multiple columns

  // Precompute Thomas algorithm coefficients
  auto coeffs = precomputeCoefficients(sys);

  // Create device views with the required layout: (nVertLevels[+1], nCells)
  // rw_p has shape (nVertLevels+1, nCells) — the boundary entry at index n is 0
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> rw_p("rw_p", n + 1, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> a_tri_dev("a_tri", n, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> alpha_tri_dev("alpha_tri", n, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> gamma_tri_dev("gamma_tri", n, nCells);

  // Fill device views: same system for all cells (could vary per cell, but this
  // tests the property across multiple columns with the same system)
  auto rw_p_host = Kokkos::create_mirror_view(rw_p);
  auto a_tri_host = Kokkos::create_mirror_view(a_tri_dev);
  auto alpha_tri_host = Kokkos::create_mirror_view(alpha_tri_dev);
  auto gamma_tri_host = Kokkos::create_mirror_view(gamma_tri_dev);

  for (int iCell = 0; iCell < nCells; ++iCell) {
    // Set the RHS as the initial rw_p values (levels 0..n-1)
    // IMPORTANT: Level 0 must be pre-divided by the diagonal (pre-scaled by
    // alpha_tri(0)) because the kernel's forward sweep starts at k=1 and does
    // NOT apply alpha_tri to level 0. This matches the Reference_Model behavior
    // where the explicit tendency at level 1 (Fortran) enters already normalized.
    rw_p_host(0, iCell) = sys.rhs[0] * coeffs.alpha_tri[0];
    for (int k = 1; k < n; ++k) {
      rw_p_host(k, iCell) = sys.rhs[k];
    }
    // Boundary value at level n is 0 (top boundary condition)
    rw_p_host(n, iCell) = Scalar{0};

    // Set coefficients
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

  // Run the tridiagonal solve (dts = 1.0 so dts^2 = 1.0)
  mpas::dycore::TridiagonalSolveParams params;
  params.nVertLevels = n;
  params.nCells = nCells;
  params.dts = Scalar{1.0};

  // Create const views for the kernel
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> a_tri_const = a_tri_dev;
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> alpha_tri_const = alpha_tri_dev;
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> gamma_tri_const = gamma_tri_dev;

  mpas::dycore::tridiagonal_solve_no_damping<ExecSpace>(
      rw_p, a_tri_const, alpha_tri_const, gamma_tri_const, params);
  Kokkos::fence("tridiagonal_solve fence");

  // Read back the solution
  Kokkos::deep_copy(rw_p_host, rw_p);

  // Verify A * x == b for each cell column
  for (int iCell = 0; iCell < nCells; ++iCell) {
    // Extract solution x from rw_p (levels 0..n-1)
    std::vector<Scalar> x(n);
    for (int k = 0; k < n; ++k) {
      x[k] = rw_p_host(k, iCell);
    }

    // Compute A * x
    auto Ax = matVecProduct(sys, x);

    // Assert A*x == b within tolerance for each level
    for (int k = 0; k < n; ++k) {
      Scalar diff = std::abs(Ax[k] - sys.rhs[k]);
      Scalar scale = std::max(Scalar{1.0}, std::abs(sys.rhs[k]));
      RC_ASSERT(diff / scale <= parity_tolerance);
    }
  }
}

/// For any diagonally-dominant tridiagonal system, the solve produces finite
/// (non-NaN, non-Inf) values in all solution entries, verifying stability of
/// the Thomas algorithm for well-conditioned systems.
RC_GTEST_PROP(TridiagonalSolveProperty,
              SolveProducesFiniteValues,
              ()) {
  auto sys = generateDiagonallyDominantSystem();
  const int n = sys.n;
  const int nCells = 1;

  auto coeffs = precomputeCoefficients(sys);

  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> rw_p("rw_p", n + 1, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> a_tri_dev("a_tri", n, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> alpha_tri_dev("alpha_tri", n, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> gamma_tri_dev("gamma_tri", n, nCells);

  auto rw_p_host = Kokkos::create_mirror_view(rw_p);
  auto a_tri_host = Kokkos::create_mirror_view(a_tri_dev);
  auto alpha_tri_host = Kokkos::create_mirror_view(alpha_tri_dev);
  auto gamma_tri_host = Kokkos::create_mirror_view(gamma_tri_dev);

  for (int k = 0; k < n; ++k) {
    rw_p_host(k, 0) = (k == 0) ? sys.rhs[k] * coeffs.alpha_tri[0] : sys.rhs[k];
    a_tri_host(k, 0) = coeffs.a_tri[k];
    alpha_tri_host(k, 0) = coeffs.alpha_tri[k];
    gamma_tri_host(k, 0) = coeffs.gamma_tri[k];
  }
  rw_p_host(n, 0) = Scalar{0};

  Kokkos::deep_copy(rw_p, rw_p_host);
  Kokkos::deep_copy(a_tri_dev, a_tri_host);
  Kokkos::deep_copy(alpha_tri_dev, alpha_tri_host);
  Kokkos::deep_copy(gamma_tri_dev, gamma_tri_host);

  mpas::dycore::TridiagonalSolveParams params;
  params.nVertLevels = n;
  params.nCells = nCells;
  params.dts = Scalar{1.0};

  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> a_tri_const = a_tri_dev;
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> alpha_tri_const = alpha_tri_dev;
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> gamma_tri_const = gamma_tri_dev;

  mpas::dycore::tridiagonal_solve_no_damping<ExecSpace>(
      rw_p, a_tri_const, alpha_tri_const, gamma_tri_const, params);
  Kokkos::fence("tridiagonal_solve finite fence");

  Kokkos::deep_copy(rw_p_host, rw_p);

  // Every solution entry must be finite
  for (int k = 0; k < n; ++k) {
    RC_ASSERT(std::isfinite(rw_p_host(k, 0)));
  }
}

/// For any diagonally-dominant tridiagonal system with varying RHS per column,
/// the solve produces correct solutions independently in each column (verifying
/// the parallelization over horizontal elements preserves the vertical dependency).
RC_GTEST_PROP(TridiagonalSolveProperty,
              IndependentColumnsSolveCorrectly,
              ()) {
  // Generate a system (matrix is the same for all columns)
  auto sys = generateDiagonallyDominantSystem();
  const int n = sys.n;
  const int nCells = *rc::gen::inRange(2, 9); // Multiple columns

  auto coeffs = precomputeCoefficients(sys);

  // Generate different RHS for each column
  std::vector<std::vector<Scalar>> rhs_per_cell(nCells);
  for (int iCell = 0; iCell < nCells; ++iCell) {
    rhs_per_cell[iCell].resize(n);
    for (int k = 0; k < n; ++k) {
      double raw = *rc::gen::inRange(-1000, 1001);
      rhs_per_cell[iCell][k] = static_cast<Scalar>(raw * 0.01);
    }
  }

  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> rw_p("rw_p", n + 1, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> a_tri_dev("a_tri", n, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> alpha_tri_dev("alpha_tri", n, nCells);
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace> gamma_tri_dev("gamma_tri", n, nCells);

  auto rw_p_host = Kokkos::create_mirror_view(rw_p);
  auto a_tri_host = Kokkos::create_mirror_view(a_tri_dev);
  auto alpha_tri_host = Kokkos::create_mirror_view(alpha_tri_dev);
  auto gamma_tri_host = Kokkos::create_mirror_view(gamma_tri_dev);

  for (int iCell = 0; iCell < nCells; ++iCell) {
    for (int k = 0; k < n; ++k) {
      // Pre-scale level 0 by alpha_tri(0) (kernel expects this)
      rw_p_host(k, iCell) = (k == 0)
          ? rhs_per_cell[iCell][k] * coeffs.alpha_tri[0]
          : rhs_per_cell[iCell][k];
      a_tri_host(k, iCell) = coeffs.a_tri[k];
      alpha_tri_host(k, iCell) = coeffs.alpha_tri[k];
      gamma_tri_host(k, iCell) = coeffs.gamma_tri[k];
    }
    rw_p_host(n, iCell) = Scalar{0};
  }

  Kokkos::deep_copy(rw_p, rw_p_host);
  Kokkos::deep_copy(a_tri_dev, a_tri_host);
  Kokkos::deep_copy(alpha_tri_dev, alpha_tri_host);
  Kokkos::deep_copy(gamma_tri_dev, gamma_tri_host);

  mpas::dycore::TridiagonalSolveParams params;
  params.nVertLevels = n;
  params.nCells = nCells;
  params.dts = Scalar{1.0};

  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> a_tri_const = a_tri_dev;
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> alpha_tri_const = alpha_tri_dev;
  Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace> gamma_tri_const = gamma_tri_dev;

  mpas::dycore::tridiagonal_solve_no_damping<ExecSpace>(
      rw_p, a_tri_const, alpha_tri_const, gamma_tri_const, params);
  Kokkos::fence("tridiagonal_solve columns fence");

  Kokkos::deep_copy(rw_p_host, rw_p);

  // Verify A * x == b for each cell column independently
  for (int iCell = 0; iCell < nCells; ++iCell) {
    std::vector<Scalar> x(n);
    for (int k = 0; k < n; ++k) {
      x[k] = rw_p_host(k, iCell);
    }

    // Use the per-cell RHS
    TridiagonalSystem cell_sys = sys;
    cell_sys.rhs = rhs_per_cell[iCell];

    auto Ax = matVecProduct(cell_sys, x);

    for (int k = 0; k < n; ++k) {
      Scalar diff = std::abs(Ax[k] - cell_sys.rhs[k]);
      Scalar scale = std::max(Scalar{1.0}, std::abs(cell_sys.rhs[k]));
      RC_ASSERT(diff / scale <= parity_tolerance);
    }
  }
}
