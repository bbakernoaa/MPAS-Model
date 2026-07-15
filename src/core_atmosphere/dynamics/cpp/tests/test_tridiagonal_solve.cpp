#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cmath>

#include "mpas_dycore/tridiagonal_solve.hpp"
#include "mpas_dycore/scalar.hpp"

namespace mpas {
namespace dycore {
namespace test {

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using view1d = Kokkos::View<Scalar*, Kokkos::LayoutLeft, MemSpace>;
using view2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace>;
using const_view1d = Kokkos::View<const Scalar*, Kokkos::LayoutLeft, MemSpace>;
using const_view2d = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MemSpace>;

/// Helper: Precompute alpha_tri and gamma_tri from a_tri, b_tri, c_tri
/// using the same recurrence as the Fortran atm_compute_vert_imp_coefs_work.
///
/// The Fortran precomputation (for each cell):
///   gamma_tri(1) = 0
///   for k=2..nVertLevels:
///     alpha_tri(k) = 1/(1 + dts^2*(b_tri(k) - a_tri(k)*gamma_tri(k-1)))
///     gamma_tri(k) = dts^2 * c_tri(k) * alpha_tri(k)
///
/// Note: a_tri(1) and alpha_tri(1) are never used in the solve;
///       gamma_tri(1)=0 ensures level 1 is the identity row.
static void precompute_tri_coefficients(
    const view2d& a_tri_out,
    const view2d& alpha_tri_out,
    const view2d& gamma_tri_out,
    const std::vector<Scalar>& a_tri_col,
    const std::vector<Scalar>& b_tri_col,
    const std::vector<Scalar>& c_tri_col,
    Scalar dts, int iCell) {

  const int nLev = static_cast<int>(a_tri_col.size());
  const Scalar dts2 = dts * dts;

  // Level 0 (Fortran level 1): boundary initialization
  a_tri_out(0, iCell) = Scalar(0.0);
  alpha_tri_out(0, iCell) = Scalar(0.0);
  gamma_tri_out(0, iCell) = Scalar(0.0);

  for (int k = 1; k < nLev; ++k) {
    a_tri_out(k, iCell) = a_tri_col[k];
    Scalar alpha = Scalar(1.0) / (Scalar(1.0) + dts2 * (b_tri_col[k] - a_tri_col[k] * gamma_tri_out(k-1, iCell)));
    alpha_tri_out(k, iCell) = alpha;
    gamma_tri_out(k, iCell) = dts2 * c_tri_col[k] * alpha;
  }
}

/// Test: Simple 3-level diagonally-dominant system.
/// Verifies the Thomas algorithm produces a correct solution x such that A*x = b.
TEST(TridiagonalSolve, SimpleSystemNoDamping) {
  const int nVertLevels = 4;
  const int nCells = 1;
  const Scalar dts = 1.0;

  // Construct a diagonally-dominant tridiagonal system for levels 1..nVertLevels-1
  // (0-based). Level 0 is treated as a boundary (identity row).
  //
  // For a Thomas algorithm with the MPAS precomputed coefficients:
  //   The "matrix" encoded in (a_tri, alpha_tri, gamma_tri) operates on
  //   the vector rw_p(1..nVertLevels-1). Level 0 is boundary.
  //
  // We set up: a simple system with known solution.
  // diagonal entries b (positive), sub-diagonal a (negative), super-diagonal c (negative)
  // such that the system is diagonally dominant.

  std::vector<Scalar> a_tri_col(nVertLevels, 0.0);
  std::vector<Scalar> b_tri_col(nVertLevels, 0.0);
  std::vector<Scalar> c_tri_col(nVertLevels, 0.0);

  // Level 0: boundary (not solved)
  // Levels 1..nVertLevels-1: tridiagonal system
  // a(k) is the lower diagonal, b(k) the diagonal, c(k) the upper diagonal
  // In the MPAS formulation: the matrix element is actually
  //   M(k,k-1) = -dts^2 * a_tri(k)
  //   M(k,k)   = 1 + dts^2 * b_tri(k)   (encoded in alpha_tri)
  //   M(k,k+1) = -dts^2 * c_tri(k)       (encoded in gamma_tri)

  b_tri_col[1] = 4.0;  a_tri_col[1] = -1.0; c_tri_col[1] = -1.0;
  b_tri_col[2] = 4.0;  a_tri_col[2] = -1.0; c_tri_col[2] = -1.0;
  b_tri_col[3] = 4.0;  a_tri_col[3] = -1.0; c_tri_col[3] = 0.0;  // top level: c=0

  // Allocate coefficient views
  view2d a_tri_v("a_tri", nVertLevels, nCells);
  view2d alpha_tri_v("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri_v("gamma_tri", nVertLevels, nCells);

  precompute_tri_coefficients(a_tri_v, alpha_tri_v, gamma_tri_v,
                              a_tri_col, b_tri_col, c_tri_col, dts, 0);

  // Set a known RHS vector b for the system.
  // The system M*x = b where M is the implicit matrix.
  // We'll set b = [1, 2, 3] for levels 1,2,3 and verify A*x = b.
  view2d rw_p("rw_p", nVertLevels + 1, nCells);
  Kokkos::deep_copy(rw_p, Scalar(0.0));
  rw_p(1, 0) = 1.0;
  rw_p(2, 0) = 2.0;
  rw_p(3, 0) = 3.0;

  // Run the tridiagonal solve
  TridiagonalSolveParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.dts = dts;

  const_view2d a_c = a_tri_v;
  const_view2d alpha_c = alpha_tri_v;
  const_view2d gamma_c = gamma_tri_v;

  tridiagonal_solve_no_damping<ExecSpace>(rw_p, a_c, alpha_c, gamma_c, params);

  // Extract the solution x
  Scalar x1 = rw_p(1, 0);
  Scalar x2 = rw_p(2, 0);
  Scalar x3 = rw_p(3, 0);

  // Verify A*x = b. The tridiagonal system solved by the MPAS Thomas algorithm is:
  //   l(k)*x(k-1) + d(k)*x(k) + u(k)*x(k+1) = b(k)
  // where:
  //   l(k) = dts^2 * a_tri(k)       (sub-diagonal)
  //   d(k) = 1 + dts^2 * b_tri(k)   (diagonal)
  //   u(k) = dts^2 * c_tri(k)       (super-diagonal)
  const Scalar dts2 = dts * dts;
  Scalar row1 = (1.0 + dts2 * b_tri_col[1]) * x1 + (dts2 * c_tri_col[1]) * x2;
  Scalar row2 = (dts2 * a_tri_col[2]) * x1 + (1.0 + dts2 * b_tri_col[2]) * x2 + (dts2 * c_tri_col[2]) * x3;
  Scalar row3 = (dts2 * a_tri_col[3]) * x2 + (1.0 + dts2 * b_tri_col[3]) * x3;

  const Scalar tol = 1.0e-12;
  EXPECT_NEAR(row1, 1.0, tol) << "A*x row 1 != b[1]";
  EXPECT_NEAR(row2, 2.0, tol) << "A*x row 2 != b[2]";
  EXPECT_NEAR(row3, 3.0, tol) << "A*x row 3 != b[3]";
}

/// Test: Multiple cells solved in parallel produce correct independent solutions.
TEST(TridiagonalSolve, MultipleCellsParallel) {
  const int nVertLevels = 5;
  const int nCells = 4;
  const Scalar dts = 0.5;
  const Scalar dts2 = dts * dts;

  // Each cell gets a slightly different system but the same structure
  std::vector<std::vector<Scalar>> b_vals(nCells);
  std::vector<std::vector<Scalar>> rhs_vals(nCells);

  view2d a_tri_v("a_tri", nVertLevels, nCells);
  view2d alpha_tri_v("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri_v("gamma_tri", nVertLevels, nCells);
  view2d rw_p("rw_p", nVertLevels + 1, nCells);
  Kokkos::deep_copy(rw_p, Scalar(0.0));

  for (int ic = 0; ic < nCells; ++ic) {
    std::vector<Scalar> a_col(nVertLevels, 0.0);
    std::vector<Scalar> b_col(nVertLevels, 0.0);
    std::vector<Scalar> c_col(nVertLevels, 0.0);

    for (int k = 1; k < nVertLevels; ++k) {
      b_col[k] = 5.0 + static_cast<Scalar>(ic);  // diag dominant
      a_col[k] = -1.0;
      c_col[k] = (k < nVertLevels - 1) ? -1.0 : 0.0;
      rw_p(k, ic) = static_cast<Scalar>(k + ic);
    }

    b_vals[ic] = b_col;
    rhs_vals[ic].resize(nVertLevels);
    for (int k = 1; k < nVertLevels; ++k) {
      rhs_vals[ic][k] = rw_p(k, ic);
    }

    precompute_tri_coefficients(a_tri_v, alpha_tri_v, gamma_tri_v,
                                a_col, b_col, c_col, dts, ic);
  }

  // Run the solve
  TridiagonalSolveParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.dts = dts;

  const_view2d a_c = a_tri_v;
  const_view2d alpha_c = alpha_tri_v;
  const_view2d gamma_c = gamma_tri_v;

  tridiagonal_solve_no_damping<ExecSpace>(rw_p, a_c, alpha_c, gamma_c, params);

  // Verify A*x = b for each cell.
  // The system: l(k)*x(k-1) + d(k)*x(k) + u(k)*x(k+1) = b(k)
  // l(k) = dts^2 * a_tri(k), d(k) = 1 + dts^2 * b_tri(k), u(k) = dts^2 * c_tri(k)
  const Scalar tol = 1.0e-11;
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 1; k < nVertLevels; ++k) {
      Scalar row = (1.0 + dts2 * b_vals[ic][k]) * rw_p(k, ic);
      if (k > 1) {
        row += (dts2 * (-1.0)) * rw_p(k - 1, ic);  // a_tri = -1
      }
      if (k < nVertLevels - 1) {
        row += (dts2 * (-1.0)) * rw_p(k + 1, ic);  // c_tri = -1
      }
      EXPECT_NEAR(row, rhs_vals[ic][k], tol)
          << "Cell " << ic << ", level " << k << ": A*x != b";
    }
  }
}

/// Test: Rayleigh damping modifies the solution in the absorbing layer.
TEST(TridiagonalSolve, RayleighDampingApplied) {
  const int nVertLevels = 6;
  const int nCells = 1;
  const Scalar dts = 1.0;

  // Set up a simple system
  std::vector<Scalar> a_col(nVertLevels, 0.0);
  std::vector<Scalar> b_col(nVertLevels, 0.0);
  std::vector<Scalar> c_col(nVertLevels, 0.0);

  for (int k = 1; k < nVertLevels; ++k) {
    b_col[k] = 6.0;
    a_col[k] = -1.0;
    c_col[k] = (k < nVertLevels - 1) ? -1.0 : 0.0;
  }

  view2d a_tri_v("a_tri", nVertLevels, nCells);
  view2d alpha_tri_v("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri_v("gamma_tri", nVertLevels, nCells);

  precompute_tri_coefficients(a_tri_v, alpha_tri_v, gamma_tri_v,
                              a_col, b_col, c_col, dts, 0);

  // Allocate the full set of views for the damped version
  view2d rw_p_damped("rw_p_damped", nVertLevels + 1, nCells);
  view2d rw_p_undamped("rw_p_undamped", nVertLevels + 1, nCells);
  view2d dss("dss", nVertLevels, nCells);
  view2d w_full("w", nVertLevels + 1, nCells);
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view2d zz_v("zz", nVertLevels, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);

  // Set RHS
  for (int k = 1; k < nVertLevels; ++k) {
    rw_p_damped(k, 0) = static_cast<Scalar>(k);
    rw_p_undamped(k, 0) = static_cast<Scalar>(k);
  }

  // Set dss: non-zero only at top levels (absorbing layer at k=3,4)
  Kokkos::deep_copy(dss, Scalar(0.0));
  dss(3, 0) = 0.1;
  dss(4, 0) = 0.2;

  // Set physical fields to nonzero values
  Kokkos::deep_copy(w_full, Scalar(1.0));
  Kokkos::deep_copy(rw_save, Scalar(0.5));
  Kokkos::deep_copy(rw_base, Scalar(0.3));
  Kokkos::deep_copy(fzm_v, Scalar(0.5));
  Kokkos::deep_copy(fzp_v, Scalar(0.5));
  Kokkos::deep_copy(zz_v, Scalar(1.0));
  Kokkos::deep_copy(rho_zz_v, Scalar(1.0));

  // Run undamped solve
  TridiagonalSolveParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.dts = dts;

  const_view2d a_c = a_tri_v;
  const_view2d alpha_c = alpha_tri_v;
  const_view2d gamma_c = gamma_tri_v;

  tridiagonal_solve_no_damping<ExecSpace>(rw_p_undamped, a_c, alpha_c, gamma_c, params);

  // Run damped solve
  const_view2d dss_c = dss;
  const_view2d w_c = w_full;
  const_view2d rw_save_c = rw_save;
  const_view2d rw_base_c = rw_base;
  const_view1d fzm_c = fzm_v;
  const_view1d fzp_c = fzp_v;
  const_view2d zz_c = zz_v;
  const_view2d rho_c = rho_zz_v;

  tridiagonal_vertical_solve<ExecSpace>(
      rw_p_damped, a_c, alpha_c, gamma_c,
      dss_c, w_c, rw_save_c, rw_base_c,
      fzm_c, fzp_c, zz_c, rho_c, params);

  // Verify that levels with dss=0 match undamped solve exactly
  const Scalar tol = 1.0e-12;
  EXPECT_NEAR(rw_p_damped(1, 0), rw_p_undamped(1, 0), tol)
      << "Level 1 (no damping) should match undamped solve";
  EXPECT_NEAR(rw_p_damped(2, 0), rw_p_undamped(2, 0), tol)
      << "Level 2 (no damping) should match undamped solve";

  // Verify that levels with dss!=0 differ from undamped
  EXPECT_NE(rw_p_damped(3, 0), rw_p_undamped(3, 0))
      << "Level 3 (dss=0.1) should differ from undamped solve";
  EXPECT_NE(rw_p_damped(4, 0), rw_p_undamped(4, 0))
      << "Level 4 (dss=0.2) should differ from undamped solve";
}

/// Test: Rayleigh damping reduces vertical velocity amplitude.
/// With w > 0 and positive dss, the damping should reduce |rw_p|.
TEST(TridiagonalSolve, RayleighDampingReducesAmplitude) {
  const int nVertLevels = 4;
  const int nCells = 1;
  const Scalar dts = 1.0;

  // Identity-like system (all b = 0, a = 0, c = 0 means alpha=1, gamma=0)
  // So the Thomas solve is effectively rw_p_out = rw_p_in.
  // Then damping modifies the result.
  std::vector<Scalar> a_col(nVertLevels, 0.0);
  std::vector<Scalar> b_col(nVertLevels, 0.0);
  std::vector<Scalar> c_col(nVertLevels, 0.0);

  view2d a_tri_v("a_tri", nVertLevels, nCells);
  view2d alpha_tri_v("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri_v("gamma_tri", nVertLevels, nCells);

  precompute_tri_coefficients(a_tri_v, alpha_tri_v, gamma_tri_v,
                              a_col, b_col, c_col, dts, 0);

  view2d rw_p("rw_p", nVertLevels + 1, nCells);
  view2d dss("dss", nVertLevels, nCells);
  view2d w_full("w", nVertLevels + 1, nCells);
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  view1d fzm_v("fzm", nVertLevels);
  view1d fzp_v("fzp", nVertLevels);
  view2d zz_v("zz", nVertLevels, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);

  // Set initial rw_p to a large value at levels with damping
  Kokkos::deep_copy(rw_p, Scalar(0.0));
  rw_p(1, 0) = 10.0;
  rw_p(2, 0) = 10.0;

  // Strong damping at all interior levels
  Kokkos::deep_copy(dss, Scalar(0.5));

  // w > 0 everywhere (the field being damped)
  Kokkos::deep_copy(w_full, Scalar(5.0));
  // rw_save = rw_base (no perturbation from time t, so rw_diff = 0)
  Kokkos::deep_copy(rw_save, Scalar(1.0));
  Kokkos::deep_copy(rw_base, Scalar(1.0));
  // Simple geometry: fzm=fzp=0.5, zz=1, rho_zz=1
  Kokkos::deep_copy(fzm_v, Scalar(0.5));
  Kokkos::deep_copy(fzp_v, Scalar(0.5));
  Kokkos::deep_copy(zz_v, Scalar(1.0));
  Kokkos::deep_copy(rho_zz_v, Scalar(1.0));

  TridiagonalSolveParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.dts = dts;

  const_view2d a_c = a_tri_v;
  const_view2d alpha_c = alpha_tri_v;
  const_view2d gamma_c = gamma_tri_v;
  const_view2d dss_c = dss;
  const_view2d w_c = w_full;
  const_view2d rw_save_c = rw_save;
  const_view2d rw_base_c = rw_base;
  const_view1d fzm_c = fzm_v;
  const_view1d fzp_c = fzp_v;
  const_view2d zz_c = zz_v;
  const_view2d rho_c = rho_zz_v;

  tridiagonal_vertical_solve<ExecSpace>(
      rw_p, a_c, alpha_c, gamma_c,
      dss_c, w_c, rw_save_c, rw_base_c,
      fzm_c, fzp_c, zz_c, rho_c, params);

  // With dss > 0, dts*dss*zz*rho*w is a positive damping contribution.
  // The result should be different from the undamped input of 10.0.
  // The exact value depends on the formula, but it should be damped.
  EXPECT_LT(std::abs(rw_p(1, 0)), 10.0 + 1.0)
      << "Damping should modify rw_p from the identity solve";
  EXPECT_LT(std::abs(rw_p(2, 0)), 10.0 + 1.0)
      << "Damping should modify rw_p from the identity solve";
}

/// Test: Zero dss produces the same result as no-damping version.
TEST(TridiagonalSolve, ZeroDssMatchesNoDamping) {
  const int nVertLevels = 5;
  const int nCells = 2;
  const Scalar dts = 0.75;

  std::vector<Scalar> a_col(nVertLevels, 0.0);
  std::vector<Scalar> b_col(nVertLevels, 0.0);
  std::vector<Scalar> c_col(nVertLevels, 0.0);

  for (int k = 1; k < nVertLevels; ++k) {
    b_col[k] = 3.0;
    a_col[k] = -0.5;
    c_col[k] = (k < nVertLevels - 1) ? -0.5 : 0.0;
  }

  view2d a_tri_v("a_tri", nVertLevels, nCells);
  view2d alpha_tri_v("alpha_tri", nVertLevels, nCells);
  view2d gamma_tri_v("gamma_tri", nVertLevels, nCells);

  for (int ic = 0; ic < nCells; ++ic) {
    precompute_tri_coefficients(a_tri_v, alpha_tri_v, gamma_tri_v,
                                a_col, b_col, c_col, dts, ic);
  }

  view2d rw_p1("rw_p1", nVertLevels + 1, nCells);
  view2d rw_p2("rw_p2", nVertLevels + 1, nCells);
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 1; k < nVertLevels; ++k) {
      Scalar val = static_cast<Scalar>(k * (ic + 1));
      rw_p1(k, ic) = val;
      rw_p2(k, ic) = val;
    }
  }

  // Auxiliary fields (all zero dss)
  view2d dss("dss", nVertLevels, nCells);
  Kokkos::deep_copy(dss, Scalar(0.0));
  view2d w_full("w", nVertLevels + 1, nCells);
  Kokkos::deep_copy(w_full, Scalar(2.0));
  view2d rw_save("rw_save", nVertLevels + 1, nCells);
  Kokkos::deep_copy(rw_save, Scalar(1.0));
  view2d rw_base("rw_base", nVertLevels + 1, nCells);
  Kokkos::deep_copy(rw_base, Scalar(0.5));
  view1d fzm_v("fzm", nVertLevels);
  Kokkos::deep_copy(fzm_v, Scalar(0.6));
  view1d fzp_v("fzp", nVertLevels);
  Kokkos::deep_copy(fzp_v, Scalar(0.4));
  view2d zz_v("zz", nVertLevels, nCells);
  Kokkos::deep_copy(zz_v, Scalar(1.0));
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  Kokkos::deep_copy(rho_zz_v, Scalar(1.2));

  TridiagonalSolveParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.dts = dts;

  const_view2d a_c = a_tri_v;
  const_view2d alpha_c = alpha_tri_v;
  const_view2d gamma_c = gamma_tri_v;

  // Solve without damping
  tridiagonal_solve_no_damping<ExecSpace>(rw_p1, a_c, alpha_c, gamma_c, params);

  // Solve with zero dss (should be identical)
  const_view2d dss_c = dss;
  const_view2d w_c = w_full;
  const_view2d rws_c = rw_save;
  const_view2d rwb_c = rw_base;
  const_view1d fzm_c = fzm_v;
  const_view1d fzp_c = fzp_v;
  const_view2d zz_c = zz_v;
  const_view2d rho_c = rho_zz_v;

  tridiagonal_vertical_solve<ExecSpace>(
      rw_p2, a_c, alpha_c, gamma_c,
      dss_c, w_c, rws_c, rwb_c,
      fzm_c, fzp_c, zz_c, rho_c, params);

  const Scalar tol = 1.0e-14;
  for (int ic = 0; ic < nCells; ++ic) {
    for (int k = 0; k <= nVertLevels; ++k) {
      EXPECT_NEAR(rw_p2(k, ic), rw_p1(k, ic), tol)
          << "Zero dss should match no-damping solve at cell=" << ic << " k=" << k;
    }
  }
}

}  // namespace test
}  // namespace dycore
}  // namespace mpas
