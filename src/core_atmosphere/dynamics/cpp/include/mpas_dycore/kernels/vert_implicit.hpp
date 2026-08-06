#pragma once

/// @file vert_implicit.hpp
/// @brief Vertical implicit coefficient computation kernel for the MPAS dynamical core.
///
/// This kernel computes the tridiagonal matrix coefficients used in the vertically
/// implicit treatment of acoustic/gravity wave modes during the acoustic timestep.
/// It is a faithful line-by-line port of the Fortran subroutine
/// `atm_compute_vert_imp_coefs_work` from mpas_atm_time_integration.F.
///
/// @section algorithm Algorithm
/// The vertically implicit solve couples the w (vertical velocity), theta
/// (potential temperature), and rho (density) equations through a tridiagonal
/// system in the vertical. The kernel computes intermediate coupling coefficients
/// (cofwr, cofwz, coftz, cofwt) and then assembles and LU-factorizes the
/// tridiagonal matrix (a_tri, alpha_tri, gamma_tri) via a forward sweep.
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief 1D mutable mdspan (dynamic extent).
using Span1D = std::mdspan<real_type,
    std::extents<index_type, std::dynamic_extent>>;

/// @brief 1D const mdspan (dynamic extent).
using ConstSpan1D = std::mdspan<const real_type,
    std::extents<index_type, std::dynamic_extent>>;

/// @brief Compute vertical implicit solve coefficients for the acoustic step.
///
/// Faithful port of `atm_compute_vert_imp_coefs_work`. Computes the intermediate
/// coefficients (cofwr, cofwz, coftz, cofwt) and performs the tridiagonal LU
/// factorization (a_tri, alpha_tri, gamma_tri) for the vertically implicit solver.
///
/// Key differences from the old formulation:
/// - cofrz is 1D: simply rdzw[k]
/// - cofwr = 0.5*gravity*(fzm[k]*zz[k] + fzp[k]*zz[k-1])  (no dts or epssm)
/// - cofwz = c2*(fzm[k]*zz[k]+fzp[k]*zz[k-1])*rdzu[k]*cqw[k]*(fzm[k]*p[k]+fzp[k]*p[k-1])
/// - coftz = theta at interfaces: fzm[k]*t[k] + fzp[k]*t[k-1]
/// - cofwt = 0.5*rcv*zz[k]*gravity*rb[k]/(1+qtot[k]) * p[k]/((rtb[k]+rt[k])*pb[k])
/// - Tridiagonal assembly uses etp/ewp weighting
/// - LU factorization: alpha_tri = 1/(1+dts^2*(b_tri - a_tri*gamma_tri(k-1)))
///                     gamma_tri = dts^2 * c_tri * alpha_tri
///
/// Indexing convention (0-based C++ vs 1-based Fortran):
/// - Fortran k=1 → C++ k=0
/// - Fortran k=2..nVertLevels → C++ k=1..nVertLevels-1
/// - coftz has nVertLevels+1 entries: coftz[0]=0, coftz[k] for k=1..nVL-1, coftz[nVL]=0
/// - cofwr, cofwz are at interfaces k=1..nVertLevels-1 (Fortran k=2..nVertLevels)
/// - cofwt is at levels k=0..nVertLevels-1 (Fortran k=1..nVertLevels)
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy.
///
/// @param[in]  policy       Execution policy (accepted for API consistency).
/// @param[out] cofwr        Interface coefficient for w-rho coupling (nVertLevels, nCells).
/// @param[out] cofwz        Interface coefficient for w-pressure coupling (nVertLevels, nCells).
/// @param[out] coftz        Theta at interfaces (nVertLevels+1, nCells).
/// @param[out] cofwt        Level coefficient for w-theta coupling (nVertLevels, nCells).
/// @param[out] a_tri        Tridiagonal lower diagonal (nVertLevels, nCells).
/// @param[out] alpha_tri    Tridiagonal LU factored inverse (nVertLevels, nCells).
/// @param[out] gamma_tri    Tridiagonal LU factored upper (nVertLevels, nCells).
/// @param[out] cofrz        1D output: simply rdzw[k] (nVertLevels).
/// @param[in]  zz           Terrain height factor (nVertLevels, nCells).
/// @param[in]  p            Modified pressure / Exner function (nVertLevels, nCells).
/// @param[in]  t            Moist potential temperature theta_m (nVertLevels, nCells).
/// @param[in]  rb           Base-state dry density (nVertLevels, nCells).
/// @param[in]  rtb          Base-state rho*theta (nVertLevels, nCells).
/// @param[in]  pb           Base-state pressure (nVertLevels, nCells).
/// @param[in]  rt           Perturbation rho*theta (nVertLevels, nCells).
/// @param[in]  cqw          Moist coefficient at interfaces (nVertLevels, nCells).
/// @param[in]  qtot         Total moisture mixing ratio (nVertLevels, nCells).
/// @param[in]  rdzw         Reciprocal of dz at cell centers (nVertLevels).
/// @param[in]  fzm          Interpolation weight from level k (nVertLevels).
/// @param[in]  fzp          Interpolation weight from level k-1 (nVertLevels).
/// @param[in]  rdzu         Reciprocal of dz at interfaces (nVertLevels).
/// @param[in]  etp          Off-centering weight at levels (nVertLevels).
/// @param[in]  ewp          Off-centering weight at interfaces (nVertLevels+1).
/// @param[in]  dts          Acoustic timestep.
/// @param[in]  nCells       Number of cells.
/// @param[in]  nVertLevels  Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_vert_imp_coefs(
    Policy policy,
    Field2D<Layout, unchecked_accessor> cofwr,       // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> cofwz,       // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> coftz,       // output: (nVertLevels+1, nCells)
    Field2D<Layout, unchecked_accessor> cofwt,       // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> a_tri,       // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> alpha_tri,   // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> gamma_tri,   // output: (nVertLevels, nCells)
    Span1D cofrz,                                    // output: 1D (nVertLevels)
    ConstField2D<Layout, unchecked_accessor> zz,     // input: terrain height factor (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> p,      // input: modified pressure (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> t,      // input: theta_m (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rb,     // input: base-state density (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rtb,    // input: base-state rho*theta (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> pb,     // input: base-state pressure (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rt,     // input: perturbation rho*theta (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> cqw,    // input: moist coefficient (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> qtot,   // input: total moisture (nVertLevels, nCells)
    ConstSpan1D rdzw,                                // input: reciprocal dz at centers (nVertLevels)
    ConstSpan1D fzm,                                 // input: interp weight from level k (nVertLevels)
    ConstSpan1D fzp,                                 // input: interp weight from level k-1 (nVertLevels)
    ConstSpan1D rdzu,                                // input: reciprocal dz at interfaces (nVertLevels)
    ConstSpan1D etp,                                 // input: off-centering at levels (nVertLevels)
    ConstSpan1D ewp,                                 // input: off-centering at interfaces (nVertLevels+1)
    real_type dts,                                   // acoustic timestep
    index_type nCells,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
