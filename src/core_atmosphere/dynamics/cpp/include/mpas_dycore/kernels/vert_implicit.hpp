#pragma once

/// @file vert_implicit.hpp
/// @brief Vertical implicit coefficient computation kernel for the MPAS dynamical core.
///
/// This kernel computes the tridiagonal matrix coefficients used in the vertically
/// implicit treatment of acoustic/gravity wave modes during the acoustic timestep.
/// These coefficients depend on the acoustic timestep (dts) and the thermodynamic
/// state, and need recomputation only when dts changes between RK stages.
///
/// @section algorithm Algorithm
/// The vertically implicit solve couples the w (vertical velocity), theta
/// (potential temperature), and rho (density) equations through a tridiagonal
/// system in the vertical. The coupling coefficients are derived from the
/// linearized equations with an off-centering parameter (epssm = 0.5) that
/// controls the implicit/explicit split of the acoustic mode treatment.
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Compute vertical implicit solve coefficients for the acoustic step.
///
/// Produces the tridiagonal matrix coefficients used in the vertically implicit
/// treatment of acoustic/gravity wave modes. These coefficients depend on the
/// acoustic timestep (dts) and the thermodynamic state, and need recomputation
/// only when dts changes between RK stages.
///
/// The formulation follows Klemp et al. (2007) Eqs. 3.12-3.17. For interior
/// interfaces k = 1 to nVertLevels-1:
///
///   theta_interface = fzm[k] * theta_m[k, iCell] + fzp[k] * theta_m[k-1, iCell]
///   rho_interface   = fzm[k] * rho_zz[k, iCell]  + fzp[k] * rho_zz[k-1, iCell]
///
///   cofwr(k, iCell) = epssm * dts * rho_interface * rdzu[k]
///   cofwz(k, iCell) = epssm * dts * (rdry/cvdry) * theta_interface * rdzu[k] * cqw[k, iCell]
///   coftz(k, iCell) = epssm * dts * gravity * rho_zz[k, iCell]
///   cofwt(k, iCell) = epssm * dts * (theta_m[k, iCell] - theta_m[k-1, iCell]) * rdzu[k]
///
/// Tridiagonal assembly:
///   a_tri(k)     = -cofwz(k) * coftz(k)              (lower diagonal)
///   gamma_tri(k) = -cofwz(k+1) * coftz(k)            (upper diagonal)
///   alpha_tri(k) = 1 + cofwz(k)*coftz(k-1) + cofwz(k+1)*coftz(k) (main diagonal)
///   cofrz(k)     = cofwr(k) * rho_zz(k, iCell)
///
/// Boundary conditions: cofwr, cofwz, cofwt = 0 at k=0 and k=nVertLevels.
///
/// Output fields:
///   cofwr: coefficient for w in the rho equation (nVertLevels+1, nCells)
///   cofwz: coefficient for w in the z-momentum equation (nVertLevels+1, nCells)
///   coftz: coefficient for theta in z-momentum (nVertLevels, nCells)
///   cofwt: coefficient for w in the theta equation (nVertLevels+1, nCells)
///   a_tri: tridiagonal lower diagonal (nVertLevels, nCells)
///   alpha_tri: tridiagonal main diagonal factor (nVertLevels, nCells)
///   gamma_tri: tridiagonal upper diagonal (nVertLevels, nCells)
///   cofrz: coefficient for rho-zz in the z equation (nVertLevels, nCells)
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy.
///
/// @param[in]  policy       Execution policy (accepted for API consistency).
/// @param[out] cofwr        Coefficient for w in rho equation (nVertLevels+1, nCells).
/// @param[out] cofwz        Coefficient for w in z-momentum equation (nVertLevels+1, nCells).
/// @param[out] coftz        Coefficient for theta in z equation (nVertLevels, nCells).
/// @param[out] cofwt        Coefficient for w in theta equation (nVertLevels+1, nCells).
/// @param[out] a_tri        Tridiagonal lower diagonal (nVertLevels, nCells).
/// @param[out] alpha_tri    Tridiagonal main diagonal factor (nVertLevels, nCells).
/// @param[out] gamma_tri    Tridiagonal upper diagonal (nVertLevels, nCells).
/// @param[out] cofrz        Coefficient for rho-zz in z equation (nVertLevels, nCells).
/// @param[in]  theta_m      Moist potential temperature (nVertLevels, nCells).
/// @param[in]  rho_zz       Dry density times height factor (nVertLevels, nCells).
/// @param[in]  cqw          Moist coefficient at interfaces (nVertLevels+1, nCells).
/// @param[in]  rdzu         Reciprocal vertical spacing at interfaces (nVertLevels+1).
/// @param[in]  fzm          Interpolation weight to interface from level k (nVertLevels+1).
/// @param[in]  fzp          Interpolation weight to interface from level k-1 (nVertLevels+1).
/// @param[in]  dts          Acoustic timestep.
/// @param[in]  gravity      Gravitational acceleration.
/// @param[in]  rdry         Gas constant for dry air.
/// @param[in]  cvdry        Specific heat at constant volume for dry air.
/// @param[in]  nCells       Number of cells.
/// @param[in]  nVertLevels  Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_vert_imp_coefs(
    Policy policy,
    Field2D<Layout, unchecked_accessor> cofwr,       // output: (nVertLevels+1, nCells)
    Field2D<Layout, unchecked_accessor> cofwz,       // output: (nVertLevels+1, nCells)
    Field2D<Layout, unchecked_accessor> coftz,       // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> cofwt,       // output: (nVertLevels+1, nCells)
    Field2D<Layout, unchecked_accessor> a_tri,       // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> alpha_tri,   // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> gamma_tri,   // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> cofrz,       // output: (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> theta_m,// input: moist potential temp (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rho_zz, // input: dry density (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> cqw,    // input: moist coefficient (nVertLevels+1, nCells)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzu,  // input: reciprocal vertical spacing (nVertLevels+1)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,   // input: interp weight from level k (nVertLevels+1)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,   // input: interp weight from level k-1 (nVertLevels+1)
    real_type dts,                                    // acoustic timestep
    real_type gravity,                                // gravitational acceleration
    real_type rdry,                                   // gas constant for dry air
    real_type cvdry,                                  // specific heat at const volume
    index_type nCells,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
