#pragma once

/// @file acoustic.hpp
/// @brief Forward-backward acoustic sub-step kernel for the MPAS dynamical core.
///
/// Implements the acoustic sub-stepping inner loop (`atm_advance_acoustic_step_work`)
/// that advances fast acoustic and gravity wave modes within each Runge-Kutta stage.
/// This is the most computationally intensive kernel in the time integration, executing
/// multiple times per RK stage.
///
/// @section algorithm Algorithm Overview
///
/// The acoustic sub-step follows a forward-backward time integration:
///
/// **Phase 1: Update horizontal velocity perturbation (ru_p)**
///   For small_step != 1: pressure gradient + terrain metric correction
///   For small_step == 1: initialize from tendency only
///
/// **Phase 2: Save rtheta_pp_old**
///   Store current theta perturbation for use in implicit solve
///
/// **Phase 3: Column solve preparation**
///   Accumulate horizontal flux divergence for rho and theta equations
///   Combine with existing perturbations and tendencies
///   Accumulate wwAvg (pre-update contribution)
///
/// **Phase 4: Update rw_p (explicit part of vertically implicit solve)**
///   Compute the right-hand side of the vertical momentum equation coupling
///   w, theta, and rho through vertical pressure gradient
///
/// **Phase 5: Tridiagonal solve**
///   Forward sweep and backward substitution for the implicit vertical coupling
///
/// **Phase 6: Implicit Rayleigh damping on w**
///   Gravity-wave absorbing layer applied to w perturbation
///
/// **Phase 7: Accumulate wwAvg (post-update contribution)**
///
/// **Phase 8: Update rho_pp and rtheta_pp**
///   Final density and theta perturbations from updated rw_p
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.
/// @reference Klemp, J. B., Dudhia, J., and Hassiotis, A. (2008),
/// "An upper gravity-wave absorbing layer for NWP applications",
/// Mon. Wea. Rev., 136, 3987-4004.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Execute one forward-backward acoustic sub-step.
///
/// This is a faithful port of the Fortran `atm_advance_acoustic_step_work`
/// subroutine. It advances the acoustic perturbation variables (ru_p, rw_p,
/// rho_pp, rtheta_pp) by one small timestep (dts) and accumulates the
/// time-averaged fluxes (ruAvg, wwAvg) for later use in scalar transport
/// and large-step variable recovery.
///
/// The function handles both the first acoustic step (small_step == 1) where
/// perturbation variables are initialized, and subsequent steps where the
/// forward-backward integration proceeds with pressure gradient forcing.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy.
///
/// @param[in]     policy            Execution policy instance.
/// @param[in,out] ru_p              Horizontal momentum perturbation (nVertLevels, nEdges).
/// @param[in,out] rw_p              Vertical momentum perturbation (nVertLevels+1, nCells).
/// @param[in,out] rtheta_pp         Potential temperature perturbation (nVertLevels, nCells).
/// @param[in,out] rho_pp            Density perturbation (nVertLevels, nCells).
/// @param[out]    rtheta_pp_old     Previous theta perturbation for implicit solve (nVertLevels, nCells).
/// @param[in,out] ruAvg             Time-averaged horizontal mass flux (nVertLevels, nEdges).
/// @param[in,out] wwAvg             Time-averaged vertical mass flux (nVertLevels+1, nCells).
/// @param[in]     rho_zz            Base-state dry density (nVertLevels, nCells).
/// @param[in]     theta_m           Moist potential temperature (nVertLevels, nCells).
/// @param[in]     zz                Terrain height metric dz/dzeta (nVertLevels, nCells).
/// @param[in]     exner             Exner function (nVertLevels, nCells).
/// @param[in]     cqu               Moist coefficient on edges (nVertLevels, nEdges).
/// @param[in]     zxu               Terrain metric on edges (nVertLevels, nEdges).
/// @param[in]     cofwt             Implicit coefficient for w-theta coupling (nVertLevels, nCells).
/// @param[in]     coftz             Implicit coefficient theta-z (nVertLevels+1, nCells).
/// @param[in]     cofwr             Implicit coefficient w-rho (nVertLevels, nCells).
/// @param[in]     cofwz             Implicit coefficient w-z (nVertLevels, nCells).
/// @param[in]     a_tri             Tridiagonal lower diagonal (nVertLevels, nCells).
/// @param[in]     alpha_tri         Tridiagonal solve factor (nVertLevels, nCells).
/// @param[in]     gamma_tri         Tridiagonal upper diagonal (nVertLevels, nCells).
/// @param[in]     dss               Rayleigh damping coefficient (nVertLevels, nCells).
/// @param[in]     tend_ru           Horizontal momentum tendency (nVertLevels, nEdges).
/// @param[in]     tend_rho          Density tendency (nVertLevels, nCells).
/// @param[in]     tend_rt           Theta tendency (nVertLevels, nCells).
/// @param[in]     tend_rw           Vertical momentum tendency (nVertLevels+1, nCells).
/// @param[in]     w                 Vertical velocity (nVertLevels+1, nCells).
/// @param[in]     rw                Base-state rho*w (nVertLevels+1, nCells).
/// @param[in]     rw_save           Saved rho*w from RK stage start (nVertLevels+1, nCells).
/// @param[in]     mesh              Mesh connectivity views.
/// @param[in]     edgesOnCell_sign  Edge orientation signs (maxEdges, nCells) as real.
/// @param[in]     invDcEdge         Reciprocal distance between cell centers (nEdges).
/// @param[in]     invAreaCell       Reciprocal cell area (nCells).
/// @param[in]     dvEdge            Edge lengths (nEdges).
/// @param[in]     cofrz             Coefficient for rho-z coupling (nVertLevels).
/// @param[in]     rdzw              Reciprocal vertical spacing at levels (nVertLevels).
/// @param[in]     fzm               Vertical interpolation weight from level k (nVertLevels).
/// @param[in]     fzp               Vertical interpolation weight from level k-1 (nVertLevels).
/// @param[in]     etp               Explicit time-centering weight (nVertLevels).
/// @param[in]     etm               Implicit time-centering weight (nVertLevels).
/// @param[in]     ewp               Explicit w-centering weight after update (nVertLevels+1).
/// @param[in]     ewm               Implicit w-centering weight before update (nVertLevels+1).
/// @param[in]     specZoneMaskEdge  Specified zone mask on edges (nEdges).
/// @param[in]     specZoneMaskCell  Specified zone mask on cells (nCells).
/// @param[in]     dts               Acoustic sub-timestep.
/// @param[in]     small_step        Current acoustic sub-step number (1-based).
/// @param[in]     nCells            Number of owned cells (cellSolve range).
/// @param[in]     nCellsAll         Total number of cells including halos.
/// @param[in]     nEdges            Number of edges.
/// @param[in]     nVertLevels       Number of vertical levels.
/// @param[in]     maxEdges          Maximum edges per cell.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void advance_acoustic_step(
    Policy policy,
    // Prognostic perturbation fields (read-write)
    Field2D<Layout, unchecked_accessor> ru_p,            // (nVertLevels, nEdges)
    Field2D<Layout, unchecked_accessor> rw_p,            // (nVertLevels+1, nCells)
    Field2D<Layout, unchecked_accessor> rtheta_pp,       // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> rho_pp,          // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> rtheta_pp_old,   // (nVertLevels, nCells)
    // Accumulated flux fields (read-write)
    Field2D<Layout, unchecked_accessor> ruAvg,           // (nVertLevels, nEdges)
    Field2D<Layout, unchecked_accessor> wwAvg,           // (nVertLevels+1, nCells)
    // Base-state fields (read-only)
    ConstField2D<Layout, unchecked_accessor> rho_zz,     // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> theta_m,    // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> zz,         // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> exner,      // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> cqu,        // (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> zxu,        // (nVertLevels, nEdges)
    // Implicit solve coefficients (read-only)
    ConstField2D<Layout, unchecked_accessor> cofwt,      // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> coftz,      // (nVertLevels+1, nCells)
    ConstField2D<Layout, unchecked_accessor> cofwr,      // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> cofwz,      // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> a_tri,      // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> alpha_tri,  // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> gamma_tri,  // (nVertLevels, nCells)
    // Rayleigh damping coefficient
    ConstField2D<Layout, unchecked_accessor> dss,        // (nVertLevels, nCells)
    // Tendencies (read-only)
    ConstField2D<Layout, unchecked_accessor> tend_ru,    // (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> tend_rho,   // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> tend_rt,    // (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> tend_rw,    // (nVertLevels+1, nCells)
    // Velocity fields (read-only)
    ConstField2D<Layout, unchecked_accessor> w,          // (nVertLevels+1, nCells)
    ConstField2D<Layout, unchecked_accessor> rw,         // (nVertLevels+1, nCells)
    ConstField2D<Layout, unchecked_accessor> rw_save,    // (nVertLevels+1, nCells)
    // Mesh connectivity
    const MeshConnectivity& mesh,
    // Edge orientation signs: (maxEdges, nCells) stored as real
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign,
    // Geometry 1D arrays
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,    // (nEdges)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invAreaCell,  // (nCells)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,       // (nEdges)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> cofrz,        // (nVertLevels)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzw,         // (nVertLevels)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,          // (nVertLevels)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,          // (nVertLevels)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> etp,          // (nVertLevels)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> etm,          // (nVertLevels)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> ewp,          // (nVertLevels+1)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> ewm,          // (nVertLevels+1)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMaskEdge, // (nEdges)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMaskCell, // (nCells)
    // Scalar parameters
    real_type dts,
    index_type small_step,
    // Dimensions
    index_type nCells,
    index_type nCellsAll,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges);

} // namespace mpas::dycore::kernels
