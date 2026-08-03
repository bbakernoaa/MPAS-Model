#pragma once

/// @file scalars.hpp
/// @brief Scalar transport kernel with flux-corrected transport (FCT) limiting.
///
/// Implements the horizontal and vertical advection of scalar tracers with
/// monotone (Zalesak FCT) limiting. This is a faithful port of the Fortran
/// subroutine `atm_advance_scalars_mono_work` from
/// `src/core_atmosphere/dynamics/mpas_atm_time_integration.F`.
///
/// @section algorithm Algorithm Overview
///
/// For each scalar species, the transport proceeds in several phases:
///
/// **Phase 1: Vertical flux computation**
///   - Zero flux at top and bottom boundaries
///   - Linear interpolation at levels 2 and nVertLevels
///   - 3rd-order upwind-biased flux (flux3) at interior levels 3..nVertLevels-1
///   - Compute local min/max bounds (s_min, s_max) from scalar_old and neighbors
///
/// **Phase 2: High-order horizontal flux computation**
///   - For each edge, use advCellsForEdge stencil and adv_coefs/adv_coefs_3rd
///   - flux = sum_j (adv_coefs(j) + sign(1, ruAvg) * adv_coefs_3rd(j)) * scalar(cell_j)
///
/// **Phase 3: Upwind update + anti-diffusive flux decomposition**
///   - Upwind vertical flux: max(0,w)*scalar(k-1) + min(0,w)*scalar(k)
///   - Upwind horizontal flux: max(0,u)*scalar(cell1) + min(0,u)*scalar(cell2)
///   - Anti-diffusive flux: high_order_flux - upwind_flux
///   - Track incoming/outgoing anti-diffusive contributions (scale_in, scale_out)
///
/// **Phase 4: FCT limiter (Zalesak)**
///   - scale_in  = clamp((s_max * rho_new - scalar_upwind) / flux_in,  0, 1)
///   - scale_out = clamp((s_min * rho_new - scalar_upwind) / flux_out, 0, 1)
///   - Rescale anti-diffusive flux using min(sender_out, receiver_in)
///
/// **Phase 5: Final update**
///   - Apply rescaled anti-diffusive fluxes (horizontal + vertical)
///   - Divide by rho_zz_new to get mixing ratio
///   - Clamp negative values to zero (positive-definite)
///
/// @reference Skamarock, W. C. and Gassmann, A. (2011),
/// "Conservative Transport Schemes for the Climate and Weather Prediction
/// Models", Mon. Wea. Rev., 139, 3284-3300.
/// @reference Zalesak, S. T. (1979), "Fully Multidimensional Flux-Corrected
/// Transport Algorithms for Fluids", J. Comput. Phys., 31, 335-362.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Advance scalar tracers by one timestep using monotone transport.
///
/// Implements horizontal and vertical advection of scalar fields with
/// flux-corrected transport (FCT) limiting to ensure monotonicity.
/// This is a faithful port of the Fortran `atm_advance_scalars_mono_work`.
///
/// The algorithm:
///   1. Compute vertical flux using 3rd-order upwind-biased stencil
///   2. Compute high-order horizontal flux using advection coefficients
///   3. Decompose into upwind + anti-diffusive components
///   4. Determine local min/max bounds from pre-advection scalar field
///   5. Limit anti-diffusive flux using Zalesak FCT limiter
///   6. Apply limited fluxes and divide by new density
///   7. Clamp negative values to zero (positive-definite)
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy (accepted for API consistency; kernel
///                 uses serial iteration due to per-scalar phased algorithm).
///
/// @param[in]     policy              Execution policy instance.
/// @param[in,out] scalars             Scalar mixing ratios (nScalars, nVertLevels, nCells).
///                                    On input: current values (used as "scalar_new" for flux
///                                    computation and "scalar_old" for bounds/upwind).
///                                    On output: updated values after transport.
/// @param[in]     ruAvg               Time-averaged horizontal mass flux (nVertLevels, nEdges).
/// @param[in]     wwAvg               Time-averaged vertical mass flux (nVertLevels+1, nCells).
/// @param[in]     rho_zz_old          Dry density at start of timestep (nVertLevels, nCells).
/// @param[in]     rho_zz_new          Dry density at end of timestep (nVertLevels, nCells).
/// @param[in]     mesh                Mesh connectivity views.
/// @param[in]     dvEdge              Edge lengths (nEdges).
/// @param[in]     areaCell            Cell areas (nCells). Note: passed as invAreaCell
///                                    (reciprocal) for efficiency.
/// @param[in]     rdzw                Reciprocal vertical layer thickness (nVertLevels).
/// @param[in]     fzm                 Vertical interpolation weight from level k (nVertLevels).
/// @param[in]     fzp                 Vertical interpolation weight from level k-1 (nVertLevels).
/// @param[in]     advCellsForEdge     Advection stencil cells per edge (nEdges, maxAdvCells).
/// @param[in]     nAdvCellsForEdge    Number of advection cells per edge (nEdges).
/// @param[in]     adv_coefs           Advection coefficients, flattened (maxAdvCells, nEdges).
/// @param[in]     adv_coefs_3rd       3rd-order advection coefficients (maxAdvCells, nEdges).
/// @param[in]     dt                  Large timestep [s].
/// @param[in]     coef_3rd_order      3rd-order coefficient for vertical flux (typically 1.0).
/// @param[in]     nScalars            Number of scalar species.
/// @param[in]     nCells              Number of owned cells.
/// @param[in]     nEdges              Number of edges.
/// @param[in]     nVertLevels         Number of vertical levels.
/// @param[in]     maxAdvCells         Maximum advection cells per edge.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void advance_scalars_mono(
    Policy policy,
    Field3D<Layout, unchecked_accessor> scalars,          // in/out: (nScalars, nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> ruAvg,       // time-averaged mass flux (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> wwAvg,       // time-averaged vertical flux (nVertLevels+1, nCells)
    ConstField2D<Layout, unchecked_accessor> rho_zz_old,  // density at start of step (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> rho_zz_new,  // density at end of step (nVertLevels, nCells)
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invAreaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzw,   // reciprocal dz (nVertLevels)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,    // vert interp weight (nVertLevels)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,    // vert interp weight (nVertLevels)
    ConnectivityView advCellsForEdge,     // (nEdges, maxAdvCells)
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>> nAdvCellsForEdge, // (nEdges)
    ConstField2D<Layout, unchecked_accessor> adv_coefs,       // (maxAdvCells, nEdges)
    ConstField2D<Layout, unchecked_accessor> adv_coefs_3rd,   // (maxAdvCells, nEdges)
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign, // (maxEdges, nCells)
    real_type dt,
    real_type coef_3rd_order,  // 3rd order coefficient (typically 1.0 for upwind-biased)
    index_type nScalars,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxAdvCells);

} // namespace mpas::dycore::kernels
