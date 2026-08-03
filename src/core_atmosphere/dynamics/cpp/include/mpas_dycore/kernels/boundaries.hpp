#pragma once

/// @file boundaries.hpp
/// @brief Lateral boundary condition kernels for regional MPAS configurations.
///
/// Implements the specified-zone and relaxation-zone lateral boundary conditions
/// from the Fortran `mpas_atm_boundaries` module. In regional MPAS simulations,
/// the outermost cells belong to the "specified zone" where prognostic fields
/// are overwritten entirely with boundary (driving model) data. Interior to that,
/// cells in the "relaxation zone" are blended toward boundary values using a
/// relaxation coefficient that ramps smoothly from 1 (full boundary) at the
/// inner edge of the specified zone to 0 (full interior) at the interior boundary
/// of the relaxation zone.
///
/// @section spec_zone Specified Zone
///
/// For cells where specZoneMask == 1:
///   field(k, iCell) = field_bdy(k, iCell)
///
/// @section relax_zone Relaxation Zone
///
/// For cells where relax_coef > 0 and specZoneMask != 1:
///   field(k, iCell) = field(k, iCell)
///                   + dt * relax_coef(iCell) * (field_bdy(k, iCell) - field(k, iCell))
///
/// This blends the interior solution toward the driving-model values at a rate
/// controlled by the relaxation coefficient and the timestep.
///
/// @reference Skamarock, W. C. et al. (2012), "A Multiscale Nonhydrostatic
/// Atmospheric Model Using Centroidal Voronoi Tesselations and C-Grid Staggering",
/// Mon. Wea. Rev., 140, 3090-3105.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Apply lateral boundary conditions to a 2D prognostic field.
///
/// For cells in the specified zone (specZoneMask == 1.0): the field is overwritten
/// with boundary values from the driving model.
/// For cells in the relaxation zone (relax_coef > 0.0): the field is blended
/// toward boundary values using a Rayleigh-type nudging formulation.
///
/// @tparam Layout  mdspan layout policy (default_layout = column-major).
/// @tparam Policy  Execution policy satisfying ExecutionPolicy concept.
///
/// @param[in]     policy        Execution policy instance.
/// @param[in,out] field         The prognostic field to apply BCs to (nVertLevels, nCells).
/// @param[in]     field_bdy     Boundary values from driving model (nVertLevels, nCells).
/// @param[in]     relax_coef    Relaxation coefficient per cell (0=interior, 1=boundary) (nCells).
/// @param[in]     specZoneMask  Specified zone mask per cell (1.0=in spec zone, 0.0=not) (nCells).
/// @param[in]     dt            Timestep for relaxation blending.
/// @param[in]     nCells        Number of cells to process.
/// @param[in]     nVertLevels   Number of vertical levels.
template <typename Layout = default_layout, ExecutionPolicy Policy = SerialPolicy>
void apply_lateral_boundary(
    Policy policy,
    Field2D<Layout, unchecked_accessor> field,
    ConstField2D<Layout, unchecked_accessor> field_bdy,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> relax_coef,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMask,
    real_type dt,
    index_type nCells,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
