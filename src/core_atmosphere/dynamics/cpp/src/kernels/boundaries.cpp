/// @file boundaries.cpp
/// @brief Implementation of lateral boundary condition kernels for regional MPAS.
///
/// Ports the specified-zone and relaxation-zone lateral boundary condition logic
/// from the Fortran `mpas_atm_boundaries` module to C++ using mdspan views.
///
/// @section algorithm Algorithm Overview
///
/// The lateral boundary condition (LBC) application in MPAS-Regional uses a two-zone
/// approach for nesting within a driving model:
///
///   1. Specified zone (outermost cells): prognostic fields are directly overwritten
///      with time-interpolated driving-model boundary data. These cells are fully
///      prescribed by the parent domain.
///
///   2. Relaxation zone (interior to specified zone): prognostic fields are nudged
///      toward boundary values using a Rayleigh-type exponential relaxation:
///        field = field + dt * relax_coef * (field_bdy - field)
///      The relaxation coefficient ramps from 1.0 (strongest nudging, adjacent to
///      specified zone) down to 0.0 (no nudging, interior cells).
///
/// This two-zone scheme prevents spurious wave reflections at the lateral boundaries
/// and ensures smooth transition between the prescribed boundary state and the
/// freely-evolving interior solution.
///
/// @reference Skamarock, W. C. et al. (2012), Mon. Wea. Rev., 140, 3090-3105.

#include <mpas_dycore/kernels/boundaries.hpp>

namespace mpas::dycore::kernels {

// ============================================================================
// Explicit template instantiations for default layout and serial policy
// ============================================================================

template void apply_lateral_boundary<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> field,
    ConstField2D<default_layout, unchecked_accessor> field_bdy,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> relax_coef,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMask,
    real_type dt,
    index_type nCells,
    index_type nVertLevels);

// ============================================================================
// apply_lateral_boundary Implementation
// ============================================================================

/// @brief Apply lateral boundary conditions (specified zone + relaxation zone).
///
/// For each cell iCell (0 to nCells-1):
///   - If specZoneMask[iCell] == 1.0: overwrite field with boundary data.
///   - Else if relax_coef[iCell] > 0.0: blend field toward boundary values.
///   - Otherwise: leave field unchanged (interior cell).
///
/// The relaxation formula:
///   field(k, iCell) += dt * relax_coef(iCell) * (field_bdy(k, iCell) - field(k, iCell))
///
/// is equivalent to exponential damping toward the driving-model state with a
/// time scale of 1 / relax_coef.
template <typename Layout, ExecutionPolicy Policy>
void apply_lateral_boundary(
    Policy policy,
    Field2D<Layout, unchecked_accessor> field,
    ConstField2D<Layout, unchecked_accessor> field_bdy,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> relax_coef,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMask,
    real_type dt,
    index_type nCells,
    index_type nVertLevels)
{
    if (nCells == 0 || nVertLevels == 0) return;

    policy.parallel_for(nCells, nVertLevels,
        [&](index_type iCell, index_type k) {
            if (specZoneMask[iCell] == 1.0) {
                // Specified zone: overwrite with boundary data
                field[k, iCell] = field_bdy[k, iCell];
            } else if (relax_coef[iCell] > 0.0) {
                // Relaxation zone: blend toward boundary values
                field[k, iCell] = field[k, iCell]
                    + dt * relax_coef[iCell]
                    * (field_bdy[k, iCell] - field[k, iCell]);
            }
            // Interior cells (relax_coef == 0, specZoneMask == 0): no change
        });
}

} // namespace mpas::dycore::kernels
