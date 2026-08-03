/// @file les_models.cpp
/// @brief Implementation of LES turbulence model kernels.
///
/// Implements two eddy viscosity computation methods:
///   1. Fixed viscosity: constant value at all grid points.
///   2. 2D Smagorinsky: deformation-based spatially-varying viscosity.
///
/// @section smagorinsky_algorithm Smagorinsky Algorithm
///
/// For each cell, the velocity gradients are reconstructed from edge-normal (u)
/// and edge-tangential (v) velocities using precomputed deformation coefficients:
///
///   dudx(k) = sum_{j=0}^{nEdgesOnCell-1} coef_c2(j,iCell)*u(k,edge_j) - coef_cs(j,iCell)*v(k,edge_j)
///   dudy(k) = sum_{j=0}^{nEdgesOnCell-1} coef_cs(j,iCell)*u(k,edge_j) - coef_s2(j,iCell)*v(k,edge_j)
///   dvdx(k) = sum_{j=0}^{nEdgesOnCell-1} coef_cs(j,iCell)*u(k,edge_j) + coef_c2(j,iCell)*v(k,edge_j)
///   dvdy(k) = sum_{j=0}^{nEdgesOnCell-1} coef_s2(j,iCell)*u(k,edge_j) + coef_cs(j,iCell)*v(k,edge_j)
///
/// Deformation tensor:
///   D11 = 2*dudx,  D22 = 2*dvdy,  D12 = dudy + dvdx
///
/// Strain rate magnitude:
///   |S| = sqrt(0.25*(D11 - D22)^2 + D12^2)
///
/// Eddy viscosity with stability bound:
///   kdiff = (c_s * config_len_disp)^2 * |S|
///   kdiff = min(kdiff, 0.01 * config_len_disp^2 * invDt)
///
/// This matches the Fortran `smagorinsky_2d` subroutine in
/// `mpas_atm_dissipation_models.F`.

#include <mpas_dycore/kernels/les_models.hpp>

#include <algorithm>
#include <cmath>

namespace mpas::dycore::kernels {

// ============================================================================
// Explicit instantiation for default layout and serial policy
// ============================================================================

template void compute_2d_fixed_viscosity<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> eddy_visc_horz,
    real_type config_h_theta_eddy_visc2,
    index_type nCells,
    index_type nVertLevels);

template void compute_2d_smagorinsky<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> eddy_visc_horz,
    ConstField2D<default_layout, unchecked_accessor> u,
    ConstField2D<default_layout, unchecked_accessor> v,
    real_type c_s,
    real_type config_len_disp,
    real_type invDt,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_c2,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_s2,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_cs,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels);

// ============================================================================
// Implementation: compute_2d_fixed_viscosity
// ============================================================================

/// @brief Fill eddy viscosity with a constant value.
///
/// Sets eddy_visc_horz(k, iCell) = config_h_theta_eddy_visc2 for all k, iCell.
/// Gracefully handles zero-extent dimensions by returning early.
template <typename Layout, ExecutionPolicy Policy>
void compute_2d_fixed_viscosity(
    Policy policy,
    Field2D<Layout, unchecked_accessor> eddy_visc_horz,
    real_type config_h_theta_eddy_visc2,
    index_type nCells,
    index_type nVertLevels)
{
    // Zero-extent graceful handling.
    if (nCells == 0 || nVertLevels == 0) {
        return;
    }

    policy.parallel_for(nCells, nVertLevels,
        [&](index_type iCell, index_type k) {
            eddy_visc_horz[k, iCell] = config_h_theta_eddy_visc2;
        });
}

// ============================================================================
// Implementation: compute_2d_smagorinsky
// ============================================================================

/// @brief Compute 2D Smagorinsky deformation-based eddy viscosity.
///
/// For each cell, reconstructs velocity gradients from edge-based velocities
/// using deformation coefficients, computes the strain rate magnitude, and
/// derives the eddy viscosity with a stability upper bound.
///
/// The algorithm preserves the exact operation order of the Fortran reference
/// implementation to enable bit-for-bit reproducibility under strict FP semantics.
template <typename Layout, ExecutionPolicy Policy>
void compute_2d_smagorinsky(
    Policy policy,
    Field2D<Layout, unchecked_accessor> eddy_visc_horz,
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> v,
    real_type c_s,
    real_type config_len_disp,
    real_type invDt,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_c2,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_s2,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_cs,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels)
{
    // Zero-extent graceful handling.
    if (nCells == 0 || nVertLevels == 0) {
        return;
    }

    // Precompute constant factors.
    const real_type smag_factor = (c_s * config_len_disp) * (c_s * config_len_disp);
    const real_type upper_bound = 0.01 * config_len_disp * config_len_disp * invDt;

    // The outer loop is over cells. For each cell, we accumulate velocity
    // gradients from the edge velocities using the deformation coefficients,
    // then compute the deformation tensor and resulting eddy viscosity.
    //
    // Note: We use a cell-level serial loop with a vertical-level inner loop
    // to match the Fortran implementation's operation order (which accumulates
    // edge contributions for all levels before computing the final viscosity).
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const index_type n_edges = mesh.nEdgesOnCell[iCell];

        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type dudx = 0.0;
            real_type dudy = 0.0;
            real_type dvdx = 0.0;
            real_type dvdy = 0.0;

            for (index_type j = 0; j < n_edges; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                const real_type coef_c2 = deformation_coef_c2[j, iCell];
                const real_type coef_s2 = deformation_coef_s2[j, iCell];
                const real_type coef_cs = deformation_coef_cs[j, iCell];
                const real_type u_edge  = u[k, iEdge];
                const real_type v_edge  = v[k, iEdge];

                dudx += coef_c2 * u_edge - coef_cs * v_edge;
                dudy += coef_cs * u_edge - coef_s2 * v_edge;
                dvdx += coef_cs * u_edge + coef_c2 * v_edge;
                dvdy += coef_s2 * u_edge + coef_cs * v_edge;
            }

            // Deformation tensor components.
            const real_type d_11 = 2.0 * dudx;
            const real_type d_22 = 2.0 * dvdy;
            const real_type d_12 = dudy + dvdx;

            // Strain rate magnitude using the Smagorinsky formulation.
            const real_type strain_rate =
                std::sqrt(0.25 * (d_11 - d_22) * (d_11 - d_22) + d_12 * d_12);

            // Eddy viscosity with stability upper bound.
            eddy_visc_horz[k, iCell] = std::min(
                smag_factor * strain_rate, upper_bound);
        }
    }
}

} // namespace mpas::dycore::kernels
