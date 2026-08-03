/// @file tend_theta.cpp
/// @brief Implementation of the thermodynamic (theta) tendency kernel.
///
/// Computes the time tendency of potential temperature (theta) from
/// horizontal flux divergence of rho*theta on the cell stencil.
///
/// @section governing_equation Governing Equation
///
/// For each cell iCell and vertical level k:
///   tend_theta(k, iCell) = -(1/areaCell(iCell)) * sum_{j=0}^{nEdgesOnCell-1}(
///       sign_j * rtheta_flux(k, iEdge_j) * dvEdge(iEdge_j))
///
/// where sign_j = +1 if cellsOnEdge(iEdge_j, 0) == iCell, else -1.
///
/// The negative sign enforces the conservation property: the tendency is the
/// negative divergence of the flux, ensuring that theta is conserved globally
/// when integrated over the domain.
///
/// @reference Skamarock, W. C., and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/kernels/tendencies.hpp>

namespace mpas::dycore::kernels {

/// @brief Compute theta tendency from horizontal flux divergence.
///
/// Iterates over all owned cells and vertical levels, accumulating the
/// net rho*theta flux through each cell's edges. The result is divided
/// by the cell area and negated to produce the negative divergence form.
///
/// Handles zero-extent gracefully: if nCells==0 or nVertLevels==0, returns
/// immediately without touching any memory.
template <typename Layout, ExecutionPolicy Policy>
void compute_tend_theta(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_theta,
    ConstField2D<Layout, unchecked_accessor> rtheta_flux,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    index_type nCells,
    index_type nVertLevels)
{
    // Handle zero-extent gracefully.
    if (nCells == 0 || nVertLevels == 0) {
        return;
    }

    // Compute negative divergence of rho*theta flux for each cell.
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const index_type nEdgesOnThisCell = mesh.nEdgesOnCell[iCell];
        const real_type inv_area = 1.0 / areaCell[iCell];

        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type flux_sum = 0.0;

            for (index_type j = 0; j < nEdgesOnThisCell; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                // Sign convention: +1 if this cell is cellsOnEdge[iEdge, 0], -1 otherwise.
                const real_type sign =
                    (mesh.cellsOnEdge[iEdge, 0] == iCell) ? 1.0 : -1.0;

                flux_sum += sign * rtheta_flux[k, iEdge] * dvEdge[iEdge];
            }

            // Tendency is negative divergence of flux.
            tend_theta[k, iCell] = -flux_sum * inv_area;
        }
    }
}

// Explicit instantiation for default layout and serial policy.
template void compute_tend_theta<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> tend_theta,
    ConstField2D<default_layout, unchecked_accessor> rtheta_flux,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    index_type nCells,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
