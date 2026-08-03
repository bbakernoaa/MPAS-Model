/// @file divergence_damping.cpp
/// @brief Implementation of the 3D divergence damping kernel.
///
/// Applies del-2 divergence damping to the acoustic velocity perturbation (ru_p)
/// within the acoustic substep. This is a two-pass algorithm:
///   1. Compute cell-centered divergence from edge-based ru_p.
///   2. Apply a Laplacian damping correction back to ru_p on edges.
///
/// @section governing_equation Governing Equations
///
/// Pass 1 — Cell divergence accumulation:
///   For each cell iCell and each vertical level k:
///     div(k, iCell) = 0
///     for each edge j of iCell:
///       iEdge = edgesOnCell(iCell, j)
///       sign  = (cellsOnEdge(iEdge, 0) == iCell) ? +1.0 : -1.0
///       div(k, iCell) += sign * dvEdge(iEdge) * ru_p(k, iEdge)
///     div(k, iCell) *= invAreaCell(iCell)
///
/// Pass 2 — Edge damping update:
///   For each edge iEdge with two valid adjacent cells (cell0, cell1):
///     ru_p(k, iEdge) -= dts * divdamp_coef * (div(k, cell1) - div(k, cell0)) * invDcEdge(iEdge)
///
/// This matches the Fortran MPAS reference implementation in the acoustic
/// substep routine (atm_advance_acoustic_step).
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/kernels/divergence_damping.hpp>

namespace mpas::dycore::kernels {

// ============================================================================
// Explicit template instantiation for default layout and serial policy
// ============================================================================

template void apply_divergence_damping<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> ru_p,
    Field2D<default_layout, unchecked_accessor> divergence,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invAreaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,
    real_type dts,
    real_type divdamp_coef,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels);

// ============================================================================
// Implementation
// ============================================================================

/// @brief Implementation of apply_divergence_damping.
///
/// Two-pass algorithm:
///   Pass 1: Compute divergence at each cell by accumulating signed edge fluxes.
///   Pass 2: Apply damping correction to ru_p using the divergence gradient.
///
/// Gracefully handles zero-extent dimensions by returning early.
/// Boundary edges (cell1 == INVALID_INDEX) are skipped in Pass 2.
template <typename Layout, ExecutionPolicy Policy>
void apply_divergence_damping(
    Policy policy,
    Field2D<Layout, unchecked_accessor> ru_p,
    Field2D<Layout, unchecked_accessor> divergence,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invAreaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,
    real_type dts,
    real_type divdamp_coef,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels)
{
    // Zero-extent graceful handling.
    if (nCells == 0 || nEdges == 0 || nVertLevels == 0) {
        return;
    }

    // ========================================================================
    // Pass 1: Compute cell-centered divergence from edge-based ru_p.
    // ========================================================================
    // For each cell, sum signed edge contributions and scale by inverse area.
    // The sign convention is: positive when flow is outward from the cell.
    //   sign = +1 if cellsOnEdge(iEdge, 0) == iCell (edge normal points away)
    //   sign = -1 otherwise (edge normal points toward the cell)
    policy.parallel_for(nCells, nVertLevels,
        [&](index_type iCell, index_type k) {
            real_type div_sum = 0.0;
            const index_type n_edges = mesh.nEdgesOnCell[iCell];

            for (index_type j = 0; j < n_edges; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                // Determine sign: outward-pointing edge normal relative to this cell.
                const real_type sign =
                    (mesh.cellsOnEdge[iEdge, 0] == iCell) ? 1.0 : -1.0;

                div_sum += sign * dvEdge[iEdge] * ru_p[k, iEdge];
            }

            divergence[k, iCell] = div_sum * invAreaCell[iCell];
        });

    // ========================================================================
    // Pass 2: Apply divergence damping correction to ru_p on edges.
    // ========================================================================
    // For each interior edge (both adjacent cells valid), subtract the
    // Laplacian damping term proportional to the divergence gradient.
    policy.parallel_for(nEdges, nVertLevels,
        [&](index_type iEdge, index_type k) {
            const index_type cell0 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 1];

            // Skip boundary edges where one cell is missing.
            if (cell1 == INVALID_INDEX) {
                return;
            }

            ru_p[k, iEdge] -= dts * divdamp_coef
                * (divergence[k, cell1] - divergence[k, cell0])
                * invDcEdge[iEdge];
        });
}

} // namespace mpas::dycore::kernels
