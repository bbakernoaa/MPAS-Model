/// @file moist_coefficients.cpp
/// @brief Implementation of the moist coefficient computation kernel.
///
/// Computes cqw (cell interfaces) and cqu (edges) from moisture species mixing
/// ratios. This kernel is invoked at the beginning of each dynamics sub-cycle,
/// before the first RK stage, to provide moist density conversion factors for
/// the pressure gradient and buoyancy terms.
///
/// @section governing_equations Governing Equations
/// The moist coefficient at each cell and level is:
///   cq_cell(k, iCell) = 1 / (1 + sum_{s=moist_start}^{moist_end} q_s(k, iCell))
///
/// Interface values (cqw) are vertical averages:
///   cqw(k, iCell) = 0.5 * (cq_cell(k-1, iCell) + cq_cell(k, iCell))
/// with boundary conditions:
///   cqw(0, iCell) = cq_cell(0, iCell)              (top)
///   cqw(nVertLevels, iCell) = cq_cell(nVertLevels-1, iCell) (bottom)
///
/// Edge values (cqu) are horizontal averages:
///   cqu(k, iEdge) = 0.5 * (cq_cell(k, cell0) + cq_cell(k, cell1))
/// with boundary edges using the single adjacent cell value.
///
/// @reference Skamarock, W. C., et al. (2012), "A Multiscale Nonhydrostatic
/// Atmospheric Model Using Centroidal Voronoi Tesselations and C-Grid Staggering",
/// Mon. Wea. Rev., 140, 3090-3105. NCAR Tech Note NCAR/TN-497+STR.

#include <mpas_dycore/kernels/moist_coefficients.hpp>

#include <vector>

namespace mpas::dycore::kernels {

// Explicit instantiation for default layout and serial policy.
template void compute_moist_coefficients<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> cqw,
    Field2D<default_layout, unchecked_accessor> cqu,
    ConstField3D<default_layout, unchecked_accessor> scalars,
    const MeshConnectivity& mesh,
    index_type moist_start,
    index_type moist_end,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels);

/// @brief Implementation of compute_moist_coefficients.
///
/// Uses serial loops internally due to data dependencies (computing the
/// intermediate cq_cell workspace before averaging to interfaces and edges).
/// The execution policy parameter is accepted for API consistency but not used
/// for dispatch in this kernel.
template <typename Layout, ExecutionPolicy Policy>
void compute_moist_coefficients(
    Policy /*policy*/,
    Field2D<Layout, unchecked_accessor> cqw,
    Field2D<Layout, unchecked_accessor> cqu,
    ConstField3D<Layout, unchecked_accessor> scalars,
    const MeshConnectivity& mesh,
    index_type moist_start,
    index_type moist_end,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels)
{
    // Handle no-moisture case: moist_end < moist_start means no moisture species.
    if (moist_end < moist_start) {
        // Set all cqw to 1.0 (nVertLevels+1 interfaces per cell)
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type k = 0; k <= nVertLevels; ++k) {
                cqw[k, iCell] = 1.0;
            }
        }
        // Set all cqu to 1.0
        for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                cqu[k, iEdge] = 1.0;
            }
        }
        return;
    }

    // Temporary workspace for cell-level moist coefficients.
    // Layout: flat array indexed as [k * nCells + iCell] (row-major for workspace).
    std::vector<real_type> cq_cell_storage(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells));

    // Lambda to access cq_cell workspace in a 2D manner.
    auto cq_cell = [&](index_type k, index_type iCell) -> real_type& {
        return cq_cell_storage[static_cast<std::size_t>(k) * static_cast<std::size_t>(nCells)
                               + static_cast<std::size_t>(iCell)];
    };

    // Step 1: Compute cell-level moist coefficient cq_cell(k, iCell).
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type sum = 0.0;
            for (index_type s = moist_start; s <= moist_end; ++s) {
                sum += scalars[s, k, iCell];
            }
            cq_cell(k, iCell) = 1.0 / (1.0 + sum);
        }
    }

    // Step 2: Compute cqw at cell interfaces by vertical averaging.
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        // Top boundary: copy level 0 value
        cqw[0, iCell] = cq_cell(0, iCell);

        // Interior interfaces: average adjacent levels
        for (index_type k = 1; k < nVertLevels; ++k) {
            cqw[k, iCell] = 0.5 * (cq_cell(k - 1, iCell) + cq_cell(k, iCell));
        }

        // Bottom boundary: copy last level value
        cqw[nVertLevels, iCell] = cq_cell(nVertLevels - 1, iCell);
    }

    // Step 3: Compute cqu at edges by horizontal averaging of adjacent cells.
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        index_type cell0 = mesh.cellsOnEdge[iEdge, 0];
        index_type cell1 = mesh.cellsOnEdge[iEdge, 1];

        if (cell1 == INVALID_INDEX) {
            // Boundary edge: use single adjacent cell value
            for (index_type k = 0; k < nVertLevels; ++k) {
                cqu[k, iEdge] = cq_cell(k, cell0);
            }
        } else {
            // Interior edge: average the two adjacent cells
            for (index_type k = 0; k < nVertLevels; ++k) {
                cqu[k, iEdge] = 0.5 * (cq_cell(k, cell0) + cq_cell(k, cell1));
            }
        }
    }
}

} // namespace mpas::dycore::kernels
