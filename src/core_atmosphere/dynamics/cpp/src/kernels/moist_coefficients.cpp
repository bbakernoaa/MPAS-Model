/// @file moist_coefficients.cpp
/// @brief Implementation of the moist coefficient computation kernel.
///
/// Computes cqw (cell interfaces) and cqu (edges) from moisture species mixing
/// ratios. This kernel is invoked at the beginning of each dynamics sub-cycle,
/// before the first RK stage, to provide moist density conversion factors for
/// the pressure gradient and buoyancy terms.
///
/// @section governing_equations Governing Equations
/// The total moisture mixing ratio at each cell and level is:
///   qtot(k, iCell) = sum_{s=moist_start}^{moist_end} q_s(k, iCell)
///
/// Interface values (cqw) average moisture VERTICALLY, then take the reciprocal:
///   cqw(k, iCell) = 1 / (1 + 0.5*(qtot(k-1, iCell) + qtot(k, iCell)))
/// with boundary conditions:
///   cqw(0, iCell) = 1 / (1 + qtot(0, iCell))                    (top)
///   cqw(nVertLevels, iCell) = 1 / (1 + qtot(nVertLevels-1, iCell)) (bottom)
///
/// Edge values (cqu) average moisture HORIZONTALLY across adjacent cells,
/// then take the reciprocal:
///   cqu(k, iEdge) = 1 / (1 + 0.5*(qtot(k, cell0) + qtot(k, cell1)))
/// with boundary edges using the single adjacent cell value.
///
/// NOTE: The order of operations matters. Averaging the reciprocals
/// (i.e., 0.5*(1/(1+q_a) + 1/(1+q_b))) is NOT equivalent to taking the
/// reciprocal of the averaged moisture (i.e., 1/(1 + 0.5*(q_a + q_b))).
/// This implementation follows the Fortran reference which averages moisture
/// first, then takes the reciprocal.
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
/// intermediate qtot workspace before averaging to interfaces and edges).
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

    // Temporary workspace for total moisture mixing ratio per cell and level.
    // Layout: flat array indexed as [k * nCells + iCell] (row-major for workspace).
    std::vector<real_type> qtot_storage(
        static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells));

    // Lambda to access qtot workspace in a 2D manner.
    auto qtot = [&](index_type k, index_type iCell) -> real_type& {
        return qtot_storage[static_cast<std::size_t>(k) * static_cast<std::size_t>(nCells)
                            + static_cast<std::size_t>(iCell)];
    };

    // Step 1: Compute total moisture qtot(k, iCell) at each cell and level.
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type sum = 0.0;
            for (index_type s = moist_start; s <= moist_end; ++s) {
                sum += scalars[s, k, iCell];
            }
            qtot(k, iCell) = sum;
        }
    }

    // Step 2: Compute cqw at cell interfaces.
    // The reciprocal is taken AFTER averaging moisture vertically.
    // Fortran: cqw(k,iCell) = 1/(1 + 0.5*(qtot(k,iCell)+qtot(k-1,iCell))) for k=2..nVertLevels (1-based)
    // C++ 0-based: k=1..nVertLevels-1 are interior interfaces.
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        // Top boundary (k=0): use level 0 moisture directly
        cqw[0, iCell] = 1.0 / (1.0 + qtot(0, iCell));

        // Interior interfaces: reciprocal of vertically-averaged moisture
        for (index_type k = 1; k < nVertLevels; ++k) {
            real_type qtotal = 0.5 * (qtot(k - 1, iCell) + qtot(k, iCell));
            cqw[k, iCell] = 1.0 / (1.0 + qtotal);
        }

        // Bottom boundary (k=nVertLevels): use last level moisture directly
        cqw[nVertLevels, iCell] = 1.0 / (1.0 + qtot(nVertLevels - 1, iCell));
    }

    // Step 3: Compute cqu at edges.
    // The reciprocal is taken AFTER averaging moisture horizontally across adjacent cells.
    // Fortran: cqu(k,iEdge) = 1/(1 + 0.5*(qtot(k,cell1)+qtot(k,cell2)))
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        index_type cell0 = mesh.cellsOnEdge[iEdge, 0];
        index_type cell1 = mesh.cellsOnEdge[iEdge, 1];

        if (cell1 == INVALID_INDEX) {
            // Boundary edge: use single adjacent cell value
            for (index_type k = 0; k < nVertLevels; ++k) {
                cqu[k, iEdge] = 1.0 / (1.0 + qtot(k, cell0));
            }
        } else {
            // Interior edge: reciprocal of horizontally-averaged moisture
            for (index_type k = 0; k < nVertLevels; ++k) {
                real_type qtotal = 0.5 * (qtot(k, cell0) + qtot(k, cell1));
                cqu[k, iEdge] = 1.0 / (1.0 + qtotal);
            }
        }
    }
}

} // namespace mpas::dycore::kernels
