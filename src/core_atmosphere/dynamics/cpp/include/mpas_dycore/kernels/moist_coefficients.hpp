#pragma once

/// @file moist_coefficients.hpp
/// @brief Moist coefficient computation kernel for the MPAS dynamical core.
///
/// This kernel computes the moist coefficients cqw (at cell interfaces) and
/// cqu (at edges) from the moisture species mixing ratios. These coefficients
/// convert between dry and moist density and are used in the pressure gradient
/// and buoyancy terms of the dynamics equations.
///
/// The moist coefficients are recomputed at the beginning of each dynamics
/// sub-cycle, before the first RK stage.
///
/// @section algorithm Algorithm
/// For each cell and level, a cell-centered moist coefficient is computed as:
///   cq_cell(k, iCell) = 1 / (1 + sum_{s=moist_start}^{moist_end} scalars(s, k, iCell))
///
/// The interface values cqw are computed as vertical averages of adjacent levels.
/// The edge values cqu are computed as horizontal averages of the two cells
/// sharing each edge.
///
/// @reference Skamarock, W. C., et al. (2012), "A Multiscale Nonhydrostatic
/// Atmospheric Model Using Centroidal Voronoi Tesselations and C-Grid Staggering",
/// Mon. Wea. Rev., 140, 3090-3105. NCAR Tech Note NCAR/TN-497+STR.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Compute moist coefficients cqu (edges) and cqw (cell interfaces).
///
/// Governing relation:
/// @f[
///   c_q = \frac{1}{1 + \sum_{s=\text{moist\_start}}^{\text{moist\_end}} q_s}
/// @f]
/// where @f$ q_s @f$ are moisture mixing ratios from moist_start to moist_end
/// inclusive.
///
/// For cqw at cell interfaces: average of adjacent level values.
/// For cqu at edges: average of the two cells sharing the edge.
///
/// When moist_end < moist_start (no moisture species), sets cqu=1.0
/// and cqw=1.0 everywhere.
///
/// @reference Skamarock et al. (2012), "A Multiscale Nonhydrostatic
/// Atmospheric Model", NCAR Tech Note NCAR/TN-497+STR.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy for parallelization.
///
/// @param[in]  policy       Execution policy (accepted for API consistency).
/// @param[out] cqw          Moist coefficient at cell interfaces (nVertLevels+1, nCells).
/// @param[out] cqu          Moist coefficient at edges (nVertLevels, nEdges).
/// @param[in]  scalars      Scalar mixing ratios (nScalars, nVertLevels, nCells).
/// @param[in]  mesh         Mesh connectivity (cellsOnEdge used).
/// @param[in]  moist_start  First moisture species index (0-based).
/// @param[in]  moist_end    Last moisture species index (inclusive, 0-based).
/// @param[in]  nCells       Number of cells.
/// @param[in]  nEdges       Number of edges.
/// @param[in]  nVertLevels  Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_moist_coefficients(
    Policy policy,
    Field2D<Layout, unchecked_accessor> cqw,
    Field2D<Layout, unchecked_accessor> cqu,
    ConstField3D<Layout, unchecked_accessor> scalars,
    const MeshConnectivity& mesh,
    index_type moist_start,
    index_type moist_end,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
