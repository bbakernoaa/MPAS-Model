#pragma once

/// @file divergence_damping.hpp
/// @brief 3D divergence damping kernel for the acoustic substep velocity perturbation.
///
/// Applies del-2 divergence damping to the acoustic velocity perturbation (ru_p)
/// within the acoustic substep to suppress spurious computational modes that
/// arise on the C-grid. The damping is a two-step process:
///   1. Compute cell-centered divergence from ru_p on edges.
///   2. Apply a Laplacian correction to ru_p using the divergence gradient.
///
/// @section equation Damping Equations
///
/// Step 1 — Cell divergence:
///   div(k, iCell) = invAreaCell(iCell) * sum_{edges of iCell} sign * dvEdge(iEdge) * ru_p(k, iEdge)
///
/// where sign = +1 if cellsOnEdge(iEdge, 0) == iCell, else -1.
///
/// Step 2 — Edge damping update:
///   ru_p(k, iEdge) -= dts * divdamp_coef * (div(k, cell1) - div(k, cell0)) * invDcEdge(iEdge)
///
/// where cell0 = cellsOnEdge(iEdge, 0) and cell1 = cellsOnEdge(iEdge, 1).
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Apply 3D divergence damping to the acoustic velocity perturbation.
///
/// Computes cell-centered divergence from ru_p and applies del-2 damping
/// back to ru_p. This is used within the acoustic substep to suppress
/// spurious computational modes on the C-grid.
///
/// Algorithm:
///   1. div(k, iCell) = invAreaCell * sum_edges(sign * dvEdge * ru_p)
///   2. ru_p(k, iEdge) -= dts * divdamp_coef * (div(k, cell1) - div(k, cell0)) * invDcEdge
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy.
///
/// @param[in]     policy          Execution policy.
/// @param[in,out] ru_p            Acoustic velocity perturbation (nVertLevels, nEdges).
/// @param[out]    divergence      Workspace for cell divergence (nVertLevels, nCells).
/// @param[in]     mesh            Mesh connectivity (cellsOnEdge, edgesOnCell, nEdgesOnCell).
/// @param[in]     dvEdge          Edge lengths (nEdges).
/// @param[in]     invAreaCell     Inverse cell areas (nCells).
/// @param[in]     invDcEdge       Inverse distance between cell centers across edges (nEdges).
/// @param[in]     dts             Acoustic substep time step (s).
/// @param[in]     divdamp_coef    Divergence damping coefficient.
/// @param[in]     nCells          Number of cells.
/// @param[in]     nEdges          Number of edges.
/// @param[in]     nVertLevels     Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
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
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
