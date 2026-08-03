#pragma once

/// @file diagnostics.hpp
/// @brief Solve diagnostics kernel for the MPAS dynamical core.
///
/// Computes kinetic energy (ke), relative vorticity, divergence, potential
/// vorticity on edges (pv_edge), and density interpolated to edges (rho_edge).
/// These diagnostic fields are computed at the beginning of each RK sub-step
/// from the prognostic velocity and density fields.
///
/// @section ke Kinetic Energy
/// KE(k, iCell) = sum over edges of cell: 0.25 * u(k, iEdge)^2 * areaEdge / areaCell
/// where areaEdge = dcEdge(iEdge) * dvEdge(iEdge).
///
/// @section vorticity Relative Vorticity
/// vorticity(k, iVertex) = (sum of circulation contributions) / areaTriangle(iVertex)
/// Circulation on each edge = u(k, iEdge) * dcEdge(iEdge) with sign based on
/// edge orientation relative to the vertex (verticesOnEdge ordering).
///
/// @section divergence Divergence
/// divergence(k, iCell) = (sum over edges of u * dvEdge with sign) / areaCell(iCell)
/// Sign convention: +1 if cellsOnEdge[iEdge,0] == iCell, -1 otherwise.
///
/// @section rho_edge Edge Density
/// rho_edge(k, iEdge) = 0.5 * (rho_zz(k, cell0) + rho_zz(k, cell1))
/// Boundary edges: rho_edge = rho_zz of the single adjacent cell.
///
/// @section pv_edge Potential Vorticity on Edges
/// pv_edge(k, iEdge) = 0.5 * ((fVertex0 + vorticity0) + (fVertex1 + vorticity1))
/// where fVertex is the Coriolis parameter and vorticity is relative vorticity
/// at each vertex endpoint of the edge. Note: the original definition had a
/// factor of 1/density which has been removed as it was not integral to any
/// conservation property of the system.
///
/// @reference Ringler, T. D., Thuburn, J., Klemp, J. B., and Skamarock, W. C.
/// (2010), "A unified approach to energy conservation and potential vorticity
/// dynamics for arbitrarily-structured C-grids", J. Comput. Phys., 229, 3065-3090.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Compute diagnostic fields for the dynamics solve.
///
/// Computes kinetic energy (ke), relative vorticity, divergence, potential
/// vorticity on edges (pv_edge), and density interpolated to edges (rho_edge).
///
/// The computation order is:
///   1. Kinetic energy (ke) - area-weighted edge velocity squared
///   2. Relative vorticity - circulation normalized by dual-cell area
///   3. Divergence - flux divergence normalized by cell area
///   4. Edge density (rho_edge) - cell density averaged to edges
///   5. Potential vorticity (pv_edge) - average absolute vorticity at edge vertices
///
/// @reference Ringler, T. D., Thuburn, J., Klemp, J. B., and Skamarock, W. C.
/// (2010), "A unified approach to energy conservation and potential vorticity
/// dynamics for arbitrarily-structured C-grids", J. Comput. Phys., 229, 3065-3090.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy for parallelization.
///
/// @param[in]  policy         Execution policy (accepted for API consistency).
/// @param[out] ke             Kinetic energy per cell (nVertLevels, nCells).
/// @param[out] vorticity      Relative vorticity at vertices (nVertLevels, nVertices).
/// @param[out] divergence     Divergence at cells (nVertLevels, nCells).
/// @param[out] pv_edge        Potential vorticity on edges (nVertLevels, nEdges).
/// @param[out] rho_edge       Density interpolated to edges (nVertLevels, nEdges).
/// @param[in]  u              Normal velocity on edges (nVertLevels, nEdges).
/// @param[in]  rho_zz         Dry air density at cells (nVertLevels, nCells).
/// @param[in]  mesh           Mesh connectivity views.
/// @param[in]  areaCell       Cell areas (nCells).
/// @param[in]  areaTriangle   Dual-cell (triangle) areas at vertices (nVertices).
/// @param[in]  dvEdge         Distance between vertices of each edge (nEdges).
/// @param[in]  dcEdge         Distance between cell centers across each edge (nEdges).
/// @param[in]  fVertex        Coriolis parameter at vertices (nVertices).
/// @param[in]  edgesOnVertex  Edges adjacent to each vertex (nVertices, vertexDegree).
/// @param[in]  nCells         Number of cells.
/// @param[in]  nEdges         Number of edges.
/// @param[in]  nVertices      Number of vertices.
/// @param[in]  nVertLevels    Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_solve_diagnostics(
    Policy policy,
    Field2D<Layout, unchecked_accessor> ke,           // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> vorticity,    // output: (nVertLevels, nVertices)
    Field2D<Layout, unchecked_accessor> divergence,   // output: (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> pv_edge,      // output: (nVertLevels, nEdges)
    Field2D<Layout, unchecked_accessor> rho_edge,     // output: (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> u,       // input: normal velocity (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> rho_zz,  // input: dry density (nVertLevels, nCells)
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaTriangle,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dcEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fVertex,
    ConnectivityView edgesOnVertex,   // (nVertices, vertexDegree)
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
