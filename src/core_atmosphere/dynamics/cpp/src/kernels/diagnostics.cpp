/// @file diagnostics.cpp
/// @brief Implementation of the solve diagnostics kernel.
///
/// Computes kinetic energy (ke), relative vorticity, divergence, potential
/// vorticity on edges (pv_edge), and density interpolated to edges (rho_edge).
/// These diagnostic fields are used in the momentum and tracer transport
/// equations during the dynamics solve.
///
/// @section governing_equations Governing Equations
///
/// Kinetic energy at each cell:
///   KE(k, iCell) = sum_{e in edgesOnCell(iCell)} 0.25 * u(k,e)^2 * dcEdge(e) * dvEdge(e) / areaCell(iCell)
///
/// Relative vorticity at each vertex (dual cell):
///   vorticity(k, iVertex) = sum_{e in edgesOnVertex(iVertex)} sign_e * u(k,e) * dcEdge(e) / areaTriangle(iVertex)
/// where sign_e is +1 if verticesOnEdge(e,1)==iVertex, -1 otherwise.
///
/// Divergence at each cell:
///   divergence(k, iCell) = sum_{e in edgesOnCell(iCell)} sign_e * u(k,e) * dvEdge(e) / areaCell(iCell)
/// where sign_e is +1 if cellsOnEdge(e,0)==iCell, -1 otherwise.
///
/// Edge density:
///   rho_edge(k, iEdge) = 0.5 * (rho_zz(k, cell0) + rho_zz(k, cell1))
///   Boundary: rho_edge(k, iEdge) = rho_zz(k, cell0)
///
/// Potential vorticity on edges:
///   pv_edge(k, iEdge) = 0.5 * (fVertex(v0) + vorticity(k,v0) + fVertex(v1) + vorticity(k,v1))
///
/// Note: the original definition of pv_edge had a factor of 1/density. That
/// factor has been removed given that it was not integral to any conservation
/// property of the system (see Fortran reference implementation).
///
/// @reference Ringler, T. D., Thuburn, J., Klemp, J. B., and Skamarock, W. C.
/// (2010), "A unified approach to energy conservation and potential vorticity
/// dynamics for arbitrarily-structured C-grids", J. Comput. Phys., 229, 3065-3090.

#include <mpas_dycore/kernels/diagnostics.hpp>

namespace mpas::dycore::kernels {

// Explicit instantiation for default layout and serial policy.
template void compute_solve_diagnostics<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> ke,
    Field2D<default_layout, unchecked_accessor> vorticity,
    Field2D<default_layout, unchecked_accessor> divergence,
    Field2D<default_layout, unchecked_accessor> pv_edge,
    Field2D<default_layout, unchecked_accessor> rho_edge,
    ConstField2D<default_layout, unchecked_accessor> u,
    ConstField2D<default_layout, unchecked_accessor> rho_zz,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaTriangle,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dcEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fVertex,
    ConnectivityView edgesOnVertex,
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels);

/// @brief Implementation of compute_solve_diagnostics.
///
/// Computes all diagnostic fields in order: ke, vorticity, divergence,
/// rho_edge, pv_edge. Uses serial loops. The execution policy parameter
/// is accepted for API consistency but not used for dispatch in this kernel
/// due to data dependencies between computation stages.
template <typename Layout, ExecutionPolicy Policy>
void compute_solve_diagnostics(
    Policy /*policy*/,
    Field2D<Layout, unchecked_accessor> ke,
    Field2D<Layout, unchecked_accessor> vorticity,
    Field2D<Layout, unchecked_accessor> divergence,
    Field2D<Layout, unchecked_accessor> pv_edge,
    Field2D<Layout, unchecked_accessor> rho_edge,
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> rho_zz,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaTriangle,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dcEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fVertex,
    ConnectivityView edgesOnVertex,
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels)
{
    // ========================================================================
    // Step 1: Compute kinetic energy at cell centers.
    //
    // KE(k, iCell) = sum over edges of cell:
    //   0.25 * u(k, iEdge)^2 * dcEdge(iEdge) * dvEdge(iEdge) / areaCell(iCell)
    // ========================================================================
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const index_type nEdgesOnThisCell = mesh.nEdgesOnCell[iCell];
        const real_type inv_area = 1.0 / areaCell[iCell];

        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type ke_sum = 0.0;

            for (index_type j = 0; j < nEdgesOnThisCell; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                const real_type u_val = u[k, iEdge];
                ke_sum += 0.25 * u_val * u_val * dcEdge[iEdge] * dvEdge[iEdge];
            }

            ke[k, iCell] = ke_sum * inv_area;
        }
    }

    // ========================================================================
    // Step 2: Compute relative vorticity at vertices (dual cells).
    //
    // vorticity(k, iVertex) = sum over edges of vertex:
    //   sign_e * u(k, iEdge) * dcEdge(iEdge) / areaTriangle(iVertex)
    //
    // Sign convention (Fortran mpas_atm_core.F lines 1222-1226):
    //   sign = +1 when iVertex == verticesOnEdge(iEdge, 1)  (second vertex)
    //          -1 otherwise
    // ========================================================================
    const index_type vertexDegree = edgesOnVertex.extent(1);

    for (index_type iVertex = 0; iVertex < nVertices; ++iVertex) {
        const real_type inv_area_tri = 1.0 / areaTriangle[iVertex];

        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type circ = 0.0;

            for (index_type j = 0; j < vertexDegree; ++j) {
                const index_type iEdge = edgesOnVertex[iVertex, j];
                if (iEdge == INVALID_INDEX) break;

                // Sign is +1 when iVertex is the second vertex of the edge
                // (Fortran: verticesOnEdge(2,iEdge), zero-based index 1)
                const real_type sign =
                    (mesh.verticesOnEdge[iEdge, 1] == iVertex) ? 1.0 : -1.0;
                circ += sign * u[k, iEdge] * dcEdge[iEdge];
            }

            vorticity[k, iVertex] = circ * inv_area_tri;
        }
    }

    // ========================================================================
    // Step 3: Compute divergence at cell centers.
    //
    // divergence(k, iCell) = sum over edges of cell:
    //   sign_e * u(k, iEdge) * dvEdge(iEdge) / areaCell(iCell)
    //
    // Sign convention: if cellsOnEdge(iEdge, 0) == iCell then +1, else -1.
    // ========================================================================
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const index_type nEdgesOnThisCell = mesh.nEdgesOnCell[iCell];
        const real_type inv_area = 1.0 / areaCell[iCell];

        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type div_sum = 0.0;

            for (index_type j = 0; j < nEdgesOnThisCell; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                // Sign based on cellsOnEdge ordering
                const real_type sign =
                    (mesh.cellsOnEdge[iEdge, 0] == iCell) ? 1.0 : -1.0;
                div_sum += sign * u[k, iEdge] * dvEdge[iEdge];
            }

            divergence[k, iCell] = div_sum * inv_area;
        }
    }

    // ========================================================================
    // Step 4: Compute density interpolated to edges.
    //
    // rho_edge(k, iEdge) = 0.5 * (rho_zz(k, cell0) + rho_zz(k, cell1))
    // Boundary edges (cell1 == INVALID_INDEX): rho_edge = rho_zz(k, cell0)
    // ========================================================================
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        const index_type cell0 = mesh.cellsOnEdge[iEdge, 0];
        const index_type cell1 = mesh.cellsOnEdge[iEdge, 1];

        if (cell1 == INVALID_INDEX) {
            // Boundary edge: use single adjacent cell
            for (index_type k = 0; k < nVertLevels; ++k) {
                rho_edge[k, iEdge] = rho_zz[k, cell0];
            }
        } else {
            // Interior edge: average the two adjacent cells
            for (index_type k = 0; k < nVertLevels; ++k) {
                rho_edge[k, iEdge] = 0.5 * (rho_zz[k, cell0] + rho_zz[k, cell1]);
            }
        }
    }

    // ========================================================================
    // Step 5: Compute potential vorticity on edges.
    //
    // pv_edge(k, iEdge) = 0.5 * (pv_vertex(k,v0) + pv_vertex(k,v1))
    // where pv_vertex(k,v) = fVertex(v) + vorticity(k,v)
    //
    // Note: the original definition of pv_edge had a factor of 1/density.
    // That factor has been removed given that it was not integral to any
    // conservation property of the system.
    // ========================================================================
    for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
        const index_type v0 = mesh.verticesOnEdge[iEdge, 0];
        const index_type v1 = mesh.verticesOnEdge[iEdge, 1];

        for (index_type k = 0; k < nVertLevels; ++k) {
            pv_edge[k, iEdge] = 0.5 * ((fVertex[v0] + vorticity[k, v0])
                                      + (fVertex[v1] + vorticity[k, v1]));
        }
    }
}

} // namespace mpas::dycore::kernels
