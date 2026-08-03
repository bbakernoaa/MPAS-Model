#pragma once

/// @file dissipation.hpp
/// @brief Del-2 and del-4 horizontal diffusion kernels for momentum and scalars.
///
/// Implements the discrete Laplacian (del-2) and biharmonic (del-4) diffusion
/// operators on the MPAS unstructured Voronoi mesh. Del-4 is computed as two
/// successive applications of del-2 with intermediate storage.
///
/// @section u_del2 Del-2 for Momentum (Velocity)
///
/// For each edge iEdge with adjacent cells (cell0, cell1) and vertices (vertex0, vertex1):
///
///   delsq_u(k, iEdge) = (divergence(k, cell1) - divergence(k, cell0)) * invDcEdge(iEdge)
///                      - (vorticity(k, vertex1) - vorticity(k, vertex0)) * r_dv
///
/// where r_dv = min(invDvEdge(iEdge), 4 * invDcEdge(iEdge)).
///
/// The del-2 diffusion tendency is:
///   tend_u(k, iEdge) += rho_edge(k, iEdge) * eddy_visc * delsq_u(k, iEdge) * meshScalingDel2(iEdge)
///
/// @section u_del4 Del-4 for Momentum (Velocity)
///
/// Computed as del-2(del-2(u)):
///   1. Compute delsq_u (del-2 of u)
///   2. Compute delsq_divergence from delsq_u on cells
///   3. Compute delsq_vorticity from delsq_u on vertices
///   4. Apply del-2 of delsq_u using delsq_divergence and delsq_vorticity:
///      tend_u(k, iEdge) -= rho_edge(k, iEdge) * meshScalingDel4(iEdge) * eddy_visc4
///                          * ((delsq_div(cell1) - delsq_div(cell0)) * div_factor * invDcEdge
///                            -(delsq_vort(v1) - delsq_vort(v0)) * r_dv)
///
/// @section scalar_del2 Del-2 for Scalars (Cell-Centered)
///
/// For each cell iCell:
///   delsq_scalar(k, iCell) = invAreaCell(iCell) * sum_edges(
///       edgesOnCell_sign(j, iCell) * dvEdge(iEdge) * invDcEdge(iEdge)
///       * (scalar(k, cell2) - scalar(k, cell1)) * rho_edge(k, iEdge))
///
/// The del-2 scalar diffusion tendency is:
///   tend(k, iCell) += eddy_visc * prandtl_inv * meshScalingDel2(iEdge) * delsq_scalar(k, iCell)
///
/// @section smagorinsky Smagorinsky Viscosity
///
/// Computes deformation-based eddy viscosity on cells:
///   kdiff(k, iCell) = (c_s * len_disp)^2 * |S| clamped by an upper bound.
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

// ============================================================================
// 1D field view alias for convenience
// ============================================================================
using Field1D = std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>;
using MutField1D = std::mdspan<real_type, std::extents<index_type, std::dynamic_extent>>;

// ============================================================================
// Del-2 operator on velocity (edges)
// ============================================================================

/// @brief Compute the del-2 (Laplacian) of normal velocity on edges.
///
/// Uses pre-computed divergence and vorticity fields to form:
///   delsq_u(k, iEdge) = (div(k, cell1) - div(k, cell0)) * invDcEdge
///                      - (vort(k, v1) - vort(k, v0)) * r_dv
///
/// where r_dv = min(invDvEdge, 4*invDcEdge) for numerical stability.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy satisfying ExecutionPolicy concept.
///
/// @param[in]     policy       Execution policy instance.
/// @param[out]    delsq_u      Output del-2 of u (nVertLevels, nEdges).
/// @param[in]     divergence   Cell-centered divergence (nVertLevels, nCells).
/// @param[in]     vorticity    Vertex-centered relative vorticity (nVertLevels, nVertices).
/// @param[in]     mesh         Mesh connectivity (cellsOnEdge, verticesOnEdge).
/// @param[in]     invDcEdge    Inverse distance between cell centers (nEdges).
/// @param[in]     invDvEdge    Inverse distance between vertices (nEdges).
/// @param[in]     nEdges       Number of edges.
/// @param[in]     nVertLevels  Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_u_del2(
    Policy policy,
    Field2D<Layout, unchecked_accessor> delsq_u,
    ConstField2D<Layout, unchecked_accessor> divergence,
    ConstField2D<Layout, unchecked_accessor> vorticity,
    const MeshConnectivity& mesh,
    Field1D invDcEdge,
    Field1D invDvEdge,
    index_type nEdges,
    index_type nVertLevels);

// ============================================================================
// Apply del-2 and del-4 momentum diffusion
// ============================================================================

/// @brief Apply del-2 and del-4 horizontal diffusion to the momentum tendency.
///
/// 1. Computes del-2 of u (delsq_u) from divergence and vorticity.
/// 2. Applies del-2 diffusion: tend_u += rho_edge * eddy_visc * meshScalingDel2 * delsq_u
/// 3. If eddy_visc4 > 0: computes del-2 of delsq_u (using delsq_divergence and
///    delsq_vorticity workspace), then applies del-4 with negative sign.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy satisfying ExecutionPolicy concept.
///
/// @param[in]     policy              Execution policy instance.
/// @param[in,out] tend_u              Momentum tendency accumulator (nVertLevels, nEdges).
/// @param[in]     divergence          Cell-centered divergence (nVertLevels, nCells).
/// @param[in]     vorticity           Vertex-centered vorticity (nVertLevels, nVertices).
/// @param[in]     rho_edge            Density at edges (nVertLevels, nEdges).
/// @param[in]     eddy_visc_horz      Horizontal eddy viscosity on cells (nVertLevels, nCells).
/// @param[out]    delsq_u             Workspace: del-2 of u (nVertLevels, nEdges).
/// @param[out]    delsq_divergence    Workspace: divergence of delsq_u (nVertLevels, nCells).
/// @param[out]    delsq_vorticity     Workspace: vorticity of delsq_u (nVertLevels, nVertices).
/// @param[in]     mesh                Mesh connectivity.
/// @param[in]     edgesOnVertex       Edges adjacent to each vertex (nVertices, vertexDegree).
/// @param[in]     edgesOnVertex_sign  Sign convention for edges on vertices (nVertices, vertexDegree).
/// @param[in]     edgesOnCell_sign    Sign convention for edges on cells (nCells, maxEdges).
/// @param[in]     invDcEdge           Inverse distance between cell centers (nEdges).
/// @param[in]     invDvEdge           Inverse distance between vertices (nEdges).
/// @param[in]     dvEdge              Edge lengths in the dual mesh (nEdges).
/// @param[in]     dcEdge              Edge lengths in the primal mesh (nEdges).
/// @param[in]     invAreaCell         Inverse cell areas (nCells).
/// @param[in]     invAreaTriangle     Inverse dual-cell (triangle) areas (nVertices).
/// @param[in]     meshScalingDel2     Mesh scaling for del-2 per edge (nEdges).
/// @param[in]     meshScalingDel4     Mesh scaling for del-4 per edge (nEdges).
/// @param[in]     eddy_visc4          Del-4 hyperviscosity coefficient (scalar).
/// @param[in]     del4u_div_factor    Factor applied to divergent component in del-4.
/// @param[in]     nCells              Number of cells.
/// @param[in]     nEdges              Number of edges.
/// @param[in]     nVertices           Number of vertices.
/// @param[in]     nVertLevels         Number of vertical levels.
/// @param[in]     vertexDegree        Number of edges per vertex (typically 3).
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void apply_u_diffusion(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_u,
    ConstField2D<Layout, unchecked_accessor> divergence,
    ConstField2D<Layout, unchecked_accessor> vorticity,
    ConstField2D<Layout, unchecked_accessor> rho_edge,
    ConstField2D<Layout, unchecked_accessor> eddy_visc_horz,
    Field2D<Layout, unchecked_accessor> delsq_u,
    Field2D<Layout, unchecked_accessor> delsq_divergence,
    Field2D<Layout, unchecked_accessor> delsq_vorticity,
    const MeshConnectivity& mesh,
    ConnectivityView edgesOnVertex,
    ConstField2D<Layout, unchecked_accessor> edgesOnVertex_sign,
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign,
    Field1D invDcEdge,
    Field1D invDvEdge,
    Field1D dvEdge,
    Field1D dcEdge,
    Field1D invAreaCell,
    Field1D invAreaTriangle,
    Field1D meshScalingDel2,
    Field1D meshScalingDel4,
    real_type eddy_visc4,
    real_type del4u_div_factor,
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels,
    index_type vertexDegree);

// ============================================================================
// Del-2 operator on scalars (cells)
// ============================================================================

/// @brief Compute del-2 (Laplacian) of a cell-centered scalar field.
///
/// For each cell iCell:
///   delsq_scalar(k, iCell) = invAreaCell(iCell) * sum_{edges of iCell}(
///       sign * dvEdge * invDcEdge * (scalar(k, cell_neighbor) - scalar(k, iCell)) * rho_edge)
///
/// The result is a density-weighted Laplacian in conservative form.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy satisfying ExecutionPolicy concept.
///
/// @param[in]     policy          Execution policy instance.
/// @param[out]    delsq_scalar    Output del-2 of scalar (nVertLevels, nCells).
/// @param[in]     scalar          Input scalar field on cells (nVertLevels, nCells).
/// @param[in]     rho_edge        Density at edges (nVertLevels, nEdges).
/// @param[in]     mesh            Mesh connectivity (edgesOnCell, cellsOnEdge, nEdgesOnCell).
/// @param[in]     edgesOnCell_sign Sign convention for edges on cells (nCells, maxEdges).
/// @param[in]     dvEdge          Dual-mesh edge lengths (nEdges).
/// @param[in]     invDcEdge       Inverse distance between cell centers (nEdges).
/// @param[in]     invAreaCell     Inverse cell areas (nCells).
/// @param[in]     nCells          Number of cells.
/// @param[in]     nVertLevels     Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_scalar_del2(
    Policy policy,
    Field2D<Layout, unchecked_accessor> delsq_scalar,
    ConstField2D<Layout, unchecked_accessor> scalar,
    ConstField2D<Layout, unchecked_accessor> rho_edge,
    const MeshConnectivity& mesh,
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign,
    Field1D dvEdge,
    Field1D invDcEdge,
    Field1D invAreaCell,
    index_type nCells,
    index_type nVertLevels);

// ============================================================================
// Apply del-2 and del-4 scalar diffusion
// ============================================================================

/// @brief Apply del-2 and del-4 horizontal diffusion to a scalar tendency.
///
/// 1. Computes del-2 of scalar (delsq_scalar) in conservative form.
/// 2. Applies del-2 diffusion: tend += eddy_visc * prandtl_inv * meshScalingDel2 * delsq_scalar
/// 3. If eddy_visc4 > 0: computes del-2 of delsq_scalar, then subtracts del-4 correction.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy satisfying ExecutionPolicy concept.
///
/// @param[in]     policy           Execution policy instance.
/// @param[in,out] tend_scalar      Scalar tendency accumulator (nVertLevels, nCells).
/// @param[in]     scalar           Input scalar field on cells (nVertLevels, nCells).
/// @param[in]     rho_edge         Density at edges (nVertLevels, nEdges).
/// @param[in]     eddy_visc_horz   Horizontal eddy viscosity on cells (nVertLevels, nCells).
/// @param[out]    delsq_scalar     Workspace: del-2 of scalar (nVertLevels, nCells).
/// @param[in]     mesh             Mesh connectivity.
/// @param[in]     edgesOnCell_sign Sign convention for edges on cells (nCells, maxEdges).
/// @param[in]     dvEdge           Dual-mesh edge lengths (nEdges).
/// @param[in]     invDcEdge        Inverse distance between cell centers (nEdges).
/// @param[in]     invAreaCell      Inverse cell areas (nCells).
/// @param[in]     meshScalingDel2  Mesh scaling for del-2 per edge (nEdges).
/// @param[in]     meshScalingDel4  Mesh scaling for del-4 per edge (nEdges).
/// @param[in]     prandtl_inv      Inverse turbulent Prandtl number (scalar).
/// @param[in]     eddy_visc4       Del-4 hyperviscosity coefficient (scalar).
/// @param[in]     nCells           Number of cells.
/// @param[in]     nEdges           Number of edges.
/// @param[in]     nVertLevels      Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void apply_scalar_diffusion(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_scalar,
    ConstField2D<Layout, unchecked_accessor> scalar,
    ConstField2D<Layout, unchecked_accessor> rho_edge,
    ConstField2D<Layout, unchecked_accessor> eddy_visc_horz,
    Field2D<Layout, unchecked_accessor> delsq_scalar,
    const MeshConnectivity& mesh,
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign,
    Field1D dvEdge,
    Field1D invDcEdge,
    Field1D invAreaCell,
    Field1D meshScalingDel2,
    Field1D meshScalingDel4,
    real_type prandtl_inv,
    real_type eddy_visc4,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels);

// ============================================================================
// Smagorinsky deformation-based viscosity
// ============================================================================

/// @brief Compute Smagorinsky nonlinear eddy viscosity on cells.
///
/// For each cell iCell, computes the deformation tensor from edge velocities
/// and deformation coefficients, then:
///   kdiff(k, iCell) = (c_s * len_disp)^2 * |S|
///   kdiff(k, iCell) = min(kdiff, 0.01 * len_disp^2 * invDt)
///
/// where |S| = sqrt(0.25*(d_11 - d_22)^2 + d_12^2).
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy satisfying ExecutionPolicy concept.
///
/// @param[in]     policy              Execution policy instance.
/// @param[out]    eddy_visc_horz      Output eddy viscosity on cells (nVertLevels, nCells).
/// @param[in]     u                   Edge-normal velocity (nVertLevels, nEdges).
/// @param[in]     v                   Edge-tangential velocity (nVertLevels, nEdges).
/// @param[in]     deformation_coef_c2 Deformation coefficient (maxEdges, nCells).
/// @param[in]     deformation_coef_s2 Deformation coefficient (maxEdges, nCells).
/// @param[in]     deformation_coef_cs Deformation coefficient (maxEdges, nCells).
/// @param[in]     mesh                Mesh connectivity (edgesOnCell, nEdgesOnCell).
/// @param[in]     c_s                 Smagorinsky constant.
/// @param[in]     len_disp            Length dispersion scale (config_len_disp).
/// @param[in]     invDt               Inverse timestep for upper bound clamping.
/// @param[in]     nCells              Number of cells.
/// @param[in]     nVertLevels         Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_smagorinsky_visc(
    Policy policy,
    Field2D<Layout, unchecked_accessor> eddy_visc_horz,
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> v,
    ConstField2D<Layout, unchecked_accessor> deformation_coef_c2,
    ConstField2D<Layout, unchecked_accessor> deformation_coef_s2,
    ConstField2D<Layout, unchecked_accessor> deformation_coef_cs,
    const MeshConnectivity& mesh,
    real_type c_s,
    real_type len_disp,
    real_type invDt,
    index_type nCells,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
