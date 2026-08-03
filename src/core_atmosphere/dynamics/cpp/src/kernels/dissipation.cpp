/// @file dissipation.cpp
/// @brief Implementation of del-2 and del-4 horizontal diffusion kernels.
///
/// Ports the Fortran `u_dissipation_3d` and `scalar_dissipation_3d_les`
/// subroutines from `mpas_atm_dissipation_models.F` to C++ using mdspan views.
///
/// @section algorithm Algorithm Overview
///
/// Momentum diffusion (u_dissipation_3d):
///   Pass 1: delsq_u = grad(div) - k x grad(vort) on edges
///   Pass 2 (del-2): tend_u += rho_edge * eddy_visc * meshScalingDel2 * delsq_u
///   Pass 3 (del-4): compute delsq_divergence and delsq_vorticity from delsq_u,
///     then tend_u -= rho_edge * meshScalingDel4 * eddy_visc4
///                    * (grad(delsq_div) * div_factor - k x grad(delsq_vort))
///
/// Scalar diffusion (scalar_dissipation_3d_les):
///   Pass 1: delsq_scalar = conservative Laplacian on cells
///   Pass 2 (del-2): tend += eddy_visc * prandtl_inv * meshScalingDel2 * delsq_scalar
///   Pass 3 (del-4): tend -= eddy_visc4 * prandtl_inv * meshScalingDel4 * del2(delsq_scalar)
///
/// @reference Skamarock & Klemp (2008), J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/kernels/dissipation.hpp>

#include <algorithm>
#include <cmath>

namespace mpas::dycore::kernels {

// ============================================================================
// Explicit template instantiations for default layout and serial policy
// ============================================================================

template void compute_u_del2<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> delsq_u,
    ConstField2D<default_layout, unchecked_accessor> divergence,
    ConstField2D<default_layout, unchecked_accessor> vorticity,
    const MeshConnectivity& mesh,
    Field1D invDcEdge,
    Field1D invDvEdge,
    index_type nEdges,
    index_type nVertLevels);

template void apply_u_diffusion<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> tend_u,
    ConstField2D<default_layout, unchecked_accessor> divergence,
    ConstField2D<default_layout, unchecked_accessor> vorticity,
    ConstField2D<default_layout, unchecked_accessor> rho_edge,
    ConstField2D<default_layout, unchecked_accessor> eddy_visc_horz,
    Field2D<default_layout, unchecked_accessor> delsq_u,
    Field2D<default_layout, unchecked_accessor> delsq_divergence,
    Field2D<default_layout, unchecked_accessor> delsq_vorticity,
    const MeshConnectivity& mesh,
    ConnectivityView edgesOnVertex,
    ConstField2D<default_layout, unchecked_accessor> edgesOnVertex_sign,
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign,
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

template void compute_scalar_del2<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> delsq_scalar,
    ConstField2D<default_layout, unchecked_accessor> scalar,
    ConstField2D<default_layout, unchecked_accessor> rho_edge,
    const MeshConnectivity& mesh,
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign,
    Field1D dvEdge,
    Field1D invDcEdge,
    Field1D invAreaCell,
    index_type nCells,
    index_type nVertLevels);

template void apply_scalar_diffusion<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> tend_scalar,
    ConstField2D<default_layout, unchecked_accessor> scalar,
    ConstField2D<default_layout, unchecked_accessor> rho_edge,
    ConstField2D<default_layout, unchecked_accessor> eddy_visc_horz,
    Field2D<default_layout, unchecked_accessor> delsq_scalar,
    const MeshConnectivity& mesh,
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign,
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

template void compute_smagorinsky_visc<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> eddy_visc_horz,
    ConstField2D<default_layout, unchecked_accessor> u,
    ConstField2D<default_layout, unchecked_accessor> v,
    ConstField2D<default_layout, unchecked_accessor> deformation_coef_c2,
    ConstField2D<default_layout, unchecked_accessor> deformation_coef_s2,
    ConstField2D<default_layout, unchecked_accessor> deformation_coef_cs,
    const MeshConnectivity& mesh,
    real_type c_s,
    real_type len_disp,
    real_type invDt,
    index_type nCells,
    index_type nVertLevels);

// ============================================================================
// compute_u_del2 Implementation
// ============================================================================

/// @brief Compute del-2 of normal velocity on edges from divergence and vorticity.
///
/// For each edge iEdge:
///   delsq_u(k, iEdge) = (div(k, cell1) - div(k, cell0)) * invDcEdge(iEdge)
///                      - (vort(k, v1) - vort(k, v0)) * r_dv
///
/// where r_dv = min(invDvEdge, 4*invDcEdge) to ensure numerical stability
/// when dual-mesh edges are very short relative to primal edges.
///
/// Boundary edges (cell1 == INVALID_INDEX) produce delsq_u = 0.
template <typename Layout, ExecutionPolicy Policy>
void compute_u_del2(
    Policy policy,
    Field2D<Layout, unchecked_accessor> delsq_u,
    ConstField2D<Layout, unchecked_accessor> divergence,
    ConstField2D<Layout, unchecked_accessor> vorticity,
    const MeshConnectivity& mesh,
    Field1D invDcEdge,
    Field1D invDvEdge,
    index_type nEdges,
    index_type nVertLevels)
{
    if (nEdges == 0 || nVertLevels == 0) return;

    policy.parallel_for(nEdges, nVertLevels,
        [&](index_type iEdge, index_type k) {
            const index_type cell0 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 1];
            const index_type vertex0 = mesh.verticesOnEdge[iEdge, 0];
            const index_type vertex1 = mesh.verticesOnEdge[iEdge, 1];

            if (cell1 == INVALID_INDEX) {
                delsq_u[k, iEdge] = 0.0;
                return;
            }

            const real_type r_dc = invDcEdge[iEdge];
            const real_type r_dv =
                std::min(invDvEdge[iEdge], 4.0 * invDcEdge[iEdge]);

            delsq_u[k, iEdge] =
                (divergence[k, cell1] - divergence[k, cell0]) * r_dc
              - (vorticity[k, vertex1] - vorticity[k, vertex0]) * r_dv;
        });
}

// ============================================================================
// apply_u_diffusion Implementation
// ============================================================================

/// @brief Apply del-2 and del-4 horizontal diffusion to momentum tendency.
///
/// Three-pass algorithm matching the Fortran u_dissipation_3d:
///   Pass 1: Compute delsq_u and apply del-2 to tend_u.
///   Pass 2: Compute delsq_divergence and delsq_vorticity from delsq_u.
///   Pass 3: Apply del-4 to tend_u (if eddy_visc4 > 0).
template <typename Layout, ExecutionPolicy Policy>
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
    index_type vertexDegree)
{
    if (nEdges == 0 || nVertLevels == 0) return;

    // ========================================================================
    // Pass 1: Compute delsq_u and apply del-2 diffusion to tend_u.
    // ========================================================================
    // delsq_u = grad(divergence) - k x grad(vorticity)
    // tend_u += rho_edge * kdiffu * delsq_u_les * meshScalingDel2
    //
    // Note: For LES models, del-2 uses u_diffusion_les which adds an extra
    // divergence gradient term. Here we use the standard formulation where
    // the LES factor (tau_12_factor) is handled by the caller providing
    // the appropriate eddy_visc_horz field.
    policy.parallel_for(nEdges, nVertLevels,
        [&](index_type iEdge, index_type k) {
            const index_type cell0 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 1];
            const index_type vertex0 = mesh.verticesOnEdge[iEdge, 0];
            const index_type vertex1 = mesh.verticesOnEdge[iEdge, 1];

            if (cell1 == INVALID_INDEX) {
                delsq_u[k, iEdge] = 0.0;
                return;
            }

            const real_type r_dc = invDcEdge[iEdge];
            const real_type r_dv =
                std::min(invDvEdge[iEdge], 4.0 * invDcEdge[iEdge]);

            // del-2 of u: grad(div) - k x grad(vort)
            const real_type u_diffusion =
                (divergence[k, cell1] - divergence[k, cell0]) * r_dc
              - (vorticity[k, vertex1] - vorticity[k, vertex0]) * r_dv;

            delsq_u[k, iEdge] = u_diffusion;

            // Spatially varying eddy viscosity (average of two adjacent cells)
            const real_type kdiffu =
                0.5 * (eddy_visc_horz[k, cell0] + eddy_visc_horz[k, cell1]);

            // Apply del-2 diffusion to tendency
            tend_u[k, iEdge] += rho_edge[k, iEdge] * kdiffu
                * u_diffusion * meshScalingDel2[iEdge];
        });

    // ========================================================================
    // Pass 2 & 3: Del-4 hyperdiffusion (if active).
    // ========================================================================
    if (eddy_visc4 <= 0.0) return;

    // Pass 2a: Compute delsq_vorticity from delsq_u on vertices.
    // delsq_vorticity(k, iVertex) = invAreaTriangle(iVertex)
    //     * sum_edges(dcEdge(iEdge) * edgesOnVertex_sign(i, iVertex) * delsq_u(k, iEdge))
    if (nVertices > 0) {
        policy.parallel_for(nVertices, nVertLevels,
            [&](index_type iVertex, index_type k) {
                real_type vort_sum = 0.0;
                for (index_type i = 0; i < vertexDegree; ++i) {
                    const index_type iEdge = edgesOnVertex[iVertex, i];
                    if (iEdge == INVALID_INDEX) break;

                    const real_type edge_sign =
                        invAreaTriangle[iVertex] * dcEdge[iEdge]
                        * edgesOnVertex_sign[i, iVertex];

                    vort_sum += edge_sign * delsq_u[k, iEdge];
                }
                delsq_vorticity[k, iVertex] = vort_sum;
            });
    }

    // Pass 2b: Compute delsq_divergence from delsq_u on cells.
    // delsq_divergence(k, iCell) = invAreaCell(iCell)
    //     * sum_edges(dvEdge(iEdge) * edgesOnCell_sign(j, iCell) * delsq_u(k, iEdge))
    if (nCells > 0) {
        policy.parallel_for(nCells, nVertLevels,
            [&](index_type iCell, index_type k) {
                real_type div_sum = 0.0;
                const index_type n_edges = mesh.nEdgesOnCell[iCell];
                const real_type r = invAreaCell[iCell];

                for (index_type j = 0; j < n_edges; ++j) {
                    const index_type iEdge = mesh.edgesOnCell[iCell, j];
                    if (iEdge == INVALID_INDEX) break;

                    const real_type edge_sign =
                        r * dvEdge[iEdge] * edgesOnCell_sign[j, iCell];

                    div_sum += edge_sign * delsq_u[k, iEdge];
                }
                delsq_divergence[k, iCell] = div_sum;
            });
    }

    // Pass 3: Apply del-4 to tend_u on edges.
    // tend_u -= rho_edge * meshScalingDel4 * eddy_visc4
    //           * ((delsq_div(cell1) - delsq_div(cell0)) * div_factor * invDcEdge
    //             -(delsq_vort(v1) - delsq_vort(v0)) * r_dv)
    policy.parallel_for(nEdges, nVertLevels,
        [&](index_type iEdge, index_type k) {
            const index_type cell0 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 1];
            const index_type vertex0 = mesh.verticesOnEdge[iEdge, 0];
            const index_type vertex1 = mesh.verticesOnEdge[iEdge, 1];

            if (cell1 == INVALID_INDEX) return;

            const real_type u_mix_scale =
                meshScalingDel4[iEdge] * eddy_visc4;
            const real_type r_dc =
                u_mix_scale * del4u_div_factor * invDcEdge[iEdge];
            const real_type r_dv =
                u_mix_scale * std::min(invDvEdge[iEdge],
                                       4.0 * invDcEdge[iEdge]);

            const real_type u_diffusion = rho_edge[k, iEdge]
                * ((delsq_divergence[k, cell1] - delsq_divergence[k, cell0]) * r_dc
                 - (delsq_vorticity[k, vertex1] - delsq_vorticity[k, vertex0]) * r_dv);

            tend_u[k, iEdge] -= u_diffusion;
        });
}

// ============================================================================
// compute_scalar_del2 Implementation
// ============================================================================

/// @brief Compute del-2 of a cell-centered scalar field in conservative form.
///
/// For each cell iCell:
///   delsq_scalar(k, iCell) = invAreaCell(iCell) * sum_{edges of iCell}(
///       edgesOnCell_sign(j, iCell) * dvEdge(iEdge) * invDcEdge(iEdge)
///       * (scalar(k, cell_neighbor) - scalar(k, iCell)) * rho_edge(k, iEdge))
///
/// The sign convention maps to the Fortran:
///   edge_sign = invAreaCell * edgesOnCell_sign(j, iCell) * dvEdge * invDcEdge
///   flux = (scalar(cell2) - scalar(cell1)) * rho_edge
///   delsq += edge_sign * flux
///
/// where cell1 = cellsOnEdge(0, iEdge), cell2 = cellsOnEdge(1, iEdge).
template <typename Layout, ExecutionPolicy Policy>
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
    index_type nVertLevels)
{
    if (nCells == 0 || nVertLevels == 0) return;

    policy.parallel_for(nCells, nVertLevels,
        [&](index_type iCell, index_type k) {
            real_type del2_sum = 0.0;
            const index_type n_edges = mesh.nEdgesOnCell[iCell];
            const real_type r_areaCell = invAreaCell[iCell];

            for (index_type j = 0; j < n_edges; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                const real_type edge_sign = r_areaCell
                    * edgesOnCell_sign[j, iCell]
                    * dvEdge[iEdge] * invDcEdge[iEdge];

                const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
                const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

                if (cell2 == INVALID_INDEX) continue;

                const real_type flux =
                    (scalar[k, cell2] - scalar[k, cell1])
                    * rho_edge[k, iEdge];

                del2_sum += edge_sign * flux;
            }
            delsq_scalar[k, iCell] = del2_sum;
        });
}

// ============================================================================
// apply_scalar_diffusion Implementation
// ============================================================================

/// @brief Apply del-2 and del-4 horizontal diffusion to a scalar tendency.
///
/// Two-pass algorithm matching the Fortran scalar_dissipation_3d_les:
///   Pass 1: Compute delsq_scalar and apply del-2 to tend_scalar.
///   Pass 2: If eddy_visc4 > 0, apply del-4 correction using del-2(delsq_scalar).
template <typename Layout, ExecutionPolicy Policy>
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
    index_type nVertLevels)
{
    if (nCells == 0 || nVertLevels == 0) return;

    // ========================================================================
    // Pass 1: Compute delsq_scalar and apply del-2 to tend_scalar.
    // ========================================================================
    // For each cell, iterate over its edges accumulating the Laplacian flux
    // and simultaneously applying the del-2 viscous tendency.
    //
    // delsq_scalar stores the unweighted del-2 (flux without viscosity/Prandtl)
    // for later use in the del-4 computation.
    policy.parallel_for(nCells, nVertLevels,
        [&](index_type iCell, index_type k) {
            real_type del2_sum = 0.0;
            real_type tend_sum = 0.0;
            const index_type n_edges = mesh.nEdgesOnCell[iCell];
            const real_type r_areaCell = invAreaCell[iCell];

            for (index_type j = 0; j < n_edges; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                const real_type edge_sign = r_areaCell
                    * edgesOnCell_sign[j, iCell]
                    * dvEdge[iEdge] * invDcEdge[iEdge];

                const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
                const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

                if (cell2 == INVALID_INDEX) continue;

                const real_type flux =
                    (scalar[k, cell2] - scalar[k, cell1])
                    * rho_edge[k, iEdge];

                // Accumulate raw del-2 (for del-4 second pass)
                del2_sum += edge_sign * flux;

                // Del-2 tendency with spatially varying viscosity and Prandtl
                const real_type pr_scale =
                    prandtl_inv * meshScalingDel2[iEdge];
                const real_type kdiff =
                    0.5 * (eddy_visc_horz[k, cell1]
                         + eddy_visc_horz[k, cell2]);
                tend_sum += edge_sign * flux * kdiff * pr_scale;
            }
            delsq_scalar[k, iCell] = del2_sum;
            tend_scalar[k, iCell] += tend_sum;
        });

    // ========================================================================
    // Pass 2: Del-4 hyperdiffusion (if active).
    // ========================================================================
    // tend -= eddy_visc4 * prandtl_inv * invAreaCell * sum_edges(
    //     meshScalingDel4 * dvEdge * edgesOnCell_sign * invDcEdge
    //     * (delsq_scalar(cell2) - delsq_scalar(cell1)))
    if (eddy_visc4 <= 0.0) return;

    policy.parallel_for(nCells, nVertLevels,
        [&](index_type iCell, index_type k) {
            const index_type n_edges = mesh.nEdgesOnCell[iCell];
            const real_type r_areaCell =
                eddy_visc4 * prandtl_inv * invAreaCell[iCell];

            real_type del4_sum = 0.0;
            for (index_type j = 0; j < n_edges; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                const real_type edge_sign = meshScalingDel4[iEdge]
                    * r_areaCell * dvEdge[iEdge]
                    * edgesOnCell_sign[j, iCell] * invDcEdge[iEdge];

                const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
                const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

                if (cell2 == INVALID_INDEX) continue;

                del4_sum += edge_sign
                    * (delsq_scalar[k, cell2] - delsq_scalar[k, cell1]);
            }
            tend_scalar[k, iCell] -= del4_sum;
        });
}

// ============================================================================
// compute_smagorinsky_visc Implementation
// ============================================================================

/// @brief Compute Smagorinsky nonlinear eddy viscosity on cells.
///
/// For each cell iCell, reconstructs the horizontal deformation tensor
/// from edge velocities using deformation coefficients, then:
///   d_11 = 2 * dudx
///   d_22 = 2 * dvdy
///   d_12 = dudy + dvdx
///   |S| = sqrt(0.25*(d_11-d_22)^2 + d_12^2)
///   kdiff = (c_s * len_disp)^2 * |S|
///   kdiff = min(kdiff, 0.01 * len_disp^2 * invDt)
///
/// The upper bound prevents excessive viscosity in regions of strong
/// deformation from violating the CFL condition.
template <typename Layout, ExecutionPolicy Policy>
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
    index_type nVertLevels)
{
    if (nCells == 0 || nVertLevels == 0) return;

    const real_type smag_coef = (c_s * len_disp) * (c_s * len_disp);
    const real_type upper_bound = 0.01 * len_disp * len_disp * invDt;

    policy.parallel_for(nCells, nVertLevels,
        [&](index_type iCell, index_type k) {
            const index_type n_edges = mesh.nEdgesOnCell[iCell];

            real_type dudx = 0.0;
            real_type dudy = 0.0;
            real_type dvdx = 0.0;
            real_type dvdy = 0.0;

            for (index_type j = 0; j < n_edges; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                const real_type u_val = u[k, iEdge];
                const real_type v_val = v[k, iEdge];
                const real_type c2 = deformation_coef_c2[j, iCell];
                const real_type s2 = deformation_coef_s2[j, iCell];
                const real_type cs = deformation_coef_cs[j, iCell];

                dudx += c2 * u_val - cs * v_val;
                dudy += cs * u_val - s2 * v_val;
                dvdx += cs * u_val + c2 * v_val;
                dvdy += s2 * u_val + cs * v_val;
            }

            // Deformation tensor components
            const real_type d_11 = 2.0 * dudx;
            const real_type d_22 = 2.0 * dvdy;
            const real_type d_12 = dudy + dvdx;

            // Smagorinsky viscosity with upper bound
            const real_type strain_rate =
                std::sqrt(0.25 * (d_11 - d_22) * (d_11 - d_22)
                        + d_12 * d_12);
            eddy_visc_horz[k, iCell] =
                std::min(smag_coef * strain_rate, upper_bound);
        });
}

} // namespace mpas::dycore::kernels
