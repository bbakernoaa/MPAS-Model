#pragma once

/// @file tendencies.hpp
/// @brief Dynamics tendency kernels for the MPAS dynamical core.
///
/// Declares the three primary tendency sub-kernels that compose the right-hand
/// side of the MPAS-Atmosphere prognostic equations:
///
/// - compute_tend_u: Horizontal momentum tendency on edges (pressure gradient,
///   Coriolis/curvature, kinetic energy gradient).
/// - compute_tend_theta: Thermodynamic tendency on cells (horizontal flux
///   divergence of coupled potential temperature).
/// - compute_tend_w: Vertical momentum tendency on cell interfaces (vertical
///   pressure gradient and buoyancy).
///
/// Each sub-kernel is independently testable and accepts all data exclusively
/// as function parameters (mdspan views or scalars) with no global state access.
///
/// @section governing_equations Governing Equations
///
/// Horizontal momentum (Skamarock & Klemp 2008, Eq. 2.14):
///   tend_u = -(pressure_gradient) - (ke_gradient) + (pv_edge * rho_u_flux)
///
/// Thermodynamic (flux divergence form):
///   tend_theta = -(1/areaCell) * sum_edges(rtheta_flux * dvEdge * sign)
///
/// Vertical momentum (split-explicit perturbation formulation):
///   tend_w = -cqw * (rdzu * d(pp)/dz - buoyancy_at_interface)
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

// ============================================================================
// Type alias facade for readability (non-template)
// ============================================================================

/// Default 2D field view used in tendency kernels (column-major, unchecked).
using TendField2D = Field2D<default_layout, unchecked_accessor>;
/// Default 2D const field view used in tendency kernels (column-major, unchecked).
using TendConstField2D = ConstField2D<default_layout, unchecked_accessor>;

// ============================================================================
// Horizontal Momentum Tendency
// ============================================================================

/// @brief Compute horizontal momentum tendency on edges.
///
/// Evaluates the right-hand side of the horizontal momentum equation on
/// the C-grid edge normal velocity using the TRiSK discretization
/// (Ringler et al., JCP 2010). For each edge, the tendency includes:
///
///   pressure_grad = -cqu(k, iEdge) * (pressure(k, cell1) - pressure(k, cell0)) * invDcEdge(iEdge)
///   q = sum_j weightsOnEdge(j, iEdge) * u(k, eoe_j) * 0.5*(pv_edge(k, iEdge) + pv_edge(k, eoe_j))
///   ke_grad = (ke(k, cell1) - ke(k, cell0)) * invDcEdge(iEdge)
///   tend_u(k, iEdge) = pressure_grad + rho_edge(k, iEdge) * (q - ke_grad)
///
/// Where:
/// - Pressure gradient: flat-terrain simplification (zz=1, zxu=0) of the full
///   terrain-following pressure gradient with moist coefficient cqu.
/// - KE gradient: normalized by invDcEdge (reciprocal cell-center distance).
/// - Coriolis: TRiSK reconstruction summing over neighboring edges using
///   weightsOnEdge and averaged potential vorticity (Ringler et al. 2010).
///
/// Boundary edges (cell1 == INVALID_INDEX) produce tend_u = 0.
///
/// @tparam Layout   mdspan layout policy (default: column-major).
/// @tparam Policy   Execution policy for parallelization.
///
/// @param[in]  policy         Execution policy instance.
/// @param[out] tend_u         Output tendency field (nVertLevels, nEdges).
/// @param[in]  u              Normal velocity on edges (nVertLevels, nEdges).
/// @param[in]  pressure       Cell-centered pressure (nVertLevels, nCells).
/// @param[in]  pv_edge        Potential vorticity interpolated to edges (nVertLevels, nEdges).
/// @param[in]  ke             Kinetic energy on cells (nVertLevels, nCells).
/// @param[in]  rho_edge       Density interpolated to edges (nVertLevels, nEdges).
/// @param[in]  cqu            Moist coefficient on edges (nVertLevels, nEdges).
/// @param[in]  mesh           Mesh connectivity views (cellsOnEdge, etc.).
/// @param[in]  edgesOnEdge    Edge-to-edge connectivity (nEdges, maxEdges2).
/// @param[in]  weightsOnEdge  TRiSK reconstruction weights (nEdges * maxEdges2, flattened).
/// @param[in]  nEdgesOnEdge   Number of neighbor edges per edge (nEdges).
/// @param[in]  invDcEdge      Reciprocal of dcEdge for each edge (nEdges).
/// @param[in]  nEdges         Number of edges to process.
/// @param[in]  nVertLevels    Number of vertical levels.
/// @param[in]  maxEdges2      Second extent of edgesOnEdge (max neighbors per edge).
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.
/// @reference Ringler, T., Thuburn, J., Klemp, J., and Skamarock, W. (2010),
/// "A unified approach to energy conservation and potential vorticity dynamics
/// for arbitrarily-structured C-grids", J. Comput. Phys., 229, 3065-3090.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_tend_u(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_u,        // output: (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> u,        // input: normal velocity
    ConstField2D<Layout, unchecked_accessor> pressure, // input: cell pressure (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> pv_edge,  // input: PV on edges
    ConstField2D<Layout, unchecked_accessor> ke,       // input: kinetic energy on cells
    ConstField2D<Layout, unchecked_accessor> rho_edge, // input: edge density
    ConstField2D<Layout, unchecked_accessor> cqu,      // input: moist coefficient on edges
    const MeshConnectivity& mesh,
    ConnectivityView edgesOnEdge,                      // input: (nEdges, maxEdges2)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> weightsOnEdge, // flattened (nEdges * maxEdges2)
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>> nEdgesOnEdge, // (nEdges)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,     // (nEdges)
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges2);

// ============================================================================
// Thermodynamic Tendency
// ============================================================================

/// @brief Compute thermodynamic tendency on cells.
///
/// Evaluates the horizontal flux divergence of coupled potential temperature
/// (rho_zz * theta_m) on cell centers:
///
///   tend_theta(k, iCell) = -(1/areaCell) * sum_{e in edgesOnCell}
///                            sign_e * rtheta_flux(k, e) * dvEdge(e)
///
/// Sign convention: +1 if cellsOnEdge(e, 0) == iCell, -1 otherwise.
///
/// @tparam Layout   mdspan layout policy (default: column-major).
/// @tparam Policy   Execution policy for parallelization.
///
/// @param[in]  policy        Execution policy instance.
/// @param[out] tend_theta    Output tendency field (nVertLevels, nCells).
/// @param[in]  rtheta_flux   Edge flux of rho*theta (nVertLevels, nEdges).
/// @param[in]  mesh          Mesh connectivity views.
/// @param[in]  areaCell      Cell areas (nCells).
/// @param[in]  dvEdge        Edge lengths (nEdges).
/// @param[in]  nCells        Number of cells to process.
/// @param[in]  nVertLevels   Number of vertical levels.
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_tend_theta(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_theta,
    ConstField2D<Layout, unchecked_accessor> rtheta_flux, // rho*theta flux on edges
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    index_type nCells,
    index_type nVertLevels);

// ============================================================================
// Vertical Momentum Tendency
// ============================================================================

/// @brief Compute vertical momentum tendency on cell interfaces.
///
/// Evaluates the perturbation pressure gradient and buoyancy terms at cell
/// interface levels (nVertLevels + 1 per column) using the split-explicit
/// formulation where the hydrostatic base state has been analytically removed:
///
///   tend_w(k, iCell) = -cqw(k, iCell) * (
///       rdzu(k) * (pp(k, iCell) - pp(k-1, iCell))            // perturbation pressure gradient
///     - (fzm(k) * dpdz(k, iCell) + fzp(k) * dpdz(k-1, iCell)) // buoyancy at interface
///   )
///
/// Where dpdz is the buoyancy term (pre-computed at cell centers), interpolated
/// to the interface using fzm/fzp weights.
///
/// Boundary levels (k=0 and k=nVertLevels) are set to zero (rigid lid).
///
/// @tparam Layout   mdspan layout policy (default: column-major).
/// @tparam Policy   Execution policy for parallelization.
///
/// @param[in]  policy        Execution policy instance.
/// @param[out] tend_w        Output tendency field (nVertLevels+1, nCells).
/// @param[in]  pp            Perturbation pressure at cell centers (nVertLevels, nCells).
/// @param[in]  cqw           Moist coefficient at interfaces (nVertLevels+1, nCells).
/// @param[in]  rdzu          Reciprocal vertical spacing at interfaces (nVertLevels+1).
/// @param[in]  fzm           Vertical interpolation weight (nVertLevels+1).
/// @param[in]  fzp           Vertical interpolation weight (nVertLevels+1).
/// @param[in]  dpdz          Buoyancy term at cell centers (nVertLevels, nCells).
/// @param[in]  nCells        Number of cells to process.
/// @param[in]  nVertLevels   Number of vertical levels.
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative split-explicit time integration methods for the compressible
/// nonhydrostatic equations", Mon. Wea. Rev., 135, 2897-2913.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_tend_w(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_w,        // output: (nVertLevels+1, nCells)
    ConstField2D<Layout, unchecked_accessor> pp,       // input: perturbation pressure (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> cqw,      // input: moist coefficient at interfaces (nVertLevels+1, nCells)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzu, // input: reciprocal dz at interfaces (nVertLevels+1)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,  // input: interpolation weight (nVertLevels+1)
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,  // input: interpolation weight (nVertLevels+1)
    ConstField2D<Layout, unchecked_accessor> dpdz,     // input: buoyancy term (nVertLevels, nCells)
    index_type nCells,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
