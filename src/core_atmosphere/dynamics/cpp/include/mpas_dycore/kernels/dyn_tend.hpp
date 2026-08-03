#pragma once

/// @file dyn_tend.hpp
/// @brief Dynamics tendency orchestrator for the MPAS dynamical core.
///
/// Composes the full dynamics tendency computation by calling sub-kernels
/// in sequence: diagnostics, horizontal momentum tendency, optional Rayleigh
/// damping, thermodynamic tendency, and vertical momentum tendency.
///
/// This orchestrator is called once per Runge-Kutta sub-step to produce the
/// complete set of tendencies needed for acoustic sub-stepping.
///
/// @section sequence Computation Sequence
///   1. compute_solve_diagnostics (ke, vorticity, divergence, pv_edge, rho_edge)
///   2. compute_tend_u (pressure gradient + Coriolis + KE gradient)
///   3. rayleigh_damp_u (optional, applied when config.rayleigh_damp_u is true)
///   4. compute_tend_theta (flux divergence of rho*theta)
///   5. compute_tend_w (vertical pressure gradient + buoyancy)
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Configuration for the dynamics tendency orchestrator.
struct DynTendConfig {
    bool rayleigh_damp_u;                    ///< Enable Rayleigh damping on tend_u.
    index_type n_rayleigh_damp_levels;       ///< Number of uppermost levels to damp.
    real_type rayleigh_timescale_days;       ///< Damping timescale in days.
};

/// @brief Orchestrate the complete dynamics tendency computation.
///
/// Computes diagnostics (ke, vorticity, divergence, pv_edge, rho_edge) then
/// invokes tendency sub-kernels (tend_u, tend_theta, tend_w) and applies
/// Rayleigh damping when enabled.
///
/// Sequence:
///   1. compute_solve_diagnostics (ke, vorticity, divergence, pv_edge, rho_edge)
///   2. compute_tend_u (pressure gradient + Coriolis + KE gradient)
///   3. rayleigh_damp_u (optional)
///   4. compute_tend_theta (flux divergence of rho*theta)
///   5. compute_tend_w (perturbation pressure gradient + buoyancy)
///
/// If any of nCells, nEdges, or nVertLevels is zero, returns immediately
/// without modifying any output fields (zero-extent graceful return).
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy for parallelization.
///
/// @param[in]     policy         Execution policy instance.
/// @param[out]    tend_u         Horizontal velocity tendency (nVertLevels, nEdges).
/// @param[out]    tend_theta     Thermodynamic tendency (nVertLevels, nCells).
/// @param[out]    tend_w         Vertical velocity tendency (nVertLevels+1, nCells).
/// @param[out]    ke             Kinetic energy workspace (nVertLevels, nCells).
/// @param[out]    vorticity      Relative vorticity workspace (nVertLevels, nVertices).
/// @param[out]    divergence     Divergence workspace (nVertLevels, nCells).
/// @param[out]    pv_edge        Potential vorticity workspace (nVertLevels, nEdges).
/// @param[out]    rho_edge       Edge density workspace (nVertLevels, nEdges).
/// @param[in]     u              Normal velocity on edges (nVertLevels, nEdges).
/// @param[in]     pressure       Cell-centered pressure (nVertLevels, nCells).
/// @param[in]     rho_zz         Dry density at cells (nVertLevels, nCells).
/// @param[in]     theta_m        Moist potential temperature (nVertLevels, nCells).
/// @param[in]     rtheta_flux    Edge flux of rho*theta (nVertLevels, nEdges).
/// @param[in]     pp             Perturbation pressure at cell centers (nVertLevels, nCells).
/// @param[in]     cqw            Moist coefficient at interfaces (nVertLevels+1, nCells).
/// @param[in]     cqu            Moist coefficient on edges (nVertLevels, nEdges).
/// @param[in]     dpdz           Buoyancy term at cell centers (nVertLevels, nCells).
/// @param[in]     mesh           Mesh connectivity views.
/// @param[in]     edgesOnEdge    Edge-to-edge connectivity (nEdges, maxEdges2).
/// @param[in]     areaCell       Cell areas (nCells).
/// @param[in]     areaTriangle   Dual-cell (triangle) areas at vertices (nVertices).
/// @param[in]     dvEdge         Edge lengths (nEdges).
/// @param[in]     dcEdge         Distance between cell centers across each edge (nEdges).
/// @param[in]     weightsOnEdge  TRiSK reconstruction weights (nEdges * maxEdges2, flattened).
/// @param[in]     nEdgesOnEdge_arr Number of neighbor edges per edge (nEdges).
/// @param[in]     invDcEdge      Reciprocal of dcEdge for each edge (nEdges).
/// @param[in]     fVertex        Coriolis parameter at vertices (nVertices).
/// @param[in]     rdzu           Reciprocal vertical spacing at interfaces (nVertLevels+1).
/// @param[in]     fzm            Vertical interpolation weight (nVertLevels+1).
/// @param[in]     fzp            Vertical interpolation weight (nVertLevels+1).
/// @param[in]     edgesOnVertex  Edges adjacent to each vertex (nVertices, vertexDegree).
/// @param[in]     nCells         Number of cells.
/// @param[in]     nEdges         Number of edges.
/// @param[in]     nVertices      Number of vertices.
/// @param[in]     nVertLevels    Number of vertical levels.
/// @param[in]     maxEdges2      Second extent of edgesOnEdge.
/// @param[in]     config         Configuration (Rayleigh damping parameters).
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_dyn_tend(
    Policy policy,
    // Tendency outputs
    Field2D<Layout, unchecked_accessor> tend_u,       // (nVertLevels, nEdges)
    Field2D<Layout, unchecked_accessor> tend_theta,   // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> tend_w,       // (nVertLevels+1, nCells)
    // Diagnostic workspaces (pre-allocated)
    Field2D<Layout, unchecked_accessor> ke,           // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> vorticity,    // (nVertLevels, nVertices)
    Field2D<Layout, unchecked_accessor> divergence,   // (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> pv_edge,      // (nVertLevels, nEdges)
    Field2D<Layout, unchecked_accessor> rho_edge,     // (nVertLevels, nEdges)
    // Prognostic inputs
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> pressure,
    ConstField2D<Layout, unchecked_accessor> rho_zz,
    ConstField2D<Layout, unchecked_accessor> theta_m,
    ConstField2D<Layout, unchecked_accessor> rtheta_flux,  // for theta tendency
    // Perturbation fields for tend_w
    ConstField2D<Layout, unchecked_accessor> pp,       // perturbation pressure (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> cqw,      // moist coefficient at interfaces (nVertLevels+1, nCells)
    ConstField2D<Layout, unchecked_accessor> cqu,      // moist coefficient on edges (nVertLevels, nEdges)
    ConstField2D<Layout, unchecked_accessor> dpdz,     // buoyancy term (nVertLevels, nCells)
    // Mesh
    const MeshConnectivity& mesh,
    ConnectivityView edgesOnEdge,                      // (nEdges, maxEdges2)
    // Geometry 1D arrays
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> areaTriangle,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dcEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> weightsOnEdge,
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>> nEdgesOnEdge_arr,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fVertex,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzu,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    ConnectivityView edgesOnVertex,
    // Dimensions
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels,
    index_type maxEdges2,
    // Config
    const DynTendConfig& config);

} // namespace mpas::dycore::kernels
