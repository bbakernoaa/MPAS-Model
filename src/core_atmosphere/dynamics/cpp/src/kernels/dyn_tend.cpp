/// @file dyn_tend.cpp
/// @brief Implementation of the dynamics tendency orchestrator.
///
/// Composes the full dynamics tendency computation by invoking sub-kernels in
/// sequence: diagnostics, horizontal momentum tendency, optional Rayleigh
/// damping, thermodynamic tendency, and vertical momentum tendency.
///
/// @section sequence Computation Sequence
///   1. compute_solve_diagnostics — kinetic energy, vorticity, divergence,
///      potential vorticity on edges, density on edges.
///   2. compute_tend_u — pressure gradient, Coriolis, KE gradient.
///   3. rayleigh_damp_u — optional damping of velocity tendency on upper levels.
///   4. compute_tend_theta — horizontal flux divergence of rho*theta.
///   5. compute_tend_w — perturbation pressure gradient and buoyancy.
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/kernels/dyn_tend.hpp>
#include <mpas_dycore/kernels/tendencies.hpp>
#include <mpas_dycore/kernels/diagnostics.hpp>
#include <mpas_dycore/kernels/rayleigh_damping.hpp>

namespace mpas::dycore::kernels {

// ============================================================================
// Explicit template instantiation for default layout and serial policy
// ============================================================================

template void compute_dyn_tend<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> tend_u,
    Field2D<default_layout, unchecked_accessor> tend_theta,
    Field2D<default_layout, unchecked_accessor> tend_w,
    Field2D<default_layout, unchecked_accessor> ke,
    Field2D<default_layout, unchecked_accessor> vorticity,
    Field2D<default_layout, unchecked_accessor> divergence,
    Field2D<default_layout, unchecked_accessor> pv_edge,
    Field2D<default_layout, unchecked_accessor> rho_edge,
    ConstField2D<default_layout, unchecked_accessor> u,
    ConstField2D<default_layout, unchecked_accessor> pressure,
    ConstField2D<default_layout, unchecked_accessor> rho_zz,
    ConstField2D<default_layout, unchecked_accessor> theta_m,
    ConstField2D<default_layout, unchecked_accessor> rtheta_flux,
    ConstField2D<default_layout, unchecked_accessor> pp,
    ConstField2D<default_layout, unchecked_accessor> cqw,
    ConstField2D<default_layout, unchecked_accessor> cqu,
    ConstField2D<default_layout, unchecked_accessor> dpdz,
    const MeshConnectivity& mesh,
    ConnectivityView edgesOnEdge,
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
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels,
    index_type maxEdges2,
    const DynTendConfig& config);

// ============================================================================
// Implementation
// ============================================================================

/// @brief Implementation of compute_dyn_tend orchestrator.
///
/// Handles the zero-extent graceful return, then composes the sub-kernels
/// in the defined sequence. Rayleigh damping is applied conditionally based
/// on config.rayleigh_damp_u.
template <typename Layout, ExecutionPolicy Policy>
void compute_dyn_tend(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_u,
    Field2D<Layout, unchecked_accessor> tend_theta,
    Field2D<Layout, unchecked_accessor> tend_w,
    Field2D<Layout, unchecked_accessor> ke,
    Field2D<Layout, unchecked_accessor> vorticity,
    Field2D<Layout, unchecked_accessor> divergence,
    Field2D<Layout, unchecked_accessor> pv_edge,
    Field2D<Layout, unchecked_accessor> rho_edge,
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> pressure,
    ConstField2D<Layout, unchecked_accessor> rho_zz,
    ConstField2D<Layout, unchecked_accessor> theta_m,
    ConstField2D<Layout, unchecked_accessor> rtheta_flux,
    ConstField2D<Layout, unchecked_accessor> pp,
    ConstField2D<Layout, unchecked_accessor> cqw,
    ConstField2D<Layout, unchecked_accessor> cqu,
    ConstField2D<Layout, unchecked_accessor> dpdz,
    const MeshConnectivity& mesh,
    ConnectivityView edgesOnEdge,
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
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels,
    index_type maxEdges2,
    const DynTendConfig& config)
{
    // Zero-extent graceful return: do nothing if mesh has no work.
    if (nCells == 0 || nEdges == 0 || nVertLevels == 0) {
        return;
    }

    // Step 1: Compute diagnostic fields (ke, vorticity, divergence, pv_edge, rho_edge).
    compute_solve_diagnostics<Layout, Policy>(
        policy,
        ke, vorticity, divergence, pv_edge, rho_edge,
        u, rho_zz,
        mesh,
        areaCell, areaTriangle, dvEdge, dcEdge, fVertex,
        edgesOnVertex,
        nCells, nEdges, nVertices, nVertLevels);

    // Step 2: Compute horizontal momentum tendency.
    compute_tend_u<Layout, Policy>(
        policy,
        tend_u,
        u, pressure, pv_edge, ke, rho_edge, cqu,
        mesh,
        edgesOnEdge, weightsOnEdge, nEdgesOnEdge_arr, invDcEdge,
        nEdges, nVertLevels, maxEdges2);

    // Step 3: Apply Rayleigh damping to tend_u when enabled.
    if (config.rayleigh_damp_u) {
        rayleigh_damp_u<Layout, Policy>(
            policy,
            tend_u, u, rho_edge,
            config.rayleigh_damp_u,
            config.n_rayleigh_damp_levels,
            config.rayleigh_timescale_days,
            nEdges, nVertLevels);
    }

    // Step 4: Compute thermodynamic tendency (flux divergence of rho*theta).
    compute_tend_theta<Layout, Policy>(
        policy,
        tend_theta,
        rtheta_flux,
        mesh,
        areaCell, dvEdge,
        nCells, nVertLevels);

    // Step 5: Compute vertical momentum tendency (perturbation pressure gradient + buoyancy).
    compute_tend_w<Layout, Policy>(
        policy,
        tend_w,
        pp, cqw, rdzu, fzm, fzp, dpdz,
        nCells, nVertLevels);
}

} // namespace mpas::dycore::kernels
