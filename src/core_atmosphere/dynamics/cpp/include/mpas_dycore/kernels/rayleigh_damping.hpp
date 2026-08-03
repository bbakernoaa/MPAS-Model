#pragma once

/// @file rayleigh_damping.hpp
/// @brief Rayleigh damping kernel for horizontal velocity tendency.
///
/// Applies Rayleigh damping to the horizontal velocity tendency on the
/// uppermost model levels to suppress spurious wave reflections from the
/// upper boundary. The damping uses a linearly-ramped coefficient that is
/// strongest at the model top (k=0) and decreases toward the bottom of the
/// sponge layer, weighted by the air density at each edge.
///
/// @section equation Damping Equation
///
/// The per-level damping coefficient ramps linearly within the sponge layer:
///   coef(k) = (n_damp_levels - k) / (n_damp_levels * timescale_days * seconds_per_day)
///
/// The tendency modification for each edge on damped levels:
///   tend_u[k, iEdge] -= rho_edge[k, iEdge] * u[k, iEdge] * coef(k)
/// for k in [0, n_damp_levels).
///
/// @reference Klemp, J. B., Dudhia, J., and Hassiotis, A. (2008),
/// "An upper gravity-wave absorbing layer for NWP applications",
/// Mon. Wea. Rev., 136, 3987-4004.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Apply Rayleigh damping to the horizontal velocity tendency.
///
/// Applies density-weighted damping to tend_u on the uppermost n_damp_levels
/// vertical levels. The damping coefficient ramps linearly from maximum at
/// the model top (k=0) to minimum at the bottom of the sponge layer:
///   coef(k) = (n_damp_levels - k) / (n_damp_levels * timescale_days * 86400.0)
///
/// The tendency modification is:
///   tend_u[k, iEdge] -= rho_edge[k, iEdge] * u[k, iEdge] * coef(k)
/// for k in [0, n_damp_levels).
///
/// When enabled is false, this function is a no-op.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy.
///
/// @param[in]     policy              Execution policy.
/// @param[in,out] tend_u              Horizontal velocity tendency (nVertLevels, nEdges).
/// @param[in]     u                   Current horizontal velocity (nVertLevels, nEdges).
/// @param[in]     rho_edge            Air density at edges (nVertLevels, nEdges).
/// @param[in]     enabled             Whether Rayleigh damping is enabled.
/// @param[in]     n_damp_levels       Number of uppermost levels to damp.
/// @param[in]     timescale_days      Damping timescale in days.
/// @param[in]     nEdges              Number of edges.
/// @param[in]     nVertLevels         Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void rayleigh_damp_u(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_u,
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> rho_edge,
    bool enabled,
    index_type n_damp_levels,
    real_type timescale_days,
    index_type nEdges,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
