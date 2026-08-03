/// @file rayleigh_damping.cpp
/// @brief Implementation of the Rayleigh damping kernel for horizontal velocity.
///
/// Applies Rayleigh damping to the velocity tendency on the uppermost model
/// levels. This suppresses spurious wave reflections from the upper boundary
/// by relaxing velocity toward zero with a linearly-ramped, density-weighted
/// damping coefficient.
///
/// @section governing_equation Governing Equation
///
/// The base coefficient inverse is:
///   rayleigh_coef_inverse = 1.0 / (n_damp_levels * timescale_days * seconds_per_day)
///
/// The per-level coefficient ramps linearly (strongest at model top, k=0):
///   coef(k) = (n_damp_levels - k) * rayleigh_coef_inverse
///
/// For each edge and each damped level k in [0, n_damp_levels):
///   tend_u(k, iEdge) -= rho_edge(k, iEdge) * u(k, iEdge) * coef(k)
///
/// This matches the Fortran MPAS reference implementation which applies
/// density-weighted linear Rayleigh friction in the sponge layer.
///
/// @reference Klemp, J. B., Dudhia, J., and Hassiotis, A. (2008),
/// "An upper gravity-wave absorbing layer for NWP applications",
/// Mon. Wea. Rev., 136, 3987-4004.

#include <mpas_dycore/kernels/rayleigh_damping.hpp>
#include <mpas_dycore/constants.hpp>

#include <algorithm>

namespace mpas::dycore::kernels {

// Explicit instantiation for default layout and serial policy.
template void rayleigh_damp_u<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> tend_u,
    ConstField2D<default_layout, unchecked_accessor> u,
    ConstField2D<default_layout, unchecked_accessor> rho_edge,
    bool enabled,
    index_type n_damp_levels,
    real_type timescale_days,
    index_type nEdges,
    index_type nVertLevels);

/// @brief Implementation of rayleigh_damp_u.
///
/// When enabled is false, returns immediately (no-op). Otherwise computes
/// a linearly-ramped damping coefficient and applies density-weighted
/// damping to tend_u on the uppermost levels.
/// The number of damped levels is clamped to nVertLevels for safety.
template <typename Layout, ExecutionPolicy Policy>
void rayleigh_damp_u(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_u,
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> rho_edge,
    bool enabled,
    index_type n_damp_levels,
    real_type timescale_days,
    index_type nEdges,
    index_type nVertLevels)
{
    // No-op when damping is disabled (Requirement 21.3).
    if (!enabled) {
        return;
    }

    // Clamp n_damp_levels to nVertLevels for safety.
    const index_type levels = std::min(n_damp_levels, nVertLevels);

    // Compute base inverse coefficient (Requirement 21.2).
    // This matches the Fortran:
    //   rayleigh_coef_inverse = 1.0 / (n_damp_levels * timescale_days * seconds_per_day)
    const real_type rayleigh_coef_inverse =
        1.0 / (static_cast<real_type>(levels) * timescale_days * constants::seconds_per_day);

    // Apply density-weighted damping with linear ramp on uppermost levels.
    // In C++ convention: k=0 is model top (strongest damping),
    // k=levels-1 is bottom of sponge layer (weakest damping).
    // The ramp factor is (levels - k), giving maximum at k=0 and 1 at k=levels-1.
    policy.parallel_for(nEdges, levels, [&](index_type iEdge, index_type k) {
        const real_type coef = static_cast<real_type>(levels - k) * rayleigh_coef_inverse;
        tend_u[k, iEdge] -= rho_edge[k, iEdge] * u[k, iEdge] * coef;
    });
}

} // namespace mpas::dycore::kernels
