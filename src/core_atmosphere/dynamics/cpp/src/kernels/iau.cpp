/// @file iau.cpp
/// @brief Implementation of the Incremental Analysis Update (IAU) tendency kernel.
///
/// Applies analysis increments as a constant tendency over the IAU time window.
/// The increment is divided by the window duration so that by the end of the
/// window the total accumulated tendency equals the original analysis increment.
///
/// @section governing_equation Governing Equation
///
/// For each entity and vertical level during the active IAU window:
///   tend(k, iEntity) += iau_increment(k, iEntity) / iau_window_seconds
///
/// When iau_active is false, the function returns immediately with no modification.
///
/// @reference Bloom, S. C., Takacs, L. L., da Silva, A. M., and Ledvina, D. (1996),
/// "Data assimilation using incremental analysis updates",
/// Mon. Wea. Rev., 124, 1256-1271.

#include <mpas_dycore/kernels/iau.hpp>

namespace mpas::dycore::kernels {

// Explicit instantiation for default layout and serial policy.
template void apply_iau_tendency<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> tendency,
    ConstField2D<default_layout, unchecked_accessor> iau_increment,
    real_type iau_window_seconds,
    bool iau_active,
    index_type nEntities,
    index_type nVertLevels);

/// @brief Implementation of apply_iau_tendency.
///
/// When iau_active is false, returns immediately (no-op, Requirement 19.3).
/// Otherwise adds the analysis increment divided by the IAU window duration
/// to the tendency field at every (entity, level) pair (Requirements 19.1, 19.2).
template <typename Layout, ExecutionPolicy Policy>
void apply_iau_tendency(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tendency,
    ConstField2D<Layout, unchecked_accessor> iau_increment,
    real_type iau_window_seconds,
    bool iau_active,
    index_type nEntities,
    index_type nVertLevels)
{
    // No-op when IAU is inactive (outside the IAU time window).
    if (!iau_active) {
        return;
    }

    // Precompute inverse to avoid repeated division in the inner loop.
    const real_type inv_window = 1.0 / iau_window_seconds;

    // Apply IAU tendency: tend += increment / window_seconds
    policy.parallel_for(nEntities, nVertLevels, [&](index_type iEntity, index_type k) {
        tendency[k, iEntity] += iau_increment[k, iEntity] * inv_window;
    });
}

} // namespace mpas::dycore::kernels
