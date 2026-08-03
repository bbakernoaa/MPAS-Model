#pragma once

/// @file iau.hpp
/// @brief Incremental Analysis Update (IAU) tendency kernel.
///
/// Adds an analysis increment (divided by the IAU window duration) as a constant
/// tendency to prognostic fields during the IAU time window. Outside the window,
/// this is a no-op.
///
/// The IAU technique applies analysis increments gradually over a fixed time
/// window rather than as an instantaneous adjustment, reducing dynamic shock in
/// the model fields.
///
/// @section equation IAU Tendency Equation
///
/// For each entity and vertical level during the active IAU window:
///   tend(k, iEntity) += iau_increment(k, iEntity) / iau_window_seconds
///
/// When iau_active is false, no modification is made.
///
/// @reference Bloom, S. C., Takacs, L. L., da Silva, A. M., and Ledvina, D. (1996),
/// "Data assimilation using incremental analysis updates",
/// Mon. Wea. Rev., 124, 1256-1271.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Apply Incremental Analysis Update (IAU) tendency to a field.
///
/// Adds the analysis increment (divided by the IAU window length) to the
/// tendency field during the IAU window. Outside the window, this is a no-op.
///
/// Formula: tend(k, iEntity) += iau_increment(k, iEntity) / iau_window_seconds
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy.
///
/// @param[in]     policy              Execution policy.
/// @param[in,out] tendency            Tendency field to add IAU increment to (nVertLevels, nEntities).
/// @param[in]     iau_increment       Analysis increment field (same shape as tendency).
/// @param[in]     iau_window_seconds  IAU window duration in seconds.
/// @param[in]     iau_active          Whether IAU is active (true during IAU window).
/// @param[in]     nEntities           Number of entities (cells or edges).
/// @param[in]     nVertLevels         Number of vertical levels.
template <typename Layout = default_layout, ExecutionPolicy Policy = SerialPolicy>
void apply_iau_tendency(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tendency,
    ConstField2D<Layout, unchecked_accessor> iau_increment,
    real_type iau_window_seconds,
    bool iau_active,
    index_type nEntities,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
