#pragma once

/// @file execution_policy.hpp
/// @brief Execution policy abstraction for dispatching (entity, level) iteration patterns.
///
/// Defines the ExecutionPolicy concept and concrete policy implementations:
/// - SerialPolicy: sequential nested loops (default)
/// - OpenMPPolicy: outer loop parallelized with OpenMP
/// - GPUPolicy: compile-time selectable GPU dispatch stub (CUDA/HIP/SYCL)
///
/// Kernels accept the policy as a template parameter and remain agnostic to
/// the dispatch mechanism. The DefaultPolicy alias is selected via preprocessor
/// macros (MPAS_USE_GPU, MPAS_USE_OPENMP).

#include "types.hpp"

#include <concepts>
#include <cstdio>
#include <cstdlib>

namespace mpas::dycore {

// ============================================================================
// Execution Policy Concept
// ============================================================================

/// @brief Concept requiring a type to provide parallel_for over (entity, level) pairs.
///
/// A valid ExecutionPolicy must expose a `parallel_for` method that accepts:
/// - n_entities: number of mesh entities (cells, edges, or vertices)
/// - n_levels: number of vertical levels
/// - kernel: a callable invoked as kernel(iEntity, k) for each pair
///
/// This enables kernels to be written independently of the iteration strategy.
template <typename Policy>
concept ExecutionPolicy = requires(Policy p,
    index_type n_entities, index_type n_levels,
    void(*kernel)(index_type, index_type)) {
    { p.parallel_for(n_entities, n_levels, kernel) } -> std::same_as<void>;
};

// ============================================================================
// Serial Execution Policy
// ============================================================================

/// @brief Serial execution policy: nested loops in index order.
///
/// Outer loop iterates over mesh entities sequentially in ascending index order.
/// Inner loop iterates over vertical levels sequentially in ascending order.
/// This is the compile-time default when no parallel backend is configured.
struct SerialPolicy {
    /// Dispatch kernel for all (entity, level) pairs sequentially.
    void parallel_for(index_type n_entities, index_type n_levels, auto&& kernel) const {
        for (index_type iEntity = 0; iEntity < n_entities; ++iEntity)
            for (index_type k = 0; k < n_levels; ++k)
                kernel(iEntity, k);
    }
};

// ============================================================================
// OpenMP Execution Policy
// ============================================================================

/// @brief OpenMP execution policy: outer loop parallelized with #pragma omp parallel for.
///
/// The outermost mesh-entity loop is parallelized using OpenMP with a static schedule.
/// The inner vertical-level loop executes sequentially within each thread.
struct OpenMPPolicy {
    /// Dispatch kernel with OpenMP parallelism on the entity dimension.
    void parallel_for(index_type n_entities, index_type n_levels, auto&& kernel) const {
        #pragma omp parallel for schedule(static)
        for (index_type iEntity = 0; iEntity < n_entities; ++iEntity)
            for (index_type k = 0; k < n_levels; ++k)
                kernel(iEntity, k);
    }
};

// ============================================================================
// GPU Execution Policy (Stub)
// ============================================================================

/// @brief GPU execution policy stub, compile-time selectable via MPAS_GPU_BACKEND.
///
/// In production, this would dispatch kernels as GPU threads using CUDA, HIP, or SYCL
/// (selected by the MPAS_GPU_BACKEND preprocessor define), mapping mesh entities to
/// thread blocks and vertical levels to threads within a block.
///
/// Currently implemented as a serial fallback. When MPAS_USE_GPU is defined but no
/// compatible GPU device is available at runtime, the real implementation would
/// terminate with a diagnostic before executing any kernel.
struct GPUPolicy {
    /// Dispatch kernel — currently falls back to serial execution.
    /// Real implementation would dispatch via CUDA/HIP/SYCL based on MPAS_GPU_BACKEND.
    void parallel_for(index_type n_entities, index_type n_levels, auto&& kernel) const {
#if defined(MPAS_GPU_BACKEND)
        // Stub: In a full implementation, this would launch a GPU kernel using
        // the backend specified by MPAS_GPU_BACKEND (CUDA, HIP, or SYCL).
        // For now, fall back to serial execution.
        for (index_type iEntity = 0; iEntity < n_entities; ++iEntity)
            for (index_type k = 0; k < n_levels; ++k)
                kernel(iEntity, k);
#else
        // No GPU backend configured — serial fallback.
        for (index_type iEntity = 0; iEntity < n_entities; ++iEntity)
            for (index_type k = 0; k < n_levels; ++k)
                kernel(iEntity, k);
#endif
    }
};

// ============================================================================
// Concept satisfaction checks
// ============================================================================

static_assert(ExecutionPolicy<SerialPolicy>,
    "SerialPolicy must satisfy ExecutionPolicy concept");
static_assert(ExecutionPolicy<OpenMPPolicy>,
    "OpenMPPolicy must satisfy ExecutionPolicy concept");
static_assert(ExecutionPolicy<GPUPolicy>,
    "GPUPolicy must satisfy ExecutionPolicy concept");

// ============================================================================
// Default Policy Selection
// ============================================================================

/// @brief Default execution policy, selected at compile time via preprocessor macros.
///
/// - If MPAS_USE_GPU is defined: GPUPolicy
/// - Else if MPAS_USE_OPENMP is defined: OpenMPPolicy
/// - Otherwise: SerialPolicy (sequential execution)
#if defined(MPAS_USE_GPU)
using DefaultPolicy = GPUPolicy;
#elif defined(MPAS_USE_OPENMP)
using DefaultPolicy = OpenMPPolicy;
#else
using DefaultPolicy = SerialPolicy;
#endif

} // namespace mpas::dycore
