#pragma once

/// @file types.hpp
/// @brief Fundamental type aliases and mdspan field representations for the MPAS dynamical core.
///
/// Defines the core type system: real_type, index_type, layout policies, accessor policies,
/// and mdspan-based field view templates (Field2D, Field3D, ConnectivityView). Layout and
/// accessor policies are propagated as template parameters to all kernel instantiations,
/// enabling source-level portability across memory orderings and bounds-checking modes.

#if __has_include(<mdspan>)
#include <mdspan>
#elif __has_include(<experimental/mdspan>)
#include <experimental/mdspan>
// The Kokkos reference implementation polyfills std::mdspan, std::extents,
// std::layout_left, std::layout_right, and std::dynamic_extent directly
// into the std:: namespace when included via <experimental/mdspan>.
#else
#error "No mdspan implementation found. Install GCC 14+, or the Kokkos mdspan reference implementation."
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <concepts>
#include <type_traits>

namespace mpas::dycore {

// ============================================================================
// Fundamental types
// ============================================================================

/// Floating-point type for all prognostic and diagnostic fields.
using real_type = double;

/// Integer type for mesh indices and connectivity tables.
using index_type = std::int32_t;

// ============================================================================
// Layout policies
// ============================================================================

/// Column-major (Fortran-native) layout.
using layout_left = std::layout_left;

/// Row-major (C-native) layout.
using layout_right = std::layout_right;

/// Default layout: column-major to match Fortran memory ordering from the marshalling layer.
using default_layout = layout_left;

/// Safety check: The Fortran marshalling layer passes column-major arrays.
/// If this layout is changed, all field data from Fortran will be misinterpreted.
static_assert(std::is_same_v<default_layout, layout_left>,
    "default_layout must be layout_left (column-major) to match Fortran memory ordering");

// ============================================================================
// Accessor policies
// ============================================================================

/// @brief Bounds-checking accessor policy.
///
/// Validates that the linearized offset passed to `access()` is within
/// [0, total_size). On violation, prints a diagnostic message indicating
/// the offending index value and the total extent, then terminates via
/// std::abort(). The accessor is stateful: it stores the total element
/// count (product of all extents) to enable bounds checking at the
/// linearized-offset level.
///
/// Satisfies the C++ standard Accessor requirements for use with mdspan.
template <typename T>
struct checked_accessor {
    using element_type     = T;
    using reference        = T&;
    using data_handle_type = T*;

    using offset_policy = checked_accessor;

    /// Total number of elements in the span. Zero means "unknown/unchecked".
    std::size_t total_size_ = 0;

    constexpr checked_accessor() noexcept = default;

    /// Construct with a known total element count for bounds validation.
    explicit constexpr checked_accessor(std::size_t total) noexcept
        : total_size_(total) {}

    /// Converting constructor from compatible accessor (e.g., non-const to const).
    template <typename U>
        requires std::convertible_to<U*, T*>
    constexpr checked_accessor(const checked_accessor<U>& other) noexcept
        : total_size_(other.total_size_) {}

    /// Access element at linearized offset `i` from pointer `p`.
    /// If total_size_ > 0 and i >= total_size_, prints diagnostic and aborts.
    constexpr reference access(data_handle_type p, std::size_t i) const noexcept {
        if (total_size_ > 0 && i >= total_size_) {
            std::fprintf(stderr,
                "MPAS Dycore bounds violation: linearized index %zu >= total extent %zu\n",
                i, total_size_);
            std::abort();
        }
        return p[i];
    }

    constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept {
        return p + i;
    }
};

/// @brief Unchecked (zero-overhead) accessor policy.
///
/// Direct pointer dereference with no bounds checking. Suitable for
/// performance-critical release builds where indices are known valid.
/// Satisfies the C++ standard Accessor requirements for use with mdspan.
template <typename T>
struct unchecked_accessor {
    using element_type     = T;
    using reference        = T&;
    using data_handle_type = T*;

    using offset_policy = unchecked_accessor;

    constexpr unchecked_accessor() noexcept = default;

    /// Construct accepting (and ignoring) a total size for API compatibility
    /// with checked_accessor.
    explicit constexpr unchecked_accessor(std::size_t /*total*/) noexcept {}

    /// Converting constructor from compatible accessor (e.g., non-const to const).
    template <typename U>
        requires std::convertible_to<U*, T*>
    constexpr unchecked_accessor(const unchecked_accessor<U>&) noexcept {}

    /// Direct element access with no bounds checking.
    constexpr reference access(data_handle_type p, std::size_t i) const noexcept {
        return p[i];
    }

    constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept {
        return p + i;
    }
};

// ============================================================================
// Compile-time accessor selection
// ============================================================================

/// @brief Default accessor policy, selected at compile time.
///
/// When `MPAS_BOUNDS_CHECK` is defined (the default), uses `checked_accessor`
/// which validates all accesses. When not defined, uses `unchecked_accessor`
/// for zero-overhead access in performance-critical builds.
#ifdef MPAS_BOUNDS_CHECK
template <typename T> using default_accessor_policy = checked_accessor<T>;
#else
template <typename T> using default_accessor_policy = unchecked_accessor<T>;
#endif

// ============================================================================
// Layout concept enforcement
// ============================================================================

/// @brief Concept requiring that a Layout satisfies the C++ standard LayoutMapping requirements.
///
/// A valid layout policy must provide a nested `mapping` template that can be instantiated
/// with a 1D dynamic extents type. This enforces requirement 1.8 at compile time.
template <typename Layout>
concept ValidLayoutPolicy = requires {
    typename Layout::template mapping<std::extents<index_type, std::dynamic_extent>>;
};

// Static assertions to ensure built-in layouts satisfy the concept.
static_assert(ValidLayoutPolicy<layout_left>,
    "layout_left must satisfy ValidLayoutPolicy");
static_assert(ValidLayoutPolicy<layout_right>,
    "layout_right must satisfy ValidLayoutPolicy");

// ============================================================================
// Field view type aliases
// ============================================================================

/// @brief 2D mutable field view: (nVertLevels, nEntities).
///
/// Used for prognostic and diagnostic fields indexed by vertical level and
/// horizontal mesh entity (cells, edges, or vertices).
/// The Accessor template parameter defaults to `default_accessor_policy`,
/// which is `checked_accessor` when MPAS_BOUNDS_CHECK is defined, or
/// `unchecked_accessor` otherwise.
template <typename Layout = default_layout,
          template<typename> typename Accessor = default_accessor_policy>
using Field2D = std::mdspan<real_type,
    std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
    Layout, Accessor<real_type>>;

/// @brief 2D read-only field view: (nVertLevels, nEntities).
template <typename Layout = default_layout,
          template<typename> typename Accessor = default_accessor_policy>
using ConstField2D = std::mdspan<const real_type,
    std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
    Layout, Accessor<const real_type>>;

/// @brief 3D mutable field view: (nScalars, nVertLevels, nEntities).
///
/// Used for scalar transport fields where the additional leading dimension
/// indexes the tracer/scalar species.
template <typename Layout = default_layout,
          template<typename> typename Accessor = default_accessor_policy>
using Field3D = std::mdspan<real_type,
    std::extents<index_type, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
    Layout, Accessor<real_type>>;

/// @brief 3D read-only field view: (nScalars, nVertLevels, nEntities).
template <typename Layout = default_layout,
          template<typename> typename Accessor = default_accessor_policy>
using ConstField3D = std::mdspan<const real_type,
    std::extents<index_type, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
    Layout, Accessor<const real_type>>;

/// @brief Connectivity table view: (nEntities, maxNeighbors).
///
/// Read-only integer view over flat connectivity arrays produced by the
/// marshalling layer. Zero-based indices; INVALID_INDEX (-1) marks missing neighbors.
using ConnectivityView = std::mdspan<const index_type,
    std::extents<index_type, std::dynamic_extent, std::dynamic_extent>>;

} // namespace mpas::dycore
