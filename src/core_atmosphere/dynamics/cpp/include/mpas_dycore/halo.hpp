#pragma once

/// @file halo.hpp
/// @brief Non-blocking MPI halo exchange for the MPAS dynamical core.
///
/// Defines the CSRHaloDescriptor struct (CSR representation of halo communication
/// patterns from the Fortran marshalling layer) and the HaloExchange class that
/// manages pack/send/recv/unpack cycles for 2D and 3D mdspan field views.
///
/// The halo exchange is decoupled from computation kernels: kernels operate on
/// local field views, and the SRK3 orchestrator inserts start()/wait() calls at
/// algorithm-mandated synchronization points.

#include <mpas_dycore/types.hpp>

#include <mpi.h>
#include <span>
#include <vector>

namespace mpas::dycore {

// ============================================================================
// CSR Halo Descriptor
// ============================================================================

/// @brief Compressed sparse row representation of halo exchange patterns.
///
/// Produced by the Fortran marshalling layer. Contains neighbor ranks,
/// per-neighbor per-layer send/recv counts, and flattened zero-based index
/// lists for pack (send) and unpack (recv) operations.
///
/// The `counts` array is organized as (n_neighbors * n_layers) entries.
/// For neighbor i and layer j, the count is counts[i * n_layers + j].
/// The send_indices and recv_indices are flattened CSR-style: the segment
/// for neighbor i, layer j starts at the sum of all preceding counts.
struct CSRHaloDescriptor {
    /// Number of neighboring ranks this process communicates with.
    index_type n_neighbors;

    /// Number of halo layers (1 to 3).
    index_type n_layers;

    /// Ranks of neighboring processes: size (n_neighbors).
    std::span<const int> ranks;

    /// Per-neighbor, per-layer send/recv counts: size (n_neighbors * n_layers).
    /// counts[i * n_layers + j] = number of entities to exchange with
    /// neighbor i for layer j.
    std::span<const int> counts;

    /// Flattened CSR source indices for packing send buffers.
    /// These are zero-based mesh entity indices identifying which local
    /// entities to pack for each neighbor/layer combination.
    std::span<const int> send_indices;

    /// Flattened CSR destination indices for unpacking receive buffers.
    /// These are zero-based mesh entity indices identifying where to
    /// write received data for each neighbor/layer combination.
    std::span<const int> recv_indices;
};

// ============================================================================
// HaloExchange class
// ============================================================================

/// @brief Non-blocking MPI halo exchange manager.
///
/// Usage pattern:
///   HaloExchange halo;
///   halo.start(field, desc, layers, comm);
///   // ... overlap computation ...
///   halo.wait();  // blocks until exchange complete, unpacks into field
///
/// For single-process runs (n_neighbors == 0), all operations are no-ops.
class HaloExchange {
public:
    HaloExchange() = default;

    /// @brief Initiate non-blocking halo exchange for a 2D field.
    ///
    /// Packs outgoing data from the field into send buffers, posts MPI_Irecv
    /// for each neighbor, then posts MPI_Isend. Call wait() to complete.
    ///
    /// @tparam Layout  mdspan layout policy (layout_left or layout_right).
    /// @param field    Mutable 2D field view: (nVertLevels, nEntities).
    /// @param desc     CSR halo descriptor with neighbor info and index lists.
    /// @param layers   Which halo layers to exchange (0-based layer indices).
    /// @param comm     MPI communicator.
    template <typename Layout>
    void start(Field2D<Layout, unchecked_accessor> field,
               const CSRHaloDescriptor& desc,
               std::span<const int> layers,
               MPI_Comm comm);

    /// @brief Initiate non-blocking halo exchange for a 3D field.
    ///
    /// Same semantics as the 2D variant but operates on a 3D field
    /// (nScalars, nVertLevels, nEntities). All scalars and vertical levels
    /// for each indexed entity are packed/unpacked.
    ///
    /// @tparam Layout  mdspan layout policy.
    /// @param field    Mutable 3D field view: (nScalars, nVertLevels, nEntities).
    /// @param desc     CSR halo descriptor.
    /// @param layers   Which halo layers to exchange.
    /// @param comm     MPI communicator.
    template <typename Layout>
    void start(Field3D<Layout, unchecked_accessor> field,
               const CSRHaloDescriptor& desc,
               std::span<const int> layers,
               MPI_Comm comm);

    /// @brief Initiate halo exchange packing multiple 2D fields per message.
    ///
    /// Reduces message count by packing all fields into a single buffer
    /// per neighbor. All fields must share the same layout and extents.
    ///
    /// @tparam Layout  mdspan layout policy.
    /// @param fields   Span of mutable 2D field views to exchange.
    /// @param desc     CSR halo descriptor.
    /// @param layers   Which halo layers to exchange.
    /// @param comm     MPI communicator.
    template <typename Layout>
    void start_multi(std::span<Field2D<Layout, unchecked_accessor>> fields,
                     const CSRHaloDescriptor& desc,
                     std::span<const int> layers,
                     MPI_Comm comm);

    /// @brief Wait for all outstanding sends/receives to complete, then unpack.
    ///
    /// Blocks until all MPI requests posted by the most recent start() or
    /// start_multi() call have completed. After waiting, unpacks received
    /// data from recv_buffer_ into the field at the destination indices.
    void wait();

    /// @brief Returns true if this is a no-op (single-process, n_neighbors == 0).
    ///
    /// When true, start()/start_multi()/wait() return immediately without
    /// issuing any MPI calls or modifying the field.
    bool is_noop() const;

private:
    /// Outstanding MPI requests (sends + receives).
    std::vector<MPI_Request> requests_;

    /// Contiguous send buffer for packed outgoing data.
    std::vector<real_type> send_buffer_;

    /// Contiguous receive buffer for incoming data.
    std::vector<real_type> recv_buffer_;

    /// True when n_neighbors == 0 (single-process, no communication needed).
    bool noop_ = true;

    // ---- State saved for unpack during wait() ----

    /// Pointer to the field data for unpacking (set during start).
    real_type* unpack_ptr_ = nullptr;

    /// Number of vertical levels in the field (for 2D: extent(0)).
    index_type unpack_nVertLevels_ = 0;

    /// Number of scalars (for 3D fields; 0 means 2D mode).
    index_type unpack_nScalars_ = 0;

    /// Second extent of the field (nEntities for 2D, nVertLevels for 3D layout).
    index_type unpack_stride_ = 0;

    /// Number of fields packed (for multi-field mode).
    index_type unpack_nFields_ = 0;

    /// Layout tag: true = layout_left (column-major), false = layout_right.
    bool unpack_layout_left_ = true;

    /// Saved descriptor for unpack.
    CSRHaloDescriptor unpack_desc_{};

    /// Saved layers for unpack.
    std::vector<int> unpack_layers_;

    /// Saved field pointers for multi-field unpack.
    std::vector<real_type*> unpack_field_ptrs_;

    /// Helper: compute total index count for specified layers for one neighbor.
    static index_type compute_layer_count(
        const CSRHaloDescriptor& desc,
        std::span<const int> layers,
        index_type neighbor_idx);

    /// Helper: compute offset into index array for a given neighbor and layer.
    static index_type compute_index_offset(
        const CSRHaloDescriptor& desc,
        index_type neighbor_idx,
        index_type layer_idx);
};

} // namespace mpas::dycore

// ============================================================================
// Template implementations (must be in header for ODR compliance)
// ============================================================================

#include <mpas_dycore/halo.inl>
