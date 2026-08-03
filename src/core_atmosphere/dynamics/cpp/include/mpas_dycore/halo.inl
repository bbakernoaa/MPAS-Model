#pragma once

/// @file halo.inl
/// @brief Template implementations for HaloExchange start methods.
///
/// Included from halo.hpp. Not intended for direct inclusion.

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace mpas::dycore {

// ============================================================================
// start() for Field2D
// ============================================================================

template <typename Layout>
void HaloExchange::start(Field2D<Layout, unchecked_accessor> field,
                         const CSRHaloDescriptor& desc,
                         std::span<const int> layers,
                         MPI_Comm comm)
{
    noop_ = (desc.n_neighbors == 0);
    if (noop_) return;

    const index_type nVertLevels = field.extent(0);
    const index_type nEntities   = field.extent(1);

    // Compute total entities to send/recv across all neighbors and requested layers
    index_type total_send_count = 0;
    index_type total_recv_count = 0;
    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        index_type count = compute_layer_count(desc, layers, i);
        total_send_count += count;
        total_recv_count += count;
    }

    const index_type values_per_entity = nVertLevels;
    send_buffer_.resize(static_cast<std::size_t>(total_send_count) * values_per_entity);
    recv_buffer_.resize(static_cast<std::size_t>(total_recv_count) * values_per_entity);

    // Pack send buffer
    index_type buf_offset = 0;
    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        for (int layer : layers) {
            if (layer >= desc.n_layers) continue;
            index_type idx_offset = compute_index_offset(desc, i, layer);
            index_type count = desc.counts[static_cast<std::size_t>(i * desc.n_layers + layer)];
            for (index_type c = 0; c < count; ++c) {
                index_type entity_idx = desc.send_indices[static_cast<std::size_t>(idx_offset + c)];
                for (index_type k = 0; k < nVertLevels; ++k) {
                    send_buffer_[static_cast<std::size_t>(buf_offset)] = field[k, entity_idx];
                    ++buf_offset;
                }
            }
        }
    }

    // Post receives and sends
    requests_.clear();
    requests_.resize(static_cast<std::size_t>(2 * desc.n_neighbors));

    index_type recv_offset = 0;
    index_type send_offset = 0;
    constexpr int tag = 0;

    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        index_type count = compute_layer_count(desc, layers, i);
        index_type msg_size = count * values_per_entity;

        MPI_Irecv(recv_buffer_.data() + recv_offset,
                  static_cast<int>(msg_size), MPI_DOUBLE,
                  desc.ranks[static_cast<std::size_t>(i)], tag, comm,
                  &requests_[static_cast<std::size_t>(i)]);
        recv_offset += msg_size;
    }

    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        index_type count = compute_layer_count(desc, layers, i);
        index_type msg_size = count * values_per_entity;

        MPI_Isend(send_buffer_.data() + send_offset,
                  static_cast<int>(msg_size), MPI_DOUBLE,
                  desc.ranks[static_cast<std::size_t>(i)], tag, comm,
                  &requests_[static_cast<std::size_t>(desc.n_neighbors + i)]);
        send_offset += msg_size;
    }

    // Save state for unpack in wait()
    unpack_ptr_ = field.data_handle();
    unpack_nVertLevels_ = nVertLevels;
    unpack_nScalars_ = 0;  // 2D mode
    unpack_stride_ = nEntities;
    unpack_nFields_ = 0;
    unpack_layout_left_ = std::is_same_v<Layout, layout_left>;
    unpack_desc_ = desc;
    unpack_layers_.assign(layers.begin(), layers.end());
    unpack_field_ptrs_.clear();
}

// ============================================================================
// start() for Field3D
// ============================================================================

template <typename Layout>
void HaloExchange::start(Field3D<Layout, unchecked_accessor> field,
                         const CSRHaloDescriptor& desc,
                         std::span<const int> layers,
                         MPI_Comm comm)
{
    noop_ = (desc.n_neighbors == 0);
    if (noop_) return;

    const index_type nScalars    = field.extent(0);
    const index_type nVertLevels = field.extent(1);
    const index_type nEntities   = field.extent(2);

    // Compute total entities to send/recv
    index_type total_count = 0;
    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        total_count += compute_layer_count(desc, layers, i);
    }

    const index_type values_per_entity = nScalars * nVertLevels;
    send_buffer_.resize(static_cast<std::size_t>(total_count) * values_per_entity);
    recv_buffer_.resize(static_cast<std::size_t>(total_count) * values_per_entity);

    // Pack send buffer
    index_type buf_offset = 0;
    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        for (int layer : layers) {
            if (layer >= desc.n_layers) continue;
            index_type idx_offset = compute_index_offset(desc, i, layer);
            index_type count = desc.counts[static_cast<std::size_t>(i * desc.n_layers + layer)];
            for (index_type c = 0; c < count; ++c) {
                index_type entity_idx = desc.send_indices[static_cast<std::size_t>(idx_offset + c)];
                for (index_type s = 0; s < nScalars; ++s) {
                    for (index_type k = 0; k < nVertLevels; ++k) {
                        send_buffer_[static_cast<std::size_t>(buf_offset)] = field[s, k, entity_idx];
                        ++buf_offset;
                    }
                }
            }
        }
    }

    // Post receives and sends
    requests_.clear();
    requests_.resize(static_cast<std::size_t>(2 * desc.n_neighbors));

    index_type recv_offset = 0;
    index_type send_offset = 0;
    constexpr int tag = 1;

    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        index_type count = compute_layer_count(desc, layers, i);
        index_type msg_size = count * values_per_entity;

        MPI_Irecv(recv_buffer_.data() + recv_offset,
                  static_cast<int>(msg_size), MPI_DOUBLE,
                  desc.ranks[static_cast<std::size_t>(i)], tag, comm,
                  &requests_[static_cast<std::size_t>(i)]);
        recv_offset += msg_size;
    }

    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        index_type count = compute_layer_count(desc, layers, i);
        index_type msg_size = count * values_per_entity;

        MPI_Isend(send_buffer_.data() + send_offset,
                  static_cast<int>(msg_size), MPI_DOUBLE,
                  desc.ranks[static_cast<std::size_t>(i)], tag, comm,
                  &requests_[static_cast<std::size_t>(desc.n_neighbors + i)]);
        send_offset += msg_size;
    }

    // Save state for unpack in wait()
    unpack_ptr_ = field.data_handle();
    unpack_nVertLevels_ = nVertLevels;
    unpack_nScalars_ = nScalars;
    unpack_stride_ = nEntities;
    unpack_nFields_ = 0;
    unpack_layout_left_ = std::is_same_v<Layout, layout_left>;
    unpack_desc_ = desc;
    unpack_layers_.assign(layers.begin(), layers.end());
    unpack_field_ptrs_.clear();
}

// ============================================================================
// start_multi() for multiple Field2D
// ============================================================================

template <typename Layout>
void HaloExchange::start_multi(std::span<Field2D<Layout, unchecked_accessor>> fields,
                               const CSRHaloDescriptor& desc,
                               std::span<const int> layers,
                               MPI_Comm comm)
{
    noop_ = (desc.n_neighbors == 0);
    if (noop_) return;

    if (fields.empty()) {
        noop_ = true;
        return;
    }

    const index_type nFields     = static_cast<index_type>(fields.size());
    const index_type nVertLevels = fields[0].extent(0);
    const index_type nEntities   = fields[0].extent(1);

    // Compute total entities per neighbor
    index_type total_count = 0;
    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        total_count += compute_layer_count(desc, layers, i);
    }

    const index_type values_per_entity = nVertLevels * nFields;
    send_buffer_.resize(static_cast<std::size_t>(total_count) * values_per_entity);
    recv_buffer_.resize(static_cast<std::size_t>(total_count) * values_per_entity);

    // Pack send buffer: for each entity, pack all fields' vertical levels
    index_type buf_offset = 0;
    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        for (int layer : layers) {
            if (layer >= desc.n_layers) continue;
            index_type idx_offset = compute_index_offset(desc, i, layer);
            index_type count = desc.counts[static_cast<std::size_t>(i * desc.n_layers + layer)];
            for (index_type c = 0; c < count; ++c) {
                index_type entity_idx = desc.send_indices[static_cast<std::size_t>(idx_offset + c)];
                for (index_type f = 0; f < nFields; ++f) {
                    for (index_type k = 0; k < nVertLevels; ++k) {
                        send_buffer_[static_cast<std::size_t>(buf_offset)] = fields[static_cast<std::size_t>(f)][k, entity_idx];
                        ++buf_offset;
                    }
                }
            }
        }
    }

    // Post receives and sends
    requests_.clear();
    requests_.resize(static_cast<std::size_t>(2 * desc.n_neighbors));

    index_type recv_offset = 0;
    index_type send_offset = 0;
    constexpr int tag = 2;

    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        index_type count = compute_layer_count(desc, layers, i);
        index_type msg_size = count * values_per_entity;

        MPI_Irecv(recv_buffer_.data() + recv_offset,
                  static_cast<int>(msg_size), MPI_DOUBLE,
                  desc.ranks[static_cast<std::size_t>(i)], tag, comm,
                  &requests_[static_cast<std::size_t>(i)]);
        recv_offset += msg_size;
    }

    for (index_type i = 0; i < desc.n_neighbors; ++i) {
        index_type count = compute_layer_count(desc, layers, i);
        index_type msg_size = count * values_per_entity;

        MPI_Isend(send_buffer_.data() + send_offset,
                  static_cast<int>(msg_size), MPI_DOUBLE,
                  desc.ranks[static_cast<std::size_t>(i)], tag, comm,
                  &requests_[static_cast<std::size_t>(desc.n_neighbors + i)]);
        send_offset += msg_size;
    }

    // Save state for unpack in wait()
    unpack_ptr_ = nullptr;  // multi-field mode uses field_ptrs
    unpack_nVertLevels_ = nVertLevels;
    unpack_nScalars_ = 0;
    unpack_stride_ = nEntities;
    unpack_nFields_ = nFields;
    unpack_layout_left_ = std::is_same_v<Layout, layout_left>;
    unpack_desc_ = desc;
    unpack_layers_.assign(layers.begin(), layers.end());
    unpack_field_ptrs_.clear();
    unpack_field_ptrs_.reserve(static_cast<std::size_t>(nFields));
    for (index_type f = 0; f < nFields; ++f) {
        unpack_field_ptrs_.push_back(fields[static_cast<std::size_t>(f)].data_handle());
    }
}

} // namespace mpas::dycore
