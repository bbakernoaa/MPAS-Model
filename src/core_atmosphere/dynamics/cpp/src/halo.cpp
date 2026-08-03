/// @file halo.cpp
/// @brief Non-template implementation of HaloExchange methods.

#include <mpas_dycore/halo.hpp>

#include <algorithm>
#include <numeric>

namespace mpas::dycore {

// ============================================================================
// is_noop
// ============================================================================

bool HaloExchange::is_noop() const
{
    return noop_;
}

// ============================================================================
// wait
// ============================================================================

void HaloExchange::wait()
{
    if (noop_) return;

    // Block until all sends and receives complete
    MPI_Waitall(static_cast<int>(requests_.size()),
                requests_.data(), MPI_STATUSES_IGNORE);

    // Unpack receive buffer into field(s)
    std::span<const int> layers(unpack_layers_);

    index_type buf_offset = 0;

    if (unpack_nFields_ > 0) {
        // Multi-field 2D unpack
        const index_type nVertLevels = unpack_nVertLevels_;
        const index_type nEntities   = unpack_stride_;
        const index_type nFields     = unpack_nFields_;

        for (index_type i = 0; i < unpack_desc_.n_neighbors; ++i) {
            for (int layer : layers) {
                if (layer >= unpack_desc_.n_layers) continue;
                index_type idx_offset = compute_index_offset(unpack_desc_, i, layer);
                index_type count = unpack_desc_.counts[
                    static_cast<std::size_t>(i * unpack_desc_.n_layers + layer)];
                for (index_type c = 0; c < count; ++c) {
                    index_type entity_idx = unpack_desc_.recv_indices[
                        static_cast<std::size_t>(idx_offset + c)];
                    for (index_type f = 0; f < nFields; ++f) {
                        real_type* field_ptr = unpack_field_ptrs_[static_cast<std::size_t>(f)];
                        for (index_type k = 0; k < nVertLevels; ++k) {
                            // Compute linear index based on layout
                            std::size_t linear_idx;
                            if (unpack_layout_left_) {
                                // layout_left: (nVertLevels, nEntities) -> k + entity_idx * nVertLevels
                                linear_idx = static_cast<std::size_t>(k)
                                    + static_cast<std::size_t>(entity_idx) * static_cast<std::size_t>(nVertLevels);
                            } else {
                                // layout_right: (nVertLevels, nEntities) -> entity_idx + k * nEntities
                                linear_idx = static_cast<std::size_t>(entity_idx)
                                    + static_cast<std::size_t>(k) * static_cast<std::size_t>(nEntities);
                            }
                            field_ptr[linear_idx] = recv_buffer_[static_cast<std::size_t>(buf_offset)];
                            ++buf_offset;
                        }
                    }
                }
            }
        }
    } else if (unpack_nScalars_ == 0) {
        // Single-field 2D unpack
        const index_type nVertLevels = unpack_nVertLevels_;
        const index_type nEntities   = unpack_stride_;

        for (index_type i = 0; i < unpack_desc_.n_neighbors; ++i) {
            for (int layer : layers) {
                if (layer >= unpack_desc_.n_layers) continue;
                index_type idx_offset = compute_index_offset(unpack_desc_, i, layer);
                index_type count = unpack_desc_.counts[
                    static_cast<std::size_t>(i * unpack_desc_.n_layers + layer)];
                for (index_type c = 0; c < count; ++c) {
                    index_type entity_idx = unpack_desc_.recv_indices[
                        static_cast<std::size_t>(idx_offset + c)];
                    for (index_type k = 0; k < nVertLevels; ++k) {
                        std::size_t linear_idx;
                        if (unpack_layout_left_) {
                            linear_idx = static_cast<std::size_t>(k)
                                + static_cast<std::size_t>(entity_idx) * static_cast<std::size_t>(nVertLevels);
                        } else {
                            linear_idx = static_cast<std::size_t>(entity_idx)
                                + static_cast<std::size_t>(k) * static_cast<std::size_t>(nEntities);
                        }
                        unpack_ptr_[linear_idx] = recv_buffer_[static_cast<std::size_t>(buf_offset)];
                        ++buf_offset;
                    }
                }
            }
        }
    } else {
        // 3D field unpack
        const index_type nScalars    = unpack_nScalars_;
        const index_type nVertLevels = unpack_nVertLevels_;
        const index_type nEntities   = unpack_stride_;

        for (index_type i = 0; i < unpack_desc_.n_neighbors; ++i) {
            for (int layer : layers) {
                if (layer >= unpack_desc_.n_layers) continue;
                index_type idx_offset = compute_index_offset(unpack_desc_, i, layer);
                index_type count = unpack_desc_.counts[
                    static_cast<std::size_t>(i * unpack_desc_.n_layers + layer)];
                for (index_type c = 0; c < count; ++c) {
                    index_type entity_idx = unpack_desc_.recv_indices[
                        static_cast<std::size_t>(idx_offset + c)];
                    for (index_type s = 0; s < nScalars; ++s) {
                        for (index_type k = 0; k < nVertLevels; ++k) {
                            std::size_t linear_idx;
                            if (unpack_layout_left_) {
                                // layout_left: (nScalars, nVertLevels, nEntities)
                                // -> s + k * nScalars + entity_idx * nScalars * nVertLevels
                                linear_idx = static_cast<std::size_t>(s)
                                    + static_cast<std::size_t>(k) * static_cast<std::size_t>(nScalars)
                                    + static_cast<std::size_t>(entity_idx) * static_cast<std::size_t>(nScalars) * static_cast<std::size_t>(nVertLevels);
                            } else {
                                // layout_right: (nScalars, nVertLevels, nEntities)
                                // -> entity_idx + k * nEntities + s * nEntities * nVertLevels
                                linear_idx = static_cast<std::size_t>(entity_idx)
                                    + static_cast<std::size_t>(k) * static_cast<std::size_t>(nEntities)
                                    + static_cast<std::size_t>(s) * static_cast<std::size_t>(nEntities) * static_cast<std::size_t>(nVertLevels);
                            }
                            unpack_ptr_[linear_idx] = recv_buffer_[static_cast<std::size_t>(buf_offset)];
                            ++buf_offset;
                        }
                    }
                }
            }
        }
    }

    // Clear state
    requests_.clear();
}

// ============================================================================
// Static helpers
// ============================================================================

index_type HaloExchange::compute_layer_count(
    const CSRHaloDescriptor& desc,
    std::span<const int> layers,
    index_type neighbor_idx)
{
    index_type total = 0;
    for (int layer : layers) {
        if (layer >= desc.n_layers) continue;
        total += desc.counts[static_cast<std::size_t>(neighbor_idx * desc.n_layers + layer)];
    }
    return total;
}

index_type HaloExchange::compute_index_offset(
    const CSRHaloDescriptor& desc,
    index_type neighbor_idx,
    index_type layer_idx)
{
    // Sum all counts for neighbors before this one (all layers),
    // plus counts for this neighbor's layers before layer_idx
    index_type offset = 0;
    for (index_type i = 0; i < neighbor_idx; ++i) {
        for (index_type j = 0; j < desc.n_layers; ++j) {
            offset += desc.counts[static_cast<std::size_t>(i * desc.n_layers + j)];
        }
    }
    for (index_type j = 0; j < layer_idx; ++j) {
        offset += desc.counts[static_cast<std::size_t>(neighbor_idx * desc.n_layers + j)];
    }
    return offset;
}

} // namespace mpas::dycore
