#ifndef MPAS_DYCORE_ACCUMULATION_HPP
#define MPAS_DYCORE_ACCUMULATION_HPP

/// @file accumulation.hpp
/// @brief Deterministic unstructured accumulation utilities for the C++ dycore.
///
/// This header provides two reusable accumulation strategies for scatter/gather
/// operations on unstructured meshes (e.g. edge-to-cell flux gathers):
///
/// 1. **ColoredAccumulator** — Uses KokkosKernels distance-1 graph coloring to
///    partition source elements (edges) into color sets such that no two sources
///    of the same color write to the same target (cell). The accumulation iterates
///    color sets sequentially and parallelizes within a color. This removes races
///    without atomics and, because the per-target accumulation order is fixed by
///    the coloring, keeps floating-point results reproducible (Requirement 2.6).
///
/// 2. **TeamDuplicationAccumulator** — A fallback for cases where no natural
///    coloring exists. Uses thread-team local copies of the output array and a
///    sequential reduction across teams to produce deterministic results.
///
/// Both strategies are templated on `ExecSpace` so they work on host and device.
/// They are header-only (templated) and deterministic: the same result is produced
/// regardless of thread scheduling.

#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>
#include <KokkosKernels_Handle.hpp>
#include <KokkosGraph_Distance1Color.hpp>

#include <type_traits>

namespace mpas {
namespace dycore {

// ============================================================================
// ColoredAccumulator
// ============================================================================

/// @brief Graph-coloring-based deterministic accumulator for unstructured meshes.
///
/// Given a connectivity relationship (e.g. cellsOnEdge mapping edges to cells),
/// this class builds a conflict graph and colors it so that same-color source
/// elements never share a write target. The `accumulate()` method then iterates
/// colors sequentially (fixing accumulation order) and parallelizes within each
/// color (no races, no atomics).
///
/// @tparam ExecSpace  Kokkos execution space (host or device).
/// @tparam ScalarT    Numeric type for accumulated values (defaults to mpas::dycore::Scalar).
template <class ExecSpace, class ScalarT = Scalar>
class ColoredAccumulator {
 public:
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using ordinal_type = int;
  using size_type = int;

  // CRS graph views (stored in device memory)
  using rowmap_view = Kokkos::View<size_type*, memory_space>;
  using entries_view = Kokkos::View<ordinal_type*, memory_space>;
  using colors_view = Kokkos::View<ordinal_type*, memory_space>;

  // 1-D value views
  using value_view = Kokkos::View<ScalarT*, memory_space>;

  /// @brief Construct from raw connectivity arrays.
  ///
  /// Builds a source-element conflict graph (two source elements conflict if
  /// they write to the same target) and computes a distance-1 coloring.
  ///
  /// @param n_sources     Number of source elements (e.g. nEdges).
  /// @param n_targets     Number of target elements (e.g. nCells).
  /// @param sources_per_target  Max number of sources that map to one target (e.g. maxEdges).
  /// @param target_of_source    View of shape (max_targets_per_source, n_sources) giving
  ///                            target indices for each source (0-based). For cellsOnEdge
  ///                            this is (2, nEdges) in LayoutLeft, flattened to a 1-D accessor.
  ///                            Negative or out-of-range values indicate "no target".
  /// @param max_targets_per_source  How many targets each source writes to (e.g. 2 for edges).
  ColoredAccumulator(ordinal_type n_sources,
                     ordinal_type n_targets,
                     const Kokkos::View<const ordinal_type**, Kokkos::LayoutLeft, memory_space>& target_of_source,
                     ordinal_type max_targets_per_source)
      : n_sources_(n_sources),
        n_targets_(n_targets),
        max_targets_per_source_(max_targets_per_source),
        target_of_source_(target_of_source) {
    build_conflict_graph_and_color();
  }

  /// @brief Construct from raw cellsOnEdge-style connectivity (convenience).
  ///
  /// @param n_edges    Number of edges (source elements).
  /// @param n_cells    Number of cells (target elements).
  /// @param cells_on_edge  View of shape (2, nEdges) in LayoutLeft, giving the two
  ///                       cell neighbors of each edge (0-based indices, -1 for boundary).
  ColoredAccumulator(ordinal_type n_edges,
                     ordinal_type n_cells,
                     const Kokkos::View<const ordinal_type**, Kokkos::LayoutLeft, memory_space>& cells_on_edge)
      : ColoredAccumulator(n_edges, n_cells, cells_on_edge, 2) {}

  /// @brief Deterministic accumulation: scatter edge values into target cells.
  ///
  /// For each source element `e`, adds `op(e)` to `output(target_of_source(t, e))`
  /// for each valid target `t`. The operation iterates colors sequentially;
  /// within each color, sources are processed in parallel. This guarantees
  /// determinism: the per-target accumulation order is the same regardless of
  /// thread scheduling.
  ///
  /// @tparam OpFunctor  A functor with signature `ScalarT operator()(ordinal_type source, ordinal_type target_slot) const`
  ///                    returning the value to accumulate from `source` into target slot `target_slot`.
  /// @param output      1-D View of size >= n_targets, initialized to zero by caller.
  /// @param op          The scatter functor.
  template <class OpFunctor>
  void accumulate(const value_view& output, OpFunctor op) const {
    // Iterate colors sequentially for determinism
    for (ordinal_type color = 1; color <= num_colors_; ++color) {
      const auto colors = colors_;
      const auto target_of_source = target_of_source_;
      const ordinal_type max_tps = max_targets_per_source_;
      const ordinal_type n_src = n_sources_;
      const ordinal_type n_tgt = n_targets_;

      Kokkos::parallel_for(
          "ColoredAccumulator::accumulate_color",
          Kokkos::RangePolicy<exec_space>(0, n_src),
          KOKKOS_LAMBDA(const ordinal_type e) {
            if (colors(e) != color) return;
            for (ordinal_type t = 0; t < max_tps; ++t) {
              ordinal_type target = target_of_source(t, e);
              if (target >= 0 && target < n_tgt) {
                output(target) += op(e, t);
              }
            }
          });
      Kokkos::fence("ColoredAccumulator::color_fence");
    }
  }

  /// @brief Multi-level (2-D) accumulation variant.
  ///
  /// For computations where each source contributes a column of values
  /// (e.g. nVertLevels per edge), this iterates colors sequentially and
  /// parallelizes over (source, level) within each color.
  ///
  /// @tparam OpFunctor  Functor with `ScalarT operator()(ordinal_type source, ordinal_type level, ordinal_type target_slot) const`.
  /// @param output      2-D View (n_levels, n_targets) in LayoutLeft.
  /// @param n_levels    Number of vertical levels.
  /// @param op          The scatter functor.
  template <class OpFunctor>
  void accumulate_2d(const Kokkos::View<ScalarT**, Kokkos::LayoutLeft, memory_space>& output,
                     ordinal_type n_levels,
                     OpFunctor op) const {
    for (ordinal_type color = 1; color <= num_colors_; ++color) {
      const auto colors = colors_;
      const auto target_of_source = target_of_source_;
      const ordinal_type max_tps = max_targets_per_source_;
      const ordinal_type n_src = n_sources_;
      const ordinal_type n_tgt = n_targets_;

      Kokkos::parallel_for(
          "ColoredAccumulator::accumulate_2d_color",
          Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
              {0, 0}, {n_levels, n_src}),
          KOKKOS_LAMBDA(const ordinal_type k, const ordinal_type e) {
            if (colors(e) != color) return;
            for (ordinal_type t = 0; t < max_tps; ++t) {
              ordinal_type target = target_of_source(t, e);
              if (target >= 0 && target < n_tgt) {
                output(k, target) += op(e, k, t);
              }
            }
          });
      Kokkos::fence("ColoredAccumulator::color_2d_fence");
    }
  }

  /// @brief Get the number of colors used.
  ordinal_type num_colors() const { return num_colors_; }

  /// @brief Get the per-source color assignment view.
  colors_view get_colors() const { return colors_; }

 private:
  ordinal_type n_sources_;
  ordinal_type n_targets_;
  ordinal_type max_targets_per_source_;
  Kokkos::View<const ordinal_type**, Kokkos::LayoutLeft, memory_space> target_of_source_;

  ordinal_type num_colors_ = 0;
  colors_view colors_;

  /// Build a CRS conflict graph among sources (two sources conflict if they
  /// share at least one target), then run KokkosKernels distance-1 coloring.
  void build_conflict_graph_and_color() {
    // Step 1: Build an inverted mapping: for each target, list which sources write to it.
    // We need this to build the source-source conflict graph.
    //
    // First pass: count how many sources map to each target.
    Kokkos::View<ordinal_type*, memory_space> target_count("target_count", n_targets_);
    {
      const auto target_of_source = target_of_source_;
      const ordinal_type max_tps = max_targets_per_source_;
      const ordinal_type n_src = n_sources_;
      const ordinal_type n_tgt = n_targets_;

      Kokkos::parallel_for(
          "count_sources_per_target",
          Kokkos::RangePolicy<exec_space>(0, n_src),
          KOKKOS_LAMBDA(const ordinal_type e) {
            for (ordinal_type t = 0; t < max_tps; ++t) {
              ordinal_type tgt = target_of_source(t, e);
              if (tgt >= 0 && tgt < n_tgt) {
                Kokkos::atomic_inc(&target_count(tgt));
              }
            }
          });
      Kokkos::fence("count_sources_fence");
    }

    // Compute prefix sum for target->source CSR row map
    Kokkos::View<size_type*, memory_space> inv_rowmap("inv_rowmap", n_targets_ + 1);
    {
      auto target_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, target_count);
      auto inv_rowmap_h = Kokkos::create_mirror_view(inv_rowmap);
      inv_rowmap_h(0) = 0;
      for (ordinal_type i = 0; i < n_targets_; ++i) {
        inv_rowmap_h(i + 1) = inv_rowmap_h(i) + target_count_h(i);
      }
      Kokkos::deep_copy(inv_rowmap, inv_rowmap_h);
    }

    // Second pass: fill the inverted entries (target -> list of sources)
    auto inv_rowmap_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, inv_rowmap);
    size_type total_inv_entries = inv_rowmap_h(n_targets_);
    Kokkos::View<ordinal_type*, memory_space> inv_entries("inv_entries", total_inv_entries);
    // Use a counter view to track insertion position per target
    Kokkos::View<ordinal_type*, memory_space> insert_pos("insert_pos", n_targets_);
    {
      const auto target_of_source = target_of_source_;
      const auto inv_rm = inv_rowmap;
      const ordinal_type max_tps = max_targets_per_source_;
      const ordinal_type n_src = n_sources_;
      const ordinal_type n_tgt = n_targets_;

      Kokkos::parallel_for(
          "fill_inv_entries",
          Kokkos::RangePolicy<exec_space>(0, n_src),
          KOKKOS_LAMBDA(const ordinal_type e) {
            for (ordinal_type t = 0; t < max_tps; ++t) {
              ordinal_type tgt = target_of_source(t, e);
              if (tgt >= 0 && tgt < n_tgt) {
                ordinal_type pos = Kokkos::atomic_fetch_add(&insert_pos(tgt), 1);
                inv_entries(inv_rm(tgt) + pos) = e;
              }
            }
          });
      Kokkos::fence("fill_inv_entries_fence");
    }

    // Step 2: Build the source-source conflict graph in CRS format.
    // Two sources conflict if they share a target. We iterate over each target's
    // source list and form edges between all pairs.
    //
    // For efficiency, we build this on the host.
    auto inv_rowmap_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, inv_rowmap);
    auto inv_entries_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, inv_entries);

    // Count neighbors per source
    std::vector<std::vector<ordinal_type>> adj_lists(n_sources_);
    for (ordinal_type tgt = 0; tgt < n_targets_; ++tgt) {
      size_type begin = inv_rowmap_host(tgt);
      size_type end = inv_rowmap_host(tgt + 1);
      // All sources in [begin, end) share this target -> pairwise edges
      for (size_type i = begin; i < end; ++i) {
        for (size_type j = begin; j < end; ++j) {
          if (i != j) {
            adj_lists[inv_entries_host(i)].push_back(inv_entries_host(j));
          }
        }
      }
    }

    // Deduplicate adjacency lists and build CRS
    std::vector<size_type> crs_rowmap(n_sources_ + 1, 0);
    std::vector<ordinal_type> crs_entries;
    for (ordinal_type s = 0; s < n_sources_; ++s) {
      auto& adj = adj_lists[s];
      std::sort(adj.begin(), adj.end());
      adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
      crs_rowmap[s + 1] = crs_rowmap[s] + static_cast<size_type>(adj.size());
      crs_entries.insert(crs_entries.end(), adj.begin(), adj.end());
    }

    // Copy CRS to device
    size_type total_entries = static_cast<size_type>(crs_entries.size());
    rowmap_view rowmap_dev("conflict_rowmap", n_sources_ + 1);
    entries_view entries_dev("conflict_entries", total_entries);

    {
      auto rowmap_host = Kokkos::create_mirror_view(rowmap_dev);
      for (ordinal_type i = 0; i <= n_sources_; ++i) {
        rowmap_host(i) = crs_rowmap[i];
      }
      Kokkos::deep_copy(rowmap_dev, rowmap_host);

      auto entries_host = Kokkos::create_mirror_view(entries_dev);
      for (size_type i = 0; i < total_entries; ++i) {
        entries_host(i) = crs_entries[i];
      }
      Kokkos::deep_copy(entries_dev, entries_host);
    }

    // Step 3: Run KokkosKernels distance-1 graph coloring.
    using HandleType = KokkosKernels::Experimental::KokkosKernelsHandle<
        size_type, ordinal_type, ScalarT, exec_space, memory_space, memory_space>;

    HandleType handle;
    handle.create_graph_coloring_handle(KokkosGraph::COLORING_DEFAULT);

    KokkosGraph::Experimental::graph_color(
        &handle, n_sources_, n_sources_, rowmap_dev, entries_dev);

    // Extract coloring results
    auto color_handle = handle.get_graph_coloring_handle();
    num_colors_ = static_cast<ordinal_type>(color_handle->get_num_colors());

    // Copy colors to our member view
    auto colors_result = color_handle->get_vertex_colors();
    colors_ = colors_view("source_colors", n_sources_);
    Kokkos::deep_copy(colors_, colors_result);

    handle.destroy_graph_coloring_handle();
  }
};

// ============================================================================
// TeamDuplicationAccumulator
// ============================================================================

/// @brief Thread-team local-duplication fallback accumulator.
///
/// For cases where no natural coloring exists or the coloring approach is
/// impractical, this strategy creates per-thread (or per-team) local copies
/// of the output buffer and reduces them sequentially to produce deterministic
/// results. The reduction order is fixed (team 0 first, then team 1, etc.)
/// so results are reproducible regardless of scheduling.
///
/// @tparam ExecSpace  Kokkos execution space.
/// @tparam ScalarT    Numeric type for accumulated values.
template <class ExecSpace, class ScalarT = Scalar>
class TeamDuplicationAccumulator {
 public:
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using ordinal_type = int;

  using value_view = Kokkos::View<ScalarT*, memory_space>;

  /// @brief Construct a team-duplication accumulator.
  ///
  /// @param n_targets       Number of target elements to accumulate into.
  /// @param n_duplicates    Number of duplicate buffers (typically = number of
  ///                        teams or hardware threads). If 0, auto-detect from
  ///                        the execution space concurrency.
  explicit TeamDuplicationAccumulator(ordinal_type n_targets,
                                      ordinal_type n_duplicates = 0)
      : n_targets_(n_targets) {
    if (n_duplicates <= 0) {
      n_duplicates_ = static_cast<ordinal_type>(exec_space().concurrency());
      // Clamp to a reasonable maximum to avoid excessive memory usage
      if (n_duplicates_ > 256) n_duplicates_ = 256;
      if (n_duplicates_ < 1) n_duplicates_ = 1;
    } else {
      n_duplicates_ = n_duplicates;
    }
    // Allocate the duplicate buffer: (n_targets, n_duplicates) in LayoutLeft
    duplicates_ = Kokkos::View<ScalarT**, Kokkos::LayoutLeft, memory_space>(
        "team_duplicates", n_targets_, n_duplicates_);
  }

  /// @brief Deterministic accumulation using thread-local duplicates.
  ///
  /// Each work item writes into its assigned duplicate buffer. After all work
  /// items complete, the duplicates are reduced into the output in a fixed
  /// order (duplicate 0, then 1, ...) ensuring determinism.
  ///
  /// @tparam ScatterFunctor  A functor with signature
  ///   `void operator()(ordinal_type work_item, Kokkos::View<ScalarT*, memory_space> local_output) const`
  ///   that scatters contributions from `work_item` into `local_output`.
  /// @param output      1-D View of size >= n_targets, result after reduction.
  /// @param n_work      Number of work items to scatter.
  /// @param scatter     The scatter functor.
  template <class ScatterFunctor>
  void accumulate(const value_view& output,
                  ordinal_type n_work,
                  ScatterFunctor scatter) const {
    // Zero all duplicate buffers
    Kokkos::deep_copy(duplicates_, ScalarT{0});

    const auto duplicates = duplicates_;
    const ordinal_type n_dup = n_duplicates_;
    const ordinal_type n_tgt = n_targets_;

    // Phase 1: Scatter into per-duplicate buffers.
    // Work items are assigned to duplicate slots round-robin (i % n_dup).
    // Within each slot, items are processed sequentially in ascending order
    // to ensure a fixed accumulation order (deterministic FP results).
    // Slots are independent so we iterate them sequentially for simplicity
    // (this is the "fallback" strategy when coloring isn't available).
    for (ordinal_type d = 0; d < n_dup; ++d) {
      auto local = Kokkos::subview(duplicates, Kokkos::ALL, d);
      for (ordinal_type i = d; i < n_work; i += n_dup) {
        scatter(i, local);
      }
    }
    Kokkos::fence("TeamDuplication::scatter_fence");

    // Phase 2: Deterministic reduction — sum duplicates into output in fixed order.
    // Sequential over duplicates, parallel over targets within each duplicate.
    for (ordinal_type d = 0; d < n_dup; ++d) {
      Kokkos::parallel_for(
          "TeamDuplication::reduce",
          Kokkos::RangePolicy<exec_space>(0, n_tgt),
          KOKKOS_LAMBDA(const ordinal_type t) {
            output(t) += duplicates(t, d);
          });
      Kokkos::fence("TeamDuplication::reduce_fence");
    }
  }

  /// @brief Get the number of duplicate buffers.
  ordinal_type num_duplicates() const { return n_duplicates_; }

 private:
  ordinal_type n_targets_;
  ordinal_type n_duplicates_;
  Kokkos::View<ScalarT**, Kokkos::LayoutLeft, memory_space> duplicates_;
};

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_ACCUMULATION_HPP
