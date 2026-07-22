#ifndef MPAS_DYCORE_HALO_MANAGER_HPP
#define MPAS_DYCORE_HALO_MANAGER_HPP

/// @file halo_manager.hpp
/// @brief RAII halo exchange manager wrapping the Halo_Library.
///
/// `Halo_Manager` creates all named halo exchange groups from the compile-time
/// registry at construction (RAII), performs named-group exchanges via the
/// Halo_Library, supports GPU-aware exchange of device-resident data, and
/// releases all resources in the destructor.
///
/// The actual halo exchange topology (neighbor ranks, element counts per
/// neighbor) is provided at construction via `HaloTopology`. When the
/// Time_Integrator wiring is complete, the Fortran-originated decomposition
/// info (MPAS `mpas_dmpar` structures) will populate this topology.
///
/// Requirements: 11.1, 11.3, 11.4, 11.5, 11.6, 11.7, 11.10

#include "mpas_dycore/config.hpp"
#include "mpas_dycore/field_store.hpp"
#include "mpas_dycore/halo_group_registry.hpp"
#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>
#include <halo/communicator.hpp>
#include <halo/environment.hpp>
#include <halo/exchange.hpp>
#include <halo/exchange_indexed.hpp>
#include <halo/halo_plan.hpp>
#include <halo/indexed_halo_plan.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mpas {
namespace dycore {

/// @brief Halo topology information for a single neighbor of one mesh element
/// type (cells, edges, or vertices).
///
/// Describes the send/receive relationship with a single neighbor rank,
/// organized by halo layer. `layers[l]` holds the 0-based local indices for
/// halo layer `l` (layer index `l` corresponds to MPAS halo layer `l + 1`).
/// For a send neighbor these are the owned indices to gather; for a receive
/// neighbor they are the halo indices to fill. A neighbor with empty layers
/// exchanges zero elements. This information originates from the MPAS partition
/// decomposition (`mpas_dmpar` module) and is passed to the C++ dycore
/// during initialization.
struct HaloNeighborInfo {
  int rank = 0;  ///< MPI rank of the neighbor.
  /// Per-layer 0-based local index lists. `layers[l]` is the ordered index
  /// list for halo layer `l`.
  std::vector<std::vector<std::size_t>> layers;

  /// @brief Total number of indices across all halo layers.
  [[nodiscard]] std::size_t total_indices() const noexcept {
    std::size_t total = 0;
    for (const auto& layer : layers) {
      total += layer.size();
    }
    return total;
  }
};

/// @brief Complete halo topology for constructing Halo_Plans.
///
/// Contains send/receive neighbor lists for each mesh element kind
/// (cells, edges, vertices). These are sourced from the MPAS partition
/// decomposition at `dycore_init` time.
struct HaloTopology {
  std::vector<HaloNeighborInfo> cell_send_neighbors;
  std::vector<HaloNeighborInfo> cell_recv_neighbors;
  std::vector<HaloNeighborInfo> edge_send_neighbors;
  std::vector<HaloNeighborInfo> edge_recv_neighbors;
  std::vector<HaloNeighborInfo> vertex_send_neighbors;
  std::vector<HaloNeighborInfo> vertex_recv_neighbors;
};

/// @brief Minimal Domain aggregate bundling Field_Store and communicator state.
///
/// Until the full Time_Integrator wiring task lands a complete Domain type,
/// this minimal definition allows Halo_Manager (and other compute modules)
/// to compile and be tested independently.
struct Domain {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using FieldStoreT = Field_Store<Scalar, ExecSpace>;

  /// Reference to the field store holding prognostic and internal fields.
  FieldStoreT& field_store;

  /// The MPI communicator used for halo exchanges (non-owning).
  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  /// Halo topology describing the partition's neighbor relationships.
  /// Populated during `dycore_init` from MPAS decomposition structures.
  HaloTopology topology{};

  Domain(FieldStoreT& fs, MPI_Comm comm = MPI_COMM_WORLD)
      : field_store(fs), mpi_comm(comm) {}

  Domain(FieldStoreT& fs, MPI_Comm comm, HaloTopology topo)
      : field_store(fs), mpi_comm(comm), topology(std::move(topo)) {}
};

/// @brief RAII halo exchange manager over the Halo_Library.
///
/// Creates all Halo_Groups from the registry at construction, performs
/// named-group exchanges via `halo::exchange_blocking`, and releases all
/// owned resources in the destructor.
///
/// Requirements: 11.1, 11.3, 11.4, 11.5, 11.6, 11.7, 11.10
class Halo_Manager {
 public:
  /// @brief Construct all halo groups from the registry (RAII).
  ///
  /// Iterates `build_halo_group_registry()` and stores each group definition.
  /// Creates a `halo::Communicator` wrapping a duplicate of the Domain's
  /// MPI communicator and pre-builds `halo::Halo_Plan` objects for the
  /// partition's halo exchange topology.
  ///
  /// @param domain  The domain holding Field_Store, MPI communicator, and
  ///                halo topology.
  /// @param config  The immutable configuration snapshot.
  ///
  /// @throws std::runtime_error if validate_method detects an invalid method.
  ///
  /// Requirements: 11.1, 11.2, 11.8, 11.9
  explicit Halo_Manager(Domain& domain, const Config& config);

  /// @brief Releases communication buffers and group resources (RAII).
  ///
  /// The destructor releases the owned `halo::Communicator` (which calls
  /// `MPI_Comm_free`), the cached `Halo_Plan` objects, and the group
  /// definition map.
  ///
  /// Requirement: 11.7
  ~Halo_Manager();

  // Non-copyable, movable.
  Halo_Manager(const Halo_Manager&) = delete;
  Halo_Manager& operator=(const Halo_Manager&) = delete;
  Halo_Manager(Halo_Manager&&) noexcept = default;
  Halo_Manager& operator=(Halo_Manager&&) noexcept = default;

  /// @brief Exchange the halo regions of every field in the named group.
  ///
  /// Looks up the group by name and, for each field entry, resolves the field's
  /// element kind to the matching indexed plan (Req 9.2), the entry's time
  /// level to the field View (Req 9.3), and the entry's MPAS 1-based
  /// `halo_layers` to a 0-based layer subset (Req 9.4), then drives a
  /// gather/communicate/scatter exchange via `halo::exchange_indexed`. The
  /// indexed exchange internally selects the GPU-aware direct path or the
  /// host-staged path based on the view's memory space and the runtime
  /// GPU-aware MPI probe.
  ///
  /// Fields absent from the field store are skipped so the group may be
  /// exchanged before all fields are populated (Req 9.6). An element kind whose
  /// plan is `nullptr` (single-rank / no neighbors) is likewise skipped.
  ///
  /// @param group_name  The name of the halo group to exchange (e.g.
  ///                    "dynamics:exner").
  ///
  /// @throws std::runtime_error if group_name is not found in the registry
  ///                            (Req 9.5).
  ///
  /// Requirements: 9.2, 9.3, 9.4, 9.5, 9.6
  void exchange(const std::string& group_name);

  /// @brief Validate the configured halo exchange method.
  ///
  /// If `config.halo_exchange_method` is not "direct" or "grouped", throws
  /// a `std::runtime_error` identifying the unrecognized method name.
  ///
  /// @param config  The configuration to validate.
  ///
  /// @throws std::runtime_error naming the invalid exchange method.
  ///
  /// Requirement: 11.10
  void validate_method(const Config& config) const;

  /// @brief Query whether a group name is registered.
  [[nodiscard]] bool has_group(const std::string& group_name) const;

  /// @brief Return the number of registered halo groups.
  [[nodiscard]] std::size_t num_groups() const;

  /// @brief Return the group definition for a given name (for testing).
  /// @throws std::runtime_error if not found.
  [[nodiscard]] const HaloGroupDefinition& group(
      const std::string& group_name) const;

  /// @brief Return the built Indexed_Halo_Plan for an element kind (for
  /// testing / verification).
  ///
  /// Returns the plan constructed from the marshalled topology for the given
  /// element kind, or `nullptr` when that element kind carried no send/recv
  /// neighbors at construction (e.g. a single-rank run). Production behavior is
  /// unaffected: this is a read-only const accessor over already-built state.
  ///
  /// Requirement: 9.1
  [[nodiscard]] const halo::Indexed_Halo_Plan* indexed_plan(
      halo::Element_Kind kind) const {
    return indexed_plan_for(kind);
  }

 private:
  /// Reference to the domain (non-owning; domain must outlive Halo_Manager).
  Domain* domain_ = nullptr;

  /// Cached GPU-aware communication flag from config.
  bool gpu_aware_comm_ = false;

  /// The owned HALO communicator (wraps a duplicate of the domain's MPI_Comm).
  /// Destroyed by the destructor, which calls MPI_Comm_free (Req 11.7).
  std::unique_ptr<halo::Communicator> communicator_;

  /// Pre-built halo plan for cell-based fields.
  std::unique_ptr<halo::Halo_Plan> cell_plan_;

  /// Pre-built halo plan for edge-based fields.
  std::unique_ptr<halo::Halo_Plan> edge_plan_;

  /// Pre-built indexed (gather/scatter) halo plan for cell-based fields.
  /// `nullptr` when the topology carries no cell send/recv neighbors
  /// (e.g. a single-rank run).
  std::unique_ptr<halo::Indexed_Halo_Plan> cell_indexed_plan_;

  /// Pre-built indexed halo plan for edge-based fields. `nullptr` when empty.
  std::unique_ptr<halo::Indexed_Halo_Plan> edge_indexed_plan_;

  /// Pre-built indexed halo plan for vertex-based fields. `nullptr` when empty.
  std::unique_ptr<halo::Indexed_Halo_Plan> vertex_indexed_plan_;

  /// The group registry: maps group name -> group definition for O(1) lookup.
  std::unordered_map<std::string, HaloGroupDefinition> groups_;

  /// @brief Convert HaloNeighborInfo to halo::Neighbor_Info.
  static std::vector<halo::Neighbor_Info> to_halo_neighbors(
      const std::vector<HaloNeighborInfo>& infos);

  /// @brief Build a Halo_Plan from neighbor info, returning nullptr if empty.
  std::unique_ptr<halo::Halo_Plan> build_plan(
      const std::vector<HaloNeighborInfo>& send,
      const std::vector<HaloNeighborInfo>& recv) const;

  /// @brief Convert HaloNeighborInfo list to halo::Indexed_Neighbor list,
  /// preserving per-layer index lists.
  static std::vector<halo::Indexed_Neighbor> to_indexed_neighbors(
      const std::vector<HaloNeighborInfo>& infos);

  /// @brief Build an Indexed_Halo_Plan for one element kind, returning nullptr
  /// when both the send and recv neighbor lists are empty (single-rank).
  std::unique_ptr<halo::Indexed_Halo_Plan> build_indexed_plan(
      halo::Element_Kind kind,
      const std::vector<HaloNeighborInfo>& send,
      const std::vector<HaloNeighborInfo>& recv) const;

  /// @brief Map a field name to the mesh Element_Kind it is defined on.
  ///
  /// Replaces the previous hard-coded name-based edge/cell split with an
  /// explicit table. Edge-based fields (normal velocities and edge-staggered
  /// quantities) return `edge`; vertex-based fields return `vertex`; all other
  /// fields default to `cell`.
  static halo::Element_Kind element_kind_of(std::string_view field_name);

  /// @brief Select the indexed plan matching an element kind (may be nullptr).
  [[nodiscard]] halo::Indexed_Halo_Plan* indexed_plan_for(
      halo::Element_Kind kind) const;
};

// ─── Inline / template implementation ────────────────────────────────────────

inline std::vector<halo::Neighbor_Info> Halo_Manager::to_halo_neighbors(
    const std::vector<HaloNeighborInfo>& infos) {
  std::vector<halo::Neighbor_Info> result;
  result.reserve(infos.size());
  for (const auto& info : infos) {
    result.push_back(halo::Neighbor_Info{info.rank, info.total_indices()});
  }
  return result;
}

inline std::unique_ptr<halo::Halo_Plan> Halo_Manager::build_plan(
    const std::vector<HaloNeighborInfo>& send,
    const std::vector<HaloNeighborInfo>& recv) const {
  if (send.empty() && recv.empty()) {
    return nullptr;
  }
  return std::make_unique<halo::Halo_Plan>(
      *communicator_,
      to_halo_neighbors(send),
      to_halo_neighbors(recv));
}

inline std::vector<halo::Indexed_Neighbor> Halo_Manager::to_indexed_neighbors(
    const std::vector<HaloNeighborInfo>& infos) {
  std::vector<halo::Indexed_Neighbor> result;
  result.reserve(infos.size());
  for (const auto& info : infos) {
    result.push_back(halo::Indexed_Neighbor{info.rank, info.layers});
  }
  return result;
}

inline std::unique_ptr<halo::Indexed_Halo_Plan> Halo_Manager::build_indexed_plan(
    halo::Element_Kind kind,
    const std::vector<HaloNeighborInfo>& send,
    const std::vector<HaloNeighborInfo>& recv) const {
  // Single-rank (or otherwise neighborless) topology: no plan needed.
  if (send.empty() && recv.empty()) {
    return nullptr;
  }
  return std::make_unique<halo::Indexed_Halo_Plan>(
      *communicator_,
      kind,
      to_indexed_neighbors(send),
      to_indexed_neighbors(recv));
}

inline halo::Element_Kind Halo_Manager::element_kind_of(
    std::string_view field_name) {
  // Edge-based fields: normal velocities and edge-staggered quantities.
  if (field_name == "u" || field_name == "ru" || field_name == "ru_p" ||
      field_name == "pv_edge" || field_name == "rho_edge" ||
      field_name == "tend_u") {
    return halo::Element_Kind::edge;
  }
  // Vertex-based fields: potential vorticity / circulation on dual cells.
  if (field_name == "pv_vertex" || field_name == "vorticity") {
    return halo::Element_Kind::vertex;
  }
  // All remaining prognostic/diagnostic fields live on cell centers.
  return halo::Element_Kind::cell;
}

inline halo::Indexed_Halo_Plan* Halo_Manager::indexed_plan_for(
    halo::Element_Kind kind) const {
  switch (kind) {
    case halo::Element_Kind::edge:
      return edge_indexed_plan_.get();
    case halo::Element_Kind::vertex:
      return vertex_indexed_plan_.get();
    case halo::Element_Kind::cell:
    case halo::Element_Kind::generic:
    default:
      return cell_indexed_plan_.get();
  }
}

inline Halo_Manager::Halo_Manager(Domain& domain, const Config& config)
    : domain_(&domain), gpu_aware_comm_(config.gpu_aware_comm) {
  // Validate the exchange method at construction time (Req 11.10).
  validate_method(config);

  // Initialize the Halo_Library environment if not already done.
  // This is idempotent per halo::Environment's std::once_flag.
  halo::Environment::initialize();

  // Duplicate the domain's communicator for HALO-internal use so that
  // Halo_Manager owns a communicator with independent lifetime (Req 11.7).
  MPI_Comm dup_comm = MPI_COMM_NULL;
  int rc = MPI_Comm_dup(domain.mpi_comm, &dup_comm);
  if (rc != MPI_SUCCESS) {
    throw std::runtime_error(
        "Halo_Manager: MPI_Comm_dup failed during construction");
  }
  communicator_ = std::make_unique<halo::Communicator>(dup_comm);

  // Build halo plans from the domain's topology.
  // Cell plan covers fields on cell elements; edge plan covers edge fields.
  cell_plan_ = build_plan(domain.topology.cell_send_neighbors,
                          domain.topology.cell_recv_neighbors);
  edge_plan_ = build_plan(domain.topology.edge_send_neighbors,
                          domain.topology.edge_recv_neighbors);

  // Build one indexed (gather/scatter) plan per element kind directly from the
  // per-neighbor, per-layer index lists in the topology (Req 9.1). Each plan is
  // left nullptr when its element kind has no send/recv neighbors, which is the
  // single-rank case.
  cell_indexed_plan_ =
      build_indexed_plan(halo::Element_Kind::cell,
                         domain.topology.cell_send_neighbors,
                         domain.topology.cell_recv_neighbors);
  edge_indexed_plan_ =
      build_indexed_plan(halo::Element_Kind::edge,
                         domain.topology.edge_send_neighbors,
                         domain.topology.edge_recv_neighbors);
  vertex_indexed_plan_ =
      build_indexed_plan(halo::Element_Kind::vertex,
                         domain.topology.vertex_send_neighbors,
                         domain.topology.vertex_recv_neighbors);

  // Build the full group registry and store each definition in the map.
  // This creates every initialization, dynamics, and (conditional) physics
  // group defined by the Reference_Model (Req 11.1, 11.2, 11.8, 11.9).
  auto registry = build_halo_group_registry();
  groups_.reserve(registry.size());
  for (auto& group_def : registry) {
    std::string name = group_def.group_name;
    groups_.emplace(std::move(name), std::move(group_def));
  }
}

inline Halo_Manager::~Halo_Manager() {
  // RAII: destroy plans before the communicator, since Halo_Plan holds a
  // non-owning pointer to the Communicator.
  cell_plan_.reset();
  edge_plan_.reset();
  // Indexed_Halo_Plan also holds a non-owning Communicator pointer; release
  // the indexed plans before the communicator as well.
  cell_indexed_plan_.reset();
  edge_indexed_plan_.reset();
  vertex_indexed_plan_.reset();
  // halo::Communicator's destructor calls MPI_Comm_free on the duplicated
  // communicator. The groups_ map is destroyed by its own destructor.
  // This satisfies Req 11.7: release communication buffers and group resources.
  communicator_.reset();
  groups_.clear();
}

inline void Halo_Manager::exchange(const std::string& group_name) {
  // Look up the group definition.
  auto it = groups_.find(group_name);
  if (it == groups_.end()) {
    throw std::runtime_error(
        "Halo_Manager::exchange: unknown group '" + group_name + "'");
  }

  const HaloGroupDefinition& group_def = it->second;
  auto& field_store = domain_->field_store;

  // For each field in the group, resolve its element-kind indexed plan, the
  // View at the requested time level, and the 0-based layer subset, then drive
  // an indexed gather/communicate/scatter exchange (Req 9.2, 9.3, 9.4).
  for (const auto& field_entry : group_def.fields) {
    if (!field_store.has_field(field_entry.field_name)) {
      // Field not yet registered in the store (may be allocated later or
      // conditionally). Skip silently so a group may be exchanged before all
      // fields are populated (Req 9.6).
      continue;
    }

    // Resolve the field's element kind and select the matching indexed plan
    // (Req 9.2).
    const halo::Element_Kind kind = element_kind_of(field_entry.field_name);
    halo::Indexed_Halo_Plan* plan = indexed_plan_for(kind);

    if (plan == nullptr) {
      // No topology for this element kind — skip (single-rank run or a plan
      // whose send/recv neighbor lists were empty at construction).
      continue;
    }

    // Get the device-side View for this field at the specified time level
    // (Req 9.3).
    auto field_view = field_store.level(field_entry.field_name,
                                        field_entry.time_level);

    // Convert the entry's MPAS 1-based halo layer numbers to the 0-based layer
    // indices that exchange_indexed expects (Req 9.4).
    std::vector<int> layer_subset;
    layer_subset.reserve(field_entry.halo_layers.size());
    for (int mpas_layer : field_entry.halo_layers) {
      layer_subset.push_back(mpas_layer - 1);
    }

    // Drive the indexed exchange. exchange_indexed internally selects the
    // GPU-aware direct path or the host-staged path from the view's memory
    // space and the runtime GPU-aware MPI probe, so no manual host mirroring
    // is needed here.
    halo::exchange_indexed(*plan, field_view,
                           std::span<const int>(layer_subset));
  }
}

inline void Halo_Manager::validate_method(const Config& config) const {
  const auto& method = config.halo_exchange_method;
  if (method != "direct" && method != "grouped") {
    throw std::runtime_error(
        "Halo_Manager: unrecognized halo exchange method '" + method +
        "'. Valid methods are 'direct' and 'grouped'.");
  }
}

inline bool Halo_Manager::has_group(const std::string& group_name) const {
  return groups_.find(group_name) != groups_.end();
}

inline std::size_t Halo_Manager::num_groups() const {
  return groups_.size();
}

inline const HaloGroupDefinition& Halo_Manager::group(
    const std::string& group_name) const {
  auto it = groups_.find(group_name);
  if (it == groups_.end()) {
    throw std::runtime_error(
        "Halo_Manager::group: unknown group '" + group_name + "'");
  }
  return it->second;
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_HALO_MANAGER_HPP
