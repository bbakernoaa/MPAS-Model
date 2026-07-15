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
#include <halo/halo_plan.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mpas {
namespace dycore {

/// @brief Halo topology information for a single mesh element type
/// (cells, edges, or vertices).
///
/// Describes the send/receive neighbor relationships and element counts
/// for one element kind. This information originates from the MPAS partition
/// decomposition (`mpas_dmpar` module) and is passed to the C++ dycore
/// during initialization.
struct HaloNeighborInfo {
  int rank = 0;            ///< MPI rank of the neighbor.
  std::size_t count = 0;   ///< Number of elements to exchange with this neighbor.
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
  /// Looks up the group by name, iterates its field entries, and calls the
  /// Halo_Library's `exchange_blocking` for each field's View at the specified
  /// time level. When `gpu_aware_comm` is true and the runtime supports it,
  /// device-resident data is exchanged directly with no host copy (Req 11.5).
  /// Otherwise data is staged through host memory (Req 11.6).
  ///
  /// @param group_name  The name of the halo group to exchange (e.g.
  ///                    "dynamics:exner").
  ///
  /// @throws std::runtime_error if group_name is not found in the registry.
  ///
  /// Requirements: 11.3, 11.4, 11.5, 11.6
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

  /// The group registry: maps group name -> group definition for O(1) lookup.
  std::unordered_map<std::string, HaloGroupDefinition> groups_;

  /// @brief Convert HaloNeighborInfo to halo::Neighbor_Info.
  static std::vector<halo::Neighbor_Info> to_halo_neighbors(
      const std::vector<HaloNeighborInfo>& infos);

  /// @brief Build a Halo_Plan from neighbor info, returning nullptr if empty.
  std::unique_ptr<halo::Halo_Plan> build_plan(
      const std::vector<HaloNeighborInfo>& send,
      const std::vector<HaloNeighborInfo>& recv) const;
};

// ─── Inline / template implementation ────────────────────────────────────────

inline std::vector<halo::Neighbor_Info> Halo_Manager::to_halo_neighbors(
    const std::vector<HaloNeighborInfo>& infos) {
  std::vector<halo::Neighbor_Info> result;
  result.reserve(infos.size());
  for (const auto& info : infos) {
    result.push_back(halo::Neighbor_Info{info.rank, info.count});
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

  // For each field in the group, get the View at the specified time level
  // and perform a halo exchange via the Halo_Library (Req 11.3, 11.4).
  for (const auto& field_entry : group_def.fields) {
    if (!field_store.has_field(field_entry.field_name)) {
      // Field not yet registered in the store (may be allocated later or
      // conditionally). Skip silently — this allows Halo_Manager to be
      // constructed before all fields are populated.
      continue;
    }

    // Get the device-side View for this field at the specified time level.
    auto field_view = field_store.level(field_entry.field_name,
                                        field_entry.time_level);

    // Select the appropriate halo plan based on field identity.
    // Edge-based fields (u, ru, pv_edge, tend_u, etc.) use the edge plan;
    // cell-based fields use the cell plan. This mapping follows the
    // Reference_Model's element-type assignments.
    halo::Halo_Plan* plan = nullptr;
    const auto& name = field_entry.field_name;
    if (name == "u" || name == "ru" || name == "pv_edge" ||
        name == "tend_u" || name == "ru_p") {
      plan = edge_plan_.get();
    } else {
      plan = cell_plan_.get();
    }

    if (plan == nullptr) {
      // No topology for this element type — skip (single-process run or
      // topology not yet populated).
      continue;
    }

    if (gpu_aware_comm_) {
      // GPU-aware path (Req 11.5): exchange device-resident data directly.
      // The Halo_Library's exchange_blocking dispatches between GPU-direct
      // and host-staged paths based on its compile-time
      // requires_staging_v<ViewType> trait and its runtime
      // Environment::is_gpu_aware_mpi() probe. When Config says gpu_aware,
      // we pass the device View trusting the library to use the direct path.
      halo::exchange_blocking(*plan, field_view);
    } else {
      // Host-staged path (Req 11.6): sync device data to a host mirror,
      // perform the exchange on host, then copy the result back to device.
      auto host_mirror = Kokkos::create_mirror_view(
          Kokkos::HostSpace{}, field_view);
      Kokkos::deep_copy(host_mirror, field_view);
      halo::exchange_blocking(*plan, host_mirror);
      Kokkos::deep_copy(field_view, host_mirror);
    }
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
