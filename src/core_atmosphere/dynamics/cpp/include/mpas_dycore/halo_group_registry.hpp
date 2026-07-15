#ifndef MPAS_DYCORE_HALO_GROUP_REGISTRY_HPP
#define MPAS_DYCORE_HALO_GROUP_REGISTRY_HPP

/// @file halo_group_registry.hpp
/// @brief Compile-time registry of named halo exchange groups.
///
/// Defines the complete table of halo exchange groups used by initialization,
/// dynamics, and (when MPAS_PHYSICS_ENABLED) physics, matching the composition
/// of the Reference_Model (`mpas_atm_halos.F`).
///
/// Each group maps a name to an ordered list of field entries, where each entry
/// specifies the field name, time level, and set of halo layers to exchange.
///
/// The registry is a constexpr/initialization-time constant table (no runtime
/// I/O). It is queryable by group name.
///
/// Requirements: 11.2, 11.8, 11.9

#include <string>
#include <string_view>
#include <vector>

namespace mpas {
namespace dycore {

/// A single field entry within a halo exchange group.
struct HaloFieldEntry {
  std::string field_name;         ///< Name of the field (e.g. "theta_m", "u")
  int time_level;                 ///< Time level for exchange (1 or 2)
  std::vector<int> halo_layers;   ///< Set of halo layers to exchange (e.g. {1,2})
};

/// Definition of a named halo exchange group.
struct HaloGroupDefinition {
  std::string group_name;                  ///< Group name (e.g. "dynamics:exner")
  std::vector<HaloFieldEntry> fields;      ///< Ordered list of field entries
};

/// Returns the complete registry of halo exchange groups as defined by the
/// Reference_Model.
///
/// The returned vector includes all initialization and dynamics groups
/// unconditionally. Physics groups (`physics:blten`, `physics:cuten`) are
/// included only when the library was compiled with MPAS_PHYSICS_ENABLED.
///
/// The order of groups in the vector matches the construction order in the
/// Reference_Model's `atm_build_halo_groups` routine. The order of fields
/// within each group matches the Reference_Model's `add_field` call order.
inline std::vector<HaloGroupDefinition> build_halo_group_registry() {
  std::vector<HaloGroupDefinition> registry;
  registry.reserve(20);

  // ── Initialization groups ────────────────────────────────────────────────

  registry.push_back({"initialization:u", {
    {"u", 1, {1, 2, 3}},
  }});

  registry.push_back({"initialization:pv_edge,ru,rw", {
    {"pv_edge", 1, {1, 2, 3}},
    {"ru",      1, {1, 2, 3}},
    {"rw",      1, {1, 2}},
  }});

  // ── Dynamics groups ──────────────────────────────────────────────────────

  registry.push_back({"dynamics:theta_m,scalars,pressure_p,rtheta_p", {
    {"theta_m",    1, {1, 2}},
    {"scalars",    1, {1, 2}},
    {"pressure_p", 1, {1, 2}},
    {"rtheta_p",   1, {1, 2}},
  }});

  registry.push_back({"dynamics:rw_p,ru_p,rho_pp,rtheta_pp", {
    {"rw_p",      1, {1}},
    {"ru_p",      1, {2}},
    {"rho_pp",    1, {1, 2}},
    {"rtheta_pp", 1, {2}},
  }});

  registry.push_back({"dynamics:w,pv_edge,rho_edge", {
    {"w",        2, {1, 2}},
    {"pv_edge",  1, {1, 2}},
    {"rho_edge", 1, {1, 2}},
  }});

  registry.push_back({"dynamics:w,pv_edge,rho_edge,scalars", {
    {"w",        2, {1, 2}},
    {"pv_edge",  1, {1, 2}},
    {"rho_edge", 1, {1, 2}},
    {"scalars",  2, {1, 2}},
  }});

  registry.push_back({"dynamics:theta_m,pressure_p,rtheta_p", {
    {"theta_m",    2, {1, 2}},
    {"pressure_p", 1, {1, 2}},
    {"rtheta_p",   1, {1, 2}},
  }});

  registry.push_back({"dynamics:exner", {
    {"exner", 1, {1, 2}},
  }});

  registry.push_back({"dynamics:tend_u", {
    {"tend_u", 1, {1}},
  }});

  registry.push_back({"dynamics:rho_pp", {
    {"rho_pp", 1, {1}},
  }});

  registry.push_back({"dynamics:rtheta_pp", {
    {"rtheta_pp", 1, {1}},
  }});

  registry.push_back({"dynamics:u_123", {
    {"u", 2, {1, 2, 3}},
  }});

  registry.push_back({"dynamics:u_3", {
    {"u", 2, {3}},
  }});

  registry.push_back({"dynamics:scalars", {
    {"scalars", 2, {1, 2}},
  }});

  registry.push_back({"dynamics:scalars_old", {
    {"scalars", 1, {1, 2}},
  }});

  registry.push_back({"dynamics:w", {
    {"w", 2, {1, 2}},
  }});

  registry.push_back({"dynamics:scale", {
    {"scale", 1, {1, 2}},
  }});

  // ── Physics groups (conditional on MPAS_PHYSICS_ENABLED) ─────────────────

#ifdef MPAS_PHYSICS_ENABLED
  registry.push_back({"physics:blten", {
    {"rublten", 1, {1, 2}},
    {"rvblten", 1, {1, 2}},
  }});

  registry.push_back({"physics:cuten", {
    {"rucuten", 1, {1, 2}},
    {"rvcuten", 1, {1, 2}},
  }});
#endif

  return registry;
}

/// Look up a halo group definition by name.
///
/// @param registry The halo group registry (from `build_halo_group_registry()`).
/// @param group_name The name of the group to find.
/// @return Pointer to the group definition, or nullptr if not found.
inline const HaloGroupDefinition* find_halo_group(
    const std::vector<HaloGroupDefinition>& registry,
    std::string_view group_name) {
  for (const auto& group : registry) {
    if (group.group_name == group_name) {
      return &group;
    }
  }
  return nullptr;
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_HALO_GROUP_REGISTRY_HPP
