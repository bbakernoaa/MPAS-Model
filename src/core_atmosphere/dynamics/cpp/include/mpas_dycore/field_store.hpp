#ifndef MPAS_DYCORE_FIELD_STORE_HPP
#define MPAS_DYCORE_FIELD_STORE_HPP

/// @file field_store.hpp
/// @brief Performance-portable field storage with hybrid memory ownership.
///
/// `Field_Store` is the central storage abstraction for the C++ dycore. It
/// implements a hybrid memory ownership model (Requirement 1):
///
/// - **Dycore-internal fields** (tendencies, acoustic substep temporaries,
///   scratch buffers, dissipation working arrays) are C++-owned Kokkos Views
///   allocated via `allocate()` in the active Execution_Space's memory space,
///   RAII-managed, and never exposed to Fortran (Req 1.3, 1.4, 1.10).
///
/// - **Prognostic state fields** (theta_m, rho_zz, u, w, scalars, density) and
///   **mesh geometry** are Fortran-allocated and wrapped via `wrap()` as Kokkos
///   DualViews. The host mirror IS the Fortran pointer (zero-copy on host); the
///   device side is a Kokkos-allocated copy (Req 1.9).
///
/// - Explicit `sync_to_device(field)` / `sync_to_host(field)` deep-copy between
///   sides (Req 1.11, 1.12).
///
/// All Views/DualViews enforce LayoutLeft (column-major) and use RKIND precision
/// via the `Scalar` template parameter (Req 1.6, 1.8).
///
/// Views are passed by value into kernels; their accessors are usable inside a
/// Kokkos parallel region on the active Execution_Space (Req 1.7).

#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace mpas {
namespace dycore {

/// Dimension/extent information for a stored field.
struct Extents {
  int n_inner = 0;  ///< inner (fastest-varying, leftmost in LayoutLeft) dimension
  int n_elem = 0;   ///< outer (second) dimension
};

/// @brief Performance-portable field storage with hybrid memory ownership.
///
/// @tparam ScalarT  the numeric precision type (bound to RKIND at build time).
/// @tparam ExecSpace  a Kokkos execution space (e.g. Kokkos::DefaultExecutionSpace).
template <class ScalarT, class ExecSpace>
class Field_Store {
 public:
  // ── Type aliases ──────────────────────────────────────────────────────────

  using scalar_type = ScalarT;
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using host_space = Kokkos::HostSpace;
  using layout = Kokkos::LayoutLeft;

  /// C++-owned 2-D View allocated in ExecSpace::memory_space.
  using view2d = Kokkos::View<ScalarT**, layout, memory_space>;

  /// DualView wrapping Fortran-owned memory: host mirror = Fortran pointer,
  /// device side = Kokkos-allocated copy.
  using dualview2d = Kokkos::DualView<ScalarT**, layout, ExecSpace>;

  // ── Construction / destruction ────────────────────────────────────────────

  Field_Store() = default;
  ~Field_Store() = default;

  // Non-copyable (owns resource maps), movable.
  Field_Store(const Field_Store&) = delete;
  Field_Store& operator=(const Field_Store&) = delete;
  Field_Store(Field_Store&&) noexcept = default;
  Field_Store& operator=(Field_Store&&) noexcept = default;

  // ── Public interface ──────────────────────────────────────────────────────

  /// Allocate a C++-owned View in the active memory_space (RAII: freed on
  /// destruction of this Field_Store or when the entry is removed).
  /// Used for dycore-internal fields (Req 1.3, 1.4, 1.10).
  ///
  /// @param name     unique field name.
  /// @param n_inner  inner (vertical/fastest) extent.
  /// @param n_elem   outer (horizontal/element) extent.
  /// @return         a View usable inside Kokkos parallel regions (Req 1.7).
  view2d allocate(const std::string& name, int n_inner, int n_elem) {
    view2d v(Kokkos::view_alloc(name, Kokkos::WithoutInitializing),
             n_inner, n_elem);
    // Zero-initialize for safety (matches Fortran default)
    Kokkos::deep_copy(v, ScalarT{0});

    FieldEntry entry;
    entry.extents = {n_inner, n_elem};
    entry.storage = OwnedField{v};
    fields_[name] = std::move(entry);
    return v;
  }

  /// Allocate a time-leveled C++-owned field. Each time level gets its own
  /// View in the active memory_space. Levels are 1-based (Req 1.2, 1.13).
  ///
  /// @param name       base field name.
  /// @param n_inner    inner extent.
  /// @param n_elem     outer extent.
  /// @param n_levels   number of time levels (typically 1 or 2).
  /// @return           the View for time level 1.
  view2d allocate(const std::string& name, int n_inner, int n_elem, int n_levels) {
    std::vector<view2d> levels;
    levels.reserve(n_levels);
    for (int tl = 0; tl < n_levels; ++tl) {
      std::string level_name = name + "_tl" + std::to_string(tl + 1);
      view2d v(Kokkos::view_alloc(level_name, Kokkos::WithoutInitializing),
               n_inner, n_elem);
      Kokkos::deep_copy(v, ScalarT{0});
      levels.push_back(v);
    }

    FieldEntry entry;
    entry.extents = {n_inner, n_elem};
    entry.storage = OwnedTimeLeveled{std::move(levels)};
    fields_[name] = std::move(entry);
    return std::get<OwnedTimeLeveled>(fields_[name].storage).levels[0];
  }

  /// Wrap Fortran-owned memory as a DualView: the host mirror IS the Fortran
  /// pointer (zero-copy on host), and the device side is a Kokkos-allocated
  /// copy (Req 1.9). The device copy is populated by an initial deep_copy.
  ///
  /// @param name         unique field name.
  /// @param fortran_ptr  non-null Fortran-owned pointer (lifetime managed externally).
  /// @param n_inner      inner extent.
  /// @param n_elem       outer extent.
  /// @return             DualView whose host side aliases @p fortran_ptr.
  dualview2d wrap(const std::string& name, ScalarT* fortran_ptr, int n_inner, int n_elem) {
    dualview2d dv = wrap_dualview(name, fortran_ptr, n_inner, n_elem);

    FieldEntry entry;
    entry.extents = {n_inner, n_elem};
    entry.storage = WrappedField{dv};
    fields_[name] = std::move(entry);
    return dv;
  }

  /// Wrap Fortran-owned memory with multiple time levels (Req 1.2, 1.9, 1.13).
  /// Each time level gets a DualView aliasing a different region of the Fortran
  /// allocation. Levels are 1-based.
  ///
  /// @param name         base field name.
  /// @param fortran_ptrs array of pointers, one per time level.
  /// @param n_inner      inner extent.
  /// @param n_elem       outer extent.
  /// @return             DualView for time level 1.
  dualview2d wrap(const std::string& name,
                  const std::vector<ScalarT*>& fortran_ptrs,
                  int n_inner, int n_elem) {
    std::vector<dualview2d> levels;
    levels.reserve(fortran_ptrs.size());
    for (std::size_t tl = 0; tl < fortran_ptrs.size(); ++tl) {
      std::string level_name = name + "_tl" + std::to_string(tl + 1);
      levels.push_back(wrap_dualview(level_name, fortran_ptrs[tl], n_inner, n_elem));
    }

    FieldEntry entry;
    entry.extents = {n_inner, n_elem};
    entry.storage = WrappedTimeLeveled{std::move(levels)};
    fields_[name] = std::move(entry);
    return std::get<WrappedTimeLeveled>(fields_[name].storage).levels[0];
  }

  /// Deep-copy the host mirror to the device copy of a DualView-wrapped field
  /// (Req 1.11). This ensures the device side reflects the latest Fortran-written
  /// host data. No-op for C++-owned fields (they live in device memory already).
  void sync_to_device(const std::string& name) {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
      throw std::runtime_error("Field_Store::sync_to_device: unknown field '" + name + "'");
    }
    std::visit(SyncToDeviceVisitor{}, it->second.storage);
  }

  /// Deep-copy the device side back to the host mirror (the Fortran pointer) of
  /// a DualView-wrapped field (Req 1.12). No-op for C++-owned fields.
  void sync_to_host(const std::string& name) {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
      throw std::runtime_error("Field_Store::sync_to_host: unknown field '" + name + "'");
    }
    std::visit(SyncToHostVisitor{}, it->second.storage);
  }

  /// Sync all DualView-wrapped fields host→device.
  void sync_all_to_device() {
    for (auto& [name, entry] : fields_) {
      std::visit(SyncToDeviceVisitor{}, entry.storage);
    }
  }

  /// Sync all DualView-wrapped fields device→host.
  void sync_all_to_host() {
    for (auto& [name, entry] : fields_) {
      std::visit(SyncToHostVisitor{}, entry.storage);
    }
  }

  /// Access a field's View at a specific time level (1-based). Works uniformly
  /// for both owned and DualView-wrapped fields (Req 1.2, 1.13).
  /// For single-level fields, time_level must be 1.
  /// Returns a device-side View usable in Kokkos parallel regions (Req 1.7).
  view2d level(const std::string& name, int time_level) const {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
      throw std::runtime_error("Field_Store::level: unknown field '" + name + "'");
    }
    return std::visit(LevelVisitor{time_level}, it->second.storage);
  }

  /// Query the preserved extents of a stored field (Req 1.5).
  Extents extents(const std::string& name) const {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
      throw std::runtime_error("Field_Store::extents: unknown field '" + name + "'");
    }
    return it->second.extents;
  }

  /// Check whether a field name exists in the store.
  bool has_field(const std::string& name) const {
    return fields_.find(name) != fields_.end();
  }

  /// Get the number of time levels for a field.
  int num_time_levels(const std::string& name) const {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
      throw std::runtime_error("Field_Store::num_time_levels: unknown field '" + name + "'");
    }
    return std::visit(NumLevelsVisitor{}, it->second.storage);
  }

 private:
  // ── Internal storage types ────────────────────────────────────────────────

  /// Single-level C++-owned View.
  struct OwnedField {
    view2d view;
  };

  /// Multi-level C++-owned Views (one per time level).
  struct OwnedTimeLeveled {
    std::vector<view2d> levels;
  };

  /// Single-level DualView wrapping Fortran memory.
  struct WrappedField {
    dualview2d dv;
  };

  /// Multi-level DualViews wrapping Fortran memory.
  struct WrappedTimeLeveled {
    std::vector<dualview2d> levels;
  };

  using StorageVariant = std::variant<OwnedField, OwnedTimeLeveled,
                                      WrappedField, WrappedTimeLeveled>;

  struct FieldEntry {
    Extents extents;
    StorageVariant storage;
  };

  // ── Visitor implementations ───────────────────────────────────────────────

  struct SyncToDeviceVisitor {
    void operator()(OwnedField& /*f*/) const { /* no-op: already in device memory */ }
    void operator()(OwnedTimeLeveled& /*f*/) const { /* no-op */ }
    void operator()(WrappedField& f) const {
      Kokkos::deep_copy(f.dv.view_device(), f.dv.view_host());
    }
    void operator()(WrappedTimeLeveled& f) const {
      for (auto& dv : f.levels) {
        Kokkos::deep_copy(dv.view_device(), dv.view_host());
      }
    }
  };

  struct SyncToHostVisitor {
    void operator()(OwnedField& /*f*/) const { /* no-op */ }
    void operator()(OwnedTimeLeveled& /*f*/) const { /* no-op */ }
    void operator()(WrappedField& f) const {
      Kokkos::deep_copy(f.dv.view_host(), f.dv.view_device());
    }
    void operator()(WrappedTimeLeveled& f) const {
      for (auto& dv : f.levels) {
        Kokkos::deep_copy(dv.view_host(), dv.view_device());
      }
    }
  };

  struct LevelVisitor {
    int time_level;  // 1-based

    view2d operator()(const OwnedField& f) const {
      if (time_level != 1) {
        throw std::runtime_error(
            "Field_Store::level: single-level owned field accessed with time_level != 1");
      }
      return f.view;
    }

    view2d operator()(const OwnedTimeLeveled& f) const {
      int idx = time_level - 1;
      if (idx < 0 || idx >= static_cast<int>(f.levels.size())) {
        throw std::runtime_error(
            "Field_Store::level: time_level out of range for owned field");
      }
      return f.levels[idx];
    }

    view2d operator()(const WrappedField& f) const {
      if (time_level != 1) {
        throw std::runtime_error(
            "Field_Store::level: single-level wrapped field accessed with time_level != 1");
      }
      return f.dv.view_device();
    }

    view2d operator()(const WrappedTimeLeveled& f) const {
      int idx = time_level - 1;
      if (idx < 0 || idx >= static_cast<int>(f.levels.size())) {
        throw std::runtime_error(
            "Field_Store::level: time_level out of range for wrapped field");
      }
      return f.levels[idx].view_device();
    }
  };

  struct NumLevelsVisitor {
    int operator()(const OwnedField& /*f*/) const { return 1; }
    int operator()(const OwnedTimeLeveled& f) const {
      return static_cast<int>(f.levels.size());
    }
    int operator()(const WrappedField& /*f*/) const { return 1; }
    int operator()(const WrappedTimeLeveled& f) const {
      return static_cast<int>(f.levels.size());
    }
  };

  // ── Internal helpers ──────────────────────────────────────────────────────

  /// Build a DualView whose host mirror aliases `fortran_ptr` (zero-copy on
  /// host) and whose device side is a Kokkos-allocated copy (Req 1.9).
  /// This is the same construction as the `field_store::wrap_dualview_2d` shim
  /// in mesh_data.hpp, now implemented here canonically.
  static dualview2d wrap_dualview(const std::string& name,
                                  ScalarT* fortran_ptr,
                                  int n_inner, int n_elem) {
    using t_host = typename dualview2d::t_host;
    using t_dev = typename dualview2d::t_dev;

    // Host mirror aliases the Fortran pointer directly (zero-copy on host).
    // We construct a host View over the raw pointer and bind it as the
    // DualView's host member.
    t_host host_mirror(fortran_ptr, n_inner, n_elem);

    // Device copy: Kokkos-allocated in ExecSpace::memory_space.
    t_dev device_copy(
        Kokkos::view_alloc(name, Kokkos::WithoutInitializing),
        n_inner, n_elem);

    // Initial deep-copy host→device so device reflects current Fortran state.
    Kokkos::deep_copy(device_copy, host_mirror);

    return dualview2d(device_copy, host_mirror);
  }

  // ── Data members ──────────────────────────────────────────────────────────
  std::unordered_map<std::string, FieldEntry> fields_;
};

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_FIELD_STORE_HPP
