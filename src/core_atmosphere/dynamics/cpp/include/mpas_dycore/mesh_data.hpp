#ifndef MPAS_DYCORE_MESH_DATA_HPP
#define MPAS_DYCORE_MESH_DATA_HPP

/// @file mesh_data.hpp
/// @brief Read-only mesh connectivity and geometry for the C++ dycore.
///
/// `MeshData` is the read-only view of the unstructured MPAS mesh consumed by
/// every ported compute module. Its members are `Kokkos::DualView`s created via
/// `Field_Store::wrap()` (Requirement 13.2, 1.9): the host mirror IS the
/// Fortran-owned pointer (zero-copy on host) and the device side is a
/// Kokkos-allocated copy in the active `ExecSpace::memory_space`. Mesh geometry
/// is synced to device exactly once, during `dycore_init`, and is treated as
/// static thereafter — no compute module writes through a `MeshData` view and
/// no per-timestep sync is issued for it (design: "Kokkos Execution and Memory
/// Strategy / Memory spaces and layout").
///
/// All Views use `Kokkos::LayoutLeft` (column-major) storage, which matches the
/// Fortran declaration order exactly: an MPAS array declared
/// `cellsOnEdge(2, nEdges)` has its leftmost index varying fastest in memory,
/// precisely how a LayoutLeft View indexes. Wrapping the raw Fortran pointer as
/// the host mirror of a LayoutLeft DualView therefore indexes identically to
/// the Fortran array, with no data duplication on the host side
/// (Requirement 1.8, 1.9).
///
/// The connectivity/geometry arrays and their Reference_Model shapes are:
///
///   | Field           | Shape (LayoutLeft)         | Meaning                        |
///   |-----------------|----------------------------|--------------------------------|
///   | cellsOnEdge     | (2, nEdges)                | cell neighbors of each edge    |
///   | edgesOnCell     | (maxEdges, nCells)         | edges around each cell         |
///   | verticesOnEdge  | (2, nEdges)                | vertices bounding each edge    |
///   | nEdgesOnCell    | (nCells)                   | valid edge count per cell      |
///   | dvEdge          | (nEdges)                   | edge length (dual/Voronoi)     |
///   | dcEdge          | (nEdges)                   | distance between cell centers  |
///   | areaCell        | (nCells)                   | cell area                      |
///   | zgrid           | (nVertLevels+1, nCells)    | height of layer interfaces     |
///   | zz              | (nVertLevels, nCells)      | d(zeta)/dz metric              |
///   | fzm, fzp        | (nVertLevels, nCells)      | vertical interpolation weights |
///
/// Connectivity indices carry the Fortran 1-based values verbatim; the raw
/// pointers are not renumbered here. Modules that index into a View using a
/// connectivity value are responsible for the 1-based to 0-based conversion at
/// the point of use, so that `MeshData` stays a faithful, zero-copy (on host)
/// mirror of the Reference_Model arrays.
///
/// @note Field_Store (task 2.3) is the general-purpose storage abstraction
/// that owns the `allocate()`/`wrap()`/`sync_to_device()`/`sync_to_host()`
/// contract described in the design document's "Field_Store" section. Because
/// Field_Store has not landed yet at the time this header was authored,
/// `MeshData::wrap()` is written against that documented interface contract
/// (a free/static function `wrap(name, fortran_ptr, extents...) -> DualView`
/// returning a `Kokkos::DualView<T*/**, Kokkos::LayoutLeft, ExecSpace>` whose
/// host side aliases `fortran_ptr`) via the forward-declared
/// `mpas::dycore::field_store` namespace below. When Field_Store is
/// implemented, `MPAS_DYCORE_MESH_DATA_USE_FIELD_STORE` should be left defined
/// (the default) so `MeshData::wrap` calls straight into it; the local
/// `field_store::wrap_dualview` shim can then be deleted. See
/// `docs: dependency on Field_Store (task 2.3)` at the bottom of this file.

#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include <string>
#include <type_traits>

namespace mpas {
namespace dycore {

/// Integer type used for MPAS mesh connectivity indices and per-element counts.
/// MPAS declares these as the default Fortran `integer` (32-bit), so a 32-bit
/// signed integer wraps the same memory without reinterpretation.
using MeshIndex = int;

/// Extents describing an MPAS mesh partition. These are the per-build/per-run
/// dimensions sourced from the MPAS pools (Requirement 13.4). Extents are
/// stored so consumers can range Kokkos parallel constructs over the correct
/// element counts without re-deriving them from View shapes.
struct MeshDims {
  int nCells = 0;        ///< number of cells (owned + halo) in the partition
  int nEdges = 0;        ///< number of edges (owned + halo) in the partition
  int nVertices = 0;     ///< number of vertices (owned + halo) in the partition
  int nVertLevels = 0;   ///< number of vertical layers (mid-levels)
  int maxEdges = 0;      ///< maximum number of edges on any cell (fixed inner dim)
};

/// Raw, Fortran-owned pointers to the mesh connectivity/geometry arrays. This
/// is the plain-old-data hand-off across the C ABI: the interop layer fills it
/// from the MPAS pools and passes it to `MeshData::wrap`. Pointers are
/// non-owning; the Fortran side retains ownership and lifetime responsibility
/// for the underlying memory (the DualView host mirror aliases it, it is never
/// freed by `MeshData`/`Field_Store`).
struct MeshRawPointers {
  MeshIndex* cellsOnEdge = nullptr;     ///< (2, nEdges)
  MeshIndex* edgesOnCell = nullptr;     ///< (maxEdges, nCells)
  MeshIndex* verticesOnEdge = nullptr;  ///< (2, nEdges)
  MeshIndex* nEdgesOnCell = nullptr;    ///< (nCells)
  Scalar* dvEdge = nullptr;             ///< (nEdges)
  Scalar* dcEdge = nullptr;             ///< (nEdges)
  Scalar* areaCell = nullptr;           ///< (nCells)
  Scalar* zgrid = nullptr;              ///< (nVertLevels+1, nCells)
  Scalar* zz = nullptr;                 ///< (nVertLevels, nCells)
  Scalar* fzm = nullptr;                ///< (nVertLevels, nCells)
  Scalar* fzp = nullptr;                ///< (nVertLevels, nCells)
};

// ─────────────────────────────────────────────────────────────────────────────
// Field_Store::wrap() interface contract (task 2.3 dependency shim)
// ─────────────────────────────────────────────────────────────────────────────
//
// Field_Store (task 2.3) is designed to expose:
//
//   template <class Scalar, class ExecSpace>
//   class Field_Store {
//    public:
//     using dualview1d = Kokkos::DualView<Scalar*,  Kokkos::LayoutLeft, ExecSpace>;
//     using dualview2d = Kokkos::DualView<Scalar**, Kokkos::LayoutLeft, ExecSpace>;
//     dualview1d wrap(Scalar* fortran_ptr, int n_elem);
//     dualview2d wrap(Scalar* fortran_ptr, int n_inner, int n_elem);
//     ...
//   };
//
// i.e. `wrap()` returns a `Kokkos::DualView` whose host mirror IS the
// `fortran_ptr` passed in (zero-copy on host) and whose device side is a
// Kokkos-allocated copy (Requirement 1.9). `MeshData::wrap` below is written
// directly against that contract, so once task 2.3 lands, replace the
// `field_store::wrap_dualview` shim namespace with a real `Field_Store`
// instance (or keep this header decoupled from a concrete `Field_Store`
// instantiation by leaving the shim, which forwards to the exact same
// DualView construction Field_Store::wrap performs). Either way, no change to
// MeshData's public members or call sites is required.
namespace field_store {

/// Build a `Kokkos::DualView` whose host mirror aliases `fortran_ptr` (an
/// Unmanaged host View over the raw pointer) and whose device side is a
/// Kokkos-allocated copy in `ExecSpace::memory_space`, matching the exact
/// contract of `Field_Store::wrap()` (Requirement 1.9, 1.5, 1.8).
///
/// This is a standalone helper (not a `Field_Store` member) so `MeshData` can
/// be implemented and compiled independently of task 2.3. It performs the
/// same `Kokkos::DualView(t_dev, t_host)` construction that `Field_Store::wrap`
/// is designed to perform internally.
template <class T, class ExecSpace, class... Extents>
inline Kokkos::DualView<T*, Kokkos::LayoutLeft, ExecSpace> wrap_dualview_1d(
    T* fortran_ptr, Extents... extents) {
  using dualview = Kokkos::DualView<T*, Kokkos::LayoutLeft, ExecSpace>;
  using t_host = typename dualview::t_host;
  using t_dev = typename dualview::t_dev;

  // Host mirror aliases the Fortran pointer directly (zero-copy on host).
  // t_host is itself managed-View-typed by DualView's traits; we construct an
  // Unmanaged host view over fortran_ptr and let the DualView host member bind
  // to it by value (DualView's h_view stores a t_host, which for a managed
  // DualView type is a *managed* View type — to keep the alias truly
  // zero-copy we instead build the pair-constructor path below via the
  // Unmanaged host View cast, mirroring Field_Store::wrap's documented
  // zero-copy contract).
  Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_unmanaged(fortran_ptr, extents...);
  t_host host_mirror(host_unmanaged.data(), extents...);

  t_dev device_copy(
      Kokkos::view_alloc(std::string("mesh_field"), Kokkos::WithoutInitializing),
      extents...);
  Kokkos::deep_copy(device_copy, host_mirror);

  return dualview(device_copy, host_mirror);
}

template <class T, class ExecSpace, class... Extents>
inline Kokkos::DualView<T**, Kokkos::LayoutLeft, ExecSpace> wrap_dualview_2d(
    T* fortran_ptr, Extents... extents) {
  using dualview = Kokkos::DualView<T**, Kokkos::LayoutLeft, ExecSpace>;
  using t_host = typename dualview::t_host;
  using t_dev = typename dualview::t_dev;

  Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_unmanaged(fortran_ptr, extents...);
  t_host host_mirror(host_unmanaged.data(), extents...);

  t_dev device_copy(
      Kokkos::view_alloc(std::string("mesh_field"), Kokkos::WithoutInitializing),
      extents...);
  Kokkos::deep_copy(device_copy, host_mirror);

  return dualview(device_copy, host_mirror);
}

}  // namespace field_store

/// Read-only mesh connectivity and geometry as `Kokkos::DualView`s.
///
/// `MeshData` is templated on the Kokkos `ExecSpace` so the same struct serves
/// host and device execution; the device sides of the DualViews live in
/// `ExecSpace::memory_space` (Requirement 13.2). Every member is created via
/// `Field_Store::wrap()` (or, until task 2.3 lands, the equivalent
/// `field_store::wrap_dualview_*` shim above) so the host mirror IS the
/// Fortran-owned pointer and the device side is a Kokkos-allocated copy
/// (Requirement 1.9). `MeshData::wrap` calls `sync_to_device` (via a
/// `deep_copy` at construction time, matching the "synced once during
/// dycore_init" design note) so the device copy is valid immediately; no
/// further sync is performed because mesh geometry is static thereafter.
///
/// @tparam ExecSpace  a Kokkos execution space (defaults to the host space).
template <class ExecSpace = Kokkos::DefaultHostExecutionSpace>
struct MeshData {
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutLeft;

  /// DualView aliases matching the `Field_Store::wrap()` contract
  /// (Req 1.5, 1.8, 1.9): host mirror = Fortran pointer, device = Kokkos copy.
  template <class T>
  using DualView1D = Kokkos::DualView<T*, layout, ExecSpace>;
  template <class T>
  using DualView2D = Kokkos::DualView<T**, layout, ExecSpace>;

  // ── Dimensions (Requirement 13.4) ──────────────────────────────────────────
  MeshDims dims{};

  // ── Connectivity (Requirement 13.2, 1.5) ────────────────────────────────────
  DualView2D<MeshIndex> cellsOnEdge;     ///< (2, nEdges)
  DualView2D<MeshIndex> edgesOnCell;     ///< (maxEdges, nCells)
  DualView2D<MeshIndex> verticesOnEdge;  ///< (2, nEdges)
  DualView1D<MeshIndex> nEdgesOnCell;    ///< (nCells)

  // ── Horizontal geometry (Requirement 13.2, 1.5) ─────────────────────────────
  DualView1D<Scalar> dvEdge;    ///< (nEdges)
  DualView1D<Scalar> dcEdge;    ///< (nEdges)
  DualView1D<Scalar> areaCell;  ///< (nCells)

  // ── Vertical geometry / weights (Requirement 13.2, 1.5) ──────────────────────
  DualView2D<Scalar> zgrid;  ///< (nVertLevels+1, nCells)
  DualView2D<Scalar> zz;     ///< (nVertLevels, nCells)
  DualView2D<Scalar> fzm;    ///< (nVertLevels, nCells)
  DualView2D<Scalar> fzp;    ///< (nVertLevels, nCells)

  MeshData() = default;

  /// Build a `MeshData` by wrapping Fortran-owned arrays via
  /// `Field_Store::wrap()` (Requirement 13.2). Extents come from @p d and
  /// match the Reference_Model shapes exactly; each DualView's host mirror
  /// aliases the corresponding raw pointer in @p p and its device side is
  /// synced once here, matching "Mesh geometry is synced to device once
  /// during dycore_init and treated as static thereafter."
  ///
  /// @param d  mesh extents (nCells, nEdges, nVertices, nVertLevels, maxEdges).
  /// @param p  raw Fortran-owned array pointers.
  /// @return   a fully populated read-only `MeshData`, device-synced.
  static MeshData wrap(const MeshDims& d, const MeshRawPointers& p) {
    MeshData m;
    m.dims = d;

    m.cellsOnEdge =
        field_store::wrap_dualview_2d<MeshIndex, ExecSpace>(p.cellsOnEdge, 2, d.nEdges);
    m.edgesOnCell = field_store::wrap_dualview_2d<MeshIndex, ExecSpace>(
        p.edgesOnCell, d.maxEdges, d.nCells);
    m.verticesOnEdge = field_store::wrap_dualview_2d<MeshIndex, ExecSpace>(
        p.verticesOnEdge, 2, d.nEdges);
    m.nEdgesOnCell =
        field_store::wrap_dualview_1d<MeshIndex, ExecSpace>(p.nEdgesOnCell, d.nCells);

    m.dvEdge = field_store::wrap_dualview_1d<Scalar, ExecSpace>(p.dvEdge, d.nEdges);
    m.dcEdge = field_store::wrap_dualview_1d<Scalar, ExecSpace>(p.dcEdge, d.nEdges);
    m.areaCell = field_store::wrap_dualview_1d<Scalar, ExecSpace>(p.areaCell, d.nCells);

    m.zgrid = field_store::wrap_dualview_2d<Scalar, ExecSpace>(p.zgrid, d.nVertLevels + 1,
                                                                d.nCells);
    m.zz = field_store::wrap_dualview_2d<Scalar, ExecSpace>(p.zz, d.nVertLevels, d.nCells);
    m.fzm = field_store::wrap_dualview_2d<Scalar, ExecSpace>(p.fzm, d.nVertLevels, d.nCells);
    m.fzp = field_store::wrap_dualview_2d<Scalar, ExecSpace>(p.fzp, d.nVertLevels, d.nCells);

    return m;
  }

  /// Device-side accessor for a 2-D field, usable inside a Kokkos parallel
  /// region on `ExecSpace` (Requirement 1.7). Returns the DualView's device
  /// View; callers on host execution spaces where device == host get the same
  /// underlying data with no extra copy.
  template <class T>
  static KOKKOS_INLINE_FUNCTION typename DualView2D<T>::t_dev device_view(
      const DualView2D<T>& dv) {
    return dv.view_device();
  }

  template <class T>
  static KOKKOS_INLINE_FUNCTION typename DualView1D<T>::t_dev device_view(
      const DualView1D<T>& dv) {
    return dv.view_device();
  }
};

}  // namespace dycore
}  // namespace mpas

// ─────────────────────────────────────────────────────────────────────────────
// docs: dependency on Field_Store (task 2.3)
// ─────────────────────────────────────────────────────────────────────────────
// This header implements MeshData against the documented Field_Store::wrap()
// contract (design.md, "Field_Store" component) without depending on the
// Field_Store class itself, because task 2.3 has not been implemented yet.
// The `mpas::dycore::field_store::wrap_dualview_1d/2d` helpers above perform
// exactly the DualView(t_dev, t_host) construction that Field_Store::wrap is
// specified to do: host mirror aliases the Fortran pointer, device side is a
// Kokkos-allocated copy, populated by one deep_copy at wrap time.
//
// When task 2.3 lands, the recommended follow-up (tracked as a note for
// whoever implements 2.3, not a new task) is:
//   1. Give Field_Store<Scalar, ExecSpace> a `wrap(ptr, extents...)` overload
//      set matching the shim signatures here.
//   2. Point `field_store::wrap_dualview_1d/2d` at `Field_Store::wrap` (or
//      have MeshData take a `Field_Store&` and call it directly), so there is
//      a single implementation of the wrap contract.
// No change to MeshData's public members, `MeshDims`, `MeshRawPointers`, or
// `MeshData::wrap`'s signature is anticipated from that follow-up.
// ─────────────────────────────────────────────────────────────────────────────

#endif  // MPAS_DYCORE_MESH_DATA_HPP
