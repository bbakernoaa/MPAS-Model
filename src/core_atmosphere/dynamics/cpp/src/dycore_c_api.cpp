/**
 * @file dycore_c_api.cpp
 * @brief Implementation of the C API for the MPAS C++ dycore.
 *
 * This file implements the three entry points (init/timestep/finalize) that the
 * Fortran atmosphere core driver calls via iso_c_binding.  It owns the Kokkos
 * lifecycle and maintains a static DycoreContext that holds the Field_Store,
 * MeshData, and Config for the duration of the simulation.
 *
 * Requirements: 13.1, 13.2, 13.5, 13.7, 13.8, 13.9, 1.9, 1.10, 1.11, 1.12
 */

#include "mpas_dycore/dycore_c_api.h"

#include "mpas_dycore/config.hpp"
#include "mpas_dycore/field_store.hpp"
#include "mpas_dycore/mesh_data.hpp"
#include "mpas_dycore/scalar.hpp"
#include "mpas_dycore/time_integrator_advance.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

namespace {

// ─── Type aliases ────────────────────────────────────────────────────────────
using Scalar = mpas::dycore::Scalar;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using FieldStore = mpas::dycore::Field_Store<Scalar, ExecSpace>;
using MeshDataT = mpas::dycore::MeshData<ExecSpace>;
using Config = mpas::dycore::Config;
using ConfigBuilder = mpas::dycore::ConfigBuilder;
using MeshDims = mpas::dycore::MeshDims;
using MeshRawPointers = mpas::dycore::MeshRawPointers;

/// Internal context holding all persistent dycore state between init and finalize.
/// Heap-allocated via unique_ptr so that destruction order is explicit and we can
/// guarantee the Field_Store is torn down before Kokkos::finalize.
struct DycoreContext {
  MeshDims dims{};
  int num_scalars = 0;
  MeshDataT mesh{};
  FieldStore state_store{};
  Config config;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  /// Names of prognostic state fields registered in state_store (for batch sync).
  std::vector<std::string> prognostic_field_names;

  DycoreContext(Config cfg) : config(std::move(cfg)) {}
};

/// The singleton context, created by dycore_init and destroyed by dycore_finalize.
static std::unique_ptr<DycoreContext> g_context;

}  // anonymous namespace

extern "C" {
  // We use __attribute__((weak)) to provide a safe mock implementation for C++-only unit tests
  // (which don't link against the full Fortran marshalling library).
  // The Fortran-defined mpas_cpp_get_pointer symbol will override this at runtime in full builds.
  #ifdef __GNUC__
  __attribute__((weak))
  #endif
  void* mpas_cpp_get_pointer(const char* pool_name, const char* var_name, int dim_num, int time_level) {
    static std::vector<double> s_mock_mem(100000, 1.0);
    (void)pool_name;
    (void)var_name;
    (void)dim_num;
    (void)time_level;
    return s_mock_mem.data();
  }
}

struct DynFieldMetadata {
  std::string var_name;
  std::string fortran_name; // Name in MPAS Fortran pools (handles mismatches)
  std::string pool_name;
  int dimensions;       
  int time_levels;      
  std::string extent_x; 
  std::string extent_y; 
};

const std::vector<DynFieldMetadata> G_DIAGNOSTIC_FIELDS = {
  {"exner",         "exner",        "diag", 2, 0, "nVertLevels",   "nCells"},
  {"exner_base",    "exner_base",   "diag", 2, 0, "nVertLevels",   "nCells"},
  {"pressure_p",    "pressure_p",   "diag", 2, 0, "nVertLevels",   "nCells"},
  {"rho_p",         "rho_p",        "diag", 2, 0, "nVertLevels",   "nCells"},
  {"rtheta_base",   "rtheta_base",  "diag", 2, 0, "nVertLevels",   "nCells"},
  {"rtheta_p",      "rtheta_p",     "diag", 2, 0, "nVertLevels",   "nCells"},
  {"ru",            "ru",           "diag", 2, 0, "nVertLevels",   "nEdges"},
  {"rw",            "rw",           "diag", 2, 0, "nVertLevels+1", "nCells"},
  {"h_edge",        "rho_edge",     "diag", 2, 0, "nVertLevels",   "nEdges"}, // mismatch handled
  {"v",             "v",            "diag", 2, 0, "nVertLevels",   "nEdges"},
  {"vorticity",     "vorticity",    "diag", 2, 0, "nVertLevels",   "nVertices"},
  {"divergence",    "divergence",   "diag", 2, 0, "nVertLevels",   "nCells"},
  {"ke",            "ke",           "diag", 2, 0, "nVertLevels",   "nCells"},
  {"pv_edge",       "pv_edge",      "diag", 2, 0, "nVertLevels",   "nEdges"},
  {"pv_vertex",     "pv_vertex",    "diag", 2, 0, "nVertLevels",   "nVertices"},
  {"pv_cell",       "pv_cell",      "diag", 2, 0, "nVertLevels",   "nCells"},
  {"gradPVn",       "gradPVn",      "diag", 2, 0, "nVertLevels",   "nEdges"},
  {"gradPVt",       "gradPVt",      "diag", 2, 0, "nVertLevels",   "nEdges"},
  {"tend_u",        "u",            "tend", 2, 0, "nVertLevels",   "nEdges"},
  {"tend_w",        "w",            "tend", 2, 0, "nVertLevels+1", "nCells"},
  {"tend_theta_m",  "theta_m",      "tend", 2, 0, "nVertLevels",   "nCells"}, // mismatch handled
  {"tend_rho",      "rho_zz",       "tend", 2, 0, "nVertLevels",   "nCells"},
  {"ruAvg",         "ruAvg",        "diag", 2, 0, "nVertLevels",   "nEdges"},
  {"wwAvg",         "wwAvg",        "diag", 2, 0, "nVertLevels+1", "nCells"}
};

int resolve_extent(const std::string& extent_name, int nCells, int nEdges, int nVertices, int nVertLevels) {
  if (extent_name == "nCells")        return nCells;
  if (extent_name == "nEdges")        return nEdges;
  if (extent_name == "nVertices")     return nVertices;
  if (extent_name == "nVertLevels")   return nVertLevels;
  if (extent_name == "nVertLevels+1") return nVertLevels + 1;
  throw std::runtime_error("C++ Dycore: Unknown extent: " + extent_name);
}

void wrap_all_diagnostic_fields(FieldStore& store, 
                                std::vector<std::string>& prognostic_field_names,
                                int nCells, int nEdges, int nVertices, int nVertLevels) {
  for (const auto& meta : G_DIAGNOSTIC_FIELDS) {
    int n_inner = resolve_extent(meta.extent_x, nCells, nEdges, nVertices, nVertLevels);
    int n_elem  = resolve_extent(meta.extent_y, nCells, nEdges, nVertices, nVertLevels);

    if (meta.time_levels > 0) {
      std::vector<Scalar*> ptrs;
      for (int tl = 1; tl <= meta.time_levels; ++tl) {
        void* ptr = mpas_cpp_get_pointer(meta.pool_name.c_str(), meta.fortran_name.c_str(), meta.dimensions, tl);
        if (!ptr) {
          throw std::runtime_error("C++ Dycore: Missing time-leveled field '" + meta.var_name + "' (TL=" + std::to_string(tl) + ") in pool '" + meta.pool_name + "'");
        }
        ptrs.push_back(static_cast<Scalar*>(ptr));
      }
      store.wrap(meta.var_name, ptrs, n_inner, n_elem);
    } else {
      void* ptr = mpas_cpp_get_pointer(meta.pool_name.c_str(), meta.fortran_name.c_str(), meta.dimensions, 0);
      if (!ptr) {
        throw std::runtime_error("C++ Dycore: Missing field '" + meta.var_name + "' in pool '" + meta.pool_name + "'");
      }
      store.wrap(meta.var_name, static_cast<Scalar*>(ptr), n_inner, n_elem);
    }
    prognostic_field_names.push_back(meta.var_name);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// C API Implementation
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

int dycore_init(
    /* Mesh dimensions */
    int nCells, int nEdges, int nVertices, int nVertLevels, int maxEdges,
    int num_scalars,
    /* Mesh geometry / connectivity pointers */
    int* cellsOnEdge, int* edgesOnCell, int* verticesOnEdge,
    int* nEdgesOnCell_ptr,
    double* dvEdge, double* dcEdge, double* areaCell,
    double* zgrid, double* zz, double* fzm, double* fzp,
    /* Prognostic state pointers (two time levels each) */
    double* u_tl1, double* u_tl2,
    double* w_tl1, double* w_tl2,
    double* theta_m_tl1, double* theta_m_tl2,
    double* rho_zz_tl1, double* rho_zz_tl2,
    double* scalars_tl1, double* scalars_tl2,
    /* Configuration */
    int time_integration_order, int number_of_sub_steps,
    int dynamics_split_steps,
    int config_monotonic, int config_scalar_advection,
    int config_apply_lbcs, int config_mix_full,
    int config_iau, int gpu_aware_comm,
    /* MPI */
    int mpi_comm_fortran) {

  // ── 0a. Dimension validation (Req 8.3) ───────────────────────────────────
  if (nCells <= 0 || nEdges <= 0 || nVertLevels <= 0 || maxEdges <= 0) {
    return 2;  // Invalid dimensions
  }

  // ── 0b. Null pointer checks for required arrays (Req 8.2) ────────────────
  if (cellsOnEdge == nullptr || edgesOnCell == nullptr ||
      verticesOnEdge == nullptr || nEdgesOnCell_ptr == nullptr ||
      dvEdge == nullptr || dcEdge == nullptr || areaCell == nullptr ||
      zgrid == nullptr || zz == nullptr || fzm == nullptr || fzp == nullptr ||
      u_tl1 == nullptr || u_tl2 == nullptr ||
      w_tl1 == nullptr || w_tl2 == nullptr ||
      theta_m_tl1 == nullptr || theta_m_tl2 == nullptr ||
      rho_zz_tl1 == nullptr || rho_zz_tl2 == nullptr ||
      scalars_tl1 == nullptr || scalars_tl2 == nullptr) {
    return 3;  // Null pointer for a required array
  }

  // ── 1. Initialize Kokkos (Req 13.5) ──────────────────────────────────────
  try {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
  } catch (...) {
    return 1;  // Kokkos initialization failed
  }

  // ── 1b. Convert Fortran MPI communicator to C handle ─────────────────────
  MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_fortran);

  // ── 2. Build Config from scalar parameters (Req 13.2) ────────────────────
  auto cfg = ConfigBuilder{}
      .time_integration_order(time_integration_order)
      .number_of_sub_steps(number_of_sub_steps)
      .dynamics_split_steps(dynamics_split_steps)
      .config_monotonic(config_monotonic != 0)
      .config_scalar_advection(config_scalar_advection != 0)
      .config_apply_lbcs(config_apply_lbcs != 0)
      .config_mix_full(config_mix_full != 0)
      .config_les_model("none")
      .config_les_surface("none")
      .config_iau(config_iau != 0)
      .gpu_aware_comm(gpu_aware_comm != 0)
      .halo_exchange_method("direct")
      .build();

  // ── 3. Allocate the context ───────────────────────────────────────────────
  g_context = std::make_unique<DycoreContext>(std::move(cfg));
  g_context->num_scalars = num_scalars;
  g_context->mpi_comm = mpi_comm;

  MeshDims& dims = g_context->dims;
  dims.nCells = nCells;
  dims.nEdges = nEdges;
  dims.nVertices = nVertices;
  dims.nVertLevels = nVertLevels;
  dims.maxEdges = maxEdges;

  // ── 4. Wrap mesh geometry via Field_Store::wrap and sync to device (Req 1.9, 13.2) ─
  // Mesh geometry is static: synced to device once during init.
  MeshRawPointers ptrs;
  ptrs.cellsOnEdge = cellsOnEdge;
  ptrs.edgesOnCell = edgesOnCell;
  ptrs.verticesOnEdge = verticesOnEdge;
  ptrs.nEdgesOnCell = nEdgesOnCell_ptr;
  ptrs.dvEdge = dvEdge;
  ptrs.dcEdge = dcEdge;
  ptrs.areaCell = areaCell;
  ptrs.zgrid = zgrid;
  ptrs.zz = zz;
  ptrs.fzm = fzm;
  ptrs.fzp = fzp;

  g_context->mesh = MeshDataT::wrap(dims, ptrs);
  // MeshData::wrap already performs a deep_copy host→device at construction time
  // for each field, satisfying the "sync_to_device on mesh geometry once" contract.

  // ── 5. Wrap prognostic state fields as DualViews (Req 1.9) ────────────────
  // Two time levels per prognostic variable, wrapped via Field_Store::wrap.
  auto& store = g_context->state_store;
  auto& names = g_context->prognostic_field_names;

  // u: (nVertLevels, nEdges) x 2 time levels
  {
    std::vector<Scalar*> u_ptrs = {u_tl1, u_tl2};
    store.wrap("u", u_ptrs, nVertLevels, nEdges);
    names.push_back("u");
  }

  // w: (nVertLevels+1, nCells) x 2 time levels
  {
    std::vector<Scalar*> w_ptrs = {w_tl1, w_tl2};
    store.wrap("w", w_ptrs, nVertLevels + 1, nCells);
    names.push_back("w");
  }

  // theta_m: (nVertLevels, nCells) x 2 time levels
  {
    std::vector<Scalar*> theta_ptrs = {theta_m_tl1, theta_m_tl2};
    store.wrap("theta_m", theta_ptrs, nVertLevels, nCells);
    names.push_back("theta_m");
  }

  // rho_zz: (nVertLevels, nCells) x 2 time levels
  {
    std::vector<Scalar*> rho_ptrs = {rho_zz_tl1, rho_zz_tl2};
    store.wrap("rho_zz", rho_ptrs, nVertLevels, nCells);
    names.push_back("rho_zz");
  }

  // scalars: (num_scalars * nVertLevels, nCells) x 2 time levels
  // Note: scalars in MPAS are (num_scalars, nVertLevels, nCells) in Fortran,
  // but we treat the first two dims as a single inner dimension for the 2-D
  // DualView wrapping, matching the flat memory layout.
  {
    std::vector<Scalar*> sc_ptrs = {scalars_tl1, scalars_tl2};
    store.wrap("scalars", sc_ptrs, num_scalars * nVertLevels, nCells);
    names.push_back("scalars");
  }

  // Wrap all dynamic diagnostics and tendency fields from MPAS pools
  wrap_all_diagnostic_fields(store, names, nCells, nEdges, nVertices, nVertLevels);

  return 0;  // Success
}

void dycore_timestep(double dt, int itimestep) {
  if (!g_context) {
    return;  // dycore_init was not called; no-op.
  }

  auto& store = g_context->state_store;
  const auto& names = g_context->prognostic_field_names;

  // ── Timestep entry: sync_to_device on all prognostic state fields (Req 13.7, 1.11) ─
  for (const auto& name : names) {
    store.sync_to_device(name);
  }

  // ── Execute dycore kernels ────────────────────────────────────────────────
  mpas::dycore::AdvanceDomain domain{
      .config = g_context->config,
      .nCells = g_context->dims.nCells,
      .nEdges = g_context->dims.nEdges,
      .nVertices = g_context->dims.nVertices,
      .nVertLevels = g_context->dims.nVertLevels,
      .nCellsSolve = g_context->dims.nCells,
      .nEdgesSolve = g_context->dims.nEdges,
      .num_scalars = g_context->num_scalars,
      .maxEdges = g_context->dims.maxEdges,
      .itimestep = itimestep,
      .dt = dt,
      .halo_manager = nullptr,
      .scalar_advection_enabled = g_context->config.config_scalar_advection,
      .split_dynamics_transport = false,
      .field_store = &store
  };

  mpas::dycore::Time_Integrator_Advance integrator;
  integrator.advance(domain);

  // ── Timestep exit: sync_to_host on all modified prognostic state fields (Req 13.8, 1.12) ─
  // All prognostic state fields are considered potentially modified by the dycore.
  for (const auto& name : names) {
    store.sync_to_host(name);
  }
}

void dycore_finalize(void) {
  // Destroy the context before finalizing Kokkos, so that all Kokkos Views
  // are released while the runtime is still active (Req 13.5).
  g_context.reset();

  if (Kokkos::is_initialized()) {
    Kokkos::finalize();
  }
}

}  // extern "C"
