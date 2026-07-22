/**
 * @file test_dycore_init_wiring_smoke.cpp
 * @brief Smoke test for the production halo-exchange wiring in the dycore C API.
 *
 * Feature: cpp-dycore-halo-exchange, Task 8.3 (production wiring smoke test)
 *
 * Validates: Requirements 7.5, 10.1, 10.2
 *
 * What this test observes
 * -----------------------
 * Tasks 8.1 and 8.2 made `dycore_init` decode the marshalled per-element-kind,
 * per-direction CSR halo arrays into a `HaloTopology`, construct a persistent
 * `Halo_Manager` from that topology and the field store, and hold it in the
 * file-static `DycoreContext`. `dycore_timestep` then builds the `AdvanceDomain`
 * with `.halo_manager = g_context->halo_manager.get()`, replacing the old
 * `nullptr` that made every guarded exchange call site a no-op.
 *
 * Because the context is internal to `dycore_c_api.cpp`, the wiring is observed
 * through two seams that keep production behavior unchanged:
 *   1. The public C API return code: `dycore_init` returns 0 (Req 7.5, 10.1).
 *   2. A read-only test hook, `dycore_test_halo_manager_is_set()`, which reports
 *      whether the context holds a non-null `Halo_Manager`. Since the timestep
 *      path assigns that same pointer to `AdvanceDomain.halo_manager`, a
 *      non-zero result is exactly the condition Req 10.2 requires.
 * A successful `dycore_timestep` call then confirms the wired path (with a
 * non-null manager) runs without error.
 *
 * Topology
 * --------
 * The test runs as a single MPI rank (the default under ctest), so the only
 * valid synthetic topology is the single-rank case: every marshalled direction
 * carries zero neighbors. The `Halo_Manager` is still constructed (non-null),
 * with each element kind's indexed plan left null (no neighbors) — the correct
 * single-rank behavior. The CSR arrays are passed through the C API exactly as
 * the marshaller would deliver them for a one-rank decomposition, exercising the
 * acceptance and decode of the marshalled arrays (Req 7.5, 10.1).
 *
 * Lifecycle
 * ---------
 * `main` initializes MPI (required for the manager's `MPI_Comm_dup`) but does
 * NOT touch Kokkos: `dycore_init` initializes Kokkos and `dycore_finalize`
 * finalizes it. Kokkos 4.x forbids re-initialization after finalize within one
 * process, so the whole init/timestep/finalize cycle runs exactly once.
 */

#include <gtest/gtest.h>

#include <mpi.h>

#include "mpas_dycore/dycore_c_api.h"

#include <vector>

// Read-only introspection hook defined in dycore_c_api.cpp (not part of the
// public header). Returns 1 when the persistent context holds a non-null
// Halo_Manager, 0 otherwise.
extern "C" int dycore_test_halo_manager_is_set(void);

namespace {

// ─── Minimal synthetic mesh dimensions ───────────────────────────────────────
struct SmokeMeshDims {
  static constexpr int nCells = 4;
  static constexpr int nEdges = 8;
  static constexpr int nVertices = 4;
  static constexpr int nVertLevels = 3;
  static constexpr int maxEdges = 6;
  static constexpr int num_scalars = 2;
};

// ─── Heap-allocated, valid placeholder buffers for a minimal mesh ─────────────
struct SmokeBuffers {
  std::vector<int> cellsOnEdge;
  std::vector<int> edgesOnCell;
  std::vector<int> verticesOnEdge;
  std::vector<int> nEdgesOnCell;
  std::vector<double> dvEdge;
  std::vector<double> dcEdge;
  std::vector<double> areaCell;
  std::vector<double> zgrid;
  std::vector<double> zz;
  std::vector<double> fzm;
  std::vector<double> fzp;
  std::vector<double> u_tl1, u_tl2;
  std::vector<double> w_tl1, w_tl2;
  std::vector<double> theta_m_tl1, theta_m_tl2;
  std::vector<double> rho_zz_tl1, rho_zz_tl2;
  std::vector<double> scalars_tl1, scalars_tl2;

  SmokeBuffers() {
    using D = SmokeMeshDims;
    cellsOnEdge.assign(2 * D::nEdges, 1);
    edgesOnCell.assign(D::maxEdges * D::nCells, 1);
    verticesOnEdge.assign(2 * D::nEdges, 1);
    nEdgesOnCell.assign(D::nCells, D::maxEdges);

    dvEdge.assign(D::nEdges, 1000.0);
    dcEdge.assign(D::nEdges, 2000.0);
    areaCell.assign(D::nCells, 1.0e6);
    zgrid.assign((D::nVertLevels + 1) * D::nCells, 100.0);
    zz.assign(D::nVertLevels * D::nCells, 1.0);
    fzm.assign(D::nVertLevels * D::nCells, 0.5);
    fzp.assign(D::nVertLevels * D::nCells, 0.5);

    u_tl1.assign(D::nVertLevels * D::nEdges, 0.0);
    u_tl2.assign(D::nVertLevels * D::nEdges, 0.0);
    w_tl1.assign((D::nVertLevels + 1) * D::nCells, 0.0);
    w_tl2.assign((D::nVertLevels + 1) * D::nCells, 0.0);
    theta_m_tl1.assign(D::nVertLevels * D::nCells, 0.0);
    theta_m_tl2.assign(D::nVertLevels * D::nCells, 0.0);
    rho_zz_tl1.assign(D::nVertLevels * D::nCells, 0.0);
    rho_zz_tl2.assign(D::nVertLevels * D::nCells, 0.0);
    scalars_tl1.assign(D::num_scalars * D::nVertLevels * D::nCells, 0.0);
    scalars_tl2.assign(D::num_scalars * D::nVertLevels * D::nCells, 0.0);
  }

  /// Call dycore_init with a single-rank (empty) marshalled halo topology.
  int call_init(int mpi_comm_fortran) {
    using D = SmokeMeshDims;
    return dycore_init(
        D::nCells, D::nEdges, D::nVertices, D::nVertLevels, D::maxEdges,
        D::num_scalars,
        /* nCellsSolve */ D::nCells, /* nEdgesSolve */ D::nEdges,
        cellsOnEdge.data(), edgesOnCell.data(), verticesOnEdge.data(),
        nEdgesOnCell.data(),
        dvEdge.data(), dcEdge.data(), areaCell.data(),
        zgrid.data(), zz.data(), fzm.data(), fzp.data(),
        u_tl1.data(), u_tl2.data(),
        w_tl1.data(), w_tl2.data(),
        theta_m_tl1.data(), theta_m_tl2.data(),
        rho_zz_tl1.data(), rho_zz_tl2.data(),
        scalars_tl1.data(), scalars_tl2.data(),
        /* time_integration_order */ 3,
        /* number_of_sub_steps */ 6,
        /* dynamics_split_steps */ 1,
        /* config_monotonic */ 1,
        /* config_scalar_advection */ 1,
        /* config_apply_lbcs */ 0,
        /* config_mix_full */ 0,
        /* config_iau */ 0,
        /* gpu_aware_comm */ 0,
        /* config_smdiv */ 0.1,
        /* config_len_disp */ 120000.0,
        /* config_apvm_upwinding */ 0.0,
        /* config_hollingsworth */ 1,
        /* mpi_comm_fortran */ mpi_comm_fortran,
        // Single-rank marshalled topology: every direction has zero neighbors.
        // This is the CSR description the marshaller emits for a one-rank
        // decomposition; the pointer arguments are legitimately null.
        /* cell send   */ 0, 0, nullptr, nullptr, nullptr,
        /* cell recv   */ 0, 0, nullptr, nullptr, nullptr,
        /* edge send   */ 0, 0, nullptr, nullptr, nullptr,
        /* edge recv   */ 0, 0, nullptr, nullptr, nullptr,
        /* vertex send */ 0, 0, nullptr, nullptr, nullptr,
        /* vertex recv */ 0, 0, nullptr, nullptr, nullptr);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Single-lifecycle smoke test: init -> observe wiring -> timestep -> finalize.
//
// Kokkos can only be initialized once per process, so the entire init/timestep/
// finalize cycle and all wiring assertions live in one test body.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DycoreInitWiringSmoke, HaloManagerConstructedAndWiredForTimestep) {
  // Feature: cpp-dycore-halo-exchange, Task 8.3
  // Validates: Requirements 7.5, 10.1, 10.2

  // No context exists before init, so the manager must not be set yet.
  ASSERT_EQ(dycore_test_halo_manager_is_set(), 0)
      << "No Halo_Manager should exist before dycore_init";

  SmokeBuffers bufs;

  // Use a real MPI communicator handle so the manager's MPI_Comm_dup succeeds.
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  // (Req 7.5, 10.1) dycore_init accepts the marshalled CSR arrays and succeeds.
  const int rc = bufs.call_init(comm_f);
  ASSERT_EQ(rc, 0)
      << "dycore_init should succeed (return 0) with marshalled halo arrays; "
      << "got error code " << rc;

  // (Req 10.1) A non-null Halo_Manager was constructed and held in the context.
  // This is the exact pointer dycore_timestep assigns to
  // AdvanceDomain.halo_manager (Req 10.2).
  ASSERT_EQ(dycore_test_halo_manager_is_set(), 1)
      << "dycore_init should construct a non-null Halo_Manager in the context";

  // (Req 10.2) Build the AdvanceDomain for a timestep and run it. With a
  // non-null manager wired in, the previously no-op halo-exchange call sites are
  // now active; on a single rank the empty plans make each exchange a safe
  // no-op, so the timestep must complete without error.
  ASSERT_NO_THROW(dycore_timestep(/* dt */ 10.0, /* itimestep */ 1))
      << "dycore_timestep should run without error on the wired halo path";

  // The manager persists across timesteps.
  EXPECT_EQ(dycore_test_halo_manager_is_set(), 1)
      << "Halo_Manager should persist after a timestep";

  // Tear down: destroys the context (and the Halo_Manager) before Kokkos
  // finalize, so the manager must no longer be set afterward.
  dycore_finalize();
  EXPECT_EQ(dycore_test_halo_manager_is_set(), 0)
      << "Halo_Manager should be released after dycore_finalize";
}

}  // namespace

// Dedicated runner: initialize MPI (required by the Halo_Manager's
// MPI_Comm_dup) and let dycore_init / dycore_finalize own the Kokkos lifecycle.
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
