/// @file prop_api_lifecycle.cpp
/// @brief Property tests for API lifecycle and error handling.
///
/// **Validates: Requirements 7.4, 7.6, 7.8**
///
/// Property 17: Uninitialized State Guard - Calling timestep before init returns
///              error code and does not modify field data.
/// Property 18: Error Reporting Completeness - All error paths produce non-zero
///              error codes and non-empty diagnostic messages.
/// Property 19: Finalize-Init Lifecycle - After finalize, init can be called
///              again successfully.

#include <gtest/gtest.h>
#include <mpas_dycore/api.h>
#include <mpas_dycore/state.hpp>
#include "../synthetic_mesh.hpp"
#include <mpi.h>
#include <vector>
#include <cstring>

using namespace mpas::dycore;
using namespace mpas::dycore::testing;

// ============================================================================
// MPI Environment for API lifecycle tests
// ============================================================================

class ApiLifecycleTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override { MPI_Init(nullptr, nullptr); }
    void TearDown() override { MPI_Finalize(); }
};

auto* const api_lifecycle_env =
    ::testing::AddGlobalTestEnvironment(new ApiLifecycleTestEnvironment);

// ============================================================================
// Helper: call mpas_dycore_cpp_init with a synthetic mesh
// ============================================================================

/// @brief Initialize the dycore with a valid synthetic mesh.
/// Returns 0 on success, non-zero on failure.
static int init_with_valid_mesh(int nCells = 4, int nVertLevels = 10) {
    SyntheticMesh mesh = genValidMesh(nCells, nVertLevels);

    // No halo neighbors (single process)
    int halo_n_neighbors = 0;
    int halo_n_layers = 1;

    char errmsg[512] = {};

    // Get the Fortran MPI communicator handle
    int mpi_comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);

    int rc = mpas_dycore_cpp_init(
        /* Mesh dimensions */
        static_cast<int>(mesh.nCells),
        static_cast<int>(mesh.nEdges),
        static_cast<int>(mesh.nVertices),
        static_cast<int>(mesh.maxEdges),
        static_cast<int>(mesh.nVertLevels),
        /* Connectivity arrays */
        mesh.cellsOnEdge.data(),
        static_cast<int>(mesh.nEdges), 2,
        mesh.verticesOnEdge.data(),
        static_cast<int>(mesh.nEdges), 2,
        mesh.edgesOnCell.data(),
        static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.cellsOnCell.data(),
        static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.nEdgesOnCell_arr.data(),
        static_cast<int>(mesh.nCells),
        /* Halo CSR descriptors */
        nullptr, halo_n_neighbors,
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        halo_n_layers,
        /* Configuration */
        60.0, /* dt */
        6,    /* number_of_sub_steps */
        0,    /* config_apply_lbcs */
        "2d_fixed", /* config_horiz_mixing */
        0.0,  /* config_h_mom_eddy_visc2 */
        0.0,  /* config_h_mom_eddy_visc4 */
        0.0,  /* config_h_theta_eddy_visc2 */
        /* MPI communicator */
        mpi_comm_fortran,
        /* Error output */
        errmsg, 512
    );

    return rc;
}

/// @brief Ensure the dycore is in uninitialized state before each test.
class ApiLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Force state to uninitialized
        char errmsg[512] = {};
        mpas_dycore_cpp_finalize(errmsg, 512);
    }

    void TearDown() override {
        // Clean up after test
        char errmsg[512] = {};
        mpas_dycore_cpp_finalize(errmsg, 512);
    }
};

// ============================================================================
// Property 17: Uninitialized State Guard
// ============================================================================
// Calling timestep before init returns error code and does not modify field data.

TEST_F(ApiLifecycleTest, Property17_TimestepBeforeInitReturnsError) {
    // Prepare field data with known values
    const int nVertLevels = 10;
    const int nEdges = 13; // matches genValidMesh(4)
    const int nCells = 4;
    const int nScalars = 2;

    std::vector<double> u(nVertLevels * nEdges, 42.0);
    std::vector<double> theta_m(nVertLevels * nCells, 300.0);
    std::vector<double> rho_zz(nVertLevels * nCells, 1.2);
    std::vector<double> w((nVertLevels + 1) * nCells, 0.5);
    std::vector<double> scalars(nScalars * nVertLevels * nCells, 0.01);

    // Save copies for comparison
    std::vector<double> u_orig(u);
    std::vector<double> theta_m_orig(theta_m);
    std::vector<double> rho_zz_orig(rho_zz);
    std::vector<double> w_orig(w);
    std::vector<double> scalars_orig(scalars);

    char errmsg[512] = {};

    // Call timestep WITHOUT calling init first
    int rc = mpas_dycore_cpp_timestep(
        u.data(), nVertLevels, nEdges,
        theta_m.data(), nVertLevels, nCells,
        rho_zz.data(), nVertLevels, nCells,
        w.data(), nVertLevels + 1, nCells,
        scalars.data(), nScalars, nVertLevels, nCells,
        errmsg, 512
    );

    // Verify non-zero error code
    EXPECT_NE(rc, 0) << "timestep before init should return non-zero error code";

    // Verify error message contains relevant diagnostic
    EXPECT_GT(std::strlen(errmsg), 0u) << "error message should not be empty";
    std::string msg(errmsg);
    EXPECT_TRUE(msg.find("not initialized") != std::string::npos ||
                msg.find("not init") != std::string::npos)
        << "error message should indicate not initialized, got: " << msg;

    // Verify field data is unchanged
    EXPECT_EQ(u, u_orig) << "u field should not be modified";
    EXPECT_EQ(theta_m, theta_m_orig) << "theta_m field should not be modified";
    EXPECT_EQ(rho_zz, rho_zz_orig) << "rho_zz field should not be modified";
    EXPECT_EQ(w, w_orig) << "w field should not be modified";
    EXPECT_EQ(scalars, scalars_orig) << "scalars field should not be modified";
}

TEST_F(ApiLifecycleTest, Property17_TimestepAfterFinalizeReturnsError) {
    // Init then finalize, then call timestep
    int rc = init_with_valid_mesh(4, 10);
    ASSERT_EQ(rc, 0) << "init should succeed";

    char errmsg[512] = {};
    rc = mpas_dycore_cpp_finalize(errmsg, 512);
    ASSERT_EQ(rc, 0) << "finalize should succeed";

    // Now try timestep — should fail since we're uninitialized again
    const int nVertLevels = 10;
    const int nEdges = 13;
    const int nCells = 4;
    const int nScalars = 2;

    std::vector<double> u(nVertLevels * nEdges, 42.0);
    std::vector<double> theta_m(nVertLevels * nCells, 300.0);
    std::vector<double> rho_zz(nVertLevels * nCells, 1.2);
    std::vector<double> w((nVertLevels + 1) * nCells, 0.5);
    std::vector<double> scalars(nScalars * nVertLevels * nCells, 0.01);

    std::vector<double> u_orig(u);

    std::memset(errmsg, 0, sizeof(errmsg));
    rc = mpas_dycore_cpp_timestep(
        u.data(), nVertLevels, nEdges,
        theta_m.data(), nVertLevels, nCells,
        rho_zz.data(), nVertLevels, nCells,
        w.data(), nVertLevels + 1, nCells,
        scalars.data(), nScalars, nVertLevels, nCells,
        errmsg, 512
    );

    EXPECT_NE(rc, 0) << "timestep after finalize should return non-zero error code";
    EXPECT_GT(std::strlen(errmsg), 0u) << "error message should not be empty";
    EXPECT_EQ(u, u_orig) << "u field should not be modified after error";
}

// ============================================================================
// Property 18: Error Reporting Completeness
// ============================================================================
// All error paths produce non-zero error codes and non-empty diagnostic messages.

TEST_F(ApiLifecycleTest, Property18_InitWithZeroCellsReturnsError) {
    SyntheticMesh mesh = genValidMesh(4, 10);

    char errmsg[512] = {};
    int mpi_comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);

    // Call init with nCells = 0 (invalid)
    int rc = mpas_dycore_cpp_init(
        0, /* nCells = 0 (INVALID) */
        static_cast<int>(mesh.nEdges),
        static_cast<int>(mesh.nVertices),
        static_cast<int>(mesh.maxEdges),
        static_cast<int>(mesh.nVertLevels),
        mesh.cellsOnEdge.data(), static_cast<int>(mesh.nEdges), 2,
        mesh.verticesOnEdge.data(), static_cast<int>(mesh.nEdges), 2,
        mesh.edgesOnCell.data(), static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.cellsOnCell.data(), static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.nEdgesOnCell_arr.data(), static_cast<int>(mesh.nCells),
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        1,
        60.0, 6,
        0, "2d_fixed", 0.0, 0.0, 0.0,
        mpi_comm_fortran,
        errmsg, 512
    );

    EXPECT_NE(rc, 0) << "init with nCells=0 should fail";
    EXPECT_GT(std::strlen(errmsg), 0u)
        << "error message should not be empty for nCells=0";
}

TEST_F(ApiLifecycleTest, Property18_InitWithInvalidSubStepsZero) {
    SyntheticMesh mesh = genValidMesh(4, 10);

    char errmsg[512] = {};
    int mpi_comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);

    int rc = mpas_dycore_cpp_init(
        static_cast<int>(mesh.nCells),
        static_cast<int>(mesh.nEdges),
        static_cast<int>(mesh.nVertices),
        static_cast<int>(mesh.maxEdges),
        static_cast<int>(mesh.nVertLevels),
        mesh.cellsOnEdge.data(), static_cast<int>(mesh.nEdges), 2,
        mesh.verticesOnEdge.data(), static_cast<int>(mesh.nEdges), 2,
        mesh.edgesOnCell.data(), static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.cellsOnCell.data(), static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.nEdgesOnCell_arr.data(), static_cast<int>(mesh.nCells),
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        1,
        60.0,
        0, /* number_of_sub_steps = 0 (INVALID, must be 2-24) */
        0, "2d_fixed", 0.0, 0.0, 0.0,
        mpi_comm_fortran,
        errmsg, 512
    );

    EXPECT_NE(rc, 0) << "init with number_of_sub_steps=0 should fail";
    EXPECT_GT(std::strlen(errmsg), 0u)
        << "error message should not be empty for sub_steps=0";
}

TEST_F(ApiLifecycleTest, Property18_InitWithInvalidSubStepsOne) {
    SyntheticMesh mesh = genValidMesh(4, 10);

    char errmsg[512] = {};
    int mpi_comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);

    int rc = mpas_dycore_cpp_init(
        static_cast<int>(mesh.nCells),
        static_cast<int>(mesh.nEdges),
        static_cast<int>(mesh.nVertices),
        static_cast<int>(mesh.maxEdges),
        static_cast<int>(mesh.nVertLevels),
        mesh.cellsOnEdge.data(), static_cast<int>(mesh.nEdges), 2,
        mesh.verticesOnEdge.data(), static_cast<int>(mesh.nEdges), 2,
        mesh.edgesOnCell.data(), static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.cellsOnCell.data(), static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.nEdgesOnCell_arr.data(), static_cast<int>(mesh.nCells),
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        1,
        60.0,
        1, /* number_of_sub_steps = 1 (INVALID, must be 2-24) */
        0, "2d_fixed", 0.0, 0.0, 0.0,
        mpi_comm_fortran,
        errmsg, 512
    );

    EXPECT_NE(rc, 0) << "init with number_of_sub_steps=1 should fail";
    EXPECT_GT(std::strlen(errmsg), 0u)
        << "error message should not be empty for sub_steps=1";
}

TEST_F(ApiLifecycleTest, Property18_InitWithInvalidSubSteps25) {
    SyntheticMesh mesh = genValidMesh(4, 10);

    char errmsg[512] = {};
    int mpi_comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);

    int rc = mpas_dycore_cpp_init(
        static_cast<int>(mesh.nCells),
        static_cast<int>(mesh.nEdges),
        static_cast<int>(mesh.nVertices),
        static_cast<int>(mesh.maxEdges),
        static_cast<int>(mesh.nVertLevels),
        mesh.cellsOnEdge.data(), static_cast<int>(mesh.nEdges), 2,
        mesh.verticesOnEdge.data(), static_cast<int>(mesh.nEdges), 2,
        mesh.edgesOnCell.data(), static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.cellsOnCell.data(), static_cast<int>(mesh.nCells), static_cast<int>(mesh.maxEdges),
        mesh.nEdgesOnCell_arr.data(), static_cast<int>(mesh.nCells),
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        1,
        60.0,
        25, /* number_of_sub_steps = 25 (INVALID, must be 2-24) */
        0, "2d_fixed", 0.0, 0.0, 0.0,
        mpi_comm_fortran,
        errmsg, 512
    );

    EXPECT_NE(rc, 0) << "init with number_of_sub_steps=25 should fail";
    EXPECT_GT(std::strlen(errmsg), 0u)
        << "error message should not be empty for sub_steps=25";
}

TEST_F(ApiLifecycleTest, Property18_TimestepWithNullPointerReturnsError) {
    // First init successfully
    int rc = init_with_valid_mesh(4, 10);
    ASSERT_EQ(rc, 0) << "init should succeed";

    auto& state = get_dycore_state();
    ASSERT_EQ(state.status, DycoreStatus::Ready);

    const int nVertLevels = static_cast<int>(state.nVertLevels);
    const int nEdges = static_cast<int>(state.nEdges);
    const int nCells = static_cast<int>(state.nCells);
    const int nScalars = 2;

    std::vector<double> theta_m(nVertLevels * nCells, 300.0);
    std::vector<double> rho_zz(nVertLevels * nCells, 1.2);
    std::vector<double> w((nVertLevels + 1) * nCells, 0.5);
    std::vector<double> scalars(nScalars * nVertLevels * nCells, 0.01);

    char errmsg[512] = {};

    // Call timestep with null u pointer
    rc = mpas_dycore_cpp_timestep(
        nullptr, /* u = null (INVALID) */
        nVertLevels, nEdges,
        theta_m.data(), nVertLevels, nCells,
        rho_zz.data(), nVertLevels, nCells,
        w.data(), nVertLevels + 1, nCells,
        scalars.data(), nScalars, nVertLevels, nCells,
        errmsg, 512
    );

    EXPECT_NE(rc, 0) << "timestep with null pointer should return non-zero";
    EXPECT_GT(std::strlen(errmsg), 0u)
        << "error message should not be empty for null pointer";
}

TEST_F(ApiLifecycleTest, Property18_TimestepWithWrongExtentsReturnsError) {
    // First init successfully
    int rc = init_with_valid_mesh(4, 10);
    ASSERT_EQ(rc, 0) << "init should succeed";

    auto& state = get_dycore_state();
    ASSERT_EQ(state.status, DycoreStatus::Ready);

    const int nVertLevels = static_cast<int>(state.nVertLevels);
    const int nEdges = static_cast<int>(state.nEdges);
    const int nCells = static_cast<int>(state.nCells);
    const int nScalars = 2;

    std::vector<double> u(nVertLevels * nEdges, 42.0);
    std::vector<double> theta_m(nVertLevels * nCells, 300.0);
    std::vector<double> rho_zz(nVertLevels * nCells, 1.2);
    std::vector<double> w((nVertLevels + 1) * nCells, 0.5);
    std::vector<double> scalars(nScalars * nVertLevels * nCells, 0.01);

    char errmsg[512] = {};

    // Call timestep with wrong u extents (swap dimensions)
    rc = mpas_dycore_cpp_timestep(
        u.data(),
        nEdges, nVertLevels, /* wrong: should be (nVertLevels, nEdges) */
        theta_m.data(), nVertLevels, nCells,
        rho_zz.data(), nVertLevels, nCells,
        w.data(), nVertLevels + 1, nCells,
        scalars.data(), nScalars, nVertLevels, nCells,
        errmsg, 512
    );

    EXPECT_NE(rc, 0) << "timestep with wrong extents should return non-zero";
    EXPECT_GT(std::strlen(errmsg), 0u)
        << "error message should not be empty for wrong extents";
}

// ============================================================================
// Property 19: Finalize-Init Lifecycle
// ============================================================================
// After finalize, init can be called again successfully.

TEST_F(ApiLifecycleTest, Property19_FinalizeInitCycle) {
    // First init → should succeed
    int rc = init_with_valid_mesh(4, 10);
    EXPECT_EQ(rc, 0) << "first init should succeed";

    auto& state = get_dycore_state();
    EXPECT_EQ(state.status, DycoreStatus::Ready);

    // First finalize → should succeed
    char errmsg[512] = {};
    rc = mpas_dycore_cpp_finalize(errmsg, 512);
    EXPECT_EQ(rc, 0) << "first finalize should succeed";
    EXPECT_EQ(state.status, DycoreStatus::Uninitialized);

    // Second init → should succeed (re-initialization after finalize)
    rc = init_with_valid_mesh(4, 10);
    EXPECT_EQ(rc, 0) << "second init after finalize should succeed";
    EXPECT_EQ(state.status, DycoreStatus::Ready);

    // Second finalize → should succeed
    std::memset(errmsg, 0, sizeof(errmsg));
    rc = mpas_dycore_cpp_finalize(errmsg, 512);
    EXPECT_EQ(rc, 0) << "second finalize should succeed";
    EXPECT_EQ(state.status, DycoreStatus::Uninitialized);
}

TEST_F(ApiLifecycleTest, Property19_MultipleInitFinalizeCycles) {
    // Perform multiple init/finalize cycles to ensure no resource leaks
    // or state corruption across cycles
    for (int cycle = 0; cycle < 5; ++cycle) {
        int rc = init_with_valid_mesh(4, 10);
        EXPECT_EQ(rc, 0) << "init should succeed on cycle " << cycle;

        auto& state = get_dycore_state();
        EXPECT_EQ(state.status, DycoreStatus::Ready)
            << "state should be Ready on cycle " << cycle;

        char errmsg[512] = {};
        rc = mpas_dycore_cpp_finalize(errmsg, 512);
        EXPECT_EQ(rc, 0) << "finalize should succeed on cycle " << cycle;
        EXPECT_EQ(state.status, DycoreStatus::Uninitialized)
            << "state should be Uninitialized after finalize on cycle " << cycle;
    }
}

TEST_F(ApiLifecycleTest, Property19_StateIsUninitializedAfterFinalFinalize) {
    // Init, finalize, then verify state fields are reset
    int rc = init_with_valid_mesh(4, 10);
    ASSERT_EQ(rc, 0);

    auto& state = get_dycore_state();
    // Verify state was populated
    EXPECT_GT(state.nCells, 0);
    EXPECT_GT(state.nEdges, 0);

    char errmsg[512] = {};
    rc = mpas_dycore_cpp_finalize(errmsg, 512);
    ASSERT_EQ(rc, 0);

    // After finalize, state should be fully reset
    EXPECT_EQ(state.status, DycoreStatus::Uninitialized);
    EXPECT_EQ(state.nCells, 0);
    EXPECT_EQ(state.nEdges, 0);
    EXPECT_EQ(state.nVertices, 0);
    EXPECT_EQ(state.nVertLevels, 0);
    EXPECT_EQ(state.comm, MPI_COMM_NULL);
    EXPECT_TRUE(state.connectivity_storage.empty());
    EXPECT_TRUE(state.workspace.empty());
}
