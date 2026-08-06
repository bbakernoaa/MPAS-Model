/// @file prop_api_geometry.cpp
/// @brief Property tests for C API validation in mpas_dycore_cpp_set_geometry.
///
/// **Validates: Requirements 4.6, 4.7**
///
/// Property 5: C API validation rejects null pointers and mismatched dimensions
///             - Null pointer arguments return error code 1 with non-empty message
///             - Mismatched dimension extents return error code 1 with non-empty message

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/api.h>
#include <mpas_dycore/state.hpp>
#include "../synthetic_mesh.hpp"
#include <mpi.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>

using namespace mpas::dycore;
using namespace mpas::dycore::testing;

// ============================================================================
// MPI Environment
// ============================================================================

class ApiGeometryTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override { MPI_Init(nullptr, nullptr); }
    void TearDown() override { MPI_Finalize(); }
};

auto* const api_geometry_env =
    ::testing::AddGlobalTestEnvironment(new ApiGeometryTestEnvironment);

// ============================================================================
// Test Fixture: ensures dycore is initialized before geometry tests
// ============================================================================

class ApiGeometryTest : public ::testing::Test {
protected:
    // Minimal mesh dimensions used for init
    static constexpr int kNCells = 4;
    static constexpr int kNVertLevels = 10;

    void SetUp() override {
        // Ensure clean state
        char errmsg[512] = {};
        mpas_dycore_cpp_finalize(errmsg, 512);

        // Initialize the dycore so set_geometry can be called
        int rc = init_dycore();
        ASSERT_EQ(rc, 0) << "init should succeed in SetUp";
    }

    void TearDown() override {
        char errmsg[512] = {};
        mpas_dycore_cpp_finalize(errmsg, 512);
    }

    /// @brief Initialize the dycore with a valid synthetic mesh.
    static int init_dycore() {
        SyntheticMesh mesh = genValidMesh(kNCells, kNVertLevels);

        char errmsg[512] = {};
        int mpi_comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);

        return mpas_dycore_cpp_init(
            static_cast<int>(mesh.nCells),
            static_cast<int>(mesh.nEdges),
            static_cast<int>(mesh.nVertices),
            static_cast<int>(mesh.maxEdges),
            static_cast<int>(mesh.nVertLevels),
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
    }

    /// @brief Get stored mesh dimensions from the initialized state.
    struct Dims {
        int nCells;
        int nEdges;
        int nVertices;
        int nVertLevels;
        int maxEdges;
    };

    static Dims get_dims() {
        auto& state = get_dycore_state();
        return Dims{
            static_cast<int>(state.nCells),
            static_cast<int>(state.nEdges),
            static_cast<int>(state.nVertices),
            static_cast<int>(state.nVertLevels),
            static_cast<int>(state.maxEdges)
        };
    }

    /// @brief Helper struct holding all geometry arrays with valid data.
    struct GeometryArrays {
        std::vector<double> areaCell;
        std::vector<double> invAreaCell;
        std::vector<double> dvEdge;
        std::vector<double> dcEdge;
        std::vector<double> invDcEdge;
        std::vector<double> rdzw;
        std::vector<double> rdzu;
        std::vector<double> fzm;
        std::vector<double> fzp;
        std::vector<double> etp;
        std::vector<double> etm;
        std::vector<double> ewp;
        std::vector<double> ewm;
        std::vector<double> zz;
        std::vector<double> rb;
        std::vector<double> rtb;
        std::vector<double> pb;
        std::vector<double> edgesOnCell_sign;
        std::vector<double> specZoneMaskEdge;
        std::vector<double> specZoneMaskCell;
        std::vector<double> weightsOnEdge;
        std::vector<int> nEdgesOnEdge;
        std::vector<int> edgesOnEdge;
        std::vector<int> advCellsForEdge;
        std::vector<int> nAdvCellsForEdge;
        std::vector<double> adv_coefs;
        std::vector<double> adv_coefs_3rd;
        std::vector<double> fVertex;
        std::vector<double> areaTriangle;

        int maxEdges2 = 4;
        int maxAdvCells = 4;
    };

    /// @brief Create valid geometry arrays matching the initialized mesh.
    static GeometryArrays create_valid_geometry() {
        auto d = get_dims();
        GeometryArrays g;

        g.areaCell.assign(static_cast<std::size_t>(d.nCells), 1.0e9);
        g.invAreaCell.assign(static_cast<std::size_t>(d.nCells), 1.0e-9);
        g.dvEdge.assign(static_cast<std::size_t>(d.nEdges), 1.0e4);
        g.dcEdge.assign(static_cast<std::size_t>(d.nEdges), 1.2e4);
        g.invDcEdge.assign(static_cast<std::size_t>(d.nEdges), 1.0 / 1.2e4);
        g.rdzw.assign(static_cast<std::size_t>(d.nVertLevels), 1.0 / 500.0);
        g.rdzu.assign(static_cast<std::size_t>(d.nVertLevels), 1.0 / 500.0);
        g.fzm.assign(static_cast<std::size_t>(d.nVertLevels), 0.5);
        g.fzp.assign(static_cast<std::size_t>(d.nVertLevels), 0.5);
        g.etp.assign(static_cast<std::size_t>(d.nVertLevels), 1.0);
        g.etm.assign(static_cast<std::size_t>(d.nVertLevels), 0.0);
        g.ewp.assign(static_cast<std::size_t>(d.nVertLevels + 1), 0.5);
        g.ewm.assign(static_cast<std::size_t>(d.nVertLevels + 1), 0.5);
        g.zz.assign(static_cast<std::size_t>(d.nVertLevels) * d.nCells, 1.0);
        g.rb.assign(static_cast<std::size_t>(d.nVertLevels) * d.nCells, 1.2);
        g.rtb.assign(static_cast<std::size_t>(d.nVertLevels) * d.nCells, 350.0);
        g.pb.assign(static_cast<std::size_t>(d.nVertLevels) * d.nCells, 1.0e5);
        g.edgesOnCell_sign.assign(
            static_cast<std::size_t>(d.maxEdges) * d.nCells, 1.0);
        g.specZoneMaskEdge.assign(static_cast<std::size_t>(d.nEdges), 0.0);
        g.specZoneMaskCell.assign(static_cast<std::size_t>(d.nCells), 0.0);
        g.weightsOnEdge.assign(
            static_cast<std::size_t>(d.nEdges) * g.maxEdges2, 0.25);
        g.nEdgesOnEdge.assign(static_cast<std::size_t>(d.nEdges), g.maxEdges2);
        g.edgesOnEdge.assign(
            static_cast<std::size_t>(d.nEdges) * g.maxEdges2, 0);
        g.advCellsForEdge.assign(
            static_cast<std::size_t>(d.nEdges) * g.maxAdvCells, 0);
        g.nAdvCellsForEdge.assign(static_cast<std::size_t>(d.nEdges), g.maxAdvCells);
        g.adv_coefs.assign(
            static_cast<std::size_t>(g.maxAdvCells) * d.nEdges, 0.1);
        g.adv_coefs_3rd.assign(
            static_cast<std::size_t>(g.maxAdvCells) * d.nEdges, 0.05);
        g.fVertex.assign(static_cast<std::size_t>(d.nVertices), 1.0e-4);
        g.areaTriangle.assign(static_cast<std::size_t>(d.nVertices), 3.0e8);

        return g;
    }

    /// @brief Call set_geometry with all valid arrays (returns 0 on success).
    static int call_set_geometry_valid(GeometryArrays& g, char* errmsg, int errmsg_len) {
        auto d = get_dims();
        return mpas_dycore_cpp_set_geometry(
            g.areaCell.data(), g.invAreaCell.data(),
            g.dvEdge.data(), g.dcEdge.data(), g.invDcEdge.data(),
            g.rdzw.data(), g.rdzu.data(),
            g.fzm.data(), g.fzp.data(),
            g.etp.data(), g.etm.data(),
            g.ewp.data(), g.ewm.data(),
            g.zz.data(),
            g.rb.data(), g.rtb.data(), g.pb.data(),
            g.edgesOnCell_sign.data(),
            g.specZoneMaskEdge.data(), g.specZoneMaskCell.data(),
            g.weightsOnEdge.data(),
            g.nEdgesOnEdge.data(),
            g.edgesOnEdge.data(),
            g.advCellsForEdge.data(),
            g.nAdvCellsForEdge.data(),
            g.adv_coefs.data(), g.adv_coefs_3rd.data(),
            g.fVertex.data(), g.areaTriangle.data(),
            nullptr, nullptr,   // zb_cell, zb3_cell (optional)
            nullptr, nullptr, nullptr,  // rho_base, rtheta_base, exner_base (optional)
            0.0, 0.0, 0.0,     // cf1, cf2, cf3
            nullptr,            // nEdgesOnCell (optional)
            d.nCells, d.nEdges, d.nVertices, d.nVertLevels,
            d.maxEdges, g.maxEdges2, g.maxAdvCells,
            errmsg, errmsg_len
        );
    }
};

// ============================================================================
// Sanity check: valid geometry call succeeds
// ============================================================================

TEST_F(ApiGeometryTest, ValidGeometryCallSucceeds) {
    auto g = create_valid_geometry();
    char errmsg[512] = {};
    int rc = call_set_geometry_valid(g, errmsg, 512);
    EXPECT_EQ(rc, 0) << "valid geometry call should succeed, got: " << errmsg;
}

// ============================================================================
// Property 5a: Null pointer rejection
// ============================================================================

RC_GTEST_FIXTURE_PROP(ApiGeometryTest, NullPointerReturnsError, ()) {
    // There are 29 pointer arguments to set_geometry.
    // We pick one at random and set it to null.
    auto g = create_valid_geometry();
    auto d = get_dims();

    const int numPointers = 29;
    int ptrIdx = *rc::gen::inRange(0, numPointers);

    const double* null_dbl = nullptr;
    const int* null_int = nullptr;

    char errmsg[512] = {};

    int rc_val = mpas_dycore_cpp_set_geometry(
        /* 0  areaCell         */ ptrIdx == 0 ? null_dbl : g.areaCell.data(),
        /* 1  invAreaCell      */ ptrIdx == 1 ? null_dbl : g.invAreaCell.data(),
        /* 2  dvEdge           */ ptrIdx == 2 ? null_dbl : g.dvEdge.data(),
        /* 3  dcEdge           */ ptrIdx == 3 ? null_dbl : g.dcEdge.data(),
        /* 4  invDcEdge        */ ptrIdx == 4 ? null_dbl : g.invDcEdge.data(),
        /* 5  rdzw             */ ptrIdx == 5 ? null_dbl : g.rdzw.data(),
        /* 6  rdzu             */ ptrIdx == 6 ? null_dbl : g.rdzu.data(),
        /* 7  fzm              */ ptrIdx == 7 ? null_dbl : g.fzm.data(),
        /* 8  fzp              */ ptrIdx == 8 ? null_dbl : g.fzp.data(),
        /* 9  etp              */ ptrIdx == 9 ? null_dbl : g.etp.data(),
        /* 10 etm              */ ptrIdx == 10 ? null_dbl : g.etm.data(),
        /* 11 ewp              */ ptrIdx == 11 ? null_dbl : g.ewp.data(),
        /* 12 ewm              */ ptrIdx == 12 ? null_dbl : g.ewm.data(),
        /* 13 zz               */ ptrIdx == 13 ? null_dbl : g.zz.data(),
        /* 14 rb               */ ptrIdx == 14 ? null_dbl : g.rb.data(),
        /* 15 rtb              */ ptrIdx == 15 ? null_dbl : g.rtb.data(),
        /* 16 pb               */ ptrIdx == 16 ? null_dbl : g.pb.data(),
        /* 17 edgesOnCell_sign */ ptrIdx == 17 ? null_dbl : g.edgesOnCell_sign.data(),
        /* 18 specZoneMaskEdge */ ptrIdx == 18 ? null_dbl : g.specZoneMaskEdge.data(),
        /* 19 specZoneMaskCell */ ptrIdx == 19 ? null_dbl : g.specZoneMaskCell.data(),
        /* 20 weightsOnEdge    */ ptrIdx == 20 ? null_dbl : g.weightsOnEdge.data(),
        /* 21 nEdgesOnEdge     */ ptrIdx == 21 ? null_int : g.nEdgesOnEdge.data(),
        /* 22 edgesOnEdge      */ ptrIdx == 22 ? null_int : g.edgesOnEdge.data(),
        /* 23 advCellsForEdge  */ ptrIdx == 23 ? null_int : g.advCellsForEdge.data(),
        /* 24 nAdvCellsForEdge */ ptrIdx == 24 ? null_int : g.nAdvCellsForEdge.data(),
        /* 25 adv_coefs        */ ptrIdx == 25 ? null_dbl : g.adv_coefs.data(),
        /* 26 adv_coefs_3rd    */ ptrIdx == 26 ? null_dbl : g.adv_coefs_3rd.data(),
        /* 27 fVertex          */ ptrIdx == 27 ? null_dbl : g.fVertex.data(),
        /* 28 areaTriangle     */ ptrIdx == 28 ? null_dbl : g.areaTriangle.data(),
        /* zb_cell (optional)  */ nullptr,
        /* zb3_cell (optional) */ nullptr,
        /* rho_base (optional) */ nullptr,
        /* rtheta_base (opt)   */ nullptr,
        /* exner_base (opt)    */ nullptr,
        /* cf1, cf2, cf3       */ 0.0, 0.0, 0.0,
        /* nEdgesOnCell (opt)  */ nullptr,
        d.nCells, d.nEdges, d.nVertices, d.nVertLevels,
        d.maxEdges, g.maxEdges2, g.maxAdvCells,
        errmsg, 512
    );

    // Verify error code 1 (std::exception caught by api_wrap)
    RC_ASSERT(rc_val == 1);
    // Verify non-empty error message
    RC_ASSERT(std::strlen(errmsg) > 0);
}

// ============================================================================
// Property 5b: Mismatched dimensions rejection
// ============================================================================

RC_GTEST_FIXTURE_PROP(ApiGeometryTest, MismatchedDimensionsReturnsError, ()) {
    auto g = create_valid_geometry();
    auto d = get_dims();

    // Pick which dimension to perturb: 0=nCells, 1=nEdges, 2=nVertices, 3=nVertLevels, 4=maxEdges
    int dimIdx = *rc::gen::inRange(0, 5);

    // Generate a non-zero offset to make it mismatch
    int offset = *rc::gen::suchThat(rc::gen::inRange(-10, 11), [](int v) { return v != 0; });

    int pass_nCells = d.nCells;
    int pass_nEdges = d.nEdges;
    int pass_nVertices = d.nVertices;
    int pass_nVertLevels = d.nVertLevels;
    int pass_maxEdges = d.maxEdges;

    switch (dimIdx) {
        case 0: pass_nCells += offset; break;
        case 1: pass_nEdges += offset; break;
        case 2: pass_nVertices += offset; break;
        case 3: pass_nVertLevels += offset; break;
        case 4: pass_maxEdges += offset; break;
    }

    char errmsg[512] = {};

    int rc_val = mpas_dycore_cpp_set_geometry(
        g.areaCell.data(), g.invAreaCell.data(),
        g.dvEdge.data(), g.dcEdge.data(), g.invDcEdge.data(),
        g.rdzw.data(), g.rdzu.data(),
        g.fzm.data(), g.fzp.data(),
        g.etp.data(), g.etm.data(),
        g.ewp.data(), g.ewm.data(),
        g.zz.data(),
        g.rb.data(), g.rtb.data(), g.pb.data(),
        g.edgesOnCell_sign.data(),
        g.specZoneMaskEdge.data(), g.specZoneMaskCell.data(),
        g.weightsOnEdge.data(),
        g.nEdgesOnEdge.data(),
        g.edgesOnEdge.data(),
        g.advCellsForEdge.data(),
        g.nAdvCellsForEdge.data(),
        g.adv_coefs.data(), g.adv_coefs_3rd.data(),
        g.fVertex.data(), g.areaTriangle.data(),
        nullptr, nullptr,   // zb_cell, zb3_cell (optional)
        nullptr, nullptr, nullptr,  // rho_base, rtheta_base, exner_base (optional)
        0.0, 0.0, 0.0,     // cf1, cf2, cf3
        nullptr,            // nEdgesOnCell (optional)
        pass_nCells, pass_nEdges, pass_nVertices, pass_nVertLevels,
        pass_maxEdges, g.maxEdges2, g.maxAdvCells,
        errmsg, 512
    );

    // Verify error code 1
    RC_ASSERT(rc_val == 1);
    // Verify non-empty error message
    RC_ASSERT(std::strlen(errmsg) > 0);
}
