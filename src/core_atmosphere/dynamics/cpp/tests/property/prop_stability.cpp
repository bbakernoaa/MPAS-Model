/// @file prop_stability.cpp
/// @brief Property test for idealized test stability (Property 7).
///
/// **Validates: Requirement 5.4**
///
/// Property 7: Idealized test stability
/// For any gravity wave test case on a planar hex mesh with physically
/// reasonable parameters, the solution remains bounded after N timesteps:
/// max(|u|) < U_bound, max(|w|) < W_bound, and theta_m > 0 everywhere.
/// No NaN/Inf values appear in any field.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/api.h>

#include "../planar_hex_mesh.hpp"

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

using namespace mpas::dycore::testing;

// ============================================================================
// MPI Environment
// ============================================================================

class StabilityTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override { MPI_Init(nullptr, nullptr); }
    void TearDown() override { MPI_Finalize(); }
};

auto* const stability_env =
    ::testing::AddGlobalTestEnvironment(new StabilityTestEnvironment);

// ============================================================================
// Constants
// ============================================================================

// Mesh dimensions (small for property test speed)
static constexpr int NX = 5;
static constexpr int NY = 5;
static constexpr int N_VERT_LEVELS = 10;

// Timestep configuration
static constexpr double DT = 5.0;           // seconds
static constexpr int N_SUB_STEPS = 6;
static constexpr int N_TIMESTEPS = 10;

// Stability bounds
static constexpr double U_BOUND = 100.0;    // m/s
static constexpr double W_BOUND = 50.0;     // m/s

// Physical constants
static constexpr double PI = 3.14159265358979323846;
static constexpr double H = 10000.0;        // domain height (m)

// ============================================================================
// Test Fixture
// ============================================================================

class StabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure clean dycore state
        char errmsg[512] = {};
        mpas_dycore_cpp_finalize(errmsg, 512);
    }

    void TearDown() override {
        char errmsg[512] = {};
        mpas_dycore_cpp_finalize(errmsg, 512);
    }

    /// @brief Apply a gravity wave perturbation to theta_m with given amplitude.
    static void apply_gravity_wave_perturbation(
        double* theta_m, int nVertLevels, int nCells,
        int nx, int ny, double dx, double dz,
        double amplitude) {

        // Wavelength spans the full domain width for a single mode
        double wavelength = static_cast<double>(nx) * dx;

        for (int iCell = 0; iCell < nCells; ++iCell) {
            int ix = iCell % nx;
            int iy = iCell / nx;
            bool oddRow = (iy % 2 != 0);
            double x = static_cast<double>(ix) * dx + (oddRow ? 0.5 * dx : 0.0);

            for (int k = 0; k < nVertLevels; ++k) {
                double z = (static_cast<double>(k) + 0.5) * dz;
                double perturbation = amplitude
                    * std::sin(2.0 * PI * x / wavelength)
                    * std::sin(PI * z / H);

                std::size_t idx = static_cast<std::size_t>(k) * nCells + iCell;
                theta_m[idx] += perturbation;
            }
        }
    }

    /// @brief Check that all values in array are finite (no NaN/Inf).
    static bool all_finite(const double* data, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(data[i])) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Property 7: Idealized test stability
// ============================================================================

RC_GTEST_FIXTURE_PROP(StabilityTest, SolutionRemainsBounded, ()) {
    // Generate random gravity wave amplitude in [0.001, 1.0] K
    double amplitude = *rc::gen::map(
        rc::gen::inRange(1, 1001),
        [](int v) { return static_cast<double>(v) * 0.001; });

    RC_TAG("amplitude=" + std::to_string(amplitude));

    // Generate the planar hex mesh
    auto mesh = generate_planar_hex_mesh(NX, NY, N_VERT_LEVELS);

    // Apply gravity wave perturbation
    apply_gravity_wave_perturbation(
        mesh.theta_m.data(), mesh.nVertLevels, mesh.nCells,
        NX, NY, mesh.dx, mesh.dz, amplitude);

    // Initialize dycore via C API
    char errmsg[512] = {};
    int mpi_comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);

    int rc_init = mpas_dycore_cpp_init(
        static_cast<int>(mesh.nCells),
        static_cast<int>(mesh.nEdges),
        static_cast<int>(mesh.nVertices),
        static_cast<int>(mesh.maxEdges),
        static_cast<int>(mesh.nVertLevels),
        // Connectivity
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
        // Halo (single process, no halo)
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        nullptr, 0,
        1,  // halo_n_layers
        // Configuration
        DT, N_SUB_STEPS,
        0,           // config_apply_lbcs
        "2d_fixed",  // config_horiz_mixing
        0.0,         // config_h_mom_eddy_visc2
        0.0,         // config_h_mom_eddy_visc4
        0.0,         // config_h_theta_eddy_visc2
        // MPI
        mpi_comm_fortran,
        // Error
        errmsg, 512
    );
    RC_ASSERT(rc_init == 0);

    // Set geometry
    std::memset(errmsg, 0, sizeof(errmsg));
    const auto& geom = mesh.geometry;

    int rc_geom = mpas_dycore_cpp_set_geometry(
        geom.areaCell.data(), geom.invAreaCell.data(),
        geom.dvEdge.data(), geom.dcEdge.data(), geom.invDcEdge.data(),
        geom.rdzw.data(), geom.rdzu.data(),
        geom.fzm.data(), geom.fzp.data(),
        geom.etp.data(), geom.etm.data(),
        geom.ewp.data(), geom.ewm.data(),
        geom.zz.data(),
        geom.rb.data(), geom.rtb.data(), geom.pb.data(),
        geom.edgesOnCell_sign.data(),
        geom.specZoneMaskEdge.data(), geom.specZoneMaskCell.data(),
        geom.weightsOnEdge.data(),
        geom.nEdgesOnEdge.data(),
        geom.edgesOnEdge_storage.data(),
        geom.advCellsForEdge_storage.data(),
        geom.nAdvCellsForEdge.data(),
        geom.adv_coefs.data(), geom.adv_coefs_3rd.data(),
        geom.fVertex.data(), geom.areaTriangle.data(),
        geom.zb_cell.empty() ? nullptr : geom.zb_cell.data(),
        geom.zb3_cell.empty() ? nullptr : geom.zb3_cell.data(),
        geom.rho_base.empty() ? nullptr : geom.rho_base.data(),
        geom.rtheta_base.empty() ? nullptr : geom.rtheta_base.data(),
        geom.exner_base.empty() ? nullptr : geom.exner_base.data(),
        geom.cf1, geom.cf2, geom.cf3,
        geom.nEdgesOnCell.empty() ? nullptr : geom.nEdgesOnCell.data(),
        static_cast<int>(mesh.nCells),
        static_cast<int>(mesh.nEdges),
        static_cast<int>(mesh.nVertices),
        static_cast<int>(mesh.nVertLevels),
        static_cast<int>(mesh.maxEdges),
        static_cast<int>(mesh.maxEdges2),
        static_cast<int>(mesh.maxAdvCells),
        errmsg, 512
    );
    RC_ASSERT(rc_geom == 0);

    // Execute N timesteps
    const int nVL = mesh.nVertLevels;
    const int nC = mesh.nCells;
    const int nE = mesh.nEdges;
    const int nS = mesh.nScalars;

    for (int step = 1; step <= N_TIMESTEPS; ++step) {
        std::memset(errmsg, 0, sizeof(errmsg));

        int rc_ts = mpas_dycore_cpp_timestep(
            mesh.u.data(),       nVL, nE,
            mesh.theta_m.data(), nVL, nC,
            mesh.rho_zz.data(),  nVL, nC,
            mesh.w.data(),       nVL + 1, nC,
            mesh.scalars.data(), nS, nVL, nC,
            errmsg, 512
        );
        RC_ASSERT(rc_ts == 0);

        // Check for NaN/Inf after each timestep
        RC_ASSERT(all_finite(mesh.u.data(),
            static_cast<std::size_t>(nVL) * nE));
        RC_ASSERT(all_finite(mesh.theta_m.data(),
            static_cast<std::size_t>(nVL) * nC));
        RC_ASSERT(all_finite(mesh.rho_zz.data(),
            static_cast<std::size_t>(nVL) * nC));
        RC_ASSERT(all_finite(mesh.w.data(),
            static_cast<std::size_t>(nVL + 1) * nC));
    }

    // Verify stability bounds after all timesteps
    double max_u = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(nVL) * nE; ++i) {
        max_u = std::max(max_u, std::abs(mesh.u[i]));
    }

    double max_w = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(nVL + 1) * nC; ++i) {
        max_w = std::max(max_w, std::abs(mesh.w[i]));
    }

    double min_theta = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < static_cast<std::size_t>(nVL) * nC; ++i) {
        min_theta = std::min(min_theta, mesh.theta_m[i]);
    }

    RC_ASSERT(max_u < U_BOUND);
    RC_ASSERT(max_w < W_BOUND);
    RC_ASSERT(min_theta > 0.0);

    // Finalize dycore for this trial
    std::memset(errmsg, 0, sizeof(errmsg));
    mpas_dycore_cpp_finalize(errmsg, 512);
}
