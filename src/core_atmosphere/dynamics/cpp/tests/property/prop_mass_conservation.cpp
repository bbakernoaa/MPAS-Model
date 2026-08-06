/// @file prop_mass_conservation.cpp
/// @brief Property test for mass conservation through a single timestep.
///
/// **Validates: Requirement 5.5**
///
/// Property 6: Mass conservation through timestep
///   - Vary mesh size and timestep within CFL-stable range
///   - Verify total dry air mass is conserved to relative error < 1.0e-12
///     after one full timestep on a periodic planar hexagonal mesh.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/api.h>

#include "../planar_hex_mesh.hpp"

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <vector>

using namespace mpas::dycore::testing;

// ============================================================================
// MPI Environment
// ============================================================================

class MassConservationMPIEnvironment : public ::testing::Environment {
public:
    void SetUp() override { MPI_Init(nullptr, nullptr); }
    void TearDown() override { MPI_Finalize(); }
};

auto* const mass_conservation_mpi_env =
    ::testing::AddGlobalTestEnvironment(new MassConservationMPIEnvironment);

// ============================================================================
// Helper: compute total dry air mass
// ============================================================================

/// @brief Compute total mass = sum over cells and levels of rho_zz[k,iCell] * areaCell[iCell] * dz
static double compute_total_mass(const double* rho_zz, const double* areaCell,
                                 double dz, int nVertLevels, int nCells) {
    double mass = 0.0;
    for (int k = 0; k < nVertLevels; ++k) {
        for (int iCell = 0; iCell < nCells; ++iCell) {
            std::size_t idx = static_cast<std::size_t>(k) * nCells + iCell;
            mass += rho_zz[idx] * areaCell[iCell] * dz;
        }
    }
    return mass;
}

// ============================================================================
// Property 6: Mass conservation through timestep
// ============================================================================

RC_GTEST_PROP(MassConservation, TotalMassConservedAfterOneTimestep, ()) {
    // Generate random mesh parameters
    int nx = *rc::gen::inRange(3, 9);          // [3, 8]
    int ny = *rc::gen::inRange(3, 9);          // [3, 8]
    int nVertLevels = *rc::gen::inRange(3, 16); // [3, 15]

    // Generate random timestep in [1.0, 20.0] seconds (well within CFL for dx=10km)
    double dt = 1.0 + (*rc::gen::inRange(0, 1901)) / 100.0;  // [1.0, 20.0]

    RC_TAG("nx=" + std::to_string(nx) + " ny=" + std::to_string(ny) +
           " nVL=" + std::to_string(nVertLevels) +
           " dt=" + std::to_string(dt));

    // Generate the planar hex mesh
    auto mesh = generate_planar_hex_mesh(nx, ny, nVertLevels);

    // Initialize via C API
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
        dt, 6,  // number_of_sub_steps
        0,           // config_apply_lbcs (no LBCs for periodic mesh)
        "2d_fixed",  // config_horiz_mixing
        0.0,         // config_h_mom_eddy_visc2
        0.0,         // config_h_mom_eddy_visc4
        0.0,         // config_h_theta_eddy_visc2
        // MPI
        mpi_comm_fortran,
        // Error
        errmsg, 512
    );
    RC_ASSERT_FALSE(rc_init != 0);

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
    RC_ASSERT_FALSE(rc_geom != 0);

    // Compute initial mass = sum of rho_zz * areaCell * dz
    double initial_mass = compute_total_mass(
        mesh.rho_zz.data(), geom.areaCell.data(),
        mesh.dz, mesh.nVertLevels, mesh.nCells);

    RC_ASSERT(initial_mass > 0.0);

    // Execute ONE timestep via C API
    std::memset(errmsg, 0, sizeof(errmsg));
    int rc_ts = mpas_dycore_cpp_timestep(
        mesh.u.data(),
        static_cast<int>(mesh.nVertLevels), static_cast<int>(mesh.nEdges),
        mesh.theta_m.data(),
        static_cast<int>(mesh.nVertLevels), static_cast<int>(mesh.nCells),
        mesh.rho_zz.data(),
        static_cast<int>(mesh.nVertLevels), static_cast<int>(mesh.nCells),
        mesh.w.data(),
        static_cast<int>(mesh.nVertLevels + 1), static_cast<int>(mesh.nCells),
        mesh.scalars.data(),
        static_cast<int>(mesh.nScalars),
        static_cast<int>(mesh.nVertLevels), static_cast<int>(mesh.nCells),
        errmsg, 512
    );
    RC_ASSERT_FALSE(rc_ts != 0);

    // Compute final mass
    double final_mass = compute_total_mass(
        mesh.rho_zz.data(), geom.areaCell.data(),
        mesh.dz, mesh.nVertLevels, mesh.nCells);

    // Verify mass conservation: |final_mass - initial_mass| / initial_mass < 1e-12
    double relative_error = std::abs(final_mass - initial_mass) / initial_mass;
    RC_ASSERT(relative_error < 1.0e-12);

    // Finalize (must be done before next RapidCheck iteration reinitializes)
    std::memset(errmsg, 0, sizeof(errmsg));
    mpas_dycore_cpp_finalize(errmsg, 512);
}
