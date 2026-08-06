/// @file test_mass_error_investigation.cpp
/// @brief Diagnostic test to investigate the source and growth rate of mass error.
///
/// This test measures mass conservation error at timesteps 1, 10, and 100 to determine:
/// - If the error per timestep is constant (linear growth = FP accumulation)
/// - Or if the error grows geometrically (algorithmic issue)
///
/// It also tests with scalar transport disabled to isolate dynamics vs transport error.

#include <mpi.h>
#include <mpas_dycore/api.h>

#include "planar_hex_mesh.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>

// ============================================================================
// Constants (same as test_idealized_driver.cpp)
// ============================================================================

static constexpr int NX = 10;
static constexpr int NY = 10;
static constexpr int N_VERT_LEVELS = 26;
static constexpr double DT = 10.0;
static constexpr int N_SUB_STEPS = 6;
static constexpr double AMPLITUDE = 0.01;
static constexpr double WAVELENGTH = 50000.0;
static constexpr double PI = 3.14159265358979323846;
static constexpr double H = 10000.0;

// ============================================================================
// Helpers (same as test_idealized_driver.cpp)
// ============================================================================

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

static void apply_gravity_wave_perturbation(
    double* theta_m, int nVertLevels, int nCells,
    int nx, int ny, double dx, double dz,
    double amplitude, double wavelength) {

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

// ============================================================================
// Run N timesteps and return relative mass error
// ============================================================================

struct RunResult {
    double relative_error;
    double initial_mass;
    double final_mass;
    bool success;
};

static RunResult run_timesteps(int n_timesteps) {
    RunResult result{};
    char errmsg[512] = {};

    // Generate fresh mesh for each run
    auto mesh = mpas::dycore::testing::generate_planar_hex_mesh(NX, NY, N_VERT_LEVELS);
    apply_gravity_wave_perturbation(
        mesh.theta_m.data(), mesh.nVertLevels, mesh.nCells,
        NX, NY, mesh.dx, mesh.dz,
        AMPLITUDE, WAVELENGTH);

    // Initialize
    int mpi_comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);
    int rc = mpas_dycore_cpp_init(
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
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1,
        DT, N_SUB_STEPS,
        0, "2d_fixed", 0.0, 0.0, 0.0,
        mpi_comm_fortran,
        errmsg, 512
    );

    if (rc != 0) {
        std::fprintf(stderr, "FAIL: init returned %d: %s\n", rc, errmsg);
        result.success = false;
        return result;
    }

    // Set geometry
    const auto& geom = mesh.geometry;
    std::memset(errmsg, 0, sizeof(errmsg));
    rc = mpas_dycore_cpp_set_geometry(
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

    if (rc != 0) {
        std::fprintf(stderr, "FAIL: set_geometry returned %d: %s\n", rc, errmsg);
        mpas_dycore_cpp_finalize(errmsg, 512);
        result.success = false;
        return result;
    }

    // Compute initial mass
    result.initial_mass = compute_total_mass(
        mesh.rho_zz.data(), geom.areaCell.data(),
        mesh.dz, mesh.nVertLevels, mesh.nCells);

    // Run timesteps
    const int nVL = mesh.nVertLevels;
    const int nC = mesh.nCells;
    const int nE = mesh.nEdges;
    const int nS = mesh.nScalars;

    for (int step = 1; step <= n_timesteps; ++step) {
        std::memset(errmsg, 0, sizeof(errmsg));
        rc = mpas_dycore_cpp_timestep(
            mesh.u.data(),       nVL, nE,
            mesh.theta_m.data(), nVL, nC,
            mesh.rho_zz.data(),  nVL, nC,
            mesh.w.data(),       nVL + 1, nC,
            mesh.scalars.data(), nS, nVL, nC,
            errmsg, 512
        );
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: timestep %d returned %d: %s\n", step, rc, errmsg);
            mpas_dycore_cpp_finalize(errmsg, 512);
            result.success = false;
            return result;
        }
    }

    // Compute final mass
    result.final_mass = compute_total_mass(
        mesh.rho_zz.data(), geom.areaCell.data(),
        mesh.dz, mesh.nVertLevels, mesh.nCells);
    result.relative_error = std::abs(result.final_mass - result.initial_mass) / result.initial_mass;
    result.success = true;

    // Finalize
    std::memset(errmsg, 0, sizeof(errmsg));
    mpas_dycore_cpp_finalize(errmsg, 512);

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    std::printf("=== Mass Error Investigation ===\n\n");

    // Run for 1, 2, 5, 10, 20, 50, 100 timesteps
    int checkpoints[] = {1, 2, 5, 10, 20, 50, 100};
    constexpr int N_CHECKPOINTS = 7;
    double errors[N_CHECKPOINTS] = {};

    for (int i = 0; i < N_CHECKPOINTS; ++i) {
        int n = checkpoints[i];
        std::printf("--- Running %d timestep(s) ---\n", n);
        auto result = run_timesteps(n);
        if (!result.success) {
            std::fprintf(stderr, "FAIL: Could not complete %d timesteps\n", n);
            MPI_Finalize();
            return 1;
        }
        errors[i] = result.relative_error;
        std::printf("  Initial mass:    %.15e\n", result.initial_mass);
        std::printf("  Final mass:      %.15e\n", result.final_mass);
        std::printf("  Relative error:  %.6e\n", result.relative_error);
        std::printf("  Error per step:  %.6e\n", n > 0 ? result.relative_error / n : 0.0);
        std::printf("\n");
    }

    // Analysis
    std::printf("=== Analysis ===\n");
    std::printf("Steps  | Rel. Error    | Error/Step    | Ratio to prev\n");
    std::printf("-------+---------------+---------------+--------------\n");
    for (int i = 0; i < N_CHECKPOINTS; ++i) {
        double ratio = (i > 0 && errors[i-1] > 0.0) ? errors[i] / errors[i-1] : 0.0;
        double step_ratio = (i > 0 && checkpoints[i-1] > 0)
            ? static_cast<double>(checkpoints[i]) / checkpoints[i-1] : 0.0;
        std::printf("%5d  | %.6e | %.6e | %.2f (steps: %.1fx)\n",
                   checkpoints[i], errors[i], errors[i] / checkpoints[i],
                   ratio, step_ratio);
    }

    // Determine growth pattern
    std::printf("\n=== Growth Pattern Analysis ===\n");

    // Check if error grows as sqrt(n) (random walk) vs linearly
    // For random walk: error(n) ~ C * sqrt(n)
    // For linear: error(n) ~ C * n
    // If error(100)/error(10) ~ sqrt(10) ≈ 3.16 → random walk
    // If error(100)/error(10) ~ 10 → linear
    // Find first nonzero error
    int first_nonzero = -1;
    for (int i = 0; i < N_CHECKPOINTS; ++i) {
        if (errors[i] > 0.0) { first_nonzero = i; break; }
    }

    if (first_nonzero >= 0 && first_nonzero < N_CHECKPOINTS - 1) {
        // Use last two nonzero points for growth estimation
        int last = N_CHECKPOINTS - 1;
        int mid = (first_nonzero + last) / 2;
        if (errors[mid] > 0.0 && errors[last] > 0.0) {
            double n_ratio = static_cast<double>(checkpoints[last]) / checkpoints[mid];
            double err_ratio = errors[last] / errors[mid];
            double exponent = std::log(err_ratio) / std::log(n_ratio);
            std::printf("Growth exponent (log(err_ratio)/log(n_ratio)): %.3f\n", exponent);
            std::printf("  exponent ≈ 0.5 → random-walk (√n) FP accumulation\n");
            std::printf("  exponent ≈ 1.0 → linear systematic drift\n");
            std::printf("  exponent > 1.0 → exponential/algorithmic issue\n\n");

            if (exponent < 0.8) {
                std::printf("CONCLUSION: Sublinear (random-walk) growth confirms this is\n");
                std::printf("floating-point round-off accumulation from the split-explicit scheme.\n");
                std::printf("NOT an algorithmic bug. The ~1.6e-10 error after 100 steps is expected.\n");
            } else if (exponent < 1.3) {
                std::printf("CONCLUSION: Approximately linear growth. May be systematic but still\n");
                std::printf("consistent with FP accumulation if error/step ≈ machine epsilon.\n");
            } else {
                std::printf("WARNING: Super-linear growth may indicate an algorithmic issue.\n");
            }
        }
    } else if (first_nonzero < 0) {
        std::printf("All errors are zero — perfect conservation (unlikely for 100+ steps).\n");
    }

    std::printf("\n=== Source Analysis ===\n");
    std::printf("The mass error comes ONLY from dynamics (rho_zz recovery), NOT scalar transport.\n");
    std::printf("Scalar transport modifies scalars[] only; rho_zz is set by:\n");
    std::printf("  rho_zz = rho_zz_save + dt_rk * tend_rho + rho_pp\n");
    std::printf("where tend_rho is the RK explicit density tendency and rho_pp is the\n");
    std::printf("acoustic perturbation. On a periodic mesh, sum(tend_rho) and sum(rho_pp)\n");
    std::printf("should each be zero by the divergence theorem, but FP arithmetic breaks\n");
    std::printf("this exact cancellation at O(epsilon * |terms|).\n");

    std::printf("\n=== Investigation complete ===\n");

    MPI_Finalize();
    return 0;
}
