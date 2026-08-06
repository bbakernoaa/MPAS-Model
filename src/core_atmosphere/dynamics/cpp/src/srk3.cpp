/// @file srk3.cpp
/// @brief Implementation of the SRK3/RK2 time integration loop for the MPAS dynamical core.
///
/// Implements the SRK3Integrator::timestep() method following the Fortran `atm_srk3`
/// structure from mpas_atm_time_integration.F. This orchestrates one full large timestep:
///
///   1. Compute moist coefficients (cqu, cqw)
///   2. Compute vertical implicit coefficients
///   3. Dynamics sub-step loop (dynamics_split_steps iterations):
///      4. RK loop (rk_step = 1..n_rk_stages):
///         a. compute_dyn_tend (tendencies)
///         b. Halo exchange tend_u
///         c. set_smlstep_pert_variables (initialize acoustic perturbations)
///         d. Acoustic sub-step loop:
///            - advance_acoustic_step
///            - divergence_damping_3d
///         e. Recover large-step variables from perturbations
///         f. Halo exchanges for recovered variables
///         g. compute_solve_diagnostics (if needed for next RK step)
///   5. Scalar transport (advance_scalars)
///
/// The RK weights are determined by config_.n_rk_stages:
///   - SRK3 (order 3): rk_timestep = {dt/3, dt/2, dt}, sub_steps = {1, max(1,n/2), n}
///   - RK2  (order 2): rk_timestep = {dt/2, dt/2, dt}, sub_steps = {max(1,n/2), max(1,n/2), n}
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.

#include <mpas_dycore/srk3.hpp>
#include <mpas_dycore/types.hpp>
#include <mpas_dycore/constants.hpp>
#include <mpas_dycore/kernels/recover_state.hpp>
#include <mpas_dycore/kernels/moist_coefficients.hpp>
#include <mpas_dycore/kernels/vert_implicit.hpp>
#include <mpas_dycore/kernels/dyn_tend.hpp>
#include <mpas_dycore/kernels/acoustic.hpp>
#include <mpas_dycore/kernels/divergence_damping.hpp>
#include <mpas_dycore/kernels/diagnostics.hpp>
#include <mpas_dycore/kernels/scalars.hpp>
#include <mpas_dycore/state.hpp>

#include <algorithm>
#include <stdexcept>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace mpas::dycore {

// ============================================================================
// Field ID constants for halo exchange callback dispatch
// ============================================================================

namespace {

constexpr int FIELD_ID_TEND_U     = 1;
constexpr int FIELD_ID_RHO_PP     = 2;
constexpr int FIELD_ID_RTHETA_PP  = 3;
constexpr int FIELD_ID_U          = 4;
constexpr int FIELD_ID_RHO_ZZ     = 5;
constexpr int FIELD_ID_THETA_M    = 6;
constexpr int FIELD_ID_W          = 7;
constexpr int FIELD_ID_SCALARS    = 8;

} // anonymous namespace

// ============================================================================
// Helper: compute RK weights and sub-step counts
// ============================================================================

namespace {

/// @brief Per-stage RK timestep fractions and acoustic sub-step counts.
struct RKWeights {
    std::array<real_type, 3> rk_timestep;       // dt fraction for each RK stage
    std::array<real_type, 3> rk_sub_timestep;   // acoustic dts for each stage
    std::array<int, 3>       number_sub_steps;  // acoustic iterations per stage
};

/// @brief Compute the RK weights based on integration order and config.
///
/// For SRK3 (n_rk_stages == 3):
///   rk_timestep       = {dt/3, dt/2, dt}
///   rk_sub_timestep   = {dt/3, dt/n_sub, dt/n_sub}
///   number_sub_steps  = {1, max(1, n_sub/2), n_sub}
///
/// For RK2 (n_rk_stages == 2):
///   rk_timestep       = {dt/2, dt/2, dt}
///   rk_sub_timestep   = {dt/n_sub, dt/n_sub, dt/n_sub}
///   number_sub_steps  = {max(1, n_sub/2), max(1, n_sub/2), n_sub}
///
/// Note: Even for RK2, we use 3 stages in the loop, but stage 2 (index 1)
/// and stage 3 (index 2) have different weights than SRK3.
RKWeights compute_rk_weights(int n_rk_stages, real_type dt, int n_sub) {
    RKWeights w{};

    if (n_rk_stages == 3) {
        // SRK3 mode
        w.rk_timestep[0] = dt / 3.0;
        w.rk_timestep[1] = dt / 2.0;
        w.rk_timestep[2] = dt;

        w.rk_sub_timestep[0] = dt / 3.0;
        w.rk_sub_timestep[1] = dt / static_cast<real_type>(n_sub);
        w.rk_sub_timestep[2] = dt / static_cast<real_type>(n_sub);

        w.number_sub_steps[0] = 1;
        w.number_sub_steps[1] = std::max(1, n_sub / 2);
        w.number_sub_steps[2] = n_sub;
    } else {
        // RK2 mode (n_rk_stages == 2, but we still iterate 3 stages internally)
        w.rk_timestep[0] = dt / 2.0;
        w.rk_timestep[1] = dt / 2.0;
        w.rk_timestep[2] = dt;

        w.rk_sub_timestep[0] = dt / static_cast<real_type>(n_sub);
        w.rk_sub_timestep[1] = dt / static_cast<real_type>(n_sub);
        w.rk_sub_timestep[2] = dt / static_cast<real_type>(n_sub);

        w.number_sub_steps[0] = std::max(1, n_sub / 2);
        w.number_sub_steps[1] = std::max(1, n_sub / 2);
        w.number_sub_steps[2] = n_sub;
    }

    return w;
}

} // anonymous namespace

// ============================================================================
// SRK3Integrator::timestep
// ============================================================================

void SRK3Integrator::timestep() {
    const auto& cfg = config_;

    // Determine number of dynamics sub-steps for dynamics-transport splitting.
    //
    // When config_split_dynamics_transport is true, the dynamics phase is
    // sub-cycled: each sub-step executes the full RK loop with a reduced
    // timestep dt_dynamics = dt / dynamics_split. Scalar transport then
    // executes ONCE using the full dt and accumulated mass fluxes (ruAvg,
    // wwAvg) averaged over all dynamics sub-steps.
    //
    // When the split is disabled, dynamics_split == 1 and dt_dynamics == dt,
    // so the loop executes once with the full timestep — identical to the
    // non-split behavior.
    //
    // Fortran reference (mpas_atm_time_integration.F):
    //   if (config_split_dynamics_transport) then
    //     dt_dynamics = dt / real(dynamics_split)
    //   else
    //     dynamics_split = 1
    //     dt_dynamics = dt
    //   end if
    const int dynamics_split = cfg.dynamics_transport_split
                                   ? cfg.dynamics_split_steps
                                   : 1;
    const real_type dt_dynamics = cfg.dt / static_cast<real_type>(dynamics_split);

    // Compute RK weights using dt_dynamics (the per-sub-step timestep).
    // When the split is disabled, dt_dynamics == dt so this is unchanged.
    const RKWeights rk = compute_rk_weights(
        cfg.n_rk_stages, dt_dynamics, cfg.number_of_sub_steps);

    // ========================================================================
    // Step 1: Compute moist coefficients (cqu, cqw)
    // ========================================================================
    // kernels::compute_moist_coefficients(policy, cqw, cqu, scalars, mesh,
    //     moist_start, moist_end, nCells, nEdges, nVertLevels);

    // ========================================================================
    // Step 2: Dynamics sub-step loop
    //
    // Each iteration executes one full RK loop (all stages + acoustic steps)
    // using the reduced timestep dt_dynamics. The time-averaged mass fluxes
    // (ruAvg, wwAvg) are accumulated across all dynamics sub-steps for later
    // use by scalar transport.
    // ========================================================================
    for (int dynamics_substep = 0; dynamics_substep < dynamics_split; ++dynamics_substep) {

        // ====================================================================
        // Step 2a: Compute vertical implicit coefficients (first RK step,
        //          or when dts changes between stages)
        // ====================================================================
        // The vertical implicit coefficients depend on dts. For SRK3, dts
        // changes between stage 0 (dts = dt/3) and stages 1-2 (dts = dt/n_sub).
        // We compute them before the first stage and recompute when dts changes.
        //
        // kernels::compute_vert_imp_coefs(policy,
        //     cofwr, cofwz, coftz, cofwt,
        //     a_tri, alpha_tri, gamma_tri, cofrz,
        //     zz, p, t, rb, rtb, pb, rt, cqw, qtot,
        //     rdzw, fzm, fzp, rdzu, etp, ewp,
        //     rk.rk_sub_timestep[0], nCells, nVertLevels);

        real_type prev_dts = rk.rk_sub_timestep[0];

        // ====================================================================
        // Step 2b: RK stage loop
        //
        // Fortran always iterates rk_step = 1, 3 regardless of
        // config_time_integration_order. The RK weights array is 0-indexed
        // so we access rk_timestep[rk_step - 1].
        // ====================================================================
        for (int rk_step = 1; rk_step <= 3; ++rk_step) {
            const real_type dt_rk   = rk.rk_timestep[rk_step - 1];
            const real_type dts_rk  = rk.rk_sub_timestep[rk_step - 1];
            const int n_acoustic    = rk.number_sub_steps[rk_step - 1];

            // ================================================================
            // Recompute vert_imp_coefs if dts changed
            // ================================================================
            if (rk_step > 1 && std::abs(dts_rk - prev_dts) > 1.0e-14) {
                // kernels::compute_vert_imp_coefs(policy,
                //     cofwr, cofwz, coftz, cofwt,
                //     a_tri, alpha_tri, gamma_tri, cofrz,
                //     zz, p, t, rb, rtb, pb, rt, cqw, qtot,
                //     rdzw, fzm, fzp, rdzu, etp, ewp,
                //     dts_rk, nCells, nVertLevels);
                prev_dts = dts_rk;
            }

            // ================================================================
            // Step 2b.i: Compute dynamics tendencies
            // ================================================================
            // kernels::compute_dyn_tend(policy,
            //     tend_u, tend_theta, tend_w,
            //     ke, vorticity, divergence, pv_edge, rho_edge,
            //     u, pressure, rho_zz, theta_m, rtheta_flux,
            //     pp, cqw, cqu, dpdz,
            //     mesh, edgesOnEdge,
            //     areaCell, areaTriangle, dvEdge, dcEdge,
            //     weightsOnEdge, nEdgesOnEdge_arr, invDcEdge, fVertex,
            //     rdzu, fzm, fzp, edgesOnVertex,
            //     nCells, nEdges, nVertices, nVertLevels, maxEdges2,
            //     dyn_tend_config);

            // ================================================================
            // Step 2b.ii: Halo exchange tend_u
            // ================================================================
            // halo.start(tend_u, ...);
            // halo.wait();

            // ================================================================
            // Step 2b.iii: Initialize acoustic perturbation variables
            //              (set_smlstep_pert_variables)
            // ================================================================
            // ru_p = 0, rw_p = 0, rho_pp = 0, rtheta_pp = 0
            // ruAvg = 0, wwAvg = 0

            // ================================================================
            // Step 2b.iv: Apply LBC adjustments (regional configs only)
            // ================================================================
            // if (cfg.config_apply_lbcs) {
            //     kernels::apply_specified_zone(...);
            //     kernels::apply_relaxation_zone(...);
            // }

            // ================================================================
            // Step 2b.v: Acoustic sub-step loop
            // ================================================================
            for (int small_step = 1; small_step <= n_acoustic; ++small_step) {

                // ============================================================
                // Halo exchange rho_pp (needed for pressure gradient)
                // ============================================================
                // halo.start(rho_pp, ...);
                // halo.wait();

                // ============================================================
                // Advance acoustic step
                // ============================================================
                // kernels::advance_acoustic_step(policy,
                //     ru_p, rw_p, rtheta_pp, rho_pp, rtheta_pp_old,
                //     ruAvg, wwAvg,
                //     rho_zz, theta_m, zz, exner, cqu, zxu,
                //     cofwt, coftz, cofwr, cofwz,
                //     a_tri, alpha_tri, gamma_tri,
                //     dss, tend_ru, tend_rho, tend_rt, tend_rw,
                //     w, rw, rw_save,
                //     mesh, edgesOnCell_sign,
                //     invDcEdge, invAreaCell, dvEdge,
                //     cofrz, rdzw, fzm, fzp, etp, etm, ewp, ewm,
                //     specZoneMaskEdge, specZoneMaskCell,
                //     dts_rk, small_step,
                //     nCells, nCellsAll, nEdges, nVertLevels, maxEdges);

                // ============================================================
                // Halo exchange rtheta_pp
                // ============================================================
                // halo.start(rtheta_pp, ...);
                // halo.wait();

                // ============================================================
                // 3D divergence damping
                // ============================================================
                // if (cfg.config_smdiv > 0.0) {
                //     kernels::apply_divergence_damping(policy,
                //         ru_p, divergence_workspace,
                //         mesh, dvEdge, invAreaCell, invDcEdge,
                //         dts_rk, cfg.config_smdiv,
                //         nCells, nEdges, nVertLevels);
                // }

            } // end acoustic sub-step loop

            // ================================================================
            // Step 2b.vi: Recover large-step variables from accumulated
            //             acoustic perturbations
            // ================================================================
            // u_new     = u_old + dt_rk * tend_u + ruAvg / rho_edge
            // rho_new   = rho_old + dt_rk * tend_rho + rho_pp_accum
            // theta_new = theta_old + dt_rk * tend_theta + rtheta_pp_accum / rho_new
            //
            // The recovery uses ruAvg normalized by the number of acoustic steps:
            //   ruAvg /= n_acoustic
            //   wwAvg /= n_acoustic

            // ================================================================
            // Step 2b.vii: Halo exchanges for recovered variables
            // ================================================================
            // halo.start(u, ...);
            // halo.wait();
            // halo.start(rho_zz, ...);
            // halo.wait();
            // halo.start(theta_m, ...);
            // halo.wait();
            // halo.start(w, ...);
            // halo.wait();

            // ================================================================
            // Step 2b.viii: Compute solve diagnostics (unconditional per Fortran)
            // ================================================================
            // kernels::compute_solve_diagnostics(policy,
            //     ke, vorticity, divergence, pv_edge, rho_edge,
            //     u, rho_zz, mesh, ...);

        } // end RK stage loop

    } // end dynamics sub-step loop

    // ========================================================================
    // Step 3: Scalar transport (after all dynamics sub-steps)
    //
    // Transport uses the FULL timestep dt (not dt_dynamics) and the
    // accumulated/normalized ruAvg and wwAvg from all dynamics sub-steps.
    // When dynamics-transport splitting is enabled, the fluxes are averaged:
    //   ruAvg_transport = sum(ruAvg_substep) / dynamics_split
    //   wwAvg_transport = sum(wwAvg_substep) / dynamics_split
    // This ensures mass consistency between dynamics and transport.
    // ========================================================================
    // if (cfg.config_scalar_advection) {
    //     kernels::advance_scalars_mono(policy,
    //         scalars, scalars_old, ruAvg, wwAvg,
    //         rho_zz_old, rho_zz_new,
    //         mesh, advCellsForEdge, adv_coefs, adv_coefs_3rd,
    //         nAdvCellsForEdge, cellsOnCell,
    //         nCells, nEdges, nVertLevels, nScalars,
    //         cfg.coef_3rd_order, config_positive_definite);
    //     // NOTE: dt argument to transport is cfg.dt (full timestep),
    //     //       NOT dt_dynamics. The transport step spans the entire
    //     //       large timestep interval regardless of dynamics sub-cycling.
    // }

    // ========================================================================
    // Step 4: Velocity reconstruction (after final stage)
    // ========================================================================
    // kernels::reconstruct_velocity(policy,
    //     uReconstructX, uReconstructY, uReconstructZ,
    //     uReconstructZonal, uReconstructMeridional,
    //     u, coeffs_reconstruct,
    //     mesh, nCells, nEdges, nVertLevels, maxEdges);
}

// ============================================================================
// SRK3Integrator::advance_dynamics
// ============================================================================

void SRK3Integrator::advance_dynamics([[maybe_unused]] int rk_step) {
    // Placeholder for per-stage dynamics advancement.
    // The full logic is currently inlined in timestep() above.
    // This will be factored out when workspace and field references are finalized.
}

// ============================================================================
// SRK3Integrator::advance_transport
// ============================================================================

void SRK3Integrator::advance_transport() {
    // ========================================================================
    // Dynamics-Transport Split Semantics
    // ========================================================================
    //
    // When config_split_dynamics_transport is enabled, scalar transport uses:
    //   - The FULL timestep dt (NOT the reduced dt_dynamics = dt / dynamics_split)
    //   - The ACCUMULATED time-averaged mass fluxes (ruAvg, wwAvg) averaged
    //     over ALL dynamics sub-steps
    //
    // The dynamics sub-step loop (in timestep()) executes dynamics_split_steps
    // iterations, each producing its own ruAvg/wwAvg. These are accumulated
    // and then normalized (divided by dynamics_split_steps) before being passed
    // here. This ensures mass consistency: the transport step sees the mean
    // mass flux that the dynamics produced over the full timestep interval.
    //
    // When the split is disabled (dynamics_split == 1), the behavior is
    // identical to inline transport: ruAvg/wwAvg come from the single RK loop
    // and dt is used directly.
    //
    // Fortran reference (mpas_atm_time_integration.F):
    //   call atm_advance_scalars(..., dt, ruAvg, wwAvg, ...)
    //   ! where ruAvg/wwAvg are the accumulated/normalized averages
    //
    // Implementation will be wired in a subsequent task when advance_scalars_mono
    // kernel is connected.
}

// ============================================================================
// SRK3Integrator::timestep (parameterized version)
// ============================================================================

void SRK3Integrator::timestep(
    Field2D<> u,
    Field2D<> theta_m,
    Field2D<> rho_zz,
    Field2D<> w,
    Field3D<> scalars,
    SRK3Workspace& workspace,
    const MeshGeometry& geometry,
    const MeshConnectivity& mesh,
    HaloExchange& halo,
    const CSRHaloDescriptor& halo_desc,
    MPI_Comm comm,
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels) {

    const auto& cfg = config_;

    // Determine dynamics sub-step count for dynamics-transport splitting.
    const int dynamics_split = cfg.dynamics_transport_split
                                   ? cfg.dynamics_split_steps
                                   : 1;
    const real_type dt_dynamics = cfg.dt / static_cast<real_type>(dynamics_split);

    // Compute RK weights using dt_dynamics.
    // (Re-using the same local helper as the no-arg version)
    struct RKWeightsLocal {
        std::array<real_type, 3> rk_timestep;
        std::array<real_type, 3> rk_sub_timestep;
        std::array<int, 3>       number_sub_steps;
    };

    auto compute_rk = [](int n_rk_stages, real_type dt, int n_sub) -> RKWeightsLocal {
        RKWeightsLocal wt{};
        if (n_rk_stages == 3) {
            wt.rk_timestep[0] = dt / 3.0;
            wt.rk_timestep[1] = dt / 2.0;
            wt.rk_timestep[2] = dt;
            wt.rk_sub_timestep[0] = dt / 3.0;
            wt.rk_sub_timestep[1] = dt / static_cast<real_type>(n_sub);
            wt.rk_sub_timestep[2] = dt / static_cast<real_type>(n_sub);
            wt.number_sub_steps[0] = 1;
            wt.number_sub_steps[1] = std::max(1, n_sub / 2);
            wt.number_sub_steps[2] = n_sub;
        } else {
            wt.rk_timestep[0] = dt / 2.0;
            wt.rk_timestep[1] = dt / 2.0;
            wt.rk_timestep[2] = dt;
            wt.rk_sub_timestep[0] = dt / static_cast<real_type>(n_sub);
            wt.rk_sub_timestep[1] = dt / static_cast<real_type>(n_sub);
            wt.rk_sub_timestep[2] = dt / static_cast<real_type>(n_sub);
            wt.number_sub_steps[0] = std::max(1, n_sub / 2);
            wt.number_sub_steps[1] = std::max(1, n_sub / 2);
            wt.number_sub_steps[2] = n_sub;
        }
        return wt;
    };

    const auto rk = compute_rk(cfg.n_rk_stages, dt_dynamics, cfg.number_of_sub_steps);

    // --- Pre-compute connectivity conversion for recover_state_perturbation ---
    // These are constant throughout the timestep (mesh doesn't change).
    {
        const auto cellsOnEdge_size = static_cast<std::size_t>(2) * static_cast<std::size_t>(nEdges);
        workspace.cellsOnEdge_real_storage.resize(cellsOnEdge_size);
        for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
            workspace.cellsOnEdge_real_storage[static_cast<std::size_t>(0) + static_cast<std::size_t>(iEdge) * 2] =
                static_cast<real_type>(mesh.cellsOnEdge[iEdge, 0]);
            workspace.cellsOnEdge_real_storage[static_cast<std::size_t>(1) + static_cast<std::size_t>(iEdge) * 2] =
                static_cast<real_type>(mesh.cellsOnEdge[iEdge, 1]);
        }

        const auto edgesOnCell_size = static_cast<std::size_t>(geometry.maxEdges_) * static_cast<std::size_t>(nCells);
        workspace.edgesOnCell_real_storage.resize(edgesOnCell_size);
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type i = 0; i < geometry.maxEdges_; ++i) {
                workspace.edgesOnCell_real_storage[static_cast<std::size_t>(i) + static_cast<std::size_t>(iCell) * static_cast<std::size_t>(geometry.maxEdges_)] =
                    static_cast<real_type>(mesh.edgesOnCell[iCell, i]);
            }
        }

        workspace.nEdgesOnCell_real_storage.resize(static_cast<std::size_t>(nCells));
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            workspace.nEdgesOnCell_real_storage[static_cast<std::size_t>(iCell)] =
                static_cast<real_type>(geometry.nEdgesOnCell[static_cast<std::size_t>(iCell)]);
        }

        // Build edgesOnVertex from verticesOnEdge (transpose relationship).
        // For each vertex, collect which edges have that vertex as an endpoint.
        // vertexDegree is typically 3 for a Voronoi dual mesh.
        const index_type vertexDegree = geometry.vertexDegree;
        if (geometry.edgesOnVertex_storage.empty()) {
            auto& eov = const_cast<MeshGeometry&>(geometry).edgesOnVertex_storage;
            eov.assign(
                static_cast<std::size_t>(nVertices) * static_cast<std::size_t>(vertexDegree),
                INVALID_INDEX);

            // Count edges per vertex and fill connectivity
            std::vector<index_type> vertCount(static_cast<std::size_t>(nVertices), 0);
            for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
                for (index_type j = 0; j < 2; ++j) {
                    auto iVertex = mesh.verticesOnEdge[iEdge, j];
                    if (iVertex != INVALID_INDEX && iVertex < nVertices) {
                        auto slot = vertCount[static_cast<std::size_t>(iVertex)];
                        if (slot < vertexDegree) {
                            eov[static_cast<std::size_t>(iVertex) * static_cast<std::size_t>(vertexDegree)
                                + static_cast<std::size_t>(slot)] = iEdge;
                            vertCount[static_cast<std::size_t>(iVertex)]++;
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Pre-compute moist coefficients (cqw, cqu) from scalars — once before
    // the dynamics sub-step loop (per Fortran atm_srk3 structure).
    // ========================================================================
    {
        using UncheckedField2D = Field2D<default_layout, unchecked_accessor>;
        using ConstUncheckedField3D = ConstField3D<default_layout, unchecked_accessor>;
        auto cqw_view = UncheckedField2D(workspace.cqw_storage.data(), nVertLevels + 1, nCells);
        auto cqu_view = UncheckedField2D(workspace.cqu_storage.data(), nVertLevels, nEdges);
        auto scalars_const = ConstUncheckedField3D(
            scalars.data_handle(), scalars.extent(0), nVertLevels, nCells);

        kernels::compute_moist_coefficients<default_layout>(SerialPolicy{},
            cqw_view, cqu_view, scalars_const, mesh,
            cfg.moist_start, cfg.moist_end, nCells, nEdges, nVertLevels);
    }

    // ========================================================================
    // Dynamics sub-step loop
    // ========================================================================
    for (int dynamics_substep = 0; dynamics_substep < dynamics_split; ++dynamics_substep) {

        // ====================================================================
        // Step 2a: Compute perturbation state from full fields and save ONCE
        // before the RK loop (per Fortran atm_rk_integration_setup).
        //
        // This replaces the per-stage full-field save that was previously
        // done inside advance_dynamics().
        // ====================================================================

        // Compute rho_p = rho_zz - rho_base
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                workspace.rho_p_storage[static_cast<std::size_t>(k) + static_cast<std::size_t>(iCell) * static_cast<std::size_t>(nVertLevels)] =
                    rho_zz[k, iCell] - geometry.rho_base[static_cast<std::size_t>(k) + static_cast<std::size_t>(iCell) * static_cast<std::size_t>(nVertLevels)];
            }
        }

        // Compute rtheta_p = theta_m * rho_zz - rtheta_base
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                workspace.rtheta_p_storage[static_cast<std::size_t>(k) + static_cast<std::size_t>(iCell) * static_cast<std::size_t>(nVertLevels)] =
                    theta_m[k, iCell] * rho_zz[k, iCell]
                    - geometry.rtheta_base[static_cast<std::size_t>(k) + static_cast<std::size_t>(iCell) * static_cast<std::size_t>(nVertLevels)];
            }
        }

        // Compute ru = 0.5 * u * (rho_zz[cell1] + rho_zz[cell2])
        for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
            auto cell1 = mesh.cellsOnEdge[iEdge, 0];
            auto cell2 = mesh.cellsOnEdge[iEdge, 1];
            for (index_type k = 0; k < nVertLevels; ++k) {
                workspace.ru_storage[static_cast<std::size_t>(k) + static_cast<std::size_t>(iEdge) * static_cast<std::size_t>(nVertLevels)] =
                    0.5 * u[k, iEdge] * (rho_zz[k, cell1] + rho_zz[k, cell2]);
            }
        }

        // Compute rw from w, rho_zz, zz (reverse of recovery Phase 1:
        //   rw = w * (fzm*rho_zz[k]*zz[k] + fzp*rho_zz[k-1]*zz[k-1])
        // with rigid-lid BCs at top and bottom.
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            const auto nVp1 = static_cast<std::size_t>(nVertLevels + 1);
            // Bottom rigid lid
            workspace.rw_storage[static_cast<std::size_t>(0) + static_cast<std::size_t>(iCell) * nVp1] = 0.0;
            // Interior levels
            for (index_type k = 1; k < nVertLevels; ++k) {
                const auto zz_k   = geometry.zz[static_cast<std::size_t>(k)   + static_cast<std::size_t>(iCell) * static_cast<std::size_t>(nVertLevels)];
                const auto zz_km1 = geometry.zz[static_cast<std::size_t>(k-1) + static_cast<std::size_t>(iCell) * static_cast<std::size_t>(nVertLevels)];
                const auto rho_k   = rho_zz[k, iCell];
                const auto rho_km1 = rho_zz[k-1, iCell];
                const auto rho_at_w = geometry.fzm[static_cast<std::size_t>(k)] * rho_k * zz_k
                                    + geometry.fzp[static_cast<std::size_t>(k)] * rho_km1 * zz_km1;
                workspace.rw_storage[static_cast<std::size_t>(k) + static_cast<std::size_t>(iCell) * nVp1] =
                    w[k, iCell] * rho_at_w;
            }
            // Top rigid lid
            workspace.rw_storage[static_cast<std::size_t>(nVertLevels) + static_cast<std::size_t>(iCell) * nVp1] = 0.0;
        }

        // Save perturbation state (copy to save arrays) — happens ONCE per dynamics sub-step
        std::copy(workspace.rho_p_storage.begin(), workspace.rho_p_storage.end(),
                  workspace.rho_p_save_storage.begin());
        std::copy(workspace.rtheta_p_storage.begin(), workspace.rtheta_p_storage.end(),
                  workspace.rtheta_p_save_storage.begin());
        std::copy(workspace.ru_storage.begin(), workspace.ru_storage.end(),
                  workspace.ru_save_storage.begin());
        std::copy(workspace.rw_storage.begin(), workspace.rw_storage.end(),
                  workspace.rw_save_storage.begin());

        // ====================================================================
        // Step 2b: RK stage loop
        //
        // Fortran always iterates rk_step = 1, 3 regardless of
        // config_time_integration_order. The RK weights array is 0-indexed
        // so we access rk_timestep[rk_step - 1].
        // ====================================================================
        for (int rk_step = 1; rk_step <= 3; ++rk_step) {
            const real_type dt_rk  = rk.rk_timestep[rk_step - 1];
            const real_type dts_rk = rk.rk_sub_timestep[rk_step - 1];
            const int n_acoustic   = rk.number_sub_steps[rk_step - 1];

            advance_dynamics(rk_step, dt_rk, dts_rk, n_acoustic,
                             u, theta_m, rho_zz, w,
                             workspace, geometry, mesh,
                             halo, halo_desc, comm,
                             nCells, nEdges, nVertices, nVertLevels);
        }
    }

    // ========================================================================
    // Scalar transport (after all dynamics sub-steps)
    // ========================================================================
    if (cfg.config_scalar_advection) {
        advance_transport(scalars, workspace, geometry, mesh,
                          halo, halo_desc, comm,
                          nCells, nEdges, nVertLevels);
    }
}

// ============================================================================
// SRK3Integrator::advance_dynamics (parameterized version)
// ============================================================================

void SRK3Integrator::advance_dynamics(
    int rk_step,
    real_type dt_rk,
    real_type dts_rk,
    int n_acoustic,
    Field2D<> u,
    Field2D<> theta_m,
    Field2D<> rho_zz,
    Field2D<> w,
    SRK3Workspace& workspace,
    const MeshGeometry& geometry,
    const MeshConnectivity& mesh,
    HaloExchange& halo,
    const CSRHaloDescriptor& halo_desc,
    MPI_Comm comm,
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels) {

    const auto& cfg = config_;

    // Type aliases for convenience
    using UncheckedField2D = Field2D<default_layout, unchecked_accessor>;
    using ConstUncheckedField2D = ConstField2D<default_layout, unchecked_accessor>;
    using ConstSpan1D = kernels::ConstSpan1D;
    using Span1D = kernels::Span1D;

    // ========================================================================
    // Construct reusable mdspan views from workspace, geometry, and mesh
    // ========================================================================

    // --- Workspace 2D views (mutable) ---
    auto ru_p_view = UncheckedField2D(workspace.ru_p_storage.data(), nVertLevels, nEdges);
    auto rw_p_view_mut = UncheckedField2D(workspace.rw_p_storage.data(), nVertLevels + 1, nCells);
    auto rtheta_pp_view_mut = UncheckedField2D(workspace.rtheta_pp_storage.data(), nVertLevels, nCells);
    auto rho_pp_view_mut = UncheckedField2D(workspace.rho_pp_storage.data(), nVertLevels, nCells);
    auto rtheta_pp_old_view = UncheckedField2D(workspace.rtheta_pp_old_storage.data(), nVertLevels, nCells);
    auto ruAvg_view = UncheckedField2D(workspace.ruAvg_storage.data(), nVertLevels, nEdges);
    auto wwAvg_view = UncheckedField2D(workspace.wwAvg_storage.data(), nVertLevels + 1, nCells);
    auto tend_u_view = UncheckedField2D(workspace.tend_u_storage.data(), nVertLevels, nEdges);
    auto tend_theta_view = UncheckedField2D(workspace.tend_theta_storage.data(), nVertLevels, nCells);
    auto tend_w_view = UncheckedField2D(workspace.tend_w_storage.data(), nVertLevels + 1, nCells);
    auto tend_rho_view = UncheckedField2D(workspace.tend_rho_storage.data(), nVertLevels, nCells);
    auto ke_view = UncheckedField2D(workspace.ke_storage.data(), nVertLevels, nCells);
    auto vorticity_view = UncheckedField2D(workspace.vorticity_storage.data(), nVertLevels, nVertices);
    auto divergence_view = UncheckedField2D(workspace.divergence_storage.data(), nVertLevels, nCells);
    auto pv_edge_view = UncheckedField2D(workspace.pv_edge_storage.data(), nVertLevels, nEdges);
    auto rho_edge_view = UncheckedField2D(workspace.rho_edge_storage.data(), nVertLevels, nEdges);
    auto cofwr_view = UncheckedField2D(workspace.cofwr_storage.data(), nVertLevels, nCells);
    auto cofwz_view = UncheckedField2D(workspace.cofwz_storage.data(), nVertLevels, nCells);
    auto coftz_view = UncheckedField2D(workspace.coftz_storage.data(), nVertLevels + 1, nCells);
    auto cofwt_view = UncheckedField2D(workspace.cofwt_storage.data(), nVertLevels, nCells);
    auto a_tri_view = UncheckedField2D(workspace.a_tri_storage.data(), nVertLevels, nCells);
    auto alpha_tri_view = UncheckedField2D(workspace.alpha_tri_storage.data(), nVertLevels, nCells);
    auto gamma_tri_view = UncheckedField2D(workspace.gamma_tri_storage.data(), nVertLevels, nCells);
    auto exner_view = UncheckedField2D(workspace.exner_storage.data(), nVertLevels, nCells);
    auto pressure_view = UncheckedField2D(workspace.pressure_storage.data(), nVertLevels, nCells);
    auto rtheta_flux_view = UncheckedField2D(workspace.rtheta_flux_storage.data(), nVertLevels, nEdges);
    auto pp_view = UncheckedField2D(workspace.pp_storage.data(), nVertLevels, nCells);
    auto dpdz_view = UncheckedField2D(workspace.dpdz_storage.data(), nVertLevels, nCells);
    auto zxu_view = UncheckedField2D(workspace.zxu_storage.data(), nVertLevels, nEdges);
    auto dss_view = UncheckedField2D(workspace.dss_storage.data(), nVertLevels, nCells);
    auto qtot_view = UncheckedField2D(workspace.qtot_storage.data(), nVertLevels, nCells);
    auto rw_view = UncheckedField2D(workspace.rw_storage.data(), nVertLevels + 1, nCells);
    auto rw_save_view_field = UncheckedField2D(workspace.rw_save_storage.data(), nVertLevels + 1, nCells);

    // --- Workspace 1D mutable view ---
    auto cofrz_span = Span1D(workspace.cofrz_storage.data(), nVertLevels);

    // --- Geometry 2D const views ---
    auto zz_view = ConstUncheckedField2D(geometry.zz.data(), nVertLevels, nCells);
    auto rb_view = ConstUncheckedField2D(geometry.rb.data(), nVertLevels, nCells);
    auto rtb_view = ConstUncheckedField2D(geometry.rtb.data(), nVertLevels, nCells);
    auto pb_view = ConstUncheckedField2D(geometry.pb.data(), nVertLevels, nCells);
    auto edgesOnCell_sign_view = ConstUncheckedField2D(
        geometry.edgesOnCell_sign.data(), geometry.maxEdges_, nCells);

    // --- Geometry 1D const spans ---
    auto rdzw_span = ConstSpan1D(geometry.rdzw.data(), nVertLevels);
    auto rdzu_span = ConstSpan1D(geometry.rdzu.data(), nVertLevels);
    auto fzm_span = ConstSpan1D(geometry.fzm.data(), nVertLevels);
    auto fzp_span = ConstSpan1D(geometry.fzp.data(), nVertLevels);
    auto etp_span = ConstSpan1D(geometry.etp.data(), nVertLevels);
    auto etm_span = ConstSpan1D(geometry.etm.data(), nVertLevels);
    auto ewp_span = ConstSpan1D(geometry.ewp.data(), nVertLevels + 1);
    auto ewm_span = ConstSpan1D(geometry.ewm.data(), nVertLevels + 1);
    auto areaCell_span = ConstSpan1D(geometry.areaCell.data(), nCells);
    auto invAreaCell_span = ConstSpan1D(geometry.invAreaCell.data(), nCells);
    auto dvEdge_span = ConstSpan1D(geometry.dvEdge.data(), nEdges);
    auto dcEdge_span = ConstSpan1D(geometry.dcEdge.data(), nEdges);
    auto invDcEdge_span = ConstSpan1D(geometry.invDcEdge.data(), nEdges);
    auto specZoneMaskEdge_span = ConstSpan1D(geometry.specZoneMaskEdge.data(), nEdges);
    auto specZoneMaskCell_span = ConstSpan1D(geometry.specZoneMaskCell.data(), nCells);
    auto fVertex_span = ConstSpan1D(geometry.fVertex.data(), nVertices);
    auto areaTriangle_span = ConstSpan1D(geometry.areaTriangle.data(), nVertices);
    auto weightsOnEdge_span = ConstSpan1D(
        geometry.weightsOnEdge.data(),
        static_cast<index_type>(geometry.weightsOnEdge.size()));

    // --- Geometry index_type 1D spans ---
    using ConstIndexSpan1D = std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>;
    auto nEdgesOnEdge_span = ConstIndexSpan1D(geometry.nEdgesOnEdge.data(), nEdges);
    auto nAdvCellsForEdge_span = ConstIndexSpan1D(geometry.nAdvCellsForEdge.data(), nEdges);

    // --- Geometry connectivity views (2D mdspan over flat storage) ---
    auto edgesOnEdge_view = ConnectivityView(
        geometry.edgesOnEdge_storage.data(), nEdges, geometry.maxEdges2);
    auto advCellsForEdge_view = ConnectivityView(
        geometry.advCellsForEdge_storage.data(), nEdges, geometry.maxAdvCells);
    auto edgesOnVertex_view = ConnectivityView(
        geometry.edgesOnVertex_storage.data(), nVertices, geometry.vertexDegree);

    // --- Geometry 2D const field views (advection coefficients) ---
    auto adv_coefs_view = ConstUncheckedField2D(
        geometry.adv_coefs.data(), geometry.maxAdvCells, nEdges);
    auto adv_coefs_3rd_view = ConstUncheckedField2D(
        geometry.adv_coefs_3rd.data(), geometry.maxAdvCells, nEdges);

    // --- Workspace const views (moist coefficients, computed before RK loop) ---
    auto cqw_view = ConstUncheckedField2D(workspace.cqw_storage.data(), nVertLevels + 1, nCells);
    auto cqu_view = ConstUncheckedField2D(workspace.cqu_storage.data(), nVertLevels, nEdges);

    // --- Workspace perturbation rtheta (for vert imp coefs) ---
    auto rtheta_p_view = ConstUncheckedField2D(workspace.rtheta_p_storage.data(), nVertLevels, nCells);

    // --- Prognostic field const views ---
    auto u_const = ConstUncheckedField2D(u.data_handle(), nVertLevels, nEdges);
    auto theta_m_const = ConstUncheckedField2D(theta_m.data_handle(), nVertLevels, nCells);
    auto rho_zz_const = ConstUncheckedField2D(rho_zz.data_handle(), nVertLevels, nCells);
    auto w_const = ConstUncheckedField2D(w.data_handle(), nVertLevels + 1, nCells);

    // ========================================================================
    // Step 3: Compute vertical implicit coefficients
    // ========================================================================
    kernels::compute_vert_imp_coefs<default_layout>(SerialPolicy{},
        cofwr_view, cofwz_view, coftz_view, cofwt_view,
        a_tri_view, alpha_tri_view, gamma_tri_view, cofrz_span,
        zz_view, exner_view, theta_m_const, rb_view, rtb_view, pb_view,
        rtheta_p_view, cqw_view, qtot_view,
        rdzw_span, fzm_span, fzp_span, rdzu_span, etp_span, ewp_span,
        dts_rk, nCells, nVertLevels);

    // ========================================================================
    // Step 4: Compute dynamics tendencies
    // ========================================================================
    {
        auto& dstate = get_dycore_state();
        if (!dstate.use_cpp_compute_dyn_tend) {
            if (dstate.compute_dyn_tend_cb == nullptr) {
                throw std::runtime_error(
                    "compute_dyn_tend callback required but not registered");
            }
            dstate.compute_dyn_tend_cb(
                u.data_handle(),
                theta_m.data_handle(),
                rho_zz.data_handle(),
                w.data_handle(),
                workspace.tend_u_storage.data(),
                workspace.tend_theta_storage.data(),
                workspace.tend_rho_storage.data(),
                workspace.tend_w_storage.data(),
                static_cast<int>(nVertLevels),
                static_cast<int>(nCells),
                static_cast<int>(nEdges),
                rk_step,
                dt_rk);
        } else {
            kernels::DynTendConfig dyn_tend_config{};
            dyn_tend_config.rayleigh_damp_u = cfg.config_rayleigh_damp_u;
            dyn_tend_config.n_rayleigh_damp_levels = static_cast<index_type>(cfg.config_number_rayleigh_damp_u_levels);
            dyn_tend_config.rayleigh_timescale_days = cfg.config_rayleigh_damp_u_timescale_days;

            kernels::compute_dyn_tend<default_layout>(SerialPolicy{},
                tend_u_view, tend_theta_view, tend_w_view,
                ke_view, vorticity_view, divergence_view, pv_edge_view, rho_edge_view,
                u_const, pressure_view, rho_zz_const, theta_m_const, rtheta_flux_view,
                pp_view, cqw_view, cqu_view, dpdz_view,
                mesh, edgesOnEdge_view,
                areaCell_span, areaTriangle_span, dvEdge_span, dcEdge_span,
                weightsOnEdge_span, nEdgesOnEdge_span, invDcEdge_span, fVertex_span,
                rdzu_span, fzm_span, fzp_span, edgesOnVertex_view,
                nCells, nEdges, nVertices, nVertLevels, geometry.maxEdges2,
                dyn_tend_config);
        }
    }

    // ========================================================================
    // Step 5: Halo exchange tend_u
    // ========================================================================
    {
        auto& dstate = get_dycore_state();
        if (!dstate.use_cpp_halo_exchange && dstate.halo_exchange_cb != nullptr) {
            dstate.halo_exchange_cb(
                workspace.tend_u_storage.data(), FIELD_ID_TEND_U,
                static_cast<int>(nVertLevels), static_cast<int>(nEdges));
        } else {
            std::array<int, 1> layers_arr = {0};
            std::span<const int> layers_span(layers_arr);
            halo.start<default_layout>(tend_u_view, halo_desc, layers_span, comm);
            halo.wait();
        }
    }

    // ========================================================================
    // Step 6: Zero perturbation fields (set_smlstep_pert_variables)
    // ========================================================================
    std::fill(workspace.ru_p_storage.begin(), workspace.ru_p_storage.end(), 0.0);
    std::fill(workspace.rw_p_storage.begin(), workspace.rw_p_storage.end(), 0.0);
    std::fill(workspace.rho_pp_storage.begin(), workspace.rho_pp_storage.end(), 0.0);
    std::fill(workspace.rtheta_pp_storage.begin(), workspace.rtheta_pp_storage.end(), 0.0);
    std::fill(workspace.ruAvg_storage.begin(), workspace.ruAvg_storage.end(), 0.0);
    std::fill(workspace.wwAvg_storage.begin(), workspace.wwAvg_storage.end(), 0.0);

    // ========================================================================
    // Step 7: Acoustic sub-step loop
    // ========================================================================
    for (int small_step = 1; small_step <= n_acoustic; ++small_step) {

        // ====================================================================
        // Step 7a: Halo exchange rho_pp (needed for pressure gradient)
        // ====================================================================
        {
            auto& dstate = get_dycore_state();
            if (!dstate.use_cpp_halo_exchange && dstate.halo_exchange_cb != nullptr) {
                dstate.halo_exchange_cb(
                    workspace.rho_pp_storage.data(), FIELD_ID_RHO_PP,
                    static_cast<int>(nVertLevels), static_cast<int>(nCells));
            } else {
                std::array<int, 1> layers_arr = {0};
                std::span<const int> layers_span(layers_arr);
                halo.start<default_layout>(rho_pp_view_mut, halo_desc, layers_span, comm);
                halo.wait();
            }
        }

        // ====================================================================
        // Step 7b: Advance acoustic step
        // ====================================================================
        {
            auto& dstate = get_dycore_state();
            if (!dstate.use_cpp_advance_acoustic_step) {
                if (dstate.advance_acoustic_step_cb == nullptr) {
                    throw std::runtime_error(
                        "advance_acoustic_step callback required but not registered");
                }
                dstate.advance_acoustic_step_cb(
                    workspace.ru_p_storage.data(),
                    workspace.rw_p_storage.data(),
                    workspace.rtheta_pp_storage.data(),
                    workspace.rho_pp_storage.data(),
                    static_cast<int>(nVertLevels),
                    static_cast<int>(nCells),
                    static_cast<int>(nEdges),
                    small_step - 1,  // 0-based for callback
                    dts_rk);
            } else {
                kernels::advance_acoustic_step<default_layout>(SerialPolicy{},
                    ru_p_view, rw_p_view_mut, rtheta_pp_view_mut, rho_pp_view_mut,
                    rtheta_pp_old_view, ruAvg_view, wwAvg_view,
                    rho_zz_const, theta_m_const, zz_view, exner_view, cqu_view, zxu_view,
                    cofwt_view, coftz_view, cofwr_view, cofwz_view,
                    a_tri_view, alpha_tri_view, gamma_tri_view,
                    dss_view,
                    ConstUncheckedField2D(workspace.tend_u_storage.data(), nVertLevels, nEdges),
                    ConstUncheckedField2D(workspace.tend_rho_storage.data(), nVertLevels, nCells),
                    ConstUncheckedField2D(workspace.tend_theta_storage.data(), nVertLevels, nCells),
                    ConstUncheckedField2D(workspace.tend_w_storage.data(), nVertLevels + 1, nCells),
                    w_const,
                    ConstUncheckedField2D(workspace.rw_storage.data(), nVertLevels + 1, nCells),
                    ConstUncheckedField2D(workspace.rw_save_storage.data(), nVertLevels + 1, nCells),
                    mesh, edgesOnCell_sign_view,
                    invDcEdge_span, invAreaCell_span, dvEdge_span,
                    ConstSpan1D(workspace.cofrz_storage.data(), nVertLevels),
                    rdzw_span, fzm_span, fzp_span, etp_span, etm_span, ewp_span, ewm_span,
                    specZoneMaskEdge_span, specZoneMaskCell_span,
                    dts_rk, static_cast<index_type>(small_step),
                    nCells, nCells, nEdges, nVertLevels,
                    geometry.maxEdges_);
            }
        }

        // ====================================================================
        // Step 7c: Halo exchange rtheta_pp
        // ====================================================================
        {
            auto& dstate = get_dycore_state();
            if (!dstate.use_cpp_halo_exchange && dstate.halo_exchange_cb != nullptr) {
                dstate.halo_exchange_cb(
                    workspace.rtheta_pp_storage.data(), FIELD_ID_RTHETA_PP,
                    static_cast<int>(nVertLevels), static_cast<int>(nCells));
            } else {
                std::array<int, 1> layers_arr = {0};
                std::span<const int> layers_span(layers_arr);
                halo.start<default_layout>(rtheta_pp_view_mut, halo_desc, layers_span, comm);
                halo.wait();
            }
        }

        // ====================================================================
        // Step 7d: 3D divergence damping
        // ====================================================================
        if (cfg.config_h_mom_eddy_visc2 > 0.0) {
            kernels::apply_divergence_damping<default_layout>(SerialPolicy{},
                ru_p_view, divergence_view,
                mesh, dvEdge_span, invAreaCell_span, invDcEdge_span,
                dts_rk, cfg.config_h_mom_eddy_visc2,
                nCells, nEdges, nVertLevels);
        }

    } // end acoustic sub-step loop

    // ========================================================================
    // Step 8: Recover large-step variables using Fortran perturbation-variable
    //         formulation (recover_state_perturbation)
    // ========================================================================
    {
        // Output prognostic fields (reinterpret as unchecked for the kernel)
        auto u_out = UncheckedField2D(u.data_handle(), nVertLevels, nEdges);
        auto rho_zz_out = UncheckedField2D(rho_zz.data_handle(), nVertLevels, nCells);
        auto theta_m_out = UncheckedField2D(theta_m.data_handle(), nVertLevels, nCells);
        auto w_out = UncheckedField2D(w.data_handle(), nVertLevels + 1, nCells);

        // Output diagnostics (updated on rk_step==3)
        auto pressure_p_view_out = UncheckedField2D(workspace.pp_storage.data(), nVertLevels, nCells);

        // Perturbation state saves (from before RK loop)
        auto rho_p_save_view = ConstUncheckedField2D(
            workspace.rho_p_save_storage.data(), nVertLevels, nCells);
        auto rtheta_p_save_view = ConstUncheckedField2D(
            workspace.rtheta_p_save_storage.data(), nVertLevels, nCells);
        auto ru_save_view = ConstUncheckedField2D(
            workspace.ru_save_storage.data(), nVertLevels, nEdges);
        auto rw_save_view_const = ConstUncheckedField2D(
            workspace.rw_save_storage.data(), nVertLevels + 1, nCells);

        // Acoustic sub-step perturbations (const views for recovery)
        auto rho_pp_const = ConstUncheckedField2D(
            workspace.rho_pp_storage.data(), nVertLevels, nCells);
        auto rtheta_pp_const = ConstUncheckedField2D(
            workspace.rtheta_pp_storage.data(), nVertLevels, nCells);
        auto ru_p_const = ConstUncheckedField2D(
            workspace.ru_p_storage.data(), nVertLevels, nEdges);
        auto rw_p_const = ConstUncheckedField2D(
            workspace.rw_p_storage.data(), nVertLevels + 1, nCells);

        // Base-state profiles
        auto rho_base_view = ConstUncheckedField2D(
            geometry.rho_base.data(), nVertLevels, nCells);
        auto rtheta_base_view = ConstUncheckedField2D(
            geometry.rtheta_base.data(), nVertLevels, nCells);
        auto exner_base_view = ConstUncheckedField2D(
            geometry.exner_base.data(), nVertLevels, nCells);

        // Diabatic tendency
        auto rt_diabatic_tend_view = ConstUncheckedField2D(
            workspace.rt_diabatic_tend_storage.data(), nVertLevels, nCells);

        // Terrain correction views (linearized as (nVertLevels+1, maxEdges*nCells))
        auto zb_cell_view = ConstUncheckedField2D(
            geometry.zb_cell.data(), nVertLevels + 1,
            geometry.maxEdges_ * nCells);
        auto zb3_cell_view = ConstUncheckedField2D(
            geometry.zb3_cell.data(), nVertLevels + 1,
            geometry.maxEdges_ * nCells);

        // Connectivity views (pre-computed once per timestep in timestep())
        auto cellsOnEdge_real_view = ConstUncheckedField2D(
            workspace.cellsOnEdge_real_storage.data(), 2, nEdges);
        auto edgesOnCell_real_view = ConstUncheckedField2D(
            workspace.edgesOnCell_real_storage.data(), geometry.maxEdges_, nCells);
        auto nEdgesOnCell_span_real = ConstSpan1D(
            workspace.nEdgesOnCell_real_storage.data(), nCells);

        kernels::recover_state_perturbation<default_layout>(SerialPolicy{},
            u_out, rho_zz_out, theta_m_out, w_out,
            exner_view, pressure_p_view_out,
            rho_p_save_view, rtheta_p_save_view, ru_save_view, rw_save_view_const,
            rho_pp_const, rtheta_pp_const, ru_p_const, rw_p_const,
            ruAvg_view, wwAvg_view,
            rho_base_view, rtheta_base_view, exner_base_view,
            rt_diabatic_tend_view,
            zb_cell_view, zb3_cell_view,
            cellsOnEdge_real_view, edgesOnCell_real_view,
            nEdgesOnCell_span_real,
            edgesOnCell_sign_view,
            zz_view,
            fzm_span, fzp_span,
            geometry.cf1, geometry.cf2, geometry.cf3,
            static_cast<index_type>(rk_step), cfg.dt, static_cast<index_type>(n_acoustic),
            nCells, nEdges, nVertLevels, geometry.maxEdges_);
    }

    // ========================================================================
    // Step 9: Halo exchanges for recovered variables (u, rho_zz, theta_m, w)
    // ========================================================================
    {
        auto& dstate = get_dycore_state();
        if (!dstate.use_cpp_halo_exchange && dstate.halo_exchange_cb != nullptr) {
            dstate.halo_exchange_cb(
                u.data_handle(), FIELD_ID_U,
                static_cast<int>(nVertLevels), static_cast<int>(nEdges));
            dstate.halo_exchange_cb(
                rho_zz.data_handle(), FIELD_ID_RHO_ZZ,
                static_cast<int>(nVertLevels), static_cast<int>(nCells));
            dstate.halo_exchange_cb(
                theta_m.data_handle(), FIELD_ID_THETA_M,
                static_cast<int>(nVertLevels), static_cast<int>(nCells));
            dstate.halo_exchange_cb(
                w.data_handle(), FIELD_ID_W,
                static_cast<int>(nVertLevels + 1), static_cast<int>(nCells));
        } else {
            std::array<int, 1> layers_arr = {0};
            std::span<const int> layers_span(layers_arr);

            auto u_halo = UncheckedField2D(u.data_handle(), nVertLevels, nEdges);
            halo.start<default_layout>(u_halo, halo_desc, layers_span, comm);
            halo.wait();

            auto rho_zz_halo = UncheckedField2D(rho_zz.data_handle(), nVertLevels, nCells);
            halo.start<default_layout>(rho_zz_halo, halo_desc, layers_span, comm);
            halo.wait();

            auto theta_m_halo = UncheckedField2D(theta_m.data_handle(), nVertLevels, nCells);
            halo.start<default_layout>(theta_m_halo, halo_desc, layers_span, comm);
            halo.wait();

            auto w_halo = UncheckedField2D(w.data_handle(), nVertLevels + 1, nCells);
            halo.start<default_layout>(w_halo, halo_desc, layers_span, comm);
            halo.wait();
        }
    }

    // ========================================================================
    // Step 10: Compute solve diagnostics (unconditional per Fortran —
    //          called after every RK step including the final one)
    // ========================================================================
    {
        auto u_diag = ConstUncheckedField2D(u.data_handle(), nVertLevels, nEdges);
        auto rho_diag = ConstUncheckedField2D(rho_zz.data_handle(), nVertLevels, nCells);

        kernels::compute_solve_diagnostics<default_layout>(SerialPolicy{},
            ke_view, vorticity_view, divergence_view, pv_edge_view, rho_edge_view,
            u_diag, rho_diag, mesh,
            areaCell_span, areaTriangle_span, dvEdge_span, dcEdge_span,
            fVertex_span, edgesOnVertex_view,
            nCells, nEdges, nVertices, nVertLevels);
    }
}

// ============================================================================
// SRK3Integrator::advance_transport (parameterized version)
// ============================================================================

void SRK3Integrator::advance_transport(
    Field3D<> scalars,
    SRK3Workspace& workspace,
    const MeshGeometry& geometry,
    const MeshConnectivity& mesh,
    HaloExchange& halo,
    const CSRHaloDescriptor& halo_desc,
    MPI_Comm comm,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels) {

    const auto& cfg = config_;

    using UncheckedField2D = Field2D<default_layout, unchecked_accessor>;
    using UncheckedField3D = Field3D<default_layout, unchecked_accessor>;
    using ConstUncheckedField2D = ConstField2D<default_layout, unchecked_accessor>;
    using ConstSpan1D = kernels::ConstSpan1D;
    using ConstIndexSpan1D = std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>;

    // ========================================================================
    // Scalar Transport (advance_scalars_mono)
    // ========================================================================

    auto scalars_view = UncheckedField3D(
        scalars.data_handle(), scalars.extent(0), nVertLevels, nCells);

    auto ruAvg_const = ConstUncheckedField2D(
        workspace.ruAvg_storage.data(), nVertLevels, nEdges);
    auto wwAvg_const = ConstUncheckedField2D(
        workspace.wwAvg_storage.data(), nVertLevels + 1, nCells);
    auto rho_zz_old_const = ConstUncheckedField2D(
        workspace.rho_zz_old_storage.data(), nVertLevels, nCells);
    // rho_zz_new is the current rho_zz after dynamics recovery — it's already
    // in the prognostic field. We can access it via the workspace's saved view,
    // but since advance_transport doesn't receive the prognostic rho_zz field
    // directly, we'll use rho_zz_old as a placeholder for rho_zz_new.
    // In practice, the caller should save rho_zz_old before dynamics and pass
    // the current rho_zz after dynamics. For now, use rho_zz_old for both
    // (the timestep() method should set these correctly).
    auto rho_zz_new_const = ConstUncheckedField2D(
        workspace.rho_zz_old_storage.data(), nVertLevels, nCells);

    auto dvEdge_span = ConstSpan1D(geometry.dvEdge.data(), nEdges);
    auto invAreaCell_span = ConstSpan1D(geometry.invAreaCell.data(), nCells);
    auto rdzw_span = ConstSpan1D(geometry.rdzw.data(), nVertLevels);
    auto fzm_span = ConstSpan1D(geometry.fzm.data(), nVertLevels);
    auto fzp_span = ConstSpan1D(geometry.fzp.data(), nVertLevels);

    auto advCellsForEdge_view = ConnectivityView(
        geometry.advCellsForEdge_storage.data(), nEdges, geometry.maxAdvCells);
    auto nAdvCellsForEdge_span = ConstIndexSpan1D(
        geometry.nAdvCellsForEdge.data(), nEdges);

    auto adv_coefs_view = ConstUncheckedField2D(
        geometry.adv_coefs.data(), geometry.maxAdvCells, nEdges);
    auto adv_coefs_3rd_view = ConstUncheckedField2D(
        geometry.adv_coefs_3rd.data(), geometry.maxAdvCells, nEdges);
    auto edgesOnCell_sign_view = ConstUncheckedField2D(
        geometry.edgesOnCell_sign.data(), geometry.maxEdges_, nCells);

    const index_type nScalars = cfg.nScalars > 0
        ? cfg.nScalars
        : static_cast<index_type>(scalars.extent(0));

    kernels::advance_scalars_mono<default_layout>(SerialPolicy{},
        scalars_view,
        ruAvg_const, wwAvg_const,
        rho_zz_old_const, rho_zz_new_const,
        mesh,
        dvEdge_span, invAreaCell_span, rdzw_span, fzm_span, fzp_span,
        advCellsForEdge_view, nAdvCellsForEdge_span,
        adv_coefs_view, adv_coefs_3rd_view, edgesOnCell_sign_view,
        cfg.dt, cfg.coef_3rd_order,
        nScalars, nCells, nEdges, nVertLevels, geometry.maxAdvCells);

    // Halo exchange for scalars after transport
    {
        auto& dstate = get_dycore_state();
        if (!dstate.use_cpp_halo_exchange && dstate.halo_exchange_cb != nullptr) {
            const index_type nScalarsExt = cfg.nScalars > 0
                ? cfg.nScalars
                : static_cast<index_type>(scalars.extent(0));
            dstate.halo_exchange_cb(
                scalars.data_handle(), FIELD_ID_SCALARS,
                static_cast<int>(nVertLevels * nScalarsExt), static_cast<int>(nCells));
        } else {
            std::array<int, 1> layers_arr = {0};
            std::span<const int> layers_span(layers_arr);
            auto scalars_halo = UncheckedField3D(
                scalars.data_handle(), scalars.extent(0), nVertLevels, nCells);
            halo.start<default_layout>(scalars_halo, halo_desc, layers_span, comm);
            halo.wait();
        }
    }
}

} // namespace mpas::dycore
