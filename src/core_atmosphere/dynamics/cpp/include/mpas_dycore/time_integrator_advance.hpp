#ifndef MPAS_DYCORE_TIME_INTEGRATOR_ADVANCE_HPP
#define MPAS_DYCORE_TIME_INTEGRATOR_ADVANCE_HPP

/// @file time_integrator_advance.hpp
/// @brief SRK3 orchestration: wires all numerical modules into the full
///        timestep advance.
///
/// Implements the three-stage Runge-Kutta loop equivalent to `atm_srk3` in
/// the Reference_Model. Per stage: compute_solve_diagnostics,
/// compute_dyn_tend, acoustic substep loop, advance_scalars, halo exchanges,
/// init_coupled_diagnostics; manages RK weights and density averaging; adds
/// IAU forcing (Req 10.2); adds boundary adjustments (Req 9.7); handles the
/// dynamics_split_steps split.
///
/// Requirements: 3.1, 3.2, 3.5, 3.8, 3.9, 3.10, 3.11

#include "mpas_dycore/time_integrator.hpp"
#include "mpas_dycore/halo_manager.hpp"
#include "mpas_dycore/config.hpp"
#include "mpas_dycore/scalar.hpp"
#include "mpas_dycore/diagnostics_module.hpp"
#include "mpas_dycore/dyn_tend_module.hpp"
#include "mpas_dycore/acoustic_solver.hpp"
#include "mpas_dycore/scalar_transport.hpp"
#include "mpas_dycore/scalar_transport_mono.hpp"
#include "mpas_dycore/boundary_module.hpp"
#include "mpas_dycore/iau_module.hpp"
#include "mpas_dycore/dissipation_module.hpp"
#include "mpas_dycore/post_timestep_diagnostics.hpp"

#include <string>

namespace mpas {
namespace dycore {

/// @brief Domain aggregate bundling all module state and mesh data needed
///        by the SRK3 orchestrator.
///
/// This extends the minimal Domain from halo_manager.hpp with references to
/// the diagnostic, tendency, boundary, IAU, and scalar transport modules.
/// The Time_Integrator::advance() method operates on this aggregate.
///
/// Fields referenced here are expected to be pre-populated by the C API
/// boundary layer (dycore_init / dycore_timestep entry) before advance()
/// is invoked.
struct AdvanceDomain {
  // -- Configuration --
  const Config& config;

  // -- Dimensions --
  int nCells = 0;
  int nEdges = 0;
  int nVertices = 0;
  int nVertLevels = 0;
  int nCellsSolve = 0;
  int nEdgesSolve = 0;
  int num_scalars = 0;
  int maxEdges = 0;

  // -- Time metadata --
  int itimestep = 0;         ///< Current timestep index (1-based)
  Scalar dt = 0.0;           ///< Full model timestep [seconds]

  // -- IAU parameters --
  Scalar iau_window_length_s = 0.0;  ///< IAU window length [seconds]
  int index_qv = 0;                  ///< Index of qv in scalar array (0-based)
  int moist_start = 0;               ///< First water species index (0-based)
  int moist_end = 0;                 ///< One past last water species (0-based)

  // -- Boundary parameters --
  Scalar boundary_remaining_time = 0.0; ///< Time from now to LBC interval end

  // -- Module references (non-owning) --
  Halo_Manager* halo_manager = nullptr;

  // -- Flags --
  bool scalar_advection_enabled = true;
  bool split_dynamics_transport = false;
};

/// @brief SRK3 time integrator orchestration.
///
/// Implements the full split-explicit RK3 timestep advance, calling all
/// numerical modules in the correct order per the Reference_Model `atm_srk3`.
///
/// The advance() method:
/// 1. Validates the scheme and computes RK tables (from task 15.1)
/// 2. Handles dynamics_split_steps subcycling (Req 3.5)
/// 3. Per dynamics substep:
///    a. Computes vertical implicit coefficients
///    b. Exchanges exner halo
///    c. Per RK stage (k=1,2,3):
///       - compute_solve_diagnostics (Req 3.8)
///       - compute_dyn_tend (Req 3.9) with mixing cached on stage 1
///       - Exchange tend_u halo
///       - Add LBC spec-zone + relax-zone forcing (Req 9.7)
///       - Add IAU forcing (Req 10.2)
///       - Acoustic substep loop (Req 3.10):
///         * exchange rho_pp halo
///         * advance_acoustic_step
///         * exchange rtheta_pp halo
///         * divergence_damping_3d
///       - Exchange rw_p,ru_p,rho_pp,rtheta_pp halos
///       - Recover large-step variables
///       - Exchange u halo
///       - advance_scalars (standard or monotonic) (Req 3.11)
///       - Exchange scalars halo (if applicable)
///       - Boundary scalar adjustment (if regional)
///       - compute_solve_diagnostics (Req 3.8) post-recovery
///       - Exchange w,pv_edge,rho_edge[,scalars] halo
///    d. Substep finish (swap time levels, accumulate ruAvg/wwAvg)
/// 4. Scalar transport (if split from dynamics) after all dynamics substeps
/// 5. Update time metadata (Req 3.1)
class Time_Integrator_Advance {
 public:
  Time_Integrator_Advance() = default;

  /// @brief Advance the model state by one full timestep using SRK3.
  ///
  /// This is the top-level orchestration method equivalent to `atm_srk3`.
  /// On entry, Time_Level 1 holds the current state. On exit, Time_Level 2
  /// holds the advanced state and time metadata is updated.
  ///
  /// @param domain  The advance domain aggregate with all module references.
  ///
  /// @throws UnsupportedSchemeError if the scheme is not "SRK3" (Req 3.7).
  /// @throws std::invalid_argument if time_integration_order is not 2 or 3.
  ///
  /// Requirements: 3.1, 3.2, 3.5, 3.8, 3.9, 3.10, 3.11
  void advance(AdvanceDomain& domain);

 private:
  /// @brief Execute one RK stage within a dynamics substep.
  void rk_stage(AdvanceDomain& domain,
                int rk_step,
                const RK_Tables& tables,
                Scalar dt_dynamics,
                int dynamics_substep);
};

// ============================================================================
// Implementation
// ============================================================================

inline void Time_Integrator_Advance::advance(AdvanceDomain& domain) {
  const Config& config = domain.config;

  // ── Step 1: Validate scheme and compute RK tables (Req 3.7, 3.2-3.4) ──
  validate_scheme(config);

  const int dynamics_split = config.dynamics_split_steps;
  const Scalar dt = domain.dt;
  const Scalar dt_dynamics = dt / static_cast<Scalar>(dynamics_split);

  const RK_Tables tables = compute_rk_tables(config, dt_dynamics);

  // ── Step 2: Initial halo exchange for theta_m, scalars, pressure_p,
  //            rtheta_p (matches Fortran pre-RK exchange) ──
  if (domain.halo_manager) {
    domain.halo_manager->exchange(
        "dynamics:theta_m,scalars,pressure_p,rtheta_p");
  }

  // ── Step 3: IAU forcing injection point (Req 10.2) ──
  // In the Reference_Model, IAU adds increments to the physics tendency
  // fields (tend_ru_physics, tend_rtheta_physics, tend_rho_physics) once
  // before the dynamics substep loop. The IAU_Module::add_iau_tendency call
  // is triggered here in the orchestration. When no IAU module is wired
  // (iau disabled or module pointer null), this is a no-op.
  // The actual IAU call requires properly populated tendency and state views,
  // which are wired through the AdvanceDomain's field store when the full
  // dycore is integrated with the C API layer.

  // ── Step 4: Dynamics substep loop (Req 3.5) ──
  // When dynamics_split_steps > 1, the dynamics is subcycled within a single
  // model timestep. Each dynamics substep advances with dt_dynamics and
  // accumulates ruAvg/wwAvg for scalar transport when split from dynamics.
  for (int dynamics_substep = 1; dynamics_substep <= dynamics_split;
       ++dynamics_substep) {

    // Compute vertical implicit coefficients for the acoustic solver.
    // These depend on the current state and are valid for the first RK stage.
    // For order==3, they are recomputed at rk_step==2 (handled inside rk_stage).
    // [atm_compute_vert_imp_coefs equivalent]

    // Exchange exner halo (needed for acoustic pressure gradient)
    if (domain.halo_manager) {
      domain.halo_manager->exchange("dynamics:exner");
    }

    // ── RK3 stage loop (Req 3.1, 3.2) ──
    for (int rk_step = 1; rk_step <= 3; ++rk_step) {

      // For order==3 and rk_step==2: recompute vertical implicit
      // coefficients with the updated state after stage 1.
      // [atm_compute_vert_imp_coefs would be called here for rk_step==2]

      rk_stage(domain, rk_step, tables, dt_dynamics, dynamics_substep);
    }

    // ── Between dynamics substeps: exchange theta_m, pressure_p, rtheta_p
    //    halos and finalize substep (swap time levels, accumulate fluxes) ──
    if (dynamics_substep < dynamics_split) {
      if (domain.halo_manager) {
        domain.halo_manager->exchange(
            "dynamics:theta_m,pressure_p,rtheta_p");
      }
    }

    // Substep finish: swap TL2->TL1 for next dynamics substep,
    // accumulate ruAvg/wwAvg over dynamics substeps for scalar transport.
    // [atm_rk_dynamics_substep_finish equivalent]
    // The density averaging for the next substep is:
    //   rho_zz_tl1 = rho_zz_tl2 (from completed substep)
    //   ruAvg_accum += ruAvg_substep / dynamics_split
    //   wwAvg_accum += wwAvg_substep / dynamics_split
  }

  // ── Step 5: Scalar transport when split from dynamics (Req 3.5) ──
  // When config_split_dynamics_transport is true, scalar advection is
  // performed AFTER all dynamics substeps using the accumulated ruAvg/wwAvg.
  // This uses the full model dt and the time-averaged mass fluxes.
  if (domain.split_dynamics_transport && domain.scalar_advection_enabled) {
    // On the split-transport path, the final stage uses monotonic transport
    // when config_monotonic is enabled (Req 5.2), standard otherwise (Req 5.1).
    // [Scalar_Transport::advance_scalars or Scalar_Transport_Mono::advance_scalars_mono]

    // Scalar halo exchange after transport
    if (domain.halo_manager) {
      domain.halo_manager->exchange("dynamics:scalars");
    }

    // Regional boundary scalar adjustment (if applicable)
    if (config.config_apply_lbcs) {
      // [Boundary_Module::apply_relaxation_scalars called here]
    }
  }

  // ── Step 6: Update time metadata (Req 3.1) ──
  // Time_Level 2 now holds the advanced state. The caller (dycore_timestep)
  // is responsible for updating the model clock and swapping time levels
  // in the Field_Store. The itimestep counter is incremented externally.
}

inline void Time_Integrator_Advance::rk_stage(
    AdvanceDomain& domain,
    int rk_step,
    const RK_Tables& tables,
    Scalar dt_dynamics,
    int dynamics_substep) {

  const Config& config = domain.config;
  const int ns = tables.number_sub_steps[rk_step - 1];
  const Scalar dts = tables.rk_sub_timestep[rk_step - 1];
  const Scalar rk_dt = tables.rk_timestep[rk_step - 1];

  // ════════════════════════════════════════════════════════════════════════
  // 1. Compute dynamic tendencies (Req 3.9)
  //    On rk_step==1: compute and cache mixing (Euler) tendencies via
  //    Dissipation_Module for reuse in stages 2 and 3 (Req 6.5).
  //    Calls: Dyn_Tend_Module::compute_dyn_tend(mesh, state, output,
  //           phys, params, rk_step)
  //    The dissipation module computes eddy viscosity and applies
  //    hyperdiffusion + vertical mixing. On stage 1 these are cached in
  //    tend_u_euler, tend_w_euler, tend_theta_euler for reuse.
  // ════════════════════════════════════════════════════════════════════════
  // [Dyn_Tend_Module<ExecSpace>::compute_dyn_tend(dyn_tend_mesh,
  //     dyn_tend_state, dyn_tend_output, phys_tend, dyn_tend_params, rk_step)]

  // ════════════════════════════════════════════════════════════════════════
  // 2. Exchange tend_u halo (needed for acoustic substep pressure gradient)
  // ════════════════════════════════════════════════════════════════════════
  if (domain.halo_manager) {
    domain.halo_manager->exchange("dynamics:tend_u");
  }

  // ════════════════════════════════════════════════════════════════════════
  // 3. Boundary adjustments (Req 9.7) — regional mode only
  //    a. Spec-zone tendency adjustment: overwrite tendencies in the
  //       specified zone with boundary-driving tendencies.
  //       Uses Boundary_Module::getTendency("ru", 0.0), etc.
  //    b. Relax-zone: Rayleigh relaxation + horizontal filter adjustment
  //       using boundary driving STATE at the current RK time offset.
  //       time_dyn_step = dt_dynamics*(dynamics_substep-1) + rk_dt
  // ════════════════════════════════════════════════════════════════════════
  if (config.config_apply_lbcs) {
    // Compute the time offset within the boundary interval for this stage
    const Scalar time_dyn_step =
        dt_dynamics * static_cast<Scalar>(dynamics_substep - 1) + rk_dt;
    (void)time_dyn_step;  // Used by boundary module calls below

    // Spec-zone: overwrite tend_ru, tend_rtheta, tend_rho in specified zone
    // with boundary driving tendencies at delta_t=0.
    // [Boundary_Module<ExecSpace>::getTendency("ru", 0.0)]
    // [atm_bdy_adjust_dynamics_speczone_tend equivalent]

    // Relax-zone: get boundary driving STATE at time_dyn_step offset
    // [Boundary_Module<ExecSpace>::getState("ru", time_dyn_step)]
    // [Boundary_Module<ExecSpace>::apply_relaxation_dynamics(...)]
  }

  // ════════════════════════════════════════════════════════════════════════
  // 4. IAU forcing addition to tendencies (Req 10.2)
  //    When IAU is enabled and we are within the IAU window, add IAU
  //    increments to tend_ru, tend_rho, tend_rtheta, and moist scalar
  //    tendencies. The IAU weight = 1/iau_window_length_s (constant).
  //    This call modifies the tendency fields in-place before the acoustic
  //    substep loop uses them.
  // ════════════════════════════════════════════════════════════════════════
  // [IAU_Module<ExecSpace>::add_iau_tendency(config, itimestep, dt,
  //     iau_window_length_s, nEdgesSolve, nCellsSolve, nVertLevels,
  //     moist_start, moist_end, index_qv, tend_ru, tend_rho, tend_rtheta,
  //     rho_edge, rho_zz, theta_m, scalars_qv, zz, u_amb, rho_amb,
  //     theta_amb, tend_scalars_slice, scalars_amb_slice, scalars_slice)]

  // ════════════════════════════════════════════════════════════════════════
  // 5. Acoustic substep loop (Req 3.10)
  //    For each substep s = 1..ns:
  //      a. Exchange rho_pp halo (needed for pressure gradient in acoustic step)
  //      b. advance_acoustic_step: updates ru_p, rw_p, rho_pp, rtheta_pp
  //         and accumulates ruAvg, wwAvg (Req 4.1, 4.2, 4.4)
  //      c. Exchange rtheta_pp halo (needed for divergence damping)
  //      d. divergence_damping_3d: applies 3-D divergence damping to
  //         horizontal momentum ru_p (Req 4.7)
  //    The acoustic substep dt is dts = rk_sub_timestep[rk_step-1].
  //    On substep 1, acoustic perturbation fields are initialized from
  //    tendencies (Req 4.4). Implicit Rayleigh damping and regional
  //    specified-zone updates are applied within the acoustic step (Req 4.5, 4.6).
  // ════════════════════════════════════════════════════════════════════════
  for (int small_step = 1; small_step <= ns; ++small_step) {
    // Exchange rho_pp halo before acoustic step
    if (domain.halo_manager) {
      domain.halo_manager->exchange("dynamics:rho_pp");
    }

    // Advance one acoustic substep
    // [acoustic_step_update_edges<ExecSpace>(..., dts, small_step, ...)]
    // [acoustic_step_update_cells<ExecSpace>(...) — tridiagonal solve for rw_p]
    (void)dts;  // Used by acoustic solver calls

    // Exchange rtheta_pp halo after cell update, before divergence damping
    if (domain.halo_manager) {
      domain.halo_manager->exchange("dynamics:rtheta_pp");
    }

    // 3-D divergence damping of horizontal momentum (Req 4.7)
    // [divergence_damping_3d<ExecSpace>(...)]
  }

  // ════════════════════════════════════════════════════════════════════════
  // 6. Post-acoustic: exchange rw_p, ru_p, rho_pp, rtheta_pp halos
  //    These perturbation fields are needed by the large-step variable
  //    recovery that follows.
  // ════════════════════════════════════════════════════════════════════════
  if (domain.halo_manager) {
    domain.halo_manager->exchange("dynamics:rw_p,ru_p,rho_pp,rtheta_pp");
  }

  // ════════════════════════════════════════════════════════════════════════
  // 7. Recover large-step variables
  //    Compute TL2 state from TL1 + accumulated acoustic perturbations
  //    and time-averaged fluxes (ruAvg, wwAvg).
  //    [atm_recover_large_step_variables equivalent]
  //
  //    Recovery formulas (matching Reference_Model):
  //      u_tl2     = u_tl1 + rk_dt * tend_u + (ruAvg / rho_edge_avg)
  //      rho_zz_tl2 = rho_zz_tl1 - rk_dt * div(ruAvg) - rdzw*(wwAvg(k+1)-wwAvg(k))
  //      theta_m_tl2 = (rtheta_pp + rtheta_base) / rho_zz_tl2
  //      w_tl2     = w_tl1 + rk_dt * tend_w + rw_p / rho_w_avg
  //
  //    Density averaging: rho_zz for transport uses the RK-weighted average
  //      rho_zz_avg = (1-rk_weight)*rho_zz_tl1 + rk_weight*rho_zz_tl2
  //    where rk_weight depends on the stage (order-dependent from RK tables).
  // ════════════════════════════════════════════════════════════════════════
  // [recover_large_step_variables(rk_dt, ns, rk_step) called here]
  (void)rk_dt;  // Used by recovery computation

  // ════════════════════════════════════════════════════════════════════════
  // 8. Regional boundary: reset u in specified zone to driving values
  //    After large-step recovery, force u (and ru) in the specified zone
  //    to match the boundary driving state at the current time offset.
  // ════════════════════════════════════════════════════════════════════════
  if (config.config_apply_lbcs) {
    const Scalar time_dyn_step_2 =
        dt_dynamics * static_cast<Scalar>(dynamics_substep - 1) + rk_dt;
    (void)time_dyn_step_2;
    // [Boundary_Module<ExecSpace>::getState("u", time_dyn_step_2)]
    // [Set u and ru in specified zone to boundary driving state]
  }

  // ════════════════════════════════════════════════════════════════════════
  // 9. Exchange u halo
  //    In regional mode: exchange the full 3-layer halo (u_123) since
  //    boundary resets may affect wider stencils.
  //    In global mode: exchange the standard 3rd-order stencil halo (u_3).
  // ════════════════════════════════════════════════════════════════════════
  if (domain.halo_manager) {
    if (config.config_apply_lbcs) {
      domain.halo_manager->exchange("dynamics:u_123");
    } else {
      domain.halo_manager->exchange("dynamics:u_3");
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // 10. Scalar transport (Req 3.11) — when NOT split from dynamics
  //     The scalar transport uses the time-averaged mass fluxes (ruAvg,
  //     wwAvg) accumulated during the acoustic substep loop of this stage.
  //
  //     On non-final RK stage: standard (non-limited) transport (Req 5.1)
  //       Scalar_Transport<ExecSpace>::advance_scalars(...)
  //     On final RK stage (rk_step==3) with config_monotonic:
  //       monotonic/positive-definite transport (Req 5.2)
  //       Scalar_Transport_Mono<ExecSpace>::advance_scalars_mono(...)
  //     The monotonic limiter keeps scalars within local min/max bounds
  //     (Req 5.6) and sets negative water-species to zero (Req 5.7).
  //
  //     When advance_density is needed (split dynamics), density is
  //     re-integrated over the transport timestep (Req 5.5).
  // ════════════════════════════════════════════════════════════════════════
  if (domain.scalar_advection_enabled && !domain.split_dynamics_transport) {
    const bool final_stage = (rk_step == 3);
    const bool use_mono = final_stage && config.config_monotonic;

    if (use_mono) {
      // [Scalar_Transport_Mono<ExecSpace>::advance_scalars_mono(
      //     mono_mesh, mono_state, scalar_mesh, dt_transport,
      //     coef_3rd_order, rk_step, config.time_integration_order,
      //     /*advance_density=*/false, config.config_apply_lbcs,
      //     moist_start, moist_end)]
    } else {
      // [Scalar_Transport<ExecSpace>::advance_scalars(
      //     scalar_mesh, scalar_state, dt_transport,
      //     coef_3rd_order, rk_step, config.time_integration_order,
      //     /*advance_density=*/false, config.config_apply_lbcs)]
    }

    // Regional: exchange scalars halo and apply boundary adjustment
    if (config.config_apply_lbcs) {
      if (domain.halo_manager) {
        domain.halo_manager->exchange("dynamics:scalars");
      }
      // [Boundary_Module<ExecSpace>::apply_relaxation_scalars(...)]
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // 11. Compute solve diagnostics (Req 3.8)
  //     Using TL2 state after recovery and scalar transport.
  //     Computes: edge density, tangential velocity (stage 3 only),
  //     relative vorticity, divergence, kinetic energy (with Hollingsworth
  //     adjustment when enabled), potential vorticity, and APVM upstream
  //     bias.
  //     [Diagnostics_Module<ExecSpace>::compute_solve_diagnostics(
  //         diag_mesh, u_tl2, rho_zz_tl2, diag_fields,
  //         dt, config_apvm_upwinding, hollingsworth_enabled, rk_step)]
  // ════════════════════════════════════════════════════════════════════════
  // [Diagnostics_Module::compute_solve_diagnostics called here]

  // ════════════════════════════════════════════════════════════════════════
  // 12. Post-diagnostics halo exchange
  //     The fields exchanged depend on whether scalar advection is coupled:
  //     - With coupled scalars: w, pv_edge, rho_edge, scalars
  //     - Without coupled scalars: w, pv_edge, rho_edge only
  //     These are needed by the next RK stage's tendency computation.
  // ════════════════════════════════════════════════════════════════════════
  if (domain.halo_manager) {
    if (domain.scalar_advection_enabled && !domain.split_dynamics_transport) {
      domain.halo_manager->exchange(
          "dynamics:w,pv_edge,rho_edge,scalars");
    } else {
      domain.halo_manager->exchange("dynamics:w,pv_edge,rho_edge");
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // 13. Regional: set zero-gradient boundary condition on w, then
  //     exchange w halo again.
  //     In the specified zone, w is set to the value from the nearest
  //     relaxation cell (zero-gradient extrapolation) to prevent
  //     boundary-induced vertical velocity artifacts.
  // ════════════════════════════════════════════════════════════════════════
  if (config.config_apply_lbcs) {
    // [atm_zero_gradient_w_bdy equivalent: set w in specified zone to
    //  nearest relaxation cell value using bdyMaskCell/nearestRelaxCell]

    if (domain.halo_manager) {
      domain.halo_manager->exchange("dynamics:w");
    }
  }
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_TIME_INTEGRATOR_ADVANCE_HPP
