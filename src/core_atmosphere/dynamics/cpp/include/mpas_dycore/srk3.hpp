#pragma once

/// @file srk3.hpp
/// @brief Split-Explicit Runge-Kutta 3 time integrator for the MPAS dynamical core.
///
/// Defines the SRK3Config struct with all timestep configuration parameters and
/// the SRK3Integrator class that orchestrates one full large timestep including:
///   - 2 or 3 Runge-Kutta stages for slow dynamics
///   - Multiple acoustic sub-steps within each RK stage
///   - Scalar transport (optionally split from dynamics)
///   - Halo exchange between phases
///
/// The integrator is stateless except for its configuration. All field data
/// is passed by reference through the timestep() method.

#include <mpas_dycore/types.hpp>

#include <stdexcept>
#include <string>

namespace mpas::dycore {

// ============================================================================
// SRK3Config: configuration for the SRK3 time integrator
// ============================================================================

/// @brief Configuration for the SRK3 time integrator.
///
/// Contains all parameters needed to configure a single large timestep:
/// RK mode selection, acoustic sub-stepping, dynamics-transport splitting,
/// physics coupling, mixing coefficients, scalar transport, and IAU.
struct SRK3Config {
    // ---- Time integration core ----

    /// Number of Runge-Kutta stages: 2 (RK2 mode) or 3 (SRK3, default).
    int n_rk_stages = 3;

    /// Number of acoustic sub-steps per RK stage (valid range: 2 to 24).
    int number_of_sub_steps = 6;

    /// Large timestep in seconds. Must be > 0.
    real_type dt = 0.0;

    /// Acoustic sub-timestep in seconds (= dt / number_of_sub_steps).
    real_type dts = 0.0;

    // ---- Dynamics-transport splitting ----

    /// If true, dynamics and transport use separate sub-cycling.
    bool dynamics_transport_split = false;

    /// Number of dynamics sub-steps per transport step (used when split is enabled).
    int dynamics_split_steps = 1;

    // ---- Physics coupling ----

    /// Apply lateral boundary conditions (regional configurations).
    bool config_apply_lbcs = false;

    /// Enable Rayleigh damping on horizontal velocity (u).
    bool config_rayleigh_damp_u = false;

    /// Number of model levels from the top where Rayleigh damping is applied.
    int config_number_rayleigh_damp_u_levels = 0;

    /// Rayleigh damping timescale in days.
    real_type config_rayleigh_damp_u_timescale_days = 0.0;

    // ---- Horizontal mixing ----

    /// Second-order momentum eddy viscosity coefficient.
    real_type config_h_mom_eddy_visc2 = 0.0;

    /// Fourth-order momentum eddy viscosity coefficient.
    real_type config_h_mom_eddy_visc4 = 0.0;

    /// Second-order theta eddy viscosity coefficient.
    real_type config_h_theta_eddy_visc2 = 0.0;

    /// Fourth-order theta eddy viscosity coefficient.
    real_type config_h_theta_eddy_visc4 = 0.0;

    /// Factor scaling the divergent component of del-4 diffusion.
    real_type config_del4u_div_factor = 1.0;

    // ---- Scalar transport ----

    /// Third-order coefficient for scalar advection (blending with 4th-order).
    real_type coef_3rd_order = 1.0;

    /// Enable/disable scalar advection within the timestep.
    bool config_scalar_advection = true;

    // ---- Incremental Analysis Update (IAU) ----

    /// Whether IAU forcing is active for this timestep.
    bool iau_active = false;

    /// IAU window duration in seconds.
    real_type iau_window_seconds = 0.0;
};

// ============================================================================
// SRK3Config validation
// ============================================================================

/// @brief Validate an SRK3Config, throwing std::invalid_argument on failure.
///
/// Checks:
/// - n_rk_stages is 2 or 3
/// - number_of_sub_steps is in [2, 24]
/// - dt > 0
/// - dts > 0
/// - dynamics_split_steps >= 1 (when split is enabled)
///
/// @param config The configuration to validate.
/// @throws std::invalid_argument with a descriptive message on validation failure.
inline void validate_srk3_config(const SRK3Config& config) {
    if (config.n_rk_stages != 2 && config.n_rk_stages != 3) {
        throw std::invalid_argument(
            "SRK3Config: n_rk_stages must be 2 or 3, got " +
            std::to_string(config.n_rk_stages));
    }

    if (config.number_of_sub_steps < 2 || config.number_of_sub_steps > 24) {
        throw std::invalid_argument(
            "SRK3Config: number_of_sub_steps must be in [2, 24], got " +
            std::to_string(config.number_of_sub_steps));
    }

    if (config.dt <= 0.0) {
        throw std::invalid_argument(
            "SRK3Config: dt must be > 0, got " +
            std::to_string(config.dt));
    }

    if (config.dts <= 0.0) {
        throw std::invalid_argument(
            "SRK3Config: dts must be > 0, got " +
            std::to_string(config.dts));
    }

    if (config.dynamics_transport_split && config.dynamics_split_steps < 1) {
        throw std::invalid_argument(
            "SRK3Config: dynamics_split_steps must be >= 1 when split is enabled, got " +
            std::to_string(config.dynamics_split_steps));
    }
}

// ============================================================================
// SRK3Integrator: time integration driver
// ============================================================================

/// @brief Split-Explicit Runge-Kutta 3 time integrator for the MPAS dycore.
///
/// This class orchestrates one full timestep of the MPAS dynamical core,
/// including:
///   - 2 or 3 Runge-Kutta stages for slow dynamics
///   - Multiple acoustic sub-steps within each RK stage
///   - Scalar transport (optionally split from dynamics)
///   - Halo exchange between phases
///
/// The integrator is stateless except for its configuration. All field data
/// is passed by reference through the timestep() method.
class SRK3Integrator {
public:
    /// Default constructor (creates an integrator with default config; must be
    /// reconfigured before use via the parameterized constructor).
    SRK3Integrator() = default;

    /// @brief Construct an integrator with a validated configuration.
    ///
    /// @param config The SRK3 configuration parameters.
    /// @throws std::invalid_argument if config fails validation.
    explicit SRK3Integrator(const SRK3Config& config)
        : config_(config) {
        validate_srk3_config(config_);
    }

    /// @brief Execute one full large timestep.
    ///
    /// This is the top-level entry point called by mpas_dycore_cpp_timestep().
    /// All field references will be added in task 19.2 when the full method
    /// signature is defined.
    void timestep();

    /// @brief Get the current configuration (read-only).
    /// @return Const reference to the integrator's SRK3Config.
    const SRK3Config& config() const { return config_; }

private:
    /// Configuration for this integrator instance.
    SRK3Config config_{};

    /// @brief Execute the dynamics phase for one RK stage.
    ///
    /// Computes tendencies, performs acoustic sub-stepping, and recovers
    /// large-step variables for the given RK stage.
    ///
    /// @param rk_step The RK stage index (0-based: 0, 1, or 2).
    void advance_dynamics(int rk_step);

    /// @brief Execute the transport phase (scalar advection).
    ///
    /// Performs horizontal and vertical scalar advection with FCT limiting,
    /// using time-averaged mass fluxes accumulated during acoustic sub-stepping.
    void advance_transport();
};

} // namespace mpas::dycore
