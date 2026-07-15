#ifndef MPAS_DYCORE_TIME_INTEGRATOR_HPP
#define MPAS_DYCORE_TIME_INTEGRATOR_HPP

/// @file time_integrator.hpp
/// @brief SRK3 time integrator: scheme validation and RK stage/substep tables.
///
/// Implements the initialization portion of the `atm_timestep`/`atm_srk3`
/// routines from the Reference_Model (`mpas_atm_time_integration.F`):
/// - Three-stage RK substep weights for configured order 2 and 3
///   (Requirements 3.2, 3.3, 3.4)
/// - Per-stage acoustic substep counts (Requirement 3.6)
/// - Rejection of any scheme other than "SRK3" with an error naming the
///   scheme (Requirement 3.7)

#include <array>
#include <stdexcept>
#include <string>

#include "mpas_dycore/config.hpp"
#include "mpas_dycore/scalar.hpp"

namespace mpas {
namespace dycore {

/// Exception thrown when an unsupported time integration scheme is configured.
/// The error message names the unsupported scheme (Requirement 3.7).
class UnsupportedSchemeError : public std::runtime_error {
public:
  explicit UnsupportedSchemeError(const std::string& scheme)
      : std::runtime_error("Unknown time integration option " + scheme +
                           ". Currently, only 'SRK3' is supported."),
        scheme_(scheme) {}

  /// Returns the name of the unsupported scheme.
  const std::string& scheme() const noexcept { return scheme_; }

private:
  std::string scheme_;
};

/// Holds the per-stage RK timestep weights and acoustic substep counts
/// for the split-explicit SRK3 time integrator.
///
/// The tables are computed exactly as in the Reference_Model:
/// - Order 3: rk_timestep = {dt/3, dt/2, dt}
///            rk_sub_timestep = {dt/3, dt/ns, dt/ns}
///            number_sub_steps = {1, max(1, ns/2), ns}
/// - Order 2: rk_timestep = {dt/2, dt/2, dt}
///            rk_sub_timestep = {dt/ns, dt/ns, dt/ns}
///            number_sub_steps = {max(1, ns/2), max(1, ns/2), ns}
///
/// where dt = dt_dynamics and ns = config_number_of_sub_steps.
struct RK_Tables {
  /// RK stage timestep weights (3 stages).
  /// rk_timestep[k] is the outer timestep fraction for stage k (0-indexed).
  std::array<Scalar, 3> rk_timestep;

  /// Acoustic sub-timestep for each RK stage (3 stages).
  /// rk_sub_timestep[k] is the acoustic substep dt for stage k (0-indexed).
  std::array<Scalar, 3> rk_sub_timestep;

  /// Number of acoustic substeps for each RK stage (3 stages).
  /// number_sub_steps[k] is the acoustic substep count for stage k (0-indexed).
  std::array<int, 3> number_sub_steps;
};

/// Validate that the configured time integration scheme is "SRK3".
///
/// If the scheme name (trimmed comparison) is not "SRK3", throws an
/// UnsupportedSchemeError naming the configured scheme (Requirement 3.7).
///
/// @param config The immutable configuration snapshot.
/// @throws UnsupportedSchemeError if the scheme is not "SRK3".
inline void validate_scheme(const Config& config) {
  if (config.time_integration_scheme != "SRK3") {
    throw UnsupportedSchemeError(config.time_integration_scheme);
  }
}

/// Compute the RK stage tables from the Reference_Model definitions.
///
/// This replicates the logic in `atm_srk3` from `mpas_atm_time_integration.F`
/// for `config_time_integration_order` == 2 or 3.
///
/// @param config The immutable configuration snapshot (must have scheme "SRK3"
///        and time_integration_order in {2, 3}).
/// @param dt_dynamics The dynamics timestep (dt / dynamics_split_steps or dt
///        when splitting is disabled).
/// @return An RK_Tables struct populated with the stage weights and substep
///         counts.
/// @throws UnsupportedSchemeError if the scheme is not "SRK3".
/// @throws std::invalid_argument if time_integration_order is not 2 or 3.
inline RK_Tables compute_rk_tables(const Config& config, Scalar dt_dynamics) {
  // Validate scheme first (Requirement 3.7)
  validate_scheme(config);

  const int ns = config.number_of_sub_steps;
  const int order = config.time_integration_order;

  RK_Tables tables{};

  if (order == 3) {
    // Order 3 weights from the Reference_Model (Requirement 3.3)
    tables.rk_timestep[0] = dt_dynamics / Scalar(3.0);
    tables.rk_timestep[1] = dt_dynamics / Scalar(2.0);
    tables.rk_timestep[2] = dt_dynamics;

    tables.rk_sub_timestep[0] = dt_dynamics / Scalar(3.0);
    tables.rk_sub_timestep[1] = dt_dynamics / static_cast<Scalar>(ns);
    tables.rk_sub_timestep[2] = dt_dynamics / static_cast<Scalar>(ns);

    tables.number_sub_steps[0] = 1;
    tables.number_sub_steps[1] = std::max(1, ns / 2);
    tables.number_sub_steps[2] = ns;

  } else if (order == 2) {
    // Order 2 weights from the Reference_Model (Requirement 3.4)
    tables.rk_timestep[0] = dt_dynamics / Scalar(2.0);
    tables.rk_timestep[1] = dt_dynamics / Scalar(2.0);
    tables.rk_timestep[2] = dt_dynamics;

    tables.rk_sub_timestep[0] = dt_dynamics / static_cast<Scalar>(ns);
    tables.rk_sub_timestep[1] = dt_dynamics / static_cast<Scalar>(ns);
    tables.rk_sub_timestep[2] = dt_dynamics / static_cast<Scalar>(ns);

    tables.number_sub_steps[0] = std::max(1, ns / 2);
    tables.number_sub_steps[1] = std::max(1, ns / 2);
    tables.number_sub_steps[2] = ns;

  } else {
    throw std::invalid_argument(
        "config_time_integration_order must be 2 or 3, got " +
        std::to_string(order));
  }

  return tables;
}

/// Return the number of acoustic substeps for a given RK stage (1-indexed).
///
/// @param tables The precomputed RK tables.
/// @param stage The RK stage number (1, 2, or 3).
/// @return The acoustic substep count for that stage.
inline int acoustic_substeps(const RK_Tables& tables, int stage) {
  return tables.number_sub_steps[stage - 1];
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_TIME_INTEGRATOR_HPP
