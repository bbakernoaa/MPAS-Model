#ifndef MPAS_DYCORE_CONFIG_HPP
#define MPAS_DYCORE_CONFIG_HPP

/// @file config.hpp
/// @brief Immutable snapshot of namelist options consumed by the C++ dycore.
///
/// `Config` captures the subset of MPAS-Atmosphere namelist options that the
/// ported dynamical core modules reference. It is constructed once during
/// `dycore_init` from values sourced out of the existing MPAS data structures
/// (Requirement 13.2) and passed by const-reference to every compute module.
///
/// Immutability is enforced by making all data members `const`. A `Config`
/// object cannot be default-constructed; it must be aggregate-initialized (or
/// constructed via the named-parameter builder below) with all fields set.
/// This prevents accidental use of uninitialized configuration values.
///
/// Validation of configuration values (Requirement 3.7, 8.10, 8.11, 9.8-9.10,
/// 11.10) is performed by the consuming modules at construction or first use,
/// not by `Config` itself — `Config` is a plain data snapshot, not a policy
/// object.
///
/// Fixed per-build inner dimensions (nVertLevels, maxEdges, num_scalars) are
/// captured separately in `MeshDims` (Requirement 13.4) and are not repeated
/// here.

#include <string>

namespace mpas {
namespace dycore {

/// Immutable snapshot of namelist configuration options for the C++ dycore.
///
/// All members are `const` to enforce immutability after construction.
/// Construction is via aggregate initialization (C++20 designated initializers
/// are encouraged for clarity at the call site).
struct Config {
  // ── Time integration (Requirement 3) ───────────────────────────────────────

  /// Name of the time integration scheme (e.g. "SRK3").
  /// Validated by Time_Integrator::validate_scheme (Req 3.7).
  const std::string time_integration_scheme;

  /// Order of the Runge-Kutta time integration scheme (2 or 3).
  /// Used by Time_Integrator to select stage weights (Req 3.2, 3.3, 3.4).
  const int time_integration_order;

  /// Number of acoustic substeps per RK stage (Req 3.6).
  const int number_of_sub_steps;

  /// Number of dynamics split steps when dynamics-transport splitting is
  /// enabled (Req 3.5). A value of 1 means no splitting.
  const int dynamics_split_steps;

  // ── Scalar transport (Requirement 5) ───────────────────────────────────────

  /// Whether to apply the monotonic/positive-definite flux limiter on the
  /// final RK substep (Req 5.2, 5.6, 5.7).
  const bool config_monotonic;

  /// Whether scalar advection is enabled (Req 5.1).
  const bool config_scalar_advection;

  // ── Regional / limited-area (Requirement 9) ────────────────────────────────

  /// Whether limited-area lateral boundary conditions are active
  /// (Regional_Mode). Controls boundary tendency reads and relaxation-zone
  /// adjustments (Req 9.1-9.10).
  const bool config_apply_lbcs;

  // ── Dissipation / mixing (Requirement 8) ───────────────────────────────────

  /// Whether vertical mixing operates on the full state (true) or on the
  /// perturbation from the initial 1-D state (false) (Req 8.8).
  const bool config_mix_full;

  /// LES turbulence model option string (e.g. "none", "smag3d", "tke15").
  /// Converted to an internal enum by Dissipation_Module::les_model_from_string
  /// (Req 8.10).
  const std::string config_les_model;

  /// LES surface flux boundary condition option string (e.g. "none", "monin_obukhov").
  /// Converted to an internal enum by Dissipation_Module::les_surface_from_string
  /// (Req 8.11).
  const std::string config_les_surface;

  // ── Incremental Analysis Update (Requirement 10) ───────────────────────────

  /// Whether IAU forcing is enabled (Req 10.1, 10.2).
  const bool config_iau;

  // ── Communication (Requirement 11) ─────────────────────────────────────────

  /// Whether halo exchanges should use GPU-aware MPI, exchanging device-
  /// resident data directly without staging through host (Req 11.5, 11.6).
  const bool gpu_aware_comm;

  /// Halo exchange method identifier (e.g. "direct", "grouped").
  /// Validated by Halo_Manager::validate_method (Req 11.10).
  const std::string halo_exchange_method;

  const double config_smdiv;
  const double config_len_disp;
  const double config_apvm_upwinding;
  const bool config_hollingsworth;
};

/// Builder helper for constructing a `Config` with named parameters.
///
/// Because `Config` has all-const members, aggregate initialization is the
/// primary construction path (especially with C++20 designated initializers).
/// This builder is provided as a convenience for call sites where designated
/// initializers are cumbersome or where values are populated incrementally
/// (e.g. reading from an MPAS pool one field at a time).
///
/// Usage:
/// @code
///   auto cfg = ConfigBuilder{}
///       .time_integration_order(3)
///       .number_of_sub_steps(6)
///       .dynamics_split_steps(1)
///       .config_monotonic(true)
///       .config_scalar_advection(true)
///       .config_apply_lbcs(false)
///       .config_mix_full(true)
///       .config_les_model("none")
///       .config_les_surface("none")
///       .config_iau(false)
///       .gpu_aware_comm(false)
///       .halo_exchange_method("direct")
///       .build();
/// @endcode
class ConfigBuilder {
public:
  ConfigBuilder& time_integration_scheme(std::string v) { time_integration_scheme_ = std::move(v); return *this; }
  ConfigBuilder& time_integration_order(int v) { time_integration_order_ = v; return *this; }
  ConfigBuilder& number_of_sub_steps(int v) { number_of_sub_steps_ = v; return *this; }
  ConfigBuilder& dynamics_split_steps(int v) { dynamics_split_steps_ = v; return *this; }
  ConfigBuilder& config_monotonic(bool v) { config_monotonic_ = v; return *this; }
  ConfigBuilder& config_scalar_advection(bool v) { config_scalar_advection_ = v; return *this; }
  ConfigBuilder& config_apply_lbcs(bool v) { config_apply_lbcs_ = v; return *this; }
  ConfigBuilder& config_mix_full(bool v) { config_mix_full_ = v; return *this; }
  ConfigBuilder& config_les_model(std::string v) { config_les_model_ = std::move(v); return *this; }
  ConfigBuilder& config_les_surface(std::string v) { config_les_surface_ = std::move(v); return *this; }
  ConfigBuilder& config_iau(bool v) { config_iau_ = v; return *this; }
  ConfigBuilder& gpu_aware_comm(bool v) { gpu_aware_comm_ = v; return *this; }
  ConfigBuilder& halo_exchange_method(std::string v) { halo_exchange_method_ = std::move(v); return *this; }
  ConfigBuilder& config_smdiv(double v) { config_smdiv_ = v; return *this; }
  ConfigBuilder& config_len_disp(double v) { config_len_disp_ = v; return *this; }
  ConfigBuilder& config_apvm_upwinding(double v) { config_apvm_upwinding_ = v; return *this; }
  ConfigBuilder& config_hollingsworth(bool v) { config_hollingsworth_ = v; return *this; }

  /// Construct an immutable `Config` from the accumulated values.
  [[nodiscard]] Config build() const {
    return Config{
        .time_integration_scheme = time_integration_scheme_,
        .time_integration_order = time_integration_order_,
        .number_of_sub_steps = number_of_sub_steps_,
        .dynamics_split_steps = dynamics_split_steps_,
        .config_monotonic = config_monotonic_,
        .config_scalar_advection = config_scalar_advection_,
        .config_apply_lbcs = config_apply_lbcs_,
        .config_mix_full = config_mix_full_,
        .config_les_model = config_les_model_,
        .config_les_surface = config_les_surface_,
        .config_iau = config_iau_,
        .gpu_aware_comm = gpu_aware_comm_,
        .halo_exchange_method = halo_exchange_method_,
        .config_smdiv = config_smdiv_,
        .config_len_disp = config_len_disp_,
        .config_apvm_upwinding = config_apvm_upwinding_,
        .config_hollingsworth = config_hollingsworth_,
    };
  }

private:
  std::string time_integration_scheme_ = "SRK3";
  int time_integration_order_ = 3;
  int number_of_sub_steps_ = 6;
  int dynamics_split_steps_ = 1;
  bool config_monotonic_ = true;
  bool config_scalar_advection_ = true;
  bool config_apply_lbcs_ = false;
  bool config_mix_full_ = true;
  std::string config_les_model_ = "none";
  std::string config_les_surface_ = "none";
  bool config_iau_ = false;
  bool gpu_aware_comm_ = false;
  std::string halo_exchange_method_ = "direct";
  double config_smdiv_ = 0.1;
  double config_len_disp_ = 120000.0;
  double config_apvm_upwinding_ = 0.0;
  bool config_hollingsworth_ = true;
};

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_CONFIG_HPP
