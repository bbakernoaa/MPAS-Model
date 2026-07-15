#ifndef MPAS_DYCORE_LES_OPTIONS_HPP
#define MPAS_DYCORE_LES_OPTIONS_HPP

/// @file les_options.hpp
/// @brief LES option-string mapping functions for the Dissipation_Module.
///
/// These functions convert namelist option strings for the LES turbulence
/// model and LES surface flux scheme to integer option codes, matching the
/// Reference_Model (mpas_atm_dissipation_models.F) exactly.
///
/// Requirements: 8.10, 8.11

#include <string_view>

namespace mpas {
namespace dycore {

/// Integer option codes for the LES turbulence model, matching the
/// Reference_Model parameters in mpas_atm_dissipation_models.F.
inline constexpr int LES_INVALID_OPT = -1;

inline constexpr int LES_MODEL_NONE = 0;
inline constexpr int LES_MODEL_3D_SMAGORINSKY = 1;
inline constexpr int LES_MODEL_PROGNOSTIC_15_ORDER = 2;

inline constexpr int LES_SURFACE_NONE = 0;
inline constexpr int LES_SURFACE_SPECIFIED = 1;
inline constexpr int LES_SURFACE_VARYING = 2;

/// Convert an LES model option string to its integer code.
///
/// Valid inputs (matching the Reference_Model):
///   "none"                  → LES_MODEL_NONE (0)
///   "3d_smagorinsky"       → LES_MODEL_3D_SMAGORINSKY (1)
///   "prognostic_1.5_order" → LES_MODEL_PROGNOSTIC_15_ORDER (2)
///
/// Any unrecognized string returns LES_INVALID_OPT (-1).
///
/// Requirement 8.10
[[nodiscard]] constexpr int les_model_from_string(std::string_view les_model_str) noexcept {
  if (les_model_str == "none") {
    return LES_MODEL_NONE;
  }
  if (les_model_str == "3d_smagorinsky") {
    return LES_MODEL_3D_SMAGORINSKY;
  }
  if (les_model_str == "prognostic_1.5_order") {
    return LES_MODEL_PROGNOSTIC_15_ORDER;
  }
  return LES_INVALID_OPT;
}

/// Convert an LES surface option string to its integer code.
///
/// Valid inputs (matching the Reference_Model):
///   "none"      → LES_SURFACE_NONE (0)
///   "specified" → LES_SURFACE_SPECIFIED (1)
///   "varying"   → LES_SURFACE_VARYING (2)
///
/// Any unrecognized string returns LES_INVALID_OPT (-1).
///
/// Requirement 8.11
[[nodiscard]] constexpr int les_surface_from_string(std::string_view les_surface_str) noexcept {
  if (les_surface_str == "none") {
    return LES_SURFACE_NONE;
  }
  if (les_surface_str == "specified") {
    return LES_SURFACE_SPECIFIED;
  }
  if (les_surface_str == "varying") {
    return LES_SURFACE_VARYING;
  }
  return LES_INVALID_OPT;
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_LES_OPTIONS_HPP
