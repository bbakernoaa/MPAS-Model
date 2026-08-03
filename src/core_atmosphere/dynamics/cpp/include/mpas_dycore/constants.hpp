#pragma once

/// @file constants.hpp
/// @brief Named physical constants for the MPAS dynamical core.
///
/// All values match the Fortran MPAS reference implementation
/// (src/framework/mpas_constants.F) to full double precision.
/// Constants are defined as inline constexpr to ensure compile-time
/// evaluation and zero runtime overhead.
///
/// @see src/framework/mpas_constants.F in the MPAS-Model repository.

#include "mpas_dycore/types.hpp"

namespace mpas::dycore::constants {

/// @brief Gravitational acceleration [m s^-2].
/// @details Standard value used in MPAS atmospheric dynamics.
/// Reference: WMO standard atmosphere.
inline constexpr real_type gravity = 9.80616;

/// @brief Gas constant for dry air [J kg^-1 K^-1].
/// @details R_d in the ideal gas law for dry air: p = rho * R_d * T.
inline constexpr real_type rdry = 287.0;

/// @brief Gas constant for water vapor [J kg^-1 K^-1].
/// @details R_v in the ideal gas law for water vapor.
inline constexpr real_type rvapor = 461.6;

/// @brief Specific heat of dry air at constant pressure [J kg^-1 K^-1].
/// @details Computed as 7/2 * R_d (diatomic ideal gas).
/// In Fortran MPAS: cp = 7.0 * rgas / 2.0 = 1004.5.
inline constexpr real_type cpdry = 7.0 * rdry / 2.0;

/// @brief Specific heat of dry air at constant volume [J kg^-1 K^-1].
/// @details Follows from Mayer's relation: c_v = c_p - R_d.
/// In Fortran MPAS: cv = cp - rgas = 717.5.
inline constexpr real_type cvdry = cpdry - rdry;

/// @brief Reference pressure [Pa].
/// @details Standard reference pressure (1000 hPa) used in potential
/// temperature and Exner function computations.
inline constexpr real_type pref = 1.0e5;

/// @brief Mean Earth radius [m].
/// @details Spherical Earth approximation used in MPAS mesh generation
/// and curvature terms.
inline constexpr real_type rearth = 6371229.0;

/// @brief Mathematical constant pi.
/// @details Full double-precision value matching the Fortran MPAS literal.
inline constexpr real_type pi = 3.141592653589793;

/// @brief Angular rotation rate of the Earth [rad s^-1].
/// @details Used in Coriolis force computation: f = 2 * omega * sin(lat).
inline constexpr real_type omega = 7.29212e-5;

/// @brief Number of seconds in one day [s].
/// @details Used for converting timescales (e.g., Rayleigh damping)
/// from days to seconds.
inline constexpr real_type seconds_per_day = 86400.0;

} // namespace mpas::dycore::constants
