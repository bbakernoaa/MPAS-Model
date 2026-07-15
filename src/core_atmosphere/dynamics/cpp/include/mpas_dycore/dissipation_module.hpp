#ifndef MPAS_DYCORE_DISSIPATION_MODULE_HPP
#define MPAS_DYCORE_DISSIPATION_MODULE_HPP

/// @file dissipation_module.hpp
/// @brief Dissipation and LES mixing models for the C++ dycore.
///
/// Implements compute_eddy_viscosity() covering:
/// - 2-D Smagorinsky horizontal eddy viscosity (Req 8.1)
/// - Fixed horizontal eddy viscosity (Req 8.2)
/// - 3-D Smagorinsky LES (Req 8.3)
/// - Prognostic 1.5-order TKE LES with TKE tendency (Req 8.4)
/// - Moist Brunt-Väisälä frequency (Req 8.5)
/// - Eddy-viscosity stability bounding (Req 8.6)
///
/// Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6

#include "mpas_dycore/config.hpp"
#include "mpas_dycore/les_options.hpp"
#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <string>

namespace mpas {
namespace dycore {

/// Physical constants matching the Reference_Model (mpas_constants.F).
namespace constants {
inline constexpr Scalar gravity = 9.80616;
inline constexpr Scalar rgas = 287.0;
inline constexpr Scalar rv = 461.6;
inline constexpr Scalar cp = 7.0 * rgas / 2.0;
inline constexpr Scalar rvord = rv / rgas;
inline constexpr Scalar prandtl = 1.0;
}  // namespace constants

/// Constants from the Reference_Model dissipation module.
namespace dissipation_constants {
inline constexpr Scalar epsilon_bv = 1.0e-06;
inline constexpr Scalar c_k = 0.25;
inline constexpr Scalar qc_cr = 0.00001;  // cloud water threshold [kg/kg]

// Physics constants for moist Brunt-Väisälä (stand-alone definitions)
inline constexpr Scalar svp1 = 0.6112;
inline constexpr Scalar svp2 = 17.67;
inline constexpr Scalar svp3 = 29.65;
inline constexpr Scalar svpt0 = 273.15;
inline constexpr Scalar xlv = 2.50e6;  // latent heat of vaporization [J/kg]
inline constexpr Scalar R_d = 287.0;
inline constexpr Scalar R_v = 461.6;
inline constexpr Scalar ep_2 = R_d / R_v;
}  // namespace dissipation_constants

}  // namespace dycore (temporarily close for vertical_mixing.hpp inclusion)
}  // namespace mpas

// Include vertical_mixing.hpp here (after constants/dissipation_constants are
// defined) because it references symbols from those namespaces.
#include "mpas_dycore/vertical_mixing.hpp"

namespace mpas {
namespace dycore {

/// Parameters for the eddy viscosity computation, sourced from the Config
/// and mesh properties at construction time.
struct EddyViscosityParams {
  int les_model_opt = LES_MODEL_NONE;
  Scalar c_s = 0.0;             // Smagorinsky coefficient
  Scalar config_len_disp = 0.0; // horizontal filter length scale [m]
  Scalar invDt = 0.0;           // 1/dt for stability bounding
  Scalar config_visc4_2dsmag = 0.0; // 4th-order background visc coefficient
  int dynamics_substep = 1;     // current dynamics substep (for TKE tendency)
  int index_tke = -1;           // scalar index for TKE (1-based Fortran index)
  int index_qv = -1;            // scalar index for water vapor (1-based)
  int index_qc = -1;            // scalar index for cloud water (1-based, or <=0 if absent)
  int nVertLevels = 0;
  int nCells = 0;
  int nEdges = 0;
  int maxEdges = 0;
  int num_scalars = 0;
};

/// Output from compute_eddy_viscosity: the computed viscosity fields and
/// the 4th-order background filter coefficients.
struct EddyViscosityResult {
  Scalar h_mom_eddy_visc4 = 0.0;
  Scalar h_theta_eddy_visc4 = 0.0;
};

/// @brief Compute the moist Brunt-Väisälä frequency N^2 (Requirement 8.5).
///
/// Uses the dry formulation where cloud water < qc_cr threshold, and the
/// moist formulation (Durran and Klemp 1982, Eq. 36) otherwise.
///
/// @tparam ExecSpace  Kokkos execution space.
/// @param bn2         Output: (nVertLevels, nCells) Brunt-Väisälä frequency squared.
/// @param theta_m     Input: coupled potential temperature (nVertLevels, nCells).
/// @param exner       Input: Exner function (nVertLevels, nCells).
/// @param pressure_b  Input: base-state pressure (nVertLevels, nCells).
/// @param pp          Input: pressure perturbation (nVertLevels, nCells).
/// @param zgrid       Input: layer interface heights (nVertLevels+1, nCells).
/// @param scalars     Input: scalar fields (num_scalars, nVertLevels, nCells).
/// @param qtot        Input: total water mixing ratio (nVertLevels, nCells).
/// @param params      Parameters (index_qv, index_qc, nVertLevels, nCells, num_scalars).
template <class ExecSpace>
void calculate_brunt_vaisala(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& bn2,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& exner,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& pressure_b,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& pp,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zgrid,
    const Kokkos::View<const Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& scalars,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& qtot,
    const EddyViscosityParams& params) {

  namespace dc = dissipation_constants;
  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  // Fortran 1-based to C++ 0-based index conversion
  const int idx_qv = params.index_qv - 1;
  const int idx_qc = params.index_qc - 1;

  Kokkos::parallel_for(
      "calculate_brunt_vaisala",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        using Kokkos::fmax;
        using Kokkos::fmin;

        // Compute theta, temp, qvsw, coefa for this column
        // Use stack arrays for intermediate per-level values.
        // Since nVertLevels may be large, we do the computation in-line.

        for (int k = 1; k < nVertLevels - 1; ++k) {
          // theta = theta_m / (1 + rvord * qv)
          const Scalar qv_k = scalars(idx_qv, k, iCell);
          const Scalar theta_k = theta_m(k, iCell) /
                                 (Scalar(1.0) + constants::rvord * qv_k);
          const Scalar temp_k = exner(k, iCell) * theta_k;

          const Scalar p = pressure_b(k, iCell) + pp(k, iCell);
          Scalar esw = Scalar(1000.0) * dc::svp1 *
                       Kokkos::exp(dc::svp2 * (temp_k - dc::svpt0) /
                                   (temp_k - dc::svp3));
          if (p < esw) esw = p * Scalar(0.99);
          const Scalar qvsw_k = dc::ep_2 * esw / (p - esw);

          const Scalar coefa_k =
              (Scalar(1.0) + dc::xlv * qvsw_k / dc::R_d / temp_k) /
              (Scalar(1.0) + dc::xlv * dc::xlv * qvsw_k / constants::cp /
                             dc::R_v / temp_k / temp_k);

          // Need theta at k-1 and k+1 for finite differences
          const Scalar qv_km1 = scalars(idx_qv, k - 1, iCell);
          const Scalar theta_km1 = theta_m(k - 1, iCell) /
                                   (Scalar(1.0) + constants::rvord * qv_km1);
          const Scalar qv_kp1 = scalars(idx_qv, k + 1, iCell);
          const Scalar theta_kp1 = theta_m(k + 1, iCell) /
                                   (Scalar(1.0) + constants::rvord * qv_kp1);

          // qvsw at k-1 and k+1 for moist formulation
          const Scalar temp_km1 = exner(k - 1, iCell) * theta_km1;
          const Scalar p_km1 = pressure_b(k - 1, iCell) + pp(k - 1, iCell);
          Scalar esw_km1 = Scalar(1000.0) * dc::svp1 *
                           Kokkos::exp(dc::svp2 * (temp_km1 - dc::svpt0) /
                                       (temp_km1 - dc::svp3));
          if (p_km1 < esw_km1) esw_km1 = p_km1 * Scalar(0.99);
          const Scalar qvsw_km1 = dc::ep_2 * esw_km1 / (p_km1 - esw_km1);

          const Scalar temp_kp1 = exner(k + 1, iCell) * theta_kp1;
          const Scalar p_kp1 = pressure_b(k + 1, iCell) + pp(k + 1, iCell);
          Scalar esw_kp1 = Scalar(1000.0) * dc::svp1 *
                           Kokkos::exp(dc::svp2 * (temp_kp1 - dc::svpt0) /
                                       (temp_kp1 - dc::svp3));
          if (p_kp1 < esw_kp1) esw_kp1 = p_kp1 * Scalar(0.99);
          const Scalar qvsw_kp1 = dc::ep_2 * esw_kp1 / (p_kp1 - esw_kp1);

          // Vertical spacing: dz between k-1 and k+1 centers
          const Scalar dz = Scalar(0.5) * (zgrid(k + 2, iCell) + zgrid(k + 1, iCell)) -
                            Scalar(0.5) * (zgrid(k, iCell) + zgrid(k - 1, iCell));
          const Scalar rdz = Scalar(1.0) / dz;

          // Select dry or moist formulation based on cloud water
          bool dry_bv = true;
          if (idx_qc >= 0) {
            if (scalars(idx_qc, k, iCell) >= dc::qc_cr) {
              dry_bv = false;
            }
          }

          if (dry_bv) {
            // Dry Brunt-Väisälä frequency
            bn2(k, iCell) = constants::gravity *
                ((theta_kp1 - theta_km1) / theta_k * rdz +
                 constants::rvord *
                     (scalars(idx_qv, k + 1, iCell) -
                      scalars(idx_qv, k - 1, iCell)) * rdz -
                 (qtot(k + 1, iCell) - qtot(k - 1, iCell)) * rdz);
          } else {
            // Moist Brunt-Väisälä frequency (Durran & Klemp 1982 Eq. 36)
            bn2(k, iCell) = constants::gravity *
                (coefa_k *
                     ((theta_kp1 - theta_km1) / theta_k * rdz +
                      dc::xlv / constants::cp / temp_k *
                          (qvsw_kp1 - qvsw_km1) * rdz) -
                 (qtot(k + 1, iCell) - qtot(k - 1, iCell)) * rdz);
          }
        }

        // Boundary levels: copy from interior
        bn2(0, iCell) = bn2(1, iCell);
        bn2(nVertLevels - 1, iCell) = bn2(nVertLevels - 2, iCell);
      });
}

/// @brief Compute 2-D Smagorinsky horizontal eddy viscosity (Requirement 8.1).
///
/// Computes kdiff from deformation using the Reference_Model formulation,
/// then bounds by the stability limit (Requirement 8.6).
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
EddyViscosityResult smagorinsky_2d(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& kdiff,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& v,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_c2,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_s2,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_cs,
    const Kokkos::View<const int*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& nEdgesOnCell,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell,
    const EddyViscosityParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const Scalar c_s = params.c_s;
  const Scalar config_len_disp = params.config_len_disp;
  const Scalar invDt = params.invDt;
  // Stability limit: 0.01 * config_len_disp^2 * invDt (Req 8.6)
  const Scalar stability_limit = Scalar(0.01) * config_len_disp * config_len_disp * invDt;
  const Scalar smag_coeff = (c_s * config_len_disp) * (c_s * config_len_disp);

  Kokkos::parallel_for(
      "smagorinsky_2d",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        for (int k = 0; k < nVertLevels; ++k) {
          Scalar dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;

          const int nEdges_i = nEdgesOnCell(iCell);
          for (int iEdge = 0; iEdge < nEdges_i; ++iEdge) {
            // edgesOnCell stores 1-based Fortran indices
            const int ie = edgesOnCell(iEdge, iCell) - 1;
            const Scalar coef_c2 = deformation_coef_c2(iEdge, iCell);
            const Scalar coef_s2 = deformation_coef_s2(iEdge, iCell);
            const Scalar coef_cs = deformation_coef_cs(iEdge, iCell);

            dudx += coef_c2 * u(k, ie) - coef_cs * v(k, ie);
            dudy += coef_cs * u(k, ie) - coef_s2 * v(k, ie);
            dvdx += coef_cs * u(k, ie) + coef_c2 * v(k, ie);
            dvdy += coef_s2 * u(k, ie) + coef_cs * v(k, ie);
          }

          // Smagorinsky formulation
          const Scalar d_11 = Scalar(2.0) * dudx;
          const Scalar d_22 = Scalar(2.0) * dvdy;
          const Scalar d_12 = dudy + dvdx;
          Scalar visc = smag_coeff *
              Kokkos::sqrt(Scalar(0.25) * (d_11 - d_22) * (d_11 - d_22) +
                           d_12 * d_12);
          // Stability bounding (Req 8.6)
          visc = Kokkos::fmin(visc, stability_limit);
          kdiff(k, iCell) = visc;
        }
      });

  // Background 4th-order filter coefficients
  EddyViscosityResult result;
  result.h_mom_eddy_visc4 = params.config_visc4_2dsmag *
                             config_len_disp * config_len_disp * config_len_disp;
  result.h_theta_eddy_visc4 = result.h_mom_eddy_visc4;
  return result;
}

/// @brief Set horizontal eddy viscosity to a fixed configured value (Requirement 8.2).
///
/// Fills kdiff with the fixed second-order viscosity value, bounded by stability.
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void fixed_horizontal_eddy_viscosity(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& kdiff,
    Scalar fixed_visc2,
    const EddyViscosityParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const Scalar invDt = params.invDt;
  const Scalar config_len_disp = params.config_len_disp;
  // Stability limit (Req 8.6)
  const Scalar stability_limit = Scalar(0.01) * config_len_disp * config_len_disp * invDt;
  const Scalar bounded_visc = Kokkos::fmin(fixed_visc2, stability_limit);

  Kokkos::parallel_for(
      "fixed_horizontal_eddy_viscosity",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        for (int k = 0; k < nVertLevels; ++k) {
          kdiff(k, iCell) = bounded_visc;
        }
      });
}

/// @brief Compute 3-D Smagorinsky / prognostic TKE eddy viscosities (Req 8.3, 8.4).
///
/// - LES_MODEL_3D_SMAGORINSKY: horizontal and vertical eddy viscosities from
///   the three-dimensional deformation tensor and moist Brunt-Väisälä frequency.
/// - LES_MODEL_PROGNOSTIC_15_ORDER: eddy viscosities from prognostic TKE with
///   TKE tendency computed on dynamics_substep == 1.
///
/// Both models apply the stability bound (Req 8.6).
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
EddyViscosityResult les_models(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& eddy_visc_horz,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& eddy_visc_vert,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& prandtl_3d_inv,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& v,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& uCell,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& vCell,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& bv_freq2,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zgrid,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& scalars,
    const Kokkos::View<Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_scalars,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_c2,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_s2,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_cs,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_c,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_s,
    const Kokkos::View<const int*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& nEdgesOnCell,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
    const EddyViscosityParams& params) {

  namespace dc = dissipation_constants;
  const int les_model_opt = params.les_model_opt;
  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const Scalar c_s = params.c_s;
  const Scalar config_len_disp = params.config_len_disp;
  const Scalar invDt = params.invDt;
  const int dynamics_substep = params.dynamics_substep;
  const int idx_tke = params.index_tke - 1;  // 0-based

  // Horizontal stability limit
  const Scalar h_stability_limit = Scalar(0.01) * config_len_disp * config_len_disp * invDt;
  // Prandtl number inverse from Reference_Model
  const Scalar pr_inv = Scalar(1.0) / constants::prandtl;

  Kokkos::parallel_for(
      "les_models",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // ─── Compute 3-D deformation tensor ─────────────────────────────────

        // Horizontal deformation gradients accumulated over edges
        // We inline-compute per-level to avoid dynamic allocation in the kernel.
        // First pass: compute dudz, dvdz from cell-centered velocities.

        for (int k = 0; k < nVertLevels; ++k) {
          Scalar dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;

          const int nEdges_i = nEdgesOnCell(iCell);
          for (int iEdge = 0; iEdge < nEdges_i; ++iEdge) {
            const int ie = edgesOnCell(iEdge, iCell) - 1;
            const Scalar coef_c2 = deformation_coef_c2(iEdge, iCell);
            const Scalar coef_s2 = deformation_coef_s2(iEdge, iCell);
            const Scalar coef_cs = deformation_coef_cs(iEdge, iCell);

            dudx += coef_c2 * u(k, ie) - coef_cs * v(k, ie);
            dudy += coef_cs * u(k, ie) - coef_s2 * v(k, ie);
            dvdx += coef_cs * u(k, ie) + coef_c2 * v(k, ie);
            dvdy += coef_s2 * u(k, ie) + coef_cs * v(k, ie);
          }

          // dwdx, dwdy: horizontal gradient of w at level k (Fortran uses
          // the interface-level gradient at index k directly)
          Scalar dwdx_k = 0.0, dwdy_k = 0.0;
          for (int iEdge = 0; iEdge < nEdges_i; ++iEdge) {
            const int ie = edgesOnCell(iEdge, iCell) - 1;
            const int cell1 = cellsOnEdge(0, ie) - 1;
            const int cell2 = cellsOnEdge(1, ie) - 1;
            // Fortran: wk = 0.5*(w(k,cell1)+w(k,cell2)) at interface k
            // For the full-level deformation d_13(k), Fortran uses dwdx(k)
            // which is accumulated at the same index as the full level.
            const Scalar wk = Scalar(0.5) * (w(k, cell1) + w(k, cell2));
            const Scalar coef_c = deformation_coef_c(iEdge, iCell);
            const Scalar coef_s = deformation_coef_s(iEdge, iCell);
            dwdx_k += coef_c * wk;
            dwdy_k += coef_s * wk;
          }

          // dwdz at full level k
          const Scalar rdz_w = Scalar(1.0) / (zgrid(k + 1, iCell) - zgrid(k, iCell));
          const Scalar dwdz_k = (w(k + 1, iCell) - w(k, iCell)) * rdz_w;

          // dudz, dvdz: vertical gradient of cell-centered velocity
          Scalar dudz_k = 0.0, dvdz_k = 0.0;
          if (k == 0) {
            // Bottom level: one-sided
            const Scalar rdz_uv = Scalar(1.0) / (zgrid(k + 1, iCell) - zgrid(k, iCell));
            dudz_k = (uCell(k + 1, iCell) - uCell(k, iCell)) * rdz_uv;
            dvdz_k = (vCell(k + 1, iCell) - vCell(k, iCell)) * rdz_uv;
          } else if (k == nVertLevels - 1) {
            // Top level: one-sided
            const Scalar rdz_uv = Scalar(1.0) / (zgrid(k + 1, iCell) - zgrid(k, iCell));
            dudz_k = (uCell(k, iCell) - uCell(k - 1, iCell)) * rdz_uv;
            dvdz_k = (vCell(k, iCell) - vCell(k - 1, iCell)) * rdz_uv;
          } else {
            // Interior: centered
            const Scalar rdz_uv = Scalar(1.0) /
                (zgrid(k + 2, iCell) + zgrid(k + 1, iCell) -
                 zgrid(k, iCell) - zgrid(k - 1, iCell));
            dudz_k = (uCell(k + 1, iCell) - uCell(k - 1, iCell)) * rdz_uv;
            dvdz_k = (vCell(k + 1, iCell) - vCell(k - 1, iCell)) * rdz_uv;
          }

          // Deformation tensor components
          const Scalar d_11 = Scalar(2.0) * dudx;
          const Scalar d_22 = Scalar(2.0) * dvdy;
          const Scalar d_33 = Scalar(2.0) * dwdz_k;
          const Scalar d_12 = dudy + dvdx;
          const Scalar d_13 = dwdx_k + dudz_k;
          const Scalar d_23 = dwdy_k + dvdz_k;

          // ─── 3-D Smagorinsky (Req 8.3) ───────────────────────────────────
          if (les_model_opt == LES_MODEL_3D_SMAGORINSKY) {
            const Scalar def2 = Scalar(0.5) * (d_11 * d_11 + d_22 * d_22 + d_33 * d_33) +
                                d_12 * d_12 + d_13 * d_13 + d_23 * d_23;
            const Scalar arg = Kokkos::fmax(Scalar(0.0), def2 - pr_inv * bv_freq2(k, iCell));
            // Horizontal eddy viscosity
            Scalar ev_h = (c_s * config_len_disp) * (c_s * config_len_disp) * Kokkos::sqrt(arg);
            ev_h = Kokkos::fmin(ev_h, h_stability_limit);  // Stability bound (Req 8.6)
            eddy_visc_horz(k, iCell) = ev_h;

            // Vertical eddy viscosity
            const Scalar delta_z = zgrid(k + 1, iCell) - zgrid(k, iCell);
            Scalar ev_v = (c_s * delta_z) * (c_s * delta_z) * Kokkos::sqrt(arg);
            // Vertical stability bound: 0.01 * delta_z^2 * invDt
            const Scalar v_stability_limit = Scalar(0.01) * delta_z * delta_z * invDt;
            ev_v = Kokkos::fmin(ev_v, v_stability_limit);
            eddy_visc_vert(k, iCell) = ev_v;

          // ─── Prognostic 1.5-order TKE (Req 8.4) ────────────────────────
          } else if (les_model_opt == LES_MODEL_PROGNOSTIC_15_ORDER) {
            // Bound TKE: ensure non-negative
            scalars(idx_tke, k, iCell) = Kokkos::fmax(Scalar(0.0), scalars(idx_tke, k, iCell));
            const Scalar tke_val = scalars(idx_tke, k, iCell);

            const Scalar delta_z = zgrid(k + 1, iCell) - zgrid(k, iCell);
            const Scalar delta_s = Kokkos::pow(
                config_len_disp * config_len_disp * delta_z, Scalar(1.0) / Scalar(3.0));
            const Scalar bv = Kokkos::fmax(
                Kokkos::sqrt(Kokkos::fabs(bv_freq2(k, iCell))), dc::epsilon_bv);

            // Isentropic mixing length
            Scalar tke_length = delta_s;
            if (bv_freq2(k, iCell) > Scalar(1.0e-06)) {
              tke_length = Scalar(0.76) * Kokkos::sqrt(tke_val) / bv;
            }
            tke_length = Kokkos::fmin(tke_length, delta_z);

            Scalar diss_length = Kokkos::fmin(delta_s,
                Kokkos::fmax(tke_length, Scalar(0.01) * delta_s));
            if (bv_freq2(k, iCell) <= Scalar(0.0)) diss_length = delta_s;

            // Non-isotropic mixing
            const Scalar l_horizontal = config_len_disp;
            Scalar l_vertical = Kokkos::fmin(delta_z, tke_length);
            if (bv_freq2(k, iCell) <= Scalar(0.0)) diss_length = delta_z;

            // Eddy viscosities from TKE
            Scalar ev_h = dc::c_k * l_horizontal * Kokkos::sqrt(tke_val);
            ev_h = Kokkos::fmin(ev_h, h_stability_limit);  // Stability bound (Req 8.6)
            const Scalar v_stability = Scalar(0.01) * delta_z * delta_z * invDt;
            Scalar ev_v = dc::c_k * l_vertical * Kokkos::sqrt(tke_val);
            ev_v = Kokkos::fmin(ev_v, v_stability);  // Stability bound (Req 8.6)

            eddy_visc_horz(k, iCell) = ev_h;
            eddy_visc_vert(k, iCell) = ev_v;

            // TKE tendency (only on first dynamics substep)
            if (dynamics_substep == 1) {
              const Scalar shear_production =
                  ev_h * (d_11 * d_11 + d_22 * d_22 + d_12 * d_12) +
                  ev_v * (d_33 * d_33 + d_13 * d_13 + d_23 * d_23);
              const Scalar buoyancy = -ev_v * bv_freq2(k, iCell);
              const Scalar c_dissipation =
                  Scalar(1.9) * dc::c_k +
                  Kokkos::fmax(Scalar(0.0), Scalar(0.93) - Scalar(1.9) * dc::c_k) *
                      diss_length / delta_s;

              const Scalar dissipation =
                  -c_dissipation * Kokkos::pow(tke_val, Scalar(1.5)) / diss_length;

              tend_scalars(idx_tke, k, iCell) =
                  rho_zz(k, iCell) * (shear_production + buoyancy + dissipation);
            }

            // Prandtl number inverse for vertical mixing (3-D)
            prandtl_3d_inv(k, iCell) = Scalar(1.0) + Scalar(2.0) * l_vertical / delta_z;
          }
        }  // end k loop
      });  // end parallel_for

  // Background 4th-order filter coefficients (same for both LES models)
  EddyViscosityResult result;
  result.h_mom_eddy_visc4 = params.config_visc4_2dsmag *
      config_len_disp * config_len_disp * config_len_disp;
  result.h_theta_eddy_visc4 = result.h_mom_eddy_visc4;
  return result;
}

/// Parameters controlling the fourth-order horizontal hyperdiffusion filter.
struct HyperdiffusionParams {
  Scalar h_mom_eddy_visc4 = 0.0;     ///< 4th-order visc coeff for u (momentum)
  Scalar h_theta_eddy_visc4 = 0.0;   ///< 4th-order visc coeff for theta
  Scalar config_del4u_div_factor = 1.0; ///< factor scaling div part of u filter
  Scalar prandtl_inv = 1.0;          ///< inverse Prandtl number for theta/scalar filtering
  bool config_mix_scalars = false;    ///< whether to apply filter to scalars
  int nVertLevels = 0;
  int nCells = 0;
  int nEdges = 0;
  int nVertices = 0;
  int maxEdges = 0;
  int vertexDegree = 3;              ///< vertex degree (typically 3 for triangular dual)
  int num_scalars = 0;
};

/// @brief Apply fourth-order horizontal hyperdiffusion (Requirement 8.7).
///
/// Implements del2(del2(field)) for:
///   - u (horizontal momentum) on edges using the div/vort decomposition
///   - w (vertical velocity) on cells
///   - theta_m (potential temperature) on cells
///   - scalars (optional, when config_mix_scalars is true)
///
/// The pattern for cell-based fields (w, theta, scalars):
///   Step 1: delsq(k,iCell) = sum_edges[ sign * dvEdge * invDcEdge * (f(cell2)-f(cell1)) * rho_edge ] / areaCell
///   Step 2: tend(k,iCell) -= visc4 * meshScalingDel4 * sum_edges[ sign * dvEdge * invDcEdge * (delsq(cell2)-delsq(cell1)) ] / areaCell
///
/// The pattern for u (edge-based):
///   Step 1: delsq_u(k,iEdge) = grad_n(div) - grad_t(vort)
///                             = (div(cell2)-div(cell1))/dcEdge - (vort(v2)-vort(v1))/min(dvEdge, 4*dcEdge)
///   Step 2: Compute delsq_vorticity and delsq_divergence from delsq_u (same as original vort/div from u)
///   Step 3: tend_u(k,iEdge) -= rho_edge * visc4 * meshScalingDel4 *
///                              [ del4u_div_factor*(delsq_div(c2)-delsq_div(c1))/dcEdge
///                                - (delsq_vort(v2)-delsq_vort(v1))/min(dvEdge,4*dcEdge) ]
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void apply_hyperdiffusion(
    // Tendency output fields (accumulated into)
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_u,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_w,
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_theta,
    const Kokkos::View<Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_scalars,
    // Input state fields
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
    const Kokkos::View<const Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& scalars,
    // Diagnostic fields
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& divergence,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& vorticity,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_edge,
    // Mesh connectivity/geometry
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& verticesOnEdge,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnVertex,
    const Kokkos::View<const int*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& nEdgesOnCell_v,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell_sign,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnVertex_sign,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invAreaCell,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invAreaTriangle,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invDcEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invDvEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dvEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dcEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& meshScalingDel4,
    // Parameters
    const HyperdiffusionParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const int nEdges = params.nEdges;
  const int nVertices = params.nVertices;
  const int vertexDegree = params.vertexDegree;
  const Scalar h_mom_eddy_visc4 = params.h_mom_eddy_visc4;
  const Scalar h_theta_eddy_visc4 = params.h_theta_eddy_visc4;
  const Scalar config_del4u_div_factor = params.config_del4u_div_factor;
  const Scalar prandtl_inv = params.prandtl_inv;

  using view2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  // ════════════════════════════════════════════════════════════════════════════
  // PART 1: Fourth-order filter for u (horizontal momentum on edges)
  // ════════════════════════════════════════════════════════════════════════════
  if (h_mom_eddy_visc4 > Scalar(0.0)) {
    // Step 1: Compute delsq_u = grad_n(div) - grad_t(vort) at all edges
    view2d delsq_u("delsq_u", nVertLevels, nEdges);

    Kokkos::parallel_for(
        "hyperdiff::delsq_u",
        Kokkos::RangePolicy<ExecSpace>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;
          const int vertex1 = verticesOnEdge(0, iEdge) - 1;
          const int vertex2 = verticesOnEdge(1, iEdge) - 1;
          const Scalar r_dc = invDcEdge(iEdge);
          const Scalar r_dv = Kokkos::fmin(invDvEdge(iEdge),
                                           Scalar(4.0) * invDcEdge(iEdge));

          for (int k = 0; k < nVertLevels; ++k) {
            delsq_u(k, iEdge) = (divergence(k, cell2) - divergence(k, cell1)) * r_dc
                              - (vorticity(k, vertex2) - vorticity(k, vertex1)) * r_dv;
          }
        });
    Kokkos::fence("hyperdiff::delsq_u_fence");

    // Step 2: Compute delsq_vorticity at vertices from delsq_u
    view2d delsq_vorticity("delsq_vorticity", nVertLevels, nVertices);

    Kokkos::parallel_for(
        "hyperdiff::delsq_vorticity",
        Kokkos::RangePolicy<ExecSpace>(0, nVertices),
        KOKKOS_LAMBDA(const int iVertex) {
          for (int k = 0; k < nVertLevels; ++k) {
            Scalar acc = Scalar(0.0);
            for (int i = 0; i < vertexDegree; ++i) {
              const int iEdge = edgesOnVertex(i, iVertex) - 1;
              const Scalar sign_dc = invAreaTriangle(iVertex) * dcEdge(iEdge) *
                                     edgesOnVertex_sign(i, iVertex);
              acc += sign_dc * delsq_u(k, iEdge);
            }
            delsq_vorticity(k, iVertex) = acc;
          }
        });

    // Step 2b: Compute delsq_divergence at cells from delsq_u
    view2d delsq_divergence("delsq_divergence", nVertLevels, nCells);

    Kokkos::parallel_for(
        "hyperdiff::delsq_divergence",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          const Scalar r = invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);
          for (int k = 0; k < nVertLevels; ++k) {
            Scalar acc = Scalar(0.0);
            for (int i = 0; i < ne; ++i) {
              const int iEdge = edgesOnCell(i, iCell) - 1;
              const Scalar sign_dv = r * dvEdge(iEdge) * edgesOnCell_sign(i, iCell);
              acc += sign_dv * delsq_u(k, iEdge);
            }
            delsq_divergence(k, iCell) = acc;
          }
        });
    Kokkos::fence("hyperdiff::delsq_div_vort_fence");

    // Step 3: Apply del4 u filter to tend_u
    Kokkos::parallel_for(
        "hyperdiff::tend_u_del4",
        Kokkos::RangePolicy<ExecSpace>(0, nEdges),
        KOKKOS_LAMBDA(const int iEdge) {
          const int cell1 = cellsOnEdge(0, iEdge) - 1;
          const int cell2 = cellsOnEdge(1, iEdge) - 1;
          const int vertex1 = verticesOnEdge(0, iEdge) - 1;
          const int vertex2 = verticesOnEdge(1, iEdge) - 1;

          const Scalar u_mix_scale = meshScalingDel4(iEdge) * h_mom_eddy_visc4;
          const Scalar r_dc = u_mix_scale * config_del4u_div_factor * invDcEdge(iEdge);
          const Scalar r_dv = u_mix_scale *
              Kokkos::fmin(invDvEdge(iEdge), Scalar(4.0) * invDcEdge(iEdge));

          for (int k = 0; k < nVertLevels; ++k) {
            const Scalar u_diffusion = rho_edge(k, iEdge) *
                ((delsq_divergence(k, cell2) - delsq_divergence(k, cell1)) * r_dc
               - (delsq_vorticity(k, vertex2) - delsq_vorticity(k, vertex1)) * r_dv);
            tend_u(k, iEdge) -= u_diffusion;
          }
        });
    Kokkos::fence("hyperdiff::tend_u_fence");
  }  // h_mom_eddy_visc4 > 0

  // ════════════════════════════════════════════════════════════════════════════
  // PART 2: Fourth-order filter for w (vertical velocity on cells)
  // ════════════════════════════════════════════════════════════════════════════
  if (h_mom_eddy_visc4 > Scalar(0.0)) {
    // Step 1: Compute delsq_w at cells
    // delsq_w(k,iCell) = sum_edges[ 0.5*sign*dvEdge*invDcEdge*(rho_edge(k,e)+rho_edge(k-1,e))*(w(k,c2)-w(k,c1)) ] / areaCell
    view2d delsq_w("delsq_w", nVertLevels + 1, nCells);

    Kokkos::parallel_for(
        "hyperdiff::delsq_w",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          const Scalar r_areaCell = invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);

          // Initialize
          for (int k = 0; k <= nVertLevels; ++k) {
            delsq_w(k, iCell) = Scalar(0.0);
          }

          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            const Scalar edge_sign = Scalar(0.5) * r_areaCell *
                edgesOnCell_sign(i, iCell) * dvEdge(iEdge) * invDcEdge(iEdge);

            const int cell1 = cellsOnEdge(0, iEdge) - 1;
            const int cell2 = cellsOnEdge(1, iEdge) - 1;

            for (int k = 1; k < nVertLevels; ++k) {
              const Scalar w_turb_flux = edge_sign *
                  (rho_edge(k, iEdge) + rho_edge(k - 1, iEdge)) *
                  (w(k, cell2) - w(k, cell1));
              delsq_w(k, iCell) += w_turb_flux;
            }
          }
        });
    Kokkos::fence("hyperdiff::delsq_w_fence");

    // Step 2: Apply del4 w filter to tend_w
    Kokkos::parallel_for(
        "hyperdiff::tend_w_del4",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          const Scalar r_areaCell = h_mom_eddy_visc4 * invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);

          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            const int cell1 = cellsOnEdge(0, iEdge) - 1;
            const int cell2 = cellsOnEdge(1, iEdge) - 1;

            const Scalar edge_sign = meshScalingDel4(iEdge) * r_areaCell *
                dvEdge(iEdge) * edgesOnCell_sign(i, iCell) * invDcEdge(iEdge);

            for (int k = 1; k < nVertLevels; ++k) {
              tend_w(k, iCell) -= edge_sign * (delsq_w(k, cell2) - delsq_w(k, cell1));
            }
          }
        });
    Kokkos::fence("hyperdiff::tend_w_fence");
  }  // h_mom_eddy_visc4 > 0 for w

  // ════════════════════════════════════════════════════════════════════════════
  // PART 3: Fourth-order filter for theta_m (potential temperature on cells)
  // ════════════════════════════════════════════════════════════════════════════
  if (h_theta_eddy_visc4 > Scalar(0.0)) {
    // Step 1: Compute delsq_theta at cells
    // delsq_theta(k,iCell) = sum_edges[ sign*dvEdge*invDcEdge*(theta(k,c2)-theta(k,c1))*rho_edge(k,e) ] / areaCell
    view2d delsq_theta("delsq_theta", nVertLevels, nCells);

    Kokkos::parallel_for(
        "hyperdiff::delsq_theta",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          const Scalar r_areaCell = invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);

          for (int k = 0; k < nVertLevels; ++k) {
            delsq_theta(k, iCell) = Scalar(0.0);
          }

          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            const Scalar edge_sign = r_areaCell * edgesOnCell_sign(i, iCell) *
                dvEdge(iEdge) * invDcEdge(iEdge);

            const int cell1 = cellsOnEdge(0, iEdge) - 1;
            const int cell2 = cellsOnEdge(1, iEdge) - 1;

            for (int k = 0; k < nVertLevels; ++k) {
              const Scalar theta_turb_flux = edge_sign *
                  (theta_m(k, cell2) - theta_m(k, cell1)) * rho_edge(k, iEdge);
              delsq_theta(k, iCell) += theta_turb_flux;
            }
          }
        });
    Kokkos::fence("hyperdiff::delsq_theta_fence");

    // Step 2: Apply del4 theta filter to tend_theta
    Kokkos::parallel_for(
        "hyperdiff::tend_theta_del4",
        Kokkos::RangePolicy<ExecSpace>(0, nCells),
        KOKKOS_LAMBDA(const int iCell) {
          const Scalar r_areaCell = h_theta_eddy_visc4 * prandtl_inv * invAreaCell(iCell);
          const int ne = nEdgesOnCell_v(iCell);

          for (int i = 0; i < ne; ++i) {
            const int iEdge = edgesOnCell(i, iCell) - 1;
            const int cell1 = cellsOnEdge(0, iEdge) - 1;
            const int cell2 = cellsOnEdge(1, iEdge) - 1;

            const Scalar edge_sign = meshScalingDel4(iEdge) * r_areaCell *
                dvEdge(iEdge) * edgesOnCell_sign(i, iCell) * invDcEdge(iEdge);

            for (int k = 0; k < nVertLevels; ++k) {
              tend_theta(k, iCell) -= edge_sign *
                  (delsq_theta(k, cell2) - delsq_theta(k, cell1));
            }
          }
        });
    Kokkos::fence("hyperdiff::tend_theta_fence");
  }  // h_theta_eddy_visc4 > 0 for theta

  // ════════════════════════════════════════════════════════════════════════════
  // PART 4: Fourth-order filter for scalars (optional, on cells)
  // ════════════════════════════════════════════════════════════════════════════
  if (params.config_mix_scalars && h_theta_eddy_visc4 > Scalar(0.0)) {
    const int num_scalars = params.num_scalars;

    for (int iScalar = 0; iScalar < num_scalars; ++iScalar) {
      // Step 1: Compute delsq for this scalar (reusing the same pattern as theta)
      view2d delsq_scalar("delsq_scalar", nVertLevels, nCells);

      Kokkos::parallel_for(
          "hyperdiff::delsq_scalar",
          Kokkos::RangePolicy<ExecSpace>(0, nCells),
          KOKKOS_LAMBDA(const int iCell) {
            const Scalar r_areaCell = invAreaCell(iCell);
            const int ne = nEdgesOnCell_v(iCell);

            for (int k = 0; k < nVertLevels; ++k) {
              delsq_scalar(k, iCell) = Scalar(0.0);
            }

            for (int i = 0; i < ne; ++i) {
              const int iEdge = edgesOnCell(i, iCell) - 1;
              const Scalar edge_sign = r_areaCell * edgesOnCell_sign(i, iCell) *
                  dvEdge(iEdge) * invDcEdge(iEdge);

              const int cell1 = cellsOnEdge(0, iEdge) - 1;
              const int cell2 = cellsOnEdge(1, iEdge) - 1;

              for (int k = 0; k < nVertLevels; ++k) {
                const Scalar flux = edge_sign *
                    (scalars(iScalar, k, cell2) - scalars(iScalar, k, cell1)) *
                    rho_edge(k, iEdge);
                delsq_scalar(k, iCell) += flux;
              }
            }
          });
      Kokkos::fence("hyperdiff::delsq_scalar_fence");

      // Step 2: Apply del4 scalar filter to tend_scalars
      Kokkos::parallel_for(
          "hyperdiff::tend_scalar_del4",
          Kokkos::RangePolicy<ExecSpace>(0, nCells),
          KOKKOS_LAMBDA(const int iCell) {
            const Scalar r_areaCell = h_theta_eddy_visc4 * prandtl_inv * invAreaCell(iCell);
            const int ne = nEdgesOnCell_v(iCell);

            for (int i = 0; i < ne; ++i) {
              const int iEdge = edgesOnCell(i, iCell) - 1;
              const int cell1 = cellsOnEdge(0, iEdge) - 1;
              const int cell2 = cellsOnEdge(1, iEdge) - 1;

              const Scalar edge_sign = meshScalingDel4(iEdge) * r_areaCell *
                  dvEdge(iEdge) * edgesOnCell_sign(i, iCell) * invDcEdge(iEdge);

              for (int k = 0; k < nVertLevels; ++k) {
                tend_scalars(iScalar, k, iCell) -= edge_sign *
                    (delsq_scalar(k, cell2) - delsq_scalar(k, cell1));
              }
            }
          });
      Kokkos::fence("hyperdiff::tend_scalar_fence");
    }  // end scalar loop
  }  // config_mix_scalars && h_theta_eddy_visc4 > 0
}

/// @brief The Dissipation_Module class (Requirement 8.1-8.6, 8.7-8.11).
///
/// This class orchestrates the dissipation/mixing computations for the C++
/// dycore, selecting the appropriate eddy viscosity model based on
/// configuration and dispatching to the individual kernel functions above.
class Dissipation_Module {
 public:
  // Expose the LES option-string mapping functions (Req 8.10, 8.11)
  // as static members, consistent with the design document API:
  //   static int les_model_from_string(std::string_view);
  //   static int les_surface_from_string(std::string_view);
  // These are implemented in les_options.hpp and re-exported here for the
  // design document class API contract.
  static constexpr int les_model_opt_from_string(std::string_view s) noexcept {
    return les_model_from_string(s);
  }
  static constexpr int les_surface_opt_from_string(std::string_view s) noexcept {
    return les_surface_from_string(s);
  }

  /// @brief Compute eddy viscosity fields (Req 8.1-8.6).
  ///
  /// Dispatches to the appropriate model based on the LES option:
  /// - LES_MODEL_NONE: use 2-D Smagorinsky or fixed horizontal viscosity.
  /// - LES_MODEL_3D_SMAGORINSKY: 3-D deformation-based model.
  /// - LES_MODEL_PROGNOSTIC_15_ORDER: prognostic TKE-based model.
  ///
  /// All paths apply the stability bound (Req 8.6) to every computed
  /// eddy viscosity value.
  ///
  /// @tparam ExecSpace  Kokkos execution space.

  /// Computes the Brunt-Väisälä frequency and eddy viscosity, then
  /// returns the background 4th-order filter coefficients.
  template <class ExecSpace>
  static EddyViscosityResult compute_eddy_viscosity(
      // Output fields
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& eddy_visc_horz,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& eddy_visc_vert,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& prandtl_3d_inv,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& bn2,
      // Input fields
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& v,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& uCell,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& vCell,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& exner,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& pressure_b,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& pp,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zgrid,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& qtot,
      const Kokkos::View<Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& scalars,
      const Kokkos::View<Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_scalars,
      // Mesh connectivity/geometry
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_c2,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_s2,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_cs,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_c,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& deformation_coef_s,
      const Kokkos::View<const int*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& nEdgesOnCell_v,
      const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell_v,
      const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge_v,
      // Parameters
      const EddyViscosityParams& params,
      bool use_smagorinsky_2d) {

    // Step 1: Compute Brunt-Väisälä frequency if LES model is active
    if (params.les_model_opt != LES_MODEL_NONE) {
      using const_view3d = Kokkos::View<const Scalar***, Kokkos::LayoutLeft,
                                         typename ExecSpace::memory_space>;
      const_view3d scalars_const = scalars;
      calculate_brunt_vaisala<ExecSpace>(
          bn2, theta_m, exner, pressure_b, pp, zgrid, scalars_const, qtot, params);
    }

    // Step 2: Compute eddy viscosity based on the configured model
    if (params.les_model_opt == LES_MODEL_NONE) {
      if (use_smagorinsky_2d) {
        // 2-D Smagorinsky (Req 8.1)
        return smagorinsky_2d<ExecSpace>(
            eddy_visc_horz, u, v,
            deformation_coef_c2, deformation_coef_s2, deformation_coef_cs,
            nEdgesOnCell_v, edgesOnCell_v, params);
      } else {
        // Fixed horizontal eddy viscosity (Req 8.2)
        // For the fixed case, the h_mom/h_theta visc4 are zero unless
        // the 4th-order filter is separately configured.
        fixed_horizontal_eddy_viscosity<ExecSpace>(
            eddy_visc_horz, params.c_s, params);
        EddyViscosityResult result;
        result.h_mom_eddy_visc4 = 0.0;
        result.h_theta_eddy_visc4 = 0.0;
        return result;
      }
    } else {
      // 3-D Smagorinsky or prognostic TKE (Req 8.3, 8.4)
      // bn2 was just computed by calculate_brunt_vaisala above; pass it as
      // the bv_freq2 input to les_models. The View is non-const so we can
      // pass it where a const view is expected (implicit conversion).
      using const_view2d = Kokkos::View<const Scalar**, Kokkos::LayoutLeft,
                                         typename ExecSpace::memory_space>;
      const_view2d bn2_const = bn2;
      return les_models<ExecSpace>(
          eddy_visc_horz, eddy_visc_vert, prandtl_3d_inv,
          u, v, uCell, vCell, w, bn2_const, zgrid, rho_zz,
          scalars, tend_scalars,
          deformation_coef_c2, deformation_coef_s2, deformation_coef_cs,
          deformation_coef_c, deformation_coef_s,
          nEdgesOnCell_v, edgesOnCell_v, cellsOnEdge_v, params);
    }
  }

  /// @brief Apply fourth-order horizontal hyperdiffusion (Req 8.7).
  ///
  /// Dispatches to the `apply_hyperdiffusion` free function above.
  /// Applies the del2(del2(field)) filter to u, w, theta_m, and optionally
  /// scalars using the viscosity coefficients from compute_eddy_viscosity.
  template <class ExecSpace>
  static void hyperdiffusion(
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_u,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_w,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_theta,
      const Kokkos::View<Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_scalars,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
      const Kokkos::View<const Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& scalars,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& divergence,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& vorticity,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_edge,
      const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
      const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& verticesOnEdge,
      const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell,
      const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnVertex,
      const Kokkos::View<const int*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& nEdgesOnCell_v,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnCell_sign,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& edgesOnVertex_sign,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invAreaCell,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invAreaTriangle,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invDcEdge,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& invDvEdge,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dvEdge,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& dcEdge,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& meshScalingDel4,
      const HyperdiffusionParams& params) {
    apply_hyperdiffusion<ExecSpace>(
        tend_u, tend_w, tend_theta, tend_scalars,
        u, w, theta_m, scalars,
        divergence, vorticity, rho_edge,
        cellsOnEdge, verticesOnEdge, edgesOnCell, edgesOnVertex,
        nEdgesOnCell_v, edgesOnCell_sign, edgesOnVertex_sign,
        invAreaCell, invAreaTriangle, invDcEdge, invDvEdge, dvEdge, dcEdge,
        meshScalingDel4, params);
  }

  /// @brief Apply vertical mixing and LES surface boundary condition (Req 8.8, 8.9).
  ///
  /// Applies the second-order vertical filter in physical height space on
  /// either the full state or the perturbation from the initial 1-D state
  /// (config_mix_full), preserving the sequential vertical dependency (Req 2.2).
  /// When an LES model is active, also applies the LES turbulent vertical
  /// mixing using eddy_visc_vert and the configured surface flux lower
  /// boundary condition (Req 8.9).
  ///
  /// Dispatches to the free functions in vertical_mixing.hpp.
  template <class ExecSpace>
  static void vertical_mixing(
      // Tendency output fields (accumulated into)
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_u,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_w,
      const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_theta,
      const Kokkos::View<Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_scalars,
      // Input state fields
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& v,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
      const Kokkos::View<const Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& scalars,
      // Density / metric fields
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_edge,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& eddy_visc_vert,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zz,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zgrid,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& divergence,
      const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& prandtl_3d_inv,
      // Mesh connectivity / geometry
      const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzu,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzw,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzm,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzp,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u_init,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& v_init,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& angleEdge,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& t_init,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ustm,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& hfx,
      const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& qfx,
      const VerticalMixingParams& params) {

    // Step 1: Simple second-order vertical filter (Req 8.8)
    apply_vertical_mixing_u<ExecSpace>(
        tend_u, u, v, rho_edge, zgrid, cellsOnEdge, u_init, v_init, angleEdge, params);

    apply_vertical_mixing_w<ExecSpace>(tend_w, w, rho_zz, rdzw, rdzu, params);

    apply_vertical_mixing_theta<ExecSpace>(
        tend_theta, theta_m, rho_zz, zgrid, t_init, params);

    // Step 2: LES turbulent vertical mixing (Req 8.8, 8.9)
    if (params.les_model_opt != LES_MODEL_NONE) {
      apply_les_vertical_mixing_u<ExecSpace>(
          tend_u, u, v, rho_edge, rho_zz, eddy_visc_vert, zz,
          rdzu, rdzw, fzm, fzp, cellsOnEdge, ustm, params);

      apply_les_vertical_mixing_w<ExecSpace>(
          tend_w, w, rho_zz, eddy_visc_vert, zz, divergence, rdzw, rdzu, params);

      apply_les_vertical_mixing_theta_scalars<ExecSpace>(
          tend_theta, tend_scalars, theta_m, scalars,
          rho_zz, eddy_visc_vert, zz, prandtl_3d_inv,
          rdzw, rdzu, fzm, fzp, hfx, qfx, params);
    }
  }
};

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_DISSIPATION_MODULE_HPP
