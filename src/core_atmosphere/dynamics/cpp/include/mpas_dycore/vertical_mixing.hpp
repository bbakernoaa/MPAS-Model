#ifndef MPAS_DYCORE_VERTICAL_MIXING_HPP
#define MPAS_DYCORE_VERTICAL_MIXING_HPP

/// @file vertical_mixing.hpp
/// @brief Second-order vertical filter and LES surface boundary conditions.
///
/// Implements the vertical mixing for u, w, theta_m, and scalars:
/// - Second-order vertical filter in physical height space, mixing either
///   the full state or the perturbation from the initial 1-D state (Req 8.8).
/// - LES surface flux lower boundary condition (Req 8.9).
/// - Sequential vertical dependency preserved (Req 2.2).
///
/// Requirements: 8.8, 8.9, 2.2

#include "mpas_dycore/les_options.hpp"
#include "mpas_dycore/scalar.hpp"

#include <Kokkos_Core.hpp>

namespace mpas {
namespace dycore {

/// Physical constants matching the Reference_Model (duplicated from dissipation_module.hpp
/// for use in stand-alone vertical mixing kernels).
namespace vertical_mixing_constants {
inline constexpr Scalar gravity = 9.80616;
inline constexpr Scalar rgas = 287.0;
inline constexpr Scalar rv = 461.6;
inline constexpr Scalar cp = 7.0 * rgas / 2.0;
inline constexpr Scalar rvord = rv / rgas;
}  // namespace vertical_mixing_constants

/// Parameters controlling the vertical mixing filter and LES surface BC.
struct VerticalMixingParams {
  Scalar v_mom_eddy_visc2 = 0.0;       ///< 2nd-order vertical viscosity for momentum
  Scalar v_theta_eddy_visc2 = 0.0;     ///< 2nd-order vertical viscosity for theta/scalars
  Scalar prandtl_inv = 1.0;            ///< inverse Prandtl number for theta/scalar
  bool config_mix_full = true;          ///< mix full state (true) or perturbation (false)
  int les_model_opt = LES_MODEL_NONE;   ///< LES turbulence model option
  int les_surface_opt = LES_SURFACE_NONE; ///< LES surface flux option
  Scalar config_surface_drag_coefficient = 0.0; ///< surface drag coeff (specified)
  Scalar config_surface_heat_flux = 0.0;        ///< surface heat flux [K m/s]
  Scalar config_surface_moisture_flux = 0.0;    ///< surface moisture flux [kg/kg m/s]
  bool config_mix_scalars = false;      ///< whether to apply vertical mixing to scalars
  int index_qv = -1;                    ///< scalar index for water vapor (1-based)
  int index_tke = -1;                   ///< scalar index for TKE (1-based)
  int nVertLevels = 0;
  int nCells = 0;
  int nEdges = 0;
  int num_scalars = 0;
};

/// @brief Apply second-order vertical filter for u (horizontal momentum on edges).
///
/// Implements the vertical mixing for u in physical height space (Req 8.8).
/// Parallelizes over edges, sequential over vertical levels (Req 2.2).
/// Mixes either the full state or perturbation from the initial 1-D state.
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void apply_vertical_mixing_u(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_u,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& v,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_edge,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zgrid,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u_init,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& v_init,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& angleEdge,
    const VerticalMixingParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nEdges = params.nEdges;
  const Scalar v_mom_eddy_visc2 = params.v_mom_eddy_visc2;
  const bool config_mix_full = params.config_mix_full;

  if (v_mom_eddy_visc2 <= Scalar(0.0)) return;

  Kokkos::parallel_for(
      "vertical_mixing_u",
      Kokkos::RangePolicy<ExecSpace>(0, nEdges),
      KOKKOS_LAMBDA(const int iEdge) {
        const int cell1 = cellsOnEdge(0, iEdge) - 1;
        const int cell2 = cellsOnEdge(1, iEdge) - 1;

        if (config_mix_full) {
          // Mix full state
          for (int k = 1; k < nVertLevels - 1; ++k) {
            const Scalar z1 = Scalar(0.5) * (zgrid(k - 1, cell1) + zgrid(k - 1, cell2));
            const Scalar z2 = Scalar(0.5) * (zgrid(k, cell1) + zgrid(k, cell2));
            const Scalar z3 = Scalar(0.5) * (zgrid(k + 1, cell1) + zgrid(k + 1, cell2));
            const Scalar z4 = Scalar(0.5) * (zgrid(k + 2, cell1) + zgrid(k + 2, cell2));

            const Scalar zm = Scalar(0.5) * (z1 + z2);
            const Scalar z0 = Scalar(0.5) * (z2 + z3);
            const Scalar zp = Scalar(0.5) * (z3 + z4);

            tend_u(k, iEdge) += rho_edge(k, iEdge) * v_mom_eddy_visc2 * (
                (u(k + 1, iEdge) - u(k, iEdge)) / (zp - z0)
              - (u(k, iEdge) - u(k - 1, iEdge)) / (z0 - zm)
            ) / (Scalar(0.5) * (zp - zm));
          }
        } else {
          // Mix perturbation from initial 1-D state
          // Compute u_mix = u - u_init*cos(angle) + v_init*sin(angle)
          // Use sequential loop for the perturbation computation
          for (int k = 1; k < nVertLevels - 1; ++k) {
            const Scalar angle = angleEdge(iEdge);
            const Scalar cos_a = Kokkos::cos(angle);
            const Scalar sin_a = Kokkos::sin(angle);

            const Scalar u_mix_km1 = u(k - 1, iEdge)
                - u_init(k - 1) * cos_a + v_init(k - 1) * sin_a;
            const Scalar u_mix_k = u(k, iEdge)
                - u_init(k) * cos_a + v_init(k) * sin_a;
            const Scalar u_mix_kp1 = u(k + 1, iEdge)
                - u_init(k + 1) * cos_a + v_init(k + 1) * sin_a;

            const Scalar z1 = Scalar(0.5) * (zgrid(k - 1, cell1) + zgrid(k - 1, cell2));
            const Scalar z2 = Scalar(0.5) * (zgrid(k, cell1) + zgrid(k, cell2));
            const Scalar z3 = Scalar(0.5) * (zgrid(k + 1, cell1) + zgrid(k + 1, cell2));
            const Scalar z4 = Scalar(0.5) * (zgrid(k + 2, cell1) + zgrid(k + 2, cell2));

            const Scalar zm = Scalar(0.5) * (z1 + z2);
            const Scalar z0 = Scalar(0.5) * (z2 + z3);
            const Scalar zp = Scalar(0.5) * (z3 + z4);

            tend_u(k, iEdge) += rho_edge(k, iEdge) * v_mom_eddy_visc2 * (
                (u_mix_kp1 - u_mix_k) / (zp - z0)
              - (u_mix_k - u_mix_km1) / (z0 - zm)
            ) / (Scalar(0.5) * (zp - zm));
          }
        }
      });
}

/// @brief Apply second-order vertical filter for w (vertical velocity on cells).
///
/// Implements the simple vertical mixing for w using v_mom_eddy_visc2 (Req 8.8).
/// Parallelizes over cells, sequential over vertical levels (Req 2.2).
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void apply_vertical_mixing_w(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzw,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzu,
    const VerticalMixingParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const Scalar v_mom_eddy_visc2 = params.v_mom_eddy_visc2;

  if (v_mom_eddy_visc2 <= Scalar(0.0)) return;

  Kokkos::parallel_for(
      "vertical_mixing_w",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        for (int k = 1; k < nVertLevels; ++k) {
          tend_w(k, iCell) += v_mom_eddy_visc2 *
              Scalar(0.5) * (rho_zz(k, iCell) + rho_zz(k - 1, iCell)) * (
                  (w(k + 1, iCell) - w(k, iCell)) * rdzw(k)
                - (w(k, iCell) - w(k - 1, iCell)) * rdzw(k - 1)
              ) * rdzu(k);
        }
      });
}

/// @brief Apply second-order vertical filter for theta_m (potential temperature on cells).
///
/// Implements the vertical mixing for theta in physical height space (Req 8.8).
/// Parallelizes over cells, sequential over vertical levels (Req 2.2).
/// Mixes either the full state or perturbation from the initial 1-D state.
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void apply_vertical_mixing_theta(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_theta,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zgrid,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& t_init,
    const VerticalMixingParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const Scalar v_theta_eddy_visc2 = params.v_theta_eddy_visc2;
  const Scalar prandtl_inv = params.prandtl_inv;
  const bool config_mix_full = params.config_mix_full;

  if (v_theta_eddy_visc2 <= Scalar(0.0)) return;

  Kokkos::parallel_for(
      "vertical_mixing_theta",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        for (int k = 1; k < nVertLevels - 1; ++k) {
          const Scalar z1 = zgrid(k - 1, iCell);
          const Scalar z2 = zgrid(k, iCell);
          const Scalar z3 = zgrid(k + 1, iCell);
          const Scalar z4 = zgrid(k + 2, iCell);

          const Scalar zm = Scalar(0.5) * (z1 + z2);
          const Scalar z0 = Scalar(0.5) * (z2 + z3);
          const Scalar zp = Scalar(0.5) * (z3 + z4);

          if (config_mix_full) {
            tend_theta(k, iCell) += v_theta_eddy_visc2 * prandtl_inv
                * rho_zz(k, iCell) * (
                    (theta_m(k + 1, iCell) - theta_m(k, iCell)) / (zp - z0)
                  - (theta_m(k, iCell) - theta_m(k - 1, iCell)) / (z0 - zm)
                ) / (Scalar(0.5) * (zp - zm));
          } else {
            // Mix perturbation from initial 1-D state
            const Scalar pert_kp1 = theta_m(k + 1, iCell) - t_init(k + 1);
            const Scalar pert_k = theta_m(k, iCell) - t_init(k);
            const Scalar pert_km1 = theta_m(k - 1, iCell) - t_init(k - 1);

            tend_theta(k, iCell) += v_theta_eddy_visc2 * prandtl_inv
                * rho_zz(k, iCell) * (
                    (pert_kp1 - pert_k) / (zp - z0)
                  - (pert_k - pert_km1) / (z0 - zm)
                ) / (Scalar(0.5) * (zp - zm));
          }
        }
      });
}

/// @brief Apply LES vertical turbulent mixing for u (Req 8.8, 8.9).
///
/// Computes turbulent vertical fluxes using eddy_visc_vert and applies
/// the configured LES surface flux lower boundary condition (Req 8.9).
/// Parallelizes over edges, sequential over vertical levels (Req 2.2).
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void apply_les_vertical_mixing_u(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_u,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& v,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_edge,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& eddy_visc_vert,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zz,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzu,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzw,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzm,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzp,
    const Kokkos::View<const int**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& cellsOnEdge,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& ustm,
    const VerticalMixingParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nEdges = params.nEdges;
  const int les_surface_opt = params.les_surface_opt;
  const Scalar config_surface_drag_coefficient = params.config_surface_drag_coefficient;

  Kokkos::parallel_for(
      "les_vertical_mixing_u",
      Kokkos::RangePolicy<ExecSpace>(0, nEdges),
      KOKKOS_LAMBDA(const int iEdge) {
        const int cell1 = cellsOnEdge(0, iEdge) - 1;
        const int cell2 = cellsOnEdge(1, iEdge) - 1;

        // Compute turbulent vertical fluxes at interfaces
        // turb_vflux(k) at interface k between levels k-1 and k
        // Index 0 = bottom, nVertLevels = top
        // We compute sequentially to preserve vertical dependency

        // Top BC: no flux out of domain
        Scalar turb_vflux_top = Scalar(0.0);
        Scalar turb_vflux_prev = Scalar(0.0);  // will hold turb_vflux(k) for divergence

        // Bottom BC (index 1 in Fortran = index 0 here for interface level)
        // In Fortran: turb_vflux(1) = 0 or surface BC
        Scalar turb_vflux_bottom = Scalar(0.0);

        // Compute interior fluxes from level 2..nVertLevels (0-based: k=1..nVertLevels-1)
        // and apply surface BC at the bottom

        // First compute turb_vflux(1) = bottom BC
        if (les_surface_opt == LES_SURFACE_SPECIFIED) {
          const Scalar vel_mag = Kokkos::sqrt(u(0, iEdge) * u(0, iEdge)
                                            + v(0, iEdge) * v(0, iEdge));
          turb_vflux_bottom = -rho_edge(0, iEdge)
              * config_surface_drag_coefficient * u(0, iEdge) * vel_mag;
        } else if (les_surface_opt == LES_SURFACE_VARYING) {
          const Scalar ust_edge = Scalar(0.5) * (ustm(cell1) + ustm(cell2));
          const Scalar vel_mag = Kokkos::fmax(
              Kokkos::sqrt(u(0, iEdge) * u(0, iEdge) + v(0, iEdge) * v(0, iEdge)),
              Scalar(0.1));
          turb_vflux_bottom = -rho_edge(0, iEdge) * ust_edge * ust_edge
              * (u(0, iEdge) / vel_mag);
        }
        // else: turb_vflux_bottom remains 0, will be set to turb_vflux(2) below

        // Compute fluxes at each interface and apply tendency
        // We walk bottom-to-top: for level k (0-based), the tendency is:
        //   tend_u(k) -= rdzw(k) * (turb_vflux(k+1) - turb_vflux(k))
        // where turb_vflux(k) is at interface k (between level k-1 and k)

        // Compute turb_vflux at interface k=1 (between level 0 and 1)
        // In Fortran (1-based): turb_vflux(2) = ...
        Scalar flux_k1 = Scalar(0.0);
        if (nVertLevels > 1) {
          const Scalar rho_k_c1 = fzm(1) * rho_zz(1, cell1) * zz(1, cell1) * eddy_visc_vert(1, cell1)
                                + fzp(1) * rho_zz(0, cell1) * zz(0, cell1) * eddy_visc_vert(0, cell1);
          const Scalar rho_k_c2 = fzm(1) * rho_zz(1, cell2) * zz(1, cell2) * eddy_visc_vert(1, cell2)
                                + fzp(1) * rho_zz(0, cell2) * zz(0, cell2) * eddy_visc_vert(0, cell2);
          const Scalar rho_k_at_w = Scalar(0.5) * (rho_k_c1 + rho_k_c2);

          const Scalar zz_c1 = fzm(1) * zz(1, cell1) + fzp(1) * zz(0, cell1);
          const Scalar zz_c2 = fzm(1) * zz(1, cell2) + fzp(1) * zz(0, cell2);
          const Scalar zz_at_w = Scalar(0.5) * (zz_c1 + zz_c2);

          flux_k1 = -rho_k_at_w * zz_at_w * rdzu(1) * (u(1, iEdge) - u(0, iEdge));
        }

        // For surface options without explicit BC, set bottom = first interior flux
        if (les_surface_opt != LES_SURFACE_SPECIFIED &&
            les_surface_opt != LES_SURFACE_VARYING) {
          turb_vflux_bottom = flux_k1;
        }

        // Apply tendency for level 0: tend_u(0) -= rdzw(0) * (flux_k1 - turb_vflux_bottom)
        tend_u(0, iEdge) -= rdzw(0) * (flux_k1 - turb_vflux_bottom);

        // Walk through levels 1..nVertLevels-1
        Scalar flux_prev = flux_k1;
        for (int k = 1; k < nVertLevels; ++k) {
          Scalar flux_curr;
          if (k < nVertLevels - 1) {
            // Compute turb_vflux at interface k+1
            const int kp1 = k + 1;
            const Scalar rho_k_c1 = fzm(kp1) * rho_zz(kp1, cell1) * zz(kp1, cell1) * eddy_visc_vert(kp1, cell1)
                                  + fzp(kp1) * rho_zz(k, cell1) * zz(k, cell1) * eddy_visc_vert(k, cell1);
            const Scalar rho_k_c2 = fzm(kp1) * rho_zz(kp1, cell2) * zz(kp1, cell2) * eddy_visc_vert(kp1, cell2)
                                  + fzp(kp1) * rho_zz(k, cell2) * zz(k, cell2) * eddy_visc_vert(k, cell2);
            const Scalar rho_k_at_w = Scalar(0.5) * (rho_k_c1 + rho_k_c2);

            const Scalar zz_c1 = fzm(kp1) * zz(kp1, cell1) + fzp(kp1) * zz(k, cell1);
            const Scalar zz_c2 = fzm(kp1) * zz(kp1, cell2) + fzp(kp1) * zz(k, cell2);
            const Scalar zz_at_w = Scalar(0.5) * (zz_c1 + zz_c2);

            flux_curr = -rho_k_at_w * zz_at_w * rdzu(kp1) * (u(kp1, iEdge) - u(k, iEdge));
          } else {
            // Top BC: copy from level below
            if (les_surface_opt == LES_SURFACE_SPECIFIED ||
                les_surface_opt == LES_SURFACE_VARYING) {
              flux_curr = flux_prev;
            } else {
              flux_curr = flux_prev;
            }
          }

          tend_u(k, iEdge) -= rdzw(k) * (flux_curr - flux_prev);
          flux_prev = flux_curr;
        }
      });
}

/// @brief Apply LES vertical turbulent mixing for w (Req 8.8).
///
/// Computes turbulent vertical fluxes for w using eddy_visc_vert and divergence.
/// Parallelizes over cells, sequential over vertical levels (Req 2.2).
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void apply_les_vertical_mixing_w(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& w,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& eddy_visc_vert,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& divergence,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzw,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzu,
    const VerticalMixingParams& params) {

  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;

  Kokkos::parallel_for(
      "les_vertical_mixing_w",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // Compute turbulent fluxes at full levels (index 0..nVertLevels-1)
        // turb_vflux(k) = -rho_zz(k)*eddy_visc_vert(k)*zz(k)*
        //                  (2*zz(k)*rdzw(k)*(w(k+1)-w(k)) + divergence(k))
        // Top: turb_vflux(nVertLevels) = 0

        // Apply: tend_w(k) -= rdzu(k) * (turb_vflux(k) - turb_vflux(k-1))
        // for k = 1..nVertLevels-1 (interface levels for w, 0-based)

        // Compute flux at level 0 (bottom full level)
        Scalar flux_prev = -rho_zz(0, iCell) * eddy_visc_vert(0, iCell) * zz(0, iCell) * (
            Scalar(2.0) * zz(0, iCell) * rdzw(0) * (w(1, iCell) - w(0, iCell))
            + divergence(0, iCell));

        for (int k = 1; k < nVertLevels; ++k) {
          // Compute flux at level k
          Scalar flux_curr;
          if (k < nVertLevels - 1) {
            flux_curr = -rho_zz(k, iCell) * eddy_visc_vert(k, iCell) * zz(k, iCell) * (
                Scalar(2.0) * zz(k, iCell) * rdzw(k) * (w(k + 1, iCell) - w(k, iCell))
                + divergence(k, iCell));
          } else {
            // Top level: flux = 0
            flux_curr = Scalar(0.0);
          }

          tend_w(k, iCell) -= rdzu(k) * (flux_curr - flux_prev);
          flux_prev = flux_curr;
        }
      });
}

/// @brief Apply LES vertical turbulent mixing for theta_m and scalars (Req 8.8, 8.9).
///
/// Computes turbulent vertical fluxes using eddy_visc_vert with the
/// Prandtl number scaling, and applies the configured LES surface flux
/// lower boundary condition for heat and moisture (Req 8.9).
/// Parallelizes over cells, sequential over vertical levels (Req 2.2).
///
/// @tparam ExecSpace  Kokkos execution space.
template <class ExecSpace>
void apply_les_vertical_mixing_theta_scalars(
    const Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_theta,
    const Kokkos::View<Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& tend_scalars,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& theta_m,
    const Kokkos::View<const Scalar***, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& scalars,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rho_zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& eddy_visc_vert,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& zz,
    const Kokkos::View<const Scalar**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& prandtl_3d_inv,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzw,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& rdzu,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzm,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& fzp,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& hfx,
    const Kokkos::View<const Scalar*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& qfx,
    const VerticalMixingParams& params) {

  namespace vmc = vertical_mixing_constants;
  const int nVertLevels = params.nVertLevels;
  const int nCells = params.nCells;
  const int les_model_opt = params.les_model_opt;
  const int les_surface_opt = params.les_surface_opt;
  const Scalar config_surface_heat_flux = params.config_surface_heat_flux;
  const Scalar config_surface_moisture_flux = params.config_surface_moisture_flux;
  const bool config_mix_scalars = params.config_mix_scalars;
  const int num_scalars = params.num_scalars;
  const int idx_qv = params.index_qv - 1;  // 0-based
  const Scalar prandtl_inv_const = params.prandtl_inv;

  Kokkos::parallel_for(
      "les_vertical_mixing_theta_scalars",
      Kokkos::RangePolicy<ExecSpace>(0, nCells),
      KOKKOS_LAMBDA(const int iCell) {
        // Compute Prandtl inverse at interfaces
        // For 3-D Smagorinsky: constant prandtl_inv
        // For prognostic TKE: interpolated from prandtl_3d_inv field

        // Compute rho_k_at_w and zz_at_w at interfaces for theta flux
        // turb_vflux(k) = -prandtl_1d_inv(k) * rho_k_at_w(k) * zz_at_w(k)
        //                  * rdzu(k) * (theta_m(k) - theta_m(k-1))

        // Bottom BC
        Scalar turb_vflux_bottom = Scalar(0.0);
        Scalar moisture_flux_val = Scalar(0.0);

        // Compute surface BC for theta and moisture
        if (les_surface_opt == LES_SURFACE_SPECIFIED ||
            les_surface_opt == LES_SURFACE_VARYING) {
          Scalar heat_flux, moist_flux;
          if (les_surface_opt == LES_SURFACE_SPECIFIED) {
            moist_flux = config_surface_moisture_flux;
            heat_flux = config_surface_heat_flux;
          } else {  // LES_SURFACE_VARYING
            heat_flux = hfx(iCell) / rho_zz(0, iCell) / vmc::cp;
            moist_flux = qfx(iCell) / rho_zz(0, iCell);
          }

          const Scalar qv_cell = scalars(idx_qv, 0, iCell);
          const Scalar theta_m_cell = theta_m(0, iCell);
          const Scalar theta_cell = theta_m_cell /
              (Scalar(1.0) + vmc::rvord * qv_cell);

          const Scalar theta_m_flux = heat_flux * (Scalar(1.0) + vmc::rvord * qv_cell)
              + vmc::rvord * theta_cell * moist_flux;
          turb_vflux_bottom = theta_m_flux * rho_zz(0, iCell);
          moisture_flux_val = moist_flux * rho_zz(0, iCell);
        }

        // Compute flux at interface k=1 (between level 0 and 1)
        Scalar pr_inv_k1;
        if (les_model_opt == LES_MODEL_3D_SMAGORINSKY) {
          pr_inv_k1 = prandtl_inv_const;
        } else {
          pr_inv_k1 = fzm(1) * prandtl_3d_inv(1, iCell)
                    + fzp(1) * prandtl_3d_inv(0, iCell);
        }

        Scalar rho_k_at_w_k1 = fzm(1) * rho_zz(1, iCell) * zz(1, iCell) * zz(1, iCell)
                                        * eddy_visc_vert(1, iCell)
                              + fzp(1) * rho_zz(0, iCell) * zz(0, iCell) * zz(0, iCell)
                                        * eddy_visc_vert(0, iCell);
        Scalar zz_at_w_k1 = fzm(1) * zz(1, iCell) + fzp(1) * zz(0, iCell);

        Scalar flux_k1 = -pr_inv_k1 * rho_k_at_w_k1 * zz_at_w_k1 * rdzu(1)
            * (theta_m(1, iCell) - theta_m(0, iCell));

        // For non-surface-BC cases, bottom = first interior
        if (les_surface_opt != LES_SURFACE_SPECIFIED &&
            les_surface_opt != LES_SURFACE_VARYING) {
          turb_vflux_bottom = flux_k1;
        }

        // Apply tendency for level 0
        tend_theta(0, iCell) -= rdzw(0) * (flux_k1 - turb_vflux_bottom);

        // Walk levels 1..nVertLevels-1
        Scalar flux_prev = flux_k1;
        for (int k = 1; k < nVertLevels; ++k) {
          Scalar flux_curr;
          if (k < nVertLevels - 1) {
            const int kp1 = k + 1;
            Scalar pr_inv_kp1;
            if (les_model_opt == LES_MODEL_3D_SMAGORINSKY) {
              pr_inv_kp1 = prandtl_inv_const;
            } else {
              pr_inv_kp1 = fzm(kp1) * prandtl_3d_inv(kp1, iCell)
                         + fzp(kp1) * prandtl_3d_inv(k, iCell);
            }

            const Scalar rho_k = fzm(kp1) * rho_zz(kp1, iCell) * zz(kp1, iCell)
                                            * zz(kp1, iCell) * eddy_visc_vert(kp1, iCell)
                               + fzp(kp1) * rho_zz(k, iCell) * zz(k, iCell)
                                            * zz(k, iCell) * eddy_visc_vert(k, iCell);
            const Scalar zz_w = fzm(kp1) * zz(kp1, iCell) + fzp(kp1) * zz(k, iCell);

            flux_curr = -pr_inv_kp1 * rho_k * zz_w * rdzu(kp1)
                * (theta_m(kp1, iCell) - theta_m(k, iCell));
          } else {
            // Top: copy from below for surface options, else 0
            if (les_surface_opt == LES_SURFACE_SPECIFIED ||
                les_surface_opt == LES_SURFACE_VARYING) {
              flux_curr = flux_prev;
            } else {
              flux_curr = flux_prev;
            }
          }

          tend_theta(k, iCell) -= rdzw(k) * (flux_curr - flux_prev);
          flux_prev = flux_curr;
        }

        // ── Scalar vertical mixing (if enabled) ──────────────────────────
        if (config_mix_scalars) {
          for (int iScalar = 0; iScalar < num_scalars; ++iScalar) {
            // Bottom BC for scalars: moisture flux for qv, zero otherwise
            Scalar scalar_flux_bottom = Scalar(0.0);
            if ((les_surface_opt == LES_SURFACE_SPECIFIED ||
                 les_surface_opt == LES_SURFACE_VARYING) &&
                iScalar == idx_qv) {
              scalar_flux_bottom = moisture_flux_val;
            }

            // Compute flux at interface k=1
            Scalar s_flux_k1 = -pr_inv_k1 * rho_k_at_w_k1 * zz_at_w_k1 * rdzu(1)
                * (scalars(iScalar, 1, iCell) - scalars(iScalar, 0, iCell));

            if (les_surface_opt != LES_SURFACE_SPECIFIED &&
                les_surface_opt != LES_SURFACE_VARYING) {
              scalar_flux_bottom = s_flux_k1;
            }

            // Apply tendency for level 0
            tend_scalars(iScalar, 0, iCell) -= rdzw(0) * (s_flux_k1 - scalar_flux_bottom);

            // Walk levels 1..nVertLevels-1
            Scalar s_flux_prev = s_flux_k1;
            for (int k = 1; k < nVertLevels; ++k) {
              Scalar s_flux_curr;
              if (k < nVertLevels - 1) {
                const int kp1 = k + 1;
                Scalar pr_inv_kp1;
                if (les_model_opt == LES_MODEL_3D_SMAGORINSKY) {
                  pr_inv_kp1 = prandtl_inv_const;
                } else {
                  pr_inv_kp1 = fzm(kp1) * prandtl_3d_inv(kp1, iCell)
                             + fzp(kp1) * prandtl_3d_inv(k, iCell);
                }

                const Scalar rho_k = fzm(kp1) * rho_zz(kp1, iCell) * zz(kp1, iCell)
                                                * zz(kp1, iCell) * eddy_visc_vert(kp1, iCell)
                                   + fzp(kp1) * rho_zz(k, iCell) * zz(k, iCell)
                                                * zz(k, iCell) * eddy_visc_vert(k, iCell);
                const Scalar zz_w = fzm(kp1) * zz(kp1, iCell) + fzp(kp1) * zz(k, iCell);

                s_flux_curr = -pr_inv_kp1 * rho_k * zz_w * rdzu(kp1)
                    * (scalars(iScalar, kp1, iCell) - scalars(iScalar, k, iCell));
              } else {
                s_flux_curr = s_flux_prev;
              }

              tend_scalars(iScalar, k, iCell) -= rdzw(k) * (s_flux_curr - s_flux_prev);
              s_flux_prev = s_flux_curr;
            }
          }
        }
      });
}

}  // namespace dycore
}  // namespace mpas

#endif  // MPAS_DYCORE_VERTICAL_MIXING_HPP
