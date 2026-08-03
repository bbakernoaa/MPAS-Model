#pragma once

/// @file les_models.hpp
/// @brief LES turbulence model kernels for eddy viscosity computation.
///
/// Provides two methods of computing horizontal eddy viscosity on cells:
///   1. Fixed viscosity: fills eddy_visc_horz with a constant value.
///   2. 2D Smagorinsky: computes deformation-based eddy viscosity from
///      horizontal velocity gradients using the Smagorinsky formulation.
///
/// @section smagorinsky_equation 2D Smagorinsky Equations
///
/// The horizontal velocity gradients are reconstructed at each cell from
/// edge-normal (u) and edge-tangential (v) velocities using precomputed
/// deformation coefficients:
///
///   dudx = sum_edges deformation_coef_c2 * u - deformation_coef_cs * v
///   dudy = sum_edges deformation_coef_cs * u - deformation_coef_s2 * v
///   dvdx = sum_edges deformation_coef_cs * u + deformation_coef_c2 * v
///   dvdy = sum_edges deformation_coef_s2 * u + deformation_coef_cs * v
///
/// The deformation tensor components are then:
///   D11 = 2 * dudx
///   D22 = 2 * dvdy
///   D12 = dudy + dvdx
///
/// The strain rate magnitude:
///   |S| = sqrt(0.25 * (D11 - D22)^2 + D12^2)
///
/// The eddy viscosity:
///   eddy_visc_horz = (c_s * config_len_disp)^2 * |S|
///
/// A stability bound is imposed:
///   eddy_visc_horz = min(eddy_visc_horz, 0.01 * config_len_disp^2 * invDt)
///
/// @reference Smagorinsky, J. (1963), "General circulation experiments with the
/// primitive equations. I. The basic experiment", Mon. Wea. Rev., 91, 99-164.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Fill eddy viscosity with a constant value (2d_fixed option).
///
/// Sets eddy_visc_horz(k, iCell) = config_h_theta_eddy_visc2 for all k, iCell.
/// This implements the simplest diffusion option where the eddy viscosity
/// is spatially uniform.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy.
///
/// @param[in]  policy                     Execution policy.
/// @param[out] eddy_visc_horz             Horizontal eddy viscosity (nVertLevels, nCells).
/// @param[in]  config_h_theta_eddy_visc2  Constant viscosity value.
/// @param[in]  nCells                     Number of cells.
/// @param[in]  nVertLevels                Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_2d_fixed_viscosity(
    Policy policy,
    Field2D<Layout, unchecked_accessor> eddy_visc_horz,
    real_type config_h_theta_eddy_visc2,
    index_type nCells,
    index_type nVertLevels);

/// @brief Compute 2D Smagorinsky deformation-based eddy viscosity.
///
/// Computes spatially-varying horizontal eddy viscosity at each cell based on
/// the local deformation (strain rate) of the velocity field. The computation
/// uses precomputed deformation coefficients that relate edge-based velocity
/// components to cell-centered velocity gradients.
///
/// Algorithm (per cell):
///   1. Reconstruct velocity gradients (dudx, dudy, dvdx, dvdy) from edge velocities
///      using deformation coefficients.
///   2. Compute deformation tensor: D11 = 2*dudx, D22 = 2*dvdy, D12 = dudy + dvdx.
///   3. Compute strain rate magnitude: |S| = sqrt(0.25*(D11-D22)^2 + D12^2).
///   4. eddy_visc_horz = (c_s * config_len_disp)^2 * |S|.
///   5. Apply upper bound: min(eddy_visc_horz, 0.01 * config_len_disp^2 * invDt).
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy.
///
/// @param[in]  policy              Execution policy.
/// @param[out] eddy_visc_horz      Horizontal eddy viscosity output (nVertLevels, nCells).
/// @param[in]  u                   Edge-normal velocity (nVertLevels, nEdges).
/// @param[in]  v                   Edge-tangential velocity (nVertLevels, nEdges).
/// @param[in]  c_s                 Smagorinsky coefficient.
/// @param[in]  config_len_disp     Filter length scale [m].
/// @param[in]  invDt               Inverse of the large timestep [1/s] (for stability bound).
/// @param[in]  mesh                Mesh connectivity (edgesOnCell, nEdgesOnCell).
/// @param[in]  deformation_coef_c2 Deformation coefficient c2 (maxEdges, nCells), column-major.
/// @param[in]  deformation_coef_s2 Deformation coefficient s2 (maxEdges, nCells), column-major.
/// @param[in]  deformation_coef_cs Deformation coefficient cs (maxEdges, nCells), column-major.
/// @param[in]  nCells              Number of cells.
/// @param[in]  nEdges              Number of edges (for bounds context only).
/// @param[in]  nVertLevels         Number of vertical levels.
template <typename Layout = default_layout,
          ExecutionPolicy Policy = SerialPolicy>
void compute_2d_smagorinsky(
    Policy policy,
    Field2D<Layout, unchecked_accessor> eddy_visc_horz,
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> v,
    real_type c_s,
    real_type config_len_disp,
    real_type invDt,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_c2,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_s2,
    std::mdspan<const real_type,
        std::extents<index_type, std::dynamic_extent, std::dynamic_extent>,
        default_layout> deformation_coef_cs,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
