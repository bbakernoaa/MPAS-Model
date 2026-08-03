#pragma once

/// @file velocity_reconstruct.hpp
/// @brief Velocity reconstruction kernel for the MPAS dynamical core.
///
/// Reconstructs cell-center velocity vectors from edge-normal velocity
/// components using pre-computed least-squares reconstruction coefficients.
/// This is a port of the Fortran `mpas_vector_reconstruction` module.
///
/// @section formulation Reconstruction Formulation
///
/// For each cell, the zonal (u) and meridional (v) velocity components at the
/// cell center are reconstructed from the edge-normal velocities:
///
///   u_cell(k, iCell) = sum_{j=0}^{nEdgesOnCell-1} coeffs_reconstruct_u(j, iCell) * u(k, edge_j)
///   v_cell(k, iCell) = sum_{j=0}^{nEdgesOnCell-1} coeffs_reconstruct_v(j, iCell) * u(k, edge_j)
///
/// where:
/// - u(k, edge_j) is the normal velocity on the j-th edge of cell iCell
/// - coeffs_reconstruct_u and coeffs_reconstruct_v are pre-computed
///   least-squares fit coefficients that map edge-normal velocities to
///   zonal and meridional components, respectively
///
/// The reconstruction coefficients encode the geometry of the cell (edge
/// orientations, distances, etc.) and are computed once during mesh setup.
///
/// @reference Thuburn, J., Ringler, T. D., Skamarock, W. C., and Klemp, J. B.
/// (2009), "Numerical representation of geostrophic modes on arbitrarily
/// structured C-grids", J. Comput. Phys., 228, 8321-8335.

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/execution_policy.hpp>

namespace mpas::dycore::kernels {

/// @brief Reconstruct cell-center velocity vectors from edge-normal velocities.
///
/// Uses pre-computed reconstruction coefficients (from least-squares fit) to
/// reconstruct zonal (u_cell) and meridional (v_cell) velocity components at
/// cell centers from the edge-normal velocity field.
///
/// @tparam Layout  mdspan layout policy.
/// @tparam Policy  Execution policy for parallelization.
///
/// @param[in]  policy                Execution policy instance.
/// @param[out] u_cell                Zonal velocity at cell centers (nVertLevels, nCells).
/// @param[out] v_cell                Meridional velocity at cell centers (nVertLevels, nCells).
/// @param[in]  u                     Normal velocity on edges (nVertLevels, nEdges).
/// @param[in]  mesh                  Mesh connectivity views.
/// @param[in]  coeffs_reconstruct_u  Zonal reconstruction coefficients (maxEdges, nCells).
/// @param[in]  coeffs_reconstruct_v  Meridional reconstruction coefficients (maxEdges, nCells).
/// @param[in]  nCells                Number of cells.
/// @param[in]  nVertLevels           Number of vertical levels.
template <typename Layout = default_layout, ExecutionPolicy Policy = SerialPolicy>
void reconstruct_velocity(
    Policy policy,
    Field2D<Layout, unchecked_accessor> u_cell,    // output: zonal velocity (nVertLevels, nCells)
    Field2D<Layout, unchecked_accessor> v_cell,    // output: meridional velocity (nVertLevels, nCells)
    ConstField2D<Layout, unchecked_accessor> u,    // input: edge-normal velocity (nVertLevels, nEdges)
    const MeshConnectivity& mesh,
    ConstField2D<Layout, unchecked_accessor> coeffs_reconstruct_u,  // (maxEdges, nCells)
    ConstField2D<Layout, unchecked_accessor> coeffs_reconstruct_v,  // (maxEdges, nCells)
    index_type nCells,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
