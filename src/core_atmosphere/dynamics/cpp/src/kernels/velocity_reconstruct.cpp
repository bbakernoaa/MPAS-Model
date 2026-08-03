/// @file velocity_reconstruct.cpp
/// @brief Implementation of the velocity reconstruction kernel.
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
/// The coefficients encode geometric information about the mesh (edge normals,
/// cell geometry) and are computed once during mesh initialization using a
/// least-squares procedure.
///
/// @reference Thuburn, J., Ringler, T. D., Skamarock, W. C., and Klemp, J. B.
/// (2009), "Numerical representation of geostrophic modes on arbitrarily
/// structured C-grids", J. Comput. Phys., 228, 8321-8335.

#include <mpas_dycore/kernels/velocity_reconstruct.hpp>

namespace mpas::dycore::kernels {

// Explicit instantiation for default layout and serial policy.
template void reconstruct_velocity<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> u_cell,
    Field2D<default_layout, unchecked_accessor> v_cell,
    ConstField2D<default_layout, unchecked_accessor> u,
    const MeshConnectivity& mesh,
    ConstField2D<default_layout, unchecked_accessor> coeffs_reconstruct_u,
    ConstField2D<default_layout, unchecked_accessor> coeffs_reconstruct_v,
    index_type nCells,
    index_type nVertLevels);

/// @brief Implementation of reconstruct_velocity.
///
/// For each cell, iterates over its edges and accumulates the weighted
/// contribution of each edge-normal velocity to the zonal and meridional
/// cell-center velocity components using the pre-computed reconstruction
/// coefficients.
///
/// Loop ordering: outer over cells, middle over vertical levels, inner over
/// edges of cell. This ordering maximizes data locality for cell-centered
/// output fields.
template <typename Layout, ExecutionPolicy Policy>
void reconstruct_velocity(
    Policy /*policy*/,
    Field2D<Layout, unchecked_accessor> u_cell,
    Field2D<Layout, unchecked_accessor> v_cell,
    ConstField2D<Layout, unchecked_accessor> u,
    const MeshConnectivity& mesh,
    ConstField2D<Layout, unchecked_accessor> coeffs_reconstruct_u,
    ConstField2D<Layout, unchecked_accessor> coeffs_reconstruct_v,
    index_type nCells,
    index_type nVertLevels)
{
    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        const index_type nEdgesOnThisCell = mesh.nEdgesOnCell[iCell];

        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type u_sum = 0.0;
            real_type v_sum = 0.0;

            for (index_type j = 0; j < nEdgesOnThisCell; ++j) {
                const index_type iEdge = mesh.edgesOnCell[iCell, j];
                if (iEdge == INVALID_INDEX) break;

                const real_type u_edge = u[k, iEdge];
                u_sum += coeffs_reconstruct_u[j, iCell] * u_edge;
                v_sum += coeffs_reconstruct_v[j, iCell] * u_edge;
            }

            u_cell[k, iCell] = u_sum;
            v_cell[k, iCell] = v_sum;
        }
    }
}

} // namespace mpas::dycore::kernels
