/// @file tend_u.cpp
/// @brief Implementation of the horizontal momentum tendency kernel.
///
/// Computes the right-hand side of the horizontal momentum equation on edges
/// using the TRiSK discretization (Ringler et al., JCP 2010). This kernel is
/// called during the dynamics tendency computation phase of each RK sub-step,
/// after diagnostic fields (ke, pv_edge, rho_edge) have been computed.
///
/// @section role Role in the SRK3 Integration
/// Within the `compute_dyn_tend` orchestrator, this kernel runs after
/// `compute_solve_diagnostics` has produced kinetic energy, potential vorticity,
/// and edge density. The resulting tend_u is then used by the acoustic sub-stepping
/// to advance the horizontal velocity.
///
/// @section governing_equation Governing Equation
///
/// For each edge iEdge with adjacent cells cell0 and cell1:
///
///   pressure_grad = -cqu(k, iEdge) * (pressure(k, cell1) - pressure(k, cell0)) * invDcEdge(iEdge)
///
///   q = sum_{j=0}^{nEdgesOnEdge-1} weightsOnEdge(iEdge, j) * u(k, eoe_j)
///       * 0.5 * (pv_edge(k, iEdge) + pv_edge(k, eoe_j))
///
///   ke_grad = (ke(k, cell1) - ke(k, cell0)) * invDcEdge(iEdge)
///
///   tend_u(k, iEdge) = pressure_grad + rho_edge(k, iEdge) * (q - ke_grad)
///
/// The pressure gradient uses a flat-terrain simplification (zz=1, zxu=0).
/// The Coriolis term uses the TRiSK reconstruction which sums over neighboring
/// edges with averaged potential vorticity (Ringler et al. 2010). The kinetic
/// energy gradient is normalized by invDcEdge.
///
/// For boundary edges where cell1 == INVALID_INDEX, tend_u is set to zero.
///
/// @section inputs Inputs (physical terms)
/// - u: normal velocity component on edges (m/s)
/// - pressure: cell-centered modified pressure (Pa)
/// - pv_edge: potential vorticity interpolated to edges (1/(m*s))
/// - ke: kinetic energy per unit mass on cells (m^2/s^2)
/// - rho_edge: dry air density interpolated to edges (kg/m^3)
/// - cqu: moist coefficient on edges (dimensionless)
/// - invDcEdge: reciprocal distance between cell centers (1/m)
/// - edgesOnEdge: neighbor edge indices for TRiSK reconstruction
/// - weightsOnEdge: TRiSK reconstruction weights
/// - nEdgesOnEdge: number of neighbor edges per edge
///
/// @section output Output
/// - tend_u: time tendency of normal velocity on edges (m/s^2)
///
/// @reference Skamarock, W. C. and Klemp, J. B. (2008), "A time-split
/// nonhydrostatic atmospheric model for weather research and forecasting
/// applications", J. Comput. Phys., 227, 3465-3485.
/// @reference Ringler, T., Thuburn, J., Klemp, J., and Skamarock, W. (2010),
/// "A unified approach to energy conservation and potential vorticity dynamics
/// for arbitrarily-structured C-grids", J. Comput. Phys., 229, 3065-3090.

#include <mpas_dycore/kernels/tendencies.hpp>

namespace mpas::dycore::kernels {

// ============================================================================
// Explicit template instantiation for default layout and serial policy
// ============================================================================

template void compute_tend_u<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> tend_u,
    ConstField2D<default_layout, unchecked_accessor> u,
    ConstField2D<default_layout, unchecked_accessor> pressure,
    ConstField2D<default_layout, unchecked_accessor> pv_edge,
    ConstField2D<default_layout, unchecked_accessor> ke,
    ConstField2D<default_layout, unchecked_accessor> rho_edge,
    ConstField2D<default_layout, unchecked_accessor> cqu,
    const MeshConnectivity& mesh,
    ConnectivityView edgesOnEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> weightsOnEdge,
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>> nEdgesOnEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges2);

// ============================================================================
// Implementation
// ============================================================================

/// @brief Implementation of compute_tend_u.
///
/// Iterates over all edges and vertical levels computing the horizontal
/// momentum tendency using the TRiSK discretization. The execution policy
/// controls parallelization of the outer edge loop. Uses descriptive loop
/// variable names consistent with the Fortran reference (iEdge, k, cell0, cell1).
template <typename Layout, ExecutionPolicy Policy>
void compute_tend_u(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_u,
    ConstField2D<Layout, unchecked_accessor> u,
    ConstField2D<Layout, unchecked_accessor> pressure,
    ConstField2D<Layout, unchecked_accessor> pv_edge,
    ConstField2D<Layout, unchecked_accessor> ke,
    ConstField2D<Layout, unchecked_accessor> rho_edge,
    ConstField2D<Layout, unchecked_accessor> cqu,
    const MeshConnectivity& mesh,
    ConnectivityView edgesOnEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> weightsOnEdge,
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>> nEdgesOnEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges2)
{
    // Zero-extent graceful handling: produce no output if dimensions are zero.
    if (nEdges == 0 || nVertLevels == 0) {
        return;
    }

    // Dispatch over (iEdge, k) pairs via the execution policy.
    policy.parallel_for(nEdges, nVertLevels,
        [&](index_type iEdge, index_type k) {
            // Retrieve the two cells adjacent to this edge from connectivity.
            const index_type cell0 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 1];

            // Boundary edge: one side has no adjacent cell.
            if (cell1 == INVALID_INDEX) {
                tend_u[k, iEdge] = 0.0;
                return;
            }

            // --- Pressure gradient term (flat terrain simplification) ---
            // Full formula: -cqu * ((pp(cell2)-pp(cell1))*invDcEdge / (0.5*(zz(cell2)+zz(cell1)))
            //                       - 0.5*zxu*(dpdz(cell1)+dpdz(cell2)))
            // For flat terrain (zz=1, zxu=0): -cqu * (pressure(cell1) - pressure(cell0)) * invDcEdge
            const real_type pressure_grad =
                -cqu[k, iEdge] * (pressure[k, cell1] - pressure[k, cell0]) * invDcEdge[iEdge];

            // --- TRiSK Coriolis reconstruction (Ringler et al. 2010) ---
            // q = sum_j weightsOnEdge(j, iEdge) * u(k, eoe) * 0.5*(pv_edge(k, iEdge) + pv_edge(k, eoe))
            real_type q = 0.0;
            const index_type n_eoe = nEdgesOnEdge[iEdge];
            for (index_type j = 0; j < n_eoe; ++j) {
                const index_type eoe = edgesOnEdge[iEdge, j];
                if (eoe == INVALID_INDEX) break;
                const real_type workpv = 0.5 * (pv_edge[k, iEdge] + pv_edge[k, eoe]);
                q += weightsOnEdge[iEdge * maxEdges2 + j] * u[k, eoe] * workpv;
            }

            // --- Kinetic energy gradient with invDcEdge normalization ---
            const real_type ke_grad = (ke[k, cell1] - ke[k, cell0]) * invDcEdge[iEdge];

            // --- Assemble tendency ---
            // tend_u = pressure_grad + rho_edge * (Coriolis - KE_gradient)
            tend_u[k, iEdge] = pressure_grad + rho_edge[k, iEdge] * (q - ke_grad);
        });
}

} // namespace mpas::dycore::kernels
