/// @file tend_w.cpp
/// @brief Implementation of the vertical momentum tendency kernel.
///
/// Computes the vertical momentum tendency at cell interfaces using the
/// split-explicit perturbation formulation. The hydrostatic base state is
/// analytically removed, so only perturbation pressure and buoyancy drive
/// the w tendency. This kernel operates on nVertLevels+1 interfaces per cell,
/// applying rigid-lid boundary conditions at top (k=0) and surface (k=nVertLevels).
///
/// @section governing_equations Governing Equations
///
/// For interior interfaces k = 1 to nVertLevels-1:
///   pgrad = rdzu(k) * (pp(k, iCell) - pp(k-1, iCell))
///   buoyancy = fzm(k) * dpdz(k, iCell) + fzp(k) * dpdz(k-1, iCell)
///   tend_w(k, iCell) = -cqw(k, iCell) * (pgrad - buoyancy)
///
/// Boundary conditions:
///   tend_w(0, iCell) = 0.0       (rigid lid at model top)
///   tend_w(nVertLevels, iCell) = 0.0  (rigid surface)
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.

#include <mpas_dycore/kernels/tendencies.hpp>

namespace mpas::dycore::kernels {

// Explicit instantiation for default layout and serial policy.
template void compute_tend_w<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> tend_w,
    ConstField2D<default_layout, unchecked_accessor> pp,
    ConstField2D<default_layout, unchecked_accessor> cqw,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzu,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    ConstField2D<default_layout, unchecked_accessor> dpdz,
    index_type nCells,
    index_type nVertLevels);

/// @brief Implementation of compute_tend_w.
///
/// Computes vertical momentum tendency at cell interfaces using the
/// split-explicit perturbation pressure gradient and buoyancy formulation.
/// The execution policy controls the outer loop parallelization over cells;
/// the inner loop over interfaces is executed sequentially within each cell.
template <typename Layout, ExecutionPolicy Policy>
void compute_tend_w(
    Policy policy,
    Field2D<Layout, unchecked_accessor> tend_w,
    ConstField2D<Layout, unchecked_accessor> pp,
    ConstField2D<Layout, unchecked_accessor> cqw,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzu,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    ConstField2D<Layout, unchecked_accessor> dpdz,
    index_type nCells,
    index_type nVertLevels)
{
    // Handle zero-extent cases: nothing to compute.
    if (nCells == 0 || nVertLevels == 0) {
        return;
    }

    // Iterate over all cells. For each cell, compute the tendency at all
    // nVertLevels+1 interfaces. We use the policy's parallel_for with
    // nEntities=nCells and nLevels=1 to parallelize over cells only,
    // since the inner interface loop has variable bounds (0..nVertLevels).
    policy.parallel_for(nCells, index_type{1}, [&](index_type iCell, index_type /*unused*/) {
        // Top boundary: rigid lid
        tend_w[0, iCell] = 0.0;

        // Interior interfaces: k = 1 to nVertLevels-1
        for (index_type k = 1; k < nVertLevels; ++k) {
            // Perturbation pressure gradient at interface
            const real_type pgrad = rdzu[k] * (pp[k, iCell] - pp[k - 1, iCell]);

            // Buoyancy interpolated to interface from cell centers
            const real_type buoyancy = fzm[k] * dpdz[k, iCell] + fzp[k] * dpdz[k - 1, iCell];

            // Combined tendency with moist correction coefficient
            tend_w[k, iCell] = -cqw[k, iCell] * (pgrad - buoyancy);
        }

        // Surface boundary: rigid surface
        tend_w[nVertLevels, iCell] = 0.0;
    });
}

} // namespace mpas::dycore::kernels
