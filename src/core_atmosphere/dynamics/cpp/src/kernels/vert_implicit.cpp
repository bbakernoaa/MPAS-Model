/// @file vert_implicit.cpp
/// @brief Implementation of the vertical implicit coefficient computation kernel.
///
/// Computes the tridiagonal matrix coefficients (cofwr, cofwz, coftz, cofwt,
/// a_tri, alpha_tri, gamma_tri, cofrz) used in the vertically implicit treatment
/// of acoustic/gravity wave modes during the acoustic timestep. These coefficients
/// couple the w, theta, and rho equations in the vertical direction.
///
/// @section governing_equations Governing Equations
/// The off-centering parameter epssm = 0.5 controls the implicit/explicit split.
/// For interface k (interior: k = 1 to nVertLevels-1):
///
///   theta_interface = fzm[k] * theta_m[k, iCell] + fzp[k] * theta_m[k-1, iCell]
///   rho_interface   = fzm[k] * rho_zz[k, iCell]  + fzp[k] * rho_zz[k-1, iCell]
///
///   cofwr(k, iCell) = epssm * dts * rho_interface * rdzu[k]
///   cofwz(k, iCell) = epssm * dts * (rdry/cvdry) * theta_interface * rdzu[k] * cqw[k, iCell]
///   coftz(k, iCell) = epssm * dts * gravity * rho_zz(k, iCell)
///   cofwt(k, iCell) = epssm * dts * (theta_m[k, iCell] - theta_m[k-1, iCell]) * rdzu[k]
///
/// The tridiagonal coefficients are:
///   a_tri(k)     = -cofwz(k) * coftz(k)              (lower diagonal)
///   gamma_tri(k) = -cofwz(k+1) * coftz(k)            (upper diagonal)
///   alpha_tri(k) = 1 + cofwz(k)*coftz(k-1) + cofwz(k+1)*coftz(k) (main diagonal)
///   cofrz(k)     = cofwr(k) * rho_zz(k, iCell)
///
/// Boundary conditions: cofwr, cofwz, cofwt = 0 at k=0 and k=nVertLevels.
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.

#include <mpas_dycore/kernels/vert_implicit.hpp>

namespace mpas::dycore::kernels {

// Explicit instantiation for default layout and serial policy.
template void compute_vert_imp_coefs<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> cofwr,
    Field2D<default_layout, unchecked_accessor> cofwz,
    Field2D<default_layout, unchecked_accessor> coftz,
    Field2D<default_layout, unchecked_accessor> cofwt,
    Field2D<default_layout, unchecked_accessor> a_tri,
    Field2D<default_layout, unchecked_accessor> alpha_tri,
    Field2D<default_layout, unchecked_accessor> gamma_tri,
    Field2D<default_layout, unchecked_accessor> cofrz,
    ConstField2D<default_layout, unchecked_accessor> theta_m,
    ConstField2D<default_layout, unchecked_accessor> rho_zz,
    ConstField2D<default_layout, unchecked_accessor> cqw,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzu,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    real_type dts,
    real_type gravity,
    real_type rdry,
    real_type cvdry,
    index_type nCells,
    index_type nVertLevels);

/// @brief Implementation of compute_vert_imp_coefs.
///
/// Iterates column-by-column over cells, computing interface coefficients
/// and then assembling the tridiagonal system. The execution policy parameter
/// is accepted for API consistency but the kernel uses serial column iteration
/// due to vertical data dependencies in tridiagonal assembly.
///
/// The dts-change recomputation logic is handled at the call site: this kernel
/// should only be invoked when dts differs from its previous value (tracked by
/// the caller). The kernel itself is stateless and always recomputes all
/// coefficients unconditionally.
template <typename Layout, ExecutionPolicy Policy>
void compute_vert_imp_coefs(
    Policy /*policy*/,
    Field2D<Layout, unchecked_accessor> cofwr,
    Field2D<Layout, unchecked_accessor> cofwz,
    Field2D<Layout, unchecked_accessor> coftz,
    Field2D<Layout, unchecked_accessor> cofwt,
    Field2D<Layout, unchecked_accessor> a_tri,
    Field2D<Layout, unchecked_accessor> alpha_tri,
    Field2D<Layout, unchecked_accessor> gamma_tri,
    Field2D<Layout, unchecked_accessor> cofrz,
    ConstField2D<Layout, unchecked_accessor> theta_m,
    ConstField2D<Layout, unchecked_accessor> rho_zz,
    ConstField2D<Layout, unchecked_accessor> cqw,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzu,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    real_type dts,
    real_type gravity,
    real_type rdry,
    real_type cvdry,
    index_type nCells,
    index_type nVertLevels)
{
    // Off-centering parameter for the implicit treatment of acoustic modes.
    constexpr real_type epssm = 0.5;

    // Precompute constant factor: rdry / cvdry
    const real_type rcv = rdry / cvdry;

    for (index_type iCell = 0; iCell < nCells; ++iCell) {
        // ------------------------------------------------------------------
        // Step 1: Set boundary conditions at top (k=0) and bottom (k=nVertLevels).
        // ------------------------------------------------------------------
        cofwr[0, iCell] = 0.0;
        cofwz[0, iCell] = 0.0;
        cofwt[0, iCell] = 0.0;
        cofwr[nVertLevels, iCell] = 0.0;
        cofwz[nVertLevels, iCell] = 0.0;
        cofwt[nVertLevels, iCell] = 0.0;

        // ------------------------------------------------------------------
        // Step 2: Compute interface coefficients for interior interfaces
        //         k = 1 to nVertLevels-1.
        // ------------------------------------------------------------------
        for (index_type k = 1; k < nVertLevels; ++k) {
            // Interface values using fzm/fzp interpolation weights.
            real_type theta_interface = fzm[k] * theta_m[k, iCell]
                                      + fzp[k] * theta_m[k - 1, iCell];
            real_type rho_interface   = fzm[k] * rho_zz[k, iCell]
                                      + fzp[k] * rho_zz[k - 1, iCell];

            // Coefficient relating w perturbation to density perturbation.
            cofwr[k, iCell] = epssm * dts * rho_interface * rdzu[k];

            // Coefficient for w in the implicit pressure gradient force.
            // Uses (rdry/cvdry) * theta_interface * rdzu * cqw.
            cofwz[k, iCell] = epssm * dts * rcv * theta_interface
                            * rdzu[k] * cqw[k, iCell];

            // Coefficient for w in the theta equation (vertical advection).
            cofwt[k, iCell] = epssm * dts
                            * (theta_m[k, iCell] - theta_m[k - 1, iCell])
                            * rdzu[k];
        }

        // ------------------------------------------------------------------
        // Step 3: Compute level-centered coefficients for k = 0 to nVertLevels-1.
        // ------------------------------------------------------------------
        for (index_type k = 0; k < nVertLevels; ++k) {
            // Coefficient for theta in the z-momentum equation (buoyancy).
            coftz[k, iCell] = epssm * dts * gravity * rho_zz[k, iCell];

            // Coefficient for rho-zz coupling.
            cofrz[k, iCell] = cofwr[k, iCell] * rho_zz[k, iCell];
        }

        // ------------------------------------------------------------------
        // Step 4: Assemble tridiagonal coefficients.
        // ------------------------------------------------------------------
        for (index_type k = 0; k < nVertLevels; ++k) {
            // Lower diagonal: coupling from interface k below level k.
            a_tri[k, iCell] = -cofwz[k, iCell] * coftz[k, iCell];

            // Upper diagonal: coupling from interface k+1 above level k.
            gamma_tri[k, iCell] = -cofwz[k + 1, iCell] * coftz[k, iCell];

            // Main diagonal factor.
            // For k=0: cofwz[0]=0 so the lower term vanishes naturally.
            // For k=nVertLevels-1: cofwz[nVertLevels]=0 so upper term vanishes.
            real_type lower_contrib = (k > 0) ? cofwz[k, iCell] * coftz[k - 1, iCell] : 0.0;
            real_type upper_contrib = cofwz[k + 1, iCell] * coftz[k, iCell];
            alpha_tri[k, iCell] = 1.0 + lower_contrib + upper_contrib;
        }
    }
}

} // namespace mpas::dycore::kernels
