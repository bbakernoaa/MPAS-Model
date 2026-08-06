/// @file vert_implicit.cpp
/// @brief Implementation of the vertical implicit coefficient computation kernel.
///
/// Faithful line-by-line port of the Fortran subroutine `atm_compute_vert_imp_coefs_work`
/// from mpas_atm_time_integration.F. Computes the tridiagonal matrix coefficients
/// (cofwr, cofwz, coftz, cofwt, a_tri, alpha_tri, gamma_tri, cofrz) used in the
/// vertically implicit treatment of acoustic/gravity wave modes.
///
/// @section indexing Indexing Convention
/// Fortran 1-based → C++ 0-based:
///   Fortran k=1       → C++ k=0
///   Fortran k=2..nVL  → C++ k=1..nVL-1
///
/// coftz has nVertLevels+1 entries indexed 0..nVertLevels.
/// cofwr, cofwz computed at interfaces k=1..nVertLevels-1 (Fortran k=2..nVL).
/// cofwt computed at levels k=0..nVertLevels-1 (Fortran k=1..nVL).
/// Tridiagonal assembly at k=1..nVertLevels-1 (Fortran k=2..nVL).
/// LU forward sweep at k=1..nVertLevels-1 (Fortran k=2..nVL), sequential.
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.

#include <mpas_dycore/kernels/vert_implicit.hpp>
#include <mpas_dycore/constants.hpp>

namespace mpas::dycore::kernels {

/// @brief Implementation of compute_vert_imp_coefs.
///
/// Iterates column-by-column over cells, computing interface and level
/// coefficients, assembling the tridiagonal system, and performing the
/// forward sweep of LU factorization.
///
/// The execution policy parameter is accepted for API consistency but the
/// kernel uses serial column iteration due to vertical data dependencies
/// in the LU factorization sweep.
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
    Span1D cofrz,
    ConstField2D<Layout, unchecked_accessor> zz,
    ConstField2D<Layout, unchecked_accessor> p,
    ConstField2D<Layout, unchecked_accessor> t,
    ConstField2D<Layout, unchecked_accessor> rb,
    ConstField2D<Layout, unchecked_accessor> rtb,
    ConstField2D<Layout, unchecked_accessor> pb,
    ConstField2D<Layout, unchecked_accessor> rt,
    ConstField2D<Layout, unchecked_accessor> cqw,
    ConstField2D<Layout, unchecked_accessor> qtot,
    ConstSpan1D rdzw,
    ConstSpan1D fzm,
    ConstSpan1D fzp,
    ConstSpan1D rdzu,
    ConstSpan1D etp,
    ConstSpan1D ewp,
    real_type dts,
    index_type nCells,
    index_type nVertLevels)
{
    // Physical constants (matching Fortran: rgas, cp)
    const real_type rgas = constants::rdry;
    const real_type cp   = constants::cpdry;

    // rcv = rgas / (cp - rgas)  (Fortran: rcv = rgas/(cp-rgas))
    const real_type rcv = rgas / (cp - rgas);
    // c2 = cp * rcv  (Fortran: c2 = cp*rcv)
    const real_type c2 = cp * rcv;

    // dts squared for LU factorization
    const real_type dts2 = dts * dts;

    // ------------------------------------------------------------------
    // Step 1: Set cofrz (1D, level-only): cofrz(k) = rdzw(k)
    //   Fortran: do k=1,nVertLevels; cofrz(k) = rdzw(k); end do
    //   C++ 0-based: k=0..nVertLevels-1
    // ------------------------------------------------------------------
    for (index_type k = 0; k < nVertLevels; ++k) {
        cofrz[k] = rdzw[k];
    }

    // ------------------------------------------------------------------
    // Step 2: Loop over cells
    // ------------------------------------------------------------------
    for (index_type iCell = 0; iCell < nCells; ++iCell) {

        // --------------------------------------------------------------
        // cofwr at interfaces k=1..nVertLevels-1 (Fortran k=2..nVertLevels)
        //   cofwr(k,iCell) = 0.5*gravity*(fzm(k)*zz(k,iCell)+fzp(k)*zz(k-1,iCell))
        // --------------------------------------------------------------
        for (index_type k = 1; k < nVertLevels; ++k) {
            cofwr[k, iCell] = 0.5 * constants::gravity
                * (fzm[k] * zz[k, iCell] + fzp[k] * zz[k - 1, iCell]);
        }

        // --------------------------------------------------------------
        // coftz at interfaces (nVertLevels+1 entries)
        //   coftz(1,iCell) = 0.0                          → C++ coftz[0,iCell] = 0
        //   coftz(k,iCell) = fzm(k)*t(k)+fzp(k)*t(k-1)  → k=1..nVL-1
        //   coftz(nVertLevels+1,iCell) = 0.0              → C++ coftz[nVL,iCell] = 0
        // --------------------------------------------------------------
        coftz[0, iCell] = 0.0;
        for (index_type k = 1; k < nVertLevels; ++k) {
            coftz[k, iCell] = fzm[k] * t[k, iCell] + fzp[k] * t[k - 1, iCell];
        }
        coftz[nVertLevels, iCell] = 0.0;

        // --------------------------------------------------------------
        // cofwz at interfaces k=1..nVertLevels-1 (Fortran k=2..nVertLevels)
        //   cofwz(k,iCell) = c2*(fzm(k)*zz(k,iCell)+fzp(k)*zz(k-1,iCell))
        //                    *rdzu(k)*cqw(k,iCell)*(fzm(k)*p(k,iCell)+fzp(k)*p(k-1,iCell))
        // --------------------------------------------------------------
        for (index_type k = 1; k < nVertLevels; ++k) {
            cofwz[k, iCell] = c2
                * (fzm[k] * zz[k, iCell] + fzp[k] * zz[k - 1, iCell])
                * rdzu[k] * cqw[k, iCell]
                * (fzm[k] * p[k, iCell] + fzp[k] * p[k - 1, iCell]);
        }

        // --------------------------------------------------------------
        // cofwt at levels k=0..nVertLevels-1 (Fortran k=1..nVertLevels)
        //   cofwt(k,iCell) = 0.5*rcv*zz(k,iCell)*gravity*rb(k,iCell)/(1.+qtotal)
        //                    *p(k,iCell)/((rtb(k,iCell)+rt(k,iCell))*pb(k,iCell))
        // --------------------------------------------------------------
        for (index_type k = 0; k < nVertLevels; ++k) {
            real_type qtotal = qtot[k, iCell];
            cofwt[k, iCell] = 0.5 * rcv * zz[k, iCell] * constants::gravity
                * rb[k, iCell] / (1.0 + qtotal)
                * p[k, iCell] / ((rtb[k, iCell] + rt[k, iCell]) * pb[k, iCell]);
        }

        // --------------------------------------------------------------
        // Tridiagonal assembly and LU factorization
        //
        // Fortran initialization at k=1 (C++ k=0):
        //   a_tri(1) = 0, gamma_tri(1) = 0, alpha_tri(1) = 0
        // Note: b_tri(1)=1, c_tri(1)=0 are never used.
        // --------------------------------------------------------------
        a_tri[0, iCell] = 0.0;
        gamma_tri[0, iCell] = 0.0;
        alpha_tri[0, iCell] = 0.0;

        // Tridiagonal assembly: k=1..nVertLevels-1 (Fortran k=2..nVertLevels)
        // Uses local b_tri and c_tri (per-level temporaries).
        // Then sequential forward sweep for LU factorization.

        // We need b_tri and c_tri as temporaries. Since LU sweep is sequential,
        // we can compute all of them first and then sweep, or interleave.
        // For faithfulness, compute all a_tri, b_tri, c_tri first, then sweep.

        // Temporary arrays for b_tri and c_tri (stack-allocated for small nVertLevels,
        // but we'll use a simple loop approach that matches the Fortran structure).
        // Since nVertLevels is typically <=100, we can use VLA-like approach or
        // just combine the loops as the Fortran does (compute then sweep).

        // First pass: compute a_tri, b_tri, c_tri for k=1..nVertLevels-1
        // We store b_tri and c_tri temporarily. Using a small buffer on the stack.
        // Maximum typical nVertLevels in MPAS is ~55, allocate up to 256 for safety.
        real_type b_tri[256];
        real_type c_tri[256];

        for (index_type k = 1; k < nVertLevels; ++k) {
            // a_tri(k,iCell) = (-cofwz(k)*coftz(k-1)*rdzw(k-1)*zz(k-1,iCell)
            //                   +cofwr(k)*cofrz(k-1)
            //                   -cofwt(k-1)*coftz(k-1)*rdzw(k-1))
            //                  * etp(k-1)*ewp(k-1)
            //
            // Note: Fortran k maps to C++ k; Fortran k-1 maps to C++ k-1.
            // cofrz(k-1) in Fortran = cofrz[k-1] in C++ = rdzw[k-1]
            a_tri[k, iCell] = (-cofwz[k, iCell] * coftz[k - 1, iCell] * rdzw[k - 1] * zz[k - 1, iCell]
                               + cofwr[k, iCell] * cofrz[k - 1]
                               - cofwt[k - 1, iCell] * coftz[k - 1, iCell] * rdzw[k - 1])
                              * etp[k - 1] * ewp[k - 1];

            // b_tri(k) = (+cofwz(k)*coftz(k)*(etp(k)*rdzw(k)*zz(k,iCell)+etp(k-1)*rdzw(k-1)*zz(k-1,iCell))
            //             -coftz(k)*(etp(k)*cofwt(k)*rdzw(k) - etp(k-1)*cofwt(k-1)*rdzw(k-1))
            //             +cofwr(k)*(etp(k)*cofrz(k) - etp(k-1)*cofrz(k-1)))
            //            * ewp(k)
            //
            // Note: cofrz(k) = rdzw[k], cofrz(k-1) = rdzw[k-1] (0-based)
            // But wait — in Fortran, cofrz(k) for k=2..nVertLevels means C++ index k.
            // Actually in Fortran: cofrz is indexed 1..nVertLevels.
            // Fortran cofrz(k) = rdzw(k) for k=1..nVertLevels.
            // In the b_tri formula, Fortran uses cofrz(k) and cofrz(k-1) where k starts at 2.
            // So Fortran cofrz(k) at k=2 is rdzw(2), and cofrz(k-1)=cofrz(1)=rdzw(1).
            // In C++ 0-based: cofrz[k] = rdzw[k] for k=0..nVL-1.
            // Fortran b_tri at k uses cofrz(k) and cofrz(k-1):
            //   In C++ at k (where our k corresponds to Fortran k):
            //   cofrz(k) → cofrz[k] = rdzw[k]
            //   cofrz(k-1) → cofrz[k-1] = rdzw[k-1]
            b_tri[k] = (cofwz[k, iCell] * coftz[k, iCell]
                            * (etp[k] * rdzw[k] * zz[k, iCell]
                               + etp[k - 1] * rdzw[k - 1] * zz[k - 1, iCell])
                        - coftz[k, iCell]
                            * (etp[k] * cofwt[k, iCell] * rdzw[k]
                               - etp[k - 1] * cofwt[k - 1, iCell] * rdzw[k - 1])
                        + cofwr[k, iCell]
                            * (etp[k] * cofrz[k] - etp[k - 1] * cofrz[k - 1]))
                       * ewp[k];

            // c_tri(k) = (-cofwz(k)*coftz(k+1)*rdzw(k)*zz(k,iCell)
            //             -cofwr(k)*cofrz(k)
            //             +cofwt(k)*coftz(k+1)*rdzw(k))
            //            * etp(k)*ewp(k+1)
            c_tri[k] = (-cofwz[k, iCell] * coftz[k + 1, iCell] * rdzw[k] * zz[k, iCell]
                         - cofwr[k, iCell] * cofrz[k]
                         + cofwt[k, iCell] * coftz[k + 1, iCell] * rdzw[k])
                        * etp[k] * ewp[k + 1];
        }

        // c_tri(nVertLevels) = 0.0  →  C++ c_tri[nVertLevels-1] = 0.0
        c_tri[nVertLevels - 1] = 0.0;

        // Forward sweep of LU factorization (sequential, k=1..nVertLevels-1):
        //   alpha_tri(k) = 1/(1.0 + dts^2*(b_tri(k) - a_tri(k)*gamma_tri(k-1)))
        //   gamma_tri(k) = dts^2 * c_tri(k) * alpha_tri(k)
        for (index_type k = 1; k < nVertLevels; ++k) {
            alpha_tri[k, iCell] = 1.0
                / (1.0 + dts2 * (b_tri[k] - a_tri[k, iCell] * gamma_tri[k - 1, iCell]));
            gamma_tri[k, iCell] = dts2 * c_tri[k] * alpha_tri[k, iCell];
        }
    } // end loop over cells
}

// Explicit instantiation for default_layout / SerialPolicy.
template void compute_vert_imp_coefs<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> cofwr,
    Field2D<default_layout, unchecked_accessor> cofwz,
    Field2D<default_layout, unchecked_accessor> coftz,
    Field2D<default_layout, unchecked_accessor> cofwt,
    Field2D<default_layout, unchecked_accessor> a_tri,
    Field2D<default_layout, unchecked_accessor> alpha_tri,
    Field2D<default_layout, unchecked_accessor> gamma_tri,
    Span1D cofrz,
    ConstField2D<default_layout, unchecked_accessor> zz,
    ConstField2D<default_layout, unchecked_accessor> p,
    ConstField2D<default_layout, unchecked_accessor> t,
    ConstField2D<default_layout, unchecked_accessor> rb,
    ConstField2D<default_layout, unchecked_accessor> rtb,
    ConstField2D<default_layout, unchecked_accessor> pb,
    ConstField2D<default_layout, unchecked_accessor> rt,
    ConstField2D<default_layout, unchecked_accessor> cqw,
    ConstField2D<default_layout, unchecked_accessor> qtot,
    ConstSpan1D rdzw,
    ConstSpan1D fzm,
    ConstSpan1D fzp,
    ConstSpan1D rdzu,
    ConstSpan1D etp,
    ConstSpan1D ewp,
    real_type dts,
    index_type nCells,
    index_type nVertLevels);

} // namespace mpas::dycore::kernels
