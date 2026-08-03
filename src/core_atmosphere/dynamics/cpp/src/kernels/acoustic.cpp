/// @file acoustic.cpp
/// @brief Implementation of the forward-backward acoustic sub-step kernel.
///
/// Faithful port of the Fortran subroutine `atm_advance_acoustic_step_work` from
/// `src/core_atmosphere/dynamics/mpas_atm_time_integration.F`. This kernel advances
/// the acoustic perturbation variables (ru_p, rw_p, rho_pp, rtheta_pp) by one
/// small timestep and accumulates time-averaged fluxes (ruAvg, wwAvg).
///
/// @section phases Algorithm Phases
///
/// Phase 1: Update horizontal velocity perturbation (ru_p) on all edges
///   - For small_step != 1: forward-backward pressure gradient update
///   - For small_step == 1: initialize from tendency only
///
/// Phase 2: Save rtheta_pp_old (previous theta perturbation)
///
/// Phase 3: Column solve for each owned cell:
///   - Initialize perturbations on first step
///   - Accumulate horizontal flux divergence into rs (density) and ts (theta)
///   - Combine with existing perturbation + vertical coupling
///   - Accumulate wwAvg (pre-update contribution from ewm)
///
/// Phase 4: Update rw_p (explicit part of vertically implicit solve)
///
/// Phase 5: Tridiagonal solve (forward sweep + backward substitution)
///
/// Phase 6: Implicit Rayleigh damping on w (gravity-wave absorbing layer)
///
/// Phase 7: Accumulate wwAvg (post-update contribution from ewp)
///
/// Phase 8: Update rho_pp and rtheta_pp from updated rw_p
///
/// @reference Klemp, J. B., Skamarock, W. C., and Dudhia, J. (2007),
/// "Conservative Split-Explicit Time Integration Methods for the Compressible
/// Nonhydrostatic Equations", Mon. Wea. Rev., 135, 2897-2913.
/// @reference Klemp, J. B., Dudhia, J., and Hassiotis, A. (2008),
/// "An upper gravity-wave absorbing layer for NWP applications",
/// Mon. Wea. Rev., 136, 3987-4004.

#include <mpas_dycore/kernels/acoustic.hpp>
#include <mpas_dycore/constants.hpp>

#include <vector>

namespace mpas::dycore::kernels {

// ============================================================================
// Explicit template instantiation for default layout and serial policy
// ============================================================================

template void advance_acoustic_step<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field2D<default_layout, unchecked_accessor> ru_p,
    Field2D<default_layout, unchecked_accessor> rw_p,
    Field2D<default_layout, unchecked_accessor> rtheta_pp,
    Field2D<default_layout, unchecked_accessor> rho_pp,
    Field2D<default_layout, unchecked_accessor> rtheta_pp_old,
    Field2D<default_layout, unchecked_accessor> ruAvg,
    Field2D<default_layout, unchecked_accessor> wwAvg,
    ConstField2D<default_layout, unchecked_accessor> rho_zz,
    ConstField2D<default_layout, unchecked_accessor> theta_m,
    ConstField2D<default_layout, unchecked_accessor> zz,
    ConstField2D<default_layout, unchecked_accessor> exner,
    ConstField2D<default_layout, unchecked_accessor> cqu,
    ConstField2D<default_layout, unchecked_accessor> zxu,
    ConstField2D<default_layout, unchecked_accessor> cofwt,
    ConstField2D<default_layout, unchecked_accessor> coftz,
    ConstField2D<default_layout, unchecked_accessor> cofwr,
    ConstField2D<default_layout, unchecked_accessor> cofwz,
    ConstField2D<default_layout, unchecked_accessor> a_tri,
    ConstField2D<default_layout, unchecked_accessor> alpha_tri,
    ConstField2D<default_layout, unchecked_accessor> gamma_tri,
    ConstField2D<default_layout, unchecked_accessor> dss,
    ConstField2D<default_layout, unchecked_accessor> tend_ru,
    ConstField2D<default_layout, unchecked_accessor> tend_rho,
    ConstField2D<default_layout, unchecked_accessor> tend_rt,
    ConstField2D<default_layout, unchecked_accessor> tend_rw,
    ConstField2D<default_layout, unchecked_accessor> w,
    ConstField2D<default_layout, unchecked_accessor> rw,
    ConstField2D<default_layout, unchecked_accessor> rw_save,
    const MeshConnectivity& mesh,
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invAreaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> cofrz,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzw,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> etp,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> etm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> ewp,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> ewm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMaskEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMaskCell,
    real_type dts,
    index_type small_step,
    index_type nCells,
    index_type nCellsAll,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges);

// ============================================================================
// Implementation
// ============================================================================

/// @brief Implementation of advance_acoustic_step.
///
/// Faithful line-by-line port of the Fortran `atm_advance_acoustic_step_work`.
/// The execution policy is accepted for API consistency but the kernel uses
/// serial iteration due to vertical data dependencies in the tridiagonal solve.
///
/// Key constants:
///   rcv = rgas / (cp - rgas)
///   c2 = cp * rcv
///   gravity = 9.80616 m/s^2
template <typename Layout, ExecutionPolicy Policy>
void advance_acoustic_step(
    Policy /*policy*/,
    Field2D<Layout, unchecked_accessor> ru_p,
    Field2D<Layout, unchecked_accessor> rw_p,
    Field2D<Layout, unchecked_accessor> rtheta_pp,
    Field2D<Layout, unchecked_accessor> rho_pp,
    Field2D<Layout, unchecked_accessor> rtheta_pp_old,
    Field2D<Layout, unchecked_accessor> ruAvg,
    Field2D<Layout, unchecked_accessor> wwAvg,
    ConstField2D<Layout, unchecked_accessor> rho_zz,
    ConstField2D<Layout, unchecked_accessor> theta_m,
    ConstField2D<Layout, unchecked_accessor> zz,
    ConstField2D<Layout, unchecked_accessor> exner,
    ConstField2D<Layout, unchecked_accessor> cqu,
    ConstField2D<Layout, unchecked_accessor> zxu,
    ConstField2D<Layout, unchecked_accessor> cofwt,
    ConstField2D<Layout, unchecked_accessor> coftz,
    ConstField2D<Layout, unchecked_accessor> cofwr,
    ConstField2D<Layout, unchecked_accessor> cofwz,
    ConstField2D<Layout, unchecked_accessor> a_tri,
    ConstField2D<Layout, unchecked_accessor> alpha_tri,
    ConstField2D<Layout, unchecked_accessor> gamma_tri,
    ConstField2D<Layout, unchecked_accessor> dss,
    ConstField2D<Layout, unchecked_accessor> tend_ru,
    ConstField2D<Layout, unchecked_accessor> tend_rho,
    ConstField2D<Layout, unchecked_accessor> tend_rt,
    ConstField2D<Layout, unchecked_accessor> tend_rw,
    ConstField2D<Layout, unchecked_accessor> w,
    ConstField2D<Layout, unchecked_accessor> rw,
    ConstField2D<Layout, unchecked_accessor> rw_save,
    const MeshConnectivity& mesh,
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invDcEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invAreaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> cofrz,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzw,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> etp,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> etm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> ewp,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> ewm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMaskEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> specZoneMaskCell,
    real_type dts,
    index_type small_step,
    index_type nCells,
    index_type nCellsAll,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxEdges)
{
    using namespace constants;

    // Zero-extent graceful handling.
    if (nCells == 0 || nEdges == 0 || nVertLevels == 0) {
        return;
    }

    // Key constants matching Fortran: rcv = rgas/(cp-rgas), c2 = cp*rcv
    const real_type rcv = rdry / (cpdry - rdry);
    const real_type c2 = cpdry * rcv;

    // ========================================================================
    // Phase 1: Update horizontal velocity perturbation (ru_p) on all edges.
    //
    // Fortran: loop over edgeStart..edgeEnd, update edges touching owned cells.
    // C++ equivalent: loop over all edges, check if either adjacent cell is owned.
    // ========================================================================

    if (small_step != 1) {
        // Forward-backward acoustic step: update ru_p with pressure gradient.
        for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

            // Skip boundary edges.
            if (cell1 == INVALID_INDEX || cell2 == INVALID_INDEX) {
                continue;
            }

            // Update edges for block-owned cells (cell1 < nCells or cell2 < nCells).
            if (cell1 < nCells || cell2 < nCells) {
                for (index_type k = 0; k < nVertLevels; ++k) {
                    // Pressure gradient computation.
                    real_type pgrad = ((rtheta_pp[k, cell2] - rtheta_pp[k, cell1])
                                       * invDcEdge[iEdge])
                                      / (0.5 * (zz[k, cell2] + zz[k, cell1]));
                    pgrad = cqu[k, iEdge] * 0.5 * c2
                            * (exner[k, cell1] + exner[k, cell2]) * pgrad;
                    pgrad = pgrad + 0.5 * zxu[k, iEdge] * gravity
                            * (rho_pp[k, cell1] + rho_pp[k, cell2]);

                    // Update ru_p with tendency and pressure gradient.
                    // specZoneMaskEdge: 1.0 in specified zone (pgrad zeroed), 0.0 elsewhere.
                    ru_p[k, iEdge] = ru_p[k, iEdge]
                        + dts * (tend_ru[k, iEdge]
                                 - (1.0 - specZoneMaskEdge[iEdge]) * pgrad);
                }

                // Accumulate ru_p for later use in scalar transport.
                for (index_type k = 0; k < nVertLevels; ++k) {
                    ruAvg[k, iEdge] = ruAvg[k, iEdge] + ru_p[k, iEdge];
                }
            }
        }
    } else {
        // First acoustic step: initialize ru_p from tendency only.
        for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

            if (cell1 == INVALID_INDEX || cell2 == INVALID_INDEX) {
                continue;
            }

            if (cell1 < nCells || cell2 < nCells) {
                for (index_type k = 0; k < nVertLevels; ++k) {
                    ru_p[k, iEdge] = dts * tend_ru[k, iEdge];
                }
                for (index_type k = 0; k < nVertLevels; ++k) {
                    ruAvg[k, iEdge] = ru_p[k, iEdge];
                }
            }
        }
    }

    // ========================================================================
    // Phase 2: Save rtheta_pp_old.
    //
    // Fortran: loop over cellStart..cellEnd (all cells including halos).
    // ========================================================================

    if (small_step == 1) {
        for (index_type iCell = 0; iCell < nCellsAll; ++iCell) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                rtheta_pp_old[k, iCell] = 0.0;
            }
        }
    } else {
        for (index_type iCell = 0; iCell < nCellsAll; ++iCell) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                rtheta_pp_old[k, iCell] = rtheta_pp[k, iCell];
            }
        }
    }

    // ========================================================================
    // Phases 3-8: Column solve for each owned cell.
    //
    // Fortran: loop over cellSolveStart..cellSolveEnd (owned cells only).
    // C++ equivalent: loop over [0, nCells).
    // ========================================================================

    // Thread-local scratch arrays for column work (ts and rs).
    std::vector<real_type> ts_vec(static_cast<std::size_t>(nVertLevels), 0.0);
    std::vector<real_type> rs_vec(static_cast<std::size_t>(nVertLevels), 0.0);

    for (index_type iCell = 0; iCell < nCells; ++iCell) {

        real_type* ts = ts_vec.data();
        real_type* rs = rs_vec.data();

        // Phase 3a: Initialize perturbation fields on first small step.
        if (small_step == 1) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                wwAvg[k, iCell] = 0.0;
                rho_pp[k, iCell] = 0.0;
                rtheta_pp[k, iCell] = 0.0;
                rw_p[k, iCell] = 0.0;
            }
            wwAvg[nVertLevels, iCell] = 0.0;
            rw_p[nVertLevels, iCell] = 0.0;
        }

        // Check specified zone mask: if in specified zone, simplified update.
        if (specZoneMaskCell[iCell] == 0.0) {
            // Not in specified zone — full computation.

            // Phase 3b: Initialize column scratch arrays.
            for (index_type k = 0; k < nVertLevels; ++k) {
                ts[k] = 0.0;
                rs[k] = 0.0;
            }

            // Phase 3c: Accumulate horizontal flux divergence.
            const index_type nEdgesOnThisCell = mesh.nEdgesOnCell[iCell];
            for (index_type i = 0; i < nEdgesOnThisCell; ++i) {
                const index_type iEdge = mesh.edgesOnCell[iCell, i];
                if (iEdge == INVALID_INDEX) break;

                const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
                const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

                for (index_type k = 0; k < nVertLevels; ++k) {
                    const real_type flux = edgesOnCell_sign[i, iCell]
                        * dts * dvEdge[iEdge] * ru_p[k, iEdge] * invAreaCell[iCell];
                    rs[k] = rs[k] - flux;
                    ts[k] = ts[k] - flux * 0.5
                            * (theta_m[k, cell2] + theta_m[k, cell1]);
                }
            }

            // Phase 3d: Combine with existing perturbation + tendency +
            //           vertical coupling from current rw_p (using ewm weights).
            for (index_type k = 0; k < nVertLevels; ++k) {
                rs[k] = rho_pp[k, iCell] + dts * tend_rho[k, iCell] + rs[k]
                         - dts * cofrz[k] * (ewm[k + 1] * rw_p[k + 1, iCell]
                                             - ewm[k] * rw_p[k, iCell]);
                ts[k] = rtheta_pp[k, iCell] + dts * tend_rt[k, iCell] + ts[k]
                         - dts * rdzw[k] * (ewm[k + 1] * coftz[k + 1, iCell] * rw_p[k + 1, iCell]
                                            - ewm[k] * coftz[k, iCell] * rw_p[k, iCell]);
            }

            // Phase 3e: Accumulate wwAvg (pre-update contribution).
            // Fortran: k=2..nVertLevels (1-based) = k=1..nVertLevels-1 (0-based).
            for (index_type k = 1; k < nVertLevels; ++k) {
                wwAvg[k, iCell] = wwAvg[k, iCell] + ewm[k] * rw_p[k, iCell];
            }

            // Phase 4: Update rw_p (explicit part of vertically implicit solve).
            // Fortran: k=2..nVertLevels (1-based) = k=1..nVertLevels-1 (0-based).
            for (index_type k = 1; k < nVertLevels; ++k) {
                rw_p[k, iCell] = rw_p[k, iCell] + dts * (tend_rw[k, iCell]
                    - cofwz[k, iCell] * ((etp[k] * zz[k, iCell] * ts[k]
                                          - etp[k - 1] * zz[k - 1, iCell] * ts[k - 1])
                                         + (etm[k] * zz[k, iCell] * rtheta_pp[k, iCell]
                                            - etm[k - 1] * zz[k - 1, iCell] * rtheta_pp[k - 1, iCell]))
                    - cofwr[k, iCell] * ((etp[k] * rs[k] + etp[k - 1] * rs[k - 1])
                                         + (etm[k] * rho_pp[k, iCell]
                                            + etm[k - 1] * rho_pp[k - 1, iCell]))
                    + cofwt[k, iCell] * (etp[k] * ts[k] + etm[k] * rtheta_pp[k, iCell])
                    + cofwt[k - 1, iCell] * (etp[k - 1] * ts[k - 1]
                                             + etm[k - 1] * rtheta_pp[k - 1, iCell]));
            }

            // Phase 5: Tridiagonal solve.
            // Forward sweep: k=2..nVertLevels (1-based) = k=1..nVertLevels-1 (0-based).
            for (index_type k = 1; k < nVertLevels; ++k) {
                rw_p[k, iCell] = (rw_p[k, iCell]
                    - (dts * dts) * a_tri[k, iCell] * rw_p[k - 1, iCell])
                    * alpha_tri[k, iCell];
            }

            // Backward sweep: k=nVertLevels..1 (1-based, descending) =
            //                  k=nVertLevels-1..0 (0-based, descending).
            // Fortran loop: do k=nVertLevels,1,-1
            //   rw_p(k,iCell) = rw_p(k,iCell) - gamma_tri(k,iCell)*rw_p(k+1,iCell)
            // In 0-based: k goes from nVertLevels-1 down to 0.
            for (index_type k = nVertLevels - 1; k >= 0; --k) {
                rw_p[k, iCell] = rw_p[k, iCell]
                    - gamma_tri[k, iCell] * rw_p[k + 1, iCell];
            }

            // Phase 6: Implicit Rayleigh damping on w (gravity-wave absorbing).
            // Fortran: k=2..nVertLevels (1-based) = k=1..nVertLevels-1 (0-based).
            for (index_type k = 1; k < nVertLevels; ++k) {
                rw_p[k, iCell] = (rw_p[k, iCell]
                    + (rw_save[k, iCell] - rw[k, iCell])
                    - dts * dss[k, iCell]
                      * (fzm[k] * zz[k, iCell] + fzp[k] * zz[k - 1, iCell])
                      * (fzm[k] * rho_zz[k, iCell] + fzp[k] * rho_zz[k - 1, iCell])
                      * w[k, iCell])
                    / (1.0 + dts * dss[k, iCell])
                    - (rw_save[k, iCell] - rw[k, iCell]);
            }

            // Phase 7: Accumulate wwAvg (post-update contribution).
            for (index_type k = 1; k < nVertLevels; ++k) {
                wwAvg[k, iCell] = wwAvg[k, iCell] + ewp[k] * rw_p[k, iCell];
            }

            // Phase 8: Update rho_pp and rtheta_pp given updated rw_p.
            for (index_type k = 0; k < nVertLevels; ++k) {
                rho_pp[k, iCell] = rs[k]
                    - dts * cofrz[k] * (ewp[k + 1] * rw_p[k + 1, iCell]
                                        - ewp[k] * rw_p[k, iCell]);
                rtheta_pp[k, iCell] = ts[k]
                    - dts * rdzw[k] * (ewp[k + 1] * coftz[k + 1, iCell] * rw_p[k + 1, iCell]
                                       - ewp[k] * coftz[k, iCell] * rw_p[k, iCell]);
            }

        } else {
            // Specified zone in regional MPAS: simplified update.
            for (index_type k = 0; k < nVertLevels; ++k) {
                rho_pp[k, iCell] = rho_pp[k, iCell] + dts * tend_rho[k, iCell];
                rtheta_pp[k, iCell] = rtheta_pp[k, iCell] + dts * tend_rt[k, iCell];
                rw_p[k, iCell] = rw_p[k, iCell] + dts * tend_rw[k, iCell];
                wwAvg[k, iCell] = wwAvg[k, iCell] + ewp[k] * rw_p[k, iCell];
            }
        }

    } // end loop over cells
}

} // namespace mpas::dycore::kernels
