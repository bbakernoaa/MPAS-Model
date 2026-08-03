/// @file scalars.cpp
/// @brief Implementation of the scalar transport kernel with FCT limiting.
///
/// Faithful port of the Fortran subroutine `atm_advance_scalars_mono_work` from
/// `src/core_atmosphere/dynamics/mpas_atm_time_integration.F`.
///
/// The algorithm proceeds per-scalar through the following phases:
/// 1. Compute vertical flux (3rd-order interior, linear at boundaries)
/// 2. Compute local min/max bounds from pre-advection scalar values
/// 3. Compute high-order horizontal flux using advection stencil
/// 4. Decompose into upwind + anti-diffusive components
/// 5. Limit anti-diffusive fluxes using Zalesak FCT
/// 6. Apply limited fluxes, divide by new density, clamp to >= 0

#include <mpas_dycore/kernels/scalars.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace mpas::dycore::kernels {

// ============================================================================
// Explicit template instantiation for default layout and serial policy
// ============================================================================

template void advance_scalars_mono<default_layout, SerialPolicy>(
    SerialPolicy policy,
    Field3D<default_layout, unchecked_accessor> scalars,
    ConstField2D<default_layout, unchecked_accessor> ruAvg,
    ConstField2D<default_layout, unchecked_accessor> wwAvg,
    ConstField2D<default_layout, unchecked_accessor> rho_zz_old,
    ConstField2D<default_layout, unchecked_accessor> rho_zz_new,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invAreaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzw,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    ConnectivityView advCellsForEdge,
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>> nAdvCellsForEdge,
    ConstField2D<default_layout, unchecked_accessor> adv_coefs,
    ConstField2D<default_layout, unchecked_accessor> adv_coefs_3rd,
    ConstField2D<default_layout, unchecked_accessor> edgesOnCell_sign,
    real_type dt,
    real_type coef_3rd_order,
    index_type nScalars,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxAdvCells);

// ============================================================================
// Helper: 4th-order flux function (statement function from Fortran)
// ============================================================================

namespace {

/// @brief 4th-order centered flux function.
/// Fortran: flux4(q_im2, q_im1, q_i, q_ip1, ua) =
///            ua * (7*(q_i + q_im1) - (q_ip1 + q_im2)) / 12.0
inline real_type flux4(real_type q_im2, real_type q_im1,
                       real_type q_i, real_type q_ip1, real_type ua) {
    return ua * (7.0 * (q_i + q_im1) - (q_ip1 + q_im2)) / 12.0;
}

/// @brief 3rd-order upwind-biased flux function.
/// Fortran: flux3(q_im2, q_im1, q_i, q_ip1, ua, coef3) =
///            flux4(...) + coef3*|ua|*((q_ip1 - q_im2) - 3*(q_i - q_im1))/12.0
inline real_type flux3(real_type q_im2, real_type q_im1,
                       real_type q_i, real_type q_ip1,
                       real_type ua, real_type coef3) {
    return flux4(q_im2, q_im1, q_i, q_ip1, ua)
         + coef3 * std::abs(ua)
           * ((q_ip1 - q_im2) - 3.0 * (q_i - q_im1)) / 12.0;
}

} // anonymous namespace

// ============================================================================
// Implementation
// ============================================================================

template <typename Layout, ExecutionPolicy Policy>
void advance_scalars_mono(
    Policy /*policy*/,
    Field3D<Layout, unchecked_accessor> scalars,
    ConstField2D<Layout, unchecked_accessor> ruAvg,
    ConstField2D<Layout, unchecked_accessor> wwAvg,
    ConstField2D<Layout, unchecked_accessor> rho_zz_old,
    ConstField2D<Layout, unchecked_accessor> rho_zz_new,
    const MeshConnectivity& mesh,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> dvEdge,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> invAreaCell,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> rdzw,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzm,
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>> fzp,
    ConnectivityView advCellsForEdge,
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>> nAdvCellsForEdge,
    ConstField2D<Layout, unchecked_accessor> adv_coefs,
    ConstField2D<Layout, unchecked_accessor> adv_coefs_3rd,
    ConstField2D<Layout, unchecked_accessor> edgesOnCell_sign,
    real_type dt,
    real_type coef_3rd_order,
    index_type nScalars,
    index_type nCells,
    index_type nEdges,
    index_type nVertLevels,
    index_type maxAdvCells)
{
    // Zero-extent graceful handling (Requirement 4.7).
    if (nCells == 0 || nEdges == 0 || nVertLevels == 0 || nScalars == 0) {
        return;
    }

    // Small epsilon to avoid division by zero in FCT limiter.
    constexpr real_type eps = 1.0e-20;

    // Allocate workspace arrays.
    const auto nv = static_cast<std::size_t>(nVertLevels);
    const auto nc = static_cast<std::size_t>(nCells);
    const auto ne = static_cast<std::size_t>(nEdges);

    // Per-cell 2D workspace: scalar_old, scalar_new, s_max, s_min
    std::vector<real_type> scalar_old_vec(nv * nc, 0.0);
    std::vector<real_type> scalar_new_vec(nv * nc, 0.0);
    std::vector<real_type> s_max_vec(nv * nc, 0.0);
    std::vector<real_type> s_min_vec(nv * nc, 0.0);

    // Vertical anti-diffusive flux: (nVertLevels+1, nCells)
    std::vector<real_type> wdtn_vec((nv + 1) * nc, 0.0);

    // Scale factors: (nVertLevels, 2, nCells) — SCALE_IN=0, SCALE_OUT=1
    std::vector<real_type> scale_in_vec(nv * nc, 0.0);
    std::vector<real_type> scale_out_vec(nv * nc, 0.0);

    // Horizontal flux arrays: (nVertLevels, nEdges)
    std::vector<real_type> flux_arr_vec(nv * ne, 0.0);
    std::vector<real_type> flux_upwind_vec(nv * ne, 0.0);
    std::vector<real_type> flux_anti_vec(nv * ne, 0.0);

    // Helper lambdas for 2D indexing into flat workspace arrays.
    // Layout: (k, iEntity) — column-major style matching Fortran.
    auto idx2 = [nVertLevels](index_type k, index_type i) -> std::size_t {
        return static_cast<std::size_t>(k) +
               static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(i);
    };
    // For wdtn: (nVertLevels+1) levels per cell.
    auto idx2w = [nVertLevels](index_type k, index_type i) -> std::size_t {
        return static_cast<std::size_t>(k) +
               static_cast<std::size_t>(nVertLevels + 1) * static_cast<std::size_t>(i);
    };

    real_type* scalar_old = scalar_old_vec.data();
    real_type* scalar_new = scalar_new_vec.data();
    real_type* s_max = s_max_vec.data();
    real_type* s_min = s_min_vec.data();
    real_type* wdtn = wdtn_vec.data();
    real_type* scale_in = scale_in_vec.data();
    real_type* scale_out = scale_out_vec.data();
    real_type* flux_arr = flux_arr_vec.data();
    real_type* flux_upwind = flux_upwind_vec.data();
    real_type* flux_anti = flux_anti_vec.data();

    // ========================================================================
    // Loop over each scalar species independently.
    // ========================================================================

    for (index_type iScalar = 0; iScalar < nScalars; ++iScalar) {

        // ====================================================================
        // Phase 0: Copy scalar data into working arrays.
        // scalar_old = scalars (pre-advection values, used for upwind & bounds)
        // scalar_new = scalars (used for high-order flux computation)
        // ====================================================================
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                scalar_old[idx2(k, iCell)] = scalars[iScalar, k, iCell];
                scalar_new[idx2(k, iCell)] = scalars[iScalar, k, iCell];
            }
        }

        // ====================================================================
        // Phase 1: Vertical flux and local min/max bounds.
        //
        // Fortran reference: vertical flux using flux3 for interior levels,
        // linear interpolation at boundaries, and s_max/s_min from vertical
        // and horizontal neighbors.
        // ====================================================================
        for (index_type iCell = 0; iCell < nCells; ++iCell) {

            // Zero flux at top and bottom boundaries.
            // Fortran: wdtn(1,iCell) = 0; wdtn(nVertLevels+1,iCell) = 0
            wdtn[idx2w(0, iCell)] = 0.0;
            wdtn[idx2w(nVertLevels, iCell)] = 0.0;

            // k=0 (Fortran k=1): only vertical neighbors for bounds.
            // s_max/s_min from k=0 and k=1.
            s_max[idx2(0, iCell)] = std::max(scalar_old[idx2(0, iCell)],
                                             scalar_old[idx2(1, iCell)]);
            s_min[idx2(0, iCell)] = std::min(scalar_old[idx2(0, iCell)],
                                             scalar_old[idx2(1, iCell)]);

            // k=1 (Fortran k=2): linear interpolation for vertical flux.
            if (nVertLevels > 1) {
                wdtn[idx2w(1, iCell)] = wwAvg[1, iCell]
                    * (fzm[1] * scalar_new[idx2(1, iCell)]
                     + fzp[1] * scalar_new[idx2(0, iCell)]);
                s_max[idx2(1, iCell)] = std::max({scalar_old[idx2(0, iCell)],
                                                  scalar_old[idx2(1, iCell)],
                                                  scalar_old[idx2(2, iCell)]});
                s_min[idx2(1, iCell)] = std::min({scalar_old[idx2(0, iCell)],
                                                  scalar_old[idx2(1, iCell)],
                                                  scalar_old[idx2(2, iCell)]});
            }

            // Interior levels k=2..nVertLevels-2 (Fortran k=3..nVertLevels-1):
            // 3rd-order flux.
            for (index_type k = 2; k < nVertLevels - 1; ++k) {
                wdtn[idx2w(k, iCell)] = flux3(
                    scalar_new[idx2(k - 2, iCell)],
                    scalar_new[idx2(k - 1, iCell)],
                    scalar_new[idx2(k, iCell)],
                    scalar_new[idx2(k + 1, iCell)],
                    wwAvg[k, iCell],
                    coef_3rd_order);
                s_max[idx2(k, iCell)] = std::max({scalar_old[idx2(k - 1, iCell)],
                                                  scalar_old[idx2(k, iCell)],
                                                  scalar_old[idx2(k + 1, iCell)]});
                s_min[idx2(k, iCell)] = std::min({scalar_old[idx2(k - 1, iCell)],
                                                  scalar_old[idx2(k, iCell)],
                                                  scalar_old[idx2(k + 1, iCell)]});
            }

            // k=nVertLevels-1 (Fortran k=nVertLevels): linear interpolation.
            if (nVertLevels > 1) {
                index_type k = nVertLevels - 1;
                wdtn[idx2w(k, iCell)] = wwAvg[k, iCell]
                    * (fzm[k] * scalar_new[idx2(k, iCell)]
                     + fzp[k] * scalar_new[idx2(k - 1, iCell)]);
                s_max[idx2(k, iCell)] = std::max(scalar_old[idx2(k, iCell)],
                                                 scalar_old[idx2(k - 1, iCell)]);
                s_min[idx2(k, iCell)] = std::min(scalar_old[idx2(k, iCell)],
                                                 scalar_old[idx2(k - 1, iCell)]);
            }

            // Extend s_min/s_max using horizontal neighbors (cellsOnCell).
            const index_type nEdgesOnThisCell = mesh.nEdgesOnCell[iCell];
            for (index_type i = 0; i < nEdgesOnThisCell; ++i) {
                const index_type iNeighbor = mesh.cellsOnCell[iCell, i];
                if (iNeighbor == INVALID_INDEX) break;
                for (index_type k = 0; k < nVertLevels; ++k) {
                    s_max[idx2(k, iCell)] = std::max(s_max[idx2(k, iCell)],
                                                     scalar_old[idx2(k, iNeighbor)]);
                    s_min[idx2(k, iCell)] = std::min(s_min[idx2(k, iCell)],
                                                     scalar_old[idx2(k, iNeighbor)]);
                }
            }
        } // end Phase 1 loop over cells

        // ====================================================================
        // Phase 2: High-order horizontal flux computation.
        //
        // For each edge, compute flux using the advCellsForEdge stencil:
        //   flux = uhAvg * sum_j (adv_coefs(j) + sign(1,uhAvg)*adv_coefs_3rd(j))
        //                        * scalar_new(cell_j)
        // ====================================================================
        for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

            if (cell1 == INVALID_INDEX || cell2 == INVALID_INDEX) {
                for (index_type k = 0; k < nVertLevels; ++k) {
                    flux_arr[idx2(k, iEdge)] = 0.0;
                }
                continue;
            }

            // Only compute for edges touching owned cells.
            if (cell1 < nCells || cell2 < nCells) {
                for (index_type k = 0; k < nVertLevels; ++k) {
                    flux_arr[idx2(k, iEdge)] = 0.0;
                }

                const index_type nAdv = nAdvCellsForEdge[iEdge];
                for (index_type i = 0; i < nAdv; ++i) {
                    const index_type iAdv = advCellsForEdge[iEdge, i];
                    for (index_type k = 0; k < nVertLevels; ++k) {
                        real_type scalar_weight = ruAvg[k, iEdge]
                            * (adv_coefs[i, iEdge]
                             + std::copysign(1.0, ruAvg[k, iEdge])
                               * adv_coefs_3rd[i, iEdge]);
                        flux_arr[idx2(k, iEdge)] += scalar_weight
                            * scalar_new[idx2(k, iAdv)];
                    }
                }
            } else {
                for (index_type k = 0; k < nVertLevels; ++k) {
                    flux_arr[idx2(k, iEdge)] = 0.0;
                }
            }
        } // end Phase 2 loop over edges

        // ====================================================================
        // Phase 3: Upwind update + anti-diffusive flux decomposition.
        //
        // Vertical: upwind flux and incorporate into scalar_new (as rho*scalar).
        // Horizontal: upwind flux and anti-diffusive = high_order - upwind.
        // Track scale_in/scale_out for FCT limiter.
        // ====================================================================

        // --- Vertical upwind flux and partial update ---
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            // Initialize scalar_new as rho_old * scalar_old (mass form).
            for (index_type k = 0; k < nVertLevels; ++k) {
                scalar_new[idx2(k, iCell)] = scalar_old[idx2(k, iCell)]
                                           * rho_zz_old[k, iCell];
            }

            // Compute vertical upwind flux and apply to scalar_new.
            // flux_upwind_vert(k) = dt * (max(0,w(k))*s(k-1) + min(0,w(k))*s(k))
            // Then: scalar_new(k) -= flux_upwind_vert(k+1) * rdzw(k)
            //       scalar_new(k) += flux_upwind_vert(k) * rdzw(k)
            // And: wdtn(k) = dt*wdtn(k) - flux_upwind_vert(k)  (anti-diffusive)
            //
            // We use a temporary array for the vertical upwind flux.
            std::vector<real_type> flux_upwind_vert(nv + 1, 0.0);
            // Top and bottom: zero.
            flux_upwind_vert[0] = 0.0;
            flux_upwind_vert[nv] = 0.0;

            for (index_type k = 1; k < nVertLevels; ++k) {
                flux_upwind_vert[static_cast<std::size_t>(k)] = dt
                    * (std::max(0.0, wwAvg[k, iCell]) * scalar_old[idx2(k - 1, iCell)]
                     + std::min(0.0, wwAvg[k, iCell]) * scalar_old[idx2(k, iCell)]);
            }

            // Apply vertical upwind flux divergence to scalar_new.
            for (index_type k = 0; k < nVertLevels; ++k) {
                scalar_new[idx2(k, iCell)] -=
                    flux_upwind_vert[static_cast<std::size_t>(k + 1)] * rdzw[k];
                scalar_new[idx2(k, iCell)] +=
                    flux_upwind_vert[static_cast<std::size_t>(k)] * rdzw[k];
            }

            // Compute vertical anti-diffusive flux:
            // wdtn(k) = dt*wdtn(k) - flux_upwind_vert(k)
            for (index_type k = 1; k < nVertLevels; ++k) {
                wdtn[idx2w(k, iCell)] = dt * wdtn[idx2w(k, iCell)]
                    - flux_upwind_vert[static_cast<std::size_t>(k)];
            }
            // Boundaries remain zero.
            wdtn[idx2w(0, iCell)] = 0.0;
            wdtn[idx2w(nVertLevels, iCell)] = 0.0;

            // Compute scale_in and scale_out from vertical anti-diffusive flux.
            // scale_in  = -rdzw(k) * (min(0, wdtn(k+1)) - max(0, wdtn(k)))
            // scale_out = -rdzw(k) * (max(0, wdtn(k+1)) - min(0, wdtn(k)))
            for (index_type k = 0; k < nVertLevels; ++k) {
                scale_in[idx2(k, iCell)] = -rdzw[k]
                    * (std::min(0.0, wdtn[idx2w(k + 1, iCell)])
                     - std::max(0.0, wdtn[idx2w(k, iCell)]));
                scale_out[idx2(k, iCell)] = -rdzw[k]
                    * (std::max(0.0, wdtn[idx2w(k + 1, iCell)])
                     - std::min(0.0, wdtn[idx2w(k, iCell)]));
            }
        } // end vertical upwind loop over cells

        // --- Horizontal upwind flux computation ---
        for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

            if (cell1 == INVALID_INDEX || cell2 == INVALID_INDEX) {
                for (index_type k = 0; k < nVertLevels; ++k) {
                    flux_upwind[idx2(k, iEdge)] = 0.0;
                    flux_anti[idx2(k, iEdge)] = 0.0;
                }
                continue;
            }

            for (index_type k = 0; k < nVertLevels; ++k) {
                // Upwind horizontal flux (already multiplied by dvEdge and dt).
                flux_upwind[idx2(k, iEdge)] = dvEdge[iEdge] * dt
                    * (std::max(0.0, ruAvg[k, iEdge]) * scalar_old[idx2(k, cell1)]
                     + std::min(0.0, ruAvg[k, iEdge]) * scalar_old[idx2(k, cell2)]);

                // Anti-diffusive flux = dt * high_order - upwind.
                flux_anti[idx2(k, iEdge)] = dt * flux_arr[idx2(k, iEdge)]
                    - flux_upwind[idx2(k, iEdge)];
            }
        }

        // --- Apply horizontal upwind flux and accumulate scale factors ---
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            const index_type nEdgesOnThisCell = mesh.nEdgesOnCell[iCell];
            for (index_type i = 0; i < nEdgesOnThisCell; ++i) {
                const index_type iEdge = mesh.edgesOnCell[iCell, i];
                if (iEdge == INVALID_INDEX) break;

                for (index_type k = 0; k < nVertLevels; ++k) {
                    // Apply upwind flux divergence.
                    scalar_new[idx2(k, iCell)] -= edgesOnCell_sign[i, iCell]
                        * flux_upwind[idx2(k, iEdge)] * invAreaCell[iCell];

                    // Accumulate anti-diffusive flux contributions for limiter.
                    // Outgoing: positive contribution leaving the cell.
                    scale_out[idx2(k, iCell)] -= std::max(0.0,
                        edgesOnCell_sign[i, iCell] * flux_anti[idx2(k, iEdge)])
                        * invAreaCell[iCell];
                    // Incoming: negative contribution entering the cell.
                    scale_in[idx2(k, iCell)] -= std::min(0.0,
                        edgesOnCell_sign[i, iCell] * flux_anti[idx2(k, iEdge)])
                        * invAreaCell[iCell];
                }
            }
        }

        // ====================================================================
        // Phase 4: FCT limiter (Zalesak).
        //
        // Compute scale factors that limit anti-diffusive fluxes to maintain
        // monotonicity: the updated scalar must remain within [s_min, s_max].
        //
        // scale_in  = clamp((s_max * rho_new - scalar_new_upwind) / flux_in, 0, 1)
        // scale_out = clamp((s_min * rho_new - scalar_new_upwind) / flux_out, 0, 1)
        // ====================================================================
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type k = 0; k < nVertLevels; ++k) {
                real_type scale_factor;

                // Incoming limit: prevent exceeding s_max.
                scale_factor = (s_max[idx2(k, iCell)] * rho_zz_new[k, iCell]
                              - scalar_new[idx2(k, iCell)])
                             / (scale_in[idx2(k, iCell)] + eps);
                scale_in[idx2(k, iCell)] = std::min(1.0, std::max(0.0, scale_factor));

                // Outgoing limit: prevent going below s_min.
                scale_factor = (s_min[idx2(k, iCell)] * rho_zz_new[k, iCell]
                              - scalar_new[idx2(k, iCell)])
                             / (scale_out[idx2(k, iCell)] - eps);
                scale_out[idx2(k, iCell)] = std::min(1.0, std::max(0.0, scale_factor));
            }
        }

        // ====================================================================
        // Phase 5: Rescale anti-diffusive fluxes using FCT scale factors.
        //
        // For each edge, the rescaled flux uses the minimum of the sender's
        // outgoing scale factor and the receiver's incoming scale factor.
        // ====================================================================

        // --- Rescale horizontal anti-diffusive fluxes ---
        for (index_type iEdge = 0; iEdge < nEdges; ++iEdge) {
            const index_type cell1 = mesh.cellsOnEdge[iEdge, 0];
            const index_type cell2 = mesh.cellsOnEdge[iEdge, 1];

            if (cell1 == INVALID_INDEX || cell2 == INVALID_INDEX) {
                continue;
            }

            if (cell1 < nCells || cell2 < nCells) {
                for (index_type k = 0; k < nVertLevels; ++k) {
                    real_type f = flux_anti[idx2(k, iEdge)];
                    // Positive flux: flows from cell1 to cell2.
                    // Negative flux: flows from cell2 to cell1.
                    f = std::max(0.0, f)
                        * std::min(scale_out[idx2(k, cell1)],
                                   scale_in[idx2(k, cell2)])
                      + std::min(0.0, f)
                        * std::min(scale_in[idx2(k, cell1)],
                                   scale_out[idx2(k, cell2)]);
                    flux_anti[idx2(k, iEdge)] = f;
                }
            }
        }

        // --- Rescale vertical anti-diffusive fluxes ---
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            for (index_type k = 1; k < nVertLevels; ++k) {
                real_type f = wdtn[idx2w(k, iCell)];
                // Positive wdtn at level k: flux from cell k-1 to cell k.
                // Negative wdtn at level k: flux from cell k to cell k-1.
                f = std::max(0.0, f)
                    * std::min(scale_out[idx2(k - 1, iCell)],
                               scale_in[idx2(k, iCell)])
                  + std::min(0.0, f)
                    * std::min(scale_out[idx2(k, iCell)],
                               scale_in[idx2(k - 1, iCell)]);
                wdtn[idx2w(k, iCell)] = f;
            }
        }

        // ====================================================================
        // Phase 6: Final update.
        //
        // Apply the rescaled anti-diffusive fluxes (horizontal + vertical)
        // to scalar_new, then divide by rho_zz_new to recover mixing ratio.
        // Clamp negative values to zero (positive-definite).
        // ====================================================================
        for (index_type iCell = 0; iCell < nCells; ++iCell) {
            // Apply rescaled horizontal anti-diffusive flux.
            const index_type nEdgesOnThisCell = mesh.nEdgesOnCell[iCell];
            for (index_type i = 0; i < nEdgesOnThisCell; ++i) {
                const index_type iEdge = mesh.edgesOnCell[iCell, i];
                if (iEdge == INVALID_INDEX) break;

                for (index_type k = 0; k < nVertLevels; ++k) {
                    scalar_new[idx2(k, iCell)] -= edgesOnCell_sign[i, iCell]
                        * flux_anti[idx2(k, iEdge)] * invAreaCell[iCell];
                }
            }

            // Apply rescaled vertical anti-diffusive flux divergence and
            // divide by rho_zz_new.
            for (index_type k = 0; k < nVertLevels; ++k) {
                scalar_new[idx2(k, iCell)] += -rdzw[k]
                    * (wdtn[idx2w(k + 1, iCell)] - wdtn[idx2w(k, iCell)]);
                scalar_new[idx2(k, iCell)] /= rho_zz_new[k, iCell];
            }

            // Positive-definite clamp (Requirement 5.5).
            for (index_type k = 0; k < nVertLevels; ++k) {
                scalars[iScalar, k, iCell] = std::max(0.0, scalar_new[idx2(k, iCell)]);
            }
        }

    } // end loop over scalars
}

} // namespace mpas::dycore::kernels
