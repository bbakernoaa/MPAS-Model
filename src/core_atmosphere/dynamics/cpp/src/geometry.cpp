/// @file geometry.cpp
/// @brief Implementation of MeshGeometry::populate() and validate_sizes().

#include <mpas_dycore/geometry.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace mpas::dycore {

void MeshGeometry::populate(
    const real_type* areaCell_ptr,
    const real_type* invAreaCell_ptr,
    const real_type* dvEdge_ptr,
    const real_type* dcEdge_ptr,
    const real_type* invDcEdge_ptr,
    const real_type* rdzw_ptr,
    const real_type* rdzu_ptr,
    const real_type* fzm_ptr,
    const real_type* fzp_ptr,
    const real_type* etp_ptr,
    const real_type* etm_ptr,
    const real_type* ewp_ptr,
    const real_type* ewm_ptr,
    const real_type* zz_ptr,
    const real_type* rb_ptr,
    const real_type* rtb_ptr,
    const real_type* pb_ptr,
    const real_type* edgesOnCell_sign_ptr,
    const real_type* specZoneMaskEdge_ptr,
    const real_type* specZoneMaskCell_ptr,
    const real_type* weightsOnEdge_ptr,
    const index_type* nEdgesOnEdge_ptr,
    const index_type* edgesOnEdge_ptr,
    const index_type* advCellsForEdge_ptr,
    const index_type* nAdvCellsForEdge_ptr,
    const real_type* adv_coefs_ptr,
    const real_type* adv_coefs_3rd_ptr,
    const real_type* fVertex_ptr,
    const real_type* areaTriangle_ptr,
    const real_type* zb_cell_ptr,
    const real_type* zb3_cell_ptr,
    const real_type* rho_base_ptr,
    const real_type* rtheta_base_ptr,
    const real_type* exner_base_ptr,
    const index_type* nEdgesOnCell_ptr,
    real_type cf1_in,
    real_type cf2_in,
    real_type cf3_in,
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels,
    index_type maxEdges,
    index_type maxEdges2_in,
    index_type maxAdvCells_in)
{
    // --- Validate dimension parameters are positive ---
    if (nCells <= 0) {
        throw std::invalid_argument("MeshGeometry::populate: nCells must be > 0, got " +
                                    std::to_string(nCells));
    }
    if (nEdges <= 0) {
        throw std::invalid_argument("MeshGeometry::populate: nEdges must be > 0, got " +
                                    std::to_string(nEdges));
    }
    if (nVertices <= 0) {
        throw std::invalid_argument("MeshGeometry::populate: nVertices must be > 0, got " +
                                    std::to_string(nVertices));
    }
    if (nVertLevels <= 0) {
        throw std::invalid_argument("MeshGeometry::populate: nVertLevels must be > 0, got " +
                                    std::to_string(nVertLevels));
    }
    if (maxEdges <= 0) {
        throw std::invalid_argument("MeshGeometry::populate: maxEdges must be > 0, got " +
                                    std::to_string(maxEdges));
    }
    if (maxEdges2_in <= 0) {
        throw std::invalid_argument("MeshGeometry::populate: maxEdges2_in must be > 0, got " +
                                    std::to_string(maxEdges2_in));
    }
    if (maxAdvCells_in <= 0) {
        throw std::invalid_argument("MeshGeometry::populate: maxAdvCells_in must be > 0, got " +
                                    std::to_string(maxAdvCells_in));
    }

    // --- Validate all pointers are non-null ---
    if (!areaCell_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: areaCell_ptr is null");
    }
    if (!invAreaCell_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: invAreaCell_ptr is null");
    }
    if (!dvEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: dvEdge_ptr is null");
    }
    if (!dcEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: dcEdge_ptr is null");
    }
    if (!invDcEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: invDcEdge_ptr is null");
    }
    if (!rdzw_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: rdzw_ptr is null");
    }
    if (!rdzu_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: rdzu_ptr is null");
    }
    if (!fzm_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: fzm_ptr is null");
    }
    if (!fzp_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: fzp_ptr is null");
    }
    if (!etp_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: etp_ptr is null");
    }
    if (!etm_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: etm_ptr is null");
    }
    if (!ewp_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: ewp_ptr is null");
    }
    if (!ewm_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: ewm_ptr is null");
    }
    if (!zz_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: zz_ptr is null");
    }
    if (!rb_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: rb_ptr is null");
    }
    if (!rtb_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: rtb_ptr is null");
    }
    if (!pb_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: pb_ptr is null");
    }
    if (!edgesOnCell_sign_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: edgesOnCell_sign_ptr is null");
    }
    if (!specZoneMaskEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: specZoneMaskEdge_ptr is null");
    }
    if (!specZoneMaskCell_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: specZoneMaskCell_ptr is null");
    }
    if (!weightsOnEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: weightsOnEdge_ptr is null");
    }
    if (!nEdgesOnEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: nEdgesOnEdge_ptr is null");
    }
    if (!edgesOnEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: edgesOnEdge_ptr is null");
    }
    if (!advCellsForEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: advCellsForEdge_ptr is null");
    }
    if (!nAdvCellsForEdge_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: nAdvCellsForEdge_ptr is null");
    }
    if (!adv_coefs_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: adv_coefs_ptr is null");
    }
    if (!adv_coefs_3rd_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: adv_coefs_3rd_ptr is null");
    }
    if (!fVertex_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: fVertex_ptr is null");
    }
    if (!areaTriangle_ptr) {
        throw std::invalid_argument("MeshGeometry::populate: areaTriangle_ptr is null");
    }

    // --- Store dimension values ---
    nCells_ = nCells;
    nEdges_ = nEdges;
    nVertices_ = nVertices;
    nVertLevels_ = nVertLevels;
    maxEdges_ = maxEdges;
    maxEdges2 = maxEdges2_in;
    maxAdvCells = maxAdvCells_in;

    // --- Copy arrays into owned storage ---

    // Cell geometry (nCells each)
    areaCell.assign(areaCell_ptr, areaCell_ptr + nCells);
    invAreaCell.assign(invAreaCell_ptr, invAreaCell_ptr + nCells);

    // Edge geometry (nEdges each)
    dvEdge.assign(dvEdge_ptr, dvEdge_ptr + nEdges);
    dcEdge.assign(dcEdge_ptr, dcEdge_ptr + nEdges);
    invDcEdge.assign(invDcEdge_ptr, invDcEdge_ptr + nEdges);

    // Vertical metrics (nVertLevels each)
    rdzw.assign(rdzw_ptr, rdzw_ptr + nVertLevels);
    rdzu.assign(rdzu_ptr, rdzu_ptr + nVertLevels);
    fzm.assign(fzm_ptr, fzm_ptr + nVertLevels);
    fzp.assign(fzp_ptr, fzp_ptr + nVertLevels);
    etp.assign(etp_ptr, etp_ptr + nVertLevels);
    etm.assign(etm_ptr, etm_ptr + nVertLevels);

    // Vertical metrics (nVertLevels+1 each)
    const auto nVertLevelsP1 = nVertLevels + 1;
    ewp.assign(ewp_ptr, ewp_ptr + nVertLevelsP1);
    ewm.assign(ewm_ptr, ewm_ptr + nVertLevelsP1);

    // Terrain metric (nVertLevels × nCells)
    const auto nVL_x_nC = static_cast<std::size_t>(nVertLevels) * static_cast<std::size_t>(nCells);
    zz.assign(zz_ptr, zz_ptr + nVL_x_nC);

    // Base-state profiles (nVertLevels × nCells each)
    rb.assign(rb_ptr, rb_ptr + nVL_x_nC);
    rtb.assign(rtb_ptr, rtb_ptr + nVL_x_nC);
    pb.assign(pb_ptr, pb_ptr + nVL_x_nC);

    // Edge orientation signs (maxEdges × nCells)
    const auto maxE_x_nC = static_cast<std::size_t>(maxEdges) * static_cast<std::size_t>(nCells);
    edgesOnCell_sign.assign(edgesOnCell_sign_ptr, edgesOnCell_sign_ptr + maxE_x_nC);

    // Specified zone masks
    specZoneMaskEdge.assign(specZoneMaskEdge_ptr, specZoneMaskEdge_ptr + nEdges);
    specZoneMaskCell.assign(specZoneMaskCell_ptr, specZoneMaskCell_ptr + nCells);

    // Edge reconstruction data (TRiSK)
    const auto nE_x_mE2 = static_cast<std::size_t>(nEdges) * static_cast<std::size_t>(maxEdges2_in);
    weightsOnEdge.assign(weightsOnEdge_ptr, weightsOnEdge_ptr + nE_x_mE2);
    nEdgesOnEdge.assign(nEdgesOnEdge_ptr, nEdgesOnEdge_ptr + nEdges);
    edgesOnEdge_storage.assign(edgesOnEdge_ptr, edgesOnEdge_ptr + nE_x_mE2);

    // Advection stencils
    const auto nE_x_mAC = static_cast<std::size_t>(nEdges) * static_cast<std::size_t>(maxAdvCells_in);
    advCellsForEdge_storage.assign(advCellsForEdge_ptr, advCellsForEdge_ptr + nE_x_mAC);
    nAdvCellsForEdge.assign(nAdvCellsForEdge_ptr, nAdvCellsForEdge_ptr + nEdges);

    // Advection coefficients (maxAdvCells × nEdges each)
    const auto mAC_x_nE = static_cast<std::size_t>(maxAdvCells_in) * static_cast<std::size_t>(nEdges);
    adv_coefs.assign(adv_coefs_ptr, adv_coefs_ptr + mAC_x_nE);
    adv_coefs_3rd.assign(adv_coefs_3rd_ptr, adv_coefs_3rd_ptr + mAC_x_nE);

    // Vertex geometry (nVertices each)
    fVertex.assign(fVertex_ptr, fVertex_ptr + nVertices);
    areaTriangle.assign(areaTriangle_ptr, areaTriangle_ptr + nVertices);

    // --- Terrain correction arrays (optional — nullptr means flat terrain) ---
    const auto nVLp1_x_mE_x_nC = static_cast<std::size_t>(nVertLevels + 1)
                                * static_cast<std::size_t>(maxEdges)
                                * static_cast<std::size_t>(nCells);
    if (zb_cell_ptr) {
        zb_cell.assign(zb_cell_ptr, zb_cell_ptr + nVLp1_x_mE_x_nC);
    } else {
        zb_cell.assign(nVLp1_x_mE_x_nC, real_type{0});
    }
    if (zb3_cell_ptr) {
        zb3_cell.assign(zb3_cell_ptr, zb3_cell_ptr + nVLp1_x_mE_x_nC);
    } else {
        zb3_cell.assign(nVLp1_x_mE_x_nC, real_type{0});
    }

    // --- Base-state profiles for perturbation formulation (optional) ---
    if (rho_base_ptr) {
        rho_base.assign(rho_base_ptr, rho_base_ptr + nVL_x_nC);
    } else {
        rho_base.assign(nVL_x_nC, real_type{0});
    }
    if (rtheta_base_ptr) {
        rtheta_base.assign(rtheta_base_ptr, rtheta_base_ptr + nVL_x_nC);
    } else {
        rtheta_base.assign(nVL_x_nC, real_type{0});
    }
    if (exner_base_ptr) {
        exner_base.assign(exner_base_ptr, exner_base_ptr + nVL_x_nC);
    } else {
        exner_base.assign(nVL_x_nC, real_type{0});
    }

    // --- Per-cell edge count (optional) ---
    if (nEdgesOnCell_ptr) {
        nEdgesOnCell.assign(nEdgesOnCell_ptr, nEdgesOnCell_ptr + nCells);
    } else {
        nEdgesOnCell.assign(static_cast<std::size_t>(nCells), index_type{0});
        // Warn if terrain data was provided but nEdgesOnCell is null
        if (zb_cell_ptr) {
            std::fprintf(stderr,
                "WARNING: MeshGeometry::populate: zb_cell_ptr is non-null but "
                "nEdgesOnCell_ptr is null. Terrain correction will have no effect "
                "(0 edges per cell).\n");
        }
    }

    // --- Vertical extrapolation coefficients ---
    cf1 = cf1_in;
    cf2 = cf2_in;
    cf3 = cf3_in;

    // --- Mark as populated ---
    populated_ = true;
}

bool MeshGeometry::validate_sizes(
    index_type nCells,
    index_type nEdges,
    index_type nVertices,
    index_type nVertLevels,
    index_type maxEdges2_in,
    index_type maxAdvCells_in) const
{
    const auto nc = static_cast<std::size_t>(nCells);
    const auto ne = static_cast<std::size_t>(nEdges);
    const auto nv = static_cast<std::size_t>(nVertices);
    const auto nvl = static_cast<std::size_t>(nVertLevels);
    const auto nvlp1 = static_cast<std::size_t>(nVertLevels + 1);
    const auto me2 = static_cast<std::size_t>(maxEdges2_in);
    const auto mac = static_cast<std::size_t>(maxAdvCells_in);

    // Cell geometry
    if (areaCell.size() != nc) return false;
    if (invAreaCell.size() != nc) return false;

    // Edge geometry
    if (dvEdge.size() != ne) return false;
    if (dcEdge.size() != ne) return false;
    if (invDcEdge.size() != ne) return false;

    // Vertical metrics (nVertLevels)
    if (rdzw.size() != nvl) return false;
    if (rdzu.size() != nvl) return false;
    if (fzm.size() != nvl) return false;
    if (fzp.size() != nvl) return false;
    if (etp.size() != nvl) return false;
    if (etm.size() != nvl) return false;

    // Vertical metrics (nVertLevels+1)
    if (ewp.size() != nvlp1) return false;
    if (ewm.size() != nvlp1) return false;

    // Terrain metric and base-state profiles (nVertLevels × nCells)
    if (zz.size() != nvl * nc) return false;
    if (rb.size() != nvl * nc) return false;
    if (rtb.size() != nvl * nc) return false;
    if (pb.size() != nvl * nc) return false;

    // Edge orientation signs -- note: uses the edgesOnCell_sign size
    // The populate() call uses maxEdges × nCells, but validate_sizes doesn't
    // receive maxEdges. We check it's a multiple of nCells and non-empty.
    if (edgesOnCell_sign.size() % nc != 0) return false;
    if (edgesOnCell_sign.empty()) return false;

    // Specified zone masks
    if (specZoneMaskEdge.size() != ne) return false;
    if (specZoneMaskCell.size() != nc) return false;

    // Edge reconstruction data (TRiSK)
    if (weightsOnEdge.size() != ne * me2) return false;
    if (nEdgesOnEdge.size() != ne) return false;
    if (edgesOnEdge_storage.size() != ne * me2) return false;

    // Advection stencils
    if (advCellsForEdge_storage.size() != ne * mac) return false;
    if (nAdvCellsForEdge.size() != ne) return false;
    if (adv_coefs.size() != mac * ne) return false;
    if (adv_coefs_3rd.size() != mac * ne) return false;

    // Vertex geometry
    if (fVertex.size() != nv) return false;
    if (areaTriangle.size() != nv) return false;

    // Terrain correction arrays (nVertLevels+1 × maxEdges × nCells)
    // Only validate if they have been populated (non-empty)
    if (!zb_cell.empty()) {
        // We use edgesOnCell_sign.size() / nc to infer maxEdges
        const auto me = edgesOnCell_sign.size() / nc;
        const auto expected_zb = nvlp1 * me * nc;
        if (zb_cell.size() != expected_zb) return false;
        if (zb3_cell.size() != expected_zb) return false;
    }

    // Base-state profiles for perturbation formulation (nVertLevels × nCells)
    if (!rho_base.empty() && rho_base.size() != nvl * nc) return false;
    if (!rtheta_base.empty() && rtheta_base.size() != nvl * nc) return false;
    if (!exner_base.empty() && exner_base.size() != nvl * nc) return false;

    // Per-cell edge count (nCells)
    if (!nEdgesOnCell.empty() && nEdgesOnCell.size() != nc) return false;

    return true;
}

} // namespace mpas::dycore
