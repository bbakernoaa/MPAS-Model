#pragma once

/// @file geometry.hpp
/// @brief Mesh geometry data for the MPAS dynamical core.
///
/// Defines the MeshGeometry struct that holds all geometric, metric,
/// and stencil data passed from Fortran via the extended C API. Storage
/// is owned (std::vector) and populated once during initialization by
/// copying from raw Fortran array pointers.

#include <mpas_dycore/types.hpp>

#include <vector>

namespace mpas::dycore {

/// @brief Geometry data passed from Fortran via the extended C API.
///
/// Contains cell geometry, edge geometry, vertical metrics, terrain metric,
/// base-state profiles, edge orientation signs, specified zone masks, edge
/// reconstruction data (TRiSK), advection stencils for scalar transport,
/// and vertex geometry. All arrays are owned and populated via populate().
struct MeshGeometry {

    // --- Cell geometry ---
    std::vector<real_type> areaCell;              ///< (nCells)
    std::vector<real_type> invAreaCell;           ///< (nCells)

    // --- Edge geometry ---
    std::vector<real_type> dvEdge;                ///< (nEdges)
    std::vector<real_type> dcEdge;                ///< (nEdges)
    std::vector<real_type> invDcEdge;             ///< (nEdges)

    // --- Vertical metrics ---
    std::vector<real_type> rdzw;                  ///< (nVertLevels)
    std::vector<real_type> rdzu;                  ///< (nVertLevels)
    std::vector<real_type> fzm;                   ///< (nVertLevels)
    std::vector<real_type> fzp;                   ///< (nVertLevels)
    std::vector<real_type> etp;                   ///< (nVertLevels)
    std::vector<real_type> etm;                   ///< (nVertLevels)
    std::vector<real_type> ewp;                   ///< (nVertLevels+1)
    std::vector<real_type> ewm;                   ///< (nVertLevels+1)

    // --- Terrain metric ---
    std::vector<real_type> zz;                    ///< (nVertLevels × nCells) dz/dzeta

    // --- Base-state profiles ---
    std::vector<real_type> rb;                    ///< (nVertLevels × nCells) base-state density
    std::vector<real_type> rtb;                   ///< (nVertLevels × nCells) base-state rho*theta
    std::vector<real_type> pb;                    ///< (nVertLevels × nCells) base-state pressure

    // --- Edge orientation signs ---
    std::vector<real_type> edgesOnCell_sign;      ///< (maxEdges × nCells)

    // --- Specified zone masks ---
    std::vector<real_type> specZoneMaskEdge;      ///< (nEdges)
    std::vector<real_type> specZoneMaskCell;      ///< (nCells)

    // --- Edge reconstruction data (TRiSK) ---
    std::vector<real_type> weightsOnEdge;         ///< (nEdges × maxEdges2)
    std::vector<index_type> nEdgesOnEdge;         ///< (nEdges)
    std::vector<index_type> edgesOnEdge_storage;  ///< (nEdges × maxEdges2)

    // --- Advection stencils for scalar transport ---
    std::vector<index_type> advCellsForEdge_storage; ///< (nEdges × maxAdvCells)
    std::vector<index_type> nAdvCellsForEdge;        ///< (nEdges)
    std::vector<real_type> adv_coefs;                ///< (maxAdvCells × nEdges)
    std::vector<real_type> adv_coefs_3rd;            ///< (maxAdvCells × nEdges)

    // --- Terrain correction for w recovery ---
    std::vector<real_type> zb_cell;               ///< (nVertLevels+1, maxEdges, nCells) terrain slope metric
    std::vector<real_type> zb3_cell;              ///< (nVertLevels+1, maxEdges, nCells) 3rd-order terrain correction

    // --- Base-state profiles for perturbation formulation ---
    std::vector<real_type> rho_base;              ///< (nVertLevels, nCells) base-state density
    std::vector<real_type> rtheta_base;           ///< (nVertLevels, nCells) base-state rho*theta
    std::vector<real_type> exner_base;            ///< (nVertLevels, nCells) base-state Exner function

    // --- Vertical extrapolation coefficients (for k=1 in w terrain correction) ---
    real_type cf1 = 0.0;                          ///< extrapolation weight for level 1
    real_type cf2 = 0.0;                          ///< extrapolation weight for level 2
    real_type cf3 = 0.0;                          ///< extrapolation weight for level 3

    // --- Per-cell edge count (for terrain correction loop) ---
    std::vector<index_type> nEdgesOnCell;         ///< (nCells) number of edges per cell

    // --- Vertex geometry ---
    std::vector<real_type> fVertex;               ///< (nVertices) Coriolis parameter
    std::vector<real_type> areaTriangle;          ///< (nVertices)

    // --- Vertex-to-edge connectivity ---
    std::vector<index_type> edgesOnVertex_storage; ///< (nVertices × vertexDegree)
    index_type vertexDegree = 3;                   ///< Number of edges per vertex (typically 3 for Voronoi)

    // --- Dimension members ---
    index_type maxEdges_ = 0;                     ///< Max edges per cell (for edgesOnCell_sign, zb_cell, zb3_cell)
    index_type maxEdges2 = 0;                     ///< Max edges used in TRiSK reconstruction
    index_type maxAdvCells = 0;                   ///< Max advection stencil cells per edge

    // --- Stored mesh dimensions for validation ---
    index_type nCells_ = 0;                       ///< Number of cells in local partition.
    index_type nEdges_ = 0;                       ///< Number of edges in local partition.
    index_type nVertices_ = 0;                    ///< Number of vertices in local partition.
    index_type nVertLevels_ = 0;                  ///< Number of vertical levels.

    /// @brief Whether geometry has been populated.
    bool populated_ = false;

    /// @brief Populate geometry from raw Fortran array pointers.
    ///
    /// Copies data from raw pointers into owned storage vectors. The caller
    /// must ensure that each pointer points to at least the documented number
    /// of elements (per the dimension metadata).
    ///
    /// @param areaCell_ptr         Pointer to cell areas (nCells elements).
    /// @param invAreaCell_ptr      Pointer to inverse cell areas (nCells elements).
    /// @param dvEdge_ptr           Pointer to dvEdge (nEdges elements).
    /// @param dcEdge_ptr           Pointer to dcEdge (nEdges elements).
    /// @param invDcEdge_ptr        Pointer to inverse dcEdge (nEdges elements).
    /// @param rdzw_ptr             Pointer to rdzw (nVertLevels elements).
    /// @param rdzu_ptr             Pointer to rdzu (nVertLevels elements).
    /// @param fzm_ptr              Pointer to fzm (nVertLevels elements).
    /// @param fzp_ptr              Pointer to fzp (nVertLevels elements).
    /// @param etp_ptr              Pointer to etp (nVertLevels elements).
    /// @param etm_ptr              Pointer to etm (nVertLevels elements).
    /// @param ewp_ptr              Pointer to ewp (nVertLevels+1 elements).
    /// @param ewm_ptr              Pointer to ewm (nVertLevels+1 elements).
    /// @param zz_ptr               Pointer to zz (nVertLevels × nCells elements).
    /// @param rb_ptr               Pointer to rb (nVertLevels × nCells elements).
    /// @param rtb_ptr              Pointer to rtb (nVertLevels × nCells elements).
    /// @param pb_ptr               Pointer to pb (nVertLevels × nCells elements).
    /// @param edgesOnCell_sign_ptr Pointer to edge signs (maxEdges × nCells elements).
    /// @param specZoneMaskEdge_ptr Pointer to edge zone mask (nEdges elements).
    /// @param specZoneMaskCell_ptr Pointer to cell zone mask (nCells elements).
    /// @param weightsOnEdge_ptr    Pointer to TRiSK weights (nEdges × maxEdges2 elements).
    /// @param nEdgesOnEdge_ptr     Pointer to edge neighbor counts (nEdges elements).
    /// @param edgesOnEdge_ptr      Pointer to edge-on-edge connectivity (nEdges × maxEdges2 elements).
    /// @param advCellsForEdge_ptr  Pointer to advection cell stencils (nEdges × maxAdvCells elements).
    /// @param nAdvCellsForEdge_ptr Pointer to advection cell counts (nEdges elements).
    /// @param adv_coefs_ptr        Pointer to advection coefficients (maxAdvCells × nEdges elements).
    /// @param adv_coefs_3rd_ptr    Pointer to 3rd-order advection coefficients (maxAdvCells × nEdges elements).
    /// @param fVertex_ptr          Pointer to Coriolis parameter (nVertices elements).
    /// @param areaTriangle_ptr     Pointer to triangle areas (nVertices elements).
    /// @param zb_cell_ptr          Pointer to terrain slope metric ((nVertLevels+1) × maxEdges × nCells elements), or nullptr for flat terrain.
    /// @param zb3_cell_ptr         Pointer to 3rd-order terrain correction ((nVertLevels+1) × maxEdges × nCells elements), or nullptr for flat terrain.
    /// @param rho_base_ptr         Pointer to base-state density (nVertLevels × nCells elements), or nullptr.
    /// @param rtheta_base_ptr      Pointer to base-state rho*theta (nVertLevels × nCells elements), or nullptr.
    /// @param exner_base_ptr       Pointer to base-state Exner function (nVertLevels × nCells elements), or nullptr.
    /// @param nEdgesOnCell_ptr     Pointer to per-cell edge counts (nCells elements), or nullptr.
    /// @param cf1_in               Vertical extrapolation coefficient 1.
    /// @param cf2_in               Vertical extrapolation coefficient 2.
    /// @param cf3_in               Vertical extrapolation coefficient 3.
    /// @param nCells               Number of cells.
    /// @param nEdges               Number of edges.
    /// @param nVertices            Number of vertices.
    /// @param nVertLevels          Number of vertical levels.
    /// @param maxEdges             Maximum edges per cell (for edgesOnCell_sign).
    /// @param maxEdges2_in         Maximum edges for TRiSK reconstruction.
    /// @param maxAdvCells_in       Maximum advection stencil cells per edge.
    void populate(
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
        index_type maxAdvCells_in);

    /// @brief Validate that all storage vector sizes are consistent with dimensions.
    ///
    /// Checks every vector's size against the expected product of dimensions.
    /// Returns true if all sizes are correct, false otherwise.
    ///
    /// @param nCells       Expected number of cells.
    /// @param nEdges       Expected number of edges.
    /// @param nVertices    Expected number of vertices.
    /// @param nVertLevels  Expected number of vertical levels.
    /// @param maxEdges2_in Expected maxEdges2 dimension.
    /// @param maxAdvCells_in Expected maxAdvCells dimension.
    /// @return true if all vector sizes match the expected dimensions.
    [[nodiscard]] bool validate_sizes(
        index_type nCells,
        index_type nEdges,
        index_type nVertices,
        index_type nVertLevels,
        index_type maxEdges2_in,
        index_type maxAdvCells_in) const;

    /// @brief Query whether geometry data has been populated.
    /// @return true if populate() has been called successfully.
    [[nodiscard]] bool populated() const noexcept { return populated_; }
};

} // namespace mpas::dycore
