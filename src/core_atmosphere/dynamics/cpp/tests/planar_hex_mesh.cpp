/// @file planar_hex_mesh.cpp
/// @brief Implementation of the planar periodic hexagonal mesh generator.

#include "planar_hex_mesh.hpp"

#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>

namespace mpas::dycore::testing {

namespace {

// Physical constants
constexpr double g       = 9.80616;    // gravitational acceleration (m/s^2)
constexpr double Rd      = 287.0;      // dry air gas constant (J/kg/K)
constexpr double cp      = 1003.5;     // specific heat at constant pressure (J/kg/K)
constexpr double T0      = 250.0;      // isothermal temperature (K)
constexpr double p_sfc   = 101325.0;   // surface pressure (Pa)
constexpr double p_ref   = 100000.0;   // reference pressure (Pa)
constexpr double H       = 10000.0;    // domain height (m)

} // anonymous namespace

PlanarHexMesh generate_planar_hex_mesh(int nx, int ny, int nVertLevels_in) {
    assert(nx >= 2 && "nx must be at least 2");
    assert(ny >= 2 && "ny must be at least 2");
    assert(nVertLevels_in >= 1 && "nVertLevels must be at least 1");

    PlanarHexMesh mesh;

    // ---- Dimensions ----
    const index_type nxI = static_cast<index_type>(nx);
    const index_type nyI = static_cast<index_type>(ny);

    mesh.nCells      = nxI * nyI;
    mesh.nEdges      = 3 * nxI * nyI;  // 3 edges per cell in a periodic hex mesh
    mesh.nVertices   = 2 * nxI * nyI;  // 2 vertices per cell in a periodic hex mesh
    mesh.maxEdges    = 6;
    mesh.nVertLevels = static_cast<index_type>(nVertLevels_in);
    mesh.maxEdges2   = 10;
    mesh.maxAdvCells = 10;

    const auto nC = mesh.nCells;
    const auto nE = mesh.nEdges;
    const auto nV = mesh.nVertices;
    const auto nVL = mesh.nVertLevels;
    const auto mE = mesh.maxEdges;
    const auto mE2 = mesh.maxEdges2;
    const auto mAC = mesh.maxAdvCells;

    // ---- Grid spacing ----
    // Hex cell: side length s, then distance between cell centers = s*sqrt(3)
    // Use dx = 10000 m (10 km) as the distance between adjacent cell centers
    const double dx = 10000.0;
    mesh.dx = dx;
    mesh.dz = H / static_cast<double>(nVertLevels_in);

    // Cell area for a regular hexagon with center-to-center distance dx:
    // side length s = dx / sqrt(3)
    // area = (3*sqrt(3)/2) * s^2 = (3*sqrt(3)/2) * (dx^2 / 3) = sqrt(3)/2 * dx^2
    const double cellArea = std::sqrt(3.0) / 2.0 * dx * dx;

    // dvEdge: distance between the two vertices of an edge
    // For a regular hexagon, edge length = s = dx / sqrt(3)
    const double side = dx / std::sqrt(3.0);
    const double dvEdge_val = side;

    // dcEdge: distance between the two cell centers sharing an edge = dx
    const double dcEdge_val = dx;

    // ========================================================================
    // Build connectivity arrays
    // ========================================================================
    // Cell indexing: cell(ix, iy) = iy * nx + ix  (row-major)
    // We use a "brick wall" hex layout:
    //   - Even rows: cells at positions (ix*dx, iy*dy)
    //   - Odd rows: cells shifted by dx/2 in x
    //
    // Each cell has 6 edges and 6 neighbors (periodic wrapping).
    // Edge numbering: 3 edges per cell, corresponding to the "east",
    //   "northeast", and "northwest" edges of each cell. The other 3 edges
    //   are owned by neighboring cells.

    // Helper: cell index with periodic wrap
    auto cellIdx = [&](int ix, int iy) -> index_type {
        int wx = ((ix % nx) + nx) % nx;
        int wy = ((iy % ny) + ny) % ny;
        return static_cast<index_type>(wy * nx + wx);
    };

    // Edge indexing: edge(iCell, localEdge) where localEdge in {0,1,2}
    //   edge 0: "east" edge
    //   edge 1: "northeast" edge
    //   edge 2: "northwest" edge
    auto edgeIdx = [&](index_type iCell, int localEdge) -> index_type {
        return iCell * 3 + static_cast<index_type>(localEdge);
    };

    // Vertex indexing: vertex(iCell, localVertex) where localVertex in {0,1}
    //   vertex 0: "north" vertex of cell
    //   vertex 1: "south" vertex of cell
    // Actually simpler: 2 vertices per cell in the dual sense
    auto vertIdx = [&](index_type iCell, int localVert) -> index_type {
        return iCell * 2 + static_cast<index_type>(localVert);
    };

    // Allocate connectivity
    mesh.cellsOnEdge.resize(static_cast<std::size_t>(nE) * 2, 0);
    mesh.verticesOnEdge.resize(static_cast<std::size_t>(nE) * 2, 0);
    mesh.edgesOnCell.resize(static_cast<std::size_t>(nC) * mE, -1);
    mesh.cellsOnCell.resize(static_cast<std::size_t>(nC) * mE, -1);
    mesh.nEdgesOnCell_arr.resize(static_cast<std::size_t>(nC), 6);

    // For each cell, determine its 6 neighbors (depends on row parity)
    // Hex grid "brick wall" neighbor offsets:
    //   Even row (iy % 2 == 0):
    //     E:  (ix+1, iy)    W:  (ix-1, iy)
    //     NE: (ix,   iy+1)  NW: (ix-1, iy+1)
    //     SE: (ix,   iy-1)  SW: (ix-1, iy-1)
    //   Odd row (iy % 2 == 1):
    //     E:  (ix+1, iy)    W:  (ix-1, iy)
    //     NE: (ix+1, iy+1)  NW: (ix,   iy+1)
    //     SE: (ix+1, iy-1)  SW: (ix,   iy-1)

    // We assign each cell 3 "owned" edges:
    //   local edge 0 = E edge    (shared with W neighbor's edge)
    //   local edge 1 = NE edge   (shared with SW neighbor)
    //   local edge 2 = NW edge   (shared with SE neighbor)

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            index_type iCell = cellIdx(ix, iy);
            bool evenRow = (iy % 2 == 0);

            // Compute 6 neighbor cell indices
            index_type neighbor_E, neighbor_W, neighbor_NE, neighbor_NW;
            index_type neighbor_SE, neighbor_SW;

            neighbor_E  = cellIdx(ix + 1, iy);
            neighbor_W  = cellIdx(ix - 1, iy);

            if (evenRow) {
                neighbor_NE = cellIdx(ix,     iy + 1);
                neighbor_NW = cellIdx(ix - 1, iy + 1);
                neighbor_SE = cellIdx(ix,     iy - 1);
                neighbor_SW = cellIdx(ix - 1, iy - 1);
            } else {
                neighbor_NE = cellIdx(ix + 1, iy + 1);
                neighbor_NW = cellIdx(ix,     iy + 1);
                neighbor_SE = cellIdx(ix + 1, iy - 1);
                neighbor_SW = cellIdx(ix,     iy - 1);
            }

            // ---- cellsOnEdge: for each owned edge, record the two cells ----
            // Edge 0 (E): between this cell and E neighbor
            index_type e0 = edgeIdx(iCell, 0);
            mesh.cellsOnEdge[static_cast<std::size_t>(e0) * 2 + 0] = iCell;
            mesh.cellsOnEdge[static_cast<std::size_t>(e0) * 2 + 1] = neighbor_E;

            // Edge 1 (NE): between this cell and NE neighbor
            index_type e1 = edgeIdx(iCell, 1);
            mesh.cellsOnEdge[static_cast<std::size_t>(e1) * 2 + 0] = iCell;
            mesh.cellsOnEdge[static_cast<std::size_t>(e1) * 2 + 1] = neighbor_NE;

            // Edge 2 (NW): between this cell and NW neighbor
            index_type e2 = edgeIdx(iCell, 2);
            mesh.cellsOnEdge[static_cast<std::size_t>(e2) * 2 + 0] = iCell;
            mesh.cellsOnEdge[static_cast<std::size_t>(e2) * 2 + 1] = neighbor_NW;

            // ---- verticesOnEdge ----
            // Each edge has 2 vertices. For our hex mesh:
            //   Edge 0 (E):  vertices are the "NE vertex" and "SE vertex" of this cell
            //   Edge 1 (NE): vertices are the "N vertex" and "NE vertex" of this cell
            //   Edge 2 (NW): vertices are the "NW vertex" and "N vertex" of this cell
            // We use: vertex 0 of cell = "N" vertex, vertex 1 of cell = "S" vertex
            // The NE vertex of cell is vertex 0 of neighbor_E
            // The SE vertex of cell is vertex 1 of neighbor_E
            // Actually to keep things simple and consistent:
            //   vertex 0 of cell = "upper-right" type vertex
            //   vertex 1 of cell = "lower-right" type vertex

            // For simplicity, assign vertices per edge consistently:
            mesh.verticesOnEdge[static_cast<std::size_t>(e0) * 2 + 0] = vertIdx(iCell, 0);
            mesh.verticesOnEdge[static_cast<std::size_t>(e0) * 2 + 1] = vertIdx(iCell, 1);

            mesh.verticesOnEdge[static_cast<std::size_t>(e1) * 2 + 0] = vertIdx(iCell, 0);
            mesh.verticesOnEdge[static_cast<std::size_t>(e1) * 2 + 1] = vertIdx(neighbor_NE, 1);

            mesh.verticesOnEdge[static_cast<std::size_t>(e2) * 2 + 0] = vertIdx(neighbor_NW, 0);
            mesh.verticesOnEdge[static_cast<std::size_t>(e2) * 2 + 1] = vertIdx(iCell, 0);

            // ---- edgesOnCell: 6 edges for each cell ----
            // The 6 edges of this cell are:
            //   slot 0: E edge  = owned edge 0 of this cell
            //   slot 1: NE edge = owned edge 1 of this cell
            //   slot 2: NW edge = owned edge 2 of this cell
            //   slot 3: W edge  = owned edge 0 of neighbor_W
            //   slot 4: SW edge = owned edge 1 of neighbor_SW
            //   slot 5: SE edge = owned edge 2 of neighbor_SE
            auto eocBase = static_cast<std::size_t>(iCell) * mE;
            mesh.edgesOnCell[eocBase + 0] = edgeIdx(iCell, 0);      // E
            mesh.edgesOnCell[eocBase + 1] = edgeIdx(iCell, 1);      // NE
            mesh.edgesOnCell[eocBase + 2] = edgeIdx(iCell, 2);      // NW
            mesh.edgesOnCell[eocBase + 3] = edgeIdx(neighbor_W, 0); // W
            mesh.edgesOnCell[eocBase + 4] = edgeIdx(neighbor_SW, 1);// SW
            mesh.edgesOnCell[eocBase + 5] = edgeIdx(neighbor_SE, 2);// SE

            // ---- cellsOnCell: 6 neighbors ----
            auto cocBase = static_cast<std::size_t>(iCell) * mE;
            mesh.cellsOnCell[cocBase + 0] = neighbor_E;
            mesh.cellsOnCell[cocBase + 1] = neighbor_NE;
            mesh.cellsOnCell[cocBase + 2] = neighbor_NW;
            mesh.cellsOnCell[cocBase + 3] = neighbor_W;
            mesh.cellsOnCell[cocBase + 4] = neighbor_SW;
            mesh.cellsOnCell[cocBase + 5] = neighbor_SE;
        }
    }

    // ========================================================================
    // Populate MeshGeometry
    // ========================================================================
    auto& geom = mesh.geometry;

    // ---- Cell geometry ----
    geom.areaCell.assign(static_cast<std::size_t>(nC), cellArea);
    geom.invAreaCell.resize(static_cast<std::size_t>(nC));
    for (std::size_t i = 0; i < static_cast<std::size_t>(nC); ++i) {
        geom.invAreaCell[i] = 1.0 / cellArea;
    }

    // ---- Edge geometry ----
    geom.dvEdge.assign(static_cast<std::size_t>(nE), dvEdge_val);
    geom.dcEdge.assign(static_cast<std::size_t>(nE), dcEdge_val);
    geom.invDcEdge.resize(static_cast<std::size_t>(nE));
    for (std::size_t i = 0; i < static_cast<std::size_t>(nE); ++i) {
        geom.invDcEdge[i] = 1.0 / dcEdge_val;
    }

    // ---- Vertical metrics (uniform spacing) ----
    const double dz = mesh.dz;
    const double rdzw_val = 1.0 / dz;
    const double rdzu_val = 1.0 / dz;

    geom.rdzw.assign(static_cast<std::size_t>(nVL), rdzw_val);
    geom.rdzu.assign(static_cast<std::size_t>(nVL), rdzu_val);
    geom.fzm.assign(static_cast<std::size_t>(nVL), 0.5);
    geom.fzp.assign(static_cast<std::size_t>(nVL), 0.5);
    geom.etp.assign(static_cast<std::size_t>(nVL), 1.0);
    geom.etm.assign(static_cast<std::size_t>(nVL), 0.0);
    geom.ewp.assign(static_cast<std::size_t>(nVL + 1), 0.5);
    geom.ewm.assign(static_cast<std::size_t>(nVL + 1), 0.5);

    // ---- Terrain metric (flat terrain: zz = 1.0) ----
    const std::size_t zz_size = static_cast<std::size_t>(nVL) * static_cast<std::size_t>(nC);
    geom.zz.assign(zz_size, 1.0);

    // ---- Base-state profiles (isothermal atmosphere) ----
    const std::size_t profile_size = static_cast<std::size_t>(nVL) * static_cast<std::size_t>(nC);
    geom.rb.resize(profile_size);
    geom.rtb.resize(profile_size);
    geom.pb.resize(profile_size);

    for (index_type k = 0; k < nVL; ++k) {
        // Height at the center of level k
        double z = (static_cast<double>(k) + 0.5) * dz;

        // Pressure from hydrostatic balance in isothermal atmosphere
        double p = p_sfc * std::exp(-g * z / (Rd * T0));

        // Base-state density
        double rb_val = p / (Rd * T0);

        // Potential temperature: theta = T * (p_ref/p)^(Rd/cp)
        double theta = T0 * std::pow(p_ref / p, Rd / cp);

        // rho*theta base state
        double rtb_val = rb_val * theta;

        // Fill all cells at this level (uniform base state)
        for (index_type iCell = 0; iCell < nC; ++iCell) {
            std::size_t idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(nC)
                            + static_cast<std::size_t>(iCell);
            geom.rb[idx]  = rb_val;
            geom.rtb[idx] = rtb_val;
            geom.pb[idx]  = p;
        }
    }

    // ---- Edge orientation signs ----
    // For a periodic mesh with consistent edge orientation, set sign = +1
    // for cells that "own" the edge and -1 for the other cell.
    const std::size_t sign_size = static_cast<std::size_t>(mE) * static_cast<std::size_t>(nC);
    geom.edgesOnCell_sign.resize(sign_size);

    for (index_type iCell = 0; iCell < nC; ++iCell) {
        for (index_type j = 0; j < mE; ++j) {
            std::size_t sIdx = static_cast<std::size_t>(j) * static_cast<std::size_t>(nC)
                             + static_cast<std::size_t>(iCell);
            index_type eId = mesh.edgesOnCell[static_cast<std::size_t>(iCell) * mE + j];
            // Cell is "owner" (cell 0 on edge) → sign = +1, else -1
            if (mesh.cellsOnEdge[static_cast<std::size_t>(eId) * 2 + 0] == iCell) {
                geom.edgesOnCell_sign[sIdx] = 1.0;
            } else {
                geom.edgesOnCell_sign[sIdx] = -1.0;
            }
        }
    }

    // ---- Specified zone masks (no specified zones for periodic mesh) ----
    geom.specZoneMaskEdge.assign(static_cast<std::size_t>(nE), 0.0);
    geom.specZoneMaskCell.assign(static_cast<std::size_t>(nC), 0.0);

    // ---- Edge reconstruction data (TRiSK weights) ----
    // For each edge, nEdgesOnEdge is the number of edges used in the
    // TRiSK reconstruction. For a uniform hex mesh, this is typically
    // the edges of the two adjacent cells minus the edge itself.
    // For simplicity, use the edges of the two cells (up to maxEdges2).
    geom.nEdgesOnEdge.resize(static_cast<std::size_t>(nE));
    const std::size_t eoe_size = static_cast<std::size_t>(nE) * static_cast<std::size_t>(mE2);
    geom.edgesOnEdge_storage.resize(eoe_size, 0);
    geom.weightsOnEdge.resize(eoe_size, 0.0);

    for (index_type iEdge = 0; iEdge < nE; ++iEdge) {
        // Get the two cells on this edge
        index_type c0 = mesh.cellsOnEdge[static_cast<std::size_t>(iEdge) * 2 + 0];
        index_type c1 = mesh.cellsOnEdge[static_cast<std::size_t>(iEdge) * 2 + 1];

        // Collect unique edges from both cells (excluding this edge)
        std::vector<index_type> stencilEdges;
        for (index_type j = 0; j < mE; ++j) {
            index_type e = mesh.edgesOnCell[static_cast<std::size_t>(c0) * mE + j];
            if (e != iEdge && e >= 0) {
                stencilEdges.push_back(e);
            }
        }
        for (index_type j = 0; j < mE; ++j) {
            index_type e = mesh.edgesOnCell[static_cast<std::size_t>(c1) * mE + j];
            if (e != iEdge && e >= 0) {
                // Avoid duplicates
                if (std::find(stencilEdges.begin(), stencilEdges.end(), e) == stencilEdges.end()) {
                    stencilEdges.push_back(e);
                }
            }
        }

        // Limit to maxEdges2
        index_type nEoE = static_cast<index_type>(
            std::min(static_cast<std::size_t>(mE2), stencilEdges.size()));
        geom.nEdgesOnEdge[static_cast<std::size_t>(iEdge)] = nEoE;

        // Fill edgesOnEdge and uniform weights
        double weight = (nEoE > 0) ? 1.0 / static_cast<double>(nEoE) : 0.0;
        for (index_type j = 0; j < nEoE; ++j) {
            std::size_t idx = static_cast<std::size_t>(iEdge) * static_cast<std::size_t>(mE2)
                            + static_cast<std::size_t>(j);
            geom.edgesOnEdge_storage[idx] = stencilEdges[static_cast<std::size_t>(j)];
            geom.weightsOnEdge[idx] = weight;
        }
    }

    // ---- Advection stencils ----
    // For each edge, the advection stencil includes cells near the edge.
    // Simple approach: include the 2 cells on the edge plus their neighbors.
    const std::size_t acfe_size = static_cast<std::size_t>(nE) * static_cast<std::size_t>(mAC);
    geom.advCellsForEdge_storage.resize(acfe_size, 0);
    geom.nAdvCellsForEdge.resize(static_cast<std::size_t>(nE));
    geom.adv_coefs.resize(acfe_size, 0.0);
    geom.adv_coefs_3rd.resize(acfe_size, 0.0);

    for (index_type iEdge = 0; iEdge < nE; ++iEdge) {
        index_type c0 = mesh.cellsOnEdge[static_cast<std::size_t>(iEdge) * 2 + 0];
        index_type c1 = mesh.cellsOnEdge[static_cast<std::size_t>(iEdge) * 2 + 1];

        // Stencil: the two cells on the edge + their immediate neighbors
        std::vector<index_type> stencilCells;
        stencilCells.push_back(c0);
        stencilCells.push_back(c1);

        // Add neighbors of c0 (avoiding duplicates)
        for (index_type j = 0; j < mE; ++j) {
            index_type cn = mesh.cellsOnCell[static_cast<std::size_t>(c0) * mE + j];
            if (cn >= 0 && std::find(stencilCells.begin(), stencilCells.end(), cn) == stencilCells.end()) {
                stencilCells.push_back(cn);
                if (static_cast<index_type>(stencilCells.size()) >= mAC) break;
            }
        }

        // Add neighbors of c1 if space remains
        if (static_cast<index_type>(stencilCells.size()) < mAC) {
            for (index_type j = 0; j < mE; ++j) {
                index_type cn = mesh.cellsOnCell[static_cast<std::size_t>(c1) * mE + j];
                if (cn >= 0 && std::find(stencilCells.begin(), stencilCells.end(), cn) == stencilCells.end()) {
                    stencilCells.push_back(cn);
                    if (static_cast<index_type>(stencilCells.size()) >= mAC) break;
                }
            }
        }

        index_type nAdv = static_cast<index_type>(
            std::min(static_cast<std::size_t>(mAC), stencilCells.size()));
        geom.nAdvCellsForEdge[static_cast<std::size_t>(iEdge)] = nAdv;

        // Fill advCellsForEdge and uniform advection coefficients
        double coef = (nAdv > 0) ? 1.0 / static_cast<double>(nAdv) : 0.0;
        for (index_type j = 0; j < nAdv; ++j) {
            std::size_t idx = static_cast<std::size_t>(iEdge) * static_cast<std::size_t>(mAC)
                            + static_cast<std::size_t>(j);
            geom.advCellsForEdge_storage[idx] = stencilCells[static_cast<std::size_t>(j)];
            geom.adv_coefs[idx] = coef;
            geom.adv_coefs_3rd[idx] = coef;
        }
    }

    // ---- Vertex geometry ----
    // Coriolis parameter: use f = 0 for simplicity (non-rotating)
    geom.fVertex.assign(static_cast<std::size_t>(nV), 0.0);

    // Triangle area: for dual mesh vertex triangles
    // Each vertex is shared by 3 cells, triangle area ~ cellArea / 3
    const double triArea = cellArea / 3.0;
    geom.areaTriangle.assign(static_cast<std::size_t>(nV), triArea);

    // ---- Store dimension members ----
    geom.maxEdges_ = mE;
    geom.maxEdges2 = mE2;
    geom.maxAdvCells = mAC;
    geom.nCells_ = nC;
    geom.nEdges_ = nE;
    geom.nVertices_ = nV;
    geom.nVertLevels_ = nVL;

    // ---- Terrain correction arrays (flat terrain: all zeros) ----
    const std::size_t zb_size = static_cast<std::size_t>(nVL + 1)
                              * static_cast<std::size_t>(mE)
                              * static_cast<std::size_t>(nC);
    geom.zb_cell.assign(zb_size, 0.0);
    geom.zb3_cell.assign(zb_size, 0.0);

    // ---- Base-state profiles for perturbation formulation ----
    geom.rho_base.assign(profile_size, 0.0);
    geom.rtheta_base.assign(profile_size, 0.0);
    geom.exner_base.assign(profile_size, 0.0);
    for (index_type k = 0; k < nVL; ++k) {
        double z = (static_cast<double>(k) + 0.5) * dz;
        double p = p_sfc * std::exp(-g * z / (Rd * T0));
        double rb_val = p / (Rd * T0);
        double theta = T0 * std::pow(p_ref / p, Rd / cp);
        double rtb_val = rb_val * theta;
        double exner_val = std::pow(p / p_ref, Rd / cp);
        for (index_type iCell = 0; iCell < nC; ++iCell) {
            std::size_t idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(nC)
                            + static_cast<std::size_t>(iCell);
            geom.rho_base[idx] = rb_val;
            geom.rtheta_base[idx] = rtb_val;
            geom.exner_base[idx] = exner_val;
        }
    }

    // ---- Vertical extrapolation coefficients (uniform grid) ----
    geom.cf1 = 0.0;
    geom.cf2 = 0.0;
    geom.cf3 = 0.0;

    // ---- Per-cell edge count ----
    geom.nEdgesOnCell.assign(static_cast<std::size_t>(nC), mE);

    geom.populated_ = true;

    // ========================================================================
    // Allocate prognostic field arrays
    // ========================================================================
    mesh.nScalars = 1;
    mesh.u.assign(static_cast<std::size_t>(nVL) * static_cast<std::size_t>(nE), 0.0);
    mesh.theta_m.resize(static_cast<std::size_t>(nVL) * static_cast<std::size_t>(nC));
    mesh.rho_zz.resize(static_cast<std::size_t>(nVL) * static_cast<std::size_t>(nC));
    mesh.w.assign(static_cast<std::size_t>(nVL + 1) * static_cast<std::size_t>(nC), 0.0);
    mesh.scalars.assign(
        static_cast<std::size_t>(mesh.nScalars) * static_cast<std::size_t>(nVL) * static_cast<std::size_t>(nC),
        0.0);

    // Initialize theta_m and rho_zz from the base state
    for (index_type k = 0; k < nVL; ++k) {
        double z = (static_cast<double>(k) + 0.5) * dz;
        double p = p_sfc * std::exp(-g * z / (Rd * T0));
        double rb_val = p / (Rd * T0);
        double theta = T0 * std::pow(p_ref / p, Rd / cp);

        for (index_type iCell = 0; iCell < nC; ++iCell) {
            std::size_t idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(nC)
                            + static_cast<std::size_t>(iCell);
            mesh.rho_zz[idx] = rb_val;       // for flat terrain, rho_zz = rho
            mesh.theta_m[idx] = theta;        // potential temperature
        }
    }

    return mesh;
}

} // namespace mpas::dycore::testing
