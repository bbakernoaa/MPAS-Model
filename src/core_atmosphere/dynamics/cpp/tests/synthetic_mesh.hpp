#pragma once

/// @file synthetic_mesh.hpp
/// @brief Synthetic mesh generators for unit and property-based testing.
///
/// Provides small valid and invalid unstructured meshes suitable for testing
/// connectivity validation, kernel correctness, and property-based testing
/// with RapidCheck.

#include <mpas_dycore/types.hpp>

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <numeric>

#ifdef MPAS_HAVE_RAPIDCHECK
#include <rapidcheck.h>
#endif

namespace mpas::dycore::testing {

// ============================================================================
// Sentinel value for unused connectivity slots
// ============================================================================

inline constexpr index_type INVALID_INDEX = -1;

// ============================================================================
// SyntheticMesh: owns all connectivity and geometry data for testing
// ============================================================================

/// @brief A self-contained synthetic mesh for testing.
///
/// Owns all flat arrays for connectivity and geometry and provides helper
/// methods to create mdspan ConnectivityView objects over the owned data.
struct SyntheticMesh {
    // Mesh dimensions
    index_type nCells      = 0;
    index_type nEdges      = 0;
    index_type nVertices   = 0;
    index_type maxEdges    = 0;
    index_type nVertLevels = 0;

    // Connectivity arrays (flat, row-major storage)
    std::vector<index_type> cellsOnEdge;      // flat: nEdges * 2
    std::vector<index_type> verticesOnEdge;   // flat: nEdges * 2
    std::vector<index_type> edgesOnCell;      // flat: nCells * maxEdges
    std::vector<index_type> cellsOnCell;      // flat: nCells * maxEdges
    std::vector<index_type> verticesOnCell;   // flat: nCells * maxEdges
    std::vector<index_type> cellsOnVertex;    // flat: nVertices * 3
    std::vector<index_type> nEdgesOnCell_arr; // nCells

    // Geometry arrays
    std::vector<double> areaCell;      // nCells
    std::vector<double> areaTriangle;  // nVertices
    std::vector<double> dvEdge;        // nEdges
    std::vector<double> dcEdge;        // nEdges

    // ========================================================================
    // Helper to create ConnectivityView mdspans from flat arrays
    // ========================================================================

    /// @brief Create a ConnectivityView over cellsOnEdge (nEdges x 2).
    ConnectivityView cellsOnEdge_view() const {
        return ConnectivityView(cellsOnEdge.data(), nEdges, 2);
    }

    /// @brief Create a ConnectivityView over verticesOnEdge (nEdges x 2).
    ConnectivityView verticesOnEdge_view() const {
        return ConnectivityView(verticesOnEdge.data(), nEdges, 2);
    }

    /// @brief Create a ConnectivityView over edgesOnCell (nCells x maxEdges).
    ConnectivityView edgesOnCell_view() const {
        return ConnectivityView(edgesOnCell.data(), nCells, maxEdges);
    }

    /// @brief Create a ConnectivityView over cellsOnCell (nCells x maxEdges).
    ConnectivityView cellsOnCell_view() const {
        return ConnectivityView(cellsOnCell.data(), nCells, maxEdges);
    }

    /// @brief Create a ConnectivityView over verticesOnCell (nCells x maxEdges).
    ConnectivityView verticesOnCell_view() const {
        return ConnectivityView(verticesOnCell.data(), nCells, maxEdges);
    }

    /// @brief Create a ConnectivityView over cellsOnVertex (nVertices x 3).
    ConnectivityView cellsOnVertex_view() const {
        return ConnectivityView(cellsOnVertex.data(), nVertices, 3);
    }

    /// @brief Create a 1D mdspan view over nEdgesOnCell.
    std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>
    nEdgesOnCell_view() const {
        return std::mdspan<const index_type, std::extents<index_type, std::dynamic_extent>>(
            nEdgesOnCell_arr.data(), nCells);
    }
};

// ============================================================================
// genValidMesh: create a simple valid mesh with linear chain topology
// ============================================================================

/// @brief Generate a valid synthetic mesh with a linear chain topology.
///
/// Produces a chain of cells connected in sequence. Each cell has 4 edges
/// (interior cells) or fewer (boundary cells), padded to maxEdges = 6.
/// All connectivity indices are valid (in [0, N-1] or INVALID_INDEX sentinel).
///
/// @param nCells Number of cells in the chain (must be >= 2).
/// @param nVertLevels Number of vertical levels (default: 10).
/// @return A fully-populated SyntheticMesh struct.
inline SyntheticMesh genValidMesh(int nCells, int nVertLevels = 10) {
    assert(nCells >= 2 && "genValidMesh requires at least 2 cells");

    SyntheticMesh mesh;
    mesh.nCells      = static_cast<index_type>(nCells);
    mesh.maxEdges    = 6;  // hexagonal-like padding
    mesh.nVertLevels = static_cast<index_type>(nVertLevels);

    // Linear chain topology:
    // - Each pair of adjacent cells shares one internal edge
    // - Each cell also has top/bottom/left/right boundary edges (simplified)
    //
    // For a chain of N cells:
    //   nEdges = (N-1) internal edges + N top edges + N bottom edges + 2 side edges
    //          = 3N + 1
    //   nVertices = 2*N (top and bottom vertices per cell)

    const int numInternalEdges = nCells - 1;
    const int numTopEdges      = nCells;
    const int numBottomEdges   = nCells;
    const int numSideEdges     = 2;  // left and right boundary
    const int totalEdges       = numInternalEdges + numTopEdges + numBottomEdges + numSideEdges;
    const int totalVertices    = 2 * (nCells + 1);  // vertices at corners of chain

    mesh.nEdges    = static_cast<index_type>(totalEdges);
    mesh.nVertices = static_cast<index_type>(totalVertices);

    // Allocate connectivity arrays
    mesh.cellsOnEdge.resize(static_cast<std::size_t>(mesh.nEdges) * 2, INVALID_INDEX);
    mesh.verticesOnEdge.resize(static_cast<std::size_t>(mesh.nEdges) * 2, INVALID_INDEX);
    mesh.edgesOnCell.resize(static_cast<std::size_t>(mesh.nCells) * mesh.maxEdges, INVALID_INDEX);
    mesh.cellsOnCell.resize(static_cast<std::size_t>(mesh.nCells) * mesh.maxEdges, INVALID_INDEX);
    mesh.verticesOnCell.resize(static_cast<std::size_t>(mesh.nCells) * mesh.maxEdges, INVALID_INDEX);
    mesh.cellsOnVertex.resize(static_cast<std::size_t>(mesh.nVertices) * 3, INVALID_INDEX);
    mesh.nEdgesOnCell_arr.resize(static_cast<std::size_t>(mesh.nCells), 0);

    // Helper to set cellsOnEdge[edgeIdx, col]
    auto setCellsOnEdge = [&](int edgeIdx, int col, index_type val) {
        mesh.cellsOnEdge[static_cast<std::size_t>(edgeIdx) * 2 + col] = val;
    };

    // Helper to set verticesOnEdge[edgeIdx, col]
    auto setVerticesOnEdge = [&](int edgeIdx, int col, index_type val) {
        mesh.verticesOnEdge[static_cast<std::size_t>(edgeIdx) * 2 + col] = val;
    };

    // Helper to set edgesOnCell[cellIdx, slot]
    auto setEdgesOnCell = [&](int cellIdx, int slot, index_type val) {
        mesh.edgesOnCell[static_cast<std::size_t>(cellIdx) * mesh.maxEdges + slot] = val;
    };

    // Helper to set cellsOnCell[cellIdx, slot]
    auto setCellsOnCell = [&](int cellIdx, int slot, index_type val) {
        mesh.cellsOnCell[static_cast<std::size_t>(cellIdx) * mesh.maxEdges + slot] = val;
    };

    // Helper to set verticesOnCell[cellIdx, slot]
    auto setVerticesOnCell = [&](int cellIdx, int slot, index_type val) {
        mesh.verticesOnCell[static_cast<std::size_t>(cellIdx) * mesh.maxEdges + slot] = val;
    };

    // Helper to set cellsOnVertex[vertIdx, col]
    auto setCellsOnVertex = [&](int vertIdx, int col, index_type val) {
        mesh.cellsOnVertex[static_cast<std::size_t>(vertIdx) * 3 + col] = val;
    };

    // Edge layout:
    //   [0, numInternalEdges)             : internal edges between adjacent cells
    //   [numInternalEdges, numInternalEdges + numTopEdges)   : top edges
    //   [numInternalEdges + numTopEdges, ...)                : bottom edges
    //   last 2: side boundary edges

    int edgeOffset_internal = 0;
    int edgeOffset_top      = numInternalEdges;
    int edgeOffset_bottom   = numInternalEdges + numTopEdges;
    int edgeOffset_side     = numInternalEdges + numTopEdges + numBottomEdges;

    // Fill internal edges: edge i connects cell i to cell i+1
    for (int i = 0; i < numInternalEdges; ++i) {
        int edgeIdx = edgeOffset_internal + i;
        setCellsOnEdge(edgeIdx, 0, static_cast<index_type>(i));
        setCellsOnEdge(edgeIdx, 1, static_cast<index_type>(i + 1));
        // Vertices: bottom-right of cell i and top-right of cell i
        setVerticesOnEdge(edgeIdx, 0, static_cast<index_type>(2 * (i + 1)));
        setVerticesOnEdge(edgeIdx, 1, static_cast<index_type>(2 * (i + 1) + 1));
    }

    // Fill top edges: each cell has a top edge (boundary)
    for (int i = 0; i < numTopEdges; ++i) {
        int edgeIdx = edgeOffset_top + i;
        setCellsOnEdge(edgeIdx, 0, static_cast<index_type>(i));
        setCellsOnEdge(edgeIdx, 1, INVALID_INDEX); // boundary
        setVerticesOnEdge(edgeIdx, 0, static_cast<index_type>(2 * i + 1));
        setVerticesOnEdge(edgeIdx, 1, static_cast<index_type>(2 * (i + 1) + 1));
    }

    // Fill bottom edges: each cell has a bottom edge (boundary)
    for (int i = 0; i < numBottomEdges; ++i) {
        int edgeIdx = edgeOffset_bottom + i;
        setCellsOnEdge(edgeIdx, 0, static_cast<index_type>(i));
        setCellsOnEdge(edgeIdx, 1, INVALID_INDEX); // boundary
        setVerticesOnEdge(edgeIdx, 0, static_cast<index_type>(2 * i));
        setVerticesOnEdge(edgeIdx, 1, static_cast<index_type>(2 * (i + 1)));
    }

    // Fill side boundary edges (left and right caps)
    // Left cap: cell 0
    {
        int edgeIdx = edgeOffset_side;
        setCellsOnEdge(edgeIdx, 0, 0);
        setCellsOnEdge(edgeIdx, 1, INVALID_INDEX);
        setVerticesOnEdge(edgeIdx, 0, 0);
        setVerticesOnEdge(edgeIdx, 1, 1);
    }
    // Right cap: cell nCells-1
    {
        int edgeIdx = edgeOffset_side + 1;
        setCellsOnEdge(edgeIdx, 0, static_cast<index_type>(nCells - 1));
        setCellsOnEdge(edgeIdx, 1, INVALID_INDEX);
        setVerticesOnEdge(edgeIdx, 0, static_cast<index_type>(2 * nCells));
        setVerticesOnEdge(edgeIdx, 1, static_cast<index_type>(2 * nCells + 1));
    }

    // Fill edgesOnCell, cellsOnCell, verticesOnCell, and nEdgesOnCell
    for (int i = 0; i < nCells; ++i) {
        int slot = 0;

        // Internal edges (left neighbor)
        if (i > 0) {
            int leftEdge = edgeOffset_internal + (i - 1);
            setEdgesOnCell(i, slot, static_cast<index_type>(leftEdge));
            setCellsOnCell(i, slot, static_cast<index_type>(i - 1));
            slot++;
        }

        // Internal edges (right neighbor)
        if (i < nCells - 1) {
            int rightEdge = edgeOffset_internal + i;
            setEdgesOnCell(i, slot, static_cast<index_type>(rightEdge));
            setCellsOnCell(i, slot, static_cast<index_type>(i + 1));
            slot++;
        }

        // Top edge
        {
            int topEdge = edgeOffset_top + i;
            setEdgesOnCell(i, slot, static_cast<index_type>(topEdge));
            // No cell neighbor on top (boundary)
            slot++;
        }

        // Bottom edge
        {
            int bottomEdge = edgeOffset_bottom + i;
            setEdgesOnCell(i, slot, static_cast<index_type>(bottomEdge));
            // No cell neighbor on bottom (boundary)
            slot++;
        }

        // Side edges for boundary cells
        if (i == 0) {
            setEdgesOnCell(i, slot, static_cast<index_type>(edgeOffset_side));
            slot++;
        }
        if (i == nCells - 1) {
            setEdgesOnCell(i, slot, static_cast<index_type>(edgeOffset_side + 1));
            slot++;
        }

        mesh.nEdgesOnCell_arr[static_cast<std::size_t>(i)] = static_cast<index_type>(slot);

        // Vertices on cell: 4 corners (bottom-left, bottom-right, top-right, top-left)
        setVerticesOnCell(i, 0, static_cast<index_type>(2 * i));         // bottom-left
        setVerticesOnCell(i, 1, static_cast<index_type>(2 * (i + 1)));   // bottom-right
        setVerticesOnCell(i, 2, static_cast<index_type>(2 * (i + 1) + 1)); // top-right
        setVerticesOnCell(i, 3, static_cast<index_type>(2 * i + 1));     // top-left
    }

    // Fill cellsOnVertex: each vertex can be adjacent to up to 3 cells
    // For our linear chain, bottom vertices (even indices) and top vertices (odd indices)
    // are each shared by at most 2 cells.
    for (int v = 0; v < totalVertices; ++v) {
        // Determine which cells this vertex belongs to based on position
        // Vertex v: bottom row if v is even, top row if v is odd
        // Position along chain: v/2 (integer division)
        int pos = v / 2;  // position 0..nCells

        int col = 0;
        // Left cell (pos-1) if exists
        if (pos > 0 && pos <= nCells) {
            setCellsOnVertex(v, col, static_cast<index_type>(pos - 1));
            col++;
        }
        // Right cell (pos) if exists
        if (pos < nCells) {
            setCellsOnVertex(v, col, static_cast<index_type>(pos));
            col++;
        }
        // Remaining slots stay INVALID_INDEX
    }

    // ========================================================================
    // Geometry: reasonable physical values
    // ========================================================================

    // areaCell: typical MPAS cell area ~1e8 to 1e10 m^2 (120km resolution ~ 1.44e10)
    mesh.areaCell.resize(static_cast<std::size_t>(mesh.nCells));
    for (int i = 0; i < nCells; ++i) {
        mesh.areaCell[static_cast<std::size_t>(i)] = 1.0e9 + static_cast<double>(i) * 1.0e6;
    }

    // areaTriangle: dual mesh triangle area, ~1/3 of cell area
    mesh.areaTriangle.resize(static_cast<std::size_t>(mesh.nVertices));
    for (int v = 0; v < totalVertices; ++v) {
        mesh.areaTriangle[static_cast<std::size_t>(v)] = 3.0e8 + static_cast<double>(v) * 5.0e5;
    }

    // dvEdge: distance between vertices of an edge, ~1e4 m (10 km)
    mesh.dvEdge.resize(static_cast<std::size_t>(mesh.nEdges));
    for (int e = 0; e < totalEdges; ++e) {
        mesh.dvEdge[static_cast<std::size_t>(e)] = 1.0e4 + static_cast<double>(e) * 100.0;
    }

    // dcEdge: distance between cells sharing an edge, ~1e4 m (10 km)
    mesh.dcEdge.resize(static_cast<std::size_t>(mesh.nEdges));
    for (int e = 0; e < totalEdges; ++e) {
        mesh.dcEdge[static_cast<std::size_t>(e)] = 1.2e4 + static_cast<double>(e) * 120.0;
    }

    return mesh;
}

// ============================================================================
// genInvalidMesh: create a mesh with deliberately invalid connectivity
// ============================================================================

/// @brief Generate a mesh with deliberately invalid connectivity indices.
///
/// Produces a small mesh where some connectivity entries contain indices
/// that exceed entity counts or are less than -1. Useful for testing that
/// validation routines correctly reject invalid meshes.
///
/// @return A SyntheticMesh with known invalid indices.
inline SyntheticMesh genInvalidMesh() {
    // Start with a small valid mesh and corrupt it
    SyntheticMesh mesh = genValidMesh(4);

    // Corruption 1: cellsOnEdge has index exceeding nCells
    // Set cellsOnEdge[0, 0] to nCells (one past valid range)
    mesh.cellsOnEdge[0] = mesh.nCells;  // out-of-range: should be < nCells

    // Corruption 2: verticesOnEdge has negative index < -1
    // Set verticesOnEdge[1, 0] to -2
    mesh.verticesOnEdge[2] = -2;  // invalid: less than sentinel -1

    // Corruption 3: edgesOnCell has index exceeding nEdges
    // Set edgesOnCell[0, 0] to nEdges (one past valid range)
    mesh.edgesOnCell[0] = mesh.nEdges;  // out-of-range: should be < nEdges

    // Corruption 4: cellsOnVertex has index exceeding nCells
    // Set cellsOnVertex[0, 0] to nCells + 100
    mesh.cellsOnVertex[0] = mesh.nCells + 100;  // wildly out-of-range

    return mesh;
}

// ============================================================================
// RapidCheck integration: random valid mesh generator
// ============================================================================

#ifdef MPAS_HAVE_RAPIDCHECK

/// @brief RapidCheck generator producing random valid meshes with 4-20 cells.
///
/// Generates meshes with the linear chain topology and randomized cell counts
/// within [4, 20]. Geometry values are randomized within physically reasonable
/// ranges.
///
/// @return An rc::Gen<SyntheticMesh> that can be used with rc::prop.
inline rc::Gen<SyntheticMesh> genRandomValidMesh() {
    return rc::gen::mapcat(
        rc::gen::inRange(4, 21),  // nCells in [4, 20]
        [](int nCells) {
            return rc::gen::map(
                rc::gen::tuple(
                    rc::gen::inRange(5, 51),   // nVertLevels in [5, 50]
                    rc::gen::inRange(0, 1000)  // seed for geometry variation
                ),
                [nCells](std::tuple<int, int> params) {
                    auto [nVertLevels, geomSeed] = params;

                    SyntheticMesh mesh = genValidMesh(nCells, nVertLevels);

                    // Apply random variation to geometry using the seed
                    double scale = 1.0 + static_cast<double>(geomSeed) * 0.001;

                    for (auto& a : mesh.areaCell) {
                        a *= scale;
                    }
                    for (auto& a : mesh.areaTriangle) {
                        a *= scale;
                    }
                    for (auto& d : mesh.dvEdge) {
                        d *= scale;
                    }
                    for (auto& d : mesh.dcEdge) {
                        d *= scale;
                    }

                    return mesh;
                }
            );
        }
    );
}

#endif // MPAS_HAVE_RAPIDCHECK

} // namespace mpas::dycore::testing
