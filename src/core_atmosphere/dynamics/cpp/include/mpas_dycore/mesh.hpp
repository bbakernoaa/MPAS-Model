#pragma once

/// @file mesh.hpp
/// @brief Mesh connectivity representation and validation for the MPAS dynamical core.
///
/// Defines the MeshConnectivity struct holding non-owning mdspan views over
/// flat connectivity arrays produced by the marshalling layer. All index arrays
/// use zero-based indexing with INVALID_INDEX (-1) as the sentinel for missing
/// neighbors (boundary edges, under-connected cells).
///
/// Memory for connectivity arrays is allocated as single contiguous blocks
/// during initialization and never reallocated. The mdspan views are non-owning
/// references into these blocks.

#include <mpas_dycore/types.hpp>

#include <string>

namespace mpas::dycore {

// ============================================================================
// Sentinel for missing neighbors (boundary edges, under-connected cells)
// ============================================================================

/// Sentinel index value representing a missing neighbor in connectivity tables.
/// Boundary edges have one cell neighbor set to INVALID_INDEX; cells with fewer
/// than maxEdges neighbors have unused slots filled with INVALID_INDEX.
inline constexpr index_type INVALID_INDEX = -1;

// ============================================================================
// MeshConnectivity: non-owning views over flat connectivity arrays
// ============================================================================

/// @brief Non-owning mesh connectivity views for the MPAS unstructured Voronoi mesh.
///
/// Each ConnectivityView is an mdspan<const index_type, extents<dynamic, dynamic>>
/// referencing a contiguous flat array allocated during initialization. The first
/// extent is the number of owned entities; the second is the maximum neighbor count
/// for that relation.
///
/// All valid index values fall in [0, N-1] where N is the entity count for the
/// referenced type. Missing neighbors are represented by INVALID_INDEX (-1).
struct MeshConnectivity {
    /// Edge-to-cell connectivity: (nEdges, 2).
    /// cellsOnEdge[iEdge, 0] and cellsOnEdge[iEdge, 1] give the two cells
    /// sharing edge iEdge. Boundary edges have one entry set to INVALID_INDEX.
    ConnectivityView cellsOnEdge;

    /// Edge-to-vertex connectivity: (nEdges, 2).
    /// verticesOnEdge[iEdge, 0] and verticesOnEdge[iEdge, 1] give the two
    /// vertices at the endpoints of edge iEdge.
    ConnectivityView verticesOnEdge;

    /// Cell-to-edge connectivity: (nCells, maxEdges).
    /// edgesOnCell[iCell, j] gives the j-th edge of cell iCell.
    /// Unused slots (j >= nEdgesOnCell[iCell]) are INVALID_INDEX.
    ConnectivityView edgesOnCell;

    /// Cell-to-cell connectivity: (nCells, maxEdges).
    /// cellsOnCell[iCell, j] gives the neighbor cell across the j-th edge.
    /// Unused slots are INVALID_INDEX.
    ConnectivityView cellsOnCell;

    /// Cell-to-vertex connectivity: (nCells, maxEdges).
    /// verticesOnCell[iCell, j] gives the j-th vertex of cell iCell.
    /// Unused slots are INVALID_INDEX.
    ConnectivityView verticesOnCell;

    /// Vertex-to-cell connectivity: (nVertices, 3).
    /// cellsOnVertex[iVertex, j] gives the j-th cell sharing vertex iVertex.
    /// Boundary vertices may have entries set to INVALID_INDEX.
    ConnectivityView cellsOnVertex;

    /// Per-cell neighbor count: (nCells).
    /// nEdgesOnCell[iCell] gives the actual number of edges/neighbors for cell iCell.
    std::mdspan<const index_type,
        std::extents<index_type, std::dynamic_extent>> nEdgesOnCell;
};

// ============================================================================
// Validation result
// ============================================================================

/// @brief Result of mesh connectivity validation.
///
/// If valid is true, the mesh passed all range checks. Otherwise, the remaining
/// fields identify the first out-of-range entry found.
struct MeshValidationResult {
    /// True if all connectivity indices are within valid ranges.
    bool valid;

    /// Name of the connectivity table containing the invalid entry (empty if valid).
    std::string table_name;

    /// Row index of the invalid entry in the connectivity table.
    index_type entry_row;

    /// Column index of the invalid entry in the connectivity table.
    index_type entry_col;

    /// The out-of-range index value found.
    index_type bad_value;

    /// The entity count (upper bound) that was violated.
    index_type entity_count;
};

// ============================================================================
// Validation function
// ============================================================================

/// @brief Validate all connectivity indices in a MeshConnectivity struct.
///
/// Checks every entry in all connectivity tables to ensure values are either
/// INVALID_INDEX (-1) or in the valid range [0, N-1] where N is the entity
/// count for the referenced type. Returns on the first invalid entry found.
///
/// @param mesh       The mesh connectivity to validate.
/// @param nCells     Total number of cells (valid cell indices are [0, nCells-1]).
/// @param nEdges     Total number of edges (valid edge indices are [0, nEdges-1]).
/// @param nVertices  Total number of vertices (valid vertex indices are [0, nVertices-1]).
/// @return A MeshValidationResult indicating success or the first invalid entry.
MeshValidationResult validate_connectivity(
    const MeshConnectivity& mesh,
    index_type nCells,
    index_type nEdges,
    index_type nVertices);

} // namespace mpas::dycore
