/// @file mesh.cpp
/// @brief Implementation of mesh connectivity validation.
///
/// Validates that all connectivity indices in a MeshConnectivity struct fall
/// within valid ranges [0, N-1] or are the INVALID_INDEX sentinel (-1).
/// Any value less than -1 or >= the entity count is rejected.

#include <mpas_dycore/mesh.hpp>

namespace mpas::dycore {

namespace {

/// @brief Check a single connectivity table for out-of-range indices.
///
/// Iterates over all entries in the given ConnectivityView. Each value must
/// be either INVALID_INDEX (-1) or in [0, entity_count-1]. Returns an invalid
/// result on the first violation found.
///
/// @param view         The connectivity table to check.
/// @param table_name   Human-readable name for error reporting.
/// @param entity_count Upper bound: valid indices are [0, entity_count-1].
/// @return A valid result if all entries pass, or an invalid result with details.
MeshValidationResult check_table(
    ConnectivityView view,
    const char* table_name,
    index_type entity_count)
{
    const auto nRows = view.extent(0);
    const auto nCols = view.extent(1);

    for (index_type row = 0; row < static_cast<index_type>(nRows); ++row) {
        for (index_type col = 0; col < static_cast<index_type>(nCols); ++col) {
            const index_type val = view[row, col];

            // Valid values: INVALID_INDEX (-1) or [0, entity_count-1]
            if (val < INVALID_INDEX || val >= entity_count) {
                return MeshValidationResult{
                    .valid = false,
                    .table_name = table_name,
                    .entry_row = row,
                    .entry_col = col,
                    .bad_value = val,
                    .entity_count = entity_count
                };
            }
        }
    }

    return MeshValidationResult{
        .valid = true,
        .table_name = {},
        .entry_row = 0,
        .entry_col = 0,
        .bad_value = 0,
        .entity_count = 0
    };
}

} // anonymous namespace

MeshValidationResult validate_connectivity(
    const MeshConnectivity& mesh,
    index_type nCells,
    index_type nEdges,
    index_type nVertices)
{
    // Check cellsOnEdge: values must reference cells [0, nCells-1] or INVALID_INDEX
    if (auto result = check_table(mesh.cellsOnEdge, "cellsOnEdge", nCells);
        !result.valid) {
        return result;
    }

    // Check verticesOnEdge: values must reference vertices [0, nVertices-1] or INVALID_INDEX
    if (auto result = check_table(mesh.verticesOnEdge, "verticesOnEdge", nVertices);
        !result.valid) {
        return result;
    }

    // Check edgesOnCell: values must reference edges [0, nEdges-1] or INVALID_INDEX
    if (auto result = check_table(mesh.edgesOnCell, "edgesOnCell", nEdges);
        !result.valid) {
        return result;
    }

    // Check cellsOnCell: values must reference cells [0, nCells-1] or INVALID_INDEX
    if (auto result = check_table(mesh.cellsOnCell, "cellsOnCell", nCells);
        !result.valid) {
        return result;
    }

    // Check verticesOnCell: values must reference vertices [0, nVertices-1] or INVALID_INDEX
    if (auto result = check_table(mesh.verticesOnCell, "verticesOnCell", nVertices);
        !result.valid) {
        return result;
    }

    // Check cellsOnVertex: values must reference cells [0, nCells-1] or INVALID_INDEX
    if (auto result = check_table(mesh.cellsOnVertex, "cellsOnVertex", nCells);
        !result.valid) {
        return result;
    }

    // All tables passed validation
    return MeshValidationResult{
        .valid = true,
        .table_name = {},
        .entry_row = 0,
        .entry_col = 0,
        .bad_value = 0,
        .entity_count = 0
    };
}

} // namespace mpas::dycore
