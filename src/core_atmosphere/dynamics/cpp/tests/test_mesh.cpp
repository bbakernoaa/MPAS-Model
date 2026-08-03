/// @file test_mesh.cpp
/// @brief Unit tests for MeshConnectivity struct and validate_connectivity().

#include <mpas_dycore/mesh.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace mpas::dycore {
namespace {

// Helper to create a minimal valid mesh for testing
struct TestMeshData {
    std::vector<index_type> cellsOnEdge;
    std::vector<index_type> verticesOnEdge;
    std::vector<index_type> edgesOnCell;
    std::vector<index_type> cellsOnCell;
    std::vector<index_type> verticesOnCell;
    std::vector<index_type> cellsOnVertex;
    std::vector<index_type> nEdgesOnCell_arr;

    index_type nCells = 2;
    index_type nEdges = 2;
    index_type nVertices = 3;
    index_type maxEdges = 2;

    MeshConnectivity make_connectivity() const {
        MeshConnectivity mesh{};
        mesh.cellsOnEdge = ConnectivityView(cellsOnEdge.data(), nEdges, 2);
        mesh.verticesOnEdge = ConnectivityView(verticesOnEdge.data(), nEdges, 2);
        mesh.edgesOnCell = ConnectivityView(edgesOnCell.data(), nCells, maxEdges);
        mesh.cellsOnCell = ConnectivityView(cellsOnCell.data(), nCells, maxEdges);
        mesh.verticesOnCell = ConnectivityView(verticesOnCell.data(), nCells, maxEdges);
        mesh.cellsOnVertex = ConnectivityView(cellsOnVertex.data(), nVertices, 3);
        mesh.nEdgesOnCell = std::mdspan<const index_type,
            std::extents<index_type, std::dynamic_extent>>(nEdgesOnCell_arr.data(), nCells);
        return mesh;
    }
};

TestMeshData make_valid_test_data() {
    TestMeshData d;
    // 2 cells, 2 edges, 3 vertices, maxEdges=2
    d.cellsOnEdge = {0, 1, 1, INVALID_INDEX};
    d.verticesOnEdge = {0, 1, 1, 2};
    d.edgesOnCell = {0, INVALID_INDEX, 0, 1};
    d.cellsOnCell = {1, INVALID_INDEX, 0, INVALID_INDEX};
    d.verticesOnCell = {0, 1, 1, 2};
    d.cellsOnVertex = {0, INVALID_INDEX, INVALID_INDEX,
                       0, 1, INVALID_INDEX,
                       1, INVALID_INDEX, INVALID_INDEX};
    d.nEdgesOnCell_arr = {1, 2};
    return d;
}

// ---------------------------------------------------------------------------
// Tests for valid meshes
// ---------------------------------------------------------------------------

TEST(MeshValidation, EmptyMeshIsValid) {
    // Zero-extent connectivity views
    std::vector<index_type> empty;
    MeshConnectivity mesh{};
    mesh.cellsOnEdge = ConnectivityView(empty.data(), 0, 2);
    mesh.verticesOnEdge = ConnectivityView(empty.data(), 0, 2);
    mesh.edgesOnCell = ConnectivityView(empty.data(), 0, 2);
    mesh.cellsOnCell = ConnectivityView(empty.data(), 0, 2);
    mesh.verticesOnCell = ConnectivityView(empty.data(), 0, 2);
    mesh.cellsOnVertex = ConnectivityView(empty.data(), 0, 3);
    mesh.nEdgesOnCell = std::mdspan<const index_type,
        std::extents<index_type, std::dynamic_extent>>(empty.data(), 0);

    auto result = validate_connectivity(mesh, 0, 0, 0);
    EXPECT_TRUE(result.valid);
}

TEST(MeshValidation, SimpleValidMeshPasses) {
    auto data = make_valid_test_data();
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_TRUE(result.valid);
}

TEST(MeshValidation, AllSentinelValuesAreValid) {
    // A mesh where all connectivity entries are INVALID_INDEX
    std::vector<index_type> all_invalid(20, INVALID_INDEX);
    std::vector<index_type> n_edges(2, 0);

    MeshConnectivity mesh{};
    mesh.cellsOnEdge = ConnectivityView(all_invalid.data(), 2, 2);
    mesh.verticesOnEdge = ConnectivityView(all_invalid.data(), 2, 2);
    mesh.edgesOnCell = ConnectivityView(all_invalid.data(), 2, 2);
    mesh.cellsOnCell = ConnectivityView(all_invalid.data(), 2, 2);
    mesh.verticesOnCell = ConnectivityView(all_invalid.data(), 2, 2);
    mesh.cellsOnVertex = ConnectivityView(all_invalid.data(), 2, 3);
    mesh.nEdgesOnCell = std::mdspan<const index_type,
        std::extents<index_type, std::dynamic_extent>>(n_edges.data(), 2);

    auto result = validate_connectivity(mesh, 2, 2, 3);
    EXPECT_TRUE(result.valid);
}

TEST(MeshValidation, MaxValidIndexAccepted) {
    auto data = make_valid_test_data();
    // Set a value to exactly nCells-1 (the maximum valid index)
    data.cellsOnEdge[1] = data.nCells - 1;
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_TRUE(result.valid);
}

// ---------------------------------------------------------------------------
// Tests for invalid meshes - out-of-range indices
// ---------------------------------------------------------------------------

TEST(MeshValidation, CellsOnEdgeOutOfRange) {
    auto data = make_valid_test_data();
    data.cellsOnEdge[1] = data.nCells; // one past valid range
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "cellsOnEdge");
    EXPECT_EQ(result.bad_value, data.nCells);
    EXPECT_EQ(result.entity_count, data.nCells);
    EXPECT_EQ(result.entry_row, 0);
    EXPECT_EQ(result.entry_col, 1);
}

TEST(MeshValidation, VerticesOnEdgeOutOfRange) {
    auto data = make_valid_test_data();
    data.verticesOnEdge[3] = data.nVertices; // row=1, col=1
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "verticesOnEdge");
    EXPECT_EQ(result.bad_value, data.nVertices);
    EXPECT_EQ(result.entity_count, data.nVertices);
}

TEST(MeshValidation, EdgesOnCellOutOfRange) {
    auto data = make_valid_test_data();
    data.edgesOnCell[0] = data.nEdges; // row=0, col=0
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "edgesOnCell");
    EXPECT_EQ(result.bad_value, data.nEdges);
    EXPECT_EQ(result.entity_count, data.nEdges);
}

TEST(MeshValidation, CellsOnCellOutOfRange) {
    auto data = make_valid_test_data();
    data.cellsOnCell[0] = data.nCells + 100;
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "cellsOnCell");
    EXPECT_EQ(result.bad_value, data.nCells + 100);
}

TEST(MeshValidation, VerticesOnCellOutOfRange) {
    auto data = make_valid_test_data();
    data.verticesOnCell[2] = data.nVertices;
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "verticesOnCell");
    EXPECT_EQ(result.bad_value, data.nVertices);
}

TEST(MeshValidation, CellsOnVertexOutOfRange) {
    auto data = make_valid_test_data();
    data.cellsOnVertex[0] = data.nCells + 50;
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "cellsOnVertex");
    EXPECT_EQ(result.bad_value, data.nCells + 50);
}

// ---------------------------------------------------------------------------
// Tests for invalid meshes - values less than -1
// ---------------------------------------------------------------------------

TEST(MeshValidation, ValueLessThanSentinelRejected) {
    auto data = make_valid_test_data();
    data.cellsOnEdge[0] = -2; // less than INVALID_INDEX (-1)
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "cellsOnEdge");
    EXPECT_EQ(result.bad_value, -2);
}

TEST(MeshValidation, LargeNegativeValueRejected) {
    auto data = make_valid_test_data();
    data.verticesOnEdge[1] = -100;
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "verticesOnEdge");
    EXPECT_EQ(result.bad_value, -100);
}

// ---------------------------------------------------------------------------
// Tests verifying result fields are populated correctly
// ---------------------------------------------------------------------------

TEST(MeshValidation, InvalidResultReportsCorrectRowAndCol) {
    auto data = make_valid_test_data();
    // Corrupt the second row, first col of edgesOnCell
    // edgesOnCell is (nCells=2, maxEdges=2), row-major: [row0col0, row0col1, row1col0, row1col1]
    data.edgesOnCell[2] = data.nEdges + 5; // row=1, col=0
    auto mesh = data.make_connectivity();

    auto result = validate_connectivity(mesh, data.nCells, data.nEdges, data.nVertices);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.table_name, "edgesOnCell");
    EXPECT_EQ(result.entry_row, 1);
    EXPECT_EQ(result.entry_col, 0);
    EXPECT_EQ(result.bad_value, data.nEdges + 5);
    EXPECT_EQ(result.entity_count, data.nEdges);
}

TEST(MeshValidation, InvalidIndexSentinelConstant) {
    // Verify the sentinel value
    EXPECT_EQ(INVALID_INDEX, -1);
}

} // anonymous namespace
} // namespace mpas::dycore
