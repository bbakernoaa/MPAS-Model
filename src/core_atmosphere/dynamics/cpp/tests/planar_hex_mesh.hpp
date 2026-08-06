#pragma once

/// @file planar_hex_mesh.hpp
/// @brief Planar periodic hexagonal mesh generator for integration testing.
///
/// Generates a small planar hexagonal mesh with periodic boundaries in both
/// x and y directions, suitable for exercising the full SRK3 integration loop
/// via the C API (mpas_dycore_cpp_init + mpas_dycore_cpp_set_geometry).

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/geometry.hpp>

#include <vector>

namespace mpas::dycore::testing {

/// @brief Result struct from the planar hex mesh generator.
///
/// Contains all data needed to call mpas_dycore_cpp_init (connectivity) and
/// mpas_dycore_cpp_set_geometry (geometry/metrics/base-state), plus correctly
/// sized prognostic field arrays for calling mpas_dycore_cpp_timestep.
struct PlanarHexMesh {
    // ---- Mesh dimensions ----
    index_type nCells      = 0;
    index_type nEdges      = 0;
    index_type nVertices   = 0;
    index_type maxEdges    = 6;   // hexagonal cells
    index_type nVertLevels = 0;
    index_type maxEdges2   = 10;  // TRiSK stencil width
    index_type maxAdvCells = 10;  // advection stencil width

    // ---- Grid spacing ----
    double dx = 0.0;  // horizontal grid spacing (m)
    double dz = 0.0;  // vertical grid spacing (m)

    // ---- Connectivity arrays (for mpas_dycore_cpp_init) ----
    std::vector<index_type> cellsOnEdge;      // nEdges * 2
    std::vector<index_type> verticesOnEdge;   // nEdges * 2
    std::vector<index_type> edgesOnCell;      // nCells * maxEdges
    std::vector<index_type> cellsOnCell;      // nCells * maxEdges
    std::vector<index_type> nEdgesOnCell_arr; // nCells

    // ---- MeshGeometry (for mpas_dycore_cpp_set_geometry) ----
    MeshGeometry geometry;

    // ---- Prognostic field arrays (correctly sized for timestep) ----
    std::vector<real_type> u;        // nVertLevels * nEdges
    std::vector<real_type> theta_m;  // nVertLevels * nCells
    std::vector<real_type> rho_zz;   // nVertLevels * nCells
    std::vector<real_type> w;        // (nVertLevels+1) * nCells
    std::vector<real_type> scalars;  // nScalars * nVertLevels * nCells

    index_type nScalars = 1;  // at least one scalar for transport
};

/// @brief Generate a periodic planar hexagonal mesh.
///
/// Creates a hexagonal mesh with nx cells in the x-direction and ny cells
/// in the y-direction, with nVertLevels vertical levels. The mesh is
/// doubly-periodic (no open boundaries).
///
/// The mesh uses an isothermal atmosphere base state:
///   T0 = 250 K, p_surface = 101325 Pa
///   p(z) = p_surface * exp(-g*z / (R*T0))
///   rb = p / (R*T0)
///   rtb = rb * theta where theta = T0 * (p_ref/p)^(R/cp)
///   pb = p
///
/// Vertical metrics assume uniform spacing with domain height H = 10000 m:
///   dz = H / nVertLevels
///   rdzw = rdzu = 1/dz
///   fzm = fzp = 0.5 (uniform spacing)
///
/// @param nx Number of cells in the x-direction.
/// @param ny Number of cells in the y-direction.
/// @param nVertLevels Number of vertical levels.
/// @return A fully-populated PlanarHexMesh suitable for C API calls.
PlanarHexMesh generate_planar_hex_mesh(int nx, int ny, int nVertLevels);

} // namespace mpas::dycore::testing
