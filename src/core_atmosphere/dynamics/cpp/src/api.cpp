/// @file api.cpp
/// @brief C-linkage API implementation for the MPAS dynamical core.
///
/// Implements the three entry points (init, timestep, finalize) that form
/// the boundary between the Fortran marshalling layer and the C++ dycore.
/// Each function uses api_wrap() to convert exceptions into error codes.
///
/// Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.8

#include <mpas_dycore/api.h>
#include <mpas_dycore/error.hpp>
#include <mpas_dycore/state.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/types.hpp>

#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

// ============================================================================
// mpas_dycore_cpp_init
// ============================================================================

int mpas_dycore_cpp_init(
    /* Mesh dimensions */
    int nCells, int nEdges, int nVertices, int maxEdges, int nVertLevels,
    /* Connectivity arrays (zero-based from marshaller) */
    const int* cellsOnEdge,      int cellsOnEdge_d0, int cellsOnEdge_d1,
    const int* verticesOnEdge,   int verticesOnEdge_d0, int verticesOnEdge_d1,
    const int* edgesOnCell,      int edgesOnCell_d0, int edgesOnCell_d1,
    const int* cellsOnCell,      int cellsOnCell_d0, int cellsOnCell_d1,
    const int* nEdgesOnCell_arr, int nEdgesOnCell_len,
    /* Halo CSR descriptors */
    const int* halo_ranks, int halo_n_neighbors,
    const int* halo_counts, int halo_counts_len,
    const int* halo_send_indices, int halo_send_len,
    const int* halo_recv_indices, int halo_recv_len,
    int halo_n_layers,
    /* Configuration */
    double dt, int number_of_sub_steps,
    int config_apply_lbcs,
    const char* config_horiz_mixing,
    double config_h_mom_eddy_visc2,
    double config_h_mom_eddy_visc4,
    double config_h_theta_eddy_visc2,
    /* MPI communicator (Fortran handle) */
    int mpi_comm_fortran,
    /* Error output */
    char* errmsg, int errmsg_len
) {
    using namespace mpas::dycore;

    return api_wrap(errmsg, errmsg_len, [&]() {
        auto& state = get_dycore_state();

        // ---- Validate extents are positive ----
        if (nCells <= 0) {
            throw std::invalid_argument("nCells must be positive, got " + std::to_string(nCells));
        }
        if (nEdges <= 0) {
            throw std::invalid_argument("nEdges must be positive, got " + std::to_string(nEdges));
        }
        if (nVertices <= 0) {
            throw std::invalid_argument("nVertices must be positive, got " + std::to_string(nVertices));
        }
        if (maxEdges <= 0) {
            throw std::invalid_argument("maxEdges must be positive, got " + std::to_string(maxEdges));
        }
        if (nVertLevels <= 0) {
            throw std::invalid_argument("nVertLevels must be positive, got " + std::to_string(nVertLevels));
        }

        // ---- Validate number_of_sub_steps in [2, 24] ----
        if (number_of_sub_steps < 2 || number_of_sub_steps > 24) {
            throw std::invalid_argument(
                "number_of_sub_steps must be in [2, 24], got " +
                std::to_string(number_of_sub_steps));
        }

        // ---- Store mesh dimensions ----
        state.nCells = static_cast<index_type>(nCells);
        state.nEdges = static_cast<index_type>(nEdges);
        state.nVertices = static_cast<index_type>(nVertices);
        state.maxEdges = static_cast<index_type>(maxEdges);
        state.nVertLevels = static_cast<index_type>(nVertLevels);

        // ---- Copy connectivity data into owned storage ----
        // Compute total connectivity storage needed:
        //   cellsOnEdge:    nEdges * 2
        //   verticesOnEdge: nEdges * 2
        //   edgesOnCell:    nCells * maxEdges
        //   cellsOnCell:    nCells * maxEdges
        //   nEdgesOnCell:   nCells
        const std::size_t coe_size = static_cast<std::size_t>(cellsOnEdge_d0) *
                                     static_cast<std::size_t>(cellsOnEdge_d1);
        const std::size_t voe_size = static_cast<std::size_t>(verticesOnEdge_d0) *
                                     static_cast<std::size_t>(verticesOnEdge_d1);
        const std::size_t eoc_size = static_cast<std::size_t>(edgesOnCell_d0) *
                                     static_cast<std::size_t>(edgesOnCell_d1);
        const std::size_t coc_size = static_cast<std::size_t>(cellsOnCell_d0) *
                                     static_cast<std::size_t>(cellsOnCell_d1);
        const std::size_t neoc_size = static_cast<std::size_t>(nEdgesOnCell_len);

        const std::size_t total_conn_size = coe_size + voe_size + eoc_size + coc_size + neoc_size;

        state.connectivity_storage.resize(total_conn_size);

        // Copy each connectivity array into the contiguous block
        index_type* dst = state.connectivity_storage.data();

        // cellsOnEdge
        index_type* coe_ptr = dst;
        std::copy_n(cellsOnEdge, coe_size, coe_ptr);
        dst += coe_size;

        // verticesOnEdge
        index_type* voe_ptr = dst;
        std::copy_n(verticesOnEdge, voe_size, voe_ptr);
        dst += voe_size;

        // edgesOnCell
        index_type* eoc_ptr = dst;
        std::copy_n(edgesOnCell, eoc_size, eoc_ptr);
        dst += eoc_size;

        // cellsOnCell
        index_type* coc_ptr = dst;
        std::copy_n(cellsOnCell, coc_size, coc_ptr);
        dst += coc_size;

        // nEdgesOnCell
        index_type* neoc_ptr = dst;
        std::copy_n(nEdgesOnCell_arr, neoc_size, neoc_ptr);

        // ---- Build ConnectivityView mdspans over owned storage ----
        state.mesh.cellsOnEdge = ConnectivityView(
            coe_ptr,
            static_cast<index_type>(cellsOnEdge_d0),
            static_cast<index_type>(cellsOnEdge_d1));

        state.mesh.verticesOnEdge = ConnectivityView(
            voe_ptr,
            static_cast<index_type>(verticesOnEdge_d0),
            static_cast<index_type>(verticesOnEdge_d1));

        state.mesh.edgesOnCell = ConnectivityView(
            eoc_ptr,
            static_cast<index_type>(edgesOnCell_d0),
            static_cast<index_type>(edgesOnCell_d1));

        state.mesh.cellsOnCell = ConnectivityView(
            coc_ptr,
            static_cast<index_type>(cellsOnCell_d0),
            static_cast<index_type>(cellsOnCell_d1));

        // nEdgesOnCell is 1D
        state.mesh.nEdgesOnCell = std::mdspan<const index_type,
            std::extents<index_type, std::dynamic_extent>>(
            neoc_ptr, static_cast<index_type>(nEdgesOnCell_len));

        // ---- Validate connectivity ----
        auto result = validate_connectivity(
            state.mesh,
            state.nCells, state.nEdges, state.nVertices);

        if (!result.valid) {
            throw std::invalid_argument(
                "Invalid connectivity in table '" + result.table_name +
                "' at [" + std::to_string(result.entry_row) + ", " +
                std::to_string(result.entry_col) + "]: value " +
                std::to_string(result.bad_value) + " exceeds entity count " +
                std::to_string(result.entity_count));
        }

        // ---- Copy halo descriptor data into owned storage ----
        if (halo_n_neighbors > 0) {
            state.halo_ranks_storage.assign(
                halo_ranks, halo_ranks + halo_n_neighbors);
            state.halo_counts_storage.assign(
                halo_counts, halo_counts + halo_counts_len);
            state.halo_send_indices_storage.assign(
                halo_send_indices, halo_send_indices + halo_send_len);
            state.halo_recv_indices_storage.assign(
                halo_recv_indices, halo_recv_indices + halo_recv_len);
        } else {
            state.halo_ranks_storage.clear();
            state.halo_counts_storage.clear();
            state.halo_send_indices_storage.clear();
            state.halo_recv_indices_storage.clear();
        }

        // Build CSRHaloDescriptor views over owned storage
        state.halo_desc.n_neighbors = static_cast<index_type>(halo_n_neighbors);
        state.halo_desc.n_layers = static_cast<index_type>(halo_n_layers);
        state.halo_desc.ranks = std::span<const int>(
            state.halo_ranks_storage.data(),
            state.halo_ranks_storage.size());
        state.halo_desc.counts = std::span<const int>(
            state.halo_counts_storage.data(),
            state.halo_counts_storage.size());
        state.halo_desc.send_indices = std::span<const int>(
            state.halo_send_indices_storage.data(),
            state.halo_send_indices_storage.size());
        state.halo_desc.recv_indices = std::span<const int>(
            state.halo_recv_indices_storage.data(),
            state.halo_recv_indices_storage.size());

        // ---- Convert Fortran MPI communicator handle ----
        state.comm = MPI_Comm_f2c(mpi_comm_fortran);

        // ---- Store configuration ----
        state.dt = dt;
        state.number_of_sub_steps = number_of_sub_steps;

        // ---- Set status to Ready ----
        state.status = DycoreStatus::Ready;
    });
}

// ============================================================================
// mpas_dycore_cpp_timestep
// ============================================================================

int mpas_dycore_cpp_timestep(
    /* Prognostic fields (read-write) */
    double* u,       int u_d0, int u_d1,
    double* theta_m, int theta_m_d0, int theta_m_d1,
    double* rho_zz,  int rho_zz_d0, int rho_zz_d1,
    double* w,       int w_d0, int w_d1,
    /* Scalars (read-write, nScalars x nVertLevels x nCells) */
    double* scalars, int scalars_d0, int scalars_d1, int scalars_d2,
    /* Error output */
    char* errmsg, int errmsg_len
) {
    using namespace mpas::dycore;

    auto& state = get_dycore_state();

    // ---- Uninitialized-state guard ----
    if (state.status != DycoreStatus::Ready) {
        const char* msg = "dycore not initialized";
        if (errmsg != nullptr && errmsg_len > 0) {
            std::strncpy(errmsg, msg, static_cast<std::size_t>(errmsg_len - 1));
            errmsg[errmsg_len - 1] = '\0';
        }
        return 1;
    }

    return api_wrap(errmsg, errmsg_len, [&]() {
        // ---- Construct mdspan views from raw pointers + extent metadata ----
        // For now (stub): validate that fields are non-null and extents match
        // state dimensions. The actual SRK3 integration will be connected in
        // task 19.2.

        if (u == nullptr) {
            throw std::invalid_argument("u field pointer is null");
        }
        if (theta_m == nullptr) {
            throw std::invalid_argument("theta_m field pointer is null");
        }
        if (rho_zz == nullptr) {
            throw std::invalid_argument("rho_zz field pointer is null");
        }
        if (w == nullptr) {
            throw std::invalid_argument("w field pointer is null");
        }
        if (scalars == nullptr) {
            throw std::invalid_argument("scalars field pointer is null");
        }

        // Validate extents match stored dimensions
        if (u_d0 != state.nVertLevels || u_d1 != state.nEdges) {
            throw std::invalid_argument(
                "u extents (" + std::to_string(u_d0) + ", " +
                std::to_string(u_d1) + ") do not match expected (" +
                std::to_string(state.nVertLevels) + ", " +
                std::to_string(state.nEdges) + ")");
        }
        if (theta_m_d0 != state.nVertLevels || theta_m_d1 != state.nCells) {
            throw std::invalid_argument(
                "theta_m extents (" + std::to_string(theta_m_d0) + ", " +
                std::to_string(theta_m_d1) + ") do not match expected (" +
                std::to_string(state.nVertLevels) + ", " +
                std::to_string(state.nCells) + ")");
        }
        if (rho_zz_d0 != state.nVertLevels || rho_zz_d1 != state.nCells) {
            throw std::invalid_argument(
                "rho_zz extents (" + std::to_string(rho_zz_d0) + ", " +
                std::to_string(rho_zz_d1) + ") do not match expected (" +
                std::to_string(state.nVertLevels) + ", " +
                std::to_string(state.nCells) + ")");
        }
        if (w_d0 != state.nVertLevels + 1 || w_d1 != state.nCells) {
            throw std::invalid_argument(
                "w extents (" + std::to_string(w_d0) + ", " +
                std::to_string(w_d1) + ") do not match expected (" +
                std::to_string(state.nVertLevels + 1) + ", " +
                std::to_string(state.nCells) + ")");
        }
        if (scalars_d1 != state.nVertLevels || scalars_d2 != state.nCells) {
            throw std::invalid_argument(
                "scalars extents (" + std::to_string(scalars_d0) + ", " +
                std::to_string(scalars_d1) + ", " +
                std::to_string(scalars_d2) + ") do not match expected (*, " +
                std::to_string(state.nVertLevels) + ", " +
                std::to_string(state.nCells) + ")");
        }

        // Store nScalars from the first call for future use
        state.nScalars = static_cast<index_type>(scalars_d0);

        // Construct mdspan views (used by SRK3 integrator in task 19.2)
        [[maybe_unused]] Field2D<> u_view(
            u,
            static_cast<index_type>(u_d0),
            static_cast<index_type>(u_d1));

        [[maybe_unused]] Field2D<> theta_m_view(
            theta_m,
            static_cast<index_type>(theta_m_d0),
            static_cast<index_type>(theta_m_d1));

        [[maybe_unused]] Field2D<> rho_zz_view(
            rho_zz,
            static_cast<index_type>(rho_zz_d0),
            static_cast<index_type>(rho_zz_d1));

        [[maybe_unused]] Field2D<> w_view(
            w,
            static_cast<index_type>(w_d0),
            static_cast<index_type>(w_d1));

        [[maybe_unused]] Field3D<> scalars_view(
            scalars,
            static_cast<index_type>(scalars_d0),
            static_cast<index_type>(scalars_d1),
            static_cast<index_type>(scalars_d2));

        // TODO(task 19.2): Connect SRK3 integration here.
        // For now this is a validation-only stub that confirms the API
        // contract is met (non-null pointers, correct extents).
    });
}

// ============================================================================
// mpas_dycore_cpp_finalize
// ============================================================================

int mpas_dycore_cpp_finalize(char* errmsg, int errmsg_len) {
    using namespace mpas::dycore;

    return api_wrap(errmsg, errmsg_len, [&]() {
        auto& state = get_dycore_state();

        // ---- Clear workspace ----
        state.workspace.clear();
        state.workspace.shrink_to_fit();

        // ---- Clear connectivity storage ----
        state.connectivity_storage.clear();
        state.connectivity_storage.shrink_to_fit();

        // Reset mesh views to empty
        state.mesh = MeshConnectivity{};

        // ---- Clear halo storage ----
        state.halo_ranks_storage.clear();
        state.halo_ranks_storage.shrink_to_fit();
        state.halo_counts_storage.clear();
        state.halo_counts_storage.shrink_to_fit();
        state.halo_send_indices_storage.clear();
        state.halo_send_indices_storage.shrink_to_fit();
        state.halo_recv_indices_storage.clear();
        state.halo_recv_indices_storage.shrink_to_fit();

        // Reset halo descriptor
        state.halo_desc = CSRHaloDescriptor{};

        // ---- Reset MPI communicator ----
        state.comm = MPI_COMM_NULL;

        // ---- Reset dimensions ----
        state.nCells = 0;
        state.nEdges = 0;
        state.nVertices = 0;
        state.maxEdges = 0;
        state.nVertLevels = 0;
        state.nScalars = 0;

        // ---- Reset configuration ----
        state.dt = 0.0;
        state.number_of_sub_steps = 0;

        // ---- Set status to Uninitialized ----
        state.status = DycoreStatus::Uninitialized;
    });
}
