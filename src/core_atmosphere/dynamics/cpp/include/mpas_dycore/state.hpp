#pragma once

/// @file state.hpp
/// @brief Global dycore state management for the MPAS dynamical core C API.
///
/// Defines the DycoreState struct that holds all runtime state for the
/// dynamical core: mesh dimensions, MPI communicator, owned workspace memory,
/// mesh connectivity, halo exchange state, and configuration parameters.
/// A single global instance is managed through get_dycore_state().

#include <mpas_dycore/types.hpp>
#include <mpas_dycore/mesh.hpp>
#include <mpas_dycore/halo.hpp>

#include <mpi.h>
#include <vector>

namespace mpas::dycore {

// ============================================================================
// DycoreStatus: lifecycle state of the dynamical core
// ============================================================================

/// @brief Lifecycle status of the dynamical core instance.
///
/// Tracks the initialization state of the global DycoreState:
/// - Uninitialized: before mpas_dycore_init() has been called
/// - Ready: after successful initialization, ready for timesteps
/// - Error: an unrecoverable error has occurred
enum class DycoreStatus {
    Uninitialized,  ///< Not yet initialized
    Ready,          ///< Initialized and ready for computation
    Error           ///< Unrecoverable error state
};

// ============================================================================
// DycoreState: global singleton state
// ============================================================================

/// @brief Complete runtime state for the MPAS dynamical core.
///
/// Holds all data needed by the C API: mesh dimensions, MPI communicator,
/// owned workspace memory for scratch arrays, mesh connectivity (as both
/// mdspan views and owned storage), halo exchange objects, and configuration
/// parameters. A single global instance is accessed via get_dycore_state().
struct DycoreState {
    /// Current lifecycle status.
    DycoreStatus status = DycoreStatus::Uninitialized;

    // ---- Mesh dimensions ----

    /// Number of cells in the local partition.
    index_type nCells = 0;

    /// Number of edges in the local partition.
    index_type nEdges = 0;

    /// Number of vertices in the local partition.
    index_type nVertices = 0;

    /// Maximum number of edges per cell (determines connectivity table width).
    index_type maxEdges = 0;

    /// Number of vertical levels.
    index_type nVertLevels = 0;

    /// Number of scalar tracers.
    index_type nScalars = 0;

    // ---- MPI communicator ----

    /// MPI communicator for this dycore instance.
    MPI_Comm comm = MPI_COMM_NULL;

    // ---- Owned workspace memory ----

    /// Scratch arrays for timestep computation (single contiguous allocation).
    std::vector<real_type> workspace;

    // ---- Mesh connectivity ----

    /// Non-owning mdspan views over connectivity_storage.
    MeshConnectivity mesh;

    /// Single contiguous block of connectivity data (owned).
    std::vector<index_type> connectivity_storage;

    // ---- Halo exchange ----

    /// Halo exchange manager.
    HaloExchange halo;

    /// CSR halo descriptor (views into halo storage vectors below).
    CSRHaloDescriptor halo_desc{};

    /// Owned storage for neighbor ranks.
    std::vector<int> halo_ranks_storage;

    /// Owned storage for per-neighbor counts.
    std::vector<int> halo_counts_storage;

    /// Owned storage for send index lists.
    std::vector<int> halo_send_indices_storage;

    /// Owned storage for receive index lists.
    std::vector<int> halo_recv_indices_storage;

    // ---- Configuration parameters ----

    /// Timestep size in seconds (to be extended in task 19.1).
    real_type dt = 0.0;

    /// Number of acoustic sub-steps per RK3 stage.
    int number_of_sub_steps = 0;
};

// ============================================================================
// Global singleton accessor
// ============================================================================

/// @brief Access the global singleton DycoreState instance.
///
/// The C API manages a single DycoreState instance for the lifetime of the
/// program. This function returns a mutable reference to that instance.
/// Thread safety: the caller is responsible for synchronization if multiple
/// threads access the state concurrently (in practice, only the main thread
/// calls C API functions).
///
/// @return Mutable reference to the global DycoreState.
DycoreState& get_dycore_state();

} // namespace mpas::dycore
