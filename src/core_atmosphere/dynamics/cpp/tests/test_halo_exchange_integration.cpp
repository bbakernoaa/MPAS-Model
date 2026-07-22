#include "mpas_dycore/halo_manager.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <string>
#include <vector>

/// @file test_halo_exchange_integration.cpp
/// @brief Integration tests for halo exchange.
///
/// Validates that halo exchange correctly transfers data between processes,
/// exercising both the GPU-aware (direct device) and host-staged transfer
/// paths. Uses MPI_COMM_SELF with self-send/recv topology for single-process
/// testing, and multi-rank testing when run under mpirun.
///
/// Requirements: 11.3, 11.4, 11.5, 11.6

namespace {

using mpas::dycore::Config;
using mpas::dycore::ConfigBuilder;
using mpas::dycore::Domain;
using mpas::dycore::Field_Store;
using mpas::dycore::Halo_Manager;
using mpas::dycore::HaloGroupDefinition;
using mpas::dycore::HaloNeighborInfo;
using mpas::dycore::HaloTopology;
using mpas::dycore::Scalar;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using FieldStoreT = Field_Store<Scalar, ExecSpace>;
using view2d = typename FieldStoreT::view2d;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a config for integration testing with host-staged transfer.
Config make_host_staged_config() {
  return ConfigBuilder{}
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(1)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .config_mix_full(true)
      .config_les_model("none")
      .config_les_surface("none")
      .config_iau(false)
      .gpu_aware_comm(false)
      .halo_exchange_method("direct")
      .build();
}

/// Build a config for integration testing with GPU-aware transfer.
Config make_gpu_aware_config() {
  return ConfigBuilder{}
      .time_integration_order(3)
      .number_of_sub_steps(6)
      .dynamics_split_steps(1)
      .config_monotonic(true)
      .config_scalar_advection(true)
      .config_apply_lbcs(false)
      .config_mix_full(true)
      .config_les_model("none")
      .config_les_surface("none")
      .config_iau(false)
      .gpu_aware_comm(true)
      .halo_exchange_method("direct")
      .build();
}

/// Build a self-communicating topology for single-process testing.
///
/// In MPI self-communication, the send count must equal the receive count.
/// The Halo_Library's exchange_blocking lays out the view as:
///   [send_region: 0..n_exchange) [recv_region: n_exchange..2*n_exchange)
///
/// After exchange, the recv region contains a copy of the send region.
///
/// Build a HaloNeighborInfo for a neighbor exchanging `count` elements.
///
/// The elements are placed in a single halo layer as indices [0, count).
/// The contiguous Halo_Plan path only consumes the total index count, so the
/// specific index values are immaterial for this self-communication test.
HaloNeighborInfo make_neighbor(int rank, std::size_t count) {
  HaloNeighborInfo info;
  info.rank = rank;
  std::vector<std::size_t> layer;
  layer.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    layer.push_back(i);
  }
  info.layers.push_back(std::move(layer));
  return info;
}

/// @param n_exchange Number of elements to send/receive (must match).
HaloTopology make_self_topology(std::size_t n_exchange) {
  HaloTopology topo;
  topo.cell_send_neighbors = {make_neighbor(0, n_exchange)};
  topo.cell_recv_neighbors = {make_neighbor(0, n_exchange)};
  topo.edge_send_neighbors = {make_neighbor(0, n_exchange)};
  topo.edge_recv_neighbors = {make_neighbor(0, n_exchange)};
  return topo;
}

// ─── Test Fixture ────────────────────────────────────────────────────────────

/// Integration test fixture for halo exchange on a single process.
///
/// Uses a field of shape (kVertLevels, kTotalCells) where kTotalCells is chosen
/// so that the flat layout has room for both send and receive regions:
///   - Send region: first kExchangeCells cells (all levels)
///   - Recv region: next kExchangeCells cells (all levels)
///   - Extra cells: remaining cells (untouched by exchange)
///
/// After self-exchange, the recv region should contain a copy of the send region.
class HaloExchangeIntegrationTest : public ::testing::Test {
 protected:
  static constexpr int kVertLevels = 4;
  /// Number of cells in the exchange (send = recv).
  static constexpr int kExchangeCells = 3;
  /// Total cells in the field (send + recv + extra).
  static constexpr int kTotalCells = kExchangeCells * 2 + 2;
  /// Number of flat elements exchanged.
  static constexpr std::size_t kExchangeElements =
      static_cast<std::size_t>(kExchangeCells) * kVertLevels;

  void SetUp() override {
    // Allocate the "exner" field: shape (kVertLevels, kTotalCells).
    store_.allocate("exner", kVertLevels, kTotalCells);

    // Initialize field with known pattern:
    //   - Send cells [0, kExchangeCells): value = j*100 + k + 1
    //   - Recv cells [kExchangeCells, 2*kExchangeCells): sentinel = -1
    //   - Extra cells: value = 999
    auto field_view = store_.level("exner", 1);
    auto host_mirror = Kokkos::create_mirror_view(field_view);

    for (int j = 0; j < kTotalCells; ++j) {
      for (int k = 0; k < kVertLevels; ++k) {
        if (j < kExchangeCells) {
          host_mirror(k, j) = static_cast<Scalar>(j * 100 + k + 1);
        } else if (j < 2 * kExchangeCells) {
          host_mirror(k, j) = static_cast<Scalar>(-1.0);
        } else {
          host_mirror(k, j) = static_cast<Scalar>(999.0);
        }
      }
    }
    Kokkos::deep_copy(field_view, host_mirror);
  }

  FieldStoreT store_;
};

// ─── Test: Exchange executes without error ───────────────────────────────────

TEST_F(HaloExchangeIntegrationTest, ExchangeExecutesWithoutErrorSingleRank) {
  /// On a single rank with self-communicating topology, exchange should
  /// complete without error. Validates the Halo_Library path is correctly
  /// wired (Req 11.3, 11.4).
  auto topo = make_self_topology(kExchangeElements);
  Domain domain(store_, MPI_COMM_SELF, topo);
  auto config = make_host_staged_config();

  Halo_Manager hm(domain, config);
  EXPECT_NO_THROW(hm.exchange("dynamics:exner"));
}

// ─── Test: Host-staged path transfers data ──────────────────────────────────

TEST_F(HaloExchangeIntegrationTest, HostStagedExchangeTransfersData) {
  /// With GPU-aware comm disabled, the host-staged path copies device data to
  /// host, performs MPI exchange, and copies results back to device (Req 11.6).
  /// After self-exchange, the recv region should be updated from the sentinel.
  auto topo = make_self_topology(kExchangeElements);
  Domain domain(store_, MPI_COMM_SELF, topo);
  auto config = make_host_staged_config();

  Halo_Manager hm(domain, config);
  hm.exchange("dynamics:exner");

  auto field_view = store_.level("exner", 1);
  auto host_mirror = Kokkos::create_mirror_view(field_view);
  Kokkos::deep_copy(host_mirror, field_view);

  // Recv region should no longer be the sentinel (-1).
  bool halo_updated = false;
  for (int j = kExchangeCells; j < 2 * kExchangeCells; ++j) {
    for (int k = 0; k < kVertLevels; ++k) {
      if (host_mirror(k, j) != static_cast<Scalar>(-1.0)) {
        halo_updated = true;
        break;
      }
    }
    if (halo_updated) break;
  }
  EXPECT_TRUE(halo_updated)
      << "Recv region should be updated after host-staged exchange";
}

// ─── Test: GPU-aware path transfers data ────────────────────────────────────

TEST_F(HaloExchangeIntegrationTest, GpuAwareExchangeTransfersData) {
  /// With GPU-aware comm enabled, exchange should transfer device-resident
  /// data without host staging (Req 11.5). On host-only builds, this still
  /// exercises the gpu_aware code path in the Halo_Manager.
  auto topo = make_self_topology(kExchangeElements);
  Domain domain(store_, MPI_COMM_SELF, topo);
  auto config = make_gpu_aware_config();

  Halo_Manager hm(domain, config);
  hm.exchange("dynamics:exner");

  auto field_view = store_.level("exner", 1);
  auto host_mirror = Kokkos::create_mirror_view(field_view);
  Kokkos::deep_copy(host_mirror, field_view);

  // Recv region should no longer be the sentinel (-1).
  bool halo_updated = false;
  for (int j = kExchangeCells; j < 2 * kExchangeCells; ++j) {
    for (int k = 0; k < kVertLevels; ++k) {
      if (host_mirror(k, j) != static_cast<Scalar>(-1.0)) {
        halo_updated = true;
        break;
      }
    }
    if (halo_updated) break;
  }
  EXPECT_TRUE(halo_updated)
      << "Recv region should be updated after GPU-aware exchange";
}

// ─── Test: Halo receives correct owner values ───────────────────────────────

TEST_F(HaloExchangeIntegrationTest, HaloCellsReceiveOwnerValues) {
  /// On a self-communicating topology, the recv region should receive an exact
  /// copy of the send region. Validates data consistency (Req 11.3).
  ///
  /// The Halo_Library sends elements [0, n_exchange) and places them into
  /// elements [n_exchange, 2*n_exchange). In LayoutLeft column-major:
  ///   send = cells [0, kExchangeCells), all levels
  ///   recv = cells [kExchangeCells, 2*kExchangeCells), all levels
  ///
  /// After exchange, cell j in recv should equal cell j in send.
  auto topo = make_self_topology(kExchangeElements);
  Domain domain(store_, MPI_COMM_SELF, topo);
  auto config = make_host_staged_config();

  Halo_Manager hm(domain, config);
  hm.exchange("dynamics:exner");

  auto field_view = store_.level("exner", 1);
  auto host_mirror = Kokkos::create_mirror_view(field_view);
  Kokkos::deep_copy(host_mirror, field_view);

  // Recv cells [kExchangeCells, 2*kExchangeCells) should match
  // send cells [0, kExchangeCells).
  for (int j = 0; j < kExchangeCells; ++j) {
    for (int k = 0; k < kVertLevels; ++k) {
      Scalar sent = host_mirror(k, j);
      Scalar received = host_mirror(k, kExchangeCells + j);
      EXPECT_DOUBLE_EQ(sent, received)
          << "Recv cell (" << k << ", " << kExchangeCells + j
          << ") should match send cell (" << k << ", " << j << ")";
    }
  }
}

// ─── Test: Exchange all fields in a multi-field group ────────────────────────

TEST_F(HaloExchangeIntegrationTest, MultiFieldGroupExchangeAllFields) {
  /// Tests that exchange updates halo regions of EVERY field in a group
  /// (Req 11.3). Uses "dynamics:theta_m,scalars,pressure_p,rtheta_p".

  // Allocate all fields for this group.
  store_.allocate("theta_m", kVertLevels, kTotalCells);
  store_.allocate("scalars", kVertLevels, kTotalCells);
  store_.allocate("pressure_p", kVertLevels, kTotalCells);
  store_.allocate("rtheta_p", kVertLevels, kTotalCells);

  // Fill send regions with distinct patterns, recv regions with sentinel.
  auto fill_field = [&](const std::string& name, Scalar base) {
    auto v = store_.level(name, 1);
    auto h = Kokkos::create_mirror_view(v);
    for (int j = 0; j < kTotalCells; ++j) {
      for (int k = 0; k < kVertLevels; ++k) {
        if (j < kExchangeCells) {
          h(k, j) = base + static_cast<Scalar>(j * 10 + k);
        } else if (j < 2 * kExchangeCells) {
          h(k, j) = static_cast<Scalar>(-999.0);
        } else {
          h(k, j) = static_cast<Scalar>(888.0);
        }
      }
    }
    Kokkos::deep_copy(v, h);
  };

  fill_field("theta_m", 1000.0);
  fill_field("scalars", 2000.0);
  fill_field("pressure_p", 3000.0);
  fill_field("rtheta_p", 4000.0);

  auto topo = make_self_topology(kExchangeElements);
  Domain domain(store_, MPI_COMM_SELF, topo);
  auto config = make_host_staged_config();

  Halo_Manager hm(domain, config);
  hm.exchange("dynamics:theta_m,scalars,pressure_p,rtheta_p");

  // Verify all four fields have their recv regions updated.
  auto verify_field = [&](const std::string& name) {
    auto v = store_.level(name, 1);
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);

    bool updated = false;
    for (int j = kExchangeCells; j < 2 * kExchangeCells; ++j) {
      for (int k = 0; k < kVertLevels; ++k) {
        if (h(k, j) != static_cast<Scalar>(-999.0)) {
          updated = true;
          break;
        }
      }
      if (updated) break;
    }
    EXPECT_TRUE(updated)
        << "Field '" << name
        << "' recv region should be updated after group exchange";
  };

  verify_field("theta_m");
  verify_field("scalars");
  verify_field("pressure_p");
  verify_field("rtheta_p");
}

// ─── Test: Exchange with empty topology (no neighbors) ──────────────────────

TEST_F(HaloExchangeIntegrationTest, EmptyTopologyNoOp) {
  /// When the topology has no neighbors (single process, no partition overlap),
  /// exchange should be a no-op — field data should remain unchanged.
  HaloTopology empty_topo{};
  Domain domain(store_, MPI_COMM_SELF, empty_topo);
  auto config = make_host_staged_config();

  Halo_Manager hm(domain, config);

  auto field_view = store_.level("exner", 1);
  auto before = Kokkos::create_mirror_view(Kokkos::HostSpace{}, field_view);
  Kokkos::deep_copy(before, field_view);

  EXPECT_NO_THROW(hm.exchange("dynamics:exner"));

  auto after = Kokkos::create_mirror_view(Kokkos::HostSpace{}, field_view);
  Kokkos::deep_copy(after, field_view);

  for (int j = 0; j < kTotalCells; ++j) {
    for (int k = 0; k < kVertLevels; ++k) {
      EXPECT_EQ(before(k, j), after(k, j))
          << "Field should be unchanged with empty topology at ("
          << k << ", " << j << ")";
    }
  }
}

// ─── Test: Both paths produce consistent results ────────────────────────────

TEST_F(HaloExchangeIntegrationTest, HostStagedAndGpuAwarePathsConsistent) {
  /// Both the host-staged and GPU-aware code paths should produce the same
  /// result for identical input data (Req 11.5, 11.6 consistency).

  // Set up a second field store with identical data.
  FieldStoreT store2;
  store2.allocate("exner", kVertLevels, kTotalCells);
  auto v1 = store_.level("exner", 1);
  auto v2 = store2.level("exner", 1);
  Kokkos::deep_copy(v2, v1);

  auto topo = make_self_topology(kExchangeElements);

  // Host-staged exchange on store_.
  {
    Domain domain(store_, MPI_COMM_SELF, topo);
    auto config = make_host_staged_config();
    Halo_Manager hm(domain, config);
    hm.exchange("dynamics:exner");
  }

  // GPU-aware exchange on store2.
  {
    Domain domain2(store2, MPI_COMM_SELF, topo);
    auto config = make_gpu_aware_config();
    Halo_Manager hm2(domain2, config);
    hm2.exchange("dynamics:exner");
  }

  auto h1 = Kokkos::create_mirror_view(Kokkos::HostSpace{}, v1);
  auto h2 = Kokkos::create_mirror_view(Kokkos::HostSpace{}, v2);
  Kokkos::deep_copy(h1, v1);
  Kokkos::deep_copy(h2, v2);

  for (int j = 0; j < kTotalCells; ++j) {
    for (int k = 0; k < kVertLevels; ++k) {
      EXPECT_DOUBLE_EQ(h1(k, j), h2(k, j))
          << "Host-staged and GPU-aware paths should produce identical results "
          << "at (" << k << ", " << j << ")";
    }
  }
}

// ─── Test: Exchange preserves send region ───────────────────────────────────

TEST_F(HaloExchangeIntegrationTest, ExchangePreservesSendRegion) {
  /// Halo exchange should not modify the send region. Only the recv region
  /// should be updated.
  auto topo = make_self_topology(kExchangeElements);
  Domain domain(store_, MPI_COMM_SELF, topo);
  auto config = make_host_staged_config();

  auto field_view = store_.level("exner", 1);
  auto before = Kokkos::create_mirror_view(Kokkos::HostSpace{}, field_view);
  Kokkos::deep_copy(before, field_view);

  Halo_Manager hm(domain, config);
  hm.exchange("dynamics:exner");

  auto after = Kokkos::create_mirror_view(Kokkos::HostSpace{}, field_view);
  Kokkos::deep_copy(after, field_view);

  // Send region: cells [0, kExchangeCells) should be unchanged.
  for (int j = 0; j < kExchangeCells; ++j) {
    for (int k = 0; k < kVertLevels; ++k) {
      EXPECT_EQ(before(k, j), after(k, j))
          << "Send region should be unchanged after exchange at ("
          << k << ", " << j << ")";
    }
  }
}

// ─── Multi-rank integration test ────────────────────────────────────────────

/// Test fixture for multi-rank halo exchange (requires >= 2 MPI ranks).
/// When run with a single rank, tests are skipped via GTEST_SKIP.
class HaloExchangeMultiRankTest : public ::testing::Test {
 protected:
  static constexpr int kVertLevels = 4;
  static constexpr int kExchangeCells = 3;
  static constexpr int kTotalPerRank = kExchangeCells * 2 + 2;
  static constexpr std::size_t kExchangeElements =
      static_cast<std::size_t>(kExchangeCells) * kVertLevels;

  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
  }

  int rank_ = 0;
  int size_ = 1;
};

TEST_F(HaloExchangeMultiRankTest, TwoRankExchangeOwnerValues) {
  /// When run with 2+ ranks, verify that halo cells receive the correct
  /// owner values from the neighbor rank (Req 11.3).
  if (size_ < 2) {
    GTEST_SKIP() << "Multi-rank test requires >= 2 MPI processes";
  }

  // Each rank exchanges with its neighbor (rank 0 <-> rank 1).
  // Only the first 2 ranks participate.
  if (rank_ >= 2) {
    GTEST_SKIP() << "Only ranks 0 and 1 participate";
  }

  int neighbor = (rank_ == 0) ? 1 : 0;

  FieldStoreT store;
  store.allocate("exner", kVertLevels, kTotalPerRank);

  // Fill send cells with rank-specific values.
  auto field_view = store.level("exner", 1);
  auto host_mirror = Kokkos::create_mirror_view(field_view);
  for (int j = 0; j < kTotalPerRank; ++j) {
    for (int k = 0; k < kVertLevels; ++k) {
      if (j < kExchangeCells) {
        // Send region: identifiable by rank.
        host_mirror(k, j) = static_cast<Scalar>(
            (rank_ + 1) * 1000 + j * 10 + k);
      } else if (j < 2 * kExchangeCells) {
        // Recv region: sentinel.
        host_mirror(k, j) = static_cast<Scalar>(-1.0);
      } else {
        // Extra cells.
        host_mirror(k, j) = static_cast<Scalar>(777.0);
      }
    }
  }
  Kokkos::deep_copy(field_view, host_mirror);

  // Topology: exchange kExchangeElements with the neighbor.
  HaloTopology topo;
  topo.cell_send_neighbors = {make_neighbor(neighbor, kExchangeElements)};
  topo.cell_recv_neighbors = {make_neighbor(neighbor, kExchangeElements)};

  Domain domain(store, MPI_COMM_WORLD, topo);
  auto config = make_host_staged_config();

  Halo_Manager hm(domain, config);
  hm.exchange("dynamics:exner");

  // Read back and verify recv region.
  Kokkos::deep_copy(host_mirror, field_view);

  // Recv cells should now contain the neighbor's send values.
  for (int j = 0; j < kExchangeCells; ++j) {
    for (int k = 0; k < kVertLevels; ++k) {
      Scalar val = host_mirror(k, kExchangeCells + j);
      Scalar expected = static_cast<Scalar>(
          (neighbor + 1) * 1000 + j * 10 + k);
      EXPECT_DOUBLE_EQ(val, expected)
          << "Rank " << rank_ << ": recv cell (" << k << ", "
          << kExchangeCells + j << ") should contain neighbor's value";
    }
  }
}

}  // namespace
