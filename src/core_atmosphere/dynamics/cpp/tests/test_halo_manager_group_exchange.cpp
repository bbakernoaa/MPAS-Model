#include "mpas_dycore/halo_manager.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <halo/indexed_halo_plan.hpp>

#include <cstddef>
#include <string>
#include <vector>

/// @file test_halo_manager_group_exchange.cpp
/// @brief Example-based unit tests for Halo_Manager named-group exchange.
///
/// Task 6.5. These tests exercise the group-exchange behavior added in tasks
/// 6.2/6.3, where `Halo_Manager::exchange(group_name)` resolves each registry
/// field entry's element kind to the matching indexed plan, reads the field
/// view at the entry's time level, converts the MPAS 1-based `halo_layers` to a
/// 0-based layer subset, and drives `halo::exchange_indexed`.
///
/// All tests run on a single rank using a self-send `HaloTopology` over
/// `MPI_COMM_SELF`, so gather/scatter is verifiable in-process: each element
/// kind's plan gathers from a distinct owned "source" column and scatters into
/// distinct "recv" columns. Observing which recv columns change reveals which
/// plan (element kind), which time level, and which halo layers were used.
///
/// Validates: Requirements 9.2, 9.3, 9.4, 9.5, 9.6

namespace {

using mpas::dycore::Config;
using mpas::dycore::ConfigBuilder;
using mpas::dycore::Domain;
using mpas::dycore::Field_Store;
using mpas::dycore::Halo_Manager;
using mpas::dycore::HaloNeighborInfo;
using mpas::dycore::HaloTopology;
using mpas::dycore::Scalar;
using mpas::dycore::build_halo_group_registry;
using mpas::dycore::find_halo_group;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using FieldStoreT = Field_Store<Scalar, ExecSpace>;

// ─── Layout constants ────────────────────────────────────────────────────────

constexpr int kLevels = 2;   ///< Vertical levels per field (rank-2 columns).
constexpr int kCols = 40;    ///< Number of element columns per field.

constexpr Scalar kSentinel = static_cast<Scalar>(-1.0);  ///< Untouched marker.
constexpr Scalar kSource = static_cast<Scalar>(5.0);     ///< Owned source value.
constexpr Scalar kOtherSource =
    static_cast<Scalar>(99.0);  ///< Distinct source used to probe time level.

// Owned "source" columns gathered by each element kind's send plan.
constexpr std::size_t kCellSrc = 0;
constexpr std::size_t kEdgeSrc = 1;
constexpr std::size_t kVertexSrc = 2;

// Halo "recv" columns scattered into by each kind's recv plan, one per layer.
// Cell plan has 2 layers, edge plan 3 layers, vertex plan 1 layer.
const std::vector<std::size_t> kCellRecvCols = {10, 11};
const std::vector<std::size_t> kEdgeRecvCols = {20, 21, 22};
const std::vector<std::size_t> kVertexRecvCols = {30};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a default valid config (method = "direct", host-staged path).
Config make_valid_config() {
  return ConfigBuilder{}
      .gpu_aware_comm(false)
      .halo_exchange_method("direct")
      .build();
}

/// Build a HaloNeighborInfo for `rank` with the given per-layer index lists.
HaloNeighborInfo make_info(int rank,
                           std::vector<std::vector<std::size_t>> layers) {
  HaloNeighborInfo info;
  info.rank = rank;
  info.layers = std::move(layers);
  return info;
}

/// Build a single-rank self-send topology for all three element kinds.
///
/// Each kind gathers from its own source column and scatters into its own recv
/// columns (one per halo layer). Layer counts differ per kind (cell=2, edge=3,
/// vertex=1) so the built plans are also distinguishable by `num_layers()`.
/// Per selected layer the send and recv index counts match (1 each), which is
/// required for the self MPI send/recv to complete.
HaloTopology make_multi_kind_topology() {
  HaloTopology topo;

  // Cell: 2 layers -> recv cols 10 (layer 0 / MPAS layer 1), 11 (layer 1).
  topo.cell_send_neighbors = {make_info(0, {{kCellSrc}, {kCellSrc}})};
  topo.cell_recv_neighbors = {make_info(0, {{10}, {11}})};

  // Edge: 3 layers -> recv cols 20, 21, 22 (MPAS layers 1, 2, 3).
  topo.edge_send_neighbors = {
      make_info(0, {{kEdgeSrc}, {kEdgeSrc}, {kEdgeSrc}})};
  topo.edge_recv_neighbors = {make_info(0, {{20}, {21}, {22}})};

  // Vertex: 1 layer -> recv col 30 (MPAS layer 1).
  topo.vertex_send_neighbors = {make_info(0, {{kVertexSrc}})};
  topo.vertex_recv_neighbors = {make_info(0, {{30}})};

  return topo;
}

/// Allocate a time-leveled field and fill every level with source/sentinel
/// values: source columns hold `source_value`, all other columns (including the
/// recv columns) hold the sentinel.
void allocate_probe_field(FieldStoreT& store, const std::string& name,
                          int n_levels, Scalar source_value = kSource) {
  store.allocate(name, kLevels, kCols, n_levels);
  for (int tl = 1; tl <= n_levels; ++tl) {
    auto v = store.level(name, tl);
    auto h = Kokkos::create_mirror_view(v);
    for (int c = 0; c < kCols; ++c) {
      for (int k = 0; k < kLevels; ++k) {
        h(k, c) = kSentinel;
      }
    }
    for (std::size_t src : {kCellSrc, kEdgeSrc, kVertexSrc}) {
      for (int k = 0; k < kLevels; ++k) {
        h(k, static_cast<int>(src)) = source_value;
      }
    }
    Kokkos::deep_copy(v, h);
  }
}

/// Copy a field view's time level to a host mirror for inspection.
auto read_level(const FieldStoreT& store, const std::string& name,
                int time_level) {
  auto v = store.level(name, time_level);
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  return h;
}

/// Count how many of the given columns were modified away from the sentinel.
template <typename HostView>
int count_changed(const HostView& h, const std::vector<std::size_t>& cols) {
  int changed = 0;
  for (std::size_t col : cols) {
    bool col_changed = false;
    for (int k = 0; k < kLevels; ++k) {
      if (h(k, static_cast<int>(col)) != kSentinel) {
        col_changed = true;
      }
    }
    if (col_changed) {
      ++changed;
    }
  }
  return changed;
}

/// Look up a field entry's time level within a named registry group.
int time_level_of(const std::string& group_name, const std::string& field) {
  auto registry = build_halo_group_registry();
  const auto* group = find_halo_group(registry, group_name);
  if (group == nullptr) {
    return 1;
  }
  for (const auto& entry : group->fields) {
    if (entry.field_name == field) {
      return entry.time_level;
    }
  }
  return 1;
}

/// Exchange `group_name` with only `field_name` allocated, then assert which
/// element-kind recv columns changed. Because unlisted fields are skipped
/// (Req 9.6), only the single allocated field is exchanged, so the recv columns
/// that change reveal which plan (element kind) the field was routed to.
///
/// Validates: Requirement 9.2
void expect_field_routes_to(const std::string& field_name,
                            const std::string& group_name,
                            halo::Element_Kind expected_kind) {
  const int tl = time_level_of(group_name, field_name);
  const int n_levels = (tl >= 2) ? 2 : 1;

  FieldStoreT store;
  allocate_probe_field(store, field_name, n_levels);
  auto topo = make_multi_kind_topology();
  Domain domain(store, MPI_COMM_SELF, topo);

  Halo_Manager hm(domain, make_valid_config());
  hm.exchange(group_name);

  auto h = read_level(store, field_name, tl);
  const int cell_changed = count_changed(h, kCellRecvCols);
  const int edge_changed = count_changed(h, kEdgeRecvCols);
  const int vertex_changed = count_changed(h, kVertexRecvCols);

  switch (expected_kind) {
    case halo::Element_Kind::edge:
      EXPECT_GT(edge_changed, 0)
          << "'" << field_name << "' should route to the EDGE plan";
      EXPECT_EQ(cell_changed, 0)
          << "'" << field_name << "' must not touch the cell plan";
      EXPECT_EQ(vertex_changed, 0)
          << "'" << field_name << "' must not touch the vertex plan";
      break;
    case halo::Element_Kind::vertex:
      EXPECT_GT(vertex_changed, 0)
          << "'" << field_name << "' should route to the VERTEX plan";
      EXPECT_EQ(cell_changed, 0)
          << "'" << field_name << "' must not touch the cell plan";
      EXPECT_EQ(edge_changed, 0)
          << "'" << field_name << "' must not touch the edge plan";
      break;
    case halo::Element_Kind::cell:
    case halo::Element_Kind::generic:
    default:
      EXPECT_GT(cell_changed, 0)
          << "'" << field_name << "' should route to the CELL plan";
      EXPECT_EQ(edge_changed, 0)
          << "'" << field_name << "' must not touch the edge plan";
      EXPECT_EQ(vertex_changed, 0)
          << "'" << field_name << "' must not touch the vertex plan";
      break;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Req 9.2 — Element-kind mapping per representative field
// ═══════════════════════════════════════════════════════════════════════════

// Edge-based fields (normal velocities and edge-staggered quantities) must be
// exchanged with the edge plan. Each field below is allocated alone inside a
// group that references it; the remaining group fields are skipped.

TEST(HaloManagerGroupExchangeTest, EdgeFieldU_RoutesToEdgePlan) {
  expect_field_routes_to("u", "initialization:u", halo::Element_Kind::edge);
}

TEST(HaloManagerGroupExchangeTest, EdgeFieldRu_RoutesToEdgePlan) {
  expect_field_routes_to("ru", "initialization:pv_edge,ru,rw",
                         halo::Element_Kind::edge);
}

TEST(HaloManagerGroupExchangeTest, EdgeFieldRuP_RoutesToEdgePlan) {
  expect_field_routes_to("ru_p", "dynamics:rw_p,ru_p,rho_pp,rtheta_pp",
                         halo::Element_Kind::edge);
}

TEST(HaloManagerGroupExchangeTest, EdgeFieldPvEdge_RoutesToEdgePlan) {
  expect_field_routes_to("pv_edge", "dynamics:w,pv_edge,rho_edge",
                         halo::Element_Kind::edge);
}

TEST(HaloManagerGroupExchangeTest, EdgeFieldRhoEdge_RoutesToEdgePlan) {
  expect_field_routes_to("rho_edge", "dynamics:w,pv_edge,rho_edge",
                         halo::Element_Kind::edge);
}

TEST(HaloManagerGroupExchangeTest, EdgeFieldTendU_RoutesToEdgePlan) {
  expect_field_routes_to("tend_u", "dynamics:tend_u",
                         halo::Element_Kind::edge);
}

// Cell-based fields (the default) must be exchanged with the cell plan.

TEST(HaloManagerGroupExchangeTest, CellFieldExner_RoutesToCellPlan) {
  expect_field_routes_to("exner", "dynamics:exner",
                         halo::Element_Kind::cell);
}

TEST(HaloManagerGroupExchangeTest, CellFieldThetaM_RoutesToCellPlan) {
  expect_field_routes_to("theta_m", "dynamics:theta_m,scalars,pressure_p,rtheta_p",
                         halo::Element_Kind::cell);
}

TEST(HaloManagerGroupExchangeTest, CellFieldPressureP_RoutesToCellPlan) {
  expect_field_routes_to("pressure_p",
                         "dynamics:theta_m,scalars,pressure_p,rtheta_p",
                         halo::Element_Kind::cell);
}

TEST(HaloManagerGroupExchangeTest, CellFieldW_RoutesToCellPlan) {
  expect_field_routes_to("w", "dynamics:w", halo::Element_Kind::cell);
}

// The registry defines no group that references a vertex field (pv_vertex /
// vorticity), so vertex routing cannot be exercised through `exchange()`.
// Instead verify that a distinct, correctly-kinded vertex plan is built and
// selectable (Req 9.1/9.2 plan-per-kind), alongside the cell and edge plans.
TEST(HaloManagerGroupExchangeTest, PerElementKindPlansAreDistinctAndKinded) {
  FieldStoreT store;
  auto topo = make_multi_kind_topology();
  Domain domain(store, MPI_COMM_SELF, topo);
  Halo_Manager hm(domain, make_valid_config());

  const auto* cell = hm.indexed_plan(halo::Element_Kind::cell);
  const auto* edge = hm.indexed_plan(halo::Element_Kind::edge);
  const auto* vertex = hm.indexed_plan(halo::Element_Kind::vertex);

  ASSERT_NE(cell, nullptr);
  ASSERT_NE(edge, nullptr);
  ASSERT_NE(vertex, nullptr);

  EXPECT_EQ(cell->element_kind(), halo::Element_Kind::cell);
  EXPECT_EQ(edge->element_kind(), halo::Element_Kind::edge);
  EXPECT_EQ(vertex->element_kind(), halo::Element_Kind::vertex);

  // Distinct objects, and distinguishable by their per-kind layer counts.
  EXPECT_NE(cell, edge);
  EXPECT_NE(edge, vertex);
  EXPECT_NE(cell, vertex);
  EXPECT_EQ(cell->num_layers(), 2u);
  EXPECT_EQ(edge->num_layers(), 3u);
  EXPECT_EQ(vertex->num_layers(), 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Req 9.3 — Time level forwarded from the registry
// ═══════════════════════════════════════════════════════════════════════════

// Group "dynamics:w" exchanges field "w" at time level 2. With both time levels
// allocated, only time level 2's recv columns should change; time level 1 must
// remain untouched, proving the exchange reads/writes the entry's time level.
TEST(HaloManagerGroupExchangeTest, TimeLevelForwardedFromRegistry) {
  ASSERT_EQ(time_level_of("dynamics:w", "w"), 2);

  FieldStoreT store;
  store.allocate("w", kLevels, kCols, 2);

  // Time level 1 gets a distinct source value; time level 2 the canonical one.
  auto fill = [&](int tl, Scalar source_value) {
    auto v = store.level("w", tl);
    auto h = Kokkos::create_mirror_view(v);
    for (int c = 0; c < kCols; ++c) {
      for (int k = 0; k < kLevels; ++k) {
        h(k, c) = kSentinel;
      }
    }
    for (int k = 0; k < kLevels; ++k) {
      h(k, static_cast<int>(kCellSrc)) = source_value;
    }
    Kokkos::deep_copy(v, h);
  };
  fill(1, kOtherSource);
  fill(2, kSource);

  auto topo = make_multi_kind_topology();
  Domain domain(store, MPI_COMM_SELF, topo);
  Halo_Manager hm(domain, make_valid_config());
  hm.exchange("dynamics:w");

  auto h1 = read_level(store, "w", 1);
  auto h2 = read_level(store, "w", 2);

  // Time level 1 was never accessed: its recv columns remain the sentinel.
  EXPECT_EQ(count_changed(h1, kCellRecvCols), 0)
      << "Time level 1 must be untouched when the entry's time level is 2";

  // Time level 2 was exchanged: recv columns hold the level-2 source value.
  EXPECT_EQ(count_changed(h2, kCellRecvCols),
            static_cast<int>(kCellRecvCols.size()));
  for (std::size_t col : kCellRecvCols) {
    for (int k = 0; k < kLevels; ++k) {
      EXPECT_EQ(h2(k, static_cast<int>(col)), kSource)
          << "Time-level-2 recv col " << col << " should hold the level-2 "
          << "source value";
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Req 9.4 — Halo-layer subset forwarded 0-based
// ═══════════════════════════════════════════════════════════════════════════

// Group "dynamics:u_3" exchanges "u" (edge) with MPAS halo_layers {3}. After
// 0-based conversion this selects layer index 2 only, so edge recv col 22 (the
// third layer) changes while cols 20 and 21 stay untouched. If the 1-based
// value were used directly (index 3) no layer would match and nothing would
// change — so this pins down the "- 1" conversion.
TEST(HaloManagerGroupExchangeTest, LayerSubsetLayer3ConvertedToIndex2) {
  ASSERT_EQ(time_level_of("dynamics:u_3", "u"), 2);

  FieldStoreT store;
  allocate_probe_field(store, "u", 2);
  auto topo = make_multi_kind_topology();
  Domain domain(store, MPI_COMM_SELF, topo);
  Halo_Manager hm(domain, make_valid_config());
  hm.exchange("dynamics:u_3");

  auto h = read_level(store, "u", 2);
  // Only the third edge layer (col 22) should be updated.
  EXPECT_NE(h(0, 22), kSentinel) << "MPAS layer 3 -> 0-based index 2 (col 22)";
  EXPECT_EQ(h(0, 20), kSentinel) << "Layer 1 (col 20) must be excluded";
  EXPECT_EQ(h(0, 21), kSentinel) << "Layer 2 (col 21) must be excluded";
}

// Group "dynamics:tend_u" exchanges "tend_u" (edge) with halo_layers {1}. After
// conversion this selects layer index 0 only: edge recv col 20 changes; cols 21
// and 22 stay untouched. Confirms MPAS layer 1 -> 0-based index 0.
TEST(HaloManagerGroupExchangeTest, LayerSubsetLayer1ConvertedToIndex0) {
  FieldStoreT store;
  allocate_probe_field(store, "tend_u", 1);
  auto topo = make_multi_kind_topology();
  Domain domain(store, MPI_COMM_SELF, topo);
  Halo_Manager hm(domain, make_valid_config());
  hm.exchange("dynamics:tend_u");

  auto h = read_level(store, "tend_u", 1);
  EXPECT_NE(h(0, 20), kSentinel) << "MPAS layer 1 -> 0-based index 0 (col 20)";
  EXPECT_EQ(h(0, 21), kSentinel) << "Layer 2 (col 21) must be excluded";
  EXPECT_EQ(h(0, 22), kSentinel) << "Layer 3 (col 22) must be excluded";
}

// ═══════════════════════════════════════════════════════════════════════════
// Req 9.5 — Unknown group raises an error naming the group
// ═══════════════════════════════════════════════════════════════════════════

TEST(HaloManagerGroupExchangeTest, UnknownGroupThrowsNamingTheGroup) {
  FieldStoreT store;
  auto topo = make_multi_kind_topology();
  Domain domain(store, MPI_COMM_SELF, topo);
  Halo_Manager hm(domain, make_valid_config());

  try {
    hm.exchange("does:not:exist");
    FAIL() << "Expected std::runtime_error for an unknown group";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("does:not:exist"), std::string::npos)
        << "Error message should name the unknown group. Got: " << msg;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Req 9.6 — Missing fields are skipped without error
// ═══════════════════════════════════════════════════════════════════════════

// A group whose fields are all absent from the store is a no-op, not an error.
TEST(HaloManagerGroupExchangeTest, GroupWithAllFieldsMissingIsSkipped) {
  FieldStoreT store;  // no fields allocated
  auto topo = make_multi_kind_topology();
  Domain domain(store, MPI_COMM_SELF, topo);
  Halo_Manager hm(domain, make_valid_config());

  EXPECT_NO_THROW(
      hm.exchange("dynamics:theta_m,scalars,pressure_p,rtheta_p"));
}

// When only some group fields are present, the present field is exchanged and
// the absent fields are skipped without error.
TEST(HaloManagerGroupExchangeTest, PartialGroupExchangesPresentSkipsMissing) {
  FieldStoreT store;
  // Only theta_m is allocated; scalars, pressure_p, rtheta_p are missing.
  allocate_probe_field(store, "theta_m", 1);
  auto topo = make_multi_kind_topology();
  Domain domain(store, MPI_COMM_SELF, topo);
  Halo_Manager hm(domain, make_valid_config());

  EXPECT_NO_THROW(
      hm.exchange("dynamics:theta_m,scalars,pressure_p,rtheta_p"));

  // theta_m (cell) was present, so its cell recv columns are updated.
  auto h = read_level(store, "theta_m", 1);
  EXPECT_GT(count_changed(h, kCellRecvCols), 0)
      << "The present field should still be exchanged";
}

}  // namespace
