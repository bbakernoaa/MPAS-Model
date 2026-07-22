/// @file test_halo_manager_plan_property.cpp
/// @brief Property-based test: Halo_Manager builds indexed plans faithfully
///        from a randomly generated HaloTopology.
///
/// Feature: cpp-dycore-halo-exchange, Property 7
///
/// **Validates: Requirements 9.1**
///
/// For a randomly generated HaloTopology (per-element-kind send/recv
/// HaloNeighborInfo lists, each carrying per-halo-layer 0-based index lists),
/// constructing a Halo_Manager yields, for each element kind
/// (cell / edge / vertex):
///   * a nullptr Indexed_Halo_Plan when that kind has no send and no recv
///     neighbors (the single-rank / neighborless case), or
///   * an Indexed_Halo_Plan whose element kind, send neighbor ranks and
///     per-layer index lists, and recv neighbor ranks and per-layer index
///     lists exactly match the supplied topology.
///
/// The test runs under a single MPI rank (MPI_COMM_WORLD), so the valid
/// neighbor-rank range is [0, comm_size). Neighbor ranks are drawn from that
/// range and kept distinct within each list to satisfy the Indexed_Halo_Plan
/// construction contract. Even under a single rank this exercises the empty
/// (nullptr) case and the single-neighbor case with arbitrary per-layer index
/// lists, which is exactly the plan-construction faithfulness under test.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "mpas_dycore/config.hpp"
#include "mpas_dycore/halo_manager.hpp"

namespace {

using mpas::dycore::Config;
using mpas::dycore::ConfigBuilder;
using mpas::dycore::Domain;
using mpas::dycore::Field_Store;
using mpas::dycore::Halo_Manager;
using mpas::dycore::HaloNeighborInfo;
using mpas::dycore::HaloTopology;
using mpas::dycore::Scalar;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using FieldStoreT = Field_Store<Scalar, ExecSpace>;

/// Build a default valid config (method = "direct") for Halo_Manager.
Config make_valid_config() {
  return ConfigBuilder{}.halo_exchange_method("direct").build();
}

/// Generate one neighbor's per-layer 0-based index lists.
///
/// Produces between 0 and 3 halo layers, each holding between 0 and 5 indices
/// drawn from [0, 1000). Empty layers and empty per-layer lists are valid and
/// intentionally exercised.
std::vector<std::vector<std::size_t>> gen_layers() {
  const int n_layers = *rc::gen::inRange(0, 4);
  std::vector<std::vector<std::size_t>> layers;
  layers.reserve(static_cast<std::size_t>(n_layers));
  for (int l = 0; l < n_layers; ++l) {
    const int count = *rc::gen::inRange(0, 6);
    std::vector<std::size_t> indices;
    indices.reserve(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
      indices.push_back(
          static_cast<std::size_t>(*rc::gen::inRange(0, 1000)));
    }
    layers.push_back(std::move(indices));
  }
  return layers;
}

/// Generate a neighbor list with distinct ranks drawn from [0, comm_size).
///
/// Each rank in the valid range is independently included, guaranteeing rank
/// uniqueness within the list (required by Indexed_Halo_Plan construction).
std::vector<HaloNeighborInfo> gen_neighbors(int comm_size) {
  std::vector<HaloNeighborInfo> neighbors;
  for (int r = 0; r < comm_size; ++r) {
    if (*rc::gen::arbitrary<bool>()) {
      HaloNeighborInfo info;
      info.rank = r;
      info.layers = gen_layers();
      neighbors.push_back(std::move(info));
    }
  }
  return neighbors;
}

/// Assert that a built plan faithfully mirrors the supplied topology lists.
void check_plan(const halo::Indexed_Halo_Plan* plan,
                const std::vector<HaloNeighborInfo>& send,
                const std::vector<HaloNeighborInfo>& recv,
                halo::Element_Kind kind) {
  if (send.empty() && recv.empty()) {
    // No neighbors for this element kind -> no plan (single-rank case).
    RC_ASSERT(plan == nullptr);
    return;
  }

  RC_ASSERT(plan != nullptr);
  RC_ASSERT(plan->element_kind() == kind);

  // Send neighbors: ranks and per-layer index lists must match, in order.
  const auto send_info = plan->send_info();
  RC_ASSERT(send_info.size() == send.size());
  for (std::size_t i = 0; i < send.size(); ++i) {
    RC_ASSERT(send_info[i].rank == send[i].rank);
    RC_ASSERT(send_info[i].layers == send[i].layers);
  }

  // Recv neighbors: ranks and per-layer index lists must match, in order.
  const auto recv_info = plan->recv_info();
  RC_ASSERT(recv_info.size() == recv.size());
  for (std::size_t i = 0; i < recv.size(); ++i) {
    RC_ASSERT(recv_info[i].rank == recv[i].rank);
    RC_ASSERT(recv_info[i].layers == recv[i].layers);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Property 7: Halo_Manager builds plans faithfully from topology
// ---------------------------------------------------------------------------
RC_GTEST_PROP(HaloManagerPlanProperty,
              BuildsPlansFaithfullyFromTopology,
              ()) {
  // Feature: cpp-dycore-halo-exchange, Property 7
  int comm_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  // Randomly generate a full per-element-kind topology.
  HaloTopology topo;
  topo.cell_send_neighbors = gen_neighbors(comm_size);
  topo.cell_recv_neighbors = gen_neighbors(comm_size);
  topo.edge_send_neighbors = gen_neighbors(comm_size);
  topo.edge_recv_neighbors = gen_neighbors(comm_size);
  topo.vertex_send_neighbors = gen_neighbors(comm_size);
  topo.vertex_recv_neighbors = gen_neighbors(comm_size);

  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD, topo);
  auto config = make_valid_config();

  Halo_Manager hm(domain, config);

  // Each element kind's plan must mirror its topology lists exactly.
  check_plan(hm.indexed_plan(halo::Element_Kind::cell),
             topo.cell_send_neighbors, topo.cell_recv_neighbors,
             halo::Element_Kind::cell);
  check_plan(hm.indexed_plan(halo::Element_Kind::edge),
             topo.edge_send_neighbors, topo.edge_recv_neighbors,
             halo::Element_Kind::edge);
  check_plan(hm.indexed_plan(halo::Element_Kind::vertex),
             topo.vertex_send_neighbors, topo.vertex_recv_neighbors,
             halo::Element_Kind::vertex);
}
