#include "mpas_dycore/halo_manager.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using mpas::dycore::Config;
using mpas::dycore::ConfigBuilder;
using mpas::dycore::Domain;
using mpas::dycore::Field_Store;
using mpas::dycore::Halo_Manager;
using mpas::dycore::HaloGroupDefinition;
using mpas::dycore::HaloTopology;
using mpas::dycore::Scalar;
using mpas::dycore::build_halo_group_registry;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using FieldStoreT = Field_Store<Scalar, ExecSpace>;

/// Helper: build a default valid config (method = "direct").
Config make_valid_config() {
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

/// Helper: build a config with an invalid exchange method.
Config make_invalid_method_config(const std::string& method) {
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
      .halo_exchange_method(method)
      .build();
}

// ─── Test: construction creates all groups ───────────────────────────────────

TEST(HaloManagerTest, ConstructionCreatesAllRegistryGroups) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();

  Halo_Manager hm(domain, config);

  // Verify all groups from the registry are present.
  auto registry = build_halo_group_registry();
  EXPECT_EQ(hm.num_groups(), registry.size());

  for (const auto& group_def : registry) {
    EXPECT_TRUE(hm.has_group(group_def.group_name))
        << "Missing group: " << group_def.group_name;
  }
}

// ─── Test: group lookup returns correct composition ──────────────────────────

TEST(HaloManagerTest, GroupCompositionMatchesRegistry) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();

  Halo_Manager hm(domain, config);

  // Check a representative dynamics group.
  const auto& group = hm.group("dynamics:exner");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "exner");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
}

TEST(HaloManagerTest, GroupCompositionMultiField) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();

  Halo_Manager hm(domain, config);

  // Check dynamics:theta_m,scalars,pressure_p,rtheta_p group.
  const auto& group = hm.group("dynamics:theta_m,scalars,pressure_p,rtheta_p");
  ASSERT_EQ(group.fields.size(), 4u);
  EXPECT_EQ(group.fields[0].field_name, "theta_m");
  EXPECT_EQ(group.fields[1].field_name, "scalars");
  EXPECT_EQ(group.fields[2].field_name, "pressure_p");
  EXPECT_EQ(group.fields[3].field_name, "rtheta_p");
}

// ─── Test: validate_method accepts valid methods ─────────────────────────────

TEST(HaloManagerTest, ValidateMethodAcceptsDirect) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = ConfigBuilder{}
      .halo_exchange_method("direct")
      .build();

  // Construction should not throw.
  EXPECT_NO_THROW(Halo_Manager(domain, config));
}

TEST(HaloManagerTest, ValidateMethodAcceptsGrouped) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = ConfigBuilder{}
      .halo_exchange_method("grouped")
      .build();

  // Construction should not throw.
  EXPECT_NO_THROW(Halo_Manager(domain, config));
}

// ─── Test: validate_method rejects invalid methods with the method name ──────

TEST(HaloManagerTest, ValidateMethodRejectsUnknownMethod) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_invalid_method_config("scatter");

  // Construction should throw naming the invalid method.
  try {
    Halo_Manager hm(domain, config);
    FAIL() << "Expected std::runtime_error for invalid method";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("scatter"), std::string::npos)
        << "Error message should name the invalid method. Got: " << msg;
  }
}

TEST(HaloManagerTest, ValidateMethodRejectsEmptyMethod) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_invalid_method_config("");

  try {
    Halo_Manager hm(domain, config);
    FAIL() << "Expected std::runtime_error for empty method";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("unrecognized"), std::string::npos)
        << "Error message should indicate unrecognized method. Got: " << msg;
  }
}

// ─── Test: exchange throws for unknown group ─────────────────────────────────

TEST(HaloManagerTest, ExchangeThrowsForUnknownGroup) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();

  Halo_Manager hm(domain, config);

  EXPECT_THROW(hm.exchange("nonexistent:group"), std::runtime_error);
}

// ─── Test: exchange skips unregistered fields gracefully ─────────────────────

TEST(HaloManagerTest, ExchangeSkipsUnregisteredFields) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();

  Halo_Manager hm(domain, config);

  // "dynamics:exner" group references "exner" field which is not in the store.
  // Should not throw — just skip fields that aren't registered yet.
  EXPECT_NO_THROW(hm.exchange("dynamics:exner"));
}

// ─── Test: RAII destruction ──────────────────────────────────────────────────

TEST(HaloManagerTest, DestructorDoesNotThrow) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();

  // Construct and immediately destroy — should release all resources cleanly.
  { Halo_Manager hm(domain, config); }
  // If we get here without a crash or leak (ASan), RAII is working.
  SUCCEED();
}

// ─── Test: gpu_aware_comm flag is respected ──────────────────────────────────

TEST(HaloManagerTest, ConstructionWithGpuAwareComm) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = ConfigBuilder{}
      .gpu_aware_comm(true)
      .halo_exchange_method("direct")
      .build();

  // Should construct without error.
  EXPECT_NO_THROW(Halo_Manager(domain, config));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task 5.3: Comprehensive halo group composition and method error tests
// Validates: Requirements 11.2, 11.8, 11.9, 11.10
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Test: ALL groups in Halo_Manager match the registry composition ─────────
// This verifies that the Halo_Manager faithfully preserves the field membership,
// time_level, and halo_layers for every group (Req 11.2, 11.8).

TEST(HaloManagerTest, AllGroupsMatchRegistryComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();

  Halo_Manager hm(domain, config);
  auto registry = build_halo_group_registry();

  for (const auto& expected : registry) {
    ASSERT_TRUE(hm.has_group(expected.group_name))
        << "Missing group: " << expected.group_name;

    const auto& actual = hm.group(expected.group_name);
    ASSERT_EQ(actual.fields.size(), expected.fields.size())
        << "Field count mismatch for group: " << expected.group_name;

    for (std::size_t i = 0; i < expected.fields.size(); ++i) {
      EXPECT_EQ(actual.fields[i].field_name, expected.fields[i].field_name)
          << "Field name mismatch at index " << i
          << " in group: " << expected.group_name;
      EXPECT_EQ(actual.fields[i].time_level, expected.fields[i].time_level)
          << "Time level mismatch for field '"
          << expected.fields[i].field_name
          << "' in group: " << expected.group_name;
      EXPECT_EQ(actual.fields[i].halo_layers, expected.fields[i].halo_layers)
          << "Halo layers mismatch for field '"
          << expected.fields[i].field_name
          << "' in group: " << expected.group_name;
    }
  }
}

// ─── Test: initialization groups preserved (Req 11.8) ────────────────────────

TEST(HaloManagerTest, InitializationUGroupComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("initialization:u");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "u");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2, 3}));
}

TEST(HaloManagerTest, InitializationPvEdgeRuRwGroupComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("initialization:pv_edge,ru,rw");
  ASSERT_EQ(group.fields.size(), 3u);
  EXPECT_EQ(group.fields[0].field_name, "pv_edge");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(group.fields[1].field_name, "ru");
  EXPECT_EQ(group.fields[1].time_level, 1);
  EXPECT_EQ(group.fields[1].halo_layers, (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(group.fields[2].field_name, "rw");
  EXPECT_EQ(group.fields[2].time_level, 1);
  EXPECT_EQ(group.fields[2].halo_layers, (std::vector<int>{1, 2}));
}

// ─── Test: dynamics groups field/time_level/halo_layers (Req 11.2, 11.8) ─────

TEST(HaloManagerTest, DynamicsRwPRuPRhoPpRthetaPpComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:rw_p,ru_p,rho_pp,rtheta_pp");
  ASSERT_EQ(group.fields.size(), 4u);
  EXPECT_EQ(group.fields[0].field_name, "rw_p");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1}));
  EXPECT_EQ(group.fields[1].field_name, "ru_p");
  EXPECT_EQ(group.fields[1].time_level, 1);
  EXPECT_EQ(group.fields[1].halo_layers, (std::vector<int>{2}));
  EXPECT_EQ(group.fields[2].field_name, "rho_pp");
  EXPECT_EQ(group.fields[2].time_level, 1);
  EXPECT_EQ(group.fields[2].halo_layers, (std::vector<int>{1, 2}));
  EXPECT_EQ(group.fields[3].field_name, "rtheta_pp");
  EXPECT_EQ(group.fields[3].time_level, 1);
  EXPECT_EQ(group.fields[3].halo_layers, (std::vector<int>{2}));
}

TEST(HaloManagerTest, DynamicsWPvEdgeRhoEdgeComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:w,pv_edge,rho_edge");
  ASSERT_EQ(group.fields.size(), 3u);
  EXPECT_EQ(group.fields[0].field_name, "w");
  EXPECT_EQ(group.fields[0].time_level, 2);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
  EXPECT_EQ(group.fields[1].field_name, "pv_edge");
  EXPECT_EQ(group.fields[1].time_level, 1);
  EXPECT_EQ(group.fields[1].halo_layers, (std::vector<int>{1, 2}));
  EXPECT_EQ(group.fields[2].field_name, "rho_edge");
  EXPECT_EQ(group.fields[2].time_level, 1);
  EXPECT_EQ(group.fields[2].halo_layers, (std::vector<int>{1, 2}));
}

TEST(HaloManagerTest, DynamicsWPvEdgeRhoEdgeScalarsComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:w,pv_edge,rho_edge,scalars");
  ASSERT_EQ(group.fields.size(), 4u);
  EXPECT_EQ(group.fields[0].field_name, "w");
  EXPECT_EQ(group.fields[0].time_level, 2);
  EXPECT_EQ(group.fields[1].field_name, "pv_edge");
  EXPECT_EQ(group.fields[1].time_level, 1);
  EXPECT_EQ(group.fields[2].field_name, "rho_edge");
  EXPECT_EQ(group.fields[2].time_level, 1);
  EXPECT_EQ(group.fields[3].field_name, "scalars");
  EXPECT_EQ(group.fields[3].time_level, 2);
  EXPECT_EQ(group.fields[3].halo_layers, (std::vector<int>{1, 2}));
}

TEST(HaloManagerTest, DynamicsThetaMPressurePRthetaPComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:theta_m,pressure_p,rtheta_p");
  ASSERT_EQ(group.fields.size(), 3u);
  EXPECT_EQ(group.fields[0].field_name, "theta_m");
  EXPECT_EQ(group.fields[0].time_level, 2);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
  EXPECT_EQ(group.fields[1].field_name, "pressure_p");
  EXPECT_EQ(group.fields[1].time_level, 1);
  EXPECT_EQ(group.fields[2].field_name, "rtheta_p");
  EXPECT_EQ(group.fields[2].time_level, 1);
}

TEST(HaloManagerTest, DynamicsTendUComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:tend_u");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "tend_u");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1}));
}

TEST(HaloManagerTest, DynamicsRhoPpComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:rho_pp");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "rho_pp");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1}));
}

TEST(HaloManagerTest, DynamicsRthetaPpComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:rtheta_pp");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "rtheta_pp");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1}));
}

TEST(HaloManagerTest, DynamicsU123Composition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:u_123");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "u");
  EXPECT_EQ(group.fields[0].time_level, 2);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2, 3}));
}

TEST(HaloManagerTest, DynamicsU3Composition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:u_3");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "u");
  EXPECT_EQ(group.fields[0].time_level, 2);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{3}));
}

TEST(HaloManagerTest, DynamicsScalarsComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:scalars");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "scalars");
  EXPECT_EQ(group.fields[0].time_level, 2);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
}

TEST(HaloManagerTest, DynamicsScalarsOldComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:scalars_old");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "scalars");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
}

TEST(HaloManagerTest, DynamicsWComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:w");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "w");
  EXPECT_EQ(group.fields[0].time_level, 2);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
}

TEST(HaloManagerTest, DynamicsScaleComposition) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  const auto& group = hm.group("dynamics:scale");
  ASSERT_EQ(group.fields.size(), 1u);
  EXPECT_EQ(group.fields[0].field_name, "scale");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
}

// ─── Test: physics groups via Halo_Manager (Req 11.9) ────────────────────────

#ifdef MPAS_PHYSICS_ENABLED
TEST(HaloManagerTest, PhysicsBltenGroupPresent) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  ASSERT_TRUE(hm.has_group("physics:blten"));
  const auto& group = hm.group("physics:blten");
  ASSERT_EQ(group.fields.size(), 2u);
  EXPECT_EQ(group.fields[0].field_name, "rublten");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
  EXPECT_EQ(group.fields[1].field_name, "rvblten");
  EXPECT_EQ(group.fields[1].time_level, 1);
  EXPECT_EQ(group.fields[1].halo_layers, (std::vector<int>{1, 2}));
}

TEST(HaloManagerTest, PhysicsCutenGroupPresent) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  ASSERT_TRUE(hm.has_group("physics:cuten"));
  const auto& group = hm.group("physics:cuten");
  ASSERT_EQ(group.fields.size(), 2u);
  EXPECT_EQ(group.fields[0].field_name, "rucuten");
  EXPECT_EQ(group.fields[0].time_level, 1);
  EXPECT_EQ(group.fields[0].halo_layers, (std::vector<int>{1, 2}));
  EXPECT_EQ(group.fields[1].field_name, "rvcuten");
  EXPECT_EQ(group.fields[1].time_level, 1);
  EXPECT_EQ(group.fields[1].halo_layers, (std::vector<int>{1, 2}));
}
#else
TEST(HaloManagerTest, PhysicsGroupsAbsentWhenPhysicsDisabled) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  EXPECT_FALSE(hm.has_group("physics:blten"));
  EXPECT_FALSE(hm.has_group("physics:cuten"));
}
#endif

// ─── Test: validate_method error message contains the offending method name ──
// Requirement 11.10: invalid method is reported WITH the offending method name.

TEST(HaloManagerTest, ValidateMethodErrorContainsMethodName_Multiword) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_invalid_method_config("nonblocking_async");

  try {
    Halo_Manager hm(domain, config);
    FAIL() << "Expected std::runtime_error for invalid method";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("nonblocking_async"), std::string::npos)
        << "Error message must name the offending method. Got: " << msg;
  }
}

TEST(HaloManagerTest, ValidateMethodErrorContainsMethodName_Typo) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_invalid_method_config("direcT");

  try {
    Halo_Manager hm(domain, config);
    FAIL() << "Expected std::runtime_error for typo'd method";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("direcT"), std::string::npos)
        << "Error message must contain the exact invalid method string. Got: "
        << msg;
  }
}

TEST(HaloManagerTest, ValidateMethodErrorContainsMethodName_Whitespace) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_invalid_method_config(" direct");

  try {
    Halo_Manager hm(domain, config);
    FAIL() << "Expected std::runtime_error for method with leading space";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    // The leading space makes it invalid; the message should still contain
    // the offending string.
    EXPECT_NE(msg.find("direct"), std::string::npos)
        << "Error message must reference the invalid method. Got: " << msg;
  }
}

// ─── Test: group count matches expected Reference_Model count (Req 11.8) ─────

TEST(HaloManagerTest, GroupCountMatchesExpected) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

#ifdef MPAS_PHYSICS_ENABLED
  // 2 initialization + 15 dynamics + 2 physics = 19
  EXPECT_EQ(hm.num_groups(), 19u);
#else
  // 2 initialization + 15 dynamics = 17
  EXPECT_EQ(hm.num_groups(), 17u);
#endif
}

// ─── Test: group() throws for nonexistent group ──────────────────────────────

TEST(HaloManagerTest, GroupAccessorThrowsForNonexistent) {
  FieldStoreT store;
  Domain domain(store, MPI_COMM_WORLD);
  auto config = make_valid_config();
  Halo_Manager hm(domain, config);

  EXPECT_THROW((void)hm.group("nonexistent:foobar"), std::runtime_error);
}

}  // namespace
