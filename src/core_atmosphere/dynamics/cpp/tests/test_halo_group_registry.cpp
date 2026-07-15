#include "mpas_dycore/halo_group_registry.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

using mpas::dycore::HaloFieldEntry;
using mpas::dycore::HaloGroupDefinition;
using mpas::dycore::build_halo_group_registry;
using mpas::dycore::find_halo_group;

class HaloGroupRegistryTest : public ::testing::Test {
protected:
  std::vector<HaloGroupDefinition> registry = build_halo_group_registry();
};

// Helper: check that a specific field entry matches expected values
void expect_field(const HaloFieldEntry& entry,
                  const std::string& expected_name,
                  int expected_time_level,
                  const std::vector<int>& expected_layers) {
  EXPECT_EQ(entry.field_name, expected_name);
  EXPECT_EQ(entry.time_level, expected_time_level);
  EXPECT_EQ(entry.halo_layers, expected_layers);
}

TEST_F(HaloGroupRegistryTest, ContainsAllDynamicsAndInitGroups) {
  // The minimum set of groups (without physics) is 18
  const std::vector<std::string> expected_groups = {
      "initialization:u",
      "initialization:pv_edge,ru,rw",
      "dynamics:theta_m,scalars,pressure_p,rtheta_p",
      "dynamics:rw_p,ru_p,rho_pp,rtheta_pp",
      "dynamics:w,pv_edge,rho_edge",
      "dynamics:w,pv_edge,rho_edge,scalars",
      "dynamics:theta_m,pressure_p,rtheta_p",
      "dynamics:exner",
      "dynamics:tend_u",
      "dynamics:rho_pp",
      "dynamics:rtheta_pp",
      "dynamics:u_123",
      "dynamics:u_3",
      "dynamics:scalars",
      "dynamics:scalars_old",
      "dynamics:w",
      "dynamics:scale",
  };

  for (const auto& name : expected_groups) {
    EXPECT_NE(find_halo_group(registry, name), nullptr)
        << "Expected group '" << name << "' not found in registry";
  }
}

TEST_F(HaloGroupRegistryTest, InitializationU) {
  const auto* group = find_halo_group(registry, "initialization:u");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "u", 1, {1, 2, 3});
}

TEST_F(HaloGroupRegistryTest, InitializationPvEdgeRuRw) {
  const auto* group = find_halo_group(registry, "initialization:pv_edge,ru,rw");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 3u);
  expect_field(group->fields[0], "pv_edge", 1, {1, 2, 3});
  expect_field(group->fields[1], "ru", 1, {1, 2, 3});
  expect_field(group->fields[2], "rw", 1, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsThetaMScalarsPressurePRthetaP) {
  const auto* group = find_halo_group(registry,
      "dynamics:theta_m,scalars,pressure_p,rtheta_p");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 4u);
  expect_field(group->fields[0], "theta_m", 1, {1, 2});
  expect_field(group->fields[1], "scalars", 1, {1, 2});
  expect_field(group->fields[2], "pressure_p", 1, {1, 2});
  expect_field(group->fields[3], "rtheta_p", 1, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsRwPRuPRhoPpRthetaPp) {
  const auto* group = find_halo_group(registry,
      "dynamics:rw_p,ru_p,rho_pp,rtheta_pp");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 4u);
  expect_field(group->fields[0], "rw_p", 1, {1});
  expect_field(group->fields[1], "ru_p", 1, {2});
  expect_field(group->fields[2], "rho_pp", 1, {1, 2});
  expect_field(group->fields[3], "rtheta_pp", 1, {2});
}

TEST_F(HaloGroupRegistryTest, DynamicsWPvEdgeRhoEdge) {
  const auto* group = find_halo_group(registry, "dynamics:w,pv_edge,rho_edge");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 3u);
  expect_field(group->fields[0], "w", 2, {1, 2});
  expect_field(group->fields[1], "pv_edge", 1, {1, 2});
  expect_field(group->fields[2], "rho_edge", 1, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsWPvEdgeRhoEdgeScalars) {
  const auto* group = find_halo_group(registry,
      "dynamics:w,pv_edge,rho_edge,scalars");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 4u);
  expect_field(group->fields[0], "w", 2, {1, 2});
  expect_field(group->fields[1], "pv_edge", 1, {1, 2});
  expect_field(group->fields[2], "rho_edge", 1, {1, 2});
  expect_field(group->fields[3], "scalars", 2, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsThetaMPressurePRthetaP) {
  const auto* group = find_halo_group(registry,
      "dynamics:theta_m,pressure_p,rtheta_p");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 3u);
  expect_field(group->fields[0], "theta_m", 2, {1, 2});
  expect_field(group->fields[1], "pressure_p", 1, {1, 2});
  expect_field(group->fields[2], "rtheta_p", 1, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsExner) {
  const auto* group = find_halo_group(registry, "dynamics:exner");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "exner", 1, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsTendU) {
  const auto* group = find_halo_group(registry, "dynamics:tend_u");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "tend_u", 1, {1});
}

TEST_F(HaloGroupRegistryTest, DynamicsRhoPp) {
  const auto* group = find_halo_group(registry, "dynamics:rho_pp");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "rho_pp", 1, {1});
}

TEST_F(HaloGroupRegistryTest, DynamicsRthetaPp) {
  const auto* group = find_halo_group(registry, "dynamics:rtheta_pp");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "rtheta_pp", 1, {1});
}

TEST_F(HaloGroupRegistryTest, DynamicsU123) {
  const auto* group = find_halo_group(registry, "dynamics:u_123");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "u", 2, {1, 2, 3});
}

TEST_F(HaloGroupRegistryTest, DynamicsU3) {
  const auto* group = find_halo_group(registry, "dynamics:u_3");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "u", 2, {3});
}

TEST_F(HaloGroupRegistryTest, DynamicsScalars) {
  const auto* group = find_halo_group(registry, "dynamics:scalars");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "scalars", 2, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsScalarsOld) {
  const auto* group = find_halo_group(registry, "dynamics:scalars_old");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "scalars", 1, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsW) {
  const auto* group = find_halo_group(registry, "dynamics:w");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "w", 2, {1, 2});
}

TEST_F(HaloGroupRegistryTest, DynamicsScale) {
  const auto* group = find_halo_group(registry, "dynamics:scale");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 1u);
  expect_field(group->fields[0], "scale", 1, {1, 2});
}

TEST_F(HaloGroupRegistryTest, FindNonexistentGroupReturnsNull) {
  EXPECT_EQ(find_halo_group(registry, "nonexistent:group"), nullptr);
}

#ifdef MPAS_PHYSICS_ENABLED
TEST_F(HaloGroupRegistryTest, PhysicsBlten) {
  const auto* group = find_halo_group(registry, "physics:blten");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 2u);
  expect_field(group->fields[0], "rublten", 1, {1, 2});
  expect_field(group->fields[1], "rvblten", 1, {1, 2});
}

TEST_F(HaloGroupRegistryTest, PhysicsCuten) {
  const auto* group = find_halo_group(registry, "physics:cuten");
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->fields.size(), 2u);
  expect_field(group->fields[0], "rucuten", 1, {1, 2});
  expect_field(group->fields[1], "rvcuten", 1, {1, 2});
}
#else
TEST_F(HaloGroupRegistryTest, PhysicsGroupsAbsentWhenPhysicsDisabled) {
  EXPECT_EQ(find_halo_group(registry, "physics:blten"), nullptr);
  EXPECT_EQ(find_halo_group(registry, "physics:cuten"), nullptr);
}
#endif

TEST_F(HaloGroupRegistryTest, RegistryGroupCountMatchesReferenceModel) {
#ifdef MPAS_PHYSICS_ENABLED
  // 17 dynamics/init groups + 2 physics groups = 19
  EXPECT_EQ(registry.size(), 19u);
#else
  // 2 initialization + 15 dynamics = 17
  EXPECT_EQ(registry.size(), 17u);
#endif
}

// Verify group ordering matches the Reference_Model construction order
TEST_F(HaloGroupRegistryTest, GroupOrderMatchesReferenceModel) {
  const std::vector<std::string> expected_order = {
      "initialization:u",
      "initialization:pv_edge,ru,rw",
      "dynamics:theta_m,scalars,pressure_p,rtheta_p",
      "dynamics:rw_p,ru_p,rho_pp,rtheta_pp",
      "dynamics:w,pv_edge,rho_edge",
      "dynamics:w,pv_edge,rho_edge,scalars",
      "dynamics:theta_m,pressure_p,rtheta_p",
      "dynamics:exner",
      "dynamics:tend_u",
      "dynamics:rho_pp",
      "dynamics:rtheta_pp",
      "dynamics:u_123",
      "dynamics:u_3",
      "dynamics:scalars",
      "dynamics:scalars_old",
      "dynamics:w",
      "dynamics:scale",
  };

  for (size_t i = 0; i < expected_order.size(); ++i) {
    ASSERT_LT(i, registry.size());
    EXPECT_EQ(registry[i].group_name, expected_order[i])
        << "Group at index " << i << " has wrong name";
  }
}

}  // namespace
