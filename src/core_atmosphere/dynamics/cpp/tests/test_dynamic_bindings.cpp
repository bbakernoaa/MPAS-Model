#include <gtest/gtest.h>
#include "mpas_dycore/field_store.hpp"
#include "mpas_dycore/dycore_c_api.h"
#include <unordered_map>
#include <vector>

using Scalar = mpas::dycore::Scalar;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using FieldStore = mpas::dycore::Field_Store<Scalar, ExecSpace>;

// Static mock registry to store allocated arrays for the tests
static std::unordered_map<std::string, std::vector<double>> g_mock_fortran_memory;

// In this test file, our custom mpas_cpp_get_pointer implementation overrides
// the weak stub defined in dycore_c_api.cpp.
extern "C" {
  void* mpas_cpp_get_pointer(const char* pool_name, const char* var_name, int dim_num, int time_level) {
    std::string key = std::string(pool_name) + ":" + var_name + ":" + std::to_string(time_level);
    if (g_mock_fortran_memory.find(key) == g_mock_fortran_memory.end()) {
      g_mock_fortran_memory[key] = std::vector<double>(1000, 42.0); // Allocate dummy data
    }
    return g_mock_fortran_memory[key].data();
  }
}

TEST(DynamicBindingsTest, RegistryAndWrappingSucceeds) {
  // Fill required dummy pointers for dycore_init main arguments
  std::vector<int> cellsOnEdge(100, 0);
  std::vector<int> edgesOnCell(100, 0);
  std::vector<int> verticesOnEdge(100, 0);
  std::vector<int> nEdgesOnCell(100, 0);
  std::vector<double> dummy_geom(100, 1.0);
  std::vector<double> dummy_state(100, 2.0);

  int rc = dycore_init(
      10, 10, 10, 5, 5, 2,
      cellsOnEdge.data(), edgesOnCell.data(), verticesOnEdge.data(), nEdgesOnCell.data(),
      dummy_geom.data(), dummy_geom.data(), dummy_geom.data(),
      dummy_geom.data(), dummy_geom.data(), dummy_geom.data(), dummy_geom.data(),
      dummy_state.data(), dummy_state.data(), dummy_state.data(), dummy_state.data(),
      dummy_state.data(), dummy_state.data(), dummy_state.data(), dummy_state.data(),
      dummy_state.data(), dummy_state.data(),
      3, 4, 1, 0, 1, 0, 0, 0, 0,
      0.1, 120000.0, 0.0, 1,
      0,
      /* cell send   */ 0, 0, nullptr, nullptr, nullptr,
      /* cell recv   */ 0, 0, nullptr, nullptr, nullptr,
      /* edge send   */ 0, 0, nullptr, nullptr, nullptr,
      /* edge recv   */ 0, 0, nullptr, nullptr, nullptr,
      /* vertex send */ 0, 0, nullptr, nullptr, nullptr,
      /* vertex recv */ 0, 0, nullptr, nullptr, nullptr
  );

  EXPECT_EQ(rc, 0);
  dycore_finalize();
}
