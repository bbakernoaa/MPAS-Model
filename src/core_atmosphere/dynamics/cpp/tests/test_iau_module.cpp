#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mpas_dycore/iau_module.hpp"
#include "mpas_dycore/config.hpp"
#include "mpas_dycore/scalar.hpp"

namespace mpas {
namespace dycore {
namespace {

using exec_space = Kokkos::DefaultExecutionSpace;
using memory_space = typename exec_space::memory_space;
using view2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, memory_space>;

/// Helper to create a host-mirrored view, fill it, and deep-copy to device.
view2d make_filled_view(const std::string& name, int n0, int n1, Scalar fill_value) {
  view2d v(name, n0, n1);
  auto h = Kokkos::create_mirror_view(v);
  for (int j = 0; j < n1; ++j)
    for (int i = 0; i < n0; ++i)
      h(i, j) = fill_value;
  Kokkos::deep_copy(v, h);
  return v;
}

/// Helper to read a single value from a device view.
Scalar read_value(const view2d& v, int i, int j) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  return h(i, j);
}

// ---------------------------------------------------------------------------
// Req 10.1: No forcing when IAU is off
// ---------------------------------------------------------------------------

TEST(IAUModuleTest, NoForcingWhenIAUOff) {
  const int nVert = 3;
  const int nEdges = 2;
  const int nCells = 2;
  const int nMoist = 1;

  auto config = ConfigBuilder{}.config_iau(false).build();

  auto tend_ru = make_filled_view("tend_ru", nVert, nEdges, 1.0);
  auto tend_rho = make_filled_view("tend_rho", nVert, nCells, 2.0);
  auto tend_rtheta = make_filled_view("tend_rtheta", nVert, nCells, 3.0);
  auto rho_edge = make_filled_view("rho_edge", nVert, nEdges, 1.0);
  auto rho_zz = make_filled_view("rho_zz", nVert, nCells, 1.0);
  auto theta_m = make_filled_view("theta_m", nVert, nCells, 300.0);
  auto scalars_qv = make_filled_view("scalars_qv", nVert, nCells, 0.01);
  auto zz = make_filled_view("zz", nVert, nCells, 1.0);
  auto u_amb = make_filled_view("u_amb", nVert, nEdges, 5.0);
  auto rho_amb = make_filled_view("rho_amb", nVert, nCells, 0.1);
  auto theta_amb = make_filled_view("theta_amb", nVert, nCells, 1.0);
  auto tend_scalars = make_filled_view("tend_scalars", nMoist * nVert, nCells, 0.0);
  auto scalars_amb = make_filled_view("scalars_amb", nMoist * nVert, nCells, 0.001);
  auto scalars_slice = make_filled_view("scalars_slice", nMoist * nVert, nCells, 0.01);

  IAU_Module<exec_space>::add_iau_tendency(
      config, /*itimestep=*/1, /*dt=*/60.0, /*iau_window_length_s=*/3600.0,
      nEdges, nCells, nVert,
      /*moist_start=*/0, /*moist_end=*/nMoist,
      /*index_qv=*/0,
      tend_ru, tend_rho, tend_rtheta,
      rho_edge, rho_zz, theta_m, scalars_qv, zz,
      u_amb, rho_amb, theta_amb,
      tend_scalars, scalars_amb, scalars_slice);

  Kokkos::fence();

  // Tendencies should be unchanged.
  EXPECT_DOUBLE_EQ(read_value(tend_ru, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(read_value(tend_rho, 0, 0), 2.0);
  EXPECT_DOUBLE_EQ(read_value(tend_rtheta, 0, 0), 3.0);
  EXPECT_DOUBLE_EQ(read_value(tend_scalars, 0, 0), 0.0);
}

// ---------------------------------------------------------------------------
// Req 10.3: Return without modification when weight <= negligible threshold
// ---------------------------------------------------------------------------

TEST(IAUModuleTest, NoModificationWhenWeightNegligible) {
  const int nVert = 2;
  const int nEdges = 1;
  const int nCells = 1;
  const int nMoist = 1;

  auto config = ConfigBuilder{}.config_iau(true).build();

  auto tend_ru = make_filled_view("tend_ru", nVert, nEdges, 1.0);
  auto tend_rho = make_filled_view("tend_rho", nVert, nCells, 2.0);
  auto tend_rtheta = make_filled_view("tend_rtheta", nVert, nCells, 3.0);
  auto rho_edge = make_filled_view("rho_edge", nVert, nEdges, 1.0);
  auto rho_zz = make_filled_view("rho_zz", nVert, nCells, 1.0);
  auto theta_m = make_filled_view("theta_m", nVert, nCells, 300.0);
  auto scalars_qv = make_filled_view("scalars_qv", nVert, nCells, 0.01);
  auto zz = make_filled_view("zz", nVert, nCells, 1.0);
  auto u_amb = make_filled_view("u_amb", nVert, nEdges, 5.0);
  auto rho_amb = make_filled_view("rho_amb", nVert, nCells, 0.1);
  auto theta_amb = make_filled_view("theta_amb", nVert, nCells, 1.0);
  auto tend_scalars = make_filled_view("tend_scalars", nMoist * nVert, nCells, 0.0);
  auto scalars_amb = make_filled_view("scalars_amb", nMoist * nVert, nCells, 0.001);
  auto scalars_slice = make_filled_view("scalars_slice", nMoist * nVert, nCells, 0.01);

  // Use an astronomically large window so that wgt = 1/window ~ 0.
  const Scalar huge_window = Scalar(1.0e+35);

  IAU_Module<exec_space>::add_iau_tendency(
      config, /*itimestep=*/1, /*dt=*/60.0, /*iau_window_length_s=*/huge_window,
      nEdges, nCells, nVert,
      /*moist_start=*/0, /*moist_end=*/nMoist,
      /*index_qv=*/0,
      tend_ru, tend_rho, tend_rtheta,
      rho_edge, rho_zz, theta_m, scalars_qv, zz,
      u_amb, rho_amb, theta_amb,
      tend_scalars, scalars_amb, scalars_slice);

  Kokkos::fence();

  // Tendencies should be unchanged (weight was negligible).
  EXPECT_DOUBLE_EQ(read_value(tend_ru, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(read_value(tend_rho, 0, 0), 2.0);
  EXPECT_DOUBLE_EQ(read_value(tend_rtheta, 0, 0), 3.0);
  EXPECT_DOUBLE_EQ(read_value(tend_scalars, 0, 0), 0.0);
}

// ---------------------------------------------------------------------------
// Req 10.3: Return without modification when outside IAU window
// ---------------------------------------------------------------------------

TEST(IAUModuleTest, NoModificationOutsideWindow) {
  const int nVert = 2;
  const int nEdges = 1;
  const int nCells = 1;
  const int nMoist = 1;

  auto config = ConfigBuilder{}.config_iau(true).build();

  auto tend_ru = make_filled_view("tend_ru", nVert, nEdges, 1.0);
  auto tend_rho = make_filled_view("tend_rho", nVert, nCells, 2.0);
  auto tend_rtheta = make_filled_view("tend_rtheta", nVert, nCells, 3.0);
  auto rho_edge = make_filled_view("rho_edge", nVert, nEdges, 1.0);
  auto rho_zz = make_filled_view("rho_zz", nVert, nCells, 1.0);
  auto theta_m = make_filled_view("theta_m", nVert, nCells, 300.0);
  auto scalars_qv = make_filled_view("scalars_qv", nVert, nCells, 0.01);
  auto zz = make_filled_view("zz", nVert, nCells, 1.0);
  auto u_amb = make_filled_view("u_amb", nVert, nEdges, 5.0);
  auto rho_amb = make_filled_view("rho_amb", nVert, nCells, 0.1);
  auto theta_amb = make_filled_view("theta_amb", nVert, nCells, 1.0);
  auto tend_scalars = make_filled_view("tend_scalars", nMoist * nVert, nCells, 0.0);
  auto scalars_amb = make_filled_view("scalars_amb", nMoist * nVert, nCells, 0.001);
  auto scalars_slice = make_filled_view("scalars_slice", nMoist * nVert, nCells, 0.01);

  // Window is 3600s, dt = 60s => nsteps_iau = 60. timestep 61 is outside.
  IAU_Module<exec_space>::add_iau_tendency(
      config, /*itimestep=*/61, /*dt=*/60.0, /*iau_window_length_s=*/3600.0,
      nEdges, nCells, nVert,
      /*moist_start=*/0, /*moist_end=*/nMoist,
      /*index_qv=*/0,
      tend_ru, tend_rho, tend_rtheta,
      rho_edge, rho_zz, theta_m, scalars_qv, zz,
      u_amb, rho_amb, theta_amb,
      tend_scalars, scalars_amb, scalars_slice);

  Kokkos::fence();

  EXPECT_DOUBLE_EQ(read_value(tend_ru, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(read_value(tend_rho, 0, 0), 2.0);
  EXPECT_DOUBLE_EQ(read_value(tend_rtheta, 0, 0), 3.0);
}

// ---------------------------------------------------------------------------
// Req 10.2: Within IAU window, forcing is applied correctly
// ---------------------------------------------------------------------------

TEST(IAUModuleTest, ForcingAppliedWithinWindow) {
  // Simplified 1-cell, 1-edge, 1-level, 1-moist-scalar test.
  const int nVert = 1;
  const int nEdges = 1;
  const int nCells = 1;
  const int nMoist = 1;

  auto config = ConfigBuilder{}.config_iau(true).build();

  // Setup: dt=60, window=3600 => nsteps=60, weight=1/3600
  const Scalar dt = 60.0;
  const Scalar window = 3600.0;
  const Scalar wgt = Scalar(1.0) / window;

  // State values
  const Scalar rho_edge_val = 1.2;
  const Scalar rho_zz_val = 1.1;
  const Scalar theta_m_val = 300.0;
  const Scalar qv_val = 0.01;
  const Scalar zz_val = 0.98;

  // Increment values
  const Scalar u_amb_val = 2.0;
  const Scalar rho_amb_val = 0.05;
  const Scalar theta_amb_val = 0.5;
  const Scalar scalars_amb_val = 0.001;

  auto tend_ru = make_filled_view("tend_ru", nVert, nEdges, 0.0);
  auto tend_rho = make_filled_view("tend_rho", nVert, nCells, 0.0);
  auto tend_rtheta = make_filled_view("tend_rtheta", nVert, nCells, 0.0);
  auto rho_edge = make_filled_view("rho_edge", nVert, nEdges, rho_edge_val);
  auto rho_zz = make_filled_view("rho_zz", nVert, nCells, rho_zz_val);
  auto theta_m = make_filled_view("theta_m", nVert, nCells, theta_m_val);
  auto scalars_qv = make_filled_view("scalars_qv", nVert, nCells, qv_val);
  auto zz = make_filled_view("zz", nVert, nCells, zz_val);
  auto u_amb = make_filled_view("u_amb", nVert, nEdges, u_amb_val);
  auto rho_amb = make_filled_view("rho_amb", nVert, nCells, rho_amb_val);
  auto theta_amb = make_filled_view("theta_amb", nVert, nCells, theta_amb_val);
  auto tend_scalars = make_filled_view("tend_scalars", nMoist * nVert, nCells, 0.0);
  auto scalars_amb = make_filled_view("scalars_amb", nMoist * nVert, nCells, scalars_amb_val);
  auto scalars_slice = make_filled_view("scalars_slice", nMoist * nVert, nCells, qv_val);

  IAU_Module<exec_space>::add_iau_tendency(
      config, /*itimestep=*/1, dt, window,
      nEdges, nCells, nVert,
      /*moist_start=*/0, /*moist_end=*/nMoist,
      /*index_qv=*/0,
      tend_ru, tend_rho, tend_rtheta,
      rho_edge, rho_zz, theta_m, scalars_qv, zz,
      u_amb, rho_amb, theta_amb,
      tend_scalars, scalars_amb, scalars_slice);

  Kokkos::fence();

  // Expected values following Fortran logic:
  // tend_ru += wgt * rho_edge * u_amb
  const Scalar expected_tend_ru = wgt * rho_edge_val * u_amb_val;
  EXPECT_NEAR(read_value(tend_ru, 0, 0), expected_tend_ru, 1e-12);

  // tend_rho += wgt * rho_amb / zz
  const Scalar expected_tend_rho = wgt * rho_amb_val / zz_val;
  EXPECT_NEAR(read_value(tend_rho, 0, 0), expected_tend_rho, 1e-12);

  // theta = theta_m / (1 + rvord * qv)
  const Scalar theta = theta_m_val / (Scalar(1.0) + rvord * qv_val);

  // tend_th = wgt * (theta_amb * rho_zz + theta * rho_amb / zz)
  const Scalar tend_th = wgt * (theta_amb_val * rho_zz_val
                                + theta * rho_amb_val / zz_val);

  // tend_scalars (qv) += wgt * (scalars_amb * rho_zz + scalars * rho_amb / zz)
  const Scalar tend_qv = wgt * (scalars_amb_val * rho_zz_val
                                + qv_val * rho_amb_val / zz_val);

  // tend_rtheta += (1 + rvord * qv) * tend_th + rvord * theta * tend_qv
  const Scalar expected_tend_rtheta = (Scalar(1.0) + rvord * qv_val) * tend_th
                                      + rvord * theta * tend_qv;
  EXPECT_NEAR(read_value(tend_rtheta, 0, 0), expected_tend_rtheta, 1e-10);

  // Scalar tendency matches expected
  EXPECT_NEAR(read_value(tend_scalars, 0, 0), tend_qv, 1e-12);
}

// ---------------------------------------------------------------------------
// Req 10.4: Theta increment coupled to moist theta correctly
// ---------------------------------------------------------------------------

TEST(IAUModuleTest, ThetaCoupledMoistConversion) {
  // Verify the coupled conversion: (1+rvord*qv)*tend_th + rvord*theta*tend_qv
  // with a higher qv to emphasize the coupling.
  const int nVert = 1;
  const int nEdges = 1;
  const int nCells = 1;
  const int nMoist = 1;

  auto config = ConfigBuilder{}.config_iau(true).build();

  const Scalar dt = 120.0;
  const Scalar window = 7200.0;  // nsteps = 60
  const Scalar wgt = Scalar(1.0) / window;

  const Scalar rho_edge_val = 1.0;
  const Scalar rho_zz_val = 1.0;
  const Scalar theta_m_val = 310.0;
  const Scalar qv_val = 0.02;  // higher moisture
  const Scalar zz_val = 1.0;   // simplify: zz = 1

  const Scalar u_amb_val = 0.0;
  const Scalar rho_amb_val = 0.0;  // zero density increment
  const Scalar theta_amb_val = 2.0;
  const Scalar scalars_amb_val = 0.005;

  auto tend_ru = make_filled_view("tend_ru", nVert, nEdges, 0.0);
  auto tend_rho = make_filled_view("tend_rho", nVert, nCells, 0.0);
  auto tend_rtheta = make_filled_view("tend_rtheta", nVert, nCells, 0.0);
  auto rho_edge = make_filled_view("rho_edge", nVert, nEdges, rho_edge_val);
  auto rho_zz = make_filled_view("rho_zz", nVert, nCells, rho_zz_val);
  auto theta_m = make_filled_view("theta_m", nVert, nCells, theta_m_val);
  auto scalars_qv = make_filled_view("scalars_qv", nVert, nCells, qv_val);
  auto zz = make_filled_view("zz", nVert, nCells, zz_val);
  auto u_amb = make_filled_view("u_amb", nVert, nEdges, u_amb_val);
  auto rho_amb = make_filled_view("rho_amb", nVert, nCells, rho_amb_val);
  auto theta_amb = make_filled_view("theta_amb", nVert, nCells, theta_amb_val);
  auto tend_scalars = make_filled_view("tend_scalars", nMoist * nVert, nCells, 0.0);
  auto scalars_amb = make_filled_view("scalars_amb", nMoist * nVert, nCells, scalars_amb_val);
  auto scalars_slice = make_filled_view("scalars_slice", nMoist * nVert, nCells, qv_val);

  IAU_Module<exec_space>::add_iau_tendency(
      config, /*itimestep=*/30, dt, window,
      nEdges, nCells, nVert,
      /*moist_start=*/0, /*moist_end=*/nMoist,
      /*index_qv=*/0,
      tend_ru, tend_rho, tend_rtheta,
      rho_edge, rho_zz, theta_m, scalars_qv, zz,
      u_amb, rho_amb, theta_amb,
      tend_scalars, scalars_amb, scalars_slice);

  Kokkos::fence();

  // With zz=1, rho_amb=0:
  // theta = theta_m / (1 + rvord * qv)
  const Scalar theta = theta_m_val / (Scalar(1.0) + rvord * qv_val);

  // tend_th = wgt * (theta_amb * rho_zz + 0)  = wgt * theta_amb
  const Scalar tend_th = wgt * theta_amb_val * rho_zz_val;

  // tend_qv = wgt * (scalars_amb * rho_zz + qv * 0) = wgt * scalars_amb
  const Scalar tend_qv = wgt * scalars_amb_val * rho_zz_val;

  // tend_rtheta = (1 + rvord*qv) * tend_th + rvord * theta * tend_qv
  const Scalar expected = (Scalar(1.0) + rvord * qv_val) * tend_th
                          + rvord * theta * tend_qv;

  EXPECT_NEAR(read_value(tend_rtheta, 0, 0), expected, 1e-10);
}

// ---------------------------------------------------------------------------
// Req 10.2: Forcing applied at the window boundary (timestep == nsteps_iau)
// ---------------------------------------------------------------------------

TEST(IAUModuleTest, ForcingAppliedAtWindowBoundary) {
  // Verify that the forcing IS applied when itimestep == nsteps_iau (the last
  // timestep within the window), confirming the <= boundary condition.
  const int nVert = 1;
  const int nEdges = 1;
  const int nCells = 1;
  const int nMoist = 1;

  auto config = ConfigBuilder{}.config_iau(true).build();

  const Scalar dt = 60.0;
  const Scalar window = 3600.0;  // nsteps_iau = round(3600/60) = 60
  const int nsteps_iau = 60;
  const Scalar wgt = Scalar(1.0) / window;

  const Scalar rho_edge_val = 1.1;
  const Scalar u_amb_val = 3.0;

  auto tend_ru = make_filled_view("tend_ru", nVert, nEdges, 0.0);
  auto tend_rho = make_filled_view("tend_rho", nVert, nCells, 0.0);
  auto tend_rtheta = make_filled_view("tend_rtheta", nVert, nCells, 0.0);
  auto rho_edge = make_filled_view("rho_edge", nVert, nEdges, rho_edge_val);
  auto rho_zz = make_filled_view("rho_zz", nVert, nCells, 1.0);
  auto theta_m = make_filled_view("theta_m", nVert, nCells, 300.0);
  auto scalars_qv = make_filled_view("scalars_qv", nVert, nCells, 0.01);
  auto zz = make_filled_view("zz", nVert, nCells, 1.0);
  auto u_amb = make_filled_view("u_amb", nVert, nEdges, u_amb_val);
  auto rho_amb = make_filled_view("rho_amb", nVert, nCells, 0.0);
  auto theta_amb = make_filled_view("theta_amb", nVert, nCells, 0.0);
  auto tend_scalars = make_filled_view("tend_scalars", nMoist * nVert, nCells, 0.0);
  auto scalars_amb = make_filled_view("scalars_amb", nMoist * nVert, nCells, 0.0);
  auto scalars_slice = make_filled_view("scalars_slice", nMoist * nVert, nCells, 0.01);

  // Call at the last valid timestep (boundary of the window)
  IAU_Module<exec_space>::add_iau_tendency(
      config, /*itimestep=*/nsteps_iau, dt, window,
      nEdges, nCells, nVert,
      /*moist_start=*/0, /*moist_end=*/nMoist,
      /*index_qv=*/0,
      tend_ru, tend_rho, tend_rtheta,
      rho_edge, rho_zz, theta_m, scalars_qv, zz,
      u_amb, rho_amb, theta_amb,
      tend_scalars, scalars_amb, scalars_slice);

  Kokkos::fence();

  // Forcing should be applied: tend_ru = wgt * rho_edge * u_amb
  const Scalar expected_tend_ru = wgt * rho_edge_val * u_amb_val;
  EXPECT_NEAR(read_value(tend_ru, 0, 0), expected_tend_ru, 1e-12)
      << "Field divergence: tend_ru at window boundary timestep";
}

// ---------------------------------------------------------------------------
// Req 10.2, 10.4: Multi-cell, multi-level, multiple moist scalars
// ---------------------------------------------------------------------------

TEST(IAUModuleTest, MultipleCellsLevelsAndMoistScalars) {
  // Validates kernel indexing across multiple cells, levels, and moist scalars.
  // Uses 2 moist scalars to verify the scalar loop and qv identification.
  const int nVert = 3;
  const int nEdges = 2;
  const int nCells = 2;
  const int nMoist = 2;  // qv at index 0, another scalar at index 1

  auto config = ConfigBuilder{}.config_iau(true).build();

  const Scalar dt = 120.0;
  const Scalar window = 7200.0;  // nsteps = 60
  const Scalar wgt = Scalar(1.0) / window;

  // Non-uniform state values to detect indexing errors
  auto tend_ru = make_filled_view("tend_ru", nVert, nEdges, 0.0);
  auto tend_rho = make_filled_view("tend_rho", nVert, nCells, 0.0);
  auto tend_rtheta = make_filled_view("tend_rtheta", nVert, nCells, 0.0);

  // Fill state fields with non-trivial varying values via host mirrors
  view2d rho_edge("rho_edge", nVert, nEdges);
  view2d rho_zz("rho_zz", nVert, nCells);
  view2d theta_m("theta_m", nVert, nCells);
  view2d scalars_qv("scalars_qv", nVert, nCells);
  view2d zz("zz", nVert, nCells);
  view2d u_amb("u_amb", nVert, nEdges);
  view2d rho_amb("rho_amb", nVert, nCells);
  view2d theta_amb("theta_amb", nVert, nCells);
  view2d tend_scalars("tend_scalars", nMoist * nVert, nCells);
  view2d scalars_amb("scalars_amb", nMoist * nVert, nCells);
  view2d scalars_slice("scalars_slice", nMoist * nVert, nCells);

  {
    auto h_rho_edge = Kokkos::create_mirror_view(rho_edge);
    auto h_rho_zz = Kokkos::create_mirror_view(rho_zz);
    auto h_theta_m = Kokkos::create_mirror_view(theta_m);
    auto h_scalars_qv = Kokkos::create_mirror_view(scalars_qv);
    auto h_zz = Kokkos::create_mirror_view(zz);
    auto h_u_amb = Kokkos::create_mirror_view(u_amb);
    auto h_rho_amb = Kokkos::create_mirror_view(rho_amb);
    auto h_theta_amb = Kokkos::create_mirror_view(theta_amb);
    auto h_tend_scalars = Kokkos::create_mirror_view(tend_scalars);
    auto h_scalars_amb = Kokkos::create_mirror_view(scalars_amb);
    auto h_scalars_slice = Kokkos::create_mirror_view(scalars_slice);

    for (int j = 0; j < nEdges; ++j)
      for (int i = 0; i < nVert; ++i) {
        h_rho_edge(i, j) = 1.0 + 0.1 * i + 0.05 * j;
        h_u_amb(i, j) = 2.0 + 0.5 * i - 0.3 * j;
      }

    for (int j = 0; j < nCells; ++j)
      for (int i = 0; i < nVert; ++i) {
        h_rho_zz(i, j) = 1.0 + 0.05 * i + 0.02 * j;
        h_theta_m(i, j) = 300.0 + 2.0 * i + 1.0 * j;
        h_scalars_qv(i, j) = 0.01 + 0.001 * i;
        h_zz(i, j) = 0.98 + 0.005 * j;
        h_rho_amb(i, j) = 0.05 + 0.01 * i;
        h_theta_amb(i, j) = 0.5 + 0.1 * i + 0.05 * j;
      }

    for (int j = 0; j < nCells; ++j)
      for (int m = 0; m < nMoist; ++m)
        for (int i = 0; i < nVert; ++i) {
          int row = m * nVert + i;
          h_tend_scalars(row, j) = 0.0;
          h_scalars_amb(row, j) = 0.001 + 0.0005 * m + 0.0001 * i;
          h_scalars_slice(row, j) = 0.01 + 0.002 * m + 0.001 * i;
        }

    Kokkos::deep_copy(rho_edge, h_rho_edge);
    Kokkos::deep_copy(rho_zz, h_rho_zz);
    Kokkos::deep_copy(theta_m, h_theta_m);
    Kokkos::deep_copy(scalars_qv, h_scalars_qv);
    Kokkos::deep_copy(zz, h_zz);
    Kokkos::deep_copy(u_amb, h_u_amb);
    Kokkos::deep_copy(rho_amb, h_rho_amb);
    Kokkos::deep_copy(theta_amb, h_theta_amb);
    Kokkos::deep_copy(tend_scalars, h_tend_scalars);
    Kokkos::deep_copy(scalars_amb, h_scalars_amb);
    Kokkos::deep_copy(scalars_slice, h_scalars_slice);
  }

  IAU_Module<exec_space>::add_iau_tendency(
      config, /*itimestep=*/10, dt, window,
      nEdges, nCells, nVert,
      /*moist_start=*/0, /*moist_end=*/nMoist,
      /*index_qv=*/0,
      tend_ru, tend_rho, tend_rtheta,
      rho_edge, rho_zz, theta_m, scalars_qv, zz,
      u_amb, rho_amb, theta_amb,
      tend_scalars, scalars_amb, scalars_slice);

  Kokkos::fence();

  // Verify all fields against manually computed Reference_Model equivalents.
  auto h_tend_ru = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tend_ru);
  auto h_tend_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tend_rho);
  auto h_tend_rtheta = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tend_rtheta);
  auto h_tend_scalars = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tend_scalars);

  // Recompute expected values on host using the same input data
  auto h_rho_edge = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rho_edge);
  auto h_rho_zz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rho_zz);
  auto h_theta_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, theta_m);
  auto h_scalars_qv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scalars_qv);
  auto h_zz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, zz);
  auto h_u_amb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_amb);
  auto h_rho_amb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rho_amb);
  auto h_theta_amb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, theta_amb);
  auto h_scalars_amb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scalars_amb);
  auto h_scalars_slice = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scalars_slice);

  // Check tend_ru for all edges and levels
  for (int j = 0; j < nEdges; ++j)
    for (int i = 0; i < nVert; ++i) {
      Scalar expected = wgt * h_rho_edge(i, j) * h_u_amb(i, j);
      EXPECT_NEAR(h_tend_ru(i, j), expected, 1e-12)
          << "Field divergence: tend_ru at level=" << i << " edge=" << j;
    }

  // Check tend_rho for all cells and levels
  for (int j = 0; j < nCells; ++j)
    for (int i = 0; i < nVert; ++i) {
      Scalar expected = wgt * h_rho_amb(i, j) / h_zz(i, j);
      EXPECT_NEAR(h_tend_rho(i, j), expected, 1e-12)
          << "Field divergence: tend_rho at level=" << i << " cell=" << j;
    }

  // Check tend_rtheta and tend_scalars following the Reference_Model algorithm
  for (int j = 0; j < nCells; ++j)
    for (int i = 0; i < nVert; ++i) {
      const Scalar qv = h_scalars_qv(i, j);
      const Scalar theta = h_theta_m(i, j) / (Scalar(1.0) + rvord * qv);
      const Scalar rho_zz_val = h_rho_zz(i, j);
      const Scalar zz_val = h_zz(i, j);
      const Scalar rho_amb_val = h_rho_amb(i, j);

      Scalar tend_th = wgt * (h_theta_amb(i, j) * rho_zz_val
                              + theta * rho_amb_val / zz_val);

      Scalar tend_qv_local = Scalar(0.0);
      for (int m = 0; m < nMoist; ++m) {
        int row = m * nVert + i;
        Scalar scalar_val = h_scalars_slice(row, j);
        Scalar scalar_amb_val = h_scalars_amb(row, j);
        Scalar tend_s = wgt * (scalar_amb_val * rho_zz_val
                               + scalar_val * rho_amb_val / zz_val);
        EXPECT_NEAR(h_tend_scalars(row, j), tend_s, 1e-12)
            << "Field divergence: tend_scalars at moist=" << m
            << " level=" << i << " cell=" << j;
        if (m == 0) {  // index_qv == 0 - moist_start
          tend_qv_local = tend_s;
        }
      }

      Scalar expected_rtheta = (Scalar(1.0) + rvord * qv) * tend_th
                               + rvord * theta * tend_qv_local;
      EXPECT_NEAR(h_tend_rtheta(i, j), expected_rtheta, 1e-10)
          << "Field divergence: tend_rtheta at level=" << i << " cell=" << j;
    }
}

// ---------------------------------------------------------------------------
// Req 10.2, 10.3: Boundary between inside and outside window (nsteps+1)
// ---------------------------------------------------------------------------

TEST(IAUModuleTest, NoForcingOneStepPastWindow) {
  // Verify that timestep nsteps_iau + 1 does NOT apply forcing.
  const int nVert = 2;
  const int nEdges = 1;
  const int nCells = 1;
  const int nMoist = 1;

  auto config = ConfigBuilder{}.config_iau(true).build();

  const Scalar dt = 60.0;
  const Scalar window = 600.0;  // nsteps_iau = round(600/60) = 10
  const int nsteps_iau = 10;

  auto tend_ru = make_filled_view("tend_ru", nVert, nEdges, 7.0);
  auto tend_rho = make_filled_view("tend_rho", nVert, nCells, 8.0);
  auto tend_rtheta = make_filled_view("tend_rtheta", nVert, nCells, 9.0);
  auto rho_edge = make_filled_view("rho_edge", nVert, nEdges, 1.2);
  auto rho_zz = make_filled_view("rho_zz", nVert, nCells, 1.1);
  auto theta_m = make_filled_view("theta_m", nVert, nCells, 300.0);
  auto scalars_qv = make_filled_view("scalars_qv", nVert, nCells, 0.01);
  auto zz = make_filled_view("zz", nVert, nCells, 1.0);
  auto u_amb = make_filled_view("u_amb", nVert, nEdges, 5.0);
  auto rho_amb = make_filled_view("rho_amb", nVert, nCells, 0.1);
  auto theta_amb = make_filled_view("theta_amb", nVert, nCells, 1.0);
  auto tend_scalars = make_filled_view("tend_scalars", nMoist * nVert, nCells, 4.0);
  auto scalars_amb = make_filled_view("scalars_amb", nMoist * nVert, nCells, 0.001);
  auto scalars_slice = make_filled_view("scalars_slice", nMoist * nVert, nCells, 0.01);

  // One step past the window boundary
  IAU_Module<exec_space>::add_iau_tendency(
      config, /*itimestep=*/nsteps_iau + 1, dt, window,
      nEdges, nCells, nVert,
      /*moist_start=*/0, /*moist_end=*/nMoist,
      /*index_qv=*/0,
      tend_ru, tend_rho, tend_rtheta,
      rho_edge, rho_zz, theta_m, scalars_qv, zz,
      u_amb, rho_amb, theta_amb,
      tend_scalars, scalars_amb, scalars_slice);

  Kokkos::fence();

  // No modification should have occurred
  EXPECT_DOUBLE_EQ(read_value(tend_ru, 0, 0), 7.0)
      << "Field divergence: tend_ru modified outside IAU window";
  EXPECT_DOUBLE_EQ(read_value(tend_rho, 0, 0), 8.0)
      << "Field divergence: tend_rho modified outside IAU window";
  EXPECT_DOUBLE_EQ(read_value(tend_rtheta, 0, 0), 9.0)
      << "Field divergence: tend_rtheta modified outside IAU window";
  EXPECT_DOUBLE_EQ(read_value(tend_scalars, 0, 0), 4.0)
      << "Field divergence: tend_scalars modified outside IAU window";
}

}  // namespace
}  // namespace dycore
}  // namespace mpas
