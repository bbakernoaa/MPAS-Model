/// @file test_dissipation_eddy_visc_property.cpp
/// @brief Property-based test: Eddy-viscosity stability bound.
///
/// Feature: mpas-dycore-cpp-port, Property 9: Eddy-viscosity stability bound
///
/// **Validates: Requirements 8.6**
///
/// For any random configuration (c_s, config_len_disp, invDt, velocities),
/// every computed eddy viscosity NEVER exceeds 0.01 * config_len_disp^2 * invDt.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include "mpas_dycore/dissipation_module.hpp"

#include <cmath>
#include <vector>

namespace {

using ES = Kokkos::DefaultHostExecutionSpace;
using MS = ES::memory_space;
using Scalar = mpas::dycore::Scalar;
using view2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, MS>;
using cview2d = Kokkos::View<const Scalar**, Kokkos::LayoutLeft, MS>;
using view3d = Kokkos::View<Scalar***, Kokkos::LayoutLeft, MS>;
using iview1d = Kokkos::View<int*, Kokkos::LayoutLeft, MS>;
using ciview1d = Kokkos::View<const int*, Kokkos::LayoutLeft, MS>;
using iview2d = Kokkos::View<int**, Kokkos::LayoutLeft, MS>;
using ciview2d = Kokkos::View<const int**, Kokkos::LayoutLeft, MS>;

} // namespace

// ---------------------------------------------------------------------------
// Property 9: Eddy-viscosity stability bound (2-D Smagorinsky)
// Feature: mpas-dycore-cpp-port, Property 9: Eddy-viscosity stability bound
// ---------------------------------------------------------------------------

RC_GTEST_PROP(EddyViscosityStabilityBound,
              Smagorinsky2D_NeverExceedsStabilityLimit,
              ()) {
  // Generate random mesh dimensions
  const int nVertLevels = *rc::gen::inRange(1, 11);  // [1, 10]
  const int nCells = *rc::gen::inRange(1, 11);       // [1, 10]
  const int nEdges = *rc::gen::inRange(1, 21);       // [1, 20]
  const int maxEdges = *rc::gen::inRange(3, 7);      // [3, 6]

  // Generate physical parameters in positive ranges
  const Scalar c_s = static_cast<Scalar>(*rc::gen::inRange(1, 100)) * 0.01;
  const Scalar config_len_disp =
      static_cast<Scalar>(*rc::gen::inRange(100, 50001));
  const Scalar dt = static_cast<Scalar>(*rc::gen::inRange(1, 601));
  const Scalar invDt = Scalar(1.0) / dt;

  // Stability limit from the Reference_Model formula
  const Scalar stability_limit =
      Scalar(0.01) * config_len_disp * config_len_disp * invDt;

  // Create mesh connectivity
  iview1d nEdgesOnCell_v("nEdgesOnCell", nCells);
  iview2d edgesOnCell_v("edgesOnCell", maxEdges, nCells);

  for (int c = 0; c < nCells; ++c) {
    const int ne = std::min(maxEdges, nEdges);
    nEdgesOnCell_v(c) = ne;
    for (int e = 0; e < ne; ++e) {
      edgesOnCell_v(e, c) = (c * 2 + e) % nEdges + 1;  // 1-based
    }
  }

  // Generate random velocity fields (potentially large to trigger bounding)
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  for (int k = 0; k < nVertLevels; ++k) {
    for (int e = 0; e < nEdges; ++e) {
      u_v(k, e) = static_cast<Scalar>(*rc::gen::inRange(-10000, 10001)) * 0.1;
      v_v(k, e) = static_cast<Scalar>(*rc::gen::inRange(-10000, 10001)) * 0.1;
    }
  }

  // Generate random deformation coefficients
  view2d deform_c2("c2", maxEdges, nCells);
  view2d deform_s2("s2", maxEdges, nCells);
  view2d deform_cs("cs", maxEdges, nCells);
  for (int c = 0; c < nCells; ++c) {
    for (int e = 0; e < maxEdges; ++e) {
      deform_c2(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_s2(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_cs(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
    }
  }

  // Output eddy viscosity
  view2d kdiff("kdiff", nVertLevels, nCells);

  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.c_s = c_s;
  params.config_len_disp = config_len_disp;
  params.invDt = invDt;
  params.config_visc4_2dsmag = 0.05;

  mpas::dycore::smagorinsky_2d<ES>(
      kdiff, cview2d(u_v), cview2d(v_v),
      cview2d(deform_c2), cview2d(deform_s2), cview2d(deform_cs),
      ciview1d(nEdgesOnCell_v), ciview2d(edgesOnCell_v), params);

  // Assert: every computed viscosity <= stability_limit
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      RC_ASSERT(kdiff(k, c) >= Scalar(0.0));
      RC_ASSERT(kdiff(k, c) <= stability_limit + Scalar(1.0e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Property 9: Eddy-viscosity stability bound (Fixed horizontal viscosity)
// Feature: mpas-dycore-cpp-port, Property 9: Eddy-viscosity stability bound
// ---------------------------------------------------------------------------

RC_GTEST_PROP(EddyViscosityStabilityBound,
              FixedHorizontal_NeverExceedsStabilityLimit,
              ()) {
  const int nVertLevels = *rc::gen::inRange(1, 11);
  const int nCells = *rc::gen::inRange(1, 11);

  const Scalar config_len_disp =
      static_cast<Scalar>(*rc::gen::inRange(100, 50001));
  const Scalar dt = static_cast<Scalar>(*rc::gen::inRange(1, 601));
  const Scalar invDt = Scalar(1.0) / dt;

  // Generate a fixed viscosity that could be any positive value (possibly huge)
  const Scalar fixed_visc =
      static_cast<Scalar>(*rc::gen::inRange(1, 1000001));

  const Scalar stability_limit =
      Scalar(0.01) * config_len_disp * config_len_disp * invDt;

  view2d kdiff("kdiff", nVertLevels, nCells);

  mpas::dycore::EddyViscosityParams params;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.config_len_disp = config_len_disp;
  params.invDt = invDt;

  mpas::dycore::fixed_horizontal_eddy_viscosity<ES>(kdiff, fixed_visc, params);

  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      RC_ASSERT(kdiff(k, c) >= Scalar(0.0));
      RC_ASSERT(kdiff(k, c) <= stability_limit + Scalar(1.0e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Property 9: Eddy-viscosity stability bound (3-D Smagorinsky LES)
// Feature: mpas-dycore-cpp-port, Property 9: Eddy-viscosity stability bound
// ---------------------------------------------------------------------------

RC_GTEST_PROP(EddyViscosityStabilityBound,
              LES3DSmagorinsky_NeverExceedsStabilityLimit,
              ()) {
  const int nVertLevels = *rc::gen::inRange(2, 9);   // need >=2 for zgrid
  const int nCells = *rc::gen::inRange(1, 7);
  const int nEdges = *rc::gen::inRange(1, 13);
  const int maxEdges = *rc::gen::inRange(3, 6);
  const int num_scalars = 2;

  const Scalar c_s = static_cast<Scalar>(*rc::gen::inRange(1, 100)) * 0.01;
  const Scalar config_len_disp =
      static_cast<Scalar>(*rc::gen::inRange(100, 20001));
  const Scalar dt = static_cast<Scalar>(*rc::gen::inRange(1, 601));
  const Scalar invDt = Scalar(1.0) / dt;

  const Scalar h_stability_limit =
      Scalar(0.01) * config_len_disp * config_len_disp * invDt;

  // Create connectivity
  iview1d nEdgesOnCell_v("nEdgesOnCell", nCells);
  iview2d edgesOnCell_v("edgesOnCell", maxEdges, nCells);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);

  for (int c = 0; c < nCells; ++c) {
    const int ne = std::min(maxEdges, nEdges);
    nEdgesOnCell_v(c) = ne;
    for (int e = 0; e < ne; ++e) {
      edgesOnCell_v(e, c) = (c * 2 + e) % nEdges + 1;
    }
  }
  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = (e % nCells) + 1;
    cellsOnEdge_v(1, e) = ((e + 1) % nCells) + 1;
  }

  // Random velocities and deformation coefficients
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d uCell_v("uCell", nVertLevels, nCells);
  view2d vCell_v("vCell", nVertLevels, nCells);
  view2d w_v("w", nVertLevels + 1, nCells);
  view2d bv_freq2_v("bv_freq2", nVertLevels, nCells);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view3d scalars_v("scalars", num_scalars, nVertLevels, nCells);
  view3d tend_scalars_v("tend_s", num_scalars, nVertLevels, nCells);
  view2d deform_c2("c2", maxEdges, nCells);
  view2d deform_s2("s2", maxEdges, nCells);
  view2d deform_cs("cs", maxEdges, nCells);
  view2d deform_c("c", maxEdges, nCells);
  view2d deform_s("s", maxEdges, nCells);
  view2d eddy_visc_horz("ev_h", nVertLevels, nCells);
  view2d eddy_visc_vert("ev_v", nVertLevels, nCells);
  view2d prandtl_3d_inv("pr3d", nVertLevels, nCells);

  for (int k = 0; k < nVertLevels; ++k) {
    for (int e = 0; e < nEdges; ++e) {
      u_v(k, e) = static_cast<Scalar>(*rc::gen::inRange(-5000, 5001)) * 0.1;
      v_v(k, e) = static_cast<Scalar>(*rc::gen::inRange(-5000, 5001)) * 0.1;
    }
  }
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      uCell_v(k, c) = static_cast<Scalar>(*rc::gen::inRange(-5000, 5001)) * 0.1;
      vCell_v(k, c) = static_cast<Scalar>(*rc::gen::inRange(-5000, 5001)) * 0.1;
      bv_freq2_v(k, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 1001)) * 0.001;
      rho_zz_v(k, c) = 1.0;
    }
  }

  for (int k = 0; k <= nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      w_v(k, c) = static_cast<Scalar>(*rc::gen::inRange(-2000, 2001)) * 0.1;
    }
  }

  // Random but positive vertical spacing (dz between 100 and 2000 m)
  for (int c = 0; c < nCells; ++c) {
    zgrid_v(0, c) = 0.0;
    for (int k = 1; k <= nVertLevels; ++k) {
      const Scalar dz = static_cast<Scalar>(*rc::gen::inRange(100, 2001));
      zgrid_v(k, c) = zgrid_v(k - 1, c) + dz;
    }
  }

  for (int c = 0; c < nCells; ++c) {
    for (int e = 0; e < maxEdges; ++e) {
      deform_c2(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_s2(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_cs(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_c(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_s(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
    }
  }

  mpas::dycore::EddyViscosityParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_3D_SMAGORINSKY;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.maxEdges = maxEdges;
  params.c_s = c_s;
  params.config_len_disp = config_len_disp;
  params.invDt = invDt;
  params.config_visc4_2dsmag = 0.05;
  params.dynamics_substep = 1;
  params.index_tke = 1;
  params.num_scalars = num_scalars;

  mpas::dycore::les_models<ES>(
      eddy_visc_horz, eddy_visc_vert, prandtl_3d_inv,
      cview2d(u_v), cview2d(v_v), cview2d(uCell_v), cview2d(vCell_v),
      cview2d(w_v), cview2d(bv_freq2_v), cview2d(zgrid_v),
      cview2d(rho_zz_v), scalars_v, tend_scalars_v,
      cview2d(deform_c2), cview2d(deform_s2), cview2d(deform_cs),
      cview2d(deform_c), cview2d(deform_s),
      ciview1d(nEdgesOnCell_v), ciview2d(edgesOnCell_v),
      ciview2d(cellsOnEdge_v), params);

  // Assert: horizontal viscosity bounded by h_stability_limit
  // Assert: vertical viscosity bounded by 0.01 * dz^2 * invDt
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      RC_ASSERT(eddy_visc_horz(k, c) >= Scalar(0.0));
      RC_ASSERT(eddy_visc_horz(k, c) <= h_stability_limit + Scalar(1.0e-10));

      const Scalar dz = zgrid_v(k + 1, c) - zgrid_v(k, c);
      const Scalar v_stability_limit = Scalar(0.01) * dz * dz * invDt;
      RC_ASSERT(eddy_visc_vert(k, c) >= Scalar(0.0));
      RC_ASSERT(eddy_visc_vert(k, c) <= v_stability_limit + Scalar(1.0e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Property 9: Eddy-viscosity stability bound (Prognostic 1.5-order TKE)
// Feature: mpas-dycore-cpp-port, Property 9: Eddy-viscosity stability bound
// ---------------------------------------------------------------------------

RC_GTEST_PROP(EddyViscosityStabilityBound,
              PrognosticTKE_NeverExceedsStabilityLimit,
              ()) {
  const int nVertLevels = *rc::gen::inRange(2, 9);
  const int nCells = *rc::gen::inRange(1, 7);
  const int nEdges = *rc::gen::inRange(1, 13);
  const int maxEdges = *rc::gen::inRange(3, 6);
  const int num_scalars = 2;
  const int idx_tke = 1;  // 1-based

  const Scalar c_s = static_cast<Scalar>(*rc::gen::inRange(1, 100)) * 0.01;
  const Scalar config_len_disp =
      static_cast<Scalar>(*rc::gen::inRange(100, 20001));
  const Scalar dt = static_cast<Scalar>(*rc::gen::inRange(1, 601));
  const Scalar invDt = Scalar(1.0) / dt;

  const Scalar h_stability_limit =
      Scalar(0.01) * config_len_disp * config_len_disp * invDt;

  // Connectivity
  iview1d nEdgesOnCell_v("nEdgesOnCell", nCells);
  iview2d edgesOnCell_v("edgesOnCell", maxEdges, nCells);
  iview2d cellsOnEdge_v("cellsOnEdge", 2, nEdges);

  for (int c = 0; c < nCells; ++c) {
    const int ne = std::min(maxEdges, nEdges);
    nEdgesOnCell_v(c) = ne;
    for (int e = 0; e < ne; ++e) {
      edgesOnCell_v(e, c) = (c * 2 + e) % nEdges + 1;
    }
  }
  for (int e = 0; e < nEdges; ++e) {
    cellsOnEdge_v(0, e) = (e % nCells) + 1;
    cellsOnEdge_v(1, e) = ((e + 1) % nCells) + 1;
  }

  // Fields
  view2d u_v("u", nVertLevels, nEdges);
  view2d v_v("v", nVertLevels, nEdges);
  view2d uCell_v("uCell", nVertLevels, nCells);
  view2d vCell_v("vCell", nVertLevels, nCells);
  view2d w_v("w", nVertLevels + 1, nCells);
  view2d bv_freq2_v("bv_freq2", nVertLevels, nCells);
  view2d zgrid_v("zgrid", nVertLevels + 1, nCells);
  view2d rho_zz_v("rho_zz", nVertLevels, nCells);
  view3d scalars_v("scalars", num_scalars, nVertLevels, nCells);
  view3d tend_scalars_v("tend_s", num_scalars, nVertLevels, nCells);
  view2d deform_c2("c2", maxEdges, nCells);
  view2d deform_s2("s2", maxEdges, nCells);
  view2d deform_cs("cs", maxEdges, nCells);
  view2d deform_c("c", maxEdges, nCells);
  view2d deform_s("s", maxEdges, nCells);
  view2d eddy_visc_horz("ev_h", nVertLevels, nCells);
  view2d eddy_visc_vert("ev_v", nVertLevels, nCells);
  view2d prandtl_3d_inv("pr3d", nVertLevels, nCells);

  for (int k = 0; k < nVertLevels; ++k) {
    for (int e = 0; e < nEdges; ++e) {
      u_v(k, e) = static_cast<Scalar>(*rc::gen::inRange(-5000, 5001)) * 0.1;
      v_v(k, e) = static_cast<Scalar>(*rc::gen::inRange(-5000, 5001)) * 0.1;
    }
  }
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      uCell_v(k, c) = static_cast<Scalar>(*rc::gen::inRange(-5000, 5001)) * 0.1;
      vCell_v(k, c) = static_cast<Scalar>(*rc::gen::inRange(-5000, 5001)) * 0.1;
      bv_freq2_v(k, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 1001)) * 0.001;
      rho_zz_v(k, c) = 1.0;
      // TKE: random non-negative values
      scalars_v(idx_tke - 1, k, c) =
          static_cast<Scalar>(*rc::gen::inRange(0, 10001)) * 0.01;
    }
  }

  for (int k = 0; k <= nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      w_v(k, c) = static_cast<Scalar>(*rc::gen::inRange(-2000, 2001)) * 0.1;
    }
  }

  // Positive vertical spacing
  for (int c = 0; c < nCells; ++c) {
    zgrid_v(0, c) = 0.0;
    for (int k = 1; k <= nVertLevels; ++k) {
      const Scalar dz = static_cast<Scalar>(*rc::gen::inRange(100, 2001));
      zgrid_v(k, c) = zgrid_v(k - 1, c) + dz;
    }
  }

  for (int c = 0; c < nCells; ++c) {
    for (int e = 0; e < maxEdges; ++e) {
      deform_c2(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_s2(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_cs(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_c(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
      deform_s(e, c) = static_cast<Scalar>(*rc::gen::inRange(-100, 101)) * 0.01;
    }
  }

  Kokkos::deep_copy(tend_scalars_v, Scalar(0.0));

  mpas::dycore::EddyViscosityParams params;
  params.les_model_opt = mpas::dycore::LES_MODEL_PROGNOSTIC_15_ORDER;
  params.nVertLevels = nVertLevels;
  params.nCells = nCells;
  params.nEdges = nEdges;
  params.maxEdges = maxEdges;
  params.c_s = c_s;
  params.config_len_disp = config_len_disp;
  params.invDt = invDt;
  params.config_visc4_2dsmag = 0.05;
  params.dynamics_substep = 1;
  params.index_tke = idx_tke;
  params.num_scalars = num_scalars;

  mpas::dycore::les_models<ES>(
      eddy_visc_horz, eddy_visc_vert, prandtl_3d_inv,
      cview2d(u_v), cview2d(v_v), cview2d(uCell_v), cview2d(vCell_v),
      cview2d(w_v), cview2d(bv_freq2_v), cview2d(zgrid_v),
      cview2d(rho_zz_v), scalars_v, tend_scalars_v,
      cview2d(deform_c2), cview2d(deform_s2), cview2d(deform_cs),
      cview2d(deform_c), cview2d(deform_s),
      ciview1d(nEdgesOnCell_v), ciview2d(edgesOnCell_v),
      ciview2d(cellsOnEdge_v), params);

  // Assert: horizontal viscosity bounded by h_stability_limit
  // Assert: vertical viscosity bounded by 0.01 * dz^2 * invDt
  for (int k = 0; k < nVertLevels; ++k) {
    for (int c = 0; c < nCells; ++c) {
      RC_ASSERT(eddy_visc_horz(k, c) >= Scalar(0.0));
      RC_ASSERT(eddy_visc_horz(k, c) <= h_stability_limit + Scalar(1.0e-10));

      const Scalar dz = zgrid_v(k + 1, c) - zgrid_v(k, c);
      const Scalar v_stability_limit = Scalar(0.01) * dz * dz * invDt;
      RC_ASSERT(eddy_visc_vert(k, c) >= Scalar(0.0));
      RC_ASSERT(eddy_visc_vert(k, c) <= v_stability_limit + Scalar(1.0e-10));
    }
  }
}
