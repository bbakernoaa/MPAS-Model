#include <gtest/gtest.h>

#include "mpas_dycore/les_options.hpp"

namespace mpas {
namespace dycore {
namespace {

// ---------------------------------------------------------------------------
// les_model_from_string tests (Requirement 8.10)
// ---------------------------------------------------------------------------

TEST(LesOptionsTest, LesModelNone) {
  EXPECT_EQ(les_model_from_string("none"), LES_MODEL_NONE);
  EXPECT_EQ(les_model_from_string("none"), 0);
}

TEST(LesOptionsTest, LesModel3dSmagorinsky) {
  EXPECT_EQ(les_model_from_string("3d_smagorinsky"), LES_MODEL_3D_SMAGORINSKY);
  EXPECT_EQ(les_model_from_string("3d_smagorinsky"), 1);
}

TEST(LesOptionsTest, LesModelPrognostic15Order) {
  EXPECT_EQ(les_model_from_string("prognostic_1.5_order"), LES_MODEL_PROGNOSTIC_15_ORDER);
  EXPECT_EQ(les_model_from_string("prognostic_1.5_order"), 2);
}

TEST(LesOptionsTest, LesModelInvalidReturnsNegativeOne) {
  EXPECT_EQ(les_model_from_string("invalid"), LES_INVALID_OPT);
  EXPECT_EQ(les_model_from_string("invalid"), -1);
}

TEST(LesOptionsTest, LesModelEmptyStringIsInvalid) {
  EXPECT_EQ(les_model_from_string(""), LES_INVALID_OPT);
}

TEST(LesOptionsTest, LesModelCaseSensitive) {
  // The Reference_Model uses exact string comparison (no case folding)
  EXPECT_EQ(les_model_from_string("None"), LES_INVALID_OPT);
  EXPECT_EQ(les_model_from_string("NONE"), LES_INVALID_OPT);
  EXPECT_EQ(les_model_from_string("3D_SMAGORINSKY"), LES_INVALID_OPT);
}

// ---------------------------------------------------------------------------
// les_surface_from_string tests (Requirement 8.11)
// ---------------------------------------------------------------------------

TEST(LesOptionsTest, LesSurfaceNone) {
  EXPECT_EQ(les_surface_from_string("none"), LES_SURFACE_NONE);
  EXPECT_EQ(les_surface_from_string("none"), 0);
}

TEST(LesOptionsTest, LesSurfaceSpecified) {
  EXPECT_EQ(les_surface_from_string("specified"), LES_SURFACE_SPECIFIED);
  EXPECT_EQ(les_surface_from_string("specified"), 1);
}

TEST(LesOptionsTest, LesSurfaceVarying) {
  EXPECT_EQ(les_surface_from_string("varying"), LES_SURFACE_VARYING);
  EXPECT_EQ(les_surface_from_string("varying"), 2);
}

TEST(LesOptionsTest, LesSurfaceInvalidReturnsNegativeOne) {
  EXPECT_EQ(les_surface_from_string("invalid"), LES_INVALID_OPT);
  EXPECT_EQ(les_surface_from_string("invalid"), -1);
}

TEST(LesOptionsTest, LesSurfaceEmptyStringIsInvalid) {
  EXPECT_EQ(les_surface_from_string(""), LES_INVALID_OPT);
}

TEST(LesOptionsTest, LesSurfaceCaseSensitive) {
  EXPECT_EQ(les_surface_from_string("None"), LES_INVALID_OPT);
  EXPECT_EQ(les_surface_from_string("Specified"), LES_INVALID_OPT);
  EXPECT_EQ(les_surface_from_string("VARYING"), LES_INVALID_OPT);
}

// ---------------------------------------------------------------------------
// Compile-time evaluation (constexpr) verification
// ---------------------------------------------------------------------------

TEST(LesOptionsTest, ConstexprEvaluation) {
  // These functions are constexpr; verify they can be used at compile time
  static_assert(les_model_from_string("none") == 0);
  static_assert(les_model_from_string("3d_smagorinsky") == 1);
  static_assert(les_model_from_string("prognostic_1.5_order") == 2);
  static_assert(les_model_from_string("garbage") == -1);

  static_assert(les_surface_from_string("none") == 0);
  static_assert(les_surface_from_string("specified") == 1);
  static_assert(les_surface_from_string("varying") == 2);
  static_assert(les_surface_from_string("garbage") == -1);
}

}  // namespace
}  // namespace dycore
}  // namespace mpas
