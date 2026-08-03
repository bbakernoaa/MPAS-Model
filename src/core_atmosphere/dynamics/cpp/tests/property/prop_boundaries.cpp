/// @file prop_boundaries.cpp
/// @brief Property-based tests for lateral boundary condition kernels.
///
/// **Validates: Requirements 12.1, 12.2, 12.4**
///
/// Property 26: Specified Zone Overwrite
///   Cells in the specified zone (specZoneMask==1) have their field values
///   replaced entirely by boundary data.
///
/// Property 27: Relaxation Zone Coefficient Formula
///   For cells with relax_coef > 0 and specZoneMask != 1, the result is
///   blended between the original field value and the boundary value using
///   field = field + dt * relax_coef * (field_bdy - field).
///
/// Property 28: Disabled Boundaries Produce No Modification
///   Cells with relax_coef=0 and specZoneMask=0 are unchanged after
///   apply_lateral_boundary.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/kernels/boundaries.hpp>
#include <mpas_dycore/types.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace mpas::dycore;
using namespace mpas::dycore::kernels;

// ============================================================================
// Property 26: Specified Zone Overwrite
// ============================================================================
// Cells in the specified zone (specZoneMask==1.0) have their field values
// replaced entirely by boundary data regardless of original values.

RC_GTEST_PROP(SpecifiedZoneOverwrite, FieldReplacedByBoundaryData,
              ()) {
    // Generate random mesh dimensions
    const auto nCells = *rc::gen::inRange(1, 20);
    const auto nVertLevels = *rc::gen::inRange(1, 20);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Generate a positive timestep
    const int dt_raw = *rc::gen::inRange(1, 10000);
    const real_type dt = static_cast<real_type>(dt_raw) / 100.0;

    // Generate field values
    std::vector<real_type> field_data(nLev * nC);
    for (std::size_t i = 0; i < field_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        field_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Generate boundary field values
    std::vector<real_type> field_bdy_data(nLev * nC);
    for (std::size_t i = 0; i < field_bdy_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        field_bdy_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Generate specZoneMask: at least one cell in the specified zone
    std::vector<real_type> specZoneMask_data(nC, 0.0);
    // Mark some cells as specified zone
    for (std::size_t i = 0; i < nC; ++i) {
        bool inSpecZone = *rc::gen::arbitrary<bool>();
        specZoneMask_data[i] = inSpecZone ? 1.0 : 0.0;
    }
    // Ensure at least one cell is in the specified zone
    specZoneMask_data[0] = 1.0;

    // Generate relax_coef: zero for all cells (so only spec zone matters)
    std::vector<real_type> relax_coef_data(nC, 0.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> field(
        field_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> field_bdy(
        field_bdy_data.data(), nVertLevels, nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        relax_coef(relax_coef_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        specZoneMask(specZoneMask_data.data(), nCells);

    // Call the kernel
    apply_lateral_boundary<default_layout>(
        SerialPolicy{},
        field, field_bdy,
        relax_coef, specZoneMask,
        dt,
        static_cast<index_type>(nCells),
        static_cast<index_type>(nVertLevels));

    // Verify: cells in the specified zone now have boundary values
    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        if (specZoneMask_data[static_cast<std::size_t>(iCell)] == 1.0) {
            for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
                auto actual = field[k, iCell];
                auto expected = field_bdy[k, iCell];
                RC_ASSERT(actual == expected);
            }
        }
    }
}

// ============================================================================
// Property 28: Interior Cells Unchanged (Disabled Boundaries)
// ============================================================================
// Cells with relax_coef=0 and specZoneMask=0 remain unchanged after the
// boundary condition application.

RC_GTEST_PROP(InteriorCellsUnchanged, FieldUnmodifiedForInteriorCells,
              ()) {
    // Generate random mesh dimensions
    const auto nCells = *rc::gen::inRange(1, 20);
    const auto nVertLevels = *rc::gen::inRange(1, 20);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Generate a positive timestep
    const int dt_raw = *rc::gen::inRange(1, 10000);
    const real_type dt = static_cast<real_type>(dt_raw) / 100.0;

    // Generate field values
    std::vector<real_type> field_data(nLev * nC);
    for (std::size_t i = 0; i < field_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        field_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Save original for comparison
    std::vector<real_type> field_original(field_data);

    // Generate arbitrary boundary data (should not matter for interior cells)
    std::vector<real_type> field_bdy_data(nLev * nC);
    for (std::size_t i = 0; i < field_bdy_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        field_bdy_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // All cells are interior: relax_coef=0, specZoneMask=0
    std::vector<real_type> relax_coef_data(nC, 0.0);
    std::vector<real_type> specZoneMask_data(nC, 0.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> field(
        field_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> field_bdy(
        field_bdy_data.data(), nVertLevels, nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        relax_coef(relax_coef_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        specZoneMask(specZoneMask_data.data(), nCells);

    // Call the kernel
    apply_lateral_boundary<default_layout>(
        SerialPolicy{},
        field, field_bdy,
        relax_coef, specZoneMask,
        dt,
        static_cast<index_type>(nCells),
        static_cast<index_type>(nVertLevels));

    // Verify: ALL cells unchanged (bit-for-bit identical)
    for (std::size_t i = 0; i < field_data.size(); ++i) {
        RC_ASSERT(field_data[i] == field_original[i]);
    }
}

// ============================================================================
// Property 27: Relaxation Zone Blends Toward Boundary
// ============================================================================
// For cells with relax_coef > 0 and specZoneMask != 1, the result lies
// between the original field value and the boundary value (inclusive).
// Specifically: field_new = field_old + dt * relax_coef * (field_bdy - field_old)

RC_GTEST_PROP(RelaxationZoneBlends, ResultBetweenFieldAndBoundary,
              ()) {
    // Generate random mesh dimensions
    const auto nCells = *rc::gen::inRange(1, 20);
    const auto nVertLevels = *rc::gen::inRange(1, 20);

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Generate a positive timestep such that dt * max_relax_coef <= 1.0
    // to ensure blending stays between original and boundary values.
    // Use dt * relax_coef in (0, 1] to guarantee monotone blending.
    const int dt_raw = *rc::gen::inRange(1, 100); // dt in [0.01, 1.0]
    const real_type dt = static_cast<real_type>(dt_raw) / 100.0;

    // Generate field values
    std::vector<real_type> field_data(nLev * nC);
    for (std::size_t i = 0; i < field_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        field_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // Save original for comparison
    std::vector<real_type> field_original(field_data);

    // Generate boundary field values
    std::vector<real_type> field_bdy_data(nLev * nC);
    for (std::size_t i = 0; i < field_bdy_data.size(); ++i) {
        int raw = *rc::gen::inRange(-100000, 100001);
        field_bdy_data[i] = static_cast<real_type>(raw) / 1000.0;
    }

    // All cells in relaxation zone: positive relax_coef, specZoneMask=0
    // relax_coef chosen so that dt * relax_coef <= 1.0
    std::vector<real_type> relax_coef_data(nC);
    for (std::size_t i = 0; i < nC; ++i) {
        // relax_coef in (0, 1/dt] to ensure blending factor alpha = dt*relax_coef in (0, 1]
        int raw = *rc::gen::inRange(1, 101); // [0.01, 1.01) -> / (100*dt) -> in (0, 1/dt]
        relax_coef_data[i] = static_cast<real_type>(raw) / (100.0 * dt);
        // Clamp to ensure dt * relax_coef <= 1.0
        if (dt * relax_coef_data[i] > 1.0) {
            relax_coef_data[i] = 1.0 / dt;
        }
    }

    std::vector<real_type> specZoneMask_data(nC, 0.0);

    // Create mdspan views
    Field2D<default_layout, unchecked_accessor> field(
        field_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> field_bdy(
        field_bdy_data.data(), nVertLevels, nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        relax_coef(relax_coef_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        specZoneMask(specZoneMask_data.data(), nCells);

    // Call the kernel
    apply_lateral_boundary<default_layout>(
        SerialPolicy{},
        field, field_bdy,
        relax_coef, specZoneMask,
        dt,
        static_cast<index_type>(nCells),
        static_cast<index_type>(nVertLevels));

    // Verify: for each cell in the relaxation zone, the result is between
    // the original field value and the boundary value.
    // field_new = field_old + alpha * (field_bdy - field_old)
    //           = (1 - alpha) * field_old + alpha * field_bdy
    // where alpha = dt * relax_coef in (0, 1]
    for (index_type iCell = 0; iCell < static_cast<index_type>(nCells); ++iCell) {
        const real_type alpha = dt * relax_coef_data[static_cast<std::size_t>(iCell)];
        RC_PRE(alpha > 0.0 && alpha <= 1.0);

        for (index_type k = 0; k < static_cast<index_type>(nVertLevels); ++k) {
            // Compute flat index for column-major layout: k + nVertLevels * iCell
            const std::size_t idx = static_cast<std::size_t>(k) +
                                    nLev * static_cast<std::size_t>(iCell);
            const real_type old_val = field_original[idx];
            const real_type bdy_val = field_bdy_data[idx];
            const real_type new_val = field_data[idx];

            // Check exact formula: new = old + alpha * (bdy - old)
            const real_type expected = old_val + alpha * (bdy_val - old_val);
            RC_ASSERT(std::abs(new_val - expected) < 1.0e-12);

            // Also verify the result is between old and bdy (or equal to one)
            const real_type lo = std::min(old_val, bdy_val);
            const real_type hi = std::max(old_val, bdy_val);
            RC_ASSERT(new_val >= lo - 1.0e-12);
            RC_ASSERT(new_val <= hi + 1.0e-12);
        }
    }
}

// ============================================================================
// Deterministic Test: Zero-size inputs produce no modification
// ============================================================================
// When nCells=0 or nVertLevels=0, the kernel returns immediately.

TEST(BoundaryConditionsDeterministic, ZeroSizeInputsNoOp) {
    // nCells=0
    {
        std::vector<real_type> field_data(1, 42.0);
        std::vector<real_type> field_bdy_data(1, 99.0);
        std::vector<real_type> relax_coef_data(1, 1.0);
        std::vector<real_type> specZoneMask_data(1, 1.0);

        Field2D<default_layout, unchecked_accessor> field(
            field_data.data(), 1, 0);
        ConstField2D<default_layout, unchecked_accessor> field_bdy(
            field_bdy_data.data(), 1, 0);
        std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
            relax_coef(relax_coef_data.data(), 0);
        std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
            specZoneMask(specZoneMask_data.data(), 0);

        apply_lateral_boundary<default_layout>(
            SerialPolicy{},
            field, field_bdy,
            relax_coef, specZoneMask,
            1.0, 0, 1);

        // Field should be unchanged since nCells=0
        EXPECT_DOUBLE_EQ(field_data[0], 42.0);
    }

    // nVertLevels=0
    {
        std::vector<real_type> field_data(1, 42.0);
        std::vector<real_type> field_bdy_data(1, 99.0);
        std::vector<real_type> relax_coef_data(1, 1.0);
        std::vector<real_type> specZoneMask_data(1, 1.0);

        Field2D<default_layout, unchecked_accessor> field(
            field_data.data(), 0, 1);
        ConstField2D<default_layout, unchecked_accessor> field_bdy(
            field_bdy_data.data(), 0, 1);
        std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
            relax_coef(relax_coef_data.data(), 1);
        std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
            specZoneMask(specZoneMask_data.data(), 1);

        apply_lateral_boundary<default_layout>(
            SerialPolicy{},
            field, field_bdy,
            relax_coef, specZoneMask,
            1.0, 1, 0);

        // Field should be unchanged since nVertLevels=0
        EXPECT_DOUBLE_EQ(field_data[0], 42.0);
    }
}

// ============================================================================
// Deterministic Test: Mixed zones known-answer
// ============================================================================
// Verify correct behavior when cells have a mix of specified, relaxation, and
// interior zones.

TEST(BoundaryConditionsDeterministic, MixedZonesKnownAnswer) {
    const index_type nCells = 4;
    const index_type nVertLevels = 2;

    const auto nLev = static_cast<std::size_t>(nVertLevels);
    const auto nC   = static_cast<std::size_t>(nCells);

    // Field: all 10.0
    std::vector<real_type> field_data(nLev * nC, 10.0);
    // Boundary: all 20.0
    std::vector<real_type> field_bdy_data(nLev * nC, 20.0);

    // Cell 0: specified zone (mask=1, coef=0)
    // Cell 1: relaxation zone (mask=0, coef=0.5)
    // Cell 2: interior (mask=0, coef=0)
    // Cell 3: relaxation zone (mask=0, coef=1.0)
    std::vector<real_type> specZoneMask_data = {1.0, 0.0, 0.0, 0.0};
    std::vector<real_type> relax_coef_data   = {0.0, 0.5, 0.0, 1.0};

    const real_type dt = 1.0;

    Field2D<default_layout, unchecked_accessor> field(
        field_data.data(), nVertLevels, nCells);
    ConstField2D<default_layout, unchecked_accessor> field_bdy(
        field_bdy_data.data(), nVertLevels, nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        relax_coef(relax_coef_data.data(), nCells);
    std::mdspan<const real_type, std::extents<index_type, std::dynamic_extent>>
        specZoneMask(specZoneMask_data.data(), nCells);

    apply_lateral_boundary<default_layout>(
        SerialPolicy{},
        field, field_bdy,
        relax_coef, specZoneMask,
        dt,
        nCells,
        nVertLevels);

    // Cell 0 (specified zone): field = boundary = 20.0
    for (index_type k = 0; k < nVertLevels; ++k) {
        auto val = field[k, 0];
        EXPECT_DOUBLE_EQ(val, 20.0)
            << "Specified zone cell should be overwritten at k=" << k;
    }

    // Cell 1 (relaxation, coef=0.5, dt=1.0):
    // field = 10 + 1.0 * 0.5 * (20 - 10) = 10 + 5 = 15.0
    for (index_type k = 0; k < nVertLevels; ++k) {
        auto val = field[k, 1];
        EXPECT_DOUBLE_EQ(val, 15.0)
            << "Relaxation zone cell (coef=0.5) should blend at k=" << k;
    }

    // Cell 2 (interior): field = 10.0 unchanged
    for (index_type k = 0; k < nVertLevels; ++k) {
        auto val = field[k, 2];
        EXPECT_DOUBLE_EQ(val, 10.0)
            << "Interior cell should be unchanged at k=" << k;
    }

    // Cell 3 (relaxation, coef=1.0, dt=1.0):
    // field = 10 + 1.0 * 1.0 * (20 - 10) = 10 + 10 = 20.0
    for (index_type k = 0; k < nVertLevels; ++k) {
        auto val = field[k, 3];
        EXPECT_DOUBLE_EQ(val, 20.0)
            << "Relaxation zone cell (coef=1.0) should fully blend at k=" << k;
    }
}
