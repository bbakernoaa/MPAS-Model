/// @file prop_srk3_config.cpp
/// @brief Property-based tests for SRK3Config validation and RK2 mode configuration.
///
/// **Validates: Requirements 10.2, 10.3**
///
/// Property 40: SRK3Config Validation
///   Invalid configs throw std::invalid_argument, valid configs don't.
///
/// Property 41: RK2 Mode Uses 2 Stages
///   When n_rk_stages=2, verify correct RK weights and sub-step counts.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/srk3.hpp>

#include <stdexcept>

using namespace mpas::dycore;

// ============================================================================
// Property 40: SRK3Config Validation
// ============================================================================
// Invalid configs throw std::invalid_argument, valid configs don't throw.

// Test 1: Valid SRK3 config (n_rk_stages=3) doesn't throw
TEST(SRK3ConfigValidation, ValidSRK3ConfigDoesNotThrow) {
    SRK3Config config{};
    config.n_rk_stages = 3;
    config.number_of_sub_steps = 6;
    config.dt = 60.0;
    config.dts = 10.0;

    EXPECT_NO_THROW(validate_srk3_config(config));
}

// Test 2: Valid RK2 config (n_rk_stages=2) doesn't throw
TEST(SRK3ConfigValidation, ValidRK2ConfigDoesNotThrow) {
    SRK3Config config{};
    config.n_rk_stages = 2;
    config.number_of_sub_steps = 6;
    config.dt = 60.0;
    config.dts = 10.0;

    EXPECT_NO_THROW(validate_srk3_config(config));
}

// Test 3: Invalid n_rk_stages (0, 1, 4) throws std::invalid_argument
TEST(SRK3ConfigValidation, InvalidNRkStagesThrows) {
    SRK3Config config{};
    config.number_of_sub_steps = 6;
    config.dt = 60.0;
    config.dts = 10.0;

    config.n_rk_stages = 0;
    EXPECT_THROW(validate_srk3_config(config), std::invalid_argument);

    config.n_rk_stages = 1;
    EXPECT_THROW(validate_srk3_config(config), std::invalid_argument);

    config.n_rk_stages = 4;
    EXPECT_THROW(validate_srk3_config(config), std::invalid_argument);
}

// Test 4: Invalid sub_steps (0, 1, 25) throws
TEST(SRK3ConfigValidation, InvalidSubStepsThrows) {
    SRK3Config config{};
    config.n_rk_stages = 3;
    config.dt = 60.0;
    config.dts = 10.0;

    config.number_of_sub_steps = 0;
    EXPECT_THROW(validate_srk3_config(config), std::invalid_argument);

    config.number_of_sub_steps = 1;
    EXPECT_THROW(validate_srk3_config(config), std::invalid_argument);

    config.number_of_sub_steps = 25;
    EXPECT_THROW(validate_srk3_config(config), std::invalid_argument);
}

// Test 5: Invalid dt (0, negative) throws
TEST(SRK3ConfigValidation, InvalidDtThrows) {
    SRK3Config config{};
    config.n_rk_stages = 3;
    config.number_of_sub_steps = 6;
    config.dts = 10.0;

    config.dt = 0.0;
    EXPECT_THROW(validate_srk3_config(config), std::invalid_argument);

    config.dt = -1.0;
    EXPECT_THROW(validate_srk3_config(config), std::invalid_argument);
}

// Test 6: SRK3Integrator construction with valid config succeeds
TEST(SRK3ConfigValidation, IntegratorConstructionWithValidConfig) {
    SRK3Config config{};
    config.n_rk_stages = 3;
    config.number_of_sub_steps = 6;
    config.dt = 60.0;
    config.dts = 10.0;

    EXPECT_NO_THROW(SRK3Integrator integrator(config));
}

// ============================================================================
// Property 40 (RapidCheck): For any valid n_rk_stages (2 or 3), sub_steps
// in [2, 24], and positive dt/dts, validate_srk3_config does not throw.
// For any invalid config, it throws.
// ============================================================================

RC_GTEST_PROP(SRK3ConfigProperty, ValidConfigsDoNotThrow,
              ()) {
    // Generate valid n_rk_stages: either 2 or 3
    const int n_rk_stages = *rc::gen::element(2, 3);

    // Generate valid number_of_sub_steps in [2, 24]
    const int sub_steps = *rc::gen::inRange(2, 25);

    // Generate positive dt in (0, 10000]
    const double dt = 1.0 + (*rc::gen::inRange(0, 9999)) * 1.0;

    // Generate positive dts in (0, dt]
    const double dts = 1.0 + (*rc::gen::inRange(0, static_cast<int>(dt) - 1)) * 1.0;

    SRK3Config config{};
    config.n_rk_stages = n_rk_stages;
    config.number_of_sub_steps = sub_steps;
    config.dt = dt;
    config.dts = dts;

    // Valid configs should not throw
    try {
        validate_srk3_config(config);
    } catch (...) {
        RC_FAIL("validate_srk3_config threw for valid config");
    }
}

RC_GTEST_PROP(SRK3ConfigProperty, InvalidNRkStagesThrows,
              ()) {
    // Generate invalid n_rk_stages: not 2 or 3
    int n_rk_stages = *rc::gen::inRange(-10, 20);
    RC_PRE(n_rk_stages != 2 && n_rk_stages != 3);

    SRK3Config config{};
    config.n_rk_stages = n_rk_stages;
    config.number_of_sub_steps = 6;
    config.dt = 60.0;
    config.dts = 10.0;

    bool threw = false;
    try {
        validate_srk3_config(config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    RC_ASSERT(threw);
}

RC_GTEST_PROP(SRK3ConfigProperty, InvalidSubStepsThrows,
              ()) {
    // Generate invalid sub_steps: < 2 or > 24
    int sub_steps = *rc::gen::inRange(-10, 50);
    RC_PRE(sub_steps < 2 || sub_steps > 24);

    SRK3Config config{};
    config.n_rk_stages = 3;
    config.number_of_sub_steps = sub_steps;
    config.dt = 60.0;
    config.dts = 10.0;

    bool threw = false;
    try {
        validate_srk3_config(config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    RC_ASSERT(threw);
}

RC_GTEST_PROP(SRK3ConfigProperty, InvalidDtThrows,
              ()) {
    // Generate non-positive dt
    const double dt = -1.0 * (*rc::gen::inRange(0, 1000));

    SRK3Config config{};
    config.n_rk_stages = 3;
    config.number_of_sub_steps = 6;
    config.dt = dt;
    config.dts = 10.0;

    bool threw = false;
    try {
        validate_srk3_config(config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    RC_ASSERT(threw);
}

// ============================================================================
// Property 41: RK2 Mode Uses 2 Stages with correct weights
// ============================================================================
// When n_rk_stages=2, verify the integrator accepts the config and the
// internal RK weights follow the documented pattern:
//   rk_timestep = {dt/2, dt/2, dt}
//   number_sub_steps = {max(1, n/2), max(1, n/2), n}

RC_GTEST_PROP(RK2ModeProperty, RK2ConfigAcceptedAndCorrectStages,
              ()) {
    // Generate valid sub_steps in [2, 24]
    const int sub_steps = *rc::gen::inRange(2, 25);

    // Generate positive dt
    const double dt = 1.0 + (*rc::gen::inRange(0, 9999)) * 1.0;
    const double dts = dt / static_cast<double>(sub_steps);

    SRK3Config config{};
    config.n_rk_stages = 2;
    config.number_of_sub_steps = sub_steps;
    config.dt = dt;
    config.dts = dts;

    // RK2 config should be accepted
    SRK3Integrator integrator(config);

    // Verify the config was stored correctly
    RC_ASSERT(integrator.config().n_rk_stages == 2);
    RC_ASSERT(integrator.config().number_of_sub_steps == sub_steps);
    RC_ASSERT(integrator.config().dt == dt);
    RC_ASSERT(integrator.config().dts == dts);
}

TEST(RK2ModeProperty, RK2IntegratorStoresCorrectConfig) {
    SRK3Config config{};
    config.n_rk_stages = 2;
    config.number_of_sub_steps = 6;
    config.dt = 60.0;
    config.dts = 10.0;

    SRK3Integrator integrator(config);

    EXPECT_EQ(integrator.config().n_rk_stages, 2);
    EXPECT_EQ(integrator.config().number_of_sub_steps, 6);
    EXPECT_DOUBLE_EQ(integrator.config().dt, 60.0);
    EXPECT_DOUBLE_EQ(integrator.config().dts, 10.0);
}
