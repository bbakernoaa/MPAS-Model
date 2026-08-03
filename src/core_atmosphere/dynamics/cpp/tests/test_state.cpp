/// @file test_state.cpp
/// @brief Unit tests for DycoreState and error handling (api_wrap).

#include <mpas_dycore/state.hpp>
#include <mpas_dycore/error.hpp>

#include <gtest/gtest.h>
#include <stdexcept>
#include <cstring>

namespace mpas::dycore {
namespace {

// ============================================================================
// DycoreState tests
// ============================================================================

TEST(DycoreStateTest, DefaultStatusIsUninitialized) {
    DycoreState state;
    EXPECT_EQ(state.status, DycoreStatus::Uninitialized);
}

TEST(DycoreStateTest, DefaultDimensionsAreZero) {
    DycoreState state;
    EXPECT_EQ(state.nCells, 0);
    EXPECT_EQ(state.nEdges, 0);
    EXPECT_EQ(state.nVertices, 0);
    EXPECT_EQ(state.maxEdges, 0);
    EXPECT_EQ(state.nVertLevels, 0);
    EXPECT_EQ(state.nScalars, 0);
}

TEST(DycoreStateTest, DefaultCommIsNull) {
    DycoreState state;
    EXPECT_EQ(state.comm, MPI_COMM_NULL);
}

TEST(DycoreStateTest, DefaultConfigIsZero) {
    DycoreState state;
    EXPECT_EQ(state.dt, 0.0);
    EXPECT_EQ(state.number_of_sub_steps, 0);
}

TEST(DycoreStateTest, SingletonReturnsSameInstance) {
    DycoreState& a = get_dycore_state();
    DycoreState& b = get_dycore_state();
    EXPECT_EQ(&a, &b);
}

TEST(DycoreStateTest, SingletonMutationsAreVisible) {
    DycoreState& state = get_dycore_state();
    auto original_status = state.status;

    state.status = DycoreStatus::Ready;
    EXPECT_EQ(get_dycore_state().status, DycoreStatus::Ready);

    // Restore original state for other tests
    state.status = original_status;
}

// ============================================================================
// ErrorInfo tests
// ============================================================================

TEST(ErrorInfoTest, StructLayout) {
    ErrorInfo info{};
    info.code = 0;
    info.message[0] = '\0';
    EXPECT_EQ(info.code, 0);
    EXPECT_EQ(info.message[0], '\0');
}

// ============================================================================
// api_wrap tests
// ============================================================================

TEST(ApiWrapTest, SuccessReturnsZero) {
    char errmsg[256] = {};
    int result = api_wrap(errmsg, 256, []() {
        // no-op, success
    });
    EXPECT_EQ(result, 0);
}

TEST(ApiWrapTest, StdExceptionReturnsOne) {
    char errmsg[256] = {};
    int result = api_wrap(errmsg, 256, []() {
        throw std::runtime_error("test error message");
    });
    EXPECT_EQ(result, 1);
    EXPECT_STREQ(errmsg, "test error message");
}

TEST(ApiWrapTest, UnknownExceptionReturnsTwo) {
    char errmsg[256] = {};
    int result = api_wrap(errmsg, 256, []() {
        throw 42;  // non-std exception
    });
    EXPECT_EQ(result, 2);
    EXPECT_STREQ(errmsg, "unknown exception");
}

TEST(ApiWrapTest, MessageTruncatedToBufferLength) {
    char errmsg[16] = {};
    int result = api_wrap(errmsg, 16, []() {
        throw std::runtime_error("this is a very long error message that exceeds the buffer");
    });
    EXPECT_EQ(result, 1);
    // Should be null-terminated and not overflow
    EXPECT_EQ(errmsg[15], '\0');
    EXPECT_EQ(std::strlen(errmsg), 15u);
}

TEST(ApiWrapTest, NullBufferDoesNotCrash) {
    int result = api_wrap(nullptr, 0, []() {
        throw std::runtime_error("test");
    });
    EXPECT_EQ(result, 1);
}

TEST(ApiWrapTest, ZeroLengthBufferDoesNotCrash) {
    char errmsg[1] = {'X'};
    int result = api_wrap(errmsg, 0, []() {
        throw std::runtime_error("test");
    });
    EXPECT_EQ(result, 1);
    // Buffer should be unchanged when length is 0
    EXPECT_EQ(errmsg[0], 'X');
}

} // anonymous namespace
} // namespace mpas::dycore
