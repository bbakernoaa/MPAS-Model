/// @file prop_execution_policy.cpp
/// @brief Property-based tests for execution policy abstraction.
///
/// **Validates: Requirements 9.2, 9.3, 9.6**

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <mpas_dycore/execution_policy.hpp>

#include <vector>
#include <numeric>
#include <atomic>

using namespace mpas::dycore;

// ============================================================================
// Property 20: Execution Policy Complete Coverage
// ============================================================================
// For any (n_entities, n_levels), parallel_for visits all (entity, level)
// pairs exactly once.

RC_GTEST_PROP(ExecutionPolicyCoverage, SerialPolicyVisitsAllPairsExactlyOnce,
              ()) {
    const auto n_entities = *rc::gen::inRange(1, 501);
    const auto n_levels = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(n_entities) * n_levels;

    // Create a visit counter for each (entity, level) pair
    std::vector<int> visit_count(total, 0);

    SerialPolicy policy;
    policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            visit_count[idx]++;
        });

    // Every pair must be visited exactly once
    for (std::size_t idx = 0; idx < total; ++idx) {
        RC_ASSERT(visit_count[idx] == 1);
    }
}

RC_GTEST_PROP(ExecutionPolicyCoverage, OpenMPPolicyVisitsAllPairs,
              ()) {
    const auto n_entities = *rc::gen::inRange(1, 501);
    const auto n_levels = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(n_entities) * n_levels;

    // Use atomic counters since OpenMP may access concurrently
    std::vector<std::atomic<int>> visit_count(total);
    for (auto& v : visit_count) {
        v.store(0, std::memory_order_relaxed);
    }

    OpenMPPolicy policy;
    policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            visit_count[idx].fetch_add(1, std::memory_order_relaxed);
        });

    // Every pair must be visited exactly once
    for (std::size_t idx = 0; idx < total; ++idx) {
        RC_ASSERT(visit_count[idx].load(std::memory_order_relaxed) == 1);
    }
}

RC_GTEST_PROP(ExecutionPolicyCoverage, GPUPolicyVisitsAllPairsExactlyOnce,
              ()) {
    const auto n_entities = *rc::gen::inRange(1, 501);
    const auto n_levels = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(n_entities) * n_levels;

    // GPUPolicy is currently a serial fallback stub, so no atomics needed
    std::vector<int> visit_count(total, 0);

    GPUPolicy policy;
    policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            visit_count[idx]++;
        });

    // Every pair must be visited exactly once
    for (std::size_t idx = 0; idx < total; ++idx) {
        RC_ASSERT(visit_count[idx] == 1);
    }
}

// ============================================================================
// Property 21: Policy Numerical Equivalence
// ============================================================================
// SerialPolicy and OpenMPPolicy produce identical results on a reduction
// (given associative, commutative integer operations).

RC_GTEST_PROP(PolicyNumericalEquivalence, SerialAndOpenMPProduceSameIntegerSum,
              ()) {
    const auto n_entities = *rc::gen::inRange(1, 501);
    const auto n_levels = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(n_entities) * n_levels;

    // Generate random integer data to avoid FP non-associativity issues
    auto data = *rc::gen::container<std::vector<int>>(
        total, rc::gen::inRange(-1000, 1001));

    // Compute per-entity sums using SerialPolicy
    std::vector<long long> serial_result(static_cast<std::size_t>(n_entities), 0);
    SerialPolicy serial_policy;
    serial_policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            serial_result[static_cast<std::size_t>(iEntity)] += data[idx];
        });

    // Compute per-entity sums using OpenMPPolicy
    // Since each entity is processed by a single thread (outer loop parallelized),
    // the inner level accumulation is sequential and deterministic.
    std::vector<long long> omp_result(static_cast<std::size_t>(n_entities), 0);
    OpenMPPolicy omp_policy;
    omp_policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            omp_result[static_cast<std::size_t>(iEntity)] += data[idx];
        });

    // Results must be identical
    for (index_type i = 0; i < n_entities; ++i) {
        RC_ASSERT(serial_result[static_cast<std::size_t>(i)]
                  == omp_result[static_cast<std::size_t>(i)]);
    }
}

RC_GTEST_PROP(PolicyNumericalEquivalence, SerialAndGPUStubProduceSameResult,
              ()) {
    const auto n_entities = *rc::gen::inRange(1, 501);
    const auto n_levels = *rc::gen::inRange(1, 101);
    const std::size_t total = static_cast<std::size_t>(n_entities) * n_levels;

    // Generate random integer data
    auto data = *rc::gen::container<std::vector<int>>(
        total, rc::gen::inRange(-1000, 1001));

    // Compute per-entity sums using SerialPolicy
    std::vector<long long> serial_result(static_cast<std::size_t>(n_entities), 0);
    SerialPolicy serial_policy;
    serial_policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            serial_result[static_cast<std::size_t>(iEntity)] += data[idx];
        });

    // Compute per-entity sums using GPUPolicy (stub)
    std::vector<long long> gpu_result(static_cast<std::size_t>(n_entities), 0);
    GPUPolicy gpu_policy;
    gpu_policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            gpu_result[static_cast<std::size_t>(iEntity)] += data[idx];
        });

    // Results must be identical
    for (index_type i = 0; i < n_entities; ++i) {
        RC_ASSERT(serial_result[static_cast<std::size_t>(i)]
                  == gpu_result[static_cast<std::size_t>(i)]);
    }
}

RC_GTEST_PROP(PolicyNumericalEquivalence, AllPoliciesProduceSameTotalSum,
              ()) {
    const auto n_entities = *rc::gen::inRange(1, 201);
    const auto n_levels = *rc::gen::inRange(1, 51);
    const std::size_t total = static_cast<std::size_t>(n_entities) * n_levels;

    // Generate random integer data
    auto data = *rc::gen::container<std::vector<int>>(
        total, rc::gen::inRange(-500, 501));

    // Compute total sum with SerialPolicy
    long long serial_sum = 0;
    SerialPolicy serial_policy;
    serial_policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            serial_sum += data[idx];
        });

    // Compute total sum with OpenMPPolicy using atomic accumulation
    std::atomic<long long> omp_sum{0};
    OpenMPPolicy omp_policy;
    omp_policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            omp_sum.fetch_add(data[idx], std::memory_order_relaxed);
        });

    // Compute total sum with GPUPolicy
    long long gpu_sum = 0;
    GPUPolicy gpu_policy;
    gpu_policy.parallel_for(n_entities, n_levels,
        [&](index_type iEntity, index_type k) {
            const std::size_t idx = static_cast<std::size_t>(iEntity) * n_levels + k;
            gpu_sum += data[idx];
        });

    RC_ASSERT(serial_sum == omp_sum.load(std::memory_order_relaxed));
    RC_ASSERT(serial_sum == gpu_sum);
}
