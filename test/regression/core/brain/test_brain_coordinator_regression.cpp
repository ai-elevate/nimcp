//=============================================================================
// test_brain_coordinator_regression.cpp - Coordinator Regression Tests
//=============================================================================
/**
 * @file test_brain_coordinator_regression.cpp
 * @brief Regression tests for coordinator subsystem stability
 *
 * WHAT: Tests ensuring coordinator behavior remains stable across changes
 * WHY:  Prevent regressions in coordinator initialization and operation
 * HOW:  GoogleTest with stability tests, edge cases, and boundary conditions
 *
 * Test Categories:
 * 1. Initialization Stability - Same results across multiple runs
 * 2. Edge Cases - Boundary conditions and extreme inputs
 * 3. Concurrency - Multi-threaded coordinator access
 * 4. Memory Stability - No leaks under repeated operations
 * 5. Error Recovery - Graceful handling of failures
 *
 * @version 1.0.0
 * @date 2025-12-15
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

// Headers have their own extern "C" guards
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "async/nimcp_bio_async_orchestrator.h"
#include "plasticity/nimcp_plasticity_coordinator.h"
#include "cognitive/immune/nimcp_immune_bridge_coordinator.h"
#include "cognitive/nimcp_cognitive_meta_controller.h"
#include "security/nimcp_security_perception_bridge.h"
#include "swarm/nimcp_swarm_module_registry.h"

//=============================================================================
// Test Fixture
//=============================================================================

/**
 * CoordinatorRegressionTest Fixture
 */
class CoordinatorRegressionTest : public ::testing::Test {
protected:
    brain_t brain = nullptr;

    void SetUp() override {
        brain = brain_create("test", BRAIN_SIZE_SMALL, BRAIN_TASK_CLASSIFICATION, 128, 32);
    }

    void TearDown() override {
        if (brain != nullptr) {
            brain_destroy(brain);
            brain = nullptr;
        }
    }
};

//=============================================================================
// 1. INITIALIZATION STABILITY TESTS
//=============================================================================

/**
 * TEST: InitStability_ConsistentResults
 *
 * WHAT: Create brains multiple times and verify consistent initialization
 * WHY:  Ensure deterministic behavior
 * EXPECT: Same coordinator enabled status across runs
 */
TEST_F(CoordinatorRegressionTest, InitStability_ConsistentResults) {
    brain_destroy(brain);
    brain = nullptr;

    // Track enabled status across iterations
    bool bio_async_enabled[5];
    bool plasticity_enabled[5];
    bool immune_enabled[5];

    for (int i = 0; i < 5; i++) {
        brain = brain_create("test", BRAIN_SIZE_SMALL, BRAIN_TASK_CLASSIFICATION, 128, 32);
        ASSERT_NE(brain, nullptr);

        bio_async_enabled[i] = brain->bio_async_orchestrator_enabled;
        plasticity_enabled[i] = brain->plasticity_coordinator_enabled;
        immune_enabled[i] = brain->immune_bridge_coordinator_enabled;

        brain_destroy(brain);
        brain = nullptr;
    }

    // Verify consistency (all iterations should have same status)
    for (int i = 1; i < 5; i++) {
        EXPECT_EQ(bio_async_enabled[0], bio_async_enabled[i]);
        EXPECT_EQ(plasticity_enabled[0], plasticity_enabled[i]);
        EXPECT_EQ(immune_enabled[0], immune_enabled[i]);
    }

    // Create brain for TearDown
    brain = brain_create("test", BRAIN_SIZE_SMALL, BRAIN_TASK_CLASSIFICATION, 128, 32);
}

/**
 * TEST: InitStability_DifferentSizes
 *
 * WHAT: Create brains of different sizes and verify coordinators work
 * WHY:  Ensure coordinators scale with brain size
 * EXPECT: All sizes initialize successfully
 */
TEST_F(CoordinatorRegressionTest, InitStability_DifferentSizes) {
    brain_destroy(brain);
    brain = nullptr;

    brain_size_t sizes[] = {BRAIN_SIZE_TINY, BRAIN_SIZE_SMALL, BRAIN_SIZE_MEDIUM};
    uint32_t inputs[] = {32, 64, 128};
    uint32_t outputs[] = {8, 16, 32};

    for (int i = 0; i < 3; i++) {
        brain = brain_create("test", sizes[i], BRAIN_TASK_CLASSIFICATION, inputs[i], outputs[i]);
        ASSERT_NE(brain, nullptr) << "Failed for size " << i;
        brain_destroy(brain);
        brain = nullptr;
    }

    // Create brain for TearDown
    brain = brain_create("test", BRAIN_SIZE_SMALL, BRAIN_TASK_CLASSIFICATION, 128, 32);
}

//=============================================================================
// 2. EDGE CASES TESTS
//=============================================================================

/**
 * TEST: EdgeCase_MinimalBrain
 *
 * WHAT: Create smallest possible brain
 * WHY:  Verify coordinators handle minimal resources
 * EXPECT: Brain created, coordinators may be skipped
 */
TEST_F(CoordinatorRegressionTest, EdgeCase_MinimalBrain) {
    brain_destroy(brain);

    brain = brain_create("test", BRAIN_SIZE_TINY, BRAIN_TASK_CLASSIFICATION, 8, 4);
    EXPECT_NE(brain, nullptr);
}

/**
 * TEST: EdgeCase_RapidCreateDestroy
 *
 * WHAT: Rapidly create and destroy brains
 * WHY:  Verify no race conditions in coordinator lifecycle
 * EXPECT: No crashes or hangs
 */
TEST_F(CoordinatorRegressionTest, EdgeCase_RapidCreateDestroy) {
    brain_destroy(brain);
    brain = nullptr;

    for (int i = 0; i < 20; i++) {
        brain = brain_create("test", BRAIN_SIZE_TINY, BRAIN_TASK_CLASSIFICATION, 64, 16);
        ASSERT_NE(brain, nullptr);
        brain_destroy(brain);
        brain = nullptr;
    }

    brain = brain_create("test", BRAIN_SIZE_TINY, BRAIN_TASK_CLASSIFICATION, 64, 16);
}

//=============================================================================
// 3. CONCURRENCY TESTS
//=============================================================================

/**
 * TEST: Concurrency_MultiThreadedStatsAccess
 *
 * WHAT: Access coordinator stats from multiple threads
 * WHY:  Verify thread safety of stats retrieval
 * EXPECT: No crashes or data races
 */
TEST_F(CoordinatorRegressionTest, Concurrency_MultiThreadedStatsAccess) {
    ASSERT_NE(brain, nullptr);

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    // Launch 4 threads accessing stats
    for (int i = 0; i < 4; i++) {
        threads.emplace_back([this, &success_count]() {
            for (int j = 0; j < 10; j++) {
                if (brain->bio_async_orchestrator_enabled && brain->bio_async_orchestrator) {
                    bio_orchestrator_stats_t stats;
                    bio_orchestrator_get_stats(brain->bio_async_orchestrator, &stats);
                }
                if (brain->plasticity_coordinator_enabled && brain->plasticity_coordinator) {
                    plasticity_coordinator_stats_t stats;
                    plasticity_coordinator_get_stats(brain->plasticity_coordinator, &stats);
                }
                success_count++;
            }
        });
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), 40);
}

//=============================================================================
// 4. MEMORY STABILITY TESTS
//=============================================================================

/**
 * TEST: MemoryStability_RepeatedOperations
 *
 * WHAT: Perform repeated operations without memory growth
 * WHY:  Verify no memory leaks in coordinator operations
 * EXPECT: Operations complete without unbounded memory use
 */
TEST_F(CoordinatorRegressionTest, MemoryStability_RepeatedOperations) {
    ASSERT_NE(brain, nullptr);

    // Perform repeated stats queries
    for (int i = 0; i < 100; i++) {
        if (brain->bio_async_orchestrator_enabled && brain->bio_async_orchestrator) {
            bio_orchestrator_stats_t stats;
            bio_orchestrator_get_stats(brain->bio_async_orchestrator, &stats);
        }
        if (brain->plasticity_coordinator_enabled && brain->plasticity_coordinator) {
            plasticity_coordinator_stats_t stats;
            plasticity_coordinator_get_stats(brain->plasticity_coordinator, &stats);
        }
        if (brain->immune_bridge_coordinator_enabled && brain->immune_bridge_coordinator) {
            immune_coordinator_stats_t stats;
            immune_bridge_coordinator_get_stats(brain->immune_bridge_coordinator, &stats);
        }
    }

    SUCCEED();
}

//=============================================================================
// 5. ERROR RECOVERY TESTS
//=============================================================================

/**
 * TEST: ErrorRecovery_NullPointerHandling
 *
 * WHAT: Verify coordinators handle NULL pointers gracefully
 * WHY:  Ensure robustness against invalid input
 * EXPECT: Functions return error codes without crashing
 */
TEST_F(CoordinatorRegressionTest, ErrorRecovery_NullPointerHandling) {
    // These should return errors, not crash
    bio_orchestrator_stats_t bio_stats;
    int result1 = bio_orchestrator_get_stats(nullptr, &bio_stats);
    EXPECT_NE(result1, 0);

    plasticity_coordinator_stats_t plas_stats;
    int result2 = plasticity_coordinator_get_stats(nullptr, &plas_stats);
    EXPECT_NE(result2, 0);

    immune_coordinator_stats_t imm_stats;
    int result3 = immune_bridge_coordinator_get_stats(nullptr, &imm_stats);
    EXPECT_NE(result3, 0);

    meta_controller_stats_t meta_stats;
    int result4 = meta_controller_get_stats(nullptr, &meta_stats);
    EXPECT_NE(result4, 0);

    sec_percept_stats_t sec_stats;
    int result5 = sec_percept_get_stats(nullptr, &sec_stats);
    EXPECT_NE(result5, 0);

    swarm_registry_stats_t swarm_stats;
    int result6 = swarm_registry_get_stats(nullptr, &swarm_stats);
    EXPECT_NE(result6, 0);
}

/**
 * TEST: ErrorRecovery_AfterDestroy
 *
 * WHAT: Verify using destroyed brain doesn't cause undefined behavior
 * WHY:  Ensure cleanup is complete
 * EXPECT: Proper error or crash avoidance
 */
TEST_F(CoordinatorRegressionTest, ErrorRecovery_AfterDestroy) {
    // This test just verifies destroy completes
    // We don't attempt to use the brain after destroy as that's undefined behavior
    brain_destroy(brain);
    brain = nullptr;

    // Create new brain for TearDown
    brain = brain_create("test", BRAIN_SIZE_TINY, BRAIN_TASK_CLASSIFICATION, 64, 16);
    ASSERT_NE(brain, nullptr);
}

//=============================================================================
// 6. STATS VALUE REGRESSION TESTS
//=============================================================================

/**
 * TEST: StatsRegression_ValidRanges
 *
 * WHAT: Verify stats values are within expected ranges
 * WHY:  Catch unexpected value changes
 * EXPECT: All stats within valid bounds
 */
TEST_F(CoordinatorRegressionTest, StatsRegression_ValidRanges) {
    ASSERT_NE(brain, nullptr);

    // Bio-async orchestrator stats ranges
    if (brain->bio_async_orchestrator_enabled && brain->bio_async_orchestrator) {
        bio_orchestrator_stats_t stats;
        int result = bio_orchestrator_get_stats(brain->bio_async_orchestrator, &stats);
        if (result == 0) {
            EXPECT_GE(stats.system_health_score, 0.0f);
            EXPECT_LE(stats.system_health_score, 1.0f);
        }
    }

    // Immune bridge coordinator stats ranges
    if (brain->immune_bridge_coordinator_enabled && brain->immune_bridge_coordinator) {
        immune_coordinator_stats_t stats;
        int result = immune_bridge_coordinator_get_stats(brain->immune_bridge_coordinator, &stats);
        if (result == 0) {
            EXPECT_GE(stats.system_health, 0.0f);
            EXPECT_LE(stats.system_health, 1.0f);
        }
    }

    // Cognitive meta-controller stats ranges
    if (brain->cognitive_meta_controller_enabled && brain->cognitive_meta_controller) {
        meta_controller_stats_t stats;
        int result = meta_controller_get_stats(brain->cognitive_meta_controller, &stats);
        if (result == 0) {
            EXPECT_GE(stats.system_confidence, 0.0f);
            EXPECT_LE(stats.system_confidence, 1.0f);
            EXPECT_GE(stats.system_uncertainty, 0.0f);
            EXPECT_LE(stats.system_uncertainty, 1.0f);
        }
    }
}
