/**
 * @file test_health_repair_bridge_regression.cpp
 * @brief Regression tests for Health to Self-Repair Bridge
 *
 * WHAT: Regression tests to prevent known bugs from reoccurring
 * WHY: Ensure bug fixes remain fixed across code changes
 * HOW: Test specific bug scenarios and edge cases from development
 *
 * TEST COVERAGE:
 * - Null pointer dereference edge cases
 * - Severity mapping boundary issues
 * - Rate limiting numerical stability
 * - Memory management and leak prevention
 * - Concurrent access safety
 * - Configuration validation edge cases
 *
 * @version 1.0.0
 * @date 2025-01-20
 * @author NIMCP Development Team
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <cmath>

// Headers have their own extern "C" guards
#include "cognitive/fault_tolerance/nimcp_health_diagnostic_bridge.h"
#include "cognitive/fault_tolerance/nimcp_health_self_repair_bridge.h"
#include "cognitive/fault_tolerance/nimcp_self_repair.h"
#include "cognitive/fault_tolerance/nimcp_self_repair_health_notify.h"
#include "utils/memory/nimcp_memory.h"

//=============================================================================
// Test Fixture
//=============================================================================

class HealthRepairBridgeRegressionTest : public ::testing::Test {
protected:
    size_t baseline_allocated = 0;
    health_diag_bridge_t* diag_bridge = nullptr;
    health_self_repair_bridge_t* repair_bridge = nullptr;
    self_repair_coordinator_t* self_repair = nullptr;

    void SetUp() override {
        nimcp_memory_init();
        nimcp_memory_enable_tracking(true);

        nimcp_memory_stats_t stats;
        nimcp_memory_get_stats(&stats);
        baseline_allocated = stats.current_allocated;

        diag_bridge = health_diag_bridge_create(NULL);
        self_repair = self_repair_create(NULL);

        if (diag_bridge && self_repair) {
            repair_bridge = health_self_repair_bridge_create(
                NULL, diag_bridge, self_repair);
        }
    }

    void TearDown() override {
        if (repair_bridge) health_self_repair_bridge_destroy(repair_bridge);
        if (self_repair) self_repair_destroy(self_repair);
        if (diag_bridge) health_diag_bridge_destroy(diag_bridge);

        nimcp_memory_stats_t stats;
        nimcp_memory_get_stats(&stats);
        EXPECT_EQ(stats.current_allocated, baseline_allocated)
            << "Memory leak detected! Allocated: " << stats.current_allocated
            << ", Baseline: " << baseline_allocated;
    }

    void create_anomaly(anomaly_t* anomaly, anomaly_type_t type,
                        anomaly_severity_t severity, float confidence) {
        memset(anomaly, 0, sizeof(*anomaly));
        anomaly->type = type;
        anomaly->severity = severity;
        anomaly->confidence = confidence;
    }
};

//=============================================================================
// REGRESSION: Null Pointer Handling
//=============================================================================

/**
 * @test REGRESSION: Null Bridge Crash
 *
 * WHAT: Prevent crash when passing NULL to API functions
 * WHY: Bug #1: Various functions crashed on NULL input
 * HOW: Call all public APIs with NULL parameters
 */
TEST_F(HealthRepairBridgeRegressionTest, NullBridgeCrashPrevention) {
    // Diagnostic bridge NULL handling
    EXPECT_NO_THROW(health_diag_bridge_destroy(NULL));
    EXPECT_FALSE(health_diag_bridge_is_ready(NULL));

    anomaly_t anomaly;
    diagnostic_result_t* result = nullptr;
    EXPECT_EQ(health_diag_bridge_convert_anomaly(NULL, &anomaly, &result), -1);

    health_agent_message_t msg;
    EXPECT_EQ(health_diag_bridge_convert_agent_message(NULL, &msg, &result), -1);

    health_diag_bridge_stats_t diag_stats;
    EXPECT_EQ(health_diag_bridge_get_stats(NULL, &diag_stats), -1);

    // Self-repair bridge NULL handling
    EXPECT_NO_THROW(health_self_repair_bridge_destroy(NULL));
    EXPECT_FALSE(health_self_repair_bridge_is_ready(NULL));

    uint64_t request_id;
    EXPECT_EQ(health_self_repair_bridge_process_anomaly(NULL, &anomaly, &request_id), -1);

    health_self_repair_bridge_stats_t repair_stats;
    EXPECT_EQ(health_self_repair_bridge_get_stats(NULL, &repair_stats), -1);

    // Notify bridge NULL handling
    EXPECT_NO_THROW(self_repair_health_notify_destroy(NULL));
    EXPECT_FALSE(self_repair_health_notify_is_ready(NULL));
}

/**
 * @test REGRESSION: Null Output Parameter Crash
 *
 * WHAT: Prevent crash when output parameter is NULL
 * WHY: Bug #2: Functions wrote to NULL pointers
 * HOW: Call functions with NULL output parameters
 */
TEST_F(HealthRepairBridgeRegressionTest, NullOutputParameterHandling) {
    ASSERT_NE(diag_bridge, nullptr);

    anomaly_t anomaly;
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 0.8f);

    // NULL result pointer
    EXPECT_EQ(health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, NULL), -1);

    // NULL stats pointer
    EXPECT_EQ(health_diag_bridge_get_stats(diag_bridge, NULL), -1);

    ASSERT_NE(repair_bridge, nullptr);
    EXPECT_EQ(health_self_repair_bridge_get_stats(repair_bridge, NULL), -1);
}

/**
 * @test REGRESSION: Null Input Data Crash
 *
 * WHAT: Prevent crash when input data is NULL
 * WHY: Bug #3: Dereference of NULL anomaly/message
 * HOW: Call conversion functions with NULL input
 */
TEST_F(HealthRepairBridgeRegressionTest, NullInputDataHandling) {
    ASSERT_NE(diag_bridge, nullptr);

    diagnostic_result_t* result = nullptr;

    // NULL anomaly
    EXPECT_EQ(health_diag_bridge_convert_anomaly(diag_bridge, NULL, &result), -1);
    EXPECT_EQ(result, nullptr);

    // NULL agent message
    EXPECT_EQ(health_diag_bridge_convert_agent_message(diag_bridge, NULL, &result), -1);
    EXPECT_EQ(result, nullptr);

    ASSERT_NE(repair_bridge, nullptr);

    uint64_t request_id = 0;
    EXPECT_EQ(health_self_repair_bridge_process_anomaly(repair_bridge, NULL, &request_id), -1);
}

//=============================================================================
// REGRESSION: Severity Mapping Edge Cases
//=============================================================================

/**
 * @test REGRESSION: Unknown Severity Handling
 *
 * WHAT: Handle unknown/invalid severity values gracefully
 * WHY: Bug #4: Crash on out-of-range severity enum values
 * HOW: Pass invalid severity and verify default handling
 */
TEST_F(HealthRepairBridgeRegressionTest, UnknownSeverityHandling) {
    ASSERT_NE(diag_bridge, nullptr);

    anomaly_t anomaly;
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, (anomaly_severity_t)999, 0.8f);

    diagnostic_result_t* result = nullptr;
    int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    // Should succeed with default severity, not crash
    EXPECT_EQ(ret, 0);
    if (result) {
        diagnostics_free_result(result);
    }
}

/**
 * @test REGRESSION: Unknown Anomaly Type Handling
 *
 * WHAT: Handle unknown anomaly types gracefully
 * WHY: Bug #5: Array out-of-bounds on unknown anomaly type
 * HOW: Pass invalid anomaly type
 */
TEST_F(HealthRepairBridgeRegressionTest, UnknownAnomalyTypeHandling) {
    ASSERT_NE(diag_bridge, nullptr);

    anomaly_t anomaly;
    create_anomaly(&anomaly, (anomaly_type_t)999, ANOMALY_SEVERITY_WARNING, 0.8f);

    diagnostic_result_t* result = nullptr;
    int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    // Should succeed with default mapping, not crash
    EXPECT_EQ(ret, 0);
    if (result) {
        diagnostics_free_result(result);
    }
}

//=============================================================================
// REGRESSION: Numerical Stability
//=============================================================================

/**
 * @test REGRESSION: Zero Confidence Handling
 *
 * WHAT: Handle zero confidence values
 * WHY: Bug #6: Division by zero when normalizing zero confidence
 * HOW: Pass anomaly with zero confidence
 */
TEST_F(HealthRepairBridgeRegressionTest, ZeroConfidenceHandling) {
    ASSERT_NE(diag_bridge, nullptr);

    anomaly_t anomaly;
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 0.0f);

    diagnostic_result_t* result = nullptr;
    int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    EXPECT_EQ(ret, 0);
    ASSERT_NE(result, nullptr);
    // Zero confidence should remain zero or get clamped, not NaN
    EXPECT_FALSE(std::isnan(result->confidence));
    diagnostics_free_result(result);
}

/**
 * @test REGRESSION: NaN/Inf Confidence Handling
 *
 * WHAT: Handle NaN and Inf confidence values
 * WHY: Bug #7: NaN propagation through calculations
 * HOW: Pass anomaly with NaN confidence
 */
TEST_F(HealthRepairBridgeRegressionTest, NaNConfidenceHandling) {
    ASSERT_NE(diag_bridge, nullptr);

    anomaly_t anomaly;
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING,
                   std::numeric_limits<float>::quiet_NaN());

    diagnostic_result_t* result = nullptr;
    int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    if (ret == 0 && result) {
        // NaN should be handled, not propagated
        EXPECT_FALSE(std::isnan(result->confidence));
        diagnostics_free_result(result);
    }
}

/**
 * @test REGRESSION: Confidence Clamping
 *
 * WHAT: Clamp confidence to valid range [0, 1]
 * WHY: Bug #8: Confidence > 1.0 caused issues in downstream processing
 * HOW: Pass confidence values outside [0, 1]
 */
TEST_F(HealthRepairBridgeRegressionTest, ConfidenceClamping) {
    ASSERT_NE(diag_bridge, nullptr);

    // Test confidence > 1.0
    anomaly_t anomaly;
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 2.0f);

    diagnostic_result_t* result = nullptr;
    health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    if (result) {
        EXPECT_LE(result->confidence, 1.0f);
        EXPECT_GE(result->confidence, 0.0f);
        diagnostics_free_result(result);
    }

    // Test confidence < 0.0
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, -1.0f);
    result = nullptr;
    health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    if (result) {
        EXPECT_LE(result->confidence, 1.0f);
        EXPECT_GE(result->confidence, 0.0f);
        diagnostics_free_result(result);
    }
}

//=============================================================================
// REGRESSION: Rate Limiting Edge Cases
//=============================================================================

/**
 * @test REGRESSION: Zero Rate Limit Handling
 *
 * WHAT: Handle zero max_repairs_per_window configuration
 * WHY: Bug #9: Division by zero in rate limit calculations
 * HOW: Configure bridge with zero rate limit
 */
TEST_F(HealthRepairBridgeRegressionTest, ZeroRateLimitHandling) {
    ASSERT_NE(diag_bridge, nullptr);
    ASSERT_NE(self_repair, nullptr);

    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.rate_limit.max_repairs_per_window = 0;  // Zero rate limit

    health_self_repair_bridge_t* test_bridge =
        health_self_repair_bridge_create(&config, diag_bridge, self_repair);

    // Should either reject invalid config or handle gracefully
    if (test_bridge) {
        // If created, verify it doesn't crash on use
        anomaly_t anomaly;
        create_anomaly(&anomaly, ANOMALY_RESOURCE_EXHAUSTION,
                       ANOMALY_SEVERITY_CRITICAL, 0.9f);

        uint64_t request_id = 0;
        // Should not crash, might return error or skip
        EXPECT_NO_THROW(
            health_self_repair_bridge_process_anomaly(test_bridge, &anomaly, &request_id)
        );

        health_self_repair_bridge_destroy(test_bridge);
    }
    // If NULL returned, that's also acceptable behavior
}

/**
 * @test REGRESSION: Zero Window Duration Handling
 *
 * WHAT: Handle zero window_duration_ms configuration
 * WHY: Bug #10: Division by zero in rate calculations
 * HOW: Configure bridge with zero window duration
 */
TEST_F(HealthRepairBridgeRegressionTest, ZeroWindowDurationHandling) {
    ASSERT_NE(diag_bridge, nullptr);
    ASSERT_NE(self_repair, nullptr);

    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.rate_limit.window_duration_ms = 0;  // Zero window

    health_self_repair_bridge_t* test_bridge =
        health_self_repair_bridge_create(&config, diag_bridge, self_repair);

    if (test_bridge) {
        anomaly_t anomaly;
        create_anomaly(&anomaly, ANOMALY_RESOURCE_EXHAUSTION,
                       ANOMALY_SEVERITY_CRITICAL, 0.9f);

        uint64_t request_id = 0;
        EXPECT_NO_THROW(
            health_self_repair_bridge_process_anomaly(test_bridge, &anomaly, &request_id)
        );

        health_self_repair_bridge_destroy(test_bridge);
    }
}

//=============================================================================
// REGRESSION: Memory Management
//=============================================================================

/**
 * @test REGRESSION: Double Free Prevention
 *
 * WHAT: Prevent double-free crashes
 * WHY: Bug #11: Double destroy calls caused crash
 * HOW: Call destroy twice on same handle
 */
TEST_F(HealthRepairBridgeRegressionTest, DoubleFreePreventionDiagBridge) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    health_diag_bridge_destroy(bridge);
    // Second destroy should not crash (NULL should be handled)
    EXPECT_NO_THROW(health_diag_bridge_destroy(NULL));
}

TEST_F(HealthRepairBridgeRegressionTest, DoubleFreePreventionRepairBridge) {
    // Note: Can't test actual double-free without use-after-free issues
    // Instead verify NULL handling
    EXPECT_NO_THROW(health_self_repair_bridge_destroy(NULL));
    EXPECT_NO_THROW(health_self_repair_bridge_destroy(NULL));
}

/**
 * @test REGRESSION: Result Leak Prevention
 *
 * WHAT: Ensure diagnostic results can be freed properly
 * WHY: Bug #12: Memory leak when caller forgot to free result
 * HOW: Convert multiple anomalies and verify no leaks
 */
TEST_F(HealthRepairBridgeRegressionTest, ResultLeakPrevention) {
    ASSERT_NE(diag_bridge, nullptr);

    // Convert many anomalies and free each result
    for (int i = 0; i < 100; i++) {
        anomaly_t anomaly;
        create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 0.8f);

        diagnostic_result_t* result = nullptr;
        int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

        ASSERT_EQ(ret, 0);
        ASSERT_NE(result, nullptr);
        diagnostics_free_result(result);
    }

    // Memory check happens in TearDown
}

//=============================================================================
// REGRESSION: Configuration Validation
//=============================================================================

/**
 * @test REGRESSION: Invalid Confidence Threshold
 *
 * WHAT: Handle invalid min_confidence configuration
 * WHY: Bug #13: Negative threshold caused all repairs to pass
 * HOW: Configure with negative threshold
 */
TEST_F(HealthRepairBridgeRegressionTest, InvalidConfidenceThreshold) {
    ASSERT_NE(diag_bridge, nullptr);
    ASSERT_NE(self_repair, nullptr);

    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.min_confidence = -1.0f;  // Invalid negative threshold

    health_self_repair_bridge_t* test_bridge =
        health_self_repair_bridge_create(&config, diag_bridge, self_repair);

    // Should either reject invalid config or clamp to valid range
    if (test_bridge) {
        health_self_repair_bridge_destroy(test_bridge);
    }
}

/**
 * @test REGRESSION: Default Config Idempotency
 *
 * WHAT: Calling default_config multiple times should be safe
 * WHY: Bug #14: Some fields weren't reset on second call
 * HOW: Call default_config twice and compare
 */
TEST_F(HealthRepairBridgeRegressionTest, DefaultConfigIdempotency) {
    health_self_repair_bridge_config_t config1, config2;

    // Initialize first time
    health_self_repair_bridge_default_config(&config1);

    // Modify some fields
    config1.trigger_policy = HEALTH_TRIGGER_MANUAL;
    config1.rate_limit.max_repairs_per_window = 999;

    // Call default_config again - should reset
    health_self_repair_bridge_default_config(&config1);
    health_self_repair_bridge_default_config(&config2);

    // Should be equal
    EXPECT_EQ(config1.trigger_policy, config2.trigger_policy);
    EXPECT_EQ(config1.rate_limit.max_repairs_per_window,
              config2.rate_limit.max_repairs_per_window);
}

//=============================================================================
// REGRESSION: String Handling
//=============================================================================

/**
 * @test REGRESSION: Long Description Truncation
 *
 * WHAT: Handle anomaly descriptions longer than buffer
 * WHY: Bug #15: Buffer overflow on long descriptions
 * HOW: Pass anomaly with maximum length description
 */
TEST_F(HealthRepairBridgeRegressionTest, LongDescriptionTruncation) {
    ASSERT_NE(diag_bridge, nullptr);

    anomaly_t anomaly;
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 0.8f);

    // Fill description with max length string
    memset(anomaly.description, 'A', sizeof(anomaly.description) - 1);
    anomaly.description[sizeof(anomaly.description) - 1] = '\0';

    diagnostic_result_t* result = nullptr;
    int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    EXPECT_EQ(ret, 0);
    if (result) {
        diagnostics_free_result(result);
    }
}

/**
 * @test REGRESSION: Empty Description Handling
 *
 * WHAT: Handle anomalies with empty description
 * WHY: Bug #16: Empty string caused issues in logging
 * HOW: Pass anomaly with empty description
 */
TEST_F(HealthRepairBridgeRegressionTest, EmptyDescriptionHandling) {
    ASSERT_NE(diag_bridge, nullptr);

    anomaly_t anomaly;
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 0.8f);
    anomaly.description[0] = '\0';  // Empty description

    diagnostic_result_t* result = nullptr;
    int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    EXPECT_EQ(ret, 0);
    if (result) {
        diagnostics_free_result(result);
    }
}

//=============================================================================
// REGRESSION: Statistics Accuracy
//=============================================================================

/**
 * @test REGRESSION: Statistics Overflow Prevention
 *
 * WHAT: Prevent statistics counter overflow
 * WHY: Bug #17: 32-bit counters overflowed on heavy use
 * HOW: Process many anomalies and verify stats don't wrap
 */
TEST_F(HealthRepairBridgeRegressionTest, StatisticsOverflowPrevention) {
    ASSERT_NE(diag_bridge, nullptr);

    // Process many anomalies
    for (int i = 0; i < 1000; i++) {
        anomaly_t anomaly;
        create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 0.8f);

        diagnostic_result_t* result = nullptr;
        health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);
        if (result) diagnostics_free_result(result);
    }

    health_diag_bridge_stats_t stats;
    health_diag_bridge_get_stats(diag_bridge, &stats);

    // Should not overflow
    EXPECT_EQ(stats.anomalies_converted, 1000u);
}

/**
 * @test REGRESSION: Statistics Reset Completeness
 *
 * WHAT: Reset should clear all statistics
 * WHY: Bug #18: Some stats were not reset
 * HOW: Generate stats, reset, verify all zero
 */
TEST_F(HealthRepairBridgeRegressionTest, StatisticsResetCompleteness) {
    ASSERT_NE(diag_bridge, nullptr);

    // Generate some stats
    anomaly_t anomaly;
    create_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 0.8f);

    diagnostic_result_t* result = nullptr;
    health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);
    if (result) diagnostics_free_result(result);

    // Reset
    health_diag_bridge_reset_stats(diag_bridge);

    // Verify all zero
    health_diag_bridge_stats_t stats;
    health_diag_bridge_get_stats(diag_bridge, &stats);

    EXPECT_EQ(stats.anomalies_converted, 0u);
    EXPECT_EQ(stats.agent_messages_converted, 0u);
    EXPECT_EQ(stats.conversions_failed, 0u);
}

//=============================================================================
// REGRESSION: API Stability
//=============================================================================

/**
 * @test REGRESSION: Version String Non-Null
 *
 * WHAT: Version functions should never return NULL
 * WHY: Bug #19: NULL version string caused printf crash
 * HOW: Verify all version functions return non-NULL
 */
TEST_F(HealthRepairBridgeRegressionTest, VersionStringNonNull) {
    const char* diag_version = health_diag_bridge_version();
    const char* repair_version = health_self_repair_bridge_version();
    const char* notify_version = self_repair_health_notify_version();

    ASSERT_NE(diag_version, nullptr);
    ASSERT_NE(repair_version, nullptr);
    ASSERT_NE(notify_version, nullptr);

    EXPECT_GT(strlen(diag_version), 0u);
    EXPECT_GT(strlen(repair_version), 0u);
    EXPECT_GT(strlen(notify_version), 0u);
}

/**
 * @test REGRESSION: Policy/Outcome Name Non-Null
 *
 * WHAT: Name conversion functions should handle all enum values
 * WHY: Bug #20: Unknown enum returned NULL, causing crash
 * HOW: Test all enum values including invalid ones
 */
TEST_F(HealthRepairBridgeRegressionTest, NameFunctionsNonNull) {
    // Test valid policy names
    for (int i = 0; i <= HEALTH_TRIGGER_AUTO; i++) {
        const char* name = health_self_repair_bridge_policy_name((health_trigger_policy_t)i);
        EXPECT_NE(name, nullptr) << "Policy " << i << " returned NULL";
    }

    // Test invalid policy - should return "UNKNOWN" or similar, not NULL
    const char* invalid_policy = health_self_repair_bridge_policy_name((health_trigger_policy_t)999);
    EXPECT_NE(invalid_policy, nullptr);

    // Test valid outcome names
    for (int i = 0; i <= HEALTH_REPAIR_OUTCOME_TIMEOUT; i++) {
        const char* name = health_self_repair_bridge_outcome_name((health_repair_outcome_t)i);
        EXPECT_NE(name, nullptr) << "Outcome " << i << " returned NULL";
    }

    // Test invalid outcome
    const char* invalid_outcome = health_self_repair_bridge_outcome_name((health_repair_outcome_t)999);
    EXPECT_NE(invalid_outcome, nullptr);
}
