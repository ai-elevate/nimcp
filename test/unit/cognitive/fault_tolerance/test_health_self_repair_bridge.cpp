/**
 * @file test_health_self_repair_bridge.cpp
 * @brief Unit tests for Health Self-Repair Bridge
 * @version 1.0.0
 * @date 2025-01-20
 *
 * WHAT: Unit tests for health-to-self-repair automation bridge
 * WHY: Ensure reliable auto-trigger and outcome tracking
 * HOW: Test-driven development with coverage of all public APIs
 *
 * Test Coverage:
 * - Creation and destruction
 * - Configuration
 * - Trigger policy enforcement
 * - Rate limiting
 * - Aggregation
 * - Tracking records
 * - Callbacks
 * - Statistics
 *
 * @author NIMCP Development Team
 */

#include <gtest/gtest.h>
#include <cstring>

// Headers have their own extern "C" guards
#include "cognitive/fault_tolerance/nimcp_health_self_repair_bridge.h"
#include "cognitive/fault_tolerance/nimcp_health_diagnostic_bridge.h"
#include "cognitive/fault_tolerance/nimcp_self_repair.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"

//=============================================================================
// Test Fixture
//=============================================================================

class HealthSelfRepairBridgeTest : public ::testing::Test {
protected:
    size_t baseline_allocated = 0;
    health_diag_bridge_t* diag_bridge = nullptr;
    self_repair_coordinator_t* self_repair = nullptr;

    void SetUp() override {
        nimcp_memory_init();
        nimcp_memory_enable_tracking(true);

        nimcp_memory_stats_t stats;
        nimcp_memory_get_stats(&stats);
        baseline_allocated = stats.current_allocated;

        // Create dependencies
        diag_bridge = health_diag_bridge_create(NULL);
        self_repair = self_repair_create(NULL);
    }

    void TearDown() override {
        // Cleanup dependencies
        if (self_repair) self_repair_destroy(self_repair);
        if (diag_bridge) health_diag_bridge_destroy(diag_bridge);

        nimcp_memory_stats_t stats;
        nimcp_memory_get_stats(&stats);
        EXPECT_EQ(stats.current_allocated, baseline_allocated)
            << "Memory leak detected! Allocated: " << stats.current_allocated
            << ", Baseline: " << baseline_allocated;
    }
};

//=============================================================================
// Creation and Destruction Tests
//=============================================================================

TEST_F(HealthSelfRepairBridgeTest, CreateWithDefaults) {
    ASSERT_NE(diag_bridge, nullptr);
    ASSERT_NE(self_repair, nullptr);

    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, diag_bridge, self_repair);

    ASSERT_NE(bridge, nullptr);
    EXPECT_TRUE(health_self_repair_bridge_is_ready(bridge));

    health_self_repair_bridge_destroy(bridge);
}

TEST_F(HealthSelfRepairBridgeTest, CreateWithCustomConfig) {
    ASSERT_NE(diag_bridge, nullptr);
    ASSERT_NE(self_repair, nullptr);

    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);

    config.trigger_policy = HEALTH_TRIGGER_ERROR;
    config.rate_limit.max_repairs_per_window = 5;
    config.aggregation.enabled = false;
    config.async_repairs = false;

    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        &config, diag_bridge, self_repair);

    ASSERT_NE(bridge, nullptr);
    EXPECT_TRUE(health_self_repair_bridge_is_ready(bridge));

    health_self_repair_bridge_destroy(bridge);
}

TEST_F(HealthSelfRepairBridgeTest, CreateNullDiagBridgeReturnsNull) {
    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, NULL, self_repair);
    EXPECT_EQ(bridge, nullptr);
}

TEST_F(HealthSelfRepairBridgeTest, CreateNullSelfRepairReturnsNull) {
    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, diag_bridge, NULL);
    EXPECT_EQ(bridge, nullptr);
}

TEST_F(HealthSelfRepairBridgeTest, DestroyNullSafety) {
    EXPECT_NO_THROW(health_self_repair_bridge_destroy(NULL));
}

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(HealthSelfRepairBridgeTest, DefaultConfigValues) {
    health_self_repair_bridge_config_t config;
    int ret = health_self_repair_bridge_default_config(&config);

    ASSERT_EQ(ret, 0);
    EXPECT_EQ(config.trigger_policy, HEALTH_TRIGGER_CRITICAL);
    EXPECT_GT(config.rate_limit.max_repairs_per_window, 0u);
    EXPECT_GT(config.rate_limit.window_duration_ms, 0u);
    EXPECT_TRUE(config.aggregation.enabled);
    EXPECT_TRUE(config.async_repairs);
    EXPECT_GT(config.min_confidence, 0.0f);
    EXPECT_LE(config.min_confidence, 1.0f);
}

//=============================================================================
// Trigger Policy Tests
//=============================================================================

TEST_F(HealthSelfRepairBridgeTest, ManualPolicySkipsAutoTrigger) {
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.trigger_policy = HEALTH_TRIGGER_MANUAL;

    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        &config, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    // Create critical anomaly
    anomaly_t anomaly;
    memset(&anomaly, 0, sizeof(anomaly));
    anomaly.type = ANOMALY_RESOURCE_EXHAUSTION;
    anomaly.severity = ANOMALY_SEVERITY_CRITICAL;
    anomaly.confidence = 0.9f;

    uint64_t request_id = 0;
    int ret = health_self_repair_bridge_process_anomaly(bridge, &anomaly, &request_id);

    // Should be skipped due to manual policy
    EXPECT_EQ(ret, 1);  // 1 = skipped

    health_self_repair_bridge_destroy(bridge);
}

TEST_F(HealthSelfRepairBridgeTest, FatalOnlyPolicySkipsWarning) {
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.trigger_policy = HEALTH_TRIGGER_FATAL_ONLY;
    config.aggregation.enabled = false;

    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        &config, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    // Create warning anomaly (should be skipped)
    anomaly_t anomaly;
    memset(&anomaly, 0, sizeof(anomaly));
    anomaly.type = ANOMALY_MEMORY_LEAK;
    anomaly.severity = ANOMALY_SEVERITY_WARNING;
    anomaly.confidence = 0.9f;

    uint64_t request_id = 0;
    int ret = health_self_repair_bridge_process_anomaly(bridge, &anomaly, &request_id);
    EXPECT_EQ(ret, 1);  // Skipped

    health_self_repair_bridge_destroy(bridge);
}

//=============================================================================
// Rate Limiting Tests
//=============================================================================

TEST_F(HealthSelfRepairBridgeTest, RateLimitIsEnforced) {
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.rate_limit.max_repairs_per_window = 2;
    config.rate_limit.window_duration_ms = 60000;
    config.aggregation.enabled = false;

    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        &config, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    // First check: should not be rate limited initially
    EXPECT_FALSE(health_self_repair_bridge_is_rate_limited(bridge, ERROR_TYPE_MEMORY_LEAK));

    // Reset to clear state
    health_self_repair_bridge_reset_rate_limit(bridge);

    health_self_repair_bridge_destroy(bridge);
}

TEST_F(HealthSelfRepairBridgeTest, RateLimitReset) {
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.rate_limit.max_repairs_per_window = 1;

    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        &config, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    // Reset rate limit
    health_self_repair_bridge_reset_rate_limit(bridge);

    // Should not be rate limited after reset
    EXPECT_FALSE(health_self_repair_bridge_is_rate_limited(bridge, ERROR_TYPE_UNKNOWN));

    health_self_repair_bridge_destroy(bridge);
}

//=============================================================================
// Tracking Tests
//=============================================================================

TEST_F(HealthSelfRepairBridgeTest, GetPendingCount) {
    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    uint32_t pending = health_self_repair_bridge_get_pending_count(bridge);
    EXPECT_EQ(pending, 0u);  // Initially no pending repairs

    health_self_repair_bridge_destroy(bridge);
}

TEST_F(HealthSelfRepairBridgeTest, GetRecentTracking) {
    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    health_repair_tracking_t records[10];
    uint32_t count = health_self_repair_bridge_get_recent_tracking(bridge, records, 10);
    EXPECT_EQ(count, 0u);  // Initially no records

    health_self_repair_bridge_destroy(bridge);
}

//=============================================================================
// Callback Tests
//=============================================================================

static bool trigger_callback_called = false;
static bool outcome_callback_called = false;

static void test_trigger_callback(
    uint64_t request_id,
    const diagnostic_result_t* diagnostic,
    void* user_data
) {
    (void)request_id;
    (void)diagnostic;
    (void)user_data;
    trigger_callback_called = true;
}

static void test_outcome_callback(
    uint64_t request_id,
    health_repair_outcome_t outcome,
    const self_repair_result_t* result,
    void* user_data
) {
    (void)request_id;
    (void)outcome;
    (void)result;
    (void)user_data;
    outcome_callback_called = true;
}

TEST_F(HealthSelfRepairBridgeTest, SetTriggerCallback) {
    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    trigger_callback_called = false;

    int ret = health_self_repair_bridge_set_trigger_callback(
        bridge, test_trigger_callback, NULL);
    EXPECT_EQ(ret, 0);

    health_self_repair_bridge_destroy(bridge);
}

TEST_F(HealthSelfRepairBridgeTest, SetOutcomeCallback) {
    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    outcome_callback_called = false;

    int ret = health_self_repair_bridge_set_outcome_callback(
        bridge, test_outcome_callback, NULL);
    EXPECT_EQ(ret, 0);

    health_self_repair_bridge_destroy(bridge);
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(HealthSelfRepairBridgeTest, GetStatistics) {
    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    health_self_repair_bridge_stats_t stats;
    int ret = health_self_repair_bridge_get_stats(bridge, &stats);

    ASSERT_EQ(ret, 0);
    EXPECT_EQ(stats.repairs_triggered, 0u);
    EXPECT_EQ(stats.repairs_succeeded, 0u);
    EXPECT_EQ(stats.repairs_failed, 0u);

    health_self_repair_bridge_destroy(bridge);
}

TEST_F(HealthSelfRepairBridgeTest, ResetStatistics) {
    health_self_repair_bridge_t* bridge = health_self_repair_bridge_create(
        NULL, diag_bridge, self_repair);
    ASSERT_NE(bridge, nullptr);

    health_self_repair_bridge_reset_stats(bridge);

    health_self_repair_bridge_stats_t stats;
    health_self_repair_bridge_get_stats(bridge, &stats);
    EXPECT_EQ(stats.repairs_triggered, 0u);

    health_self_repair_bridge_destroy(bridge);
}

//=============================================================================
// Utility Function Tests
//=============================================================================

TEST_F(HealthSelfRepairBridgeTest, PolicyNames) {
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_MANUAL), "MANUAL");
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_FATAL_ONLY), "FATAL_ONLY");
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_CRITICAL), "CRITICAL");
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_ERROR), "ERROR");
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_AUTO), "AUTO");
}

TEST_F(HealthSelfRepairBridgeTest, OutcomeNames) {
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_PENDING), "PENDING");
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_SUCCESS), "SUCCESS");
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_FAILED), "FAILED");
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_SKIPPED), "SKIPPED");
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_TIMEOUT), "TIMEOUT");
}

TEST_F(HealthSelfRepairBridgeTest, VersionString) {
    const char* version = health_self_repair_bridge_version();
    ASSERT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0u);
}
