/**
 * @file test_health_self_repair_integration.cpp
 * @brief Integration tests for Health to Self-Repair Pipeline
 * @version 1.0.0
 * @date 2025-01-20
 *
 * WHAT: Integration tests for complete health monitoring to self-repair flow
 * WHY: Verify end-to-end operation of health detection to repair pipeline
 * HOW: Test full pipeline with diagnostic bridge, self-repair bridge, and coordinator
 *
 * TEST SCENARIOS:
 * - Health anomaly triggers diagnostic conversion
 * - Diagnostic triggers self-repair coordinator
 * - Rate limiting across the pipeline
 * - Outcome feedback to health agent
 * - Aggregation of similar anomalies
 * - Full pipeline stress test
 *
 * @author NIMCP Development Team
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <chrono>

// Headers have their own extern "C" guards
#include "cognitive/fault_tolerance/nimcp_health_diagnostic_bridge.h"
#include "cognitive/fault_tolerance/nimcp_health_self_repair_bridge.h"
#include "cognitive/fault_tolerance/nimcp_self_repair.h"
#include "cognitive/fault_tolerance/nimcp_self_repair_health_notify.h"
#include "utils/fault_tolerance/nimcp_health_agent.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"

//=============================================================================
// Test Fixture
//=============================================================================

class HealthSelfRepairIntegrationTest : public ::testing::Test {
protected:
    size_t baseline_allocated = 0;

    // Pipeline components
    health_diag_bridge_t* diag_bridge = nullptr;
    health_self_repair_bridge_t* repair_bridge = nullptr;
    self_repair_coordinator_t* self_repair = nullptr;
    self_repair_health_notify_bridge_t* notify_bridge = nullptr;
    nimcp_health_agent_t* health_agent = nullptr;

    void SetUp() override {
        nimcp_memory_init();
        nimcp_memory_enable_tracking(true);

        nimcp_memory_stats_t stats;
        nimcp_memory_get_stats(&stats);
        baseline_allocated = stats.current_allocated;

        // Create full pipeline
        health_agent = nimcp_health_agent_create(NULL);
        diag_bridge = health_diag_bridge_create(NULL);
        self_repair = self_repair_create(NULL);

        if (diag_bridge && self_repair) {
            repair_bridge = health_self_repair_bridge_create(
                NULL, diag_bridge, self_repair);
        }

        if (self_repair && health_agent) {
            notify_bridge = self_repair_health_notify_create(
                NULL, self_repair, health_agent);
        }
    }

    void TearDown() override {
        // Cleanup in reverse order of creation
        if (notify_bridge) self_repair_health_notify_destroy(notify_bridge);
        if (repair_bridge) health_self_repair_bridge_destroy(repair_bridge);
        if (self_repair) self_repair_destroy(self_repair);
        if (diag_bridge) health_diag_bridge_destroy(diag_bridge);
        if (health_agent) nimcp_health_agent_destroy(health_agent);

        nimcp_memory_stats_t stats;
        nimcp_memory_get_stats(&stats);
        EXPECT_EQ(stats.current_allocated, baseline_allocated)
            << "Memory leak detected! Allocated: " << stats.current_allocated
            << ", Baseline: " << baseline_allocated;
    }

    // Helper: Create test anomaly
    void create_test_anomaly(anomaly_t* anomaly, anomaly_type_t type,
                              anomaly_severity_t severity, float confidence) {
        memset(anomaly, 0, sizeof(*anomaly));
        anomaly->type = type;
        anomaly->severity = severity;
        anomaly->confidence = confidence;
        strncpy(anomaly->affected_component, "test_component",
                sizeof(anomaly->affected_component) - 1);
        strncpy(anomaly->description, "Test anomaly for integration",
                sizeof(anomaly->description) - 1);
    }

    // Helper: Create test agent message
    void create_test_agent_message(health_agent_message_t* message,
                                    health_agent_msg_type_t type,
                                    health_agent_severity_t severity) {
        memset(message, 0, sizeof(*message));
        message->type = type;
        message->severity = severity;
        strncpy(message->description, "Test health agent message",
                sizeof(message->description) - 1);
    }
};

//=============================================================================
// Pipeline Setup Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, FullPipelineCreation) {
    ASSERT_NE(health_agent, nullptr) << "Health agent creation failed";
    ASSERT_NE(diag_bridge, nullptr) << "Diagnostic bridge creation failed";
    ASSERT_NE(self_repair, nullptr) << "Self-repair coordinator creation failed";
    ASSERT_NE(repair_bridge, nullptr) << "Self-repair bridge creation failed";
    ASSERT_NE(notify_bridge, nullptr) << "Notify bridge creation failed";

    EXPECT_TRUE(health_diag_bridge_is_ready(diag_bridge));
    EXPECT_TRUE(health_self_repair_bridge_is_ready(repair_bridge));
    EXPECT_TRUE(self_repair_health_notify_is_ready(notify_bridge));
}

//=============================================================================
// Anomaly to Diagnostic Conversion Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, AnomalyToDiagnosticFlow) {
    ASSERT_NE(diag_bridge, nullptr);

    anomaly_t anomaly;
    create_test_anomaly(&anomaly, ANOMALY_MEMORY_LEAK,
                        ANOMALY_SEVERITY_WARNING, 0.85f);

    diagnostic_result_t* result = nullptr;
    int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

    ASSERT_EQ(ret, 0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->error_type, ERROR_TYPE_MEMORY_LEAK);
    EXPECT_EQ(result->severity, DIAG_SEVERITY_WARNING);

    // Check statistics
    health_diag_bridge_stats_t stats;
    health_diag_bridge_get_stats(diag_bridge, &stats);
    EXPECT_EQ(stats.anomalies_converted, 1u);

    diagnostics_free_result(result);
}

TEST_F(HealthSelfRepairIntegrationTest, AgentMessageToDiagnosticFlow) {
    ASSERT_NE(diag_bridge, nullptr);

    health_agent_message_t message;
    create_test_agent_message(&message, HEALTH_MSG_DEADLOCK_DETECTED,
                               HEALTH_SEVERITY_CRITICAL);

    diagnostic_result_t* result = nullptr;
    int ret = health_diag_bridge_convert_agent_message(diag_bridge, &message, &result);

    ASSERT_EQ(ret, 0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->error_type, ERROR_TYPE_DEADLOCK);
    EXPECT_EQ(result->severity, DIAG_SEVERITY_CRITICAL);

    // Check statistics
    health_diag_bridge_stats_t stats;
    health_diag_bridge_get_stats(diag_bridge, &stats);
    EXPECT_EQ(stats.agent_messages_converted, 1u);

    diagnostics_free_result(result);
}

//=============================================================================
// Full Pipeline Flow Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, CriticalAnomalyTriggersRepair) {
    ASSERT_NE(repair_bridge, nullptr);

    // Configure to trigger on CRITICAL severity
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.trigger_policy = HEALTH_TRIGGER_CRITICAL;
    config.aggregation.enabled = false;

    // Recreate bridge with new config
    health_self_repair_bridge_destroy(repair_bridge);
    repair_bridge = health_self_repair_bridge_create(&config, diag_bridge, self_repair);
    ASSERT_NE(repair_bridge, nullptr);

    // Create critical anomaly
    anomaly_t anomaly;
    create_test_anomaly(&anomaly, ANOMALY_RESOURCE_EXHAUSTION,
                        ANOMALY_SEVERITY_CRITICAL, 0.9f);

    uint64_t request_id = 0;
    int ret = health_self_repair_bridge_process_anomaly(repair_bridge, &anomaly, &request_id);

    // Should trigger repair (0 = triggered, 1 = skipped, -1 = error)
    EXPECT_EQ(ret, 0);
    EXPECT_GT(request_id, 0u);

    // Check bridge statistics
    health_self_repair_bridge_stats_t stats;
    health_self_repair_bridge_get_stats(repair_bridge, &stats);
    EXPECT_EQ(stats.repairs_triggered, 1u);
}

TEST_F(HealthSelfRepairIntegrationTest, WarningAnomalySkippedWithCriticalPolicy) {
    ASSERT_NE(repair_bridge, nullptr);

    // Configure to only trigger on CRITICAL and above
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.trigger_policy = HEALTH_TRIGGER_CRITICAL;
    config.aggregation.enabled = false;

    // Recreate bridge with new config
    health_self_repair_bridge_destroy(repair_bridge);
    repair_bridge = health_self_repair_bridge_create(&config, diag_bridge, self_repair);
    ASSERT_NE(repair_bridge, nullptr);

    // Create warning anomaly (should be skipped)
    anomaly_t anomaly;
    create_test_anomaly(&anomaly, ANOMALY_MEMORY_LEAK,
                        ANOMALY_SEVERITY_WARNING, 0.9f);

    uint64_t request_id = 0;
    int ret = health_self_repair_bridge_process_anomaly(repair_bridge, &anomaly, &request_id);

    // Should be skipped
    EXPECT_EQ(ret, 1);  // 1 = skipped

    // Check statistics
    health_self_repair_bridge_stats_t stats;
    health_self_repair_bridge_get_stats(repair_bridge, &stats);
    EXPECT_EQ(stats.repairs_triggered, 0u);
    EXPECT_EQ(stats.repairs_skipped, 1u);
}

TEST_F(HealthSelfRepairIntegrationTest, ManualPolicySkipsAllAutoTriggers) {
    ASSERT_NE(repair_bridge, nullptr);

    // Configure manual policy (no auto-triggers)
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.trigger_policy = HEALTH_TRIGGER_MANUAL;

    // Recreate bridge with new config
    health_self_repair_bridge_destroy(repair_bridge);
    repair_bridge = health_self_repair_bridge_create(&config, diag_bridge, self_repair);
    ASSERT_NE(repair_bridge, nullptr);

    // Create FATAL anomaly (even FATAL should be skipped with manual policy)
    anomaly_t anomaly;
    create_test_anomaly(&anomaly, ANOMALY_RESOURCE_EXHAUSTION,
                        ANOMALY_SEVERITY_CRITICAL, 1.0f);

    uint64_t request_id = 0;
    int ret = health_self_repair_bridge_process_anomaly(repair_bridge, &anomaly, &request_id);

    EXPECT_EQ(ret, 1);  // Skipped due to manual policy
}

//=============================================================================
// Rate Limiting Integration Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, RateLimitEnforced) {
    ASSERT_NE(repair_bridge, nullptr);

    // Configure strict rate limit: only 2 repairs per window
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.trigger_policy = HEALTH_TRIGGER_CRITICAL;
    config.rate_limit.max_repairs_per_window = 2;
    config.rate_limit.window_duration_ms = 60000;
    config.aggregation.enabled = false;

    // Recreate bridge with new config
    health_self_repair_bridge_destroy(repair_bridge);
    repair_bridge = health_self_repair_bridge_create(&config, diag_bridge, self_repair);
    ASSERT_NE(repair_bridge, nullptr);

    // Process multiple critical anomalies
    uint64_t request_id = 0;
    int triggered_count = 0;
    int skipped_count = 0;

    for (int i = 0; i < 5; i++) {
        anomaly_t anomaly;
        create_test_anomaly(&anomaly, ANOMALY_RESOURCE_EXHAUSTION,
                            ANOMALY_SEVERITY_CRITICAL, 0.95f);

        int ret = health_self_repair_bridge_process_anomaly(
            repair_bridge, &anomaly, &request_id);

        if (ret == 0) triggered_count++;
        else if (ret == 1) skipped_count++;
    }

    // Should have triggered 2 and skipped 3 due to rate limit
    EXPECT_LE(triggered_count, 2);
    EXPECT_GE(skipped_count, 3);
}

//=============================================================================
// Health Agent Notification Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, NotifyBridgeReadiness) {
    ASSERT_NE(notify_bridge, nullptr);
    EXPECT_TRUE(self_repair_health_notify_is_ready(notify_bridge));
}

TEST_F(HealthSelfRepairIntegrationTest, NotifyRepairFailure) {
    ASSERT_NE(notify_bridge, nullptr);

    // Create a failure notification
    self_repair_health_notification_t notification;
    memset(&notification, 0, sizeof(notification));
    notification.type = REPAIR_NOTIFY_FAILURE;
    notification.repair_id = 12345;
    notification.intervention = REPAIR_INTERVENTION_ALERT;
    strncpy(notification.error_message, "Fix failed validation",
            sizeof(notification.error_message) - 1);

    int ret = self_repair_health_notify_send(notify_bridge, &notification);
    EXPECT_EQ(ret, 0);

    // Check statistics
    self_repair_health_notify_stats_t stats;
    self_repair_health_notify_get_stats(notify_bridge, &stats);
    EXPECT_EQ(stats.notifications_sent, 1u);
    EXPECT_EQ(stats.failures_notified, 1u);
}

TEST_F(HealthSelfRepairIntegrationTest, SuggestInterventionForRepeatFailures) {
    ASSERT_NE(notify_bridge, nullptr);

    self_repair_result_t result;
    memset(&result, 0, sizeof(result));
    result.status = REPAIR_STATUS_ERROR;
    result.success = false;

    // Suggest intervention
    repair_intervention_t intervention = self_repair_suggest_intervention(
        notify_bridge, &result);

    // Should suggest some intervention for failures
    EXPECT_NE(intervention, REPAIR_INTERVENTION_NONE);
}

//=============================================================================
// Multiple Anomaly Types Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, ProcessMultipleAnomalyTypes) {
    ASSERT_NE(diag_bridge, nullptr);

    // Test various anomaly types
    struct {
        anomaly_type_t type;
        error_type_t expected_error;
    } test_cases[] = {
        { ANOMALY_MEMORY_LEAK, ERROR_TYPE_MEMORY_LEAK },
        { ANOMALY_RESOURCE_EXHAUSTION, ERROR_TYPE_OUT_OF_MEMORY },
        { ANOMALY_PERFORMANCE_DEGRADATION, ERROR_TYPE_INVALID_STATE },
        { ANOMALY_THREAD_CONTENTION, ERROR_TYPE_DEADLOCK },
    };

    for (const auto& tc : test_cases) {
        anomaly_t anomaly;
        create_test_anomaly(&anomaly, tc.type, ANOMALY_SEVERITY_ERROR, 0.8f);

        diagnostic_result_t* result = nullptr;
        int ret = health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);

        ASSERT_EQ(ret, 0);
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->error_type, tc.expected_error)
            << "Anomaly type " << health_diag_bridge_anomaly_type_name(tc.type)
            << " did not map to expected error type";

        diagnostics_free_result(result);
    }
}

//=============================================================================
// Severity Translation Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, SeverityTranslationConsistency) {
    // Verify severity mappings are consistent
    EXPECT_EQ(
        health_diag_bridge_translate_anomaly_severity(ANOMALY_SEVERITY_INFO),
        DIAG_SEVERITY_INFO);
    EXPECT_EQ(
        health_diag_bridge_translate_anomaly_severity(ANOMALY_SEVERITY_WARNING),
        DIAG_SEVERITY_WARNING);
    EXPECT_EQ(
        health_diag_bridge_translate_anomaly_severity(ANOMALY_SEVERITY_ERROR),
        DIAG_SEVERITY_ERROR);
    EXPECT_EQ(
        health_diag_bridge_translate_anomaly_severity(ANOMALY_SEVERITY_CRITICAL),
        DIAG_SEVERITY_CRITICAL);
    // Note: CRITICAL maps to CRITICAL, not FATAL - there is no ANOMALY_SEVERITY_FATAL
}

//=============================================================================
// Statistics Integration Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, StatisticsTrackFullPipeline) {
    ASSERT_NE(repair_bridge, nullptr);

    // Configure to trigger on all errors
    health_self_repair_bridge_config_t config;
    health_self_repair_bridge_default_config(&config);
    config.trigger_policy = HEALTH_TRIGGER_ERROR;
    config.aggregation.enabled = false;

    // Recreate bridge with new config
    health_self_repair_bridge_destroy(repair_bridge);
    repair_bridge = health_self_repair_bridge_create(&config, diag_bridge, self_repair);
    ASSERT_NE(repair_bridge, nullptr);

    // Process several anomalies
    for (int i = 0; i < 3; i++) {
        anomaly_t anomaly;
        create_test_anomaly(&anomaly, ANOMALY_MEMORY_LEAK,
                            ANOMALY_SEVERITY_ERROR, 0.8f);

        uint64_t request_id = 0;
        health_self_repair_bridge_process_anomaly(repair_bridge, &anomaly, &request_id);
    }

    // Check diagnostic bridge stats
    health_diag_bridge_stats_t diag_stats;
    health_diag_bridge_get_stats(diag_bridge, &diag_stats);
    EXPECT_EQ(diag_stats.anomalies_converted, 3u);

    // Check repair bridge stats
    health_self_repair_bridge_stats_t repair_stats;
    health_self_repair_bridge_get_stats(repair_bridge, &repair_stats);
    EXPECT_GE(repair_stats.repairs_triggered + repair_stats.repairs_skipped, 3u);
}

//=============================================================================
// Reset Functionality Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, ResetStatisticsAcrossPipeline) {
    ASSERT_NE(diag_bridge, nullptr);
    ASSERT_NE(repair_bridge, nullptr);
    ASSERT_NE(notify_bridge, nullptr);

    // Generate some activity
    anomaly_t anomaly;
    create_test_anomaly(&anomaly, ANOMALY_MEMORY_LEAK, ANOMALY_SEVERITY_WARNING, 0.8f);

    diagnostic_result_t* result = nullptr;
    health_diag_bridge_convert_anomaly(diag_bridge, &anomaly, &result);
    if (result) diagnostics_free_result(result);

    // Reset all stats
    health_diag_bridge_reset_stats(diag_bridge);
    health_self_repair_bridge_reset_stats(repair_bridge);
    self_repair_health_notify_reset_stats(notify_bridge);

    // Verify all reset
    health_diag_bridge_stats_t diag_stats;
    health_diag_bridge_get_stats(diag_bridge, &diag_stats);
    EXPECT_EQ(diag_stats.anomalies_converted, 0u);

    health_self_repair_bridge_stats_t repair_stats;
    health_self_repair_bridge_get_stats(repair_bridge, &repair_stats);
    EXPECT_EQ(repair_stats.repairs_triggered, 0u);

    self_repair_health_notify_stats_t notify_stats;
    self_repair_health_notify_get_stats(notify_bridge, &notify_stats);
    EXPECT_EQ(notify_stats.notifications_sent, 0u);
}

//=============================================================================
// Utility Function Tests
//=============================================================================

TEST_F(HealthSelfRepairIntegrationTest, PolicyNameStrings) {
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_MANUAL), "MANUAL");
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_FATAL_ONLY), "FATAL_ONLY");
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_CRITICAL), "CRITICAL");
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_ERROR), "ERROR");
    EXPECT_STREQ(health_self_repair_bridge_policy_name(HEALTH_TRIGGER_AUTO), "AUTO");
}

TEST_F(HealthSelfRepairIntegrationTest, OutcomeNameStrings) {
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_PENDING), "PENDING");
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_SUCCESS), "SUCCESS");
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_FAILED), "FAILED");
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_SKIPPED), "SKIPPED");
    EXPECT_STREQ(health_self_repair_bridge_outcome_name(HEALTH_REPAIR_OUTCOME_TIMEOUT), "TIMEOUT");
}

TEST_F(HealthSelfRepairIntegrationTest, NotifyTypeStrings) {
    EXPECT_STREQ(self_repair_notify_type_name(REPAIR_NOTIFY_FAILURE), "FAILURE");
    EXPECT_STREQ(self_repair_notify_type_name(REPAIR_NOTIFY_ROLLBACK), "ROLLBACK");
    EXPECT_STREQ(self_repair_notify_type_name(REPAIR_NOTIFY_HIGH_RISK), "HIGH_RISK");
    EXPECT_STREQ(self_repair_notify_type_name(REPAIR_NOTIFY_REPEATED_FAILURE), "REPEATED_FAILURE");
}

TEST_F(HealthSelfRepairIntegrationTest, InterventionNameStrings) {
    EXPECT_STREQ(self_repair_intervention_name(REPAIR_INTERVENTION_NONE), "NONE");
    EXPECT_STREQ(self_repair_intervention_name(REPAIR_INTERVENTION_ALERT), "ALERT");
    EXPECT_STREQ(self_repair_intervention_name(REPAIR_INTERVENTION_QUARANTINE), "QUARANTINE");
    EXPECT_STREQ(self_repair_intervention_name(REPAIR_INTERVENTION_MANUAL_REPAIR), "MANUAL_REPAIR");
}

TEST_F(HealthSelfRepairIntegrationTest, VersionStrings) {
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
