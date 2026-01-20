/**
 * @file test_health_diagnostic_bridge.cpp
 * @brief Unit tests for Health Diagnostic Bridge
 * @version 1.0.0
 * @date 2025-01-20
 *
 * WHAT: Unit tests for health monitoring to diagnostic format conversion
 * WHY: Ensure reliable conversion of health events for self-repair pipeline
 * HOW: Test-driven development with coverage of all public APIs
 *
 * Test Coverage:
 * - Creation and destruction
 * - Configuration
 * - Anomaly to diagnostic conversion
 * - Agent message to diagnostic conversion
 * - Severity translation
 * - Custom mapping support
 * - Statistics tracking
 * - Error handling
 *
 * @author NIMCP Development Team
 */

#include <gtest/gtest.h>
#include <cstring>

// Headers have their own extern "C" guards
#include "cognitive/fault_tolerance/nimcp_health_diagnostic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"

//=============================================================================
// Test Fixture
//=============================================================================

class HealthDiagnosticBridgeTest : public ::testing::Test {
protected:
    size_t baseline_allocated = 0;

    void SetUp() override {
        nimcp_memory_init();
        nimcp_memory_enable_tracking(true);

        nimcp_memory_stats_t stats;
        nimcp_memory_get_stats(&stats);
        baseline_allocated = stats.current_allocated;
    }

    void TearDown() override {
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

TEST_F(HealthDiagnosticBridgeTest, CreateWithDefaults) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);

    ASSERT_NE(bridge, nullptr);
    EXPECT_TRUE(health_diag_bridge_is_ready(bridge));

    health_diag_bridge_destroy(bridge);
}

TEST_F(HealthDiagnosticBridgeTest, CreateWithCustomConfig) {
    health_diag_bridge_config_t config;
    int ret = health_diag_bridge_default_config(&config);
    ASSERT_EQ(ret, 0);

    config.capture_stack_trace = false;
    config.capture_memory_snapshot = false;
    config.default_confidence = 0.9f;
    config.enable_pattern_analysis = false;

    health_diag_bridge_t* bridge = health_diag_bridge_create(&config);

    ASSERT_NE(bridge, nullptr);
    EXPECT_TRUE(health_diag_bridge_is_ready(bridge));

    health_diag_bridge_destroy(bridge);
}

TEST_F(HealthDiagnosticBridgeTest, DestroyNullSafety) {
    EXPECT_NO_THROW(health_diag_bridge_destroy(NULL));
}

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(HealthDiagnosticBridgeTest, DefaultConfigValues) {
    health_diag_bridge_config_t config;
    int ret = health_diag_bridge_default_config(&config);

    ASSERT_EQ(ret, 0);
    EXPECT_TRUE(config.capture_stack_trace);
    EXPECT_TRUE(config.capture_memory_snapshot);
    EXPECT_GT(config.default_confidence, 0.0f);
    EXPECT_LE(config.default_confidence, 1.0f);
    EXPECT_TRUE(config.enable_pattern_analysis);
    EXPECT_GT(config.escalation_threshold, 0u);
}

TEST_F(HealthDiagnosticBridgeTest, DefaultConfigNullReturnsError) {
    int ret = health_diag_bridge_default_config(NULL);
    EXPECT_EQ(ret, -1);
}

//=============================================================================
// Anomaly Conversion Tests
//=============================================================================

TEST_F(HealthDiagnosticBridgeTest, ConvertMemoryLeakAnomaly) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    anomaly_t anomaly;
    memset(&anomaly, 0, sizeof(anomaly));
    anomaly.type = ANOMALY_MEMORY_LEAK;
    anomaly.severity = ANOMALY_SEVERITY_WARNING;
    anomaly.confidence = 0.85f;
    strncpy(anomaly.description, "Memory leak detected in module X",
            sizeof(anomaly.description) - 1);
    strncpy(anomaly.affected_component, "memory_manager",
            sizeof(anomaly.affected_component) - 1);
    anomaly.metric_value = 1024.0;
    anomaly.expected_value = 512.0;
    anomaly.deviation = 512.0;

    diagnostic_result_t* result = NULL;
    int ret = health_diag_bridge_convert_anomaly(bridge, &anomaly, &result);

    ASSERT_EQ(ret, 0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->error_type, ERROR_TYPE_MEMORY_LEAK);
    EXPECT_EQ(result->severity, DIAG_SEVERITY_WARNING);
    EXPECT_GT(result->confidence, 0.0f);

    diagnostics_free_result(result);
    health_diag_bridge_destroy(bridge);
}

TEST_F(HealthDiagnosticBridgeTest, ConvertCriticalAnomaly) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    anomaly_t anomaly;
    memset(&anomaly, 0, sizeof(anomaly));
    anomaly.type = ANOMALY_RESOURCE_EXHAUSTION;
    anomaly.severity = ANOMALY_SEVERITY_CRITICAL;
    anomaly.confidence = 0.95f;

    diagnostic_result_t* result = NULL;
    int ret = health_diag_bridge_convert_anomaly(bridge, &anomaly, &result);

    ASSERT_EQ(ret, 0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->severity, DIAG_SEVERITY_CRITICAL);

    diagnostics_free_result(result);
    health_diag_bridge_destroy(bridge);
}

TEST_F(HealthDiagnosticBridgeTest, ConvertAnomalyNullBridgeReturnsError) {
    anomaly_t anomaly;
    memset(&anomaly, 0, sizeof(anomaly));
    diagnostic_result_t* result = NULL;

    int ret = health_diag_bridge_convert_anomaly(NULL, &anomaly, &result);
    EXPECT_EQ(ret, -1);
}

TEST_F(HealthDiagnosticBridgeTest, ConvertAnomalyNullAnomalyReturnsError) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    diagnostic_result_t* result = NULL;
    int ret = health_diag_bridge_convert_anomaly(bridge, NULL, &result);
    EXPECT_EQ(ret, -1);

    health_diag_bridge_destroy(bridge);
}

//=============================================================================
// Agent Message Conversion Tests
//=============================================================================

TEST_F(HealthDiagnosticBridgeTest, ConvertDeadlockMessage) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    health_agent_message_t message;
    memset(&message, 0, sizeof(message));
    message.type = HEALTH_MSG_DEADLOCK_DETECTED;
    message.severity = HEALTH_SEVERITY_CRITICAL;
    message.source = HEALTH_SOURCE_THREADING;
    message.data.deadlock.thread_id_1 = 1001;
    message.data.deadlock.thread_id_2 = 1002;
    strncpy(message.description, "Deadlock between threads",
            sizeof(message.description) - 1);

    diagnostic_result_t* result = NULL;
    int ret = health_diag_bridge_convert_agent_message(bridge, &message, &result);

    ASSERT_EQ(ret, 0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->error_type, ERROR_TYPE_DEADLOCK);
    EXPECT_EQ(result->severity, DIAG_SEVERITY_CRITICAL);

    diagnostics_free_result(result);
    health_diag_bridge_destroy(bridge);
}

TEST_F(HealthDiagnosticBridgeTest, ConvertNaNMessage) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    health_agent_message_t message;
    memset(&message, 0, sizeof(message));
    message.type = HEALTH_MSG_NAN_DETECTED;
    message.severity = HEALTH_SEVERITY_ERROR;
    message.source = HEALTH_SOURCE_NEURAL;
    message.data.nan.neuron_id = 42;
    message.data.nan.layer_id = 3;
    message.data.nan.nan_count = 5;

    diagnostic_result_t* result = NULL;
    int ret = health_diag_bridge_convert_agent_message(bridge, &message, &result);

    ASSERT_EQ(ret, 0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->error_type, ERROR_TYPE_NAN_DETECTED);

    diagnostics_free_result(result);
    health_diag_bridge_destroy(bridge);
}

TEST_F(HealthDiagnosticBridgeTest, ConvertMemoryCorruptionMessage) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    health_agent_message_t message;
    memset(&message, 0, sizeof(message));
    message.type = HEALTH_MSG_MEMORY_CORRUPTION;
    message.severity = HEALTH_SEVERITY_CRITICAL;
    message.data.memory.address = (void*)0x12345678;
    message.data.memory.size = 256;
    message.data.memory.expected_canary = 0xDEADBEEF;
    message.data.memory.actual_canary = 0x00000000;

    diagnostic_result_t* result = NULL;
    int ret = health_diag_bridge_convert_agent_message(bridge, &message, &result);

    ASSERT_EQ(ret, 0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->error_type, ERROR_TYPE_HEAP_CORRUPTION);
    EXPECT_TRUE(result->memory_corruption_detected);

    diagnostics_free_result(result);
    health_diag_bridge_destroy(bridge);
}

//=============================================================================
// Severity Translation Tests
//=============================================================================

TEST_F(HealthDiagnosticBridgeTest, TranslateAnomalySeverity) {
    EXPECT_EQ(health_diag_bridge_translate_anomaly_severity(ANOMALY_SEVERITY_INFO),
              DIAG_SEVERITY_INFO);
    EXPECT_EQ(health_diag_bridge_translate_anomaly_severity(ANOMALY_SEVERITY_WARNING),
              DIAG_SEVERITY_WARNING);
    EXPECT_EQ(health_diag_bridge_translate_anomaly_severity(ANOMALY_SEVERITY_ERROR),
              DIAG_SEVERITY_ERROR);
    EXPECT_EQ(health_diag_bridge_translate_anomaly_severity(ANOMALY_SEVERITY_CRITICAL),
              DIAG_SEVERITY_CRITICAL);
}

TEST_F(HealthDiagnosticBridgeTest, TranslateAgentSeverity) {
    EXPECT_EQ(health_diag_bridge_translate_agent_severity(HEALTH_SEVERITY_INFO),
              DIAG_SEVERITY_INFO);
    EXPECT_EQ(health_diag_bridge_translate_agent_severity(HEALTH_SEVERITY_WARNING),
              DIAG_SEVERITY_WARNING);
    EXPECT_EQ(health_diag_bridge_translate_agent_severity(HEALTH_SEVERITY_ERROR),
              DIAG_SEVERITY_ERROR);
    EXPECT_EQ(health_diag_bridge_translate_agent_severity(HEALTH_SEVERITY_CRITICAL),
              DIAG_SEVERITY_CRITICAL);
    EXPECT_EQ(health_diag_bridge_translate_agent_severity(HEALTH_SEVERITY_FATAL),
              DIAG_SEVERITY_FATAL);
}

//=============================================================================
// Custom Mapping Tests
//=============================================================================

TEST_F(HealthDiagnosticBridgeTest, AddCustomAnomalyMapping) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    anomaly_error_mapping_t mapping;
    mapping.anomaly_type = ANOMALY_CACHE_THRASHING;
    mapping.error_type = ERROR_TYPE_INVALID_STATE;
    mapping.default_severity = DIAG_SEVERITY_WARNING;
    mapping.default_confidence = 0.75f;
    mapping.description_template = "Custom: Cache thrashing - %s";

    int ret = health_diag_bridge_add_anomaly_mapping(bridge, &mapping);
    EXPECT_EQ(ret, 0);

    const anomaly_error_mapping_t* retrieved =
        health_diag_bridge_get_anomaly_mapping(bridge, ANOMALY_CACHE_THRASHING);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->error_type, ERROR_TYPE_INVALID_STATE);
    EXPECT_FLOAT_EQ(retrieved->default_confidence, 0.75f);

    health_diag_bridge_destroy(bridge);
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(HealthDiagnosticBridgeTest, StatisticsTrackConversions) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    // Convert an anomaly
    anomaly_t anomaly;
    memset(&anomaly, 0, sizeof(anomaly));
    anomaly.type = ANOMALY_MEMORY_LEAK;
    anomaly.severity = ANOMALY_SEVERITY_WARNING;
    anomaly.confidence = 0.8f;

    diagnostic_result_t* result = NULL;
    health_diag_bridge_convert_anomaly(bridge, &anomaly, &result);
    if (result) diagnostics_free_result(result);

    // Convert an agent message
    health_agent_message_t message;
    memset(&message, 0, sizeof(message));
    message.type = HEALTH_MSG_DEADLOCK_DETECTED;
    message.severity = HEALTH_SEVERITY_CRITICAL;

    result = NULL;
    health_diag_bridge_convert_agent_message(bridge, &message, &result);
    if (result) diagnostics_free_result(result);

    // Check statistics
    health_diag_bridge_stats_t stats;
    int ret = health_diag_bridge_get_stats(bridge, &stats);
    ASSERT_EQ(ret, 0);
    EXPECT_EQ(stats.anomalies_converted, 1u);
    EXPECT_EQ(stats.agent_messages_converted, 1u);

    health_diag_bridge_destroy(bridge);
}

TEST_F(HealthDiagnosticBridgeTest, ResetStatistics) {
    health_diag_bridge_t* bridge = health_diag_bridge_create(NULL);
    ASSERT_NE(bridge, nullptr);

    // Convert something
    anomaly_t anomaly;
    memset(&anomaly, 0, sizeof(anomaly));
    anomaly.type = ANOMALY_MEMORY_LEAK;
    anomaly.severity = ANOMALY_SEVERITY_WARNING;
    anomaly.confidence = 0.8f;

    diagnostic_result_t* result = NULL;
    health_diag_bridge_convert_anomaly(bridge, &anomaly, &result);
    if (result) diagnostics_free_result(result);

    // Reset
    health_diag_bridge_reset_stats(bridge);

    // Verify reset
    health_diag_bridge_stats_t stats;
    health_diag_bridge_get_stats(bridge, &stats);
    EXPECT_EQ(stats.anomalies_converted, 0u);
    EXPECT_EQ(stats.agent_messages_converted, 0u);

    health_diag_bridge_destroy(bridge);
}

//=============================================================================
// Utility Function Tests
//=============================================================================

TEST_F(HealthDiagnosticBridgeTest, AnomalyTypeNames) {
    EXPECT_STREQ(health_diag_bridge_anomaly_type_name(ANOMALY_NONE), "NONE");
    EXPECT_STREQ(health_diag_bridge_anomaly_type_name(ANOMALY_MEMORY_LEAK), "MEMORY_LEAK");
    EXPECT_STREQ(health_diag_bridge_anomaly_type_name(ANOMALY_UNKNOWN), "UNKNOWN");
}

TEST_F(HealthDiagnosticBridgeTest, AgentMsgTypeNames) {
    EXPECT_STREQ(health_diag_bridge_agent_msg_type_name(HEALTH_MSG_ANOMALY_DETECTED),
                 "ANOMALY_DETECTED");
    EXPECT_STREQ(health_diag_bridge_agent_msg_type_name(HEALTH_MSG_DEADLOCK_DETECTED),
                 "DEADLOCK_DETECTED");
}

TEST_F(HealthDiagnosticBridgeTest, VersionString) {
    const char* version = health_diag_bridge_version();
    ASSERT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0u);
}
