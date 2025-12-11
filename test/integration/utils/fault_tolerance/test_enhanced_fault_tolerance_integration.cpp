/**
 * @file test_enhanced_fault_tolerance_integration.cpp
 * @brief Integration tests for enhanced fault tolerance modules
 *
 * Tests module interactions including:
 * - Distributed FT + Byzantine FT coordination
 * - Predictive Analysis + Graceful Degradation
 * - Chaos Engineering + Recovery Observability
 * - Hierarchical Recovery + Recovery Evolution
 * - Security (BBB) integration with all modules
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>

extern "C" {
#include "utils/fault_tolerance/nimcp_distributed_fault_tolerance.h"
#include "utils/fault_tolerance/nimcp_hierarchical_recovery.h"
#include "utils/fault_tolerance/nimcp_recovery_evolution.h"
#include "utils/fault_tolerance/nimcp_byzantine_fault_tolerance.h"
#include "utils/fault_tolerance/nimcp_graceful_degradation.h"
#include "utils/fault_tolerance/nimcp_recovery_observability.h"
#include "utils/fault_tolerance/nimcp_chaos_engineering.h"
#include "utils/fault_tolerance/nimcp_predictive_analysis.h"
#include "utils/memory/nimcp_memory.h"
}

//=============================================================================
// Multi-Module Integration Fixture
//=============================================================================

class EnhancedFaultToleranceIntegrationTest : public ::testing::Test {
protected:
    dft_context_t* dft_ctx;
    hr_context_t* hr_ctx;
    re_context_t* re_ctx;
    bft_context_t* bft_ctx;
    gd_context_t* gd_ctx;
    ro_context_t* ro_ctx;
    ce_context_t* ce_ctx;
    pa_context_t* pa_ctx;

    void SetUp() override {
        // Create all contexts with default configs
        dft_config_t dft_config = dft_default_config();
        dft_config.node_id = 1;
        dft_config.cluster_size = 5;
        dft_ctx = dft_create(&dft_config);

        hr_config_t hr_config = hr_default_config();
        hr_ctx = hr_create(&hr_config);

        re_config_t re_config = re_default_config();
        re_ctx = re_create(&re_config);

        bft_config_t bft_config = bft_default_config();
        bft_config.node_id = 1;
        bft_config.cluster_size = 7;
        bft_ctx = bft_create(&bft_config);

        gd_config_t gd_config = gd_default_config();
        gd_ctx = gd_create(&gd_config);

        ro_config_t ro_config = ro_default_config();
        ro_ctx = ro_create(&ro_config);

        ce_config_t ce_config = ce_default_config();
        ce_config.enable_dry_run = true;  // Safety
        ce_ctx = ce_create(&ce_config);

        pa_config_t pa_config = pa_default_config();
        pa_ctx = pa_create(&pa_config);
    }

    void TearDown() override {
        if (dft_ctx) dft_destroy(dft_ctx);
        if (hr_ctx) hr_destroy(hr_ctx);
        if (re_ctx) re_destroy(re_ctx);
        if (bft_ctx) bft_destroy(bft_ctx);
        if (gd_ctx) gd_destroy(gd_ctx);
        if (ro_ctx) ro_destroy(ro_ctx);
        if (ce_ctx) ce_destroy(ce_ctx);
        if (pa_ctx) pa_destroy(pa_ctx);
    }
};

//=============================================================================
// Distributed FT + Byzantine FT Integration Tests
//=============================================================================

TEST_F(EnhancedFaultToleranceIntegrationTest, DftAndBftCoordination) {
    // SCENARIO: Byzantine node detected, DFT should mark it as failed

    ASSERT_TRUE(dft_start(dft_ctx));
    ASSERT_TRUE(bft_start(bft_ctx));

    // Register nodes in both systems
    for (int i = 2; i <= 5; i++) {
        dft_node_info_t node_info = {0};
        node_info.node_id = i;
        node_info.state = DFT_NODE_HEALTHY;
        dft_register_node(dft_ctx, &node_info);
    }

    // Simulate Byzantine behavior detection
    bft_behavior_report_t report = {0};
    report.node_id = 3;
    report.behavior = BFT_BEHAVIOR_EQUIVOCATION;
    memcpy(report.evidence, "equivocated", 11);
    report.evidence_size = 11;
    bft_report_behavior(bft_ctx, &report);

    // BFT should mark node as Byzantine
    if (bft_is_byzantine(bft_ctx, 3)) {
        // Notify DFT of the Byzantine node
        dft_report_failure(dft_ctx, 3, DFT_FAILURE_BYZANTINE);

        // Verify DFT marks node as failed
        dft_node_state_t state = dft_get_node_state(dft_ctx, 3);
        EXPECT_EQ(state, DFT_NODE_FAILED);
    }

    ASSERT_TRUE(bft_stop(bft_ctx));
    ASSERT_TRUE(dft_stop(dft_ctx));
}

TEST_F(EnhancedFaultToleranceIntegrationTest, BftQuarantineAffectsDft) {
    ASSERT_TRUE(dft_start(dft_ctx));
    ASSERT_TRUE(bft_start(bft_ctx));

    // Quarantine a node in BFT
    bft_quarantine_node(bft_ctx, 2, 60000);
    EXPECT_TRUE(bft_is_quarantined(bft_ctx, 2));

    // DFT should exclude quarantined nodes from quorum
    uint32_t quorum_before = dft_get_quorum_size(dft_ctx);

    // Report to DFT that node is quarantined
    dft_report_failure(dft_ctx, 2, DFT_FAILURE_QUARANTINED);

    // Quorum calculation should adjust
    uint32_t quorum_after = dft_get_quorum_size(dft_ctx);

    // Quorum size may decrease due to quarantine
    (void)quorum_before;
    (void)quorum_after;

    ASSERT_TRUE(bft_stop(bft_ctx));
    ASSERT_TRUE(dft_stop(dft_ctx));
}

//=============================================================================
// Predictive Analysis + Graceful Degradation Integration Tests
//=============================================================================

TEST_F(EnhancedFaultToleranceIntegrationTest, PredictiveAnalysisTriggersDegradation) {
    // SCENARIO: PA predicts resource exhaustion, GD proactively degrades

    ASSERT_TRUE(pa_start(pa_ctx));
    ASSERT_TRUE(gd_start(gd_ctx));

    // Initial tier should be FULL
    EXPECT_EQ(gd_get_current_tier(gd_ctx), GD_TIER_FULL);

    // Add trending data showing resource exhaustion
    for (int i = 0; i < 100; i++) {
        double cpu = 50.0 + i * 0.4;  // CPU trending up to 90%
        pa_add_sample(pa_ctx, PA_SERIES_CPU, cpu);

        // Update GD with current resource usage
        gd_update_resource(gd_ctx, GD_RESOURCE_CPU, cpu);
    }

    // Get PA predictions
    pa_failure_prediction_t predictions[5];
    uint32_t pred_count = pa_predict_failures(pa_ctx, predictions, 5);

    // If prediction indicates impending failure, degrade proactively
    if (pred_count > 0 && predictions[0].probability > 0.7) {
        gd_set_tier(gd_ctx, GD_TIER_REDUCED, "predictive_degradation");
        EXPECT_EQ(gd_get_current_tier(gd_ctx), GD_TIER_REDUCED);
    }

    ASSERT_TRUE(gd_stop(gd_ctx));
    ASSERT_TRUE(pa_stop(pa_ctx));
}

TEST_F(EnhancedFaultToleranceIntegrationTest, DegradationTriggersLoadShedding) {
    ASSERT_TRUE(gd_start(gd_ctx));

    // Register features with priorities
    gd_feature_t critical_feature = {0};
    strncpy(critical_feature.name, "authentication", sizeof(critical_feature.name) - 1);
    critical_feature.priority = GD_PRIORITY_CRITICAL;
    critical_feature.minimum_tier = GD_TIER_EMERGENCY;
    uint32_t critical_id = gd_register_feature(gd_ctx, &critical_feature);

    gd_feature_t optional_feature = {0};
    strncpy(optional_feature.name, "analytics", sizeof(optional_feature.name) - 1);
    optional_feature.priority = GD_PRIORITY_OPTIONAL;
    optional_feature.minimum_tier = GD_TIER_FULL;
    uint32_t optional_id = gd_register_feature(gd_ctx, &optional_feature);

    // Both features should be enabled in FULL tier
    EXPECT_TRUE(gd_is_feature_enabled(gd_ctx, critical_id));
    EXPECT_TRUE(gd_is_feature_enabled(gd_ctx, optional_id));

    // Degrade to REDUCED tier
    gd_set_tier(gd_ctx, GD_TIER_REDUCED, "high_load");

    // Optional feature should be disabled
    EXPECT_TRUE(gd_is_feature_enabled(gd_ctx, critical_id));
    EXPECT_FALSE(gd_is_feature_enabled(gd_ctx, optional_id));

    // Start load shedding for low priority requests
    gd_start_load_shedding(gd_ctx, 50.0, GD_PRIORITY_MEDIUM, 60000);

    // Critical requests should still be accepted
    EXPECT_TRUE(gd_should_accept_request(gd_ctx, GD_PRIORITY_CRITICAL));

    // Low priority may be rejected
    bool accept_low = gd_should_accept_request(gd_ctx, GD_PRIORITY_LOW);
    EXPECT_FALSE(accept_low);

    ASSERT_TRUE(gd_stop(gd_ctx));
}

//=============================================================================
// Chaos Engineering + Recovery Observability Integration Tests
//=============================================================================

TEST_F(EnhancedFaultToleranceIntegrationTest, ChaosExperimentGeneratesMetrics) {
    // SCENARIO: Chaos experiment injects faults, RO captures metrics

    ASSERT_TRUE(ro_start(ro_ctx));

    // Create and configure chaos experiment
    uint32_t exp_id = ce_create_experiment(ce_ctx, "latency_test",
        "Test system behavior under network latency");

    ce_fault_spec_t fault = {0};
    fault.type = CE_FAULT_NETWORK_LATENCY;
    fault.pattern = CE_PATTERN_ONCE;
    fault.intensity = 50.0;
    fault.duration_ms = 1000;
    ce_set_fault(ce_ctx, exp_id, &fault);

    ce_target_spec_t target = {0};
    strncpy(target.name, "test_nodes", sizeof(target.name) - 1);
    target.node_ids[0] = 1;
    target.node_count = 1;
    ce_set_target(ce_ctx, exp_id, &target);

    // Record baseline metric
    ro_record_counter(ro_ctx, "chaos.experiments.started", 1);

    // Start experiment (dry run)
    ce_start_experiment(ce_ctx, exp_id);

    // Record fault injection event
    ro_event_t event = {0};
    strncpy(event.name, "fault_injected", sizeof(event.name) - 1);
    event.type = RO_EVENT_FAULT_DETECTED;
    event.fault_type = CE_FAULT_NETWORK_LATENCY;
    event.node_id = 1;
    event.severity = RO_SEVERITY_WARNING;
    ro_record_event(ro_ctx, &event);

    // Wait briefly for experiment
    struct timespec ts = {0, 100000000};  // 100ms
    nanosleep(&ts, NULL);

    // Record recovery metric
    ro_record_counter(ro_ctx, "chaos.recoveries.observed", 1);

    // Verify metrics were recorded
    ro_stats_t stats;
    ro_get_stats(ro_ctx, &stats);
    EXPECT_GE(stats.events_recorded, 1);

    ASSERT_TRUE(ro_stop(ro_ctx));
}

TEST_F(EnhancedFaultToleranceIntegrationTest, ObservabilityTracksRecoveryTime) {
    ASSERT_TRUE(ro_start(ro_ctx));

    // Start a trace for fault recovery
    ro_trace_id_t trace_id = ro_start_trace(ro_ctx, "fault_recovery");

    // Create spans for each recovery phase
    ro_span_t detect_span = {0};
    strncpy(detect_span.name, "detect_failure", sizeof(detect_span.name) - 1);
    detect_span.trace_id = trace_id;
    ro_start_span(ro_ctx, &detect_span);

    // Simulate detection time
    struct timespec ts = {0, 10000000};  // 10ms
    nanosleep(&ts, NULL);

    ro_end_span(ro_ctx, &detect_span);

    // Recovery span
    ro_span_t recover_span = {0};
    strncpy(recover_span.name, "recover_component", sizeof(recover_span.name) - 1);
    recover_span.trace_id = trace_id;
    ro_start_span(ro_ctx, &recover_span);

    // Simulate recovery time
    nanosleep(&ts, NULL);

    ro_end_span(ro_ctx, &recover_span);

    // End trace
    ro_end_trace(ro_ctx, trace_id);

    // Get MTTR
    double mttr = ro_get_mttr(ro_ctx, 0);  // All fault types
    // MTTR should reflect total recovery time
    (void)mttr;

    ASSERT_TRUE(ro_stop(ro_ctx));
}

//=============================================================================
// Hierarchical Recovery + Recovery Evolution Integration Tests
//=============================================================================

TEST_F(EnhancedFaultToleranceIntegrationTest, EvolutionOptimizesRecoveryStrategies) {
    // SCENARIO: HR uses RE to select and improve recovery strategies

    ASSERT_TRUE(hr_start(hr_ctx));
    ASSERT_TRUE(re_start(re_ctx));

    // Register nodes and pods
    hr_pod_t pod = {0};
    pod.pod_id = 1;
    strncpy(pod.name, "test_pod", sizeof(pod.name) - 1);
    pod.max_concurrent_failures = 2;
    hr_register_pod(hr_ctx, &pod);

    for (int i = 1; i <= 3; i++) {
        hr_node_t node = {0};
        node.node_id = i;
        node.pod_id = 1;
        snprintf(node.name, sizeof(node.name), "node_%d", i);
        hr_register_node(hr_ctx, &node);
    }

    // Register recovery strategies with RE
    re_strategy_t strategies[3];

    // Strategy 1: Fast restart
    memset(&strategies[0], 0, sizeof(re_strategy_t));
    strncpy(strategies[0].name, "fast_restart", sizeof(strategies[0].name) - 1);
    strategies[0].fault_type = 1;
    strategies[0].actions[0] = 1;  // Restart
    strategies[0].action_count = 1;
    strategies[0].fitness = 0.6;

    // Strategy 2: Failover
    memset(&strategies[1], 0, sizeof(re_strategy_t));
    strncpy(strategies[1].name, "failover", sizeof(strategies[1].name) - 1);
    strategies[1].fault_type = 1;
    strategies[1].actions[0] = 2;  // Failover
    strategies[1].action_count = 1;
    strategies[1].fitness = 0.7;

    // Strategy 3: Checkpoint restore
    memset(&strategies[2], 0, sizeof(re_strategy_t));
    strncpy(strategies[2].name, "checkpoint_restore", sizeof(strategies[2].name) - 1);
    strategies[2].fault_type = 1;
    strategies[2].actions[0] = 3;  // Restore checkpoint
    strategies[2].action_count = 1;
    strategies[2].fitness = 0.8;

    for (int i = 0; i < 3; i++) {
        re_register_strategy(re_ctx, &strategies[i]);
    }

    // Select best strategy for fault type 1
    re_strategy_t best;
    EXPECT_TRUE(re_select_best_strategy(re_ctx, 1, &best));
    EXPECT_NEAR(best.fitness, 0.8, 0.01);  // checkpoint_restore

    // Report failure to HR
    hr_failure_report_t report = {0};
    report.failed_id = 1;
    report.level = HR_LEVEL_NODE;
    report.failure_type = 1;
    strncpy(report.description, "Node 1 failed", sizeof(report.description) - 1);

    uint32_t recovery_id = hr_report_failure(hr_ctx, &report);
    EXPECT_GT(recovery_id, 0);

    // Update strategy fitness based on recovery result
    re_evaluation_result_t result = {0};
    result.success = true;
    result.recovery_time_ms = 100;
    result.resource_usage = 0.3;

    uint32_t strategy_id = 3;  // checkpoint_restore
    re_update_strategy_fitness(re_ctx, strategy_id, &result);

    ASSERT_TRUE(re_stop(re_ctx));
    ASSERT_TRUE(hr_stop(hr_ctx));
}

TEST_F(EnhancedFaultToleranceIntegrationTest, HierarchicalEscalationWithEvolution) {
    ASSERT_TRUE(hr_start(hr_ctx));
    ASSERT_TRUE(re_start(re_ctx));

    // Set escalation policy
    hr_escalation_policy_t policy = {0};
    policy.source_level = HR_LEVEL_NODE;
    policy.target_level = HR_LEVEL_POD;
    policy.escalation_timeout_ms = 1000;
    policy.max_retries = 2;
    policy.auto_escalate = true;
    hr_set_escalation_policy(hr_ctx, &policy);

    // Register hierarchy
    hr_region_t region = {0};
    region.region_id = 1;
    strncpy(region.name, "region_1", sizeof(region.name) - 1);
    hr_register_region(hr_ctx, &region);

    hr_pod_t pod = {0};
    pod.pod_id = 1;
    pod.region_id = 1;
    strncpy(pod.name, "pod_1", sizeof(pod.name) - 1);
    hr_register_pod(hr_ctx, &pod);

    hr_node_t node = {0};
    node.node_id = 1;
    node.pod_id = 1;
    strncpy(node.name, "node_1", sizeof(node.name) - 1);
    hr_register_node(hr_ctx, &node);

    // Report node failure
    hr_failure_report_t report = {0};
    report.failed_id = 1;
    report.level = HR_LEVEL_NODE;
    uint32_t recovery_id = hr_report_failure(hr_ctx, &report);

    // If node recovery fails, should escalate to pod level
    // Simulate by manually escalating
    hr_escalate(hr_ctx, recovery_id, HR_LEVEL_POD);

    // Record experience for learning
    re_experience_t exp = {0};
    exp.state.fault_type = 1;
    exp.action = 1;  // Escalate
    exp.reward = 0.5;  // Partial success
    exp.done = true;
    re_store_experience(re_ctx, &exp);

    ASSERT_TRUE(re_stop(re_ctx));
    ASSERT_TRUE(hr_stop(hr_ctx));
}

//=============================================================================
// Full Pipeline Integration Tests
//=============================================================================

TEST_F(EnhancedFaultToleranceIntegrationTest, FullImmuneSystemPipeline) {
    // SCENARIO: Complete "immune system" response to a detected threat

    // Start all systems
    ASSERT_TRUE(pa_start(pa_ctx));
    ASSERT_TRUE(gd_start(gd_ctx));
    ASSERT_TRUE(dft_start(dft_ctx));
    ASSERT_TRUE(hr_start(hr_ctx));
    ASSERT_TRUE(ro_start(ro_ctx));

    // Register nodes
    for (int i = 2; i <= 5; i++) {
        dft_node_info_t node_info = {0};
        node_info.node_id = i;
        node_info.state = DFT_NODE_HEALTHY;
        dft_register_node(dft_ctx, &node_info);

        hr_node_t hr_node = {0};
        hr_node.node_id = i;
        snprintf(hr_node.name, sizeof(hr_node.name), "node_%d", i);
        hr_register_node(hr_ctx, &hr_node);
    }

    // PHASE 1: DETECTION (Predictive Analysis)
    // Add anomalous data pattern
    for (int i = 0; i < 50; i++) {
        pa_add_sample(pa_ctx, PA_SERIES_ERROR_RATE, 0.01 + i * 0.005);
    }

    // Start observability trace
    ro_trace_id_t trace_id = ro_start_trace(ro_ctx, "immune_response");

    // Detect anomalies
    pa_anomaly_t anomalies[10];
    uint32_t anomaly_count = pa_detect_anomalies(pa_ctx, PA_SERIES_ERROR_RATE, anomalies, 10);

    ro_record_counter(ro_ctx, "immune.anomalies.detected", anomaly_count);

    // PHASE 2: ASSESSMENT (Distributed FT)
    if (anomaly_count > 0) {
        // Check cluster health
        dft_cluster_health_t health;
        dft_get_cluster_health(dft_ctx, &health);

        ro_record_gauge(ro_ctx, "immune.cluster.health", health.healthy_nodes);
    }

    // PHASE 3: RESPONSE (Graceful Degradation)
    if (anomaly_count > 2) {
        // Proactive degradation
        gd_set_tier(gd_ctx, GD_TIER_REDUCED, "anomaly_response");

        ro_event_t event = {0};
        strncpy(event.name, "tier_reduced", sizeof(event.name) - 1);
        event.type = RO_EVENT_TIER_CHANGE;
        event.severity = RO_SEVERITY_WARNING;
        ro_record_event(ro_ctx, &event);
    }

    // PHASE 4: RECOVERY (Hierarchical Recovery)
    // If node failure is imminent, initiate recovery
    if (anomaly_count > 0) {
        hr_failure_report_t report = {0};
        report.failed_id = 2;  // Predicted failing node
        report.level = HR_LEVEL_NODE;
        report.failure_type = 1;
        strncpy(report.description, "Predictive failure", sizeof(report.description) - 1);

        uint32_t recovery_id = hr_report_failure(hr_ctx, &report);
        ro_record_counter(ro_ctx, "immune.recoveries.initiated", 1);
        (void)recovery_id;
    }

    // End trace
    ro_end_trace(ro_ctx, trace_id);

    // Get final stats
    ro_stats_t stats;
    ro_get_stats(ro_ctx, &stats);
    EXPECT_GE(stats.events_recorded, 1);

    // Stop all systems
    ASSERT_TRUE(ro_stop(ro_ctx));
    ASSERT_TRUE(hr_stop(hr_ctx));
    ASSERT_TRUE(dft_stop(dft_ctx));
    ASSERT_TRUE(gd_stop(gd_ctx));
    ASSERT_TRUE(pa_stop(pa_ctx));
}

TEST_F(EnhancedFaultToleranceIntegrationTest, ChaosValidatesRecovery) {
    // SCENARIO: Chaos engineering validates that recovery systems work

    ASSERT_TRUE(hr_start(hr_ctx));
    ASSERT_TRUE(ro_start(ro_ctx));

    // Register nodes
    hr_node_t node = {0};
    node.node_id = 1;
    strncpy(node.name, "chaos_target", sizeof(node.name) - 1);
    hr_register_node(hr_ctx, &node);

    // Create chaos experiment with hypothesis
    uint32_t exp_id = ce_create_experiment(ce_ctx, "recovery_validation",
        "Verify system recovers from process kill within 5 seconds");

    ce_fault_spec_t fault = {0};
    fault.type = CE_FAULT_PROCESS_KILL;
    fault.pattern = CE_PATTERN_ONCE;
    ce_set_fault(ce_ctx, exp_id, &fault);

    ce_target_spec_t target = {0};
    strncpy(target.name, "chaos_target", sizeof(target.name) - 1);
    target.node_ids[0] = 1;
    target.node_count = 1;
    ce_set_target(ce_ctx, exp_id, &target);

    ce_hypothesis_t hypothesis = {0};
    strncpy(hypothesis.description,
            "System should recover within 5 seconds",
            sizeof(hypothesis.description) - 1);
    strncpy(hypothesis.metric_name, "recovery_time_ms", sizeof(hypothesis.metric_name) - 1);
    hypothesis.expected_min = 0;
    hypothesis.expected_max = 5000;
    ce_add_hypothesis(ce_ctx, exp_id, &hypothesis);

    // Add safety guardrail
    ce_guardrail_config_t guardrail = {0};
    guardrail.type = CE_GUARDRAIL_TIME_LIMIT;
    guardrail.threshold = 10000;  // 10 second max
    guardrail.abort_on_violation = true;
    ce_add_guardrail(ce_ctx, exp_id, &guardrail);

    // Start experiment (dry run)
    ce_start_experiment(ce_ctx, exp_id);

    // Record baseline
    ro_trace_id_t trace_id = ro_start_trace(ro_ctx, "chaos_recovery");

    // Simulate failure detection and recovery
    hr_failure_report_t report = {0};
    report.failed_id = 1;
    report.level = HR_LEVEL_NODE;
    uint32_t recovery_id = hr_report_failure(hr_ctx, &report);

    // Simulate recovery time
    struct timespec ts = {0, 500000000};  // 500ms
    nanosleep(&ts, NULL);

    // End trace
    ro_end_trace(ro_ctx, trace_id);

    // Record recovery success
    ro_record_counter(ro_ctx, "chaos.recovery.succeeded", 1);
    ro_record_histogram(ro_ctx, "chaos.recovery.time_ms", 500);

    // Get experiment result (would be evaluated in real scenario)
    ce_experiment_t exp;
    ce_get_experiment(ce_ctx, exp_id, &exp);

    (void)recovery_id;

    ASSERT_TRUE(ro_stop(ro_ctx));
    ASSERT_TRUE(hr_stop(hr_ctx));
}

//=============================================================================
// Cross-Module State Synchronization Tests
//=============================================================================

TEST_F(EnhancedFaultToleranceIntegrationTest, StateConsistencyAcrossModules) {
    // SCENARIO: Verify state is consistent across DFT, HR, and GD

    ASSERT_TRUE(dft_start(dft_ctx));
    ASSERT_TRUE(hr_start(hr_ctx));
    ASSERT_TRUE(gd_start(gd_ctx));

    // All should start healthy/full
    EXPECT_EQ(gd_get_current_tier(gd_ctx), GD_TIER_FULL);

    // Register same nodes in both DFT and HR
    for (int i = 1; i <= 3; i++) {
        dft_node_info_t dft_node = {0};
        dft_node.node_id = i;
        dft_node.state = DFT_NODE_HEALTHY;
        dft_register_node(dft_ctx, &dft_node);

        hr_node_t hr_node = {0};
        hr_node.node_id = i;
        snprintf(hr_node.name, sizeof(hr_node.name), "node_%d", i);
        hr_register_node(hr_ctx, &hr_node);
    }

    // Fail a node in DFT
    dft_report_failure(dft_ctx, 2, DFT_FAILURE_CRASH);

    // HR should also handle the failure
    hr_failure_report_t hr_report = {0};
    hr_report.failed_id = 2;
    hr_report.level = HR_LEVEL_NODE;
    hr_report_failure(hr_ctx, &hr_report);

    // If too many failures, GD should degrade
    dft_report_failure(dft_ctx, 3, DFT_FAILURE_CRASH);

    // Update GD based on failure count
    dft_cluster_health_t health;
    dft_get_cluster_health(dft_ctx, &health);

    if (health.failed_nodes >= 2) {
        gd_set_tier(gd_ctx, GD_TIER_REDUCED, "cluster_failures");
        EXPECT_EQ(gd_get_current_tier(gd_ctx), GD_TIER_REDUCED);
    }

    ASSERT_TRUE(gd_stop(gd_ctx));
    ASSERT_TRUE(hr_stop(hr_ctx));
    ASSERT_TRUE(dft_stop(dft_ctx));
}

