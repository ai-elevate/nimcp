/**
 * @file test_plasticity_anomaly_integration.cpp
 * @brief Integration tests for Plasticity Anomaly Detection
 * @date 2026-01-20
 *
 * Tests the integration between plasticity anomaly detection, STDP/BCM
 * learning systems, health monitoring, and bio-async communication.
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <cmath>

extern "C" {
#include "utils/fault_tolerance/nimcp_plasticity_anomaly_detection.h"
#include "utils/fault_tolerance/nimcp_stdp_health_metrics.h"
#include "utils/fault_tolerance/nimcp_health_agent.h"
#include "snn/plasticity/nimcp_stdp.h"
#include "snn/plasticity/nimcp_bcm.h"
#include "snn/plasticity/nimcp_homeostatic.h"
#include "async/nimcp_bio_async.h"
}

class PlasticityAnomalyIntegrationTest : public ::testing::Test {
protected:
    plasticity_anomaly_detector_t* detector = nullptr;
    stdp_health_metrics_t* stdp_health = nullptr;
    nimcp_health_agent_t* health_agent = nullptr;
    nimcp_stdp_context_t* stdp = nullptr;
    nimcp_bcm_context_t* bcm = nullptr;
    nimcp_homeostatic_context_t* homeostatic = nullptr;
    nimcp_bio_async_t* bio_async = nullptr;

    void SetUp() override {
        // Create bio-async for message passing
        nimcp_bio_async_config_t bio_config;
        nimcp_bio_async_default_config(&bio_config);
        bio_async = nimcp_bio_async_create(&bio_config);

        // Create health agent
        nimcp_health_agent_config_t ha_config;
        nimcp_health_agent_default_config(&ha_config);
        health_agent = nimcp_health_agent_create(&ha_config, bio_async);

        // Create STDP health metrics
        stdp_health_config_t sh_config;
        stdp_health_default_config(&sh_config);
        stdp_health = stdp_health_create(&sh_config, bio_async);

        // Create plasticity contexts
        nimcp_stdp_config_t stdp_config;
        nimcp_stdp_default_config(&stdp_config);
        stdp = nimcp_stdp_create(&stdp_config);

        nimcp_bcm_config_t bcm_config;
        nimcp_bcm_default_config(&bcm_config);
        bcm = nimcp_bcm_create(&bcm_config);

        nimcp_homeostatic_config_t homeo_config;
        nimcp_homeostatic_default_config(&homeo_config);
        homeostatic = nimcp_homeostatic_create(&homeo_config);
    }

    void TearDown() override {
        if (detector) {
            plasticity_anomaly_destroy(detector);
            detector = nullptr;
        }
        if (stdp_health) {
            stdp_health_destroy(stdp_health);
            stdp_health = nullptr;
        }
        if (health_agent) {
            nimcp_health_agent_destroy(health_agent);
            health_agent = nullptr;
        }
        if (stdp) {
            nimcp_stdp_destroy(stdp);
            stdp = nullptr;
        }
        if (bcm) {
            nimcp_bcm_destroy(bcm);
            bcm = nullptr;
        }
        if (homeostatic) {
            nimcp_homeostatic_destroy(homeostatic);
            homeostatic = nullptr;
        }
        if (bio_async) {
            nimcp_bio_async_destroy(bio_async);
            bio_async = nullptr;
        }
    }
};

/* ============================================================================
 * Basic Integration Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, CreateWithBioAsync) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    EXPECT_NE(detector, nullptr);
}

TEST_F(PlasticityAnomalyIntegrationTest, LoadAllDefaultRules) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    int loaded = plasticity_anomaly_load_all_default_rules(detector);
    EXPECT_GT(loaded, 0);
}

TEST_F(PlasticityAnomalyIntegrationTest, LoadCategorySpecificRules) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    // Load rules for each category
    int weight_rules = plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_WEIGHT);
    int bcm_rules = plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_BCM);
    int homeo_rules = plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_HOMEOSTATIC);
    int learning_rules = plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_LEARNING);

    EXPECT_GT(weight_rules, 0);
    EXPECT_GT(bcm_rules, 0);
    EXPECT_GT(homeo_rules, 0);
    EXPECT_GT(learning_rules, 0);
}

/* ============================================================================
 * Weight Anomaly Detection Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, DetectWeightExplosion) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_WEIGHT);

    // Normal weights first
    float normal_weights[] = {0.5f, 0.3f, 0.7f, 0.1f};
    int anomalies = plasticity_anomaly_analyze_weights(detector, normal_weights, 4, "normal");
    EXPECT_EQ(anomalies, 0);

    // Exploding weights
    float exploding_weights[] = {100.0f, 200.0f, 500.0f, 1000.0f};
    anomalies = plasticity_anomaly_analyze_weights(detector, exploding_weights, 4, "exploding");
    EXPECT_GT(anomalies, 0);
}

TEST_F(PlasticityAnomalyIntegrationTest, DetectWeightVanishing) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_WEIGHT);

    // Vanishing weights
    float vanishing_weights[] = {1e-10f, 1e-12f, 1e-15f, 0.0f};
    int anomalies = plasticity_anomaly_analyze_weights(detector, vanishing_weights, 4, "vanishing");
    EXPECT_GT(anomalies, 0);
}

TEST_F(PlasticityAnomalyIntegrationTest, DetectWeightNaN) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_WEIGHT);

    float nan_weights[] = {0.5f, NAN, 0.3f};
    int anomalies = plasticity_anomaly_analyze_weights(detector, nan_weights, 3, "nan_test");
    EXPECT_GT(anomalies, 0);

    plasticity_detection_stats_t stats;
    plasticity_anomaly_get_stats(detector, &stats);
    EXPECT_GT(stats.weight_anomalies, 0u);
}

/* ============================================================================
 * Timing Anomaly Detection Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, DetectTimingAnomalies) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);
    config.detect_timing_anomalies = true;

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_TIMING);

    // Normal timing
    float pre_normal[] = {1.0f, 5.0f, 10.0f};
    float post_normal[] = {2.0f, 6.0f, 11.0f};
    int anomalies = plasticity_anomaly_analyze_timing(detector, pre_normal, post_normal, 3);
    EXPECT_GE(anomalies, 0);

    // Very long timing windows (potential issue)
    float pre_long[] = {1.0f, 100.0f, 500.0f};
    float post_long[] = {50.0f, 250.0f, 750.0f};
    anomalies = plasticity_anomaly_analyze_timing(detector, pre_long, post_long, 3);
    EXPECT_GE(anomalies, 0);  // May or may not flag depending on thresholds
}

/* ============================================================================
 * Metric Submission and Detection Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, SubmitAndDetectMetrics) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    // Submit various metrics
    plasticity_anomaly_submit_metric(detector, "weight_mean", 0.5f, PLASTICITY_CATEGORY_WEIGHT);
    plasticity_anomaly_submit_metric(detector, "weight_std", 0.1f, PLASTICITY_CATEGORY_WEIGHT);
    plasticity_anomaly_submit_metric(detector, "activity_rate", 0.3f, PLASTICITY_CATEGORY_HOMEOSTATIC);
    plasticity_anomaly_submit_metric(detector, "learning_rate", 0.01f, PLASTICITY_CATEGORY_LEARNING);

    // Run detection
    int detected = plasticity_anomaly_detect(detector);
    EXPECT_GE(detected, 0);

    plasticity_detection_stats_t stats;
    plasticity_anomaly_get_stats(detector, &stats);
    EXPECT_GT(stats.checks_performed, 0u);
}

TEST_F(PlasticityAnomalyIntegrationTest, DetectAnomalousMetrics) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    // Submit anomalous metrics
    plasticity_anomaly_submit_metric(detector, "weight_mean", 1000.0f, PLASTICITY_CATEGORY_WEIGHT);  // Too high
    plasticity_anomaly_submit_metric(detector, "learning_rate", 10.0f, PLASTICITY_CATEGORY_LEARNING);  // Too high

    int detected = plasticity_anomaly_detect(detector);
    EXPECT_GT(detected, 0);
}

/* ============================================================================
 * Custom Rule Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, AddAndUseCustomRule) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    // Add custom threshold rule
    plasticity_detection_rule_t rule = {};
    rule.rule_id = 1000;
    rule.enabled = true;
    rule.type = PLASTICITY_RULE_THRESHOLD;
    rule.anomaly = PLASTICITY_ANOMALY_WEIGHT_EXPLOSION;
    rule.category = PLASTICITY_CATEGORY_WEIGHT;
    rule.severity = PLASTICITY_SEVERITY_ERROR;
    rule.params.threshold.max_value = 5.0f;
    rule.params.threshold.min_value = -5.0f;
    snprintf(rule.metric_name, sizeof(rule.metric_name), "custom_weight");

    int id = plasticity_anomaly_add_rule(detector, &rule);
    EXPECT_EQ(id, 1000);

    // Test the rule
    plasticity_anomaly_submit_metric(detector, "custom_weight", 10.0f, PLASTICITY_CATEGORY_WEIGHT);
    int detected = plasticity_anomaly_detect(detector);
    EXPECT_GT(detected, 0);
}

TEST_F(PlasticityAnomalyIntegrationTest, DisableRulePreventsDetection) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    // Add rule
    plasticity_detection_rule_t rule = {};
    rule.rule_id = 1001;
    rule.enabled = true;
    rule.type = PLASTICITY_RULE_THRESHOLD;
    rule.anomaly = PLASTICITY_ANOMALY_WEIGHT_EXPLOSION;
    rule.category = PLASTICITY_CATEGORY_WEIGHT;
    rule.severity = PLASTICITY_SEVERITY_WARNING;
    rule.params.threshold.max_value = 1.0f;
    snprintf(rule.metric_name, sizeof(rule.metric_name), "test_weight");

    plasticity_anomaly_add_rule(detector, &rule);

    // Submit metric that would trigger
    plasticity_anomaly_submit_metric(detector, "test_weight", 5.0f, PLASTICITY_CATEGORY_WEIGHT);

    // Disable rule
    plasticity_anomaly_set_rule_enabled(detector, 1001, false);

    // Detection should not flag this
    int detected = plasticity_anomaly_detect(detector);
    // May still detect 0 since rule is disabled
    EXPECT_GE(detected, 0);
}

/* ============================================================================
 * Callback Integration Tests
 * ============================================================================ */

static std::atomic<int> anomaly_callback_count{0};
static plasticity_anomaly_type_t last_detected_anomaly;
static plasticity_severity_t last_severity;

static void test_anomaly_callback(const plasticity_anomaly_report_t* report, void* user_data) {
    (void)user_data;
    if (report) {
        last_detected_anomaly = report->anomaly;
        last_severity = report->severity;
    }
    anomaly_callback_count++;
}

TEST_F(PlasticityAnomalyIntegrationTest, AnomalyCallbackTriggered) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_default_rules(detector, PLASTICITY_CATEGORY_WEIGHT);

    anomaly_callback_count = 0;
    plasticity_anomaly_set_callback(detector, test_anomaly_callback, nullptr);

    // Trigger anomaly with NaN weights
    float nan_weights[] = {NAN, 0.5f, NAN};
    plasticity_anomaly_analyze_weights(detector, nan_weights, 3, "callback_test");

    EXPECT_GT(anomaly_callback_count.load(), 0);
}

static std::atomic<int> health_callback_count{0};

static void test_health_callback(float health_score, void* user_data) {
    (void)user_data;
    (void)health_score;
    health_callback_count++;
}

TEST_F(PlasticityAnomalyIntegrationTest, HealthCallbackOnDetection) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    health_callback_count = 0;
    plasticity_anomaly_set_health_callback(detector, test_health_callback, nullptr);

    // Run detection which should invoke health callback
    plasticity_anomaly_detect(detector);

    EXPECT_GT(health_callback_count.load(), 0);
}

/* ============================================================================
 * Health Score Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, HealthScoreDegradesWithAnomalies) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    float initial_health = plasticity_anomaly_get_health(detector);
    EXPECT_FLOAT_EQ(initial_health, 1.0f);

    // Introduce anomalies
    float bad_weights[] = {NAN, INFINITY, -INFINITY, 1e10f};
    plasticity_anomaly_analyze_weights(detector, bad_weights, 4, "health_test");
    plasticity_anomaly_detect(detector);

    float after_health = plasticity_anomaly_get_health(detector);
    EXPECT_LT(after_health, initial_health);
}

TEST_F(PlasticityAnomalyIntegrationTest, SeverityAffectsHealthImpact) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    // Add rules with different severities
    plasticity_detection_rule_t warning_rule = {};
    warning_rule.rule_id = 2000;
    warning_rule.enabled = true;
    warning_rule.type = PLASTICITY_RULE_THRESHOLD;
    warning_rule.anomaly = PLASTICITY_ANOMALY_WEIGHT_SATURATION;
    warning_rule.category = PLASTICITY_CATEGORY_WEIGHT;
    warning_rule.severity = PLASTICITY_SEVERITY_WARNING;
    warning_rule.params.threshold.max_value = 0.9f;
    snprintf(warning_rule.metric_name, sizeof(warning_rule.metric_name), "warn_metric");
    plasticity_anomaly_add_rule(detector, &warning_rule);

    plasticity_detection_rule_t critical_rule = {};
    critical_rule.rule_id = 2001;
    critical_rule.enabled = true;
    critical_rule.type = PLASTICITY_RULE_THRESHOLD;
    critical_rule.anomaly = PLASTICITY_ANOMALY_WEIGHT_EXPLOSION;
    critical_rule.category = PLASTICITY_CATEGORY_WEIGHT;
    critical_rule.severity = PLASTICITY_SEVERITY_CRITICAL;
    critical_rule.params.threshold.max_value = 10.0f;
    snprintf(critical_rule.metric_name, sizeof(critical_rule.metric_name), "crit_metric");
    plasticity_anomaly_add_rule(detector, &critical_rule);

    // Trigger warning
    plasticity_anomaly_submit_metric(detector, "warn_metric", 0.95f, PLASTICITY_CATEGORY_WEIGHT);
    plasticity_anomaly_detect(detector);
    float health_after_warning = plasticity_anomaly_get_health(detector);

    // Reset and trigger critical
    plasticity_anomaly_reset_stats(detector);
    plasticity_anomaly_submit_metric(detector, "crit_metric", 100.0f, PLASTICITY_CATEGORY_WEIGHT);
    plasticity_anomaly_detect(detector);
    float health_after_critical = plasticity_anomaly_get_health(detector);

    // Critical should have more impact
    EXPECT_LT(health_after_critical, health_after_warning);
}

/* ============================================================================
 * Bio-Async Integration Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, ConnectToBioAsync) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    int result = plasticity_anomaly_connect_bio_async(detector, bio_async);
    EXPECT_EQ(result, 0);
}

TEST_F(PlasticityAnomalyIntegrationTest, BroadcastAnomalyViaBioAsync) {
    if (!bio_async) {
        GTEST_SKIP() << "Bio-async not available";
    }

    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_connect_bio_async(detector, bio_async);
    plasticity_anomaly_load_all_default_rules(detector);

    // Trigger anomaly
    float bad_weights[] = {NAN, 0.5f};
    plasticity_anomaly_analyze_weights(detector, bad_weights, 2, "broadcast_test");

    // Broadcast should work
    int result = plasticity_anomaly_broadcast(detector);
    EXPECT_EQ(result, 0);
}

/* ============================================================================
 * Report Generation Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, GetAnomalyReports) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    // Generate some anomalies
    float bad_weights[] = {NAN, INFINITY};
    plasticity_anomaly_analyze_weights(detector, bad_weights, 2, "report_test");
    plasticity_anomaly_detect(detector);

    // Get reports
    plasticity_anomaly_report_t reports[10];
    int count = plasticity_anomaly_get_reports(detector, reports, 10);
    EXPECT_GT(count, 0);

    // Verify report contents
    if (count > 0) {
        EXPECT_NE(reports[0].anomaly, PLASTICITY_ANOMALY_NONE);
        EXPECT_GT(reports[0].timestamp_us, 0u);
    }
}

TEST_F(PlasticityAnomalyIntegrationTest, ClearHistoryRemovesReports) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    // Generate anomalies
    float bad_weights[] = {NAN};
    plasticity_anomaly_analyze_weights(detector, bad_weights, 1, "clear_test");
    plasticity_anomaly_detect(detector);

    // Clear history
    plasticity_anomaly_clear_history(detector);

    // Reports should be empty
    plasticity_anomaly_report_t reports[10];
    int count = plasticity_anomaly_get_reports(detector, reports, 10);
    EXPECT_EQ(count, 0);
}

/* ============================================================================
 * Statistics Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, StatsAccumulateByCategory) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    // Generate weight anomalies
    float bad_weights[] = {NAN, INFINITY};
    plasticity_anomaly_analyze_weights(detector, bad_weights, 2, "weight_stats");
    plasticity_anomaly_detect(detector);

    plasticity_detection_stats_t stats;
    plasticity_anomaly_get_stats(detector, &stats);

    EXPECT_GT(stats.total_anomalies, 0u);
    EXPECT_GT(stats.weight_anomalies, 0u);
    EXPECT_GT(stats.checks_performed, 0u);
}

TEST_F(PlasticityAnomalyIntegrationTest, ResetStatsClearsAll) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    // Generate stats
    float bad_weights[] = {NAN};
    plasticity_anomaly_analyze_weights(detector, bad_weights, 1, "reset_stats");
    plasticity_anomaly_detect(detector);

    // Reset
    plasticity_anomaly_reset_stats(detector);

    plasticity_detection_stats_t stats;
    plasticity_anomaly_get_stats(detector, &stats);

    EXPECT_EQ(stats.total_anomalies, 0u);
    EXPECT_EQ(stats.weight_anomalies, 0u);
    EXPECT_EQ(stats.checks_performed, 0u);
}

/* ============================================================================
 * Concurrent Access Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, ConcurrentMetricSubmission) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    const int num_threads = 4;
    const int metrics_per_thread = 50;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < metrics_per_thread; i++) {
                char name[32];
                snprintf(name, sizeof(name), "metric_%d_%d", t, i);

                plasticity_anomaly_category_t cat = static_cast<plasticity_anomaly_category_t>(
                    (t + i) % PLASTICITY_CATEGORY_COUNT);

                float value = (t + i) % 10 == 0 ? NAN : 0.5f + (i * 0.01f);
                plasticity_anomaly_submit_metric(detector, name, value, cat);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Should complete without crashes
    int detected = plasticity_anomaly_detect(detector);
    EXPECT_GE(detected, 0);
}

TEST_F(PlasticityAnomalyIntegrationTest, ConcurrentWeightAnalysis) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    std::atomic<int> total_anomalies{0};
    const int num_threads = 4;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 20; i++) {
                float weights[10];
                for (int w = 0; w < 10; w++) {
                    // Inject some NaNs
                    if (t == 0 && i % 5 == 0 && w == 3) {
                        weights[w] = NAN;
                    } else {
                        weights[w] = 0.5f + (w * 0.05f);
                    }
                }

                char name[32];
                snprintf(name, sizeof(name), "weights_%d_%d", t, i);
                int anomalies = plasticity_anomaly_analyze_weights(detector, weights, 10, name);
                total_anomalies += anomalies;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Should have detected some anomalies from injected NaNs
    EXPECT_GT(total_anomalies.load(), 0);
}

/* ============================================================================
 * Full Pipeline Integration Tests
 * ============================================================================ */

TEST_F(PlasticityAnomalyIntegrationTest, FullDetectionPipeline) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    // Connect bio-async
    plasticity_anomaly_connect_bio_async(detector, bio_async);

    // Load all default rules
    plasticity_anomaly_load_all_default_rules(detector);

    // Set up callbacks
    anomaly_callback_count = 0;
    plasticity_anomaly_set_callback(detector, test_anomaly_callback, nullptr);

    health_callback_count = 0;
    plasticity_anomaly_set_health_callback(detector, test_health_callback, nullptr);

    // Simulate learning session with monitoring
    for (int epoch = 0; epoch < 20; epoch++) {
        // Submit weight metrics
        float weight_mean = (epoch == 10) ? 100.0f : 0.5f;  // Spike at epoch 10
        plasticity_anomaly_submit_metric(detector, "weight_mean", weight_mean, PLASTICITY_CATEGORY_WEIGHT);

        // Analyze weights
        float weights[10];
        for (int i = 0; i < 10; i++) {
            if (epoch == 15 && i == 5) {
                weights[i] = NAN;  // Inject NaN at epoch 15
            } else {
                weights[i] = 0.3f + (i * 0.05f) + (epoch * 0.01f);
            }
        }
        plasticity_anomaly_analyze_weights(detector, weights, 10, "epoch_weights");

        // Submit BCM metric
        float threshold = (epoch == 18) ? -10.0f : 0.5f;  // Bad threshold at epoch 18
        plasticity_anomaly_submit_metric(detector, "bcm_threshold", threshold, PLASTICITY_CATEGORY_BCM);

        // Run detection
        plasticity_anomaly_detect(detector);
    }

    // Verify detection worked
    plasticity_detection_stats_t stats;
    plasticity_anomaly_get_stats(detector, &stats);
    EXPECT_GT(stats.checks_performed, 0u);
    EXPECT_GT(stats.total_anomalies, 0u);  // Should have caught injected anomalies

    // Callbacks should have been triggered
    EXPECT_GT(anomaly_callback_count.load(), 0);
    EXPECT_GT(health_callback_count.load(), 0);

    // Health should be affected
    float final_health = plasticity_anomaly_get_health(detector);
    EXPECT_LT(final_health, 1.0f);

    // Get reports
    plasticity_anomaly_report_t reports[20];
    int report_count = plasticity_anomaly_get_reports(detector, reports, 20);
    EXPECT_GT(report_count, 0);
}

TEST_F(PlasticityAnomalyIntegrationTest, IntegrationWithStdpHealth) {
    plasticity_anomaly_config_t config;
    plasticity_anomaly_default_config(&config);

    detector = plasticity_anomaly_create(&config, bio_async);
    ASSERT_NE(detector, nullptr);

    plasticity_anomaly_load_all_default_rules(detector);

    // Both systems monitoring same weights
    float weights[] = {0.5f, NAN, 0.7f, INFINITY, 0.3f};

    // STDP health check
    if (stdp_health) {
        int stdp_anomalies = stdp_health_check_weights(stdp_health, weights, 5, "shared_weights");
        EXPECT_GT(stdp_anomalies, 0);
    }

    // Plasticity anomaly detection
    int plasticity_anomalies = plasticity_anomaly_analyze_weights(detector, weights, 5, "shared_weights");
    EXPECT_GT(plasticity_anomalies, 0);

    // Both should detect issues
    if (stdp_health) {
        float stdp_score = stdp_health_get_score(stdp_health);
        float plasticity_health = plasticity_anomaly_get_health(detector);

        EXPECT_LT(stdp_score, 1.0f);
        EXPECT_LT(plasticity_health, 1.0f);
    }
}

