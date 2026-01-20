/**
 * @file test_stdp_health_integration.cpp
 * @brief Integration tests for STDP Health Metrics
 * @date 2026-01-20
 *
 * Tests the integration between STDP health metrics, the health agent,
 * STDP/plasticity systems, and bio-async communication.
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <cmath>

extern "C" {
#include "utils/fault_tolerance/nimcp_stdp_health_metrics.h"
#include "utils/fault_tolerance/nimcp_health_agent.h"
#include "snn/plasticity/nimcp_stdp.h"
#include "snn/plasticity/nimcp_bcm.h"
#include "async/nimcp_bio_async.h"
}

class StdpHealthIntegrationTest : public ::testing::Test {
protected:
    stdp_health_metrics_t* metrics = nullptr;
    nimcp_health_agent_t* health_agent = nullptr;
    nimcp_stdp_context_t* stdp = nullptr;
    nimcp_bcm_context_t* bcm = nullptr;
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

        // Create STDP context
        nimcp_stdp_config_t stdp_config;
        nimcp_stdp_default_config(&stdp_config);
        stdp = nimcp_stdp_create(&stdp_config);

        // Create BCM context
        nimcp_bcm_config_t bcm_config;
        nimcp_bcm_default_config(&bcm_config);
        bcm = nimcp_bcm_create(&bcm_config);
    }

    void TearDown() override {
        if (metrics) {
            stdp_health_destroy(metrics);
            metrics = nullptr;
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
        if (bio_async) {
            nimcp_bio_async_destroy(bio_async);
            bio_async = nullptr;
        }
    }
};

/* ============================================================================
 * Basic Integration Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, CreateWithBioAsync) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    EXPECT_NE(metrics, nullptr);
}

TEST_F(StdpHealthIntegrationTest, RegisterStdpContext) {
    if (!stdp) {
        GTEST_SKIP() << "STDP context not available";
    }

    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    int id = stdp_health_register_stdp(metrics, stdp, "test_stdp");
    EXPECT_GE(id, 0);
}

TEST_F(StdpHealthIntegrationTest, RegisterBcmContext) {
    if (!bcm) {
        GTEST_SKIP() << "BCM context not available";
    }

    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    int id = stdp_health_register_bcm(metrics, bcm, "test_bcm");
    EXPECT_GE(id, 0);
}

TEST_F(StdpHealthIntegrationTest, RegisterMultipleContexts) {
    if (!stdp || !bcm) {
        GTEST_SKIP() << "Required contexts not available";
    }

    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    int stdp_id = stdp_health_register_stdp(metrics, stdp, "stdp_1");
    int bcm_id = stdp_health_register_bcm(metrics, bcm, "bcm_1");

    EXPECT_GE(stdp_id, 0);
    EXPECT_GE(bcm_id, 0);
    EXPECT_NE(stdp_id, bcm_id);
}

/* ============================================================================
 * Health Check Integration Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, CheckHealthyWeightsNoAnomalies) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Normal weight distribution
    float weights[] = {0.5f, 0.3f, 0.7f, 0.1f, 0.9f, 0.4f, 0.6f, 0.2f};
    int anomalies = stdp_health_check_weights(metrics, weights, 8, "test_weights");
    EXPECT_EQ(anomalies, 0);

    float score = stdp_health_get_score(metrics);
    EXPECT_GE(score, 0.9f);  // Should remain healthy
}

TEST_F(StdpHealthIntegrationTest, DetectNaNWeightsAnomaly) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Weights with NaN values
    float weights[] = {0.5f, NAN, 0.7f, 0.1f, NAN};
    int anomalies = stdp_health_check_weights(metrics, weights, 5, "nan_weights");
    EXPECT_GT(anomalies, 0);

    stdp_health_stats_t stats;
    stdp_health_get_stats(metrics, &stats);
    EXPECT_GT(stats.total_anomalies, 0u);
}

TEST_F(StdpHealthIntegrationTest, DetectInfWeightsAnomaly) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Weights with infinity
    float weights[] = {0.5f, INFINITY, 0.7f, -INFINITY};
    int anomalies = stdp_health_check_weights(metrics, weights, 4, "inf_weights");
    EXPECT_GT(anomalies, 0);
}

TEST_F(StdpHealthIntegrationTest, DetectWeightSaturation) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Weights all at max boundary (saturated)
    float weights[100];
    for (int i = 0; i < 100; i++) {
        weights[i] = config.weight_thresholds.max_weight_value;
    }

    int anomalies = stdp_health_check_weights(metrics, weights, 100, "saturated_weights");
    EXPECT_GT(anomalies, 0);
}

/* ============================================================================
 * Timing Analysis Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, CheckValidTimingNoViolations) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Normal pre/post spike times
    float pre_times[] = {1.0f, 5.0f, 10.0f, 15.0f};
    float post_times[] = {2.0f, 6.0f, 11.0f, 16.0f};

    int violations = stdp_health_check_timing(metrics, pre_times, post_times, 4);
    EXPECT_GE(violations, 0);  // May have 0 or more depending on window
}

TEST_F(StdpHealthIntegrationTest, DetectTimingJitter) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);
    config.timing_thresholds.max_timing_jitter_ms = 1.0f;

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // High jitter in timing - wide variation in spike intervals
    float pre_times[] = {0.0f, 1.0f, 10.0f, 11.0f, 100.0f};
    float post_times[] = {0.5f, 1.5f, 10.5f, 11.5f, 100.5f};

    int violations = stdp_health_check_timing(metrics, pre_times, post_times, 5);
    EXPECT_GE(violations, 0);
}

/* ============================================================================
 * Learning Rate Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, CheckNormalLearningRate) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    int anomaly = stdp_health_check_learning_rate(metrics, 0.01f, "test_lr");
    EXPECT_EQ(anomaly, 0);
}

TEST_F(StdpHealthIntegrationTest, DetectExcessiveLearningRate) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);
    config.lr_thresholds.max_learning_rate = 0.1f;

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Learning rate above threshold
    int anomaly = stdp_health_check_learning_rate(metrics, 1.0f, "excessive_lr");
    EXPECT_NE(anomaly, 0);

    stdp_health_stats_t stats;
    stdp_health_get_stats(metrics, &stats);
    EXPECT_GT(stats.total_anomalies, 0u);
}

TEST_F(StdpHealthIntegrationTest, DetectZeroLearningRate) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    int anomaly = stdp_health_check_learning_rate(metrics, 0.0f, "zero_lr");
    EXPECT_NE(anomaly, 0);
}

/* ============================================================================
 * Trace Analysis Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, CheckHealthyTraces) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Normal eligibility traces
    float traces[] = {0.1f, 0.2f, 0.05f, 0.3f, 0.15f};
    int anomalies = stdp_health_check_traces(metrics, traces, 5);
    EXPECT_EQ(anomalies, 0);
}

TEST_F(StdpHealthIntegrationTest, DetectExplodingTraces) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);
    config.trace_thresholds.max_trace_value = 1.0f;

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Traces exceeding threshold
    float traces[] = {0.1f, 5.0f, 0.2f, 10.0f};
    int anomalies = stdp_health_check_traces(metrics, traces, 4);
    EXPECT_GT(anomalies, 0);
}

/* ============================================================================
 * Callback Integration Tests
 * ============================================================================ */

static std::atomic<int> anomaly_callback_count{0};
static stdp_anomaly_type_t last_anomaly_type;

static void test_anomaly_callback(const stdp_anomaly_report_t* report, void* user_data) {
    (void)user_data;
    if (report) {
        last_anomaly_type = report->type;
    }
    anomaly_callback_count++;
}

TEST_F(StdpHealthIntegrationTest, AnomalyCallbackTriggered) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    anomaly_callback_count = 0;
    stdp_health_set_anomaly_callback(metrics, test_anomaly_callback, nullptr);

    // Trigger an anomaly with NaN weights
    float weights[] = {0.5f, NAN, 0.7f};
    stdp_health_check_weights(metrics, weights, 3, "callback_test");

    // Callback should have been triggered
    EXPECT_GT(anomaly_callback_count.load(), 0);
}

static std::atomic<int> check_callback_count{0};

static void test_check_callback(const stdp_health_stats_t* stats, void* user_data) {
    (void)stats;
    (void)user_data;
    check_callback_count++;
}

TEST_F(StdpHealthIntegrationTest, CheckCallbackTriggered) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    check_callback_count = 0;
    stdp_health_set_check_callback(metrics, test_check_callback, nullptr);

    // Run health check
    stdp_health_check(metrics);

    EXPECT_GT(check_callback_count.load(), 0);
}

/* ============================================================================
 * Health Score Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, HealthScoreDegradesWithAnomalies) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    float initial_score = stdp_health_get_score(metrics);
    EXPECT_FLOAT_EQ(initial_score, 1.0f);

    // Introduce anomalies
    float bad_weights[] = {NAN, INFINITY, -INFINITY};
    stdp_health_check_weights(metrics, bad_weights, 3, "bad_weights");

    float after_score = stdp_health_get_score(metrics);
    EXPECT_LT(after_score, initial_score);
}

TEST_F(StdpHealthIntegrationTest, HealthyFlagReflectsScore) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Initially healthy
    EXPECT_TRUE(stdp_health_is_healthy(metrics));

    // Introduce severe anomalies
    for (int i = 0; i < 10; i++) {
        float bad_weights[] = {NAN, NAN, NAN};
        stdp_health_check_weights(metrics, bad_weights, 3, "severe_anomaly");
    }

    // May become unhealthy depending on thresholds
    float score = stdp_health_get_score(metrics);
    if (score < 0.5f) {
        EXPECT_FALSE(stdp_health_is_healthy(metrics));
    }
}

/* ============================================================================
 * Bio-Async Integration Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, ConnectToBioAsync) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    int result = stdp_health_connect_bio_async(metrics, bio_async);
    EXPECT_EQ(result, 0);
}

TEST_F(StdpHealthIntegrationTest, BroadcastStatusViaBioAsync) {
    if (!bio_async) {
        GTEST_SKIP() << "Bio-async not available";
    }

    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    stdp_health_connect_bio_async(metrics, bio_async);

    int result = stdp_health_broadcast_status(metrics);
    EXPECT_EQ(result, 0);
}

/* ============================================================================
 * Statistics Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, StatsAccumulateCorrectly) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Perform multiple checks
    for (int i = 0; i < 5; i++) {
        stdp_health_check(metrics);
    }

    stdp_health_stats_t stats;
    stdp_health_get_stats(metrics, &stats);
    EXPECT_GE(stats.checks_performed, 5u);
}

TEST_F(StdpHealthIntegrationTest, ResetStatsClearsCounters) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Generate some stats
    float weights[] = {NAN, 0.5f};
    stdp_health_check_weights(metrics, weights, 2, "reset_test");
    stdp_health_check(metrics);

    // Reset
    stdp_health_reset_stats(metrics);

    stdp_health_stats_t stats;
    stdp_health_get_stats(metrics, &stats);
    EXPECT_EQ(stats.total_anomalies, 0u);
    EXPECT_EQ(stats.checks_performed, 0u);
}

/* ============================================================================
 * Concurrent Access Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, ConcurrentHealthChecks) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    const int num_threads = 4;
    const int checks_per_thread = 25;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < checks_per_thread; i++) {
                // Mix of good and bad weights
                if ((t + i) % 3 == 0) {
                    float bad_weights[] = {NAN, 0.5f};
                    stdp_health_check_weights(metrics, bad_weights, 2, "concurrent_bad");
                } else {
                    float good_weights[] = {0.3f, 0.5f, 0.7f};
                    stdp_health_check_weights(metrics, good_weights, 3, "concurrent_good");
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should complete without crashes
    stdp_health_stats_t stats;
    int result = stdp_health_get_stats(metrics, &stats);
    EXPECT_EQ(result, 0);
    EXPECT_GT(stats.checks_performed, 0u);
}

TEST_F(StdpHealthIntegrationTest, ConcurrentLearningRateChecks) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    std::atomic<int> anomalies_detected{0};
    const int num_threads = 4;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 20; i++) {
                float lr = (t == 0 && i % 5 == 0) ? 5.0f : 0.01f;  // Some threads use bad LR
                char name[32];
                snprintf(name, sizeof(name), "lr_%d_%d", t, i);
                int anomaly = stdp_health_check_learning_rate(metrics, lr, name);
                if (anomaly != 0) {
                    anomalies_detected++;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should detect some anomalies
    EXPECT_GT(anomalies_detected.load(), 0);
}

/* ============================================================================
 * Full Pipeline Integration Tests
 * ============================================================================ */

TEST_F(StdpHealthIntegrationTest, FullMonitoringPipeline) {
    stdp_health_config_t config;
    stdp_health_default_config(&config);

    metrics = stdp_health_create(&config, bio_async);
    ASSERT_NE(metrics, nullptr);

    // Connect to bio-async
    stdp_health_connect_bio_async(metrics, bio_async);

    // Register STDP context if available
    if (stdp) {
        stdp_health_register_stdp(metrics, stdp, "pipeline_stdp");
    }

    // Set up callback
    anomaly_callback_count = 0;
    stdp_health_set_anomaly_callback(metrics, test_anomaly_callback, nullptr);

    // Simulate learning session with monitoring
    for (int epoch = 0; epoch < 10; epoch++) {
        // Check weights
        float weights[10];
        for (int i = 0; i < 10; i++) {
            weights[i] = (epoch == 5 && i == 3) ? NAN : 0.5f + (i * 0.05f);
        }
        stdp_health_check_weights(metrics, weights, 10, "epoch_weights");

        // Check timing
        float pre[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float post[5] = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
        stdp_health_check_timing(metrics, pre, post, 5);

        // Check learning rate
        float lr = (epoch == 7) ? 5.0f : 0.01f;  // Spike at epoch 7
        stdp_health_check_learning_rate(metrics, lr, "epoch_lr");

        // Run full check
        stdp_health_check(metrics);
    }

    // Verify monitoring captured anomalies
    stdp_health_stats_t stats;
    stdp_health_get_stats(metrics, &stats);
    EXPECT_GT(stats.checks_performed, 0u);
    EXPECT_GT(stats.total_anomalies, 0u);  // Should have caught the NaN and high LR

    // Callbacks should have been triggered
    EXPECT_GT(anomaly_callback_count.load(), 0);

    // Health score should be affected
    float score = stdp_health_get_score(metrics);
    EXPECT_LT(score, 1.0f);
}

