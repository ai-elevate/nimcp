/**
 * @file test_plasticity_health_e2e.cpp
 * @brief End-to-end tests for plasticity health monitoring pipeline
 * @date 2026-01-20
 *
 * Tests the complete pipeline from parietal training signals through
 * STDP health metrics to plasticity anomaly detection and response.
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <cmath>

extern "C" {
#include "cognitive/parietal/nimcp_parietal_training_bridge.h"
#include "cognitive/parietal/nimcp_parietal_cortex.h"
#include "utils/fault_tolerance/nimcp_stdp_health_metrics.h"
#include "utils/fault_tolerance/nimcp_plasticity_anomaly_detection.h"
#include "utils/fault_tolerance/nimcp_health_agent.h"
#include "training/nimcp_training.h"
#include "snn/plasticity/nimcp_stdp.h"
#include "snn/plasticity/nimcp_bcm.h"
#include "async/nimcp_bio_async.h"
}

/**
 * E2E test fixture that creates a complete plasticity health monitoring
 * pipeline including all components from brain regions to health monitoring.
 */
class PlasticityHealthE2ETest : public ::testing::Test {
protected:
    // Bio-async messaging
    nimcp_bio_async_t* bio_async = nullptr;

    // Training components
    nimcp_training_ctx_t* training = nullptr;

    // Parietal components
    parietal_cortex_t* parietal = nullptr;
    parietal_training_bridge_t* parietal_bridge = nullptr;

    // Plasticity components
    nimcp_stdp_context_t* stdp = nullptr;
    nimcp_bcm_context_t* bcm = nullptr;

    // Health monitoring components
    nimcp_health_agent_t* health_agent = nullptr;
    stdp_health_metrics_t* stdp_health = nullptr;
    plasticity_anomaly_detector_t* anomaly_detector = nullptr;

    // Test state
    std::atomic<int> anomaly_count{0};
    std::atomic<int> health_update_count{0};
    std::atomic<float> last_health_score{1.0f};

    void SetUp() override {
        // Create bio-async messaging backbone
        nimcp_bio_async_config_t bio_config;
        nimcp_bio_async_default_config(&bio_config);
        bio_async = nimcp_bio_async_create(&bio_config);

        // Create health agent
        nimcp_health_agent_config_t ha_config;
        nimcp_health_agent_default_config(&ha_config);
        health_agent = nimcp_health_agent_create(&ha_config, bio_async);

        // Create training context
        nimcp_training_config_t train_config;
        nimcp_training_default_config(&train_config);
        training = nimcp_training_create(&train_config);

        // Create parietal cortex
        parietal_cortex_config_t parietal_config;
        parietal_cortex_default_config(&parietal_config);
        parietal = parietal_cortex_create(&parietal_config, bio_async);

        // Create plasticity contexts
        nimcp_stdp_config_t stdp_config;
        nimcp_stdp_default_config(&stdp_config);
        stdp = nimcp_stdp_create(&stdp_config);

        nimcp_bcm_config_t bcm_config;
        nimcp_bcm_default_config(&bcm_config);
        bcm = nimcp_bcm_create(&bcm_config);

        // Create STDP health metrics
        stdp_health_config_t sh_config;
        stdp_health_default_config(&sh_config);
        stdp_health = stdp_health_create(&sh_config, bio_async);

        // Create plasticity anomaly detector
        plasticity_anomaly_config_t pa_config;
        plasticity_anomaly_default_config(&pa_config);
        anomaly_detector = plasticity_anomaly_create(&pa_config, bio_async);
    }

    void TearDown() override {
        if (anomaly_detector) {
            plasticity_anomaly_destroy(anomaly_detector);
            anomaly_detector = nullptr;
        }
        if (stdp_health) {
            stdp_health_destroy(stdp_health);
            stdp_health = nullptr;
        }
        if (parietal_bridge) {
            parietal_training_destroy(parietal_bridge);
            parietal_bridge = nullptr;
        }
        if (bcm) {
            nimcp_bcm_destroy(bcm);
            bcm = nullptr;
        }
        if (stdp) {
            nimcp_stdp_destroy(stdp);
            stdp = nullptr;
        }
        if (parietal) {
            parietal_cortex_destroy(parietal);
            parietal = nullptr;
        }
        if (training) {
            nimcp_training_destroy(training);
            training = nullptr;
        }
        if (health_agent) {
            nimcp_health_agent_destroy(health_agent);
            health_agent = nullptr;
        }
        if (bio_async) {
            nimcp_bio_async_destroy(bio_async);
            bio_async = nullptr;
        }
    }

    bool setupFullPipeline() {
        if (!parietal || !training || !stdp || !bcm || !stdp_health || !anomaly_detector) {
            return false;
        }

        // Create parietal training bridge
        parietal_training_config_t pt_config;
        parietal_training_default_config(&pt_config);
        parietal_bridge = parietal_training_create(&pt_config, parietal, bio_async);
        if (!parietal_bridge) {
            return false;
        }

        // Connect parietal to training
        parietal_training_connect(parietal_bridge, training);

        // Register STDP with health metrics
        stdp_health_register_stdp(stdp_health, stdp, "main_stdp");
        stdp_health_register_bcm(stdp_health, bcm, "main_bcm");

        // Connect bio-async
        stdp_health_connect_bio_async(stdp_health, bio_async);
        plasticity_anomaly_connect_bio_async(anomaly_detector, bio_async);

        // Load default detection rules
        plasticity_anomaly_load_all_default_rules(anomaly_detector);

        return true;
    }

    // Callback for anomaly detection
    static void anomalyCallback(const plasticity_anomaly_report_t* report, void* user_data) {
        if (user_data && report) {
            auto* test = static_cast<PlasticityHealthE2ETest*>(user_data);
            test->anomaly_count++;
        }
    }

    // Callback for health updates
    static void healthCallback(float health_score, void* user_data) {
        if (user_data) {
            auto* test = static_cast<PlasticityHealthE2ETest*>(user_data);
            test->last_health_score = health_score;
            test->health_update_count++;
        }
    }
};

/* ============================================================================
 * Pipeline Setup E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, FullPipelineSetup) {
    bool setup_ok = setupFullPipeline();

    if (!parietal) {
        GTEST_SKIP() << "Parietal cortex not available";
    }

    EXPECT_TRUE(setup_ok);
    EXPECT_NE(parietal_bridge, nullptr);
    EXPECT_TRUE(parietal_training_is_connected(parietal_bridge));
}

/* ============================================================================
 * Healthy Operation E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, HealthyLearningSession) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    // Set up callbacks
    anomaly_count = 0;
    health_update_count = 0;
    plasticity_anomaly_set_callback(anomaly_detector, anomalyCallback, this);
    plasticity_anomaly_set_health_callback(anomaly_detector, healthCallback, this);

    // Simulate a healthy learning session
    for (int epoch = 0; epoch < 10; epoch++) {
        // Generate healthy learning signals
        parietal_learning_signal_t signal = {};
        signal.domain = static_cast<parietal_learning_domain_t>(epoch % PARIETAL_DOMAIN_COUNT);
        signal.signal_strength = 0.3f + (epoch * 0.05f);
        signal.error_gradient = 0.05f - (epoch * 0.004f);  // Decreasing error
        signal.timestamp_us = epoch * 1000;

        parietal_training_process_signal(parietal_bridge, &signal);

        // Check healthy weights with STDP health
        float weights[20];
        for (int i = 0; i < 20; i++) {
            weights[i] = 0.1f + (i * 0.04f) + (epoch * 0.01f);
        }
        stdp_health_check_weights(stdp_health, weights, 20, "session_weights");

        // Check with plasticity anomaly detector
        plasticity_anomaly_analyze_weights(anomaly_detector, weights, 20, "session_weights");

        // Submit metrics
        plasticity_anomaly_submit_metric(anomaly_detector, "weight_mean",
            0.5f + (epoch * 0.02f), PLASTICITY_CATEGORY_WEIGHT);
        plasticity_anomaly_submit_metric(anomaly_detector, "learning_rate",
            0.01f, PLASTICITY_CATEGORY_LEARNING);

        // Run detection
        plasticity_anomaly_detect(anomaly_detector);
    }

    // Verify healthy session had no anomalies
    EXPECT_EQ(anomaly_count.load(), 0);

    // Health should remain high
    float final_health = plasticity_anomaly_get_health(anomaly_detector);
    EXPECT_GE(final_health, 0.9f);

    float stdp_score = stdp_health_get_score(stdp_health);
    EXPECT_GE(stdp_score, 0.9f);
}

/* ============================================================================
 * Anomaly Detection E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, DetectWeightExplosionDuringLearning) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    anomaly_count = 0;
    plasticity_anomaly_set_callback(anomaly_detector, anomalyCallback, this);

    // Normal learning for first few epochs
    for (int epoch = 0; epoch < 5; epoch++) {
        float weights[10];
        for (int i = 0; i < 10; i++) {
            weights[i] = 0.5f + (i * 0.05f);
        }
        plasticity_anomaly_analyze_weights(anomaly_detector, weights, 10, "early_weights");
        plasticity_anomaly_detect(anomaly_detector);
    }

    // Inject weight explosion
    float exploding_weights[10];
    for (int i = 0; i < 10; i++) {
        exploding_weights[i] = 100.0f * (i + 1);  // Unrealistically large
    }
    plasticity_anomaly_analyze_weights(anomaly_detector, exploding_weights, 10, "exploding_weights");
    plasticity_anomaly_detect(anomaly_detector);

    // Should have detected anomaly
    EXPECT_GT(anomaly_count.load(), 0);

    // Health should decrease
    float health = plasticity_anomaly_get_health(anomaly_detector);
    EXPECT_LT(health, 1.0f);
}

TEST_F(PlasticityHealthE2ETest, DetectNaNCorruption) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    anomaly_count = 0;
    plasticity_anomaly_set_callback(anomaly_detector, anomalyCallback, this);

    // Inject NaN corruption (simulating numerical instability)
    float corrupted_weights[] = {0.5f, NAN, 0.7f, NAN, 0.3f};

    // Both systems should detect
    int stdp_anomalies = stdp_health_check_weights(stdp_health, corrupted_weights, 5, "corrupted");
    int plasticity_anomalies = plasticity_anomaly_analyze_weights(anomaly_detector,
        corrupted_weights, 5, "corrupted");
    plasticity_anomaly_detect(anomaly_detector);

    EXPECT_GT(stdp_anomalies, 0);
    EXPECT_GT(plasticity_anomalies, 0);
    EXPECT_GT(anomaly_count.load(), 0);
}

TEST_F(PlasticityHealthE2ETest, DetectLearningRateInstability) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    anomaly_count = 0;
    plasticity_anomaly_set_callback(anomaly_detector, anomalyCallback, this);

    // Simulate unstable learning rate
    float learning_rates[] = {0.01f, 0.5f, 0.001f, 10.0f, 0.0001f};  // Wild fluctuations

    for (int i = 0; i < 5; i++) {
        stdp_health_check_learning_rate(stdp_health, learning_rates[i], "unstable_lr");
        plasticity_anomaly_submit_metric(anomaly_detector, "learning_rate",
            learning_rates[i], PLASTICITY_CATEGORY_LEARNING);
    }

    plasticity_anomaly_detect(anomaly_detector);

    // Should detect high learning rate
    plasticity_detection_stats_t stats;
    plasticity_anomaly_get_stats(anomaly_detector, &stats);
    EXPECT_GT(stats.total_anomalies, 0u);
}

/* ============================================================================
 * Multi-Domain Learning E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, MultiDomainLearningWithMonitoring) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    anomaly_count = 0;
    plasticity_anomaly_set_callback(anomaly_detector, anomalyCallback, this);

    // Process signals for all parietal domains
    for (int domain = 0; domain < PARIETAL_DOMAIN_COUNT; domain++) {
        // Send learning signal
        parietal_learning_signal_t signal = {};
        signal.domain = static_cast<parietal_learning_domain_t>(domain);
        signal.signal_strength = 0.5f;
        signal.error_gradient = 0.08f;
        signal.timestamp_us = domain * 100;

        parietal_training_process_signal(parietal_bridge, &signal);

        // Generate domain-specific weights
        float weights[15];
        for (int i = 0; i < 15; i++) {
            weights[i] = 0.2f + (domain * 0.1f) + (i * 0.03f);
        }

        char weight_name[64];
        snprintf(weight_name, sizeof(weight_name), "%s_weights",
                 parietal_training_domain_name(static_cast<parietal_learning_domain_t>(domain)));

        stdp_health_check_weights(stdp_health, weights, 15, weight_name);
        plasticity_anomaly_analyze_weights(anomaly_detector, weights, 15, weight_name);
    }

    // Run detection cycle
    stdp_health_check(stdp_health);
    plasticity_anomaly_detect(anomaly_detector);

    // All domains should have healthy weights (no anomalies)
    EXPECT_EQ(anomaly_count.load(), 0);

    // Get stats
    parietal_training_stats_t parietal_stats;
    parietal_training_get_stats(parietal_bridge, &parietal_stats);
    EXPECT_EQ(parietal_stats.signals_processed, static_cast<uint32_t>(PARIETAL_DOMAIN_COUNT));
}

/* ============================================================================
 * Recovery E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, HealthRecoveryAfterAnomaly) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    // First cause an anomaly
    float bad_weights[] = {NAN, INFINITY, 100.0f};
    plasticity_anomaly_analyze_weights(anomaly_detector, bad_weights, 3, "bad");
    plasticity_anomaly_detect(anomaly_detector);

    float health_after_anomaly = plasticity_anomaly_get_health(anomaly_detector);
    EXPECT_LT(health_after_anomaly, 1.0f);

    // Now clear history and reset
    plasticity_anomaly_clear_history(anomaly_detector);
    plasticity_anomaly_reset_stats(anomaly_detector);

    // Process healthy data
    for (int i = 0; i < 10; i++) {
        float good_weights[] = {0.3f, 0.5f, 0.7f, 0.4f, 0.6f};
        plasticity_anomaly_analyze_weights(anomaly_detector, good_weights, 5, "recovery");
        plasticity_anomaly_detect(anomaly_detector);
    }

    // Health should be restored
    float recovered_health = plasticity_anomaly_get_health(anomaly_detector);
    EXPECT_GT(recovered_health, health_after_anomaly);
}

/* ============================================================================
 * Concurrent Processing E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, ConcurrentLearningAndMonitoring) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    std::atomic<bool> running{true};
    std::atomic<int> learning_iterations{0};
    std::atomic<int> monitoring_iterations{0};

    // Learning thread - processes parietal signals
    std::thread learning_thread([&]() {
        while (running && learning_iterations < 50) {
            parietal_learning_signal_t signal = {};
            signal.domain = static_cast<parietal_learning_domain_t>(
                learning_iterations % PARIETAL_DOMAIN_COUNT);
            signal.signal_strength = 0.5f;
            signal.error_gradient = 0.05f;

            parietal_training_process_signal(parietal_bridge, &signal);
            learning_iterations++;

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Monitoring thread - runs health checks
    std::thread monitoring_thread([&]() {
        while (running && monitoring_iterations < 50) {
            float weights[10];
            for (int i = 0; i < 10; i++) {
                weights[i] = 0.1f + (monitoring_iterations % 10) * 0.08f;
            }

            stdp_health_check_weights(stdp_health, weights, 10, "concurrent");
            plasticity_anomaly_analyze_weights(anomaly_detector, weights, 10, "concurrent");
            plasticity_anomaly_detect(anomaly_detector);

            monitoring_iterations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Let threads run
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    running = false;

    learning_thread.join();
    monitoring_thread.join();

    // Both threads should have completed work
    EXPECT_GT(learning_iterations.load(), 0);
    EXPECT_GT(monitoring_iterations.load(), 0);

    // Stats should reflect the work
    parietal_training_stats_t parietal_stats;
    parietal_training_get_stats(parietal_bridge, &parietal_stats);
    EXPECT_GT(parietal_stats.signals_processed, 0u);

    plasticity_detection_stats_t anomaly_stats;
    plasticity_anomaly_get_stats(anomaly_detector, &anomaly_stats);
    EXPECT_GT(anomaly_stats.checks_performed, 0u);
}

/* ============================================================================
 * Full Pipeline Flow E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, CompleteTrainingSessionWithMonitoring) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    // Set up all callbacks
    anomaly_count = 0;
    health_update_count = 0;
    last_health_score = 1.0f;

    plasticity_anomaly_set_callback(anomaly_detector, anomalyCallback, this);
    plasticity_anomaly_set_health_callback(anomaly_detector, healthCallback, this);

    // Simulate complete training session with epochs
    const int num_epochs = 20;
    const int samples_per_epoch = 10;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Process samples in epoch
        for (int sample = 0; sample < samples_per_epoch; sample++) {
            // Generate learning signal
            parietal_learning_signal_t signal = {};
            signal.domain = static_cast<parietal_learning_domain_t>(
                (epoch + sample) % PARIETAL_DOMAIN_COUNT);
            signal.signal_strength = 0.3f + (0.5f * sample / samples_per_epoch);
            signal.error_gradient = 0.1f * (1.0f - (float)epoch / num_epochs);  // Decreasing error
            signal.timestamp_us = (epoch * samples_per_epoch + sample) * 100;

            parietal_training_process_signal(parietal_bridge, &signal);

            // Generate weights based on learning progress
            float weights[15];
            for (int i = 0; i < 15; i++) {
                // Weights converge over epochs
                float base = 0.5f + (0.3f * sample / samples_per_epoch);
                float convergence = 1.0f - (0.5f * epoch / num_epochs);
                weights[i] = base * convergence + (i * 0.02f);

                // Inject anomaly at epoch 12
                if (epoch == 12 && sample == 5 && i == 7) {
                    weights[i] = NAN;
                }
            }

            // Monitor weights
            stdp_health_check_weights(stdp_health, weights, 15, "training_weights");
            plasticity_anomaly_analyze_weights(anomaly_detector, weights, 15, "training_weights");
        }

        // End of epoch monitoring
        plasticity_anomaly_submit_metric(anomaly_detector, "epoch_loss",
            0.5f * (1.0f - (float)epoch / num_epochs), PLASTICITY_CATEGORY_LEARNING);

        // Run detection
        stdp_health_check(stdp_health);
        plasticity_anomaly_detect(anomaly_detector);
    }

    // Verify training completed
    parietal_training_stats_t parietal_stats;
    parietal_training_get_stats(parietal_bridge, &parietal_stats);
    EXPECT_EQ(parietal_stats.signals_processed,
              static_cast<uint32_t>(num_epochs * samples_per_epoch));

    // Verify monitoring worked
    stdp_health_stats_t health_stats;
    stdp_health_get_stats(stdp_health, &health_stats);
    EXPECT_GT(health_stats.checks_performed, 0u);

    plasticity_detection_stats_t anomaly_stats;
    plasticity_anomaly_get_stats(anomaly_detector, &anomaly_stats);
    EXPECT_GT(anomaly_stats.checks_performed, 0u);

    // Should have detected the injected NaN anomaly
    EXPECT_GT(anomaly_stats.weight_anomalies, 0u);
    EXPECT_GT(anomaly_count.load(), 0);

    // Health should have recovered (mostly)
    float final_health = plasticity_anomaly_get_health(anomaly_detector);
    EXPECT_GT(final_health, 0.0f);
}

/* ============================================================================
 * Report Generation E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, GenerateAnomalyReport) {
    if (!setupFullPipeline()) {
        GTEST_SKIP() << "Pipeline setup failed";
    }

    // Generate multiple anomalies
    float bad_weights1[] = {NAN, 0.5f, 0.3f};
    float bad_weights2[] = {INFINITY, 0.5f, 0.3f};
    float bad_weights3[] = {100.0f, 200.0f, 300.0f};

    plasticity_anomaly_analyze_weights(anomaly_detector, bad_weights1, 3, "nan_test");
    plasticity_anomaly_analyze_weights(anomaly_detector, bad_weights2, 3, "inf_test");
    plasticity_anomaly_analyze_weights(anomaly_detector, bad_weights3, 3, "explosion_test");
    plasticity_anomaly_detect(anomaly_detector);

    // Get reports
    plasticity_anomaly_report_t reports[10];
    int count = plasticity_anomaly_get_reports(anomaly_detector, reports, 10);

    EXPECT_GT(count, 0);

    // Verify report contents
    for (int i = 0; i < count; i++) {
        EXPECT_NE(reports[i].anomaly, PLASTICITY_ANOMALY_NONE);
        EXPECT_GT(reports[i].timestamp_us, 0u);
        EXPECT_GE(reports[i].confidence, 0.0f);
        EXPECT_LE(reports[i].confidence, 1.0f);
    }
}

/* ============================================================================
 * Bio-Async Communication E2E Tests
 * ============================================================================ */

TEST_F(PlasticityHealthE2ETest, BioAsyncBroadcastOnAnomaly) {
    if (!setupFullPipeline() || !bio_async) {
        GTEST_SKIP() << "Pipeline or bio-async not available";
    }

    // Ensure bio-async connections
    stdp_health_connect_bio_async(stdp_health, bio_async);
    plasticity_anomaly_connect_bio_async(anomaly_detector, bio_async);

    // Generate anomaly
    float bad_weights[] = {NAN, 0.5f};
    plasticity_anomaly_analyze_weights(anomaly_detector, bad_weights, 2, "broadcast_test");
    plasticity_anomaly_detect(anomaly_detector);

    // Broadcast status
    int broadcast_result = plasticity_anomaly_broadcast(anomaly_detector);
    EXPECT_EQ(broadcast_result, 0);

    int stdp_broadcast = stdp_health_broadcast_status(stdp_health);
    EXPECT_EQ(stdp_broadcast, 0);

    // Give time for async processing
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

