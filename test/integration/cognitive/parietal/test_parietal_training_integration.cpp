/**
 * @file test_parietal_training_integration.cpp
 * @brief Integration tests for Parietal-Training Bridge
 * @date 2026-01-20
 *
 * Tests the integration between parietal lobe modules and the training
 * system, including plasticity connections and bio-async communication.
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

extern "C" {
#include "cognitive/parietal/nimcp_parietal_training_bridge.h"
#include "cognitive/parietal/nimcp_parietal_cortex.h"
#include "cognitive/parietal/nimcp_parietal_plasticity_bridge.h"
#include "training/nimcp_training.h"
#include "snn/plasticity/nimcp_stdp.h"
#include "async/nimcp_bio_async.h"
}

class ParietalTrainingIntegrationTest : public ::testing::Test {
protected:
    parietal_cortex_t* parietal = nullptr;
    parietal_training_bridge_t* bridge = nullptr;
    nimcp_training_ctx_t* training = nullptr;
    nimcp_bio_async_t* bio_async = nullptr;

    void SetUp() override {
        // Create bio-async for message passing
        nimcp_bio_async_config_t bio_config;
        nimcp_bio_async_default_config(&bio_config);
        bio_async = nimcp_bio_async_create(&bio_config);

        // Create parietal cortex
        parietal_cortex_config_t parietal_config;
        parietal_cortex_default_config(&parietal_config);
        parietal = parietal_cortex_create(&parietal_config, bio_async);

        // Create training context
        nimcp_training_config_t train_config;
        nimcp_training_default_config(&train_config);
        training = nimcp_training_create(&train_config);
    }

    void TearDown() override {
        if (bridge) {
            parietal_training_destroy(bridge);
            bridge = nullptr;
        }
        if (parietal) {
            parietal_cortex_destroy(parietal);
            parietal = nullptr;
        }
        if (training) {
            nimcp_training_destroy(training);
            training = nullptr;
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

TEST_F(ParietalTrainingIntegrationTest, CreateBridgeWithValidComponents) {
    if (!parietal) {
        GTEST_SKIP() << "Parietal cortex not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    EXPECT_NE(bridge, nullptr);
}

TEST_F(ParietalTrainingIntegrationTest, ConnectToTrainingSystem) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    int result = parietal_training_connect(bridge, training);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(parietal_training_is_connected(bridge));
}

TEST_F(ParietalTrainingIntegrationTest, DisconnectFromTrainingSystem) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);
    EXPECT_TRUE(parietal_training_is_connected(bridge));

    int result = parietal_training_disconnect(bridge);
    EXPECT_EQ(result, 0);
    EXPECT_FALSE(parietal_training_is_connected(bridge));
}

/* ============================================================================
 * Learning Signal Processing Tests
 * ============================================================================ */

static std::atomic<int> learning_callback_count{0};
static void test_learning_callback(const parietal_learning_signal_t* signal,
                                   parietal_train_response_t response,
                                   void* user_data) {
    (void)signal;
    (void)response;
    (void)user_data;
    learning_callback_count++;
}

TEST_F(ParietalTrainingIntegrationTest, ProcessLearningSignal) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    learning_callback_count = 0;
    parietal_training_set_learning_callback(bridge, test_learning_callback, nullptr);

    // Create and process a learning signal
    parietal_learning_signal_t signal = {};
    signal.domain = PARIETAL_DOMAIN_COORDINATE_TRANSFORM;
    signal.signal_strength = 0.8f;
    signal.error_gradient = 0.1f;
    signal.timestamp_us = 1000;

    parietal_train_response_t response = parietal_training_process_signal(bridge, &signal);
    EXPECT_NE(response, PARIETAL_TRAIN_RESPONSE_NONE);
    EXPECT_GT(learning_callback_count.load(), 0);
}

TEST_F(ParietalTrainingIntegrationTest, ProcessMultipleDomainsSequentially) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    // Process signals for multiple domains
    parietal_learning_domain_t domains[] = {
        PARIETAL_DOMAIN_COORDINATE_TRANSFORM,
        PARIETAL_DOMAIN_SPATIAL_ATTENTION,
        PARIETAL_DOMAIN_BODY_SCHEMA,
        PARIETAL_DOMAIN_MOTOR_PLANNING
    };

    for (auto domain : domains) {
        parietal_learning_signal_t signal = {};
        signal.domain = domain;
        signal.signal_strength = 0.5f;
        signal.error_gradient = 0.05f;

        parietal_train_response_t response = parietal_training_process_signal(bridge, &signal);
        EXPECT_NE(response, PARIETAL_TRAIN_RESPONSE_NONE)
            << "Failed for domain " << parietal_training_domain_name(domain);
    }

    // Verify stats updated
    parietal_training_stats_t stats;
    parietal_training_get_stats(bridge, &stats);
    EXPECT_GE(stats.signals_processed, 4u);
}

/* ============================================================================
 * Weight Update Tests
 * ============================================================================ */

TEST_F(ParietalTrainingIntegrationTest, UpdateWeightsForDomain) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    int result = parietal_training_update_weights(bridge,
        PARIETAL_DOMAIN_COORDINATE_TRANSFORM, 0.01f);
    EXPECT_EQ(result, 0);

    parietal_training_stats_t stats;
    parietal_training_get_stats(bridge, &stats);
    EXPECT_GT(stats.weight_updates, 0u);
}

TEST_F(ParietalTrainingIntegrationTest, BatchFlushAccumulatesUpdates) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    // Process multiple signals to accumulate updates
    for (int i = 0; i < 10; i++) {
        parietal_learning_signal_t signal = {};
        signal.domain = PARIETAL_DOMAIN_SPATIAL_ATTENTION;
        signal.signal_strength = 0.3f + (i * 0.05f);
        signal.error_gradient = 0.02f;

        parietal_training_process_signal(bridge, &signal);
    }

    // Flush batch
    int result = parietal_training_flush_batch(bridge);
    EXPECT_EQ(result, 0);

    parietal_training_stats_t stats;
    parietal_training_get_stats(bridge, &stats);
    EXPECT_GT(stats.batch_flushes, 0u);
}

/* ============================================================================
 * Domain Configuration Tests
 * ============================================================================ */

TEST_F(ParietalTrainingIntegrationTest, DisableDomainPreventsLearning) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    // Disable spatial attention domain
    parietal_training_set_domain_enabled(bridge, PARIETAL_DOMAIN_SPATIAL_ATTENTION, false);

    // Process signal for disabled domain
    parietal_learning_signal_t signal = {};
    signal.domain = PARIETAL_DOMAIN_SPATIAL_ATTENTION;
    signal.signal_strength = 0.8f;
    signal.error_gradient = 0.1f;

    parietal_train_response_t response = parietal_training_process_signal(bridge, &signal);
    EXPECT_EQ(response, PARIETAL_TRAIN_RESPONSE_NONE);
}

TEST_F(ParietalTrainingIntegrationTest, AdjustDomainLearningRate) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    // Adjust learning rate for motor planning
    int result = parietal_training_set_domain_lr(bridge,
        PARIETAL_DOMAIN_MOTOR_PLANNING, 0.05f);
    EXPECT_EQ(result, 0);

    // Process signal and verify it still works
    parietal_learning_signal_t signal = {};
    signal.domain = PARIETAL_DOMAIN_MOTOR_PLANNING;
    signal.signal_strength = 0.6f;
    signal.error_gradient = 0.08f;

    parietal_train_response_t response = parietal_training_process_signal(bridge, &signal);
    EXPECT_NE(response, PARIETAL_TRAIN_RESPONSE_NONE);
}

/* ============================================================================
 * Plasticity Connection Tests
 * ============================================================================ */

TEST_F(ParietalTrainingIntegrationTest, ConnectToPlasticityBridge) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);
    config.connect_to_plasticity = true;

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    // Get parietal plasticity bridge if available
    parietal_plasticity_bridge_t* plasticity = parietal_cortex_get_plasticity_bridge(parietal);
    if (plasticity) {
        int result = parietal_training_connect_plasticity(bridge, plasticity);
        EXPECT_EQ(result, 0);

        parietal_train_state_t state = parietal_training_get_state(bridge);
        EXPECT_EQ(state, PARIETAL_TRAIN_STATE_ACTIVE);
    }
}

/* ============================================================================
 * Bio-Async Integration Tests
 * ============================================================================ */

TEST_F(ParietalTrainingIntegrationTest, ConnectToBioAsync) {
    if (!parietal) {
        GTEST_SKIP() << "Parietal cortex not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    int result = parietal_training_connect_bio_async(bridge, bio_async);
    EXPECT_EQ(result, 0);
}

TEST_F(ParietalTrainingIntegrationTest, BroadcastLearningEventViaBioAsync) {
    if (!parietal || !bio_async) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect_bio_async(bridge, bio_async);

    if (training) {
        parietal_training_connect(bridge, training);

        // Process signal which should broadcast
        parietal_learning_signal_t signal = {};
        signal.domain = PARIETAL_DOMAIN_COORDINATE_TRANSFORM;
        signal.signal_strength = 0.9f;
        signal.error_gradient = 0.15f;

        parietal_training_process_signal(bridge, &signal);

        // Give time for async broadcast
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        parietal_training_stats_t stats;
        parietal_training_get_stats(bridge, &stats);
        EXPECT_GE(stats.signals_processed, 1u);
    }
}

/* ============================================================================
 * State Machine Tests
 * ============================================================================ */

TEST_F(ParietalTrainingIntegrationTest, StateTransitions) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    // Initial state should be IDLE
    parietal_train_state_t state = parietal_training_get_state(bridge);
    EXPECT_EQ(state, PARIETAL_TRAIN_STATE_IDLE);

    // Connect to training - state should become ACTIVE
    parietal_training_connect(bridge, training);
    state = parietal_training_get_state(bridge);
    EXPECT_EQ(state, PARIETAL_TRAIN_STATE_ACTIVE);

    // Disconnect - state should return to IDLE
    parietal_training_disconnect(bridge);
    state = parietal_training_get_state(bridge);
    EXPECT_EQ(state, PARIETAL_TRAIN_STATE_IDLE);
}

/* ============================================================================
 * Statistics and Monitoring Tests
 * ============================================================================ */

TEST_F(ParietalTrainingIntegrationTest, StatsAccumulateCorrectly) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    // Process several signals
    for (int i = 0; i < 5; i++) {
        parietal_learning_signal_t signal = {};
        signal.domain = static_cast<parietal_learning_domain_t>(i % PARIETAL_DOMAIN_COUNT);
        signal.signal_strength = 0.5f;
        signal.error_gradient = 0.05f;

        parietal_training_process_signal(bridge, &signal);
    }

    parietal_training_stats_t stats;
    int result = parietal_training_get_stats(bridge, &stats);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(stats.signals_processed, 5u);
}

TEST_F(ParietalTrainingIntegrationTest, ResetStatsClearsCounters) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    // Process some signals
    for (int i = 0; i < 3; i++) {
        parietal_learning_signal_t signal = {};
        signal.domain = PARIETAL_DOMAIN_BODY_SCHEMA;
        signal.signal_strength = 0.6f;

        parietal_training_process_signal(bridge, &signal);
    }

    // Reset stats
    parietal_training_reset_stats(bridge);

    parietal_training_stats_t stats;
    parietal_training_get_stats(bridge, &stats);
    EXPECT_EQ(stats.signals_processed, 0u);
    EXPECT_EQ(stats.weight_updates, 0u);
}

/* ============================================================================
 * Concurrent Access Tests
 * ============================================================================ */

TEST_F(ParietalTrainingIntegrationTest, ConcurrentSignalProcessing) {
    if (!parietal || !training) {
        GTEST_SKIP() << "Required components not available";
    }

    parietal_training_config_t config;
    parietal_training_default_config(&config);

    bridge = parietal_training_create(&config, parietal, bio_async);
    ASSERT_NE(bridge, nullptr);

    parietal_training_connect(bridge, training);

    std::atomic<int> processed_count{0};
    const int num_threads = 4;
    const int signals_per_thread = 25;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < signals_per_thread; i++) {
                parietal_learning_signal_t signal = {};
                signal.domain = static_cast<parietal_learning_domain_t>(
                    (t + i) % PARIETAL_DOMAIN_COUNT);
                signal.signal_strength = 0.3f + (i * 0.01f);
                signal.error_gradient = 0.02f;

                parietal_train_response_t response =
                    parietal_training_process_signal(bridge, &signal);
                if (response != PARIETAL_TRAIN_RESPONSE_NONE) {
                    processed_count++;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should have processed some signals without crashing
    EXPECT_GT(processed_count.load(), 0);

    parietal_training_stats_t stats;
    parietal_training_get_stats(bridge, &stats);
    EXPECT_GT(stats.signals_processed, 0u);
}

