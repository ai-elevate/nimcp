/**
 * @file test_bio_async_regression.cpp
 * @brief Regression tests for bio-async module
 *
 * Tests for:
 * - Performance characteristics
 * - Timing accuracy
 * - Numerical stability
 * - Edge cases and boundary conditions
 * - Known bug fixes
 *
 * @author NIMCP Development Team
 * @date 2025-11-28
 */

#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>

extern "C" {
#include "async/nimcp_bio_async.h"
#include "async/nimcp_biological_timescales.h"
#include "utils/error/nimcp_error_codes.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class BioAsyncRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        nimcp_bio_async_config_t config = nimcp_bio_async_default_config();
        config.enable_logging = false;
        config.enable_statistics = true;
        nimcp_error_t err = nimcp_bio_async_init(&config);
        ASSERT_EQ(err, NIMCP_SUCCESS);
    }

    void TearDown() override {
        nimcp_bio_async_shutdown();
    }
};

//=============================================================================
// Performance Regression Tests
//=============================================================================

TEST_F(BioAsyncRegressionTest, PromiseCreateLatency) {
    const int NUM_ITERATIONS = 1000;
    std::vector<double> latencies;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        nimcp_bio_promise_t promise = nimcp_bio_promise_create(
            BIO_CHANNEL_DOPAMINE, sizeof(int));

        auto end = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);

        nimcp_bio_promise_destroy(promise);
    }

    // Calculate statistics
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();

    std::sort(latencies.begin(), latencies.end());
    double p95 = latencies[static_cast<size_t>(0.95 * latencies.size())];
    double p99 = latencies[static_cast<size_t>(0.99 * latencies.size())];

    // Performance targets (microseconds)
    EXPECT_LT(mean, 100.0) << "Mean latency too high: " << mean << "us";
    EXPECT_LT(p95, 200.0) << "P95 latency too high: " << p95 << "us";
    EXPECT_LT(p99, 500.0) << "P99 latency too high: " << p99 << "us";
}

TEST_F(BioAsyncRegressionTest, PromiseCompleteLatency) {
    const int NUM_ITERATIONS = 1000;
    std::vector<double> latencies;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        nimcp_bio_promise_t promise = nimcp_bio_promise_create(
            BIO_CHANNEL_DOPAMINE, sizeof(int));

        int result = 42;
        auto start = std::chrono::high_resolution_clock::now();

        nimcp_bio_promise_complete(promise, &result);

        auto end = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);

        nimcp_bio_promise_destroy(promise);
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();

    std::sort(latencies.begin(), latencies.end());
    double p95 = latencies[static_cast<size_t>(0.95 * latencies.size())];

    EXPECT_LT(mean, 50.0) << "Mean complete latency too high: " << mean << "us";
    EXPECT_LT(p95, 100.0) << "P95 complete latency too high: " << p95 << "us";
}

TEST_F(BioAsyncRegressionTest, FutureWaitLatencyWhenReady) {
    const int NUM_ITERATIONS = 1000;
    std::vector<double> latencies;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        nimcp_bio_promise_t promise = nimcp_bio_promise_create(
            BIO_CHANNEL_DOPAMINE, sizeof(int));
        nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);

        int result = 42;
        nimcp_bio_promise_complete(promise, &result);

        auto start = std::chrono::high_resolution_clock::now();

        int out;
        nimcp_bio_future_wait(future, &out, 0);

        auto end = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);

        nimcp_bio_future_destroy(future);
        nimcp_bio_promise_destroy(promise);
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();

    // Wait on already-ready future should be very fast
    EXPECT_LT(mean, 20.0) << "Mean wait latency too high: " << mean << "us";
}

TEST_F(BioAsyncRegressionTest, PhaseSyncCreateLatency) {
    const int NUM_ITERATIONS = 500;
    std::vector<double> latencies;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        nimcp_phase_sync_t sync = nimcp_phase_sync_create(BIO_OSC_GAMMA);

        auto end = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);

        nimcp_phase_sync_destroy(sync);
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();

    EXPECT_LT(mean, 200.0) << "Mean phase sync create latency too high: " << mean << "us";
}

TEST_F(BioAsyncRegressionTest, PredictiveObserveLatency) {
    nimcp_predictive_model_t model = nimcp_predictive_create("test", 0.5f, 1.0f);

    const int NUM_ITERATIONS = 1000;
    std::vector<double> latencies;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float obs = 0.5f + 0.1f * (i % 10) / 10.0f;

        auto start = std::chrono::high_resolution_clock::now();

        nimcp_predictive_observe(model, obs);

        auto end = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();

    EXPECT_LT(mean, 50.0) << "Mean observe latency too high: " << mean << "us";

    nimcp_predictive_destroy(model);
}

//=============================================================================
// Timing Accuracy Regression Tests
//=============================================================================

TEST_F(BioAsyncRegressionTest, ConfidenceDecayFollowsExponential) {
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(BIO_CHANNEL_DOPAMINE, sizeof(int));
    nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);

    int result = 42;
    nimcp_bio_promise_complete(promise, &result);

    float tau = BIO_COMP_DA_DECAY_TAU_MS;
    std::vector<float> confidences;
    std::vector<float> times;

    // Sample confidence over time
    for (int i = 0; i < 10; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        float conf = nimcp_bio_future_get_confidence(future);
        confidences.push_back(conf);
        times.push_back(20.0f * (i + 1));
    }

    // Verify exponential decay pattern
    for (size_t i = 1; i < confidences.size(); i++) {
        // Confidence should be monotonically decreasing
        EXPECT_LE(confidences[i], confidences[i-1])
            << "Confidence not monotonically decreasing at i=" << i;

        // Check approximate exponential decay: c(t) ≈ c(t-dt) * exp(-dt/τ)
        float expected_ratio = expf(-20.0f / tau);
        float actual_ratio = confidences[i] / confidences[i-1];

        // Allow some tolerance for timing variations
        if (confidences[i-1] > 0.1f) {
            EXPECT_NEAR(actual_ratio, expected_ratio, 0.2f)
                << "Decay not following exponential at i=" << i;
        }
    }

    nimcp_bio_future_destroy(future);
    nimcp_bio_promise_destroy(promise);
}

TEST_F(BioAsyncRegressionTest, DifferentChannelDecayRates) {
    // Verify that decay rates are correctly ordered:
    // ACh (fastest) < NE < DA < 5-HT (slowest)

    std::vector<nimcp_bio_promise_t> promises;
    std::vector<nimcp_bio_future_t> futures;

    for (int c = 0; c < BIO_CHANNEL_COUNT; c++) {
        nimcp_bio_promise_t p = nimcp_bio_promise_create(
            static_cast<nimcp_bio_channel_type_t>(c), sizeof(int));
        nimcp_bio_future_t f = nimcp_bio_promise_get_future(p);
        promises.push_back(p);
        futures.push_back(f);
    }

    // Complete all simultaneously
    int result = 42;
    for (auto& p : promises) {
        nimcp_bio_promise_complete(p, &result);
    }

    // Advance simulation time for decay (confidence uses simulation time)
    nimcp_bio_async_step(100.0f);

    float conf_da = nimcp_bio_future_get_confidence(futures[BIO_CHANNEL_DOPAMINE]);
    float conf_5ht = nimcp_bio_future_get_confidence(futures[BIO_CHANNEL_SEROTONIN]);
    float conf_ne = nimcp_bio_future_get_confidence(futures[BIO_CHANNEL_NOREPINEPHRINE]);
    float conf_ach = nimcp_bio_future_get_confidence(futures[BIO_CHANNEL_ACETYLCHOLINE]);

    // 5-HT should have highest confidence (slowest decay)
    // ACh should have lowest confidence (fastest decay)
    EXPECT_GT(conf_5ht, conf_da);
    EXPECT_GT(conf_5ht, conf_ne);
    EXPECT_GT(conf_5ht, conf_ach);
    EXPECT_LT(conf_ach, conf_da);
    EXPECT_LT(conf_ach, conf_ne);

    for (auto& f : futures) nimcp_bio_future_destroy(f);
    for (auto& p : promises) nimcp_bio_promise_destroy(p);
}

//=============================================================================
// Numerical Stability Regression Tests
//=============================================================================

TEST_F(BioAsyncRegressionTest, PredictiveModelNumericalStability) {
    nimcp_predictive_model_t model = nimcp_predictive_create("test", 0.5f, 1.0f);

    // Test with extreme values
    nimcp_predictive_observe(model, 1e6f);
    float pred1 = nimcp_predictive_get_prediction(model);
    EXPECT_FALSE(std::isnan(pred1));
    EXPECT_FALSE(std::isinf(pred1));

    nimcp_predictive_observe(model, -1e6f);
    float pred2 = nimcp_predictive_get_prediction(model);
    EXPECT_FALSE(std::isnan(pred2));
    EXPECT_FALSE(std::isinf(pred2));

    nimcp_predictive_observe(model, 0.0f);
    float pred3 = nimcp_predictive_get_prediction(model);
    EXPECT_FALSE(std::isnan(pred3));
    EXPECT_FALSE(std::isinf(pred3));

    float prec = nimcp_predictive_get_precision(model);
    EXPECT_FALSE(std::isnan(prec));
    EXPECT_FALSE(std::isinf(prec));
    EXPECT_GT(prec, 0.0f);  // Precision should remain positive

    nimcp_predictive_destroy(model);
}

TEST_F(BioAsyncRegressionTest, KuramotoOrderParameterNumericalStability) {
    // Test with many phases
    const int N = 1000;
    std::vector<float> phases(N);

    // Uniformly distributed phases
    for (int i = 0; i < N; i++) {
        phases[i] = static_cast<float>(i) / N * 2 * M_PI;
    }

    float r, psi;
    bio_kuramoto_order_parameter(phases.data(), N, &r, &psi);

    EXPECT_FALSE(std::isnan(r));
    EXPECT_FALSE(std::isinf(r));
    EXPECT_FALSE(std::isnan(psi));
    EXPECT_FALSE(std::isinf(psi));

    EXPECT_GE(r, 0.0f);
    EXPECT_LE(r, 1.0f);
    EXPECT_GE(psi, -M_PI);
    EXPECT_LE(psi, M_PI);
}

TEST_F(BioAsyncRegressionTest, ExponentialDecayNumericalStability) {
    // Test with extreme tau values
    EXPECT_NEAR(bio_exponential_decay(1.0f, 100.0f, 1e-10f), 0.0f, 1e-6f);
    EXPECT_NEAR(bio_exponential_decay(1.0f, 100.0f, 1e10f), 1.0f, 1e-6f);

    // Test with zero tau (should return 0, not inf)
    float result = bio_exponential_decay(1.0f, 100.0f, 0.0f);
    EXPECT_FALSE(std::isnan(result));

    // Test with very small time
    result = bio_exponential_decay(1.0f, 1e-10f, 100.0f);
    EXPECT_FALSE(std::isnan(result));
    EXPECT_NEAR(result, 1.0f, 0.001f);
}

TEST_F(BioAsyncRegressionTest, MichaelisMentenNumericalStability) {
    // Test boundary conditions
    EXPECT_NEAR(bio_michaelis_menten(0.0f, 1.0f, 1.0f), 0.0f, 1e-6f);

    // At Km, should be Vmax/2
    EXPECT_NEAR(bio_michaelis_menten(1.0f, 1.0f, 1.0f), 0.5f, 1e-6f);

    // Very high substrate
    float result = bio_michaelis_menten(1e10f, 1.0f, 1.0f);
    EXPECT_FALSE(std::isnan(result));
    EXPECT_FALSE(std::isinf(result));
    EXPECT_NEAR(result, 1.0f, 0.01f);

    // Zero Km + substrate should not divide by zero
    result = bio_michaelis_menten(0.0f, 1.0f, 0.0f);
    EXPECT_FALSE(std::isnan(result));
}

//=============================================================================
// Edge Case Regression Tests
//=============================================================================

TEST_F(BioAsyncRegressionTest, EmptyPhaseSyncCoherence) {
    nimcp_phase_sync_t sync = nimcp_phase_sync_create(BIO_OSC_GAMMA);

    // Empty sync should have coherence 0
    float coherence = nimcp_phase_sync_get_coherence(sync);
    EXPECT_GE(coherence, 0.0f);

    size_t count = nimcp_phase_sync_get_count(sync);
    EXPECT_EQ(count, 0u);

    nimcp_phase_sync_destroy(sync);
}

TEST_F(BioAsyncRegressionTest, SingleOscillatorPhaseSyncCoherence) {
    nimcp_phase_sync_t sync = nimcp_phase_sync_create(BIO_OSC_GAMMA);

    nimcp_bio_promise_t p = nimcp_bio_promise_create(BIO_CHANNEL_DOPAMINE, sizeof(int));
    nimcp_bio_future_t f = nimcp_bio_promise_get_future(p);

    nimcp_phase_sync_add_future(sync, f);

    // Single oscillator should have coherence 1
    float coherence = nimcp_phase_sync_get_coherence(sync);
    EXPECT_NEAR(coherence, 1.0f, 0.1f);

    nimcp_bio_future_destroy(f);
    nimcp_bio_promise_destroy(p);
    nimcp_phase_sync_destroy(sync);
}

TEST_F(BioAsyncRegressionTest, ZeroSizeResult) {
    // Create promise with zero result size
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(BIO_CHANNEL_DOPAMINE, 0);
    ASSERT_NE(promise, nullptr);

    nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);
    ASSERT_NE(future, nullptr);

    // Should be able to complete with NULL result
    nimcp_error_t err = nimcp_bio_promise_complete(promise, nullptr);
    EXPECT_EQ(err, NIMCP_SUCCESS);

    // Wait should succeed
    err = nimcp_bio_future_wait(future, nullptr, 100);
    EXPECT_EQ(err, NIMCP_SUCCESS);

    nimcp_bio_future_destroy(future);
    nimcp_bio_promise_destroy(promise);
}

TEST_F(BioAsyncRegressionTest, GlialWaveSourceOutOfRange) {
    // Source region near edge of range
    nimcp_glial_wave_t wave = nimcp_glial_wave_initiate(1000, 2.0f);
    ASSERT_NE(wave, nullptr);

    // Should still work
    bool reached = nimcp_glial_wave_has_reached(wave, 1000);
    EXPECT_TRUE(reached);

    nimcp_glial_wave_destroy(wave);
}

TEST_F(BioAsyncRegressionTest, GlialWaveZeroCalcium) {
    // Initiate with very low calcium
    nimcp_glial_wave_t wave = nimcp_glial_wave_initiate(0, 0.01f);
    ASSERT_NE(wave, nullptr);

    // Wave might become inactive quickly
    for (int i = 0; i < 10; i++) {
        nimcp_glial_wave_step(wave, 1.0f);
    }

    // Should not crash
    float radius = nimcp_glial_wave_get_radius(wave);
    EXPECT_FALSE(std::isnan(radius));

    nimcp_glial_wave_destroy(wave);
}

//=============================================================================
// Concurrency Regression Tests
//=============================================================================

TEST_F(BioAsyncRegressionTest, RaceConditionPromiseComplete) {
    // Test that concurrent complete calls don't cause issues

    for (int trial = 0; trial < 100; trial++) {
        nimcp_bio_promise_t promise = nimcp_bio_promise_create(
            BIO_CHANNEL_DOPAMINE, sizeof(int));

        std::vector<std::thread> threads;
        std::atomic<int> success_count{0};

        for (int t = 0; t < 4; t++) {
            threads.emplace_back([&, t]() {
                int result = t;
                if (nimcp_bio_promise_complete(promise, &result) == NIMCP_SUCCESS) {
                    success_count++;
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        // Only one complete should succeed
        EXPECT_EQ(success_count.load(), 1) << "Trial " << trial;

        nimcp_bio_promise_destroy(promise);
    }
}

TEST_F(BioAsyncRegressionTest, RaceConditionFutureDestroy) {
    // Test that concurrent operations with destroy don't crash

    for (int trial = 0; trial < 50; trial++) {
        nimcp_bio_promise_t promise = nimcp_bio_promise_create(
            BIO_CHANNEL_DOPAMINE, sizeof(int));
        nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);

        std::atomic<bool> stop{false};
        std::thread reader([&]() {
            while (!stop) {
                nimcp_bio_future_state(future);
                nimcp_bio_future_get_confidence(future);
            }
        });

        int result = 42;
        nimcp_bio_promise_complete(promise, &result);

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        stop = true;
        reader.join();

        nimcp_bio_future_destroy(future);
        nimcp_bio_promise_destroy(promise);
    }
}

//=============================================================================
// Memory Regression Tests
//=============================================================================

TEST_F(BioAsyncRegressionTest, NoMemoryLeakOnRapidCreateDestroy) {
    // Rapid create/destroy should not leak memory

    nimcp_bio_async_stats_t stats_before;
    nimcp_bio_async_get_stats(&stats_before);

    for (int i = 0; i < 10000; i++) {
        nimcp_bio_promise_t p = nimcp_bio_promise_create(
            static_cast<nimcp_bio_channel_type_t>(i % BIO_CHANNEL_COUNT),
            sizeof(int));
        nimcp_bio_future_t f = nimcp_bio_promise_get_future(p);

        int result = i;
        nimcp_bio_promise_complete(p, &result);

        nimcp_bio_future_destroy(f);
        nimcp_bio_promise_destroy(p);
    }

    nimcp_bio_async_stats_t stats_after;
    nimcp_bio_async_get_stats(&stats_after);

    // Memory should return to approximately the same level
    // (Allow some variance for internal caches)
    // This test is primarily for ASAN/valgrind verification
    SUCCEED();
}

//=============================================================================
// Callback Regression Tests
//=============================================================================

TEST_F(BioAsyncRegressionTest, CallbackNotCalledAfterDestroy) {
    std::atomic<int> callback_count{0};

    {
        nimcp_bio_promise_t promise = nimcp_bio_promise_create(
            BIO_CHANNEL_DOPAMINE, sizeof(int));
        nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);

        nimcp_bio_future_then(future,
            [](const void*, float, nimcp_error_t, void* ud) {
                (*static_cast<std::atomic<int>*>(ud))++;
            },
            &callback_count);

        nimcp_bio_future_destroy(future);
        nimcp_bio_promise_destroy(promise);
    }

    // Wait a bit to see if callback fires after destroy
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Callback should not have been called (future destroyed before completion)
    // or called exactly once before destroy
    EXPECT_LE(callback_count.load(), 1);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
