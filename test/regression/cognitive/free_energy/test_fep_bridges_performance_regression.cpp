/**
 * @file test_fep_bridges_performance_regression.cpp
 * @brief Performance regression tests for FEP bridge modules
 *
 * WHAT: Tests performance benchmarks for bridge operations
 * WHY:  Prevent performance regressions, ensure operations complete in acceptable time
 * HOW:  Measure timing for critical operations, validate against thresholds
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include "cognitive/free_energy/nimcp_free_energy.h"
#include "cognitive/free_energy/nimcp_fep_immune_bridge.h"
#include "cognitive/free_energy/nimcp_fep_consciousness.h"
#include "cognitive/free_energy/nimcp_fep_context.h"
#include "cognitive/free_energy/nimcp_fep_curiosity.h"
#include "cognitive/free_energy/nimcp_fep_learning.h"
#include "cognitive/free_energy/nimcp_fep_neuromod.h"
#include "cognitive/free_energy/nimcp_fep_planning.h"
#include "cognitive/free_energy/nimcp_fep_sleep.h"
#include "cognitive/free_energy/nimcp_fep_evidence.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "cognitive/introspection/nimcp_introspection.h"

class FEPPerformanceRegressionTest : public ::testing::Test {
protected:
    static const int WARMUP_ITERATIONS = 100;
    static const int BENCHMARK_ITERATIONS = 1000;
    static const uint64_t TIMESTEP_MS = 16;

    // Performance thresholds (microseconds)
    static constexpr int64_t UPDATE_THRESHOLD_US = 100;        // <100μs per update
    static constexpr int64_t CONNECTION_THRESHOLD_US = 50;     // <50μs to connect
    static constexpr int64_t STATE_QUERY_THRESHOLD_US = 10;    // <10μs to query state
    static constexpr int64_t MODULATION_THRESHOLD_US = 200;    // <200μs to apply modulation

    fep_system_t* fep = nullptr;
    brain_immune_system_t* immune = nullptr;

    static const uint32_t OBS_DIM = 16;
    static const uint32_t ACTION_DIM = 8;

    void SetUp() override {
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep_config.num_levels = 2;
        fep = fep_create(&fep_config, OBS_DIM, ACTION_DIM);
        ASSERT_NE(fep, nullptr);

        brain_immune_config_t immune_config;
        brain_immune_default_config(&immune_config);
        immune = brain_immune_create(&immune_config);
        ASSERT_NE(immune, nullptr);
    }

    void TearDown() override {
        if (fep) {
            fep_destroy(fep);
            fep = nullptr;
        }
        if (immune) {
            brain_immune_destroy(immune);
            immune = nullptr;
        }
    }

    // Utility to measure operation time
    template<typename Func>
    int64_t measure_time_us(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    // Benchmark with warmup and statistics
    template<typename Func>
    void benchmark_operation(const char* name, Func&& func, int64_t threshold_us) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            func();
        }

        // Measure
        std::vector<int64_t> timings;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            int64_t time_us = measure_time_us(func);
            timings.push_back(time_us);
        }

        // Statistics
        std::sort(timings.begin(), timings.end());
        int64_t min = timings.front();
        int64_t max = timings.back();
        int64_t median = timings[timings.size() / 2];
        int64_t p95 = timings[(timings.size() * 95) / 100];

        int64_t sum = 0;
        for (auto t : timings) sum += t;
        int64_t avg = sum / timings.size();

        // Report
        printf("  %s: avg=%lldμs, median=%lldμs, p95=%lldμs, min=%lldμs, max=%lldμs\n",
               name, (long long)avg, (long long)median, (long long)p95,
               (long long)min, (long long)max);

        // Verify performance
        EXPECT_LT(avg, threshold_us) << name << " average exceeds threshold";
        EXPECT_LT(p95, threshold_us * 2) << name << " p95 exceeds 2x threshold";
    }
};

/* ============================================================================
 * Immune Bridge Performance Tests
 * ============================================================================ */

TEST_F(FEPPerformanceRegressionTest, ImmuneBridgeUpdatePerformance) {
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);
    fep_immune_bridge_connect_immune(bridge, immune);

    printf("\n[Immune Bridge Update Performance]\n");
    benchmark_operation("update", [&]() {
        fep_immune_bridge_update(bridge, TIMESTEP_MS);
    }, UPDATE_THRESHOLD_US);

    fep_immune_bridge_destroy(bridge);
}

TEST_F(FEPPerformanceRegressionTest, ImmuneBridgeConnectionPerformance) {
    printf("\n[Immune Bridge Connection Performance]\n");
    benchmark_operation("connect_fep", [&]() {
        fep_immune_bridge_t* bridge = fep_immune_bridge_create(nullptr);
        fep_immune_bridge_connect_fep(bridge, fep);
        fep_immune_bridge_destroy(bridge);
    }, CONNECTION_THRESHOLD_US);
}

TEST_F(FEPPerformanceRegressionTest, ImmuneBridgeStateQueryPerformance) {
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);
    fep_immune_bridge_connect_immune(bridge, immune);

    printf("\n[Immune Bridge State Query Performance]\n");
    benchmark_operation("get_state", [&]() {
        fep_immune_state_t state;
        fep_immune_bridge_get_state(bridge, &state);
    }, STATE_QUERY_THRESHOLD_US);

    fep_immune_bridge_destroy(bridge);
}

TEST_F(FEPPerformanceRegressionTest, ImmuneBridgeCytokineUpdatePerformance) {
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);
    fep_immune_bridge_connect_immune(bridge, immune);

    printf("\n[Immune Bridge Cytokine Update Performance]\n");
    benchmark_operation("update_cytokines", [&]() {
        fep_immune_update_cytokine_effects(bridge);
    }, UPDATE_THRESHOLD_US / 2);

    fep_immune_bridge_destroy(bridge);
}

/* ============================================================================
 * Learning Bridge Performance Tests
 * ============================================================================ */

TEST_F(FEPPerformanceRegressionTest, LearningBridgeUpdatePerformance) {
    fep_learning_system_t* learning = fep_learning_create(nullptr);
    ASSERT_NE(learning, nullptr);

    fep_learning_connect_fep(learning, fep);

    float observations[OBS_DIM];
    float states[8];
    for (int i = 0; i < OBS_DIM; i++) observations[i] = 0.5f;
    for (int i = 0; i < 8; i++) states[i] = 0.5f;

    printf("\n[Learning Bridge Update Performance]\n");
    benchmark_operation("update_likelihood", [&]() {
        fep_learning_update_likelihood(learning, observations, OBS_DIM, states, 8);
    }, UPDATE_THRESHOLD_US);

    fep_learning_destroy(learning);
}

TEST_F(FEPPerformanceRegressionTest, LearningBridgeGradientPerformance) {
    fep_learning_system_t* learning = fep_learning_create(nullptr);
    ASSERT_NE(learning, nullptr);

    fep_learning_connect_fep(learning, fep);

    printf("\n[Learning Bridge Gradient Performance]\n");
    benchmark_operation("compute_gradient", [&]() {
        float grad_norm;
        fep_learning_get_gradient_norm(learning, &grad_norm);
    }, STATE_QUERY_THRESHOLD_US);

    fep_learning_destroy(learning);
}

TEST_F(FEPPerformanceRegressionTest, LearningBridgeBatchPerformance) {
    fep_learning_system_t* learning = fep_learning_create(nullptr);
    ASSERT_NE(learning, nullptr);

    fep_learning_connect_fep(learning, fep);

    printf("\n[Learning Bridge Batch Performance]\n");
    benchmark_operation("batch_update_32", [&]() {
        fep_learning_step_batch(learning, 32);
    }, UPDATE_THRESHOLD_US * 32);

    fep_learning_destroy(learning);
}

/* ============================================================================
 * Context Bridge Performance Tests
 * ============================================================================ */

TEST_F(FEPPerformanceRegressionTest, ContextBridgeSwitchPerformance) {
    fep_context_system_t* context = fep_context_create(nullptr);
    ASSERT_NE(context, nullptr);

    fep_context_connect_fep(context, fep);

    uint32_t ctx1, ctx2;
    fep_context_add_context(context, "context_1", &ctx1);
    fep_context_add_context(context, "context_2", &ctx2);

    printf("\n[Context Bridge Switch Performance]\n");
    bool toggle = false;
    benchmark_operation("switch_context", [&]() {
        fep_context_switch_to(context, toggle ? ctx1 : ctx2);
        toggle = !toggle;
    }, UPDATE_THRESHOLD_US);

    fep_context_destroy(context);
}

TEST_F(FEPPerformanceRegressionTest, ContextBridgeInferencePerformance) {
    fep_context_system_t* context = fep_context_create(nullptr);
    ASSERT_NE(context, nullptr);

    fep_context_connect_fep(context, fep);

    uint32_t ctx1, ctx2;
    fep_context_add_context(context, "context_1", &ctx1);
    fep_context_add_context(context, "context_2", &ctx2);

    float observations[OBS_DIM];
    for (int i = 0; i < OBS_DIM; i++) observations[i] = 0.5f;

    printf("\n[Context Bridge Inference Performance]\n");
    benchmark_operation("infer_context", [&]() {
        uint32_t inferred;
        float confidence;
        fep_context_infer(context, observations, OBS_DIM, &inferred, &confidence);
    }, MODULATION_THRESHOLD_US);

    fep_context_destroy(context);
}

/* ============================================================================
 * Neuromodulation Bridge Performance Tests
 * ============================================================================ */

TEST_F(FEPPerformanceRegressionTest, NeuromodBridgeUpdatePerformance) {
    fep_neuromod_system_t* neuromod = fep_neuromod_create(nullptr);
    ASSERT_NE(neuromod, nullptr);

    fep_neuromod_connect_fep(neuromod, fep);

    printf("\n[Neuromod Bridge Update Performance]\n");
    benchmark_operation("update", [&]() {
        fep_neuromod_update(neuromod, TIMESTEP_MS);
    }, UPDATE_THRESHOLD_US);

    fep_neuromod_destroy(neuromod);
}

TEST_F(FEPPerformanceRegressionTest, NeuromodBridgeModulationPerformance) {
    fep_neuromod_system_t* neuromod = fep_neuromod_create(nullptr);
    ASSERT_NE(neuromod, nullptr);

    fep_neuromod_connect_fep(neuromod, fep);

    printf("\n[Neuromod Bridge Modulation Performance]\n");
    benchmark_operation("apply_modulation", [&]() {
        fep_neuromod_apply_modulation(neuromod);
    }, MODULATION_THRESHOLD_US);

    fep_neuromod_destroy(neuromod);
}

TEST_F(FEPPerformanceRegressionTest, NeuromodBridgeSetLevelsPerformance) {
    fep_neuromod_system_t* neuromod = fep_neuromod_create(nullptr);
    ASSERT_NE(neuromod, nullptr);

    fep_neuromod_connect_fep(neuromod, fep);

    printf("\n[Neuromod Bridge Set Levels Performance]\n");
    float level = 0.5f;
    benchmark_operation("set_dopamine", [&]() {
        fep_neuromod_set_dopamine(neuromod, level);
        level = 1.0f - level;
    }, STATE_QUERY_THRESHOLD_US);

    fep_neuromod_destroy(neuromod);
}

/* ============================================================================
 * Sleep Bridge Performance Tests
 * ============================================================================ */

TEST_F(FEPPerformanceRegressionTest, SleepBridgeUpdatePerformance) {
    fep_sleep_system_t* sleep = fep_sleep_create(nullptr);
    ASSERT_NE(sleep, nullptr);

    fep_sleep_connect_fep(sleep, fep);

    printf("\n[Sleep Bridge Update Performance]\n");
    benchmark_operation("update", [&]() {
        fep_sleep_update(sleep, TIMESTEP_MS);
    }, UPDATE_THRESHOLD_US);

    fep_sleep_destroy(sleep);
}

TEST_F(FEPPerformanceRegressionTest, SleepBridgeConsolidationPerformance) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.enable_memory_consolidation = true;

    fep_sleep_system_t* sleep = fep_sleep_create(&config);
    ASSERT_NE(sleep, nullptr);

    fep_sleep_connect_fep(sleep, fep);
    fep_sleep_enter_sleep(sleep);

    printf("\n[Sleep Bridge Consolidation Performance]\n");
    benchmark_operation("consolidate_memories", [&]() {
        fep_sleep_consolidate_memories(sleep);
    }, MODULATION_THRESHOLD_US * 2);

    fep_sleep_destroy(sleep);
}

/* ============================================================================
 * Curiosity Bridge Performance Tests
 * ============================================================================ */

TEST_F(FEPPerformanceRegressionTest, CuriosityBridgeUpdatePerformance) {
    fep_curiosity_config_t config;
    fep_curiosity_default_config(&config);

    fep_curiosity_system_t* curiosity = fep_curiosity_create(&config);
    ASSERT_NE(curiosity, nullptr);

    fep_curiosity_connect_fep(curiosity, fep);

    printf("\n[Curiosity Bridge Update Performance]\n");
    benchmark_operation("update", [&]() {
        fep_curiosity_update(curiosity, TIMESTEP_MS);
    }, UPDATE_THRESHOLD_US);

    fep_curiosity_destroy(curiosity);
}

TEST_F(FEPPerformanceRegressionTest, CuriosityBridgeNoveltyPerformance) {
    fep_curiosity_config_t config;
    fep_curiosity_default_config(&config);

    fep_curiosity_system_t* curiosity = fep_curiosity_create(&config);
    ASSERT_NE(curiosity, nullptr);

    fep_curiosity_connect_fep(curiosity, fep);

    float observations[OBS_DIM];
    for (int i = 0; i < OBS_DIM; i++) observations[i] = 0.5f;

    printf("\n[Curiosity Bridge Novelty Performance]\n");
    benchmark_operation("compute_novelty", [&]() {
        float novelty;
        fep_curiosity_compute_novelty(curiosity, observations, OBS_DIM, &novelty);
    }, UPDATE_THRESHOLD_US);

    fep_curiosity_destroy(curiosity);
}

/* ============================================================================
 * Evidence Bridge Performance Tests
 * ============================================================================ */

TEST_F(FEPPerformanceRegressionTest, EvidenceBridgeUpdatePerformance) {
    fep_evidence_config_t config;
    fep_evidence_default_config(&config);

    fep_evidence_system_t* evidence = fep_evidence_create(&config);
    ASSERT_NE(evidence, nullptr);

    fep_evidence_connect_fep(evidence, fep);

    printf("\n[Evidence Bridge Update Performance]\n");
    benchmark_operation("update", [&]() {
        fep_evidence_update(evidence, TIMESTEP_MS);
    }, UPDATE_THRESHOLD_US);

    fep_evidence_destroy(evidence);
}

TEST_F(FEPPerformanceRegressionTest, EvidenceBridgeComputePerformance) {
    fep_evidence_config_t config;
    fep_evidence_default_config(&config);

    fep_evidence_system_t* evidence = fep_evidence_create(&config);
    ASSERT_NE(evidence, nullptr);

    fep_evidence_connect_fep(evidence, fep);

    printf("\n[Evidence Bridge Compute Performance]\n");
    benchmark_operation("compute_evidence", [&]() {
        float evidence_val;
        fep_evidence_compute(evidence, &evidence_val);
    }, UPDATE_THRESHOLD_US);

    fep_evidence_destroy(evidence);
}

/* ============================================================================
 * Multi-Bridge Performance Tests
 * ============================================================================ */

TEST_F(FEPPerformanceRegressionTest, MultiBridgeSequentialUpdatePerformance) {
    fep_immune_bridge_t* immune_bridge = fep_immune_bridge_create(nullptr);
    fep_neuromod_system_t* neuromod = fep_neuromod_create(nullptr);
    fep_learning_system_t* learning = fep_learning_create(nullptr);

    fep_immune_bridge_connect_fep(immune_bridge, fep);
    fep_immune_bridge_connect_immune(immune_bridge, immune);
    fep_neuromod_connect_fep(neuromod, fep);
    fep_learning_connect_fep(learning, fep);

    printf("\n[Multi-Bridge Sequential Update Performance]\n");
    benchmark_operation("all_updates", [&]() {
        fep_immune_bridge_update(immune_bridge, TIMESTEP_MS);
        fep_neuromod_update(neuromod, TIMESTEP_MS);
    }, UPDATE_THRESHOLD_US * 3);

    fep_immune_bridge_destroy(immune_bridge);
    fep_neuromod_destroy(neuromod);
    fep_learning_destroy(learning);
}

TEST_F(FEPPerformanceRegressionTest, BioAsyncMessageLatency) {
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);

    printf("\n[Bio-Async Message Latency]\n");
    benchmark_operation("bio_async_connect", [&]() {
        fep_immune_bridge_connect_bio_async(bridge);
        fep_immune_bridge_disconnect_bio_async(bridge);
    }, CONNECTION_THRESHOLD_US * 2);

    fep_immune_bridge_destroy(bridge);
}
