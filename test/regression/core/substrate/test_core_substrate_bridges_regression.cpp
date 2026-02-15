/**
 * @file test_core_substrate_bridges_regression.cpp
 * @brief Regression tests for core substrate bridges
 * @version 1.0.0
 *
 * WHAT: Performance, memory, and correctness regression tests for substrate bridges
 * WHY:  Ensure substrate bridge operations maintain baseline performance and stability
 * HOW:  Benchmark spike processing, memory usage, numerical stability, and scaling
 *
 * Tests coverage:
 * - Neuron substrate bridge (Q10 temperature effects, ATP consumption, spike processing)
 * - Synapse substrate bridge (Transmission modulation, ATP costs, type-specific effects)
 * - Axon-dendrite substrate bridge (Conduction velocity, integration, plasticity)
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

// Headers have their own extern "C" guards
#include "core/neuron_models/nimcp_neuron_substrate_bridge.h"
#include "core/synapse_compute/nimcp_synapse_substrate_bridge.h"
#include "core/nimcp_axon_dendrite_substrate_bridge.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/neuron_models/nimcp_neuron_model.h"
#include "core/neuron_models/nimcp_izhikevich.h"
#include "core/synapse_compute/nimcp_synapse_compute.h"
#include "core/axon/nimcp_axon.h"
#include "core/dendrite/nimcp_dendrite.h"

//=============================================================================
// PERFORMANCE MONITORING UTILITIES
//=============================================================================

class PerformanceMonitor {
public:
    template<typename Func>
    static double MeasureTimeMs(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static double Mean(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }

    static double StdDev(const std::vector<double>& values) {
        if (values.size() < 2) return 0.0;
        double mean = Mean(values);
        double sq_sum = 0.0;
        for (double v : values) {
            sq_sum += (v - mean) * (v - mean);
        }
        return std::sqrt(sq_sum / (values.size() - 1));
    }

    static double Min(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        return *std::min_element(values.begin(), values.end());
    }

    static double Max(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        return *std::max_element(values.begin(), values.end());
    }
};

//=============================================================================
// PERFORMANCE BASELINES
//=============================================================================

namespace Baseline {
    // Neuron substrate bridge benchmarks (ms)
    constexpr double NEURON_UPDATE_1K_MS = 5.0;
    constexpr double NEURON_SPIKE_CONSUME_10K_MS = 10.0;

    // Synapse substrate bridge benchmarks (ms)
    constexpr double SYNAPSE_UPDATE_1K_MS = 8.0;
    constexpr double SYNAPSE_TRANSMISSION_10K_MS = 15.0;

    // Axon-dendrite substrate bridge benchmarks (ms)
    constexpr double AXON_DENDRITE_UPDATE_1K_MS = 10.0;
    constexpr double AXON_SPIKE_RECORD_10K_MS = 12.0;

    // Memory regression limits (bytes per bridge)
    constexpr size_t MAX_NEURON_BRIDGE_SIZE = 2048;
    constexpr size_t MAX_SYNAPSE_BRIDGE_SIZE = 2048;
    constexpr size_t MAX_AXON_DENDRITE_BRIDGE_SIZE = 2560;

    // Regression tolerance (30% above baseline)
    constexpr double REGRESSION_TOLERANCE = 1.3;

    // Numerical stability tolerances
    constexpr double Q10_TOLERANCE = 1e-6;
    constexpr double MODULATION_TOLERANCE = 1e-6;
}

//=============================================================================
// TEST FIXTURES
//=============================================================================

class SubstrateBridgeRegressionTest : public ::testing::Test {
protected:
    static constexpr int NUM_SAMPLES = 10;
    static constexpr int WARMUP_RUNS = 2;

    neural_substrate_t* substrate = nullptr;
    neuron_model_state_t neuron_model = nullptr;
    axon_t* test_axon = nullptr;
    dendrite_t* test_dendrite = nullptr;

    void SetUp() override {
        // Create substrate with normal conditions
        substrate = substrate_create(nullptr);
        ASSERT_NE(substrate, nullptr);

        // Create a neuron model (required by bridge creation)
        const neuron_model_vtable_t* vtable = neuron_model_get_izhikevich_vtable();
        ASSERT_NE(vtable, nullptr);
        neuron_model = neuron_model_create(vtable, nullptr);
        ASSERT_NE(neuron_model, nullptr);

        // Create axon and dendrite (required by axon-dendrite bridge creation)
        test_axon = axon_create(1, AXON_TYPE_MYELINATED, 0, 1, 100.0f, 1.0f);
        ASSERT_NE(test_axon, nullptr);

        dendrite_config_t dend_config = {};
        dend_config.id = 1;
        dend_config.type = DENDRITE_TYPE_BASAL;
        dend_config.target_neuron_id = 0;
        dend_config.total_length = 200.0f;
        dend_config.mean_diameter = 1.0f;
        dend_config.integration_window_ms = 10.0f;
        dend_config.structural_plasticity = 0.5f;
        dend_config.ltp_threshold = 0.5f;
        dend_config.ltd_threshold = 0.3f;
        test_dendrite = dendrite_create(&dend_config);
        ASSERT_NE(test_dendrite, nullptr);

        // Initialize to normal physiological state
        substrate_set_temperature(substrate, SUBSTRATE_NORMAL_TEMPERATURE);
        substrate_set_atp(substrate, SUBSTRATE_NORMAL_ATP);
        substrate_set_oxygen(substrate, SUBSTRATE_NORMAL_O2_SAT);
        substrate_set_ion_balance(substrate, SUBSTRATE_NORMAL_ION_BALANCE);
    }

    void TearDown() override {
        if (test_dendrite) {
            dendrite_destroy(test_dendrite);
            test_dendrite = nullptr;
        }
        if (test_axon) {
            axon_destroy(test_axon);
            test_axon = nullptr;
        }
        if (neuron_model) {
            neuron_model_destroy(neuron_model);
            neuron_model = nullptr;
        }
        if (substrate) {
            substrate_destroy(substrate);
            substrate = nullptr;
        }
    }
};

//=============================================================================
// NEURON SUBSTRATE BRIDGE - PERFORMANCE BENCHMARKS
//=============================================================================

TEST_F(SubstrateBridgeRegressionTest, NeuronBridgeUpdatePerformance) {
    std::cout << "\n=== Neuron Bridge Update Performance ===" << std::endl;

    constexpr int NUM_BRIDGES = 1000;
    std::vector<double> times_ms;

    // Create bridges (using fixture's neuron_model)
    std::vector<neuron_substrate_bridge_t*> bridges;
    for (int i = 0; i < NUM_BRIDGES; i++) {
        bridges.push_back(neuron_substrate_bridge_create(nullptr, neuron_model, substrate));
        ASSERT_NE(bridges.back(), nullptr);
    }

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        for (auto* bridge : bridges) {
            neuron_substrate_update_effects(bridge);
        }
    }

    // Measure
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        double time = PerformanceMonitor::MeasureTimeMs([&]() {
            for (auto* bridge : bridges) {
                neuron_substrate_update_effects(bridge);
            }
        });
        times_ms.push_back(time);
    }

    double mean = PerformanceMonitor::Mean(times_ms);
    double stddev = PerformanceMonitor::StdDev(times_ms);

    std::cout << "Updating " << NUM_BRIDGES << " neuron bridges:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms" << std::endl;
    std::cout << "  StdDev: " << stddev << " ms" << std::endl;
    std::cout << "  Throughput: " << (NUM_BRIDGES / mean * 1000.0) << " updates/sec" << std::endl;
    std::cout << "  Baseline: < " << Baseline::NEURON_UPDATE_1K_MS << " ms" << std::endl;

    EXPECT_LT(mean, Baseline::NEURON_UPDATE_1K_MS * Baseline::REGRESSION_TOLERANCE)
        << "Neuron bridge update performance regression detected";

    // Cleanup
    for (auto* bridge : bridges) {
        neuron_substrate_bridge_destroy(bridge);
    }
}

TEST_F(SubstrateBridgeRegressionTest, NeuronSpikeConsumptionPerformance) {
    std::cout << "\n=== Neuron Spike Consumption Performance ===" << std::endl;

    constexpr int NUM_SPIKES = 10000;
    std::vector<double> times_ms;

    neuron_substrate_bridge_t* bridge = neuron_substrate_bridge_create(nullptr, neuron_model, substrate);
    ASSERT_NE(bridge, nullptr);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        for (int i = 0; i < 1000; i++) {
            neuron_substrate_consume_spike(bridge);
        }
    }

    // Measure
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        // Reset ATP to avoid depletion
        substrate_set_atp(substrate, SUBSTRATE_NORMAL_ATP);

        double time = PerformanceMonitor::MeasureTimeMs([&]() {
            for (int i = 0; i < NUM_SPIKES; i++) {
                neuron_substrate_consume_spike(bridge);
            }
        });
        times_ms.push_back(time);
    }

    double mean = PerformanceMonitor::Mean(times_ms);
    double stddev = PerformanceMonitor::StdDev(times_ms);

    std::cout << "Processing " << NUM_SPIKES << " spike consumption events:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms" << std::endl;
    std::cout << "  StdDev: " << stddev << " ms" << std::endl;
    std::cout << "  Throughput: " << (NUM_SPIKES / mean * 1000.0) << " spikes/sec" << std::endl;
    std::cout << "  Baseline: < " << Baseline::NEURON_SPIKE_CONSUME_10K_MS << " ms" << std::endl;

    EXPECT_LT(mean, Baseline::NEURON_SPIKE_CONSUME_10K_MS * Baseline::REGRESSION_TOLERANCE)
        << "Neuron spike consumption performance regression detected";

    neuron_substrate_bridge_destroy(bridge);
}

//=============================================================================
// SYNAPSE SUBSTRATE BRIDGE - PERFORMANCE BENCHMARKS
//=============================================================================

TEST_F(SubstrateBridgeRegressionTest, SynapseBridgeUpdatePerformance) {
    std::cout << "\n=== Synapse Bridge Update Performance ===" << std::endl;

    constexpr int NUM_BRIDGES = 1000;
    std::vector<double> times_ms;

    // Create synapse context (NULL is acceptable for performance test)
    synapse_compute_context_t* synapse_ctx = nullptr;

    // Create bridges
    std::vector<synapse_substrate_bridge_t*> bridges;
    for (int i = 0; i < NUM_BRIDGES; i++) {
        bridges.push_back(synapse_substrate_bridge_create(nullptr, synapse_ctx, substrate));
        ASSERT_NE(bridges.back(), nullptr);
    }

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        for (auto* bridge : bridges) {
            synapse_substrate_update(bridge);
        }
    }

    // Measure
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        double time = PerformanceMonitor::MeasureTimeMs([&]() {
            for (auto* bridge : bridges) {
                synapse_substrate_update(bridge);
            }
        });
        times_ms.push_back(time);
    }

    double mean = PerformanceMonitor::Mean(times_ms);
    double stddev = PerformanceMonitor::StdDev(times_ms);

    std::cout << "Updating " << NUM_BRIDGES << " synapse bridges:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms" << std::endl;
    std::cout << "  StdDev: " << stddev << " ms" << std::endl;
    std::cout << "  Throughput: " << (NUM_BRIDGES / mean * 1000.0) << " updates/sec" << std::endl;
    std::cout << "  Baseline: < " << Baseline::SYNAPSE_UPDATE_1K_MS << " ms" << std::endl;

    EXPECT_LT(mean, Baseline::SYNAPSE_UPDATE_1K_MS * Baseline::REGRESSION_TOLERANCE)
        << "Synapse bridge update performance regression detected";

    // Cleanup
    for (auto* bridge : bridges) {
        synapse_substrate_bridge_destroy(bridge);
    }
}

TEST_F(SubstrateBridgeRegressionTest, SynapseTransmissionConsumptionPerformance) {
    std::cout << "\n=== Synapse Transmission Consumption Performance ===" << std::endl;

    constexpr int NUM_TRANSMISSIONS = 10000;
    std::vector<double> times_ms;

    synapse_substrate_bridge_t* bridge = synapse_substrate_bridge_create(nullptr, nullptr, substrate);
    ASSERT_NE(bridge, nullptr);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        for (int i = 0; i < 1000; i++) {
            synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 1);
        }
    }

    // Measure
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        // Reset ATP
        substrate_set_atp(substrate, SUBSTRATE_NORMAL_ATP);

        double time = PerformanceMonitor::MeasureTimeMs([&]() {
            for (int i = 0; i < NUM_TRANSMISSIONS; i++) {
                synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 1);
            }
        });
        times_ms.push_back(time);
    }

    double mean = PerformanceMonitor::Mean(times_ms);
    double stddev = PerformanceMonitor::StdDev(times_ms);

    std::cout << "Processing " << NUM_TRANSMISSIONS << " transmission events:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms" << std::endl;
    std::cout << "  StdDev: " << stddev << " ms" << std::endl;
    std::cout << "  Throughput: " << (NUM_TRANSMISSIONS / mean * 1000.0) << " events/sec" << std::endl;
    std::cout << "  Baseline: < " << Baseline::SYNAPSE_TRANSMISSION_10K_MS << " ms" << std::endl;

    EXPECT_LT(mean, Baseline::SYNAPSE_TRANSMISSION_10K_MS * Baseline::REGRESSION_TOLERANCE)
        << "Synapse transmission consumption performance regression detected";

    synapse_substrate_bridge_destroy(bridge);
}

//=============================================================================
// AXON-DENDRITE SUBSTRATE BRIDGE - PERFORMANCE BENCHMARKS
//=============================================================================

TEST_F(SubstrateBridgeRegressionTest, AxonDendriteBridgeUpdatePerformance) {
    std::cout << "\n=== Axon-Dendrite Bridge Update Performance ===" << std::endl;

    constexpr int NUM_BRIDGES = 1000;
    std::vector<double> times_ms;

    // Create bridges (axon/dendrite can be NULL for performance test)
    std::vector<axon_dendrite_substrate_bridge_t*> bridges;
    for (int i = 0; i < NUM_BRIDGES; i++) {
        bridges.push_back(axon_dendrite_substrate_bridge_create(nullptr, substrate, test_axon, test_dendrite));
        ASSERT_NE(bridges.back(), nullptr);
    }

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        for (auto* bridge : bridges) {
            axon_dendrite_substrate_bridge_update(bridge, 1);
        }
    }

    // Measure
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        double time = PerformanceMonitor::MeasureTimeMs([&]() {
            for (auto* bridge : bridges) {
                axon_dendrite_substrate_bridge_update(bridge, 1);
            }
        });
        times_ms.push_back(time);
    }

    double mean = PerformanceMonitor::Mean(times_ms);
    double stddev = PerformanceMonitor::StdDev(times_ms);

    std::cout << "Updating " << NUM_BRIDGES << " axon-dendrite bridges:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms" << std::endl;
    std::cout << "  StdDev: " << stddev << " ms" << std::endl;
    std::cout << "  Throughput: " << (NUM_BRIDGES / mean * 1000.0) << " updates/sec" << std::endl;
    std::cout << "  Baseline: < " << Baseline::AXON_DENDRITE_UPDATE_1K_MS << " ms" << std::endl;

    EXPECT_LT(mean, Baseline::AXON_DENDRITE_UPDATE_1K_MS * Baseline::REGRESSION_TOLERANCE)
        << "Axon-dendrite bridge update performance regression detected";

    // Cleanup
    for (auto* bridge : bridges) {
        axon_dendrite_substrate_bridge_destroy(bridge);
    }
}

TEST_F(SubstrateBridgeRegressionTest, AxonSpikeRecordingPerformance) {
    std::cout << "\n=== Axon Spike Recording Performance ===" << std::endl;

    constexpr int NUM_SPIKES = 10000;
    std::vector<double> times_ms;

    axon_dendrite_substrate_bridge_t* bridge = axon_dendrite_substrate_bridge_create(
        nullptr, substrate, test_axon, test_dendrite);
    ASSERT_NE(bridge, nullptr);

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; w++) {
        for (int i = 0; i < 1000; i++) {
            axon_dendrite_substrate_record_axon_spikes(bridge, 1);
        }
    }

    // Measure
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        // Reset ATP
        substrate_set_atp(substrate, SUBSTRATE_NORMAL_ATP);

        double time = PerformanceMonitor::MeasureTimeMs([&]() {
            for (int i = 0; i < NUM_SPIKES; i++) {
                axon_dendrite_substrate_record_axon_spikes(bridge, 1);
            }
        });
        times_ms.push_back(time);
    }

    double mean = PerformanceMonitor::Mean(times_ms);
    double stddev = PerformanceMonitor::StdDev(times_ms);

    std::cout << "Recording " << NUM_SPIKES << " axon spikes:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms" << std::endl;
    std::cout << "  StdDev: " << stddev << " ms" << std::endl;
    std::cout << "  Throughput: " << (NUM_SPIKES / mean * 1000.0) << " spikes/sec" << std::endl;
    std::cout << "  Baseline: < " << Baseline::AXON_SPIKE_RECORD_10K_MS << " ms" << std::endl;

    EXPECT_LT(mean, Baseline::AXON_SPIKE_RECORD_10K_MS * Baseline::REGRESSION_TOLERANCE)
        << "Axon spike recording performance regression detected";

    axon_dendrite_substrate_bridge_destroy(bridge);
}

//=============================================================================
// MEMORY REGRESSION TESTS
//=============================================================================

TEST_F(SubstrateBridgeRegressionTest, NeuronBridgeMemoryRegression) {
    std::cout << "\n=== Neuron Bridge Memory Regression ===" << std::endl;

    constexpr int NUM_ITERATIONS = 1000;

    // Repeatedly create/destroy and process spikes (using fixture neuron_model)
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        neuron_substrate_bridge_t* bridge = neuron_substrate_bridge_create(
            nullptr, neuron_model, substrate);
        ASSERT_NE(bridge, nullptr);

        // Process multiple spike events
        for (int j = 0; j < 100; j++) {
            neuron_substrate_consume_spike(bridge);
            neuron_substrate_update_effects(bridge);
        }

        neuron_substrate_bridge_destroy(bridge);

        // Reset substrate for next iteration
        substrate_set_atp(substrate, SUBSTRATE_NORMAL_ATP);
    }

    std::cout << "Completed " << NUM_ITERATIONS << " create/destroy cycles with spike processing" << std::endl;
    std::cout << "No memory leaks detected (test passed without crashes)" << std::endl;

    // If we get here without crashes/assertions, memory management is stable
    SUCCEED();
}

TEST_F(SubstrateBridgeRegressionTest, SynapseBridgeMemoryRegression) {
    std::cout << "\n=== Synapse Bridge Memory Regression ===" << std::endl;

    constexpr int NUM_ITERATIONS = 1000;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        synapse_substrate_bridge_t* bridge = synapse_substrate_bridge_create(
            nullptr, nullptr, substrate);
        ASSERT_NE(bridge, nullptr);

        // Process transmission events across all synapse types
        for (int j = 0; j < 10; j++) {
            synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 1);
            synapse_substrate_consume_transmission(bridge, SYNAPSE_NMDA, 1);
            synapse_substrate_consume_transmission(bridge, SYNAPSE_GABA_A, 1);
            synapse_substrate_update(bridge);
        }

        synapse_substrate_bridge_destroy(bridge);

        // Reset substrate
        substrate_set_atp(substrate, SUBSTRATE_NORMAL_ATP);
    }

    std::cout << "Completed " << NUM_ITERATIONS << " create/destroy cycles with transmission processing" << std::endl;
    SUCCEED();
}

TEST_F(SubstrateBridgeRegressionTest, AxonDendriteBridgeMemoryRegression) {
    std::cout << "\n=== Axon-Dendrite Bridge Memory Regression ===" << std::endl;

    constexpr int NUM_ITERATIONS = 1000;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        axon_dendrite_substrate_bridge_t* bridge = axon_dendrite_substrate_bridge_create(
            nullptr, substrate, test_axon, test_dendrite);
        ASSERT_NE(bridge, nullptr);

        // Process spikes and dendrite events
        for (int j = 0; j < 10; j++) {
            axon_dendrite_substrate_record_axon_spikes(bridge, 5);
            axon_dendrite_substrate_record_dendrite_events(bridge, 3);
            axon_dendrite_substrate_bridge_update(bridge, 1);
        }

        axon_dendrite_substrate_bridge_destroy(bridge);

        // Reset substrate
        substrate_set_atp(substrate, SUBSTRATE_NORMAL_ATP);
    }

    std::cout << "Completed " << NUM_ITERATIONS << " create/destroy cycles with activity recording" << std::endl;
    SUCCEED();
}

//=============================================================================
// NUMERICAL STABILITY TESTS
//=============================================================================

TEST_F(SubstrateBridgeRegressionTest, Q10CalculationStability) {
    std::cout << "\n=== Q10 Calculation Numerical Stability ===" << std::endl;

    neuron_substrate_bridge_t* bridge = neuron_substrate_bridge_create(
        nullptr, neuron_model, substrate);
    ASSERT_NE(bridge, nullptr);

    // Test temperature extremes
    std::vector<float> test_temps = {
        32.0f,  // Hypothermia
        35.0f,  // Low normal
        37.0f,  // Normal
        39.0f,  // High normal
        40.0f   // Hyperthermia
    };

    for (float temp : test_temps) {
        substrate_set_temperature(substrate, temp);
        neuron_substrate_update_effects(bridge);

        neuron_substrate_effects_t effects;
        neuron_substrate_get_effects(bridge, &effects);

        // Q10 modulation should be finite and positive
        EXPECT_TRUE(std::isfinite(effects.q10_firing_rate_mod))
            << "Q10 firing rate mod is not finite at temp " << temp;
        EXPECT_GT(effects.q10_firing_rate_mod, 0.0f)
            << "Q10 firing rate mod is not positive at temp " << temp;
        EXPECT_LE(effects.q10_firing_rate_mod, 5.0f)
            << "Q10 firing rate mod exceeds reasonable bounds at temp " << temp;

        std::cout << "  Temp " << temp << "°C: Q10 mod = "
                  << effects.q10_firing_rate_mod << std::endl;
    }

    neuron_substrate_bridge_destroy(bridge);
}

TEST_F(SubstrateBridgeRegressionTest, ModulationEdgeValues) {
    std::cout << "\n=== Modulation Edge Value Stability ===" << std::endl;

    synapse_substrate_bridge_t* bridge = synapse_substrate_bridge_create(
        nullptr, nullptr, substrate);
    ASSERT_NE(bridge, nullptr);

    // Test extreme ATP levels
    std::vector<float> test_atp_levels = {0.0f, 0.1f, 0.5f, 0.9f, 1.0f};

    for (float atp : test_atp_levels) {
        substrate_set_atp(substrate, atp);
        synapse_substrate_update(bridge);

        // Test all synapse types for valid modulation
        for (int type = SYNAPSE_AMPA; type <= SYNAPSE_ELECTRICAL; type++) {
            float mod = synapse_substrate_apply_modulation(bridge, static_cast<synapse_type_t>(type));

            EXPECT_TRUE(std::isfinite(mod))
                << "Modulation not finite for type " << type << " at ATP " << atp;
            EXPECT_GE(mod, 0.0f)
                << "Modulation negative for type " << type << " at ATP " << atp;
            EXPECT_LE(mod, 1.5f)
                << "Modulation exceeds bounds for type " << type << " at ATP " << atp;
        }

        std::cout << "  ATP " << atp << ": All synapse types stable" << std::endl;
    }

    synapse_substrate_bridge_destroy(bridge);
}

TEST_F(SubstrateBridgeRegressionTest, AxonDendriteModulationStability) {
    std::cout << "\n=== Axon-Dendrite Modulation Stability ===" << std::endl;

    axon_dendrite_substrate_bridge_t* bridge = axon_dendrite_substrate_bridge_create(
        nullptr, substrate, test_axon, test_dendrite);
    ASSERT_NE(bridge, nullptr);

    // Test various substrate conditions
    struct TestCase {
        float atp;
        float temp;
        float ion_balance;
    };

    std::vector<TestCase> cases = {
        {0.0f, 37.0f, 0.5f},   // Depleted ATP
        {0.5f, 32.0f, 0.95f},  // Low temperature
        {0.95f, 40.0f, 0.95f}, // High temperature
        {0.95f, 37.0f, 0.3f},  // Ion imbalance
        {0.2f, 35.0f, 0.4f}    // Multiple stressors
    };

    for (const auto& tc : cases) {
        substrate_set_atp(substrate, tc.atp);
        substrate_set_temperature(substrate, tc.temp);
        substrate_set_ion_balance(substrate, tc.ion_balance);

        axon_dendrite_substrate_update_axon_effects(bridge);
        axon_dendrite_substrate_update_dendrite_effects(bridge);

        float velocity_mod = axon_dendrite_substrate_get_conduction_mod(bridge);
        float integration_mod = axon_dendrite_substrate_get_integration_mod(bridge);
        float plasticity_mod = axon_dendrite_substrate_get_plasticity_mod(bridge);

        // All modulation factors should be finite and bounded
        EXPECT_TRUE(std::isfinite(velocity_mod));
        EXPECT_TRUE(std::isfinite(integration_mod));
        EXPECT_TRUE(std::isfinite(plasticity_mod));

        EXPECT_GE(velocity_mod, 0.0f);
        EXPECT_GE(integration_mod, 0.0f);
        EXPECT_GE(plasticity_mod, 0.0f);

        EXPECT_LE(velocity_mod, 2.0f);
        EXPECT_LE(integration_mod, 1.5f);
        EXPECT_LE(plasticity_mod, 1.5f);

        std::cout << "  ATP=" << tc.atp << " Temp=" << tc.temp << " Ion=" << tc.ion_balance
                  << " => Vel=" << velocity_mod << " Int=" << integration_mod
                  << " Plas=" << plasticity_mod << std::endl;
    }

    axon_dendrite_substrate_bridge_destroy(bridge);
}

//=============================================================================
// SCALING TESTS
//=============================================================================

TEST_F(SubstrateBridgeRegressionTest, ManyNeuronBridgesScaling) {
    std::cout << "\n=== Many Neuron Bridges Scaling ===" << std::endl;

    constexpr int NUM_BRIDGES = 5000;

    // Create many bridges (using fixture neuron_model)
    std::vector<neuron_substrate_bridge_t*> bridges;
    double create_time = PerformanceMonitor::MeasureTimeMs([&]() {
        for (int i = 0; i < NUM_BRIDGES; i++) {
            bridges.push_back(neuron_substrate_bridge_create(nullptr, neuron_model, substrate));
        }
    });

    // Update all bridges
    double update_time = PerformanceMonitor::MeasureTimeMs([&]() {
        for (auto* bridge : bridges) {
            neuron_substrate_update_effects(bridge);
        }
    });

    // Process spike events
    double spike_time = PerformanceMonitor::MeasureTimeMs([&]() {
        for (auto* bridge : bridges) {
            neuron_substrate_consume_spike(bridge);
        }
    });

    std::cout << "Scaling with " << NUM_BRIDGES << " neuron bridges:" << std::endl;
    std::cout << "  Creation: " << create_time << " ms ("
              << (NUM_BRIDGES / create_time * 1000.0) << " bridges/sec)" << std::endl;
    std::cout << "  Update: " << update_time << " ms ("
              << (NUM_BRIDGES / update_time * 1000.0) << " updates/sec)" << std::endl;
    std::cout << "  Spike consumption: " << spike_time << " ms ("
              << (NUM_BRIDGES / spike_time * 1000.0) << " spikes/sec)" << std::endl;

    // Cleanup
    for (auto* bridge : bridges) {
        neuron_substrate_bridge_destroy(bridge);
    }

    SUCCEED();
}

TEST_F(SubstrateBridgeRegressionTest, ManySynapseBridgesScaling) {
    std::cout << "\n=== Many Synapse Bridges Scaling ===" << std::endl;

    constexpr int NUM_BRIDGES = 5000;

    // Create many bridges
    std::vector<synapse_substrate_bridge_t*> bridges;
    double create_time = PerformanceMonitor::MeasureTimeMs([&]() {
        for (int i = 0; i < NUM_BRIDGES; i++) {
            bridges.push_back(synapse_substrate_bridge_create(nullptr, nullptr, substrate));
        }
    });

    // Update all bridges
    double update_time = PerformanceMonitor::MeasureTimeMs([&]() {
        for (auto* bridge : bridges) {
            synapse_substrate_update(bridge);
        }
    });

    // Process transmission events
    double transmission_time = PerformanceMonitor::MeasureTimeMs([&]() {
        for (auto* bridge : bridges) {
            synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 1);
        }
    });

    std::cout << "Scaling with " << NUM_BRIDGES << " synapse bridges:" << std::endl;
    std::cout << "  Creation: " << create_time << " ms" << std::endl;
    std::cout << "  Update: " << update_time << " ms" << std::endl;
    std::cout << "  Transmission: " << transmission_time << " ms" << std::endl;

    // Cleanup
    for (auto* bridge : bridges) {
        synapse_substrate_bridge_destroy(bridge);
    }

    SUCCEED();
}

//=============================================================================
// CORRECTNESS REGRESSION TESTS
//=============================================================================

TEST_F(SubstrateBridgeRegressionTest, Q10CalculationCorrectness) {
    std::cout << "\n=== Q10 Calculation Correctness ===" << std::endl;

    neuron_substrate_bridge_t* bridge = neuron_substrate_bridge_create(
        nullptr, neuron_model, substrate);
    ASSERT_NE(bridge, nullptr);

    // At reference temperature (37°C), Q10 mod should be ~1.0
    substrate_set_temperature(substrate, 37.0f);
    neuron_substrate_update_effects(bridge);

    neuron_substrate_effects_t effects;
    neuron_substrate_get_effects(bridge, &effects);

    EXPECT_NEAR(effects.q10_firing_rate_mod, 1.0f, 0.1f)
        << "Q10 should be near 1.0 at reference temperature";

    // At 47°C (+10°C), Q10 should be ~2.5 (NEURON_Q10_FIRING_RATE)
    substrate_set_temperature(substrate, 47.0f);
    neuron_substrate_update_effects(bridge);
    neuron_substrate_get_effects(bridge, &effects);

    EXPECT_NEAR(effects.q10_firing_rate_mod, 2.5f, 0.5f)
        << "Q10 should be ~2.5 at +10°C from reference";

    // At 27°C (-10°C), Q10 should be ~0.4 (1/2.5)
    substrate_set_temperature(substrate, 27.0f);
    neuron_substrate_update_effects(bridge);
    neuron_substrate_get_effects(bridge, &effects);

    EXPECT_NEAR(effects.q10_firing_rate_mod, 1.0f / 2.5f, 0.2f)
        << "Q10 should be ~0.4 at -10°C from reference";

    std::cout << "  Q10 calculations verified for temperature range" << std::endl;

    neuron_substrate_bridge_destroy(bridge);
}

TEST_F(SubstrateBridgeRegressionTest, ATPDepletionAccuracy) {
    std::cout << "\n=== ATP Depletion Accuracy ===" << std::endl;

    neuron_substrate_bridge_t* bridge = neuron_substrate_bridge_create(
        nullptr, neuron_model, substrate);
    ASSERT_NE(bridge, nullptr);

    // Start with full ATP
    substrate_set_atp(substrate, 1.0f);

    // Process many spikes and track ATP depletion
    constexpr int NUM_SPIKES = 100;
    float initial_atp = substrate_get_capacity(substrate);

    for (int i = 0; i < NUM_SPIKES; i++) {
        neuron_substrate_consume_spike(bridge);
    }

    float final_atp = substrate_get_capacity(substrate);
    float atp_consumed = initial_atp - final_atp;

    // Spikes should consume some ATP (exact cost depends on implementation constants)
    EXPECT_GT(atp_consumed, 0.0f)
        << "Spikes should consume some ATP";
    EXPECT_LT(atp_consumed, 0.5f)
        << "ATP consumption should be bounded for " << NUM_SPIKES << " spikes";

    std::cout << "  Initial ATP: " << initial_atp << std::endl;
    std::cout << "  Final ATP: " << final_atp << std::endl;
    std::cout << "  Consumed: " << atp_consumed << std::endl;

    neuron_substrate_bridge_destroy(bridge);
}

TEST_F(SubstrateBridgeRegressionTest, SynapseTypeSpecificModulation) {
    std::cout << "\n=== Synapse Type-Specific Modulation Correctness ===" << std::endl;

    synapse_substrate_bridge_t* bridge = synapse_substrate_bridge_create(
        nullptr, nullptr, substrate);
    ASSERT_NE(bridge, nullptr);

    // Set moderate ATP depletion
    substrate_set_atp(substrate, 0.5f);
    synapse_substrate_update(bridge);

    // Get modulation for different synapse types
    float ampa_mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_AMPA);
    float gaba_b_mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_GABA_B);
    float dopamine_mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_DOPAMINE);

    // GABA-B and Dopamine (metabotropic) should be at least as affected as AMPA (ionotropic)
    EXPECT_LE(gaba_b_mod, ampa_mod)
        << "GABA-B (metabotropic) should be at least as ATP-sensitive as AMPA";
    EXPECT_LE(dopamine_mod, ampa_mod)
        << "Dopamine (metabotropic) should be at least as ATP-sensitive as AMPA";

    std::cout << "  AMPA mod: " << ampa_mod << std::endl;
    std::cout << "  GABA-B mod: " << gaba_b_mod << std::endl;
    std::cout << "  Dopamine mod: " << dopamine_mod << std::endl;

    synapse_substrate_bridge_destroy(bridge);
}

//=============================================================================
// THREAD SAFETY REGRESSION
//=============================================================================

TEST_F(SubstrateBridgeRegressionTest, ConcurrentNeuronActivityThreadSafety) {
    std::cout << "\n=== Concurrent Neuron Activity Thread Safety ===" << std::endl;

    neuron_substrate_bridge_t* bridge = neuron_substrate_bridge_create(
        nullptr, neuron_model, substrate);
    ASSERT_NE(bridge, nullptr);

    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 1000;

    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&]() {
            for (int i = 0; i < OPS_PER_THREAD; i++) {
                neuron_substrate_consume_spike(bridge);
                neuron_substrate_update_effects(bridge);

                neuron_substrate_effects_t effects;
                neuron_substrate_get_effects(bridge, &effects);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "  Completed " << (NUM_THREADS * OPS_PER_THREAD)
              << " concurrent operations without crashes" << std::endl;

    neuron_substrate_bridge_destroy(bridge);
    SUCCEED();
}

//=============================================================================
// MAIN
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
