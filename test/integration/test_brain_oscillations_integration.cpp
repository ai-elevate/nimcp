/**
 * @file test_brain_oscillations_integration.cpp
 * @brief Integration tests for brain oscillations with real neural networks
 *
 * WHAT: Test oscillation analysis with actual brain networks and dynamics
 * WHY:  Verify oscillations emerge from network activity and coupling works
 * HOW:  Create brain networks, simulate activity, analyze oscillations
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 1.0
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
#include "core/brain_oscillations/nimcp_brain_oscillations.h"
#include "core/brain/nimcp_brain.h"
#include "core/synapse/nimcp_synapse.h"
#include "utils/memory/nimcp_memory.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class BrainOscillationIntegrationTest : public ::testing::Test {
protected:
    brain_t brain;
    brain_oscillation_analyzer_t* analyzer;

    static constexpr uint32_t NUM_NEURONS = 1000;
    static constexpr uint32_t WINDOW_SIZE_MS = 1000;
    static constexpr uint32_t SAMPLING_RATE_HZ = 250;
    static constexpr uint32_t SIMULATION_STEPS = 1000;
    static constexpr float PI = 3.14159265358979323846f;

    void SetUp() override {
        // Create brain with realistic network
        brain_config_t config = {0};
        config.num_neurons = NUM_NEURONS;
        config.max_synapses = NUM_NEURONS * 100;
        brain = brain_create(&config);
        ASSERT_NE(brain, nullptr);

        // Create analyzer
        analyzer = brain_oscillation_create(brain, WINDOW_SIZE_MS, SAMPLING_RATE_HZ);
        ASSERT_NE(analyzer, nullptr);
    }

    void TearDown() override {
        if (analyzer) {
            brain_oscillation_destroy(analyzer);
            analyzer = nullptr;
        }
        if (brain) {
            brain_destroy(brain);
            brain = nullptr;
        }
    }

    /**
     * @brief Create network with specific connectivity pattern
     */
    void createOscillatoryNetwork() {
        // Create recurrent connections for oscillations
        // This approximates an E-I network that can generate rhythms

        for (uint32_t i = 0; i < NUM_NEURONS / 2; i++) {
            // Excitatory neurons connect to nearby neurons
            for (int j = -10; j <= 10; j++) {
                if (j == 0) continue;
                int target = ((int)i + j + NUM_NEURONS) % NUM_NEURONS;

                synapse_config_t syn_config = {0};
                syn_config.pre_neuron_id = i;
                syn_config.post_neuron_id = (neuron_id_t)target;
                syn_config.weight = 0.05f;
                syn_config.delay_ms = 1.0f;

                brain_add_synapse(brain, &syn_config);
            }
        }

        // Inhibitory neurons for rhythm generation
        for (uint32_t i = NUM_NEURONS / 2; i < NUM_NEURONS; i++) {
            for (int j = -5; j <= 5; j++) {
                if (j == 0) continue;
                int target = ((int)i + j) % NUM_NEURONS;

                synapse_config_t syn_config = {0};
                syn_config.pre_neuron_id = i;
                syn_config.post_neuron_id = (neuron_id_t)target;
                syn_config.weight = -0.03f;
                syn_config.delay_ms = 1.0f;

                brain_add_synapse(brain, &syn_config);
            }
        }
    }

    /**
     * @brief Simulate network and record activity
     */
    void simulateAndRecord(uint32_t num_steps) {
        float dt = 0.001f;  // 1ms timestep

        for (uint32_t step = 0; step < num_steps; step++) {
            // Update brain dynamics
            brain_update(brain, dt);

            // Record activity every 4ms (250 Hz sampling)
            if (step % 4 == 0) {
                // Compute average firing rate
                float avg_activity = 0.0f;
                for (uint32_t i = 0; i < NUM_NEURONS; i++) {
                    neuron_state_t state;
                    if (brain_get_neuron_state(brain, i, &state)) {
                        avg_activity += state.membrane_potential > -50.0f ? 1.0f : 0.0f;
                    }
                }
                avg_activity /= NUM_NEURONS;

                brain_oscillation_record_value(analyzer, avg_activity);
            }
        }
    }
};

//=============================================================================
// Network Oscillation Tests
//=============================================================================

TEST_F(BrainOscillationIntegrationTest, NetworkGeneratesOscillations) {
    createOscillatoryNetwork();
    simulateAndRecord(SIMULATION_STEPS);

    brain_wave_power_t power;
    bool success = brain_oscillation_get_wave_power(analyzer, &power);
    EXPECT_TRUE(success);

    EXPECT_GT(power.total_power, 0.0f);
    EXPECT_GT(power.dominant_freq, 0.0f);
}

TEST_F(BrainOscillationIntegrationTest, NetworkSynchrony) {
    createOscillatoryNetwork();
    simulateAndRecord(SIMULATION_STEPS);

    float synchrony = brain_oscillation_compute_synchrony(analyzer);
    EXPECT_GE(synchrony, 0.0f);
    EXPECT_LE(synchrony, 1.0f);

    // Oscillatory network should have measurable synchrony
    EXPECT_GT(synchrony, 0.1f);
}

TEST_F(BrainOscillationIntegrationTest, NetworkCoherence) {
    createOscillatoryNetwork();
    simulateAndRecord(SIMULATION_STEPS);

    float coherence = brain_oscillation_compute_coherence(analyzer);
    EXPECT_GE(coherence, 0.0f);
    EXPECT_LE(coherence, 1.0f);
}

TEST_F(BrainOscillationIntegrationTest, FullAnalysisOnNetwork) {
    createOscillatoryNetwork();
    simulateAndRecord(SIMULATION_STEPS);

    oscillation_analysis_t results;
    bool success = brain_oscillation_analyze(analyzer, &results);
    EXPECT_TRUE(success);

    EXPECT_GT(results.wave_power.total_power, 0.0f);
    EXPECT_NE(results.state, COGNITIVE_STATE_UNKNOWN);
    EXPECT_GT(results.state_confidence, 0.0f);

    EXPECT_GE(results.synchrony, 0.0f);
    EXPECT_LE(results.synchrony, 1.0f);

    EXPECT_GE(results.coherence, 0.0f);
    EXPECT_LE(results.coherence, 1.0f);
}

//=============================================================================
// Cross-Frequency Coupling in Networks
//=============================================================================

TEST_F(BrainOscillationIntegrationTest, NetworkPAC) {
    createOscillatoryNetwork();
    simulateAndRecord(SIMULATION_STEPS * 2);  // Longer for PAC

    float pac = brain_oscillation_compute_pac(analyzer, BRAIN_WAVE_THETA, BRAIN_WAVE_GAMMA);
    EXPECT_GE(pac, 0.0f);
    EXPECT_LE(pac, 1.0f);
}

//=============================================================================
// State Transitions
//=============================================================================

TEST_F(BrainOscillationIntegrationTest, StateInference) {
    createOscillatoryNetwork();
    simulateAndRecord(SIMULATION_STEPS);

    cognitive_state_t state;
    float confidence;
    bool success = brain_oscillation_get_state(analyzer, &state, &confidence);
    EXPECT_TRUE(success);

    // State should be determined
    EXPECT_NE(state, COGNITIVE_STATE_UNKNOWN);
    EXPECT_GT(confidence, 0.0f);
    EXPECT_LE(confidence, 1.0f);
}

//=============================================================================
// Large Network Performance
//=============================================================================

TEST_F(BrainOscillationIntegrationTest, LargeNetworkPerformance) {
    // Test with larger network
    brain_destroy(brain);
    brain_oscillation_destroy(analyzer);

    brain_config_t config = {0};
    config.num_neurons = 10000;  // 10k neurons
    config.max_synapses = 100000;
    brain = brain_create(&config);
    ASSERT_NE(brain, nullptr);

    analyzer = brain_oscillation_create(brain, WINDOW_SIZE_MS, SAMPLING_RATE_HZ);
    ASSERT_NE(analyzer, nullptr);

    // Just verify analysis completes in reasonable time
    createOscillatoryNetwork();
    simulateAndRecord(500);  // Shorter simulation for large network

    oscillation_analysis_t results;
    bool success = brain_oscillation_analyze(analyzer, &results);
    EXPECT_TRUE(success);
}

//=============================================================================
// Stability Tests
//=============================================================================

TEST_F(BrainOscillationIntegrationTest, RepeatedAnalysis) {
    createOscillatoryNetwork();
    simulateAndRecord(SIMULATION_STEPS);

    // Run analysis multiple times - should be stable
    oscillation_analysis_t results1, results2, results3;

    EXPECT_TRUE(brain_oscillation_analyze(analyzer, &results1));
    EXPECT_TRUE(brain_oscillation_analyze(analyzer, &results2));
    EXPECT_TRUE(brain_oscillation_analyze(analyzer, &results3));

    // Results should be identical (same data)
    EXPECT_FLOAT_EQ(results1.synchrony, results2.synchrony);
    EXPECT_FLOAT_EQ(results2.synchrony, results3.synchrony);

    EXPECT_FLOAT_EQ(results1.coherence, results2.coherence);
    EXPECT_FLOAT_EQ(results2.coherence, results3.coherence);
}

//=============================================================================
// Long-Running Stability
//=============================================================================

TEST_F(BrainOscillationIntegrationTest, LongSimulation) {
    createOscillatoryNetwork();

    // Run longer simulation and verify continuous analysis
    for (int epoch = 0; epoch < 5; epoch++) {
        simulateAndRecord(500);

        oscillation_analysis_t results;
        bool success = brain_oscillation_analyze(analyzer, &results);
        EXPECT_TRUE(success);

        // Verify metrics stay in valid ranges
        EXPECT_GE(results.synchrony, 0.0f);
        EXPECT_LE(results.synchrony, 1.0f);
        EXPECT_GE(results.coherence, 0.0f);
        EXPECT_LE(results.coherence, 1.0f);
    }
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
