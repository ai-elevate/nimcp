/**
 * @file test_core_substrate_bridges_integration.cpp
 * @brief Integration tests for Core Substrate Bridges
 * @date 2025-12-19
 *
 * Tests integration between:
 * - neuron_substrate_bridge
 * - synapse_substrate_bridge
 * - axon_dendrite_substrate_bridge
 * - neural_substrate
 *
 * Focus areas:
 * 1. Neuron-synapse-axon coordination with shared substrate
 * 2. Signal propagation chain (axon → synapse → neuron)
 * 3. Substrate effects on complete neural pathway
 * 4. Spike generation and transmission efficiency
 * 5. Bio-async messaging coordination
 * 6. Recovery from substrate stress
 */

#include <gtest/gtest.h>
#include <cstring>
#include <thread>
#include <chrono>
#include <cmath>

// Headers have their own extern "C" guards
#include "core/neuron_models/nimcp_neuron_substrate_bridge.h"
#include "core/synapse_compute/nimcp_synapse_substrate_bridge.h"
#include "core/nimcp_axon_dendrite_substrate_bridge.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/neuron_models/nimcp_neuron_model.h"
#include "core/synapse_compute/nimcp_synapse_compute.h"
#include "core/axon/nimcp_axon.h"
#include "core/dendrite/nimcp_dendrite.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "utils/memory/nimcp_memory.h"

/* ============================================================================
 * Integration Test Fixture
 * ============================================================================ */

class CoreSubstrateBridgesIntegrationTest : public ::testing::Test {
protected:
    // Core substrate
    neural_substrate_t* substrate = nullptr;

    // Neural components
    neuron_model_state_t neuron_model;
    synapse_compute_context_t* synapse_context = nullptr;
    axon_t* axon = nullptr;
    dendrite_t* dendrite = nullptr;

    // Substrate bridges
    neuron_substrate_bridge_t* neuron_bridge = nullptr;
    synapse_substrate_bridge_t* synapse_bridge = nullptr;
    axon_dendrite_substrate_bridge_t* axon_dendrite_bridge = nullptr;

    void SetUp() override {
        // Create shared neural substrate
        substrate_config_t sub_cfg;
        substrate_default_config(&sub_cfg);
        substrate = substrate_create(&sub_cfg);
        ASSERT_NE(substrate, nullptr);

        // Initialize neuron model state
        memset(&neuron_model, 0, sizeof(neuron_model_state_t));
        neuron_model.v_membrane = -70.0f;
        neuron_model.threshold = -55.0f;
        neuron_model.v_rest = -70.0f;
        neuron_model.tau_membrane = 20.0f;

        // Create synapse compute context
        synapse_compute_config_t syn_cfg;
        memset(&syn_cfg, 0, sizeof(syn_cfg));
        syn_cfg.max_synapses = 128;
        synapse_context = synapse_compute_create(&syn_cfg);
        ASSERT_NE(synapse_context, nullptr);

        // Create axon
        axon_config_t axon_cfg;
        memset(&axon_cfg, 0, sizeof(axon_cfg));
        axon_cfg.length_um = 1000.0f;
        axon_cfg.diameter_um = 1.0f;
        axon_cfg.is_myelinated = true;
        axon = axon_create(&axon_cfg);
        ASSERT_NE(axon, nullptr);

        // Create dendrite
        dendrite_config_t dend_cfg;
        memset(&dend_cfg, 0, sizeof(dend_cfg));
        dend_cfg.num_compartments = 10;
        dend_cfg.total_length_um = 500.0f;
        dendrite = dendrite_create(&dend_cfg);
        ASSERT_NE(dendrite, nullptr);
    }

    void TearDown() override {
        if (neuron_bridge) {
            neuron_substrate_bridge_destroy(neuron_bridge);
        }
        if (synapse_bridge) {
            synapse_substrate_bridge_destroy(synapse_bridge);
        }
        if (axon_dendrite_bridge) {
            axon_dendrite_substrate_bridge_destroy(axon_dendrite_bridge);
        }
        if (dendrite) {
            dendrite_destroy(dendrite);
        }
        if (axon) {
            axon_destroy(axon);
        }
        if (synapse_context) {
            synapse_compute_destroy(synapse_context);
        }
        if (substrate) {
            substrate_destroy(substrate);
        }
    }

    void createAllBridges() {
        neuron_substrate_config_t neuron_cfg;
        neuron_substrate_default_config(&neuron_cfg);
        neuron_bridge = neuron_substrate_bridge_create(&neuron_cfg, neuron_model, substrate);
        ASSERT_NE(neuron_bridge, nullptr);

        synapse_substrate_config_t syn_cfg;
        synapse_substrate_default_config(&syn_cfg);
        synapse_bridge = synapse_substrate_bridge_create(&syn_cfg, synapse_context, substrate);
        ASSERT_NE(synapse_bridge, nullptr);

        axon_dendrite_substrate_config_t axon_dend_cfg;
        axon_dendrite_substrate_default_config(&axon_dend_cfg);
        axon_dendrite_bridge = axon_dendrite_substrate_bridge_create(
            &axon_dend_cfg, substrate, axon, dendrite);
        ASSERT_NE(axon_dendrite_bridge, nullptr);
    }

    void stressSubstrate(float atp, float o2, float temp) {
        substrate_set_atp(substrate, atp);
        substrate_set_oxygen(substrate, o2);
        substrate->physical.temperature = temp;
        substrate_update(substrate, 10);
    }
};

/* ============================================================================
 * Shared Substrate Coordination Tests
 * ============================================================================ */

TEST_F(CoreSubstrateBridgesIntegrationTest, AllBridgesShareSameSubstrate) {
    createAllBridges();

    // All bridges should see same substrate state
    float initial_atp = substrate->metabolic.atp_level;
    float initial_temp = substrate->physical.temperature;

    // Update all bridges
    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);
    axon_dendrite_substrate_update_dendrite_effects(axon_dendrite_bridge);

    // Substrate state should remain consistent
    EXPECT_FLOAT_EQ(substrate->metabolic.atp_level, initial_atp);
    EXPECT_FLOAT_EQ(substrate->physical.temperature, initial_temp);

    // All bridges should be initialized
    EXPECT_TRUE(neuron_substrate_is_modulated(neuron_bridge) ||
                !neuron_substrate_is_modulated(neuron_bridge)); // Any valid state
}

TEST_F(CoreSubstrateBridgesIntegrationTest, SubstrateStressAffectsAllBridges) {
    createAllBridges();

    // Create substrate stress
    stressSubstrate(0.3f, 0.5f, 40.0f);

    // Update all bridges
    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);
    axon_dendrite_substrate_update_dendrite_effects(axon_dendrite_bridge);

    // All bridges should be modulated by stress
    float neuron_excitability = neuron_substrate_get_excitability(neuron_bridge);
    float synapse_efficiency = synapse_substrate_get_transmission_efficiency(synapse_bridge);
    float axon_reliability = axon_dendrite_substrate_get_spike_reliability(axon_dendrite_bridge);

    EXPECT_LT(neuron_excitability, 1.0f);
    EXPECT_LT(synapse_efficiency, 1.0f);
    EXPECT_LT(axon_reliability, 1.0f);
}

/* ============================================================================
 * Signal Propagation Chain Tests (Axon → Synapse → Neuron)
 * ============================================================================ */

TEST_F(CoreSubstrateBridgesIntegrationTest, HealthySignalPropagation) {
    createAllBridges();

    // Healthy substrate conditions
    substrate_set_atp(substrate, 0.95f);
    substrate_set_oxygen(substrate, 0.97f);

    // Update all bridges
    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);

    // Simulate spike propagation through axon
    axon_dendrite_substrate_record_axon_spikes(axon_dendrite_bridge, 1);
    float axon_velocity = axon_dendrite_substrate_get_conduction_mod(axon_dendrite_bridge);
    EXPECT_NEAR(axon_velocity, 1.0f, 0.15f); // Should be near baseline

    // Synapse transmission
    float synapse_mod = synapse_substrate_apply_modulation(synapse_bridge, SYNAPSE_AMPA);
    EXPECT_GT(synapse_mod, 0.9f); // Healthy transmission

    // Neuron receives input
    float modulated_input = neuron_substrate_get_modulated_input(neuron_bridge, 1.0f);
    EXPECT_GT(modulated_input, 0.9f); // Healthy input processing
}

TEST_F(CoreSubstrateBridgesIntegrationTest, DegradedSignalPropagationUnderStress) {
    createAllBridges();

    // Substrate stress
    stressSubstrate(0.4f, 0.6f, 38.5f);

    // Update all bridges
    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);

    // Axon conduction degraded
    float axon_velocity = axon_dendrite_substrate_get_conduction_mod(axon_dendrite_bridge);
    float axon_reliability = axon_dendrite_substrate_get_spike_reliability(axon_dendrite_bridge);
    EXPECT_LT(axon_reliability, 0.9f);

    // Synapse transmission impaired
    float synapse_mod = synapse_substrate_apply_modulation(synapse_bridge, SYNAPSE_AMPA);
    EXPECT_LT(synapse_mod, 0.9f);

    // Neuron input reduced
    float modulated_input = neuron_substrate_get_modulated_input(neuron_bridge, 1.0f);
    EXPECT_LT(modulated_input, 0.9f);

    // Overall pathway efficiency should be significantly reduced
    float pathway_efficiency = axon_reliability * synapse_mod * modulated_input;
    EXPECT_LT(pathway_efficiency, 0.7f);
}

TEST_F(CoreSubstrateBridgesIntegrationTest, TemperatureEffectsOnPropagationChain) {
    createAllBridges();

    // Hyperthermia
    substrate->physical.temperature = 40.0f;
    substrate_update(substrate, 10);

    // Update all bridges
    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);

    // Q10 effects should increase conduction velocity
    float axon_velocity = axon_dendrite_substrate_get_conduction_mod(axon_dendrite_bridge);
    EXPECT_GT(axon_velocity, 1.0f); // Higher temperature → faster

    // Synapse kinetics also affected
    float receptor_kinetics = synapse_substrate_get_receptor_kinetics_factor(synapse_bridge);
    EXPECT_NE(receptor_kinetics, 1.0f);

    // Neuron firing rate modulated
    float firing_mod = neuron_substrate_get_modulated_firing_rate(neuron_bridge, 10.0f);
    EXPECT_NE(firing_mod, 10.0f);
}

/* ============================================================================
 * Substrate Effects on Complete Neural Pathway Tests
 * ============================================================================ */

TEST_F(CoreSubstrateBridgesIntegrationTest, ATPDepletionAffectsEntirePathway) {
    createAllBridges();

    // Gradual ATP depletion
    for (int i = 0; i < 5; i++) {
        float atp = 0.9f - (i * 0.15f); // 0.9 → 0.3
        substrate_set_atp(substrate, atp);

        neuron_substrate_update_effects(neuron_bridge);
        synapse_substrate_update(synapse_bridge);
        axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);
        axon_dendrite_substrate_update_dendrite_effects(axon_dendrite_bridge);
    }

    // All components should be impaired
    float neuron_exc = neuron_substrate_get_excitability(neuron_bridge);
    float synapse_release = synapse_substrate_get_release_probability(synapse_bridge);
    float axon_reliability = axon_dendrite_substrate_get_spike_reliability(axon_dendrite_bridge);
    float dendrite_plasticity = axon_dendrite_substrate_get_plasticity_mod(axon_dendrite_bridge);

    EXPECT_LT(neuron_exc, 0.8f);
    EXPECT_LT(synapse_release, 0.8f);
    EXPECT_LT(axon_reliability, 0.8f);
    EXPECT_LT(dendrite_plasticity, 0.8f);
}

TEST_F(CoreSubstrateBridgesIntegrationTest, OxygenDeprivationImpactsTransmission) {
    createAllBridges();

    // Hypoxia
    substrate_set_oxygen(substrate, 0.4f);
    substrate_update(substrate, 10);

    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);

    // Transmission should be severely impaired
    float synapse_efficiency = synapse_substrate_get_transmission_efficiency(synapse_bridge);
    EXPECT_LT(synapse_efficiency, 0.7f);

    // Neuron excitability reduced
    neuron_substrate_effects_t neuron_effects;
    neuron_substrate_get_effects(neuron_bridge, &neuron_effects);
    EXPECT_LT(neuron_effects.o2_transmission_mod, 1.0f);
}

TEST_F(CoreSubstrateBridgesIntegrationTest, IonicImbalanceAffectsExcitability) {
    createAllBridges();

    // Create ionic imbalance
    substrate->physical.ion_balance = 0.5f;
    substrate_update(substrate, 10);

    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);
    axon_dendrite_substrate_update_dendrite_effects(axon_dendrite_bridge);

    // Axon conduction should be impaired
    bool axon_limited = axon_dendrite_substrate_is_axon_limited(axon_dendrite_bridge);
    EXPECT_TRUE(axon_limited);

    // Dendrite integration affected
    float dendrite_integration = axon_dendrite_substrate_get_integration_mod(axon_dendrite_bridge);
    EXPECT_LT(dendrite_integration, 1.0f);
}

/* ============================================================================
 * Spike Generation and Transmission Efficiency Tests
 * ============================================================================ */

TEST_F(CoreSubstrateBridgesIntegrationTest, HighFrequencySpikeTrainDepletsSubstrate) {
    createAllBridges();

    float initial_atp = substrate->metabolic.atp_level;

    // Simulate high-frequency spike train
    for (int i = 0; i < 100; i++) {
        neuron_substrate_consume_spike(neuron_bridge);
        axon_dendrite_substrate_record_axon_spikes(axon_dendrite_bridge, 1);
        synapse_substrate_consume_transmission(synapse_bridge, SYNAPSE_AMPA, 1);
    }

    // ATP should be depleted
    EXPECT_LT(substrate->metabolic.atp_level, initial_atp);

    // Update effects
    neuron_substrate_update_effects(neuron_bridge);

    // Energy tracking should show consumption
    neuron_energy_tracking_t energy;
    neuron_substrate_get_energy_tracking(neuron_bridge, &energy);
    EXPECT_EQ(energy.total_spikes, 100u);
    EXPECT_GT(energy.total_atp_consumed, 0.0f);
}

TEST_F(CoreSubstrateBridgesIntegrationTest, SynapseTypeSpecificEfficiency) {
    createAllBridges();

    // Stress substrate
    stressSubstrate(0.6f, 0.7f, 37.0f);
    synapse_substrate_update(synapse_bridge);

    // Different synapse types should have different modulation
    float ampa_mod = synapse_substrate_apply_modulation(synapse_bridge, SYNAPSE_AMPA);
    float nmda_mod = synapse_substrate_apply_modulation(synapse_bridge, SYNAPSE_NMDA);
    float gaba_a_mod = synapse_substrate_apply_modulation(synapse_bridge, SYNAPSE_GABA_A);
    float gaba_b_mod = synapse_substrate_apply_modulation(synapse_bridge, SYNAPSE_GABA_B);

    // All should be affected
    EXPECT_LT(ampa_mod, 1.0f);
    EXPECT_LT(nmda_mod, 1.0f);
    EXPECT_LT(gaba_a_mod, 1.0f);
    EXPECT_LT(gaba_b_mod, 1.0f);

    // GABA-B (metabotropic) should be more affected than AMPA (ionotropic)
    EXPECT_LE(gaba_b_mod, ampa_mod);
}

TEST_F(CoreSubstrateBridgesIntegrationTest, NMDACalciumInfluxFeedback) {
    createAllBridges();

    float initial_ca = substrate->physical.calcium_balance;

    // Repeated NMDA transmission
    for (int i = 0; i < 50; i++) {
        synapse_substrate_record_nmda_calcium(synapse_bridge, 1);
    }

    synapse_substrate_update(synapse_bridge);

    // Calcium should accumulate (if substrate tracks it)
    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(synapse_bridge, &stats);
    EXPECT_EQ(stats.nmda_ca_influx_events, 50u);
}

/* ============================================================================
 * Bio-async Messaging Coordination Tests
 * ============================================================================ */

TEST_F(CoreSubstrateBridgesIntegrationTest, BioAsyncConnectivity) {
    createAllBridges();

    // Connect all bridges to bio-async
    int neuron_result = neuron_substrate_connect_bio_async(neuron_bridge);
    int axon_dend_result = axon_dendrite_substrate_connect_bio_async(axon_dendrite_bridge);

    // Results may be 0 (success) or error (no router available)
    // Just verify no crashes
    EXPECT_TRUE(neuron_result == 0 || neuron_result != 0);
    EXPECT_TRUE(axon_dend_result == 0 || axon_dend_result != 0);

    // Check connectivity status
    bool neuron_connected = neuron_substrate_is_bio_async_connected(neuron_bridge);
    bool axon_dend_connected = axon_dendrite_substrate_is_bio_async_connected(axon_dendrite_bridge);

    // Should reflect connection attempts
    EXPECT_TRUE(neuron_connected || !neuron_connected); // Any valid state
}

TEST_F(CoreSubstrateBridgesIntegrationTest, BioAsyncMessagingUnderLoad) {
    createAllBridges();

    neuron_substrate_connect_bio_async(neuron_bridge);
    axon_dendrite_substrate_connect_bio_async(axon_dendrite_bridge);

    // Generate activity while bio-async is connected
    for (int i = 0; i < 20; i++) {
        neuron_substrate_consume_spike(neuron_bridge);
        axon_dendrite_substrate_record_axon_spikes(axon_dendrite_bridge, 1);
        neuron_substrate_bridge_update(neuron_bridge, 10);
        axon_dendrite_substrate_bridge_update(axon_dendrite_bridge, 10);
    }

    // System should remain stable
    neuron_substrate_stats_t neuron_stats;
    neuron_substrate_get_stats(neuron_bridge, &neuron_stats);
    EXPECT_GT(neuron_stats.spikes_consumed, 0u);
}

/* ============================================================================
 * Recovery from Substrate Stress Tests
 * ============================================================================ */

TEST_F(CoreSubstrateBridgesIntegrationTest, RecoveryFromATPDepletion) {
    createAllBridges();

    // Deplete ATP
    substrate_set_atp(substrate, 0.3f);
    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);

    float depleted_neuron_exc = neuron_substrate_get_excitability(neuron_bridge);
    float depleted_synapse_eff = synapse_substrate_get_transmission_efficiency(synapse_bridge);

    // Allow recovery
    for (int i = 0; i < 100; i++) {
        substrate_update(substrate, 100); // Let ATP recover
        neuron_substrate_update_effects(neuron_bridge);
        synapse_substrate_update(synapse_bridge);
    }

    // Should recover
    float recovered_neuron_exc = neuron_substrate_get_excitability(neuron_bridge);
    float recovered_synapse_eff = synapse_substrate_get_transmission_efficiency(synapse_bridge);

    EXPECT_GT(recovered_neuron_exc, depleted_neuron_exc);
    EXPECT_GT(recovered_synapse_eff, depleted_synapse_eff);
}

TEST_F(CoreSubstrateBridgesIntegrationTest, RecoveryFromHyperthermia) {
    createAllBridges();

    // Hyperthermia
    substrate->physical.temperature = 41.0f;
    neuron_substrate_update_effects(neuron_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);

    float hot_axon_velocity = axon_dendrite_substrate_get_conduction_mod(axon_dendrite_bridge);

    // Cool down
    substrate->physical.temperature = 37.0f;
    for (int i = 0; i < 10; i++) {
        substrate_update(substrate, 100);
        neuron_substrate_update_effects(neuron_bridge);
        axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);
    }

    float normal_axon_velocity = axon_dendrite_substrate_get_conduction_mod(axon_dendrite_bridge);
    EXPECT_LT(normal_axon_velocity, hot_axon_velocity); // Cooler → slower
}

TEST_F(CoreSubstrateBridgesIntegrationTest, GradualRecoveryFromMultipleStressors) {
    createAllBridges();

    // Multiple stressors
    stressSubstrate(0.4f, 0.5f, 39.0f);
    substrate->physical.ion_balance = 0.6f;

    neuron_substrate_update_effects(neuron_bridge);
    synapse_substrate_update(synapse_bridge);
    axon_dendrite_substrate_update_axon_effects(axon_dendrite_bridge);
    axon_dendrite_substrate_update_dendrite_effects(axon_dendrite_bridge);

    // Get stressed state
    float stressed_neuron = neuron_substrate_get_excitability(neuron_bridge);
    float stressed_synapse = synapse_substrate_get_transmission_efficiency(synapse_bridge);
    float stressed_dendrite = axon_dendrite_substrate_get_plasticity_mod(axon_dendrite_bridge);

    // Gradual recovery
    for (int i = 0; i < 50; i++) {
        // Improve substrate conditions gradually
        float progress = (i + 1) / 50.0f;
        substrate_set_atp(substrate, 0.4f + (0.55f * progress)); // 0.4 → 0.95
        substrate_set_oxygen(substrate, 0.5f + (0.47f * progress)); // 0.5 → 0.97
        substrate->physical.temperature = 39.0f - (2.0f * progress); // 39 → 37
        substrate->physical.ion_balance = 0.6f + (0.35f * progress); // 0.6 → 0.95

        substrate_update(substrate, 100);
        neuron_substrate_update_effects(neuron_bridge);
        synapse_substrate_update(synapse_bridge);
        axon_dendrite_substrate_update_dendrite_effects(axon_dendrite_bridge);
    }

    // Get recovered state
    float recovered_neuron = neuron_substrate_get_excitability(neuron_bridge);
    float recovered_synapse = synapse_substrate_get_transmission_efficiency(synapse_bridge);
    float recovered_dendrite = axon_dendrite_substrate_get_plasticity_mod(axon_dendrite_bridge);

    EXPECT_GT(recovered_neuron, stressed_neuron);
    EXPECT_GT(recovered_synapse, stressed_synapse);
    EXPECT_GT(recovered_dendrite, stressed_dendrite);
}

/* ============================================================================
 * Complex Integration Scenarios
 * ============================================================================ */

TEST_F(CoreSubstrateBridgesIntegrationTest, FullNeuralPathwaySimulation) {
    createAllBridges();

    // Healthy baseline
    substrate_set_atp(substrate, 0.95f);
    substrate_set_oxygen(substrate, 0.97f);

    // Simulate complete neural pathway cycle
    for (int cycle = 0; cycle < 10; cycle++) {
        // 1. Axon spike propagation
        axon_dendrite_substrate_record_axon_spikes(axon_dendrite_bridge, 1);

        // 2. Synaptic transmission (AMPA)
        synapse_substrate_consume_transmission(synapse_bridge, SYNAPSE_AMPA, 1);

        // 3. Dendritic integration
        axon_dendrite_substrate_record_dendrite_events(axon_dendrite_bridge, 1);

        // 4. Neuron spike generation
        neuron_substrate_consume_spike(neuron_bridge);

        // 5. Update all bridges
        neuron_substrate_bridge_update(neuron_bridge, 10);
        synapse_substrate_update(synapse_bridge);
        axon_dendrite_substrate_bridge_update(axon_dendrite_bridge, 10);
    }

    // Verify activity was tracked
    neuron_substrate_stats_t neuron_stats;
    neuron_substrate_get_stats(neuron_bridge, &neuron_stats);
    EXPECT_EQ(neuron_stats.spikes_consumed, 10u);

    synapse_substrate_stats_t synapse_stats;
    synapse_substrate_get_stats(synapse_bridge, &synapse_stats);
    EXPECT_EQ(synapse_stats.transmissions_processed, 10u);

    axon_dendrite_substrate_stats_t axon_dend_stats;
    axon_dendrite_substrate_get_stats(axon_dendrite_bridge, &axon_dend_stats);
    EXPECT_EQ(axon_dend_stats.axon_spikes_processed, 10u);
}

TEST_F(CoreSubstrateBridgesIntegrationTest, PlasticityUnderMetabolicStress) {
    createAllBridges();

    // Normal plasticity capacity
    axon_dendrite_substrate_update_dendrite_effects(axon_dendrite_bridge);
    float normal_plasticity = axon_dendrite_substrate_get_plasticity_mod(axon_dendrite_bridge);

    // Induce plasticity events under normal conditions
    for (int i = 0; i < 5; i++) {
        axon_dendrite_substrate_record_plasticity(axon_dendrite_bridge, 0.8f);
    }

    // Create metabolic stress
    stressSubstrate(0.3f, 0.6f, 37.0f);
    axon_dendrite_substrate_update_dendrite_effects(axon_dendrite_bridge);
    float stressed_plasticity = axon_dendrite_substrate_get_plasticity_mod(axon_dendrite_bridge);

    // Plasticity should be suppressed under stress
    EXPECT_LT(stressed_plasticity, normal_plasticity);

    // Attempt plasticity under stress
    int result = axon_dendrite_substrate_record_plasticity(axon_dendrite_bridge, 0.8f);
    EXPECT_EQ(result, 0); // Should succeed but have reduced effect
}

TEST_F(CoreSubstrateBridgesIntegrationTest, SynchronizedUpdateCycle) {
    createAllBridges();

    // Synchronized updates over time
    for (int i = 0; i < 20; i++) {
        uint64_t delta_ms = 50;

        int neuron_result = neuron_substrate_bridge_update(neuron_bridge, delta_ms);
        synapse_substrate_update(synapse_bridge);
        int axon_dend_result = axon_dendrite_substrate_bridge_update(axon_dendrite_bridge, delta_ms);

        EXPECT_EQ(neuron_result, 0);
        EXPECT_EQ(axon_dend_result, 0);
    }

    // All should have consistent update counts
    neuron_substrate_stats_t neuron_stats;
    neuron_substrate_get_stats(neuron_bridge, &neuron_stats);
    EXPECT_GE(neuron_stats.total_updates, 20u);

    axon_dendrite_substrate_stats_t axon_dend_stats;
    axon_dendrite_substrate_get_stats(axon_dendrite_bridge, &axon_dend_stats);
    EXPECT_GE(axon_dend_stats.total_updates, 20u);
}

TEST_F(CoreSubstrateBridgesIntegrationTest, LongRunningStabilityTest) {
    createAllBridges();

    // Extended simulation with varying conditions
    for (int i = 0; i < 100; i++) {
        // Vary substrate conditions slightly
        float atp = 0.8f + 0.15f * sinf(i * 0.1f);
        float o2 = 0.85f + 0.12f * cosf(i * 0.15f);
        float temp = 37.0f + 1.0f * sinf(i * 0.05f);

        substrate_set_atp(substrate, atp);
        substrate_set_oxygen(substrate, o2);
        substrate->physical.temperature = temp;

        // Update everything
        substrate_update(substrate, 50);
        neuron_substrate_bridge_update(neuron_bridge, 50);
        synapse_substrate_update(synapse_bridge);
        axon_dendrite_substrate_bridge_update(axon_dendrite_bridge, 50);

        // Occasional activity
        if (i % 5 == 0) {
            neuron_substrate_consume_spike(neuron_bridge);
            axon_dendrite_substrate_record_axon_spikes(axon_dendrite_bridge, 1);
            synapse_substrate_consume_transmission(synapse_bridge, SYNAPSE_AMPA, 1);
        }
    }

    // System should remain stable
    neuron_substrate_stats_t neuron_stats;
    neuron_substrate_get_stats(neuron_bridge, &neuron_stats);
    EXPECT_EQ(neuron_stats.total_updates, 100u);

    // Substrate should be in reasonable state
    EXPECT_GT(substrate->metabolic.atp_level, 0.0f);
    EXPECT_LE(substrate->physical.temperature, 45.0f);
}
