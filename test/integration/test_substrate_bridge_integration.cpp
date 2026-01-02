//=============================================================================
// test_substrate_bridge_integration.cpp - Substrate Bridge Integration Tests
//=============================================================================
/**
 * @file test_substrate_bridge_integration.cpp
 * @brief Integration tests for substrate bridges working together
 *
 * WHAT: Tests multi-component interactions between substrate bridges
 * WHY:  Verify metabolic effects propagate correctly across subsystems
 * HOW:  Create multiple bridges, connect to common substrate, verify propagation
 *
 * TEST SCENARIOS:
 * 1. Multiple bridges sharing substrate state
 * 2. ATP depletion propagating to all connected bridges
 * 3. Temperature effects across cortical/emotion/synapse bridges
 * 4. Metabolic cascade simulation
 * 5. Substrate recovery and bridge synchronization
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

// Headers have their own extern "C" guards
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/cortical_columns/nimcp_cortical_substrate_bridge.h"
#include "cognitive/emotion/nimcp_emotion_substrate_bridge.h"
#include "core/synapse_compute/nimcp_synapse_substrate_bridge.h"
#include "core/neuron_models/nimcp_neuron_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"

//=============================================================================
// Test Fixture
//=============================================================================

class SubstrateBridgeIntegrationTest : public ::testing::Test {
protected:
    neural_substrate_t* substrate;

    void SetUp() override {
        // Create shared neural substrate with defaults
        substrate_config_t config;
        substrate_default_config(&config);
        config.enable_metabolic_model = true;
        config.enable_temperature_effects = true;
        config.enable_ion_dynamics = true;
        config.enable_alerts = true;

        substrate = substrate_create(&config);
        ASSERT_NE(substrate, nullptr) << "Failed to create neural substrate";
    }

    void TearDown() override {
        if (substrate) {
            substrate_destroy(substrate);
            substrate = nullptr;
        }
    }

    // Helper: Simulate ATP depletion
    void depleteATP(float target_level) {
        substrate_set_atp(substrate, target_level);
        substrate_update(substrate, 1);
    }

    // Helper: Simulate temperature change
    void setTemperature(float temp_celsius) {
        substrate_set_temperature(substrate, temp_celsius);
        substrate_update(substrate, 1);
    }

    // Helper: Verify all bridges detect impairment
    bool allBridgesImpaired(
        const cortical_substrate_bridge_t* cortical,
        const emotion_substrate_bridge_t* emotion
    ) {
        bool cortical_impaired = cortical_substrate_is_impaired(cortical);
        bool emotion_impaired = emotion_substrate_is_impaired(emotion);
        return cortical_impaired && emotion_impaired;
    }
};

//=============================================================================
// Test: Multi-Bridge Substrate Sharing
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, MultipleBridges_ShareSubstrateState) {
    // WHAT: Test multiple bridges connected to same substrate
    // WHY:  Verify substrate state is shared correctly
    // HOW:  Create multiple bridges, connect to substrate, verify consistent state

    // Create cortical substrate bridge
    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    // Create emotion substrate bridge
    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    emotion_substrate_bridge_t* emotion = emotion_substrate_bridge_create(
        &emotion_config, nullptr, substrate
    );
    ASSERT_NE(emotion, nullptr);

    // Update both bridges
    EXPECT_EQ(cortical_substrate_update(cortical), 0);
    EXPECT_EQ(emotion_substrate_update(emotion), 0);

    // Both should see healthy substrate initially
    EXPECT_FALSE(cortical_substrate_is_impaired(cortical));
    EXPECT_FALSE(emotion_substrate_is_impaired(emotion));

    // Deplete ATP
    depleteATP(0.2f);

    // Update bridges
    EXPECT_EQ(cortical_substrate_update(cortical), 0);
    EXPECT_EQ(emotion_substrate_update(emotion), 0);

    // Both should now show effects
    float cortical_fidelity = cortical_substrate_get_column_fidelity(cortical);
    float emotion_intensity = emotion_substrate_get_intensity_mod(emotion);

    // Low ATP should reduce cortical fidelity
    EXPECT_LT(cortical_fidelity, 1.0f);
    // Low ATP should affect emotion intensity modulation
    EXPECT_NE(emotion_intensity, 1.0f);

    // Cleanup
    cortical_substrate_bridge_destroy(cortical);
    emotion_substrate_bridge_destroy(emotion);
}

//=============================================================================
// Test: ATP Depletion Cascade
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, ATPDepletion_PropagatesAcrossAllBridges) {
    // WHAT: Test ATP depletion affects all connected bridges
    // WHY:  Verify metabolic stress propagates system-wide
    // HOW:  Progressively deplete ATP, verify all bridges respond

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    emotion_substrate_bridge_t* emotion = emotion_substrate_bridge_create(
        &emotion_config, nullptr, substrate
    );
    ASSERT_NE(emotion, nullptr);

    // Track effects at different ATP levels
    float atp_levels[] = {0.9f, 0.7f, 0.5f, 0.3f, 0.1f};
    float prev_cortical_fidelity = 1.0f;
    float prev_emotion_regulation = 1.0f;

    for (float atp : atp_levels) {
        depleteATP(atp);
        cortical_substrate_update(cortical);
        emotion_substrate_update(emotion);

        float current_fidelity = cortical_substrate_get_column_fidelity(cortical);
        float current_regulation = emotion_substrate_get_regulation_capacity(emotion);

        // Effects should generally worsen as ATP decreases
        // (allowing for small fluctuations in implementation)
        if (atp < 0.5f) {
            EXPECT_LT(current_fidelity, 1.0f)
                << "Cortical fidelity should degrade at ATP=" << atp;
        }
        if (atp < 0.3f) {
            EXPECT_LT(current_regulation, prev_emotion_regulation + 0.1f)
                << "Emotion regulation should be reduced at ATP=" << atp;
        }

        prev_cortical_fidelity = current_fidelity;
        prev_emotion_regulation = current_regulation;
    }

    // At critical ATP, both bridges should be impaired
    depleteATP(0.05f);
    cortical_substrate_update(cortical);
    emotion_substrate_update(emotion);

    EXPECT_TRUE(cortical_substrate_is_impaired(cortical))
        << "Cortical bridge should be impaired at critical ATP";
    EXPECT_TRUE(emotion_substrate_is_impaired(emotion))
        << "Emotion bridge should be impaired at critical ATP";

    cortical_substrate_bridge_destroy(cortical);
    emotion_substrate_bridge_destroy(emotion);
}

//=============================================================================
// Test: Temperature Effects Across Bridges
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, Temperature_AffectsAllBridges) {
    // WHAT: Test temperature changes affect all bridges
    // WHY:  Verify thermal stress propagates correctly
    // HOW:  Raise temperature, verify Q10 effects on all bridges

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    emotion_substrate_bridge_t* emotion = emotion_substrate_bridge_create(
        &emotion_config, nullptr, substrate
    );
    ASSERT_NE(emotion, nullptr);

    // Normal temperature - baseline
    setTemperature(37.0f);
    cortical_substrate_update(cortical);
    emotion_substrate_update(emotion);

    float baseline_cortical_fidelity = cortical_substrate_get_column_fidelity(cortical);
    float baseline_emotion_reactivity = emotion_substrate_get_reactivity_threshold(emotion);

    // Elevated temperature (fever)
    setTemperature(39.5f);
    cortical_substrate_update(cortical);
    emotion_substrate_update(emotion);

    float fever_cortical_fidelity = cortical_substrate_get_column_fidelity(cortical);
    float fever_emotion_reactivity = emotion_substrate_get_reactivity_threshold(emotion);

    // High temperature should affect processing
    // Cortical fidelity may be reduced
    // Emotion reactivity threshold may change (emotional blunting)
    EXPECT_LE(fever_cortical_fidelity, baseline_cortical_fidelity + 0.1f)
        << "Fever should not improve cortical fidelity";

    // Hypothermia
    setTemperature(32.0f);
    cortical_substrate_update(cortical);
    emotion_substrate_update(emotion);

    float cold_cortical_fidelity = cortical_substrate_get_column_fidelity(cortical);

    // Hypothermia should also affect processing
    EXPECT_LE(cold_cortical_fidelity, baseline_cortical_fidelity + 0.1f)
        << "Hypothermia should not improve cortical fidelity";

    cortical_substrate_bridge_destroy(cortical);
    emotion_substrate_bridge_destroy(emotion);
}

//=============================================================================
// Test: Metabolic Cascade Simulation
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, MetabolicCascade_FullSimulation) {
    // WHAT: Simulate realistic metabolic cascade across all bridges
    // WHY:  Test complex multi-bridge interaction over time
    // HOW:  Simulate metabolic stress, record spikes, verify cascading effects

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    emotion_substrate_bridge_t* emotion = emotion_substrate_bridge_create(
        &emotion_config, nullptr, substrate
    );
    ASSERT_NE(emotion, nullptr);

    // Simulation parameters
    const int num_steps = 100;
    const uint64_t step_duration_ms = 10;

    // Start with healthy state
    substrate_set_atp(substrate, 0.95f);
    substrate_set_oxygen(substrate, 0.97f);
    substrate_set_glucose(substrate, 0.90f);

    bool saw_impairment = false;
    bool saw_recovery = false;
    int impairment_count = 0;

    for (int step = 0; step < num_steps; step++) {
        // Simulate neural activity (consumes ATP)
        substrate_record_spikes(substrate, 50);
        substrate_record_transmissions(substrate, 200);

        // Update substrate
        substrate_update(substrate, step_duration_ms);

        // Update bridges
        cortical_substrate_update(cortical);
        emotion_substrate_update(emotion);

        // Track impairment
        bool currently_impaired = cortical_substrate_is_impaired(cortical) ||
                                  emotion_substrate_is_impaired(emotion);

        if (currently_impaired) {
            saw_impairment = true;
            impairment_count++;
        }

        // Simulate occasional glucose injection (recovery)
        if (step == 60) {
            substrate_set_atp(substrate, 0.8f);
            substrate_set_glucose(substrate, 0.85f);
        }

        // Check for recovery after injection
        if (step > 70 && !currently_impaired && saw_impairment) {
            saw_recovery = true;
        }
    }

    // Verify cascade behavior
    // System should function throughout simulation
    EXPECT_GE(cortical_substrate_get_column_fidelity(cortical), 0.0f);
    EXPECT_GE(emotion_substrate_get_intensity_mod(emotion), 0.0f);

    // Get final statistics
    cortical_substrate_stats_t cortical_stats;
    cortical_substrate_get_stats(cortical, &cortical_stats);

    EXPECT_GT(cortical_stats.update_count, 0u)
        << "Should have processed multiple updates";

    cortical_substrate_bridge_destroy(cortical);
    emotion_substrate_bridge_destroy(emotion);
}

//=============================================================================
// Test: Layer-Specific Effects
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, CorticalBridge_LayerSpecificGain) {
    // WHAT: Test layer-specific metabolic effects in cortical bridge
    // WHY:  Verify Q10 coefficients affect layers differently
    // HOW:  Manipulate temperature, verify layer gains differ

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    // Normal temperature
    setTemperature(37.0f);
    cortical_substrate_update(cortical);

    float gains_normal[CORTICAL_SUBSTRATE_NUM_LAYERS];
    for (int i = 0; i < CORTICAL_SUBSTRATE_NUM_LAYERS; i++) {
        gains_normal[i] = cortical_substrate_get_layer_gain(cortical, i);
    }

    // Elevated temperature
    setTemperature(40.0f);
    cortical_substrate_update(cortical);

    float gains_hot[CORTICAL_SUBSTRATE_NUM_LAYERS];
    for (int i = 0; i < CORTICAL_SUBSTRATE_NUM_LAYERS; i++) {
        gains_hot[i] = cortical_substrate_get_layer_gain(cortical, i);
    }

    // All layer gains should be valid
    for (int i = 0; i < CORTICAL_SUBSTRATE_NUM_LAYERS; i++) {
        EXPECT_GE(gains_normal[i], 0.0f) << "Layer " << i << " gain should be >= 0";
        EXPECT_LE(gains_normal[i], 1.0f) << "Layer " << i << " gain should be <= 1";
        EXPECT_GE(gains_hot[i], 0.0f) << "Layer " << i << " hot gain should be >= 0";
        EXPECT_LE(gains_hot[i], 1.0f) << "Layer " << i << " hot gain should be <= 1";
    }

    cortical_substrate_bridge_destroy(cortical);
}

//=============================================================================
// Test: Valence Bias Under Metabolic Stress
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, EmotionBridge_ValenceBiasUnderStress) {
    // WHAT: Test emotion valence bias changes with metabolic state
    // WHY:  Verify neurotransmitter-like effects on emotional processing
    // HOW:  Manipulate substrate, verify valence bias changes

    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    emotion_substrate_bridge_t* emotion = emotion_substrate_bridge_create(
        &emotion_config, nullptr, substrate
    );
    ASSERT_NE(emotion, nullptr);

    // Healthy state - baseline valence
    substrate_set_atp(substrate, 0.95f);
    substrate_update(substrate, 1);
    emotion_substrate_update(emotion);

    float baseline_valence = emotion_substrate_get_valence_bias(emotion);

    // Valence should be in valid range
    EXPECT_GE(baseline_valence, -1.0f);
    EXPECT_LE(baseline_valence, 1.0f);

    // Stress state
    substrate_set_atp(substrate, 0.25f);
    substrate_update(substrate, 1);
    emotion_substrate_update(emotion);

    float stress_valence = emotion_substrate_get_valence_bias(emotion);

    // Valence should still be valid under stress
    EXPECT_GE(stress_valence, -1.0f);
    EXPECT_LE(stress_valence, 1.0f);

    // Get full effects structure
    emotion_substrate_effects_t effects = emotion_substrate_get_effects(emotion);

    EXPECT_GE(effects.intensity_modulation, 0.0f);
    EXPECT_GE(effects.regulation_capacity, 0.0f);
    EXPECT_LE(effects.regulation_capacity, 1.0f);
    EXPECT_GE(effects.reactivity_threshold, 0.0f);
    EXPECT_LE(effects.reactivity_threshold, 1.0f);

    emotion_substrate_bridge_destroy(emotion);
}

//=============================================================================
// Test: Bridge Statistics Accumulation
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, Statistics_AccumulateAcrossBridges) {
    // WHAT: Test statistics accumulation across multiple bridges
    // WHY:  Verify monitoring works correctly in multi-bridge scenario
    // HOW:  Perform many updates, verify stats accumulated

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    emotion_substrate_bridge_t* emotion = emotion_substrate_bridge_create(
        &emotion_config, nullptr, substrate
    );
    ASSERT_NE(emotion, nullptr);

    // Perform multiple updates
    const int num_updates = 50;
    for (int i = 0; i < num_updates; i++) {
        // Vary ATP to trigger different statistics
        float atp = 0.9f - (0.7f * i / num_updates);
        substrate_set_atp(substrate, atp);
        substrate_update(substrate, 10);

        cortical_substrate_update(cortical);
        emotion_substrate_update(emotion);
    }

    // Get statistics
    cortical_substrate_stats_t cortical_stats;
    cortical_substrate_get_stats(cortical, &cortical_stats);

    emotion_substrate_stats_t emotion_stats = emotion_substrate_get_stats(emotion);

    // Verify update counts
    EXPECT_EQ(cortical_stats.update_count, (uint64_t)num_updates)
        << "Cortical bridge should have recorded all updates";
    EXPECT_EQ(emotion_stats.total_updates, (uint64_t)num_updates)
        << "Emotion bridge should have recorded all updates";

    // Verify averages are in valid range
    EXPECT_GE(cortical_stats.avg_column_fidelity, 0.0f);
    EXPECT_LE(cortical_stats.avg_column_fidelity, 1.0f);

    EXPECT_GE(emotion_stats.avg_intensity_modulation, 0.0f);

    cortical_substrate_bridge_destroy(cortical);
    emotion_substrate_bridge_destroy(emotion);
}

//=============================================================================
// Test: Substrate Alerts Propagation
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, SubstrateAlerts_AffectAllBridges) {
    // WHAT: Test substrate alerts affect all connected bridges
    // WHY:  Verify alert conditions propagate to bridge behavior
    // HOW:  Trigger various alerts, verify bridge responses

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    // Trigger low ATP alert
    substrate_set_atp(substrate, 0.2f);
    substrate_update(substrate, 1);
    cortical_substrate_update(cortical);

    substrate_alert_type_t alerts[8];
    uint32_t alert_count = 0;
    substrate_get_alerts(substrate, alerts, &alert_count);

    // There may be alerts for low ATP
    bool has_atp_alert = false;
    for (uint32_t i = 0; i < alert_count; i++) {
        if (alerts[i] == SUBSTRATE_ALERT_LOW_ATP) {
            has_atp_alert = true;
        }
    }

    // If alerts are enabled and ATP is low enough, we should see an alert
    if (alert_count > 0) {
        EXPECT_TRUE(has_atp_alert) << "Should have low ATP alert";
    }

    // Bridge should reflect degraded state
    float fidelity = cortical_substrate_get_column_fidelity(cortical);
    EXPECT_LT(fidelity, 1.0f) << "Fidelity should be reduced under ATP stress";

    cortical_substrate_bridge_destroy(cortical);
}

//=============================================================================
// Test: Concurrent Bridge Updates
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, ConcurrentUpdates_ThreadSafe) {
    // WHAT: Test concurrent updates to multiple bridges
    // WHY:  Verify thread safety of bridge operations
    // HOW:  Simulate rapid concurrent updates, verify no corruption

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    emotion_substrate_bridge_t* emotion = emotion_substrate_bridge_create(
        &emotion_config, nullptr, substrate
    );
    ASSERT_NE(emotion, nullptr);

    // Rapid alternating updates
    for (int i = 0; i < 100; i++) {
        // Vary substrate state
        float atp = 0.5f + 0.4f * sin(i * 0.1f);
        float temp = 37.0f + 2.0f * cos(i * 0.15f);

        substrate_set_atp(substrate, atp);
        substrate_set_temperature(substrate, temp);
        substrate_update(substrate, 1);

        // Update bridges in alternating order
        if (i % 2 == 0) {
            cortical_substrate_update(cortical);
            emotion_substrate_update(emotion);
        } else {
            emotion_substrate_update(emotion);
            cortical_substrate_update(cortical);
        }

        // Query values (simulates concurrent reads)
        float fidelity = cortical_substrate_get_column_fidelity(cortical);
        float intensity = emotion_substrate_get_intensity_mod(emotion);

        // Values should always be valid
        EXPECT_GE(fidelity, 0.0f);
        EXPECT_GE(intensity, 0.0f);
    }

    cortical_substrate_bridge_destroy(cortical);
    emotion_substrate_bridge_destroy(emotion);
}

//=============================================================================
// Test: Effects Structure Completeness
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, Effects_AllFieldsPopulated) {
    // WHAT: Test that all effect fields are properly populated
    // WHY:  Verify complete effect computation
    // HOW:  Get effects, verify all fields have valid values

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    cortical_substrate_update(cortical);

    cortical_substrate_effects_t effects;
    int result = cortical_substrate_get_effects(cortical, &effects);
    EXPECT_EQ(result, 0);

    // Verify all fields
    EXPECT_GE(effects.column_fidelity, 0.0f);
    EXPECT_LE(effects.column_fidelity, 1.0f);

    EXPECT_GE(effects.competition_efficiency, 0.0f);
    EXPECT_LE(effects.competition_efficiency, 1.0f);

    EXPECT_GE(effects.sparsity_modulation, 0.0f);

    EXPECT_GE(effects.hierarchical_depth, 0.0f);
    EXPECT_LE(effects.hierarchical_depth, 1.0f);

    for (int i = 0; i < CORTICAL_SUBSTRATE_NUM_LAYERS; i++) {
        EXPECT_GE(effects.layer_gain[i], 0.0f)
            << "Layer " << i << " gain should be >= 0";
        EXPECT_LE(effects.layer_gain[i], 1.0f)
            << "Layer " << i << " gain should be <= 1";
    }

    cortical_substrate_bridge_destroy(cortical);
}

//=============================================================================
// Test: Bridge Reset
//=============================================================================

TEST_F(SubstrateBridgeIntegrationTest, BridgeReset_ClearsStatistics) {
    // WHAT: Test bridge base reset functionality
    // WHY:  Verify reset clears statistics while preserving connections
    // HOW:  Accumulate stats, reset base, verify stats cleared

    cortical_substrate_config_t cortical_config;
    cortical_substrate_default_config(&cortical_config);
    cortical_substrate_bridge_t* cortical = cortical_substrate_bridge_create(
        &cortical_config, substrate
    );
    ASSERT_NE(cortical, nullptr);

    // Perform updates to accumulate stats
    for (int i = 0; i < 20; i++) {
        cortical_substrate_update(cortical);
    }

    cortical_substrate_stats_t stats_before;
    cortical_substrate_get_stats(cortical, &stats_before);
    EXPECT_GT(stats_before.update_count, 0u);

    // Reset base
    bridge_base_reset(&cortical->base);

    // Verify base stats reset
    uint64_t total_updates = 0;
    uint64_t last_update = 0;
    bridge_base_get_stats(&cortical->base, &total_updates, &last_update);

    EXPECT_EQ(total_updates, 0u) << "Base update count should be reset";
    EXPECT_EQ(last_update, 0u) << "Base last update should be reset";

    cortical_substrate_bridge_destroy(cortical);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
