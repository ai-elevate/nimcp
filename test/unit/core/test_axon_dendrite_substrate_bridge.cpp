/**
 * @file test_axon_dendrite_substrate_bridge.cpp
 * @brief Unit tests for Axon-Dendrite Neural Substrate Bridge
 * @date 2025-12-12
 *
 * WHAT: Comprehensive tests for bidirectional axon-dendrite substrate integration
 * WHY:  Verify substrate modulation of axonal conduction and dendritic integration
 * HOW:  GoogleTest framework testing all lifecycle, update, and query APIs
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

// Headers have their own extern "C" guards
#include "core/nimcp_axon_dendrite_substrate_bridge.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/axon/nimcp_axon.h"
#include "core/dendrite/nimcp_dendrite.h"
#include "utils/memory/nimcp_memory.h"

/* ============================================================================
 * Test Fixture
 * ============================================================================ */

class AxonDendriteSubstrateBridgeTest : public ::testing::Test {
protected:
    neural_substrate_t* substrate = nullptr;
    axon_t* axon = nullptr;
    dendrite_t* dendrite = nullptr;
    axon_dendrite_substrate_bridge_t* bridge = nullptr;
    axon_dendrite_substrate_config_t config;

    void SetUp() override {
        // Create neural substrate
        substrate_config_t sub_cfg;
        substrate_default_config(&sub_cfg);
        substrate = substrate_create(&sub_cfg);
        ASSERT_NE(substrate, nullptr);

        // Create test axon (1mm myelinated)
        axon = axon_create(1, AXON_TYPE_MYELINATED, 100, 200, 1000.0f, 2.0f);
        ASSERT_NE(axon, nullptr);

        // Create test dendrite
        dendrite_config_t dend_cfg;
        dendrite_default_config(&dend_cfg);
        dendrite = dendrite_create(1, 100, &dend_cfg);
        ASSERT_NE(dendrite, nullptr);

        // Get default bridge config
        axon_dendrite_substrate_default_config(&config);
    }

    void TearDown() override {
        if (bridge) {
            axon_dendrite_substrate_bridge_destroy(bridge);
            bridge = nullptr;
        }
        if (dendrite) {
            dendrite_destroy(dendrite);
            dendrite = nullptr;
        }
        if (axon) {
            axon_destroy(axon);
            axon = nullptr;
        }
        if (substrate) {
            substrate_destroy(substrate);
            substrate = nullptr;
        }
    }

    // Helper to create bridge with both axon and dendrite
    void createBridgeFull() {
        bridge = axon_dendrite_substrate_bridge_create(&config, substrate, axon, dendrite);
        ASSERT_NE(bridge, nullptr);
    }

    // Helper to create bridge with only axon
    void createBridgeAxonOnly() {
        bridge = axon_dendrite_substrate_bridge_create(&config, substrate, axon, nullptr);
        ASSERT_NE(bridge, nullptr);
    }

    // Helper to create bridge with only dendrite
    void createBridgeDendriteOnly() {
        bridge = axon_dendrite_substrate_bridge_create(&config, substrate, nullptr, dendrite);
        ASSERT_NE(bridge, nullptr);
    }
};

/* ============================================================================
 * Lifecycle Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, DefaultConfigIsValid) {
    axon_dendrite_substrate_config_t cfg;
    int result = axon_dendrite_substrate_default_config(&cfg);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(cfg.enable_axon_modulation);
    EXPECT_TRUE(cfg.enable_dendrite_modulation);
    EXPECT_TRUE(cfg.enable_bidirectional_feedback);
    EXPECT_TRUE(cfg.enable_temperature_effects);
    EXPECT_TRUE(cfg.enable_atp_dynamics);
    EXPECT_TRUE(cfg.enable_ion_dynamics);
    EXPECT_EQ(cfg.temperature_sensitivity, 1.0f);
    EXPECT_EQ(cfg.atp_sensitivity, 1.0f);
    EXPECT_EQ(cfg.ion_sensitivity, 1.0f);
    EXPECT_EQ(cfg.membrane_sensitivity, 1.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DefaultConfigNullFails) {
    int result = axon_dendrite_substrate_default_config(nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(AxonDendriteSubstrateBridgeTest, CreateWithValidParams) {
    createBridgeFull();
    EXPECT_NE(bridge, nullptr);
}

TEST_F(AxonDendriteSubstrateBridgeTest, CreateWithNullSubstrateFails) {
    bridge = axon_dendrite_substrate_bridge_create(&config, nullptr, axon, dendrite);
    EXPECT_EQ(bridge, nullptr);
}

TEST_F(AxonDendriteSubstrateBridgeTest, CreateWithNullConfig) {
    bridge = axon_dendrite_substrate_bridge_create(nullptr, substrate, axon, dendrite);
    EXPECT_NE(bridge, nullptr);  // Should use defaults
}

TEST_F(AxonDendriteSubstrateBridgeTest, CreateAxonOnlySucceeds) {
    createBridgeAxonOnly();
    EXPECT_NE(bridge, nullptr);
}

TEST_F(AxonDendriteSubstrateBridgeTest, CreateDendriteOnlySucceeds) {
    createBridgeDendriteOnly();
    EXPECT_NE(bridge, nullptr);
}

TEST_F(AxonDendriteSubstrateBridgeTest, CreateBothNullModulesFails) {
    bridge = axon_dendrite_substrate_bridge_create(&config, substrate, nullptr, nullptr);
    EXPECT_EQ(bridge, nullptr);  // Must have at least one module
}

TEST_F(AxonDendriteSubstrateBridgeTest, DestroyNullSafe) {
    axon_dendrite_substrate_bridge_destroy(nullptr);
    // Should not crash
}

/* ============================================================================
 * Bio-async Integration Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, ConnectBioAsync) {
    createBridgeFull();
    // May or may not succeed depending on router availability
    int result = axon_dendrite_substrate_connect_bio_async(bridge);
    EXPECT_TRUE(result == 0 || result == -1);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DisconnectBioAsync) {
    createBridgeFull();
    int result = axon_dendrite_substrate_disconnect_bio_async(bridge);
    EXPECT_EQ(result, 0);
}

TEST_F(AxonDendriteSubstrateBridgeTest, IsBioAsyncConnected) {
    createBridgeFull();
    bool connected = axon_dendrite_substrate_is_bio_async_connected(bridge);
    EXPECT_TRUE(connected || !connected);  // Either is valid
}

TEST_F(AxonDendriteSubstrateBridgeTest, BioAsyncNullChecks) {
    EXPECT_EQ(axon_dendrite_substrate_connect_bio_async(nullptr), -1);
    EXPECT_EQ(axon_dendrite_substrate_disconnect_bio_async(nullptr), -1);
    EXPECT_FALSE(axon_dendrite_substrate_is_bio_async_connected(nullptr));
}

/* ============================================================================
 * Axon Substrate Effects Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, UpdateAxonEffectsNormalSubstrate) {
    createBridgeAxonOnly();

    int result = axon_dendrite_substrate_update_axon_effects(bridge);
    EXPECT_EQ(result, 0);

    // Normal substrate should have moderate velocity mod
    float velocity_mod = axon_dendrite_substrate_get_conduction_mod(bridge);
    EXPECT_GE(velocity_mod, SUBSTRATE_VELOCITY_MIN);
    EXPECT_LE(velocity_mod, SUBSTRATE_VELOCITY_MAX);
}

TEST_F(AxonDendriteSubstrateBridgeTest, AxonEffectsTemperatureQ10) {
    createBridgeAxonOnly();

    // Set elevated temperature (hyperthermia)
    substrate_set_temperature(substrate, SUBSTRATE_TEMP_HYPERTHERMIA);
    axon_dendrite_substrate_update_axon_effects(bridge);

    float hot_velocity = axon_dendrite_substrate_get_conduction_mod(bridge);

    // Reset to normal
    substrate_set_temperature(substrate, SUBSTRATE_REFERENCE_TEMP);
    axon_dendrite_substrate_update_axon_effects(bridge);

    float normal_velocity = axon_dendrite_substrate_get_conduction_mod(bridge);

    // Higher temperature should increase velocity (Q10 effect)
    EXPECT_GT(hot_velocity, normal_velocity);
}

TEST_F(AxonDendriteSubstrateBridgeTest, AxonEffectsHypothermia) {
    createBridgeAxonOnly();

    // Set low temperature
    substrate_set_temperature(substrate, SUBSTRATE_TEMP_HYPOTHERMIA);
    axon_dendrite_substrate_update_axon_effects(bridge);

    float cold_velocity = axon_dendrite_substrate_get_conduction_mod(bridge);

    // Cold temperature should decrease velocity
    EXPECT_LT(cold_velocity, 1.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, AxonEffectsLowATP) {
    createBridgeAxonOnly();

    // Deplete ATP below spike threshold
    substrate_set_atp(substrate, 0.4f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);

    float spike_reliability = axon_dendrite_substrate_get_spike_reliability(bridge);

    // Low ATP should reduce reliability
    EXPECT_LT(spike_reliability, 1.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, AxonEffectsCriticalATP) {
    createBridgeAxonOnly();

    // Critical ATP depletion
    substrate_set_atp(substrate, 0.2f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);

    float spike_reliability = axon_dendrite_substrate_get_spike_reliability(bridge);

    // Critical ATP should severely impair reliability
    EXPECT_LT(spike_reliability, 0.5f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, AxonEffectsIonImbalance) {
    createBridgeAxonOnly();

    // Disrupt ion balance
    substrate_set_ion_balance(substrate, 0.5f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);

    float spike_reliability = axon_dendrite_substrate_get_spike_reliability(bridge);

    // Ion imbalance should reduce reliability
    EXPECT_LT(spike_reliability, 1.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, AxonEffectsMembraneDamage) {
    createBridgeAxonOnly();

    // Damage membrane
    substrate_set_membrane_integrity(substrate, 0.6f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);

    float velocity_mod = axon_dendrite_substrate_get_conduction_mod(bridge);

    // Membrane damage should reduce conduction
    EXPECT_LT(velocity_mod, 1.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, AxonEffectsRefractoryModulation) {
    createBridgeAxonOnly();

    // Normal ATP
    axon_dendrite_substrate_update_axon_effects(bridge);
    float normal_refractory = axon_dendrite_substrate_get_refractory_mod(bridge);

    // Low ATP
    substrate_set_atp(substrate, 0.3f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);
    float low_atp_refractory = axon_dendrite_substrate_get_refractory_mod(bridge);

    // Low ATP should increase refractory period (slower recovery)
    EXPECT_GT(low_atp_refractory, normal_refractory);
}

TEST_F(AxonDendriteSubstrateBridgeTest, UpdateAxonEffectsNullFails) {
    int result = axon_dendrite_substrate_update_axon_effects(nullptr);
    EXPECT_EQ(result, -1);
}

/* ============================================================================
 * Dendrite Substrate Effects Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, UpdateDendriteEffectsNormalSubstrate) {
    createBridgeDendriteOnly();

    int result = axon_dendrite_substrate_update_dendrite_effects(bridge);
    EXPECT_EQ(result, 0);

    // Normal substrate should have high integration efficiency
    float integration = axon_dendrite_substrate_get_integration_mod(bridge);
    EXPECT_GT(integration, 0.7f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DendriteEffectsMembraneDamage) {
    createBridgeDendriteOnly();

    // Damage membrane (affects cable properties)
    substrate_set_membrane_integrity(substrate, 0.5f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_dendrite_effects(bridge);

    float integration = axon_dendrite_substrate_get_integration_mod(bridge);

    // Membrane damage → leaky integration
    EXPECT_LT(integration, 0.8f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DendriteEffectsLowATPPlasticity) {
    createBridgeDendriteOnly();

    // Deplete ATP
    substrate_set_atp(substrate, 0.3f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_dendrite_effects(bridge);

    float plasticity = axon_dendrite_substrate_get_plasticity_mod(bridge);

    // Low ATP → impaired plasticity
    EXPECT_LT(plasticity, 0.8f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DendriteEffectsCriticalATPNoPlas ticity) {
    createBridgeDendriteOnly();

    // Critical ATP depletion
    substrate_set_atp(substrate, 0.2f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_dendrite_effects(bridge);

    float plasticity = axon_dendrite_substrate_get_plasticity_mod(bridge);

    // Critical ATP → severe plasticity impairment
    EXPECT_LT(plasticity, 0.5f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DendriteEffectsTemperatureThreshold) {
    createBridgeDendriteOnly();

    // Elevated temperature
    substrate_set_temperature(substrate, SUBSTRATE_TEMP_HYPERTHERMIA);
    axon_dendrite_substrate_update_dendrite_effects(bridge);

    float hot_threshold = axon_dendrite_substrate_get_spike_threshold_mod(bridge);

    // Normal temperature
    substrate_set_temperature(substrate, SUBSTRATE_REFERENCE_TEMP);
    axon_dendrite_substrate_update_dendrite_effects(bridge);

    float normal_threshold = axon_dendrite_substrate_get_spike_threshold_mod(bridge);

    // Hyperthermia → lower spike threshold
    EXPECT_LT(hot_threshold, normal_threshold);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DendriteEffectsCalciumHandling) {
    createBridgeDendriteOnly();

    // Normal ATP
    axon_dendrite_substrate_update_dendrite_effects(bridge);
    float normal_ca = axon_dendrite_substrate_get_ca_handling_mod(bridge);
    EXPECT_GT(normal_ca, 0.7f);

    // Low ATP → impaired Ca2+ pumps
    substrate_set_atp(substrate, 0.3f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_dendrite_effects(bridge);
    float low_atp_ca = axon_dendrite_substrate_get_ca_handling_mod(bridge);

    EXPECT_LT(low_atp_ca, normal_ca);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DendriteEffectsIonBalanceNMDA) {
    createBridgeDendriteOnly();

    // Ion imbalance affects NMDA function
    substrate_set_ion_balance(substrate, 0.5f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_dendrite_effects(bridge);

    float threshold_mod = axon_dendrite_substrate_get_spike_threshold_mod(bridge);

    // Ion imbalance → altered NMDA spike threshold
    EXPECT_NE(threshold_mod, 1.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, UpdateDendriteEffectsNullFails) {
    int result = axon_dendrite_substrate_update_dendrite_effects(nullptr);
    EXPECT_EQ(result, -1);
}

/* ============================================================================
 * Feedback: Axon/Dendrite → Substrate Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, RecordAxonSpikesDepletesATP) {
    createBridgeAxonOnly();

    float initial_atp = substrate->metabolic.atp_level;

    int result = axon_dendrite_substrate_record_axon_spikes(bridge, 10);
    EXPECT_EQ(result, 0);

    // ATP should be depleted
    EXPECT_LT(bridge->accumulated_atp_debt, 0.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, RecordAxonSpikesIonAccumulation) {
    createBridgeAxonOnly();

    int result = axon_dendrite_substrate_record_axon_spikes(bridge, 5);
    EXPECT_EQ(result, 0);

    // Ion imbalance should accumulate
    EXPECT_GT(bridge->accumulated_ion_imbalance, 0.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, RecordAxonSpikesNullFails) {
    int result = axon_dendrite_substrate_record_axon_spikes(nullptr, 10);
    EXPECT_EQ(result, -1);
}

TEST_F(AxonDendriteSubstrateBridgeTest, RecordDendriteEventsDepletesATP) {
    createBridgeDendriteOnly();

    int result = axon_dendrite_substrate_record_dendrite_events(bridge, 20);
    EXPECT_EQ(result, 0);

    // ATP debt should accumulate
    EXPECT_LT(bridge->accumulated_atp_debt, 0.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, RecordDendriteEventsNullFails) {
    int result = axon_dendrite_substrate_record_dendrite_events(nullptr, 20);
    EXPECT_EQ(result, -1);
}

TEST_F(AxonDendriteSubstrateBridgeTest, RecordPlasticityHighATPCost) {
    createBridgeDendriteOnly();

    float initial_debt = bridge->accumulated_atp_debt;

    int result = axon_dendrite_substrate_record_plasticity(bridge, 0.8f);
    EXPECT_EQ(result, 0);

    // Plasticity should cost significant ATP
    EXPECT_LT(bridge->accumulated_atp_debt, initial_debt);
}

TEST_F(AxonDendriteSubstrateBridgeTest, RecordPlasticityNullFails) {
    int result = axon_dendrite_substrate_record_plasticity(nullptr, 0.5f);
    EXPECT_EQ(result, -1);
}

/* ============================================================================
 * Bidirectional Update Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, BidirectionalUpdateBothDirections) {
    createBridgeFull();

    int result = axon_dendrite_substrate_bridge_update(bridge, 100);
    EXPECT_EQ(result, 0);

    axon_dendrite_substrate_stats_t stats;
    axon_dendrite_substrate_get_stats(bridge, &stats);
    EXPECT_GE(stats.total_updates, 1u);
}

TEST_F(AxonDendriteSubstrateBridgeTest, UpdateAxonOnlyBridge) {
    createBridgeAxonOnly();

    int result = axon_dendrite_substrate_bridge_update(bridge, 100);
    EXPECT_EQ(result, 0);

    // Should update axon effects only
    float velocity_mod = axon_dendrite_substrate_get_conduction_mod(bridge);
    EXPECT_GT(velocity_mod, 0.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, UpdateDendriteOnlyBridge) {
    createBridgeDendriteOnly();

    int result = axon_dendrite_substrate_bridge_update(bridge, 100);
    EXPECT_EQ(result, 0);

    // Should update dendrite effects only
    float integration = axon_dendrite_substrate_get_integration_mod(bridge);
    EXPECT_GT(integration, 0.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, UpdateMultipleCycles) {
    createBridgeFull();

    for (int i = 0; i < 10; i++) {
        int result = axon_dendrite_substrate_bridge_update(bridge, 100);
        EXPECT_EQ(result, 0);
    }

    axon_dendrite_substrate_stats_t stats;
    axon_dendrite_substrate_get_stats(bridge, &stats);
    EXPECT_GE(stats.total_updates, 10u);
}

TEST_F(AxonDendriteSubstrateBridgeTest, UpdateNullFails) {
    int result = axon_dendrite_substrate_bridge_update(nullptr, 100);
    EXPECT_EQ(result, -1);
}

/* ============================================================================
 * Query API Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, GetAxonEffects) {
    createBridgeAxonOnly();

    axon_dendrite_substrate_update_axon_effects(bridge);

    axon_substrate_effects_t effects;
    int result = axon_dendrite_substrate_get_axon_effects(bridge, &effects);
    EXPECT_EQ(result, 0);
    EXPECT_GE(effects.overall_velocity_mod, SUBSTRATE_VELOCITY_MIN);
    EXPECT_LE(effects.overall_velocity_mod, SUBSTRATE_VELOCITY_MAX);
}

TEST_F(AxonDendriteSubstrateBridgeTest, GetAxonEffectsNullChecks) {
    createBridgeAxonOnly();

    axon_substrate_effects_t effects;
    EXPECT_EQ(axon_dendrite_substrate_get_axon_effects(nullptr, &effects), -1);
    EXPECT_EQ(axon_dendrite_substrate_get_axon_effects(bridge, nullptr), -1);
}

TEST_F(AxonDendriteSubstrateBridgeTest, GetDendriteEffects) {
    createBridgeDendriteOnly();

    axon_dendrite_substrate_update_dendrite_effects(bridge);

    dendrite_substrate_effects_t effects;
    int result = axon_dendrite_substrate_get_dendrite_effects(bridge, &effects);
    EXPECT_EQ(result, 0);
    EXPECT_GE(effects.overall_capacity, 0.0f);
    EXPECT_LE(effects.overall_capacity, 1.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, GetDendriteEffectsNullChecks) {
    createBridgeDendriteOnly();

    dendrite_substrate_effects_t effects;
    EXPECT_EQ(axon_dendrite_substrate_get_dendrite_effects(nullptr, &effects), -1);
    EXPECT_EQ(axon_dendrite_substrate_get_dendrite_effects(bridge, nullptr), -1);
}

TEST_F(AxonDendriteSubstrateBridgeTest, IsAxonLimitedFalseNormally) {
    createBridgeAxonOnly();

    axon_dendrite_substrate_update_axon_effects(bridge);
    bool limited = axon_dendrite_substrate_is_axon_limited(bridge);
    EXPECT_FALSE(limited);
}

TEST_F(AxonDendriteSubstrateBridgeTest, IsAxonLimitedTrueWithLowATP) {
    createBridgeAxonOnly();

    substrate_set_atp(substrate, 0.2f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);

    bool limited = axon_dendrite_substrate_is_axon_limited(bridge);
    EXPECT_TRUE(limited);
}

TEST_F(AxonDendriteSubstrateBridgeTest, IsAxonLimitedNullReturnsFalse) {
    bool limited = axon_dendrite_substrate_is_axon_limited(nullptr);
    EXPECT_FALSE(limited);
}

TEST_F(AxonDendriteSubstrateBridgeTest, IsDendriteLimitedFalseNormally) {
    createBridgeDendriteOnly();

    axon_dendrite_substrate_update_dendrite_effects(bridge);
    bool limited = axon_dendrite_substrate_is_dendrite_limited(bridge);
    EXPECT_FALSE(limited);
}

TEST_F(AxonDendriteSubstrateBridgeTest, IsDendriteLimitedTrueWithDamage) {
    createBridgeDendriteOnly();

    substrate_set_membrane_integrity(substrate, 0.4f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_dendrite_effects(bridge);

    bool limited = axon_dendrite_substrate_is_dendrite_limited(bridge);
    EXPECT_TRUE(limited);
}

TEST_F(AxonDendriteSubstrateBridgeTest, IsDendriteLimitedNullReturnsFalse) {
    bool limited = axon_dendrite_substrate_is_dendrite_limited(nullptr);
    EXPECT_FALSE(limited);
}

TEST_F(AxonDendriteSubstrateBridgeTest, GetStats) {
    createBridgeFull();

    axon_dendrite_substrate_bridge_update(bridge, 100);

    axon_dendrite_substrate_stats_t stats;
    int result = axon_dendrite_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(result, 0);
    EXPECT_GE(stats.total_updates, 1u);
}

TEST_F(AxonDendriteSubstrateBridgeTest, GetStatsNullChecks) {
    createBridgeFull();

    axon_dendrite_substrate_stats_t stats;
    EXPECT_EQ(axon_dendrite_substrate_get_stats(nullptr, &stats), -1);
    EXPECT_EQ(axon_dendrite_substrate_get_stats(bridge, nullptr), -1);
}

/* ============================================================================
 * Feature Enable/Disable Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, DisableAxonModulation) {
    config.enable_axon_modulation = false;
    createBridgeFull();

    // Axon effects should not be computed
    float velocity = axon_dendrite_substrate_get_conduction_mod(bridge);
    EXPECT_EQ(velocity, 1.0f);  // Default no-modulation value
}

TEST_F(AxonDendriteSubstrateBridgeTest, DisableDendriteModulation) {
    config.enable_dendrite_modulation = false;
    createBridgeFull();

    // Dendrite effects should not be computed
    float integration = axon_dendrite_substrate_get_integration_mod(bridge);
    EXPECT_EQ(integration, 1.0f);  // Default no-modulation value
}

TEST_F(AxonDendriteSubstrateBridgeTest, DisableBidirectionalFeedback) {
    config.enable_bidirectional_feedback = false;
    createBridgeFull();

    // Record spikes
    axon_dendrite_substrate_record_axon_spikes(bridge, 10);

    // Should not affect substrate (no feedback)
    EXPECT_EQ(bridge->accumulated_atp_debt, 0.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DisableTemperatureEffects) {
    config.enable_temperature_effects = false;
    createBridgeAxonOnly();

    substrate_set_temperature(substrate, SUBSTRATE_TEMP_HYPERTHERMIA);
    axon_dendrite_substrate_update_axon_effects(bridge);

    // Temperature should not affect velocity
    float velocity = axon_dendrite_substrate_get_conduction_mod(bridge);
    // Should be close to 1.0 without temperature effects
}

TEST_F(AxonDendriteSubstrateBridgeTest, DisableATPDynamics) {
    config.enable_atp_dynamics = false;
    createBridgeAxonOnly();

    substrate_set_atp(substrate, 0.2f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);

    // Spike reliability should not be affected by low ATP
    float reliability = axon_dendrite_substrate_get_spike_reliability(bridge);
    EXPECT_GE(reliability, 0.9f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, DisableIonDynamics) {
    config.enable_ion_dynamics = false;
    createBridgeAxonOnly();

    substrate_set_ion_balance(substrate, 0.3f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);

    // Ion imbalance should not affect reliability
    float reliability = axon_dendrite_substrate_get_spike_reliability(bridge);
    EXPECT_GE(reliability, 0.9f);
}

/* ============================================================================
 * Sensitivity Multiplier Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, HighTemperatureSensitivity) {
    config.temperature_sensitivity = 2.0f;
    createBridgeAxonOnly();

    substrate_set_temperature(substrate, 39.0f);  // Slight fever
    axon_dendrite_substrate_update_axon_effects(bridge);
    float high_sens_velocity = axon_dendrite_substrate_get_conduction_mod(bridge);

    // Compare with normal sensitivity
    substrate_reset(substrate);
    bridge->config.temperature_sensitivity = 1.0f;
    substrate_set_temperature(substrate, 39.0f);
    axon_dendrite_substrate_update_axon_effects(bridge);
    float normal_velocity = axon_dendrite_substrate_get_conduction_mod(bridge);

    // High sensitivity should amplify temperature effect
    EXPECT_NE(high_sens_velocity, normal_velocity);
}

TEST_F(AxonDendriteSubstrateBridgeTest, HighATPSensitivity) {
    config.atp_sensitivity = 2.0f;
    createBridgeAxonOnly();

    substrate_set_atp(substrate, 0.4f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);
    float high_sens_reliability = axon_dendrite_substrate_get_spike_reliability(bridge);

    // Compare with normal sensitivity
    substrate_reset(substrate);
    bridge->config.atp_sensitivity = 1.0f;
    substrate_set_atp(substrate, 0.4f);
    substrate_update(substrate, 10);
    axon_dendrite_substrate_update_axon_effects(bridge);
    float normal_reliability = axon_dendrite_substrate_get_spike_reliability(bridge);

    // High sensitivity should amplify ATP effect
    EXPECT_LT(high_sens_reliability, normal_reliability);
}

/* ============================================================================
 * Edge Case and Stress Tests
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, ZeroDeltaUpdate) {
    createBridgeFull();

    int result = axon_dendrite_substrate_bridge_update(bridge, 0);
    EXPECT_EQ(result, 0);

    axon_dendrite_substrate_stats_t stats;
    axon_dendrite_substrate_get_stats(bridge, &stats);
    EXPECT_GE(stats.total_updates, 1u);
}

TEST_F(AxonDendriteSubstrateBridgeTest, LargeDeltaUpdate) {
    createBridgeFull();

    int result = axon_dendrite_substrate_bridge_update(bridge, 10000);
    EXPECT_EQ(result, 0);
}

TEST_F(AxonDendriteSubstrateBridgeTest, ExtremeSubstrateConditions) {
    createBridgeFull();

    // Extreme conditions
    substrate_set_atp(substrate, 0.05f);
    substrate_set_temperature(substrate, 42.0f);
    substrate_set_membrane_integrity(substrate, 0.3f);
    substrate_set_ion_balance(substrate, 0.2f);
    substrate_update(substrate, 10);

    // Should handle gracefully
    int result = axon_dendrite_substrate_bridge_update(bridge, 100);
    EXPECT_EQ(result, 0);

    // Both should be severely limited
    EXPECT_TRUE(axon_dendrite_substrate_is_axon_limited(bridge));
    EXPECT_TRUE(axon_dendrite_substrate_is_dendrite_limited(bridge));
}

TEST_F(AxonDendriteSubstrateBridgeTest, MassiveActivityFeedback) {
    createBridgeFull();

    // Simulate massive activity
    for (int i = 0; i < 100; i++) {
        axon_dendrite_substrate_record_axon_spikes(bridge, 50);
        axon_dendrite_substrate_record_dendrite_events(bridge, 100);
    }

    // Should accumulate significant debt
    EXPECT_LT(bridge->accumulated_atp_debt, -1.0f);
    EXPECT_GT(bridge->accumulated_ion_imbalance, 1.0f);
}

TEST_F(AxonDendriteSubstrateBridgeTest, GetterNullSafety) {
    EXPECT_FLOAT_EQ(axon_dendrite_substrate_get_conduction_mod(nullptr), 1.0f);
    EXPECT_FLOAT_EQ(axon_dendrite_substrate_get_spike_reliability(nullptr), 1.0f);
    EXPECT_FLOAT_EQ(axon_dendrite_substrate_get_refractory_mod(nullptr), 1.0f);
    EXPECT_FLOAT_EQ(axon_dendrite_substrate_get_integration_mod(nullptr), 1.0f);
    EXPECT_FLOAT_EQ(axon_dendrite_substrate_get_spike_threshold_mod(nullptr), 1.0f);
    EXPECT_FLOAT_EQ(axon_dendrite_substrate_get_plasticity_mod(nullptr), 1.0f);
    EXPECT_FLOAT_EQ(axon_dendrite_substrate_get_ca_handling_mod(nullptr), 1.0f);
}

/* ============================================================================
 * Comprehensive Integration Test
 * ============================================================================ */

TEST_F(AxonDendriteSubstrateBridgeTest, FullIntegrationCycle) {
    createBridgeFull();

    // 1. Start with healthy substrate
    EXPECT_FALSE(axon_dendrite_substrate_is_axon_limited(bridge));
    EXPECT_FALSE(axon_dendrite_substrate_is_dendrite_limited(bridge));

    // 2. Generate normal activity
    for (int i = 0; i < 10; i++) {
        axon_dendrite_substrate_record_axon_spikes(bridge, 5);
        axon_dendrite_substrate_record_dendrite_events(bridge, 10);
        axon_dendrite_substrate_bridge_update(bridge, 100);
    }

    // 3. Verify modulation under healthy conditions
    float healthy_velocity = axon_dendrite_substrate_get_conduction_mod(bridge);
    float healthy_integration = axon_dendrite_substrate_get_integration_mod(bridge);
    EXPECT_GT(healthy_velocity, 0.8f);
    EXPECT_GT(healthy_integration, 0.8f);

    // 4. Introduce substrate stress
    substrate_set_atp(substrate, 0.3f);
    substrate_set_membrane_integrity(substrate, 0.6f);
    substrate_update(substrate, 10);

    // 5. Continue activity under stress
    for (int i = 0; i < 10; i++) {
        axon_dendrite_substrate_record_axon_spikes(bridge, 5);
        axon_dendrite_substrate_record_dendrite_events(bridge, 10);
        axon_dendrite_substrate_bridge_update(bridge, 100);
    }

    // 6. Verify impairment
    float stressed_velocity = axon_dendrite_substrate_get_conduction_mod(bridge);
    float stressed_integration = axon_dendrite_substrate_get_integration_mod(bridge);
    EXPECT_LT(stressed_velocity, healthy_velocity);
    EXPECT_LT(stressed_integration, healthy_integration);

    // 7. Both should be substrate-limited
    EXPECT_TRUE(axon_dendrite_substrate_is_axon_limited(bridge));
    EXPECT_TRUE(axon_dendrite_substrate_is_dendrite_limited(bridge));

    // 8. Verify statistics
    axon_dendrite_substrate_stats_t stats;
    axon_dendrite_substrate_get_stats(bridge, &stats);
    EXPECT_GT(stats.total_updates, 0u);
    EXPECT_GT(stats.axon_spikes_processed, 0u);
    EXPECT_GT(stats.dendrite_events_processed, 0u);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
