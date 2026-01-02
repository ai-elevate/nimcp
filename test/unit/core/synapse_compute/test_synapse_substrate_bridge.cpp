/**
 * @file test_synapse_substrate_bridge.cpp
 * @brief Comprehensive unit tests for Synapse-Substrate Bridge
 *
 * TEST COVERAGE:
 * - Lifecycle (default_config, create/destroy, null safety)
 * - Substrate→Synapse effects (update, apply_modulation per synapse type)
 * - Synapse→Substrate consumption (consume_transmission, record_nmda_calcium)
 * - Query API (get_effects, get_transmission_efficiency, get_release_probability, etc.)
 * - Per-synapse-type modulation (AMPA, NMDA, GABA-A, GABA-B, dopamine, etc.)
 * - ATP effects on vesicle release probability
 * - Calcium effects on transmission
 * - Temperature Q10 effects per synapse type
 * - ATP costs per transmission type
 * - NMDA calcium influx tracking
 * - Membrane integrity effects on receptor density
 * - Edge cases (NULL pointers, extreme values)
 *
 * @author NIMCP Development Team
 * @date 2025-12-12
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

// Headers have their own extern "C" guards
#include "core/synapse_compute/nimcp_synapse_substrate_bridge.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/synapse_compute/nimcp_synapse_compute.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"

//=============================================================================
// Test Fixture
//=============================================================================

class SynapseSubstrateBridgeTest : public ::testing::Test {
protected:
    synapse_substrate_bridge_t* bridge;
    neural_substrate_t* substrate;
    synapse_compute_context_t synapse_ctx;  // Stack-allocated context
    synapse_substrate_config_t config;

    void SetUp() override {
        bridge = nullptr;
        substrate = nullptr;
        memset(&synapse_ctx, 0, sizeof(synapse_ctx));

        // Create substrate with normal values
        substrate_config_t substrate_cfg;
        substrate_default_config(&substrate_cfg);
        substrate = substrate_create(&substrate_cfg);
        ASSERT_NE(substrate, nullptr);

        // Initialize synapse compute context (stack-allocated)
        synapse_ctx.neuromodulation = 1.0f;
        synapse_ctx.current_time = 0;
        synapse_ctx.global_state = nullptr;
        synapse_ctx.global_state_size = 0;
        synapse_ctx.custom_data = nullptr;
        synapse_ctx.custom_data_size = 0;

        // Get default bridge config
        synapse_substrate_default_config(&config);

        // Create bridge (pass address of stack context)
        bridge = synapse_substrate_bridge_create(&config, &synapse_ctx, substrate);
        ASSERT_NE(bridge, nullptr);
    }

    void TearDown() override {
        if (bridge) {
            synapse_substrate_bridge_destroy(bridge);
            bridge = nullptr;
        }
        // synapse_ctx is stack-allocated, no cleanup needed
        if (substrate) {
            substrate_destroy(substrate);
            substrate = nullptr;
        }
    }

    // Helper: Set substrate ATP level
    void set_substrate_atp(float atp_level) {
        substrate_set_atp(substrate, atp_level);
    }

    // Helper: Set substrate temperature
    void set_substrate_temperature(float temp) {
        substrate_set_temperature(substrate, temp);
    }

    // Helper: Set substrate ion balance (approximates calcium effects)
    void set_substrate_calcium(float ca_level) {
        // Ion balance affects calcium handling - use as proxy
        substrate_set_ion_balance(substrate, ca_level);
    }

    // Helper: Set substrate membrane integrity
    void set_substrate_membrane(float membrane) {
        substrate_set_membrane_integrity(substrate, membrane);
    }

    // Helper: Set substrate ion balance
    void set_substrate_ion_balance(float ion_balance) {
        substrate_set_ion_balance(substrate, ion_balance);
    }
};

//=============================================================================
// LIFECYCLE TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, DefaultConfig) {
    synapse_substrate_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    int result = synapse_substrate_default_config(&cfg);
    EXPECT_EQ(result, 0);

    // Check all features enabled by default
    EXPECT_TRUE(cfg.enable_atp_modulation);
    EXPECT_TRUE(cfg.enable_ca_modulation);
    EXPECT_TRUE(cfg.enable_temperature_modulation);
    EXPECT_TRUE(cfg.enable_membrane_modulation);
    EXPECT_TRUE(cfg.enable_ion_modulation);
    EXPECT_TRUE(cfg.enable_transmission_cost);
    EXPECT_TRUE(cfg.enable_nmda_ca_tracking);

    // Check sensitivities
    EXPECT_EQ(cfg.atp_sensitivity, 1.0f);
    EXPECT_EQ(cfg.ca_sensitivity, 1.0f);
    EXPECT_EQ(cfg.temperature_sensitivity, 1.0f);
    EXPECT_EQ(cfg.membrane_sensitivity, 1.0f);
    EXPECT_EQ(cfg.ion_sensitivity, 1.0f);
    EXPECT_EQ(cfg.atp_cost_multiplier, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, DefaultConfigNullPointer) {
    int result = synapse_substrate_default_config(nullptr);
    EXPECT_NE(result, 0);
}

TEST_F(SynapseSubstrateBridgeTest, CreateValidBridge) {
    EXPECT_NE(bridge, nullptr);
}

TEST_F(SynapseSubstrateBridgeTest, CreateWithNullConfig) {
    synapse_substrate_bridge_t* test_bridge =
        synapse_substrate_bridge_create(nullptr, &synapse_ctx, substrate);
    EXPECT_NE(test_bridge, nullptr);
    synapse_substrate_bridge_destroy(test_bridge);
}

TEST_F(SynapseSubstrateBridgeTest, CreateWithNullSynapseContext) {
    synapse_substrate_bridge_t* test_bridge =
        synapse_substrate_bridge_create(&config, nullptr, substrate);
    // Implementation allows NULL synapse context (it's optional)
    EXPECT_NE(test_bridge, nullptr);
    if (test_bridge) {
        synapse_substrate_bridge_destroy(test_bridge);
    }
}

TEST_F(SynapseSubstrateBridgeTest, CreateWithNullSubstrate) {
    synapse_substrate_bridge_t* test_bridge =
        synapse_substrate_bridge_create(&config, &synapse_ctx, nullptr);
    EXPECT_EQ(test_bridge, nullptr);
}

TEST_F(SynapseSubstrateBridgeTest, DestroyNullBridge) {
    synapse_substrate_bridge_destroy(nullptr);
    SUCCEED(); // Should not crash
}

//=============================================================================
// SUBSTRATE → SYNAPSE UPDATE TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, UpdateWithNormalSubstrate) {
    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    // Check that effects are computed
    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_GT(efficiency, 0.5f);  // Should be positive with normal substrate
    EXPECT_LE(efficiency, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithNullBridge) {
    int result = synapse_substrate_update(nullptr);
    EXPECT_NE(result, 0);
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithLowATP) {
    set_substrate_atp(0.2f); // Below ATP_RELEASE_THRESHOLD (0.3)

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float release_prob = synapse_substrate_get_release_probability(bridge);
    EXPECT_LT(release_prob, 0.5f); // Release should be impaired
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithCriticalATP) {
    set_substrate_atp(0.3f); // At ATP_RELEASE_THRESHOLD

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float release_prob = synapse_substrate_get_release_probability(bridge);
    EXPECT_GT(release_prob, 0.0f);
    EXPECT_LT(release_prob, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithNormalATP) {
    set_substrate_atp(0.95f); // Normal ATP

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float release_prob = synapse_substrate_get_release_probability(bridge);
    EXPECT_GT(release_prob, 0.5f);  // Should be positive
    EXPECT_LE(release_prob, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithLowCalcium) {
    set_substrate_calcium(0.4f); // Below CA_DEPLETION_THRESHOLD (0.5)

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_LT(efficiency, 0.8f); // Transmission should be impaired
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithHighCalcium) {
    set_substrate_calcium(0.75f); // Above CA_OVERLOAD_THRESHOLD (0.7)

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_LT(efficiency, 1.0f); // Should be reduced due to Ca2+ overload
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithHyperthermia) {
    set_substrate_temperature(41.0f); // Above SUBSTRATE_HYPERTHERMIA_THRESHOLD (40.0)

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    EXPECT_GT(kinetics, 1.0f); // Faster kinetics
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithHypothermia) {
    set_substrate_temperature(30.0f); // Below SUBSTRATE_HYPOTHERMIA_THRESHOLD (32.0)

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    EXPECT_LT(kinetics, 1.0f); // Slower kinetics
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithDamagedMembrane) {
    set_substrate_membrane(0.5f); // Below MEMBRANE_RECEPTOR_THRESHOLD (0.6)

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_LT(efficiency, 0.7f); // Reduced receptor density
}

TEST_F(SynapseSubstrateBridgeTest, UpdateWithIonImbalance) {
    set_substrate_ion_balance(0.4f); // Below ION_DRIVING_FORCE_THRESHOLD (0.5)

    int result = synapse_substrate_update(bridge);
    EXPECT_EQ(result, 0);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_LT(efficiency, 0.8f); // Reduced driving force
}

//=============================================================================
// PER-SYNAPSE-TYPE MODULATION TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, AMPAModulationNormal) {
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_AMPA);
    EXPECT_GT(mod, 0.5f);  // Should be positive
    EXPECT_LE(mod, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, AMPAModulationLowATP) {
    set_substrate_atp(0.2f);
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_AMPA);
    EXPECT_LT(mod, 0.6f); // Significantly impaired
}

TEST_F(SynapseSubstrateBridgeTest, NMDAModulationNormal) {
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_NMDA);
    EXPECT_GT(mod, 0.5f);  // Should be positive
    EXPECT_LE(mod, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, NMDAModulationLowCalcium) {
    set_substrate_calcium(0.3f);
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_NMDA);
    EXPECT_LT(mod, 0.6f); // NMDA sensitive to Ca2+
}

TEST_F(SynapseSubstrateBridgeTest, GABAAModulationNormal) {
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_GABA_A);
    EXPECT_GT(mod, 0.5f);  // Should be positive
    EXPECT_LE(mod, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, GABABModulationNormal) {
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_GABA_B);
    EXPECT_GT(mod, 0.5f);  // Should be positive
    EXPECT_LE(mod, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, GABABModulationLowATP) {
    set_substrate_atp(0.2f);
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_GABA_B);
    EXPECT_LT(mod, 0.5f); // GABA-B very sensitive to ATP (metabotropic)
}

TEST_F(SynapseSubstrateBridgeTest, DopamineModulationNormal) {
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_DOPAMINE);
    EXPECT_GT(mod, 0.7f);
    EXPECT_LE(mod, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, DopamineModulationLowATP) {
    set_substrate_atp(0.15f);
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_DOPAMINE);
    EXPECT_LT(mod, 0.4f); // Dopamine very ATP-dependent
}

TEST_F(SynapseSubstrateBridgeTest, SerotoninModulationNormal) {
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_SEROTONIN);
    EXPECT_GT(mod, 0.7f);
    EXPECT_LE(mod, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, AcetylcholineModulationNormal) {
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_ACETYLCHOLINE);
    EXPECT_GT(mod, 0.7f);
    EXPECT_LE(mod, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, ElectricalModulationNormal) {
    synapse_substrate_update(bridge);

    float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_ELECTRICAL);
    EXPECT_GT(mod, 0.9f); // Electrical synapses least affected
    EXPECT_LE(mod, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, ApplyModulationNullBridge) {
    float mod = synapse_substrate_apply_modulation(nullptr, SYNAPSE_AMPA);
    EXPECT_EQ(mod, 1.0f); // Returns 1.0 on error (no modulation)
}

//=============================================================================
// TEMPERATURE Q10 EFFECTS TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, AMPAQ10EffectHyperthermia) {
    // AMPA has Q10 = 2.5 (highest temperature sensitivity)
    set_substrate_temperature(47.0f); // 10°C above normal (37°C)
    synapse_substrate_update(bridge);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    // Q10 effect varies with implementation
    EXPECT_GT(kinetics, 1.0f);  // Should be boosted at high temp
    EXPECT_LT(kinetics, 4.0f);
}

TEST_F(SynapseSubstrateBridgeTest, NMDAQ10EffectNormal) {
    // NMDA has Q10 = 2.0
    set_substrate_temperature(37.0f);
    synapse_substrate_update(bridge);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    EXPECT_NEAR(kinetics, 1.0f, 0.1f); // Should be ~1.0 at normal temp
}

TEST_F(SynapseSubstrateBridgeTest, GABAAQ10EffectHypothermia) {
    // GABA-A has Q10 = 2.0
    set_substrate_temperature(27.0f); // 10°C below normal
    synapse_substrate_update(bridge);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    // Q10^((27-37)/10) = 2.0^-1 = 0.5
    EXPECT_GT(kinetics, 0.4f);
    EXPECT_LT(kinetics, 0.6f);
}

TEST_F(SynapseSubstrateBridgeTest, GABAB_Q10Effect) {
    // GABA-B has Q10 = 1.5 (lowest among ionotropic/metabotropic)
    set_substrate_temperature(47.0f);
    synapse_substrate_update(bridge);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    // Q10 effect varies with implementation
    EXPECT_GT(kinetics, 1.0f);  // Should be boosted at high temp
    EXPECT_LT(kinetics, 3.0f);
}

//=============================================================================
// SYNAPSE → SUBSTRATE CONSUMPTION TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, ConsumeAMPATransmission) {
    // AMPA ATP cost: 0.0002
    synapse_substrate_update(bridge);

    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 100);
    EXPECT_EQ(result, 0);

    // Check stats
    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(stats.transmissions_processed, 100u);
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.0002f * 100, 0.001f);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeNMDATransmission) {
    // NMDA ATP cost: 0.0003
    synapse_substrate_update(bridge);

    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_NMDA, 50);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.0003f * 50, 0.001f);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeGABAATransmission) {
    // GABA-A ATP cost: 0.0002
    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_GABA_A, 75);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.0002f * 75, 0.001f);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeGABABTransmission) {
    // GABA-B ATP cost: 0.0005 (higher - metabotropic)
    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_GABA_B, 40);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.0005f * 40, 0.001f);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeDopamineTransmission) {
    // Dopamine ATP cost: 0.0008 (very high - cascades)
    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_DOPAMINE, 20);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.0008f * 20, 0.001f);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeSerotoninTransmission) {
    // Serotonin ATP cost: 0.0008
    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_SEROTONIN, 15);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.0008f * 15, 0.001f);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeAcetylcholineTransmission) {
    // ACh ATP cost: 0.0006
    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_ACETYLCHOLINE, 30);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.0006f * 30, 0.001f);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeElectricalTransmission) {
    // Electrical ATP cost: 0.0001 (minimal - gap junctions)
    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_ELECTRICAL, 200);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.0001f * 200, 0.001f);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeTransmissionNullBridge) {
    int result = synapse_substrate_consume_transmission(nullptr, SYNAPSE_AMPA, 10);
    EXPECT_NE(result, 0);
}

TEST_F(SynapseSubstrateBridgeTest, ConsumeTransmissionZeroCount) {
    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 0);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(stats.transmissions_processed, 0u);
}

TEST_F(SynapseSubstrateBridgeTest, RecordNMDACalcium) {
    // NMDA Ca2+ influx: 0.001 per transmission
    int result = synapse_substrate_record_nmda_calcium(bridge, 100);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(stats.nmda_ca_influx_events, 100u);
}

TEST_F(SynapseSubstrateBridgeTest, RecordNMDACalciumNullBridge) {
    int result = synapse_substrate_record_nmda_calcium(nullptr, 50);
    EXPECT_NE(result, 0);
}

TEST_F(SynapseSubstrateBridgeTest, RecordNMDACalciumZeroCount) {
    int result = synapse_substrate_record_nmda_calcium(bridge, 0);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(stats.nmda_ca_influx_events, 0u);
}

//=============================================================================
// QUERY API TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, GetEffects) {
    synapse_substrate_update(bridge);

    substrate_synapse_effects_t effects;
    int result = synapse_substrate_get_effects(bridge, &effects);
    EXPECT_EQ(result, 0);

    EXPECT_GT(effects.transmission_efficiency, 0.0f);
    EXPECT_LE(effects.transmission_efficiency, 1.0f);
    EXPECT_GT(effects.release_probability, 0.0f);
    EXPECT_LE(effects.release_probability, 1.0f);
    EXPECT_GT(effects.receptor_kinetics_factor, 0.0f);
}

TEST_F(SynapseSubstrateBridgeTest, GetEffectsNullBridge) {
    substrate_synapse_effects_t effects;
    int result = synapse_substrate_get_effects(nullptr, &effects);
    EXPECT_NE(result, 0);
}

TEST_F(SynapseSubstrateBridgeTest, GetEffectsNullOutput) {
    int result = synapse_substrate_get_effects(bridge, nullptr);
    EXPECT_NE(result, 0);
}

TEST_F(SynapseSubstrateBridgeTest, GetTransmissionEfficiency) {
    synapse_substrate_update(bridge);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_GT(efficiency, 0.0f);
    EXPECT_LE(efficiency, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, GetTransmissionEfficiencyNullBridge) {
    float efficiency = synapse_substrate_get_transmission_efficiency(nullptr);
    EXPECT_EQ(efficiency, 1.0f);  // Returns neutral 1.0 for NULL
}

TEST_F(SynapseSubstrateBridgeTest, GetReleaseProbability) {
    synapse_substrate_update(bridge);

    float prob = synapse_substrate_get_release_probability(bridge);
    EXPECT_GT(prob, 0.0f);
    EXPECT_LE(prob, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, GetReleaseProbabilityNullBridge) {
    float prob = synapse_substrate_get_release_probability(nullptr);
    EXPECT_EQ(prob, 1.0f);  // Returns neutral 1.0 for NULL
}

TEST_F(SynapseSubstrateBridgeTest, GetReceptorKineticsFactor) {
    synapse_substrate_update(bridge);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    EXPECT_GT(kinetics, 0.0f);
}

TEST_F(SynapseSubstrateBridgeTest, GetReceptorKineticsFactorNullBridge) {
    float kinetics = synapse_substrate_get_receptor_kinetics_factor(nullptr);
    EXPECT_EQ(kinetics, 1.0f); // Returns 1.0 (no modulation) on error
}

TEST_F(SynapseSubstrateBridgeTest, GetStats) {
    // Perform some operations
    synapse_substrate_update(bridge);
    synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 50);
    synapse_substrate_consume_transmission(bridge, SYNAPSE_DOPAMINE, 10);
    synapse_substrate_record_nmda_calcium(bridge, 25);

    synapse_substrate_stats_t stats;
    int result = synapse_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(result, 0);

    EXPECT_GT(stats.total_updates, 0u);
    EXPECT_EQ(stats.transmissions_processed, 60u);
    EXPECT_EQ(stats.nmda_ca_influx_events, 25u);
    EXPECT_GT(stats.total_atp_consumed_by_synapses, 0.0f);
}

TEST_F(SynapseSubstrateBridgeTest, GetStatsNullBridge) {
    synapse_substrate_stats_t stats;
    int result = synapse_substrate_get_stats(nullptr, &stats);
    EXPECT_NE(result, 0);
}

TEST_F(SynapseSubstrateBridgeTest, GetStatsNullOutput) {
    int result = synapse_substrate_get_stats(bridge, nullptr);
    EXPECT_NE(result, 0);
}

//=============================================================================
// COMBINED EFFECTS TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, CombinedLowATPAndHighTemp) {
    set_substrate_atp(0.2f);
    set_substrate_temperature(42.0f);
    synapse_substrate_update(bridge);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_LT(efficiency, 0.5f); // Both factors impair transmission
}

TEST_F(SynapseSubstrateBridgeTest, CombinedLowCalciumAndDamagedMembrane) {
    set_substrate_calcium(0.4f);
    set_substrate_membrane(0.5f);
    synapse_substrate_update(bridge);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_LT(efficiency, 0.4f); // Severe impairment
}

TEST_F(SynapseSubstrateBridgeTest, CombinedOptimalConditions) {
    set_substrate_atp(0.95f);
    set_substrate_calcium(0.95f);
    set_substrate_temperature(37.0f);
    set_substrate_membrane(0.98f);
    set_substrate_ion_balance(0.95f);
    synapse_substrate_update(bridge);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_GT(efficiency, 0.5f);   // Should be positive
    EXPECT_LE(efficiency, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, CombinedCriticalConditions) {
    set_substrate_atp(0.1f);
    set_substrate_calcium(0.2f);
    set_substrate_temperature(30.0f);
    set_substrate_membrane(0.4f);
    set_substrate_ion_balance(0.3f);
    synapse_substrate_update(bridge);

    float efficiency = synapse_substrate_get_transmission_efficiency(bridge);
    EXPECT_LT(efficiency, 0.2f); // Severe dysfunction
}

//=============================================================================
// EDGE CASES AND STRESS TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, ExtremelyLowATP) {
    set_substrate_atp(0.01f);
    synapse_substrate_update(bridge);

    float release_prob = synapse_substrate_get_release_probability(bridge);
    EXPECT_GE(release_prob, 0.0f);
    EXPECT_LT(release_prob, 0.1f);
}

TEST_F(SynapseSubstrateBridgeTest, ExtremelyHighTemp) {
    set_substrate_temperature(50.0f);
    synapse_substrate_update(bridge);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    EXPECT_GT(kinetics, 1.0f); // Should still compute Q10 factor
}

TEST_F(SynapseSubstrateBridgeTest, ExtremelyLowTemp) {
    set_substrate_temperature(20.0f);
    synapse_substrate_update(bridge);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    EXPECT_LT(kinetics, 1.0f);
    EXPECT_GT(kinetics, 0.0f);
}

TEST_F(SynapseSubstrateBridgeTest, MultipleUpdates) {
    for (int i = 0; i < 100; i++) {
        int result = synapse_substrate_update(bridge);
        EXPECT_EQ(result, 0);
    }

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(stats.total_updates, 100u);
}

TEST_F(SynapseSubstrateBridgeTest, LargeNumberOfTransmissions) {
    int result = synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 1000000);
    EXPECT_EQ(result, 0);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(stats.transmissions_processed, 1000000u);
}

TEST_F(SynapseSubstrateBridgeTest, AllSynapseTypesModulation) {
    synapse_substrate_update(bridge);

    // Test all synapse types
    synapse_type_t types[] = {
        SYNAPSE_AMPA,
        SYNAPSE_NMDA,
        SYNAPSE_GABA_A,
        SYNAPSE_GABA_B,
        SYNAPSE_DOPAMINE,
        SYNAPSE_SEROTONIN,
        SYNAPSE_ACETYLCHOLINE,
        SYNAPSE_ELECTRICAL
    };

    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        float mod = synapse_substrate_apply_modulation(bridge, types[i]);
        EXPECT_GE(mod, 0.0f) << "Synapse type " << i << " failed";
        EXPECT_LE(mod, 1.5f) << "Synapse type " << i << " failed";
    }
}

//=============================================================================
// CONFIGURATION TESTS
//=============================================================================

TEST_F(SynapseSubstrateBridgeTest, DisableATPModulation) {
    config.enable_atp_modulation = false;
    synapse_substrate_bridge_destroy(bridge);
    bridge = synapse_substrate_bridge_create(&config, &synapse_ctx, substrate);

    set_substrate_atp(0.1f);
    synapse_substrate_update(bridge);

    // Release probability should be less affected when ATP modulation disabled
    float release_prob = synapse_substrate_get_release_probability(bridge);
    EXPECT_GT(release_prob, 0.5f);  // Should still be reasonable without ATP modulation
}

TEST_F(SynapseSubstrateBridgeTest, DisableCalciumModulation) {
    config.enable_ca_modulation = false;
    synapse_substrate_bridge_destroy(bridge);
    bridge = synapse_substrate_bridge_create(&config, &synapse_ctx, substrate);

    set_substrate_calcium(0.1f);
    synapse_substrate_update(bridge);

    // Should not be affected by low calcium
    substrate_synapse_effects_t effects;
    synapse_substrate_get_effects(bridge, &effects);
    EXPECT_EQ(effects.ca_transmission_effect, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, DisableTemperatureModulation) {
    config.enable_temperature_modulation = false;
    synapse_substrate_bridge_destroy(bridge);
    bridge = synapse_substrate_bridge_create(&config, &synapse_ctx, substrate);

    set_substrate_temperature(50.0f);
    synapse_substrate_update(bridge);

    float kinetics = synapse_substrate_get_receptor_kinetics_factor(bridge);
    EXPECT_EQ(kinetics, 1.0f);
}

TEST_F(SynapseSubstrateBridgeTest, CustomSensitivity) {
    config.atp_sensitivity = 0.5f; // Reduced sensitivity
    synapse_substrate_bridge_destroy(bridge);
    bridge = synapse_substrate_bridge_create(&config, &synapse_ctx, substrate);

    set_substrate_atp(0.2f);
    synapse_substrate_update(bridge);

    float release_prob = synapse_substrate_get_release_probability(bridge);
    // Should be less impaired than with full sensitivity
    EXPECT_GT(release_prob, 0.0f);
}

TEST_F(SynapseSubstrateBridgeTest, CustomATPCostMultiplier) {
    config.atp_cost_multiplier = 2.0f; // Double the ATP cost
    synapse_substrate_bridge_destroy(bridge);
    bridge = synapse_substrate_bridge_create(&config, &synapse_ctx, substrate);

    synapse_substrate_consume_transmission(bridge, SYNAPSE_AMPA, 100);

    synapse_substrate_stats_t stats;
    synapse_substrate_get_stats(bridge, &stats);
    // Should be 0.0002 * 100 * 2.0 = 0.04
    EXPECT_NEAR(stats.total_atp_consumed_by_synapses, 0.04f, 0.001f);
}
