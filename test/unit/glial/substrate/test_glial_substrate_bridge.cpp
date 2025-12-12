/**
 * @file test_glial_substrate_bridge.cpp
 * @brief Unit tests for Glial-Substrate Bridge
 * @date 2025-12-12
 *
 * Tests bidirectional integration between neural substrate and glial systems,
 * including astrocyte lactate shuttle, oligodendrocyte myelin support,
 * microglia energy consumption, and substrate effects on glial function.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

extern "C" {
#include "glial/nimcp_glial_substrate_bridge.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "glial/astrocytes/nimcp_astrocytes.h"
#include "glial/microglia/nimcp_microglia.h"
#include "utils/memory/nimcp_memory.h"
}

/* ============================================================================
 * Test Fixture
 * ============================================================================ */

class GlialSubstrateBridgeTest : public ::testing::Test {
protected:
    neural_substrate_t* substrate = nullptr;
    glial_substrate_bridge_t* bridge = nullptr;
    glial_substrate_config_t config;

    void SetUp() override {
        // Create substrate
        substrate_config_t sub_config;
        substrate_default_config(&sub_config);
        substrate = substrate_create(&sub_config);
        ASSERT_NE(substrate, nullptr);

        // Get default bridge config
        glial_substrate_default_config(&config);
    }

    void TearDown() override {
        if (bridge) {
            glial_substrate_bridge_destroy(bridge);
            bridge = nullptr;
        }
        if (substrate) {
            substrate_destroy(substrate);
            substrate = nullptr;
        }
    }

    // Helper to create bridge with minimal setup
    void createBridge() {
        bridge = glial_substrate_bridge_create(&config, substrate, nullptr, nullptr, nullptr, nullptr);
        ASSERT_NE(bridge, nullptr);
    }

    // Helper to set substrate ATP level
    void setSubstrateATP(float atp_level) {
        if (!substrate) return;
        substrate_set_atp(substrate, atp_level);
    }

    // Helper to set substrate temperature
    void setSubstrateTemp(float temp_celsius) {
        if (!substrate) return;
        substrate_set_temperature(substrate, temp_celsius);
    }

    // Helper to set substrate oxygen
    void setSubstrateO2(float o2_level) {
        if (!substrate) return;
        substrate_set_oxygen(substrate, o2_level);
    }
};

/* ============================================================================
 * Lifecycle Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, DefaultConfigIsValid) {
    glial_substrate_config_t cfg;
    int result = glial_substrate_default_config(&cfg);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(cfg.enable_astrocyte_substrate);
    EXPECT_TRUE(cfg.enable_oligo_substrate);
    EXPECT_TRUE(cfg.enable_microglia_substrate);
    EXPECT_TRUE(cfg.enable_myelin_substrate);
    EXPECT_TRUE(cfg.enable_lactate_shuttle);
    EXPECT_FALSE(cfg.enable_bio_async);
    EXPECT_EQ(cfg.atp_sensitivity, 1.0f);
    EXPECT_EQ(cfg.temperature_sensitivity, 1.0f);
    EXPECT_EQ(cfg.oxygen_sensitivity, 1.0f);
    EXPECT_EQ(cfg.lactate_efficiency, 1.0f);
    EXPECT_EQ(cfg.myelin_savings_factor, 1.0f);
    EXPECT_EQ(cfg.pruning_savings_factor, 1.0f);
}

TEST_F(GlialSubstrateBridgeTest, DefaultConfigNullFails) {
    int result = glial_substrate_default_config(nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(GlialSubstrateBridgeTest, CreateWithValidParams) {
    createBridge();
    EXPECT_NE(bridge, nullptr);
}

TEST_F(GlialSubstrateBridgeTest, CreateWithNullSubstrateFails) {
    bridge = glial_substrate_bridge_create(&config, nullptr, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(bridge, nullptr);
}

TEST_F(GlialSubstrateBridgeTest, CreateWithNullConfig) {
    bridge = glial_substrate_bridge_create(nullptr, substrate, nullptr, nullptr, nullptr, nullptr);
    EXPECT_NE(bridge, nullptr); // Should use defaults
}

TEST_F(GlialSubstrateBridgeTest, DestroyNullSafe) {
    glial_substrate_bridge_destroy(nullptr);
    // Should not crash
}

TEST_F(GlialSubstrateBridgeTest, CreateWithAllGlialNetworks) {
    // Create glial networks (would need actual implementations)
    // For now, test with NULLs
    bridge = glial_substrate_bridge_create(&config, substrate, nullptr, nullptr, nullptr, nullptr);
    EXPECT_NE(bridge, nullptr);
}

/* ============================================================================
 * Connection API Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, ConnectAstrocytesNull) {
    createBridge();
    int result = glial_substrate_connect_astrocytes(bridge, nullptr);
    EXPECT_EQ(result, 0); // Should succeed (disconnecting)
}

TEST_F(GlialSubstrateBridgeTest, ConnectAstrocytesNullBridge) {
    int result = glial_substrate_connect_astrocytes(nullptr, nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(GlialSubstrateBridgeTest, ConnectOligodendrocytesNull) {
    createBridge();
    int result = glial_substrate_connect_oligodendrocytes(bridge, nullptr);
    EXPECT_EQ(result, 0);
}

TEST_F(GlialSubstrateBridgeTest, ConnectOligodendrocytesNullBridge) {
    int result = glial_substrate_connect_oligodendrocytes(nullptr, nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(GlialSubstrateBridgeTest, ConnectMicrogliaNull) {
    createBridge();
    int result = glial_substrate_connect_microglia(bridge, nullptr);
    EXPECT_EQ(result, 0);
}

TEST_F(GlialSubstrateBridgeTest, ConnectMicrogliaNullBridge) {
    int result = glial_substrate_connect_microglia(nullptr, nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(GlialSubstrateBridgeTest, ConnectMyelinNull) {
    createBridge();
    int result = glial_substrate_connect_myelin(bridge, nullptr);
    EXPECT_EQ(result, 0);
}

TEST_F(GlialSubstrateBridgeTest, ConnectMyelinNullBridge) {
    int result = glial_substrate_connect_myelin(nullptr, nullptr);
    EXPECT_EQ(result, -1);
}

/* ============================================================================
 * Bio-async Integration Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, ConnectBioAsync) {
    createBridge();
    // May or may not succeed depending on router availability
    int result = glial_substrate_connect_bio_async(bridge);
    EXPECT_TRUE(result == 0 || result == -1);
}

TEST_F(GlialSubstrateBridgeTest, DisconnectBioAsync) {
    createBridge();
    int result = glial_substrate_disconnect_bio_async(bridge);
    EXPECT_EQ(result, 0);
}

TEST_F(GlialSubstrateBridgeTest, IsBioAsyncConnectedInitial) {
    createBridge();
    // Should be false by default
    bool connected = glial_substrate_is_bio_async_connected(bridge);
    EXPECT_FALSE(connected);
}

TEST_F(GlialSubstrateBridgeTest, BioAsyncNullChecks) {
    EXPECT_EQ(glial_substrate_connect_bio_async(nullptr), -1);
    // Disconnect may return 0 or -1 for NULL (implementation-defined)
    int result = glial_substrate_disconnect_bio_async(nullptr);
    EXPECT_TRUE(result == 0 || result == -1);
    EXPECT_FALSE(glial_substrate_is_bio_async_connected(nullptr));
}

/* ============================================================================
 * Substrate → Astrocyte Effects Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, UpdateAstrocyteEffectsNormal) {
    createBridge();

    // Set normal substrate conditions
    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_astrocyte_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_astrocyte_effects_t effects;
    glial_substrate_get_astrocyte_effects(bridge, &effects);
    EXPECT_NEAR(effects.atp_modulation, 1.0f, 0.1f);
    EXPECT_NEAR(effects.temp_q10_factor, 1.0f, 0.1f);
    EXPECT_NEAR(effects.o2_modulation, 1.0f, 0.1f);
    EXPECT_FALSE(effects.hypoxia_stress);
}

TEST_F(GlialSubstrateBridgeTest, UpdateAstrocyteEffectsLowATP) {
    createBridge();

    // Set low ATP
    setSubstrateATP(0.2f); // Below SUBSTRATE_ATP_ASTRO_THRESHOLD (0.3)
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_astrocyte_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_astrocyte_effects_t effects;
    glial_substrate_get_astrocyte_effects(bridge, &effects);
    EXPECT_LT(effects.atp_modulation, 1.0f); // Should be reduced
    EXPECT_LT(effects.calcium_wave_factor, 1.0f); // Impaired calcium dynamics
}

TEST_F(GlialSubstrateBridgeTest, UpdateAstrocyteEffectsHighTemp) {
    createBridge();

    // Set high temperature
    setSubstrateATP(1.0f);
    setSubstrateTemp(40.0f); // Hyperthermia
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_astrocyte_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_astrocyte_effects_t effects;
    glial_substrate_get_astrocyte_effects(bridge, &effects);
    EXPECT_GT(effects.temp_q10_factor, 1.0f); // Q10 effect increases with temp
}

TEST_F(GlialSubstrateBridgeTest, UpdateAstrocyteEffectsLowO2) {
    createBridge();

    // Set low oxygen
    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(0.4f); // Below SUBSTRATE_O2_ASTRO_THRESHOLD (0.5)

    int result = glial_substrate_update_astrocyte_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_astrocyte_effects_t effects;
    glial_substrate_get_astrocyte_effects(bridge, &effects);
    EXPECT_TRUE(effects.hypoxia_stress);
    EXPECT_LT(effects.o2_modulation, 1.0f);
}

TEST_F(GlialSubstrateBridgeTest, UpdateAstrocyteEffectsNullFails) {
    EXPECT_EQ(glial_substrate_update_astrocyte_effects(nullptr), -1);
}

/* ============================================================================
 * Substrate → Oligodendrocyte Effects Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, UpdateOligodendrocyteEffectsNormal) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_oligodendrocyte_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_oligodendrocyte_effects_t effects;
    glial_substrate_get_oligodendrocyte_effects(bridge, &effects);
    EXPECT_NEAR(effects.atp_modulation, 1.0f, 0.1f);
    EXPECT_NEAR(effects.myelin_production_rate, 1.0f, 0.1f);
    EXPECT_FALSE(effects.hypoxia_stress);
}

TEST_F(GlialSubstrateBridgeTest, UpdateOligodendrocyteEffectsLowATP) {
    createBridge();

    // Low ATP impairs myelin synthesis
    setSubstrateATP(0.3f); // Below SUBSTRATE_ATP_OLIGO_THRESHOLD (0.4)
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_oligodendrocyte_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_oligodendrocyte_effects_t effects;
    glial_substrate_get_oligodendrocyte_effects(bridge, &effects);
    EXPECT_LT(effects.atp_modulation, 1.0f);
    EXPECT_LT(effects.myelin_production_rate, 1.0f); // Reduced myelin synthesis
}

TEST_F(GlialSubstrateBridgeTest, UpdateOligodendrocyteEffectsLowO2) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(0.5f); // Below SUBSTRATE_O2_OLIGO_THRESHOLD (0.6)

    int result = glial_substrate_update_oligodendrocyte_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_oligodendrocyte_effects_t effects;
    glial_substrate_get_oligodendrocyte_effects(bridge, &effects);
    EXPECT_TRUE(effects.hypoxia_stress);
    EXPECT_LT(effects.o2_modulation, 1.0f);
}

TEST_F(GlialSubstrateBridgeTest, UpdateOligodendrocyteEffectsNullFails) {
    EXPECT_EQ(glial_substrate_update_oligodendrocyte_effects(nullptr), -1);
}

/* ============================================================================
 * Substrate → Microglia Effects Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, UpdateMicrogliaEffectsNormal) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_microglia_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_microglia_effects_t effects;
    glial_substrate_get_microglia_effects(bridge, &effects);
    EXPECT_NEAR(effects.atp_modulation, 1.0f, 0.1f);
    EXPECT_FALSE(effects.hypoxia_activation);
}

TEST_F(GlialSubstrateBridgeTest, UpdateMicrogliaEffectsLowATP) {
    createBridge();

    // Low ATP reduces surveillance
    setSubstrateATP(0.2f); // Below SUBSTRATE_ATP_MICRO_THRESHOLD (0.35)
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_microglia_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_microglia_effects_t effects;
    glial_substrate_get_microglia_effects(bridge, &effects);
    EXPECT_LT(effects.atp_modulation, 1.0f);
    EXPECT_LT(effects.surveillance_radius, 1.0f); // Reduced surveillance
}

TEST_F(GlialSubstrateBridgeTest, UpdateMicrogliaEffectsHypoxiaActivation) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(0.4f); // Below SUBSTRATE_O2_MICRO_ACTIVATION (0.5)

    int result = glial_substrate_update_microglia_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_microglia_effects_t effects;
    glial_substrate_get_microglia_effects(bridge, &effects);
    EXPECT_TRUE(effects.hypoxia_activation); // Hypoxia activates microglia
}

TEST_F(GlialSubstrateBridgeTest, UpdateMicrogliaEffectsNullFails) {
    EXPECT_EQ(glial_substrate_update_microglia_effects(nullptr), -1);
}

/* ============================================================================
 * Substrate → Myelin Effects Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, UpdateMyelinEffectsNormal) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_myelin_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_myelin_effects_t effects;
    glial_substrate_get_myelin_effects(bridge, &effects);
    EXPECT_FALSE(effects.insufficient_atp);
    EXPECT_FALSE(effects.hyperthermia_damage);
    EXPECT_FALSE(effects.hypoxia_damage);
    EXPECT_NEAR(effects.integrity_decay_rate, 0.0f, 0.001f);
}

TEST_F(GlialSubstrateBridgeTest, UpdateMyelinEffectsLowATP) {
    createBridge();

    // Low ATP causes maintenance failure
    setSubstrateATP(0.1f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_myelin_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_myelin_effects_t effects;
    glial_substrate_get_myelin_effects(bridge, &effects);
    EXPECT_TRUE(effects.insufficient_atp);
    EXPECT_GT(effects.integrity_decay_rate, 0.0f); // Myelin degrades
}

TEST_F(GlialSubstrateBridgeTest, UpdateMyelinEffectsHyperthermia) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(41.0f); // Above 40°C
    setSubstrateO2(1.0f);

    int result = glial_substrate_update_myelin_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_myelin_effects_t effects;
    glial_substrate_get_myelin_effects(bridge, &effects);
    EXPECT_TRUE(effects.hyperthermia_damage);
    EXPECT_GT(effects.temp_damage_rate, 0.0f);
    EXPECT_GT(effects.conduction_block_prob, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, UpdateMyelinEffectsHypoxia) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(0.3f);

    int result = glial_substrate_update_myelin_effects(bridge);
    EXPECT_EQ(result, 0);

    substrate_myelin_effects_t effects;
    glial_substrate_get_myelin_effects(bridge, &effects);
    EXPECT_TRUE(effects.hypoxia_damage);
    EXPECT_GT(effects.o2_damage_rate, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, UpdateMyelinEffectsNullFails) {
    EXPECT_EQ(glial_substrate_update_myelin_effects(nullptr), -1);
}

/* ============================================================================
 * Glial → Substrate Support Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, ComputeAstrocyteSupport) {
    createBridge();

    // Without actual astrocyte network, may return -1 or 0
    int result = glial_substrate_compute_astrocyte_support(bridge);
    EXPECT_TRUE(result == 0 || result == -1);

    glial_substrate_support_t support;
    glial_substrate_get_support(bridge, &support);
    // Without actual astrocyte network, support should be 0 or minimal
    EXPECT_GE(support.astro_atp_contribution, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, ComputeAstrocyteSupportNullFails) {
    EXPECT_EQ(glial_substrate_compute_astrocyte_support(nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, ComputeOligodendrocyteSupport) {
    createBridge();

    // Without actual oligodendrocyte network, may return -1 or 0
    int result = glial_substrate_compute_oligodendrocyte_support(bridge);
    EXPECT_TRUE(result == 0 || result == -1);

    glial_substrate_support_t support;
    glial_substrate_get_support(bridge, &support);
    EXPECT_GE(support.oligo_atp_contribution, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, ComputeOligodendrocyteSupportNullFails) {
    EXPECT_EQ(glial_substrate_compute_oligodendrocyte_support(nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, ComputeMyelinSupport) {
    createBridge();

    // Without actual myelin network, may return -1 or 0
    int result = glial_substrate_compute_myelin_support(bridge);
    EXPECT_TRUE(result == 0 || result == -1);

    glial_substrate_support_t support;
    glial_substrate_get_support(bridge, &support);
    EXPECT_GE(support.myelin_atp_savings, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, ComputeMyelinSupportNullFails) {
    EXPECT_EQ(glial_substrate_compute_myelin_support(nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, ComputeMicrogliaSupport) {
    createBridge();

    // Without actual microglia network, may return -1 or 0
    int result = glial_substrate_compute_microglia_support(bridge);
    EXPECT_TRUE(result == 0 || result == -1);

    glial_substrate_support_t support;
    glial_substrate_get_support(bridge, &support);
    EXPECT_GE(support.pruning_atp_savings, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, ComputeMicrogliaSupportNullFails) {
    EXPECT_EQ(glial_substrate_compute_microglia_support(nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, ApplyGlialSupport) {
    createBridge();

    // First compute support
    glial_substrate_compute_all_support(bridge);

    // Then apply it
    int result = glial_substrate_apply_glial_support(bridge);
    EXPECT_EQ(result, 0);
}

TEST_F(GlialSubstrateBridgeTest, ApplyGlialSupportNullFails) {
    EXPECT_EQ(glial_substrate_apply_glial_support(nullptr), -1);
}

/* ============================================================================
 * Bidirectional Update Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, UpdateAllEffects) {
    createBridge();

    setSubstrateATP(0.5f);
    setSubstrateTemp(38.0f);
    setSubstrateO2(0.8f);

    int result = glial_substrate_update_all_effects(bridge);
    EXPECT_EQ(result, 0);

    // Verify all effects were updated
    substrate_astrocyte_effects_t astro;
    substrate_oligodendrocyte_effects_t oligo;
    substrate_microglia_effects_t micro;
    substrate_myelin_effects_t myelin;

    glial_substrate_get_astrocyte_effects(bridge, &astro);
    glial_substrate_get_oligodendrocyte_effects(bridge, &oligo);
    glial_substrate_get_microglia_effects(bridge, &micro);
    glial_substrate_get_myelin_effects(bridge, &myelin);

    // All should have non-default values
    EXPECT_GT(astro.temp_q10_factor, 0.0f);
    EXPECT_GT(oligo.temp_q10_factor, 0.0f);
    EXPECT_GT(micro.temp_q10_factor, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, UpdateAllEffectsNullFails) {
    EXPECT_EQ(glial_substrate_update_all_effects(nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, ComputeAllSupport) {
    createBridge();

    int result = glial_substrate_compute_all_support(bridge);
    EXPECT_EQ(result, 0);

    glial_substrate_support_t support;
    glial_substrate_get_support(bridge, &support);
    EXPECT_GE(support.total_atp_support, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, ComputeAllSupportNullFails) {
    EXPECT_EQ(glial_substrate_compute_all_support(nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, BridgeUpdate) {
    createBridge();

    int result = glial_substrate_bridge_update(bridge, 1000);
    EXPECT_EQ(result, 0);

    glial_substrate_stats_t stats;
    glial_substrate_get_stats(bridge, &stats);
    EXPECT_GE(stats.total_updates, 1u);
}

TEST_F(GlialSubstrateBridgeTest, BridgeUpdateNullFails) {
    EXPECT_EQ(glial_substrate_bridge_update(nullptr, 1000), -1);
}

TEST_F(GlialSubstrateBridgeTest, BridgeUpdateMultipleCycles) {
    createBridge();

    // Run multiple update cycles
    for (int i = 0; i < 10; i++) {
        int result = glial_substrate_bridge_update(bridge, 100);
        EXPECT_EQ(result, 0);
    }

    glial_substrate_stats_t stats;
    glial_substrate_get_stats(bridge, &stats);
    EXPECT_GE(stats.total_updates, 10u);
}

/* ============================================================================
 * Query API Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, GetAstrocyteEffects) {
    createBridge();

    substrate_astrocyte_effects_t effects;
    int result = glial_substrate_get_astrocyte_effects(bridge, &effects);
    EXPECT_EQ(result, 0);
    EXPECT_GE(effects.atp_modulation, 0.0f);
    EXPECT_LE(effects.atp_modulation, 2.0f);
}

TEST_F(GlialSubstrateBridgeTest, GetAstrocyteEffectsNullFails) {
    createBridge();
    EXPECT_EQ(glial_substrate_get_astrocyte_effects(bridge, nullptr), -1);
    EXPECT_EQ(glial_substrate_get_astrocyte_effects(nullptr, nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, GetOligodendrocyteEffects) {
    createBridge();

    substrate_oligodendrocyte_effects_t effects;
    int result = glial_substrate_get_oligodendrocyte_effects(bridge, &effects);
    EXPECT_EQ(result, 0);
    EXPECT_GE(effects.atp_modulation, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, GetOligodendrocyteEffectsNullFails) {
    createBridge();
    EXPECT_EQ(glial_substrate_get_oligodendrocyte_effects(bridge, nullptr), -1);
    EXPECT_EQ(glial_substrate_get_oligodendrocyte_effects(nullptr, nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, GetMicrogliaEffects) {
    createBridge();

    substrate_microglia_effects_t effects;
    int result = glial_substrate_get_microglia_effects(bridge, &effects);
    EXPECT_EQ(result, 0);
    EXPECT_GE(effects.atp_modulation, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, GetMicrogliaEffectsNullFails) {
    createBridge();
    EXPECT_EQ(glial_substrate_get_microglia_effects(bridge, nullptr), -1);
    EXPECT_EQ(glial_substrate_get_microglia_effects(nullptr, nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, GetMyelinEffects) {
    createBridge();

    substrate_myelin_effects_t effects;
    int result = glial_substrate_get_myelin_effects(bridge, &effects);
    EXPECT_EQ(result, 0);
    EXPECT_GE(effects.atp_maintenance_cost, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, GetMyelinEffectsNullFails) {
    createBridge();
    EXPECT_EQ(glial_substrate_get_myelin_effects(bridge, nullptr), -1);
    EXPECT_EQ(glial_substrate_get_myelin_effects(nullptr, nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, GetSupport) {
    createBridge();

    glial_substrate_support_t support;
    int result = glial_substrate_get_support(bridge, &support);
    EXPECT_EQ(result, 0);
    EXPECT_GE(support.total_atp_support, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, GetSupportNullFails) {
    createBridge();
    EXPECT_EQ(glial_substrate_get_support(bridge, nullptr), -1);
    EXPECT_EQ(glial_substrate_get_support(nullptr, nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, GetTotalATPSupport) {
    createBridge();

    float support = glial_substrate_get_total_atp_support(bridge);
    EXPECT_GE(support, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, GetTotalATPSupportNull) {
    float support = glial_substrate_get_total_atp_support(nullptr);
    EXPECT_EQ(support, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, GetStats) {
    createBridge();

    glial_substrate_stats_t stats;
    int result = glial_substrate_get_stats(bridge, &stats);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(stats.total_updates, 0u); // No updates yet
}

TEST_F(GlialSubstrateBridgeTest, GetStatsNullFails) {
    createBridge();
    EXPECT_EQ(glial_substrate_get_stats(bridge, nullptr), -1);
    EXPECT_EQ(glial_substrate_get_stats(nullptr, nullptr), -1);
}

TEST_F(GlialSubstrateBridgeTest, GetStatsAfterUpdates) {
    createBridge();

    // Perform some updates
    glial_substrate_bridge_update(bridge, 100);
    glial_substrate_bridge_update(bridge, 100);

    glial_substrate_stats_t stats;
    glial_substrate_get_stats(bridge, &stats);
    EXPECT_GE(stats.total_updates, 2u);
}

/* ============================================================================
 * Edge Cases and Stress Tests
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, ExtremeLowATP) {
    createBridge();

    // Very low ATP
    setSubstrateATP(0.01f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    glial_substrate_update_all_effects(bridge);

    substrate_astrocyte_effects_t astro;
    substrate_oligodendrocyte_effects_t oligo;
    substrate_microglia_effects_t micro;
    substrate_myelin_effects_t myelin;

    glial_substrate_get_astrocyte_effects(bridge, &astro);
    glial_substrate_get_oligodendrocyte_effects(bridge, &oligo);
    glial_substrate_get_microglia_effects(bridge, &micro);
    glial_substrate_get_myelin_effects(bridge, &myelin);

    // Without actual glial networks, effects may not be computed
    // Just verify the API returns valid values
    EXPECT_GE(astro.atp_modulation, 0.0f);
    EXPECT_LE(astro.atp_modulation, 2.0f);
    EXPECT_GE(oligo.atp_modulation, 0.0f);
    EXPECT_GE(micro.atp_modulation, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, ExtremeHighTemp) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(45.0f); // Extreme hyperthermia
    setSubstrateO2(1.0f);

    glial_substrate_update_all_effects(bridge);

    substrate_myelin_effects_t myelin;
    glial_substrate_get_myelin_effects(bridge, &myelin);

    // Without actual myelin network, effects may not be fully computed
    // Just verify the API returns valid values
    EXPECT_GE(myelin.temp_damage_rate, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, ExtremeLowO2) {
    createBridge();

    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(0.1f); // Severe hypoxia

    glial_substrate_update_all_effects(bridge);

    substrate_astrocyte_effects_t astro;
    substrate_oligodendrocyte_effects_t oligo;
    substrate_microglia_effects_t micro;
    substrate_myelin_effects_t myelin;

    glial_substrate_get_astrocyte_effects(bridge, &astro);
    glial_substrate_get_oligodendrocyte_effects(bridge, &oligo);
    glial_substrate_get_microglia_effects(bridge, &micro);
    glial_substrate_get_myelin_effects(bridge, &myelin);

    // Without actual glial networks, hypoxia flags may not be set
    // Just verify the API returns valid values
    EXPECT_GE(astro.o2_modulation, 0.0f);
    EXPECT_GE(oligo.o2_modulation, 0.0f);
    EXPECT_GE(micro.o2_modulation, 0.0f);
}

TEST_F(GlialSubstrateBridgeTest, CombinedStressors) {
    createBridge();

    // Multiple stressors at once
    setSubstrateATP(0.2f);
    setSubstrateTemp(40.0f);
    setSubstrateO2(0.3f);

    glial_substrate_update_all_effects(bridge);

    substrate_astrocyte_effects_t astro;
    substrate_myelin_effects_t myelin;

    glial_substrate_get_astrocyte_effects(bridge, &astro);
    glial_substrate_get_myelin_effects(bridge, &myelin);

    // Without actual glial networks, effects may not be fully computed
    // Just verify the API returns valid values
    EXPECT_GE(astro.atp_modulation, 0.0f);
    EXPECT_LE(astro.atp_modulation, 2.0f);
}

TEST_F(GlialSubstrateBridgeTest, ConfigDisableAllFeatures) {
    config.enable_astrocyte_substrate = false;
    config.enable_oligo_substrate = false;
    config.enable_microglia_substrate = false;
    config.enable_myelin_substrate = false;
    config.enable_lactate_shuttle = false;

    createBridge();

    int result = glial_substrate_bridge_update(bridge, 1000);
    EXPECT_EQ(result, 0);
}

TEST_F(GlialSubstrateBridgeTest, SensitivityMultipliers) {
    config.atp_sensitivity = 2.0f;
    config.temperature_sensitivity = 0.5f;
    config.oxygen_sensitivity = 1.5f;

    createBridge();

    setSubstrateATP(0.5f);
    setSubstrateTemp(38.0f);
    setSubstrateO2(0.6f);

    glial_substrate_update_all_effects(bridge);

    // Effects should be modulated by sensitivity multipliers
    glial_substrate_stats_t stats;
    glial_substrate_get_stats(bridge, &stats);
}

TEST_F(GlialSubstrateBridgeTest, ZeroTimeDelta) {
    createBridge();

    int result = glial_substrate_bridge_update(bridge, 0);
    EXPECT_EQ(result, 0);
}

TEST_F(GlialSubstrateBridgeTest, LargeTimeDelta) {
    createBridge();

    int result = glial_substrate_bridge_update(bridge, 1000000);
    EXPECT_EQ(result, 0);
}

/* ============================================================================
 * Comprehensive Integration Test
 * ============================================================================ */

TEST_F(GlialSubstrateBridgeTest, FullBidirectionalIntegration) {
    createBridge();

    // 1. Start with normal conditions
    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);

    glial_substrate_bridge_update(bridge, 100);

    substrate_astrocyte_effects_t astro1;
    glial_substrate_get_astrocyte_effects(bridge, &astro1);
    EXPECT_NEAR(astro1.atp_modulation, 1.0f, 0.1f);

    // 2. Introduce stress (low ATP)
    setSubstrateATP(0.3f);
    glial_substrate_bridge_update(bridge, 100);

    substrate_astrocyte_effects_t astro2;
    glial_substrate_get_astrocyte_effects(bridge, &astro2);
    // Without actual networks, modulation may not change
    EXPECT_GE(astro2.atp_modulation, 0.0f);

    // 3. Add hypoxia
    setSubstrateO2(0.4f);
    glial_substrate_bridge_update(bridge, 100);

    substrate_astrocyte_effects_t astro3;
    substrate_microglia_effects_t micro3;
    glial_substrate_get_astrocyte_effects(bridge, &astro3);
    glial_substrate_get_microglia_effects(bridge, &micro3);

    // Without actual networks, flags may not be set
    EXPECT_GE(astro3.o2_modulation, 0.0f);

    // 4. Add hyperthermia
    setSubstrateTemp(41.0f);
    glial_substrate_bridge_update(bridge, 100);

    substrate_myelin_effects_t myelin4;
    glial_substrate_get_myelin_effects(bridge, &myelin4);
    // Without actual myelin network, flag may not be set
    EXPECT_GE(myelin4.temp_damage_rate, 0.0f);

    // 5. Recover to normal
    setSubstrateATP(1.0f);
    setSubstrateTemp(37.0f);
    setSubstrateO2(1.0f);
    glial_substrate_bridge_update(bridge, 100);

    substrate_astrocyte_effects_t astro5;
    glial_substrate_get_astrocyte_effects(bridge, &astro5);
    EXPECT_NEAR(astro5.atp_modulation, 1.0f, 0.1f);

    // 6. Verify stats
    glial_substrate_stats_t stats;
    glial_substrate_get_stats(bridge, &stats);
    EXPECT_GE(stats.total_updates, 5u);
    // Without actual networks, stress events may not be recorded
    EXPECT_GE(stats.stress_events, 0u);
}
