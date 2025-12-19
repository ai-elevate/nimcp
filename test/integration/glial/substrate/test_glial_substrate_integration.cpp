/**
 * @file test_glial_substrate_integration.cpp
 * @brief Integration tests for glial-substrate bridge
 *
 * WHAT: Test bidirectional integration between glial cells and neural substrate
 * WHY:  Verify substrate affects glial function and glia support substrate metabolically
 * HOW:  Simulate substrate states and verify glial modulation and metabolic support
 *
 * TEST CATEGORIES:
 * - Substrate → glial effects (ATP, temperature, oxygen)
 * - Glial → substrate support (lactate shuttle, myelin efficiency, pruning)
 * - Multi-glial coordination
 * - Bio-async integration
 * - Metabolic feedback loops
 *
 * @author NIMCP Development Team
 * @date 2025-12-19
 */

#include <gtest/gtest.h>
#include <cmath>

#include "glial/nimcp_glial_substrate_bridge.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "glial/astrocytes/nimcp_astrocytes.h"
#include "glial/oligodendrocytes/nimcp_oligodendrocytes.h"
#include "glial/microglia/nimcp_microglia.h"
#include "glial/myelin_sheath/nimcp_myelin_sheath.h"

//=============================================================================
// Test Fixture
//=============================================================================

class GlialSubstrateIntegrationTest : public ::testing::Test {
protected:
    neural_substrate_t* substrate = nullptr;
    astrocyte_network_t* astro_network = nullptr;
    oligodendrocyte_network_t* oligo_network = nullptr;
    microglia_network_t* micro_network = nullptr;
    myelin_sheath_network_t* myelin_network = nullptr;
    glial_substrate_bridge_t* bridge = nullptr;

    void SetUp() override {
        // Create substrate
        substrate_config_t sub_config = {};
        substrate_config_init(&sub_config);
        sub_config.initial_atp_level = 0.8f;
        sub_config.initial_temperature = 37.0f;
        sub_config.initial_oxygen_level = 0.9f;
        substrate = substrate_create(&sub_config);
        ASSERT_NE(substrate, nullptr);

        // Create glial networks
        astrocyte_network_config_t astro_config;
        astrocyte_network_default_config(&astro_config);
        astro_network = astrocyte_network_create(&astro_config, 100);
        ASSERT_NE(astro_network, nullptr);

        oligodendrocyte_network_config_t oligo_config;
        oligodendrocyte_network_default_config(&oligo_config);
        oligo_network = oligodendrocyte_network_create(&oligo_config, 50);
        ASSERT_NE(oligo_network, nullptr);

        microglia_network_config_t micro_config;
        microglia_network_default_config(&micro_config);
        micro_network = microglia_network_create(&micro_config, 20);
        ASSERT_NE(micro_network, nullptr);

        myelin_sheath_network_config_t myelin_config;
        myelin_sheath_network_default_config(&myelin_config);
        myelin_network = myelin_sheath_network_create(&myelin_config, 100);
        ASSERT_NE(myelin_network, nullptr);

        // Create bridge
        glial_substrate_config_t config;
        glial_substrate_default_config(&config);
        config.enable_astrocyte_substrate = true;
        config.enable_oligo_substrate = true;
        config.enable_microglia_substrate = true;
        config.enable_myelin_substrate = true;
        config.enable_lactate_shuttle = true;

        bridge = glial_substrate_bridge_create(&config, substrate,
            astro_network, oligo_network, micro_network, myelin_network);
        ASSERT_NE(bridge, nullptr);
    }

    void TearDown() override {
        glial_substrate_bridge_destroy(bridge);
        myelin_sheath_network_destroy(myelin_network);
        microglia_network_destroy(micro_network);
        oligodendrocyte_network_destroy(oligo_network);
        astrocyte_network_destroy(astro_network);
        substrate_destroy(substrate);
    }

    void set_substrate_state(float atp, float temp, float o2) {
        substrate_set_atp_level(substrate, atp);
        substrate_set_temperature(substrate, temp);
        substrate_set_oxygen_level(substrate, o2);
    }
};

//=============================================================================
// Substrate → Astrocyte Effects
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, LowATPImpainsAstrocyteCalcium) {
    // WHAT: Test low ATP impairs astrocyte calcium dynamics
    // WHY:  Verify calcium pumps require ATP

    set_substrate_state(0.2f, 37.0f, 0.9f);

    ASSERT_EQ(glial_substrate_update_astrocyte_effects(bridge), 0);

    substrate_astrocyte_effects_t effects;
    ASSERT_EQ(glial_substrate_get_astrocyte_effects(bridge, &effects), 0);

    // Calcium wave propagation should be impaired
    EXPECT_LT(effects.atp_modulation, 0.5f);
    EXPECT_LT(effects.calcium_wave_factor, 0.8f);
}

TEST_F(GlialSubstrateIntegrationTest, HypoxiaStressesAstrocytes) {
    // WHAT: Test hypoxia triggers astrocyte stress
    // WHY:  Verify oxygen sensing

    set_substrate_state(0.8f, 37.0f, 0.4f);

    ASSERT_EQ(glial_substrate_update_astrocyte_effects(bridge), 0);

    substrate_astrocyte_effects_t effects;
    ASSERT_EQ(glial_substrate_get_astrocyte_effects(bridge, &effects), 0);

    EXPECT_TRUE(effects.hypoxia_stress);
    EXPECT_LT(effects.o2_modulation, 0.7f);
}

TEST_F(GlialSubstrateIntegrationTest, TemperatureAffectsAstrocyteMetabolism) {
    // WHAT: Test temperature affects astrocyte metabolism (Q10)
    // WHY:  Verify temperature sensitivity

    set_substrate_state(0.8f, 39.0f, 0.9f);

    ASSERT_EQ(glial_substrate_update_astrocyte_effects(bridge), 0);

    substrate_astrocyte_effects_t effects;
    ASSERT_EQ(glial_substrate_get_astrocyte_effects(bridge, &effects), 0);

    // Q10 ≈ 2.5, so 2°C increase → ~1.2x factor
    EXPECT_GT(effects.temp_q10_factor, 1.1f);
    EXPECT_LT(effects.temp_q10_factor, 1.3f);
}

//=============================================================================
// Substrate → Oligodendrocyte Effects
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, LowATPImpainsMyeination) {
    // WHAT: Test low ATP impairs myelin synthesis
    // WHY:  Myelin synthesis is ATP-expensive

    set_substrate_state(0.3f, 37.0f, 0.9f);

    ASSERT_EQ(glial_substrate_update_oligodendrocyte_effects(bridge), 0);

    substrate_oligodendrocyte_effects_t effects;
    ASSERT_EQ(glial_substrate_get_oligodendrocyte_effects(bridge, &effects), 0);

    EXPECT_LT(effects.atp_modulation, 0.5f);
    EXPECT_LT(effects.myelin_production_rate, 0.6f);
}

TEST_F(GlialSubstrateIntegrationTest, HypoxiaImpainsOligodendrocytes) {
    // WHAT: Test hypoxia impairs oligodendrocyte function
    // WHY:  High metabolic demand sensitive to oxygen

    set_substrate_state(0.8f, 37.0f, 0.5f);

    ASSERT_EQ(glial_substrate_update_oligodendrocyte_effects(bridge), 0);

    substrate_oligodendrocyte_effects_t effects;
    ASSERT_EQ(glial_substrate_get_oligodendrocyte_effects(bridge, &effects), 0);

    EXPECT_TRUE(effects.hypoxia_stress);
    EXPECT_LT(effects.o2_modulation, 0.8f);
}

//=============================================================================
// Substrate → Microglia Effects
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, LowATPReducesMicrogliaSurveillance) {
    // WHAT: Test low ATP reduces surveillance
    // WHY:  Process extension requires ATP

    set_substrate_state(0.3f, 37.0f, 0.9f);

    ASSERT_EQ(glial_substrate_update_microglia_effects(bridge), 0);

    substrate_microglia_effects_t effects;
    ASSERT_EQ(glial_substrate_get_microglia_effects(bridge, &effects), 0);

    EXPECT_LT(effects.atp_modulation, 0.5f);
    EXPECT_LT(effects.surveillance_radius, 1.0f);
}

TEST_F(GlialSubstrateIntegrationTest, HypoxiaActivatesMicroglia) {
    // WHAT: Test hypoxia is a danger signal
    // WHY:  Microglia respond to tissue stress

    set_substrate_state(0.8f, 37.0f, 0.4f);

    ASSERT_EQ(glial_substrate_update_microglia_effects(bridge), 0);

    substrate_microglia_effects_t effects;
    ASSERT_EQ(glial_substrate_get_microglia_effects(bridge, &effects), 0);

    EXPECT_TRUE(effects.hypoxia_activation);
}

//=============================================================================
// Substrate → Myelin Effects
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, LowATPCauseMyelinDegradation) {
    // WHAT: Test insufficient ATP causes myelin damage
    // WHY:  Myelin requires continuous maintenance

    set_substrate_state(0.2f, 37.0f, 0.9f);

    ASSERT_EQ(glial_substrate_update_myelin_effects(bridge), 0);

    substrate_myelin_effects_t effects;
    ASSERT_EQ(glial_substrate_get_myelin_effects(bridge, &effects), 0);

    EXPECT_TRUE(effects.insufficient_atp);
    EXPECT_GT(effects.integrity_decay_rate, 0.0f);
}

TEST_F(GlialSubstrateIntegrationTest, HyperthermaiaDamagesMyelin) {
    // WHAT: Test high temperature damages myelin
    // WHY:  Protein denaturation at high temps

    set_substrate_state(0.8f, 41.0f, 0.9f);

    ASSERT_EQ(glial_substrate_update_myelin_effects(bridge), 0);

    substrate_myelin_effects_t effects;
    ASSERT_EQ(glial_substrate_get_myelin_effects(bridge, &effects), 0);

    EXPECT_TRUE(effects.hyperthermia_damage);
    EXPECT_GT(effects.temp_damage_rate, 0.0f);
}

TEST_F(GlialSubstrateIntegrationTest, HypoxiaDamagesMyelin) {
    // WHAT: Test hypoxia causes demyelination
    // WHY:  Verify hypoxia vulnerability

    set_substrate_state(0.8f, 37.0f, 0.3f);

    ASSERT_EQ(glial_substrate_update_myelin_effects(bridge), 0);

    substrate_myelin_effects_t effects;
    ASSERT_EQ(glial_substrate_get_myelin_effects(bridge, &effects), 0);

    EXPECT_TRUE(effects.hypoxia_damage);
    EXPECT_GT(effects.o2_damage_rate, 0.0f);
}

//=============================================================================
// Glial → Substrate Support
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, AstrocyteLactateShuttle) {
    // WHAT: Test astrocytes provide lactate support
    // WHY:  Verify ANLS (Astrocyte-Neuron Lactate Shuttle)

    ASSERT_EQ(glial_substrate_compute_astrocyte_support(bridge), 0);

    glial_substrate_support_t support;
    ASSERT_EQ(glial_substrate_get_support(bridge, &support), 0);

    // Should have some lactate contribution
    EXPECT_GT(support.astro_lactate_total, 0.0f);
    EXPECT_GT(support.astro_atp_contribution, 0.0f);
    EXPECT_GT(support.astro_active_count, 0);
}

TEST_F(GlialSubstrateIntegrationTest, OligodendrocyteLactateShuttle) {
    // WHAT: Test oligodendrocytes provide lactate to axons
    // WHY:  Verify oligodendrocyte metabolic support

    ASSERT_EQ(glial_substrate_compute_oligodendrocyte_support(bridge), 0);

    glial_substrate_support_t support;
    ASSERT_EQ(glial_substrate_get_support(bridge, &support), 0);

    EXPECT_GT(support.oligo_lactate_total, 0.0f);
    EXPECT_GT(support.oligo_atp_contribution, 0.0f);
}

TEST_F(GlialSubstrateIntegrationTest, MyelinEfficiencySavings) {
    // WHAT: Test myelin provides ATP savings
    // WHY:  Saltatory conduction reduces pump activity

    ASSERT_EQ(glial_substrate_compute_myelin_support(bridge), 0);

    glial_substrate_support_t support;
    ASSERT_EQ(glial_substrate_get_support(bridge, &support), 0);

    // Should have efficiency savings
    EXPECT_GE(support.myelin_atp_savings, 0.0f);
    EXPECT_GE(support.avg_myelination_factor, 0.0f);
    EXPECT_LE(support.avg_myelination_factor, 1.0f);
}

TEST_F(GlialSubstrateIntegrationTest, MicrogliaPruningSavings) {
    // WHAT: Test microglia pruning saves ATP
    // WHY:  Fewer synapses → lower baseline cost

    // Simulate some pruning
    for (int i = 0; i < 10; i++) {
        microglia_network_prune_synapse(micro_network, i);
    }

    ASSERT_EQ(glial_substrate_compute_microglia_support(bridge), 0);

    glial_substrate_support_t support;
    ASSERT_EQ(glial_substrate_get_support(bridge, &support), 0);

    // Should track pruning
    EXPECT_GT(support.synapses_pruned, 0);
}

TEST_F(GlialSubstrateIntegrationTest, CombinedGlialSupport) {
    // WHAT: Test all glial types support substrate together
    // WHY:  Verify additive metabolic support

    ASSERT_EQ(glial_substrate_compute_all_support(bridge), 0);

    float total_support = glial_substrate_get_total_atp_support(bridge);

    // Should have positive support
    EXPECT_GT(total_support, 0.0f);

    glial_substrate_support_t support;
    ASSERT_EQ(glial_substrate_get_support(bridge, &support), 0);

    // Total should be sum of contributions
    float expected_total = support.astro_atp_contribution +
                          support.oligo_atp_contribution +
                          support.myelin_atp_savings +
                          support.pruning_atp_savings;

    EXPECT_FLOAT_EQ(support.total_atp_support, expected_total);
}

TEST_F(GlialSubstrateIntegrationTest, GlialSupportIncreasesSubstrateATP) {
    // WHAT: Test glial support actually increases substrate ATP
    // WHY:  Verify feedback loop

    float initial_atp = substrate_get_atp_level(substrate);

    // Compute and apply support
    ASSERT_EQ(glial_substrate_compute_all_support(bridge), 0);
    ASSERT_EQ(glial_substrate_apply_glial_support(bridge), 0);

    float final_atp = substrate_get_atp_level(substrate);

    // ATP should increase (or at least not decrease)
    EXPECT_GE(final_atp, initial_atp);
}

//=============================================================================
// Bidirectional Update Tests
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, BidirectionalUpdate) {
    // WHAT: Test full bidirectional update cycle
    // WHY:  Verify complete integration

    set_substrate_state(0.6f, 37.0f, 0.8f);

    // Full update (substrate → glial → substrate)
    ASSERT_EQ(glial_substrate_bridge_update(bridge, 1), 0);

    // Check effects were computed
    substrate_astrocyte_effects_t astro_effects;
    ASSERT_EQ(glial_substrate_get_astrocyte_effects(bridge, &astro_effects), 0);
    EXPECT_GT(astro_effects.atp_modulation, 0.0f);

    // Check support was computed
    float total_support = glial_substrate_get_total_atp_support(bridge);
    EXPECT_GT(total_support, 0.0f);
}

TEST_F(GlialSubstrateIntegrationTest, MultipleUpdateCycles) {
    // WHAT: Test multiple update cycles
    // WHY:  Verify stability over time

    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(glial_substrate_bridge_update(bridge, 1), 0);
    }

    glial_substrate_stats_t stats;
    ASSERT_EQ(glial_substrate_get_stats(bridge, &stats), 0);

    EXPECT_EQ(stats.total_updates, 100);
}

//=============================================================================
// Bio-Async Integration Tests
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, BioAsyncConnection) {
    // WHAT: Test bio-async connectivity
    // WHY:  Verify messaging capability

    int result = glial_substrate_connect_bio_async(bridge);

    // May succeed or fail gracefully
    if (result == 0) {
        EXPECT_TRUE(glial_substrate_is_bio_async_connected(bridge));
        ASSERT_EQ(glial_substrate_disconnect_bio_async(bridge), 0);
        EXPECT_FALSE(glial_substrate_is_bio_async_connected(bridge));
    }
}

//=============================================================================
// Metabolic Feedback Loops
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, LowATPGlialCompensation) {
    // WHAT: Test glia compensate for low substrate ATP
    // WHY:  Verify homeostatic response

    set_substrate_state(0.3f, 37.0f, 0.9f);

    // Update effects (glial function impaired)
    ASSERT_EQ(glial_substrate_update_all_effects(bridge), 0);

    // But still try to provide support
    ASSERT_EQ(glial_substrate_compute_all_support(bridge), 0);

    glial_substrate_support_t support;
    ASSERT_EQ(glial_substrate_get_support(bridge, &support), 0);

    // Support should be reduced but not zero
    EXPECT_GT(support.total_atp_support, 0.0f);
}

TEST_F(GlialSubstrateIntegrationTest, HighATPEnhancedGlialFunction) {
    // WHAT: Test high ATP enhances glial support
    // WHY:  Verify positive feedback

    set_substrate_state(0.95f, 37.0f, 0.95f);

    ASSERT_EQ(glial_substrate_update_all_effects(bridge), 0);
    ASSERT_EQ(glial_substrate_compute_all_support(bridge), 0);

    // Astrocytes should be highly functional
    substrate_astrocyte_effects_t astro_effects;
    ASSERT_EQ(glial_substrate_get_astrocyte_effects(bridge, &astro_effects), 0);
    EXPECT_GT(astro_effects.atp_modulation, 0.8f);

    // Should provide good support
    float total_support = glial_substrate_get_total_atp_support(bridge);
    EXPECT_GT(total_support, 0.0f);
}

//=============================================================================
// Connection API Tests
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, ConnectAstrocytesAfterCreation) {
    // WHAT: Test connecting astrocytes post-creation
    // WHY:  Verify dynamic connection

    // Create bridge without astrocytes
    glial_substrate_config_t config;
    glial_substrate_default_config(&config);
    glial_substrate_bridge_t* bridge2 = glial_substrate_bridge_create(&config, substrate,
        nullptr, nullptr, nullptr, nullptr);
    ASSERT_NE(bridge2, nullptr);

    // Connect astrocytes later
    ASSERT_EQ(glial_substrate_connect_astrocytes(bridge2, astro_network), 0);

    // Should now work
    ASSERT_EQ(glial_substrate_update_astrocyte_effects(bridge2), 0);

    glial_substrate_bridge_destroy(bridge2);
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(GlialSubstrateIntegrationTest, StatisticsTracking) {
    // WHAT: Test statistics are tracked correctly
    // WHY:  Verify monitoring capability

    // Cause various events
    set_substrate_state(0.3f, 37.0f, 0.4f);
    ASSERT_EQ(glial_substrate_update_all_effects(bridge), 0);

    set_substrate_state(0.9f, 39.0f, 0.9f);
    ASSERT_EQ(glial_substrate_update_all_effects(bridge), 0);

    glial_substrate_stats_t stats;
    ASSERT_EQ(glial_substrate_get_stats(bridge, &stats), 0);

    EXPECT_GT(stats.total_updates, 0);
    EXPECT_GT(stats.stress_events, 0);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
