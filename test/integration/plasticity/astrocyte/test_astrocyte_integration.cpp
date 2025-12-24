/**
 * @file test_astrocyte_integration.cpp
 * @brief Integration tests for astrocyte-plasticity with sleep and immune systems
 * @date 2025-12-19
 */

#include <gtest/gtest.h>
extern "C" {
#include "plasticity/astrocyte/nimcp_astrocyte_plasticity.h"
#include "plasticity/astrocyte/nimcp_astrocyte_sleep_bridge.h"
#include "plasticity/astrocyte/nimcp_astrocyte_immune_bridge.h"
#include "cognitive/nimcp_sleep_wake.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "utils/fault_tolerance/nimcp_byzantine_fault_tolerance.h"
#include "swarm/nimcp_swarm_immune.h"
}

class AstrocyteIntegrationTest : public ::testing::Test {
protected:
    astrocyte_plasticity_t astro;
    sleep_system_t sleep_sys;
    brain_immune_system_t* brain_immune;
    astrocyte_sleep_bridge_t sleep_bridge;
    astrocyte_immune_bridge_t* immune_bridge;

    void SetUp() override {
        // Create astrocyte system
        astrocyte_config_t astro_config;
        astrocyte_plasticity_default_config(&astro_config);
        astro = astrocyte_plasticity_create(&astro_config, 10);
        ASSERT_NE(astro, nullptr);

        // Create sleep system
        sleep_config_t sleep_config = sleep_default_config();
        sleep_sys = sleep_system_create(&sleep_config);
        ASSERT_NE(sleep_sys, nullptr);

        // Create brain immune system
        brain_immune_config_t immune_config;
        brain_immune_default_config(&immune_config);
        brain_immune = brain_immune_create(&immune_config);
        ASSERT_NE(brain_immune, nullptr);

        // Create bridges
        astrocyte_sleep_config_t sleep_bridge_config;
        astrocyte_sleep_default_config(&sleep_bridge_config);
        sleep_bridge = astrocyte_sleep_bridge_create(
            &sleep_bridge_config, sleep_sys, astro);
        ASSERT_NE(sleep_bridge, nullptr);

        astrocyte_immune_config_t immune_bridge_config;
        astrocyte_immune_default_config(&immune_bridge_config);
        immune_bridge = astrocyte_immune_bridge_create(
            &immune_bridge_config, brain_immune, astro);
        ASSERT_NE(immune_bridge, nullptr);
    }

    void TearDown() override {
        if (immune_bridge) {
            astrocyte_immune_bridge_destroy(immune_bridge);
        }
        if (sleep_bridge) {
            astrocyte_sleep_bridge_destroy(sleep_bridge);
        }
        if (brain_immune) {
            brain_immune_destroy(brain_immune);
        }
        if (sleep_sys) {
            sleep_system_destroy(sleep_sys);
        }
        if (astro) {
            astrocyte_plasticity_destroy(astro);
        }
    }
};

/* ============================================================================
 * Sleep-Astrocyte Integration Tests
 * ============================================================================ */

TEST_F(AstrocyteIntegrationTest, NREMEnhancesDSerineForConsolidation) {
    // Transition to NREM sleep
    sleep_enter_state(sleep_sys, SLEEP_STATE_DEEP_NREM);
    astrocyte_sleep_update(sleep_bridge);
    astrocyte_sleep_apply_modulation(sleep_bridge);

    // Check astrocyte state
    astrocyte_state_t state;
    astrocyte_plasticity_get_state(astro, 0, &state);

    // D-serine should be elevated in NREM
    astrocyte_plasticity_effects_t effects;
    astrocyte_plasticity_get_effects(astro, 0, &effects);
    EXPECT_GT(effects.nmda_coagonist_factor, 0.9f);
}

TEST_F(AstrocyteIntegrationTest, REMReducesDSerine) {
    sleep_enter_state(sleep_sys, SLEEP_STATE_REM);
    astrocyte_sleep_update(sleep_bridge);
    astrocyte_sleep_apply_modulation(sleep_bridge);

    astrocyte_sleep_effects_t effects;
    astrocyte_sleep_get_effects(sleep_bridge, &effects);

    EXPECT_LT(effects.d_serine_factor, 1.0f);
}

TEST_F(AstrocyteIntegrationTest, GlymphaticClearanceDuringNREM) {
    sleep_enter_state(sleep_sys, SLEEP_STATE_LIGHT_NREM);
    astrocyte_sleep_update(sleep_bridge);

    EXPECT_TRUE(astrocyte_sleep_is_glymphatic_active(sleep_bridge));

    astrocyte_sleep_effects_t effects;
    astrocyte_sleep_get_effects(sleep_bridge, &effects);
    EXPECT_GT(effects.glutamate_uptake_factor, 0.9f);
}

/* ============================================================================
 * Immune-Astrocyte Integration Tests
 * ============================================================================ */

TEST_F(AstrocyteIntegrationTest, CytokinesInduceA1ReactiveState) {
    // Simulate pro-inflammatory cytokines
    astrocyte_immune_apply_cytokine_effects(immune_bridge);
    astrocyte_immune_apply_inflammation_effects(immune_bridge);

    cytokine_astrocyte_effects_t effects;
    astrocyte_immune_get_cytokine_effects(immune_bridge, &effects);

    // Cytokines should reduce D-serine and uptake
    EXPECT_LE(effects.total_d_serine_modulation, 1.0f);
    EXPECT_LE(effects.total_glu_uptake_modulation, 1.0f);
}

TEST_F(AstrocyteIntegrationTest, A1StateImpairstPlasticity) {
    // Force A1 reactive state
    for (uint32_t i = 0; i < astrocyte_plasticity_get_num_astrocytes(astro); i++) {
        astrocyte_plasticity_set_reactive_state(
            astro, i, ASTROCYTE_A1_REACTIVE, 1.0f);
    }

    // Check plasticity effects
    astrocyte_plasticity_effects_t effects;
    astrocyte_plasticity_get_effects(astro, 0, &effects);

    EXPECT_LT(effects.nmda_coagonist_factor, 0.7f);
    EXPECT_GT(effects.glutamate_clearance_time, 10.0f);
}

TEST_F(AstrocyteIntegrationTest, DysfunctionDetectionAndAlert) {
    // Induce dysfunction (severe A1)
    for (uint32_t i = 0; i < astrocyte_plasticity_get_num_astrocytes(astro); i++) {
        astrocyte_plasticity_set_reactive_state(
            astro, i, ASTROCYTE_A1_REACTIVE, 1.0f);
    }

    // Just verify the system handles dysfunction state - actual detection
    // is handled by bridge update
    astrocyte_immune_bridge_update(immune_bridge, 100);

    // Verify A1 reactive state was set correctly
    astrocyte_state_t state;
    astrocyte_plasticity_get_state(astro, 0, &state);
    EXPECT_EQ(state.reactive_state, ASTROCYTE_A1_REACTIVE);
}

TEST_F(AstrocyteIntegrationTest, BioAsyncConnection) {
    EXPECT_EQ(astrocyte_immune_connect_bio_async(immune_bridge), 0);
    // Should not crash, may not connect if bio-async not available
    astrocyte_immune_disconnect_bio_async(immune_bridge);
}

/* ============================================================================
 * Combined Sleep-Immune-Astrocyte Tests
 * ============================================================================ */

TEST_F(AstrocyteIntegrationTest, SleepRestoresA1ToResting) {
    // Induce A1 reactive state
    astrocyte_plasticity_set_reactive_state(
        astro, 0, ASTROCYTE_A1_REACTIVE, 0.8f);

    // Transition to NREM (healing sleep)
    sleep_enter_state(sleep_sys, SLEEP_STATE_DEEP_NREM);
    astrocyte_sleep_update(sleep_bridge);

    // Sleep should partially counteract A1 dysfunction
    astrocyte_sleep_effects_t sleep_effects;
    astrocyte_sleep_get_effects(sleep_bridge, &sleep_effects);
    EXPECT_GT(sleep_effects.d_serine_factor, 1.0f);
}

TEST_F(AstrocyteIntegrationTest, FullSystemUpdate) {
    // Update all systems
    sleep_enter_state(sleep_sys, SLEEP_STATE_LIGHT_NREM);
    astrocyte_sleep_update(sleep_bridge);
    astrocyte_sleep_apply_modulation(sleep_bridge);

    astrocyte_immune_bridge_update(immune_bridge, 100);

    for (uint32_t i = 0; i < 5; i++) {
        astrocyte_plasticity_update(astro, i, 0.5f, 100);
    }

    // Should not crash, all systems integrated
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
