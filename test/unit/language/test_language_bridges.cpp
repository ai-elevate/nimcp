//=============================================================================
// test_language_bridges.cpp - Language Layer Bridges Unit Tests
//=============================================================================
/**
 * @file test_language_bridges.cpp
 * @brief Unit tests for Language Layer Bridges
 *
 * WHAT: Tests all 9 language layer bridges
 * WHY:  Verify correct bridge lifecycle, configuration, and operation
 * HOW:  gtest framework testing each bridge independently
 *
 * BRIDGES TESTED:
 * - Perception bridge (speech cortex, audio cortex)
 * - Cognitive bridge (working memory, attention)
 * - Training bridge (plasticity, learning)
 * - Omni bridge (predictive inference)
 * - Immune bridge (inflammation effects)
 * - GPU bridge (acceleration)
 * - Thalamic bridge (signal routing)
 * - Substrate bridge (metabolic modulation)
 * - Logic bridge (symbolic reasoning)
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

extern "C" {
#include "language/nimcp_language_orchestrator.h"
#include "language/nimcp_language_config.h"
#include "language/bridges/nimcp_language_perception_bridge.h"
#include "language/bridges/nimcp_language_cognitive_bridge.h"
#include "language/bridges/nimcp_language_training_bridge.h"
#include "language/bridges/nimcp_language_omni_bridge.h"
#include "language/bridges/nimcp_language_immune_bridge.h"
#include "language/bridges/nimcp_language_gpu_bridge.h"
#include "language/bridges/nimcp_language_thalamic_bridge.h"
#include "language/bridges/nimcp_language_substrate_bridge.h"
#include "language/bridges/nimcp_language_logic_bridge.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class LanguageBridgesTest : public ::testing::Test {
protected:
    language_orchestrator_t* orchestrator = nullptr;

    void SetUp() override {
        // Create orchestrator for bridges that require it
        orchestrator = language_orchestrator_create(nullptr);
    }

    void TearDown() override {
        if (orchestrator) {
            language_orchestrator_destroy(orchestrator);
            orchestrator = nullptr;
        }
    }
};

//=============================================================================
// Perception Bridge Tests
//=============================================================================

/**
 * @test Create perception bridge
 * WHAT: Test language_perception_bridge_create
 * WHY:  Verify speech/audio integration setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(LanguageBridgesTest, CreatePerceptionBridge) {
    language_perception_config_t config;
    language_perception_default_config(&config);

    language_perception_bridge_t* bridge = language_perception_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        language_perception_bridge_destroy(bridge);
    }
}

/**
 * @test Perception bridge default config
 * WHAT: Test language_perception_default_config
 * WHY:  Verify sensible defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, PerceptionBridgeDefaultConfig) {
    language_perception_config_t config;
    memset(&config, 0, sizeof(config));
    language_perception_default_config(&config);

    // Speech should be enabled
    EXPECT_TRUE(config.enable_speech_cortex);
}

//=============================================================================
// Cognitive Bridge Tests
//=============================================================================

/**
 * @test Create cognitive bridge
 * WHAT: Test language_cognitive_bridge_create
 * WHY:  Verify working memory/attention integration
 * HOW:  Create bridge, verify non-null
 */
TEST_F(LanguageBridgesTest, CreateCognitiveBridge) {
    language_cognitive_config_t config;
    language_cognitive_default_config(&config);

    language_cognitive_bridge_t* bridge = language_cognitive_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        language_cognitive_bridge_destroy(bridge);
    }
}

/**
 * @test Cognitive bridge default config
 * WHAT: Test language_cognitive_default_config
 * WHY:  Verify cognitive defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, CognitiveBridgeDefaultConfig) {
    language_cognitive_config_t config;
    memset(&config, 0, sizeof(config));
    language_cognitive_default_config(&config);

    // Phonological buffer should match Miller's number
    EXPECT_GE(config.phonological_buffer_size, 5u);
    EXPECT_LE(config.phonological_buffer_size, 12u);

    // Working memory should be enabled
    EXPECT_TRUE(config.enable_working_memory);
}

//=============================================================================
// Training Bridge Tests
//=============================================================================

/**
 * @test Create training bridge
 * WHAT: Test language_training_bridge_create
 * WHY:  Verify plasticity integration
 * HOW:  Create bridge, verify non-null
 */
TEST_F(LanguageBridgesTest, CreateTrainingBridge) {
    language_training_config_t config;
    language_training_default_config(&config);

    language_training_bridge_t* bridge = language_training_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        language_training_bridge_destroy(bridge);
    }
}

/**
 * @test Training bridge default config
 * WHAT: Test language_training_default_config
 * WHY:  Verify training defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, TrainingBridgeDefaultConfig) {
    language_training_config_t config;
    memset(&config, 0, sizeof(config));
    language_training_default_config(&config);

    // Learning rates should be positive and small
    EXPECT_GT(config.vocabulary_learning_rate, 0.0f);
    EXPECT_LT(config.vocabulary_learning_rate, 1.0f);
}

//=============================================================================
// Omni Bridge Tests
//=============================================================================

/**
 * @test Create omni bridge
 * WHAT: Test language_omni_bridge_create
 * WHY:  Verify predictive integration
 * HOW:  Create bridge, verify non-null
 */
TEST_F(LanguageBridgesTest, CreateOmniBridge) {
    language_omni_config_t config;
    language_omni_default_config(&config);

    language_omni_bridge_t* bridge = language_omni_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        language_omni_bridge_destroy(bridge);
    }
}

/**
 * @test Omni bridge default config
 * WHAT: Test language_omni_default_config
 * WHY:  Verify prediction defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, OmniBridgeDefaultConfig) {
    language_omni_config_t config;
    memset(&config, 0, sizeof(config));
    language_omni_default_config(&config);

    // Prediction horizon should be positive
    EXPECT_GT(config.phoneme_prediction_horizon, 0u);
    EXPECT_GT(config.word_prediction_horizon, 0u);
}

//=============================================================================
// Immune Bridge Tests
//=============================================================================

/**
 * @test Create immune bridge
 * WHAT: Test language_immune_bridge_create
 * WHY:  Verify neuroinflammation integration
 * HOW:  Create bridge, verify non-null
 */
TEST_F(LanguageBridgesTest, CreateImmuneBridge) {
    language_immune_config_t config;
    language_immune_default_config(&config);

    language_immune_bridge_t* bridge = language_immune_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        language_immune_bridge_destroy(bridge);
    }
}

/**
 * @test Immune bridge default config
 * WHAT: Test language_immune_default_config
 * WHY:  Verify immune defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, ImmuneBridgeDefaultConfig) {
    language_immune_config_t config;
    memset(&config, 0, sizeof(config));
    language_immune_default_config(&config);

    // Cytokine sensitivity should be positive
    EXPECT_GT(config.il1b_sensitivity, 0.0f);
    EXPECT_GT(config.il6_sensitivity, 0.0f);
    EXPECT_GT(config.tnfa_sensitivity, 0.0f);
}

//=============================================================================
// GPU Bridge Tests
//=============================================================================

/**
 * @test Create GPU bridge
 * WHAT: Test language_gpu_bridge_create
 * WHY:  Verify GPU acceleration setup
 * HOW:  Create bridge (may be null if no GPU)
 */
TEST_F(LanguageBridgesTest, CreateGPUBridge) {
    language_gpu_config_t config;
    language_gpu_default_config(&config);

    language_gpu_bridge_t* bridge = language_gpu_bridge_create(&config);
    // May be null if GPU not available

    if (bridge) {
        language_gpu_bridge_destroy(bridge);
    }
}

/**
 * @test GPU bridge default config
 * WHAT: Test language_gpu_default_config
 * WHY:  Verify GPU defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, GPUBridgeDefaultConfig) {
    language_gpu_config_t config;
    memset(&config, 0, sizeof(config));
    language_gpu_default_config(&config);

    // Batch size should be reasonable
    EXPECT_GT(config.batch_size, 0u);
}

//=============================================================================
// Thalamic Bridge Tests
//=============================================================================

/**
 * @test Create thalamic bridge
 * WHAT: Test language_thalamic_bridge_create
 * WHY:  Verify signal routing setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(LanguageBridgesTest, CreateThalamicBridge) {
    ASSERT_NE(orchestrator, nullptr) << "Orchestrator required for thalamic bridge";

    language_thalamic_config_t config;
    language_thalamic_default_config(&config);

    language_thalamic_bridge_t* bridge = language_thalamic_bridge_create(orchestrator, &config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        language_thalamic_bridge_destroy(bridge);
    }
}

/**
 * @test Thalamic bridge default config
 * WHAT: Test language_thalamic_default_config
 * WHY:  Verify thalamic defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, ThalamicBridgeDefaultConfig) {
    language_thalamic_config_t config;
    memset(&config, 0, sizeof(config));
    language_thalamic_default_config(&config);

    // Attention gating should be enabled
    EXPECT_TRUE(config.enable_attention_gating);
}

//=============================================================================
// Substrate Bridge Tests
//=============================================================================

/**
 * @test Create substrate bridge
 * WHAT: Test language_substrate_bridge_create
 * WHY:  Verify metabolic modulation setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(LanguageBridgesTest, CreateSubstrateBridge) {
    ASSERT_NE(orchestrator, nullptr) << "Orchestrator required for substrate bridge";

    language_substrate_config_t config;
    language_substrate_default_config(&config);

    language_substrate_bridge_t* bridge = language_substrate_bridge_create(orchestrator, &config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        language_substrate_bridge_destroy(bridge);
    }
}

/**
 * @test Substrate bridge default config
 * WHAT: Test language_substrate_default_config
 * WHY:  Verify substrate defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, SubstrateBridgeDefaultConfig) {
    language_substrate_config_t config;
    memset(&config, 0, sizeof(config));
    language_substrate_default_config(&config);

    // ATP modulation should be enabled
    EXPECT_TRUE(config.enable_atp_modulation);
}

//=============================================================================
// Logic Bridge Tests
//=============================================================================

/**
 * @test Create logic bridge
 * WHAT: Test language_logic_bridge_create
 * WHY:  Verify symbolic reasoning setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(LanguageBridgesTest, CreateLogicBridge) {
    ASSERT_NE(orchestrator, nullptr) << "Orchestrator required for logic bridge";

    language_logic_config_t config;
    language_logic_default_config(&config);

    language_logic_bridge_t* bridge = language_logic_bridge_create(orchestrator, &config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        language_logic_bridge_destroy(bridge);
    }
}

/**
 * @test Logic bridge default config
 * WHAT: Test language_logic_default_config
 * WHY:  Verify logic defaults
 * HOW:  Get defaults, check key fields
 */
TEST_F(LanguageBridgesTest, LogicBridgeDefaultConfig) {
    language_logic_config_t config;
    memset(&config, 0, sizeof(config));
    language_logic_default_config(&config);

    // Entailment checking should be enabled
    EXPECT_TRUE(config.enable_entailment_checking);
    EXPECT_TRUE(config.enable_consistency_checking);

    // Inference depth should be reasonable
    EXPECT_GT(config.max_inference_depth, 0u);
    EXPECT_LT(config.max_inference_depth, 100u);
}

//=============================================================================
// All Bridges Null Safety Tests
//=============================================================================

/**
 * @test All bridges with null config
 * WHAT: Test all bridge creates with null config
 * WHY:  Verify null safety across all bridges
 * HOW:  Create each bridge with null config
 */
TEST_F(LanguageBridgesTest, AllBridgesNullConfig) {
    // These should either return null or create with defaults
    language_perception_bridge_t* pb = language_perception_bridge_create(nullptr);
    language_cognitive_bridge_t* cb = language_cognitive_bridge_create(nullptr);
    language_training_bridge_t* tb = language_training_bridge_create(nullptr);
    language_omni_bridge_t* ob = language_omni_bridge_create(nullptr);
    language_immune_bridge_t* ib = language_immune_bridge_create(nullptr);
    language_gpu_bridge_t* gb = language_gpu_bridge_create(nullptr);
    // Thalamic, substrate, logic bridges require orchestrator
    language_thalamic_bridge_t* thb = language_thalamic_bridge_create(orchestrator, nullptr);
    language_substrate_bridge_t* sb = language_substrate_bridge_create(orchestrator, nullptr);
    language_logic_bridge_t* lb = language_logic_bridge_create(orchestrator, nullptr);

    // Clean up any that were created
    if (pb) language_perception_bridge_destroy(pb);
    if (cb) language_cognitive_bridge_destroy(cb);
    if (tb) language_training_bridge_destroy(tb);
    if (ob) language_omni_bridge_destroy(ob);
    if (ib) language_immune_bridge_destroy(ib);
    if (gb) language_gpu_bridge_destroy(gb);
    if (thb) language_thalamic_bridge_destroy(thb);
    if (sb) language_substrate_bridge_destroy(sb);
    if (lb) language_logic_bridge_destroy(lb);
}

/**
 * @test All bridge destroys with null
 * WHAT: Test all bridge destroys with null
 * WHY:  Verify null safety in destruction
 * HOW:  Call each destroy with null
 */
TEST_F(LanguageBridgesTest, AllBridgesDestroyNull) {
    language_perception_bridge_destroy(nullptr);
    language_cognitive_bridge_destroy(nullptr);
    language_training_bridge_destroy(nullptr);
    language_omni_bridge_destroy(nullptr);
    language_immune_bridge_destroy(nullptr);
    language_gpu_bridge_destroy(nullptr);
    language_thalamic_bridge_destroy(nullptr);
    language_substrate_bridge_destroy(nullptr);
    language_logic_bridge_destroy(nullptr);
    // No crash = success
}

/**
 * @test All default config with null
 * WHAT: Test all default config functions with null
 * WHY:  Verify null safety in config functions
 * HOW:  Call each default config with null
 */
TEST_F(LanguageBridgesTest, AllDefaultConfigNull) {
    language_perception_default_config(nullptr);
    language_cognitive_default_config(nullptr);
    language_training_default_config(nullptr);
    language_omni_default_config(nullptr);
    language_immune_default_config(nullptr);
    language_gpu_default_config(nullptr);
    language_thalamic_default_config(nullptr);
    language_substrate_default_config(nullptr);
    language_logic_default_config(nullptr);
    // No crash = success
}

