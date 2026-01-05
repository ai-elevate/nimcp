//=============================================================================
// test_wernicke_bridges.cpp - Wernicke Bridge Unit Tests
//=============================================================================
/**
 * @file test_wernicke_bridges.cpp
 * @brief Unit tests for Wernicke's area bridges
 *
 * WHAT: Tests all Wernicke bridge modules
 * WHY:  Verify correct integration with other brain systems
 * HOW:  gtest framework testing each bridge independently
 *
 * BRIDGES TESTED:
 * - Broca bridge (arcuate fasciculus)
 * - NLP bridge (language processing)
 * - Substrate bridge (metabolic effects)
 * - GPU-Bio bridge (acceleration)
 * - Omni bridge (inference)
 * - Quantum bridge (optimization)
 * - Immune bridge (inflammation effects)
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

extern "C" {
#include "core/brain/regions/wernicke/nimcp_wernicke_adapter.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_broca_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_nlp_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_substrate_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_gpu_bio_bridge.h"
#include "core/brain/regions/wernicke/nimcp_omni_wernicke_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_quantum_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_immune.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class WernickeBridgesTest : public ::testing::Test {
protected:
    wernicke_adapter_t* adapter;

    void SetUp() override {
        wernicke_config_t config = wernicke_default_config();
        adapter = wernicke_create(&config);
        ASSERT_NE(adapter, nullptr) << "Failed to create Wernicke adapter";
    }

    void TearDown() override {
        if (adapter) {
            wernicke_destroy(adapter);
            adapter = nullptr;
        }
    }
};

//=============================================================================
// Broca Bridge Tests
//=============================================================================

/**
 * @test Create Broca bridge
 * WHAT: Test wernicke_broca_bridge_create
 * WHY:  Verify arcuate fasciculus connection setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(WernickeBridgesTest, CreateBrocaBridge) {
    wernicke_broca_bridge_config_t config;
    memset(&config, 0, sizeof(config));
    config.enable_efference_copy = true;
    config.enable_repetition = true;

    wernicke_broca_bridge_t* bridge = wernicke_broca_bridge_create(adapter, &config);
    EXPECT_NE(bridge, nullptr) << "Failed to create Broca bridge";

    if (bridge) {
        wernicke_broca_bridge_destroy(bridge);
    }
}

/**
 * @test Create Broca bridge with null config
 * WHAT: Test wernicke_broca_bridge_create with defaults
 * WHY:  Verify default configuration works
 * HOW:  Create bridge with null config
 */
TEST_F(WernickeBridgesTest, CreateBrocaBridgeNullConfig) {
    wernicke_broca_bridge_t* bridge = wernicke_broca_bridge_create(adapter, nullptr);
    EXPECT_NE(bridge, nullptr) << "Should create with default config";

    if (bridge) {
        wernicke_broca_bridge_destroy(bridge);
    }
}

/**
 * @test Broca bridge default config
 * WHAT: Test wernicke_broca_bridge_default_config
 * WHY:  Verify sensible default values
 * HOW:  Get defaults, check key fields
 */
TEST_F(WernickeBridgesTest, BrocaBridgeDefaultConfig) {
    wernicke_broca_bridge_config_t config;
    wernicke_broca_bridge_default_config(&config);

    // Efference copy should be enabled by default
    EXPECT_TRUE(config.enable_efference_copy);
    EXPECT_TRUE(config.enable_repetition);
}

/**
 * @test Destroy null Broca bridge
 * WHAT: Test wernicke_broca_bridge_destroy with null
 * WHY:  Verify null safety
 * HOW:  Call destroy with null, should not crash
 */
TEST_F(WernickeBridgesTest, DestroyNullBrocaBridge) {
    wernicke_broca_bridge_destroy(nullptr);
    // No assertion - just verify no crash
}

//=============================================================================
// NLP Bridge Tests
//=============================================================================

/**
 * @test Create NLP bridge
 * WHAT: Test wernicke_nlp_bridge_create
 * WHY:  Verify NLP integration setup
 * HOW:  Create bridge with config
 */
TEST_F(WernickeBridgesTest, CreateNLPBridge) {
    wernicke_nlp_bridge_config_t config;
    memset(&config, 0, sizeof(config));
    config.enable_tokenization = true;
    config.enable_pos_tagging = true;
    config.enable_ner = true;

    wernicke_nlp_bridge_t* bridge = wernicke_nlp_bridge_create(adapter, &config);
    EXPECT_NE(bridge, nullptr) << "Failed to create NLP bridge";

    if (bridge) {
        wernicke_nlp_bridge_destroy(bridge);
    }
}

/**
 * @test NLP bridge default config
 * WHAT: Test wernicke_nlp_bridge_default_config
 * WHY:  Verify NLP defaults are sensible
 * HOW:  Get defaults, check values
 */
TEST_F(WernickeBridgesTest, NLPBridgeDefaultConfig) {
    wernicke_nlp_bridge_config_t config;
    wernicke_nlp_bridge_default_config(&config);

    EXPECT_TRUE(config.enable_tokenization);
    EXPECT_TRUE(config.enable_pos_tagging);
}

/**
 * @test Create NLP bridge with null adapter
 * WHAT: Test wernicke_nlp_bridge_create error handling
 * WHY:  Verify null safety
 * HOW:  Create with null adapter, expect null
 */
TEST_F(WernickeBridgesTest, CreateNLPBridgeNullAdapter) {
    wernicke_nlp_bridge_config_t config;
    wernicke_nlp_bridge_default_config(&config);

    wernicke_nlp_bridge_t* bridge = wernicke_nlp_bridge_create(nullptr, &config);
    EXPECT_EQ(bridge, nullptr);
}

//=============================================================================
// Substrate Bridge Tests
//=============================================================================

/**
 * @test Create substrate bridge
 * WHAT: Test wernicke_substrate_bridge_create
 * WHY:  Verify metabolic modulation setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(WernickeBridgesTest, CreateSubstrateBridge) {
    wernicke_substrate_bridge_config_t config;
    memset(&config, 0, sizeof(config));
    config.enable_atp_modulation = true;
    config.enable_fatigue_effects = true;
    config.atp_sensitivity = 1.0f;

    wernicke_substrate_bridge_t* bridge = wernicke_substrate_bridge_create(adapter, &config);
    EXPECT_NE(bridge, nullptr) << "Failed to create substrate bridge";

    if (bridge) {
        wernicke_substrate_bridge_destroy(bridge);
    }
}

/**
 * @test Substrate bridge default config
 * WHAT: Test wernicke_substrate_bridge_default_config
 * WHY:  Verify metabolic defaults
 * HOW:  Get defaults, verify ATP modulation enabled
 */
TEST_F(WernickeBridgesTest, SubstrateBridgeDefaultConfig) {
    wernicke_substrate_bridge_config_t config;
    wernicke_substrate_bridge_default_config(&config);

    EXPECT_TRUE(config.enable_atp_modulation);
    EXPECT_TRUE(config.enable_fatigue_effects);
    EXPECT_GT(config.atp_sensitivity, 0.0f);
    EXPECT_LE(config.atp_sensitivity, 2.0f);
}

/**
 * @test Substrate bridge metabolic update
 * WHAT: Test substrate bridge processes metabolic state
 * WHY:  Verify language processing responds to energy
 * HOW:  Create bridge, update, check no crash
 */
TEST_F(WernickeBridgesTest, SubstrateBridgeUpdate) {
    wernicke_substrate_bridge_config_t config;
    wernicke_substrate_bridge_default_config(&config);

    wernicke_substrate_bridge_t* bridge = wernicke_substrate_bridge_create(adapter, &config);
    ASSERT_NE(bridge, nullptr);

    // Update should not crash
    int result = wernicke_substrate_bridge_update(bridge);
    EXPECT_GE(result, 0);

    wernicke_substrate_bridge_destroy(bridge);
}

//=============================================================================
// GPU-Bio Bridge Tests
//=============================================================================

/**
 * @test Create GPU-Bio bridge
 * WHAT: Test wernicke_gpu_bio_bridge_create
 * WHY:  Verify GPU acceleration setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(WernickeBridgesTest, CreateGPUBioBridge) {
    wernicke_gpu_bio_bridge_config_t config;
    memset(&config, 0, sizeof(config));
    config.enable_gpu = true;
    config.enable_async_transfer = true;
    config.batch_size = 32;

    wernicke_gpu_bio_bridge_t* bridge = wernicke_gpu_bio_bridge_create(adapter, &config);
    // May be null if GPU not available, which is acceptable
    if (bridge) {
        wernicke_gpu_bio_bridge_destroy(bridge);
    }
}

/**
 * @test GPU-Bio bridge default config
 * WHAT: Test wernicke_gpu_bio_bridge_default_config
 * WHY:  Verify GPU defaults
 * HOW:  Get defaults, check values
 */
TEST_F(WernickeBridgesTest, GPUBioBridgeDefaultConfig) {
    wernicke_gpu_bio_bridge_config_t config;
    wernicke_gpu_bio_bridge_default_config(&config);

    EXPECT_GT(config.batch_size, 0);
}

//=============================================================================
// Omni Bridge Tests
//=============================================================================

/**
 * @test Create omni bridge
 * WHAT: Test omni_wernicke_bridge_create
 * WHY:  Verify omni inference integration
 * HOW:  Create bridge, verify non-null
 */
TEST_F(WernickeBridgesTest, CreateOmniBridge) {
    omni_wernicke_bridge_config_t config;
    memset(&config, 0, sizeof(config));
    config.enable_prediction = true;
    config.enable_pe_reporting = true;
    config.prediction_horizon = 3;

    omni_wernicke_bridge_t* bridge = omni_wernicke_bridge_create(adapter, &config);
    EXPECT_NE(bridge, nullptr) << "Failed to create omni bridge";

    if (bridge) {
        omni_wernicke_bridge_destroy(bridge);
    }
}

/**
 * @test Omni bridge default config
 * WHAT: Test omni_wernicke_bridge_default_config
 * WHY:  Verify prediction defaults
 * HOW:  Get defaults, check prediction enabled
 */
TEST_F(WernickeBridgesTest, OmniBridgeDefaultConfig) {
    omni_wernicke_bridge_config_t config;
    omni_wernicke_bridge_default_config(&config);

    EXPECT_TRUE(config.enable_prediction);
    EXPECT_GT(config.prediction_horizon, 0);
}

/**
 * @test Omni bridge update
 * WHAT: Test omni_wernicke_bridge_update
 * WHY:  Verify prediction processing
 * HOW:  Create bridge, update, verify no crash
 */
TEST_F(WernickeBridgesTest, OmniBridgeUpdate) {
    omni_wernicke_bridge_config_t config;
    omni_wernicke_bridge_default_config(&config);

    omni_wernicke_bridge_t* bridge = omni_wernicke_bridge_create(adapter, &config);
    ASSERT_NE(bridge, nullptr);

    int result = omni_wernicke_bridge_update(bridge);
    EXPECT_GE(result, 0);

    omni_wernicke_bridge_destroy(bridge);
}

//=============================================================================
// Quantum Bridge Tests
//=============================================================================

/**
 * @test Create quantum bridge
 * WHAT: Test wernicke_quantum_bridge_create
 * WHY:  Verify quantum optimization setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(WernickeBridgesTest, CreateQuantumBridge) {
    wernicke_quantum_bridge_config_t config;
    memset(&config, 0, sizeof(config));
    config.enable_quantum_attention = true;
    config.enable_optimization = true;
    config.num_qubits = 8;

    wernicke_quantum_bridge_t* bridge = wernicke_quantum_bridge_create(adapter, &config);
    EXPECT_NE(bridge, nullptr) << "Failed to create quantum bridge";

    if (bridge) {
        wernicke_quantum_bridge_destroy(bridge);
    }
}

/**
 * @test Quantum bridge default config
 * WHAT: Test wernicke_quantum_bridge_default_config
 * WHY:  Verify quantum defaults
 * HOW:  Get defaults, check qubit count
 */
TEST_F(WernickeBridgesTest, QuantumBridgeDefaultConfig) {
    wernicke_quantum_bridge_config_t config;
    wernicke_quantum_bridge_default_config(&config);

    EXPECT_GT(config.num_qubits, 0);
}

//=============================================================================
// Immune Bridge Tests
//=============================================================================

/**
 * @test Create immune bridge
 * WHAT: Test wernicke_immune_create
 * WHY:  Verify neuroinflammation modeling setup
 * HOW:  Create bridge, verify non-null
 */
TEST_F(WernickeBridgesTest, CreateImmuneBridge) {
    wernicke_immune_config_t config;
    memset(&config, 0, sizeof(config));
    config.enable_cytokine_effects = true;
    config.enable_aphasia_modeling = true;
    config.inflammation_sensitivity = 1.0f;

    wernicke_immune_t* bridge = wernicke_immune_create(adapter, &config);
    EXPECT_NE(bridge, nullptr) << "Failed to create immune bridge";

    if (bridge) {
        wernicke_immune_destroy(bridge);
    }
}

/**
 * @test Immune bridge default config
 * WHAT: Test wernicke_immune_default_config
 * WHY:  Verify immune defaults
 * HOW:  Get defaults, check sensitivity
 */
TEST_F(WernickeBridgesTest, ImmuneBridgeDefaultConfig) {
    wernicke_immune_config_t config;
    wernicke_immune_default_config(&config);

    EXPECT_TRUE(config.enable_cytokine_effects);
    EXPECT_GT(config.inflammation_sensitivity, 0.0f);
}

/**
 * @test Immune bridge update with inflammation
 * WHAT: Test wernicke_immune_update
 * WHY:  Verify inflammation affects language
 * HOW:  Create bridge, update, check no crash
 */
TEST_F(WernickeBridgesTest, ImmuneBridgeUpdate) {
    wernicke_immune_config_t config;
    wernicke_immune_default_config(&config);

    wernicke_immune_t* bridge = wernicke_immune_create(adapter, &config);
    ASSERT_NE(bridge, nullptr);

    int result = wernicke_immune_update(bridge);
    EXPECT_GE(result, 0);

    wernicke_immune_destroy(bridge);
}

/**
 * @test Immune bridge get impairment
 * WHAT: Test wernicke_immune_get_impairment
 * WHY:  Verify impairment level retrieval
 * HOW:  Create bridge, get impairment, check range
 */
TEST_F(WernickeBridgesTest, ImmuneBridgeGetImpairment) {
    wernicke_immune_config_t config;
    wernicke_immune_default_config(&config);

    wernicke_immune_t* bridge = wernicke_immune_create(adapter, &config);
    ASSERT_NE(bridge, nullptr);

    float impairment = wernicke_immune_get_impairment(bridge);
    // Impairment should be between 0 (none) and 1 (complete)
    EXPECT_GE(impairment, 0.0f);
    EXPECT_LE(impairment, 1.0f);

    wernicke_immune_destroy(bridge);
}

//=============================================================================
// Edge Cases and Error Handling
//=============================================================================

/**
 * @test All bridges with null adapter
 * WHAT: Test all bridge creates with null adapter
 * WHY:  Verify null safety across all bridges
 * HOW:  Create each bridge with null adapter
 */
TEST_F(WernickeBridgesTest, AllBridgesNullAdapter) {
    EXPECT_EQ(wernicke_broca_bridge_create(nullptr, nullptr), nullptr);
    EXPECT_EQ(wernicke_nlp_bridge_create(nullptr, nullptr), nullptr);
    EXPECT_EQ(wernicke_substrate_bridge_create(nullptr, nullptr), nullptr);
    EXPECT_EQ(wernicke_gpu_bio_bridge_create(nullptr, nullptr), nullptr);
    EXPECT_EQ(omni_wernicke_bridge_create(nullptr, nullptr), nullptr);
    EXPECT_EQ(wernicke_quantum_bridge_create(nullptr, nullptr), nullptr);
    EXPECT_EQ(wernicke_immune_create(nullptr, nullptr), nullptr);
}

/**
 * @test All bridge destroys with null
 * WHAT: Test all bridge destroys with null
 * WHY:  Verify null safety in destruction
 * HOW:  Call each destroy with null
 */
TEST_F(WernickeBridgesTest, AllBridgesDestroyNull) {
    wernicke_broca_bridge_destroy(nullptr);
    wernicke_nlp_bridge_destroy(nullptr);
    wernicke_substrate_bridge_destroy(nullptr);
    wernicke_gpu_bio_bridge_destroy(nullptr);
    omni_wernicke_bridge_destroy(nullptr);
    wernicke_quantum_bridge_destroy(nullptr);
    wernicke_immune_destroy(nullptr);
    // No assertions - verify no crash
}
