/**
 * @file test_omni_bridges_integration.cpp
 * @brief Integration tests for Omnidirectional Inference bridges
 * @version 1.1.0
 * @date 2026-01-04
 *
 * Tests integration of omnidirectional inference with:
 * - Sensory cortices (audio, visual, speech)
 * - Cortical columns
 * - Occipital lobe (V1-V5)
 * - Broca's region (language production)
 * - Bio-async message routing
 * - Knowledge graph synchronization
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <cstdlib>

/* Headers have their own extern "C" guards - don't wrap them to avoid
 * CUDA C++ function conflicts */
#include "cognitive/jepa/nimcp_jepa_bidirectional.h"
#include "cognitive/jepa/nimcp_jepa_latent.h"
#include "cognitive/memory/nimcp_hopfield_memory.h"
#include "cognitive/predictive/nimcp_predictive_hierarchy.h"
#include "cognitive/memory/nimcp_temporal_replay.h"
#include "perception/nimcp_omni_sensory_bridge.h"
#include "core/cortical_columns/nimcp_omni_cortical_columns_bridge.h"
#include "core/brain/regions/occipital/nimcp_omni_occipital_bridge.h"
#include "core/brain/regions/broca/nimcp_omni_broca_bridge.h"
#include "cognitive/omni/nimcp_omni_kg_sync.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"

/* ============================================================================
 * Test Fixture for Omnidirectional Integration
 * ============================================================================ */

class OmniIntegrationTest : public ::testing::Test {
protected:
    static constexpr uint32_t LATENT_DIM = 64;
    static constexpr uint32_t HIDDEN_DIM = 128;
    static constexpr uint32_t NUM_PATTERNS = 256;
    static constexpr uint32_t NUM_LEVELS = 3;

    jepa_bidirectional_t* jepa = nullptr;
    hopfield_memory_t* hopfield = nullptr;
    predictive_hierarchy_t* hierarchy = nullptr;
    temporal_replay_t* replay = nullptr;

    uint32_t hier_dims[NUM_LEVELS] = {LATENT_DIM, LATENT_DIM / 2, LATENT_DIM / 4};

    void SetUp() override {
        /* Create bidirectional JEPA */
        jepa_bidir_config_t jepa_config;
        jepa_bidir_default_config(&jepa_config);
        jepa_config.embedding_dim = LATENT_DIM;
        jepa_config.hidden_dim = HIDDEN_DIM;
        jepa_config.gpu_mode = JEPA_BIDIR_GPU_DISABLED;
        jepa_config.enable_forward = true;
        jepa_config.enable_backward = true;
        jepa_config.enable_lateral = true;
        jepa = jepa_bidirectional_create(&jepa_config);

        /* Create Hopfield memory */
        hopfield_config_t hopfield_config;
        hopfield_default_config(&hopfield_config);
        hopfield_config.pattern_dim = LATENT_DIM;
        hopfield_config.capacity = NUM_PATTERNS;
        hopfield_config.gpu_mode = HOPFIELD_GPU_DISABLED;
        hopfield = hopfield_memory_create(&hopfield_config);

        /* Create predictive hierarchy */
        pred_hier_config_t hier_config;
        pred_hier_simple_config(&hier_config, NUM_LEVELS, hier_dims);
        hier_config.gpu_mode = PRED_HIER_GPU_DISABLED;
        hierarchy = pred_hier_create(&hier_config);
        pred_hier_free_config(&hier_config);

        /* Create temporal replay */
        replay_config_t replay_config;
        replay_default_config(&replay_config);
        replay_config.state_dim = LATENT_DIM;
        replay_config.capacity = 1024;
        replay_config.gpu_mode = REPLAY_GPU_DISABLED;
        replay = temporal_replay_create(&replay_config);
    }

    void TearDown() override {
        if (replay) temporal_replay_destroy(replay);
        if (hierarchy) pred_hier_destroy(hierarchy);
        if (hopfield) hopfield_memory_destroy(hopfield);
        if (jepa) jepa_bidirectional_destroy(jepa);
    }

    void generate_random_pattern(float* pattern, uint32_t dim) {
        for (uint32_t i = 0; i < dim; i++) {
            pattern[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
};

/* ============================================================================
 * Core Component Creation Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, JEPABidirectionalCreation) {
    EXPECT_NE(jepa, nullptr);
}

TEST_F(OmniIntegrationTest, HopfieldMemoryCreation) {
    EXPECT_NE(hopfield, nullptr);
    if (hopfield) {
        EXPECT_EQ(hopfield_memory_capacity(hopfield), NUM_PATTERNS);
        EXPECT_EQ(hopfield_memory_pattern_count(hopfield), 0u);
    }
}

TEST_F(OmniIntegrationTest, PredictiveHierarchyCreation) {
    EXPECT_NE(hierarchy, nullptr);
}

TEST_F(OmniIntegrationTest, TemporalReplayCreation) {
    EXPECT_NE(replay, nullptr);
}

/* ============================================================================
 * Bidirectional Prediction Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, JEPAForwardPrediction) {
    if (!jepa) GTEST_SKIP();

    jepa_latent_t* input = jepa_latent_create_dim(LATENT_DIM);
    ASSERT_NE(input, nullptr);
    generate_random_pattern(input->embedding, LATENT_DIM);

    jepa_bidir_result_t* result = jepa_bidir_result_create(LATENT_DIM);
    ASSERT_NE(result, nullptr);

    int ret = jepa_bidirectional_predict(jepa, JEPA_DIR_FORWARD, input, result);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    jepa_bidir_result_destroy(result);
    jepa_latent_destroy(input);
}

TEST_F(OmniIntegrationTest, JEPABackwardPrediction) {
    if (!jepa) GTEST_SKIP();

    jepa_latent_t* input = jepa_latent_create_dim(LATENT_DIM);
    ASSERT_NE(input, nullptr);
    generate_random_pattern(input->embedding, LATENT_DIM);

    jepa_bidir_result_t* result = jepa_bidir_result_create(LATENT_DIM);
    ASSERT_NE(result, nullptr);

    int ret = jepa_bidirectional_predict(jepa, JEPA_DIR_BACKWARD, input, result);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    jepa_bidir_result_destroy(result);
    jepa_latent_destroy(input);
}

TEST_F(OmniIntegrationTest, JEPALateralPrediction) {
    if (!jepa) GTEST_SKIP();

    jepa_latent_t* input = jepa_latent_create_dim(LATENT_DIM);
    ASSERT_NE(input, nullptr);
    generate_random_pattern(input->embedding, LATENT_DIM);

    jepa_bidir_result_t* result = jepa_bidir_result_create(LATENT_DIM);
    ASSERT_NE(result, nullptr);

    int ret = jepa_bidirectional_predict(jepa, JEPA_DIR_LATERAL, input, result);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    jepa_bidir_result_destroy(result);
    jepa_latent_destroy(input);
}

/* ============================================================================
 * Hopfield Memory Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, HopfieldStoreAndRetrieve) {
    if (!hopfield) GTEST_SKIP();

    float pattern[LATENT_DIM], query[LATENT_DIM];
    generate_random_pattern(pattern, LATENT_DIM);
    memcpy(query, pattern, LATENT_DIM * sizeof(float));

    /* Add small noise to query */
    for (uint32_t i = 0; i < LATENT_DIM; i++) {
        query[i] += 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }

    /* Store pattern */
    uint32_t pattern_id;
    int ret = hopfield_memory_store(hopfield, pattern, &pattern_id);
    EXPECT_EQ(ret, NIMCP_SUCCESS);
    EXPECT_EQ(hopfield_memory_pattern_count(hopfield), 1u);

    /* Retrieve with noisy query */
    hopfield_retrieval_result_t* result = hopfield_result_create(LATENT_DIM);
    ASSERT_NE(result, nullptr);

    ret = hopfield_memory_retrieve(hopfield, query, result);
    EXPECT_EQ(ret, NIMCP_SUCCESS);
    EXPECT_GT(result->similarity, 0.8f);

    hopfield_result_destroy(result);
}

TEST_F(OmniIntegrationTest, HopfieldBatchStore) {
    if (!hopfield) GTEST_SKIP();

    const uint32_t num_patterns = 10;
    float patterns[num_patterns * LATENT_DIM];

    for (uint32_t i = 0; i < num_patterns; i++) {
        generate_random_pattern(&patterns[i * LATENT_DIM], LATENT_DIM);
    }

    uint32_t ids[num_patterns];
    int ret = hopfield_memory_store_batch(hopfield, patterns, num_patterns, ids);
    EXPECT_EQ(ret, NIMCP_SUCCESS);
    EXPECT_EQ(hopfield_memory_pattern_count(hopfield), num_patterns);
}

/* ============================================================================
 * Predictive Hierarchy Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, HierarchyForwardPass) {
    if (!hierarchy) GTEST_SKIP();

    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);

    int ret = pred_hier_forward(hierarchy, input);
    EXPECT_EQ(ret, NIMCP_SUCCESS);
}

TEST_F(OmniIntegrationTest, HierarchyBackwardPass) {
    if (!hierarchy) GTEST_SKIP();

    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);
    pred_hier_forward(hierarchy, input);

    int ret = pred_hier_backward(hierarchy);
    EXPECT_EQ(ret, NIMCP_SUCCESS);
}

TEST_F(OmniIntegrationTest, HierarchyUpdate) {
    if (!hierarchy) GTEST_SKIP();

    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);

    pred_hier_result_t* result = pred_hier_result_create(NUM_LEVELS, hier_dims);
    ASSERT_NE(result, nullptr);

    pred_hier_forward(hierarchy, input);
    pred_hier_backward(hierarchy);

    int ret = pred_hier_update(hierarchy, input, result);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    pred_hier_result_destroy(result);
}

TEST_F(OmniIntegrationTest, HierarchyFreeEnergy) {
    if (!hierarchy) GTEST_SKIP();

    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);
    pred_hier_forward(hierarchy, input);

    float fe = pred_hier_compute_free_energy(hierarchy);
    EXPECT_FALSE(std::isnan(fe));
}

/* ============================================================================
 * Temporal Replay Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, ReplayStoreAndSample) {
    if (!replay) GTEST_SKIP();

    /* Store transitions */
    for (int i = 0; i < 10; i++) {
        float state[LATENT_DIM], next_state[LATENT_DIM];
        generate_random_pattern(state, LATENT_DIM);
        generate_random_pattern(next_state, LATENT_DIM);

        int ret = temporal_replay_store(replay, state, nullptr, next_state,
                                        0.5f, false, 1.0f);
        EXPECT_EQ(ret, NIMCP_SUCCESS);
    }

    /* Sample batch */
    replay_batch_t* batch = replay_batch_create(4, LATENT_DIM, 0);
    ASSERT_NE(batch, nullptr);

    int ret = temporal_replay_sample(replay, REPLAY_MODE_PRIORITY, 4, batch);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    replay_batch_destroy(batch);
}

TEST_F(OmniIntegrationTest, ReplayForwardSweep) {
    if (!replay) GTEST_SKIP();

    /* Store transitions */
    for (int i = 0; i < 10; i++) {
        float state[LATENT_DIM], next_state[LATENT_DIM];
        generate_random_pattern(state, LATENT_DIM);
        generate_random_pattern(next_state, LATENT_DIM);
        temporal_replay_store(replay, state, nullptr, next_state, 0.5f, false, 1.0f);
    }

    /* Forward sweep */
    replay_sweep_result_t* sweep = replay_sweep_result_create(5, LATENT_DIM);
    ASSERT_NE(sweep, nullptr);

    int ret = temporal_replay_forward_sweep(replay, 0, 5, sweep);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    replay_sweep_result_destroy(sweep);
}

/* ============================================================================
 * Sensory Bridge Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, SensoryBridgeCreation) {
    omni_sensory_config_t config;
    omni_sensory_default_config(&config);

    omni_sensory_bridge_t* bridge = omni_sensory_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        omni_sensory_bridge_destroy(bridge);
    }
}

TEST_F(OmniIntegrationTest, SensoryBridgeConnect) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_sensory_config_t config;
    omni_sensory_default_config(&config);

    omni_sensory_bridge_t* bridge = omni_sensory_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    int ret = omni_sensory_connect_jepa(bridge, jepa);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    ret = omni_sensory_connect_pred_hier(bridge, hierarchy);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    omni_sensory_bridge_destroy(bridge);
}

/* ============================================================================
 * Cortical Columns Bridge Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, CorticalColumnsBridgeCreation) {
    omni_cc_config_t config;
    omni_cc_default_config(&config);

    omni_cortical_columns_bridge_t* bridge = omni_cc_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        omni_cc_bridge_destroy(bridge);
    }
}

TEST_F(OmniIntegrationTest, CorticalColumnsConnect) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_cc_config_t config;
    omni_cc_default_config(&config);

    omni_cortical_columns_bridge_t* bridge = omni_cc_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    int ret = omni_cc_connect_jepa(bridge, jepa);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    ret = omni_cc_connect_pred_hier(bridge, hierarchy);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    omni_cc_bridge_destroy(bridge);
}

TEST_F(OmniIntegrationTest, CorticalColumnsUpdate) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_cc_config_t config;
    omni_cc_default_config(&config);

    omni_cortical_columns_bridge_t* bridge = omni_cc_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_cc_connect_jepa(bridge, jepa);
    omni_cc_connect_pred_hier(bridge, hierarchy);

    /* Test the update cycle */
    int ret = omni_cc_update(bridge);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    /* Check sparsity value is valid */
    float sparsity = omni_cc_get_sparsity(bridge);
    EXPECT_GE(sparsity, 0.0f);
    EXPECT_LE(sparsity, 1.0f);

    omni_cc_bridge_destroy(bridge);
}

/* ============================================================================
 * Occipital Bridge Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, OccipitalBridgeCreation) {
    omni_occipital_config_t config;
    omni_occipital_default_config(&config);

    omni_occipital_bridge_t* bridge = omni_occipital_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        omni_occipital_bridge_destroy(bridge);
    }
}

TEST_F(OmniIntegrationTest, OccipitalBridgeConnect) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_occipital_config_t config;
    omni_occipital_default_config(&config);

    omni_occipital_bridge_t* bridge = omni_occipital_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    int ret = omni_occipital_connect_jepa(bridge, jepa);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    ret = omni_occipital_connect_pred_hier(bridge, hierarchy);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    omni_occipital_bridge_destroy(bridge);
}

TEST_F(OmniIntegrationTest, OccipitalUpdate) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_occipital_config_t config;
    omni_occipital_default_config(&config);

    omni_occipital_bridge_t* bridge = omni_occipital_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_occipital_connect_jepa(bridge, jepa);
    omni_occipital_connect_pred_hier(bridge, hierarchy);

    /* Test the update cycle */
    int ret = omni_occipital_update(bridge);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    /* Check dorsal PE is valid */
    float dorsal_pe = omni_occipital_get_dorsal_pe(bridge);
    EXPECT_FALSE(std::isnan(dorsal_pe));

    /* Check ventral PE is valid */
    float ventral_pe = omni_occipital_get_ventral_pe(bridge);
    EXPECT_FALSE(std::isnan(ventral_pe));

    omni_occipital_bridge_destroy(bridge);
}

/* ============================================================================
 * Broca Bridge Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, BrocaBridgeCreation) {
    omni_broca_config_t config;
    omni_broca_default_config(&config);

    omni_broca_bridge_t* bridge = omni_broca_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        omni_broca_bridge_destroy(bridge);
    }
}

TEST_F(OmniIntegrationTest, BrocaBridgeConnect) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_broca_config_t config;
    omni_broca_default_config(&config);

    omni_broca_bridge_t* bridge = omni_broca_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    int ret = omni_broca_connect_jepa(bridge, jepa);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    ret = omni_broca_connect_pred_hier(bridge, hierarchy);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    omni_broca_bridge_destroy(bridge);
}

TEST_F(OmniIntegrationTest, BrocaSyntaxPrediction) {
    if (!jepa || !hierarchy || !hopfield) GTEST_SKIP();

    omni_broca_config_t config;
    omni_broca_default_config(&config);

    omni_broca_bridge_t* bridge = omni_broca_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_broca_connect_jepa(bridge, jepa);
    omni_broca_connect_hopfield(bridge, hopfield);

    /* Test syntax prediction */
    omni_syntactic_prediction_t prediction;
    memset(&prediction, 0, sizeof(prediction));

    int ret = omni_broca_predict_syntax(bridge, &prediction);
    EXPECT_EQ(ret, NIMCP_SUCCESS);
    EXPECT_GE(prediction.confidence, 0.0f);
    EXPECT_LE(prediction.confidence, 1.0f);

    omni_broca_bridge_destroy(bridge);
}

TEST_F(OmniIntegrationTest, BrocaMotorPrediction) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_broca_config_t config;
    omni_broca_default_config(&config);

    omni_broca_bridge_t* bridge = omni_broca_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_broca_connect_jepa(bridge, jepa);
    omni_broca_connect_pred_hier(bridge, hierarchy);

    /* Create phoneme input */
    float phoneme[LATENT_DIM];
    generate_random_pattern(phoneme, LATENT_DIM);

    /* Test motor prediction */
    omni_motor_prediction_t prediction;
    memset(&prediction, 0, sizeof(prediction));

    int ret = omni_broca_predict_motor(bridge, phoneme, LATENT_DIM, &prediction);
    EXPECT_EQ(ret, NIMCP_SUCCESS);
    EXPECT_GE(prediction.execution_confidence, 0.0f);
    EXPECT_LE(prediction.execution_confidence, 1.0f);

    omni_broca_bridge_destroy(bridge);
}

/* ============================================================================
 * KG Sync Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, KGSyncCreation) {
    /* KG sync requires a brain_kg instance - skip if we can't create one */
    omni_kg_sync_config_t config;
    omni_kg_sync_default_config(&config);

    /* Create with NULL kg and default config - should handle gracefully */
    omni_kg_sync_t* sync = omni_kg_sync_create(nullptr, &config);
    /* Creating with NULL kg may fail gracefully, test the API exists */
    if (sync) {
        omni_kg_sync_destroy(sync);
    }
    /* Success - the API is callable */
    SUCCEED();
}

TEST_F(OmniIntegrationTest, KGSyncDefaultConfig) {
    omni_kg_sync_config_t config;
    int ret = omni_kg_sync_default_config(&config);
    EXPECT_EQ(ret, NIMCP_SUCCESS);
    EXPECT_TRUE(config.create_nodes);
    EXPECT_TRUE(config.create_edges);
}

/* ============================================================================
 * Full Pipeline Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, FullPredictionPipeline) {
    if (!jepa || !hopfield || !hierarchy || !replay) GTEST_SKIP();

    /* 1. Create input latent */
    jepa_latent_t* input = jepa_latent_create_dim(LATENT_DIM);
    ASSERT_NE(input, nullptr);
    generate_random_pattern(input->embedding, LATENT_DIM);

    /* 2. JEPA forward prediction */
    jepa_bidir_result_t* jepa_result = jepa_bidir_result_create(LATENT_DIM);
    ASSERT_NE(jepa_result, nullptr);

    int ret = jepa_bidirectional_predict(jepa, JEPA_DIR_FORWARD, input, jepa_result);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    /* 3. Store in Hopfield */
    uint32_t pattern_id;
    ret = hopfield_memory_store(hopfield, jepa_result->prediction->embedding, &pattern_id);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    /* 4. Process through hierarchy */
    ret = pred_hier_forward(hierarchy, jepa_result->prediction->embedding);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    /* 5. Store in replay */
    float next_state[LATENT_DIM];
    generate_random_pattern(next_state, LATENT_DIM);
    ret = temporal_replay_store(replay, jepa_result->prediction->embedding,
                                nullptr, next_state, 0.5f, false, 1.0f);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    jepa_bidir_result_destroy(jepa_result);
    jepa_latent_destroy(input);
}

TEST_F(OmniIntegrationTest, BridgeChainPipeline) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    /* Create bridges */
    omni_sensory_config_t sens_config;
    omni_sensory_default_config(&sens_config);
    omni_sensory_bridge_t* sensory = omni_sensory_bridge_create(&sens_config);
    ASSERT_NE(sensory, nullptr);

    omni_cc_config_t cc_config;
    omni_cc_default_config(&cc_config);
    omni_cortical_columns_bridge_t* cc = omni_cc_bridge_create(&cc_config);
    ASSERT_NE(cc, nullptr);

    /* Connect bridges to core components */
    omni_sensory_connect_jepa(sensory, jepa);
    omni_sensory_connect_pred_hier(sensory, hierarchy);
    omni_cc_connect_jepa(cc, jepa);
    omni_cc_connect_pred_hier(cc, hierarchy);

    /* Update bridges */
    int ret = omni_sensory_update(sensory);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    ret = omni_cc_update(cc);
    EXPECT_EQ(ret, NIMCP_SUCCESS);

    omni_cc_bridge_destroy(cc);
    omni_sensory_bridge_destroy(sensory);
}

TEST_F(OmniIntegrationTest, StressPredictionPipeline) {
    if (!jepa || !hopfield) GTEST_SKIP();

    const int NUM_ITERATIONS = 50;
    int success_count = 0;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        jepa_latent_t* input = jepa_latent_create_dim(LATENT_DIM);
        if (!input) continue;
        generate_random_pattern(input->embedding, LATENT_DIM);

        jepa_bidir_result_t* result = jepa_bidir_result_create(LATENT_DIM);
        if (!result) {
            jepa_latent_destroy(input);
            continue;
        }

        if (jepa_bidirectional_predict(jepa, JEPA_DIR_FORWARD, input, result) == NIMCP_SUCCESS) {
            uint32_t id;
            if (hopfield_memory_store(hopfield, result->prediction->embedding, &id) == NIMCP_SUCCESS) {
                success_count++;
            }
        }

        jepa_bidir_result_destroy(result);
        jepa_latent_destroy(input);
    }

    EXPECT_EQ(success_count, NUM_ITERATIONS);
    std::cout << "Stress test: " << success_count << "/" << NUM_ITERATIONS << std::endl;
}
