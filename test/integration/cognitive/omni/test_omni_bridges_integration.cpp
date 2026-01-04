/**
 * @file test_omni_bridges_integration.cpp
 * @brief Integration tests for Omnidirectional Inference bridges
 * @version 1.0.0
 * @date 2025-01-04
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

#include "cognitive/jepa/nimcp_jepa_bidirectional.h"
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
    static const uint32_t LATENT_DIM = 64;
    static const uint32_t HIDDEN_DIM = 128;
    static const uint32_t NUM_PATTERNS = 256;
    static const uint32_t NUM_LEVELS = 4;

    jepa_bidirectional_t* jepa = nullptr;
    hopfield_memory_t* hopfield = nullptr;
    predictive_hierarchy_t* hierarchy = nullptr;
    temporal_replay_t* replay = nullptr;

    void SetUp() override {
        /* Create bidirectional JEPA */
        jepa_bidir_config_t jepa_config;
        jepa_bidir_default_config(&jepa_config);
        jepa_config.embedding_dim = LATENT_DIM;
        jepa_config.hidden_dim = HIDDEN_DIM;
        jepa = jepa_bidirectional_create(&jepa_config);

        /* Create Hopfield memory */
        hopfield_config_t hopfield_config;
        hopfield_default_config(&hopfield_config);
        hopfield_config.pattern_dim = LATENT_DIM;
        hopfield_config.capacity = NUM_PATTERNS;
        hopfield = hopfield_memory_create(&hopfield_config);

        /* Create predictive hierarchy */
        pred_hier_config_t hier_config;
        pred_hier_default_config(&hier_config);
        hier_config.num_levels = NUM_LEVELS;
        hier_config.num_levels = LATENT_DIM;
        hierarchy = pred_hier_create(&hier_config);

        /* Create temporal replay */
        replay_config_t replay_config;
        replay_default_config(&replay_config);
        replay_config.state_dim = LATENT_DIM;
        replay_config.capacity = 1024;
        replay = temporal_replay_create(&replay_config);
    }

    void TearDown() override {
        if (jepa) {
            jepa_bidirectional_destroy(jepa);
            jepa = nullptr;
        }
        if (hopfield) {
            hopfield_memory_destroy(hopfield);
            hopfield = nullptr;
        }
        if (hierarchy) {
            pred_hier_destroy(hierarchy);
            hierarchy = nullptr;
        }
        if (replay) {
            temporal_replay_destroy(replay);
            replay = nullptr;
        }
    }

    /* Helper to generate random pattern */
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

    float input[LATENT_DIM], output[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);

    int ret = jepa_bidirectional_predict(jepa, JEPA_DIR_FORWARD, input, output, LATENT_DIM);
    EXPECT_EQ(ret, 0);
}

TEST_F(OmniIntegrationTest, JEPABackwardPrediction) {
    if (!jepa) GTEST_SKIP();

    float input[LATENT_DIM], output[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);

    int ret = jepa_bidirectional_predict(jepa, JEPA_DIR_BACKWARD, input, output, LATENT_DIM);
    EXPECT_EQ(ret, 0);
}

TEST_F(OmniIntegrationTest, JEPALateralPrediction) {
    if (!jepa) GTEST_SKIP();

    float input[LATENT_DIM], output[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);

    int ret = jepa_bidirectional_predict(jepa, JEPA_DIR_LATERAL, input, output, LATENT_DIM);
    EXPECT_EQ(ret, 0);
}

TEST_F(OmniIntegrationTest, JEPAMultiDirectionPrediction) {
    if (!jepa) GTEST_SKIP();

    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);

    jepa_multi_result_t result;
    memset(&result, 0, sizeof(result));

    jepa_direction_t directions[] = {JEPA_DIR_FORWARD, JEPA_DIR_BACKWARD};
    int ret = jepa_bidirectional_predict_multi(jepa, directions, 2, input, &result, LATENT_DIM);
    EXPECT_EQ(ret, 0);
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
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(hopfield_memory_pattern_count(hopfield), 1u);

    /* Retrieve with noisy query */
    hopfield_retrieval_result_t result;
    result.pattern = (float*)malloc(LATENT_DIM * sizeof(float));
    ret = hopfield_memory_retrieve(hopfield, query, &result);
    EXPECT_EQ(ret, 0);
    EXPECT_GT(result.similarity, 0.8f); /* Should be similar */
    free(result.pattern);
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
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(hopfield_memory_pattern_count(hopfield), num_patterns);
}

/* ============================================================================
 * Predictive Hierarchy Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, HierarchyForwardPass) {
    if (!hierarchy) GTEST_SKIP();

    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);

    int ret = predictive_hierarchy_forward(hierarchy, input, LATENT_DIM);
    EXPECT_EQ(ret, 0);
}

TEST_F(OmniIntegrationTest, HierarchyBackwardPass) {
    if (!hierarchy) GTEST_SKIP();

    /* First do forward pass */
    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);
    predictive_hierarchy_forward(hierarchy, input, LATENT_DIM);

    /* Then backward predictions */
    int ret = predictive_hierarchy_backward(hierarchy);
    EXPECT_EQ(ret, 0);
}

TEST_F(OmniIntegrationTest, HierarchyBeliefsUpdate) {
    if (!hierarchy) GTEST_SKIP();

    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);
    predictive_hierarchy_forward(hierarchy, input, LATENT_DIM);
    predictive_hierarchy_backward(hierarchy);

    /* Update beliefs from prediction errors */
    int ret = predictive_hierarchy_update_beliefs(hierarchy);
    EXPECT_EQ(ret, 0);
}

TEST_F(OmniIntegrationTest, HierarchyFreeEnergy) {
    if (!hierarchy) GTEST_SKIP();

    float input[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);
    predictive_hierarchy_forward(hierarchy, input, LATENT_DIM);

    float fe = predictive_hierarchy_compute_free_energy(hierarchy);
    EXPECT_GE(fe, 0.0f); /* Free energy should be non-negative */
}

/* ============================================================================
 * Temporal Replay Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, ReplayStoreAndSample) {
    if (!replay) GTEST_SKIP();

    /* Store a sequence */
    const uint32_t seq_len = 10;
    float sequence[seq_len * LATENT_DIM];
    for (uint32_t t = 0; t < seq_len; t++) {
        generate_random_pattern(&sequence[t * LATENT_DIM], LATENT_DIM);
    }

    int ret = temporal_replay_store_sequence(replay, sequence, seq_len, 1.0f);
    EXPECT_EQ(ret, 0);

    /* Sample */
    temporal_sample_result_t sample;
    ret = temporal_replay_sample(replay, REPLAY_MODE_PRIORITY, &sample);
    EXPECT_EQ(ret, 0);
}

TEST_F(OmniIntegrationTest, ReplayForwardSweep) {
    if (!replay) GTEST_SKIP();

    const uint32_t seq_len = 10;
    float sequence[seq_len * LATENT_DIM];
    for (uint32_t t = 0; t < seq_len; t++) {
        generate_random_pattern(&sequence[t * LATENT_DIM], LATENT_DIM);
    }
    temporal_replay_store_sequence(replay, sequence, seq_len, 1.0f);

    /* Forward sweep */
    float output[5 * LATENT_DIM];
    int ret = temporal_replay_forward_sweep(replay, 0, 0, 5, output);
    EXPECT_EQ(ret, 0);
}

TEST_F(OmniIntegrationTest, ReplayBackwardSweep) {
    if (!replay) GTEST_SKIP();

    const uint32_t seq_len = 10;
    float sequence[seq_len * LATENT_DIM];
    for (uint32_t t = 0; t < seq_len; t++) {
        generate_random_pattern(&sequence[t * LATENT_DIM], LATENT_DIM);
    }
    temporal_replay_store_sequence(replay, sequence, seq_len, 1.0f);

    /* Backward sweep */
    float output[5 * LATENT_DIM];
    int ret = temporal_replay_backward_sweep(replay, 0, 9, 5, output);
    EXPECT_EQ(ret, 0);
}

/* ============================================================================
 * Sensory Bridge Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, SensoryBridgeCreation) {
    omni_sensory_bridge_config_t config;
    omni_sensory_bridge_default_config(&config);
    config.latent_dim = LATENT_DIM;

    omni_sensory_bridge_t* bridge = omni_sensory_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        omni_sensory_bridge_destroy(bridge);
    }
}

TEST_F(OmniIntegrationTest, SensoryBridgeConnect) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_sensory_bridge_config_t config;
    omni_sensory_bridge_default_config(&config);
    config.latent_dim = LATENT_DIM;

    omni_sensory_bridge_t* bridge = omni_sensory_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    int ret = omni_sensory_bridge_connect_jepa(bridge, jepa);
    EXPECT_EQ(ret, 0);

    ret = omni_sensory_bridge_connect_hierarchy(bridge, hierarchy);
    EXPECT_EQ(ret, 0);

    omni_sensory_bridge_destroy(bridge);
}

TEST_F(OmniIntegrationTest, SensoryBridgeCrossModalBinding) {
    if (!jepa || !hopfield) GTEST_SKIP();

    omni_sensory_bridge_config_t config;
    omni_sensory_bridge_default_config(&config);
    config.latent_dim = LATENT_DIM;
    config.enable_crossmodal = true;

    omni_sensory_bridge_t* bridge = omni_sensory_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_sensory_bridge_connect_jepa(bridge, jepa);
    omni_sensory_bridge_connect_hopfield(bridge, hopfield);

    /* Create cross-modal binding between visual and audio */
    float visual_repr[LATENT_DIM], audio_repr[LATENT_DIM];
    generate_random_pattern(visual_repr, LATENT_DIM);
    generate_random_pattern(audio_repr, LATENT_DIM);

    int ret = omni_sensory_bridge_bind_modalities(
        bridge, OMNI_MODALITY_VISUAL, visual_repr,
        OMNI_MODALITY_AUDIO, audio_repr
    );
    EXPECT_EQ(ret, 0);

    omni_sensory_bridge_destroy(bridge);
}

/* ============================================================================
 * Cortical Columns Bridge Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, CorticalColumnsBridgeCreation) {
    omni_cortical_columns_bridge_config_t config;
    omni_cortical_columns_bridge_default_config(&config);

    omni_cortical_columns_bridge_t* bridge = omni_cortical_columns_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        omni_cortical_columns_bridge_destroy(bridge);
    }
}

TEST_F(OmniIntegrationTest, CorticalColumnsBridgePredictionBias) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_cortical_columns_bridge_config_t config;
    omni_cortical_columns_bridge_default_config(&config);
    config.competition_mode = OMNI_CC_COMPETITION_K_WINNERS;

    omni_cortical_columns_bridge_t* bridge = omni_cortical_columns_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_cortical_columns_bridge_connect_jepa(bridge, jepa);
    omni_cortical_columns_bridge_connect_hierarchy(bridge, hierarchy);

    /* Test prediction-based column activation */
    float prediction[LATENT_DIM];
    generate_random_pattern(prediction, LATENT_DIM);

    int ret = omni_cortical_columns_bridge_apply_prediction_bias(bridge, prediction, LATENT_DIM);
    EXPECT_EQ(ret, 0);

    omni_cortical_columns_bridge_destroy(bridge);
}

/* ============================================================================
 * Occipital Bridge Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, OccipitalBridgeCreation) {
    omni_occipital_bridge_config_t config;
    omni_occipital_bridge_default_config(&config);

    omni_occipital_bridge_t* bridge = omni_occipital_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        omni_occipital_bridge_destroy(bridge);
    }
}

TEST_F(OmniIntegrationTest, OccipitalBridgeVisualPrediction) {
    if (!jepa || !hierarchy || !hopfield) GTEST_SKIP();

    omni_occipital_bridge_config_t config;
    omni_occipital_bridge_default_config(&config);

    omni_occipital_bridge_t* bridge = omni_occipital_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_occipital_bridge_connect_jepa(bridge, jepa);
    omni_occipital_bridge_connect_hierarchy(bridge, hierarchy);

    /* Test visual prediction through V1-V5 */
    float input_repr[LATENT_DIM], output_repr[LATENT_DIM];
    generate_random_pattern(input_repr, LATENT_DIM);

    int ret = omni_occipital_bridge_forward_visual(
        bridge, OMNI_VISUAL_AREA_V1, input_repr, output_repr, LATENT_DIM
    );
    EXPECT_EQ(ret, 0);

    omni_occipital_bridge_destroy(bridge);
}

TEST_F(OmniIntegrationTest, OccipitalBridgeDorsalVentralStreams) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_occipital_bridge_config_t config;
    omni_occipital_bridge_default_config(&config);

    omni_occipital_bridge_t* bridge = omni_occipital_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_occipital_bridge_connect_jepa(bridge, jepa);

    float input[LATENT_DIM];
    float dorsal_out[LATENT_DIM], ventral_out[LATENT_DIM];
    generate_random_pattern(input, LATENT_DIM);

    /* Test dorsal (where) stream */
    int ret = omni_occipital_bridge_process_stream(
        bridge, OMNI_VISUAL_STREAM_DORSAL, input, dorsal_out, LATENT_DIM
    );
    EXPECT_EQ(ret, 0);

    /* Test ventral (what) stream */
    ret = omni_occipital_bridge_process_stream(
        bridge, OMNI_VISUAL_STREAM_VENTRAL, input, ventral_out, LATENT_DIM
    );
    EXPECT_EQ(ret, 0);

    omni_occipital_bridge_destroy(bridge);
}

/* ============================================================================
 * Broca Bridge Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, BrocaBridgeCreation) {
    omni_broca_bridge_config_t config;
    omni_broca_bridge_default_config(&config);

    omni_broca_bridge_t* bridge = omni_broca_bridge_create(&config);
    EXPECT_NE(bridge, nullptr);

    if (bridge) {
        omni_broca_bridge_destroy(bridge);
    }
}

TEST_F(OmniIntegrationTest, BrocaBridgeSyntacticPrediction) {
    if (!jepa || !hierarchy || !hopfield) GTEST_SKIP();

    omni_broca_bridge_config_t config;
    omni_broca_bridge_default_config(&config);

    omni_broca_bridge_t* bridge = omni_broca_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_broca_bridge_connect_jepa(bridge, jepa);
    omni_broca_bridge_connect_hopfield(bridge, hopfield);

    /* Test syntactic prediction (BA45) */
    float syntactic_context[LATENT_DIM], predicted_structure[LATENT_DIM];
    generate_random_pattern(syntactic_context, LATENT_DIM);

    int ret = omni_broca_bridge_predict_syntax(
        bridge, syntactic_context, predicted_structure, LATENT_DIM
    );
    EXPECT_EQ(ret, 0);

    omni_broca_bridge_destroy(bridge);
}

TEST_F(OmniIntegrationTest, BrocaBridgeMotorPrediction) {
    if (!jepa || !hierarchy) GTEST_SKIP();

    omni_broca_bridge_config_t config;
    omni_broca_bridge_default_config(&config);

    omni_broca_bridge_t* bridge = omni_broca_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    omni_broca_bridge_connect_jepa(bridge, jepa);
    omni_broca_bridge_connect_hierarchy(bridge, hierarchy);

    /* Test motor prediction (BA44) */
    float motor_plan[LATENT_DIM], motor_prediction[LATENT_DIM];
    generate_random_pattern(motor_plan, LATENT_DIM);

    int ret = omni_broca_bridge_predict_motor(
        bridge, motor_plan, motor_prediction, LATENT_DIM
    );
    EXPECT_EQ(ret, 0);

    omni_broca_bridge_destroy(bridge);
}

/* ============================================================================
 * KG Sync Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, KGSyncCreation) {
    /* Create a mock brain KG for testing */
    brain_kg_t* kg = brain_kg_create(1024);
    ASSERT_NE(kg, nullptr);

    omni_kg_sync_config_t config;
    omni_kg_sync_default_config(&config);

    omni_kg_sync_t* sync = omni_kg_sync_create(kg, &config);
    EXPECT_NE(sync, nullptr);

    if (sync) {
        omni_kg_sync_destroy(sync);
    }
    brain_kg_destroy(kg);
}

TEST_F(OmniIntegrationTest, KGSyncRegisterModules) {
    brain_kg_t* kg = brain_kg_create(1024);
    ASSERT_NE(kg, nullptr);

    omni_kg_sync_config_t config;
    omni_kg_sync_default_config(&config);
    omni_kg_sync_t* sync = omni_kg_sync_create(kg, &config);
    ASSERT_NE(sync, nullptr);

    /* Register JEPA */
    brain_kg_node_id_t jepa_node = omni_kg_register_jepa(sync, "test_jepa", jepa);
    EXPECT_NE(jepa_node, BRAIN_KG_INVALID_NODE);

    /* Register Hopfield */
    brain_kg_node_id_t hopfield_node = omni_kg_register_hopfield(sync, "test_hopfield", hopfield);
    EXPECT_NE(hopfield_node, BRAIN_KG_INVALID_NODE);

    /* Add prediction edge */
    int ret = omni_kg_add_prediction_edge(
        sync, jepa_node, hopfield_node,
        OMNI_KG_EDGE_PREDICTS_FORWARD, 1.0f
    );
    EXPECT_EQ(ret, 0);

    omni_kg_sync_destroy(sync);
    brain_kg_destroy(kg);
}

TEST_F(OmniIntegrationTest, KGSyncQueryCapabilities) {
    brain_kg_t* kg = brain_kg_create(1024);
    ASSERT_NE(kg, nullptr);

    omni_kg_sync_config_t config;
    omni_kg_sync_default_config(&config);
    omni_kg_sync_t* sync = omni_kg_sync_create(kg, &config);
    ASSERT_NE(sync, nullptr);

    /* Register modules with different capabilities */
    brain_kg_node_id_t node1 = omni_kg_register_jepa(sync, "jepa1", jepa);
    brain_kg_node_id_t node2 = omni_kg_register_hopfield(sync, "hopfield1", hopfield);

    /* Query modules with forward capability */
    brain_kg_node_id_t forward_nodes[16];
    int count = omni_kg_get_modules_with_capability(
        sync, OMNI_KG_CAP_FORWARD, forward_nodes, 16
    );
    EXPECT_GE(count, 0);

    /* Query modules with associative capability */
    brain_kg_node_id_t assoc_nodes[16];
    count = omni_kg_get_modules_with_capability(
        sync, OMNI_KG_CAP_ASSOCIATIVE, assoc_nodes, 16
    );
    EXPECT_GE(count, 0);

    omni_kg_sync_destroy(sync);
    brain_kg_destroy(kg);
}

/* ============================================================================
 * Full Pipeline Integration Tests
 * ============================================================================ */

TEST_F(OmniIntegrationTest, FullBidirectionalPipeline) {
    if (!jepa || !hopfield || !hierarchy) GTEST_SKIP();

    /* Create bridges */
    omni_sensory_bridge_config_t sensory_config;
    omni_sensory_bridge_default_config(&sensory_config);
    sensory_config.latent_dim = LATENT_DIM;
    omni_sensory_bridge_t* sensory = omni_sensory_bridge_create(&sensory_config);
    ASSERT_NE(sensory, nullptr);

    omni_occipital_bridge_config_t occipital_config;
    omni_occipital_bridge_default_config(&occipital_config);
    omni_occipital_bridge_t* occipital = omni_occipital_bridge_create(&occipital_config);
    ASSERT_NE(occipital, nullptr);

    /* Connect components */
    omni_sensory_bridge_connect_jepa(sensory, jepa);
    omni_sensory_bridge_connect_hierarchy(sensory, hierarchy);
    omni_sensory_bridge_connect_hopfield(sensory, hopfield);

    omni_occipital_bridge_connect_jepa(occipital, jepa);
    omni_occipital_bridge_connect_hierarchy(occipital, hierarchy);

    /* Simulate visual input -> prediction -> memory storage */
    float visual_input[LATENT_DIM], prediction[LATENT_DIM];
    generate_random_pattern(visual_input, LATENT_DIM);

    /* Forward through sensory */
    int ret = omni_sensory_bridge_forward(sensory, OMNI_MODALITY_VISUAL, visual_input, LATENT_DIM);
    EXPECT_EQ(ret, 0);

    /* Forward through occipital V1->V4 */
    float v1_out[LATENT_DIM], v4_out[LATENT_DIM];
    omni_occipital_bridge_forward_visual(occipital, OMNI_VISUAL_AREA_V1, visual_input, v1_out, LATENT_DIM);
    omni_occipital_bridge_forward_visual(occipital, OMNI_VISUAL_AREA_V4, v1_out, v4_out, LATENT_DIM);

    /* Generate forward prediction using JEPA */
    ret = jepa_bidirectional_predict(jepa, JEPA_DIR_FORWARD, v4_out, prediction, LATENT_DIM);
    EXPECT_EQ(ret, 0);

    /* Store in Hopfield memory */
    uint32_t pattern_id;
    ret = hopfield_memory_store(hopfield, prediction, &pattern_id);
    EXPECT_EQ(ret, 0);

    /* Generate backward prediction */
    float backward_pred[LATENT_DIM];
    ret = jepa_bidirectional_predict(jepa, JEPA_DIR_BACKWARD, prediction, backward_pred, LATENT_DIM);
    EXPECT_EQ(ret, 0);

    /* Cleanup */
    omni_occipital_bridge_destroy(occipital);
    omni_sensory_bridge_destroy(sensory);
}

TEST_F(OmniIntegrationTest, FullLanguageProductionPipeline) {
    if (!jepa || !hopfield || !hierarchy) GTEST_SKIP();

    /* Create Broca bridge */
    omni_broca_bridge_config_t broca_config;
    omni_broca_bridge_default_config(&broca_config);
    omni_broca_bridge_t* broca = omni_broca_bridge_create(&broca_config);
    ASSERT_NE(broca, nullptr);

    omni_broca_bridge_connect_jepa(broca, jepa);
    omni_broca_bridge_connect_hopfield(broca, hopfield);
    omni_broca_bridge_connect_hierarchy(broca, hierarchy);

    /* Simulate concept -> syntactic structure -> motor plan */
    float concept[LATENT_DIM], syntax[LATENT_DIM], motor[LATENT_DIM];
    generate_random_pattern(concept, LATENT_DIM);

    /* Store concept in Hopfield for retrieval */
    uint32_t concept_id;
    hopfield_memory_store(hopfield, concept, &concept_id);

    /* Predict syntactic structure */
    int ret = omni_broca_bridge_predict_syntax(broca, concept, syntax, LATENT_DIM);
    EXPECT_EQ(ret, 0);

    /* Predict motor plan */
    ret = omni_broca_bridge_predict_motor(broca, syntax, motor, LATENT_DIM);
    EXPECT_EQ(ret, 0);

    /* Verify phonological working memory */
    ret = omni_broca_bridge_phon_wm_store(broca, motor, LATENT_DIM);
    EXPECT_EQ(ret, 0);

    omni_broca_bridge_destroy(broca);
}

TEST_F(OmniIntegrationTest, FullReplayConsolidationPipeline) {
    if (!jepa || !hopfield || !hierarchy || !replay) GTEST_SKIP();

    /* Store multiple experiences */
    const uint32_t num_experiences = 5;
    const uint32_t seq_len = 8;

    for (uint32_t exp = 0; exp < num_experiences; exp++) {
        float sequence[seq_len * LATENT_DIM];
        for (uint32_t t = 0; t < seq_len; t++) {
            generate_random_pattern(&sequence[t * LATENT_DIM], LATENT_DIM);
        }

        float priority = 1.0f - (float)exp / num_experiences;
        temporal_replay_store_sequence(replay, sequence, seq_len, priority);
    }

    /* Perform forward replay sweep and store consolidated patterns */
    float sweep_output[seq_len * LATENT_DIM];
    int ret = temporal_replay_forward_sweep(replay, 0, 0, seq_len, sweep_output);
    EXPECT_EQ(ret, 0);

    /* Predict forward through sweep and store in Hopfield */
    for (uint32_t t = 0; t < seq_len - 1; t++) {
        float* current = &sweep_output[t * LATENT_DIM];
        float* next = &sweep_output[(t + 1) * LATENT_DIM];
        float predicted[LATENT_DIM];

        ret = jepa_bidirectional_predict(jepa, JEPA_DIR_FORWARD, current, predicted, LATENT_DIM);
        EXPECT_EQ(ret, 0);

        /* Store prediction in Hopfield */
        uint32_t pid;
        hopfield_memory_store(hopfield, predicted, &pid);
    }

    EXPECT_GE(hopfield_memory_pattern_count(hopfield), seq_len - 1);

    /* Perform backward replay for planning */
    ret = temporal_replay_backward_sweep(replay, 0, seq_len - 1, seq_len, sweep_output);
    EXPECT_EQ(ret, 0);

    /* Generate backward predictions */
    for (uint32_t t = seq_len - 1; t > 0; t--) {
        float* current = &sweep_output[(seq_len - 1 - t) * LATENT_DIM];
        float backward_pred[LATENT_DIM];

        ret = jepa_bidirectional_predict(jepa, JEPA_DIR_BACKWARD, current, backward_pred, LATENT_DIM);
        EXPECT_EQ(ret, 0);
    }
}
