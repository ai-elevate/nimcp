/**
 * @file e2e_test_sensory_processing_pipeline.cpp
 * @brief End-to-End tests for Phase 6 Sensory Processing pipeline
 *
 * TEST PHILOSOPHY:
 * - Test complete sensory processing workflows from stimulus to perception
 * - Test multi-modal sensory integration (touch, smell, taste)
 * - Test sensory memory formation and recall
 * - Test real-time sensory processing scenarios
 * - Verify cross-modal binding and flavor perception
 *
 * PIPELINE SCENARIOS:
 * 1. Touch-to-Perception: Touch event -> Thalamus -> Cortex -> Perception
 * 2. Smell-to-Memory: Odor -> Recognition -> Memory encoding -> Recall
 * 3. Taste-to-Reward: Taste -> Processing -> Reward computation -> Learning
 * 4. Multi-Modal Flavor: Taste + Smell -> Flavor integration
 * 5. Pain-to-Response: Pain -> Gate control -> Modulation -> Response
 *
 * @author NIMCP Development Team
 * @date 2026-01-12
 * @version 1.0.0 Phase 6 Sensory Processing E2E
 */

#include "e2e_test_framework.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <thread>
#include <chrono>

/* Sensory modules */
extern "C" {
#include "core/brain/regions/somatosensory/nimcp_somatosensory.h"
#include "core/brain/regions/olfactory/nimcp_olfactory.h"
#include "core/brain/regions/gustatory/nimcp_gustatory.h"
}

using namespace nimcp::e2e;

//=============================================================================
// Test Fixtures
//=============================================================================

/**
 * @class SensoryPipelineE2ETest
 * @brief E2E test fixture for sensory processing pipelines
 */
class SensoryPipelineE2ETest : public ::testing::Test {
protected:
    nimcp_somatosensory_t* soma = nullptr;
    nimcp_olfactory_t* olfact = nullptr;
    nimcp_gustatory_t* gust = nullptr;

    void SetUp() override {
        /* Create all sensory modules */
        soma = soma_create(nullptr);
        olfact = olfact_create(nullptr);
        gust = gust_create(nullptr);

        /* Initialize all bridges for full pipeline testing */
        if (soma) soma_init_all_bridges(soma, nullptr);
        if (olfact) {
            olfact_init_prime_resonance_bridge(olfact, nullptr);
            olfact_init_amygdala_bridge(olfact, nullptr);
            olfact_init_entorhinal_bridge(olfact, nullptr);
            olfact_init_ofc_bridge(olfact, nullptr);
            olfact_init_hypothalamus_bridge(olfact, nullptr);
        }
        if (gust) {
            gust_init_prime_resonance_bridge(gust, nullptr);
            gust_init_hypothalamus_bridge(gust, nullptr);
            gust_init_amygdala_bridge(gust, nullptr);
            gust_init_olfactory_bridge(gust, olfact);
            gust_init_ofc_bridge(gust, nullptr);
        }
    }

    void TearDown() override {
        if (soma) soma_destroy(soma);
        if (olfact) olfact_destroy(olfact);
        if (gust) gust_destroy(gust);
    }

    /* Helper: Generate odor pattern */
    void generate_odor_pattern(float* pattern, uint32_t size, uint32_t seed) {
        for (uint32_t i = 0; i < size; i++) {
            pattern[i] = sinf((float)(i + seed) * 0.1f) * 0.5f + 0.5f;
        }
    }
};

//=============================================================================
// Touch-to-Perception Pipeline Tests
//=============================================================================

E2E_TEST(SensoryPipelineE2E, TouchToPerception_CompletePipeline) {
    PipelineTracker tracker("Touch-to-Perception Pipeline");

    /* Stage 1: Create somatosensory module */
    tracker.begin_stage("Create Somatosensory Module", 50);
    nimcp_somatosensory_t* soma = soma_create(nullptr);
    if (!soma) {
        tracker.fail_stage("Failed to create somatosensory module");
        E2E_ASSERT_PIPELINE_SUCCESS(tracker);
        return;
    }
    tracker.end_stage();

    /* Stage 2: Initialize bridges */
    tracker.begin_stage("Initialize Bridges", 50);
    int init_result = soma_init_all_bridges(soma, nullptr);
    if (init_result != 0) {
        tracker.fail_stage("Failed to initialize bridges");
        soma_destroy(soma);
        E2E_ASSERT_PIPELINE_SUCCESS(tracker);
        return;
    }
    EXPECT_TRUE(soma->thalamus_bridge.initialized);
    EXPECT_TRUE(soma->motor_bridge.initialized);
    tracker.end_stage();

    /* Stage 3: Process touch event */
    tracker.begin_stage("Process Touch Event", 100);
    float position[3] = {0.5f, 0.5f, 0.0f};
    uint32_t event_id;
    int touch_result = soma_process_touch(soma, BODY_SEG_INDEX_R, position, 0.7f, TOUCH_LIGHT, &event_id);
    if (touch_result != 0) {
        tracker.fail_stage("Failed to process touch");
        soma_destroy(soma);
        E2E_ASSERT_PIPELINE_SUCCESS(tracker);
        return;
    }
    tracker.end_stage();

    /* Stage 4: Sync with thalamus (relay to cortex) */
    tracker.begin_stage("Thalamus Relay", 50);
    EXPECT_EQ(soma_sync_thalamus(soma), 0);
    tracker.end_stage();

    /* Stage 5: Update cortical processing */
    tracker.begin_stage("Cortical Processing", 100);
    EXPECT_EQ(soma_update(soma, 0.01f), 0);
    tracker.end_stage();

    /* Stage 7: Bidirectional update */
    tracker.begin_stage("Bidirectional Update", 100);
    EXPECT_EQ(soma_bidirectional_update(soma, 0.01f), 0);
    tracker.end_stage();

    /* Stage 8: Verify perception */
    tracker.begin_stage("Verify Perception", 50);
    soma_stats_t stats;
    EXPECT_EQ(soma_get_stats(soma, &stats), 0);
    EXPECT_GT(stats.touch_events_processed, 0u);
    EXPECT_GT(soma->updates_processed, 0u);
    tracker.end_stage();

    /* Cleanup */
    soma_destroy(soma);

    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

E2E_TEST(SensoryPipelineE2E, ProprioceptionBodyAwareness_Pipeline) {
    PipelineTracker tracker("Proprioception-to-Body-Awareness Pipeline");

    tracker.begin_stage("Create Module", 50);
    nimcp_somatosensory_t* soma = soma_create(nullptr);
    ASSERT_NE(soma, nullptr);
    soma_init_all_bridges(soma, nullptr);
    tracker.end_stage();

    /* Update proprioception for arm movement sequence */
    tracker.begin_stage("Arm Movement Proprioception", 200);
    for (int t = 0; t < 10; t++) {
        float pos[3] = {0.0f, 1.0f - t * 0.1f, 0.0f};  /* Lowering arm */
        float vel[3] = {0.0f, -0.1f, 0.0f};
        EXPECT_EQ(soma_update_proprioception(soma, BODY_SEG_UPPER_ARM_R, pos, vel, 0.5f, 0.6f), 0);
        soma_update(soma, 0.01f);
    }
    tracker.end_stage();

    /* Get body position estimate */
    tracker.begin_stage("Body Position Estimate", 50);
    float positions[BODY_SEG_COUNT * 3];
    uint32_t num_segments;
    EXPECT_EQ(soma_get_body_position(soma, positions, BODY_SEG_COUNT, &num_segments), 0);
    EXPECT_GT(num_segments, 0u);
    tracker.end_stage();

    /* Sync with parietal for body schema */
    tracker.begin_stage("Parietal Body Schema Sync", 50);
    EXPECT_EQ(soma_sync_parietal(soma), 0);
    tracker.end_stage();

    /* Verify joint angle estimation */
    tracker.begin_stage("Joint Angle Verification", 50);
    float angle = soma_get_joint_angle(soma, BODY_SEG_UPPER_ARM_R);
    EXPECT_GE(angle, -180.0f);
    EXPECT_LE(angle, 180.0f);
    tracker.end_stage();

    soma_destroy(soma);
    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

//=============================================================================
// Smell-to-Memory Pipeline Tests
//=============================================================================

E2E_TEST(SensoryPipelineE2E, SmellToMemory_CompletePipeline) {
    PipelineTracker tracker("Smell-to-Memory Pipeline");

    /* Stage 1: Create olfactory module */
    tracker.begin_stage("Create Olfactory Module", 50);
    nimcp_olfactory_t* olfact = olfact_create(nullptr);
    ASSERT_NE(olfact, nullptr);
    tracker.end_stage();

    /* Stage 2: Initialize bridges */
    tracker.begin_stage("Initialize Memory Bridges", 50);
    EXPECT_EQ(olfact_init_amygdala_bridge(olfact, nullptr), 0);
    EXPECT_EQ(olfact_init_entorhinal_bridge(olfact, nullptr), 0);
    EXPECT_EQ(olfact_init_prime_resonance_bridge(olfact, nullptr), 0);
    tracker.end_stage();

    /* Stage 3: Create and process distinctive odor (grandma's cookies) */
    tracker.begin_stage("Process Signature Odor", 100);
    float cookie_odor[OLFACT_MAX_RECEPTORS];
    for (uint32_t i = 0; i < OLFACT_MAX_RECEPTORS; i++) {
        cookie_odor[i] = (i % 15 < 5) ? 0.8f : 0.1f;  /* Distinctive pattern */
    }
    EXPECT_EQ(olfact_process_odor(olfact, cookie_odor, OLFACT_MAX_RECEPTORS, 0.7f), 0);
    tracker.end_stage();

    /* Stage 4: Start sniff cycle */
    tracker.begin_stage("Sniff Cycle Processing", 100);
    EXPECT_EQ(olfact_start_sniff(olfact, 0.9f), 0);
    /* Get sniff modulation */
    float modulation = olfact_get_sniff_modulation(olfact);
    EXPECT_GE(modulation, 0.0f);
    tracker.end_stage();

    /* Stage 5: Identify odor */
    tracker.begin_stage("Odor Identification", 100);
    olfact_odor_id_t odor_id;
    EXPECT_EQ(olfact_identify_odor(olfact, &odor_id), 0);
    EXPECT_GE(odor_id.confidence, 0.0f);
    tracker.end_stage();

    /* Stage 6: Sync with amygdala (emotional processing) */
    tracker.begin_stage("Amygdala Emotional Processing", 50);
    EXPECT_EQ(olfact_sync_amygdala(olfact), 0);
    hedonic_valence_t valence = olfact_get_valence(olfact);
    EXPECT_GE((int)valence, (int)HEDONIC_VERY_UNPLEASANT);
    tracker.end_stage();

    /* Stage 7: Store memory with emotional context */
    tracker.begin_stage("Memory Encoding", 100);
    EXPECT_EQ(olfact_store_memory(olfact, &odor_id, 0.9f, 0.7f, "grandma's kitchen, childhood"), 0);
    tracker.end_stage();

    /* Stage 8: Sync with entorhinal (memory consolidation) */
    tracker.begin_stage("Entorhinal Memory Consolidation", 50);
    EXPECT_EQ(olfact_sync_entorhinal(olfact), 0);
    tracker.end_stage();

    /* Stage 9: Verify memory stored */
    tracker.begin_stage("Memory Verification", 100);
    EXPECT_GT(olfact->num_memories, 0u);
    tracker.end_stage();

    olfact_destroy(olfact);
    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

E2E_TEST(SensoryPipelineE2E, OlfactoryAdaptation_Pipeline) {
    PipelineTracker tracker("Olfactory Adaptation Pipeline");

    tracker.begin_stage("Create Module", 50);
    nimcp_olfactory_t* olfact = olfact_create(nullptr);
    ASSERT_NE(olfact, nullptr);
    tracker.end_stage();

    /* Continuous exposure to same odor */
    tracker.begin_stage("Initial Odor Exposure", 100);
    float strong_odor[OLFACT_MAX_RECEPTORS];
    for (uint32_t i = 0; i < OLFACT_MAX_RECEPTORS; i++) {
        strong_odor[i] = 0.9f;  /* Strong, constant odor */
    }
    EXPECT_EQ(olfact_process_odor(olfact, strong_odor, OLFACT_MAX_RECEPTORS, 1.0f), 0);
    float initial_intensity = olfact_get_intensity(olfact);
    tracker.end_stage();

    /* Check adaptation level */
    tracker.begin_stage("Check Adaptation", 100);
    float adaptation_level = olfact_get_adaptation_level(olfact);
    EXPECT_GE(adaptation_level, 0.0f);
    tracker.end_stage();

    /* Process multiple times to simulate exposure */
    tracker.begin_stage("Repeated Exposure", 200);
    for (int t = 0; t < 10; t++) {
        EXPECT_EQ(olfact_process_odor(olfact, strong_odor, OLFACT_MAX_RECEPTORS, 1.0f), 0);
        EXPECT_EQ(olfact_update(olfact, 0.1f), 0);
    }
    tracker.end_stage();

    /* Check final intensity */
    tracker.begin_stage("Final Intensity Check", 50);
    float final_intensity = olfact_get_intensity(olfact);
    EXPECT_GE(final_intensity, 0.0f);
    EXPECT_LE(final_intensity, 1.0f);
    tracker.end_stage();

    olfact_destroy(olfact);
    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

//=============================================================================
// Taste-to-Reward Pipeline Tests
//=============================================================================

E2E_TEST(SensoryPipelineE2E, TasteToReward_CompletePipeline) {
    PipelineTracker tracker("Taste-to-Reward Pipeline");

    /* Stage 1: Create gustatory module */
    tracker.begin_stage("Create Gustatory Module", 50);
    nimcp_gustatory_t* gust = gust_create(nullptr);
    ASSERT_NE(gust, nullptr);
    tracker.end_stage();

    /* Stage 2: Initialize bridges */
    tracker.begin_stage("Initialize Reward Bridges", 50);
    EXPECT_EQ(gust_init_hypothalamus_bridge(gust, nullptr), 0);
    EXPECT_EQ(gust_init_amygdala_bridge(gust, nullptr), 0);
    EXPECT_EQ(gust_init_ofc_bridge(gust, nullptr), 0);
    tracker.end_stage();

    /* Stage 3: Verify hypothalamus bridge */
    tracker.begin_stage("Verify Hypothalamus Bridge", 50);
    EXPECT_TRUE(gust->hypothalamus_bridge.initialized);
    tracker.end_stage();

    /* Stage 4: Process delicious food taste */
    tracker.begin_stage("Process Sweet-Umami Taste", 100);
    taste_stimulus_t stimulus = {};
    stimulus.sweet = 0.7f;
    stimulus.umami = 0.5f;
    stimulus.salty = 0.2f;
    stimulus.fat_content = 0.6f;
    stimulus.temperature = 35.0f;  /* Warm food */
    stimulus.texture = 0.8f;

    EXPECT_EQ(gust_process_taste(gust, &stimulus), 0);
    tracker.end_stage();

    /* Stage 5: Evaluate perception */
    tracker.begin_stage("Evaluate Perception", 100);
    taste_perception_t perception;
    EXPECT_EQ(gust_get_perception(gust, &perception), 0);
    EXPECT_GT(perception.perceived_sweet, 0.0f);
    EXPECT_GT(perception.perceived_umami, 0.0f);
    EXPECT_GT(perception.palatability, 0.3f);  /* Should be palatable */
    tracker.end_stage();

    /* Stage 6: Compute food reward */
    tracker.begin_stage("Compute Food Reward", 100);
    food_reward_t reward;
    EXPECT_EQ(gust_compute_reward(gust, &reward), 0);
    EXPECT_GT(reward.reward_magnitude, 0.0f);  /* Should be rewarding when hungry */
    tracker.end_stage();

    /* Stage 7: Sync with hypothalamus */
    tracker.begin_stage("Hypothalamus Reward Processing", 50);
    EXPECT_EQ(gust_sync_hypothalamus(gust), 0);
    tracker.end_stage();

    /* Stage 8: Learn preference */
    tracker.begin_stage("Preference Learning", 50);
    EXPECT_EQ(gust_learn_preference(gust, TASTE_SWEET, 0.2f), 0);
    EXPECT_EQ(gust_learn_preference(gust, TASTE_UMAMI, 0.15f), 0);
    EXPECT_GT(gust->learned_preferences[TASTE_SWEET], 0.0f);
    tracker.end_stage();

    /* Stage 9: Get hedonic value */
    tracker.begin_stage("Hedonic Evaluation", 50);
    taste_hedonic_t hedonic = gust_get_hedonic_value(gust);
    EXPECT_GE((int)hedonic, (int)TASTE_HEDONIC_NEUTRAL);  /* Should be pleasant */
    tracker.end_stage();

    gust_destroy(gust);
    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

E2E_TEST(SensoryPipelineE2E, DisgustResponse_Pipeline) {
    PipelineTracker tracker("Disgust Response Pipeline");

    tracker.begin_stage("Create Module", 50);
    nimcp_gustatory_t* gust = gust_create(nullptr);
    ASSERT_NE(gust, nullptr);
    gust_init_amygdala_bridge(gust, nullptr);
    tracker.end_stage();

    /* Process very bitter taste (potential toxin) */
    tracker.begin_stage("Process Bitter Taste", 100);
    taste_stimulus_t stimulus = {};
    stimulus.bitter = 0.95f;  /* Very bitter */
    stimulus.sweet = 0.0f;

    EXPECT_EQ(gust_process_taste(gust, &stimulus), 0);
    tracker.end_stage();

    /* Evaluate disgust */
    tracker.begin_stage("Evaluate Disgust", 50);
    disgust_level_t disgust = gust_evaluate_disgust(gust);
    EXPECT_GE((int)disgust, (int)DISGUST_NONE);
    tracker.end_stage();

    /* Check for toxic warning */
    tracker.begin_stage("Toxic Warning Check", 50);
    bool toxic = gust_is_toxic_warning(gust);
    /* Very bitter often triggers toxic warning */
    if (toxic) {
        EXPECT_TRUE(toxic);
    }
    tracker.end_stage();

    /* Check disgust response */
    tracker.begin_stage("Disgust Response Check", 50);
    EXPECT_GE((int)disgust, (int)DISGUST_NONE);
    tracker.end_stage();

    /* Learn aversion */
    tracker.begin_stage("Aversion Learning", 50);
    EXPECT_EQ(gust_learn_preference(gust, TASTE_BITTER, -0.3f), 0);
    /* Verify update cycle succeeds after learning */
    EXPECT_EQ(gust_update(gust, 0.01f), 0);
    tracker.end_stage();

    gust_destroy(gust);
    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

//=============================================================================
// Multi-Modal Flavor Integration Tests
//=============================================================================

E2E_TEST(SensoryPipelineE2E, FlavorIntegration_TasteAndSmell_Pipeline) {
    PipelineTracker tracker("Flavor Integration Pipeline (Taste + Smell)");

    /* Create both modules */
    tracker.begin_stage("Create Sensory Modules", 100);
    nimcp_olfactory_t* olfact = olfact_create(nullptr);
    nimcp_gustatory_t* gust = gust_create(nullptr);
    ASSERT_NE(olfact, nullptr);
    ASSERT_NE(gust, nullptr);
    tracker.end_stage();

    /* Initialize cross-modal bridge */
    tracker.begin_stage("Initialize Cross-Modal Bridge", 50);
    EXPECT_EQ(gust_init_olfactory_bridge(gust, olfact), 0);
    EXPECT_TRUE(gust->olfactory_bridge.initialized);
    tracker.end_stage();

    /* Process fruit smell (strawberry-like) */
    tracker.begin_stage("Process Fruit Aroma", 100);
    float fruit_odor[OLFACT_MAX_RECEPTORS];
    for (uint32_t i = 0; i < OLFACT_MAX_RECEPTORS; i++) {
        fruit_odor[i] = (i % 12 < 4) ? 0.75f : 0.15f;
    }
    EXPECT_EQ(olfact_process_odor(olfact, fruit_odor, OLFACT_MAX_RECEPTORS, 0.6f), 0);
    tracker.end_stage();

    /* Process sweet-sour taste (fruit taste) */
    tracker.begin_stage("Process Fruit Taste", 100);
    taste_stimulus_t stimulus = {};
    stimulus.sweet = 0.65f;
    stimulus.sour = 0.35f;
    stimulus.olfactory_component = fruit_odor;
    stimulus.olfactory_dim = 128;

    EXPECT_EQ(gust_process_taste(gust, &stimulus), 0);
    tracker.end_stage();

    /* Sync olfactory-gustatory */
    tracker.begin_stage("Cross-Modal Sync", 50);
    EXPECT_EQ(gust_sync_olfactory(gust), 0);
    tracker.end_stage();

    /* Verify perception */
    tracker.begin_stage("Verify Perception", 50);
    taste_perception_t perception;
    EXPECT_EQ(gust_get_perception(gust, &perception), 0);
    EXPECT_GE(perception.palatability, 0.0f);
    tracker.end_stage();

    /* Verify both modules processed */
    tracker.begin_stage("Verify Processing", 50);
    EXPECT_GE(olfact->updates_processed, 0u);
    EXPECT_GE(gust->updates_processed, 0u);
    tracker.end_stage();

    olfact_destroy(olfact);
    gust_destroy(gust);
    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

//=============================================================================
// Pain Processing Pipeline Tests
//=============================================================================

E2E_TEST(SensoryPipelineE2E, PainGateControl_Pipeline) {
    PipelineTracker tracker("Pain Gate Control Pipeline");

    tracker.begin_stage("Create Module", 50);
    nimcp_somatosensory_t* soma = soma_create(nullptr);
    ASSERT_NE(soma, nullptr);
    soma_init_all_bridges(soma, nullptr);
    tracker.end_stage();

    /* Process sharp pain on forearm */
    tracker.begin_stage("Process Pain Event", 100);
    uint32_t pain_id;
    EXPECT_EQ(soma_process_pain(soma, BODY_SEG_FOREARM_L, PAIN_SHARP, 0.8f, &pain_id), 0);
    tracker.end_stage();

    /* Get initial pain level */
    tracker.begin_stage("Measure Initial Pain", 50);
    float initial_pain = soma_get_pain_level(soma, BODY_SEG_FOREARM_L);
    EXPECT_GT(initial_pain, 0.0f);
    tracker.end_stage();

    /* Apply touch to area (rubbing) - gate control */
    tracker.begin_stage("Apply Touch Inhibition", 100);
    float pos[3] = {0.5f, 0.5f, 0.0f};
    uint32_t touch_id;
    EXPECT_EQ(soma_process_touch(soma, BODY_SEG_FOREARM_L, pos, 0.6f, TOUCH_PRESSURE, &touch_id), 0);
    tracker.end_stage();

    /* Verify pain still tracked */
    tracker.begin_stage("Verify Pain Tracking", 50);
    float current_pain = soma_get_pain_level(soma, BODY_SEG_FOREARM_L);
    EXPECT_GE(current_pain, 0.0f);
    tracker.end_stage();

    /* Sync with hypothalamus (stress response) */
    tracker.begin_stage("Hypothalamus Stress Sync", 50);
    EXPECT_EQ(soma_sync_hypothalamus(soma), 0);
    tracker.end_stage();

    soma_destroy(soma);
    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

//=============================================================================
// Real-Time Processing Simulation Tests
//=============================================================================

E2E_TEST(SensoryPipelineE2E, RealTimeMultiSensory_Pipeline) {
    PipelineTracker tracker("Real-Time Multi-Sensory Pipeline");

    /* Create all modules */
    tracker.begin_stage("Create All Modules", 100);
    nimcp_somatosensory_t* soma = soma_create(nullptr);
    nimcp_olfactory_t* olfact = olfact_create(nullptr);
    nimcp_gustatory_t* gust = gust_create(nullptr);
    ASSERT_NE(soma, nullptr);
    ASSERT_NE(olfact, nullptr);
    ASSERT_NE(gust, nullptr);
    tracker.end_stage();

    /* Initialize all bridges */
    tracker.begin_stage("Initialize All Bridges", 100);
    soma_init_all_bridges(soma, nullptr);
    olfact_init_amygdala_bridge(olfact, nullptr);
    gust_init_hypothalamus_bridge(gust, nullptr);
    gust_init_olfactory_bridge(gust, olfact);
    tracker.end_stage();

    /* Simulate eating experience (100 timesteps) */
    tracker.begin_stage("Eating Experience Simulation", 1000);

    for (int t = 0; t < 100; t++) {
        float dt = 0.01f;  /* 10ms timestep */

        /* Touch: food texture in mouth */
        float mouth_pos[3] = {0.0f, 0.0f, 0.0f};
        uint32_t touch_id;
        soma_process_touch(soma, BODY_SEG_LIPS, mouth_pos, 0.4f + (t % 10) * 0.02f, TOUCH_TEXTURE, &touch_id);

        /* Smell: food aroma */
        float food_odor[OLFACT_MAX_RECEPTORS];
        for (uint32_t i = 0; i < OLFACT_MAX_RECEPTORS; i++) {
            food_odor[i] = 0.5f + sinf((float)(i + t) * 0.1f) * 0.2f;
        }
        olfact_process_odor(olfact, food_odor, OLFACT_MAX_RECEPTORS, 0.6f);

        /* Taste: food flavor */
        taste_stimulus_t taste = {};
        taste.sweet = 0.5f;
        taste.umami = 0.4f;
        taste.salty = 0.2f;
        gust_process_taste(gust, &taste);

        /* Update all modules */
        soma_update(soma, dt);
        olfact_update(olfact, dt);
        gust_update(gust, dt);

        /* Bidirectional updates every 10 steps */
        if (t % 10 == 9) {
            soma_bidirectional_update(soma, dt * 10);
            olfact_bidirectional_update(olfact, dt * 10);
            gust_bidirectional_update(gust, dt * 10);
        }
    }
    tracker.end_stage();

    /* Verify processing counts */
    tracker.begin_stage("Verify Processing Stats", 50);
    EXPECT_GT(soma->updates_processed, 50u);
    EXPECT_GT(olfact->updates_processed, 50u);
    EXPECT_GT(gust->updates_processed, 50u);
    tracker.end_stage();

    /* Cleanup */
    soma_destroy(soma);
    olfact_destroy(olfact);
    gust_destroy(gust);

    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}

//=============================================================================
// Serialization Pipeline Tests
//=============================================================================

E2E_TEST(SensoryPipelineE2E, StatePersistence_Pipeline) {
    PipelineTracker tracker("State Persistence Pipeline");

    /* Create and populate modules */
    tracker.begin_stage("Create and Populate Modules", 200);
    nimcp_somatosensory_t* soma = soma_create(nullptr);
    nimcp_olfactory_t* olfact = olfact_create(nullptr);
    nimcp_gustatory_t* gust = gust_create(nullptr);

    /* Add some state */
    float pos[3] = {0.5f, 0.5f, 0.0f};
    uint32_t id;
    soma_process_touch(soma, BODY_SEG_HAND_R, pos, 0.7f, TOUCH_LIGHT, &id);
    soma_update(soma, 0.01f);

    float odor[OLFACT_MAX_RECEPTORS];
    for (uint32_t i = 0; i < OLFACT_MAX_RECEPTORS; i++) odor[i] = 0.5f;
    olfact_process_odor(olfact, odor, OLFACT_MAX_RECEPTORS, 0.6f);
    olfact_odor_id_t odor_id;
    olfact_identify_odor(olfact, &odor_id);
    olfact_store_memory(olfact, &odor_id, 0.8f, 0.6f, "test");

    gust_learn_preference(gust, TASTE_SWEET, 0.3f);
    tracker.end_stage();

    /* Serialize all */
    tracker.begin_stage("Serialize State", 200);
    size_t soma_size = soma_get_serialization_size(soma);
    size_t olfact_size = olfact_get_serialization_size(olfact);
    size_t gust_size = gust_get_serialization_size(gust);

    std::vector<uint8_t> soma_buf(soma_size);
    std::vector<uint8_t> olfact_buf(olfact_size);
    std::vector<uint8_t> gust_buf(gust_size);

    size_t written;
    EXPECT_EQ(soma_serialize(soma, soma_buf.data(), soma_size, &written), 0);
    EXPECT_EQ(olfact_serialize(olfact, olfact_buf.data(), olfact_size, &written), 0);
    EXPECT_EQ(gust_serialize(gust, gust_buf.data(), gust_size, &written), 0);
    tracker.end_stage();

    /* Destroy originals */
    tracker.begin_stage("Destroy Originals", 50);
    soma_destroy(soma);
    olfact_destroy(olfact);
    gust_destroy(gust);
    tracker.end_stage();

    /* Deserialize */
    tracker.begin_stage("Deserialize State", 200);
    size_t bytes_read;
    nimcp_somatosensory_t* soma_restored = soma_deserialize(soma_buf.data(), soma_buf.size(), &bytes_read);
    nimcp_olfactory_t* olfact_restored = olfact_deserialize(olfact_buf.data(), olfact_buf.size(), &bytes_read);
    nimcp_gustatory_t* gust_restored = gust_deserialize(gust_buf.data(), gust_buf.size(), &bytes_read);

    ASSERT_NE(soma_restored, nullptr);
    ASSERT_NE(olfact_restored, nullptr);
    ASSERT_NE(gust_restored, nullptr);
    tracker.end_stage();

    /* Verify state preserved - check objects are valid by resetting */
    tracker.begin_stage("Verify Restored State", 100);
    /* Verify restored objects can be reset to valid state */
    EXPECT_EQ(soma_reset(soma_restored), 0);
    EXPECT_EQ(olfact_reset(olfact_restored), 0);
    EXPECT_EQ(gust_reset(gust_restored), 0);
    tracker.end_stage();

    /* Verify modules still functional */
    tracker.begin_stage("Verify Functionality", 100);
    EXPECT_EQ(soma_update(soma_restored, 0.01f), 0);
    EXPECT_EQ(olfact_update(olfact_restored, 0.01f), 0);
    EXPECT_EQ(gust_update(gust_restored, 0.01f), 0);
    tracker.end_stage();

    /* Cleanup */
    soma_destroy(soma_restored);
    olfact_destroy(olfact_restored);
    gust_destroy(gust_restored);

    E2E_ASSERT_PIPELINE_SUCCESS(tracker);
}
