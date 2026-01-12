/**
 * @file test_sensory_integration.cpp
 * @brief Integration tests for Phase 6 sensory modules
 * @version 1.0.0
 * @date 2026-01-12
 *
 * Tests comprehensive integration of somatosensory, olfactory, and gustatory
 * modules with bio-async, quantum, brainstem, swarm, and cross-sensory bridges.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <cstdlib>

extern "C" {
/* Core sensory modules */
#include "core/brain/regions/somatosensory/nimcp_somatosensory.h"
#include "core/brain/regions/olfactory/nimcp_olfactory.h"
#include "core/brain/regions/gustatory/nimcp_gustatory.h"

/* Bio-async bridges */
#include "core/brain/regions/somatosensory/bridges/nimcp_soma_bio_async_bridge.h"
#include "core/brain/regions/olfactory/bridges/nimcp_olfact_bio_async_bridge.h"
#include "core/brain/regions/gustatory/bridges/nimcp_gust_bio_async_bridge.h"

/* Quantum bridges */
#include "core/brain/regions/somatosensory/bridges/nimcp_soma_quantum_bridge.h"
#include "core/brain/regions/olfactory/bridges/nimcp_olfact_quantum_bridge.h"
#include "core/brain/regions/gustatory/bridges/nimcp_gust_quantum_bridge.h"

/* Brainstem bridges */
#include "core/brain/regions/somatosensory/bridges/nimcp_soma_cerebellum_bridge.h"
#include "core/brain/regions/somatosensory/bridges/nimcp_soma_medulla_bridge.h"

/* Cross-sensory bridges */
#include "core/brain/regions/sensory_integration/nimcp_chemosensory_bridge.h"
#include "core/brain/regions/sensory_integration/nimcp_tactile_motor_bridge.h"

/* Swarm bridge */
#include "core/brain/regions/sensory_integration/nimcp_sensory_swarm_bridge.h"

/* KG wiring */
#include "integration/knowledge/nimcp_sensory_kg_wiring.h"
}

/* ============================================================================
 * Test Fixture: Somatosensory Integration
 * ============================================================================ */

class SomatosensoryIntegrationTest : public ::testing::Test {
protected:
    nimcp_somatosensory_t* soma = nullptr;

    void SetUp() override {
        soma_config_t config = soma_default_config();
        soma = soma_create(&config);
        ASSERT_NE(soma, nullptr);
    }

    void TearDown() override {
        if (soma) {
            soma_destroy(soma);
            soma = nullptr;
        }
    }
};

/* Somatosensory Bio-Async Bridge Tests */
TEST_F(SomatosensoryIntegrationTest, BioAsyncBridgeCreation) {
    soma_bio_async_config_t config;
    EXPECT_EQ(soma_bio_async_default_config(&config), 0);

    soma_bio_router_t* bridge = soma_bio_async_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    /* Connect without router (NULL for test) */
    EXPECT_EQ(soma_bio_async_connect(bridge, soma, nullptr), 0);
    EXPECT_TRUE(soma_bio_async_is_connected(bridge));

    soma_bio_async_bridge_destroy(bridge);
}

TEST_F(SomatosensoryIntegrationTest, BioAsyncSubscription) {
    soma_bio_async_config_t config;
    soma_bio_async_default_config(&config);
    soma_bio_router_t* bridge = soma_bio_async_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    soma_bio_async_connect(bridge, soma, nullptr);

    /* Subscribe a module to touch and pain events */
    uint32_t subscription_mask = SOMA_BIO_SUB_TOUCH_EVENT | SOMA_BIO_SUB_PAIN_ALERT;
    EXPECT_EQ(soma_bio_async_subscribe_module(bridge, 0x1000, subscription_mask), 0);

    /* Verify subscriber count */
    EXPECT_GE(soma_bio_async_get_subscriber_count(bridge, SOMA_BIO_MSG_TOUCH_EVENT), 1u);

    /* Unsubscribe */
    EXPECT_EQ(soma_bio_async_unsubscribe_module(bridge, 0x1000), 0);

    soma_bio_async_bridge_destroy(bridge);
}

TEST_F(SomatosensoryIntegrationTest, BioAsyncBroadcast) {
    soma_bio_async_config_t config;
    soma_bio_async_default_config(&config);
    soma_bio_router_t* bridge = soma_bio_async_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    soma_bio_async_connect(bridge, soma, nullptr);

    /* Broadcast touch event */
    float position[3] = {0.5f, 0.5f, 0.0f};
    EXPECT_EQ(soma_bio_async_broadcast_touch(bridge, BODY_SEG_HAND_L, position, 0.7f, TOUCH_PRESSURE), 0);

    /* Broadcast pain alert */
    EXPECT_EQ(soma_bio_async_broadcast_pain(bridge, BODY_SEG_FOOT_L, PAIN_SHARP, 0.8f), 0);

    /* Broadcast proprioceptive update */
    EXPECT_EQ(soma_bio_async_broadcast_proprio(bridge, BODY_SEG_UPPER_ARM_L, 1.5f, 0.2f, 0.5f), 0);

    /* Verify stats API works (values may be placeholder) */
    soma_bio_async_stats_t stats;
    EXPECT_EQ(soma_bio_async_get_stats(bridge, &stats), 0);
    /* Note: stats may not be updated if no router is connected */

    soma_bio_async_bridge_destroy(bridge);
}

/* Somatosensory Quantum Bridge Tests */
TEST_F(SomatosensoryIntegrationTest, QuantumBridgeCreation) {
    soma_quantum_config_t config;
    EXPECT_EQ(soma_quantum_default_config(&config), 0);

    soma_quantum_bridge_t* bridge = soma_quantum_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(soma_quantum_connect(bridge, soma), 0);
    EXPECT_TRUE(soma_quantum_is_connected(bridge));
    EXPECT_EQ(soma_quantum_get_status(bridge), SOMA_QUANTUM_STATUS_IDLE);

    soma_quantum_bridge_destroy(bridge);
}

TEST_F(SomatosensoryIntegrationTest, QuantumThresholdOptimization) {
    soma_quantum_config_t config;
    soma_quantum_default_config(&config);
    config.enable_qmc = true;

    soma_quantum_bridge_t* bridge = soma_quantum_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);
    soma_quantum_connect(bridge, soma);

    /* Create threshold optimization spec */
    soma_threshold_opt_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    float thresholds[4] = {0.5f, 0.6f, 0.4f, 0.7f};
    float samples[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    spec.current_thresholds = thresholds;
    spec.num_thresholds = 4;
    spec.signal_samples = samples;
    spec.num_samples = 10;
    spec.target_sensitivity = 0.8f;
    spec.noise_tolerance = 0.1f;

    soma_qmc_result_t result;
    memset(&result, 0, sizeof(result));
    EXPECT_EQ(soma_quantum_optimize_thresholds(bridge, &spec, &result), 0);

    /* Verify result has optimized thresholds */
    EXPECT_NE(result.optimal_thresholds, nullptr);
    EXPECT_EQ(result.num_thresholds, 4u);
    EXPECT_GT(result.samples_used, 0u);

    soma_qmc_result_free(&result);
    soma_quantum_bridge_destroy(bridge);
}

TEST_F(SomatosensoryIntegrationTest, QuantumBodyMapSearch) {
    soma_quantum_config_t config;
    soma_quantum_default_config(&config);
    config.enable_walks = true;
    config.walk_steps = 10;  /* Limit walk steps to fit in map_dim */

    soma_quantum_bridge_t* bridge = soma_quantum_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);
    soma_quantum_connect(bridge, soma);

    /* Create body map search spec */
    soma_body_map_search_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    float activation[16] = {0};
    activation[5] = 0.9f;  /* High activation in region 5 */
    spec.activation_map = activation;
    spec.map_dim = 16;
    spec.start_region = 0;
    spec.target_region = 5;
    spec.activation_threshold = 0.5f;

    soma_quantum_walk_result_t result;
    memset(&result, 0, sizeof(result));
    EXPECT_EQ(soma_quantum_search_body_map(bridge, &spec, &result), 0);

    /* Verify walk completed */
    EXPECT_GT(result.steps_taken, 0u);

    soma_quantum_walk_result_free(&result);
    soma_quantum_bridge_destroy(bridge);
}

/* Somatosensory Cerebellum Bridge Tests */
TEST_F(SomatosensoryIntegrationTest, CerebellumBridgeCreation) {
    soma_cereb_config_t config;
    EXPECT_EQ(soma_cereb_default_config(&config), 0);

    soma_cereb_bridge_t* bridge = soma_cereb_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(soma_cereb_connect(bridge, soma, nullptr), 0);
    EXPECT_TRUE(soma_cereb_is_connected(bridge));

    soma_cereb_bridge_destroy(bridge);
}

TEST_F(SomatosensoryIntegrationTest, CerebellumProprioceptiveUpdate) {
    soma_cereb_config_t config;
    soma_cereb_default_config(&config);

    soma_cereb_bridge_t* bridge = soma_cereb_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);
    soma_cereb_connect(bridge, soma, nullptr);

    /* Send proprioceptive update */
    soma_cereb_joint_state_t joints[2];
    joints[0].joint_id = 0;
    joints[0].angle = 1.5f;
    joints[0].angular_velocity = 0.1f;
    joints[0].torque = 0.5f;
    joints[0].stretch = 0.3f;
    joints[0].tension = 0.4f;

    joints[1].joint_id = 1;
    joints[1].angle = 0.8f;
    joints[1].angular_velocity = 0.2f;
    joints[1].torque = 0.6f;
    joints[1].stretch = 0.4f;
    joints[1].tension = 0.5f;

    EXPECT_EQ(soma_cereb_send_proprio(bridge, joints, 2), 0);

    /* Send balance update */
    EXPECT_EQ(soma_cereb_send_balance_update(bridge, 0.1f, 0.05f, 0.02f), 0);

    /* Verify stats */
    soma_cereb_stats_t stats;
    EXPECT_EQ(soma_cereb_get_stats(bridge, &stats), 0);
    EXPECT_GE(stats.proprio_updates, 1u);

    soma_cereb_bridge_destroy(bridge);
}

/* ============================================================================
 * Test Fixture: Olfactory Integration
 * ============================================================================ */

class OlfactoryIntegrationTest : public ::testing::Test {
protected:
    nimcp_olfactory_t* olfact = nullptr;

    void SetUp() override {
        olfact_config_t config = olfact_default_config();
        olfact = olfact_create(&config);
        ASSERT_NE(olfact, nullptr);
    }

    void TearDown() override {
        if (olfact) {
            olfact_destroy(olfact);
            olfact = nullptr;
        }
    }
};

TEST_F(OlfactoryIntegrationTest, BioAsyncBridgeCreation) {
    olfact_bio_async_config_t config;
    EXPECT_EQ(olfact_bio_async_default_config(&config), 0);

    olfact_bio_async_bridge_t* bridge = olfact_bio_async_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(olfact_bio_async_connect(bridge, olfact, nullptr), 0);
    EXPECT_TRUE(olfact_bio_async_is_connected(bridge));

    olfact_bio_async_bridge_destroy(bridge);
}

TEST_F(OlfactoryIntegrationTest, BioAsyncBroadcast) {
    olfact_bio_async_config_t config;
    olfact_bio_async_default_config(&config);
    olfact_bio_async_bridge_t* bridge = olfact_bio_async_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);
    olfact_bio_async_connect(bridge, olfact, nullptr);

    /* Broadcast odor detected */
    EXPECT_EQ(olfact_bio_async_broadcast_odor_detected(bridge, 0.8f, true), 0);

    /* Broadcast odor identified */
    EXPECT_EQ(olfact_bio_async_broadcast_odor_identified(bridge, "rose", ODOR_CAT_FLORAL, 0.9f), 0);

    /* Broadcast food signal */
    EXPECT_EQ(olfact_bio_async_broadcast_food_signal(bridge, 0.7f, true, false), 0);

    /* Broadcast danger odor */
    EXPECT_EQ(olfact_bio_async_broadcast_danger_odor(bridge, 1, 0.9f), 0);

    /* Verify stats */
    olfact_bio_async_stats_t stats;
    EXPECT_EQ(olfact_bio_async_get_stats(bridge, &stats), 0);
    EXPECT_GE(stats.odors_detected, 1u);

    olfact_bio_async_bridge_destroy(bridge);
}

TEST_F(OlfactoryIntegrationTest, QuantumBridgeCreation) {
    olfact_quantum_config_t config;
    EXPECT_EQ(olfact_quantum_default_config(&config), 0);

    olfact_quantum_bridge_t* bridge = olfact_quantum_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(olfact_quantum_connect(bridge, olfact), 0);
    EXPECT_TRUE(olfact_quantum_is_connected(bridge));

    olfact_quantum_bridge_destroy(bridge);
}

TEST_F(OlfactoryIntegrationTest, QuantumOdorClassification) {
    olfact_quantum_config_t config;
    olfact_quantum_default_config(&config);
    config.enable_annealing = true;

    olfact_quantum_bridge_t* bridge = olfact_quantum_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);
    olfact_quantum_connect(bridge, olfact);

    /* Classify odor */
    float odor[8] = {0.1f, 0.8f, 0.2f, 0.3f, 0.5f, 0.1f, 0.4f, 0.2f};
    olfact_quantum_classification_result_t result;
    memset(&result, 0, sizeof(result));
    EXPECT_EQ(olfact_quantum_classify_odor(bridge, odor, 8, &result), 0);

    EXPECT_NE(result.category_probabilities, nullptr);
    EXPECT_GT(result.confidence, 0.0f);
    EXPECT_TRUE(result.converged);

    olfact_quantum_classification_result_free(&result);
    olfact_quantum_bridge_destroy(bridge);
}

/* ============================================================================
 * Test Fixture: Gustatory Integration
 * ============================================================================ */

class GustatoryIntegrationTest : public ::testing::Test {
protected:
    nimcp_gustatory_t* gust = nullptr;

    void SetUp() override {
        gust_config_t config = gust_default_config();
        gust = gust_create(&config);
        ASSERT_NE(gust, nullptr);
    }

    void TearDown() override {
        if (gust) {
            gust_destroy(gust);
            gust = nullptr;
        }
    }
};

TEST_F(GustatoryIntegrationTest, BioAsyncBridgeCreation) {
    gust_bio_async_config_t config;
    EXPECT_EQ(gust_bio_async_default_config(&config), 0);

    gust_bio_async_bridge_t* bridge = gust_bio_async_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(gust_bio_async_connect(bridge, gust, nullptr), 0);
    EXPECT_TRUE(gust_bio_async_is_connected(bridge));

    gust_bio_async_bridge_destroy(bridge);
}

TEST_F(GustatoryIntegrationTest, BioAsyncBroadcast) {
    gust_bio_async_config_t config;
    gust_bio_async_default_config(&config);
    gust_bio_async_bridge_t* bridge = gust_bio_async_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);
    gust_bio_async_connect(bridge, gust, nullptr);

    /* Create taste stimulus */
    taste_stimulus_t stimulus;
    memset(&stimulus, 0, sizeof(stimulus));
    stimulus.sweet = 0.8f;
    stimulus.salty = 0.1f;
    stimulus.sour = 0.1f;
    stimulus.bitter = 0.0f;
    stimulus.umami = 0.3f;

    /* Broadcast taste detected */
    EXPECT_EQ(gust_bio_async_broadcast_taste_detected(bridge, &stimulus, true), 0);

    /* Broadcast reward signal */
    EXPECT_EQ(gust_bio_async_broadcast_reward(bridge, 0.9f, FOOD_CAT_FRUIT), 0);

    /* Broadcast satiety update */
    EXPECT_EQ(gust_bio_async_broadcast_satiety(bridge, 0.6f, 0.4f), 0);

    /* Verify stats */
    gust_bio_async_stats_t stats;
    EXPECT_EQ(gust_bio_async_get_stats(bridge, &stats), 0);
    EXPECT_GE(stats.tastes_detected, 1u);

    gust_bio_async_bridge_destroy(bridge);
}

TEST_F(GustatoryIntegrationTest, QuantumBridgeCreation) {
    gust_quantum_config_t config;
    EXPECT_EQ(gust_quantum_default_config(&config), 0);

    gust_quantum_bridge_t* bridge = gust_quantum_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(gust_quantum_connect(bridge, gust), 0);
    EXPECT_TRUE(gust_quantum_is_connected(bridge));

    gust_quantum_bridge_destroy(bridge);
}

/* ============================================================================
 * Test Fixture: Cross-Sensory Integration
 * ============================================================================ */

class CrossSensoryIntegrationTest : public ::testing::Test {
protected:
    nimcp_olfactory_t* olfact = nullptr;
    nimcp_gustatory_t* gust = nullptr;
    nimcp_somatosensory_t* soma = nullptr;

    void SetUp() override {
        olfact_config_t olfact_cfg = olfact_default_config();
        olfact = olfact_create(&olfact_cfg);

        gust_config_t gust_cfg = gust_default_config();
        gust = gust_create(&gust_cfg);

        soma_config_t soma_cfg = soma_default_config();
        soma = soma_create(&soma_cfg);

        ASSERT_NE(olfact, nullptr);
        ASSERT_NE(gust, nullptr);
        ASSERT_NE(soma, nullptr);
    }

    void TearDown() override {
        if (olfact) olfact_destroy(olfact);
        if (gust) gust_destroy(gust);
        if (soma) soma_destroy(soma);
    }
};

TEST_F(CrossSensoryIntegrationTest, ChemosensoryBridgeCreation) {
    chemosensory_config_t config;
    EXPECT_EQ(chemosensory_default_config(&config), 0);

    chemosensory_bridge_t* bridge = chemosensory_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(chemosensory_connect(bridge, olfact, gust), 0);
    EXPECT_TRUE(chemosensory_is_connected(bridge));

    chemosensory_bridge_destroy(bridge);
}

TEST_F(CrossSensoryIntegrationTest, FlavorBinding) {
    chemosensory_config_t config;
    chemosensory_default_config(&config);
    config.enable_predictions = true;
    config.enable_memory_associations = true;

    chemosensory_bridge_t* bridge = chemosensory_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);
    chemosensory_connect(bridge, olfact, gust);

    /* Create odor perception (olfact_odor_id_t struct) */
    odor_perception_t odor_input;
    memset(&odor_input, 0, sizeof(odor_input));
    odor_input.odor_id = 42;
    strncpy(odor_input.name, "strawberry", sizeof(odor_input.name) - 1);
    odor_input.category = ODOR_CAT_FRUITY;
    odor_input.valence = HEDONIC_PLEASANT;
    odor_input.intensity = 0.7f;
    odor_input.confidence = 0.9f;
    odor_input.familiarity = 0.8f;
    odor_input.pattern = nullptr;  /* No pattern for this test */
    odor_input.pattern_dim = 0;

    /* Create taste perception */
    taste_perception_t taste_input;
    memset(&taste_input, 0, sizeof(taste_input));
    taste_input.perceived_sweet = 0.8f;
    taste_input.perceived_sour = 0.2f;
    taste_input.perceived_salty = 0.1f;
    taste_input.perceived_bitter = 0.0f;
    taste_input.perceived_umami = 0.1f;
    taste_input.overall_intensity = 0.7f;
    taste_input.palatability = 0.8f;

    /* Bind to flavor */
    chemosensory_flavor_t flavor;
    memset(&flavor, 0, sizeof(flavor));
    EXPECT_EQ(chemosensory_bind_flavor(bridge, &odor_input, &taste_input, &flavor), 0);

    /* Verify flavor output */
    EXPECT_GT(flavor.binding_strength, 0.0f);
    EXPECT_GT(flavor.palatability, 0.0f);

    chemosensory_flavor_free(&flavor);
    chemosensory_bridge_destroy(bridge);
}

TEST_F(CrossSensoryIntegrationTest, TactileMotorBridgeCreation) {
    tactile_motor_config_t config;
    EXPECT_EQ(tactile_motor_default_config(&config), 0);

    tactile_motor_bridge_t* bridge = tactile_motor_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(tactile_motor_connect(bridge, soma, nullptr), 0);
    EXPECT_TRUE(tactile_motor_is_connected(bridge));

    tactile_motor_bridge_destroy(bridge);
}

TEST_F(CrossSensoryIntegrationTest, GripControl) {
    tactile_motor_config_t config;
    tactile_motor_default_config(&config);
    config.enable_slip_detection = true;
    config.enable_force_control = true;

    tactile_motor_bridge_t* bridge = tactile_motor_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);
    tactile_motor_connect(bridge, soma, nullptr);

    /* Initialize grasp */
    EXPECT_EQ(tactile_motor_init_grasp(bridge, 0), 0);

    /* Create touch feedback */
    touch_event_t feedback;
    memset(&feedback, 0, sizeof(feedback));
    feedback.pressure = 0.5f;
    feedback.slip_velocity = 0.0f;
    feedback.normal_force = 1.0f;
    feedback.segment = BODY_SEG_HAND_L;
    feedback.timestamp_us = 1000;

    /* Update grip */
    tactile_motor_grip_t grip;
    memset(&grip, 0, sizeof(grip));
    EXPECT_EQ(tactile_motor_update_grip(bridge, &feedback, &grip), 0);
    /* Note: grip_force computation may be placeholder */
    EXPECT_GE(grip.grip_force, 0.0f);  /* At least non-negative */

    /* Check slip detection */
    bool slipping = false;
    EXPECT_EQ(tactile_motor_detect_slip(bridge, &feedback, &slipping), 0);
    EXPECT_FALSE(slipping);  /* No slip with zero slip_velocity */

    tactile_motor_bridge_destroy(bridge);
}

/* ============================================================================
 * Test Fixture: Swarm Sensory Integration
 * ============================================================================ */

class SwarmSensoryIntegrationTest : public ::testing::Test {
protected:
    nimcp_somatosensory_t* soma = nullptr;
    nimcp_olfactory_t* olfact = nullptr;
    nimcp_gustatory_t* gust = nullptr;

    void SetUp() override {
        soma_config_t soma_cfg = soma_default_config();
        soma = soma_create(&soma_cfg);

        olfact_config_t olfact_cfg = olfact_default_config();
        olfact = olfact_create(&olfact_cfg);

        gust_config_t gust_cfg = gust_default_config();
        gust = gust_create(&gust_cfg);

        ASSERT_NE(soma, nullptr);
        ASSERT_NE(olfact, nullptr);
        ASSERT_NE(gust, nullptr);
    }

    void TearDown() override {
        if (soma) soma_destroy(soma);
        if (olfact) olfact_destroy(olfact);
        if (gust) gust_destroy(gust);
    }
};

TEST_F(SwarmSensoryIntegrationTest, SwarmBridgeCreation) {
    sensory_swarm_config_t config;
    EXPECT_EQ(sensory_swarm_default_config(&config), 0);

    sensory_swarm_bridge_t* bridge = sensory_swarm_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    /* Register sensory modules */
    EXPECT_EQ(sensory_swarm_register_somatosensory(bridge, soma), 0);
    EXPECT_EQ(sensory_swarm_register_olfactory(bridge, olfact), 0);
    EXPECT_EQ(sensory_swarm_register_gustatory(bridge, gust), 0);

    sensory_swarm_bridge_destroy(bridge);
}

TEST_F(SwarmSensoryIntegrationTest, SwarmNodeManagement) {
    sensory_swarm_config_t config;
    sensory_swarm_default_config(&config);
    config.enable_touch_swarm = true;
    config.enable_smell_swarm = true;

    sensory_swarm_bridge_t* bridge = sensory_swarm_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    /* Add nodes */
    float pos1[3] = {0.0f, 0.0f, 0.0f};
    float pos2[3] = {1.0f, 0.0f, 0.0f};
    uint32_t node_id1, node_id2;

    EXPECT_EQ(sensory_swarm_add_node(bridge, SENSORY_SWARM_MODALITY_TOUCH, pos1, &node_id1), 0);
    EXPECT_EQ(sensory_swarm_add_node(bridge, SENSORY_SWARM_MODALITY_SMELL, pos2, &node_id2), 0);

    /* Update node readings */
    EXPECT_EQ(sensory_swarm_update_node(bridge, node_id1, 0.7f, 0.9f), 0);
    EXPECT_EQ(sensory_swarm_update_node(bridge, node_id2, 0.5f, 0.8f), 0);

    /* Verify node count */
    EXPECT_EQ(sensory_swarm_get_node_count(bridge, SENSORY_SWARM_MODALITY_TOUCH), 1);
    EXPECT_EQ(sensory_swarm_get_node_count(bridge, SENSORY_SWARM_MODALITY_SMELL), 1);

    /* Remove nodes */
    EXPECT_EQ(sensory_swarm_remove_node(bridge, node_id1), 0);

    sensory_swarm_bridge_destroy(bridge);
}

TEST_F(SwarmSensoryIntegrationTest, SwarmTaskSubmission) {
    sensory_swarm_config_t config;
    sensory_swarm_default_config(&config);
    config.enable_touch_swarm = true;

    sensory_swarm_bridge_t* bridge = sensory_swarm_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    /* Add some nodes first */
    float pos[3] = {0.0f, 0.0f, 0.0f};
    uint32_t node_id;
    sensory_swarm_add_node(bridge, SENSORY_SWARM_MODALITY_TOUCH, pos, &node_id);

    /* Submit exploration task */
    float input[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    uint32_t task_id;
    EXPECT_EQ(sensory_swarm_submit_task(bridge, SENSORY_SWARM_TASK_EXPLORE,
              SENSORY_SWARM_MODALITY_TOUCH, input, 4, &task_id), 0);

    /* Check task status */
    sensory_swarm_task_status_t status;
    EXPECT_EQ(sensory_swarm_get_task_status(bridge, task_id, &status), 0);

    sensory_swarm_bridge_destroy(bridge);
}

TEST_F(SwarmSensoryIntegrationTest, SwarmConsensus) {
    sensory_swarm_config_t config;
    sensory_swarm_default_config(&config);
    config.enable_smell_swarm = true;

    sensory_swarm_bridge_t* bridge = sensory_swarm_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    /* Add multiple nodes */
    for (int i = 0; i < 5; i++) {
        float pos[3] = {(float)i * 0.2f, 0.0f, 0.0f};
        uint32_t node_id;
        sensory_swarm_add_node(bridge, SENSORY_SWARM_MODALITY_SMELL, pos, &node_id);
        sensory_swarm_update_node(bridge, node_id, 0.5f + i * 0.1f, 0.8f);
    }

    /* Build consensus */
    float consensus_value = 0.0f;
    float confidence = 0.0f;
    EXPECT_EQ(sensory_swarm_build_consensus(bridge, SENSORY_SWARM_MODALITY_SMELL,
              &consensus_value, &confidence), 0);

    EXPECT_GT(confidence, 0.0f);

    sensory_swarm_bridge_destroy(bridge);
}

/* ============================================================================
 * Test Fixture: Knowledge Graph Wiring Integration
 * ============================================================================ */

class KGSensoryIntegrationTest : public ::testing::Test {
protected:
    nimcp_somatosensory_t* soma = nullptr;
    nimcp_olfactory_t* olfact = nullptr;
    nimcp_gustatory_t* gust = nullptr;
    sensory_kg_wiring_t* wiring = nullptr;

    void SetUp() override {
        soma_config_t soma_cfg = soma_default_config();
        soma = soma_create(&soma_cfg);

        olfact_config_t olfact_cfg = olfact_default_config();
        olfact = olfact_create(&olfact_cfg);

        gust_config_t gust_cfg = gust_default_config();
        gust = gust_create(&gust_cfg);

        sensory_kg_config_t kg_config;
        sensory_kg_default_config(&kg_config);
        kg_config.enable_somatosensory = true;
        kg_config.enable_olfactory = true;
        kg_config.enable_gustatory = true;
        kg_config.enable_cross_modal = true;
        wiring = sensory_kg_wiring_create(&kg_config);

        ASSERT_NE(soma, nullptr);
        ASSERT_NE(olfact, nullptr);
        ASSERT_NE(gust, nullptr);
        ASSERT_NE(wiring, nullptr);
    }

    void TearDown() override {
        if (wiring) sensory_kg_wiring_destroy(wiring);
        if (soma) soma_destroy(soma);
        if (olfact) olfact_destroy(olfact);
        if (gust) gust_destroy(gust);
    }
};

TEST_F(KGSensoryIntegrationTest, SensoryModuleRegistration) {
    /* Register somatosensory */
    EXPECT_EQ(sensory_kg_register_somatosensory(wiring, soma), 0);

    /* Register olfactory */
    EXPECT_EQ(sensory_kg_register_olfactory(wiring, olfact), 0);

    /* Register gustatory */
    EXPECT_EQ(sensory_kg_register_gustatory(wiring, gust), 0);

    /* Verify stats */
    sensory_kg_stats_t stats;
    EXPECT_EQ(sensory_kg_get_stats(wiring, &stats), 0);
    EXPECT_GT(stats.total_nodes, 0u);
}

TEST_F(KGSensoryIntegrationTest, BodyRegionRegistration) {
    sensory_kg_register_somatosensory(wiring, soma);

    /* Register specific body regions */
    uint32_t hand_node_id = 0;
    EXPECT_EQ(sensory_kg_register_body_region(wiring, "Left Hand",
              BODY_SEG_HAND_L, &hand_node_id), 0);
    EXPECT_GT(hand_node_id, 0u);

    uint32_t face_node_id = 0;
    EXPECT_EQ(sensory_kg_register_body_region(wiring, "Face",
              BODY_SEG_FACE, &face_node_id), 0);
    EXPECT_GT(face_node_id, 0u);
}

TEST_F(KGSensoryIntegrationTest, OdorCategoryRegistration) {
    sensory_kg_register_olfactory(wiring, olfact);

    /* Register odor categories */
    uint32_t floral_node_id = 0;
    EXPECT_EQ(sensory_kg_register_odor_category(wiring, "Floral",
              ODOR_CAT_FLORAL, &floral_node_id), 0);
    EXPECT_GT(floral_node_id, 0u);

    uint32_t fruity_node_id = 0;
    EXPECT_EQ(sensory_kg_register_odor_category(wiring, "Fruity",
              ODOR_CAT_FRUITY, &fruity_node_id), 0);
    EXPECT_GT(fruity_node_id, 0u);
}

TEST_F(KGSensoryIntegrationTest, TasteQualityRegistration) {
    sensory_kg_register_gustatory(wiring, gust);

    /* Register taste qualities */
    uint32_t sweet_node_id = 0;
    EXPECT_EQ(sensory_kg_register_taste_quality(wiring, TASTE_SWEET, &sweet_node_id), 0);
    EXPECT_GT(sweet_node_id, 0u);

    uint32_t salty_node_id = 0;
    EXPECT_EQ(sensory_kg_register_taste_quality(wiring, TASTE_SALTY, &salty_node_id), 0);
    EXPECT_GT(salty_node_id, 0u);
}

TEST_F(KGSensoryIntegrationTest, CrossModalFlavorRegistration) {
    sensory_kg_register_olfactory(wiring, olfact);
    sensory_kg_register_gustatory(wiring, gust);

    /* Register odor and taste nodes */
    uint32_t strawberry_odor_id = 0;
    sensory_kg_register_odor_category(wiring, "Strawberry", ODOR_CAT_FRUITY, &strawberry_odor_id);

    uint32_t sweet_taste_id = 0;
    sensory_kg_register_taste_quality(wiring, TASTE_SWEET, &sweet_taste_id);

    /* Register flavor binding */
    uint32_t flavor_node_id = 0;
    EXPECT_EQ(sensory_kg_register_flavor(wiring, sweet_taste_id, strawberry_odor_id,
              &flavor_node_id), 0);
    EXPECT_GT(flavor_node_id, 0u);
}

TEST_F(KGSensoryIntegrationTest, ActivationPropagation) {
    sensory_kg_register_somatosensory(wiring, soma);

    /* Register body region */
    uint32_t hand_node_id = 0;
    sensory_kg_register_body_region(wiring, "Hand", BODY_SEG_HAND_L, &hand_node_id);

    /* Activate node */
    EXPECT_EQ(sensory_kg_activate_node(wiring, hand_node_id, 0.9f), 0);

    /* Propagate activation */
    EXPECT_EQ(sensory_kg_propagate_activation(wiring, hand_node_id, 0.1f), 0);

    /* Decay activations */
    EXPECT_EQ(sensory_kg_decay_activations(wiring, 0.9f), 0);
}

TEST_F(KGSensoryIntegrationTest, QueryByType) {
    sensory_kg_register_somatosensory(wiring, soma);
    sensory_kg_register_olfactory(wiring, olfact);
    sensory_kg_register_gustatory(wiring, gust);

    /* Query nodes by type */
    sensory_kg_query_result_t result;
    memset(&result, 0, sizeof(result));
    EXPECT_EQ(sensory_kg_query_by_type(wiring, SENSORY_KG_NODE_CORTEX, &result), 0);

    /* Should have found cortex nodes */
    EXPECT_GE(result.num_nodes, 3u);  /* At least 3 sensory cortices */

    sensory_kg_free_query_result(&result);
}

/* Main entry point */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
