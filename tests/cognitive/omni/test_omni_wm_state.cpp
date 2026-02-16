/**
 * @file test_omni_wm_state.cpp
 * @brief Unit tests for Omni World Model State Management
 * @version 1.0.0
 * @date 2026-02-16
 *
 * Tests for nimcp_omni_wm_state.c module covering:
 * - State lifecycle (create, destroy, clone, from_values)
 * - RSSM state lifecycle
 * - Latent representation lifecycle
 * - Encode/decode operations
 * - Get/set operations
 * - Error handling
 */

#include <gtest/gtest.h>
#include "cognitive/omni/nimcp_omni_world_model.h"

class OmniWMStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize exception system if needed
        // Most tests don't require full world model initialization
    }

    void TearDown() override {
        // Cleanup
    }
};

/* ============================================================================
 * State Lifecycle Tests
 * ============================================================================ */

TEST_F(OmniWMStateTest, StateCreation) {
    omni_wm_state_t* state = omni_wm_state_create(64);
    ASSERT_NE(nullptr, state);
    EXPECT_EQ(64u, state->dim);
    EXPECT_NE(nullptr, state->values);
    EXPECT_FLOAT_EQ(1.0f, state->uncertainty);
    EXPECT_EQ(0u, state->level);
    omni_wm_state_destroy(state);
}

TEST_F(OmniWMStateTest, StateCreationInvalidDim) {
    // Zero dimension
    omni_wm_state_t* state1 = omni_wm_state_create(0);
    EXPECT_EQ(nullptr, state1);

    // Exceeds maximum
    omni_wm_state_t* state2 = omni_wm_state_create(OMNI_WM_MAX_STATE_DIM + 1);
    EXPECT_EQ(nullptr, state2);
}

TEST_F(OmniWMStateTest, StateFromValues) {
    float values[32];
    for (int i = 0; i < 32; i++) {
        values[i] = (float)i * 0.5f;
    }

    omni_wm_state_t* state = omni_wm_state_from_values(values, 32);
    ASSERT_NE(nullptr, state);
    EXPECT_EQ(32u, state->dim);

    // Verify values were copied
    for (int i = 0; i < 32; i++) {
        EXPECT_FLOAT_EQ(values[i], state->values[i]);
    }

    omni_wm_state_destroy(state);
}

TEST_F(OmniWMStateTest, StateFromValuesNullArray) {
    omni_wm_state_t* state = omni_wm_state_from_values(nullptr, 32);
    EXPECT_EQ(nullptr, state);
}

TEST_F(OmniWMStateTest, StateClone) {
    omni_wm_state_t* state = omni_wm_state_create(32);
    ASSERT_NE(nullptr, state);

    // Set up state with specific values
    state->values[0] = 42.0f;
    state->values[31] = 99.0f;
    state->uncertainty = 0.5f;
    state->timestamp = 123.456;
    state->level = 2;

    // Clone it
    omni_wm_state_t* clone = omni_wm_state_clone(state);
    ASSERT_NE(nullptr, clone);
    EXPECT_EQ(state->dim, clone->dim);
    EXPECT_FLOAT_EQ(42.0f, clone->values[0]);
    EXPECT_FLOAT_EQ(99.0f, clone->values[31]);
    EXPECT_FLOAT_EQ(0.5f, clone->uncertainty);
    EXPECT_DOUBLE_EQ(123.456, clone->timestamp);
    EXPECT_EQ(2u, clone->level);

    // Verify independence (modify original, clone unchanged)
    state->values[0] = 1000.0f;
    EXPECT_FLOAT_EQ(42.0f, clone->values[0]);

    omni_wm_state_destroy(clone);
    omni_wm_state_destroy(state);
}

TEST_F(OmniWMStateTest, StateCloneNull) {
    omni_wm_state_t* clone = omni_wm_state_clone(nullptr);
    EXPECT_EQ(nullptr, clone);
}

TEST_F(OmniWMStateTest, StateDestroyNull) {
    // Should not crash
    omni_wm_state_destroy(nullptr);
}

/* ============================================================================
 * RSSM State Lifecycle Tests
 * ============================================================================ */

TEST_F(OmniWMStateTest, RSSMStateCreation) {
    omni_wm_rssm_state_t* state = omni_wm_rssm_state_create(64, 32);
    ASSERT_NE(nullptr, state);
    EXPECT_EQ(64u, state->h_dim);
    EXPECT_EQ(32u, state->z_dim);
    EXPECT_NE(nullptr, state->h);
    EXPECT_NE(nullptr, state->z);
    EXPECT_NE(nullptr, state->z_mean);
    EXPECT_NE(nullptr, state->z_std);

    // Check std initialized to 1.0
    for (uint32_t i = 0; i < state->z_dim; i++) {
        EXPECT_FLOAT_EQ(1.0f, state->z_std[i]);
    }

    omni_wm_rssm_state_destroy(state);
}

TEST_F(OmniWMStateTest, RSSMStateCreationInvalidDims) {
    omni_wm_rssm_state_t* state1 = omni_wm_rssm_state_create(0, 32);
    EXPECT_EQ(nullptr, state1);

    omni_wm_rssm_state_t* state2 = omni_wm_rssm_state_create(64, 0);
    EXPECT_EQ(nullptr, state2);
}

TEST_F(OmniWMStateTest, RSSMStateClone) {
    omni_wm_rssm_state_t* state = omni_wm_rssm_state_create(64, 32);
    ASSERT_NE(nullptr, state);

    // Set specific values
    state->h[0] = 10.0f;
    state->h[63] = 20.0f;
    state->z[0] = 30.0f;
    state->z[31] = 40.0f;
    state->z_mean[0] = 5.0f;
    state->z_std[0] = 2.0f;
    state->timestamp = 999.0;

    // Clone it
    omni_wm_rssm_state_t* clone = omni_wm_rssm_state_clone(state);
    ASSERT_NE(nullptr, clone);
    EXPECT_EQ(64u, clone->h_dim);
    EXPECT_EQ(32u, clone->z_dim);
    EXPECT_FLOAT_EQ(10.0f, clone->h[0]);
    EXPECT_FLOAT_EQ(20.0f, clone->h[63]);
    EXPECT_FLOAT_EQ(30.0f, clone->z[0]);
    EXPECT_FLOAT_EQ(40.0f, clone->z[31]);
    EXPECT_FLOAT_EQ(5.0f, clone->z_mean[0]);
    EXPECT_FLOAT_EQ(2.0f, clone->z_std[0]);
    EXPECT_DOUBLE_EQ(999.0, clone->timestamp);

    // Verify independence
    state->h[0] = 100.0f;
    EXPECT_FLOAT_EQ(10.0f, clone->h[0]);

    omni_wm_rssm_state_destroy(clone);
    omni_wm_rssm_state_destroy(state);
}

TEST_F(OmniWMStateTest, RSSMStateCloneNull) {
    omni_wm_rssm_state_t* clone = omni_wm_rssm_state_clone(nullptr);
    EXPECT_EQ(nullptr, clone);
}

TEST_F(OmniWMStateTest, RSSMStateDestroyNull) {
    // Should not crash
    omni_wm_rssm_state_destroy(nullptr);
}

/* ============================================================================
 * Latent Representation Tests
 * ============================================================================ */

TEST_F(OmniWMStateTest, LatentCreation) {
    omni_wm_latent_t* latent = omni_wm_latent_create(64);
    ASSERT_NE(nullptr, latent);
    EXPECT_EQ(64u, latent->dim);
    EXPECT_NE(nullptr, latent->embedding);
    omni_wm_latent_destroy(latent);
}

TEST_F(OmniWMStateTest, LatentCreationInvalidDim) {
    omni_wm_latent_t* latent1 = omni_wm_latent_create(0);
    EXPECT_EQ(nullptr, latent1);

    omni_wm_latent_t* latent2 = omni_wm_latent_create(OMNI_WM_MAX_LATENT_DIM + 1);
    EXPECT_EQ(nullptr, latent2);
}

TEST_F(OmniWMStateTest, LatentDestroyNull) {
    // Should not crash
    omni_wm_latent_destroy(nullptr);
}

TEST_F(OmniWMStateTest, EncodeDecodeBasic) {
    // Create a world model for encode/decode
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    config.obs_dim = 64;
    config.latent_dim = 32;

    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    // Create observation
    float observation[64];
    for (int i = 0; i < 64; i++) {
        observation[i] = (float)i * 0.1f;
    }

    // Create latent
    omni_wm_latent_t* latent = omni_wm_latent_create(32);
    ASSERT_NE(nullptr, latent);

    // Encode
    nimcp_error_t err = omni_wm_encode(wm, observation, 64, latent);
    EXPECT_EQ(NIMCP_SUCCESS, err);

    // Check that embedding was populated (values should be non-zero due to encoding)
    bool has_nonzero = false;
    for (uint32_t i = 0; i < latent->dim; i++) {
        if (latent->embedding[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);

    // Decode back
    float decoded[64];
    err = omni_wm_decode(wm, latent, decoded, 64);
    EXPECT_EQ(NIMCP_SUCCESS, err);

    // Decoded should have values (though not necessarily matching original)
    bool decoded_has_values = false;
    for (int i = 0; i < 64; i++) {
        if (decoded[i] != 0.0f) {
            decoded_has_values = true;
            break;
        }
    }
    EXPECT_TRUE(decoded_has_values);

    omni_wm_latent_destroy(latent);
    omni_wm_destroy(wm);
}

TEST_F(OmniWMStateTest, EncodeNullArgs) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    float observation[64];
    omni_wm_latent_t* latent = omni_wm_latent_create(32);
    ASSERT_NE(nullptr, latent);

    // Null world model
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_encode(nullptr, observation, 64, latent));

    // Null observation
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_encode(wm, nullptr, 64, latent));

    // Null latent
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_encode(wm, observation, 64, nullptr));

    omni_wm_latent_destroy(latent);
    omni_wm_destroy(wm);
}

TEST_F(OmniWMStateTest, DecodeNullArgs) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    float observation[64];
    omni_wm_latent_t* latent = omni_wm_latent_create(32);
    ASSERT_NE(nullptr, latent);

    // Null world model
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_decode(nullptr, latent, observation, 64));

    // Null latent
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_decode(wm, nullptr, observation, 64));

    // Null observation
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_decode(wm, latent, nullptr, 64));

    omni_wm_latent_destroy(latent);
    omni_wm_destroy(wm);
}

/* ============================================================================
 * Get/Set State Tests
 * ============================================================================ */

TEST_F(OmniWMStateTest, SetGetState) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    // Create a state
    omni_wm_state_t* state = omni_wm_state_create(64);
    ASSERT_NE(nullptr, state);
    state->values[0] = 123.0f;
    state->uncertainty = 0.3f;

    // Set it
    nimcp_error_t err = omni_wm_set_state(wm, state);
    EXPECT_EQ(NIMCP_SUCCESS, err);

    // Get it back
    const omni_wm_state_t* retrieved = omni_wm_get_state(wm);
    ASSERT_NE(nullptr, retrieved);
    EXPECT_FLOAT_EQ(123.0f, retrieved->values[0]);
    EXPECT_FLOAT_EQ(0.3f, retrieved->uncertainty);

    // Modify original - retrieved should be independent
    state->values[0] = 999.0f;
    EXPECT_FLOAT_EQ(123.0f, retrieved->values[0]);

    omni_wm_state_destroy(state);
    omni_wm_destroy(wm);
}

TEST_F(OmniWMStateTest, SetStateNullArgs) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    omni_wm_state_t* state = omni_wm_state_create(64);
    ASSERT_NE(nullptr, state);

    // Null world model
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_set_state(nullptr, state));

    // Null state
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_set_state(wm, nullptr));

    omni_wm_state_destroy(state);
    omni_wm_destroy(wm);
}

TEST_F(OmniWMStateTest, GetStateNull) {
    const omni_wm_state_t* state = omni_wm_get_state(nullptr);
    EXPECT_EQ(nullptr, state);
}

TEST_F(OmniWMStateTest, SetGetRSSMState) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    // Create RSSM state
    omni_wm_rssm_state_t* state = omni_wm_rssm_state_create(64, 32);
    ASSERT_NE(nullptr, state);
    state->h[0] = 55.0f;
    state->z[0] = 66.0f;
    state->timestamp = 777.0;

    // Set it
    nimcp_error_t err = omni_wm_set_rssm_state(wm, state);
    EXPECT_EQ(NIMCP_SUCCESS, err);

    // Get it back
    const omni_wm_rssm_state_t* retrieved = omni_wm_get_rssm_state(wm);
    ASSERT_NE(nullptr, retrieved);
    EXPECT_FLOAT_EQ(55.0f, retrieved->h[0]);
    EXPECT_FLOAT_EQ(66.0f, retrieved->z[0]);
    EXPECT_DOUBLE_EQ(777.0, retrieved->timestamp);

    // Modify original - retrieved should be independent
    state->h[0] = 999.0f;
    EXPECT_FLOAT_EQ(55.0f, retrieved->h[0]);

    omni_wm_rssm_state_destroy(state);
    omni_wm_destroy(wm);
}

TEST_F(OmniWMStateTest, SetRSSMStateNullArgs) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    omni_wm_rssm_state_t* state = omni_wm_rssm_state_create(64, 32);
    ASSERT_NE(nullptr, state);

    // Null world model
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_set_rssm_state(nullptr, state));

    // Null state
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_set_rssm_state(wm, nullptr));

    omni_wm_rssm_state_destroy(state);
    omni_wm_destroy(wm);
}

TEST_F(OmniWMStateTest, GetRSSMStateNull) {
    const omni_wm_rssm_state_t* state = omni_wm_get_rssm_state(nullptr);
    EXPECT_EQ(nullptr, state);
}

/* ============================================================================
 * Predict Latent Test
 * ============================================================================ */

TEST_F(OmniWMStateTest, PredictLatentBasic) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    omni_wm_latent_t* latent = omni_wm_latent_create(32);
    ASSERT_NE(nullptr, latent);
    latent->embedding[0] = 1.0f;
    latent->information_content = 5.0f;

    omni_wm_latent_t* predicted = omni_wm_latent_create(32);
    ASSERT_NE(nullptr, predicted);

    float action[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    nimcp_error_t err = omni_wm_predict_latent(wm, latent, action, 8, predicted);
    EXPECT_EQ(NIMCP_SUCCESS, err);

    // Prediction should have modified the first few values (action influence)
    EXPECT_NE(1.0f, predicted->embedding[0]);
    EXPECT_FLOAT_EQ(5.0f, predicted->information_content);

    omni_wm_latent_destroy(predicted);
    omni_wm_latent_destroy(latent);
    omni_wm_destroy(wm);
}

TEST_F(OmniWMStateTest, PredictLatentNullArgs) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    omni_world_model_t* wm = omni_wm_create(&config);
    ASSERT_NE(nullptr, wm);

    omni_wm_latent_t* latent = omni_wm_latent_create(32);
    ASSERT_NE(nullptr, latent);
    omni_wm_latent_t* predicted = omni_wm_latent_create(32);
    ASSERT_NE(nullptr, predicted);
    float action[8] = {0};

    // Null world model
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_predict_latent(nullptr, latent, action, 8, predicted));

    // Null latent
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_predict_latent(wm, nullptr, action, 8, predicted));

    // Null action
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_predict_latent(wm, latent, nullptr, 8, predicted));

    // Null predicted
    EXPECT_NE(NIMCP_SUCCESS, omni_wm_predict_latent(wm, latent, action, 8, nullptr));

    omni_wm_latent_destroy(predicted);
    omni_wm_latent_destroy(latent);
    omni_wm_destroy(wm);
}
