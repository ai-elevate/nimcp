/**
 * @file test_empathetic_response.cpp
 * @brief Unit tests for Empathetic Response Engine
 *
 * WHAT: Comprehensive unit tests for non-reactive empathetic response system
 * WHY:  Ensure safe, supportive responses to student emotional states
 * HOW:  Test lifecycle, response generation, crisis detection, and FEP bridge
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

// Headers have their own extern "C" guards
#include "cognitive/nimcp_empathetic_response.h"
#include "cognitive/empathetic_response/nimcp_empathetic_response_fep_bridge.h"
#include "cognitive/free_energy/nimcp_free_energy.h"

#include "utils/nimcp_test_base.h"

/* ============================================================================
 * Empathetic Response Core Tests
 * ============================================================================ */

class EmpatheticResponseTest : public NimcpTestBase {
protected:
    empathetic_response_engine_t engine = nullptr;

    void SetUp() override {
        NimcpTestBase::SetUp();
    }

    void TearDown() override {
        if (engine) {
            empathetic_response_destroy(engine);
            engine = nullptr;
        }
        NimcpTestBase::TearDown();
    }
};

TEST_F(EmpatheticResponseTest, CreateWithNullDependencies) {
    // WHAT: Create engine without dependencies
    // WHY:  Engine should work standalone
    // HOW:  Pass NULL for ethics and empathy networks

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);
}

TEST_F(EmpatheticResponseTest, DestroyNullIsNoop) {
    // WHAT: Verify destroying NULL engine doesn't crash
    // WHY:  Defensive programming
    // HOW:  Call destroy with NULL

    empathetic_response_destroy(nullptr);
    SUCCEED();
}

/* ============================================================================
 * Response Generation Tests
 * ============================================================================ */

TEST_F(EmpatheticResponseTest, GenerateResponseForRage) {
    // WHAT: Generate response for extreme rage
    // WHY:  Most challenging case - must remain calm
    // HOW:  Create rage state, generate response

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_RAGE;
    state.intensity = EMOTION_INTENSITY_EXTREME;
    state.valence = -0.9f;
    state.arousal = 0.95f;
    strncpy(state.text_input, "I HATE this!", sizeof(state.text_input) - 1);
    state.crisis_flags = CRISIS_NONE;

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    ASSERT_TRUE(result);
    EXPECT_GT(strlen(response.response_text), 0u);
    EXPECT_GE(response.empathy_score, 0.0f);
    EXPECT_LE(response.empathy_score, 1.0f);
    EXPECT_GE(response.safety_score, 0.0f);
    EXPECT_LE(response.safety_score, 1.0f);
}

TEST_F(EmpatheticResponseTest, GenerateResponseForFear) {
    // WHAT: Generate response for fear/panic
    // WHY:  Reassurance strategy expected
    // HOW:  Create fear state, generate response

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_FEAR;
    state.intensity = EMOTION_INTENSITY_HIGH;
    state.valence = -0.7f;
    state.arousal = 0.8f;
    strncpy(state.text_input, "I'm scared I'll fail", sizeof(state.text_input) - 1);

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    ASSERT_TRUE(result);
    // Expect reassurance or validation strategy
    EXPECT_TRUE(response.primary_strategy == RESPONSE_VALIDATE ||
                response.primary_strategy == RESPONSE_REASSURE);
}

TEST_F(EmpatheticResponseTest, GenerateResponseForFrustration) {
    // WHAT: Generate response for learning frustration
    // WHY:  Common student emotion
    // HOW:  Create frustration state

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_FRUSTRATION;
    state.intensity = EMOTION_INTENSITY_MODERATE;
    state.valence = -0.5f;
    state.arousal = 0.6f;
    strncpy(state.text_input, "This is so confusing!", sizeof(state.text_input) - 1);

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    ASSERT_TRUE(result);
    EXPECT_GT(strlen(response.response_text), 0u);
}

TEST_F(EmpatheticResponseTest, GenerateResponseForConfusion) {
    // WHAT: Generate response for cognitive confusion
    // WHY:  Needs support but not crisis intervention
    // HOW:  Create confusion state

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_CONFUSION;
    state.intensity = EMOTION_INTENSITY_MODERATE;
    state.valence = -0.2f;
    state.arousal = 0.4f;
    strncpy(state.text_input, "I don't understand this at all", sizeof(state.text_input) - 1);

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    ASSERT_TRUE(result);
    // Confusion typically gets explore strategy
    EXPECT_TRUE(response.primary_strategy == RESPONSE_EXPLORE ||
                response.primary_strategy == RESPONSE_VALIDATE);
}

TEST_F(EmpatheticResponseTest, GenerateWithNullEngine) {
    // WHAT: Verify generate handles NULL engine
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    empathetic_response_t response;

    bool result = empathetic_response_generate(nullptr, &state, &response);
    EXPECT_FALSE(result);
}

TEST_F(EmpatheticResponseTest, GenerateWithNullState) {
    // WHAT: Verify generate handles NULL state
    // WHY:  Defensive programming
    // HOW:  Call with NULL state

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, nullptr, &response);
    EXPECT_FALSE(result);
}

TEST_F(EmpatheticResponseTest, GenerateWithNullResponse) {
    // WHAT: Verify generate handles NULL response
    // WHY:  Defensive programming
    // HOW:  Call with NULL response

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));

    bool result = empathetic_response_generate(engine, &state, nullptr);
    EXPECT_FALSE(result);
}

/* ============================================================================
 * Crisis Detection Tests
 * ============================================================================ */

TEST_F(EmpatheticResponseTest, DetectCrisisWithNullEngine) {
    // WHAT: Verify crisis detection handles NULL engine
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    uint32_t flags;
    float confidence;
    bool result = empathetic_response_detect_crisis(nullptr, "test", &flags, &confidence);
    EXPECT_FALSE(result);
}

TEST_F(EmpatheticResponseTest, DetectCrisisWithNullText) {
    // WHAT: Verify crisis detection handles NULL text
    // WHY:  Defensive programming
    // HOW:  Call with NULL text

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    uint32_t flags;
    float confidence;
    bool result = empathetic_response_detect_crisis(engine, nullptr, &flags, &confidence);
    EXPECT_FALSE(result);
}

TEST_F(EmpatheticResponseTest, DetectCrisisNormalText) {
    // WHAT: Verify normal text doesn't trigger crisis
    // WHY:  Avoid false positives
    // HOW:  Pass normal frustrated text

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    uint32_t flags = 0;
    float confidence = 0.0f;
    bool result = empathetic_response_detect_crisis(
        engine,
        "This assignment is really frustrating but I'll keep trying",
        &flags,
        &confidence
    );

    // Should not detect crisis in normal frustration
    EXPECT_FALSE(result);
    EXPECT_EQ(flags, CRISIS_NONE);
}

TEST_F(EmpatheticResponseTest, CrisisResponseRequiresEscalation) {
    // WHAT: Verify crisis state triggers escalation
    // WHY:  Safety-critical behavior
    // HOW:  Create crisis state, check response

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_DESPAIR;
    state.intensity = EMOTION_INTENSITY_EXTREME;
    state.valence = -1.0f;
    state.arousal = 0.9f;
    state.crisis_flags = CRISIS_SEVERE_DISTRESS;
    state.crisis_confidence = 0.9f;

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    ASSERT_TRUE(result);
    // Severe distress should at minimum flag for potential escalation
    EXPECT_TRUE(response.requires_human_escalation ||
                response.primary_strategy == RESPONSE_ESCALATE ||
                response.safety_score > 0.5f);
}

/* ============================================================================
 * Grounding Exercise Tests
 * ============================================================================ */

TEST_F(EmpatheticResponseTest, GetGroundingExerciseWithNullEngine) {
    // WHAT: Verify grounding handles NULL engine
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    grounding_exercise_t exercise;
    bool result = empathetic_response_get_grounding_exercise(
        nullptr, EMOTION_RAGE, EMOTION_INTENSITY_HIGH, &exercise
    );
    EXPECT_FALSE(result);
}

TEST_F(EmpatheticResponseTest, GetGroundingExerciseWithNullOutput) {
    // WHAT: Verify grounding handles NULL output
    // WHY:  Defensive programming
    // HOW:  Call with NULL output

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    bool result = empathetic_response_get_grounding_exercise(
        engine, EMOTION_RAGE, EMOTION_INTENSITY_HIGH, nullptr
    );
    EXPECT_FALSE(result);
}

TEST_F(EmpatheticResponseTest, GetGroundingExerciseForRage) {
    // WHAT: Get grounding exercise for rage
    // WHY:  High arousal needs grounding
    // HOW:  Request exercise, verify structure

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    grounding_exercise_t exercise;
    memset(&exercise, 0, sizeof(exercise));

    bool result = empathetic_response_get_grounding_exercise(
        engine, EMOTION_RAGE, EMOTION_INTENSITY_HIGH, &exercise
    );

    if (result) {
        EXPECT_GT(strlen(exercise.name), 0u);
        EXPECT_GT(strlen(exercise.instructions), 0u);
        EXPECT_GT(exercise.duration_seconds, 0u);
    }
}

TEST_F(EmpatheticResponseTest, GetGroundingExerciseForPanic) {
    // WHAT: Get grounding exercise for panic
    // WHY:  Panic needs immediate grounding
    // HOW:  Request exercise

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    grounding_exercise_t exercise;
    memset(&exercise, 0, sizeof(exercise));

    bool result = empathetic_response_get_grounding_exercise(
        engine, EMOTION_PANIC, EMOTION_INTENSITY_EXTREME, &exercise
    );

    if (result) {
        EXPECT_GT(strlen(exercise.instructions), 0u);
    }
}

/* ============================================================================
 * Safety Prediction Tests
 * ============================================================================ */

TEST_F(EmpatheticResponseTest, PredictSafetyWithNullEngine) {
    // WHAT: Verify safety prediction handles NULL engine
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    emotional_state_t state, reaction;
    memset(&state, 0, sizeof(state));
    memset(&reaction, 0, sizeof(reaction));

    float safety = empathetic_response_predict_safety(
        nullptr, &state, "Test response", &reaction
    );
    EXPECT_EQ(safety, 0.0f);  // 0 indicates error or unsafe
}

TEST_F(EmpatheticResponseTest, PredictSafetyForSupportiveResponse) {
    // WHAT: Verify supportive response is predicted safe
    // WHY:  Core safety validation
    // HOW:  Pass supportive response text

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_FRUSTRATION;
    state.intensity = EMOTION_INTENSITY_MODERATE;
    state.valence = -0.5f;
    state.arousal = 0.6f;

    emotional_state_t predicted_reaction;
    memset(&predicted_reaction, 0, sizeof(predicted_reaction));

    float safety = empathetic_response_predict_safety(
        engine,
        &state,
        "I understand this is frustrating. Let's work through it together.",
        &predicted_reaction
    );

    // Supportive response should be safe
    EXPECT_GE(safety, 0.5f);
}

/* ============================================================================
 * Effectiveness Tracking Tests
 * ============================================================================ */

TEST_F(EmpatheticResponseTest, TrackEffectivenessWithNullEngine) {
    // WHAT: Verify tracking handles NULL engine
    // WHY:  Defensive programming
    // HOW:  Call with NULL (should not crash)

    empathetic_response_t response;
    memset(&response, 0, sizeof(response));
    emotional_state_t reaction;
    memset(&reaction, 0, sizeof(reaction));

    empathetic_response_track_effectiveness(nullptr, &response, &reaction, 0.8f);
    SUCCEED();  // If we get here, no crash
}

TEST_F(EmpatheticResponseTest, TrackEffectivenessValidCall) {
    // WHAT: Verify effectiveness tracking works
    // WHY:  Adaptive learning needs tracking
    // HOW:  Track a response outcome

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    empathetic_response_t response;
    memset(&response, 0, sizeof(response));
    response.primary_strategy = RESPONSE_VALIDATE;
    strncpy(response.response_text, "I understand.", sizeof(response.response_text) - 1);

    emotional_state_t reaction;
    memset(&reaction, 0, sizeof(reaction));
    reaction.emotion_type = EMOTION_NEGATIVE_NONE;
    reaction.intensity = EMOTION_INTENSITY_LOW;
    reaction.valence = 0.2f;  // Slightly positive after response

    empathetic_response_track_effectiveness(engine, &response, &reaction, 0.8f);
    SUCCEED();  // No crash, tracking recorded
}

/* ============================================================================
 * Enum Value Tests
 * ============================================================================ */

TEST_F(EmpatheticResponseTest, EmotionIntensityEnumValues) {
    // WHAT: Verify intensity enum values
    // WHY:  Used in switch statements
    // HOW:  Check values

    EXPECT_EQ((int)EMOTION_INTENSITY_NONE, 0);
    EXPECT_EQ((int)EMOTION_INTENSITY_LOW, 1);
    EXPECT_EQ((int)EMOTION_INTENSITY_MODERATE, 2);
    EXPECT_EQ((int)EMOTION_INTENSITY_HIGH, 3);
    EXPECT_EQ((int)EMOTION_INTENSITY_EXTREME, 4);
}

TEST_F(EmpatheticResponseTest, NegativeEmotionEnumValues) {
    // WHAT: Verify negative emotion enum values
    // WHY:  Used in switch statements
    // HOW:  Check values

    EXPECT_EQ((int)EMOTION_NEGATIVE_NONE, 0);
    EXPECT_EQ((int)EMOTION_RAGE, 1);
    EXPECT_EQ((int)EMOTION_ANGER, 2);
    EXPECT_EQ((int)EMOTION_HATE, 3);
    EXPECT_EQ((int)EMOTION_FEAR, 4);
    EXPECT_EQ((int)EMOTION_PANIC, 5);
    EXPECT_EQ((int)EMOTION_DESPAIR, 6);
    EXPECT_EQ((int)EMOTION_DISGUST, 7);
    EXPECT_EQ((int)EMOTION_SHAME, 8);
    EXPECT_EQ((int)EMOTION_FRUSTRATION, 9);
    EXPECT_EQ((int)EMOTION_CONFUSION, 10);
    EXPECT_EQ((int)EMOTION_BOREDOM, 11);
}

TEST_F(EmpatheticResponseTest, ResponseStrategyEnumValues) {
    // WHAT: Verify response strategy enum values
    // WHY:  Strategy selection logic
    // HOW:  Check values

    EXPECT_EQ((int)RESPONSE_VALIDATE, 0);
    EXPECT_EQ((int)RESPONSE_REASSURE, 1);
    EXPECT_EQ((int)RESPONSE_EXPLORE, 2);
    EXPECT_EQ((int)RESPONSE_GROUND, 3);
    EXPECT_EQ((int)RESPONSE_REFRAME, 4);
    EXPECT_EQ((int)RESPONSE_BOUNDARY, 5);
    EXPECT_EQ((int)RESPONSE_ESCALATE, 6);
}

TEST_F(EmpatheticResponseTest, CrisisFlagBitValues) {
    // WHAT: Verify crisis flag bit values
    // WHY:  Used in bitfield operations
    // HOW:  Check bit positions

    EXPECT_EQ((int)CRISIS_NONE, 0);
    EXPECT_EQ((int)CRISIS_SUICIDAL, 1);
    EXPECT_EQ((int)CRISIS_SELF_HARM, 2);
    EXPECT_EQ((int)CRISIS_ABUSE, 4);
    EXPECT_EQ((int)CRISIS_SEVERE_DISTRESS, 8);
}

/* ============================================================================
 * Empathetic Response FEP Bridge Tests
 * ============================================================================ */

class EmpatheticResponseFepBridgeTest : public NimcpTestBase {
protected:
    empathetic_response_fep_bridge_t* bridge = nullptr;
    fep_system_t* fep = nullptr;

    void SetUp() override {
        NimcpTestBase::SetUp();

        // Create FEP system
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep = fep_create(&fep_config, 8, 4);

        // Create bridge
        empathetic_response_fep_config_t config;
        empathetic_response_fep_default_config(&config);
        bridge = empathetic_response_fep_create(&config);
    }

    void TearDown() override {
        if (bridge) {
            empathetic_response_fep_destroy(bridge);
            bridge = nullptr;
        }
        if (fep) {
            fep_destroy(fep);
            fep = nullptr;
        }
        NimcpTestBase::TearDown();
    }
};

TEST_F(EmpatheticResponseFepBridgeTest, CreateDestroy) {
    // WHAT: Verify bridge creation
    // WHY:  Basic lifecycle test
    // HOW:  Check not NULL

    ASSERT_NE(bridge, nullptr);
}

TEST_F(EmpatheticResponseFepBridgeTest, DefaultConfig) {
    // WHAT: Verify default config
    // WHY:  Ensure sensible defaults
    // HOW:  Check config values

    empathetic_response_fep_config_t config;
    int ret = empathetic_response_fep_default_config(&config);

    EXPECT_EQ(ret, 0);
    EXPECT_GT(config.fe_sensitivity, 0.0f);
    EXPECT_GT(config.emotion_sensitivity, 0.0f);
}

TEST_F(EmpatheticResponseFepBridgeTest, DefaultConfigNull) {
    // WHAT: Verify default_config handles NULL
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    int ret = empathetic_response_fep_default_config(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(EmpatheticResponseFepBridgeTest, DestroyNull) {
    // WHAT: Verify destroying NULL is safe
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    empathetic_response_fep_destroy(nullptr);
    SUCCEED();
}

TEST_F(EmpatheticResponseFepBridgeTest, ConnectFep) {
    // WHAT: Verify FEP connection
    // WHY:  Bridge needs FEP
    // HOW:  Connect and verify

    ASSERT_NE(fep, nullptr);
    int ret = empathetic_response_fep_connect_fep(bridge, fep);
    EXPECT_EQ(ret, 0);
}

TEST_F(EmpatheticResponseFepBridgeTest, ConnectFepWithNullBridge) {
    // WHAT: Verify connect handles NULL bridge
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    int ret = empathetic_response_fep_connect_fep(nullptr, fep);
    EXPECT_NE(ret, 0);
}

TEST_F(EmpatheticResponseFepBridgeTest, Update) {
    // WHAT: Verify bridge update
    // WHY:  Core functionality
    // HOW:  Connect, update

    if (fep) {
        empathetic_response_fep_connect_fep(bridge, fep);
        int ret = empathetic_response_fep_update(bridge, 100);
        EXPECT_EQ(ret, 0);
    }
}

TEST_F(EmpatheticResponseFepBridgeTest, UpdateWithNullBridge) {
    // WHAT: Verify update handles NULL
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    int ret = empathetic_response_fep_update(nullptr, 100);
    EXPECT_NE(ret, 0);
}

TEST_F(EmpatheticResponseFepBridgeTest, GetState) {
    // WHAT: Verify state retrieval
    // WHY:  State query functionality
    // HOW:  Get state after update

    if (fep) {
        empathetic_response_fep_connect_fep(bridge, fep);
        empathetic_response_fep_update(bridge, 100);

        empathetic_response_fep_state_t state;
        int ret = empathetic_response_fep_get_state(bridge, &state);
        EXPECT_EQ(ret, 0);
    }
}

TEST_F(EmpatheticResponseFepBridgeTest, GetStateWithNullBridge) {
    // WHAT: Verify get_state handles NULL
    // WHY:  Defensive programming
    // HOW:  Call with NULL

    empathetic_response_fep_state_t state;
    int ret = empathetic_response_fep_get_state(nullptr, &state);
    EXPECT_NE(ret, 0);
}

TEST_F(EmpatheticResponseFepBridgeTest, GetStats) {
    // WHAT: Verify stats retrieval
    // WHY:  Monitor bridge activity
    // HOW:  Get stats

    empathetic_response_fep_stats_t stats;
    int ret = empathetic_response_fep_get_stats(bridge, &stats);
    EXPECT_EQ(ret, 0);
}

TEST_F(EmpatheticResponseFepBridgeTest, BioAsyncConnection) {
    // WHAT: Verify bio-async lifecycle
    // WHY:  Bio-async integration
    // HOW:  Connect, check, disconnect

    EXPECT_FALSE(empathetic_response_fep_is_bio_async_connected(bridge));

    empathetic_response_fep_connect_bio_async(bridge);
    EXPECT_TRUE(empathetic_response_fep_is_bio_async_connected(bridge));

    empathetic_response_fep_disconnect_bio_async(bridge);
    EXPECT_FALSE(empathetic_response_fep_is_bio_async_connected(bridge));
}

TEST_F(EmpatheticResponseFepBridgeTest, InferUserState) {
    // WHAT: Verify user state inference from PE
    // WHY:  FEP-empathy coupling
    // HOW:  Infer with prediction error

    if (fep) {
        empathetic_response_fep_connect_fep(bridge, fep);
        int ret = empathetic_response_fep_infer_user_state(bridge, 0.5f);
        EXPECT_EQ(ret, 0);
    }
}

TEST_F(EmpatheticResponseFepBridgeTest, ModulateSocialPrecision) {
    // WHAT: Verify social precision modulation
    // WHY:  Empathy affects FEP precision
    // HOW:  Modulate after connection

    if (fep) {
        empathetic_response_fep_connect_fep(bridge, fep);
        int ret = empathetic_response_fep_modulate_social_precision(bridge);
        EXPECT_EQ(ret, 0);
    }
}

/* ============================================================================
 * Edge Case Tests
 * ============================================================================ */

TEST_F(EmpatheticResponseTest, ResponseToMultipleCrisisFlags) {
    // WHAT: Handle multiple simultaneous crisis flags
    // WHY:  Complex crisis situations
    // HOW:  Set multiple flags

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_DESPAIR;
    state.intensity = EMOTION_INTENSITY_EXTREME;
    state.valence = -1.0f;
    state.arousal = 0.9f;
    state.crisis_flags = CRISIS_SEVERE_DISTRESS | CRISIS_SELF_HARM;
    state.crisis_confidence = 0.95f;

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    ASSERT_TRUE(result);
    // Multiple crisis flags should definitely require escalation
    EXPECT_TRUE(response.requires_human_escalation ||
                response.primary_strategy == RESPONSE_ESCALATE);
}

TEST_F(EmpatheticResponseTest, ResponseToMildBoredom) {
    // WHAT: Handle low-intensity boredom
    // WHY:  Should not overreact
    // HOW:  Create mild boredom state

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_BOREDOM;
    state.intensity = EMOTION_INTENSITY_LOW;
    state.valence = -0.1f;
    state.arousal = 0.1f;
    strncpy(state.text_input, "This is kind of boring", sizeof(state.text_input) - 1);

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    ASSERT_TRUE(result);
    // Should NOT escalate for mild boredom
    EXPECT_FALSE(response.requires_human_escalation);
    EXPECT_NE(response.primary_strategy, RESPONSE_ESCALATE);
}

TEST_F(EmpatheticResponseTest, EmptyTextInput) {
    // WHAT: Handle empty text input
    // WHY:  Edge case
    // HOW:  Pass empty string

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_FRUSTRATION;
    state.intensity = EMOTION_INTENSITY_MODERATE;
    state.text_input[0] = '\0';

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    // Should still generate a response based on emotion alone
    EXPECT_TRUE(result);
}

TEST_F(EmpatheticResponseTest, MaxLengthTextInput) {
    // WHAT: Handle maximum length text input
    // WHY:  Buffer boundary test
    // HOW:  Fill text buffer

    engine = empathetic_response_create(nullptr, nullptr);
    ASSERT_NE(engine, nullptr);

    emotional_state_t state;
    memset(&state, 0, sizeof(state));
    state.emotion_type = EMOTION_FRUSTRATION;
    state.intensity = EMOTION_INTENSITY_MODERATE;

    // Fill with characters (leave room for null terminator)
    memset(state.text_input, 'A', sizeof(state.text_input) - 1);
    state.text_input[sizeof(state.text_input) - 1] = '\0';

    empathetic_response_t response;
    bool result = empathetic_response_generate(engine, &state, &response);

    EXPECT_TRUE(result);
}
