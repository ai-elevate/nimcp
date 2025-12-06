//=============================================================================
// test_brain_learning_bio_async.cpp - Unit Tests for Brain Learning Bio-Async
//=============================================================================
/**
 * @file test_brain_learning_bio_async.cpp
 * @brief Unit tests for bio-async integration in brain learning module
 *
 * WHAT: Tests bio-async message handling and event broadcasting
 * WHY:  Ensure learning events are correctly published and handled
 * HOW:  Mock bio_ctx, verify message content and channel usage
 *
 * @author NIMCP Development Team
 * @date 2025-12-05
 */

#include <gtest/gtest.h>
extern "C" {
#include "core/brain/nimcp_brain.h"
#include "core/brain/learning/nimcp_brain_learning.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class BrainLearningBioAsyncTest : public ::testing::Test {
protected:
    brain_t brain;
    bio_ctx_t bio_ctx;

    void SetUp() override {
        // Create brain with basic config
        brain_config_t config;
        memset(&config, 0, sizeof(config));
        config.num_inputs = 10;
        config.num_outputs = 3;
        config.num_hidden_neurons = 20;
        config.learning_rate = 0.01f;

        brain = brain_create(&config);
        ASSERT_NE(brain, nullptr);

        // Create bio-async context
        bio_ctx = bio_ctx_create();
        ASSERT_NE(bio_ctx, nullptr);

        // Attach to brain
        brain->bio_ctx = bio_ctx;

        // Register learning module
        ASSERT_TRUE(brain_learning_register_bio_async(brain));
    }

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
        }
        if (bio_ctx) {
            bio_ctx_destroy(bio_ctx);
        }
    }
};

//=============================================================================
// Test: Learning Episode Event Broadcasting
//=============================================================================

TEST_F(BrainLearningBioAsyncTest, LearningEpisodeBroadcastsEvents) {
    // Prepare test data
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    const char* label = "test_class";
    float confidence = 0.9f;

    // Track published messages
    int message_count = 0;
    auto handler = [](bio_ctx_t ctx, bio_msg_t msg, void* user_data) {
        int* count = (int*)user_data;
        (*count)++;

        // Verify message contains expected fields
        EXPECT_NE(bio_msg_get_string(msg, "label", nullptr), nullptr);
        EXPECT_GE(bio_msg_get_float(msg, "confidence", 0.0f), 0.0f);
    };

    bio_ctx_subscribe(bio_ctx, BIO_MSG_TRAINING_STEP, handler, &message_count);
    bio_ctx_subscribe(bio_ctx, BIO_MSG_BRAIN_LEARN_COMPLETE, handler, &message_count);

    // Perform learning
    float loss = brain_learn_example(brain, features, 10, label, confidence);

    // Verify learning succeeded
    ASSERT_GE(loss, 0.0f);

    // Process pending messages
    bio_ctx_process(bio_ctx);

    // Verify events were published (start + complete = 2 messages)
    EXPECT_GE(message_count, 2);
}

//=============================================================================
// Test: Dopamine Channel Usage
//=============================================================================

TEST_F(BrainLearningBioAsyncTest, DopamineChannelForRewardSignals) {
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    bool dopamine_received = false;
    float dopamine_strength = 0.0f;

    auto handler = [](bio_ctx_t ctx, bio_msg_t msg, void* user_data) {
        bool* received = (bool*)user_data;
        *received = true;

        // Check dopamine strength field exists
        float strength = bio_msg_get_float(msg, "dopamine_strength", -1.0f);
        EXPECT_GE(strength, 0.0f);
        EXPECT_LE(strength, 1.0f);
    };

    bio_ctx_subscribe(bio_ctx, BIO_MSG_BRAIN_LEARN_COMPLETE, handler, &dopamine_received);

    // Learn example
    brain_learn_example(brain, features, 10, "good_example", 0.95f);
    bio_ctx_process(bio_ctx);

    EXPECT_TRUE(dopamine_received);
}

//=============================================================================
// Test: Training Start Handler
//=============================================================================

TEST_F(BrainLearningBioAsyncTest, TrainingStartHandlerUpdatesLearningRate) {
    float original_lr = brain->config.learning_rate;
    float new_lr = 0.05f;

    // Send training start message
    bio_msg_t msg = bio_msg_create(BIO_MSG_TRAINING_START);
    ASSERT_NE(msg, nullptr);

    bio_msg_set_string(msg, "mode", "supervised");
    bio_msg_set_float(msg, "learning_rate", new_lr);
    bio_msg_set_uint32(msg, "batch_size", 32);

    bio_ctx_publish(bio_ctx, BIO_CHANNEL_DOPAMINE, msg);
    bio_msg_destroy(msg);

    bio_ctx_process(bio_ctx);

    // Verify learning rate was updated
    EXPECT_NEAR(brain->config.learning_rate, new_lr, 1e-6f);
}

//=============================================================================
// Test: Message Handler Registration
//=============================================================================

TEST_F(BrainLearningBioAsyncTest, AllRequiredHandlersRegistered) {
    // Verify module is registered
    bool is_registered = bio_router_is_registered(bio_ctx, BIO_MODULE_BRAIN_LEARNING);
    EXPECT_TRUE(is_registered);

    // Verify handlers respond to messages
    int handler_calls = 0;
    auto counter = [](bio_ctx_t ctx, bio_msg_t msg, void* user_data) {
        int* count = (int*)user_data;
        (*count)++;
    };

    bio_ctx_subscribe(bio_ctx, BIO_MSG_TRAINING_START, counter, &handler_calls);

    bio_msg_t test_msg = bio_msg_create(BIO_MSG_TRAINING_START);
    bio_ctx_publish(bio_ctx, BIO_CHANNEL_DOPAMINE, test_msg);
    bio_msg_destroy(test_msg);

    bio_ctx_process(bio_ctx);

    EXPECT_GT(handler_calls, 0);
}

//=============================================================================
// Test: Learn Request Message Handling
//=============================================================================

TEST_F(BrainLearningBioAsyncTest, LearnRequestMessageTriggersResponse) {
    bool response_received = false;

    auto response_handler = [](bio_ctx_t ctx, bio_msg_t msg, void* user_data) {
        bool* received = (bool*)user_data;
        *received = true;

        // Verify response contains success flag
        bool success = bio_msg_get_bool(msg, "success", false);
        EXPECT_TRUE(success);
    };

    bio_ctx_subscribe(bio_ctx, BIO_MSG_BRAIN_LEARN_RESPONSE, response_handler, &response_received);

    // Send learn request
    bio_msg_t request = bio_msg_create(BIO_MSG_BRAIN_LEARN_REQUEST);
    bio_msg_set_string(request, "label", "test_label");
    bio_msg_set_float(request, "confidence", 0.8f);

    bio_ctx_publish(bio_ctx, BIO_CHANNEL_DOPAMINE, request);
    bio_msg_destroy(request);

    bio_ctx_process(bio_ctx);

    EXPECT_TRUE(response_received);
}

//=============================================================================
// Main Test Runner
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
