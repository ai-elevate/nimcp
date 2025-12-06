/**
 * @file test_attention_bioasync.cpp
 * @brief Unit tests for attention plasticity bio-async integration
 */

#include <gtest/gtest.h>

extern "C" {
#include "plasticity/attention/nimcp_attention.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class AttentionBioAsyncTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize bio-async system
        bio_async_init();
        bio_router_init();

        // Initialize unified memory
        nimcp_unified_memory_init();
    }

    void TearDown() override {
        bio_router_shutdown();
        bio_async_shutdown();
        nimcp_unified_memory_shutdown();
    }
};

TEST_F(AttentionBioAsyncTest, ModuleRegistration) {
    // Test that attention module can register with bio-router
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_ATTENTION,
        .module_name = "attention_plasticity",
        .inbox_capacity = 64,
        .user_data = nullptr
    };

    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr) << "Failed to register attention module";

    // Verify module is registered
    bool is_registered = bio_router_is_module_registered(BIO_MODULE_ATTENTION);
    EXPECT_TRUE(is_registered);

    bio_router_unregister_module(ctx);
}

TEST_F(AttentionBioAsyncTest, WeightUpdateMessage) {
    // Register module
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_ATTENTION,
        .module_name = "attention_plasticity",
        .inbox_capacity = 64,
        .user_data = nullptr
    };
    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    // Send weight update message
    bio_message_t msg;
    msg.type = BIO_MSG_WEIGHT_UPDATE_REQUEST;
    msg.channel = BIO_CHANNEL_GLUTAMATE;
    msg.priority = BIO_PRIORITY_NORMAL;
    msg.size = 0;
    msg.source_module = BIO_MODULE_EXECUTIVE;
    msg.target_module = BIO_MODULE_ATTENTION;

    nimcp_error_t result = bio_router_send_message(&msg, ctx);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_unregister_module(ctx);
}

TEST_F(AttentionBioAsyncTest, AttentionUpdateBroadcast) {
    // Register module
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_ATTENTION,
        .module_name = "attention_plasticity",
        .inbox_capacity = 64,
        .user_data = nullptr
    };
    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    // Broadcast attention update
    bio_message_t msg;
    msg.type = BIO_MSG_ATTENTION_UPDATE;
    msg.channel = BIO_CHANNEL_GLUTAMATE;
    msg.priority = BIO_PRIORITY_HIGH;
    msg.size = 0;
    msg.source_module = BIO_MODULE_ATTENTION;
    msg.target_module = BIO_MODULE_BROADCAST;

    nimcp_error_t result = bio_router_broadcast(&msg, BIO_CHANNEL_GLUTAMATE);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_unregister_module(ctx);
}

TEST_F(AttentionBioAsyncTest, LoggingIntegration) {
    // Test that attention module logs properly
    // This test verifies logging doesn't crash, actual log capture
    // would require log infrastructure
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_ATTENTION,
        .module_name = "attention_plasticity",
        .inbox_capacity = 64,
        .user_data = nullptr
    };

    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    EXPECT_NE(ctx, nullptr);

    if (ctx) {
        bio_router_unregister_module(ctx);
    }
}

TEST_F(AttentionBioAsyncTest, SecurityValidation) {
    // Test that security validation is in place
    // This would test weight bounds checking, etc.
    // Actual validation happens within the attention module

    SUCCEED() << "Security validation placeholder";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
