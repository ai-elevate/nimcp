/**
 * @file test_adaptive_bioasync.cpp
 * @brief Unit tests for adaptive threshold plasticity bio-async integration
 */

#include <gtest/gtest.h>

extern "C" {
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class AdaptiveBioAsyncTest : public ::testing::Test {
protected:
    void SetUp() override {
        bio_async_init();
        bio_router_init();
        nimcp_unified_memory_init();
    }

    void TearDown() override {
        bio_router_shutdown();
        bio_async_shutdown();
        nimcp_unified_memory_shutdown();
    }
};

TEST_F(AdaptiveBioAsyncTest, ModuleRegistration) {
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_ADAPTIVE,
        .module_name = "adaptive_threshold",
        .inbox_capacity = 64,
        .user_data = nullptr
    };

    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    bool is_registered = bio_router_is_module_registered(BIO_MODULE_ADAPTIVE);
    EXPECT_TRUE(is_registered);

    bio_router_unregister_module(ctx);
}

TEST_F(AdaptiveBioAsyncTest, ThresholdAdaptationMessage) {
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_ADAPTIVE,
        .module_name = "adaptive_threshold",
        .inbox_capacity = 64,
        .user_data = nullptr
    };
    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    bio_message_t msg;
    msg.type = BIO_MSG_THRESHOLD_UPDATE;
    msg.channel = BIO_CHANNEL_ACETYLCHOLINE;
    msg.priority = BIO_PRIORITY_NORMAL;
    msg.size = 0;
    msg.source_module = BIO_MODULE_EXECUTIVE;
    msg.target_module = BIO_MODULE_ADAPTIVE;

    nimcp_error_t result = bio_router_send_message(&msg, ctx);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_unregister_module(ctx);
}

TEST_F(AdaptiveBioAsyncTest, LearningRateModulation) {
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_ADAPTIVE,
        .module_name = "adaptive_threshold",
        .inbox_capacity = 64,
        .user_data = nullptr
    };
    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    bio_message_t msg;
    msg.type = BIO_MSG_LEARNING_RATE;
    msg.channel = BIO_CHANNEL_ACETYLCHOLINE;
    msg.priority = BIO_PRIORITY_NORMAL;
    msg.size = 0;
    msg.source_module = BIO_MODULE_NEUROMODULATOR;
    msg.target_module = BIO_MODULE_ADAPTIVE;

    nimcp_error_t result = bio_router_send_message(&msg, ctx);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_unregister_module(ctx);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
