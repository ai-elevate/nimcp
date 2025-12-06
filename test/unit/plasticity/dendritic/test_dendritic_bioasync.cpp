/**
 * @file test_dendritic_bioasync.cpp
 * @brief Unit tests for dendritic plasticity bio-async integration
 */

#include <gtest/gtest.h>

extern "C" {
#include "plasticity/dendritic/nimcp_dendritic.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class DendriticBioAsyncTest : public ::testing::Test {
protected:
    dendritic_tree_t tree;

    void SetUp() override {
        bio_async_init();
        bio_router_init();
        nimcp_unified_memory_init();

        // Create test dendritic tree
        dendritic_tree_config_t config = dendritic_tree_config_default();
        tree = dendritic_tree_create(&config);
        ASSERT_NE(tree, nullptr);
    }

    void TearDown() override {
        if (tree) {
            dendritic_tree_destroy(tree);
        }
        bio_router_shutdown();
        bio_async_shutdown();
        nimcp_unified_memory_shutdown();
    }
};

TEST_F(DendriticBioAsyncTest, ModuleRegistration) {
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_DENDRITIC,
        .module_name = "dendritic_plasticity",
        .inbox_capacity = 64,
        .user_data = nullptr
    };

    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    bool is_registered = bio_router_is_module_registered(BIO_MODULE_DENDRITIC);
    EXPECT_TRUE(is_registered);

    bio_router_unregister_module(ctx);
}

TEST_F(DendriticBioAsyncTest, DendriticSpikeBroadcast) {
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_DENDRITIC,
        .module_name = "dendritic_plasticity",
        .inbox_capacity = 64,
        .user_data = nullptr
    };
    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    // Simulate dendritic spike event
    bio_message_t msg;
    msg.type = BIO_MSG_DENDRITIC_SPIKE;
    msg.channel = BIO_CHANNEL_CALCIUM;
    msg.priority = BIO_PRIORITY_HIGH;
    msg.size = 0;
    msg.source_module = BIO_MODULE_DENDRITIC;
    msg.target_module = BIO_MODULE_BROADCAST;

    nimcp_error_t result = bio_router_broadcast(&msg, BIO_CHANNEL_CALCIUM);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_unregister_module(ctx);
}

TEST_F(DendriticBioAsyncTest, CalciumDynamicsUpdate) {
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_DENDRITIC,
        .module_name = "dendritic_plasticity",
        .inbox_capacity = 64,
        .user_data = nullptr
    };
    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    // Send calcium update message
    bio_message_t msg;
    msg.type = BIO_MSG_CALCIUM_UPDATE;
    msg.channel = BIO_CHANNEL_CALCIUM;
    msg.priority = BIO_PRIORITY_NORMAL;
    msg.size = 0;
    msg.source_module = BIO_MODULE_EXECUTIVE;
    msg.target_module = BIO_MODULE_DENDRITIC;

    nimcp_error_t result = bio_router_send_message(&msg, ctx);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_unregister_module(ctx);
}

TEST_F(DendriticBioAsyncTest, CalciumConcentrationValidation) {
    // Test that calcium concentrations are validated
    // This would check security validation of calcium levels

    float total_calcium = dendritic_tree_get_total_calcium(tree);
    EXPECT_GE(total_calcium, 0.0f);
    EXPECT_LE(total_calcium, 100.0f); // Reasonable upper bound
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
