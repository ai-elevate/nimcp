/**
 * @file test_predictive_bioasync.cpp
 * @brief Unit tests for predictive coding plasticity bio-async integration
 */

#include <gtest/gtest.h>

extern "C" {
#include "plasticity/predictive/nimcp_predictive_coding.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class PredictiveBioAsyncTest : public ::testing::Test {
protected:
    pc_hierarchy_t hierarchy;

    void SetUp() override {
        bio_async_init();
        bio_router_init();
        nimcp_unified_memory_init();

        // Create test hierarchy
        uint32_t units[] = {10, 5, 2};
        pc_hierarchy_config_t config = pc_hierarchy_config_default(3, units);
        hierarchy = pc_hierarchy_create(&config);
        ASSERT_NE(hierarchy, nullptr);
    }

    void TearDown() override {
        if (hierarchy) {
            pc_hierarchy_destroy(hierarchy);
        }
        bio_router_shutdown();
        bio_async_shutdown();
        nimcp_unified_memory_shutdown();
    }
};

TEST_F(PredictiveBioAsyncTest, ModuleRegistration) {
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_PREDICTIVE,
        .module_name = "predictive_coding",
        .inbox_capacity = 64,
        .user_data = nullptr
    };

    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    bool is_registered = bio_router_is_module_registered(BIO_MODULE_PREDICTIVE);
    EXPECT_TRUE(is_registered);

    bio_router_unregister_module(ctx);
}

TEST_F(PredictiveBioAsyncTest, PredictionErrorBroadcast) {
    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_PREDICTIVE,
        .module_name = "predictive_coding",
        .inbox_capacity = 64,
        .user_data = nullptr
    };
    bio_module_context_t ctx = bio_router_register_module(&bio_info);
    ASSERT_NE(ctx, nullptr);

    bio_message_t msg;
    msg.type = BIO_MSG_ERROR_SIGNAL;
    msg.channel = BIO_CHANNEL_DOPAMINE;
    msg.priority = BIO_PRIORITY_HIGH;
    msg.size = 0;
    msg.source_module = BIO_MODULE_PREDICTIVE;
    msg.target_module = BIO_MODULE_BROADCAST;

    nimcp_error_t result = bio_router_broadcast(&msg, BIO_CHANNEL_DOPAMINE);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_unregister_module(ctx);
}

TEST_F(PredictiveBioAsyncTest, FreeEnergyComputation) {
    // Run one inference step
    float input[] = {0.5f, 0.3f, 0.8f, 0.1f, 0.6f, 0.4f, 0.9f, 0.2f, 0.7f, 0.0f};
    pc_hierarchy_set_input(hierarchy, input);
    pc_hierarchy_inference_step(hierarchy, 1.0f, false);

    float free_energy = pc_hierarchy_get_free_energy(hierarchy);
    EXPECT_GE(free_energy, 0.0f);
    EXPECT_LT(free_energy, 1000.0f); // Reasonable bound
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
