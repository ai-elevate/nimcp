/**
 * @file test_plasticity_bioasync_integration.cpp
 * @brief Integration tests for plasticity modules with bio-async
 *
 * These tests verify that multiple plasticity modules can work together
 * through the bio-async messaging system.
 */

#include <gtest/gtest.h>
#include <vector>

extern "C" {
#include "plasticity/stdp/nimcp_stdp.h"
#include "plasticity/homeostatic/nimcp_homeostatic.h"
#include "plasticity/attention/nimcp_attention.h"
#include "plasticity/dendritic/nimcp_dendritic.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "plasticity/predictive/nimcp_predictive_coding.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "plasticity/neuromodulators/nimcp_receptor_subtypes.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class PlasticityBioAsyncIntegrationTest : public ::testing::Test {
protected:
    std::vector<bio_module_context_t> registered_modules;

    void SetUp() override {
        bio_async_init();
        bio_router_init();
        nimcp_unified_memory_init();
    }

    void TearDown() override {
        // Unregister all modules
        for (auto ctx : registered_modules) {
            if (ctx) {
                bio_router_unregister_module(ctx);
            }
        }
        registered_modules.clear();

        bio_router_shutdown();
        bio_async_shutdown();
        nimcp_unified_memory_shutdown();
    }

    bio_module_context_t RegisterModule(bio_module_id_t id, const char* name) {
        bio_module_info_t info = {
            .module_id = id,
            .module_name = name,
            .inbox_capacity = 64,
            .user_data = nullptr
        };
        bio_module_context_t ctx = bio_router_register_module(&info);
        if (ctx) {
            registered_modules.push_back(ctx);
        }
        return ctx;
    }
};

TEST_F(PlasticityBioAsyncIntegrationTest, MultiModuleRegistration) {
    // Register multiple plasticity modules
    auto attention_ctx = RegisterModule(BIO_MODULE_ATTENTION, "attention");
    auto dendritic_ctx = RegisterModule(BIO_MODULE_DENDRITIC, "dendritic");
    auto adaptive_ctx = RegisterModule(BIO_MODULE_ADAPTIVE, "adaptive");
    auto predictive_ctx = RegisterModule(BIO_MODULE_PREDICTIVE, "predictive");

    ASSERT_NE(attention_ctx, nullptr);
    ASSERT_NE(dendritic_ctx, nullptr);
    ASSERT_NE(adaptive_ctx, nullptr);
    ASSERT_NE(predictive_ctx, nullptr);

    // Verify all modules are registered
    EXPECT_TRUE(bio_router_is_module_registered(BIO_MODULE_ATTENTION));
    EXPECT_TRUE(bio_router_is_module_registered(BIO_MODULE_DENDRITIC));
    EXPECT_TRUE(bio_router_is_module_registered(BIO_MODULE_ADAPTIVE));
    EXPECT_TRUE(bio_router_is_module_registered(BIO_MODULE_PREDICTIVE));
}

TEST_F(PlasticityBioAsyncIntegrationTest, STDPToNeuromodulatorFlow) {
    // Test: STDP module sends weight update, neuromodulator system responds
    auto stdp_ctx = RegisterModule(BIO_MODULE_STDP, "stdp");
    auto neuromod_ctx = RegisterModule(BIO_MODULE_NEUROMODULATOR, "neuromodulator");

    ASSERT_NE(stdp_ctx, nullptr);
    ASSERT_NE(neuromod_ctx, nullptr);

    // STDP sends weight update
    bio_message_t msg;
    msg.type = BIO_MSG_WEIGHT_UPDATE_REQUEST;
    msg.channel = BIO_CHANNEL_DOPAMINE;
    msg.priority = BIO_PRIORITY_NORMAL;
    msg.size = 0;
    msg.source_module = BIO_MODULE_STDP;
    msg.target_module = BIO_MODULE_NEUROMODULATOR;

    nimcp_error_t result = bio_router_send_message(&msg, stdp_ctx);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Process messages
    bio_router_process_messages();
}

TEST_F(PlasticityBioAsyncIntegrationTest, PredictiveErrorToDopamine) {
    // Test: Predictive coding error signal triggers dopamine release
    auto predictive_ctx = RegisterModule(BIO_MODULE_PREDICTIVE, "predictive");
    auto neuromod_ctx = RegisterModule(BIO_MODULE_NEUROMODULATOR, "neuromodulator");

    ASSERT_NE(predictive_ctx, nullptr);
    ASSERT_NE(neuromod_ctx, nullptr);

    // Predictive coding broadcasts error signal
    bio_message_t msg;
    msg.type = BIO_MSG_ERROR_SIGNAL;
    msg.channel = BIO_CHANNEL_DOPAMINE;
    msg.priority = BIO_PRIORITY_HIGH;
    msg.size = 0;
    msg.source_module = BIO_MODULE_PREDICTIVE;
    msg.target_module = BIO_MODULE_BROADCAST;

    nimcp_error_t result = bio_router_broadcast(&msg, BIO_CHANNEL_DOPAMINE);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Process messages
    bio_router_process_messages();
}

TEST_F(PlasticityBioAsyncIntegrationTest, DendriticSpikeToSTDP) {
    // Test: Dendritic spike triggers STDP update
    auto dendritic_ctx = RegisterModule(BIO_MODULE_DENDRITIC, "dendritic");
    auto stdp_ctx = RegisterModule(BIO_MODULE_STDP, "stdp");

    ASSERT_NE(dendritic_ctx, nullptr);
    ASSERT_NE(stdp_ctx, nullptr);

    // Dendritic module broadcasts spike event
    bio_message_t msg;
    msg.type = BIO_MSG_DENDRITIC_SPIKE;
    msg.channel = BIO_CHANNEL_CALCIUM;
    msg.priority = BIO_PRIORITY_HIGH;
    msg.size = 0;
    msg.source_module = BIO_MODULE_DENDRITIC;
    msg.target_module = BIO_MODULE_BROADCAST;

    nimcp_error_t result = bio_router_broadcast(&msg, BIO_CHANNEL_CALCIUM);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_process_messages();
}

TEST_F(PlasticityBioAsyncIntegrationTest, AttentionModulatesLearning) {
    // Test: Attention system modulates learning rate in other modules
    auto attention_ctx = RegisterModule(BIO_MODULE_ATTENTION, "attention");
    auto adaptive_ctx = RegisterModule(BIO_MODULE_ADAPTIVE, "adaptive");
    auto neuromod_ctx = RegisterModule(BIO_MODULE_NEUROMODULATOR, "neuromodulator");

    ASSERT_NE(attention_ctx, nullptr);
    ASSERT_NE(adaptive_ctx, nullptr);
    ASSERT_NE(neuromod_ctx, nullptr);

    // Attention broadcasts update
    bio_message_t msg;
    msg.type = BIO_MSG_ATTENTION_UPDATE;
    msg.channel = BIO_CHANNEL_ACETYLCHOLINE;
    msg.priority = BIO_PRIORITY_NORMAL;
    msg.size = 0;
    msg.source_module = BIO_MODULE_ATTENTION;
    msg.target_module = BIO_MODULE_BROADCAST;

    nimcp_error_t result = bio_router_broadcast(&msg, BIO_CHANNEL_ACETYLCHOLINE);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_process_messages();
}

TEST_F(PlasticityBioAsyncIntegrationTest, MultiChannelBroadcast) {
    // Test: Multiple channels can operate simultaneously
    auto predictive_ctx = RegisterModule(BIO_MODULE_PREDICTIVE, "predictive");
    auto dendritic_ctx = RegisterModule(BIO_MODULE_DENDRITIC, "dendritic");
    auto attention_ctx = RegisterModule(BIO_MODULE_ATTENTION, "attention");

    ASSERT_NE(predictive_ctx, nullptr);
    ASSERT_NE(dendritic_ctx, nullptr);
    ASSERT_NE(attention_ctx, nullptr);

    // Broadcast on dopamine channel
    bio_message_t msg1;
    msg1.type = BIO_MSG_ERROR_SIGNAL;
    msg1.channel = BIO_CHANNEL_DOPAMINE;
    msg1.priority = BIO_PRIORITY_HIGH;
    msg1.size = 0;
    msg1.source_module = BIO_MODULE_PREDICTIVE;
    msg1.target_module = BIO_MODULE_BROADCAST;

    // Broadcast on calcium channel
    bio_message_t msg2;
    msg2.type = BIO_MSG_DENDRITIC_SPIKE;
    msg2.channel = BIO_CHANNEL_CALCIUM;
    msg2.priority = BIO_PRIORITY_HIGH;
    msg2.size = 0;
    msg2.source_module = BIO_MODULE_DENDRITIC;
    msg2.target_module = BIO_MODULE_BROADCAST;

    // Broadcast on acetylcholine channel
    bio_message_t msg3;
    msg3.type = BIO_MSG_ATTENTION_UPDATE;
    msg3.channel = BIO_CHANNEL_ACETYLCHOLINE;
    msg3.priority = BIO_PRIORITY_NORMAL;
    msg3.size = 0;
    msg3.source_module = BIO_MODULE_ATTENTION;
    msg3.target_module = BIO_MODULE_BROADCAST;

    EXPECT_EQ(bio_router_broadcast(&msg1, BIO_CHANNEL_DOPAMINE), NIMCP_SUCCESS);
    EXPECT_EQ(bio_router_broadcast(&msg2, BIO_CHANNEL_CALCIUM), NIMCP_SUCCESS);
    EXPECT_EQ(bio_router_broadcast(&msg3, BIO_CHANNEL_ACETYLCHOLINE), NIMCP_SUCCESS);

    bio_router_process_messages();
}

TEST_F(PlasticityBioAsyncIntegrationTest, ReceptorSubtypeModulation) {
    // Test: Receptor subtypes respond to neuromodulator signals
    auto receptor_ctx = RegisterModule(BIO_MODULE_RECEPTOR, "receptor");
    auto neuromod_ctx = RegisterModule(BIO_MODULE_NEUROMODULATOR, "neuromodulator");

    ASSERT_NE(receptor_ctx, nullptr);
    ASSERT_NE(neuromod_ctx, nullptr);

    // Neuromodulator sends release signal
    bio_message_t msg;
    msg.type = BIO_MSG_NEUROMODULATOR_RELEASE;
    msg.channel = BIO_CHANNEL_DOPAMINE;
    msg.priority = BIO_PRIORITY_NORMAL;
    msg.size = 0;
    msg.source_module = BIO_MODULE_NEUROMODULATOR;
    msg.target_module = BIO_MODULE_RECEPTOR;

    nimcp_error_t result = bio_router_send_message(&msg, neuromod_ctx);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    bio_router_process_messages();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
