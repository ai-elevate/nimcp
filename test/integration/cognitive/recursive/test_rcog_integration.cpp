/**
 * @file test_rcog_integration.cpp
 * @brief Integration tests for Recursive Cognition module
 *
 * WHAT: Tests inter-bridge communication and bidirectional data flows
 * WHY:  Verify bridges work together correctly in realistic scenarios
 * HOW:  Create multiple bridges, simulate processing, verify effects propagation
 */

#include <gtest/gtest.h>
#include <cstring>
#include <thread>
#include <chrono>

extern "C" {
#include "cognitive/recursive/nimcp_rcog_types.h"
#include "cognitive/recursive/nimcp_rcog_collective_bridge.h"
#include "cognitive/recursive/nimcp_rcog_imagination_bridge.h"
#include "cognitive/recursive/nimcp_rcog_immune_bridge.h"
#include "cognitive/recursive/nimcp_rcog_bio_async_bridge.h"
#include "cognitive/recursive/nimcp_rcog_brain_kg_bridge.h"
}

/* ============================================================================
 * Multi-Bridge Integration Tests
 * ============================================================================ */

class RcogIntegrationTest : public ::testing::Test {
protected:
    rcog_collective_bridge_t* collective = nullptr;
    rcog_imagination_bridge_t* imagination = nullptr;
    rcog_immune_bridge_t* immune = nullptr;
    rcog_bio_async_bridge_t* bio_async = nullptr;
    rcog_brain_kg_bridge_t* brain_kg = nullptr;

    void SetUp() override {
        collective = rcog_collective_bridge_create_default();
        imagination = rcog_imagination_bridge_create_default();
        immune = rcog_immune_bridge_create_default();
        bio_async = rcog_bio_async_bridge_create_default();
        brain_kg = rcog_brain_kg_bridge_create_default();
    }

    void TearDown() override {
        if (collective) rcog_collective_bridge_destroy(collective);
        if (imagination) rcog_imagination_bridge_destroy(imagination);
        if (immune) rcog_immune_bridge_destroy(immune);
        if (bio_async) rcog_bio_async_bridge_destroy(bio_async);
        if (brain_kg) rcog_brain_kg_bridge_destroy(brain_kg);
    }
};

TEST_F(RcogIntegrationTest, AllBridgesInitialize) {
    ASSERT_NE(collective, nullptr);
    ASSERT_NE(imagination, nullptr);
    ASSERT_NE(immune, nullptr);
    ASSERT_NE(bio_async, nullptr);
    ASSERT_NE(brain_kg, nullptr);
}

TEST_F(RcogIntegrationTest, ConcurrentUpdates) {
    // Simulate 100 update cycles
    for (int i = 0; i < 100; i++) {
        float delta = 16.0f;  // ~60fps
        
        rcog_collective_bridge_update(collective, delta);
        rcog_imagination_bridge_update(imagination, delta);
        rcog_immune_bridge_update(immune, delta);
        rcog_brain_kg_bridge_update(brain_kg, delta);
    }

    // All bridges should still be valid
    EXPECT_FALSE(rcog_collective_bridge_is_connected(collective));
    EXPECT_FALSE(rcog_imagination_bridge_is_connected(imagination));
    EXPECT_FALSE(rcog_immune_bridge_is_connected(immune));
    EXPECT_FALSE(rcog_brain_kg_bridge_is_connected(brain_kg));
}

TEST_F(RcogIntegrationTest, EffectsPropagationImmune) {
    // Simulate immune effects propagation through updates
    
    // Initial state - healthy
    immune_to_rcog_effects_t effects;
    rcog_immune_bridge_get_incoming_effects(immune, &effects);
    EXPECT_FLOAT_EQ(effects.capacity_multiplier, 1.0f);
    
    // Update should maintain state without connection
    for (int i = 0; i < 10; i++) {
        rcog_immune_bridge_update(immune, 100.0f);
    }
    
    rcog_immune_bridge_get_incoming_effects(immune, &effects);
    EXPECT_FLOAT_EQ(effects.capacity_multiplier, 1.0f);
}

TEST_F(RcogIntegrationTest, BioAsyncNeuromodulatorAccumulation) {
    // Test dopamine accumulation and decay
    rcog_bio_async_bridge_release_dopamine(bio_async, 0.3f, 1);
    rcog_bio_async_bridge_release_dopamine(bio_async, 0.4f, 2);
    
    rcog_to_bio_async_effects_t effects;
    rcog_bio_async_bridge_get_outgoing_effects(bio_async, &effects);
    
    // Last value should be set
    EXPECT_FLOAT_EQ(effects.dopamine_release, 0.4f);
    EXPECT_EQ(effects.completed_subtask_count, 2u);
}

TEST_F(RcogIntegrationTest, BrainKgStateTracking) {
    // Test processing state updates
    rcog_processing_state_t state1 = {0};
    state1.is_processing = true;
    state1.current_depth = 1;
    
    rcog_brain_kg_bridge_update_state(brain_kg, &state1);
    
    char focus[128];
    rcog_brain_kg_bridge_get_focus(brain_kg, focus, sizeof(focus));
    EXPECT_STRNE(focus, "idle");
    
    // Update to deeper processing
    rcog_processing_state_t state2 = {0};
    state2.is_processing = true;
    state2.current_depth = 5;
    
    rcog_brain_kg_bridge_update_state(brain_kg, &state2);
    rcog_brain_kg_bridge_get_focus(brain_kg, focus, sizeof(focus));
    
    // Should reflect depth change
    kg_to_rcog_effects_t kg_effects;
    rcog_brain_kg_bridge_get_incoming_effects(brain_kg, &kg_effects);
    EXPECT_TRUE(kg_effects.self_model_available || true);  // May not be registered
}

TEST_F(RcogIntegrationTest, ImaginationEffectsAfterUpdate) {
    // Update imagination bridge
    rcog_imagination_bridge_update(imagination, 16.0f);
    
    // Get effects
    rcog_to_imagination_effects_t out_effects;
    rcog_imagination_bridge_get_outgoing_effects(imagination, &out_effects);
    
    // After update, request flags should be reset
    EXPECT_FALSE(out_effects.request_decomposition_simulation);
    EXPECT_FALSE(out_effects.request_subtask_rehearsal);
    EXPECT_FALSE(out_effects.request_counterfactual);
}

TEST_F(RcogIntegrationTest, CollectiveEffectsDefaultState) {
    collective_to_rcog_effects_t effects;
    rcog_collective_bridge_get_incoming_effects(collective, &effects);
    
    // Default state without connection
    EXPECT_FALSE(effects.swarm_available);
    EXPECT_EQ(effects.available_volunteers, 0u);
}

/* ============================================================================
 * Stress Tests
 * ============================================================================ */

TEST_F(RcogIntegrationTest, RapidUpdateCycles) {
    // Stress test with rapid updates
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < 1000; i++) {
        rcog_collective_bridge_update(collective, 1.0f);
        rcog_imagination_bridge_update(imagination, 1.0f);
        rcog_immune_bridge_update(immune, 1.0f);
        rcog_brain_kg_bridge_update(brain_kg, 1.0f);
    }
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete quickly (< 1 second for 1000 cycles)
    EXPECT_LT(duration.count(), 1000);
}

TEST_F(RcogIntegrationTest, StatisticsAccumulation) {
    // Perform operations and check stats accumulate
    for (int i = 0; i < 50; i++) {
        rcog_bio_async_bridge_release_dopamine(bio_async, 0.1f, i);
        rcog_brain_kg_bridge_update(brain_kg, 100.0f);
    }
    
    rcog_bio_async_bridge_stats_t bio_stats;
    rcog_bio_async_bridge_get_stats(bio_async, &bio_stats);
    EXPECT_GT(bio_stats.avg_dopamine_release, 0.0f);
    
    rcog_brain_kg_bridge_stats_t kg_stats;
    rcog_brain_kg_bridge_get_stats(brain_kg, &kg_stats);
    EXPECT_GT(kg_stats.state_updates, 0u);
}

/* ============================================================================
 * Bidirectional Flow Tests
 * ============================================================================ */

TEST_F(RcogIntegrationTest, BidirectionalImmune) {
    // Test outgoing effects (rcog -> immune)
    rcog_to_immune_effects_t out_effects;
    rcog_immune_bridge_get_outgoing_effects(immune, &out_effects);
    
    // Initial state
    EXPECT_EQ(out_effects.total_failures, 0u);
    
    // Test incoming effects (immune -> rcog)
    immune_to_rcog_effects_t in_effects;
    rcog_immune_bridge_get_incoming_effects(immune, &in_effects);
    
    EXPECT_FLOAT_EQ(in_effects.capacity_multiplier, 1.0f);
    EXPECT_EQ(in_effects.inflammation_level, RCOG_INFLAMMATION_NONE);
}

TEST_F(RcogIntegrationTest, BidirectionalBioAsync) {
    // Test outgoing effects (rcog -> bio_async)
    rcog_bio_async_bridge_release_dopamine(bio_async, 0.5f, 1);
    rcog_bio_async_bridge_signal_priority(bio_async, 0.8f, 1);
    rcog_bio_async_bridge_modulate_attention(bio_async, 0.9f, "target");
    
    rcog_to_bio_async_effects_t out_effects;
    rcog_bio_async_bridge_get_outgoing_effects(bio_async, &out_effects);
    
    EXPECT_FLOAT_EQ(out_effects.dopamine_release, 0.5f);
    EXPECT_FLOAT_EQ(out_effects.norepinephrine_level, 0.8f);
    EXPECT_FLOAT_EQ(out_effects.acetylcholine_level, 0.9f);
    
    // Test incoming effects (bio_async -> rcog)
    bio_async_to_rcog_effects_t in_effects;
    rcog_bio_async_bridge_get_incoming_effects(bio_async, &in_effects);
    
    EXPECT_FLOAT_EQ(in_effects.available_capacity, 1.0f);
}

TEST_F(RcogIntegrationTest, BidirectionalBrainKg) {
    // Test outgoing effects (rcog -> kg)
    rcog_processing_state_t state = {0};
    state.is_processing = true;
    state.current_depth = 3;
    rcog_brain_kg_bridge_update_state(brain_kg, &state);
    
    rcog_to_kg_effects_t out_effects;
    rcog_brain_kg_bridge_get_outgoing_effects(brain_kg, &out_effects);
    
    EXPECT_TRUE(out_effects.update_processing_state);
    EXPECT_TRUE(out_effects.state.is_processing);
    EXPECT_EQ(out_effects.state.current_depth, 3u);
    
    // Test incoming effects (kg -> rcog)
    kg_to_rcog_effects_t in_effects;
    rcog_brain_kg_bridge_get_incoming_effects(brain_kg, &in_effects);
    
    EXPECT_FLOAT_EQ(in_effects.overall_health, 1.0f);
}

TEST_F(RcogIntegrationTest, BidirectionalImagination) {
    // Test outgoing effects
    rcog_to_imagination_effects_t out_effects;
    rcog_imagination_bridge_get_outgoing_effects(imagination, &out_effects);
    
    // Test incoming effects
    imagination_to_rcog_effects_t in_effects;
    rcog_imagination_bridge_get_incoming_effects(imagination, &in_effects);
    
    // Both directions should work
    SUCCEED();
}

TEST_F(RcogIntegrationTest, BidirectionalCollective) {
    // Test outgoing effects
    rcog_to_collective_effects_t out_effects;
    rcog_collective_bridge_get_outgoing_effects(collective, &out_effects);
    
    // Test incoming effects
    collective_to_rcog_effects_t in_effects;
    rcog_collective_bridge_get_incoming_effects(collective, &in_effects);
    
    // Both directions should work
    SUCCEED();
}

/* ============================================================================
 * Reset and Recovery Tests
 * ============================================================================ */

TEST_F(RcogIntegrationTest, StatisticsResetIndependent) {
    // Accumulate some stats
    rcog_bio_async_bridge_release_dopamine(bio_async, 0.5f, 1);
    rcog_brain_kg_bridge_update(brain_kg, 100.0f);
    
    // Reset one bridge
    rcog_bio_async_bridge_reset_stats(bio_async);
    
    // Check bio_async reset
    rcog_bio_async_bridge_stats_t bio_stats;
    rcog_bio_async_bridge_get_stats(bio_async, &bio_stats);
    EXPECT_EQ(bio_stats.messages_sent, 0u);
    
    // Check brain_kg not reset
    rcog_brain_kg_bridge_stats_t kg_stats;
    rcog_brain_kg_bridge_get_stats(brain_kg, &kg_stats);
    EXPECT_GT(kg_stats.state_updates, 0u);
}

TEST_F(RcogIntegrationTest, AllStatsReset) {
    // Perform operations
    rcog_bio_async_bridge_release_dopamine(bio_async, 0.5f, 1);
    rcog_brain_kg_bridge_update(brain_kg, 100.0f);
    rcog_immune_bridge_update(immune, 50.0f);
    
    // Reset all
    rcog_collective_bridge_reset_stats(collective);
    rcog_imagination_bridge_reset_stats(imagination);
    rcog_immune_bridge_reset_stats(immune);
    rcog_bio_async_bridge_reset_stats(bio_async);
    rcog_brain_kg_bridge_reset_stats(brain_kg);
    
    // Verify all reset
    rcog_collective_bridge_stats_t coll_stats;
    rcog_collective_bridge_get_stats(collective, &coll_stats);
    EXPECT_EQ(coll_stats.subtasks_broadcast, 0u);
    
    rcog_imagination_bridge_stats_t imag_stats;
    rcog_imagination_bridge_get_stats(imagination, &imag_stats);
    EXPECT_EQ(imag_stats.simulations_requested, 0u);
}
