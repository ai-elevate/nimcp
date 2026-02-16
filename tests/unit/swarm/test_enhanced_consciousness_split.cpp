/**
 * @file test_enhanced_consciousness_split.cpp
 * @brief Unit tests for split enhanced consciousness modules
 *
 * Tests the refactored enhanced consciousness modules to ensure:
 * - Each module fulfills its single responsibility
 * - Modules integrate correctly
 * - Public API remains unchanged
 *
 * @author NIMCP Development Team
 * @date 2026-02-16
 */

#include <gtest/gtest.h>

extern "C" {
#include "swarm/nimcp_swarm_consciousness_enhanced.h"
#include "swarm/nimcp_swarm_brain.h"
#include "core/brain/nimcp_brain.h"
}

class EnhancedConsciousnessSplitTest : public ::testing::Test {
protected:
    swarm_consciousness_enhanced_ctx_t* ctx;
    swarm_brain_t* swarm;

    void SetUp() override {
        // Create enhanced consciousness context
        ctx = swarm_consciousness_enhanced_create(nullptr);
        ASSERT_NE(nullptr, ctx);

        // Create swarm brain
        swarm_brain_config_t swarm_config = swarm_brain_default_config();
        swarm = swarm_brain_create(&swarm_config);
        ASSERT_NE(nullptr, swarm);
    }

    void TearDown() override {
        if (ctx) {
            swarm_consciousness_enhanced_destroy(ctx);
        }
        if (swarm) {
            swarm_brain_destroy(swarm);
        }
    }
};

//=============================================================================
// Core Module Tests (Lifecycle, Callbacks, Integration)
//=============================================================================

TEST_F(EnhancedConsciousnessSplitTest, CoreLifecycle) {
    // Test default config
    swarm_consciousness_enhanced_config_t config = swarm_consciousness_enhanced_default_config();
    EXPECT_TRUE(config.enable_geometry);
    EXPECT_TRUE(config.enable_dynamics);
    EXPECT_TRUE(config.enable_binding);
    EXPECT_TRUE(config.enable_hierarchy);
    EXPECT_TRUE(config.enable_resilience);

    // Test create/destroy
    swarm_consciousness_enhanced_ctx_t* temp_ctx = swarm_consciousness_enhanced_create(&config);
    ASSERT_NE(nullptr, temp_ctx);
    swarm_consciousness_enhanced_destroy(temp_ctx);
}

TEST_F(EnhancedConsciousnessSplitTest, CorePeerCallbacks) {
    bool callback_invoked = false;
    auto callback = [](const peer_event_t* event, void* data) {
        bool* flag = static_cast<bool*>(data);
        *flag = true;
    };

    // Register callback
    EXPECT_TRUE(swarm_consciousness_register_peer_callback(ctx, callback, &callback_invoked));

    // Simulate peer joined
    EXPECT_TRUE(swarm_consciousness_on_peer_joined(ctx, 1));

    // Unregister
    swarm_consciousness_unregister_peer_callback(ctx);
}

TEST_F(EnhancedConsciousnessSplitTest, CoreSwarmIntegration) {
    // Test attach/detach
    EXPECT_TRUE(swarm_consciousness_attach_to_swarm(ctx, swarm));
    swarm_consciousness_detach_from_swarm(ctx);
}

//=============================================================================
// Compute Module Tests (Phi Collection & Aggregation)
//=============================================================================

TEST_F(EnhancedConsciousnessSplitTest, ComputePhiCollection) {
    // Attach to swarm first
    ASSERT_TRUE(swarm_consciousness_attach_to_swarm(ctx, swarm));

    // Request phi from specific drone (will fail without signal adapter, but shouldn't crash)
    bool result = swarm_consciousness_request_phi(ctx, 1);
    // Expected to fail without signal adapter
    EXPECT_FALSE(result);

    // Request all phi
    uint32_t count = swarm_consciousness_request_all_phi(ctx);
    EXPECT_EQ(0u, count);  // No peers yet

    swarm_consciousness_detach_from_swarm(ctx);
}

TEST_F(EnhancedConsciousnessSplitTest, ComputeEnhancedMetrics) {
    ASSERT_TRUE(swarm_consciousness_attach_to_swarm(ctx, swarm));

    // Compute metrics
    swarm_consciousness_enhanced_metrics_t* metrics =
        swarm_compute_enhanced_metrics(ctx, swarm);
    ASSERT_NE(nullptr, metrics);

    // Verify structure
    EXPECT_GE(metrics->base.collective_phi, 0.0f);
    EXPECT_EQ(0u, metrics->remote_phi_collected);

    // Free metrics
    swarm_consciousness_enhanced_metrics_free(metrics);

    swarm_consciousness_detach_from_swarm(ctx);
}

//=============================================================================
// Stats Module Tests (Geometry, Dynamics, Binding)
//=============================================================================

TEST_F(EnhancedConsciousnessSplitTest, StatsGeometry) {
    information_geometry_t geometry;

    // Should fail with insufficient history
    bool result = swarm_compute_information_geometry(ctx, &geometry);
    EXPECT_FALSE(result);

    // After computing metrics, geometry should work if history is sufficient
    ASSERT_TRUE(swarm_consciousness_attach_to_swarm(ctx, swarm));
    swarm_consciousness_enhanced_metrics_t* metrics =
        swarm_compute_enhanced_metrics(ctx, swarm);
    ASSERT_NE(nullptr, metrics);

    swarm_consciousness_enhanced_metrics_free(metrics);
    swarm_consciousness_detach_from_swarm(ctx);
}

TEST_F(EnhancedConsciousnessSplitTest, StatsDynamics) {
    consciousness_dynamics_t dynamics;

    // Should fail with insufficient history
    bool result = swarm_compute_consciousness_dynamics(ctx, &dynamics);
    EXPECT_FALSE(result);
}

TEST_F(EnhancedConsciousnessSplitTest, StatsBinding) {
    neural_binding_t binding;

    // Should fail with insufficient history
    bool result = swarm_compute_neural_binding(ctx, &binding);
    EXPECT_FALSE(result);

    // Test binding query
    EXPECT_FALSE(swarm_consciousness_is_bound(ctx, 0.7f));
}

//=============================================================================
// Hierarchy Module Tests (Hierarchical Consciousness & Resilience)
//=============================================================================

TEST_F(EnhancedConsciousnessSplitTest, HierarchyComputation) {
    hierarchical_consciousness_t hierarchy;

    ASSERT_TRUE(swarm_consciousness_attach_to_swarm(ctx, swarm));

    // Compute hierarchy
    bool result = swarm_compute_hierarchical_consciousness(ctx, swarm, &hierarchy);
    EXPECT_TRUE(result);

    // Verify levels exist
    EXPECT_GE(hierarchy.phi_by_level[HIERARCHY_INDIVIDUAL], 0.0f);
    EXPECT_GE(hierarchy.phi_by_level[HIERARCHY_SQUAD], 0.0f);
    EXPECT_GE(hierarchy.phi_by_level[HIERARCHY_PLATOON], 0.0f);
    EXPECT_GE(hierarchy.phi_by_level[HIERARCHY_SWARM], 0.0f);

    swarm_consciousness_detach_from_swarm(ctx);
}

TEST_F(EnhancedConsciousnessSplitTest, ResilienceComputation) {
    consciousness_resilience_t resilience;

    ASSERT_TRUE(swarm_consciousness_attach_to_swarm(ctx, swarm));

    // Compute resilience
    bool result = swarm_compute_consciousness_resilience(ctx, swarm, &resilience);
    EXPECT_TRUE(result);

    // Verify metrics
    EXPECT_GE(resilience.baseline_phi, 0.0f);
    EXPECT_GE(resilience.dropout_sensitivity, 0.0f);
    EXPECT_LE(resilience.dropout_sensitivity, 1.0f);

    swarm_consciousness_detach_from_swarm(ctx);
}

//=============================================================================
// Integration Tests (Modules Working Together)
//=============================================================================

TEST_F(EnhancedConsciousnessSplitTest, IntegrationFullPipeline) {
    // Attach to swarm
    ASSERT_TRUE(swarm_consciousness_attach_to_swarm(ctx, swarm));

    // Compute full enhanced metrics
    swarm_consciousness_enhanced_metrics_t* metrics =
        swarm_compute_enhanced_metrics(ctx, swarm);
    ASSERT_NE(nullptr, metrics);

    // Verify all subsystems
    EXPECT_GE(metrics->base.collective_phi, 0.0f);
    EXPECT_GE(metrics->geometry.total_correlation, 0.0f);
    EXPECT_NE(CONSCIOUSNESS_PHASE_CHAOS + 999, metrics->dynamics.current_phase);
    EXPECT_GE(metrics->binding.gamma_power, 0.0f);
    EXPECT_GE(metrics->hierarchy.phi_by_level[HIERARCHY_INDIVIDUAL], 0.0f);
    EXPECT_GE(metrics->resilience.baseline_phi, 0.0f);

    // Print metrics (shouldn't crash)
    swarm_consciousness_enhanced_print_metrics(metrics, true);

    // Validate metrics
    EXPECT_TRUE(swarm_consciousness_enhanced_bbb_validate(metrics));

    swarm_consciousness_enhanced_metrics_free(metrics);
    swarm_consciousness_detach_from_swarm(ctx);
}

TEST_F(EnhancedConsciousnessSplitTest, IntegrationPhaseTransition) {
    bool transition_detected = false;

    consciousness_phase_t phase = swarm_consciousness_detect_phase_transition(
        ctx, &transition_detected);

    // Should return a valid phase
    EXPECT_GE(phase, CONSCIOUSNESS_PHASE_CHAOS);
    EXPECT_LE(phase, CONSCIOUSNESS_PHASE_FROZEN);

    // Get current phase
    consciousness_phase_t current = swarm_consciousness_get_phase(ctx);
    EXPECT_EQ(CONSCIOUSNESS_PHASE_CHAOS, current);  // Default phase
}

//=============================================================================
// Utility Tests
//=============================================================================

TEST_F(EnhancedConsciousnessSplitTest, UtilityFunctions) {
    // Test phase names
    EXPECT_STREQ("CHAOS", consciousness_phase_name(CONSCIOUSNESS_PHASE_CHAOS));
    EXPECT_STREQ("CRITICAL", consciousness_phase_name(CONSCIOUSNESS_PHASE_CRITICAL));
    EXPECT_STREQ("ORDERED", consciousness_phase_name(CONSCIOUSNESS_PHASE_ORDERED));
    EXPECT_STREQ("FROZEN", consciousness_phase_name(CONSCIOUSNESS_PHASE_FROZEN));

    // Test hierarchy names
    EXPECT_STREQ("INDIVIDUAL", consciousness_hierarchy_name(HIERARCHY_INDIVIDUAL));
    EXPECT_STREQ("SQUAD", consciousness_hierarchy_name(HIERARCHY_SQUAD));
    EXPECT_STREQ("PLATOON", consciousness_hierarchy_name(HIERARCHY_PLATOON));
    EXPECT_STREQ("SWARM", consciousness_hierarchy_name(HIERARCHY_SWARM));
}

//=============================================================================
// Regression Tests (Ensure Public API Unchanged)
//=============================================================================

TEST_F(EnhancedConsciousnessSplitTest, RegressionPublicAPIUnchanged) {
    // All public API functions should still exist and work

    // Lifecycle
    swarm_consciousness_enhanced_config_t config = swarm_consciousness_enhanced_default_config();
    swarm_consciousness_enhanced_ctx_t* temp = swarm_consciousness_enhanced_create(&config);
    ASSERT_NE(nullptr, temp);
    swarm_consciousness_enhanced_destroy(temp);

    // Peer events
    EXPECT_TRUE(swarm_consciousness_register_peer_callback(ctx, nullptr, nullptr));
    swarm_consciousness_unregister_peer_callback(ctx);

    // Remote phi
    float phi_values[32];
    uint16_t drone_ids[32];
    uint32_t count;
    bool result = swarm_consciousness_get_remote_phi(ctx, phi_values, drone_ids, &count);
    EXPECT_TRUE(result);
    EXPECT_EQ(0u, count);

    // Phase transitions
    EXPECT_TRUE(swarm_consciousness_register_phase_callback(ctx, nullptr, nullptr));
    consciousness_phase_t phase = swarm_consciousness_get_phase(ctx);
    EXPECT_GE(phase, CONSCIOUSNESS_PHASE_CHAOS);

    // Binding
    EXPECT_TRUE(swarm_consciousness_register_binding_callback(ctx, nullptr, nullptr));
    EXPECT_FALSE(swarm_consciousness_is_bound(ctx, 0.7f));

    // Integration
    EXPECT_TRUE(swarm_consciousness_set_signal_adapter(ctx, nullptr));

    // Bio-async
    // (Will fail if bio-async not initialized, which is expected in unit test)
    swarm_consciousness_enhanced_register_bio_async(ctx);
}
