/**
 * @file test_fep_orchestrator.cpp
 * @brief Unit tests for FEP Orchestrator
 * @version 1.0.0
 * @date 2025-12-15
 *
 * Tests for the FEP Orchestrator including:
 * - Lifecycle (create, destroy, start, stop, pause, resume)
 * - Bridge registration and unregistration
 * - Update cycles and category-based updates
 * - Bio-async integration
 * - Immune system integration
 * - Statistics tracking
 */

#include <gtest/gtest.h>

extern "C" {
#include "cognitive/free_energy/nimcp_fep_orchestrator.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_time.h"
}

/* ============================================================================
 * Mock Bridge for Testing
 * ============================================================================ */

static int g_mock_update_count = 0;
static int g_mock_update_return = 0;

static int mock_bridge_update(void* bridge) {
    (void)bridge;
    g_mock_update_count++;
    return g_mock_update_return;
}

static void mock_bridge_destroy(void* bridge) {
    (void)bridge;
    /* No-op for mock */
}

/* ============================================================================
 * Test Fixture
 * ============================================================================ */

class FepOrchestratorTest : public ::testing::Test {
protected:
    fep_orchestrator_t* orchestrator = nullptr;
    fep_orchestrator_config_t config;

    void SetUp() override {
        g_mock_update_count = 0;
        g_mock_update_return = 0;
        fep_orchestrator_default_config(&config);
        config.enable_logging = false;  /* Reduce test output */
        orchestrator = fep_orchestrator_create(&config);
        ASSERT_NE(orchestrator, nullptr);
    }

    void TearDown() override {
        if (orchestrator) {
            fep_orchestrator_destroy(orchestrator);
            orchestrator = nullptr;
        }
    }
};

/* ============================================================================
 * Lifecycle Tests
 * ============================================================================ */

TEST_F(FepOrchestratorTest, DefaultConfigIsValid) {
    fep_orchestrator_config_t cfg;
    int result = fep_orchestrator_default_config(&cfg);
    EXPECT_EQ(result, 0);
    EXPECT_GT(cfg.max_bridges, 0u);
    EXPECT_TRUE(cfg.enable_auto_update);
    EXPECT_TRUE(cfg.enable_bio_async);
    EXPECT_TRUE(cfg.enable_brain_immune);
    EXPECT_TRUE(cfg.enable_statistics);
    
    /* Check category intervals */
    EXPECT_TRUE(cfg.categories[FEP_BRIDGE_CATEGORY_COGNITIVE].enabled);
    EXPECT_EQ(cfg.categories[FEP_BRIDGE_CATEGORY_COGNITIVE].update_interval_ms, FEP_UPDATE_INTERVAL_COGNITIVE);
}

TEST_F(FepOrchestratorTest, DefaultConfigNullFails) {
    int result = fep_orchestrator_default_config(nullptr);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

TEST_F(FepOrchestratorTest, CreateWithNullConfigUsesDefaults) {
    fep_orchestrator_t* orch = fep_orchestrator_create(nullptr);
    ASSERT_NE(orch, nullptr);
    EXPECT_EQ(orch->config.max_bridges, FEP_ORCHESTRATOR_MAX_BRIDGES);
    fep_orchestrator_destroy(orch);
}

TEST_F(FepOrchestratorTest, CreateWithCustomConfig) {
    fep_orchestrator_config_t custom_cfg;
    fep_orchestrator_default_config(&custom_cfg);
    custom_cfg.max_bridges = 64;
    custom_cfg.enable_logging = false;
    
    fep_orchestrator_t* orch = fep_orchestrator_create(&custom_cfg);
    ASSERT_NE(orch, nullptr);
    EXPECT_EQ(orch->bridge_capacity, 64u);
    fep_orchestrator_destroy(orch);
}

TEST_F(FepOrchestratorTest, DestroyNullIsSafe) {
    fep_orchestrator_destroy(nullptr);
    /* Should not crash */
}

TEST_F(FepOrchestratorTest, StartSucceeds) {
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_STOPPED);
    int result = fep_orchestrator_start(orchestrator);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_RUNNING);
}

TEST_F(FepOrchestratorTest, StartNullFails) {
    int result = fep_orchestrator_start(nullptr);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

TEST_F(FepOrchestratorTest, StopSucceeds) {
    fep_orchestrator_start(orchestrator);
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_RUNNING);
    
    int result = fep_orchestrator_stop(orchestrator);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_STOPPED);
}

TEST_F(FepOrchestratorTest, PauseAndResumeSucceeds) {
    fep_orchestrator_start(orchestrator);
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_RUNNING);
    
    int result = fep_orchestrator_pause(orchestrator);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_PAUSED);
    
    result = fep_orchestrator_resume(orchestrator);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_RUNNING);
}

TEST_F(FepOrchestratorTest, PauseWhenNotRunningFails) {
    /* Orchestrator is stopped by default */
    int result = fep_orchestrator_pause(orchestrator);
    EXPECT_EQ(result, NIMCP_ERROR_INVALID_STATE);
}

TEST_F(FepOrchestratorTest, ResumeWhenNotPausedFails) {
    fep_orchestrator_start(orchestrator);
    int result = fep_orchestrator_resume(orchestrator);
    EXPECT_EQ(result, NIMCP_ERROR_INVALID_STATE);
}

/* ============================================================================
 * Bridge Registration Tests
 * ============================================================================ */

TEST_F(FepOrchestratorTest, RegisterBridgeSucceeds) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    int result = fep_orchestrator_register_bridge(
        orchestrator,
        "test_bridge",
        FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge,
        mock_bridge_update,
        mock_bridge_destroy,
        &bridge_id
    );
    
    EXPECT_EQ(result, 0);
    EXPECT_GT(bridge_id, 0u);
    EXPECT_EQ(orchestrator->bridge_count, 1u);
}

TEST_F(FepOrchestratorTest, RegisterMultipleBridges) {
    int bridges[5];
    uint32_t ids[5];
    
    for (int i = 0; i < 5; i++) {
        bridges[i] = i;
        char name[32];
        snprintf(name, sizeof(name), "bridge_%d", i);
        
        int result = fep_orchestrator_register_bridge(
            orchestrator,
            name,
            (fep_bridge_category_t)(i % FEP_BRIDGE_CATEGORY_COUNT),
            &bridges[i],
            mock_bridge_update,
            nullptr,
            &ids[i]
        );
        EXPECT_EQ(result, 0);
    }
    
    EXPECT_EQ(orchestrator->bridge_count, 5u);
}

TEST_F(FepOrchestratorTest, RegisterBridgeNullParamsFail) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    /* Null orchestrator */
    EXPECT_EQ(fep_orchestrator_register_bridge(
        nullptr, "test", FEP_BRIDGE_CATEGORY_COGNITIVE, &mock_bridge,
        mock_bridge_update, nullptr, &bridge_id
    ), NIMCP_ERROR_NULL_POINTER);
    
    /* Null name */
    EXPECT_EQ(fep_orchestrator_register_bridge(
        orchestrator, nullptr, FEP_BRIDGE_CATEGORY_COGNITIVE, &mock_bridge,
        mock_bridge_update, nullptr, &bridge_id
    ), NIMCP_ERROR_NULL_POINTER);
    
    /* Null handle */
    EXPECT_EQ(fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE, nullptr,
        mock_bridge_update, nullptr, &bridge_id
    ), NIMCP_ERROR_NULL_POINTER);
    
    /* Null update function */
    EXPECT_EQ(fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE, &mock_bridge,
        nullptr, nullptr, &bridge_id
    ), NIMCP_ERROR_NULL_POINTER);
}

TEST_F(FepOrchestratorTest, RegisterBridgeInvalidCategoryFails) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    int result = fep_orchestrator_register_bridge(
        orchestrator,
        "test_bridge",
        (fep_bridge_category_t)999,  /* Invalid category */
        &mock_bridge,
        mock_bridge_update,
        nullptr,
        &bridge_id
    );
    
    EXPECT_EQ(result, NIMCP_ERROR_INVALID_PARAMETER);
}

TEST_F(FepOrchestratorTest, UnregisterBridgeSucceeds) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    EXPECT_EQ(orchestrator->bridge_count, 1u);
    
    int result = fep_orchestrator_unregister_bridge(orchestrator, bridge_id);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(orchestrator->bridge_count, 0u);
}

TEST_F(FepOrchestratorTest, UnregisterNonExistentBridgeFails) {
    int result = fep_orchestrator_unregister_bridge(orchestrator, 9999);
    EXPECT_EQ(result, NIMCP_ERROR_INVALID_PARAMETER);
}

TEST_F(FepOrchestratorTest, SetBridgeEnabledSucceeds) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    
    /* Disable */
    int result = fep_orchestrator_set_bridge_enabled(orchestrator, bridge_id, false);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(orchestrator->stats.active_bridges, 0u);
    
    /* Re-enable */
    result = fep_orchestrator_set_bridge_enabled(orchestrator, bridge_id, true);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(orchestrator->stats.active_bridges, 1u);
}

TEST_F(FepOrchestratorTest, GetBridgeSucceeds) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test_bridge", FEP_BRIDGE_CATEGORY_SWARM,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    
    const fep_bridge_entry_t* entry = fep_orchestrator_get_bridge(orchestrator, bridge_id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->bridge_id, bridge_id);
    EXPECT_STREQ(entry->bridge_name, "test_bridge");
    EXPECT_EQ(entry->category, FEP_BRIDGE_CATEGORY_SWARM);
}

TEST_F(FepOrchestratorTest, GetBridgeNonExistentReturnsNull) {
    const fep_bridge_entry_t* entry = fep_orchestrator_get_bridge(orchestrator, 9999);
    EXPECT_EQ(entry, nullptr);
}

/* ============================================================================
 * Update Tests
 * ============================================================================ */

TEST_F(FepOrchestratorTest, UpdateWhenNotRunningDoesNothing) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    
    /* Orchestrator is stopped */
    uint64_t now = nimcp_platform_time_monotonic_ms();
    int updated = fep_orchestrator_update(orchestrator, now);
    EXPECT_EQ(updated, 0);
    EXPECT_EQ(g_mock_update_count, 0);
}

TEST_F(FepOrchestratorTest, UpdateCallsBridges) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    
    fep_orchestrator_start(orchestrator);
    
    /* Wait for update interval and call update */
    uint64_t start = nimcp_platform_time_monotonic_ms();
    uint64_t now = start + FEP_UPDATE_INTERVAL_COGNITIVE + 1;
    
    int updated = fep_orchestrator_update(orchestrator, now);
    EXPECT_GE(updated, 1);
    EXPECT_EQ(g_mock_update_count, 1);
}

TEST_F(FepOrchestratorTest, UpdateRespectsInterval) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    
    fep_orchestrator_start(orchestrator);
    
    uint64_t start = nimcp_platform_time_monotonic_ms();
    
    /* First update - triggers immediately after start (interval passed) */
    int updated = fep_orchestrator_update(orchestrator, start + FEP_UPDATE_INTERVAL_COGNITIVE + 1);
    EXPECT_GE(updated, 1);
    
    /* Second update too soon - should not trigger */
    g_mock_update_count = 0;
    updated = fep_orchestrator_update(orchestrator, start + FEP_UPDATE_INTERVAL_COGNITIVE + 2);
    EXPECT_EQ(updated, 0);
    EXPECT_EQ(g_mock_update_count, 0);
}

TEST_F(FepOrchestratorTest, ForceUpdateAllUpdatesBridges) {
    int bridges[3];
    
    for (int i = 0; i < 3; i++) {
        bridges[i] = i;
        fep_orchestrator_register_bridge(
            orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
            &bridges[i], mock_bridge_update, nullptr, nullptr
        );
    }
    
    fep_orchestrator_start(orchestrator);
    
    int updated = fep_orchestrator_force_update_all(orchestrator);
    EXPECT_EQ(updated, 3);
    EXPECT_EQ(g_mock_update_count, 3);
}

TEST_F(FepOrchestratorTest, UpdateCategoryUpdatesOnlyMatchingBridges) {
    int cognitive_bridge = 1;
    int swarm_bridge = 2;
    
    fep_orchestrator_register_bridge(
        orchestrator, "cognitive", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &cognitive_bridge, mock_bridge_update, nullptr, nullptr
    );
    fep_orchestrator_register_bridge(
        orchestrator, "swarm", FEP_BRIDGE_CATEGORY_SWARM,
        &swarm_bridge, mock_bridge_update, nullptr, nullptr
    );
    
    fep_orchestrator_start(orchestrator);
    
    uint64_t now = nimcp_platform_time_monotonic_ms() + 100;
    int updated = fep_orchestrator_update_category(orchestrator, FEP_BRIDGE_CATEGORY_COGNITIVE, now);
    EXPECT_EQ(updated, 1);
    EXPECT_EQ(g_mock_update_count, 1);
}

TEST_F(FepOrchestratorTest, UpdateBridgeUpdatesSpecificBridge) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    
    fep_orchestrator_start(orchestrator);
    
    int result = fep_orchestrator_update_bridge(orchestrator, bridge_id);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(g_mock_update_count, 1);
}

TEST_F(FepOrchestratorTest, UpdateBridgeFailedUpdatesTrackErrors) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    
    fep_orchestrator_start(orchestrator);
    
    /* Make update fail */
    g_mock_update_return = -1;
    
    int result = fep_orchestrator_update_bridge(orchestrator, bridge_id);
    EXPECT_EQ(result, NIMCP_ERROR_OPERATION_FAILED);
    EXPECT_EQ(orchestrator->stats.update_errors, 1u);
}

/* ============================================================================
 * Category Configuration Tests
 * ============================================================================ */

TEST_F(FepOrchestratorTest, SetUpdateIntervalSucceeds) {
    int result = fep_orchestrator_set_update_interval(
        orchestrator, FEP_BRIDGE_CATEGORY_COGNITIVE, 100
    );
    EXPECT_EQ(result, 0);
    EXPECT_EQ(orchestrator->config.categories[FEP_BRIDGE_CATEGORY_COGNITIVE].update_interval_ms, 100u);
}

TEST_F(FepOrchestratorTest, SetCategoryEnabledSucceeds) {
    int result = fep_orchestrator_set_category_enabled(
        orchestrator, FEP_BRIDGE_CATEGORY_PLASTICITY, false
    );
    EXPECT_EQ(result, 0);
    EXPECT_FALSE(orchestrator->config.categories[FEP_BRIDGE_CATEGORY_PLASTICITY].enabled);
}

TEST_F(FepOrchestratorTest, GetCategoryConfigSucceeds) {
    fep_category_config_t cat_cfg;
    int result = fep_orchestrator_get_category_config(
        orchestrator, FEP_BRIDGE_CATEGORY_COGNITIVE, &cat_cfg
    );
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(cat_cfg.enabled);
    EXPECT_EQ(cat_cfg.update_interval_ms, FEP_UPDATE_INTERVAL_COGNITIVE);
}

/* ============================================================================
 * Statistics Tests
 * ============================================================================ */

TEST_F(FepOrchestratorTest, GetStatsSucceeds) {
    fep_orchestrator_stats_t stats;
    int result = fep_orchestrator_get_stats(orchestrator, &stats);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(stats.total_bridges, 0u);
}

TEST_F(FepOrchestratorTest, ResetStatsPreservesBridgeCounts) {
    int mock_bridge = 42;
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, nullptr
    );
    
    fep_orchestrator_start(orchestrator);
    fep_orchestrator_force_update_all(orchestrator);
    
    EXPECT_GT(orchestrator->stats.total_bridge_updates, 0u);
    
    fep_orchestrator_reset_stats(orchestrator);
    
    EXPECT_EQ(orchestrator->stats.total_bridge_updates, 0u);
    EXPECT_EQ(orchestrator->stats.total_bridges, 1u);  /* Preserved */
}

TEST_F(FepOrchestratorTest, GetLoadReturnsZeroWithoutBudget) {
    float load = fep_orchestrator_get_load(orchestrator);
    EXPECT_EQ(load, 0.0f);  /* No budget set */
}

/* ============================================================================
 * Integration Tests
 * ============================================================================ */

TEST_F(FepOrchestratorTest, ConnectBrainImmuneSucceeds) {
    /* Note: We don't have a real brain_immune_system_t for unit tests */
    int result = fep_orchestrator_connect_brain_immune(orchestrator, nullptr);
    EXPECT_EQ(result, 0);
    EXPECT_FALSE(orchestrator->immune_connected);  /* NULL immune = not connected */
}

TEST_F(FepOrchestratorTest, DisconnectBrainImmuneSucceeds) {
    int result = fep_orchestrator_disconnect_brain_immune(orchestrator);
    EXPECT_EQ(result, 0);
}

TEST_F(FepOrchestratorTest, ConnectBioAsyncSucceeds) {
    /* Bio-async may not be initialized in unit test environment */
    int result = fep_orchestrator_connect_bio_async(orchestrator);
    EXPECT_EQ(result, 0);  /* Should succeed even if router not initialized */
}

TEST_F(FepOrchestratorTest, DisconnectBioAsyncSucceeds) {
    int result = fep_orchestrator_disconnect_bio_async(orchestrator);
    EXPECT_EQ(result, 0);
}

/* ============================================================================
 * String Conversion Tests
 * ============================================================================ */

TEST_F(FepOrchestratorTest, CategoryToStringIsValid) {
    EXPECT_STREQ(fep_bridge_category_to_string(FEP_BRIDGE_CATEGORY_COGNITIVE), "cognitive");
    EXPECT_STREQ(fep_bridge_category_to_string(FEP_BRIDGE_CATEGORY_SWARM), "swarm");
    EXPECT_STREQ(fep_bridge_category_to_string(FEP_BRIDGE_CATEGORY_SECURITY), "security");
    EXPECT_STREQ(fep_bridge_category_to_string(FEP_BRIDGE_CATEGORY_PLASTICITY), "plasticity");
    EXPECT_STREQ(fep_bridge_category_to_string((fep_bridge_category_t)999), "unknown");
}

TEST_F(FepOrchestratorTest, StateToStringIsValid) {
    EXPECT_STREQ(fep_orchestrator_state_to_string(FEP_ORCHESTRATOR_STOPPED), "stopped");
    EXPECT_STREQ(fep_orchestrator_state_to_string(FEP_ORCHESTRATOR_RUNNING), "running");
    EXPECT_STREQ(fep_orchestrator_state_to_string(FEP_ORCHESTRATOR_PAUSED), "paused");
    EXPECT_STREQ(fep_orchestrator_state_to_string((fep_orchestrator_state_t)999), "unknown");
}

/* ============================================================================
 * Edge Case Tests
 * ============================================================================ */

TEST_F(FepOrchestratorTest, DisabledBridgeNotUpdated) {
    int mock_bridge = 42;
    uint32_t bridge_id = 0;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_COGNITIVE,
        &mock_bridge, mock_bridge_update, nullptr, &bridge_id
    );
    
    /* Disable the bridge */
    fep_orchestrator_set_bridge_enabled(orchestrator, bridge_id, false);
    
    fep_orchestrator_start(orchestrator);
    fep_orchestrator_force_update_all(orchestrator);
    
    EXPECT_EQ(g_mock_update_count, 0);  /* Bridge was disabled */
}

TEST_F(FepOrchestratorTest, DisabledCategoryNotUpdated) {
    int mock_bridge = 42;
    
    fep_orchestrator_register_bridge(
        orchestrator, "test", FEP_BRIDGE_CATEGORY_PLASTICITY,
        &mock_bridge, mock_bridge_update, nullptr, nullptr
    );
    
    /* Disable the category */
    fep_orchestrator_set_category_enabled(orchestrator, FEP_BRIDGE_CATEGORY_PLASTICITY, false);
    
    fep_orchestrator_start(orchestrator);
    
    uint64_t now = nimcp_platform_time_monotonic_ms() + 2000;  /* Well past interval */
    fep_orchestrator_update(orchestrator, now);
    
    EXPECT_EQ(g_mock_update_count, 0);  /* Category was disabled */
}

TEST_F(FepOrchestratorTest, DoubleStartIsSafe) {
    fep_orchestrator_start(orchestrator);
    int result = fep_orchestrator_start(orchestrator);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_RUNNING);
}

TEST_F(FepOrchestratorTest, DoubleStopIsSafe) {
    fep_orchestrator_start(orchestrator);
    fep_orchestrator_stop(orchestrator);
    int result = fep_orchestrator_stop(orchestrator);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(fep_orchestrator_get_state(orchestrator), FEP_ORCHESTRATOR_STOPPED);
}

