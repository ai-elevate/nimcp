//=============================================================================
// test_physics_bridges_integration.cpp - Physics Bridges Integration Tests
//=============================================================================
/**
 * @file test_physics_bridges_integration.cpp
 * @brief Integration tests for all physics layer bridges
 *
 * Tests bidirectional integration between physics modules and:
 * - Bio-async messaging system
 * - QMC (Quantum Monte Carlo)
 * - FFT spectral analysis
 * - Immune system
 * - Knowledge Graph
 * - Security (BBB)
 * - LNN (Liquid Neural Networks)
 * - Hypothalamus homeostasis
 * - Cognitive layer
 * - Training layer
 * - Swarm distribution
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

// Physics headers already have extern "C" guards internally
#include "physics/biophysics/nimcp_hodgkin_huxley.h"
#include "physics/thermodynamics/nimcp_thermodynamics.h"
#include "physics/ephaptic/nimcp_ephaptic.h"
#include "physics/bridges/nimcp_physics_lnn_bridge.h"
#include "physics/bridges/nimcp_physics_hypothalamus_bridge.h"
#include "physics/bridges/nimcp_physics_cognitive_bridge.h"
#include "physics/bridges/nimcp_physics_training_bridge.h"
#include "physics/bridges/nimcp_physics_swarm_bridge.h"
#include "physics/bridges/nimcp_physics_immune_bridge.h"
#include "physics/bridges/nimcp_physics_brain_init.h"

//=============================================================================
// Test Fixture
//=============================================================================

class PhysicsBridgesIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize HH population
        nimcp_hh_config_t hh_config = {
            .g_Na = 120.0f,
            .g_K = 36.0f,
            .g_L = 0.3f,
            .g_Ca_L = 0.0f,
            .g_Ca_T = 0.0f,
            .g_K_Ca = 0.0f,
            .g_K_A = 0.0f,
            .g_H = 0.0f,
            .E_Na = 50.0f,
            .E_K = -77.0f,
            .E_L = -54.4f,
            .E_Ca = 120.0f,
            .E_H = -30.0f,
            .C_m = 1.0f,
            .V_rest = -65.0f,
            .temperature = 37.0f,
            .surface_area = 0.01f,
            .length = 100.0f,
            .diameter = 10.0f,
            .enable_calcium = false,
            .enable_adaptation = false,
            .enable_h_current = false,
            .dt_max = 0.1f,
            .adaptive_dt = false
        };

        hh_initialized_ = (nimcp_hh_population_create(&hh_pop_, 10, &hh_config) == NIMCP_SUCCESS);

        // Initialize thermodynamics
        nimcp_thermo_config_t thermo_config = nimcp_thermo_default_config();
        thermo_config.temperature_k = 310.15f;  // 37°C
        thermo_initialized_ = (nimcp_thermo_init(&thermo_state_, &thermo_config) == NIMCP_SUCCESS);

        // Initialize ephaptic
        nimcp_ephaptic_config_t eph_config = nimcp_ephaptic_default_config();
        ephaptic_initialized_ = (nimcp_ephaptic_init(&ephaptic_, &eph_config) == NIMCP_SUCCESS);
    }

    void TearDown() override {
        if (hh_initialized_) {
            nimcp_hh_population_destroy(&hh_pop_);
        }
        if (thermo_initialized_) {
            nimcp_thermo_destroy(&thermo_state_);
        }
        if (ephaptic_initialized_) {
            nimcp_ephaptic_destroy(&ephaptic_);
        }
    }

    nimcp_hh_population_t hh_pop_;
    nimcp_thermodynamic_state_t thermo_state_;
    nimcp_ephaptic_system_t ephaptic_;
    bool hh_initialized_ = false;
    bool thermo_initialized_ = false;
    bool ephaptic_initialized_ = false;
};

//=============================================================================
// Physics LNN Bridge Tests
//=============================================================================

TEST_F(PhysicsBridgesIntegrationTest, LNNBridgeCreation) {
    physics_lnn_config_t config;
    ASSERT_EQ(physics_lnn_default_config(&config), 0);

    physics_lnn_bridge_t* bridge = physics_lnn_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    physics_lnn_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, LNNBridgeConnectionAndReset) {
    physics_lnn_bridge_t* bridge = physics_lnn_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Connect HH (LNN would need separate setup)
    if (hh_initialized_) {
        EXPECT_EQ(physics_lnn_connect_hh(bridge, &hh_pop_), 0);
    }

    // Reset should succeed
    EXPECT_EQ(physics_lnn_reset(bridge), 0);

    physics_lnn_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, LNNBridgeSpikeEncoding) {
    physics_lnn_bridge_t* bridge = physics_lnn_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    if (hh_initialized_) {
        physics_lnn_connect_hh(bridge, &hh_pop_);
    }

    // Register some spikes
    EXPECT_EQ(physics_lnn_register_spike(bridge, 0, 10.0f), 0);
    EXPECT_EQ(physics_lnn_register_spike(bridge, 1, 12.0f), 0);
    EXPECT_EQ(physics_lnn_register_spike(bridge, 2, 15.0f), 0);

    // Encode spikes
    physics_lnn_encoded_t encoded;
    EXPECT_EQ(physics_lnn_encode_spikes(bridge, 20.0f, &encoded), 0);

    // Should have encoded currents
    EXPECT_NE(encoded.currents, nullptr);
    EXPECT_GT(encoded.total_spikes, 0UL);

    // Get stats
    physics_lnn_stats_t stats;
    EXPECT_EQ(physics_lnn_get_stats(bridge, &stats), 0);
    EXPECT_EQ(stats.spikes_encoded, 3UL);

    physics_lnn_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, LNNBridgeTemperatureCoupling) {
    physics_lnn_config_t config;
    physics_lnn_default_config(&config);
    config.enable_temp_coupling = true;
    config.q10 = 2.3f;
    config.temp_ref = 25.0f;

    physics_lnn_bridge_t* bridge = physics_lnn_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    // Set temperature above reference
    EXPECT_EQ(physics_lnn_set_temperature(bridge, 35.0f), 0);

    // Tau scale should be less than 1 (faster at higher temp)
    float tau_scale = physics_lnn_get_tau_scale(bridge);
    EXPECT_LT(tau_scale, 1.0f);
    EXPECT_GT(tau_scale, 0.0f);

    // Set temperature below reference
    EXPECT_EQ(physics_lnn_set_temperature(bridge, 15.0f), 0);

    // Tau scale should be greater than 1 (slower at lower temp)
    tau_scale = physics_lnn_get_tau_scale(bridge);
    EXPECT_GT(tau_scale, 1.0f);

    physics_lnn_bridge_destroy(bridge);
}

//=============================================================================
// Physics Hypothalamus Bridge Tests
//=============================================================================

TEST_F(PhysicsBridgesIntegrationTest, HypothalamusBridgeCreation) {
    physics_hypo_config_t config;
    ASSERT_EQ(physics_hypo_default_config(&config), 0);

    physics_hypo_bridge_t* bridge = physics_hypo_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    physics_hypo_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, HypothalamusBridgeTemperatureReporting) {
    physics_hypo_bridge_t* bridge = physics_hypo_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Report temperature
    EXPECT_EQ(physics_hypo_report_temperature(bridge, 38.5f), 0);

    // Get state
    physics_hypo_state_t state;
    EXPECT_EQ(physics_hypo_get_state(bridge, &state), 0);
    EXPECT_FLOAT_EQ(state.temperature, 38.5f);

    // Run update to compute modulation (thermoregulation is calculated in update)
    EXPECT_EQ(physics_hypo_update(bridge, 0.01f), 0);

    // Get thermoregulation status
    physics_hypo_thermo_dir_t dir;
    float strength;
    EXPECT_EQ(physics_hypo_get_thermo_status(bridge, &dir, &strength), 0);

    // Above setpoint should trigger cooling
    EXPECT_EQ(dir, PHYSICS_HYPO_THERMO_COOLING);
    EXPECT_GT(strength, 0.0f);

    physics_hypo_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, HypothalamusBridgeCircadian) {
    physics_hypo_config_t config;
    physics_hypo_default_config(&config);
    config.enable_circadian = true;

    physics_hypo_bridge_t* bridge = physics_hypo_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    // Set circadian phase to midnight
    EXPECT_EQ(physics_hypo_set_circadian_phase(bridge, 0.0f), 0);

    float mult_midnight = physics_hypo_get_circadian_multiplier(bridge);

    // Set to noon
    EXPECT_EQ(physics_hypo_set_circadian_phase(bridge, 12.0f), 0);

    float mult_noon = physics_hypo_get_circadian_multiplier(bridge);

    // Noon should have higher multiplier than midnight
    EXPECT_GT(mult_noon, mult_midnight);

    physics_hypo_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, HypothalamusBridgePhysicsConnection) {
    physics_hypo_bridge_t* bridge = physics_hypo_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Connect physics modules
    if (thermo_initialized_ && hh_initialized_) {
        EXPECT_EQ(physics_hypo_connect_physics(bridge, &thermo_state_, &hh_pop_), 0);
    }

    // Update should work
    EXPECT_EQ(physics_hypo_update(bridge, 10.0f), 0);

    // Get modulation
    physics_hypo_modulation_t mod;
    EXPECT_EQ(physics_hypo_get_modulation(bridge, &mod), 0);

    physics_hypo_bridge_destroy(bridge);
}

//=============================================================================
// Physics Cognitive Bridge Tests
//=============================================================================

TEST_F(PhysicsBridgesIntegrationTest, CognitiveBridgeCreation) {
    physics_cog_config_t config;
    ASSERT_EQ(physics_cog_default_config(&config), 0);

    physics_cog_bridge_t* bridge = physics_cog_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    physics_cog_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, CognitiveBridgeCapacityComputation) {
    physics_cog_bridge_t* bridge = physics_cog_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Connect physics
    if (thermo_initialized_ && hh_initialized_ && ephaptic_initialized_) {
        EXPECT_EQ(physics_cog_connect_physics(bridge, &thermo_state_, &hh_pop_, &ephaptic_), 0);
    }

    // Compute capacity
    physics_cog_capacity_t capacity;
    EXPECT_EQ(physics_cog_compute_capacity(bridge, &capacity), 0);

    // All factors should be positive
    EXPECT_GT(capacity.speed_factor, 0.0f);
    EXPECT_GT(capacity.capacity_factor, 0.0f);
    EXPECT_GT(capacity.attention_factor, 0.0f);
    EXPECT_GT(capacity.binding_factor, 0.0f);
    EXPECT_GT(capacity.control_factor, 0.0f);
    EXPECT_GT(capacity.overall_efficiency, 0.0f);

    physics_cog_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, CognitiveBridgeAttentionModulation) {
    physics_cog_bridge_t* bridge = physics_cog_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Set low attention
    EXPECT_EQ(physics_cog_set_attention(bridge, 0.2f), 0);

    physics_cog_feedback_t feedback;
    EXPECT_EQ(physics_cog_get_feedback(bridge, &feedback), 0);
    EXPECT_FLOAT_EQ(feedback.attention_level, 0.2f);

    // Set high attention
    EXPECT_EQ(physics_cog_set_attention(bridge, 0.9f), 0);

    EXPECT_EQ(physics_cog_get_feedback(bridge, &feedback), 0);
    EXPECT_FLOAT_EQ(feedback.attention_level, 0.9f);

    physics_cog_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, CognitiveBridgeImpairmentDetection) {
    physics_cog_config_t config;
    physics_cog_default_config(&config);
    config.atp_impairment_threshold = 0.5f;

    physics_cog_bridge_t* bridge = physics_cog_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    // Initially should not be impaired
    EXPECT_FALSE(physics_cog_is_impaired(bridge));

    // Get impairment reasons (should be none)
    uint32_t reasons = physics_cog_get_impairment(bridge);
    EXPECT_EQ(reasons, (uint32_t)PHYSICS_COG_IMPAIR_NONE);

    physics_cog_bridge_destroy(bridge);
}

//=============================================================================
// Physics Training Bridge Tests
//=============================================================================

TEST_F(PhysicsBridgesIntegrationTest, TrainingBridgeCreation) {
    physics_train_config_t config;
    ASSERT_EQ(physics_train_default_config(&config), 0);

    physics_train_bridge_t* bridge = physics_train_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    physics_train_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, TrainingBridgePlasticityGating) {
    physics_train_config_t config;
    physics_train_default_config(&config);
    config.enable_metabolic_gating = true;
    config.atp_threshold = 0.5f;

    physics_train_bridge_t* bridge = physics_train_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    // Connect thermodynamics for ATP
    if (thermo_initialized_) {
        EXPECT_EQ(physics_train_connect_physics(bridge, &thermo_state_, nullptr), 0);
    }

    // Get modulation
    physics_train_modulation_t mod;
    EXPECT_EQ(physics_train_get_modulation(bridge, &mod), 0);

    // With full ATP, plasticity should be enabled
    EXPECT_TRUE(mod.plasticity_enabled);
    EXPECT_GT(mod.learning_rate_scale, 0.0f);

    physics_train_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, TrainingBridgeFeedback) {
    physics_train_bridge_t* bridge = physics_train_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Report training feedback
    physics_train_feedback_t feedback;
    feedback.update_magnitude = 0.1f;
    feedback.synapses_updated = 100;
    feedback.learning_signal = 0.5f;
    feedback.timestamp_ms = 1000.0f;

    EXPECT_EQ(physics_train_report_feedback(bridge, &feedback), 0);

    // Get stats
    physics_train_stats_t stats;
    EXPECT_EQ(physics_train_get_stats(bridge, &stats), 0);
    EXPECT_GT(stats.total_learning_cost, 0.0f);

    physics_train_bridge_destroy(bridge);
}

//=============================================================================
// Physics Swarm Bridge Tests
//=============================================================================

TEST_F(PhysicsBridgesIntegrationTest, SwarmBridgeCreation) {
    physics_swarm_config_t config;
    ASSERT_EQ(physics_swarm_default_config(&config), 0);

    physics_swarm_bridge_t* bridge = physics_swarm_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    physics_swarm_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, SwarmBridgePartitionManagement) {
    physics_swarm_config_t config;
    physics_swarm_default_config(&config);
    config.node_id = 0;
    config.total_nodes = 4;

    physics_swarm_bridge_t* bridge = physics_swarm_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    // Register a partition
    physics_swarm_partition_t partition;
    partition.partition_id = 0;
    partition.node_id = 0;
    partition.neuron_start = 0;
    partition.neuron_count = 1000;
    partition.is_boundary = true;

    EXPECT_EQ(physics_swarm_register_partition(bridge, &partition), 0);

    // Get partition back
    physics_swarm_partition_t retrieved;
    EXPECT_EQ(physics_swarm_get_partition(bridge, 0, &retrieved), 0);
    EXPECT_EQ(retrieved.partition_id, 0U);
    EXPECT_EQ(retrieved.neuron_count, 1000U);

    physics_swarm_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, SwarmBridgeStateSerialization) {
    physics_swarm_bridge_t* bridge = physics_swarm_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Serialize state
    physics_swarm_state_t state;
    EXPECT_EQ(physics_swarm_serialize_state(bridge, &state), 0);

    // State should have valid values
    EXPECT_GT(state.temperature, 0.0f);
    EXPECT_GE(state.atp_level, 0.0f);
    EXPECT_LE(state.atp_level, 1.0f);

    // Receive state from another node
    physics_swarm_state_t remote_state;
    remote_state.temperature = 36.5f;
    remote_state.atp_level = 0.9f;
    remote_state.coherence = 0.7f;
    remote_state.total_rate = 50.0f;
    remote_state.active_partitions = 2;
    remote_state.sim_time_ms = 1000.0f;

    EXPECT_EQ(physics_swarm_receive_state(bridge, 1, &remote_state), 0);

    physics_swarm_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, SwarmBridgeSyncTiming) {
    physics_swarm_config_t config;
    physics_swarm_default_config(&config);
    config.sync_interval_ms = 10.0f;

    physics_swarm_bridge_t* bridge = physics_swarm_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    // Initially no sync needed
    EXPECT_FALSE(physics_swarm_needs_sync(bridge));

    // Advance time past sync interval
    for (int i = 0; i < 15; i++) {
        physics_swarm_update(bridge, 1.0f);
    }

    // Get stats
    physics_swarm_stats_t stats;
    EXPECT_EQ(physics_swarm_get_stats(bridge, &stats), 0);
    EXPECT_GT(stats.sync_count, 0UL);

    physics_swarm_bridge_destroy(bridge);
}

//=============================================================================
// Physics Immune Bridge Tests
//=============================================================================

TEST_F(PhysicsBridgesIntegrationTest, ImmuneBridgeCreation) {
    physics_immune_config_t config;
    ASSERT_EQ(physics_immune_default_config(&config), 0);

    physics_immune_bridge_t* bridge = physics_immune_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    physics_immune_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, ImmuneBridgePhysicsConnection) {
    physics_immune_bridge_t* bridge = physics_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Connect physics modules
    if (thermo_initialized_ && hh_initialized_ && ephaptic_initialized_) {
        EXPECT_EQ(physics_immune_connect_physics(bridge, &thermo_state_, &hh_pop_, &ephaptic_), 0);
    }

    physics_immune_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, ImmuneBridgeTemperatureCheck) {
    physics_immune_config_t config;
    physics_immune_default_config(&config);
    config.monitor_temperature = true;

    physics_immune_bridge_t* bridge = physics_immune_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    // Check temperature - should return no interaction for normal temp
    physics_immune_interaction_t interaction = physics_immune_check_temperature(bridge);
    // At initialization, temperature is normal
    EXPECT_EQ(interaction, PHYSICS_IMMUNE_INTERACTION_NONE);

    physics_immune_bridge_destroy(bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, ImmuneBridgeCytokineModulation) {
    physics_immune_bridge_t* bridge = physics_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    // Receive cytokine signal
    EXPECT_EQ(physics_immune_receive_cytokine(bridge, 0, 0.5f), 0);

    // Get modulation
    physics_immune_modulation_t mod;
    EXPECT_EQ(physics_immune_get_modulation(bridge, &mod), 0);

    // Modulation should reflect cytokine effect
    // Cytokine type 0 reduces Na conductance
    EXPECT_LE(mod.g_na_modifier, 1.0f);

    physics_immune_bridge_destroy(bridge);
}

//=============================================================================
// Physics Brain Init Tests
//=============================================================================

TEST_F(PhysicsBridgesIntegrationTest, BrainInitDefaultConfig) {
    physics_init_config_t config;
    ASSERT_EQ(physics_init_default_config(&config), 0);

    // Check default values
    EXPECT_GT(config.default_hh_population_size, 0U);
    EXPECT_TRUE(config.enable_ephaptic);
    EXPECT_TRUE(config.enable_qmc);
    EXPECT_TRUE(config.enable_fft);
}

TEST_F(PhysicsBridgesIntegrationTest, PhysicsVersionString) {
    const char* version = physics_get_version();
    ASSERT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0U);
}

//=============================================================================
// Cross-Bridge Integration Tests
//=============================================================================

TEST_F(PhysicsBridgesIntegrationTest, MultipleBridgesSimultaneous) {
    // Create multiple bridges simultaneously
    physics_lnn_bridge_t* lnn_bridge = physics_lnn_bridge_create(nullptr);
    physics_hypo_bridge_t* hypo_bridge = physics_hypo_bridge_create(nullptr);
    physics_cog_bridge_t* cog_bridge = physics_cog_bridge_create(nullptr);
    physics_train_bridge_t* train_bridge = physics_train_bridge_create(nullptr);
    physics_swarm_bridge_t* swarm_bridge = physics_swarm_bridge_create(nullptr);

    ASSERT_NE(lnn_bridge, nullptr);
    ASSERT_NE(hypo_bridge, nullptr);
    ASSERT_NE(cog_bridge, nullptr);
    ASSERT_NE(train_bridge, nullptr);
    ASSERT_NE(swarm_bridge, nullptr);

    // Connect to shared physics modules
    if (hh_initialized_) {
        physics_lnn_connect_hh(lnn_bridge, &hh_pop_);
    }
    if (thermo_initialized_ && hh_initialized_) {
        physics_hypo_connect_physics(hypo_bridge, &thermo_state_, &hh_pop_);
        physics_train_connect_physics(train_bridge, &thermo_state_, &hh_pop_);
    }
    if (thermo_initialized_ && hh_initialized_ && ephaptic_initialized_) {
        physics_cog_connect_physics(cog_bridge, &thermo_state_, &hh_pop_, &ephaptic_);
    }

    // Update all bridges
    float dt = 1.0f;
    physics_lnn_update(lnn_bridge, dt);
    physics_hypo_update(hypo_bridge, dt);
    physics_cog_update(cog_bridge, dt);
    physics_train_update(train_bridge, dt);
    physics_swarm_update(swarm_bridge, dt);

    // Clean up
    physics_lnn_bridge_destroy(lnn_bridge);
    physics_hypo_bridge_destroy(hypo_bridge);
    physics_cog_bridge_destroy(cog_bridge);
    physics_train_bridge_destroy(train_bridge);
    physics_swarm_bridge_destroy(swarm_bridge);
}

TEST_F(PhysicsBridgesIntegrationTest, BridgeUpdateLoop) {
    // Simulate a realistic update loop with multiple bridges
    physics_hypo_bridge_t* hypo = physics_hypo_bridge_create(nullptr);
    physics_cog_bridge_t* cog = physics_cog_bridge_create(nullptr);

    ASSERT_NE(hypo, nullptr);
    ASSERT_NE(cog, nullptr);

    // Simulate 100ms of updates
    float dt = 1.0f;
    for (int i = 0; i < 100; i++) {
        // Update bridges
        physics_hypo_update(hypo, dt);
        physics_cog_update(cog, dt);

        // Get outputs
        physics_hypo_modulation_t hypo_mod;
        physics_hypo_get_modulation(hypo, &hypo_mod);

        physics_cog_capacity_t cog_cap;
        physics_cog_compute_capacity(cog, &cog_cap);

        // Verify outputs are valid
        EXPECT_GE(hypo_mod.q10_modifier, 0.0f);
        EXPECT_GE(cog_cap.overall_efficiency, 0.0f);
    }

    // Check statistics
    physics_hypo_stats_t hypo_stats;
    physics_hypo_get_stats(hypo, &hypo_stats);
    EXPECT_GT(hypo_stats.physics_to_hypo_count, 0UL);

    physics_cog_stats_t cog_stats;
    physics_cog_get_stats(cog, &cog_stats);
    EXPECT_GT(cog_stats.physics_to_cog_count, 0UL);

    physics_hypo_bridge_destroy(hypo);
    physics_cog_bridge_destroy(cog);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
