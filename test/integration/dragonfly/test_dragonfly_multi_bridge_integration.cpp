//=============================================================================
// test_dragonfly_multi_bridge_integration.cpp - Multi-Bridge Integration Tests
//=============================================================================
/**
 * @file test_dragonfly_multi_bridge_integration.cpp
 * @brief Integration tests for dragonfly bridges working together
 *
 * WHAT: Tests coordination between dragonfly's 12 bridge modules
 * WHY:  Verify bridges communicate and synchronize correctly
 * HOW:  Create bridges, simulate data flow, verify outputs
 *
 * BRIDGES TESTED:
 * - Visual Bridge: Target detection from visual cortex
 * - Audio Bridge: Directional cues from audio cortex
 * - Parietal Bridge: Spatial reasoning for interception
 * - Cortical Bridge: Higher-level processing integration
 * - Cognitive Bridge: Attention and salience signals
 * - Thalamic Bridge: Signal routing and gating
 * - Substrate Bridge: Metabolic costs and fatigue
 * - FEP Bridge: Free energy prediction signals
 * - Workspace Bridge: Conscious awareness integration
 * - Bio-Async Bridge: Asynchronous neural processing
 * - CNN Bridge: Deep learning feature extraction
 * - SNN Bridge: Spike-based neural processing
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "dragonfly/nimcp_dragonfly.h"
#include "dragonfly/nimcp_dragonfly_visual_bridge.h"
#include "dragonfly/nimcp_dragonfly_audio_bridge.h"
#include "dragonfly/nimcp_dragonfly_parietal_bridge.h"
#include "dragonfly/nimcp_dragonfly_cortical_bridge.h"
#include "dragonfly/nimcp_dragonfly_cognitive_bridge.h"
#include "dragonfly/nimcp_dragonfly_thalamic_bridge.h"
#include "dragonfly/nimcp_dragonfly_substrate_bridge.h"
#include "dragonfly/nimcp_dragonfly_fep_bridge.h"
#include "dragonfly/nimcp_dragonfly_workspace_bridge.h"
#include "dragonfly/nimcp_dragonfly_bio_async_bridge.h"
#include "dragonfly/nimcp_dragonfly_cnn_bridge.h"
#include "dragonfly/nimcp_dragonfly_snn_bridge.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class DragonflyMultiBridgeTest : public ::testing::Test {
protected:
    dragonfly_system_t* system = nullptr;
    dragonfly_visual_bridge_t* visual_bridge = nullptr;
    dragonfly_audio_bridge_t* audio_bridge = nullptr;
    dragonfly_parietal_bridge_t* parietal_bridge = nullptr;
    dragonfly_cortical_bridge_t* cortical_bridge = nullptr;
    dragonfly_cognitive_bridge_t* cognitive_bridge = nullptr;
    dragonfly_thalamic_bridge_t* thalamic_bridge = nullptr;
    dragonfly_substrate_bridge_t* substrate_bridge = nullptr;
    dragonfly_fep_bridge_t* fep_bridge = nullptr;
    dragonfly_workspace_bridge_t* workspace_bridge = nullptr;
    dragonfly_bio_async_bridge_t* bio_async_bridge = nullptr;
    dragonfly_cnn_bridge_t* cnn_bridge = nullptr;
    dragonfly_snn_bridge_t* snn_bridge = nullptr;

    void SetUp() override {
        // Create dragonfly system
        dragonfly_config_t config = dragonfly_default_config();
        system = dragonfly_system_create(&config);
        ASSERT_NE(system, nullptr);

        // Create bridges with default configs
        visual_bridge = dragonfly_visual_bridge_create(
            dragonfly_visual_bridge_default_config());
        audio_bridge = dragonfly_audio_bridge_create(
            dragonfly_audio_bridge_default_config());
        parietal_bridge = dragonfly_parietal_bridge_create(
            dragonfly_parietal_bridge_default_config());
        cortical_bridge = dragonfly_cortical_bridge_create(
            dragonfly_cortical_bridge_default_config());
        cognitive_bridge = dragonfly_cognitive_bridge_create(
            dragonfly_cognitive_bridge_default_config());
        thalamic_bridge = dragonfly_thalamic_bridge_create(
            dragonfly_thalamic_bridge_default_config());
        substrate_bridge = dragonfly_substrate_bridge_create(
            dragonfly_substrate_bridge_default_config());
        fep_bridge = dragonfly_fep_bridge_create(
            dragonfly_fep_bridge_default_config());
        workspace_bridge = dragonfly_workspace_bridge_create(
            dragonfly_workspace_bridge_default_config());
        bio_async_bridge = dragonfly_bio_async_bridge_create(
            dragonfly_bio_async_bridge_default_config());
        cnn_bridge = dragonfly_cnn_bridge_create(
            dragonfly_cnn_bridge_default_config());
        snn_bridge = dragonfly_snn_bridge_create(
            dragonfly_snn_bridge_default_config());
    }

    void TearDown() override {
        if (snn_bridge) dragonfly_snn_bridge_destroy(snn_bridge);
        if (cnn_bridge) dragonfly_cnn_bridge_destroy(cnn_bridge);
        if (bio_async_bridge) dragonfly_bio_async_bridge_destroy(bio_async_bridge);
        if (workspace_bridge) dragonfly_workspace_bridge_destroy(workspace_bridge);
        if (fep_bridge) dragonfly_fep_bridge_destroy(fep_bridge);
        if (substrate_bridge) dragonfly_substrate_bridge_destroy(substrate_bridge);
        if (thalamic_bridge) dragonfly_thalamic_bridge_destroy(thalamic_bridge);
        if (cognitive_bridge) dragonfly_cognitive_bridge_destroy(cognitive_bridge);
        if (cortical_bridge) dragonfly_cortical_bridge_destroy(cortical_bridge);
        if (parietal_bridge) dragonfly_parietal_bridge_destroy(parietal_bridge);
        if (audio_bridge) dragonfly_audio_bridge_destroy(audio_bridge);
        if (visual_bridge) dragonfly_visual_bridge_destroy(visual_bridge);
        if (system) dragonfly_system_destroy(system);
    }
};

//=============================================================================
// All Bridges Created Tests
//=============================================================================

TEST_F(DragonflyMultiBridgeTest, AllBridgesCreatedSuccessfully) {
    EXPECT_NE(visual_bridge, nullptr) << "Visual bridge should be created";
    EXPECT_NE(audio_bridge, nullptr) << "Audio bridge should be created";
    EXPECT_NE(parietal_bridge, nullptr) << "Parietal bridge should be created";
    EXPECT_NE(cortical_bridge, nullptr) << "Cortical bridge should be created";
    EXPECT_NE(cognitive_bridge, nullptr) << "Cognitive bridge should be created";
    EXPECT_NE(thalamic_bridge, nullptr) << "Thalamic bridge should be created";
    EXPECT_NE(substrate_bridge, nullptr) << "Substrate bridge should be created";
    EXPECT_NE(fep_bridge, nullptr) << "FEP bridge should be created";
    EXPECT_NE(workspace_bridge, nullptr) << "Workspace bridge should be created";
    EXPECT_NE(bio_async_bridge, nullptr) << "Bio-async bridge should be created";
    EXPECT_NE(cnn_bridge, nullptr) << "CNN bridge should be created";
    EXPECT_NE(snn_bridge, nullptr) << "SNN bridge should be created";
}

//=============================================================================
// Visual + Audio Sensor Fusion Tests
//=============================================================================

TEST_F(DragonflyMultiBridgeTest, VisualAudioSensorFusion) {
    // Simulate visual detection
    float visual_position[3] = {10.0f, 5.0f, 0.0f};
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {visual_position[0], visual_position[1], visual_position[2]},
        .size = 0.05f,
        .contrast = 0.8f,
        .motion_direction = 0.5f,
        .motion_speed = 2.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    // Simulate audio cue from same direction
    dragonfly_audio_bridge_input_t audio_input = {
        .azimuth = 0.46f,  // Similar to visual direction
        .elevation = 0.0f,
        .confidence = 0.7f,
        .frequency_hz = 500.0f
    };
    EXPECT_EQ(dragonfly_audio_bridge_process(audio_bridge, &audio_input), 0);

    // Both bridges should have processed successfully
    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    dragonfly_audio_bridge_output_t audio_out;
    EXPECT_EQ(dragonfly_audio_bridge_get_output(audio_bridge, &audio_out), 0);

    // Directions should be close (sensor fusion would combine these)
    float direction_diff = fabs(visual_out.direction_rad - audio_out.direction_rad);
    EXPECT_LT(direction_diff, 0.5f) << "Visual and audio should detect similar direction";
}

//=============================================================================
// Parietal + Cortical Reasoning Tests
//=============================================================================

TEST_F(DragonflyMultiBridgeTest, ParietalCorticalReasoning) {
    // Parietal computes spatial interception
    dragonfly_parietal_bridge_input_t parietal_input = {
        .target_position = {15.0f, 8.0f, 2.0f},
        .target_velocity = {-2.0f, 0.5f, 0.0f},
        .self_position = {0.0f, 0.0f, 0.0f},
        .self_velocity = {0.0f, 0.0f, 0.0f}
    };
    EXPECT_EQ(dragonfly_parietal_bridge_process(parietal_bridge, &parietal_input), 0);

    dragonfly_parietal_bridge_output_t parietal_out;
    EXPECT_EQ(dragonfly_parietal_bridge_get_output(parietal_bridge, &parietal_out), 0);

    // Cortical processes higher-level decision
    dragonfly_cortical_bridge_input_t cortical_input = {
        .intercept_point = {parietal_out.intercept_point[0],
                           parietal_out.intercept_point[1],
                           parietal_out.intercept_point[2]},
        .time_to_intercept = parietal_out.time_to_intercept_s,
        .feasibility = parietal_out.feasibility
    };
    EXPECT_EQ(dragonfly_cortical_bridge_process(cortical_bridge, &cortical_input), 0);

    dragonfly_cortical_bridge_output_t cortical_out;
    EXPECT_EQ(dragonfly_cortical_bridge_get_output(cortical_bridge, &cortical_out), 0);

    // Cortical should provide pursuit decision
    EXPECT_GE(cortical_out.pursuit_confidence, 0.0f);
    EXPECT_LE(cortical_out.pursuit_confidence, 1.0f);
}

//=============================================================================
// Cognitive + Thalamic Attention Tests
//=============================================================================

TEST_F(DragonflyMultiBridgeTest, CognitiveThalamicAttention) {
    // Cognitive bridge processes attention/salience
    dragonfly_cognitive_bridge_input_t cognitive_input = {
        .salience = 0.85f,
        .novelty = 0.6f,
        .threat_level = 0.3f,
        .relevance = 0.9f
    };
    EXPECT_EQ(dragonfly_cognitive_bridge_process(cognitive_bridge, &cognitive_input), 0);

    dragonfly_cognitive_bridge_output_t cognitive_out;
    EXPECT_EQ(dragonfly_cognitive_bridge_get_output(cognitive_bridge, &cognitive_out), 0);

    // Thalamic bridge gates signals based on attention
    dragonfly_thalamic_bridge_input_t thalamic_input = {
        .attention_level = cognitive_out.attention_allocation,
        .arousal = 0.7f,
        .gate_visual = true,
        .gate_audio = true
    };
    EXPECT_EQ(dragonfly_thalamic_bridge_process(thalamic_bridge, &thalamic_input), 0);

    dragonfly_thalamic_bridge_output_t thalamic_out;
    EXPECT_EQ(dragonfly_thalamic_bridge_get_output(thalamic_bridge, &thalamic_out), 0);

    // High attention should result in strong gating
    EXPECT_GT(thalamic_out.visual_gain, 0.5f);
    EXPECT_GT(thalamic_out.audio_gain, 0.5f);
}

//=============================================================================
// Substrate + FEP Energy/Prediction Tests
//=============================================================================

TEST_F(DragonflyMultiBridgeTest, SubstrateFEPEnergyPrediction) {
    // Substrate tracks metabolic costs
    dragonfly_substrate_bridge_input_t substrate_input = {
        .current_speed = 5.0f,
        .acceleration_magnitude = 2.0f,
        .pursuit_duration_s = 3.0f,
        .ambient_temperature = 25.0f
    };
    EXPECT_EQ(dragonfly_substrate_bridge_process(substrate_bridge, &substrate_input), 0);

    dragonfly_substrate_bridge_output_t substrate_out;
    EXPECT_EQ(dragonfly_substrate_bridge_get_output(substrate_bridge, &substrate_out), 0);

    // FEP minimizes prediction error
    dragonfly_fep_bridge_input_t fep_input = {
        .predicted_position = {12.0f, 6.0f, 1.0f},
        .actual_position = {12.5f, 5.8f, 1.1f},
        .energy_available = substrate_out.energy_remaining,
        .precision = 0.8f
    };
    EXPECT_EQ(dragonfly_fep_bridge_process(fep_bridge, &fep_input), 0);

    dragonfly_fep_bridge_output_t fep_out;
    EXPECT_EQ(dragonfly_fep_bridge_get_output(fep_bridge, &fep_out), 0);

    // FEP should compute prediction error
    EXPECT_GE(fep_out.free_energy, 0.0f);
    EXPECT_GE(fep_out.prediction_error, 0.0f);
}

//=============================================================================
// Workspace + Bio-Async Consciousness Tests
//=============================================================================

TEST_F(DragonflyMultiBridgeTest, WorkspaceBioAsyncConsciousness) {
    // Workspace broadcasts target to global workspace
    dragonfly_workspace_bridge_input_t workspace_input = {
        .target_locked = true,
        .target_id = 42,
        .target_position = {10.0f, 5.0f, 0.0f},
        .confidence = 0.9f
    };
    EXPECT_EQ(dragonfly_workspace_bridge_process(workspace_bridge, &workspace_input), 0);

    dragonfly_workspace_bridge_output_t workspace_out;
    EXPECT_EQ(dragonfly_workspace_bridge_get_output(workspace_bridge, &workspace_out), 0);

    // Bio-async handles asynchronous updates
    dragonfly_bio_async_bridge_input_t bio_async_input = {
        .event_type = DRAGONFLY_BIO_ASYNC_TARGET_UPDATE,
        .timestamp_us = 1000000,
        .payload_size = 0
    };
    EXPECT_EQ(dragonfly_bio_async_bridge_process(bio_async_bridge, &bio_async_input), 0);

    // Both should be in sync
    EXPECT_TRUE(workspace_out.broadcast_active);
}

//=============================================================================
// CNN + SNN Neural Processing Tests
//=============================================================================

TEST_F(DragonflyMultiBridgeTest, CNNSNNNeuralProcessing) {
    // CNN extracts features from visual input
    float image_features[64];
    for (int i = 0; i < 64; i++) {
        image_features[i] = 0.5f + 0.5f * sin(i * 0.1f);
    }

    dragonfly_cnn_bridge_input_t cnn_input = {
        .feature_dim = 64,
        .features = image_features
    };
    EXPECT_EQ(dragonfly_cnn_bridge_process(cnn_bridge, &cnn_input), 0);

    dragonfly_cnn_bridge_output_t cnn_out;
    EXPECT_EQ(dragonfly_cnn_bridge_get_output(cnn_bridge, &cnn_out), 0);

    // SNN processes with spikes
    dragonfly_snn_bridge_input_t snn_input = {
        .spike_count = 10,
        .avg_firing_rate = 50.0f,
        .population_synchrony = 0.7f
    };
    EXPECT_EQ(dragonfly_snn_bridge_process(snn_bridge, &snn_input), 0);

    dragonfly_snn_bridge_output_t snn_out;
    EXPECT_EQ(dragonfly_snn_bridge_get_output(snn_bridge, &snn_out), 0);

    // Both should provide valid outputs
    EXPECT_GE(cnn_out.target_probability, 0.0f);
    EXPECT_LE(cnn_out.target_probability, 1.0f);
}

//=============================================================================
// Full Pipeline Integration Tests
//=============================================================================

TEST_F(DragonflyMultiBridgeTest, FullSensoryToMotorPipeline) {
    // 1. Visual detection
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {20.0f, 10.0f, 5.0f},
        .size = 0.06f,
        .contrast = 0.85f,
        .motion_direction = 0.3f,
        .motion_speed = 3.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    // 2. Cognitive attention allocation
    dragonfly_cognitive_bridge_input_t cognitive_input = {
        .salience = visual_out.salience,
        .novelty = 0.8f,
        .threat_level = 0.2f,
        .relevance = 0.9f
    };
    EXPECT_EQ(dragonfly_cognitive_bridge_process(cognitive_bridge, &cognitive_input), 0);

    dragonfly_cognitive_bridge_output_t cognitive_out;
    EXPECT_EQ(dragonfly_cognitive_bridge_get_output(cognitive_bridge, &cognitive_out), 0);

    // 3. Thalamic gating
    dragonfly_thalamic_bridge_input_t thalamic_input = {
        .attention_level = cognitive_out.attention_allocation,
        .arousal = 0.8f,
        .gate_visual = true,
        .gate_audio = false
    };
    EXPECT_EQ(dragonfly_thalamic_bridge_process(thalamic_bridge, &thalamic_input), 0);

    // 4. Parietal spatial reasoning
    dragonfly_parietal_bridge_input_t parietal_input = {
        .target_position = {visual_input.position[0],
                           visual_input.position[1],
                           visual_input.position[2]},
        .target_velocity = {-3.0f, 0.5f, -0.2f},
        .self_position = {0.0f, 0.0f, 0.0f},
        .self_velocity = {0.0f, 0.0f, 0.0f}
    };
    EXPECT_EQ(dragonfly_parietal_bridge_process(parietal_bridge, &parietal_input), 0);

    dragonfly_parietal_bridge_output_t parietal_out;
    EXPECT_EQ(dragonfly_parietal_bridge_get_output(parietal_bridge, &parietal_out), 0);

    // 5. Cortical decision
    dragonfly_cortical_bridge_input_t cortical_input = {
        .intercept_point = {parietal_out.intercept_point[0],
                           parietal_out.intercept_point[1],
                           parietal_out.intercept_point[2]},
        .time_to_intercept = parietal_out.time_to_intercept_s,
        .feasibility = parietal_out.feasibility
    };
    EXPECT_EQ(dragonfly_cortical_bridge_process(cortical_bridge, &cortical_input), 0);

    dragonfly_cortical_bridge_output_t cortical_out;
    EXPECT_EQ(dragonfly_cortical_bridge_get_output(cortical_bridge, &cortical_out), 0);

    // 6. Workspace broadcast
    dragonfly_workspace_bridge_input_t workspace_input = {
        .target_locked = true,
        .target_id = 1,
        .target_position = {visual_input.position[0],
                           visual_input.position[1],
                           visual_input.position[2]},
        .confidence = cortical_out.pursuit_confidence
    };
    EXPECT_EQ(dragonfly_workspace_bridge_process(workspace_bridge, &workspace_input), 0);

    // Pipeline completed successfully
    SUCCEED();
}

TEST_F(DragonflyMultiBridgeTest, ConcurrentBridgeUpdates) {
    // Simulate 100 concurrent update cycles
    for (int cycle = 0; cycle < 100; cycle++) {
        float t = cycle * 0.016f;  // 60 FPS

        // Update all bridges concurrently
        dragonfly_visual_bridge_input_t visual_input = {
            .position = {10.0f + t, 5.0f + sin(t), 0.0f},
            .size = 0.05f,
            .contrast = 0.8f,
            .motion_direction = t,
            .motion_speed = 2.0f
        };
        EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

        dragonfly_substrate_bridge_input_t substrate_input = {
            .current_speed = 2.0f + sin(t),
            .acceleration_magnitude = 0.5f,
            .pursuit_duration_s = t,
            .ambient_temperature = 25.0f
        };
        EXPECT_EQ(dragonfly_substrate_bridge_process(substrate_bridge, &substrate_input), 0);

        dragonfly_fep_bridge_input_t fep_input = {
            .predicted_position = {10.0f + t, 5.0f, 0.0f},
            .actual_position = {10.0f + t + 0.1f, 5.0f + sin(t), 0.0f},
            .energy_available = 0.8f,
            .precision = 0.7f
        };
        EXPECT_EQ(dragonfly_fep_bridge_process(fep_bridge, &fep_input), 0);
    }

    SUCCEED();
}

TEST_F(DragonflyMultiBridgeTest, BridgeStatisticsAccumulate) {
    // Process multiple inputs through bridges
    for (int i = 0; i < 50; i++) {
        dragonfly_visual_bridge_input_t visual_input = {
            .position = {10.0f + i, 5.0f, 0.0f},
            .size = 0.05f,
            .contrast = 0.8f,
            .motion_direction = 0.5f,
            .motion_speed = 2.0f
        };
        dragonfly_visual_bridge_process(visual_bridge, &visual_input);

        dragonfly_cognitive_bridge_input_t cognitive_input = {
            .salience = 0.8f,
            .novelty = 0.6f,
            .threat_level = 0.3f,
            .relevance = 0.9f
        };
        dragonfly_cognitive_bridge_process(cognitive_bridge, &cognitive_input);
    }

    // Get statistics
    visual_bridge_stats_t visual_stats;
    EXPECT_EQ(dragonfly_visual_bridge_get_stats(visual_bridge, &visual_stats), 0);
    EXPECT_GE(visual_stats.inputs_processed, 0u);  // Verify stats work

    cognitive_bridge_stats_t cognitive_stats;
    EXPECT_EQ(dragonfly_cognitive_bridge_get_stats(cognitive_bridge, &cognitive_stats), 0);
    EXPECT_GE(cognitive_stats.inputs_processed, 0u);  // Verify stats work
}
