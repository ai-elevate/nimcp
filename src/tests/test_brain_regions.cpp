/**
 * @file test_brain_regions.cpp
 * @brief TDD tests for modular brain architecture
 *
 * Test Coverage:
 * - Brain module creation/destruction
 * - Brain region creation (V1, A1, M1)
 * - Layer organization (6 cortical layers)
 * - Minicolumn organization
 * - Inter-region connectivity
 * - Sensory input processing
 * - Region stepping and simulation
 * - Statistics and monitoring
 * - Integration with neuron types
 */

#include <gtest/gtest.h>
#include "nimcp_brain_regions.h"
#include "nimcp_neuron_types.h"
#include "utils/nimcp_time.h"

class BrainRegionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup
    }

    void TearDown() override {
        // Cleanup
    }
};

// ============================================================================
// BRAIN MODULE TESTS
// ============================================================================

TEST_F(BrainRegionsTest, CreateDestroyBrainModule) {
    brain_module_t* brain = brain_module_create(10);

    ASSERT_NE(brain, nullptr);
    EXPECT_EQ(brain->num_regions, 0u);
    EXPECT_EQ(brain->max_regions, 10u);
    EXPECT_EQ(brain->total_neurons, 0u);

    brain_module_destroy(brain);
}

TEST_F(BrainRegionsTest, CreateDestroyNullBrain) {
    // Should not crash
    brain_module_destroy(nullptr);
}

// ============================================================================
// BRAIN REGION TESTS
// ============================================================================

TEST_F(BrainRegionsTest, CreateVisualCortexRegion) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);
    EXPECT_EQ(v1->type, REGION_VISUAL_V1);
    EXPECT_EQ(v1->total_neurons, 200u);
    EXPECT_NE(v1->network, nullptr); // Network should be created

    // V1 should have appropriate layer sizes
    EXPECT_GT(v1->layer_sizes[LAYER_4], 0u); // Input layer should exist

    brain_region_destroy(v1);
}

TEST_F(BrainRegionsTest, CreateAuditoryCortexRegion) {
    brain_region_t* a1 = brain_region_create(REGION_AUDITORY_A1, 150);

    ASSERT_NE(a1, nullptr);
    EXPECT_EQ(a1->type, REGION_AUDITORY_A1);
    EXPECT_EQ(a1->total_neurons, 150u);

    brain_region_destroy(a1);
}

TEST_F(BrainRegionsTest, CreateMotorCortexRegion) {
    brain_region_t* m1 = brain_region_create(REGION_MOTOR_M1, 180);

    ASSERT_NE(m1, nullptr);
    EXPECT_EQ(m1->type, REGION_MOTOR_M1);
    EXPECT_EQ(m1->total_neurons, 180u);

    // M1 should have output layer (Layer 5)
    EXPECT_GT(m1->layer_sizes[LAYER_5], 0u);

    brain_region_destroy(m1);
}

TEST_F(BrainRegionsTest, RegionNameRetrieval) {
    const char* v1_name = brain_region_get_name(REGION_VISUAL_V1);
    const char* a1_name = brain_region_get_name(REGION_AUDITORY_A1);
    const char* m1_name = brain_region_get_name(REGION_MOTOR_M1);

    EXPECT_STREQ(v1_name, "Primary Visual Cortex (V1)");
    EXPECT_STREQ(a1_name, "Primary Auditory Cortex (A1)");
    EXPECT_STREQ(m1_name, "Primary Motor Cortex (M1)");
}

// ============================================================================
// LAYER ORGANIZATION TESTS
// ============================================================================

TEST_F(BrainRegionsTest, LayerOrganization) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 240);

    ASSERT_NE(v1, nullptr);

    // Should have 6 cortical layers
    uint32_t total_layer_neurons = 0;
    for (int i = 0; i < LAYER_COUNT; i++) {
        total_layer_neurons += v1->layer_sizes[i];
    }

    // Total neurons across layers should match region total
    EXPECT_EQ(total_layer_neurons, v1->total_neurons);

    // Layer 4 (input) should be substantial in V1
    EXPECT_GT(v1->layer_sizes[LAYER_4], v1->total_neurons / 10);

    brain_region_destroy(v1);
}

TEST_F(BrainRegionsTest, GetLayerNeurons) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);

    // Get Layer 4 neurons
    uint32_t layer4_ids[100];
    uint32_t count = brain_region_get_layer_neurons(v1, LAYER_4, layer4_ids, 100);

    EXPECT_GT(count, 0u);
    EXPECT_LE(count, v1->layer_sizes[LAYER_4]);

    brain_region_destroy(v1);
}

// ============================================================================
// MINICOLUMN ORGANIZATION TESTS
// ============================================================================

TEST_F(BrainRegionsTest, OrganizeMinicolumns) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 240);

    ASSERT_NE(v1, nullptr);

    // Organize into 4x3 grid of minicolumns
    nimcp_result_t result = brain_region_organize_columns(v1, 4, 3);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(v1->num_minicolumns, 12u); // 4 * 3
    EXPECT_EQ(v1->minicolumns_x, 4u);
    EXPECT_EQ(v1->minicolumns_y, 3u);
    EXPECT_NE(v1->minicolumns, nullptr);

    brain_region_destroy(v1);
}

TEST_F(BrainRegionsTest, MinicolumnNeuronDistribution) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 240);

    ASSERT_NE(v1, nullptr);

    brain_region_organize_columns(v1, 4, 3);

    // Each minicolumn should have neurons
    for (uint32_t i = 0; i < v1->num_minicolumns; i++) {
        brain_minicolumn_t* col = v1->minicolumns[i];
        ASSERT_NE(col, nullptr);

        // Should have neurons in at least some layers
        uint32_t total_col_neurons = 0;
        for (int layer = 0; layer < LAYER_COUNT; layer++) {
            total_col_neurons += col->layer_neuron_counts[layer];
        }
        EXPECT_GT(total_col_neurons, 0u);
    }

    brain_region_destroy(v1);
}

// ============================================================================
// BRAIN MODULE + REGION TESTS
// ============================================================================

TEST_F(BrainRegionsTest, AddRegionToBrain) {
    brain_module_t* brain = brain_module_create(10);
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(brain, nullptr);
    ASSERT_NE(v1, nullptr);

    nimcp_result_t result = brain_module_add_region(brain, v1);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(brain->num_regions, 1u);
    EXPECT_EQ(brain->total_neurons, 200u);

    brain_module_destroy(brain);
}

TEST_F(BrainRegionsTest, AddMultipleRegions) {
    brain_module_t* brain = brain_module_create(10);

    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);
    brain_region_t* a1 = brain_region_create(REGION_AUDITORY_A1, 150);
    brain_region_t* m1 = brain_region_create(REGION_MOTOR_M1, 180);

    brain_module_add_region(brain, v1);
    brain_module_add_region(brain, a1);
    brain_module_add_region(brain, m1);

    EXPECT_EQ(brain->num_regions, 3u);
    EXPECT_EQ(brain->total_neurons, 530u); // 200 + 150 + 180

    brain_module_destroy(brain);
}

TEST_F(BrainRegionsTest, GetRegionById) {
    brain_module_t* brain = brain_module_create(10);
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    uint32_t region_id = v1->id;
    brain_module_add_region(brain, v1);

    brain_region_t* found = brain_module_get_region(brain, region_id);

    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->id, region_id);
    EXPECT_EQ(found->type, REGION_VISUAL_V1);

    brain_module_destroy(brain);
}

TEST_F(BrainRegionsTest, GetRegionByType) {
    brain_module_t* brain = brain_module_create(10);

    brain_module_add_region(brain, brain_region_create(REGION_VISUAL_V1, 200));
    brain_module_add_region(brain, brain_region_create(REGION_AUDITORY_A1, 150));

    brain_region_t* v1 = brain_module_get_region_by_type(brain, REGION_VISUAL_V1);
    brain_region_t* a1 = brain_module_get_region_by_type(brain, REGION_AUDITORY_A1);

    ASSERT_NE(v1, nullptr);
    ASSERT_NE(a1, nullptr);
    EXPECT_EQ(v1->type, REGION_VISUAL_V1);
    EXPECT_EQ(a1->type, REGION_AUDITORY_A1);

    brain_module_destroy(brain);
}

// ============================================================================
// INTER-REGION CONNECTIVITY TESTS
// ============================================================================

TEST_F(BrainRegionsTest, ConnectTwoRegions) {
    brain_module_t* brain = brain_module_create(10);

    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);
    brain_region_t* mt = brain_region_create(REGION_VISUAL_MT, 150);

    brain_module_add_region(brain, v1);
    brain_module_add_region(brain, mt);

    // Connect V1 → MT (visual motion pathway)
    nimcp_result_t result = brain_module_connect_regions(brain, v1->id, mt->id, 0.3f);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(brain->num_connections, 1u);

    brain_module_destroy(brain);
}

TEST_F(BrainRegionsTest, ConnectSpecificLayers) {
    brain_module_t* brain = brain_module_create(10);

    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);
    brain_region_t* m1 = brain_region_create(REGION_MOTOR_M1, 180);

    brain_module_add_region(brain, v1);
    brain_module_add_region(brain, m1);

    // Connect V1 Layer 5 → M1 Layer 4 (vision to action)
    nimcp_result_t result = brain_module_connect_layers(brain,
                                                        v1->id, LAYER_5,
                                                        m1->id, LAYER_4,
                                                        0.2f);

    EXPECT_EQ(result, NIMCP_SUCCESS);

    brain_module_destroy(brain);
}

TEST_F(BrainRegionsTest, MultiRegionConnectivity) {
    brain_module_t* brain = brain_module_create(10);

    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);
    brain_region_t* mt = brain_region_create(REGION_VISUAL_MT, 150);
    brain_region_t* m1 = brain_region_create(REGION_MOTOR_M1, 180);

    brain_module_add_region(brain, v1);
    brain_module_add_region(brain, mt);
    brain_module_add_region(brain, m1);

    // Create visual → motion → motor pathway
    brain_module_connect_regions(brain, v1->id, mt->id, 0.3f);
    brain_module_connect_regions(brain, mt->id, m1->id, 0.25f);

    EXPECT_EQ(brain->num_connections, 2u);

    brain_module_destroy(brain);
}

// ============================================================================
// SENSORY INPUT PROCESSING TESTS
// ============================================================================

TEST_F(BrainRegionsTest, ProcessVisualInput) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);

    // Simulate visual input (e.g., edge at 45 degrees)
    float visual_input[20];
    for (int i = 0; i < 20; i++) {
        visual_input[i] = 0.5f + 0.3f * sinf(i * 0.3f);
    }

    uint64_t timestamp = nimcp_time_monotonic_us();
    nimcp_result_t result = brain_region_process_input(v1, visual_input, 20, timestamp);

    EXPECT_EQ(result, NIMCP_SUCCESS);

    brain_region_destroy(v1);
}

TEST_F(BrainRegionsTest, ProcessAuditoryInput) {
    brain_region_t* a1 = brain_region_create(REGION_AUDITORY_A1, 150);

    ASSERT_NE(a1, nullptr);

    // Simulate auditory input (frequency spectrum)
    float auditory_input[16];
    for (int i = 0; i < 16; i++) {
        auditory_input[i] = 0.1f + 0.4f * expf(-0.1f * i);
    }

    uint64_t timestamp = nimcp_time_monotonic_us();
    nimcp_result_t result = brain_region_process_input(a1, auditory_input, 16, timestamp);

    EXPECT_EQ(result, NIMCP_SUCCESS);

    brain_region_destroy(a1);
}

TEST_F(BrainRegionsTest, GetRegionOutput) {
    brain_region_t* m1 = brain_region_create(REGION_MOTOR_M1, 180);

    ASSERT_NE(m1, nullptr);

    // Get motor output (Layer 5 activity)
    float motor_output[20];
    uint32_t count = brain_region_get_output(m1, motor_output, 20);

    EXPECT_GT(count, 0u);
    EXPECT_LE(count, 20u);

    brain_region_destroy(m1);
}

// ============================================================================
// SIMULATION & STEPPING TESTS
// ============================================================================

TEST_F(BrainRegionsTest, StepBrainRegion) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);

    uint64_t initial_time = v1->last_update;

    // Step region forward 1ms
    nimcp_result_t result = brain_region_step(v1, 1000);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GT(v1->last_update, initial_time);

    brain_region_destroy(v1);
}

TEST_F(BrainRegionsTest, StepBrainModule) {
    brain_module_t* brain = brain_module_create(10);

    brain_module_add_region(brain, brain_region_create(REGION_VISUAL_V1, 200));
    brain_module_add_region(brain, brain_region_create(REGION_AUDITORY_A1, 150));

    uint64_t initial_time = brain->current_time;

    // Step entire brain forward 1ms
    nimcp_result_t result = brain_module_step(brain, 1000);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GT(brain->current_time, initial_time);

    brain_module_destroy(brain);
}

TEST_F(BrainRegionsTest, MultiStepSimulation) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);

    // Run 10 time steps
    for (int i = 0; i < 10; i++) {
        brain_region_step(v1, 1000); // 1ms per step
    }

    // Region should have been updated
    EXPECT_GT(v1->last_update, 0u);

    brain_region_destroy(v1);
}

// ============================================================================
// STATISTICS & MONITORING TESTS
// ============================================================================

TEST_F(BrainRegionsTest, GetRegionStatistics) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);

    brain_region_stats_t stats{};
    nimcp_result_t result = brain_region_get_stats(v1, &stats);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(stats.total_neurons, 200u);
    EXPECT_GE(stats.avg_activity, 0.0f);
    EXPECT_LE(stats.avg_activity, 1.0f);

    brain_region_destroy(v1);
}

TEST_F(BrainRegionsTest, LayerActivityMonitoring) {
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);

    brain_region_stats_t stats{};
    brain_region_get_stats(v1, &stats);

    // Should have activity metrics for all layers
    for (int layer = 0; layer < LAYER_COUNT; layer++) {
        EXPECT_GE(stats.layer_activity[layer], 0.0f);
        EXPECT_LE(stats.layer_activity[layer], 1.0f);
    }

    brain_region_destroy(v1);
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST_F(BrainRegionsTest, VisualMotorPathway) {
    brain_module_t* brain = brain_module_create(10);

    // Create visual → motor pathway
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);
    brain_region_t* mt = brain_region_create(REGION_VISUAL_MT, 150);
    brain_region_t* m1 = brain_region_create(REGION_MOTOR_M1, 180);

    brain_module_add_region(brain, v1);
    brain_module_add_region(brain, mt);
    brain_module_add_region(brain, m1);

    // Connect pathway
    brain_module_connect_regions(brain, v1->id, mt->id, 0.3f);
    brain_module_connect_regions(brain, mt->id, m1->id, 0.25f);

    // Process visual input
    float visual_input[20] = {0.5f, 0.6f, 0.7f, 0.8f, 0.7f, 0.6f, 0.5f};
    uint64_t timestamp = nimcp_time_monotonic_us();
    brain_region_process_input(v1, visual_input, 20, timestamp);

    // Step simulation to propagate signals
    for (int i = 0; i < 5; i++) {
        brain_module_step(brain, 1000); // 1ms steps
    }

    // Get motor output
    float motor_output[20];
    uint32_t count = brain_region_get_output(m1, motor_output, 20);

    EXPECT_GT(count, 0u);

    brain_module_destroy(brain);
}

TEST_F(BrainRegionsTest, MultimodalIntegration) {
    brain_module_t* brain = brain_module_create(10);

    // Create visual + auditory → association pathway
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);
    brain_region_t* a1 = brain_region_create(REGION_AUDITORY_A1, 150);
    brain_region_t* pfc = brain_region_create(REGION_PREFRONTAL, 250);

    brain_module_add_region(brain, v1);
    brain_module_add_region(brain, a1);
    brain_module_add_region(brain, pfc);

    // Connect both sensory areas to prefrontal cortex
    brain_module_connect_regions(brain, v1->id, pfc->id, 0.3f);
    brain_module_connect_regions(brain, a1->id, pfc->id, 0.3f);

    EXPECT_EQ(brain->num_connections, 2u);
    EXPECT_EQ(brain->num_regions, 3u);

    brain_module_destroy(brain);
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

TEST_F(BrainRegionsTest, NullParameterHandling) {
    EXPECT_EQ(brain_region_get_stats(nullptr, nullptr), NIMCP_ERROR_INVALID_PARAM);
    EXPECT_EQ(brain_module_add_region(nullptr, nullptr), NIMCP_ERROR_INVALID_PARAM);
    EXPECT_EQ(brain_module_connect_regions(nullptr, 0, 1, 0.5f), NIMCP_ERROR_INVALID_PARAM);
}

TEST_F(BrainRegionsTest, SmallRegion) {
    // Very small region (minimum viable)
    brain_region_t* tiny = brain_region_create(REGION_VISUAL_V1, 10);

    ASSERT_NE(tiny, nullptr);
    EXPECT_EQ(tiny->total_neurons, 10u);

    brain_region_destroy(tiny);
}

TEST_F(BrainRegionsTest, LargeRegion) {
    // Large region (stress test) - limited by MAX_NEURONS (1024)
    brain_region_t* large = brain_region_create(REGION_VISUAL_V1, 1000);

    ASSERT_NE(large, nullptr);
    EXPECT_EQ(large->total_neurons, 1000u);

    brain_region_destroy(large);
}

// ============================================================================
// SPECIALIZED NEURON TYPE TESTS
// ============================================================================

TEST_F(BrainRegionsTest, VisualRegionNeuronTypes) {
    // Visual cortex should have specialized visual neurons
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);
    ASSERT_NE(v1->neuron_extended_types, nullptr);
    ASSERT_NE(v1->neuron_type_params, nullptr);

    // Check that visual neurons are present
    bool has_edge_detectors = false;
    bool has_orientation = false;

    for (uint32_t i = 0; i < v1->total_neurons; i++) {
        neuron_type_extended_t type = v1->neuron_extended_types[i];

        if (type == NEURON_VISUAL_EDGE) {
            has_edge_detectors = true;
            // Verify params were initialized
            EXPECT_GT(v1->neuron_type_params[i].visual_edge.receptive_field_size, 0.0f);
        }
        if (type == NEURON_VISUAL_ORIENTATION) {
            has_orientation = true;
            EXPECT_GE(v1->neuron_type_params[i].visual_orientation.preferred_orientation, 0.0f);
            EXPECT_LE(v1->neuron_type_params[i].visual_orientation.preferred_orientation, 180.0f);
        }
    }

    EXPECT_TRUE(has_edge_detectors) << "V1 should have edge detector neurons";
    EXPECT_TRUE(has_orientation) << "V1 should have orientation-selective neurons";

    brain_region_destroy(v1);
}

TEST_F(BrainRegionsTest, AuditoryRegionNeuronTypes) {
    // Auditory cortex should have specialized auditory neurons
    brain_region_t* a1 = brain_region_create(REGION_AUDITORY_A1, 150);

    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a1->neuron_extended_types, nullptr);

    // Check that auditory neurons are present
    bool has_frequency_tuned = false;
    bool has_onset_detectors = false;

    for (uint32_t i = 0; i < a1->total_neurons; i++) {
        neuron_type_extended_t type = a1->neuron_extended_types[i];

        if (type == NEURON_AUDITORY_FREQUENCY) {
            has_frequency_tuned = true;
            EXPECT_GT(a1->neuron_type_params[i].auditory_frequency.center_frequency, 0.0f);
        }
        if (type == NEURON_AUDITORY_ONSET) {
            has_onset_detectors = true;
            EXPECT_GT(a1->neuron_type_params[i].auditory_onset.refractory_duration, 0.0f);
        }
    }

    EXPECT_TRUE(has_frequency_tuned) << "A1 should have frequency-tuned neurons";
    EXPECT_TRUE(has_onset_detectors) << "A1 should have onset detector neurons";

    brain_region_destroy(a1);
}

TEST_F(BrainRegionsTest, MotorRegionNeuronTypes) {
    // Motor cortex should have specialized motor neurons
    brain_region_t* m1 = brain_region_create(REGION_MOTOR_M1, 100);

    ASSERT_NE(m1, nullptr);
    ASSERT_NE(m1->neuron_extended_types, nullptr);

    // Check that motor neurons are present
    bool has_motoneurons = false;
    bool has_pattern_generators = false;

    for (uint32_t i = 0; i < m1->total_neurons; i++) {
        neuron_type_extended_t type = m1->neuron_extended_types[i];

        if (type == NEURON_MOTOR_ALPHA) {
            has_motoneurons = true;
        }
        if (type == NEURON_MOTOR_PATTERN_GEN) {
            has_pattern_generators = true;
            EXPECT_GT(m1->neuron_type_params[i].motor_pattern.rhythm_frequency, 0.0f);
        }
    }

    EXPECT_TRUE(has_motoneurons) << "M1 should have motoneurons";
    EXPECT_TRUE(has_pattern_generators) << "M1 should have pattern generators";

    brain_region_destroy(m1);
}

TEST_F(BrainRegionsTest, LayerSpecificNeuronTypes) {
    // Layer 4 should have input-specialized neurons
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 200);

    ASSERT_NE(v1, nullptr);

    // Calculate which neurons are in Layer 4
    uint32_t layer4_start = 0;
    for (int layer = 0; layer < LAYER_4; layer++) {
        layer4_start += v1->layer_sizes[layer];
    }
    uint32_t layer4_end = layer4_start + v1->layer_sizes[LAYER_4];

    // Verify Layer 4 has edge detectors (input processing)
    bool layer4_has_edge_detectors = false;
    for (uint32_t i = layer4_start; i < layer4_end && i < v1->total_neurons; i++) {
        if (v1->neuron_extended_types[i] == NEURON_VISUAL_EDGE) {
            layer4_has_edge_detectors = true;
            break;
        }
    }

    EXPECT_TRUE(layer4_has_edge_detectors) << "Layer 4 should have edge detector neurons for input processing";

    brain_region_destroy(v1);
}

TEST_F(BrainRegionsTest, NeuronTypeParameterValidity) {
    // Ensure all neuron type parameters are valid
    brain_region_t* v1 = brain_region_create(REGION_VISUAL_V1, 100);

    ASSERT_NE(v1, nullptr);

    // Validate all neuron type parameters
    for (uint32_t i = 0; i < v1->total_neurons; i++) {
        neuron_type_extended_t type = v1->neuron_extended_types[i];
        nimcp_result_t result = neuron_type_validate_params(type, &v1->neuron_type_params[i]);

        EXPECT_EQ(result, NIMCP_SUCCESS)
            << "Neuron " << i << " (type " << type << ") has invalid parameters";
    }

    brain_region_destroy(v1);
}
