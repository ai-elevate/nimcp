//=============================================================================
// test_cortical_columns_brain_integration.cpp
// Comprehensive integration tests for cortical columns with brain system
//=============================================================================
/**
 * @file test_cortical_columns_brain_integration.cpp
 * @brief Integration tests for cortical columns with NIMCP brain
 *
 * WHAT: Complete integration testing of cortical column modules with brain
 * WHY:  Verify end-to-end processing from sensory input to brain decision
 * HOW:  GTest framework with realistic visual/audio pipelines
 *
 * TEST CATEGORIES:
 * 1. BrainIntegration: Brain create/destroy with columns
 * 2. VisualPipeline: End-to-end visual processing
 * 3. AudioPipeline: Tonotopic processing
 * 4. CrossModuleConnectivity: Inter-module communication
 * 5. LearningIntegration: Plasticity across modules
 * 6. EndToEndProcessing: Full sensory-to-decision
 *
 * @author NIMCP Development Team
 * @date 2025-11-25
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <memory>
#include <cstring>

// Core brain
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/neuron/nimcp_neuron.h"
#include "core/synapse/nimcp_synapse.h"

// Cortical columns modules
#include "src/core/cortical_columns/nimcp_cortical_column.h"
#include "src/core/cortical_columns/nimcp_cortical_layers.h"
#include "src/core/cortical_columns/nimcp_topographic_maps.h"
#include "src/core/cortical_columns/nimcp_orientation_columns.h"
#include "src/core/cortical_columns/nimcp_feature_hypercolumns.h"
#include "src/core/cortical_columns/nimcp_columnar_connectivity.h"

// Plasticity
#include "plasticity/stdp/nimcp_stdp.h"
#include "plasticity/hebbian/nimcp_hebbian.h"

// Glial (if applicable)
#include "glial/astrocytes/nimcp_astrocytes.h"
#include "glial/oligodendrocytes/nimcp_oligodendrocytes.h"
#include "glial/microglia/nimcp_microglia.h"

// Utils
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_time.h"

//=============================================================================
// Test Fixtures
//=============================================================================

/**
 * Base fixture for cortical column integration tests
 */
class CorticalColumnsBrainIntegrationTest : public ::testing::Test {
protected:
    brain_t brain;
    cortical_column_pool_t* pool;
    columnar_connectivity_t* connectivity;

    // Test constants
    static constexpr uint32_t NUM_NEURONS = 1000;
    static constexpr uint32_t NUM_INPUTS = 64;
    static constexpr uint32_t NUM_OUTPUTS = 10;
    static constexpr uint32_t NUM_MINICOLUMNS = 20;
    static constexpr uint32_t NEURONS_PER_MINICOLUMN = 80;

    void SetUp() override {
        // Create brain with cortical columns support
        brain_config_t config = {0};
        config.size = BRAIN_SIZE_SMALL;
        config.task = BRAIN_TASK_PATTERN_MATCHING;
        config.num_inputs = NUM_INPUTS;
        config.num_outputs = NUM_OUTPUTS;
        config.learning_rate = 0.01f;
        config.sparsity_target = 0.85f;
        config.enable_explanations = true;
        strncpy(config.task_name, "cortical_test", sizeof(config.task_name) - 1);

        brain = brain_create_custom(&config);
        ASSERT_NE(brain, nullptr);

        // Create cortical column pool
        cortical_column_pool_config_t pool_config = {
            .max_minicolumns = 1000,
            .max_hypercolumns = 100,
            .max_neurons_per_minicolumn = 100,
            .enable_cow_support = true
        };
        pool = cortical_column_pool_create(&pool_config);
        ASSERT_NE(pool, nullptr);

        // Create connectivity manager
        connectivity = columnar_connectivity_create(50000);
        ASSERT_NE(connectivity, nullptr);
    }

    void TearDown() override {
        if (connectivity) {
            columnar_connectivity_destroy(connectivity);
        }
        if (pool) {
            cortical_column_pool_destroy(pool);
        }
        if (brain) {
            brain_destroy(brain);
        }
    }

    // Helper: Create minicolumn with brain neurons
    minicolumn_t* create_minicolumn_for_brain(
        uint32_t start_neuron_id,
        float preferred_orientation,
        float x, float y
    ) {
        std::vector<uint32_t> neuron_ids;
        for (uint32_t i = 0; i < NEURONS_PER_MINICOLUMN; ++i) {
            neuron_ids.push_back(start_neuron_id + i);
        }

        minicolumn_config_t config = {
            .neuron_ids = neuron_ids.data(),
            .num_neurons = NEURONS_PER_MINICOLUMN,
            .receptive_field = {x, y, 0.0f, 2.0f},
            .tuning_preference = preferred_orientation,
            .layers = {
                .layer_2_3_count = 32,
                .layer_4_count = 12,
                .layer_5_6_count = 36
            }
        };

        return minicolumn_create(pool, &config);
    }

    // Helper: Generate test image with oriented edge
    void generate_edge_image(float* image, uint32_t width, uint32_t height,
                            float orientation_deg) {
        float theta = orientation_deg * M_PI / 180.0f;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                float cx = static_cast<float>(x) - width / 2.0f;
                float cy = static_cast<float>(y) - height / 2.0f;
                float proj = cx * cos_t + cy * sin_t;
                image[y * width + x] = 0.5f + 0.5f * tanhf(proj / 2.0f);
            }
        }
    }

    // Helper: Generate frequency sweep for audio
    void generate_frequency_sweep(float* signal, uint32_t length,
                                 float start_freq, float end_freq,
                                 float sample_rate) {
        for (uint32_t i = 0; i < length; ++i) {
            float t = static_cast<float>(i) / sample_rate;
            float freq = start_freq + (end_freq - start_freq) * t;
            signal[i] = sinf(2.0f * M_PI * freq * t);
        }
    }
};

//=============================================================================
// 1. BRAIN INTEGRATION TESTS
//=============================================================================

TEST_F(CorticalColumnsBrainIntegrationTest, CreateCorticalColumnsInBrain) {
    // Create multiple minicolumns associated with brain neurons
    std::vector<minicolumn_t*> minicolumns;

    for (uint32_t i = 0; i < NUM_MINICOLUMNS; ++i) {
        float orientation = i * 180.0f / NUM_MINICOLUMNS;
        float x = i % 5;
        float y = i / 5;

        minicolumn_t* col = create_minicolumn_for_brain(
            i * NEURONS_PER_MINICOLUMN, orientation, x, y
        );
        ASSERT_NE(col, nullptr);
        minicolumns.push_back(col);
    }

    // Verify minicolumns created successfully
    EXPECT_EQ(minicolumns.size(), NUM_MINICOLUMNS);

    // Verify statistics
    minicolumn_stats_t stats;
    minicolumn_get_stats(minicolumns[0], &stats);
    EXPECT_EQ(stats.num_neurons, NEURONS_PER_MINICOLUMN);
    EXPECT_FLOAT_EQ(stats.tuning_preference, 0.0f);

    // Cleanup
    for (auto* col : minicolumns) {
        minicolumn_destroy(col);
    }
}

TEST_F(CorticalColumnsBrainIntegrationTest, ConnectColumnsToNeurons) {
    // Create minicolumns
    minicolumn_t* col1 = create_minicolumn_for_brain(0, 0.0f, 0.0f, 0.0f);
    minicolumn_t* col2 = create_minicolumn_for_brain(100, 45.0f, 1.0f, 0.0f);

    ASSERT_NE(col1, nullptr);
    ASSERT_NE(col2, nullptr);

    // Create laminar structures for connectivity
    laminar_structure_t layers1 = {0};
    layers1.layer_sizes[CORTICAL_LAYER_II_III] = 32;
    layers1.layer_sizes[CORTICAL_LAYER_IV] = 12;
    layers1.layer_sizes[CORTICAL_LAYER_V] = 36;
    layers1.total_neurons = 80;

    // Generate intracolumnar connections
    uint32_t conn_count = connectivity_generate_intracolumnar(
        connectivity, 0, &layers1
    );
    EXPECT_GT(conn_count, 0);

    // Verify connectivity stats
    connectivity_stats_t stats;
    nimcp_result_t result = connectivity_get_stats(connectivity, &stats);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GT(stats.total_connections, 0);
    EXPECT_GT(stats.intracolumnar_count, 0);

    // Cleanup
    minicolumn_destroy(col1);
    minicolumn_destroy(col2);
}

TEST_F(CorticalColumnsBrainIntegrationTest, ProcessInputThroughColumns) {
    // Create orientation hypercolumn
    hypercolumn_config_t hc_config = {0};
    hc_config.num_minicolumns = 8;
    hc_config.feature_space_min = 0.0f;
    hc_config.feature_space_max = 180.0f;
    hc_config.topographic_x = 0.0f;
    hc_config.topographic_y = 0.0f;
    hc_config.competition = COMPETITION_SOFTMAX;
    hc_config.k_winners = 3;
    hc_config.temperature = 1.0f;
    hc_config.lateral_inhibition_strength = 0.5f;
    hc_config.lateral_inhibition_sigma1 = 1.0f;
    hc_config.lateral_inhibition_sigma2 = 3.0f;

    // Create minicolumn configs
    std::vector<minicolumn_config_t> mc_configs(8);
    std::vector<std::vector<uint32_t>> neuron_id_arrays(8);

    for (uint32_t i = 0; i < 8; ++i) {
        neuron_id_arrays[i].resize(NEURONS_PER_MINICOLUMN);
        for (uint32_t j = 0; j < NEURONS_PER_MINICOLUMN; ++j) {
            neuron_id_arrays[i][j] = i * NEURONS_PER_MINICOLUMN + j;
        }

        mc_configs[i].neuron_ids = neuron_id_arrays[i].data();
        mc_configs[i].num_neurons = NEURONS_PER_MINICOLUMN;
        mc_configs[i].receptive_field = {0.0f, 0.0f, 0.0f, 2.0f};
        mc_configs[i].tuning_preference = i * 180.0f / 8.0f;
        mc_configs[i].layers = {32, 12, 36};
    }

    hc_config.minicolumn_configs = mc_configs.data();

    // Create hypercolumn
    hypercolumn_t* hcol = hypercolumn_create(pool, &hc_config);
    ASSERT_NE(hcol, nullptr);

    // Generate input representing 45° edge
    std::vector<float> input(64);
    for (size_t i = 0; i < input.size(); ++i) {
        float angle = i * 180.0f / 64.0f;
        float diff = fabsf(angle - 45.0f);
        input[i] = expf(-diff * diff / (2.0f * 15.0f * 15.0f));
    }

    // Process through hypercolumn
    hypercolumn_compute(hcol, input.data(), input.size());

    // Get winner
    uint32_t winner = hypercolumn_get_winner(hcol);
    EXPECT_LT(winner, 8);

    // Winner should be closest to 45° (index 2 or 3)
    EXPECT_TRUE(winner == 2 || winner == 3);

    // Get distribution
    std::vector<float> distribution(8);
    hypercolumn_get_distribution(hcol, distribution.data(), 8);

    float max_activation = 0.0f;
    for (float act : distribution) {
        max_activation = std::max(max_activation, act);
    }
    EXPECT_GT(max_activation, 0.0f);

    // Cleanup
    hypercolumn_destroy(hcol);
}

//=============================================================================
// 2. VISUAL PROCESSING PIPELINE TESTS
//=============================================================================

TEST_F(CorticalColumnsBrainIntegrationTest, VisualPipeline_Retinotopic) {
    // Create retinotopic map
    retinotopic_params_t retino_params = {
        .foveal_radius = 5.0f,
        .cortical_magnification = 2.0f,
        .log_polar_a = 1.0f,
        .aspect_ratio = 1.0f,
        .eccentricity_half = 10.0f,
        .angle_coverage = 2.0f * M_PI
    };

    topographic_map_t* retino_map = topographic_map_create_retinotopic(
        &retino_params, 32, 32
    );
    ASSERT_NE(retino_map, nullptr);

    // Generate visual input (8x8 image)
    std::vector<float> image(64, 0.0f);
    // Create center stimulus
    image[3 * 8 + 3] = 1.0f;
    image[3 * 8 + 4] = 1.0f;
    image[4 * 8 + 3] = 1.0f;
    image[4 * 8 + 4] = 1.0f;

    // Project to cortical surface
    std::vector<float> cortical_activity(32 * 32, 0.0f);
    topographic_map_project_activity(
        retino_map,
        image.data(), 8, 8,
        cortical_activity.data(), 32, 32
    );

    // Verify activity projected
    float total_activity = 0.0f;
    for (float act : cortical_activity) {
        total_activity += act;
    }
    EXPECT_GT(total_activity, 0.0f);

    // Get statistics
    topographic_stats_t stats;
    topographic_map_get_stats(retino_map, &stats);
    EXPECT_GT(stats.mean_magnification, 0.0f);
    EXPECT_GT(stats.total_cortical_area, 0.0f);

    // Cleanup
    topographic_map_destroy(retino_map);
}

TEST_F(CorticalColumnsBrainIntegrationTest, VisualPipeline_OrientationColumns) {
    // Create orientation hypercolumn
    orientation_hypercolumn_t* hcol = orientation_hypercolumn_create(
        16,    // num_orientations
        2.0f,  // spatial_frequency
        30.0f  // tuning_width
    );
    ASSERT_NE(hcol, nullptr);

    // Generate test image with 45° edge
    std::vector<float> image(16 * 16);
    generate_edge_image(image.data(), 16, 16, 45.0f);

    // Process through orientation columns
    bool result = orientation_hypercolumn_process(
        hcol, image.data(), 16, 16
    );
    EXPECT_TRUE(result);

    // Get dominant orientation
    float dominant = orientation_hypercolumn_get_dominant(hcol);
    EXPECT_GE(dominant, 0.0f);
    EXPECT_LE(dominant, 180.0f);

    // Should be close to 45°
    float error = fabsf(dominant - 45.0f);
    EXPECT_LT(error, 20.0f);  // Within 20° tolerance

    // Compute OSI
    float osi = orientation_hypercolumn_compute_osi(hcol);
    EXPECT_GE(osi, 0.0f);
    EXPECT_LE(osi, 1.0f);

    // Apply normalization
    result = orientation_hypercolumn_normalize(hcol);
    EXPECT_TRUE(result);

    // Get statistics
    hypercolumn_stats_t stats;
    result = orientation_hypercolumn_get_stats(hcol, &stats);
    EXPECT_TRUE(result);
    EXPECT_GT(stats.num_active_columns, 0);

    // Cleanup
    orientation_hypercolumn_destroy(hcol);
}

TEST_F(CorticalColumnsBrainIntegrationTest, VisualPipeline_FeatureHypercolumns) {
    // Create orientation feature hypercolumn
    feature_hypercolumn_t* fhcol = feature_hypercolumn_create_orientation(12);
    ASSERT_NE(fhcol, nullptr);

    // Process with 60° orientation
    float features[] = {60.0f};
    feature_hypercolumn_process(fhcol, features, 1);

    // Apply softmax competition
    feature_hypercolumn_softmax(fhcol, 0.5f);

    // Get winner
    uint32_t winner = feature_hypercolumn_get_winner(fhcol);
    EXPECT_LT(winner, 12);

    // Decode orientation
    float decoded_features[1];
    feature_hypercolumn_decode(fhcol, decoded_features);

    // Should be close to 60°
    float error = fabsf(decoded_features[0] - 60.0f);
    EXPECT_LT(error, 20.0f);

    // Get statistics
    feature_hypercolumn_stats_t stats;
    feature_hypercolumn_get_stats(fhcol, &stats);
    EXPECT_GT(stats.max_activation, 0.0f);
    EXPECT_GT(stats.num_active, 0);

    // Cleanup
    feature_hypercolumn_destroy(fhcol);
}

TEST_F(CorticalColumnsBrainIntegrationTest, VisualPipeline_EndToEnd) {
    // STAGE 1: Retinotopic mapping
    retinotopic_params_t retino_params = {
        .foveal_radius = 5.0f,
        .cortical_magnification = 2.0f,
        .log_polar_a = 1.0f,
        .aspect_ratio = 1.0f,
        .eccentricity_half = 10.0f,
        .angle_coverage = 2.0f * M_PI
    };

    topographic_map_t* retino_map = topographic_map_create_retinotopic(
        &retino_params, 16, 16
    );
    ASSERT_NE(retino_map, nullptr);

    // STAGE 2: Orientation columns
    orientation_hypercolumn_t* orient_hcol = orientation_hypercolumn_create(
        8, 2.0f, 30.0f
    );
    ASSERT_NE(orient_hcol, nullptr);

    // STAGE 3: Feature hypercolumns
    feature_hypercolumn_t* feature_hcol = feature_hypercolumn_create_orientation(8);
    ASSERT_NE(feature_hcol, nullptr);

    // INPUT: Generate 90° edge image
    std::vector<float> input_image(16 * 16);
    generate_edge_image(input_image.data(), 16, 16, 90.0f);

    // PROCESS: Retinotopic projection
    std::vector<float> cortical_image(16 * 16);
    topographic_map_project_activity(
        retino_map,
        input_image.data(), 16, 16,
        cortical_image.data(), 16, 16
    );

    // PROCESS: Orientation detection
    bool result = orientation_hypercolumn_process(
        orient_hcol, cortical_image.data(), 16, 16
    );
    EXPECT_TRUE(result);

    // Get orientation distribution
    std::vector<float> orientations(8);
    std::vector<float> responses(8);
    uint32_t num_orients = 0;
    result = orientation_hypercolumn_get_distribution(
        orient_hcol, orientations.data(), responses.data(), &num_orients
    );
    EXPECT_TRUE(result);
    EXPECT_EQ(num_orients, 8);

    // PROCESS: Feature binding
    float dominant_orient = orientation_hypercolumn_get_dominant(orient_hcol);
    float features[] = {dominant_orient};
    feature_hypercolumn_process(feature_hcol, features, 1);

    // DECODE: Extract final representation
    float decoded[1];
    feature_hypercolumn_decode(feature_hcol, decoded);

    // VERIFY: Should detect ~90° orientation
    float error = fabsf(decoded[0] - 90.0f);
    EXPECT_LT(error, 30.0f);

    // Cleanup
    topographic_map_destroy(retino_map);
    orientation_hypercolumn_destroy(orient_hcol);
    feature_hypercolumn_destroy(feature_hcol);
}

//=============================================================================
// 3. AUDIO PROCESSING PIPELINE TESTS
//=============================================================================

TEST_F(CorticalColumnsBrainIntegrationTest, AudioPipeline_Tonotopic) {
    // Create tonotopic map (A1 frequency mapping)
    tonotopic_params_t tono_params = {
        .min_frequency = 200.0f,
        .max_frequency = 8000.0f,
        .octave_span = 1.0f,
        .is_logarithmic = true,
        .q_factor = 4.0f
    };

    topographic_map_t* tono_map = topographic_map_create_tonotopic(
        &tono_params, 32
    );
    ASSERT_NE(tono_map, nullptr);

    // Get dimensions
    uint32_t width, height;
    topographic_map_get_dimensions(tono_map, &width, &height);
    EXPECT_GT(width, 0);

    // Get statistics
    topographic_stats_t stats;
    topographic_map_get_stats(tono_map, &stats);
    EXPECT_GT(stats.total_cortical_area, 0.0f);

    // Test frequency mapping
    float input_coords[] = {1000.0f};  // 1kHz
    float cortical_coords[2];
    topographic_map_input_to_cortex(
        tono_map, input_coords, cortical_coords, 1
    );

    EXPECT_GE(cortical_coords[0], 0.0f);

    // Cleanup
    topographic_map_destroy(tono_map);
}

TEST_F(CorticalColumnsBrainIntegrationTest, AudioPipeline_FrequencyColumns) {
    // Create spatial frequency hypercolumn
    feature_hypercolumn_t* freq_hcol = feature_hypercolumn_create_spatial_freq(
        6,       // num_octaves
        0.5f,    // min_freq
        8.0f     // max_freq
    );
    ASSERT_NE(freq_hcol, nullptr);

    // Process 2.0 cycles/degree frequency
    float features[] = {2.0f};
    feature_hypercolumn_process(freq_hcol, features, 1);

    // Apply normalization
    feature_hypercolumn_normalize(freq_hcol);

    // Decode
    float decoded[1];
    feature_hypercolumn_decode(freq_hcol, decoded);

    // Should be close to 2.0
    float error = fabsf(decoded[0] - 2.0f);
    EXPECT_LT(error, 1.0f);

    // Cleanup
    feature_hypercolumn_destroy(freq_hcol);
}

//=============================================================================
// 4. CROSS-MODULE CONNECTIVITY TESTS
//=============================================================================

TEST_F(CorticalColumnsBrainIntegrationTest, CrossModule_ColumnarConnectivity) {
    // Apply canonical microcircuit
    nimcp_result_t result = connectivity_apply_canonical_rules(connectivity);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Generate intercolumnar connections
    std::vector<uint32_t> column_ids = {0, 1, 2, 3, 4};
    std::vector<float> positions = {
        0.0f, 0.0f,  // col 0
        1.0f, 0.0f,  // col 1
        2.0f, 0.0f,  // col 2
        0.0f, 1.0f,  // col 3
        1.0f, 1.0f   // col 4
    };

    uint32_t conn_count = connectivity_generate_intercolumnar(
        connectivity,
        column_ids.data(), column_ids.size(),
        positions.data(), 2
    );
    EXPECT_GT(conn_count, 0);

    // Get connectivity stats
    connectivity_stats_t stats;
    result = connectivity_get_stats(connectivity, &stats);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GT(stats.total_connections, 0);
    EXPECT_GT(stats.intercolumnar_count, 0);
}

TEST_F(CorticalColumnsBrainIntegrationTest, CrossModule_LayerConnections) {
    // Create laminar structure
    laminar_structure_t* laminar = laminar_structure_create(nullptr);
    ASSERT_NE(laminar, nullptr);

    // Apply canonical circuit
    laminar_apply_canonical_circuit(laminar);

    // Process feedforward
    std::vector<float> input(64, 0.5f);
    laminar_process_input(laminar, CORTICAL_LAYER_IV, input.data(), input.size());
    laminar_process_feedforward(laminar);

    // Get Layer II/III activation
    float activation = laminar_get_layer_activation(laminar, CORTICAL_LAYER_II_III);
    EXPECT_GE(activation, 0.0f);

    // Process feedback
    laminar_process_feedback(laminar);

    // Get profile
    laminar_profile_t profile;
    laminar_get_profile(laminar, &profile);
    EXPECT_GT(profile.timestamp, 0);

    // Cleanup
    laminar_structure_destroy(laminar);
}

TEST_F(CorticalColumnsBrainIntegrationTest, CrossModule_LongRangeConnections) {
    // Generate long-range connections (V1 -> V2 style)
    std::vector<uint32_t> source_cols = {0, 1, 2};
    std::vector<uint32_t> target_cols = {10, 11, 12};

    uint32_t conn_count = connectivity_generate_long_range(
        connectivity,
        source_cols.data(), source_cols.size(),
        target_cols.data(), target_cols.size()
    );
    EXPECT_GT(conn_count, 0);

    // Verify connections created
    connectivity_stats_t stats;
    nimcp_result_t result = connectivity_get_stats(connectivity, &stats);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GT(stats.long_range_count, 0);
}

//=============================================================================
// 5. LEARNING INTEGRATION TESTS
//=============================================================================

TEST_F(CorticalColumnsBrainIntegrationTest, Learning_STDPAcrossColumns) {
    // Create two minicolumns
    minicolumn_t* col1 = create_minicolumn_for_brain(0, 0.0f, 0.0f, 0.0f);
    minicolumn_t* col2 = create_minicolumn_for_brain(100, 45.0f, 1.0f, 0.0f);

    ASSERT_NE(col1, nullptr);
    ASSERT_NE(col2, nullptr);

    // Create connections
    laminar_structure_t layers = {0};
    layers.layer_sizes[CORTICAL_LAYER_II_III] = 32;
    layers.layer_sizes[CORTICAL_LAYER_IV] = 12;
    layers.layer_sizes[CORTICAL_LAYER_V] = 36;
    layers.total_neurons = 80;

    connectivity_generate_intracolumnar(connectivity, 0, &layers);
    connectivity_generate_intracolumnar(connectivity, 1, &layers);

    // Simulate spike times
    std::vector<uint64_t> spike_times(200, 0);
    uint64_t current_time = nimcp_platform_get_time_us();

    // Pre-synaptic spikes
    for (size_t i = 0; i < 80; ++i) {
        spike_times[i] = current_time;
    }

    // Post-synaptic spikes (10ms later)
    for (size_t i = 100; i < 180; ++i) {
        spike_times[i] = current_time + 10000;  // 10ms
    }

    // Apply STDP
    connectivity_apply_stdp(
        connectivity,
        spike_times.data(),
        spike_times.data() + 100,
        100
    );

    // Verify connectivity updated
    connectivity_stats_t stats;
    nimcp_result_t result = connectivity_get_stats(connectivity, &stats);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Cleanup
    minicolumn_destroy(col1);
    minicolumn_destroy(col2);
}

TEST_F(CorticalColumnsBrainIntegrationTest, Learning_HebbianFeatureLearning) {
    // Create feature hypercolumn
    feature_hypercolumn_t* fhcol = feature_hypercolumn_create_orientation(8);
    ASSERT_NE(fhcol, nullptr);

    // Generate training input (45° orientation)
    std::vector<float> input(64);
    for (size_t i = 0; i < input.size(); ++i) {
        float angle = i * 180.0f / 64.0f;
        float diff = fabsf(angle - 45.0f);
        input[i] = expf(-diff * diff / (2.0f * 10.0f * 10.0f));
    }

    // Train with Hebbian learning
    for (int iter = 0; iter < 100; ++iter) {
        feature_hypercolumn_process_with_input(fhcol, input.data(), input.size());
        feature_hypercolumn_learn_hebbian(fhcol, input.data(), 0.01f);
    }

    // Test learned representation
    feature_hypercolumn_process_with_input(fhcol, input.data(), input.size());

    uint32_t winner = feature_hypercolumn_get_winner(fhcol);
    EXPECT_LT(winner, 8);

    // Winner should prefer ~45°
    float winner_pref = winner * 180.0f / 8.0f;
    float error = fabsf(winner_pref - 45.0f);
    EXPECT_LT(error, 30.0f);

    // Cleanup
    feature_hypercolumn_destroy(fhcol);
}

TEST_F(CorticalColumnsBrainIntegrationTest, Learning_BrainWithColumns) {
    // Process pattern through brain
    std::vector<float> pattern(NUM_INPUTS);
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = sinf(i * 2.0f * M_PI / pattern.size());
    }

    // Learn example
    brain_learn_example(brain, pattern.data(), "pattern_a", 0.9f);

    // Test decision
    brain_decision_t decision = brain_decide(brain, pattern.data());
    EXPECT_NE(decision.label, nullptr);
    EXPECT_GT(decision.confidence, 0.0f);

    // Verify brain stats
    brain_stats_t stats = brain_get_stats(brain);
    EXPECT_GT(stats.total_neurons, 0);
    EXPECT_GT(stats.total_synapses, 0);
}

//=============================================================================
// 6. END-TO-END PROCESSING TESTS
//=============================================================================

TEST_F(CorticalColumnsBrainIntegrationTest, EndToEnd_VisualDecision) {
    // Complete visual processing pipeline to brain decision

    // STAGE 1: Topographic mapping
    retinotopic_params_t retino_params = {
        .foveal_radius = 5.0f,
        .cortical_magnification = 2.0f,
        .log_polar_a = 1.0f,
        .aspect_ratio = 1.0f,
        .eccentricity_half = 10.0f,
        .angle_coverage = 2.0f * M_PI
    };

    topographic_map_t* retino_map = topographic_map_create_retinotopic(
        &retino_params, 8, 8
    );
    ASSERT_NE(retino_map, nullptr);

    // STAGE 2: Orientation processing
    orientation_hypercolumn_t* orient_hcol = orientation_hypercolumn_create(
        8, 2.0f, 30.0f
    );
    ASSERT_NE(orient_hcol, nullptr);

    // STAGE 3: Feature extraction
    feature_hypercolumn_t* feature_hcol = feature_hypercolumn_create_orientation(8);
    ASSERT_NE(feature_hcol, nullptr);

    // INPUT: Vertical edge (90°)
    std::vector<float> input_image(8 * 8);
    generate_edge_image(input_image.data(), 8, 8, 90.0f);

    // PROCESS: Full pipeline
    std::vector<float> cortical_image(8 * 8);
    topographic_map_project_activity(
        retino_map,
        input_image.data(), 8, 8,
        cortical_image.data(), 8, 8
    );

    orientation_hypercolumn_process(orient_hcol, cortical_image.data(), 8, 8);

    float dominant = orientation_hypercolumn_get_dominant(orient_hcol);
    float features[] = {dominant};
    feature_hypercolumn_process(feature_hcol, features, 1);

    // STAGE 4: Brain decision
    std::vector<float> brain_input(NUM_INPUTS, 0.0f);
    feature_hypercolumn_get_all_activations(feature_hcol, brain_input.data());

    // Pad input to brain size
    brain_learn_example(brain, brain_input.data(), "vertical_edge", 0.95f);

    brain_decision_t decision = brain_decide(brain, brain_input.data());
    EXPECT_NE(decision.label, nullptr);
    EXPECT_GT(decision.confidence, 0.0f);
    EXPECT_STREQ(decision.label, "vertical_edge");

    // Cleanup
    topographic_map_destroy(retino_map);
    orientation_hypercolumn_destroy(orient_hcol);
    feature_hypercolumn_destroy(feature_hcol);
}

TEST_F(CorticalColumnsBrainIntegrationTest, EndToEnd_MultipleOrientations) {
    // Test brain learning multiple orientations

    orientation_hypercolumn_t* orient_hcol = orientation_hypercolumn_create(
        16, 2.0f, 20.0f
    );
    ASSERT_NE(orient_hcol, nullptr);

    feature_hypercolumn_t* feature_hcol = feature_hypercolumn_create_orientation(16);
    ASSERT_NE(feature_hcol, nullptr);

    // Train on multiple orientations
    float orientations[] = {0.0f, 45.0f, 90.0f, 135.0f};
    const char* labels[] = {"horizontal", "diagonal_right", "vertical", "diagonal_left"};

    for (int i = 0; i < 4; ++i) {
        // Generate image
        std::vector<float> image(16 * 16);
        generate_edge_image(image.data(), 16, 16, orientations[i]);

        // Process
        orientation_hypercolumn_process(orient_hcol, image.data(), 16, 16);
        float dominant = orientation_hypercolumn_get_dominant(orient_hcol);

        float features[] = {dominant};
        feature_hypercolumn_process(feature_hcol, features, 1);

        std::vector<float> brain_input(NUM_INPUTS, 0.0f);
        feature_hypercolumn_get_all_activations(feature_hcol, brain_input.data());

        // Learn
        brain_learn_example(brain, brain_input.data(), labels[i], 0.9f);
    }

    // Test recognition
    std::vector<float> test_image(16 * 16);
    generate_edge_image(test_image.data(), 16, 16, 90.0f);

    orientation_hypercolumn_process(orient_hcol, test_image.data(), 16, 16);
    float dominant = orientation_hypercolumn_get_dominant(orient_hcol);

    float features[] = {dominant};
    feature_hypercolumn_process(feature_hcol, features, 1);

    std::vector<float> brain_input(NUM_INPUTS, 0.0f);
    feature_hypercolumn_get_all_activations(feature_hcol, brain_input.data());

    brain_decision_t decision = brain_decide(brain, brain_input.data());
    EXPECT_NE(decision.label, nullptr);
    EXPECT_STREQ(decision.label, "vertical");
    EXPECT_GT(decision.confidence, 0.5f);

    // Cleanup
    orientation_hypercolumn_destroy(orient_hcol);
    feature_hypercolumn_destroy(feature_hcol);
}

//=============================================================================
// 7. PERFORMANCE TESTS
//=============================================================================

TEST_F(CorticalColumnsBrainIntegrationTest, Performance_LargeScaleProcessing) {
    // Test processing with many hypercolumns
    const uint32_t NUM_HYPERCOLUMNS = 10;
    std::vector<feature_hypercolumn_t*> hypercolumns;

    // Create hypercolumns
    for (uint32_t i = 0; i < NUM_HYPERCOLUMNS; ++i) {
        feature_hypercolumn_t* hcol = feature_hypercolumn_create_orientation(8);
        ASSERT_NE(hcol, nullptr);
        hypercolumns.push_back(hcol);
    }

    // Process through all
    uint64_t start_time = nimcp_platform_get_time_us();

    for (int iter = 0; iter < 100; ++iter) {
        float features[] = {static_cast<float>(iter % 180)};

        for (auto* hcol : hypercolumns) {
            feature_hypercolumn_process(hcol, features, 1);
            feature_hypercolumn_normalize(hcol);
        }
    }

    uint64_t end_time = nimcp_platform_get_time_us();
    float elapsed_ms = (end_time - start_time) / 1000.0f;

    EXPECT_LT(elapsed_ms, 1000.0f);  // Should complete in < 1 second

    // Cleanup
    for (auto* hcol : hypercolumns) {
        feature_hypercolumn_destroy(hcol);
    }
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize NIMCP
    nimcp_logging_init(NIMCP_LOG_LEVEL_INFO);

    int result = RUN_ALL_TESTS();

    // Cleanup
    nimcp_logging_shutdown();

    return result;
}
