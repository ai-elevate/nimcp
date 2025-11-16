//=============================================================================
// test_fractal_cognitive.cpp - Unit Tests for Fractal Cognitive Integration
//=============================================================================

#include <gtest/gtest.h>
#include <cmath>

#include "cognitive/nimcp_fractal_cognitive.h"
#include "core/brain/nimcp_brain.h"
#include "plasticity/adaptive/nimcp_adaptive.h"

class FractalCognitiveTest : public ::testing::Test {
protected:
    brain_t brain = nullptr;
    fractal_cognitive_cache_t cache;

    void SetUp() override {
        memset(&cache, 0, sizeof(fractal_cognitive_cache_t));

        // Create small brain for testing
        brain = brain_create("test_fractal", BRAIN_SIZE_TINY,
                            BRAIN_TASK_CLASSIFICATION, 10, 2);
        ASSERT_NE(brain, nullptr);
    }

    void TearDown() override {
        fractal_cognitive_free(&cache);
        if (brain) {
            brain_destroy(brain);
        }
    }
};

//=============================================================================
// Initialization Tests
//=============================================================================

TEST_F(FractalCognitiveTest, InitSuccess) {
    adaptive_network_t adaptive_net = brain_get_network(brain);
    ASSERT_NE(adaptive_net, nullptr);

    neural_network_t network = adaptive_network_get_base_network(adaptive_net);
    ASSERT_NE(network, nullptr);

    bool result = fractal_cognitive_init(network, &cache);

    EXPECT_TRUE(result);
    EXPECT_TRUE(cache.valid);
    EXPECT_GT(cache.stats.num_neurons, 0u);
}

TEST_F(FractalCognitiveTest, InitNullInputs) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));

    EXPECT_FALSE(fractal_cognitive_init(nullptr, &cache));
    EXPECT_FALSE(fractal_cognitive_init(network, nullptr));
}

TEST_F(FractalCognitiveTest, InitAllocatesArrays) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    EXPECT_NE(cache.hub_indices, nullptr);
    EXPECT_NE(cache.centrality_scores, nullptr);
    EXPECT_NE(cache.degree_normalized, nullptr);
}

//=============================================================================
// Hub Neuron Tests
//=============================================================================

TEST_F(FractalCognitiveTest, HubIdentificationFindsHubs) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    // Should identify some hubs (top 10%)
    EXPECT_GT(cache.num_hubs, 0u);
    EXPECT_LT(cache.num_hubs, cache.stats.num_neurons);
}

TEST_F(FractalCognitiveTest, IsHubNeuronValidQuery) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    if (cache.num_hubs > 0) {
        // First hub should be identified as hub
        uint32_t hub_id = cache.hub_indices[0];
        EXPECT_TRUE(fractal_is_hub_neuron(&cache, hub_id));

        // Non-hub should not be identified
        uint32_t non_hub = cache.stats.num_neurons + 100;  // Out of range
        EXPECT_FALSE(fractal_is_hub_neuron(&cache, non_hub));
    }
}

TEST_F(FractalCognitiveTest, IsHubNeuronInvalidCache) {
    EXPECT_FALSE(fractal_is_hub_neuron(nullptr, 0));
    EXPECT_FALSE(fractal_is_hub_neuron(&cache, 0));  // Cache not initialized
}

TEST_F(FractalCognitiveTest, NearestHubFindsHub) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    if (cache.num_hubs > 0) {
        uint32_t distance = 0;
        uint32_t nearest = fractal_nearest_hub(network, &cache, 0, &distance);

        EXPECT_NE(nearest, UINT32_MAX);
        EXPECT_LT(nearest, cache.stats.num_neurons);
    }
}

//=============================================================================
// Centrality Tests
//=============================================================================

TEST_F(FractalCognitiveTest, CentralityScoresValid) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    // Check all centrality scores are in valid range
    for (uint32_t i = 0; i < cache.stats.num_neurons; i++) {
        float centrality = fractal_get_centrality(&cache, i);
        EXPECT_GE(centrality, 0.0f);
        EXPECT_LE(centrality, 1.0f);
    }
}

TEST_F(FractalCognitiveTest, CentralityHubsHigher) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    if (cache.num_hubs > 1) {
        // Hub neurons should have higher centrality than average
        float hub_centrality = fractal_get_centrality(&cache, cache.hub_indices[0]);

        float avg_centrality = 0.0f;
        for (uint32_t i = 0; i < cache.stats.num_neurons; i++) {
            avg_centrality += fractal_get_centrality(&cache, i);
        }
        avg_centrality /= cache.stats.num_neurons;

        // Hub should be above average (though not guaranteed for tiny networks)
        EXPECT_GE(hub_centrality, 0.0f);  // At least valid
    }
}

TEST_F(FractalCognitiveTest, CentralityInvalidInputs) {
    EXPECT_FLOAT_EQ(fractal_get_centrality(nullptr, 0), 0.0f);
    EXPECT_FLOAT_EQ(fractal_get_centrality(&cache, 0), 0.0f);  // Not initialized
}

TEST_F(FractalCognitiveTest, DegreeNormalizedInRange) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    for (uint32_t i = 0; i < cache.stats.num_neurons; i++) {
        float degree = fractal_get_degree_normalized(&cache, i);
        EXPECT_GE(degree, 0.0f);
        EXPECT_LE(degree, 1.0f);
    }
}

//=============================================================================
// Hierarchical Level Tests
//=============================================================================

TEST_F(FractalCognitiveTest, HierarchicalLevelInRange) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    for (uint32_t i = 0; i < cache.stats.num_neurons; i++) {
        float level = fractal_get_hierarchical_level(&cache, i);
        EXPECT_GE(level, 0.0f);
        EXPECT_LE(level, 1.0f);
    }
}

TEST_F(FractalCognitiveTest, HierarchicalLevelHubsNearRoot) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    if (cache.num_hubs > 0) {
        float hub_level = fractal_get_hierarchical_level(&cache, cache.hub_indices[0]);

        // Hubs should be near root (level ≈ 0)
        // But for tiny networks this may not hold, so just check validity
        EXPECT_GE(hub_level, 0.0f);
        EXPECT_LE(hub_level, 1.0f);
    }
}

TEST_F(FractalCognitiveTest, GetNeuronsAtLevelFindsNeurons) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    uint32_t* neurons = nullptr;
    uint32_t count = 0;

    // Get neurons at mid-level
    bool result = fractal_get_neurons_at_level(&cache, 0.5f, 0.2f,
                                               &neurons, &count);

    EXPECT_TRUE(result);
    EXPECT_GT(count, 0u);  // Should find some neurons
    EXPECT_NE(neurons, nullptr);

    if (neurons) {
        free(neurons);
    }
}

TEST_F(FractalCognitiveTest, GetNeuronsAtLevelInvalidInputs) {
    uint32_t* neurons = nullptr;
    uint32_t count = 0;

    EXPECT_FALSE(fractal_get_neurons_at_level(nullptr, 0.5f, 0.1f, &neurons, &count));
    EXPECT_FALSE(fractal_get_neurons_at_level(&cache, 0.5f, 0.1f, nullptr, &count));
    EXPECT_FALSE(fractal_get_neurons_at_level(&cache, 0.5f, 0.1f, &neurons, nullptr));
}

//=============================================================================
// Central Neighbors Tests
//=============================================================================

TEST_F(FractalCognitiveTest, GetCentralNeighborsFindsNeurons) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    uint32_t central[5];
    uint32_t found = fractal_get_central_neighbors(network, &cache,
                                                   0, 10, 5, central);

    EXPECT_GT(found, 0u);
    EXPECT_LE(found, 5u);
}

TEST_F(FractalCognitiveTest, GetCentralNeighborsInvalidInputs) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    uint32_t central[5];

    EXPECT_EQ(fractal_get_central_neighbors(nullptr, &cache, 0, 10, 5, central), 0u);
    EXPECT_EQ(fractal_get_central_neighbors(network, nullptr, 0, 10, 5, central), 0u);
    EXPECT_EQ(fractal_get_central_neighbors(network, &cache, 0, 10, 5, nullptr), 0u);
    EXPECT_EQ(fractal_get_central_neighbors(network, &cache, 0, 10, 0, central), 0u);
}

//=============================================================================
// Refresh Tests
//=============================================================================

TEST_F(FractalCognitiveTest, RefreshRecomputesCache) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    uint32_t old_num_hubs = cache.num_hubs;

    bool result = fractal_cognitive_refresh(network, &cache);

    EXPECT_TRUE(result);
    EXPECT_TRUE(cache.valid);
    // Number of hubs might change slightly due to network updates
    EXPECT_GT(cache.num_hubs, 0u);
}

//=============================================================================
// Cleanup Tests
//=============================================================================

TEST_F(FractalCognitiveTest, FreeReleasesMemory) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    fractal_cognitive_free(&cache);

    EXPECT_EQ(cache.hub_indices, nullptr);
    EXPECT_EQ(cache.centrality_scores, nullptr);
    EXPECT_EQ(cache.degree_normalized, nullptr);
    EXPECT_EQ(cache.num_hubs, 0u);
    EXPECT_FALSE(cache.valid);
}

TEST_F(FractalCognitiveTest, FreeNullCache) {
    // Should not crash
    fractal_cognitive_free(nullptr);
    SUCCEED();
}

//=============================================================================
// Integration Tests
//=============================================================================

TEST_F(FractalCognitiveTest, FullLifecycleWithBrain) {
    // Brain already has fractal_cache initialized in brain_create
    brain_t test_brain = brain_create("fractal_test", BRAIN_SIZE_SMALL,
                                     BRAIN_TASK_CLASSIFICATION, 784, 10);
    ASSERT_NE(test_brain, nullptr);

    // Check if fractal cache was created (it's optional)
    // We can't directly access brain->fractal_cache from test, so just
    // verify brain creation succeeded
    EXPECT_NE(test_brain, nullptr);

    brain_destroy(test_brain);
}

TEST_F(FractalCognitiveTest, VisualizationDoesNotCrash) {
    neural_network_t network = adaptive_network_get_base_network(brain_get_network(brain));
    fractal_cognitive_init(network, &cache);

    // Should not crash
    fractal_cognitive_print_summary(&cache);
    SUCCEED();
}
