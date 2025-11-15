/**
 * @file test_glial_integration.cpp
 * @brief TDD tests for glial-neuron integration
 *
 * Test coverage:
 * - Creation/destruction
 * - Glial network assignment
 * - Neuron/synapse → glial cell mapping
 * - Event notifications (synapse/neuron fired)
 * - Glial modulation queries
 * - Simulation step
 * - Statistics and monitoring
 * - Integration with real neural network
 *
 * TDD APPROACH: Write tests first, then implement
 */

#include <gtest/gtest.h>

#include "glial/integration/nimcp_glial_integration.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "glial/astrocytes/nimcp_astrocytes.h"
#include "glial/oligodendrocytes/nimcp_oligodendrocytes.h"
#include "glial/microglia/nimcp_microglia.h"
#include "utils/time/nimcp_time.h"

class GlialIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple neural network
        network_config_t config{};
        config.num_neurons = 10;
        config.ei_ratio = 0.8f;
        config.learning_rate = 0.01f;
        config.min_weight = -1.0f;
        config.max_weight = 1.0f;
        network = neural_network_create(&config);
        ASSERT_NE(network, nullptr);

        // Create glial cell networks
        astrocyte_net = astrocyte_network_create(5);
        ASSERT_NE(astrocyte_net, nullptr);

        oligo_net = oligodendrocyte_network_create(5);
        ASSERT_NE(oligo_net, nullptr);

        microglia_net = microglia_network_create(5);
        ASSERT_NE(microglia_net, nullptr);

        // Add some glial cells
        for (uint32_t i = 0; i < 3; i++) {
            astrocyte_t* ast = astrocyte_create(i, ASTROCYTE_TYPE_GENERIC, i * 100.0f, 0.0f, 0.0f, 50.0f);
            ASSERT_NE(ast, nullptr);
            astrocyte_network_add(astrocyte_net, ast);

            oligodendrocyte_t* oligo = oligodendrocyte_create(i, 10); // id, max_axons
            ASSERT_NE(oligo, nullptr);
            oligodendrocyte_network_add(oligo_net, oligo);

            microglia_t* mg = microglia_create(i, i * 100.0f, 0.0f, 0.0f, 100.0f);
            ASSERT_NE(mg, nullptr);
            microglia_network_add(microglia_net, mg);
        }

        // Add some connections to the network
        neural_network_add_connection(network, 0, 1, 0.5f);
        neural_network_add_connection(network, 1, 2, 0.7f);
        neural_network_add_connection(network, 2, 3, 0.3f); // Weak synapse
    }

    void TearDown() override {
        if (gi) {
            glial_integration_destroy(gi);
        }
        if (network) {
            neural_network_destroy(network);
        }
        if (astrocyte_net) {
            astrocyte_network_destroy(astrocyte_net);
        }
        if (oligo_net) {
            oligodendrocyte_network_destroy(oligo_net);
        }
        if (microglia_net) {
            microglia_network_destroy(microglia_net);
        }
    }

    neural_network_t network = nullptr;
    astrocyte_network_t* astrocyte_net = nullptr;
    oligodendrocyte_network_t* oligo_net = nullptr;
    microglia_network_t* microglia_net = nullptr;
    glial_integration_t* gi = nullptr;
};

// ============================================================================
// CREATION & DESTRUCTION
// ============================================================================

TEST_F(GlialIntegrationTest, CreateDestroy) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    EXPECT_EQ(gi->network, network);
    EXPECT_FALSE(gi->enable_astrocyte_modulation); // Default: disabled
    EXPECT_FALSE(gi->enable_oligodendrocyte_myelination);
    EXPECT_FALSE(gi->enable_microglia_pruning);

    glial_integration_destroy(gi);
    gi = nullptr; // Prevent double-free in TearDown
}

TEST_F(GlialIntegrationTest, CreateWithNullNetwork) {
    gi = glial_integration_create(nullptr, 100);
    EXPECT_EQ(gi, nullptr); // Should fail gracefully
}

TEST_F(GlialIntegrationTest, DestroyNull) {
    glial_integration_destroy(nullptr); // Should not crash
}

// ============================================================================
// GLIAL NETWORK ASSIGNMENT
// ============================================================================

TEST_F(GlialIntegrationTest, AssignGlialNetworks) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);

    nimcp_result_t result = glial_integration_set_astrocyte_network(gi, astrocyte_net);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(gi->astrocyte_network, astrocyte_net);

    result = glial_integration_set_oligodendrocyte_network(gi, oligo_net);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(gi->oligodendrocyte_network, oligo_net);

    result = glial_integration_set_microglia_network(gi, microglia_net);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(gi->microglia_network, microglia_net);
}

// ============================================================================
// GLIAL CELL ASSIGNMENT
// ============================================================================

TEST_F(GlialIntegrationTest, AssignAstrocyteToSynapse) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_astrocyte_network(gi, astrocyte_net);

    // Synapse ID encoding: pre_neuron_id * 10000 + post_neuron_id
    uint32_t synapse_id = 0 * 10000 + 1; // Neuron 0 → 1

    nimcp_result_t result = glial_integration_assign_astrocyte_to_synapse(gi, 0, synapse_id);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Verify assignment
    uint32_t count = glial_integration_get_astrocyte_synapse_count(gi, 0);
    EXPECT_EQ(count, 1);
}

TEST_F(GlialIntegrationTest, AssignOligodendrocyteToNeuron) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_oligodendrocyte_network(gi, oligo_net);

    nimcp_result_t result = glial_integration_assign_oligodendrocyte_to_neuron(gi, 0, 5);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Verify assignment
    uint32_t count = glial_integration_get_oligodendrocyte_neuron_count(gi, 0);
    EXPECT_EQ(count, 1);
}

TEST_F(GlialIntegrationTest, AssignMicrogliaToSynapse) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_microglia_network(gi, microglia_net);

    uint32_t synapse_id = 1 * 10000 + 2; // Neuron 1 → 2

    nimcp_result_t result = glial_integration_assign_microglia_to_synapse(gi, 0, synapse_id);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(GlialIntegrationTest, MultipleAssignments) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_astrocyte_network(gi, astrocyte_net);

    // Assign multiple synapses to same astrocyte
    glial_integration_assign_astrocyte_to_synapse(gi, 0, 0 * 10000 + 1);
    glial_integration_assign_astrocyte_to_synapse(gi, 0, 1 * 10000 + 2);
    glial_integration_assign_astrocyte_to_synapse(gi, 0, 2 * 10000 + 3);

    uint32_t count = glial_integration_get_astrocyte_synapse_count(gi, 0);
    EXPECT_EQ(count, 3);
}

// ============================================================================
// EVENT NOTIFICATIONS
// ============================================================================

TEST_F(GlialIntegrationTest, SynapseFiredNotification) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_astrocyte_network(gi, astrocyte_net);
    glial_integration_set_astrocyte_modulation_enabled(gi, true);

    uint32_t synapse_id = 0 * 10000 + 1;
    glial_integration_assign_astrocyte_to_synapse(gi, 0, synapse_id);

    uint64_t timestamp = nimcp_time_monotonic_us();

    // Notify that synapse fired
    glial_integration_on_synapse_fired(gi, 0, 1, 0.5f, timestamp);

    // Astrocyte should have increased calcium (test indirectly via modulation)
    float modulation = glial_integration_get_synaptic_modulation(gi, 0, 1);
    EXPECT_GT(modulation, 0.0f); // Should return some modulation
}

TEST_F(GlialIntegrationTest, NeuronFiredNotification) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_oligodendrocyte_network(gi, oligo_net);
    glial_integration_set_oligodendrocyte_myelination_enabled(gi, true);

    glial_integration_assign_oligodendrocyte_to_neuron(gi, 0, 5);

    uint64_t timestamp = nimcp_time_monotonic_us();

    // Notify that neuron fired
    glial_integration_on_neuron_fired(gi, 5, timestamp);

    // Oligodendrocyte should track activity for myelination
    float myelination = glial_integration_get_myelination_factor(gi, 5);
    EXPECT_GE(myelination, 0.0f);
}

// ============================================================================
// GLIAL MODULATION QUERIES
// ============================================================================

TEST_F(GlialIntegrationTest, SynapticModulation_NoAstrocyte) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);

    // No astrocyte assigned → modulation should be 1.0 (neutral)
    float modulation = glial_integration_get_synaptic_modulation(gi, 0, 1);
    EXPECT_FLOAT_EQ(modulation, 1.0f);
}

TEST_F(GlialIntegrationTest, SynapticModulation_WithAstrocyte) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_astrocyte_network(gi, astrocyte_net);
    glial_integration_set_astrocyte_modulation_enabled(gi, true);

    uint32_t synapse_id = 0 * 10000 + 1;
    glial_integration_assign_astrocyte_to_synapse(gi, 0, synapse_id);

    // Get astrocyte and increase calcium
    astrocyte_t* ast = astrocyte_net->astrocytes[0];
    ASSERT_NE(ast, nullptr);

    // Update calcium (increases it)
    astrocyte_update_calcium(ast, 50.0f, nimcp_time_monotonic_us());

    // Should get modulation factor (0.8 - 1.2 range)
    float modulation = glial_integration_get_synaptic_modulation(gi, 0, 1);
    EXPECT_GE(modulation, 0.8f);
    EXPECT_LE(modulation, 1.2f);
}

TEST_F(GlialIntegrationTest, MyelinationFactor_NoOligodendrocyte) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);

    // No oligodendrocyte → myelination = 0.0
    float myelination = glial_integration_get_myelination_factor(gi, 5);
    EXPECT_FLOAT_EQ(myelination, 0.0f);
}

TEST_F(GlialIntegrationTest, MyelinationFactor_WithOligodendrocyte) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_oligodendrocyte_network(gi, oligo_net);
    glial_integration_set_oligodendrocyte_myelination_enabled(gi, true);

    glial_integration_assign_oligodendrocyte_to_neuron(gi, 0, 5);

    // Get oligodendrocyte and assign neuron for myelination
    oligodendrocyte_t* oligo = oligo_net->oligodendrocytes[0];
    ASSERT_NE(oligo, nullptr);

    // Assign neuron (this initiates myelination)
    oligodendrocyte_assign_neuron(oligo, 5);

    // Track some activity for neuron 5 (the assigned neuron)
    uint64_t timestamp = nimcp_time_monotonic_us();
    for (int i = 0; i < 10; i++) {
        oligodendrocyte_track_activity(oligo, 5, 1.0f, timestamp + i * 1000);
    }

    // Remodel myelination
    oligodendrocyte_remodel_myelination(oligo, timestamp + 20000);

    // Should get myelination factor (0.0 - 1.0)
    float myelination = glial_integration_get_myelination_factor(gi, 5);
    EXPECT_GT(myelination, 0.0f);
    EXPECT_LE(myelination, 1.0f);
}

TEST_F(GlialIntegrationTest, ShouldPruneSynapse_NoMicroglia) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);

    // No microglia → should not prune
    bool should_prune = glial_integration_should_prune_synapse(gi, 0, 1);
    EXPECT_FALSE(should_prune);
}

TEST_F(GlialIntegrationTest, ShouldPruneSynapse_WeakSynapse) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_microglia_network(gi, microglia_net);
    glial_integration_set_microglia_pruning_enabled(gi, true);

    uint32_t synapse_id = 2 * 10000 + 3; // Weak synapse (weight 0.3)
    glial_integration_assign_microglia_to_synapse(gi, 0, synapse_id);

    // Get microglia and monitor synapse
    microglia_t* mg = microglia_net->microglia[0];
    ASSERT_NE(mg, nullptr);
    microglia_monitor_synapse(mg, synapse_id);

    // Track low activity
    uint64_t timestamp = nimcp_time_monotonic_us();
    microglia_track_synapse_activity(mg, synapse_id, 0.05f, timestamp); // Very low activity

    // Update scores and identify weak synapses
    microglia_update_activity_scores(mg, timestamp + 1000000); // 1 sec later

    uint32_t weak_synapses[10];
    uint32_t num_weak = microglia_identify_weak_synapses(mg, weak_synapses, 10);

    if (num_weak > 0) {
        // Should mark for pruning
        bool should_prune = glial_integration_should_prune_synapse(gi, 2, 3);
        EXPECT_TRUE(should_prune);
    }
}

// ============================================================================
// SIMULATION STEP
// ============================================================================

TEST_F(GlialIntegrationTest, SimulationStep) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_astrocyte_network(gi, astrocyte_net);
    glial_integration_set_oligodendrocyte_network(gi, oligo_net);
    glial_integration_set_microglia_network(gi, microglia_net);

    uint64_t timestamp = nimcp_time_monotonic_us();

    // Step should not crash
    glial_integration_step(gi, timestamp);
    glial_integration_step(gi, timestamp + 1000);
    glial_integration_step(gi, timestamp + 2000);
}

TEST_F(GlialIntegrationTest, SimulationStepWithActivity) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_astrocyte_network(gi, astrocyte_net);
    glial_integration_set_astrocyte_modulation_enabled(gi, true);

    uint32_t synapse_id = 0 * 10000 + 1;
    glial_integration_assign_astrocyte_to_synapse(gi, 0, synapse_id);

    uint64_t timestamp = nimcp_time_monotonic_us();

    // Fire synapse multiple times
    for (int i = 0; i < 10; i++) {
        glial_integration_on_synapse_fired(gi, 0, 1, 0.5f, timestamp + i * 1000);
        glial_integration_step(gi, timestamp + i * 1000);
    }

    // Astrocyte calcium should have increased
    astrocyte_t* ast = astrocyte_net->astrocytes[0];
    ASSERT_NE(ast, nullptr);

    // Check calcium increased (compare to baseline)
    EXPECT_GT(ast->calcium_concentration, ast->calcium_baseline);
}

// ============================================================================
// STATISTICS
// ============================================================================

TEST_F(GlialIntegrationTest, GetStatistics) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_astrocyte_network(gi, astrocyte_net);
    glial_integration_set_oligodendrocyte_network(gi, oligo_net);
    glial_integration_set_microglia_network(gi, microglia_net);

    glial_integration_stats_t stats{};
    nimcp_result_t result = glial_integration_get_stats(gi, &stats);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(stats.num_astrocytes, 3);
    EXPECT_EQ(stats.num_oligodendrocytes, 3);
    EXPECT_EQ(stats.num_microglia, 3);
}

TEST_F(GlialIntegrationTest, StatisticsWithAssignments) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);
    glial_integration_set_astrocyte_network(gi, astrocyte_net);
    glial_integration_set_oligodendrocyte_network(gi, oligo_net);
    glial_integration_set_microglia_network(gi, microglia_net);

    // Make assignments
    glial_integration_assign_astrocyte_to_synapse(gi, 0, 0 * 10000 + 1);
    glial_integration_assign_astrocyte_to_synapse(gi, 0, 1 * 10000 + 2);
    glial_integration_assign_oligodendrocyte_to_neuron(gi, 0, 5);
    glial_integration_assign_microglia_to_synapse(gi, 0, 2 * 10000 + 3);

    glial_integration_stats_t stats{};
    glial_integration_get_stats(gi, &stats);

    EXPECT_EQ(stats.num_tripartite_synapses, 2);
    EXPECT_EQ(stats.num_myelinated_neurons, 1);
    EXPECT_EQ(stats.num_monitored_synapses, 1);
}

// ============================================================================
// CONFIGURATION
// ============================================================================

TEST_F(GlialIntegrationTest, EnableDisableModulation) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);

    // Initially disabled
    EXPECT_FALSE(gi->enable_astrocyte_modulation);
    EXPECT_FALSE(gi->enable_oligodendrocyte_myelination);
    EXPECT_FALSE(gi->enable_microglia_pruning);

    // Enable
    glial_integration_set_astrocyte_modulation_enabled(gi, true);
    EXPECT_TRUE(gi->enable_astrocyte_modulation);

    glial_integration_set_oligodendrocyte_myelination_enabled(gi, true);
    EXPECT_TRUE(gi->enable_oligodendrocyte_myelination);

    glial_integration_set_microglia_pruning_enabled(gi, true);
    EXPECT_TRUE(gi->enable_microglia_pruning);

    // Disable
    glial_integration_set_astrocyte_modulation_enabled(gi, false);
    EXPECT_FALSE(gi->enable_astrocyte_modulation);
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEST_F(GlialIntegrationTest, QueryUnassignedSynapse) {
    gi = glial_integration_create(network, 100);
    ASSERT_NE(gi, nullptr);

    // Query synapse that has no glial cells assigned
    float modulation = glial_integration_get_synaptic_modulation(gi, 99, 100);
    EXPECT_FLOAT_EQ(modulation, 1.0f); // Neutral

    bool should_prune = glial_integration_should_prune_synapse(gi, 99, 100);
    EXPECT_FALSE(should_prune);
}

TEST_F(GlialIntegrationTest, NullParameterHandling) {
    glial_integration_on_synapse_fired(nullptr, 0, 1, 0.5f, 0);
    glial_integration_on_neuron_fired(nullptr, 0, 0);
    glial_integration_step(nullptr, 0);

    float modulation = glial_integration_get_synaptic_modulation(nullptr, 0, 1);
    EXPECT_FLOAT_EQ(modulation, 1.0f);

    float myelination = glial_integration_get_myelination_factor(nullptr, 0);
    EXPECT_FLOAT_EQ(myelination, 0.0f);
}

// ============================================================================
// PERFORMANCE
// ============================================================================

TEST_F(GlialIntegrationTest, PerformanceLargeNetwork) {
    // Create larger network
    network_config_t config{};
    config.num_neurons = 1000;
    config.ei_ratio = 0.8f;
    config.learning_rate = 0.01f;
    neural_network_t large_network = neural_network_create(&config);
    ASSERT_NE(large_network, nullptr);

    gi = glial_integration_create(large_network, 10000);
    ASSERT_NE(gi, nullptr);

    glial_integration_set_astrocyte_network(gi, astrocyte_net);

    // Assign many synapses
    for (uint32_t i = 0; i < 100; i++) {
        uint32_t pre = i;
        uint32_t post = (i + 1) % 1000;
        uint32_t synapse_id = pre * 10000 + post;
        glial_integration_assign_astrocyte_to_synapse(gi, i % 3, synapse_id);
    }

    // Query should be fast (O(1) hash lookup)
    uint64_t start = nimcp_time_monotonic_us();
    for (uint32_t i = 0; i < 100; i++) {
        uint32_t pre = i;
        uint32_t post = (i + 1) % 1000;
        glial_integration_get_synaptic_modulation(gi, pre, post);
    }
    uint64_t end = nimcp_time_monotonic_us();

    uint64_t elapsed = end - start;
    EXPECT_LT(elapsed, 1000); // Should complete in < 1ms

    neural_network_destroy(large_network);
}
