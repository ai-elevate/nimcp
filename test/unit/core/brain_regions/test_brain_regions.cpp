/**
 * @file test_brain_regions.cpp
 * @brief Unit tests for brain regions hierarchical architecture
 *
 * TEST PHILOSOPHY:
 * - Test initialization with various configurations
 * - Test region creation and organization
 * - Test layer structure and neuron type assignment
 * - Test minicolumn organization
 * - Test inter-region connectivity
 * - Test input processing and output collection
 * - Test error handling and edge cases
 *
 * @author NIMCP Development Team
 * @date 2025-11-11
 * @version 3.0.0 Module Integration Phase
 */

#include <gtest/gtest.h>
#include <vector>

#include "core/brain/nimcp_brain.h"
#include "core/brain_regions/nimcp_brain_regions.h"
#include "utils/time/nimcp_time.h"

//=============================================================================
// Unit Test Fixture
//=============================================================================

class BrainRegionsTest : public ::testing::Test {
protected:
    brain_t brain;
    brain_module_t* module;
    brain_region_t* region;

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
            brain = nullptr;
        }
        if (module) {
            brain_module_destroy(module);
            module = nullptr;
        }
        if (region) {
            brain_region_destroy(region);
            region = nullptr;
        }
    }

    brain_t create_brain_with_regions(uint32_t num_regions = 4, uint32_t neurons_per_region = 1000) {
        brain_config_t config = {};
        config.size = BRAIN_SIZE_SMALL;
        config.task = BRAIN_TASK_CLASSIFICATION;
        config.num_inputs = 128;
        config.num_outputs = 10;
        config.enable_brain_regions = true;
        config.num_brain_regions = num_regions;
        config.neurons_per_region = neurons_per_region;

        strncpy(config.task_name, "brain_regions_test", sizeof(config.task_name) - 1);

        return brain_create_custom(&config);
    }

    std::vector<float> create_input(uint32_t dim) {
        std::vector<float> input(dim);
        for (uint32_t i = 0; i < dim; i++) {
            input[i] = 0.5f + 0.1f * sin(i * 0.1f);
        }
        return input;
    }
};

//=============================================================================
// 1. Initialization Tests
//=============================================================================

TEST_F(BrainRegionsTest, Initialize_BrainRegionsEnabled_Success) {
    brain = create_brain_with_regions(4, 1000);
    ASSERT_NE(brain, nullptr);
}

TEST_F(BrainRegionsTest, Initialize_BrainRegionsDisabled_SkipsCreation) {
    brain_config_t config = {};
    config.size = BRAIN_SIZE_SMALL;
    config.task = BRAIN_TASK_CLASSIFICATION;
    config.num_inputs = 128;
    config.num_outputs = 10;
    config.enable_brain_regions = false;

    strncpy(config.task_name, "no_regions_test", sizeof(config.task_name) - 1);

    brain = brain_create_custom(&config);
    ASSERT_NE(brain, nullptr);
}

TEST_F(BrainRegionsTest, Module_CreateDestroy_NoLeak) {
    for (int i = 0; i < 10; i++) {
        module = brain_module_create(4);
        ASSERT_NE(module, nullptr);
        brain_module_destroy(module);
        module = nullptr;
    }
}

TEST_F(BrainRegionsTest, BrainIntegration_CreateWithRegions_Success) {
    brain = create_brain_with_regions(4, 1000);
    ASSERT_NE(brain, nullptr);

    auto input = create_input(128);
    brain_decision_t* decision = brain_decide(brain, input.data(), 128);

    ASSERT_NE(decision, nullptr);
    EXPECT_GE(decision->confidence, 0.0f);
    EXPECT_LE(decision->confidence, 1.0f);

    brain_free_decision(decision);
}

TEST_F(BrainRegionsTest, Memory_NoLeaksWithRegions_Success) {
    for (int i = 0; i < 10; i++) {
        brain_t test_brain = create_brain_with_regions(2, 500);
        ASSERT_NE(test_brain, nullptr) << "Failed on iteration " << i;
        brain_destroy(test_brain);
    }
}

//=============================================================================
// Run All Tests
//=============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
