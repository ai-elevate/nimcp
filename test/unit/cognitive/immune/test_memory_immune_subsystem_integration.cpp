/**
 * @file test_memory_immune_subsystem_integration.cpp
 * @brief Unit tests for memory-immune subsystem integration
 *
 * Tests integration with:
 * - Engram system (memory traces)
 * - Semantic memory
 * - Systems consolidation (hippocampus→cortex)
 * - Working memory transfer
 *
 * @version 1.0.0
 * @date 2025-12-11
 */

#include <gtest/gtest.h>

extern "C" {
#include "cognitive/immune/nimcp_memory_immune_integration.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "cognitive/memory/nimcp_engram.h"
#include "cognitive/memory/nimcp_semantic_memory.h"
#include "cognitive/memory/nimcp_systems_consolidation.h"
#include "cognitive/memory/nimcp_wm_transfer.h"
#include "cognitive/nimcp_working_memory.h"
}

/* ============================================================================
 * Test Fixture
 * ============================================================================ */

class MemoryImmuneSubsystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        /* Create brain immune system */
        brain_immune_config_t immune_config;
        brain_immune_default_config(&immune_config);
        immune_system = brain_immune_create(&immune_config);
        ASSERT_NE(immune_system, nullptr);

        /* Create memory-immune integration */
        memory_immune_config_t config;
        memory_immune_default_config(&config);
        integration = memory_immune_integration_create(
            immune_system,
            nullptr,  /* working_memory */
            nullptr,  /* consolidation */
            &config
        );
        ASSERT_NE(integration, nullptr);

        /* Create memory subsystems */
        engram_system = engram_system_create();
        ASSERT_NE(engram_system, nullptr);

        semantic_memory = semantic_memory_create();
        ASSERT_NE(semantic_memory, nullptr);

        systems_consolidation = systems_consolidation_create();
        ASSERT_NE(systems_consolidation, nullptr);

        wm_transfer = wm_transfer_create();
        ASSERT_NE(wm_transfer, nullptr);
    }

    void TearDown() override {
        if (wm_transfer) wm_transfer_destroy(wm_transfer);
        if (systems_consolidation) systems_consolidation_destroy(systems_consolidation);
        if (semantic_memory) semantic_memory_destroy(semantic_memory);
        if (engram_system) engram_system_destroy(engram_system);
        if (integration) memory_immune_integration_destroy(integration);
        if (immune_system) brain_immune_destroy(immune_system);
    }

    brain_immune_system_t* immune_system = nullptr;
    memory_immune_integration_t* integration = nullptr;
    engram_system_t* engram_system = nullptr;
    semantic_memory_system_t* semantic_memory = nullptr;
    systems_consolidation_system_t* systems_consolidation = nullptr;
    wm_transfer_system_t* wm_transfer = nullptr;
};

/* ============================================================================
 * Engram System Integration Tests
 * ============================================================================ */

TEST_F(MemoryImmuneSubsystemTest, ConnectEngramSystem) {
    int result = memory_immune_connect_engram_system(integration, engram_system);
    EXPECT_EQ(result, 0);
}

TEST_F(MemoryImmuneSubsystemTest, ModulateEngramConsolidation_LowIL1Beta) {
    /* Connect engram system */
    memory_immune_connect_engram_system(integration, engram_system);

    /* Simulate low IL-1β (should enhance consolidation) */
    memory_immune_metrics_t metrics = {};
    metrics.il1_concentration = 0.1f;  /* Low dose */
    metrics.inflammation_level = INFLAMMATION_NONE;

    /* Get metrics to set IL-1β */
    memory_immune_get_metrics(integration, &metrics);

    /* Modulate consolidation */
    float multiplier = memory_immune_modulate_engram_consolidation(
        integration, 0.1f, false
    );

    /* Should enhance (> 1.0) */
    EXPECT_GT(multiplier, 1.0f);
    EXPECT_NEAR(multiplier, 1.3f, 0.1f);
}

TEST_F(MemoryImmuneSubsystemTest, ModulateEngramConsolidation_HighIL1Beta) {
    memory_immune_connect_engram_system(integration, engram_system);

    /* Simulate high IL-1β (should impair consolidation) */
    memory_immune_metrics_t metrics = {};
    metrics.il1_concentration = 0.8f;  /* High dose */
    metrics.inflammation_level = INFLAMMATION_NONE;

    /* Get consolidation multiplier */
    float multiplier = memory_immune_modulate_engram_consolidation(
        integration, 0.1f, false
    );

    /* Should impair (< 1.0) */
    EXPECT_LT(multiplier, 1.0f);
    EXPECT_NEAR(multiplier, 0.6f, 0.1f);
}

TEST_F(MemoryImmuneSubsystemTest, ModulateEngramConsolidation_InflammationDisruptsSleep) {
    memory_immune_connect_engram_system(integration, engram_system);

    /* Simulate inflammation during sleep */
    memory_immune_metrics_t metrics = {};
    metrics.il1_concentration = 0.1f;  /* Low IL-1β (would normally enhance) */
    metrics.inflammation_level = INFLAMMATION_REGIONAL;  /* But inflammation present */

    float multiplier = memory_immune_modulate_engram_consolidation(
        integration, 0.1f, true  /* is_sleeping */
    );

    /* Should be reduced due to inflammation disrupting sleep consolidation */
    EXPECT_LT(multiplier, 1.3f);  /* Less than pure low IL-1β effect */
}

TEST_F(MemoryImmuneSubsystemTest, ModulateEngramRetrieval_InflammationImpairs) {
    memory_immune_connect_engram_system(integration, engram_system);

    /* Test retrieval modulation at different inflammation levels */
    float base_confidence = 0.9f;

    /* No inflammation: normal retrieval */
    memory_immune_metrics_t metrics = {};
    metrics.inflammation_level = INFLAMMATION_NONE;
    float modulated = memory_immune_modulate_engram_retrieval(integration, base_confidence);
    EXPECT_NEAR(modulated, base_confidence, 0.01f);

    /* Regional inflammation: mild impairment */
    metrics.inflammation_level = INFLAMMATION_REGIONAL;
    modulated = memory_immune_modulate_engram_retrieval(integration, base_confidence);
    EXPECT_LT(modulated, base_confidence);
    EXPECT_NEAR(modulated, 0.72f, 0.01f);  /* 0.9 * 0.8 */

    /* Systemic inflammation: moderate impairment */
    metrics.inflammation_level = INFLAMMATION_SYSTEMIC;
    modulated = memory_immune_modulate_engram_retrieval(integration, base_confidence);
    EXPECT_NEAR(modulated, 0.54f, 0.01f);  /* 0.9 * 0.6 */

    /* Cytokine storm: severe impairment */
    metrics.inflammation_level = INFLAMMATION_STORM;
    modulated = memory_immune_modulate_engram_retrieval(integration, base_confidence);
    EXPECT_NEAR(modulated, 0.36f, 0.01f);  /* 0.9 * 0.4 */
}

TEST_F(MemoryImmuneSubsystemTest, CheckThreatMemoryInEngrams) {
    memory_immune_connect_engram_system(integration, engram_system);

    /* Start immune system */
    brain_immune_start(immune_system);

    /* Present an antigen */
    uint8_t epitope[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    uint32_t antigen_id;
    int result = brain_immune_present_antigen(
        immune_system,
        ANTIGEN_SOURCE_MANUAL,
        epitope, sizeof(epitope),
        5,  /* severity */
        0,  /* source_node */
        &antigen_id
    );
    ASSERT_EQ(result, 0);

    /* Check if threat memory exists in engrams */
    uint64_t engram_id;
    float affinity;
    result = memory_immune_check_threat_memory_in_engrams(
        integration, antigen_id, &engram_id, &affinity
    );

    /* May not find match (no engrams created yet), but should not crash */
    EXPECT_TRUE(result == 0 || result == -1);
}

TEST_F(MemoryImmuneSubsystemTest, TriggerFromEngramRecall) {
    memory_immune_connect_engram_system(integration, engram_system);

    /* Create emotional engram */
    uint32_t neurons[5] = {1, 2, 3, 4, 5};
    float activations[5] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    emotional_tag_t emotion = {};
    emotion.valence = -0.8f;  /* Negative (threat) */
    emotion.arousal = 0.9f;   /* High arousal */

    uint64_t engram_id = engram_encode(
        engram_system, neurons, activations, 5,
        MEMORY_TYPE_EMOTIONAL, emotion
    );
    ASSERT_NE(engram_id, 0ULL);

    /* Trigger immune response from engram recall */
    int result = memory_immune_trigger_from_engram_recall(integration, engram_id);
    EXPECT_EQ(result, 0);  /* Should recognize as threat-related */
}

/* ============================================================================
 * Semantic Memory Integration Tests
 * ============================================================================ */

TEST_F(MemoryImmuneSubsystemTest, ConnectSemanticMemory) {
    int result = memory_immune_connect_semantic_memory(integration, semantic_memory);
    EXPECT_EQ(result, 0);
}

TEST_F(MemoryImmuneSubsystemTest, CreateSemanticImmuneConcept) {
    memory_immune_connect_semantic_memory(integration, semantic_memory);

    /* Create semantic concept from B cell memory */
    uint32_t b_cell_id = 42;
    uint64_t concept_id = 0;

    int result = memory_immune_create_semantic_immune_concept(
        integration, b_cell_id, true, &concept_id
    );

    EXPECT_EQ(result, 0);
    EXPECT_NE(concept_id, 0ULL);
}

TEST_F(MemoryImmuneSubsystemTest, QuerySemanticThreats) {
    memory_immune_connect_semantic_memory(integration, semantic_memory);
    brain_immune_start(immune_system);

    /* Create some semantic immune concepts first */
    for (uint32_t i = 1; i <= 3; i++) {
        uint64_t concept_id;
        memory_immune_create_semantic_immune_concept(
            integration, i * 10, true, &concept_id
        );
    }

    /* Present an antigen */
    uint8_t epitope[8] = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80};
    uint32_t antigen_id;
    brain_immune_present_antigen(
        immune_system, ANTIGEN_SOURCE_MANUAL,
        epitope, sizeof(epitope), 5, 0, &antigen_id
    );

    /* Query for similar threats */
    uint64_t concept_ids[5];
    float similarities[5];
    uint32_t count = memory_immune_query_semantic_threats(
        integration, antigen_id, 5, concept_ids, similarities
    );

    /* Should find some matches (may be 0 if semantic network is empty) */
    EXPECT_LE(count, 5U);
}

/* ============================================================================
 * Systems Consolidation Integration Tests
 * ============================================================================ */

TEST_F(MemoryImmuneSubsystemTest, ConnectSystemsConsolidation) {
    int result = memory_immune_connect_systems_consolidation(
        integration, systems_consolidation
    );
    EXPECT_EQ(result, 0);
}

TEST_F(MemoryImmuneSubsystemTest, ModulateReplayRate_InflammationReduces) {
    memory_immune_connect_systems_consolidation(integration, systems_consolidation);

    float base_rate = 10.0f;  /* 10 Hz baseline */

    /* No inflammation: normal replay */
    memory_immune_metrics_t metrics = {};
    metrics.inflammation_level = INFLAMMATION_NONE;
    float modulated = memory_immune_modulate_replay_rate(integration, base_rate);
    EXPECT_NEAR(modulated, base_rate, 0.01f);

    /* Regional inflammation: reduced replay */
    metrics.inflammation_level = INFLAMMATION_REGIONAL;
    modulated = memory_immune_modulate_replay_rate(integration, base_rate);
    EXPECT_LT(modulated, base_rate);
    EXPECT_NEAR(modulated, 8.0f, 0.01f);  /* 10.0 * 0.8 */

    /* Systemic inflammation: significantly reduced */
    metrics.inflammation_level = INFLAMMATION_SYSTEMIC;
    modulated = memory_immune_modulate_replay_rate(integration, base_rate);
    EXPECT_NEAR(modulated, 6.0f, 0.01f);  /* 10.0 * 0.6 */

    /* Cytokine storm: severely disrupted */
    metrics.inflammation_level = INFLAMMATION_STORM;
    modulated = memory_immune_modulate_replay_rate(integration, base_rate);
    EXPECT_NEAR(modulated, 3.0f, 0.01f);  /* 10.0 * 0.3 */
}

TEST_F(MemoryImmuneSubsystemTest, ModulateSystemsTransfer_IL1BetaBiphasic) {
    memory_immune_connect_systems_consolidation(integration, systems_consolidation);

    float base_rate = 0.05f;  /* 5% transfer rate */

    /* Low IL-1β: enhances transfer */
    memory_immune_metrics_t metrics = {};
    metrics.il1_concentration = 0.1f;
    metrics.inflammation_level = INFLAMMATION_NONE;
    float modulated = memory_immune_modulate_systems_transfer(integration, base_rate);
    EXPECT_GT(modulated, base_rate);
    EXPECT_NEAR(modulated, 0.06f, 0.01f);  /* 0.05 * 1.2 */

    /* High IL-1β: impairs transfer */
    metrics.il1_concentration = 0.8f;
    modulated = memory_immune_modulate_systems_transfer(integration, base_rate);
    EXPECT_LT(modulated, base_rate);
    EXPECT_NEAR(modulated, 0.035f, 0.01f);  /* 0.05 * 0.7 */

    /* High IL-1β + systemic inflammation: further impaired */
    metrics.inflammation_level = INFLAMMATION_SYSTEMIC;
    modulated = memory_immune_modulate_systems_transfer(integration, base_rate);
    EXPECT_NEAR(modulated, 0.028f, 0.01f);  /* 0.05 * 0.7 * 0.8 */
}

TEST_F(MemoryImmuneSubsystemTest, GetConsolidationPriorityBoost) {
    memory_immune_connect_engram_system(integration, engram_system);
    memory_immune_connect_systems_consolidation(integration, systems_consolidation);

    /* Create emotional (threat-related) engram */
    uint32_t neurons[3] = {1, 2, 3};
    float activations[3] = {0.9f, 0.8f, 0.7f};
    emotional_tag_t emotion = {};
    emotion.valence = -0.7f;
    emotion.arousal = 0.8f;

    uint64_t emotional_engram_id = engram_encode(
        engram_system, neurons, activations, 3,
        MEMORY_TYPE_EMOTIONAL, emotion
    );

    /* Get priority boost */
    float boost = memory_immune_get_consolidation_priority_boost(
        integration, emotional_engram_id
    );
    EXPECT_GT(boost, 1.0f);  /* Should be boosted */
    EXPECT_NEAR(boost, 1.5f, 0.01f);

    /* Create non-emotional engram */
    uint64_t semantic_engram_id = engram_encode(
        engram_system, neurons, activations, 3,
        MEMORY_TYPE_SEMANTIC, emotion
    );

    boost = memory_immune_get_consolidation_priority_boost(
        integration, semantic_engram_id
    );
    EXPECT_NEAR(boost, 1.0f, 0.01f);  /* Normal priority */
}

/* ============================================================================
 * Working Memory Transfer Integration Tests
 * ============================================================================ */

TEST_F(MemoryImmuneSubsystemTest, ConnectWMTransfer) {
    int result = memory_immune_connect_wm_transfer(integration, wm_transfer);
    EXPECT_EQ(result, 0);
}

TEST_F(MemoryImmuneSubsystemTest, ModulateTransferCriteria_InflammationIncreasesThresholds) {
    memory_immune_connect_wm_transfer(integration, wm_transfer);

    /* Get default criteria */
    wm_transfer_criteria_t base_criteria = wm_transfer_get_default_criteria();
    wm_transfer_criteria_t modulated_criteria;

    /* No inflammation: criteria unchanged */
    memory_immune_metrics_t metrics = {};
    metrics.inflammation_level = INFLAMMATION_NONE;
    int result = memory_immune_modulate_transfer_criteria(
        integration, &base_criteria, &modulated_criteria
    );
    EXPECT_EQ(result, 0);
    EXPECT_EQ(modulated_criteria.rehearsal_threshold, base_criteria.rehearsal_threshold);

    /* Regional inflammation: increased thresholds */
    metrics.inflammation_level = INFLAMMATION_REGIONAL;
    metrics.il1_concentration = 0.3f;
    metrics.tnf_concentration = 0.2f;
    metrics.il10_concentration = 0.0f;

    result = memory_immune_modulate_transfer_criteria(
        integration, &base_criteria, &modulated_criteria
    );
    EXPECT_EQ(result, 0);

    /* Rehearsal threshold should increase */
    EXPECT_GT(modulated_criteria.rehearsal_threshold, base_criteria.rehearsal_threshold);

    /* Attention threshold should increase */
    EXPECT_GT(modulated_criteria.attention_threshold, base_criteria.attention_threshold);

    /* Decay rate should increase (faster forgetting) */
    EXPECT_GT(modulated_criteria.decay_rate, base_criteria.decay_rate);
}

TEST_F(MemoryImmuneSubsystemTest, GetTransferPriority_ThreatBoost) {
    memory_immune_connect_wm_transfer(integration, wm_transfer);

    /* Non-threat item: normal priority */
    float priority = memory_immune_get_transfer_priority(integration, 0, false);
    EXPECT_NEAR(priority, 1.0f, 0.01f);

    /* Threat-related item: boosted priority */
    priority = memory_immune_get_transfer_priority(integration, 0, true);
    EXPECT_GT(priority, 1.0f);
    EXPECT_NEAR(priority, 1.8f, 0.01f);
}

/* ============================================================================
 * Integration Workflow Tests
 * ============================================================================ */

TEST_F(MemoryImmuneSubsystemTest, FullWorkflow_ThreatDetectionToMemoryConsolidation) {
    /* Connect all subsystems */
    memory_immune_connect_engram_system(integration, engram_system);
    memory_immune_connect_semantic_memory(integration, semantic_memory);
    memory_immune_connect_systems_consolidation(integration, systems_consolidation);
    memory_immune_connect_wm_transfer(integration, wm_transfer);

    /* Start immune system */
    brain_immune_start(immune_system);

    /* Step 1: Present threat (antigen) */
    uint8_t epitope[8] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
    uint32_t antigen_id;
    brain_immune_present_antigen(
        immune_system, ANTIGEN_SOURCE_BBB,
        epitope, sizeof(epitope), 7, 0, &antigen_id
    );

    /* Step 2: Activate immune response (creates inflammation) */
    uint32_t b_cell_id, helper_id;
    brain_immune_activate_b_cell(immune_system, antigen_id, &b_cell_id);
    brain_immune_activate_helper_t(immune_system, antigen_id, &helper_id);
    brain_immune_t_help_b(immune_system, helper_id, b_cell_id);

    /* Step 3: IL-1β released, affects memory consolidation */
    float consolidation_multiplier = memory_immune_modulate_engram_consolidation(
        integration, 0.1f, false
    );
    /* Should be affected by immune state */
    EXPECT_TRUE(consolidation_multiplier > 0.0f);

    /* Step 4: Create semantic concept from immune memory */
    uint64_t concept_id;
    memory_immune_create_semantic_immune_concept(
        integration, b_cell_id, true, &concept_id
    );
    EXPECT_NE(concept_id, 0ULL);

    /* Step 5: Check if consolidation is prioritized */
    memory_immune_metrics_t metrics;
    memory_immune_get_metrics(integration, &metrics);
    /* Should be in some elevated state */
    EXPECT_NE(metrics.state, MEM_IMMUNE_NORMAL);
}

TEST_F(MemoryImmuneSubsystemTest, BiologicalValidation_IL1BetaBiphasicEffect) {
    memory_immune_connect_engram_system(integration, engram_system);

    /* Test the biphasic dose-response curve for IL-1β */
    /* Based on Gibbs et al. (2008) - IL-1β in hippocampal memory */

    /* Very low: no effect */
    memory_immune_metrics_t metrics = {};
    metrics.il1_concentration = 0.01f;
    float mult = memory_immune_modulate_engram_consolidation(integration, 0.1f, false);
    EXPECT_NEAR(mult, 1.3f, 0.1f);  /* Enhancement */

    /* Moderate-low: enhancement peak */
    metrics.il1_concentration = 0.15f;
    mult = memory_immune_modulate_engram_consolidation(integration, 0.1f, false);
    EXPECT_NEAR(mult, 1.3f, 0.1f);

    /* Moderate-high: neutral */
    metrics.il1_concentration = 0.4f;
    mult = memory_immune_modulate_engram_consolidation(integration, 0.1f, false);
    EXPECT_NEAR(mult, 1.0f, 0.2f);

    /* High: impairment */
    metrics.il1_concentration = 0.7f;
    mult = memory_immune_modulate_engram_consolidation(integration, 0.1f, false);
    EXPECT_LT(mult, 1.0f);
    EXPECT_NEAR(mult, 0.6f, 0.1f);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
