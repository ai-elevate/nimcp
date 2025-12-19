/**
 * @file e2e_test_substrate_bridges_pipeline.cpp
 * @brief End-to-End Tests for Substrate Bridge Integration
 *
 * WHAT: Comprehensive E2E tests for all substrate bridges across NIMCP
 * WHY:  Verify complete substrate-cognitive-neural integration and stress scenarios
 * HOW:  Test pipelines involving attention, memory, reasoning, executive, emotion,
 *       introspection, working memory, ToM, neuron, synapse, axon/dendrite,
 *       plasticity, and glial substrate bridges
 *
 * TEST CATEGORIES:
 * 1. Cognitive Pipeline (5 tests)
 *    - Full cognitive cycle with substrate modulation
 *    - Emotion-cognition interaction under stress
 *    - Introspection monitoring of cognitive bridges
 *    - ToM processing with shared substrate
 *    - Memory consolidation during substrate recovery
 *
 * 2. Neural Pipeline (4 tests)
 *    - Complete spike processing with substrate
 *    - Signal propagation with substrate modulation
 *    - Plasticity during learning with substrate effects
 *    - Glial support during sustained activity
 *
 * 3. Stress Scenarios (4 tests)
 *    - Low ATP cascade across all systems
 *    - Hyperthermia effects on complete brain
 *    - Recovery sequence from critical state
 *    - Gradual degradation simulation
 *
 * 4. Full Brain Integration (3 tests)
 *    - Complete sensory-to-motor pathway
 *    - Learning and adaptation cycle
 *    - Multi-modal processing with substrate
 *
 * BIOLOGICAL BASIS:
 * - ATP depletion affects all neural/cognitive processes
 * - Temperature modulates reaction rates (Q10 effect)
 * - Substrate health determines cognitive capacity
 * - Glial cells provide metabolic support
 * - Stress cascades across coupled systems
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>

// E2E Test Framework
#include "e2e_test_framework.h"

// Neural Substrate
#include "core/neural_substrate/nimcp_neural_substrate.h"

// Cognitive Substrate Bridges
#include "cognitive/attention/nimcp_attention_substrate_bridge.h"
#include "cognitive/emotion/nimcp_emotion_substrate_bridge.h"
#include "cognitive/executive/nimcp_executive_substrate_bridge.h"
#include "cognitive/introspection/nimcp_introspection_substrate_bridge.h"
#include "cognitive/memory/nimcp_memory_consolidation_substrate_bridge.h"
#include "cognitive/reasoning/nimcp_reasoning_substrate_bridge.h"
#include "cognitive/working_memory/nimcp_working_memory_substrate_bridge.h"
#include "cognitive/tom/nimcp_tom_substrate_bridge.h"

// Core Neural Substrate Bridges
#include "core/neuron_models/nimcp_neuron_substrate_bridge.h"
#include "core/synapse_compute/nimcp_synapse_substrate_bridge.h"
#include "core/nimcp_axon_dendrite_substrate_bridge.h"
#include "core/cortical_columns/nimcp_cortical_substrate_bridge.h"

// Plasticity Substrate Bridge
#include "plasticity/nimcp_plasticity_substrate_bridge.h"

// Glial Substrate Bridge
#include "glial/nimcp_glial_substrate_bridge.h"

using namespace nimcp::e2e;

//=============================================================================
// Test Fixture
//=============================================================================

class SubstrateBridgesPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create neural substrate with default configuration
        substrate_config_t config;
        substrate_default_config(&config);
        substrate_ = substrate_create(&config);
        ASSERT_NE(substrate_, nullptr);
    }

    void TearDown() override {
        if (substrate_) {
            substrate_destroy(substrate_);
            substrate_ = nullptr;
        }
    }

    // Helper: Set substrate ATP level
    void set_atp_level(float atp) {
        substrate_set_atp_level(substrate_, atp);
    }

    // Helper: Set substrate temperature
    void set_temperature(float temp_celsius) {
        substrate_set_temperature(substrate_, temp_celsius);
    }

    // Helper: Set substrate oxygen saturation
    void set_oxygen_saturation(float o2) {
        substrate_set_oxygen_saturation(substrate_, o2);
    }

    // Helper: Get substrate ATP level
    float get_atp_level() const {
        return substrate_get_atp_level(substrate_);
    }

    // Helper: Get substrate temperature
    float get_temperature() const {
        return substrate_get_temperature(substrate_);
    }

    neural_substrate_t* substrate_ = nullptr;
};

//=============================================================================
// 1. COGNITIVE PIPELINE TESTS (5 tests)
//=============================================================================

/**
 * Test: Full Cognitive Cycle with Substrate
 * WHAT: Test attention → working_memory → reasoning → executive cycle
 * WHY: Verify complete cognitive processing under substrate modulation
 * HOW: Create all cognitive bridges, process information through pipeline
 */
TEST_F(SubstrateBridgesPipelineTest, CognitiveCycleWithSubstrate) {
    PipelineTracker tracker("CognitiveCycleWithSubstrate");

    // Stage 1: Create cognitive bridges
    tracker.begin_stage("CreateCognitiveBridges", 100);

    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    attention_substrate_bridge_t* attn_bridge =
        attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    working_memory_substrate_config_t wm_config;
    working_memory_substrate_default_config(&wm_config);
    working_memory_substrate_bridge_t* wm_bridge =
        working_memory_substrate_bridge_create(&wm_config, substrate_, nullptr);

    reasoning_substrate_config_t reasoning_config;
    reasoning_substrate_default_config(&reasoning_config);
    reasoning_substrate_bridge_t* reasoning_bridge =
        reasoning_substrate_bridge_create(&reasoning_config, substrate_, nullptr);

    executive_substrate_config_t exec_config;
    executive_substrate_default_config(&exec_config);
    executive_substrate_bridge_t* exec_bridge =
        executive_substrate_bridge_create(&exec_config, substrate_, nullptr);

    ASSERT_NE(attn_bridge, nullptr);
    ASSERT_NE(wm_bridge, nullptr);
    ASSERT_NE(reasoning_bridge, nullptr);
    ASSERT_NE(exec_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Normal operation (high ATP)
    tracker.begin_stage("NormalOperation", 100);
    set_atp_level(0.9f);
    set_temperature(37.0f);

    attention_substrate_update(attn_bridge);
    working_memory_substrate_update(wm_bridge);
    reasoning_substrate_update(reasoning_bridge);
    executive_substrate_update(exec_bridge);

    float focus = attention_substrate_get_focus_capacity(attn_bridge);
    float wm_capacity = working_memory_substrate_get_capacity(wm_bridge);
    float reasoning_ability = reasoning_substrate_get_ability(reasoning_bridge);
    float exec_control = executive_substrate_get_control(exec_bridge);

    EXPECT_GT(focus, 0.8f) << "High ATP should maintain focus";
    EXPECT_GT(wm_capacity, 0.8f) << "High ATP should maintain WM capacity";
    EXPECT_GT(reasoning_ability, 0.8f) << "High ATP should maintain reasoning";
    EXPECT_GT(exec_control, 0.8f) << "High ATP should maintain executive control";
    tracker.end_stage();

    // Stage 3: ATP depletion (fatigue simulation)
    tracker.begin_stage("ATPDepletion", 100);
    set_atp_level(0.4f);  // Low ATP

    attention_substrate_update(attn_bridge);
    working_memory_substrate_update(wm_bridge);
    reasoning_substrate_update(reasoning_bridge);
    executive_substrate_update(exec_bridge);

    focus = attention_substrate_get_focus_capacity(attn_bridge);
    wm_capacity = working_memory_substrate_get_capacity(wm_bridge);
    reasoning_ability = reasoning_substrate_get_ability(reasoning_bridge);
    exec_control = executive_substrate_get_control(exec_bridge);

    EXPECT_LT(focus, 0.7f) << "Low ATP should reduce focus";
    EXPECT_LT(wm_capacity, 0.7f) << "Low ATP should reduce WM capacity";
    EXPECT_LT(reasoning_ability, 0.7f) << "Low ATP should impair reasoning";
    EXPECT_LT(exec_control, 0.7f) << "Low ATP should reduce exec control";
    tracker.end_stage();

    // Stage 4: Recovery
    tracker.begin_stage("Recovery", 100);
    set_atp_level(0.85f);

    attention_substrate_update(attn_bridge);
    working_memory_substrate_update(wm_bridge);
    reasoning_substrate_update(reasoning_bridge);
    executive_substrate_update(exec_bridge);

    focus = attention_substrate_get_focus_capacity(attn_bridge);
    wm_capacity = working_memory_substrate_get_capacity(wm_bridge);

    EXPECT_GT(focus, 0.75f) << "Recovery should improve focus";
    EXPECT_GT(wm_capacity, 0.75f) << "Recovery should improve WM";
    tracker.end_stage();

    // Cleanup
    attention_substrate_bridge_destroy(attn_bridge);
    working_memory_substrate_bridge_destroy(wm_bridge);
    reasoning_substrate_bridge_destroy(reasoning_bridge);
    executive_substrate_bridge_destroy(exec_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Emotion-Cognition Interaction Under Stress
 * WHAT: Test emotion substrate bridge interaction with cognitive systems
 * WHY: Verify emotion modulation under substrate stress
 * HOW: Create emotion and attention bridges, test substrate stress effects
 */
TEST_F(SubstrateBridgesPipelineTest, EmotionCognitionUnderStress) {
    PipelineTracker tracker("EmotionCognitionUnderStress");

    // Stage 1: Create bridges
    tracker.begin_stage("CreateBridges", 100);

    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    emotion_substrate_bridge_t* emotion_bridge =
        emotion_substrate_bridge_create(&emotion_config, substrate_, nullptr);

    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    attention_substrate_bridge_t* attn_bridge =
        attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    ASSERT_NE(emotion_bridge, nullptr);
    ASSERT_NE(attn_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Normal emotion processing
    tracker.begin_stage("NormalEmotionProcessing", 100);
    set_atp_level(0.9f);
    set_temperature(37.0f);

    emotion_substrate_update(emotion_bridge);
    attention_substrate_update(attn_bridge);

    float emotion_regulation = emotion_substrate_get_regulation_capacity(emotion_bridge);
    float emotional_stability = emotion_substrate_get_stability(emotion_bridge);

    EXPECT_GT(emotion_regulation, 0.8f) << "Normal substrate supports emotion regulation";
    EXPECT_GT(emotional_stability, 0.7f) << "Normal substrate maintains stability";
    tracker.end_stage();

    // Stage 3: Substrate stress (low ATP + fever)
    tracker.begin_stage("SubstrateStress", 100);
    set_atp_level(0.3f);
    set_temperature(39.5f);  // Fever

    emotion_substrate_update(emotion_bridge);
    attention_substrate_update(attn_bridge);

    emotion_regulation = emotion_substrate_get_regulation_capacity(emotion_bridge);
    emotional_stability = emotion_substrate_get_stability(emotion_bridge);
    float focus = attention_substrate_get_focus_capacity(attn_bridge);

    EXPECT_LT(emotion_regulation, 0.5f) << "Stress impairs emotion regulation";
    EXPECT_LT(emotional_stability, 0.5f) << "Stress reduces stability";
    EXPECT_LT(focus, 0.5f) << "Stress impairs attention focus";

    // Check for impairment flags
    EXPECT_TRUE(emotion_substrate_is_impaired(emotion_bridge));
    EXPECT_TRUE(attention_substrate_is_impaired(attn_bridge));
    tracker.end_stage();

    // Cleanup
    emotion_substrate_bridge_destroy(emotion_bridge);
    attention_substrate_bridge_destroy(attn_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Introspection Monitoring of Cognitive Bridges
 * WHAT: Test introspection bridge monitoring other cognitive systems
 * WHY: Verify metacognitive awareness of substrate effects
 * HOW: Create introspection bridge, monitor attention/memory under stress
 */
TEST_F(SubstrateBridgesPipelineTest, IntrospectionMonitoringCognitive) {
    PipelineTracker tracker("IntrospectionMonitoringCognitive");

    // Stage 1: Create bridges
    tracker.begin_stage("CreateBridges", 100);

    introspection_substrate_config_t intro_config;
    introspection_substrate_default_config(&intro_config);
    introspection_substrate_bridge_t* intro_bridge =
        introspection_substrate_bridge_create(&intro_config, substrate_, nullptr);

    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    attention_substrate_bridge_t* attn_bridge =
        attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    ASSERT_NE(intro_bridge, nullptr);
    ASSERT_NE(attn_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Monitor under normal conditions
    tracker.begin_stage("MonitorNormalConditions", 100);
    set_atp_level(0.85f);
    set_temperature(37.0f);

    introspection_substrate_update(intro_bridge);
    attention_substrate_update(attn_bridge);

    float metacog_accuracy = introspection_substrate_get_metacognitive_accuracy(intro_bridge);
    float awareness = introspection_substrate_get_awareness(intro_bridge);

    EXPECT_GT(metacog_accuracy, 0.7f) << "Normal substrate supports metacognition";
    EXPECT_GT(awareness, 0.7f) << "Normal substrate maintains awareness";
    tracker.end_stage();

    // Stage 3: Monitor under degraded substrate
    tracker.begin_stage("MonitorDegradedSubstrate", 100);
    set_atp_level(0.35f);  // Critical ATP

    introspection_substrate_update(intro_bridge);
    attention_substrate_update(attn_bridge);

    metacog_accuracy = introspection_substrate_get_metacognitive_accuracy(intro_bridge);
    awareness = introspection_substrate_get_awareness(intro_bridge);

    EXPECT_LT(metacog_accuracy, 0.6f) << "Low ATP impairs metacognitive accuracy";
    EXPECT_LT(awareness, 0.6f) << "Low ATP reduces awareness";

    // Introspection should detect impairment
    bool detected_impairment = introspection_substrate_detects_impairment(intro_bridge);
    EXPECT_TRUE(detected_impairment) << "Introspection should detect substrate stress";
    tracker.end_stage();

    // Cleanup
    introspection_substrate_bridge_destroy(intro_bridge);
    attention_substrate_bridge_destroy(attn_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Theory of Mind Processing with Shared Substrate
 * WHAT: Test ToM bridge under substrate constraints
 * WHY: Verify social cognition sensitivity to metabolic state
 * HOW: Create ToM bridge, test mentalizing under various ATP levels
 */
TEST_F(SubstrateBridgesPipelineTest, ToMProcessingSharedSubstrate) {
    PipelineTracker tracker("ToMProcessingSharedSubstrate");

    // Stage 1: Create ToM bridge
    tracker.begin_stage("CreateToMBridge", 100);

    tom_substrate_config_t tom_config;
    tom_substrate_default_config(&tom_config);
    tom_substrate_bridge_t* tom_bridge =
        tom_substrate_bridge_create(&tom_config, substrate_, nullptr);

    ASSERT_NE(tom_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: High-order mentalizing (ATP-intensive)
    tracker.begin_stage("HighOrderMentalizing", 100);
    set_atp_level(0.9f);

    tom_substrate_update(tom_bridge);

    float mentalizing_depth = tom_substrate_get_mentalizing_depth(tom_bridge);
    float perspective_taking = tom_substrate_get_perspective_taking(tom_bridge);

    EXPECT_GT(mentalizing_depth, 0.8f) << "High ATP supports deep mentalizing";
    EXPECT_GT(perspective_taking, 0.8f) << "High ATP enables perspective-taking";
    tracker.end_stage();

    // Stage 3: Fatigue reduces ToM capacity
    tracker.begin_stage("FatigueReducesToM", 100);
    set_atp_level(0.4f);

    tom_substrate_update(tom_bridge);

    mentalizing_depth = tom_substrate_get_mentalizing_depth(tom_bridge);
    perspective_taking = tom_substrate_get_perspective_taking(tom_bridge);

    EXPECT_LT(mentalizing_depth, 0.6f) << "Low ATP reduces mentalizing depth";
    EXPECT_LT(perspective_taking, 0.6f) << "Low ATP impairs perspective-taking";
    tracker.end_stage();

    // Cleanup
    tom_substrate_bridge_destroy(tom_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Memory Consolidation During Substrate Recovery
 * WHAT: Test memory consolidation bridge during ATP recovery
 * WHY: Verify memory formation sensitivity to substrate dynamics
 * HOW: Create memory bridge, test consolidation during recovery cycle
 */
TEST_F(SubstrateBridgesPipelineTest, MemoryConsolidationRecovery) {
    PipelineTracker tracker("MemoryConsolidationRecovery");

    // Stage 1: Create memory bridge
    tracker.begin_stage("CreateMemoryBridge", 100);

    memory_consolidation_substrate_config_t mem_config;
    memory_consolidation_substrate_default_config(&mem_config);
    memory_consolidation_substrate_bridge_t* mem_bridge =
        memory_consolidation_substrate_bridge_create(&mem_config, substrate_, nullptr);

    ASSERT_NE(mem_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Low ATP blocks consolidation
    tracker.begin_stage("LowATPBlocksConsolidation", 100);
    set_atp_level(0.25f);

    memory_consolidation_substrate_update(mem_bridge);

    float consolidation_rate = memory_consolidation_substrate_get_consolidation_rate(mem_bridge);
    bool can_consolidate = memory_consolidation_substrate_can_consolidate(mem_bridge);

    EXPECT_LT(consolidation_rate, 0.3f) << "Critical ATP blocks consolidation";
    EXPECT_FALSE(can_consolidate) << "Should not consolidate at critical ATP";
    tracker.end_stage();

    // Stage 3: Gradual ATP recovery enables consolidation
    tracker.begin_stage("GradualRecovery", 200);

    // Simulate recovery over time
    for (int i = 0; i < 5; i++) {
        float atp = 0.25f + (i * 0.15f);  // 0.25 → 0.85
        set_atp_level(atp);

        memory_consolidation_substrate_update(mem_bridge);
        consolidation_rate = memory_consolidation_substrate_get_consolidation_rate(mem_bridge);

        // Consolidation should improve with ATP
        EXPECT_GE(consolidation_rate, atp * 0.9f) << "Consolidation scales with ATP";
    }
    tracker.end_stage();

    // Stage 4: Full recovery
    tracker.begin_stage("FullRecovery", 100);
    set_atp_level(0.9f);

    memory_consolidation_substrate_update(mem_bridge);
    consolidation_rate = memory_consolidation_substrate_get_consolidation_rate(mem_bridge);
    can_consolidate = memory_consolidation_substrate_can_consolidate(mem_bridge);

    EXPECT_GT(consolidation_rate, 0.8f) << "High ATP enables full consolidation";
    EXPECT_TRUE(can_consolidate) << "Should consolidate at high ATP";
    tracker.end_stage();

    // Cleanup
    memory_consolidation_substrate_bridge_destroy(mem_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

//=============================================================================
// 2. NEURAL PIPELINE TESTS (4 tests)
//=============================================================================

/**
 * Test: Complete Spike Processing with Substrate
 * WHAT: Test neuron → synapse spike processing with substrate effects
 * WHY: Verify neural computation under metabolic constraints
 * HOW: Create neuron and synapse bridges, simulate spike processing
 */
TEST_F(SubstrateBridgesPipelineTest, CompleteSpikeProcessing) {
    PipelineTracker tracker("CompleteSpikeProcessing");

    // Stage 1: Create neural bridges
    tracker.begin_stage("CreateNeuralBridges", 100);

    neuron_substrate_config_t neuron_config;
    neuron_substrate_default_config(&neuron_config);
    neuron_model_state_t neuron_model = {0};  // Simplified neuron model
    neuron_substrate_bridge_t* neuron_bridge =
        neuron_substrate_bridge_create(&neuron_config, neuron_model, substrate_);

    synapse_substrate_config_t synapse_config;
    synapse_substrate_default_config(&synapse_config);
    synapse_substrate_bridge_t* synapse_bridge =
        synapse_substrate_bridge_create(&synapse_config, substrate_);

    ASSERT_NE(neuron_bridge, nullptr);
    ASSERT_NE(synapse_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Normal spike processing
    tracker.begin_stage("NormalSpikeProcessing", 100);
    set_atp_level(0.9f);
    set_temperature(37.0f);

    neuron_substrate_bridge_update(neuron_bridge, 1);  // 1ms timestep
    synapse_substrate_bridge_update(synapse_bridge, 1);

    float excitability = neuron_substrate_get_excitability(neuron_bridge);
    float transmission = synapse_substrate_get_transmission_efficiency(synapse_bridge);

    EXPECT_GT(excitability, 0.9f) << "High ATP maintains excitability";
    EXPECT_GT(transmission, 0.9f) << "High ATP maintains transmission";
    tracker.end_stage();

    // Stage 3: ATP depletion from sustained spiking
    tracker.begin_stage("ATPDepletionFromSpiking", 200);

    // Simulate sustained spiking
    for (int i = 0; i < 100; i++) {
        neuron_substrate_consume_spike(neuron_bridge);  // Each spike consumes ATP
    }

    neuron_substrate_bridge_update(neuron_bridge, 10);
    synapse_substrate_bridge_update(synapse_bridge, 10);

    float atp_after = get_atp_level();
    excitability = neuron_substrate_get_excitability(neuron_bridge);

    EXPECT_LT(atp_after, 0.85f) << "Sustained spiking depletes ATP";
    EXPECT_LT(excitability, 0.9f) << "ATP depletion reduces excitability";
    tracker.end_stage();

    // Cleanup
    neuron_substrate_bridge_destroy(neuron_bridge);
    synapse_substrate_bridge_destroy(synapse_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Signal Propagation with Substrate Modulation
 * WHAT: Test axon/dendrite signal propagation under substrate effects
 * WHY: Verify conduction velocity and integrity under stress
 * HOW: Create axon/dendrite bridge, test propagation under various conditions
 */
TEST_F(SubstrateBridgesPipelineTest, SignalPropagationModulation) {
    PipelineTracker tracker("SignalPropagationModulation");

    // Stage 1: Create axon/dendrite bridge
    tracker.begin_stage("CreateAxonDendriteBridge", 100);

    axon_dendrite_substrate_config_t ad_config;
    axon_dendrite_substrate_default_config(&ad_config);
    axon_dendrite_substrate_bridge_t* ad_bridge =
        axon_dendrite_substrate_bridge_create(&ad_config, substrate_);

    ASSERT_NE(ad_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Normal propagation
    tracker.begin_stage("NormalPropagation", 100);
    set_atp_level(0.9f);
    set_temperature(37.0f);

    axon_dendrite_substrate_update(ad_bridge);

    float conduction_velocity = axon_dendrite_substrate_get_conduction_velocity(ad_bridge);
    float integrity = axon_dendrite_substrate_get_integrity(ad_bridge);

    EXPECT_GT(conduction_velocity, 0.9f) << "Normal substrate supports fast conduction";
    EXPECT_GT(integrity, 0.95f) << "Normal substrate maintains integrity";
    tracker.end_stage();

    // Stage 3: Temperature modulation (Q10 effect)
    tracker.begin_stage("TemperatureModulation", 100);
    set_temperature(40.0f);  // Fever

    axon_dendrite_substrate_update(ad_bridge);

    conduction_velocity = axon_dendrite_substrate_get_conduction_velocity(ad_bridge);

    // Q10 effect: higher temperature increases velocity
    EXPECT_GT(conduction_velocity, 1.0f) << "Hyperthermia increases conduction (Q10)";
    tracker.end_stage();

    // Stage 4: ATP depletion impairs propagation
    tracker.begin_stage("ATPDepletionImpairsPropagation", 100);
    set_atp_level(0.3f);
    set_temperature(37.0f);

    axon_dendrite_substrate_update(ad_bridge);

    conduction_velocity = axon_dendrite_substrate_get_conduction_velocity(ad_bridge);
    integrity = axon_dendrite_substrate_get_integrity(ad_bridge);

    EXPECT_LT(conduction_velocity, 0.6f) << "Low ATP slows conduction";
    EXPECT_LT(integrity, 0.7f) << "Low ATP degrades integrity";
    tracker.end_stage();

    // Cleanup
    axon_dendrite_substrate_bridge_destroy(ad_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Plasticity During Learning with Substrate Effects
 * WHAT: Test plasticity bridge during learning under substrate constraints
 * WHY: Verify learning modulation by metabolic state
 * HOW: Create plasticity bridge, test LTP/LTD under various ATP levels
 */
TEST_F(SubstrateBridgesPipelineTest, PlasticityDuringLearning) {
    PipelineTracker tracker("PlasticityDuringLearning");

    // Stage 1: Create plasticity bridge
    tracker.begin_stage("CreatePlasticityBridge", 100);

    plasticity_substrate_config_t plasticity_config;
    plasticity_substrate_default_config(&plasticity_config);
    plasticity_substrate_bridge_t* plasticity_bridge =
        plasticity_substrate_bridge_create(&plasticity_config, substrate_);

    ASSERT_NE(plasticity_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Normal learning (high ATP)
    tracker.begin_stage("NormalLearning", 100);
    set_atp_level(0.9f);
    set_temperature(37.0f);

    plasticity_substrate_update_all(plasticity_bridge);

    float learning_rate = plasticity_substrate_get_learning_rate_mod(plasticity_bridge);
    float plasticity_capacity = plasticity_substrate_get_capacity(plasticity_bridge);

    EXPECT_GT(learning_rate, 0.85f) << "High ATP supports full learning rate";
    EXPECT_GT(plasticity_capacity, 0.85f) << "High ATP enables full plasticity";
    tracker.end_stage();

    // Stage 3: Learning impairment (low ATP)
    tracker.begin_stage("LearningImpairment", 100);
    set_atp_level(0.35f);  // Below LTP threshold

    plasticity_substrate_update_all(plasticity_bridge);

    learning_rate = plasticity_substrate_get_learning_rate_mod(plasticity_bridge);
    plasticity_capacity = plasticity_substrate_get_capacity(plasticity_bridge);
    bool is_limited = plasticity_substrate_is_limited(plasticity_bridge);

    EXPECT_LT(learning_rate, 0.5f) << "Low ATP reduces learning rate";
    EXPECT_LT(plasticity_capacity, 0.5f) << "Low ATP limits plasticity";
    EXPECT_TRUE(is_limited) << "Should detect substrate limitation";
    tracker.end_stage();

    // Stage 4: Temperature effects on STDP window
    tracker.begin_stage("TemperatureSTDPWindow", 100);
    set_atp_level(0.85f);
    set_temperature(39.0f);  // Fever

    plasticity_substrate_update_stdp(plasticity_bridge);

    float stdp_window = plasticity_substrate_get_stdp_window_mod(plasticity_bridge);

    // Temperature shifts STDP window (Q10 effect)
    EXPECT_NE(stdp_window, 1.0f) << "Temperature modulates STDP window";
    tracker.end_stage();

    // Cleanup
    plasticity_substrate_bridge_destroy(plasticity_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Glial Support During Sustained Activity
 * WHAT: Test glial substrate bridge providing metabolic support
 * WHY: Verify glial lactate shuttle and ATP support
 * HOW: Create glial bridge, simulate sustained activity, verify ATP support
 */
TEST_F(SubstrateBridgesPipelineTest, GlialSupportSustainedActivity) {
    PipelineTracker tracker("GlialSupportSustainedActivity");

    // Stage 1: Create glial bridge
    tracker.begin_stage("CreateGlialBridge", 100);

    glial_substrate_config_t glial_config;
    glial_substrate_default_config(&glial_config);
    glial_substrate_bridge_t* glial_bridge =
        glial_substrate_bridge_create(&glial_config, substrate_, nullptr, nullptr, nullptr, nullptr);

    ASSERT_NE(glial_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Baseline ATP without glial support
    tracker.begin_stage("BaselineWithoutSupport", 100);
    set_atp_level(0.7f);

    float atp_baseline = get_atp_level();
    EXPECT_FLOAT_EQ(atp_baseline, 0.7f);
    tracker.end_stage();

    // Stage 3: Glial support provides ATP boost
    tracker.begin_stage("GlialSupport", 100);

    // Simulate glial lactate shuttle and support
    glial_substrate_compute_all_support(glial_bridge);
    glial_substrate_apply_glial_support(glial_bridge);

    float total_support = glial_substrate_get_total_atp_support(glial_bridge);

    EXPECT_GT(total_support, 0.0f) << "Glial cells should provide ATP support";
    tracker.end_stage();

    // Stage 4: Sustained activity with glial support
    tracker.begin_stage("SustainedActivityWithSupport", 200);

    // Simulate sustained activity over multiple timesteps
    for (int i = 0; i < 10; i++) {
        glial_substrate_bridge_update(glial_bridge, 10);  // 10ms per step
    }

    float atp_with_support = get_atp_level();

    // ATP should be maintained or improved with glial support
    EXPECT_GE(atp_with_support, 0.65f) << "Glial support maintains ATP during activity";
    tracker.end_stage();

    // Cleanup
    glial_substrate_bridge_destroy(glial_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

//=============================================================================
// 3. STRESS SCENARIOS (4 tests)
//=============================================================================

/**
 * Test: Low ATP Cascade Across All Systems
 * WHAT: Test cascade effects of ATP depletion across all bridges
 * WHY: Verify system-wide degradation under energy crisis
 * HOW: Create multiple bridges, deplete ATP, verify cascade
 */
TEST_F(SubstrateBridgesPipelineTest, LowATPCascade) {
    PipelineTracker tracker("LowATPCascade");

    // Stage 1: Create multiple bridges
    tracker.begin_stage("CreateMultipleBridges", 200);

    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    auto* attn_bridge = attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    working_memory_substrate_config_t wm_config;
    working_memory_substrate_default_config(&wm_config);
    auto* wm_bridge = working_memory_substrate_bridge_create(&wm_config, substrate_, nullptr);

    executive_substrate_config_t exec_config;
    executive_substrate_default_config(&exec_config);
    auto* exec_bridge = executive_substrate_bridge_create(&exec_config, substrate_, nullptr);

    neuron_substrate_config_t neuron_config;
    neuron_substrate_default_config(&neuron_config);
    neuron_model_state_t neuron_model = {0};
    auto* neuron_bridge = neuron_substrate_bridge_create(&neuron_config, neuron_model, substrate_);

    ASSERT_NE(attn_bridge, nullptr);
    ASSERT_NE(wm_bridge, nullptr);
    ASSERT_NE(exec_bridge, nullptr);
    ASSERT_NE(neuron_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Progressive ATP depletion
    tracker.begin_stage("ProgressiveATPDepletion", 300);

    float atp_levels[] = {0.8f, 0.6f, 0.4f, 0.2f};
    int impairment_count = 0;

    for (float atp : atp_levels) {
        set_atp_level(atp);

        attention_substrate_update(attn_bridge);
        working_memory_substrate_update(wm_bridge);
        executive_substrate_update(exec_bridge);
        neuron_substrate_bridge_update(neuron_bridge, 1);

        // Count impairments
        if (attention_substrate_is_impaired(attn_bridge)) impairment_count++;
        if (working_memory_substrate_is_impaired(wm_bridge)) impairment_count++;
        if (executive_substrate_is_impaired(exec_bridge)) impairment_count++;
    }

    EXPECT_GT(impairment_count, 5) << "Progressive ATP depletion should cause cascading impairments";
    tracker.end_stage();

    // Stage 3: Critical ATP (all systems fail)
    tracker.begin_stage("CriticalATP", 100);
    set_atp_level(0.15f);  // Critical

    attention_substrate_update(attn_bridge);
    working_memory_substrate_update(wm_bridge);
    executive_substrate_update(exec_bridge);
    neuron_substrate_bridge_update(neuron_bridge, 1);

    EXPECT_TRUE(attention_substrate_is_impaired(attn_bridge));
    EXPECT_TRUE(working_memory_substrate_is_impaired(wm_bridge));
    EXPECT_TRUE(executive_substrate_is_impaired(exec_bridge));

    float excitability = neuron_substrate_get_excitability(neuron_bridge);
    EXPECT_LT(excitability, 0.3f) << "Critical ATP should severely impair neurons";
    tracker.end_stage();

    // Cleanup
    attention_substrate_bridge_destroy(attn_bridge);
    working_memory_substrate_bridge_destroy(wm_bridge);
    executive_substrate_bridge_destroy(exec_bridge);
    neuron_substrate_bridge_destroy(neuron_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Hyperthermia Effects on Complete Brain
 * WHAT: Test temperature effects across all bridges
 * WHY: Verify Q10 temperature scaling and fever effects
 * HOW: Create bridges, increase temperature, verify effects
 */
TEST_F(SubstrateBridgesPipelineTest, HyperthermiaEffects) {
    PipelineTracker tracker("HyperthermiaEffects");

    // Stage 1: Create bridges
    tracker.begin_stage("CreateBridges", 200);

    neuron_substrate_config_t neuron_config;
    neuron_substrate_default_config(&neuron_config);
    neuron_model_state_t neuron_model = {0};
    auto* neuron_bridge = neuron_substrate_bridge_create(&neuron_config, neuron_model, substrate_);

    plasticity_substrate_config_t plasticity_config;
    plasticity_substrate_default_config(&plasticity_config);
    auto* plasticity_bridge = plasticity_substrate_bridge_create(&plasticity_config, substrate_);

    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    auto* attn_bridge = attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    ASSERT_NE(neuron_bridge, nullptr);
    ASSERT_NE(plasticity_bridge, nullptr);
    ASSERT_NE(attn_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Normal temperature (37°C)
    tracker.begin_stage("NormalTemperature", 100);
    set_atp_level(0.9f);
    set_temperature(37.0f);

    neuron_substrate_bridge_update(neuron_bridge, 1);
    plasticity_substrate_update_all(plasticity_bridge);
    attention_substrate_update(attn_bridge);

    neuron_substrate_effects_t neuron_effects;
    neuron_substrate_get_effects(neuron_bridge, &neuron_effects);
    float baseline_firing_mod = neuron_effects.firing_rate_mod;

    EXPECT_NEAR(baseline_firing_mod, 1.0f, 0.1f) << "Normal temp should have ~1.0 firing mod";
    tracker.end_stage();

    // Stage 3: Moderate fever (39°C)
    tracker.begin_stage("ModerateFever", 100);
    set_temperature(39.0f);

    neuron_substrate_bridge_update(neuron_bridge, 1);
    plasticity_substrate_update_all(plasticity_bridge);
    attention_substrate_update(attn_bridge);

    neuron_substrate_get_effects(neuron_bridge, &neuron_effects);
    float fever_firing_mod = neuron_effects.firing_rate_mod;

    // Q10 effect: 2°C increase with Q10=2.5 → ~1.19x
    EXPECT_GT(fever_firing_mod, baseline_firing_mod) << "Fever should increase firing rate (Q10)";
    tracker.end_stage();

    // Stage 4: High fever (41°C) - dangerous
    tracker.begin_stage("HighFever", 100);
    set_temperature(41.0f);

    neuron_substrate_bridge_update(neuron_bridge, 1);
    plasticity_substrate_update_all(plasticity_bridge);
    attention_substrate_update(attn_bridge);

    float shifting = attention_substrate_get_shifting_efficiency(attn_bridge);

    EXPECT_LT(shifting, 0.7f) << "High fever impairs cognitive function";
    tracker.end_stage();

    // Cleanup
    neuron_substrate_bridge_destroy(neuron_bridge);
    plasticity_substrate_bridge_destroy(plasticity_bridge);
    attention_substrate_bridge_destroy(attn_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Recovery Sequence from Critical State
 * WHAT: Test system recovery from critical substrate failure
 * WHY: Verify recovery dynamics and ordering
 * HOW: Create bridges, induce critical state, simulate recovery
 */
TEST_F(SubstrateBridgesPipelineTest, RecoverySequence) {
    PipelineTracker tracker("RecoverySequence");

    // Stage 1: Create bridges
    tracker.begin_stage("CreateBridges", 200);

    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    auto* attn_bridge = attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    memory_consolidation_substrate_config_t mem_config;
    memory_consolidation_substrate_default_config(&mem_config);
    auto* mem_bridge = memory_consolidation_substrate_bridge_create(&mem_config, substrate_, nullptr);

    executive_substrate_config_t exec_config;
    executive_substrate_default_config(&exec_config);
    auto* exec_bridge = executive_substrate_bridge_create(&exec_config, substrate_, nullptr);

    ASSERT_NE(attn_bridge, nullptr);
    ASSERT_NE(mem_bridge, nullptr);
    ASSERT_NE(exec_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Critical state
    tracker.begin_stage("CriticalState", 100);
    set_atp_level(0.15f);
    set_temperature(40.5f);

    attention_substrate_update(attn_bridge);
    memory_consolidation_substrate_update(mem_bridge);
    executive_substrate_update(exec_bridge);

    EXPECT_TRUE(attention_substrate_is_impaired(attn_bridge));
    EXPECT_FALSE(memory_consolidation_substrate_can_consolidate(mem_bridge));
    EXPECT_TRUE(executive_substrate_is_impaired(exec_bridge));
    tracker.end_stage();

    // Stage 3: Gradual recovery
    tracker.begin_stage("GradualRecovery", 300);

    float recovery_steps[] = {0.3f, 0.5f, 0.7f, 0.85f};

    for (float atp : recovery_steps) {
        set_atp_level(atp);
        set_temperature(37.0f + (40.5f - 37.0f) * (1.0f - atp / 0.85f));  // Temperature normalizes

        attention_substrate_update(attn_bridge);
        memory_consolidation_substrate_update(mem_bridge);
        executive_substrate_update(exec_bridge);

        float focus = attention_substrate_get_focus_capacity(attn_bridge);

        // Recovery should be proportional to ATP
        EXPECT_GT(focus, atp * 0.8f) << "Focus should recover with ATP";
    }
    tracker.end_stage();

    // Stage 4: Full recovery
    tracker.begin_stage("FullRecovery", 100);
    set_atp_level(0.9f);
    set_temperature(37.0f);

    attention_substrate_update(attn_bridge);
    memory_consolidation_substrate_update(mem_bridge);
    executive_substrate_update(exec_bridge);

    EXPECT_FALSE(attention_substrate_is_impaired(attn_bridge));
    EXPECT_TRUE(memory_consolidation_substrate_can_consolidate(mem_bridge));
    EXPECT_FALSE(executive_substrate_is_impaired(exec_bridge));
    tracker.end_stage();

    // Cleanup
    attention_substrate_bridge_destroy(attn_bridge);
    memory_consolidation_substrate_bridge_destroy(mem_bridge);
    executive_substrate_bridge_destroy(exec_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Gradual Degradation Simulation
 * WHAT: Test slow substrate degradation effects
 * WHY: Verify aging/fatigue simulation
 * HOW: Create bridges, simulate gradual ATP decline over time
 */
TEST_F(SubstrateBridgesPipelineTest, GradualDegradation) {
    PipelineTracker tracker("GradualDegradation");

    // Stage 1: Create bridges
    tracker.begin_stage("CreateBridges", 200);

    working_memory_substrate_config_t wm_config;
    working_memory_substrate_default_config(&wm_config);
    auto* wm_bridge = working_memory_substrate_bridge_create(&wm_config, substrate_, nullptr);

    reasoning_substrate_config_t reasoning_config;
    reasoning_substrate_default_config(&reasoning_config);
    auto* reasoning_bridge = reasoning_substrate_bridge_create(&reasoning_config, substrate_, nullptr);

    ASSERT_NE(wm_bridge, nullptr);
    ASSERT_NE(reasoning_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Gradual degradation simulation
    tracker.begin_stage("GradualDegradation", 400);

    float initial_atp = 0.9f;
    float degradation_rate = 0.05f;  // 5% per step

    std::vector<float> wm_capacity_history;
    std::vector<float> reasoning_history;

    for (int step = 0; step < 10; step++) {
        float atp = initial_atp - (step * degradation_rate);
        set_atp_level(atp);

        working_memory_substrate_update(wm_bridge);
        reasoning_substrate_update(reasoning_bridge);

        float wm_capacity = working_memory_substrate_get_capacity(wm_bridge);
        float reasoning_ability = reasoning_substrate_get_ability(reasoning_bridge);

        wm_capacity_history.push_back(wm_capacity);
        reasoning_history.push_back(reasoning_ability);
    }

    // Verify monotonic decline
    for (size_t i = 1; i < wm_capacity_history.size(); i++) {
        EXPECT_LE(wm_capacity_history[i], wm_capacity_history[i-1] + 0.05f)
            << "WM capacity should decline or stabilize";
        EXPECT_LE(reasoning_history[i], reasoning_history[i-1] + 0.05f)
            << "Reasoning should decline or stabilize";
    }
    tracker.end_stage();

    // Cleanup
    working_memory_substrate_bridge_destroy(wm_bridge);
    reasoning_substrate_bridge_destroy(reasoning_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

//=============================================================================
// 4. FULL BRAIN INTEGRATION TESTS (3 tests)
//=============================================================================

/**
 * Test: Complete Sensory-to-Motor Pathway
 * WHAT: Test full pipeline from perception through cognition to action
 * WHY: Verify complete brain integration with substrate
 * HOW: Create perception, cognitive, and motor bridges, simulate pathway
 */
TEST_F(SubstrateBridgesPipelineTest, SensoryToMotorPathway) {
    PipelineTracker tracker("SensoryToMotorPathway");

    // Stage 1: Create complete pathway bridges
    tracker.begin_stage("CreatePathwayBridges", 200);

    // Perception (simplified: attention as sensory gate)
    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    auto* attn_bridge = attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    // Cognition
    working_memory_substrate_config_t wm_config;
    working_memory_substrate_default_config(&wm_config);
    auto* wm_bridge = working_memory_substrate_bridge_create(&wm_config, substrate_, nullptr);

    reasoning_substrate_config_t reasoning_config;
    reasoning_substrate_default_config(&reasoning_config);
    auto* reasoning_bridge = reasoning_substrate_bridge_create(&reasoning_config, substrate_, nullptr);

    // Action
    executive_substrate_config_t exec_config;
    executive_substrate_default_config(&exec_config);
    auto* exec_bridge = executive_substrate_bridge_create(&exec_config, substrate_, nullptr);

    ASSERT_NE(attn_bridge, nullptr);
    ASSERT_NE(wm_bridge, nullptr);
    ASSERT_NE(reasoning_bridge, nullptr);
    ASSERT_NE(exec_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Normal pathway processing
    tracker.begin_stage("NormalPathwayProcessing", 200);
    set_atp_level(0.9f);
    set_temperature(37.0f);

    // Simulate pipeline: attention → WM → reasoning → executive
    attention_substrate_update(attn_bridge);
    float sensory_gate = attention_substrate_get_focus_capacity(attn_bridge);

    working_memory_substrate_update(wm_bridge);
    float wm_capacity = working_memory_substrate_get_capacity(wm_bridge);

    reasoning_substrate_update(reasoning_bridge);
    float reasoning_ability = reasoning_substrate_get_ability(reasoning_bridge);

    executive_substrate_update(exec_bridge);
    float motor_control = executive_substrate_get_control(exec_bridge);

    EXPECT_GT(sensory_gate, 0.8f) << "High ATP enables sensory processing";
    EXPECT_GT(wm_capacity, 0.8f) << "High ATP maintains working memory";
    EXPECT_GT(reasoning_ability, 0.8f) << "High ATP supports reasoning";
    EXPECT_GT(motor_control, 0.8f) << "High ATP enables motor control";
    tracker.end_stage();

    // Stage 3: Pathway under substrate stress
    tracker.begin_stage("PathwayUnderStress", 200);
    set_atp_level(0.35f);

    attention_substrate_update(attn_bridge);
    sensory_gate = attention_substrate_get_focus_capacity(attn_bridge);

    working_memory_substrate_update(wm_bridge);
    wm_capacity = working_memory_substrate_get_capacity(wm_bridge);

    reasoning_substrate_update(reasoning_bridge);
    reasoning_ability = reasoning_substrate_get_ability(reasoning_bridge);

    executive_substrate_update(exec_bridge);
    motor_control = executive_substrate_get_control(exec_bridge);

    // All stages should be impaired
    EXPECT_LT(sensory_gate, 0.6f) << "Low ATP impairs sensory gating";
    EXPECT_LT(wm_capacity, 0.6f) << "Low ATP reduces WM capacity";
    EXPECT_LT(reasoning_ability, 0.6f) << "Low ATP impairs reasoning";
    EXPECT_LT(motor_control, 0.6f) << "Low ATP degrades motor control";
    tracker.end_stage();

    // Cleanup
    attention_substrate_bridge_destroy(attn_bridge);
    working_memory_substrate_bridge_destroy(wm_bridge);
    reasoning_substrate_bridge_destroy(reasoning_bridge);
    executive_substrate_bridge_destroy(exec_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Learning and Adaptation Cycle
 * WHAT: Test complete learning cycle with substrate effects
 * WHY: Verify plasticity-cognition-substrate integration
 * HOW: Create plasticity, memory, attention bridges, simulate learning
 */
TEST_F(SubstrateBridgesPipelineTest, LearningAdaptationCycle) {
    PipelineTracker tracker("LearningAdaptationCycle");

    // Stage 1: Create learning bridges
    tracker.begin_stage("CreateLearningBridges", 200);

    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    auto* attn_bridge = attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    plasticity_substrate_config_t plasticity_config;
    plasticity_substrate_default_config(&plasticity_config);
    auto* plasticity_bridge = plasticity_substrate_bridge_create(&plasticity_config, substrate_);

    memory_consolidation_substrate_config_t mem_config;
    memory_consolidation_substrate_default_config(&mem_config);
    auto* mem_bridge = memory_consolidation_substrate_bridge_create(&mem_config, substrate_, nullptr);

    ASSERT_NE(attn_bridge, nullptr);
    ASSERT_NE(plasticity_bridge, nullptr);
    ASSERT_NE(mem_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Learning phase (high attention, high plasticity)
    tracker.begin_stage("LearningPhase", 200);
    set_atp_level(0.9f);

    attention_substrate_update(attn_bridge);
    plasticity_substrate_update_all(plasticity_bridge);
    memory_consolidation_substrate_update(mem_bridge);

    float focus = attention_substrate_get_focus_capacity(attn_bridge);
    float learning_rate = plasticity_substrate_get_learning_rate_mod(plasticity_bridge);
    float consolidation = memory_consolidation_substrate_get_consolidation_rate(mem_bridge);

    EXPECT_GT(focus, 0.8f) << "High focus during learning";
    EXPECT_GT(learning_rate, 0.8f) << "High learning rate with good substrate";
    EXPECT_GT(consolidation, 0.8f) << "High consolidation rate";
    tracker.end_stage();

    // Stage 3: Consolidation phase (lower attention, continued plasticity)
    tracker.begin_stage("ConsolidationPhase", 200);
    set_atp_level(0.7f);  // Moderate ATP

    attention_substrate_update(attn_bridge);
    plasticity_substrate_update_all(plasticity_bridge);
    memory_consolidation_substrate_update(mem_bridge);

    focus = attention_substrate_get_focus_capacity(attn_bridge);
    consolidation = memory_consolidation_substrate_get_consolidation_rate(mem_bridge);

    EXPECT_GT(consolidation, 0.6f) << "Consolidation continues with moderate ATP";
    tracker.end_stage();

    // Stage 4: Fatigue blocks new learning
    tracker.begin_stage("FatigueBlocksLearning", 200);
    set_atp_level(0.3f);

    attention_substrate_update(attn_bridge);
    plasticity_substrate_update_all(plasticity_bridge);
    memory_consolidation_substrate_update(mem_bridge);

    focus = attention_substrate_get_focus_capacity(attn_bridge);
    learning_rate = plasticity_substrate_get_learning_rate_mod(plasticity_bridge);
    bool can_consolidate = memory_consolidation_substrate_can_consolidate(mem_bridge);

    EXPECT_LT(focus, 0.5f) << "Fatigue reduces focus";
    EXPECT_LT(learning_rate, 0.5f) << "Fatigue impairs learning";
    EXPECT_FALSE(can_consolidate) << "Fatigue blocks consolidation";
    tracker.end_stage();

    // Cleanup
    attention_substrate_bridge_destroy(attn_bridge);
    plasticity_substrate_bridge_destroy(plasticity_bridge);
    memory_consolidation_substrate_bridge_destroy(mem_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

/**
 * Test: Multi-Modal Processing with Substrate
 * WHAT: Test parallel processing across multiple cognitive domains
 * WHY: Verify substrate sharing and competition
 * HOW: Create multiple cognitive bridges, simulate concurrent processing
 */
TEST_F(SubstrateBridgesPipelineTest, MultiModalProcessing) {
    PipelineTracker tracker("MultiModalProcessing");

    // Stage 1: Create multi-modal bridges
    tracker.begin_stage("CreateMultiModalBridges", 200);

    attention_substrate_config_t attn_config;
    attention_substrate_default_config(&attn_config);
    auto* attn_bridge = attention_substrate_bridge_create(&attn_config, substrate_, nullptr);

    working_memory_substrate_config_t wm_config;
    working_memory_substrate_default_config(&wm_config);
    auto* wm_bridge = working_memory_substrate_bridge_create(&wm_config, substrate_, nullptr);

    emotion_substrate_config_t emotion_config;
    emotion_substrate_default_config(&emotion_config);
    auto* emotion_bridge = emotion_substrate_bridge_create(&emotion_config, substrate_, nullptr);

    reasoning_substrate_config_t reasoning_config;
    reasoning_substrate_default_config(&reasoning_config);
    auto* reasoning_bridge = reasoning_substrate_bridge_create(&reasoning_config, substrate_, nullptr);

    ASSERT_NE(attn_bridge, nullptr);
    ASSERT_NE(wm_bridge, nullptr);
    ASSERT_NE(emotion_bridge, nullptr);
    ASSERT_NE(reasoning_bridge, nullptr);
    tracker.end_stage();

    // Stage 2: Parallel processing with high ATP
    tracker.begin_stage("ParallelProcessingHighATP", 200);
    set_atp_level(0.9f);

    attention_substrate_update(attn_bridge);
    working_memory_substrate_update(wm_bridge);
    emotion_substrate_update(emotion_bridge);
    reasoning_substrate_update(reasoning_bridge);

    float focus = attention_substrate_get_focus_capacity(attn_bridge);
    float wm_capacity = working_memory_substrate_get_capacity(wm_bridge);
    float emotion_reg = emotion_substrate_get_regulation_capacity(emotion_bridge);
    float reasoning_ability = reasoning_substrate_get_ability(reasoning_bridge);

    EXPECT_GT(focus, 0.8f);
    EXPECT_GT(wm_capacity, 0.8f);
    EXPECT_GT(emotion_reg, 0.8f);
    EXPECT_GT(reasoning_ability, 0.8f);
    tracker.end_stage();

    // Stage 3: Substrate competition (moderate ATP)
    tracker.begin_stage("SubstrateCompetition", 200);
    set_atp_level(0.5f);  // Limited ATP

    attention_substrate_update(attn_bridge);
    working_memory_substrate_update(wm_bridge);
    emotion_substrate_update(emotion_bridge);
    reasoning_substrate_update(reasoning_bridge);

    focus = attention_substrate_get_focus_capacity(attn_bridge);
    wm_capacity = working_memory_substrate_get_capacity(wm_bridge);
    emotion_reg = emotion_substrate_get_regulation_capacity(emotion_bridge);
    reasoning_ability = reasoning_substrate_get_ability(reasoning_bridge);

    // All should be moderately impaired due to substrate sharing
    float total_capacity = focus + wm_capacity + emotion_reg + reasoning_ability;
    EXPECT_LT(total_capacity, 3.0f) << "Limited ATP should constrain total capacity";
    tracker.end_stage();

    // Stage 4: Critical ATP (all systems compete for resources)
    tracker.begin_stage("CriticalCompetition", 200);
    set_atp_level(0.25f);

    attention_substrate_update(attn_bridge);
    working_memory_substrate_update(wm_bridge);
    emotion_substrate_update(emotion_bridge);
    reasoning_substrate_update(reasoning_bridge);

    EXPECT_TRUE(attention_substrate_is_impaired(attn_bridge));
    EXPECT_TRUE(working_memory_substrate_is_impaired(wm_bridge));
    EXPECT_TRUE(emotion_substrate_is_impaired(emotion_bridge));
    EXPECT_TRUE(reasoning_substrate_is_impaired(reasoning_bridge));
    tracker.end_stage();

    // Cleanup
    attention_substrate_bridge_destroy(attn_bridge);
    working_memory_substrate_bridge_destroy(wm_bridge);
    emotion_substrate_bridge_destroy(emotion_bridge);
    reasoning_substrate_bridge_destroy(reasoning_bridge);

    EXPECT_TRUE(tracker.is_successful());
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
