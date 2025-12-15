/**
 * @file test_immune_plasticity_full.cpp
 * @brief Comprehensive unit tests for brain immune-plasticity integration
 *
 * Tests immune modulation of ALL plasticity mechanisms:
 * - BCM, STDP (existing)
 * - STP (short-term plasticity)
 * - Homeostatic plasticity
 * - Dendritic nonlinearities (NMDA)
 * - Adaptive plasticity
 * - Eligibility traces
 * - Predictive coding
 *
 * @version 1.0.0
 * @date 2025-12-11
 */

#include <gtest/gtest.h>
#include "cognitive/immune/nimcp_brain_immune_plasticity.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "utils/memory/nimcp_memory.h"
#include <cmath>

/* ============================================================================
 * Test Fixture
 * ============================================================================ */

class ImmunePlasticityFullTest : public ::testing::Test {
protected:
    brain_immune_system_t* immune_system;
    immune_plasticity_config_t config;
    immune_plasticity_modulation_t modulation;

    void SetUp() override {
        /* Initialize immune system */
        brain_immune_config_t brain_config;
        brain_immune_default_config(&brain_config);
        immune_system = brain_immune_create(&brain_config);
        ASSERT_NE(immune_system, nullptr);

        /* Start immune system */
        brain_immune_start(immune_system);

        /* Get default plasticity config */
        immune_plasticity_default_config(&config);

        /* Initialize modulation to baseline */
        memset(&modulation, 0, sizeof(modulation));
    }

    void TearDown() override {
        if (immune_system) {
            brain_immune_destroy(immune_system);
        }
    }

    /* Helper: Induce inflammation */
    void induce_inflammation(brain_inflammation_level_t level) {
        /* Present multiple threats to escalate inflammation */
        for (uint32_t i = 0; i < (uint32_t)level * 5; i++) {
            uint8_t epitope[64];
            memset(epitope, 0xAA + i, sizeof(epitope));
            uint32_t antigen_id;
            brain_immune_present_antigen(immune_system, ANTIGEN_SOURCE_BFT,
                                        epitope, sizeof(epitope),
                                        8, 0, &antigen_id);
        }
    }

    /* Helper: Release specific cytokine */
    void release_cytokine(brain_cytokine_type_t type, float concentration) {
        uint32_t cytokine_id;
        brain_immune_release_cytokine(immune_system, type, 0,
                                     concentration, 1000,
                                     &cytokine_id);
    }
};

/* ============================================================================
 * Modulation Computation Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, BaselineModulation) {
    /* WHAT: Test that baseline immune state produces identity modulation
     * WHY:  No inflammation should mean no modulation
     * HOW:  Compute modulation, check all factors are ~1.0
     */

    int result = immune_plasticity_compute_modulation(immune_system, &config, &modulation);
    ASSERT_EQ(result, 0);

    /* BCM should be baseline */
    EXPECT_FLOAT_EQ(modulation.bcm_threshold_scale, 1.0f);
    EXPECT_FLOAT_EQ(modulation.bcm_learning_rate_scale, 1.0f);

    /* STDP should be baseline */
    EXPECT_FLOAT_EQ(modulation.stdp_tau_plus_scale, 1.0f);
    EXPECT_FLOAT_EQ(modulation.stdp_learning_rate_scale, 1.0f);

    /* STP should be baseline */
    EXPECT_FLOAT_EQ(modulation.stp_u_scale, 1.0f);
    EXPECT_FLOAT_EQ(modulation.stp_tau_d_scale, 1.0f);

    /* Homeostatic should be baseline */
    EXPECT_FLOAT_EQ(modulation.homeostatic_scaling_rate, 1.0f);
    EXPECT_FLOAT_EQ(modulation.homeostatic_target_shift, 0.0f);

    /* Dendritic should be baseline */
    EXPECT_FLOAT_EQ(modulation.nmda_conductance_scale, 1.0f);
    EXPECT_FLOAT_EQ(modulation.dendritic_spike_threshold_shift, 0.0f);

    /* Eligibility should be baseline */
    EXPECT_FLOAT_EQ(modulation.eligibility_decay_scale, 1.0f);
    EXPECT_FLOAT_EQ(modulation.eligibility_learning_rate_scale, 1.0f);

    /* Predictive coding should be baseline */
    EXPECT_FLOAT_EQ(modulation.pc_prediction_precision_scale, 1.0f);
    EXPECT_FLOAT_EQ(modulation.pc_learning_rate_scale, 1.0f);

    /* Global should be baseline */
    EXPECT_FLOAT_EQ(modulation.global_plasticity_scale, 1.0f);
}

TEST_F(ImmunePlasticityFullTest, InflammationModulation) {
    /* WHAT: Test that inflammation impairs all plasticity
     * WHY:  Validate global immune effects
     * HOW:  Induce inflammation, check all factors degraded
     */

    /* Induce systemic inflammation */
    induce_inflammation(INFLAMMATION_SYSTEMIC);

    int result = immune_plasticity_compute_modulation(immune_system, &config, &modulation);
    ASSERT_EQ(result, 0);

    /* All learning rates should be reduced */
    EXPECT_LT(modulation.bcm_learning_rate_scale, 1.0f);
    EXPECT_LT(modulation.stdp_learning_rate_scale, 1.0f);
    EXPECT_LT(modulation.stp_u_scale, 1.0f);  /* Reduced release probability */
    EXPECT_LT(modulation.homeostatic_scaling_rate, 1.0f);
    EXPECT_LT(modulation.nmda_conductance_scale, 1.0f);
    EXPECT_LT(modulation.eligibility_learning_rate_scale, 1.0f);
    EXPECT_LT(modulation.pc_learning_rate_scale, 1.0f);

    /* Global plasticity should be impaired */
    EXPECT_LT(modulation.global_plasticity_scale, 0.8f);
}

TEST_F(ImmunePlasticityFullTest, IL10Recovery) {
    /* WHAT: Test that IL-10 partially restores plasticity
     * WHY:  Validate anti-inflammatory effects
     * HOW:  Induce inflammation, release IL-10, check recovery
     */

    /* Induce inflammation */
    induce_inflammation(INFLAMMATION_REGIONAL);

    /* Release IL-10 */
    release_cytokine(BRAIN_CYTOKINE_IL10, 0.8f);

    int result = immune_plasticity_compute_modulation(immune_system, &config, &modulation);
    ASSERT_EQ(result, 0);

    /* IL-10 should partially restore plasticity */
    EXPECT_GT(modulation.global_plasticity_scale, 0.5f);

    /* STP should be partially recovered */
    EXPECT_GT(modulation.stp_u_scale, 0.6f);

    /* NMDA should be partially recovered */
    EXPECT_GT(modulation.nmda_conductance_scale, 0.6f);
}

/* ============================================================================
 * STP Integration Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, STPModulation) {
    /* WHAT: Test STP parameter modulation
     * WHY:  Inflammation should reduce release probability
     * HOW:  Release IL-1β, modulate STP, check U reduced
     */

    /* Release IL-1β (reduces neurotransmitter release) */
    release_cytokine(BRAIN_CYTOKINE_IL1, 0.6f);

    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    /* Create STP parameters */
    stp_params_t stp_params;
    stp_params.U = 0.5f;
    stp_params.tau_D = 200.0f;
    stp_params.tau_F = 50.0f;

    /* Apply modulation */
    int result = immune_plasticity_modulate_stp(&stp_params, &modulation);
    ASSERT_EQ(result, 0);

    /* Release probability should be reduced */
    EXPECT_LT(stp_params.U, 0.5f);

    /* Recovery times should be slowed */
    EXPECT_GT(stp_params.tau_D, 200.0f);
}

TEST_F(ImmunePlasticityFullTest, STPStateModulation) {
    /* Test STP state modulation */
    release_cytokine(BRAIN_CYTOKINE_IL1, 0.5f);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    stp_state_t stp_state;
    stp_state.params.U = 0.5f;
    stp_state.params.tau_D = 200.0f;
    stp_state.params.tau_F = 50.0f;

    int result = immune_plasticity_modulate_stp_state(&stp_state, &modulation);
    ASSERT_EQ(result, 0);

    EXPECT_LT(stp_state.params.U, 0.5f);
}

/* ============================================================================
 * Homeostatic Plasticity Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, SynapticScalingModulation) {
    /* WHAT: Test synaptic scaling modulation
     * WHY:  Inflammation should shift target rate and slow scaling
     * HOW:  Induce inflammation, modulate scaling params
     */

    induce_inflammation(INFLAMMATION_REGIONAL);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    synaptic_scaling_params_t scaling_params;
    scaling_params.target_rate = 5.0f;
    scaling_params.scaling_time_constant = 3600.0f;  /* 1 hour */
    scaling_params.scaling_exponent = 1.0f;
    scaling_params.min_scaling_factor = 0.1f;
    scaling_params.max_scaling_factor = 10.0f;
    scaling_params.rate_averaging_tau = 10.0f;

    int result = immune_plasticity_modulate_synaptic_scaling(&scaling_params, &modulation);
    ASSERT_EQ(result, 0);

    /* Target rate should be shifted upward */
    EXPECT_GT(scaling_params.target_rate, 5.0f);

    /* Scaling should be slowed */
    EXPECT_GT(scaling_params.scaling_time_constant, 3600.0f);
}

TEST_F(ImmunePlasticityFullTest, MetaplasticityModulation) {
    /* Test metaplasticity threshold shifting */
    release_cytokine(BRAIN_CYTOKINE_IL1, 0.7f);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    metaplasticity_params_t meta_params;
    meta_params.min_theta = 0.1f;
    meta_params.max_theta = 1.0f;
    meta_params.theta_tau = 1000.0f;
    meta_params.activity_tau = 100.0f;
    meta_params.theta_power = 2.0f;

    int result = immune_plasticity_modulate_metaplasticity(&meta_params, &modulation);
    ASSERT_EQ(result, 0);

    /* Theta should be elevated */
    EXPECT_GT(meta_params.min_theta, 0.1f);
    EXPECT_GT(meta_params.max_theta, 1.0f);
}

/* ============================================================================
 * Dendritic Nonlinearity Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, NMDAModulation) {
    /* WHAT: Test NMDA receptor modulation
     * WHY:  IL-1β should reduce NMDA conductance
     * HOW:  Release IL-1β, modulate NMDA params
     */

    release_cytokine(BRAIN_CYTOKINE_IL1, 0.6f);
    release_cytokine(CYTOKINE_TNF_ALPHA, 0.5f);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    nmda_params_t nmda_params;
    nmda_params.g_max = 1.0f;
    nmda_params.tau_rise = 2.0f;
    nmda_params.tau_decay = 100.0f;
    nmda_params.mg_concentration = 1.0f;
    nmda_params.mg_sensitivity = 3.57f;
    nmda_params.voltage_slope = 0.062f;
    nmda_params.ca_permeability = 0.1f;

    int result = immune_plasticity_modulate_nmda(&nmda_params, &modulation);
    ASSERT_EQ(result, 0);

    /* NMDA conductance should be reduced */
    EXPECT_LT(nmda_params.g_max, 1.0f);

    /* Calcium permeability should be reduced */
    EXPECT_LT(nmda_params.ca_permeability, 0.1f);
}

TEST_F(ImmunePlasticityFullTest, DendriticCompartmentModulation) {
    /* Test dendritic spike threshold modulation */
    induce_inflammation(INFLAMMATION_REGIONAL);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    compartment_params_t comp_params;
    comp_params.type = COMPARTMENT_DISTAL;
    comp_params.length = 100.0f;
    comp_params.diameter = 1.0f;
    comp_params.membrane_capacitance = 1.0f;
    comp_params.axial_resistance = 100.0f;
    comp_params.leak_conductance = 0.1f;
    comp_params.spike_threshold = -40.0f;
    comp_params.spike_amplitude = 40.0f;
    comp_params.spike_duration = 2.0f;
    comp_params.supralinearity_factor = 1.5f;

    int result = immune_plasticity_modulate_dendritic_compartment(&comp_params, &modulation);
    ASSERT_EQ(result, 0);

    /* Spike threshold should be raised (less negative) */
    EXPECT_GT(comp_params.spike_threshold, -40.0f);

    /* Supralinearity should be reduced */
    EXPECT_LT(comp_params.supralinearity_factor, 1.5f);
}

/* ============================================================================
 * Adaptive Plasticity Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, AdaptivePlasticityModulation) {
    /* WHAT: Test adaptive spiking modulation
     * WHY:  Inflammation should increase sparsity
     * HOW:  Induce inflammation, modulate adaptive params
     */

    induce_inflammation(INFLAMMATION_REGIONAL);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    adaptive_spike_params_t adaptive_params;
    adaptive_params.k_factor = 0.5f;
    adaptive_params.sparsity_target = 0.7f;
    adaptive_params.encoding = SPIKE_ENCODING_INTEGER;
    adaptive_params.enable_soft_reset = true;
    adaptive_params.enable_adaptation = true;
    adaptive_params.adaptation_window = 100;
    adaptive_params.min_threshold = 0.1f;
    adaptive_params.max_threshold = 10.0f;

    int result = immune_plasticity_modulate_adaptive_params(&adaptive_params, &modulation);
    ASSERT_EQ(result, 0);

    /* Sparsity target should be increased */
    EXPECT_GT(adaptive_params.sparsity_target, 0.7f);
    EXPECT_LE(adaptive_params.sparsity_target, 1.0f);  /* Clamped to max */
}

/* ============================================================================
 * Eligibility Trace Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, EligibilityTraceModulation) {
    /* WHAT: Test eligibility trace modulation
     * WHY:  Inflammation should shorten credit assignment window
     * HOW:  Induce inflammation, modulate eligibility config
     */

    induce_inflammation(INFLAMMATION_REGIONAL);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    eligibility_config_t elig_config;
    elig_config.decay_lambda = 0.95f;
    elig_config.learning_rate = 0.001f;
    elig_config.use_neuromodulation = true;
    elig_config.trace_threshold = 0.01f;
    elig_config.burst_triggered_mode = false;
    elig_config.burst_lr_multiplier = 3.0f;
    elig_config.min_burst_concentration = 0.3f;

    int result = immune_plasticity_modulate_eligibility_config(&elig_config, &modulation);
    ASSERT_EQ(result, 0);

    /* Decay should be faster (lambda changes, but clamped) */
    EXPECT_GE(elig_config.decay_lambda, 0.7f);
    EXPECT_LE(elig_config.decay_lambda, 0.99f);

    /* Learning rate should be reduced */
    EXPECT_LT(elig_config.learning_rate, 0.001f);
}

/* ============================================================================
 * Predictive Coding Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, PredictiveCodingLayerModulation) {
    /* WHAT: Test predictive coding layer modulation
     * WHY:  Inflammation should reduce precision and learning
     * HOW:  Induce inflammation, modulate PC layer params
     */

    induce_inflammation(INFLAMMATION_REGIONAL);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    pc_layer_params_t pc_params;
    pc_params.num_units = 100;
    pc_params.pred_type = PC_PREDICT_LINEAR;
    pc_params.error_type = PC_ERROR_PRECISION_WEIGHTED;
    pc_params.learning_rate_mu = 0.01f;
    pc_params.learning_rate_precision = 0.001f;
    pc_params.learning_rate_weights = 0.01f;
    pc_params.prediction_tau = 10.0f;
    pc_params.error_tau = 5.0f;
    pc_params.min_precision = 0.01f;
    pc_params.max_precision = 100.0f;

    int result = immune_plasticity_modulate_predictive_coding_layer(&pc_params, &modulation);
    ASSERT_EQ(result, 0);

    /* All learning rates should be reduced */
    EXPECT_LT(pc_params.learning_rate_mu, 0.01f);
    EXPECT_LT(pc_params.learning_rate_precision, 0.001f);
    EXPECT_LT(pc_params.learning_rate_weights, 0.01f);

    /* Minimum precision should be reduced (increased uncertainty) */
    EXPECT_LT(pc_params.min_precision, 0.01f);
}

TEST_F(ImmunePlasticityFullTest, PredictiveCodingHierarchyModulation) {
    /* Test PC hierarchy modulation */
    induce_inflammation(INFLAMMATION_REGIONAL);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    uint32_t units_per_level[] = {100, 50, 25};
    pc_hierarchy_config_t pc_config;
    pc_config.num_levels = 3;
    pc_config.units_per_level = units_per_level;
    pc_config.pred_type = PC_PREDICT_LINEAR;
    pc_config.error_type = PC_ERROR_PRECISION_WEIGHTED;
    pc_config.learning_rate = 0.01f;
    pc_config.precision_learning_rate = 0.001f;
    pc_config.learn_precisions = true;
    pc_config.use_lateral_connections = false;
    pc_config.dt = 1.0f;

    int result = immune_plasticity_modulate_predictive_coding_hierarchy(&pc_config, &modulation);
    ASSERT_EQ(result, 0);

    /* Global learning rate should be reduced */
    EXPECT_LT(pc_config.learning_rate, 0.01f);
    EXPECT_LT(pc_config.precision_learning_rate, 0.001f);
}

/* ============================================================================
 * Cross-Module Integration Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, GlobalPlasticityImpairment) {
    /* WHAT: Test that severe inflammation globally impairs all plasticity
     * WHY:  Cytokine storm should have system-wide effects
     * HOW:  Induce storm, check global plasticity scale
     */

    /* Induce cytokine storm */
    induce_inflammation(INFLAMMATION_STORM);
    release_cytokine(BRAIN_CYTOKINE_IL1, 0.9f);
    release_cytokine(BRAIN_CYTOKINE_IL6, 0.9f);
    release_cytokine(CYTOKINE_TNF_ALPHA, 0.9f);

    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    /* Global plasticity should be severely impaired */
    EXPECT_LT(modulation.global_plasticity_scale, 0.5f);

    /* Check impairment flag */
    bool impaired = immune_plasticity_is_impaired(&modulation, 0.7f);
    EXPECT_TRUE(impaired);
}

TEST_F(ImmunePlasticityFullTest, SelectiveCytokineEffects) {
    /* WHAT: Test that different cytokines have distinct effects
     * WHY:  Validate biological specificity
     * HOW:  Release different cytokines, compare effects
     */

    /* Test IL-1β (primarily affects thresholds) */
    release_cytokine(BRAIN_CYTOKINE_IL1, 0.6f);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);
    float bcm_thresh_il1 = modulation.bcm_threshold_scale;
    float stp_u_il1 = modulation.stp_u_scale;

    /* Reset and test IL-6 (affects timing windows) */
    brain_immune_destroy(immune_system);
    SetUp();
    release_cytokine(BRAIN_CYTOKINE_IL6, 0.6f);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);
    float stdp_tau_il6 = modulation.stdp_tau_plus_scale;

    /* Reset and test TNF-α (severe effects) */
    brain_immune_destroy(immune_system);
    SetUp();
    release_cytokine(CYTOKINE_TNF_ALPHA, 0.6f);
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);
    float attention_tnf = modulation.attention_gate_scale;

    /* IL-1β should elevate BCM threshold */
    EXPECT_GT(bcm_thresh_il1, 1.0f);

    /* IL-1β should reduce STP release probability */
    EXPECT_LT(stp_u_il1, 1.0f);

    /* IL-6 should narrow STDP windows */
    EXPECT_LT(stdp_tau_il6, 1.0f);

    /* TNF-α should impair attention */
    EXPECT_LT(attention_tnf, 1.0f);
}

/* ============================================================================
 * Edge Case Tests
 * ============================================================================ */

TEST_F(ImmunePlasticityFullTest, NullPointerHandling) {
    /* Test null pointer handling in all functions */
    stp_params_t stp;
    EXPECT_EQ(immune_plasticity_modulate_stp(nullptr, &modulation), -1);
    EXPECT_EQ(immune_plasticity_modulate_stp(&stp, nullptr), -1);

    homeostatic_config_t homeo;
    EXPECT_EQ(immune_plasticity_modulate_homeostatic_config(nullptr, &modulation), -1);
    EXPECT_EQ(immune_plasticity_modulate_homeostatic_config(&homeo, nullptr), -1);

    nmda_params_t nmda;
    EXPECT_EQ(immune_plasticity_modulate_nmda(nullptr, &modulation), -1);
    EXPECT_EQ(immune_plasticity_modulate_nmda(&nmda, nullptr), -1);

    adaptive_spike_params_t adaptive;
    EXPECT_EQ(immune_plasticity_modulate_adaptive_params(nullptr, &modulation), -1);
    EXPECT_EQ(immune_plasticity_modulate_adaptive_params(&adaptive, nullptr), -1);

    eligibility_config_t elig;
    EXPECT_EQ(immune_plasticity_modulate_eligibility_config(nullptr, &modulation), -1);
    EXPECT_EQ(immune_plasticity_modulate_eligibility_config(&elig, nullptr), -1);

    pc_layer_params_t pc;
    EXPECT_EQ(immune_plasticity_modulate_predictive_coding_layer(nullptr, &modulation), -1);
    EXPECT_EQ(immune_plasticity_modulate_predictive_coding_layer(&pc, nullptr), -1);
}

TEST_F(ImmunePlasticityFullTest, ExtremeCytokineConcentrations) {
    /* Test behavior with extreme cytokine levels */
    release_cytokine(BRAIN_CYTOKINE_IL1, 100.0f);  /* Way above normal */
    immune_plasticity_compute_modulation(immune_system, &config, &modulation);

    /* All factors should be clamped to valid ranges */
    EXPECT_GE(modulation.stp_u_scale, 0.1f);
    EXPECT_LE(modulation.stp_u_scale, 1.0f);
    EXPECT_GE(modulation.nmda_conductance_scale, 0.3f);
    EXPECT_LE(modulation.nmda_conductance_scale, 1.0f);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
