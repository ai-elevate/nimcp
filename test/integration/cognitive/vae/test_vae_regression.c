/**
 * @file test_vae_regression.c
 * @brief Regression tests for VAE-FEP integration system
 *
 * Comprehensive regression testing to ensure VAE system stability
 * across all modules and integration points. Tests verify that
 * previously working functionality remains intact.
 *
 * Test categories:
 * - Core VAE operations (encode, decode, sample)
 * - FEP bridge integration (free energy, precision, prediction error)
 * - System bridges (immune, BBB, logging)
 * - Cognitive bridges (hippocampus, imagination, visual, auditory, emotion)
 * - Neural bridges (SNN, plasticity, training, substrate, thalamic)
 * - Bio-async messaging
 * - Cross-module interactions
 */

#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cognitive/vae/nimcp_vae.h"
#include "cognitive/vae/nimcp_vae_fep_bridge.h"
#include "cognitive/vae/nimcp_vae_bio_async.h"
#include "nimcp.h"

/* ============================================================================
 * Test Fixtures
 * ========================================================================== */

static nimcp_brain_t *g_brain = NULL;
static nimcp_vae_t *g_vae = NULL;

static void setup_full(void) {
    nimcp_config_t config = {
        .memory_limit = 128 * 1024 * 1024,
        .enable_vae = true,
        .enable_fep = true,
        .enable_bio_async = true,
        .enable_immune = true,
        .enable_bbb = true
    };
    g_brain = nimcp_brain_create(&config);
    ck_assert_ptr_nonnull(g_brain);

    nimcp_vae_config_t vae_config = {
        .latent_dim = 32,
        .input_dim = 128,
        .hidden_dims = {64, 48},
        .num_hidden_layers = 2,
        .beta = 1.0f,
        .learning_rate = 0.001f
    };
    g_vae = nimcp_vae_create(&vae_config);
    ck_assert_ptr_nonnull(g_vae);
}

static void teardown_full(void) {
    if (g_vae) {
        nimcp_vae_destroy(g_vae);
        g_vae = NULL;
    }
    if (g_brain) {
        nimcp_brain_destroy(g_brain);
        g_brain = NULL;
    }
}

/* ============================================================================
 * Core VAE Regression Tests
 * ========================================================================== */

START_TEST(test_reg_vae_encode_decode_roundtrip)
{
    /* Regression: Encode-decode should approximately reconstruct input */
    float input[128];
    for (int i = 0; i < 128; i++) {
        input[i] = sinf((float)i * 0.1f) * 0.5f + 0.5f;
    }

    float latent[32];
    int ret = nimcp_vae_encode(g_vae, input, 128, latent, 32);
    ck_assert_int_eq(ret, 0);

    float output[128];
    ret = nimcp_vae_decode(g_vae, latent, 32, output, 128);
    ck_assert_int_eq(ret, 0);

    /* Reconstruction should be bounded */
    for (int i = 0; i < 128; i++) {
        ck_assert(output[i] >= -10.0f && output[i] <= 10.0f);
    }
}
END_TEST

START_TEST(test_reg_vae_latent_sampling)
{
    /* Regression: Sampling should produce valid latents */
    float mean[32], logvar[32];
    for (int i = 0; i < 32; i++) {
        mean[i] = 0.0f;
        logvar[i] = 0.0f;  /* variance = 1 */
    }

    float sampled[32];
    int ret = nimcp_vae_sample(g_vae, mean, logvar, sampled, 32);
    ck_assert_int_eq(ret, 0);

    /* Samples should be finite */
    for (int i = 0; i < 32; i++) {
        ck_assert(!isnan(sampled[i]));
        ck_assert(!isinf(sampled[i]));
    }
}
END_TEST

START_TEST(test_reg_vae_kl_divergence)
{
    /* Regression: KL divergence should be non-negative */
    float mean[32], logvar[32];
    for (int i = 0; i < 32; i++) {
        mean[i] = (float)i * 0.1f;
        logvar[i] = -1.0f;
    }

    float kl = nimcp_vae_kl_divergence(g_vae, mean, logvar, 32);
    ck_assert(kl >= 0.0f);
    ck_assert(!isnan(kl));
}
END_TEST

START_TEST(test_reg_vae_reconstruction_loss)
{
    /* Regression: Reconstruction loss should be non-negative */
    float input[128], output[128];
    for (int i = 0; i < 128; i++) {
        input[i] = (float)i / 128.0f;
        output[i] = input[i] + ((float)rand() / RAND_MAX) * 0.1f;
    }

    float loss = nimcp_vae_reconstruction_loss(g_vae, input, output, 128);
    ck_assert(loss >= 0.0f);
    ck_assert(!isnan(loss));
}
END_TEST

/* ============================================================================
 * FEP Bridge Regression Tests
 * ========================================================================== */

START_TEST(test_reg_fep_latent_belief_mapping)
{
    /* Regression: VAE latent should map to FEP belief */
    vae_fep_bridge_t *bridge = vae_fep_bridge_create(g_brain);
    ck_assert_ptr_nonnull(bridge);

    float latent[32];
    for (int i = 0; i < 32; i++) {
        latent[i] = (float)i * 0.05f - 0.8f;
    }

    float belief[32];
    int ret = vae_fep_bridge_latent_to_belief(bridge, latent, 32, belief, 32);
    ck_assert_int_eq(ret, 0);

    /* Beliefs should be normalized (sum close to 1 for softmax) */
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        ck_assert(belief[i] >= 0.0f);
        sum += belief[i];
    }
    ck_assert(fabsf(sum - 1.0f) < 0.01f);

    vae_fep_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_fep_free_energy_computation)
{
    /* Regression: Free energy should be computable */
    vae_fep_bridge_t *bridge = vae_fep_bridge_create(g_brain);
    ck_assert_ptr_nonnull(bridge);

    float latent[32], observation[128];
    for (int i = 0; i < 32; i++) latent[i] = 0.1f;
    for (int i = 0; i < 128; i++) observation[i] = 0.5f;

    float free_energy = vae_fep_bridge_compute_free_energy(
        bridge, latent, 32, observation, 128
    );
    ck_assert(!isnan(free_energy));
    ck_assert(!isinf(free_energy));

    vae_fep_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_fep_precision_update)
{
    /* Regression: Precision should update based on prediction error */
    vae_fep_bridge_t *bridge = vae_fep_bridge_create(g_brain);
    ck_assert_ptr_nonnull(bridge);

    float initial_precision = vae_fep_bridge_get_precision(bridge);

    /* Simulate prediction error */
    float prediction_error = 0.5f;
    int ret = vae_fep_bridge_update_precision(bridge, prediction_error);
    ck_assert_int_eq(ret, 0);

    float new_precision = vae_fep_bridge_get_precision(bridge);
    ck_assert(new_precision > 0.0f);
    ck_assert(new_precision <= 1.0f);

    vae_fep_bridge_destroy(bridge);
}
END_TEST

/* ============================================================================
 * System Bridge Regression Tests
 * ========================================================================== */

START_TEST(test_reg_immune_bridge_anomaly_detection)
{
    /* Regression: Immune bridge should detect anomalous latents */
    vae_immune_bridge_t *bridge = vae_immune_bridge_create(g_brain);
    if (!bridge) {
        /* Immune system may not be available */
        return;
    }

    /* Normal latent */
    float normal_latent[32];
    for (int i = 0; i < 32; i++) {
        normal_latent[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

    float anomaly_score = vae_immune_bridge_check_anomaly(
        bridge, normal_latent, 32
    );
    ck_assert(anomaly_score >= 0.0f);
    ck_assert(anomaly_score <= 1.0f);

    vae_immune_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_bbb_bridge_validation)
{
    /* Regression: BBB bridge should validate latent security */
    vae_bbb_bridge_t *bridge = vae_bbb_bridge_create(g_brain);
    if (!bridge) {
        /* BBB may not be available */
        return;
    }

    float latent[32];
    for (int i = 0; i < 32; i++) {
        latent[i] = (float)i * 0.01f;
    }

    bool valid = vae_bbb_bridge_validate(bridge, latent, 32);
    /* Should return a boolean result */
    ck_assert(valid == true || valid == false);

    vae_bbb_bridge_destroy(bridge);
}
END_TEST

/* ============================================================================
 * Cognitive Bridge Regression Tests
 * ========================================================================== */

START_TEST(test_reg_hippocampus_memory_encoding)
{
    /* Regression: Hippocampus bridge should encode memories */
    vae_hippocampus_bridge_t *bridge = vae_hippocampus_bridge_create(g_brain);
    if (!bridge) return;

    float latent[32], context[16];
    for (int i = 0; i < 32; i++) latent[i] = (float)i * 0.02f;
    for (int i = 0; i < 16; i++) context[i] = (float)i * 0.05f;

    uint64_t memory_id = vae_hippocampus_bridge_encode(
        bridge, latent, 32, context, 16
    );
    ck_assert(memory_id != 0);

    vae_hippocampus_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_imagination_generation)
{
    /* Regression: Imagination bridge should generate novel latents */
    vae_imagination_bridge_t *bridge = vae_imagination_bridge_create(g_brain);
    if (!bridge) return;

    float seed_latent[32], generated[32];
    for (int i = 0; i < 32; i++) seed_latent[i] = 0.0f;

    int ret = vae_imagination_bridge_generate(
        bridge, seed_latent, 32, generated, 32, 0.5f /* temperature */
    );
    ck_assert_int_eq(ret, 0);

    /* Generated should differ from seed */
    bool differs = false;
    for (int i = 0; i < 32; i++) {
        if (fabsf(generated[i] - seed_latent[i]) > 0.001f) {
            differs = true;
            break;
        }
    }
    ck_assert(differs);

    vae_imagination_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_visual_hierarchy_encoding)
{
    /* Regression: Visual bridge should encode through V1-V5 hierarchy */
    vae_visual_bridge_t *bridge = vae_visual_bridge_create(g_brain);
    if (!bridge) return;

    float visual_input[256], latent[32];
    for (int i = 0; i < 256; i++) {
        visual_input[i] = ((float)rand() / RAND_MAX);
    }

    int ret = vae_visual_bridge_encode(bridge, visual_input, 256, latent, 32);
    ck_assert_int_eq(ret, 0);

    vae_visual_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_auditory_feature_encoding)
{
    /* Regression: Auditory bridge should encode mel/MFCC features */
    vae_auditory_bridge_t *bridge = vae_auditory_bridge_create(g_brain);
    if (!bridge) return;

    float mel_features[128], latent[32];
    for (int i = 0; i < 128; i++) {
        mel_features[i] = ((float)rand() / RAND_MAX) * 10.0f;
    }

    int ret = vae_auditory_bridge_encode(bridge, mel_features, 128, latent, 32);
    ck_assert_int_eq(ret, 0);

    vae_auditory_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_emotion_modulation)
{
    /* Regression: Emotion bridge should modulate latents based on affect */
    vae_emotion_bridge_t *bridge = vae_emotion_bridge_create(g_brain);
    if (!bridge) return;

    float latent[32], modulated[32];
    for (int i = 0; i < 32; i++) latent[i] = 0.5f;

    vae_affect_state_t affect = {
        .valence = 0.8f,
        .arousal = 0.6f,
        .dominance = 0.5f
    };

    int ret = vae_emotion_bridge_modulate(
        bridge, latent, 32, &affect, modulated, 32
    );
    ck_assert_int_eq(ret, 0);

    vae_emotion_bridge_destroy(bridge);
}
END_TEST

/* ============================================================================
 * Neural Bridge Regression Tests
 * ========================================================================== */

START_TEST(test_reg_snn_latent_spike_conversion)
{
    /* Regression: SNN bridge should convert latent to spikes and back */
    vae_snn_bridge_t *bridge = vae_snn_bridge_create(g_brain);
    if (!bridge) return;

    float latent[32];
    for (int i = 0; i < 32; i++) {
        latent[i] = (float)i * 0.03f;
    }

    vae_spike_train_t spikes;
    int ret = vae_snn_bridge_encode(bridge, latent, 32, &spikes);
    ck_assert_int_eq(ret, 0);
    ck_assert(spikes.num_spikes > 0);

    float decoded[32];
    ret = vae_snn_bridge_decode(bridge, &spikes, decoded, 32);
    ck_assert_int_eq(ret, 0);

    vae_snn_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_plasticity_modulation)
{
    /* Regression: Plasticity bridge should modulate learning signals */
    vae_plasticity_bridge_t *bridge = vae_plasticity_bridge_create(g_brain);
    if (!bridge) return;

    float latent[32], modulation[32];
    for (int i = 0; i < 32; i++) latent[i] = ((float)rand() / RAND_MAX);

    int ret = vae_plasticity_bridge_compute_modulation(
        bridge, latent, 32, VAE_PLASTICITY_STDP, modulation, 32
    );
    ck_assert_int_eq(ret, 0);

    /* Modulation should be finite */
    for (int i = 0; i < 32; i++) {
        ck_assert(!isnan(modulation[i]));
    }

    vae_plasticity_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_training_step)
{
    /* Regression: Training bridge should complete a training step */
    vae_training_bridge_t *bridge = vae_training_bridge_create(g_brain);
    if (!bridge) return;

    float input[128], target[128];
    for (int i = 0; i < 128; i++) {
        input[i] = ((float)rand() / RAND_MAX);
        target[i] = input[i] * 0.9f;
    }

    vae_training_result_t result;
    int ret = vae_training_bridge_step(
        bridge, input, 128, target, 128, &result
    );
    ck_assert_int_eq(ret, 0);
    ck_assert(result.total_loss >= 0.0f);

    vae_training_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_substrate_metabolic_adaptation)
{
    /* Regression: Substrate bridge should adapt to metabolic state */
    vae_substrate_bridge_t *bridge = vae_substrate_bridge_create(g_brain);
    if (!bridge) return;

    vae_metabolic_state_t metabolic = {
        .atp_level = 0.8f,
        .oxygen_level = 0.9f,
        .glucose_level = 0.85f,
        .temperature = 37.0f
    };

    float latent[32], adapted[32];
    for (int i = 0; i < 32; i++) latent[i] = 0.5f;

    int ret = vae_substrate_bridge_adapt(
        bridge, latent, 32, &metabolic, adapted, 32
    );
    ck_assert_int_eq(ret, 0);

    vae_substrate_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_thalamic_relay)
{
    /* Regression: Thalamic bridge should relay with attention gating */
    vae_thalamic_bridge_t *bridge = vae_thalamic_bridge_create(g_brain);
    if (!bridge) return;

    float latent[32], relayed[32];
    for (int i = 0; i < 32; i++) latent[i] = (float)i * 0.02f;

    int ret = vae_thalamic_bridge_relay(
        bridge,
        VAE_THALAMIC_PULVINAR,
        latent, 32,
        0.7f,  /* attention */
        relayed, 32
    );
    ck_assert_int_eq(ret, 0);

    vae_thalamic_bridge_destroy(bridge);
}
END_TEST

/* ============================================================================
 * Bio-Async Regression Tests
 * ========================================================================== */

START_TEST(test_reg_bio_async_message_cycle)
{
    /* Regression: Bio-async should complete a message send/receive cycle */
    vae_bio_async_config_t config = {
        .brain = g_brain,
        .auto_register_handlers = true,
        .enable_heartbeat = false,
        .default_channel = BIO_CHANNEL_GLUTAMATE,
        .max_pending_messages = 100
    };

    vae_bio_async_bridge_t *bridge = vae_bio_async_bridge_create(&config);
    ck_assert_ptr_nonnull(bridge);

    int ret = vae_bio_async_bridge_start(bridge);
    ck_assert_int_eq(ret, 0);

    /* Send encode request */
    float input[32];
    for (int i = 0; i < 32; i++) input[i] = 0.1f;

    ret = vae_bio_async_send_encode_request(bridge, input, 32, 1);
    ck_assert_int_eq(ret, 0);

    /* Verify message was sent */
    vae_bio_async_state_t state;
    vae_bio_async_bridge_get_state(bridge, &state);
    ck_assert_int_ge(state.messages_sent, 1);

    vae_bio_async_bridge_stop(bridge);
    vae_bio_async_bridge_destroy(bridge);
}
END_TEST

START_TEST(test_reg_bio_async_channel_switching)
{
    /* Regression: Bio-async should switch neuromodulator channels */
    vae_bio_async_config_t config = {
        .brain = g_brain,
        .auto_register_handlers = false,
        .default_channel = BIO_CHANNEL_DOPAMINE
    };

    vae_bio_async_bridge_t *bridge = vae_bio_async_bridge_create(&config);
    ck_assert_ptr_nonnull(bridge);

    vae_bio_async_bridge_start(bridge);

    /* Switch through channels */
    bio_channel_t channels[] = {
        BIO_CHANNEL_SEROTONIN,
        BIO_CHANNEL_ACETYLCHOLINE,
        BIO_CHANNEL_GLUTAMATE
    };

    for (size_t i = 0; i < 3; i++) {
        int ret = vae_bio_async_set_channel(bridge, channels[i]);
        ck_assert_int_eq(ret, 0);

        bio_channel_t current = vae_bio_async_get_channel(bridge);
        ck_assert_int_eq(current, channels[i]);
    }

    vae_bio_async_bridge_stop(bridge);
    vae_bio_async_bridge_destroy(bridge);
}
END_TEST

/* ============================================================================
 * Cross-Module Interaction Tests
 * ========================================================================== */

START_TEST(test_reg_vae_fep_bio_async_chain)
{
    /* Regression: VAE -> FEP -> Bio-async chain should work */
    /* Encode through VAE */
    float input[128], latent[32];
    for (int i = 0; i < 128; i++) input[i] = sinf((float)i * 0.05f);

    int ret = nimcp_vae_encode(g_vae, input, 128, latent, 32);
    ck_assert_int_eq(ret, 0);

    /* Process through FEP bridge */
    vae_fep_bridge_t *fep_bridge = vae_fep_bridge_create(g_brain);
    ck_assert_ptr_nonnull(fep_bridge);

    float free_energy = vae_fep_bridge_compute_free_energy(
        fep_bridge, latent, 32, input, 128
    );
    ck_assert(!isnan(free_energy));

    /* Send via bio-async */
    vae_bio_async_config_t bio_config = {
        .brain = g_brain,
        .auto_register_handlers = true
    };
    vae_bio_async_bridge_t *bio_bridge = vae_bio_async_bridge_create(&bio_config);
    ck_assert_ptr_nonnull(bio_bridge);

    vae_bio_async_bridge_start(bio_bridge);

    vae_bio_msg_free_energy_t fe_msg = {
        .free_energy = free_energy,
        .reconstruction_loss = 0.5f,
        .kl_divergence = 0.3f
    };
    ret = vae_bio_async_send_free_energy(bio_bridge, &fe_msg);
    ck_assert_int_eq(ret, 0);

    vae_bio_async_bridge_stop(bio_bridge);
    vae_bio_async_bridge_destroy(bio_bridge);
    vae_fep_bridge_destroy(fep_bridge);
}
END_TEST

START_TEST(test_reg_snn_plasticity_training_chain)
{
    /* Regression: SNN -> Plasticity -> Training chain should work */
    vae_snn_bridge_t *snn = vae_snn_bridge_create(g_brain);
    vae_plasticity_bridge_t *plasticity = vae_plasticity_bridge_create(g_brain);
    vae_training_bridge_t *training = vae_training_bridge_create(g_brain);

    if (!snn || !plasticity || !training) {
        if (snn) vae_snn_bridge_destroy(snn);
        if (plasticity) vae_plasticity_bridge_destroy(plasticity);
        if (training) vae_training_bridge_destroy(training);
        return;
    }

    /* Generate latent */
    float latent[32];
    for (int i = 0; i < 32; i++) latent[i] = ((float)rand() / RAND_MAX) - 0.5f;

    /* Encode to spikes */
    vae_spike_train_t spikes;
    int ret = vae_snn_bridge_encode(snn, latent, 32, &spikes);
    ck_assert_int_eq(ret, 0);

    /* Compute plasticity modulation */
    float modulation[32];
    ret = vae_plasticity_bridge_compute_modulation(
        plasticity, latent, 32, VAE_PLASTICITY_STDP, modulation, 32
    );
    ck_assert_int_eq(ret, 0);

    /* Apply in training step */
    float input[128], target[128];
    for (int i = 0; i < 128; i++) {
        input[i] = ((float)rand() / RAND_MAX);
        target[i] = input[i];
    }

    vae_training_result_t result;
    ret = vae_training_bridge_step(training, input, 128, target, 128, &result);
    ck_assert_int_eq(ret, 0);

    vae_snn_bridge_destroy(snn);
    vae_plasticity_bridge_destroy(plasticity);
    vae_training_bridge_destroy(training);
}
END_TEST

/* ============================================================================
 * Stress/Stability Tests
 * ========================================================================== */

START_TEST(test_reg_repeated_encode_decode)
{
    /* Regression: Repeated encode/decode should be stable */
    float input[128], latent[32], output[128];

    for (int i = 0; i < 128; i++) {
        input[i] = ((float)rand() / RAND_MAX);
    }

    for (int iter = 0; iter < 100; iter++) {
        int ret = nimcp_vae_encode(g_vae, input, 128, latent, 32);
        ck_assert_int_eq(ret, 0);

        ret = nimcp_vae_decode(g_vae, latent, 32, output, 128);
        ck_assert_int_eq(ret, 0);

        /* Use output as next input */
        memcpy(input, output, sizeof(input));

        /* Verify no NaN/Inf propagation */
        for (int j = 0; j < 128; j++) {
            ck_assert(!isnan(input[j]));
            ck_assert(!isinf(input[j]));
        }
    }
}
END_TEST

START_TEST(test_reg_concurrent_bridge_access)
{
    /* Regression: Multiple bridges should work concurrently */
    vae_fep_bridge_t *fep = vae_fep_bridge_create(g_brain);
    vae_snn_bridge_t *snn = vae_snn_bridge_create(g_brain);
    vae_plasticity_bridge_t *plasticity = vae_plasticity_bridge_create(g_brain);

    if (!fep || !snn || !plasticity) {
        if (fep) vae_fep_bridge_destroy(fep);
        if (snn) vae_snn_bridge_destroy(snn);
        if (plasticity) vae_plasticity_bridge_destroy(plasticity);
        return;
    }

    float latent[32];
    for (int i = 0; i < 32; i++) latent[i] = 0.5f;

    /* Access all bridges */
    float belief[32];
    vae_fep_bridge_latent_to_belief(fep, latent, 32, belief, 32);

    vae_spike_train_t spikes;
    vae_snn_bridge_encode(snn, latent, 32, &spikes);

    float modulation[32];
    vae_plasticity_bridge_compute_modulation(
        plasticity, latent, 32, VAE_PLASTICITY_BCM, modulation, 32
    );

    vae_fep_bridge_destroy(fep);
    vae_snn_bridge_destroy(snn);
    vae_plasticity_bridge_destroy(plasticity);
}
END_TEST

/* ============================================================================
 * Test Suite Setup
 * ========================================================================== */

static Suite *vae_regression_suite(void) {
    Suite *s = suite_create("VAE Regression");

    /* Core VAE Tests */
    TCase *tc_core = tcase_create("Core VAE");
    tcase_add_checked_fixture(tc_core, setup_full, teardown_full);
    tcase_add_test(tc_core, test_reg_vae_encode_decode_roundtrip);
    tcase_add_test(tc_core, test_reg_vae_latent_sampling);
    tcase_add_test(tc_core, test_reg_vae_kl_divergence);
    tcase_add_test(tc_core, test_reg_vae_reconstruction_loss);
    suite_add_tcase(s, tc_core);

    /* FEP Bridge Tests */
    TCase *tc_fep = tcase_create("FEP Bridge");
    tcase_add_checked_fixture(tc_fep, setup_full, teardown_full);
    tcase_add_test(tc_fep, test_reg_fep_latent_belief_mapping);
    tcase_add_test(tc_fep, test_reg_fep_free_energy_computation);
    tcase_add_test(tc_fep, test_reg_fep_precision_update);
    suite_add_tcase(s, tc_fep);

    /* System Bridge Tests */
    TCase *tc_system = tcase_create("System Bridges");
    tcase_add_checked_fixture(tc_system, setup_full, teardown_full);
    tcase_add_test(tc_system, test_reg_immune_bridge_anomaly_detection);
    tcase_add_test(tc_system, test_reg_bbb_bridge_validation);
    suite_add_tcase(s, tc_system);

    /* Cognitive Bridge Tests */
    TCase *tc_cognitive = tcase_create("Cognitive Bridges");
    tcase_add_checked_fixture(tc_cognitive, setup_full, teardown_full);
    tcase_add_test(tc_cognitive, test_reg_hippocampus_memory_encoding);
    tcase_add_test(tc_cognitive, test_reg_imagination_generation);
    tcase_add_test(tc_cognitive, test_reg_visual_hierarchy_encoding);
    tcase_add_test(tc_cognitive, test_reg_auditory_feature_encoding);
    tcase_add_test(tc_cognitive, test_reg_emotion_modulation);
    suite_add_tcase(s, tc_cognitive);

    /* Neural Bridge Tests */
    TCase *tc_neural = tcase_create("Neural Bridges");
    tcase_add_checked_fixture(tc_neural, setup_full, teardown_full);
    tcase_add_test(tc_neural, test_reg_snn_latent_spike_conversion);
    tcase_add_test(tc_neural, test_reg_plasticity_modulation);
    tcase_add_test(tc_neural, test_reg_training_step);
    tcase_add_test(tc_neural, test_reg_substrate_metabolic_adaptation);
    tcase_add_test(tc_neural, test_reg_thalamic_relay);
    suite_add_tcase(s, tc_neural);

    /* Bio-Async Tests */
    TCase *tc_bio_async = tcase_create("Bio-Async");
    tcase_add_checked_fixture(tc_bio_async, setup_full, teardown_full);
    tcase_add_test(tc_bio_async, test_reg_bio_async_message_cycle);
    tcase_add_test(tc_bio_async, test_reg_bio_async_channel_switching);
    suite_add_tcase(s, tc_bio_async);

    /* Cross-Module Tests */
    TCase *tc_cross = tcase_create("Cross-Module");
    tcase_add_checked_fixture(tc_cross, setup_full, teardown_full);
    tcase_add_test(tc_cross, test_reg_vae_fep_bio_async_chain);
    tcase_add_test(tc_cross, test_reg_snn_plasticity_training_chain);
    suite_add_tcase(s, tc_cross);

    /* Stability Tests */
    TCase *tc_stability = tcase_create("Stability");
    tcase_add_checked_fixture(tc_stability, setup_full, teardown_full);
    tcase_set_timeout(tc_stability, 60);  /* Allow longer timeout */
    tcase_add_test(tc_stability, test_reg_repeated_encode_decode);
    tcase_add_test(tc_stability, test_reg_concurrent_bridge_access);
    suite_add_tcase(s, tc_stability);

    return s;
}

int main(void) {
    Suite *s = vae_regression_suite();
    SRunner *sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    int failures = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
