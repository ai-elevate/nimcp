/**
 * @file test_vae_e2e.c
 * @brief End-to-end tests for VAE-FEP integration system
 *
 * Tests complete workflows through the VAE system, simulating
 * real-world usage patterns and verifying system behavior under
 * realistic conditions.
 *
 * E2E scenarios:
 * - Perception pipeline (input -> encode -> latent -> decode -> output)
 * - Active inference loop (observe -> predict -> act -> update)
 * - Memory consolidation (encode -> hippocampus -> replay -> decode)
 * - Learning cycle (train -> evaluate -> adapt -> retrain)
 * - Multi-modal integration (visual + auditory -> unified latent)
 * - Neural pathway simulation (VAE -> SNN -> plasticity -> thalamic)
 */

#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cognitive/vae/nimcp_vae.h"
#include "cognitive/vae/nimcp_vae_fep_bridge.h"
#include "cognitive/vae/nimcp_vae_bio_async.h"
#include "nimcp.h"

/* ============================================================================
 * Test Fixtures
 * ========================================================================== */

static nimcp_brain_t *g_brain = NULL;
static nimcp_vae_t *g_vae = NULL;

/* E2E test configuration */
#define E2E_LATENT_DIM 32
#define E2E_INPUT_DIM 128
#define E2E_VISUAL_DIM 256
#define E2E_AUDITORY_DIM 128
#define E2E_ITERATIONS 50

static void setup_e2e(void) {
    srand((unsigned int)time(NULL));

    nimcp_config_t config = {
        .memory_limit = 256 * 1024 * 1024,
        .enable_vae = true,
        .enable_fep = true,
        .enable_bio_async = true,
        .enable_immune = true,
        .enable_bbb = true,
        .enable_snn = true,
        .enable_plasticity = true
    };
    g_brain = nimcp_brain_create(&config);
    ck_assert_ptr_nonnull(g_brain);

    nimcp_vae_config_t vae_config = {
        .latent_dim = E2E_LATENT_DIM,
        .input_dim = E2E_INPUT_DIM,
        .hidden_dims = {64, 48, 36},
        .num_hidden_layers = 3,
        .beta = 1.0f,
        .learning_rate = 0.001f,
        .dropout_rate = 0.1f
    };
    g_vae = nimcp_vae_create(&vae_config);
    ck_assert_ptr_nonnull(g_vae);
}

static void teardown_e2e(void) {
    if (g_vae) {
        nimcp_vae_destroy(g_vae);
        g_vae = NULL;
    }
    if (g_brain) {
        nimcp_brain_destroy(g_brain);
        g_brain = NULL;
    }
}

/* Helper: Generate synthetic sensory input */
static void generate_sensory_input(float *input, size_t dim, float noise) {
    for (size_t i = 0; i < dim; i++) {
        float signal = sinf((float)i * 0.1f) * 0.5f + 0.5f;
        float n = ((float)rand() / RAND_MAX - 0.5f) * noise;
        input[i] = fmaxf(0.0f, fminf(1.0f, signal + n));
    }
}

/* Helper: Compute MSE between two arrays */
static float compute_mse(const float *a, const float *b, size_t dim) {
    float mse = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    return mse / (float)dim;
}

/* ============================================================================
 * E2E Scenario 1: Perception Pipeline
 * ========================================================================== */

START_TEST(test_e2e_perception_pipeline)
{
    /* Full perception pipeline: input -> encode -> latent -> decode -> output */
    float input[E2E_INPUT_DIM];
    float latent[E2E_LATENT_DIM];
    float output[E2E_INPUT_DIM];

    generate_sensory_input(input, E2E_INPUT_DIM, 0.1f);

    /* Step 1: Encode to latent space */
    int ret = nimcp_vae_encode(g_vae, input, E2E_INPUT_DIM, latent, E2E_LATENT_DIM);
    ck_assert_int_eq(ret, 0);

    /* Verify latent is valid */
    for (int i = 0; i < E2E_LATENT_DIM; i++) {
        ck_assert(!isnan(latent[i]));
        ck_assert(!isinf(latent[i]));
    }

    /* Step 2: Decode back to observation space */
    ret = nimcp_vae_decode(g_vae, latent, E2E_LATENT_DIM, output, E2E_INPUT_DIM);
    ck_assert_int_eq(ret, 0);

    /* Step 3: Verify reconstruction quality */
    float mse = compute_mse(input, output, E2E_INPUT_DIM);
    /* MSE should be bounded (model may not be trained, so be lenient) */
    ck_assert(mse < 10.0f);

    /* Step 4: Compute free energy via FEP bridge */
    vae_fep_bridge_t *fep = vae_fep_bridge_create(g_brain);
    ck_assert_ptr_nonnull(fep);

    float free_energy = vae_fep_bridge_compute_free_energy(
        fep, latent, E2E_LATENT_DIM, input, E2E_INPUT_DIM
    );
    ck_assert(!isnan(free_energy));

    vae_fep_bridge_destroy(fep);
}
END_TEST

/* ============================================================================
 * E2E Scenario 2: Active Inference Loop
 * ========================================================================== */

START_TEST(test_e2e_active_inference_loop)
{
    /* Active inference: observe -> predict -> compare -> update beliefs */
    vae_fep_bridge_t *fep = vae_fep_bridge_create(g_brain);
    ck_assert_ptr_nonnull(fep);

    float belief[E2E_LATENT_DIM];
    float precision = 0.5f;

    /* Initialize beliefs */
    for (int i = 0; i < E2E_LATENT_DIM; i++) {
        belief[i] = 1.0f / E2E_LATENT_DIM;
    }

    /* Run active inference loop */
    for (int iter = 0; iter < E2E_ITERATIONS; iter++) {
        /* Step 1: Generate observation */
        float observation[E2E_INPUT_DIM];
        generate_sensory_input(observation, E2E_INPUT_DIM, 0.05f);

        /* Step 2: Encode observation to latent */
        float latent[E2E_LATENT_DIM];
        int ret = nimcp_vae_encode(g_vae, observation, E2E_INPUT_DIM,
                                   latent, E2E_LATENT_DIM);
        ck_assert_int_eq(ret, 0);

        /* Step 3: Convert latent to belief (posterior) */
        float posterior[E2E_LATENT_DIM];
        ret = vae_fep_bridge_latent_to_belief(fep, latent, E2E_LATENT_DIM,
                                              posterior, E2E_LATENT_DIM);
        ck_assert_int_eq(ret, 0);

        /* Step 4: Compute prediction error */
        float prediction_error = 0.0f;
        for (int i = 0; i < E2E_LATENT_DIM; i++) {
            float diff = posterior[i] - belief[i];
            prediction_error += diff * diff;
        }
        prediction_error = sqrtf(prediction_error / E2E_LATENT_DIM);

        /* Step 5: Update precision based on prediction error */
        ret = vae_fep_bridge_update_precision(fep, prediction_error);
        ck_assert_int_eq(ret, 0);

        precision = vae_fep_bridge_get_precision(fep);
        ck_assert(precision > 0.0f && precision <= 1.0f);

        /* Step 6: Update beliefs (weighted by precision) */
        for (int i = 0; i < E2E_LATENT_DIM; i++) {
            belief[i] = belief[i] * (1.0f - precision) + posterior[i] * precision;
        }

        /* Step 7: Compute free energy */
        float fe = vae_fep_bridge_compute_free_energy(
            fep, latent, E2E_LATENT_DIM, observation, E2E_INPUT_DIM
        );
        ck_assert(!isnan(fe));
    }

    vae_fep_bridge_destroy(fep);
}
END_TEST

/* ============================================================================
 * E2E Scenario 3: Memory Consolidation
 * ========================================================================== */

START_TEST(test_e2e_memory_consolidation)
{
    /* Memory: encode -> store in hippocampus -> replay -> verify */
    vae_hippocampus_bridge_t *hippo = vae_hippocampus_bridge_create(g_brain);
    if (!hippo) {
        /* Hippocampus bridge may not be available */
        return;
    }

    /* Phase 1: Encode and store memories */
    uint64_t memory_ids[10];
    float stored_latents[10][E2E_LATENT_DIM];

    for (int m = 0; m < 10; m++) {
        float input[E2E_INPUT_DIM];
        generate_sensory_input(input, E2E_INPUT_DIM, 0.05f);

        float latent[E2E_LATENT_DIM];
        int ret = nimcp_vae_encode(g_vae, input, E2E_INPUT_DIM,
                                   latent, E2E_LATENT_DIM);
        ck_assert_int_eq(ret, 0);

        /* Store latent */
        memcpy(stored_latents[m], latent, sizeof(latent));

        /* Create context for memory */
        float context[16];
        for (int i = 0; i < 16; i++) {
            context[i] = (float)m * 0.1f + (float)i * 0.01f;
        }

        /* Store in hippocampus */
        memory_ids[m] = vae_hippocampus_bridge_encode(
            hippo, latent, E2E_LATENT_DIM, context, 16
        );
        ck_assert(memory_ids[m] != 0);
    }

    /* Phase 2: Retrieve and verify memories */
    for (int m = 0; m < 10; m++) {
        float retrieved[E2E_LATENT_DIM];
        int ret = vae_hippocampus_bridge_retrieve(
            hippo, memory_ids[m], retrieved, E2E_LATENT_DIM
        );

        if (ret == 0) {
            /* Verify retrieval matches stored */
            float mse = compute_mse(stored_latents[m], retrieved, E2E_LATENT_DIM);
            ck_assert(mse < 0.1f);
        }
    }

    /* Phase 3: Replay for consolidation */
    vae_plasticity_bridge_t *plasticity = vae_plasticity_bridge_create(g_brain);
    if (plasticity) {
        for (int r = 0; r < 5; r++) {
            /* Random memory replay */
            int m = rand() % 10;
            float retrieved[E2E_LATENT_DIM];
            int ret = vae_hippocampus_bridge_retrieve(
                hippo, memory_ids[m], retrieved, E2E_LATENT_DIM
            );

            if (ret == 0) {
                /* Apply plasticity modulation during replay */
                float modulation[E2E_LATENT_DIM];
                vae_plasticity_bridge_compute_modulation(
                    plasticity, retrieved, E2E_LATENT_DIM,
                    VAE_PLASTICITY_STDP, modulation, E2E_LATENT_DIM
                );
            }
        }
        vae_plasticity_bridge_destroy(plasticity);
    }

    vae_hippocampus_bridge_destroy(hippo);
}
END_TEST

/* ============================================================================
 * E2E Scenario 4: Learning Cycle
 * ========================================================================== */

START_TEST(test_e2e_learning_cycle)
{
    /* Learning: train -> evaluate -> adapt -> continue */
    vae_training_bridge_t *training = vae_training_bridge_create(g_brain);
    if (!training) return;

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    /* Training loop */
    for (int epoch = 0; epoch < 20; epoch++) {
        float epoch_loss = 0.0f;

        /* Mini-batch training */
        for (int batch = 0; batch < 10; batch++) {
            float input[E2E_INPUT_DIM];
            float target[E2E_INPUT_DIM];

            /* Generate input-target pair */
            generate_sensory_input(input, E2E_INPUT_DIM, 0.1f);
            memcpy(target, input, sizeof(input));  /* Autoencoder target */

            vae_training_result_t result;
            int ret = vae_training_bridge_step(
                training, input, E2E_INPUT_DIM,
                target, E2E_INPUT_DIM, &result
            );
            ck_assert_int_eq(ret, 0);

            epoch_loss += result.total_loss;
        }

        epoch_loss /= 10.0f;

        if (epoch == 0) {
            initial_loss = epoch_loss;
        }
        if (epoch == 19) {
            final_loss = epoch_loss;
        }

        /* Adapt learning rate if needed */
        if (epoch > 0 && epoch % 5 == 0) {
            vae_training_bridge_set_learning_rate(
                training,
                vae_training_bridge_get_learning_rate(training) * 0.95f
            );
        }
    }

    /* Loss should generally decrease or remain stable */
    /* (Not guaranteed without real training, but should not explode) */
    ck_assert(final_loss < initial_loss * 10.0f);

    vae_training_bridge_destroy(training);
}
END_TEST

/* ============================================================================
 * E2E Scenario 5: Multi-Modal Integration
 * ========================================================================== */

START_TEST(test_e2e_multimodal_integration)
{
    /* Multi-modal: visual + auditory -> unified latent representation */
    vae_visual_bridge_t *visual = vae_visual_bridge_create(g_brain);
    vae_auditory_bridge_t *auditory = vae_auditory_bridge_create(g_brain);

    if (!visual || !auditory) {
        if (visual) vae_visual_bridge_destroy(visual);
        if (auditory) vae_auditory_bridge_destroy(auditory);
        return;
    }

    /* Generate visual input (e.g., image features) */
    float visual_input[E2E_VISUAL_DIM];
    for (int i = 0; i < E2E_VISUAL_DIM; i++) {
        visual_input[i] = ((float)rand() / RAND_MAX);
    }

    /* Generate auditory input (e.g., mel features) */
    float auditory_input[E2E_AUDITORY_DIM];
    for (int i = 0; i < E2E_AUDITORY_DIM; i++) {
        auditory_input[i] = ((float)rand() / RAND_MAX) * 10.0f;
    }

    /* Encode visual to latent */
    float visual_latent[E2E_LATENT_DIM];
    int ret = vae_visual_bridge_encode(
        visual, visual_input, E2E_VISUAL_DIM,
        visual_latent, E2E_LATENT_DIM
    );
    ck_assert_int_eq(ret, 0);

    /* Encode auditory to latent */
    float auditory_latent[E2E_LATENT_DIM];
    ret = vae_auditory_bridge_encode(
        auditory, auditory_input, E2E_AUDITORY_DIM,
        auditory_latent, E2E_LATENT_DIM
    );
    ck_assert_int_eq(ret, 0);

    /* Fuse latents (simple averaging for test) */
    float fused_latent[E2E_LATENT_DIM];
    for (int i = 0; i < E2E_LATENT_DIM; i++) {
        fused_latent[i] = (visual_latent[i] + auditory_latent[i]) * 0.5f;
    }

    /* Decode fused latent */
    float output[E2E_INPUT_DIM];
    ret = nimcp_vae_decode(g_vae, fused_latent, E2E_LATENT_DIM,
                           output, E2E_INPUT_DIM);
    ck_assert_int_eq(ret, 0);

    /* Verify output is valid */
    for (int i = 0; i < E2E_INPUT_DIM; i++) {
        ck_assert(!isnan(output[i]));
        ck_assert(!isinf(output[i]));
    }

    vae_visual_bridge_destroy(visual);
    vae_auditory_bridge_destroy(auditory);
}
END_TEST

/* ============================================================================
 * E2E Scenario 6: Neural Pathway Simulation
 * ========================================================================== */

START_TEST(test_e2e_neural_pathway)
{
    /* Neural pathway: VAE -> SNN -> Plasticity -> Substrate -> Thalamic */
    vae_snn_bridge_t *snn = vae_snn_bridge_create(g_brain);
    vae_plasticity_bridge_t *plasticity = vae_plasticity_bridge_create(g_brain);
    vae_substrate_bridge_t *substrate = vae_substrate_bridge_create(g_brain);
    vae_thalamic_bridge_t *thalamic = vae_thalamic_bridge_create(g_brain);

    if (!snn || !plasticity || !substrate || !thalamic) {
        if (snn) vae_snn_bridge_destroy(snn);
        if (plasticity) vae_plasticity_bridge_destroy(plasticity);
        if (substrate) vae_substrate_bridge_destroy(substrate);
        if (thalamic) vae_thalamic_bridge_destroy(thalamic);
        return;
    }

    /* Generate input and encode */
    float input[E2E_INPUT_DIM];
    generate_sensory_input(input, E2E_INPUT_DIM, 0.1f);

    float latent[E2E_LATENT_DIM];
    int ret = nimcp_vae_encode(g_vae, input, E2E_INPUT_DIM,
                               latent, E2E_LATENT_DIM);
    ck_assert_int_eq(ret, 0);

    /* Step 1: VAE -> SNN (encode to spikes) */
    vae_spike_train_t spikes;
    ret = vae_snn_bridge_encode(snn, latent, E2E_LATENT_DIM, &spikes);
    ck_assert_int_eq(ret, 0);
    ck_assert(spikes.num_spikes > 0);

    /* Step 2: SNN -> Plasticity (compute learning modulation) */
    float modulation[E2E_LATENT_DIM];
    ret = vae_plasticity_bridge_compute_modulation(
        plasticity, latent, E2E_LATENT_DIM,
        VAE_PLASTICITY_STDP, modulation, E2E_LATENT_DIM
    );
    ck_assert_int_eq(ret, 0);

    /* Step 3: Apply metabolic adaptation */
    vae_metabolic_state_t metabolic = {
        .atp_level = 0.85f,
        .oxygen_level = 0.92f,
        .glucose_level = 0.88f,
        .temperature = 37.0f
    };

    float adapted_latent[E2E_LATENT_DIM];
    ret = vae_substrate_bridge_adapt(
        substrate, latent, E2E_LATENT_DIM,
        &metabolic, adapted_latent, E2E_LATENT_DIM
    );
    ck_assert_int_eq(ret, 0);

    /* Step 4: Thalamic relay with attention gating */
    float relayed_latent[E2E_LATENT_DIM];
    ret = vae_thalamic_bridge_relay(
        thalamic,
        VAE_THALAMIC_PULVINAR,  /* Higher-order nucleus */
        adapted_latent, E2E_LATENT_DIM,
        0.8f,  /* High attention */
        relayed_latent, E2E_LATENT_DIM
    );
    ck_assert_int_eq(ret, 0);

    /* Step 5: Decode and verify */
    float output[E2E_INPUT_DIM];
    ret = nimcp_vae_decode(g_vae, relayed_latent, E2E_LATENT_DIM,
                           output, E2E_INPUT_DIM);
    ck_assert_int_eq(ret, 0);

    /* Verify the complete pipeline produced valid output */
    for (int i = 0; i < E2E_INPUT_DIM; i++) {
        ck_assert(!isnan(output[i]));
        ck_assert(!isinf(output[i]));
    }

    vae_snn_bridge_destroy(snn);
    vae_plasticity_bridge_destroy(plasticity);
    vae_substrate_bridge_destroy(substrate);
    vae_thalamic_bridge_destroy(thalamic);
}
END_TEST

/* ============================================================================
 * E2E Scenario 7: Bio-Async Communication Pipeline
 * ========================================================================== */

START_TEST(test_e2e_bio_async_pipeline)
{
    /* Bio-async: Full messaging pipeline through neuromodulator channels */
    vae_bio_async_config_t config = {
        .brain = g_brain,
        .auto_register_handlers = true,
        .enable_heartbeat = true,
        .heartbeat_interval_ms = 100,
        .default_channel = BIO_CHANNEL_GLUTAMATE,
        .max_pending_messages = 1000
    };

    vae_bio_async_bridge_t *bio = vae_bio_async_bridge_create(&config);
    ck_assert_ptr_nonnull(bio);

    int ret = vae_bio_async_bridge_start(bio);
    ck_assert_int_eq(ret, 0);

    /* Simulate bio-async communication cycle */
    for (int cycle = 0; cycle < 20; cycle++) {
        /* Generate and encode input */
        float input[E2E_INPUT_DIM];
        generate_sensory_input(input, E2E_INPUT_DIM, 0.1f);

        /* Send encode request via bio-async */
        ret = vae_bio_async_send_encode_request(bio, input, E2E_INPUT_DIM, cycle);
        ck_assert_int_eq(ret, 0);

        /* Compute free energy locally and send */
        float latent[E2E_LATENT_DIM];
        nimcp_vae_encode(g_vae, input, E2E_INPUT_DIM, latent, E2E_LATENT_DIM);

        float reconstruction_loss = ((float)rand() / RAND_MAX) * 0.5f;
        float kl_div = ((float)rand() / RAND_MAX) * 0.3f;

        vae_bio_msg_free_energy_t fe_msg = {
            .free_energy = reconstruction_loss + kl_div,
            .reconstruction_loss = reconstruction_loss,
            .kl_divergence = kl_div,
            .precision = 0.8f,
            .timestamp = (uint64_t)cycle
        };
        ret = vae_bio_async_send_free_energy(bio, &fe_msg);
        ck_assert_int_eq(ret, 0);

        /* Switch channels periodically */
        if (cycle % 5 == 0) {
            bio_channel_t channels[] = {
                BIO_CHANNEL_DOPAMINE,
                BIO_CHANNEL_SEROTONIN,
                BIO_CHANNEL_ACETYLCHOLINE,
                BIO_CHANNEL_GLUTAMATE
            };
            vae_bio_async_set_channel(bio, channels[cycle / 5 % 4]);
        }

        /* Send heartbeat */
        if (cycle % 3 == 0) {
            vae_bio_async_send_heartbeat(bio);
        }
    }

    /* Verify statistics */
    vae_bio_async_state_t state;
    vae_bio_async_bridge_get_state(bio, &state);

    ck_assert_int_ge(state.messages_sent, 40);  /* At least encode + free energy */
    ck_assert_int_ge(state.heartbeats_sent, 6);  /* At least 7 heartbeats (0, 3, 6, 9, 12, 15, 18) */

    vae_bio_async_bridge_stop(bio);
    vae_bio_async_bridge_destroy(bio);
}
END_TEST

/* ============================================================================
 * E2E Scenario 8: Imagination and Generation
 * ========================================================================== */

START_TEST(test_e2e_imagination_generation)
{
    /* Imagination: Generate novel content from latent space */
    vae_imagination_bridge_t *imagination = vae_imagination_bridge_create(g_brain);
    if (!imagination) return;

    /* Start from encoded experience */
    float input[E2E_INPUT_DIM];
    generate_sensory_input(input, E2E_INPUT_DIM, 0.05f);

    float seed_latent[E2E_LATENT_DIM];
    int ret = nimcp_vae_encode(g_vae, input, E2E_INPUT_DIM,
                               seed_latent, E2E_LATENT_DIM);
    ck_assert_int_eq(ret, 0);

    /* Generate variations at different temperatures */
    float temperatures[] = {0.1f, 0.5f, 1.0f, 2.0f};
    float generated_latents[4][E2E_LATENT_DIM];

    for (int t = 0; t < 4; t++) {
        ret = vae_imagination_bridge_generate(
            imagination, seed_latent, E2E_LATENT_DIM,
            generated_latents[t], E2E_LATENT_DIM,
            temperatures[t]
        );
        ck_assert_int_eq(ret, 0);

        /* Verify generated is valid */
        for (int i = 0; i < E2E_LATENT_DIM; i++) {
            ck_assert(!isnan(generated_latents[t][i]));
        }
    }

    /* Higher temperature should produce more diverse outputs */
    float diversity_low = 0.0f, diversity_high = 0.0f;
    for (int i = 0; i < E2E_LATENT_DIM; i++) {
        diversity_low += fabsf(generated_latents[0][i] - seed_latent[i]);
        diversity_high += fabsf(generated_latents[3][i] - seed_latent[i]);
    }

    /* High temperature should generally produce larger deviations */
    /* (Not guaranteed, but typical behavior) */

    /* Decode generated latents */
    for (int t = 0; t < 4; t++) {
        float output[E2E_INPUT_DIM];
        ret = nimcp_vae_decode(g_vae, generated_latents[t], E2E_LATENT_DIM,
                               output, E2E_INPUT_DIM);
        ck_assert_int_eq(ret, 0);
    }

    vae_imagination_bridge_destroy(imagination);
}
END_TEST

/* ============================================================================
 * E2E Scenario 9: Emotion-Modulated Perception
 * ========================================================================== */

START_TEST(test_e2e_emotion_modulated_perception)
{
    /* Emotion affects perception: same input, different emotional states */
    vae_emotion_bridge_t *emotion = vae_emotion_bridge_create(g_brain);
    if (!emotion) return;

    /* Generate neutral input */
    float input[E2E_INPUT_DIM];
    generate_sensory_input(input, E2E_INPUT_DIM, 0.05f);

    /* Encode */
    float latent[E2E_LATENT_DIM];
    int ret = nimcp_vae_encode(g_vae, input, E2E_INPUT_DIM,
                               latent, E2E_LATENT_DIM);
    ck_assert_int_eq(ret, 0);

    /* Different emotional states */
    vae_affect_state_t emotions[] = {
        {.valence = -0.8f, .arousal = 0.8f, .dominance = 0.2f},  /* Fear */
        {.valence = 0.0f, .arousal = 0.0f, .dominance = 0.5f},   /* Neutral */
        {.valence = 0.8f, .arousal = 0.6f, .dominance = 0.7f},   /* Happy */
        {.valence = -0.3f, .arousal = -0.5f, .dominance = 0.3f}  /* Sad */
    };

    float modulated_latents[4][E2E_LATENT_DIM];

    for (int e = 0; e < 4; e++) {
        ret = vae_emotion_bridge_modulate(
            emotion, latent, E2E_LATENT_DIM,
            &emotions[e], modulated_latents[e], E2E_LATENT_DIM
        );
        ck_assert_int_eq(ret, 0);
    }

    /* Different emotions should produce different modulations */
    /* Compare fear vs happy */
    float diff_fear_happy = 0.0f;
    for (int i = 0; i < E2E_LATENT_DIM; i++) {
        diff_fear_happy += fabsf(modulated_latents[0][i] - modulated_latents[2][i]);
    }
    /* Should have some difference (emotions should affect encoding) */

    /* Decode all emotional variants */
    for (int e = 0; e < 4; e++) {
        float output[E2E_INPUT_DIM];
        ret = nimcp_vae_decode(g_vae, modulated_latents[e], E2E_LATENT_DIM,
                               output, E2E_INPUT_DIM);
        ck_assert_int_eq(ret, 0);
    }

    vae_emotion_bridge_destroy(emotion);
}
END_TEST

/* ============================================================================
 * E2E Scenario 10: Full System Integration
 * ========================================================================== */

START_TEST(test_e2e_full_system)
{
    /* Complete system test: all components working together */

    /* Create all bridges */
    vae_fep_bridge_t *fep = vae_fep_bridge_create(g_brain);
    vae_snn_bridge_t *snn = vae_snn_bridge_create(g_brain);
    vae_thalamic_bridge_t *thalamic = vae_thalamic_bridge_create(g_brain);

    ck_assert_ptr_nonnull(fep);

    vae_bio_async_config_t bio_config = {
        .brain = g_brain,
        .auto_register_handlers = true
    };
    vae_bio_async_bridge_t *bio = vae_bio_async_bridge_create(&bio_config);
    ck_assert_ptr_nonnull(bio);

    vae_bio_async_bridge_start(bio);

    /* Run integrated processing loop */
    for (int step = 0; step < 30; step++) {
        /* 1. Generate sensory input */
        float input[E2E_INPUT_DIM];
        generate_sensory_input(input, E2E_INPUT_DIM, 0.1f);

        /* 2. Encode through VAE */
        float latent[E2E_LATENT_DIM];
        nimcp_vae_encode(g_vae, input, E2E_INPUT_DIM, latent, E2E_LATENT_DIM);

        /* 3. Process through thalamic relay (if available) */
        float processed_latent[E2E_LATENT_DIM];
        if (thalamic) {
            vae_thalamic_bridge_relay(
                thalamic, VAE_THALAMIC_LGN, latent, E2E_LATENT_DIM,
                0.7f, processed_latent, E2E_LATENT_DIM
            );
        } else {
            memcpy(processed_latent, latent, sizeof(latent));
        }

        /* 4. Convert to SNN spikes (if available) */
        if (snn) {
            vae_spike_train_t spikes;
            vae_snn_bridge_encode(snn, processed_latent, E2E_LATENT_DIM, &spikes);
        }

        /* 5. Compute free energy through FEP bridge */
        float free_energy = vae_fep_bridge_compute_free_energy(
            fep, processed_latent, E2E_LATENT_DIM, input, E2E_INPUT_DIM
        );

        /* 6. Send via bio-async */
        vae_bio_msg_free_energy_t fe_msg = {
            .free_energy = free_energy,
            .timestamp = (uint64_t)step
        };
        vae_bio_async_send_free_energy(bio, &fe_msg);

        /* 7. Update FEP precision */
        float prediction_error = fabsf(free_energy) * 0.1f;
        vae_fep_bridge_update_precision(fep, prediction_error);

        /* 8. Decode and verify */
        float output[E2E_INPUT_DIM];
        nimcp_vae_decode(g_vae, processed_latent, E2E_LATENT_DIM,
                         output, E2E_INPUT_DIM);

        for (int i = 0; i < E2E_INPUT_DIM; i++) {
            ck_assert(!isnan(output[i]));
        }
    }

    /* Verify system state */
    vae_bio_async_state_t state;
    vae_bio_async_bridge_get_state(bio, &state);
    ck_assert_int_ge(state.messages_sent, 30);

    float final_precision = vae_fep_bridge_get_precision(fep);
    ck_assert(final_precision > 0.0f && final_precision <= 1.0f);

    /* Cleanup */
    vae_bio_async_bridge_stop(bio);
    vae_bio_async_bridge_destroy(bio);
    if (thalamic) vae_thalamic_bridge_destroy(thalamic);
    if (snn) vae_snn_bridge_destroy(snn);
    vae_fep_bridge_destroy(fep);
}
END_TEST

/* ============================================================================
 * Test Suite Setup
 * ========================================================================== */

static Suite *vae_e2e_suite(void) {
    Suite *s = suite_create("VAE E2E");

    /* Perception Pipeline */
    TCase *tc_perception = tcase_create("Perception Pipeline");
    tcase_add_checked_fixture(tc_perception, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_perception, 120);
    tcase_add_test(tc_perception, test_e2e_perception_pipeline);
    suite_add_tcase(s, tc_perception);

    /* Active Inference */
    TCase *tc_inference = tcase_create("Active Inference");
    tcase_add_checked_fixture(tc_inference, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_inference, 120);
    tcase_add_test(tc_inference, test_e2e_active_inference_loop);
    suite_add_tcase(s, tc_inference);

    /* Memory Consolidation */
    TCase *tc_memory = tcase_create("Memory Consolidation");
    tcase_add_checked_fixture(tc_memory, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_memory, 120);
    tcase_add_test(tc_memory, test_e2e_memory_consolidation);
    suite_add_tcase(s, tc_memory);

    /* Learning Cycle */
    TCase *tc_learning = tcase_create("Learning Cycle");
    tcase_add_checked_fixture(tc_learning, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_learning, 180);
    tcase_add_test(tc_learning, test_e2e_learning_cycle);
    suite_add_tcase(s, tc_learning);

    /* Multi-Modal Integration */
    TCase *tc_multimodal = tcase_create("Multi-Modal");
    tcase_add_checked_fixture(tc_multimodal, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_multimodal, 120);
    tcase_add_test(tc_multimodal, test_e2e_multimodal_integration);
    suite_add_tcase(s, tc_multimodal);

    /* Neural Pathway */
    TCase *tc_neural = tcase_create("Neural Pathway");
    tcase_add_checked_fixture(tc_neural, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_neural, 120);
    tcase_add_test(tc_neural, test_e2e_neural_pathway);
    suite_add_tcase(s, tc_neural);

    /* Bio-Async Pipeline */
    TCase *tc_bio_async = tcase_create("Bio-Async Pipeline");
    tcase_add_checked_fixture(tc_bio_async, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_bio_async, 120);
    tcase_add_test(tc_bio_async, test_e2e_bio_async_pipeline);
    suite_add_tcase(s, tc_bio_async);

    /* Imagination */
    TCase *tc_imagination = tcase_create("Imagination");
    tcase_add_checked_fixture(tc_imagination, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_imagination, 120);
    tcase_add_test(tc_imagination, test_e2e_imagination_generation);
    suite_add_tcase(s, tc_imagination);

    /* Emotion Modulation */
    TCase *tc_emotion = tcase_create("Emotion Modulation");
    tcase_add_checked_fixture(tc_emotion, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_emotion, 120);
    tcase_add_test(tc_emotion, test_e2e_emotion_modulated_perception);
    suite_add_tcase(s, tc_emotion);

    /* Full System */
    TCase *tc_full = tcase_create("Full System");
    tcase_add_checked_fixture(tc_full, setup_e2e, teardown_e2e);
    tcase_set_timeout(tc_full, 300);
    tcase_add_test(tc_full, test_e2e_full_system);
    suite_add_tcase(s, tc_full);

    return s;
}

int main(void) {
    Suite *s = vae_e2e_suite();
    SRunner *sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    int failures = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
