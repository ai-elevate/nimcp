/**
 * @file nimcp_snn_speech_bridge.c
 * @brief SNN-Speech Cortex integration bridge implementation
 *
 * WHAT: Bidirectional bridge between SNN and speech_cortex_t
 * WHY:  Enable spike-based speech processing with phoneme encoding/decoding
 * HOW:  Population coding for phonemes, temporal coding for sequences
 *
 * BIOLOGICAL BASIS:
 * - Superior Temporal Gyrus (STG) contains phoneme-selective spiking neurons
 * - Speech perception relies on precise spike timing in auditory cortex
 * - Motor cortex (Broca's area) generates spike sequences for articulation
 * - Phonological working memory exhibits spike-based serial order encoding
 *
 * @author NIMCP Team
 * @date 2024-12-20
 */

#include "snn/bridges/nimcp_snn_speech_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <math.h>
#include <string.h>

//=============================================================================
// Constants
//=============================================================================

/* Standard English phoneme formant centers (F1, F2 in Hz) */
static const float PHONEME_FORMANTS[44][2] = {
    {270, 2290},  /* /i/ */
    {390, 1990},  /* /ɪ/ */
    {530, 1840},  /* /e/ */
    {660, 1720},  /* /ɛ/ */
    {730, 1090},  /* /æ/ */
    {570, 840},   /* /ɑ/ */
    {440, 1020},  /* /ɔ/ */
    {370, 950},   /* /o/ */
    {440, 1020},  /* /ʊ/ */
    {300, 870},   /* /u/ */
    {640, 1190},  /* /ʌ/ */
    {490, 1350},  /* /ə/ */
    /* Consonants use simplified representations */
    {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 0}, {0, 0}
};

//=============================================================================
// Default Configuration
//=============================================================================

/**
 * WHAT: Initialize config with biologically-plausible defaults
 * WHY:  Convenient starting point for speech processing
 * HOW:  Literature-based parameter values from STG studies
 */
void snn_speech_config_default(snn_speech_config_t* config) {
    if (!config) return;

    /* Encoding configuration */
    config->encoding_method = SNN_ENCODE_POPULATION;
    config->max_spike_rate = 150.0f;         /* STG phoneme neurons */
    config->min_spike_rate = 5.0f;
    config->temporal_window_ms = 50.0f;      /* Phoneme duration ~50-100ms */
    config->neurons_per_phoneme = 10;        /* Population per phoneme */

    /* Decoding configuration */
    config->decoding_method = SNN_DECODE_POPULATION;
    config->decode_window_ms = 50.0f;
    config->use_winner_take_all = true;      /* WTA for phoneme selection */

    /* Phoneme processing */
    config->num_phonemes = 44;               /* English IPA subset */
    config->num_formants = 4;                /* F1, F2, F3, F4 */
    config->encode_formants = true;
    config->encode_prosody = true;

    /* Temporal coding */
    config->use_sequence_encoding = true;
    config->use_position_encoding = true;
    config->max_sequence_length = 20;
    config->inter_phoneme_interval_ms = 80.0f;

    /* Phonological working memory */
    config->encode_buffer_position = true;
    config->buffer_capacity = 9;             /* 7±2 items */

    /* Bio-async */
    config->enable_bio_async = false;
    config->update_interval_ms = 50.0f;
}

//=============================================================================
// Bridge Lifecycle
//=============================================================================

/**
 * WHAT: Create SNN-speech bridge
 * WHY:  Initialize bidirectional speech processing
 * HOW:  Allocate buffers, create encoder/decoder, init tuning curves
 */
snn_speech_bridge_t* snn_speech_bridge_create(
    const snn_speech_config_t* config,
    snn_network_t* snn,
    speech_cortex_t* speech_cortex
) {
    /* Guard: Validate inputs */
    if (!config || !snn || !speech_cortex) {
        NIMCP_LOGGING_ERROR("Null parameters to snn_speech_bridge_create");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_speech_bridge_create: config/snn/speech_cortex is NULL");
        return NULL;
    }

    /* Allocate bridge */
    snn_speech_bridge_t* bridge = nimcp_malloc(sizeof(snn_speech_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate SNN-speech bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_speech_bridge_create: failed to allocate bridge");
        return NULL;
    }

    /* Initialize structure */
    memset(bridge, 0, sizeof(snn_speech_bridge_t));
    bridge->snn = snn;
    bridge->speech_cortex = speech_cortex;
    bridge->config = *config;

    /* Calculate buffer sizes */
    uint32_t num_neurons = config->num_phonemes * config->neurons_per_phoneme;

    /* Allocate working buffers */
    bridge->formant_buffer = nimcp_malloc(config->num_formants * sizeof(float));
    bridge->phoneme_features_buffer = nimcp_malloc(num_neurons * sizeof(float));
    bridge->spike_input_buffer = nimcp_malloc(num_neurons * sizeof(float));
    bridge->spike_output_buffer = nimcp_malloc(num_neurons * sizeof(float));
    bridge->spike_mask = nimcp_malloc(num_neurons);

    if (!bridge->formant_buffer || !bridge->phoneme_features_buffer ||
        !bridge->spike_input_buffer || !bridge->spike_output_buffer ||
        !bridge->spike_mask) {
        NIMCP_LOGGING_ERROR("Failed to allocate speech bridge buffers");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_speech_bridge_create: failed to allocate buffers");
        snn_speech_bridge_destroy(bridge);
        return NULL;
    }

    /* Allocate sequence buffers */
    bridge->phoneme_sequence = nimcp_malloc(config->max_sequence_length *
                                            sizeof(phoneme_t));
    bridge->sequence_spike_times = nimcp_malloc(config->max_sequence_length *
                                                num_neurons * sizeof(float));
    if (!bridge->phoneme_sequence || !bridge->sequence_spike_times) {
        NIMCP_LOGGING_ERROR("Failed to allocate sequence buffers");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_speech_bridge_create: failed to allocate sequence buffers");
        snn_speech_bridge_destroy(bridge);
        return NULL;
    }

    /* Allocate phonological buffer */
    bridge->phonological_buffer = nimcp_malloc(config->buffer_capacity *
                                               sizeof(phoneme_t));
    bridge->buffer_position_encoding = nimcp_malloc(config->buffer_capacity *
                                                    num_neurons * sizeof(float));
    if (!bridge->phonological_buffer || !bridge->buffer_position_encoding) {
        NIMCP_LOGGING_ERROR("Failed to allocate phonological buffer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_speech_bridge_create: failed to allocate phonological buffer");
        snn_speech_bridge_destroy(bridge);
        return NULL;
    }

    /* Allocate tuning curves */
    bridge->phoneme_tuning_curves = nimcp_malloc(config->num_phonemes *
                                                  sizeof(float*));
    if (!bridge->phoneme_tuning_curves) {
        NIMCP_LOGGING_ERROR("Failed to allocate tuning curves array");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_speech_bridge_create: failed to allocate tuning curves");
        snn_speech_bridge_destroy(bridge);
        return NULL;
    }

    for (uint32_t p = 0; p < config->num_phonemes; p++) {
        bridge->phoneme_tuning_curves[p] = nimcp_malloc(config->neurons_per_phoneme *
                                                        sizeof(float));
        if (!bridge->phoneme_tuning_curves[p]) {
            NIMCP_LOGGING_ERROR("Failed to allocate tuning curve for phoneme %u", p);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_speech_bridge_create: failed to allocate tuning curve");
            snn_speech_bridge_destroy(bridge);
            return NULL;
        }
    }

    bridge->phoneme_preferred_values = nimcp_malloc(num_neurons * sizeof(float));
    if (!bridge->phoneme_preferred_values) {
        NIMCP_LOGGING_ERROR("Failed to allocate preferred values");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_speech_bridge_create: failed to allocate preferred values");
        snn_speech_bridge_destroy(bridge);
        return NULL;
    }

    /* Create rate encoder - uses rate coding for speech phoneme input */
    snn_rate_encoder_config_t rate_enc_cfg;
    snn_rate_encoder_config_default(&rate_enc_cfg);
    rate_enc_cfg.max_rate = config->max_spike_rate;
    rate_enc_cfg.min_rate = config->min_spike_rate;
    rate_enc_cfg.value_min = 0.0f;
    rate_enc_cfg.value_max = 1.0f;
    rate_enc_cfg.use_poisson = true;  /* Biologically realistic */
    bridge->encoder = snn_encoder_create_rate(num_neurons, &rate_enc_cfg);

    /* Create rate decoder */
    snn_rate_decoder_config_t rate_dec_cfg;
    snn_rate_decoder_config_default(&rate_dec_cfg);
    rate_dec_cfg.time_window = config->decode_window_ms;
    rate_dec_cfg.max_rate = config->max_spike_rate;
    rate_dec_cfg.use_exponential = false;
    bridge->decoder = snn_decoder_create_rate(num_neurons, num_neurons, &rate_dec_cfg);

    /* Initialize tuning curves */
    snn_speech_bridge_init_tuning_curves(bridge);

    /* Mark as connected */
    bridge->connected = true;

    NIMCP_LOGGING_INFO("Created SNN-speech bridge (%u phonemes, %u neurons)",
                       config->num_phonemes, num_neurons);
    return bridge;
}

/**
 * WHAT: Destroy bridge and free resources
 * WHY:  Proper cleanup and memory management
 * HOW:  Disconnect, free all buffers, destroy encoder/decoder
 */
void snn_speech_bridge_destroy(snn_speech_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        snn_speech_bridge_disconnect_bio_async(bridge);
    }

    /* Free working buffers */
    if (bridge->formant_buffer) nimcp_free(bridge->formant_buffer);
    if (bridge->phoneme_features_buffer) nimcp_free(bridge->phoneme_features_buffer);
    if (bridge->spike_input_buffer) nimcp_free(bridge->spike_input_buffer);
    if (bridge->spike_output_buffer) nimcp_free(bridge->spike_output_buffer);
    if (bridge->spike_mask) nimcp_free(bridge->spike_mask);

    /* Free sequence buffers */
    if (bridge->phoneme_sequence) nimcp_free(bridge->phoneme_sequence);
    if (bridge->sequence_spike_times) nimcp_free(bridge->sequence_spike_times);

    /* Free phonological buffer */
    if (bridge->phonological_buffer) nimcp_free(bridge->phonological_buffer);
    if (bridge->buffer_position_encoding) nimcp_free(bridge->buffer_position_encoding);

    /* Free tuning curves */
    if (bridge->phoneme_tuning_curves) {
        for (uint32_t p = 0; p < bridge->config.num_phonemes; p++) {
            if (bridge->phoneme_tuning_curves[p]) {
                nimcp_free(bridge->phoneme_tuning_curves[p]);
            }
        }
        nimcp_free(bridge->phoneme_tuning_curves);
    }
    if (bridge->phoneme_preferred_values) nimcp_free(bridge->phoneme_preferred_values);

    /* Destroy encoder/decoder */
    if (bridge->encoder) snn_encoder_destroy(bridge->encoder);
    if (bridge->decoder) snn_decoder_destroy(bridge->decoder);

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed SNN-speech bridge");
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

/**
 * WHAT: Connect to bio-async messaging
 * WHY:  Enable distributed speech event coordination
 * HOW:  Register with router as BIO_MODULE_SNN_SPEECH
 */
int snn_speech_bridge_connect_bio_async(snn_speech_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SNN_SPEECH,
        .module_name = "snn_speech_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Speech bridge connected to bio-async router");
        return 0;
    }

    NIMCP_LOGGING_WARN("Bio-async router not available for speech bridge");
    return -1;
}

/**
 * WHAT: Disconnect from bio-async
 * WHY:  Clean shutdown
 * HOW:  Unregister from router
 */
int snn_speech_bridge_disconnect_bio_async(snn_speech_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;

    bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Speech bridge disconnected from bio-async");
    return 0;
}

/**
 * WHAT: Check bio-async connection status
 * WHY:  Query before sending messages
 * HOW:  Return flag
 */
bool snn_speech_bridge_is_bio_async_connected(const snn_speech_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

//=============================================================================
// Tuning Curve Functions
//=============================================================================

/**
 * WHAT: Initialize phoneme tuning curves
 * WHY:  Model phoneme-selective neurons in STG
 * HOW:  Gaussian receptive fields in formant space (F1, F2)
 */
int snn_speech_bridge_init_tuning_curves(snn_speech_bridge_t* bridge) {
    /* Guard: Validate bridge */
    if (!bridge || !bridge->phoneme_tuning_curves) {
        return -1;
    }

    uint32_t npn = bridge->config.neurons_per_phoneme;
    float sigma = 100.0f;  /* Tuning curve width in Hz */

    for (uint32_t p = 0; p < bridge->config.num_phonemes; p++) {
        float f1_center = PHONEME_FORMANTS[p][0];
        float f2_center = PHONEME_FORMANTS[p][1];

        for (uint32_t n = 0; n < npn; n++) {
            /* Distribute neurons around formant center */
            float offset = (float)(n - npn/2) * sigma / 2.0f;
            bridge->phoneme_preferred_values[p * npn + n] = f1_center + offset;

            /* Initialize tuning curve amplitude */
            bridge->phoneme_tuning_curves[p][n] = 1.0f;
        }
    }

    NIMCP_LOGGING_INFO("Initialized %u phoneme tuning curves",
                       bridge->config.num_phonemes);
    return 0;
}

/**
 * WHAT: Compute population activity for phoneme features
 * WHY:  Convert features to population spike rates
 * HOW:  Gaussian activation based on feature distance
 */
int snn_speech_bridge_compute_population_activity(
    snn_speech_bridge_t* bridge,
    const phoneme_features_t* features,
    float* activities_out
) {
    /* Guard: Validate inputs */
    if (!bridge || !features || !activities_out) {
        return -1;
    }

    uint32_t npn = bridge->config.neurons_per_phoneme;
    float sigma_sq = 10000.0f;  /* Tuning width squared */

    for (uint32_t p = 0; p < bridge->config.num_phonemes; p++) {
        float f1_center = PHONEME_FORMANTS[p][0];
        float f2_center = PHONEME_FORMANTS[p][1];

        /* Skip non-vowels with zero formants */
        if (f1_center == 0.0f && f2_center == 0.0f) {
            for (uint32_t n = 0; n < npn; n++) {
                activities_out[p * npn + n] = 0.0f;
            }
            continue;
        }

        /* Compute distance in formant space */
        float d1 = features->formant_f1 - f1_center;
        float d2 = features->formant_f2 - f2_center;
        float dist_sq = d1 * d1 + d2 * d2;

        /* Gaussian activation */
        float base_activity = expf(-dist_sq / (2.0f * sigma_sq));

        for (uint32_t n = 0; n < npn; n++) {
            /* Apply tuning curve modulation */
            float activity = base_activity * bridge->phoneme_tuning_curves[p][n];
            activities_out[p * npn + n] = activity;
        }
    }

    return 0;
}

//=============================================================================
// Encoding Functions
//=============================================================================

/**
 * WHAT: Encode phoneme to spike trains
 * WHY:  Convert phoneme features to SNN input
 * HOW:  Population coding with Gaussian tuning curves
 */
int snn_speech_bridge_encode_phoneme(
    snn_speech_bridge_t* bridge,
    phoneme_t phoneme,
    const phoneme_features_t* features,
    snn_spike_train_t** spike_trains
) {
    /* Guard: Validate inputs */
    if (!bridge || !spike_trains) {
        NIMCP_LOGGING_ERROR("Null parameters to encode_phoneme");
        return -1;
    }

    /* Validate phoneme */
    if (phoneme >= bridge->config.num_phonemes) {
        NIMCP_LOGGING_ERROR("Invalid phoneme: %u", phoneme);
        return -1;
    }

    uint32_t num_neurons = bridge->config.num_phonemes *
                           bridge->config.neurons_per_phoneme;

    /* Compute population activity */
    if (features) {
        int ret = snn_speech_bridge_compute_population_activity(
            bridge, features, bridge->spike_input_buffer);
        if (ret != 0) {
            return ret;
        }
    } else {
        /* Use one-hot encoding if no features */
        memset(bridge->spike_input_buffer, 0, num_neurons * sizeof(float));
        uint32_t start = phoneme * bridge->config.neurons_per_phoneme;
        for (uint32_t n = 0; n < bridge->config.neurons_per_phoneme; n++) {
            bridge->spike_input_buffer[start + n] = 1.0f;
        }
    }

    /* Encode to spike mask using rate coding */
    float dt = bridge->config.update_interval_ms;
    int ret = snn_encode_rate(bridge->encoder, bridge->spike_input_buffer,
                               dt, bridge->spike_mask);
    if (ret != 0) {
        NIMCP_LOGGING_ERROR("Failed to encode phoneme to spikes");
        return ret;
    }

    /* Allocate spike trains if needed */
    if (!*spike_trains) {
        *spike_trains = (snn_spike_train_t*)nimcp_calloc(num_neurons,
                                                          sizeof(snn_spike_train_t));
        if (!*spike_trains) {
            return SNN_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < num_neurons; i++) {
            (*spike_trains)[i].neuron_id = i;
        }
    }

    /* Convert binary spike mask to spike trains */
    uint64_t current_time_us = (uint64_t)(bridge->last_update_time_ms * 1000.0f);
    for (uint32_t i = 0; i < num_neurons; i++) {
        if (bridge->spike_mask[i]) {
            snn_spike_train_t* train = &((*spike_trains)[i]);
            train->spike_times[train->write_idx] = current_time_us;
            train->write_idx = (train->write_idx + 1) % SNN_SPIKE_BUFFER_SIZE;
            if (train->count < SNN_SPIKE_BUFFER_SIZE) {
                train->count++;
            }
            train->total_spikes++;
        }
    }

    /* Update statistics */
    bridge->encode_stats.phonemes_encoded++;
    bridge->phoneme_count++;

    return 0;
}

/**
 * WHAT: Encode phoneme sequence to spike patterns
 * WHY:  Enable SNN processing of words/sentences
 * HOW:  Temporal coding with inter-phoneme intervals
 */
int snn_speech_bridge_encode_sequence(
    snn_speech_bridge_t* bridge,
    const phoneme_t* phonemes,
    uint32_t num_phonemes,
    const phoneme_features_t* features,
    snn_spike_train_t** spike_trains
) {
    /* Guard: Validate inputs */
    if (!bridge || !phonemes || !spike_trains) {
        return -1;
    }

    if (num_phonemes > bridge->config.max_sequence_length) {
        NIMCP_LOGGING_WARN("Sequence truncated from %u to %u",
                           num_phonemes, bridge->config.max_sequence_length);
        num_phonemes = bridge->config.max_sequence_length;
    }

    /* Store sequence */
    memcpy(bridge->phoneme_sequence, phonemes, num_phonemes * sizeof(phoneme_t));
    bridge->sequence_length = num_phonemes;

    uint32_t num_neurons = bridge->config.num_phonemes *
                           bridge->config.neurons_per_phoneme;
    float interval = bridge->config.inter_phoneme_interval_ms;

    /* Clear spike times buffer */
    memset(bridge->sequence_spike_times, 0,
           bridge->config.max_sequence_length * num_neurons * sizeof(float));

    /* Encode each phoneme with temporal offset */
    for (uint32_t i = 0; i < num_phonemes; i++) {
        phoneme_t p = phonemes[i];
        float time_offset = i * interval;

        /* Compute population activity for this phoneme */
        if (features) {
            snn_speech_bridge_compute_population_activity(
                bridge, &features[i], bridge->spike_input_buffer);
        } else {
            memset(bridge->spike_input_buffer, 0, num_neurons * sizeof(float));
            uint32_t start = p * bridge->config.neurons_per_phoneme;
            for (uint32_t n = 0; n < bridge->config.neurons_per_phoneme; n++) {
                bridge->spike_input_buffer[start + n] = 1.0f;
            }
        }

        /* Add position encoding if enabled */
        if (bridge->config.use_position_encoding) {
            float pos_factor = 1.0f - 0.1f * i;  /* Primacy effect */
            if (i == num_phonemes - 1) {
                pos_factor += 0.2f;  /* Recency effect */
            }
            for (uint32_t n = 0; n < num_neurons; n++) {
                bridge->spike_input_buffer[n] *= pos_factor;
            }
        }

        /* Store spike times with offset */
        for (uint32_t n = 0; n < num_neurons; n++) {
            if (bridge->spike_input_buffer[n] > 0.1f) {
                bridge->sequence_spike_times[i * num_neurons + n] =
                    time_offset + (1.0f - bridge->spike_input_buffer[n]) * interval * 0.5f;
            }
        }
    }

    /* Convert accumulated activity to spike mask using rate coding */
    float dt = bridge->config.update_interval_ms;
    int ret = snn_encode_rate(bridge->encoder, bridge->spike_input_buffer,
                               dt, bridge->spike_mask);
    if (ret != 0) {
        return ret;
    }

    /* Allocate spike trains if needed */
    if (!*spike_trains) {
        *spike_trains = (snn_spike_train_t*)nimcp_calloc(num_neurons,
                                                          sizeof(snn_spike_train_t));
        if (!*spike_trains) {
            return SNN_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < num_neurons; i++) {
            (*spike_trains)[i].neuron_id = i;
        }
    }

    /* Convert binary spike mask to spike trains */
    uint64_t current_time_us = (uint64_t)(bridge->last_update_time_ms * 1000.0f);
    for (uint32_t i = 0; i < num_neurons; i++) {
        if (bridge->spike_mask[i]) {
            snn_spike_train_t* train = &((*spike_trains)[i]);
            train->spike_times[train->write_idx] = current_time_us;
            train->write_idx = (train->write_idx + 1) % SNN_SPIKE_BUFFER_SIZE;
            if (train->count < SNN_SPIKE_BUFFER_SIZE) {
                train->count++;
            }
            train->total_spikes++;
        }
    }

    /* Update statistics */
    bridge->encode_stats.sequences_encoded++;
    bridge->encode_stats.avg_sequence_length =
        (bridge->encode_stats.avg_sequence_length *
         (bridge->encode_stats.sequences_encoded - 1) + num_phonemes) /
        bridge->encode_stats.sequences_encoded;

    return 0;
}

/**
 * WHAT: Encode phonological buffer to spikes
 * WHY:  Enable SNN processing of phonological working memory
 * HOW:  Position-dependent encoding of buffer items (7±2)
 */
int snn_speech_bridge_encode_phonological_buffer(
    snn_speech_bridge_t* bridge,
    const phoneme_t* buffer,
    uint32_t buffer_size,
    snn_spike_train_t** spike_trains
) {
    /* Guard: Validate inputs */
    if (!bridge || !buffer || !spike_trains) {
        return -1;
    }

    if (buffer_size > bridge->config.buffer_capacity) {
        buffer_size = bridge->config.buffer_capacity;
    }

    /* Copy to internal buffer */
    memcpy(bridge->phonological_buffer, buffer, buffer_size * sizeof(phoneme_t));
    bridge->buffer_fill = buffer_size;

    uint32_t num_neurons = bridge->config.num_phonemes *
                           bridge->config.neurons_per_phoneme;

    /* Clear input buffer */
    memset(bridge->spike_input_buffer, 0, num_neurons * sizeof(float));

    /* Encode each buffer position with position-dependent modulation */
    for (uint32_t pos = 0; pos < buffer_size; pos++) {
        phoneme_t p = buffer[pos];
        if (p >= bridge->config.num_phonemes) continue;

        /* Position-dependent gain (primacy/recency) */
        float pos_gain = 1.0f;
        if (pos < 2) {
            pos_gain = 1.2f;  /* Primacy boost */
        } else if (pos >= buffer_size - 2) {
            pos_gain = 1.1f;  /* Recency boost */
        }

        /* Add to population activity */
        uint32_t start = p * bridge->config.neurons_per_phoneme;
        for (uint32_t n = 0; n < bridge->config.neurons_per_phoneme; n++) {
            bridge->spike_input_buffer[start + n] += pos_gain / buffer_size;
        }
    }

    /* Normalize */
    for (uint32_t n = 0; n < num_neurons; n++) {
        if (bridge->spike_input_buffer[n] > 1.0f) {
            bridge->spike_input_buffer[n] = 1.0f;
        }
    }

    /* Encode to spike mask using rate coding */
    float dt = bridge->config.update_interval_ms;
    int ret = snn_encode_rate(bridge->encoder, bridge->spike_input_buffer,
                               dt, bridge->spike_mask);
    if (ret != 0) {
        return ret;
    }

    /* Allocate spike trains if needed */
    if (!*spike_trains) {
        *spike_trains = (snn_spike_train_t*)nimcp_calloc(num_neurons,
                                                          sizeof(snn_spike_train_t));
        if (!*spike_trains) {
            return SNN_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < num_neurons; i++) {
            (*spike_trains)[i].neuron_id = i;
        }
    }

    /* Convert binary spike mask to spike trains */
    uint64_t current_time_us = (uint64_t)(bridge->last_update_time_ms * 1000.0f);
    for (uint32_t i = 0; i < num_neurons; i++) {
        if (bridge->spike_mask[i]) {
            snn_spike_train_t* train = &((*spike_trains)[i]);
            train->spike_times[train->write_idx] = current_time_us;
            train->write_idx = (train->write_idx + 1) % SNN_SPIKE_BUFFER_SIZE;
            if (train->count < SNN_SPIKE_BUFFER_SIZE) {
                train->count++;
            }
            train->total_spikes++;
        }
    }

    return 0;
}

//=============================================================================
// Decoding Functions
//=============================================================================

/**
 * WHAT: Decode spike trains to phoneme
 * WHY:  Recognize phoneme from spike activity
 * HOW:  Population vector or winner-take-all decoding
 */
int snn_speech_bridge_decode_phoneme(
    snn_speech_bridge_t* bridge,
    const snn_spike_train_t* spike_trains,
    uint32_t num_trains,
    phoneme_t* phoneme_out,
    float* confidence_out
) {
    /* Guard: Validate inputs */
    if (!bridge || !spike_trains || !phoneme_out) {
        return -1;
    }

    /* Extract spike counts from spike trains into input buffer */
    for (uint32_t i = 0; i < num_trains; i++) {
        bridge->spike_input_buffer[i] = (float)spike_trains[i].count;
    }

    /* Decode spike counts to activity using rate decoding */
    int ret = snn_decode_rate(bridge->decoder, bridge->spike_input_buffer,
                               bridge->spike_output_buffer);
    if (ret != 0) {
        return ret;
    }

    /* Sum activity per phoneme (population pooling) */
    uint32_t npn = bridge->config.neurons_per_phoneme;
    float max_activity = 0.0f;
    phoneme_t winner = 0;
    float total_activity = 0.0f;

    for (uint32_t p = 0; p < bridge->config.num_phonemes; p++) {
        float sum = 0.0f;
        for (uint32_t n = 0; n < npn && p * npn + n < num_trains; n++) {
            sum += bridge->spike_output_buffer[p * npn + n];
        }

        total_activity += sum;
        if (sum > max_activity) {
            max_activity = sum;
            winner = p;
        }
    }

    *phoneme_out = winner;
    if (confidence_out) {
        *confidence_out = (total_activity > 0.0f) ?
            max_activity / total_activity : 0.0f;
    }

    /* Update statistics */
    bridge->decode_stats.phonemes_decoded++;

    return 0;
}

/**
 * WHAT: Decode spike patterns to phoneme sequence
 * WHY:  Recognize word/sentence from spike activity
 * HOW:  Temporal segmentation + phoneme decoding
 */
int snn_speech_bridge_decode_sequence(
    snn_speech_bridge_t* bridge,
    const snn_spike_train_t* spike_trains,
    uint32_t num_trains,
    phoneme_t* phonemes_out,
    uint32_t max_phonemes,
    uint32_t* num_decoded
) {
    /* Guard: Validate inputs */
    if (!bridge || !spike_trains || !phonemes_out || !num_decoded) {
        return -1;
    }

    /* Extract spike counts from spike trains into input buffer */
    for (uint32_t i = 0; i < num_trains; i++) {
        bridge->spike_input_buffer[i] = (float)spike_trains[i].count;
    }

    /* Decode spike counts to activity using rate decoding */
    int ret = snn_decode_rate(bridge->decoder, bridge->spike_input_buffer,
                               bridge->spike_output_buffer);
    if (ret != 0) {
        return ret;
    }

    /* Simple decoding: find peaks in activity */
    uint32_t npn = bridge->config.neurons_per_phoneme;
    uint32_t decoded_count = 0;
    float threshold = 0.3f;

    for (uint32_t p = 0; p < bridge->config.num_phonemes && decoded_count < max_phonemes; p++) {
        float sum = 0.0f;
        for (uint32_t n = 0; n < npn && p * npn + n < num_trains; n++) {
            sum += bridge->spike_output_buffer[p * npn + n];
        }

        if (sum / npn > threshold) {
            phonemes_out[decoded_count++] = p;
        }
    }

    *num_decoded = decoded_count;
    return 0;
}

/**
 * WHAT: Decode spikes to phoneme features
 * WHY:  Extract acoustic features from spike representation
 * HOW:  Population vector decoding to formant space
 */
int snn_speech_bridge_decode_features(
    snn_speech_bridge_t* bridge,
    const snn_spike_train_t* spike_trains,
    uint32_t num_trains,
    phoneme_features_t* features_out
) {
    /* Guard: Validate inputs */
    if (!bridge || !spike_trains || !features_out) {
        return -1;
    }

    /* Extract spike counts from spike trains into input buffer */
    for (uint32_t i = 0; i < num_trains; i++) {
        bridge->spike_input_buffer[i] = (float)spike_trains[i].count;
    }

    /* Decode spike counts to activity using rate decoding */
    int ret = snn_decode_rate(bridge->decoder, bridge->spike_input_buffer,
                               bridge->spike_output_buffer);
    if (ret != 0) {
        return ret;
    }

    /* Population vector decoding for formants */
    float f1_sum = 0.0f, f2_sum = 0.0f;
    float weight_sum = 0.0f;
    uint32_t npn = bridge->config.neurons_per_phoneme;

    for (uint32_t p = 0; p < bridge->config.num_phonemes; p++) {
        float f1_center = PHONEME_FORMANTS[p][0];
        float f2_center = PHONEME_FORMANTS[p][1];

        if (f1_center == 0.0f) continue;  /* Skip consonants */

        float phoneme_activity = 0.0f;
        for (uint32_t n = 0; n < npn && p * npn + n < num_trains; n++) {
            phoneme_activity += bridge->spike_output_buffer[p * npn + n];
        }

        f1_sum += f1_center * phoneme_activity;
        f2_sum += f2_center * phoneme_activity;
        weight_sum += phoneme_activity;
    }

    if (weight_sum > 0.0f) {
        features_out->formant_f1 = f1_sum / weight_sum;
        features_out->formant_f2 = f2_sum / weight_sum;
    } else {
        features_out->formant_f1 = 0.0f;
        features_out->formant_f2 = 0.0f;
    }

    return 0;
}

//=============================================================================
// Update Functions
//=============================================================================

/**
 * WHAT: Update bridge state
 * WHY:  Full update cycle for speech-SNN integration
 * HOW:  Process current speech from cortex
 */
int snn_speech_bridge_update(snn_speech_bridge_t* bridge, float dt) {
    /* Guard: Validate bridge */
    if (!bridge || !bridge->connected) {
        return -1;
    }

    /* Check update interval */
    bridge->last_update_time_ms += dt;
    if (bridge->last_update_time_ms < bridge->config.update_interval_ms) {
        return 0;
    }
    bridge->last_update_time_ms = 0.0f;

    return 0;
}

//=============================================================================
// Query Functions
//=============================================================================

/**
 * WHAT: Get encoding statistics
 * WHY:  Monitor encoding performance
 * HOW:  Copy statistics structure
 */
int snn_speech_bridge_get_encode_stats(
    const snn_speech_bridge_t* bridge,
    snn_speech_encode_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->encode_stats;
    return 0;
}

/**
 * WHAT: Get decoding statistics
 * WHY:  Monitor decoding performance
 * HOW:  Copy statistics structure
 */
int snn_speech_bridge_get_decode_stats(
    const snn_speech_bridge_t* bridge,
    snn_speech_decode_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->decode_stats;
    return 0;
}

/**
 * WHAT: Get current spike rate for phoneme
 * WHY:  Query individual phoneme activity
 * HOW:  Return average rate for phoneme population from input buffer
 */
float snn_speech_bridge_get_phoneme_spike_rate(
    const snn_speech_bridge_t* bridge,
    phoneme_t phoneme
) {
    if (!bridge || !bridge->spike_input_buffer) return -1.0f;
    if (phoneme >= bridge->config.num_phonemes) return -1.0f;

    uint32_t start = phoneme * bridge->config.neurons_per_phoneme;
    float sum = 0.0f;

    for (uint32_t n = 0; n < bridge->config.neurons_per_phoneme; n++) {
        float normalized = bridge->spike_input_buffer[start + n];
        float rate_range = bridge->config.max_spike_rate - bridge->config.min_spike_rate;
        sum += bridge->config.min_spike_rate + normalized * rate_range;
    }

    return sum / bridge->config.neurons_per_phoneme;
}

/**
 * WHAT: Check if bridge is active
 * WHY:  Validate bridge state
 * HOW:  Return connected flag
 */
bool snn_speech_bridge_is_active(const snn_speech_bridge_t* bridge) {
    return bridge ? bridge->connected : false;
}

//=============================================================================
// Statistics
//=============================================================================

/**
 * WHAT: Reset bridge statistics
 * WHY:  Start fresh measurement
 * HOW:  Zero all counters
 */
void snn_speech_bridge_reset_stats(snn_speech_bridge_t* bridge) {
    if (!bridge) return;

    memset(&bridge->encode_stats, 0, sizeof(snn_speech_encode_stats_t));
    memset(&bridge->decode_stats, 0, sizeof(snn_speech_decode_stats_t));
    bridge->phoneme_count = 0;
}
