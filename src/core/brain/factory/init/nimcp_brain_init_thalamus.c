//=============================================================================
// nimcp_brain_init_thalamus.c - Thalamus Subsystem Initialization
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_thalamus.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/medulla/nimcp_medulla.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
// Configuration
//=============================================================================

void nimcp_brain_thal_default_config(brain_t brain, thalamus_config_t* config) {
    if (!config) return;

    /* Start with module defaults */
    thalamus_default_config(config);

    if (!brain) return;

    struct brain_struct* b = (struct brain_struct*)brain;

    /* Enable features based on brain's enabled subsystems */

    /* Always enable TRN for proper gating */
    config->enable_trn = true;

    /* Enable mode switching for arousal-based relay */
    config->enable_mode_switching = true;

    /* Enable attention gating if attention system is present */
    config->enable_attention_gating = (b->multihead_attention != NULL);

    /* Configure nucleus channels based on enabled cortices */

    /* LGN: Visual relay */
    if (b->visual_cortex != NULL) {
        config->lgn_config.num_channels = 64;  /* Larger for visual processing */
        config->lgn_config.attention_weight = 0.7f;  /* Vision gets high attention */
    }

    /* MGN: Auditory relay */
    if (b->audio_cortex != NULL) {
        config->mgn_config.num_channels = 32;
        config->mgn_config.attention_weight = 0.6f;
    }

    /* VA: Motor from BG */
    if (b->basal_ganglia != NULL || b->basal_ganglia_enabled) {
        uint32_t num_actions = b->config.num_outputs;
        if (num_actions == 0) num_actions = 16;
        if (num_actions > THAL_MAX_CHANNELS) num_actions = THAL_MAX_CHANNELS;
        config->va_config.num_channels = num_actions;
    }

    /* MD: Executive relay */
    if (b->executive != NULL) {
        config->md_config.num_channels = 32;
        config->md_config.attention_weight = 0.8f;  /* Executive gets high priority */
    }

    /* Pulvinar: Attention coordination */
    if (b->multihead_attention != NULL || b->salience != NULL) {
        config->pulvinar_config.num_channels = 64;
        config->pulvinar_config.attention_weight = 1.0f;
    }

    /* TRN inhibition strength based on medulla */
    if (b->medulla_enabled && b->medulla) {
        config->trn_strength = 0.7f;  /* Moderate TRN influence */
    }

    NIMCP_LOGGING_DEBUG("Thalamus config: TRN=%d, mode_switch=%d, attn_gate=%d, "
        "lgn_ch=%u, mgn_ch=%u, va_ch=%u, md_ch=%u",
        config->enable_trn,
        config->enable_mode_switching,
        config->enable_attention_gating,
        config->lgn_config.num_channels,
        config->mgn_config.num_channels,
        config->va_config.num_channels,
        config->md_config.num_channels);
}

//=============================================================================
// Lifecycle
//=============================================================================

bool nimcp_brain_factory_init_thalamus_subsystem(brain_t brain) {
    if (!brain) {
        NIMCP_LOGGING_ERROR("Cannot init thalamus: NULL brain");
        return false;
    }

    struct brain_struct* b = (struct brain_struct*)brain;

    /* Skip if already initialized */
    if (b->thalamus != NULL) {
        NIMCP_LOGGING_WARN("Thalamus already initialized");
        return true;
    }

    NIMCP_LOGGING_INFO("Initializing thalamus subsystem...");

    /* Get brain-appropriate configuration */
    thalamus_config_t config;
    nimcp_brain_thal_default_config(brain, &config);

    /* Create thalamus */
    b->thalamus = thalamus_create(&config);
    if (!b->thalamus) {
        NIMCP_LOGGING_ERROR("Failed to create thalamus");
        return false;
    }

    b->thalamus_enabled = true;
    b->last_thalamus_update_us = 0;

    /* Set initial arousal from medulla if available */
    if (b->medulla_enabled && b->medulla) {
        float arousal = medulla_get_arousal_level(b->medulla);
        thalamus_set_arousal(b->thalamus, arousal);
        NIMCP_LOGGING_DEBUG("Thalamus arousal set from medulla: %.2f", arousal);
    }

    NIMCP_LOGGING_INFO("Thalamus initialized successfully with %d nuclei",
                        THAL_NUCLEUS_COUNT);

    /* Log enabled nuclei */
    NIMCP_LOGGING_DEBUG("Thalamus nuclei enabled:");
    NIMCP_LOGGING_DEBUG("  - LGN (visual relay)");
    NIMCP_LOGGING_DEBUG("  - MGN (auditory relay)");
    NIMCP_LOGGING_DEBUG("  - VPL/VPM (somatosensory relay)");
    NIMCP_LOGGING_DEBUG("  - VA/VL (motor relay)");
    NIMCP_LOGGING_DEBUG("  - Pulvinar (attention)");
    NIMCP_LOGGING_DEBUG("  - MD (executive)");
    NIMCP_LOGGING_DEBUG("  - Anterior (limbic)");
    NIMCP_LOGGING_DEBUG("  - TRN (gating)");

    return true;
}

void nimcp_brain_thal_destroy(brain_t brain) {
    if (!brain) return;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (b->thalamus) {
        thalamus_destroy(b->thalamus);
        b->thalamus = NULL;
        b->thalamus_enabled = false;
        NIMCP_LOGGING_DEBUG("Thalamus destroyed");
    }
}

//=============================================================================
// Processing
//=============================================================================

int nimcp_brain_thal_step(brain_t brain, float dt_ms) {
    if (!brain) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return 0;  /* Not enabled, skip silently */
    }

    /* Update thalamus */
    int result = thalamus_step(b->thalamus, dt_ms);

    if (result == 0) {
        b->last_thalamus_update_us = b->current_time_us;
    }

    return result;
}

//=============================================================================
// Relay Functions
//=============================================================================

int nimcp_brain_thal_relay_visual(brain_t brain,
                                   const float* retinal_input,
                                   uint32_t input_size,
                                   float* v1_output,
                                   uint32_t output_size) {
    if (!brain || !retinal_input || !v1_output) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        /* No thalamus - pass through directly */
        uint32_t copy_size = (input_size < output_size) ? input_size : output_size;
        memcpy(v1_output, retinal_input, copy_size * sizeof(float));
        return 0;
    }

    return thalamus_relay_visual(b->thalamus, retinal_input, input_size,
                                  v1_output, output_size);
}

int nimcp_brain_thal_relay_auditory(brain_t brain,
                                     const float* ic_input,
                                     uint32_t input_size,
                                     float* a1_output,
                                     uint32_t output_size) {
    if (!brain || !ic_input || !a1_output) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        /* No thalamus - pass through directly */
        uint32_t copy_size = (input_size < output_size) ? input_size : output_size;
        memcpy(a1_output, ic_input, copy_size * sizeof(float));
        return 0;
    }

    return thalamus_relay_auditory(b->thalamus, ic_input, input_size,
                                    a1_output, output_size);
}

int nimcp_brain_thal_relay_motor(brain_t brain,
                                  const float* bg_input,
                                  uint32_t bg_size,
                                  float* motor_output,
                                  uint32_t output_size) {
    if (!brain || !bg_input || !motor_output) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        /* No thalamus - pass through directly */
        uint32_t copy_size = (bg_size < output_size) ? bg_size : output_size;
        memcpy(motor_output, bg_input, copy_size * sizeof(float));
        return 0;
    }

    return thalamus_relay_motor(b->thalamus, bg_input, bg_size,
                                 motor_output, output_size);
}

int nimcp_brain_thal_relay_executive(brain_t brain,
                                      const float* input,
                                      uint32_t input_size,
                                      float* pfc_output,
                                      uint32_t output_size) {
    if (!brain || !input || !pfc_output) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        /* No thalamus - pass through directly */
        uint32_t copy_size = (input_size < output_size) ? input_size : output_size;
        memcpy(pfc_output, input, copy_size * sizeof(float));
        return 0;
    }

    return thalamus_relay_executive(b->thalamus, input, input_size,
                                     pfc_output, output_size);
}

int nimcp_brain_thal_relay(brain_t brain,
                            thal_nucleus_type_t nucleus_type,
                            const float* input,
                            uint32_t input_size,
                            float* output,
                            uint32_t output_size) {
    if (!brain || !input || !output) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        /* No thalamus - pass through directly */
        uint32_t copy_size = (input_size < output_size) ? input_size : output_size;
        memcpy(output, input, copy_size * sizeof(float));
        return 0;
    }

    return thalamus_relay(b->thalamus, nucleus_type, input, input_size,
                           output, output_size);
}

//=============================================================================
// Attention and Gating
//=============================================================================

int nimcp_brain_thal_set_attention(brain_t brain, float attention) {
    if (!brain) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1;
    }

    /* Set attention for all nuclei */
    for (int i = 0; i < THAL_NUCLEUS_COUNT; i++) {
        thalamus_set_attention(b->thalamus, (thal_nucleus_type_t)i, attention);
    }

    return 0;
}

int nimcp_brain_thal_set_nucleus_attention(brain_t brain,
                                            thal_nucleus_type_t nucleus_type,
                                            float attention) {
    if (!brain) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1;
    }

    return thalamus_set_attention(b->thalamus, nucleus_type, attention);
}

int nimcp_brain_thal_set_arousal(brain_t brain, float arousal) {
    if (!brain) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1;
    }

    return thalamus_set_arousal(b->thalamus, arousal);
}

int nimcp_brain_thal_apply_trn_inhibition(brain_t brain,
                                           thal_nucleus_type_t nucleus_type,
                                           float inhibition) {
    if (!brain) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1;
    }

    return thalamus_apply_trn_inhibition(b->thalamus, nucleus_type, inhibition);
}

//=============================================================================
// Firing Mode Control
//=============================================================================

int nimcp_brain_thal_set_mode(brain_t brain,
                               thal_nucleus_type_t nucleus_type,
                               thal_firing_mode_t mode) {
    if (!brain) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1;
    }

    return thalamus_set_mode(b->thalamus, nucleus_type, mode);
}

thal_firing_mode_t nimcp_brain_thal_get_mode(brain_t brain,
                                              thal_nucleus_type_t nucleus_type) {
    if (!brain) return THAL_MODE_TONIC;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return THAL_MODE_TONIC;
    }

    return thalamus_get_mode(b->thalamus, nucleus_type);
}

int nimcp_brain_thal_trigger_burst(brain_t brain,
                                    thal_nucleus_type_t nucleus_type) {
    if (!brain) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1;
    }

    return thalamus_trigger_burst(b->thalamus, nucleus_type);
}

//=============================================================================
// Integration Callbacks
//=============================================================================

void nimcp_brain_thal_on_arousal_change(brain_t brain, float arousal_level) {
    if (!brain) return;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) return;

    /* Update thalamus arousal */
    thalamus_set_arousal(b->thalamus, arousal_level);

    /* Switch firing modes based on arousal */
    if (arousal_level < 0.3f) {
        /* Low arousal -> burst mode (drowsy/sleep) */
        for (int i = 0; i < THAL_NUCLEUS_COUNT; i++) {
            if (i != THAL_NUCLEUS_TRN) {  /* TRN doesn't switch modes */
                thalamus_set_mode(b->thalamus, (thal_nucleus_type_t)i, THAL_MODE_BURST);
            }
        }
    } else if (arousal_level > 0.7f) {
        /* High arousal -> tonic mode (alert) */
        for (int i = 0; i < THAL_NUCLEUS_COUNT; i++) {
            if (i != THAL_NUCLEUS_TRN) {
                thalamus_set_mode(b->thalamus, (thal_nucleus_type_t)i, THAL_MODE_TONIC);
            }
        }
    }
}

void nimcp_brain_thal_on_attention_change(brain_t brain,
                                           uint32_t channel_id,
                                           float attention_weight) {
    if (!brain) return;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) return;

    /* Update TRN inhibition based on attention */
    /* Lower attention -> higher TRN inhibition (gate closed) */
    float inhibition = 1.0f - attention_weight;

    /* Apply to relevant nuclei based on channel */
    /* Simplified: distribute across sensory nuclei */
    thalamus_apply_channel_inhibition(b->thalamus, THAL_NUCLEUS_LGN,
                                       channel_id % 64, inhibition);
    thalamus_apply_channel_inhibition(b->thalamus, THAL_NUCLEUS_MGN,
                                       channel_id % 32, inhibition);
}

void nimcp_brain_thal_on_sleep_wake_change(brain_t brain, bool is_awake) {
    if (!brain) return;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) return;

    /* Update firing modes for all nuclei */
    thal_firing_mode_t mode = is_awake ? THAL_MODE_TONIC : THAL_MODE_BURST;

    for (int i = 0; i < THAL_NUCLEUS_COUNT; i++) {
        if (i != THAL_NUCLEUS_TRN) {
            thalamus_set_mode(b->thalamus, (thal_nucleus_type_t)i, mode);
        }
    }

    /* Update arousal to match */
    float arousal = is_awake ? 0.8f : 0.2f;
    thalamus_set_arousal(b->thalamus, arousal);
}

//=============================================================================
// Query Functions
//=============================================================================

int nimcp_brain_thal_get_stats(brain_t brain, thalamus_stats_t* stats) {
    if (!brain || !stats) return -1;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        memset(stats, 0, sizeof(*stats));
        return -1;
    }

    return thalamus_get_stats(b->thalamus, stats);
}

bool nimcp_brain_thal_is_enabled(brain_t brain) {
    if (!brain) return false;
    struct brain_struct* b = (struct brain_struct*)brain;
    return b->thalamus_enabled && b->thalamus != NULL;
}

float nimcp_brain_thal_get_attention(brain_t brain) {
    if (!brain) return -1.0f;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1.0f;
    }

    return b->thalamus->global_attention;
}

float nimcp_brain_thal_get_arousal(brain_t brain) {
    if (!brain) return -1.0f;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1.0f;
    }

    return b->thalamus->global_arousal;
}

float nimcp_brain_thal_get_firing_rate(brain_t brain,
                                        thal_nucleus_type_t nucleus_type) {
    if (!brain) return -1.0f;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1.0f;
    }

    return thalamus_get_firing_rate(b->thalamus, nucleus_type);
}

float nimcp_brain_thal_get_tonic_fraction(brain_t brain,
                                           thal_nucleus_type_t nucleus_type) {
    if (!brain) return -1.0f;

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return -1.0f;
    }

    return thalamus_get_tonic_fraction(b->thalamus, nucleus_type);
}

//=============================================================================
// Direct Access
//=============================================================================

thalamus_t* nimcp_brain_thal_get_handle(brain_t brain) {
    if (!brain) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;

    }

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled) {
        return NULL;
    }

    return b->thalamus;
}

thal_nucleus_t* nimcp_brain_thal_get_nucleus(brain_t brain,
                                              thal_nucleus_type_t nucleus_type) {
    if (!brain) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;

    }

    struct brain_struct* b = (struct brain_struct*)brain;

    if (!b->thalamus_enabled || !b->thalamus) {
        return NULL;
    }

    return thalamus_get_nucleus(b->thalamus, nucleus_type);
}
