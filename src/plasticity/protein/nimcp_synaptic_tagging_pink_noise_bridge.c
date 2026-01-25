//=============================================================================
// nimcp_synaptic_tagging_pink_noise_bridge.c - Pink Noise for Synaptic Tagging
//=============================================================================

#include "plasticity/protein/nimcp_synaptic_tagging_pink_noise_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

static float clamp(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

tag_pink_noise_bridge_t* tag_pink_noise_create(const tag_pink_noise_config_t* config) {
    tag_pink_noise_bridge_t* bridge = nimcp_calloc(1, sizeof(tag_pink_noise_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "tag_pink_noise_create: failed to allocate bridge");
        return NULL;
    }

    bridge->config = config ? *config : tag_pink_noise_default_config();

    pink_noise_config_t nc;
    pink_noise_default_config(&nc);
    nc.alpha = bridge->config.noise_alpha;
    nc.amplitude = bridge->config.noise_amplitude;
    nc.seed = bridge->config.seed;

    bridge->noise_gen = pink_noise_create(&nc);
    if (!bridge->noise_gen) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "tag_pink_noise_create: failed to create noise generator");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->noise_connected = true;
    bridge->is_enabled = bridge->config.enabled;
    bridge->noisy_tag_decay = 0.1f;
    bridge->noisy_capture_prob = 0.5f;
    bridge->noisy_prp_level = 1.0f;

    NIMCP_LOGGING_INFO("Created synaptic tagging pink noise bridge");
    return bridge;
}

void tag_pink_noise_destroy(tag_pink_noise_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->noise_gen) pink_noise_destroy(bridge->noise_gen);
    nimcp_free(bridge);
}

int tag_pink_noise_connect(tag_pink_noise_bridge_t* bridge, void* state) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "tag_pink_noise_connect: bridge is NULL");
        return -1;
    }
    bridge->tagging_state = state;
    bridge->tag_connected = (state != NULL);
    return 0;
}

int tag_pink_noise_update(tag_pink_noise_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "tag_pink_noise_update: bridge is NULL");
        return -1;
    }
    if (!bridge->is_enabled) {
        return -1;  /* Not an error, just disabled */
    }
    if (!bridge->noise_gen) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "tag_pink_noise_update: noise_gen is NULL");
        return -1;
    }

    float amp = bridge->config.noise_amplitude;

    bridge->tag_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.tag_decay_noise_scale;
    bridge->capture_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.capture_prob_noise_scale;
    bridge->prp_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.prp_noise_scale;

    bridge->noisy_tag_decay = 0.1f * (1.0f + amp * bridge->tag_noise);
    bridge->noisy_capture_prob = 0.5f * (1.0f + amp * bridge->capture_noise);
    bridge->noisy_prp_level = 1.0f * (1.0f + amp * bridge->prp_noise);

    bridge->noisy_tag_decay = clamp(bridge->noisy_tag_decay, 0.01f, 0.5f);
    bridge->noisy_capture_prob = clamp(bridge->noisy_capture_prob, 0.1f, 1.0f);
    bridge->noisy_prp_level = clamp(bridge->noisy_prp_level, 0.1f, 2.0f);

    bridge->samples_generated++;
    return 0;
}

float tag_pink_noise_get_decay(const tag_pink_noise_bridge_t* bridge) {
    return bridge ? bridge->noisy_tag_decay : 0.1f;
}

float tag_pink_noise_get_capture_prob(const tag_pink_noise_bridge_t* bridge) {
    return bridge ? bridge->noisy_capture_prob : 0.5f;
}

int tag_pink_noise_reset(tag_pink_noise_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "tag_pink_noise_reset: bridge is NULL");
        return -1;
    }
    bridge->tag_noise = bridge->capture_noise = bridge->prp_noise = 0.0f;
    bridge->noisy_tag_decay = 0.1f;
    bridge->noisy_capture_prob = 0.5f;
    bridge->noisy_prp_level = 1.0f;
    bridge->samples_generated = 0;
    if (bridge->noise_gen) pink_noise_reset(bridge->noise_gen);
    return 0;
}
