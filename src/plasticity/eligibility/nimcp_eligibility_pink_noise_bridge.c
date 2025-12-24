//=============================================================================
// nimcp_eligibility_pink_noise_bridge.c - Pink Noise for Eligibility Traces
//=============================================================================

#include "plasticity/eligibility/nimcp_eligibility_pink_noise_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

static float clamp(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

elig_pink_noise_bridge_t* elig_pink_noise_create(const elig_pink_noise_config_t* config) {
    elig_pink_noise_bridge_t* bridge = nimcp_calloc(1, sizeof(elig_pink_noise_bridge_t));
    if (!bridge) return NULL;

    bridge->config = config ? *config : elig_pink_noise_default_config();

    pink_noise_config_t nc;
    pink_noise_default_config(&nc);
    nc.alpha = bridge->config.noise_alpha;
    nc.amplitude = bridge->config.noise_amplitude;
    nc.seed = bridge->config.seed;

    bridge->noise_gen = pink_noise_create(&nc);
    if (!bridge->noise_gen) { nimcp_free(bridge); return NULL; }

    bridge->noise_connected = true;
    bridge->is_enabled = bridge->config.enabled;
    bridge->noisy_decay_rate = 0.1f;
    bridge->noisy_threshold = 0.5f;
    bridge->noisy_boost = 1.0f;

    NIMCP_LOGGING_INFO("Created eligibility pink noise bridge");
    return bridge;
}

void elig_pink_noise_destroy(elig_pink_noise_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->noise_gen) pink_noise_destroy(bridge->noise_gen);
    nimcp_free(bridge);
}

int elig_pink_noise_connect(elig_pink_noise_bridge_t* bridge, void* elig_state) {
    if (!bridge) return -1;
    bridge->eligibility_state = elig_state;
    bridge->elig_connected = (elig_state != NULL);
    return 0;
}

int elig_pink_noise_update(elig_pink_noise_bridge_t* bridge) {
    if (!bridge || !bridge->is_enabled || !bridge->noise_gen) return -1;

    float amp = bridge->config.noise_amplitude;

    if (bridge->config.noise_targets & ELIG_NOISE_TARGET_DECAY) {
        bridge->decay_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.decay_noise_scale;
        bridge->noisy_decay_rate = 0.1f * (1.0f + amp * bridge->decay_noise);
        bridge->noisy_decay_rate = clamp(bridge->noisy_decay_rate, bridge->config.decay_min, bridge->config.decay_max);
    }
    if (bridge->config.noise_targets & ELIG_NOISE_TARGET_THRESHOLD) {
        bridge->threshold_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.threshold_noise_scale;
        bridge->noisy_threshold = 0.5f * (1.0f + amp * bridge->threshold_noise);
        bridge->noisy_threshold = clamp(bridge->noisy_threshold, 0.1f, 1.0f);
    }
    if (bridge->config.noise_targets & ELIG_NOISE_TARGET_BOOST) {
        bridge->boost_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.boost_noise_scale;
        bridge->noisy_boost = 1.0f * (1.0f + amp * bridge->boost_noise);
        bridge->noisy_boost = clamp(bridge->noisy_boost, 0.5f, 2.0f);
    }

    bridge->samples_generated++;
    bridge->avg_noise_amplitude = bridge->avg_noise_amplitude * 0.99f + fabsf(bridge->decay_noise) * amp * 0.01f;
    return 0;
}

float elig_pink_noise_get_noisy_decay(const elig_pink_noise_bridge_t* bridge) {
    return bridge ? bridge->noisy_decay_rate : 0.1f;
}

int elig_pink_noise_reset(elig_pink_noise_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->decay_noise = bridge->threshold_noise = bridge->boost_noise = 0.0f;
    bridge->noisy_decay_rate = 0.1f;
    bridge->noisy_threshold = 0.5f;
    bridge->noisy_boost = 1.0f;
    bridge->samples_generated = 0;
    bridge->avg_noise_amplitude = 0.0f;
    if (bridge->noise_gen) pink_noise_reset(bridge->noise_gen);
    return 0;
}
