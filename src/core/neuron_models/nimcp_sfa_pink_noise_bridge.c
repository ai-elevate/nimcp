//=============================================================================
// nimcp_sfa_pink_noise_bridge.c - Pink Noise for Spike Frequency Adaptation
//=============================================================================

#include "core/neuron_models/nimcp_sfa_pink_noise_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

static float clamp(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

sfa_pink_noise_bridge_t* sfa_pink_noise_create(const sfa_pink_noise_config_t* config) {
    sfa_pink_noise_bridge_t* bridge = nimcp_calloc(1, sizeof(sfa_pink_noise_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->config = config ? *config : sfa_pink_noise_default_config();

    pink_noise_config_t nc;
    pink_noise_default_config(&nc);
    nc.alpha = bridge->config.noise_alpha;
    nc.amplitude = bridge->config.noise_amplitude;
    nc.seed = bridge->config.seed;

    bridge->noise_gen = pink_noise_create(&nc);
    if (!bridge->noise_gen) { nimcp_free(bridge); return NULL; }

    bridge->noise_connected = true;
    bridge->is_enabled = bridge->config.enabled;
    bridge->noisy_tau = SFA_DEFAULT_TAU;
    bridge->noisy_strength = SFA_DEFAULT_STRENGTH;
    bridge->noisy_threshold = -50.0f;  // mV

    NIMCP_LOGGING_INFO("Created SFA pink noise bridge");
    return bridge;
}

void sfa_pink_noise_destroy(sfa_pink_noise_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->noise_gen) pink_noise_destroy(bridge->noise_gen);
    nimcp_free(bridge);
}

int sfa_pink_noise_connect(sfa_pink_noise_bridge_t* bridge, void* neuron_state) {
    if (!bridge) return -1;
    bridge->neuron_state = neuron_state;
    bridge->neuron_connected = (neuron_state != NULL);
    return 0;
}

int sfa_pink_noise_update(sfa_pink_noise_bridge_t* bridge) {
    if (!bridge || !bridge->is_enabled || !bridge->noise_gen) return -1;

    float amp = bridge->config.noise_amplitude;

    bridge->tau_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.tau_noise_scale;
    bridge->strength_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.strength_noise_scale;
    bridge->threshold_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.threshold_noise_scale;

    bridge->noisy_tau = SFA_DEFAULT_TAU * (1.0f + amp * bridge->tau_noise);
    bridge->noisy_strength = SFA_DEFAULT_STRENGTH * (1.0f + amp * bridge->strength_noise);
    bridge->noisy_threshold = -50.0f + amp * bridge->threshold_noise * 5.0f;  // ±5mV noise

    bridge->noisy_tau = clamp(bridge->noisy_tau, bridge->config.tau_min, bridge->config.tau_max);
    bridge->noisy_strength = clamp(bridge->noisy_strength, bridge->config.strength_min, bridge->config.strength_max);
    bridge->noisy_threshold = clamp(bridge->noisy_threshold, -60.0f, -40.0f);

    bridge->samples_generated++;
    return 0;
}

float sfa_pink_noise_get_tau(const sfa_pink_noise_bridge_t* bridge) {
    return bridge ? bridge->noisy_tau : SFA_DEFAULT_TAU;
}

float sfa_pink_noise_get_strength(const sfa_pink_noise_bridge_t* bridge) {
    return bridge ? bridge->noisy_strength : SFA_DEFAULT_STRENGTH;
}

int sfa_pink_noise_reset(sfa_pink_noise_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->tau_noise = bridge->strength_noise = bridge->threshold_noise = 0.0f;
    bridge->noisy_tau = SFA_DEFAULT_TAU;
    bridge->noisy_strength = SFA_DEFAULT_STRENGTH;
    bridge->noisy_threshold = -50.0f;
    bridge->samples_generated = 0;
    if (bridge->noise_gen) pink_noise_reset(bridge->noise_gen);
    return 0;
}
