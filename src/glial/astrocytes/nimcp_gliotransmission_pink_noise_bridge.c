//=============================================================================
// nimcp_gliotransmission_pink_noise_bridge.c - Pink Noise for Gliotransmission
//=============================================================================

#include "glial/astrocytes/nimcp_gliotransmission_pink_noise_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

static float clamp(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

glio_pink_noise_bridge_t* glio_pink_noise_create(const glio_pink_noise_config_t* config) {
    glio_pink_noise_bridge_t* bridge = nimcp_calloc(1, sizeof(glio_pink_noise_bridge_t));
    if (!bridge) return NULL;

    bridge->config = config ? *config : glio_pink_noise_default_config();

    pink_noise_config_t nc;
    pink_noise_default_config(&nc);
    nc.alpha = bridge->config.noise_alpha;
    nc.amplitude = bridge->config.noise_amplitude;
    nc.seed = bridge->config.seed;

    bridge->noise_gen = pink_noise_create(&nc);
    if (!bridge->noise_gen) { nimcp_free(bridge); return NULL; }

    bridge->noise_connected = true;
    bridge->is_enabled = bridge->config.enabled;
    bridge->noisy_glutamate_release = 0.5f;
    bridge->noisy_atp_release = 0.3f;
    bridge->noisy_dserine_release = 0.2f;
    bridge->noisy_calcium_wave = 1.0f;

    NIMCP_LOGGING_INFO("Created gliotransmission pink noise bridge");
    return bridge;
}

void glio_pink_noise_destroy(glio_pink_noise_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->noise_gen) pink_noise_destroy(bridge->noise_gen);
    nimcp_free(bridge);
}

int glio_pink_noise_connect(glio_pink_noise_bridge_t* bridge, void* astro_state) {
    if (!bridge) return -1;
    bridge->astrocyte_state = astro_state;
    bridge->astro_connected = (astro_state != NULL);
    return 0;
}

int glio_pink_noise_update(glio_pink_noise_bridge_t* bridge) {
    if (!bridge || !bridge->is_enabled || !bridge->noise_gen) return -1;

    float amp = bridge->config.noise_amplitude;

    bridge->glu_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.glutamate_noise_scale;
    bridge->atp_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.atp_noise_scale;
    bridge->dser_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.dserine_noise_scale;
    bridge->ca_noise = pink_noise_sample(bridge->noise_gen) * bridge->config.calcium_noise_scale;

    bridge->noisy_glutamate_release = 0.5f * (1.0f + amp * bridge->glu_noise);
    bridge->noisy_atp_release = 0.3f * (1.0f + amp * bridge->atp_noise);
    bridge->noisy_dserine_release = 0.2f * (1.0f + amp * bridge->dser_noise);
    bridge->noisy_calcium_wave = 1.0f * (1.0f + amp * bridge->ca_noise);

    bridge->noisy_glutamate_release = clamp(bridge->noisy_glutamate_release, 0.0f, 1.0f);
    bridge->noisy_atp_release = clamp(bridge->noisy_atp_release, 0.0f, 1.0f);
    bridge->noisy_dserine_release = clamp(bridge->noisy_dserine_release, 0.0f, 1.0f);
    bridge->noisy_calcium_wave = clamp(bridge->noisy_calcium_wave, 0.1f, 2.0f);

    bridge->samples_generated++;
    return 0;
}

float glio_pink_noise_get_glutamate(const glio_pink_noise_bridge_t* bridge) {
    return bridge ? bridge->noisy_glutamate_release : 0.5f;
}

float glio_pink_noise_get_calcium_wave(const glio_pink_noise_bridge_t* bridge) {
    return bridge ? bridge->noisy_calcium_wave : 1.0f;
}

int glio_pink_noise_reset(glio_pink_noise_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->glu_noise = bridge->atp_noise = bridge->dser_noise = bridge->ca_noise = 0.0f;
    bridge->noisy_glutamate_release = 0.5f;
    bridge->noisy_atp_release = 0.3f;
    bridge->noisy_dserine_release = 0.2f;
    bridge->noisy_calcium_wave = 1.0f;
    bridge->samples_generated = 0;
    if (bridge->noise_gen) pink_noise_reset(bridge->noise_gen);
    return 0;
}
