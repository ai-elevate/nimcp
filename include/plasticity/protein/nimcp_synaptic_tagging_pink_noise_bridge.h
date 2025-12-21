//=============================================================================
// nimcp_synaptic_tagging_pink_noise_bridge.h - Pink Noise for Synaptic Tagging
//=============================================================================
/**
 * @file nimcp_synaptic_tagging_pink_noise_bridge.h
 * @brief Integrates 1/f pink noise with synaptic tagging and capture
 * @version 1.0.0
 * @date 2025-12-21
 *
 * WHAT: Applies 1/f noise to tag decay and capture probability
 * WHY:  Synaptic tags have noisy lifetimes (Frey & Morris 1997)
 * BIOLOGICAL: Tag persistence and PRPs follow 1/f fluctuations
 */

#ifndef NIMCP_SYNAPTIC_TAGGING_PINK_NOISE_BRIDGE_H
#define NIMCP_SYNAPTIC_TAGGING_PINK_NOISE_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "plasticity/protein/nimcp_protein_synthesis.h"
#include "plasticity/noise/nimcp_pink_noise.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TAG_PINK_NOISE_DEFAULT_ALPHA     1.0f
#define TAG_PINK_NOISE_DEFAULT_AMPLITUDE 0.05f

typedef struct {
    float noise_alpha;
    float noise_amplitude;
    float tag_decay_noise_scale;
    float capture_prob_noise_scale;
    float prp_noise_scale;
    bool enabled;
    uint32_t seed;
} tag_pink_noise_config_t;

typedef struct {
    tag_pink_noise_config_t config;
    void* tagging_state;
    pink_noise_generator_t noise_gen;
    float noisy_tag_decay;
    float noisy_capture_prob;
    float noisy_prp_level;
    float tag_noise;
    float capture_noise;
    float prp_noise;
    uint64_t samples_generated;
    bool is_enabled;
    bool tag_connected;
    bool noise_connected;
} tag_pink_noise_bridge_t;

static inline tag_pink_noise_config_t tag_pink_noise_default_config(void) {
    return (tag_pink_noise_config_t){
        .noise_alpha = TAG_PINK_NOISE_DEFAULT_ALPHA,
        .noise_amplitude = TAG_PINK_NOISE_DEFAULT_AMPLITUDE,
        .tag_decay_noise_scale = 1.0f,
        .capture_prob_noise_scale = 0.5f,
        .prp_noise_scale = 0.7f,
        .enabled = true,
        .seed = 44444
    };
}

tag_pink_noise_bridge_t* tag_pink_noise_create(const tag_pink_noise_config_t* config);
void tag_pink_noise_destroy(tag_pink_noise_bridge_t* bridge);
int tag_pink_noise_connect(tag_pink_noise_bridge_t* bridge, void* state);
int tag_pink_noise_update(tag_pink_noise_bridge_t* bridge);
float tag_pink_noise_get_decay(const tag_pink_noise_bridge_t* bridge);
float tag_pink_noise_get_capture_prob(const tag_pink_noise_bridge_t* bridge);
int tag_pink_noise_reset(tag_pink_noise_bridge_t* bridge);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_SYNAPTIC_TAGGING_PINK_NOISE_BRIDGE_H
