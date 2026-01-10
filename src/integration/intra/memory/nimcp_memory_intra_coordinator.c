/**
 * @file nimcp_memory_intra_coordinator.c
 * @brief Memory Layer Intra-Layer Coordinator Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/intra/memory/nimcp_memory_intra_coordinator.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    void* module;
    nimcp_module_interface_t interface;
    bool connected;
} module_slot_t;

struct nimcp_memory_intra_struct {
    nimcp_memory_intra_config_t config;
    nimcp_layer_registry_t registry;
    module_slot_t entorhinal;
    module_slot_t perirhinal;
    module_slot_t parahippocampal;
    module_slot_t mammillary;
    nimcp_memory_intra_state_t state;
    nimcp_memory_intra_stats_t stats;
    bool is_initialized;
};

nimcp_memory_intra_config_t nimcp_memory_intra_default_config(void) {
    nimcp_memory_intra_config_t config = {
        .enable_entorhinal = true,
        .enable_perirhinal = true,
        .enable_parahippocampal = true,
        .enable_mammillary = true,
        .ec_prc_coupling = 0.6f,
        .ec_phc_coupling = 0.6f,
        .ec_mb_coupling = 0.4f,
        .prc_phc_coupling = 0.5f,
        .prc_mb_coupling = 0.3f,
        .phc_mb_coupling = 0.4f,
        .sync_interval_ms = 10,
        .coherence_threshold = 0.7f,
        .enable_theta_gating = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_memory_intra_t nimcp_memory_intra_create(const nimcp_memory_intra_config_t* config) {
    nimcp_memory_intra_t coord = (nimcp_memory_intra_t)calloc(1, sizeof(struct nimcp_memory_intra_struct));
    if (!coord) return NULL;
    coord->config = config ? *config : nimcp_memory_intra_default_config();
    return coord;
}

void nimcp_memory_intra_destroy(nimcp_memory_intra_t coord) {
    if (!coord) return;
    if (coord->is_initialized) nimcp_memory_intra_shutdown(coord);
    free(coord);
}

nimcp_layer_error_t nimcp_memory_intra_init(nimcp_memory_intra_t coord, nimcp_layer_registry_t registry) {
    if (!coord || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (coord->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    coord->registry = registry;
    coord->is_initialized = true;
    coord->state.layer_coherence = 1.0f;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_shutdown(nimcp_memory_intra_t coord) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!coord->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    coord->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_connect_entorhinal(nimcp_memory_intra_t coord, void* module, nimcp_module_interface_t* interface) {
    if (!coord || !module || !interface) return NIMCP_LAYER_ERR_NULL_PTR;
    coord->entorhinal.module = module;
    coord->entorhinal.interface = *interface;
    coord->entorhinal.connected = true;
    coord->state.entorhinal_active = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_connect_perirhinal(nimcp_memory_intra_t coord, void* module, nimcp_module_interface_t* interface) {
    if (!coord || !module || !interface) return NIMCP_LAYER_ERR_NULL_PTR;
    coord->perirhinal.module = module;
    coord->perirhinal.interface = *interface;
    coord->perirhinal.connected = true;
    coord->state.perirhinal_active = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_connect_parahippocampal(nimcp_memory_intra_t coord, void* module, nimcp_module_interface_t* interface) {
    if (!coord || !module || !interface) return NIMCP_LAYER_ERR_NULL_PTR;
    coord->parahippocampal.module = module;
    coord->parahippocampal.interface = *interface;
    coord->parahippocampal.connected = true;
    coord->state.parahippocampal_active = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_connect_mammillary(nimcp_memory_intra_t coord, void* module, nimcp_module_interface_t* interface) {
    if (!coord || !module || !interface) return NIMCP_LAYER_ERR_NULL_PTR;
    coord->mammillary.module = module;
    coord->mammillary.interface = *interface;
    coord->mammillary.connected = true;
    coord->state.mammillary_active = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_update(nimcp_memory_intra_t coord, float dt) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!coord->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* Update theta oscillation (4-8 Hz) */
    if (coord->config.enable_theta_gating) {
        coord->state.theta_phase += dt * 6.0f * 2.0f * 3.14159f; /* ~6 Hz theta */
        if (coord->state.theta_phase > 2.0f * 3.14159f) {
            coord->state.theta_phase -= 2.0f * 3.14159f;
            coord->stats.theta_cycles++;
        }
    }

    /* Encoding vs retrieval depends on theta phase */
    float theta_sin = sinf(coord->state.theta_phase);
    coord->state.encoding_strength = (theta_sin > 0) ? theta_sin : 0.0f;
    coord->state.retrieval_strength = (theta_sin < 0) ? -theta_sin : 0.0f;

    /* Update stats */
    coord->stats.avg_encoding_strength = coord->stats.avg_encoding_strength * 0.99f + coord->state.encoding_strength * 0.01f;
    coord->stats.avg_coherence = coord->stats.avg_coherence * 0.99f + coord->state.layer_coherence * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_sync(nimcp_memory_intra_t coord) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_send(nimcp_memory_intra_t coord, uint32_t target_module, nimcp_layer_msg_t* msg) {
    if (!coord || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    (void)target_module;
    coord->stats.messages_sent++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_broadcast(nimcp_memory_intra_t coord, uint32_t source_module, const nimcp_layer_msg_t* msg) {
    if (!coord || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    (void)source_module;
    coord->stats.messages_sent += MEMORY_MODULE_COUNT;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_get_state(nimcp_memory_intra_t coord, nimcp_memory_intra_state_t* state_out) {
    if (!coord || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = coord->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_memory_intra_get_stats(nimcp_memory_intra_t coord, nimcp_memory_intra_stats_t* stats_out) {
    if (!coord || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = coord->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_memory_intra_get_coherence(nimcp_memory_intra_t coord) {
    return coord ? coord->state.layer_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_memory_intra_reset_stats(nimcp_memory_intra_t coord) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&coord->stats, 0, sizeof(coord->stats));
    return NIMCP_LAYER_OK;
}
