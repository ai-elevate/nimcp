/**
 * @file nimcp_physics_intra_coordinator.c
 * @brief Physics Layer Intra-Layer Coordinator Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/intra/physics/nimcp_physics_intra_coordinator.h"
#include <string.h>
#include <stdlib.h>

typedef struct {
    void* module;
    nimcp_module_interface_t interface;
    bool connected;
} module_slot_t;

struct nimcp_physics_intra_struct {
    nimcp_physics_intra_config_t config;
    nimcp_layer_registry_t registry;
    module_slot_t ephaptic;
    module_slot_t info_geometry;
    module_slot_t hh_dynamics;
    module_slot_t thermodynamics;
    nimcp_physics_intra_state_t state;
    nimcp_physics_intra_stats_t stats;
    bool is_initialized;
};

nimcp_physics_intra_config_t nimcp_physics_intra_default_config(void) {
    nimcp_physics_intra_config_t config = {
        .enable_ephaptic = true,
        .enable_info_geometry = true,
        .enable_hh_dynamics = true,
        .enable_thermodynamics = true,
        .ephaptic_hh_coupling = 0.5f,
        .ephaptic_geometry_coupling = 0.3f,
        .hh_thermo_coupling = 0.4f,
        .geometry_thermo_coupling = 0.3f,
        .sync_interval_ms = 10,
        .coherence_threshold = 0.7f,
        .enforce_energy_conservation = true,
        .enforce_entropy_increase = true,
        .temperature_kelvin = 310.0f,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_physics_intra_t nimcp_physics_intra_create(const nimcp_physics_intra_config_t* config) {
    nimcp_physics_intra_t coord = (nimcp_physics_intra_t)calloc(1, sizeof(struct nimcp_physics_intra_struct));
    if (!coord) return NULL;
    coord->config = config ? *config : nimcp_physics_intra_default_config();
    coord->state.temperature = coord->config.temperature_kelvin;
    return coord;
}

void nimcp_physics_intra_destroy(nimcp_physics_intra_t coord) {
    if (!coord) return;
    if (coord->is_initialized) nimcp_physics_intra_shutdown(coord);
    free(coord);
}

nimcp_layer_error_t nimcp_physics_intra_init(nimcp_physics_intra_t coord, nimcp_layer_registry_t registry) {
    if (!coord || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (coord->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    coord->registry = registry;
    coord->is_initialized = true;
    coord->state.layer_coherence = 1.0f;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_shutdown(nimcp_physics_intra_t coord) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!coord->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    coord->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_connect_ephaptic(nimcp_physics_intra_t coord, void* module, nimcp_module_interface_t* interface) {
    if (!coord || !module || !interface) return NIMCP_LAYER_ERR_NULL_PTR;
    coord->ephaptic.module = module;
    coord->ephaptic.interface = *interface;
    coord->ephaptic.connected = true;
    coord->state.ephaptic_active = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_connect_info_geometry(nimcp_physics_intra_t coord, void* module, nimcp_module_interface_t* interface) {
    if (!coord || !module || !interface) return NIMCP_LAYER_ERR_NULL_PTR;
    coord->info_geometry.module = module;
    coord->info_geometry.interface = *interface;
    coord->info_geometry.connected = true;
    coord->state.info_geometry_active = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_connect_hh_dynamics(nimcp_physics_intra_t coord, void* module, nimcp_module_interface_t* interface) {
    if (!coord || !module || !interface) return NIMCP_LAYER_ERR_NULL_PTR;
    coord->hh_dynamics.module = module;
    coord->hh_dynamics.interface = *interface;
    coord->hh_dynamics.connected = true;
    coord->state.hh_dynamics_active = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_connect_thermodynamics(nimcp_physics_intra_t coord, void* module, nimcp_module_interface_t* interface) {
    if (!coord || !module || !interface) return NIMCP_LAYER_ERR_NULL_PTR;
    coord->thermodynamics.module = module;
    coord->thermodynamics.interface = *interface;
    coord->thermodynamics.connected = true;
    coord->state.thermodynamics_active = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_update(nimcp_physics_intra_t coord, float dt) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!coord->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* Accumulate energy/entropy stats */
    coord->stats.avg_energy = coord->stats.avg_energy * 0.99f + coord->state.total_energy * 0.01f;
    coord->stats.avg_entropy = coord->stats.avg_entropy * 0.99f + coord->state.entropy * 0.01f;
    coord->stats.avg_field_magnitude = coord->stats.avg_field_magnitude * 0.99f + coord->state.field_magnitude * 0.01f;

    /* Update coherences */
    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < PHYSICS_MODULE_COUNT; i++) {
        sum += coord->state.module_coherences[i];
        count++;
    }
    coord->state.layer_coherence = count > 0 ? sum / count : 1.0f;
    coord->stats.avg_coherence = coord->stats.avg_coherence * 0.99f + coord->state.layer_coherence * 0.01f;

    (void)dt;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_sync(nimcp_physics_intra_t coord) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!coord->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    coord->stats.sync_events++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_send(nimcp_physics_intra_t coord, uint32_t target_module, nimcp_layer_msg_t* msg) {
    if (!coord || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    (void)target_module;
    coord->stats.messages_sent++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_broadcast(nimcp_physics_intra_t coord, uint32_t source_module, const nimcp_layer_msg_t* msg) {
    if (!coord || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    (void)source_module;
    coord->stats.messages_sent += PHYSICS_MODULE_COUNT;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_update_energy(nimcp_physics_intra_t coord, float delta_energy) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;

    if (coord->config.enforce_energy_conservation) {
        /* In closed system, energy should be conserved - track violations */
        if (delta_energy != 0.0f) {
            coord->stats.constraint_violations++;
        }
    }

    coord->state.total_energy += delta_energy;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_update_entropy(nimcp_physics_intra_t coord, float delta_entropy) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;

    if (coord->config.enforce_entropy_increase && delta_entropy < 0.0f) {
        coord->stats.constraint_violations++;
    }

    coord->state.entropy += delta_entropy;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_check_constraints(nimcp_physics_intra_t coord, uint32_t* violations_out) {
    if (!coord || !violations_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *violations_out = (uint32_t)coord->stats.constraint_violations;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_get_state(nimcp_physics_intra_t coord, nimcp_physics_intra_state_t* state_out) {
    if (!coord || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = coord->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_physics_intra_get_stats(nimcp_physics_intra_t coord, nimcp_physics_intra_stats_t* stats_out) {
    if (!coord || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = coord->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_physics_intra_get_coherence(nimcp_physics_intra_t coord) {
    return coord ? coord->state.layer_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_physics_intra_reset_stats(nimcp_physics_intra_t coord) {
    if (!coord) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&coord->stats, 0, sizeof(coord->stats));
    return NIMCP_LAYER_OK;
}
