/**
 * @file nimcp_glial_substrate_bridge.c
 * @brief Neural Substrate-Glial System Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 */

#include "glial/nimcp_glial_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/validation/nimcp_common.h"
#include "api/nimcp_api_exception.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Compute Q10 temperature factor
 *
 * WHAT: Calculate temperature effect on reaction rates
 * WHY:  Biological processes have Q10 = 2-3
 * HOW:  Q10_factor = Q10^((T - T0) / 10)
 *
 * @param current_temp Current temperature (°C)
 * @param base_temp Base temperature (°C, typically 37)
 * @param q10 Q10 coefficient
 * @return Temperature factor
 */
static float compute_q10_factor(float current_temp, float base_temp, float q10) {
    float delta_temp = current_temp - base_temp;
    return powf(q10, delta_temp / 10.0f);
}

/**
 * @brief Clamp value to range [min, max]
 */
static inline float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int glial_substrate_default_config(glial_substrate_config_t* config) {
    if (!config) {
        LOG_ERROR("glial_substrate_default_config: NULL config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "glial_substrate_default_config: NULL config");
        return -1;
    }

    config->enable_astrocyte_substrate = true;
    config->enable_oligo_substrate = true;
    config->enable_microglia_substrate = true;
    config->enable_myelin_substrate = true;
    config->enable_lactate_shuttle = true;
    config->enable_bio_async = false;

    config->atp_sensitivity = 1.0f;
    config->temperature_sensitivity = 1.0f;
    config->oxygen_sensitivity = 1.0f;

    config->lactate_efficiency = 1.0f;
    config->myelin_savings_factor = 1.0f;
    config->pruning_savings_factor = 1.0f;

    return 0;
}

glial_substrate_bridge_t* glial_substrate_bridge_create(
    const glial_substrate_config_t* config,
    neural_substrate_t* substrate,
    astrocyte_network_t* astro_network,
    oligodendrocyte_network_t* oligo_network,
    microglia_network_t* micro_network,
    myelin_sheath_network_t* myelin_network
) {
    NIMCP_API_CHECK_NULL_RET_NULL(substrate, "glial_substrate_bridge_create: NULL substrate");

    glial_substrate_bridge_t* bridge = (glial_substrate_bridge_t*)nimcp_malloc(
        sizeof(glial_substrate_bridge_t)
    );
    NIMCP_API_CHECK_ALLOC(bridge, "glial_substrate_bridge_create: allocation failed");

    memset(bridge, 0, sizeof(glial_substrate_bridge_t));

    /* Set configuration */
    if (config) {
        bridge->config = *config;
    } else {
        glial_substrate_default_config(&bridge->config);
    }

    /* Connect systems */
    bridge->substrate = substrate;
    bridge->astrocyte_network = astro_network;
    bridge->oligo_network = oligo_network;
    bridge->microglia_network = micro_network;
    bridge->myelin_network = myelin_network;

    /* Create mutex using platform API (allocates + initializes internally) */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        LOG_ERROR("glial_substrate_bridge_create: mutex creation failed");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "glial_substrate_bridge_create: mutex creation failed");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize effects to neutral state */
    bridge->astro_effects.atp_modulation = 1.0f;
    bridge->astro_effects.calcium_wave_factor = 1.0f;
    bridge->astro_effects.glutamate_release_factor = 1.0f;
    bridge->astro_effects.temp_q10_factor = 1.0f;
    bridge->astro_effects.metabolic_rate_factor = 1.0f;
    bridge->astro_effects.o2_modulation = 1.0f;

    bridge->oligo_effects.atp_modulation = 1.0f;
    bridge->oligo_effects.myelin_production_rate = 1.0f;
    bridge->oligo_effects.g_ratio_optimization = 1.0f;
    bridge->oligo_effects.temp_q10_factor = 1.0f;
    bridge->oligo_effects.maturation_rate = 1.0f;
    bridge->oligo_effects.o2_modulation = 1.0f;

    bridge->micro_effects.atp_modulation = 1.0f;
    bridge->micro_effects.surveillance_radius = 1.0f;
    bridge->micro_effects.pruning_threshold = 1.0f;
    bridge->micro_effects.temp_q10_factor = 1.0f;
    bridge->micro_effects.activation_rate = 1.0f;
    bridge->micro_effects.o2_modulation = 1.0f;

    bridge->myelin_effects.atp_maintenance_cost = SUBSTRATE_ATP_MYELIN_MAINTENANCE;
    bridge->myelin_effects.insufficient_atp = false;
    bridge->myelin_effects.integrity_decay_rate = 0.0f;
    bridge->myelin_effects.hyperthermia_damage = false;
    bridge->myelin_effects.temp_damage_rate = 0.0f;
    bridge->myelin_effects.conduction_block_prob = 0.0f;
    bridge->myelin_effects.hypoxia_damage = false;
    bridge->myelin_effects.o2_damage_rate = 0.0f;

    NIMCP_LOGGING_INFO("Created glial-substrate bridge");

    return bridge;
}

void glial_substrate_bridge_destroy(glial_substrate_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->base.bio_async_enabled) {
        glial_substrate_disconnect_bio_async(bridge);
    }

    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed glial-substrate bridge");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int glial_substrate_connect_astrocytes(
    glial_substrate_bridge_t* bridge,
    astrocyte_network_t* astro_network
) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->astrocyte_network = astro_network;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    if (astro_network) {
        NIMCP_LOGGING_INFO("Connected astrocyte network to substrate bridge");
    }

    return 0;
}

int glial_substrate_connect_oligodendrocytes(
    glial_substrate_bridge_t* bridge,
    oligodendrocyte_network_t* oligo_network
) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->oligo_network = oligo_network;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    if (oligo_network) {
        NIMCP_LOGGING_INFO("Connected oligodendrocyte network to substrate bridge");
    }

    return 0;
}

int glial_substrate_connect_microglia(
    glial_substrate_bridge_t* bridge,
    microglia_network_t* micro_network
) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->microglia_network = micro_network;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    if (micro_network) {
        NIMCP_LOGGING_INFO("Connected microglia network to substrate bridge");
    }

    return 0;
}

int glial_substrate_connect_myelin(
    glial_substrate_bridge_t* bridge,
    myelin_sheath_network_t* myelin_network
) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->myelin_network = myelin_network;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    if (myelin_network) {
        NIMCP_LOGGING_INFO("Connected myelin sheath network to substrate bridge");
    }

    return 0;
}

/* ============================================================================
 * Bio-async Integration API
 * ============================================================================ */

int glial_substrate_connect_bio_async(glial_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_GLIAL_SUBSTRATE,
        .module_name = "glial_substrate_bridge",
        .inbox_capacity = NIMCP_INBOX_CAPACITY_SMALL,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected glial-substrate bridge to bio-async router");
        return 0;
    }

    NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    return -1;
}

int glial_substrate_disconnect_bio_async(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected glial-substrate bridge from bio-async");

    return 0;
}

bool glial_substrate_is_bio_async_connected(const glial_substrate_bridge_t* bridge) {
    return bridge && bridge->base.bio_async_enabled;
}

/* ============================================================================
 * Substrate → Glial API
 * ============================================================================ */

int glial_substrate_update_astrocyte_effects(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    if (!bridge->config.enable_astrocyte_substrate) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get substrate state */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    /* ATP effects on astrocyte function */
    if (metabolic.atp_level < SUBSTRATE_ATP_ASTRO_THRESHOLD) {
        float deficit = (SUBSTRATE_ATP_ASTRO_THRESHOLD - metabolic.atp_level) /
                        SUBSTRATE_ATP_ASTRO_THRESHOLD;
        bridge->astro_effects.atp_modulation = 1.0f - (deficit * SUBSTRATE_ATP_ASTRO_CALCIUM_FACTOR);
    } else {
        bridge->astro_effects.atp_modulation = 1.0f;
    }
    bridge->astro_effects.atp_modulation *= bridge->config.atp_sensitivity;
    bridge->astro_effects.atp_modulation = clamp(bridge->astro_effects.atp_modulation, 0.0f, 1.0f);

    /* Calcium wave propagation affected by ATP */
    bridge->astro_effects.calcium_wave_factor = bridge->astro_effects.atp_modulation;
    bridge->astro_effects.glutamate_release_factor = bridge->astro_effects.atp_modulation;

    /* Temperature effects (Q10) */
    bridge->astro_effects.temp_q10_factor = compute_q10_factor(
        physical.temperature,
        SUBSTRATE_NORMAL_TEMPERATURE,
        SUBSTRATE_TEMP_ASTRO_Q10
    );
    bridge->astro_effects.temp_q10_factor *= bridge->config.temperature_sensitivity;
    bridge->astro_effects.metabolic_rate_factor = bridge->astro_effects.temp_q10_factor;

    /* Oxygen effects */
    if (metabolic.oxygen_saturation < SUBSTRATE_O2_ASTRO_THRESHOLD) {
        bridge->astro_effects.hypoxia_stress = true;
        float o2_deficit = (SUBSTRATE_O2_ASTRO_THRESHOLD - metabolic.oxygen_saturation) /
                           SUBSTRATE_O2_ASTRO_THRESHOLD;
        bridge->astro_effects.o2_modulation = 1.0f - o2_deficit;
    } else {
        bridge->astro_effects.hypoxia_stress = false;
        bridge->astro_effects.o2_modulation = 1.0f;
    }
    bridge->astro_effects.o2_modulation *= bridge->config.oxygen_sensitivity;
    bridge->astro_effects.o2_modulation = clamp(bridge->astro_effects.o2_modulation, 0.0f, 1.0f);

    bridge->stats.astrocyte_modulations++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int glial_substrate_update_oligodendrocyte_effects(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    if (!bridge->config.enable_oligo_substrate) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get substrate state */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    /* ATP effects on myelin synthesis (very ATP-expensive) */
    if (metabolic.atp_level < SUBSTRATE_ATP_OLIGO_THRESHOLD) {
        float deficit = (SUBSTRATE_ATP_OLIGO_THRESHOLD - metabolic.atp_level) /
                        SUBSTRATE_ATP_OLIGO_THRESHOLD;
        bridge->oligo_effects.atp_modulation = 1.0f - (deficit * SUBSTRATE_ATP_OLIGO_MYELIN_FACTOR);
    } else {
        bridge->oligo_effects.atp_modulation = 1.0f;
    }
    bridge->oligo_effects.atp_modulation *= bridge->config.atp_sensitivity;
    bridge->oligo_effects.atp_modulation = clamp(bridge->oligo_effects.atp_modulation, 0.0f, 1.0f);

    /* Myelin production and g-ratio optimization affected by ATP */
    bridge->oligo_effects.myelin_production_rate = bridge->oligo_effects.atp_modulation;
    bridge->oligo_effects.g_ratio_optimization = bridge->oligo_effects.atp_modulation;

    /* Temperature effects (Q10) */
    bridge->oligo_effects.temp_q10_factor = compute_q10_factor(
        physical.temperature,
        SUBSTRATE_NORMAL_TEMPERATURE,
        SUBSTRATE_TEMP_OLIGO_Q10
    );
    bridge->oligo_effects.temp_q10_factor *= bridge->config.temperature_sensitivity;
    bridge->oligo_effects.maturation_rate = bridge->oligo_effects.temp_q10_factor;

    /* Oxygen effects */
    if (metabolic.oxygen_saturation < SUBSTRATE_O2_OLIGO_THRESHOLD) {
        bridge->oligo_effects.hypoxia_stress = true;
        float o2_deficit = (SUBSTRATE_O2_OLIGO_THRESHOLD - metabolic.oxygen_saturation) /
                           SUBSTRATE_O2_OLIGO_THRESHOLD;
        bridge->oligo_effects.o2_modulation = 1.0f - o2_deficit;
    } else {
        bridge->oligo_effects.hypoxia_stress = false;
        bridge->oligo_effects.o2_modulation = 1.0f;
    }
    bridge->oligo_effects.o2_modulation *= bridge->config.oxygen_sensitivity;
    bridge->oligo_effects.o2_modulation = clamp(bridge->oligo_effects.o2_modulation, 0.0f, 1.0f);

    bridge->stats.oligo_modulations++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int glial_substrate_update_microglia_effects(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    if (!bridge->config.enable_microglia_substrate) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get substrate state */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    /* ATP effects on surveillance */
    if (metabolic.atp_level < SUBSTRATE_ATP_MICRO_THRESHOLD) {
        float deficit = (SUBSTRATE_ATP_MICRO_THRESHOLD - metabolic.atp_level) /
                        SUBSTRATE_ATP_MICRO_THRESHOLD;
        bridge->micro_effects.atp_modulation = 1.0f - (deficit * SUBSTRATE_ATP_MICRO_SURVEILLANCE_FACTOR);
    } else {
        bridge->micro_effects.atp_modulation = 1.0f;
    }
    bridge->micro_effects.atp_modulation *= bridge->config.atp_sensitivity;
    bridge->micro_effects.atp_modulation = clamp(bridge->micro_effects.atp_modulation, 0.0f, 1.0f);

    /* Surveillance radius and pruning affected by ATP */
    bridge->micro_effects.surveillance_radius = bridge->micro_effects.atp_modulation;
    bridge->micro_effects.pruning_threshold = 1.0f / bridge->micro_effects.atp_modulation;

    /* Temperature effects (Q10) */
    bridge->micro_effects.temp_q10_factor = compute_q10_factor(
        physical.temperature,
        SUBSTRATE_NORMAL_TEMPERATURE,
        SUBSTRATE_TEMP_MICRO_Q10
    );
    bridge->micro_effects.temp_q10_factor *= bridge->config.temperature_sensitivity;
    bridge->micro_effects.activation_rate = bridge->micro_effects.temp_q10_factor;

    /* Oxygen effects - hypoxia activates microglia */
    if (metabolic.oxygen_saturation < SUBSTRATE_O2_MICRO_ACTIVATION) {
        bridge->micro_effects.hypoxia_activation = true;
        bridge->micro_effects.o2_modulation = 1.5f; /* Hypoxia increases activity */
    } else {
        bridge->micro_effects.hypoxia_activation = false;
        bridge->micro_effects.o2_modulation = 1.0f;
    }
    bridge->micro_effects.o2_modulation *= bridge->config.oxygen_sensitivity;

    bridge->stats.microglia_modulations++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int glial_substrate_update_myelin_effects(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    if (!bridge->config.enable_myelin_substrate) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get substrate state */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    /* ATP maintenance cost */
    bridge->myelin_effects.atp_maintenance_cost = SUBSTRATE_ATP_MYELIN_MAINTENANCE *
                                                   bridge->config.atp_sensitivity;

    /* Insufficient ATP → integrity decay */
    if (metabolic.atp_level < 0.5f) {
        bridge->myelin_effects.insufficient_atp = true;
        float deficit = (0.5f - metabolic.atp_level) / 0.5f;
        bridge->myelin_effects.integrity_decay_rate = deficit * 0.01f;
    } else {
        bridge->myelin_effects.insufficient_atp = false;
        bridge->myelin_effects.integrity_decay_rate = 0.0f;
    }

    /* Hyperthermia damage */
    if (physical.temperature > SUBSTRATE_HYPERTHERMIA_THRESHOLD) {
        bridge->myelin_effects.hyperthermia_damage = true;
        float temp_excess = physical.temperature - SUBSTRATE_HYPERTHERMIA_THRESHOLD;
        bridge->myelin_effects.temp_damage_rate = temp_excess * SUBSTRATE_TEMP_MYELIN_INTEGRITY_LOSS;
        bridge->myelin_effects.conduction_block_prob = temp_excess * 0.05f;
    } else {
        bridge->myelin_effects.hyperthermia_damage = false;
        bridge->myelin_effects.temp_damage_rate = 0.0f;
        bridge->myelin_effects.conduction_block_prob = 0.0f;
    }
    bridge->myelin_effects.temp_damage_rate *= bridge->config.temperature_sensitivity;

    /* Hypoxia damage */
    if (metabolic.oxygen_saturation < SUBSTRATE_CRITICAL_O2) {
        bridge->myelin_effects.hypoxia_damage = true;
        float o2_deficit = (SUBSTRATE_CRITICAL_O2 - metabolic.oxygen_saturation) /
                           SUBSTRATE_CRITICAL_O2;
        bridge->myelin_effects.o2_damage_rate = o2_deficit * SUBSTRATE_O2_MYELIN_DAMAGE_RATE;
    } else {
        bridge->myelin_effects.hypoxia_damage = false;
        bridge->myelin_effects.o2_damage_rate = 0.0f;
    }
    bridge->myelin_effects.o2_damage_rate *= bridge->config.oxygen_sensitivity;

    bridge->stats.myelin_modulations++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Glial → Substrate API
 * ============================================================================ */

int glial_substrate_compute_astrocyte_support(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->astrocyte_network) return -1;
    if (!bridge->config.enable_lactate_shuttle) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Count active astrocytes and sum lactate production */
    uint32_t active_count = 0;
    float total_lactate = 0.0f;

    astrocyte_network_t* network = bridge->astrocyte_network;
    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];
        if (astro && astro->atp_level > 0.2f) {
            active_count++;
            /* Lactate production proportional to ATP level */
            total_lactate += ASTRO_LACTATE_PRODUCTION_RATE * astro->atp_level;
        }
    }

    /* Convert lactate to ATP contribution */
    float atp_contribution = total_lactate * ASTRO_LACTATE_ATP_CONVERSION *
                             bridge->config.lactate_efficiency;

    bridge->glial_support.astro_lactate_total = total_lactate;
    bridge->glial_support.astro_atp_contribution = atp_contribution;
    bridge->glial_support.astro_active_count = active_count;

    bridge->stats.lactate_shuttles++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int glial_substrate_compute_oligodendrocyte_support(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->oligo_network) return -1;
    if (!bridge->config.enable_lactate_shuttle) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Count active oligodendrocytes and sum lactate production */
    uint32_t active_count = 0;
    float total_lactate = 0.0f;

    oligodendrocyte_network_t* network = bridge->oligo_network;
    for (uint32_t i = 0; i < network->num_oligodendrocytes; i++) {
        oligodendrocyte_t* oligo = network->oligodendrocytes[i];
        if (oligo && oligo->atp_level > 0.2f) {
            active_count++;
            /* Lactate production for axon support */
            total_lactate += OLIGO_LACTATE_PRODUCTION_RATE * oligo->atp_level;
        }
    }

    /* Convert lactate to ATP contribution */
    float atp_contribution = total_lactate * OLIGO_LACTATE_ATP_CONVERSION *
                             bridge->config.lactate_efficiency;

    bridge->glial_support.oligo_lactate_total = total_lactate;
    bridge->glial_support.oligo_atp_contribution = atp_contribution;
    bridge->glial_support.oligo_active_count = active_count;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int glial_substrate_compute_myelin_support(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->myelin_network) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get network-wide myelination statistics */
    myelin_network_stats_t stats;
    myelin_network_get_stats(bridge->myelin_network, &stats);

    /* Average myelination factor */
    float avg_myelin = 0.0f;
    if (stats.total_sheaths > 0) {
        avg_myelin = stats.mean_integrity; /* Use integrity as myelination proxy */
    }

    /* ATP savings from efficient conduction */
    /* Myelinated axons use 10-100x less ATP per action potential */
    float savings_factor = avg_myelin * MYELIN_EFFICIENCY_ATP_SAVINGS *
                           bridge->config.myelin_savings_factor;

    bridge->glial_support.myelin_atp_savings = savings_factor;
    bridge->glial_support.avg_myelination_factor = avg_myelin;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int glial_substrate_compute_microglia_support(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->microglia_network) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get network statistics */
    microglia_network_stats_t stats;
    microglia_network_get_stats(bridge->microglia_network, &stats);

    /* ATP savings from pruned synapses */
    float savings = stats.total_pruned * MICROGLIA_PRUNING_ATP_SAVINGS *
                    bridge->config.pruning_savings_factor;

    bridge->glial_support.pruning_atp_savings = savings;
    bridge->glial_support.synapses_pruned = stats.total_pruned;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int glial_substrate_apply_glial_support(glial_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Sum all ATP contributions */
    float total_support = bridge->glial_support.astro_atp_contribution +
                          bridge->glial_support.oligo_atp_contribution +
                          bridge->glial_support.myelin_atp_savings +
                          bridge->glial_support.pruning_atp_savings;

    bridge->glial_support.total_atp_support = total_support;

    /* Convert to metabolic boost (normalized 0-1) */
    float boost = clamp(total_support, 0.0f, 0.3f); /* Cap at 30% boost */
    bridge->glial_support.total_metabolic_boost = boost;

    /* Apply to substrate */
    substrate_metabolic_state_t metabolic;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);

    float new_atp = clamp(metabolic.atp_level + boost, 0.0f, 1.0f);
    substrate_set_atp(bridge->substrate, new_atp);

    /* Track max support */
    if (total_support > bridge->stats.max_atp_support) {
        bridge->stats.max_atp_support = total_support;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Bidirectional Update API
 * ============================================================================ */

int glial_substrate_update_all_effects(glial_substrate_bridge_t* bridge) {
    if (!bridge) return -1;

    int result = 0;

    if (bridge->config.enable_astrocyte_substrate && bridge->astrocyte_network) {
        result |= glial_substrate_update_astrocyte_effects(bridge);
    }

    if (bridge->config.enable_oligo_substrate && bridge->oligo_network) {
        result |= glial_substrate_update_oligodendrocyte_effects(bridge);
    }

    if (bridge->config.enable_microglia_substrate && bridge->microglia_network) {
        result |= glial_substrate_update_microglia_effects(bridge);
    }

    if (bridge->config.enable_myelin_substrate && bridge->myelin_network) {
        result |= glial_substrate_update_myelin_effects(bridge);
    }

    return result;
}

int glial_substrate_compute_all_support(glial_substrate_bridge_t* bridge) {
    if (!bridge) return -1;

    int result = 0;

    if (bridge->config.enable_lactate_shuttle) {
        if (bridge->astrocyte_network) {
            result |= glial_substrate_compute_astrocyte_support(bridge);
        }
        if (bridge->oligo_network) {
            result |= glial_substrate_compute_oligodendrocyte_support(bridge);
        }
    }

    if (bridge->myelin_network) {
        result |= glial_substrate_compute_myelin_support(bridge);
    }

    if (bridge->microglia_network) {
        result |= glial_substrate_compute_microglia_support(bridge);
    }

    return result;
}

int glial_substrate_bridge_update(
    glial_substrate_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->stats.total_updates++;

    /* Track substrate health */
    substrate_health_level_t health = substrate_get_health_level(bridge->substrate);
    float health_score = (float)(4 - health) / 4.0f; /* Convert enum to 0-1 */
    bridge->stats.avg_substrate_health =
        (bridge->stats.avg_substrate_health * 0.95f) + (health_score * 0.05f);

    if (health >= SUBSTRATE_HEALTH_COMPROMISED) {
        bridge->stats.stress_events++;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    /* Update substrate effects on glial cells */
    int result = glial_substrate_update_all_effects(bridge);

    /* Compute glial support to substrate */
    result |= glial_substrate_compute_all_support(bridge);

    /* Apply support to substrate */
    result |= glial_substrate_apply_glial_support(bridge);

    return result;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int glial_substrate_get_astrocyte_effects(
    const glial_substrate_bridge_t* bridge,
    substrate_astrocyte_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->astro_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

int glial_substrate_get_oligodendrocyte_effects(
    const glial_substrate_bridge_t* bridge,
    substrate_oligodendrocyte_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->oligo_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

int glial_substrate_get_microglia_effects(
    const glial_substrate_bridge_t* bridge,
    substrate_microglia_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->micro_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

int glial_substrate_get_myelin_effects(
    const glial_substrate_bridge_t* bridge,
    substrate_myelin_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->myelin_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

int glial_substrate_get_support(
    const glial_substrate_bridge_t* bridge,
    glial_substrate_support_t* support
) {
    if (!bridge || !support) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *support = bridge->glial_support;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

float glial_substrate_get_total_atp_support(const glial_substrate_bridge_t* bridge) {
    if (!bridge) return 0.0f;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    float support = bridge->glial_support.total_atp_support;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return support;
}

int glial_substrate_get_stats(
    const glial_substrate_bridge_t* bridge,
    glial_substrate_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}
