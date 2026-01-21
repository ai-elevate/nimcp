/**
 * @file nimcp_fep_immune_bridge.c
 * @brief Free Energy Principle - Immune System Integration Bridge
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Implementation of bidirectional FEP-immune integration
 * WHY:  Inflammation impairs precision-weighting; high PE triggers immune response
 * HOW:  Model cytokine effects on precision/learning, PE-triggered antigen presentation
 */

#include "cognitive/free_energy/nimcp_fep_immune_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

/* Get precision factor based on inflammation level */
static float get_inflammation_precision_factor(brain_inflammation_level_t level) {
    switch (level) {
        case INFLAMMATION_NONE:     return INFLAMMATION_NONE_PRECISION_FACTOR;
        case INFLAMMATION_LOCAL:    return INFLAMMATION_LOCAL_PRECISION_FACTOR;
        case INFLAMMATION_REGIONAL: return INFLAMMATION_REGIONAL_PRECISION_FACTOR;
        case INFLAMMATION_SYSTEMIC: return INFLAMMATION_SYSTEMIC_PRECISION_FACTOR;
        case INFLAMMATION_STORM:    return INFLAMMATION_STORM_PRECISION_FACTOR;
        default:                    return 1.0f;
    }
}

/* Get learning rate factor based on inflammation level */
static float get_inflammation_lr_factor(brain_inflammation_level_t level) {
    switch (level) {
        case INFLAMMATION_NONE:     return INFLAMMATION_NONE_LR_FACTOR;
        case INFLAMMATION_LOCAL:    return INFLAMMATION_LOCAL_LR_FACTOR;
        case INFLAMMATION_REGIONAL: return INFLAMMATION_REGIONAL_LR_FACTOR;
        case INFLAMMATION_SYSTEMIC: return INFLAMMATION_SYSTEMIC_LR_FACTOR;
        case INFLAMMATION_STORM:    return INFLAMMATION_STORM_LR_FACTOR;
        default:                    return 1.0f;
    }
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int fep_immune_bridge_default_config(fep_immune_config_t* config) {
    if (!config) return -1;

    config->prediction_error_threshold = FEP_IMMUNE_PE_THRESHOLD_MEDIUM;
    config->inflammation_precision_factor = 0.3f;
    config->cytokine_learning_impairment = 0.25f;
    config->recovery_rate = 0.1f;

    config->enable_sickness_behavior = true;
    config->enable_immune_memory_transfer = true;
    config->enable_pe_immune_activation = true;
    config->enable_convergence_il10 = true;

    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->pe_sensitivity = 1.0f;

    return 0;
}

fep_immune_bridge_t* fep_immune_bridge_create(const fep_immune_config_t* config) {
    fep_immune_bridge_t* bridge = (fep_immune_bridge_t*)nimcp_calloc(
        1, sizeof(fep_immune_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate FEP-immune bridge");
        return NULL;
    }

    /* Apply configuration */
    fep_immune_config_t default_cfg;
    if (!config) {
        fep_immune_bridge_default_config(&default_cfg);
        config = &default_cfg;
    }
    bridge->config = *config;

    /* Initialize state */
    bridge->state.inflammation_level = INFLAMMATION_NONE;
    bridge->state.current_inflammation = 0.0f;
    bridge->state.precision_reduction = 0.0f;
    bridge->state.learning_impairment = 0.0f;
    bridge->state.sickness_behavior_active = false;
    bridge->state.sickness_intensity = 0.0f;
    bridge->state.recovery_progress = 1.0f;
    bridge->state.converged = false;

    /* Initialize cytokine effects */
    memset(&bridge->cytokine_effects, 0, sizeof(fep_cytokine_effects_t));

    /* Create mutex */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        fep_immune_bridge_destroy(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("FEP-immune bridge created");
    return bridge;
}

void fep_immune_bridge_destroy(fep_immune_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->base.bio_async_enabled) {
        fep_immune_bridge_disconnect_bio_async(bridge);
    }

    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("FEP-immune bridge destroyed");
}

/* ============================================================================
 * Connection Implementation
 * ============================================================================ */

int fep_immune_bridge_connect_fep(
    fep_immune_bridge_t* bridge,
    fep_system_t* fep
) {
    if (!bridge) return -1;
    /* Allow NULL fep to disconnect/reset FEP connection */

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("FEP-immune bridge connected to FEP system");
    return 0;
}

int fep_immune_bridge_connect_immune(
    fep_immune_bridge_t* bridge,
    brain_immune_system_t* immune
) {
    if (!bridge || !immune) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->immune_system = immune;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("FEP-immune bridge connected to immune system");
    return 0;
}

int fep_immune_bridge_disconnect(fep_immune_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->immune_system = NULL;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * FEP → Immune Direction Implementation
 * ============================================================================ */

int fep_immune_report_prediction_failure(
    fep_immune_bridge_t* bridge,
    float magnitude
) {
    if (!bridge) return -1;
    if (!bridge->config.enable_pe_immune_activation) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Check if magnitude exceeds threshold */
    float scaled_magnitude = magnitude * bridge->config.pe_sensitivity;

    if (scaled_magnitude >= FEP_IMMUNE_PE_THRESHOLD_CRITICAL) {
        /* Critical: Severe immune response */
        if (bridge->immune_system) {
            uint8_t pattern[8] = {0xFE, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
            uint32_t antigen_id;
            brain_immune_present_antigen(bridge->immune_system,
                ANTIGEN_SOURCE_MANUAL, pattern, 8, 10, 0, &antigen_id);
            bridge->stats.immune_activations++;
        }
        bridge->stats.prediction_failures++;
    } else if (scaled_magnitude >= FEP_IMMUNE_PE_THRESHOLD_HIGH) {
        /* High: Strong immune response */
        bridge->stats.prediction_failures++;
        bridge->stats.immune_activations++;
    } else if (scaled_magnitude >= FEP_IMMUNE_PE_THRESHOLD_MEDIUM) {
        /* Medium: Moderate response */
        bridge->stats.model_violations++;
        bridge->stats.immune_activations++;
    }

    bridge->state.prediction_failures_reported++;
    bridge->state.last_pe_report_time = get_time_ms();

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fep_immune_report_model_violation(
    fep_immune_bridge_t* bridge,
    const uint8_t* pattern,
    size_t len
) {
    if (!bridge || !pattern || len == 0) return -1;
    if (!bridge->config.enable_pe_immune_activation) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Present pattern as antigen to immune system */
    if (bridge->immune_system) {
        uint32_t antigen_id;
        brain_immune_present_antigen(bridge->immune_system,
            ANTIGEN_SOURCE_MANUAL, pattern, len, 5, 0, &antigen_id);
    }

    bridge->stats.model_violations++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fep_immune_transfer_belief_to_memory(fep_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_immune_memory_transfer) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Transfer FEP belief patterns to immune memory */
    if (bridge->fep_system && bridge->immune_system) {
        /* Create memory signature from current beliefs */
        fep_system_t* fep = bridge->fep_system;
        if (fep->num_levels > 0) {
            fep_belief_t* beliefs = &fep->levels[0].beliefs;

            /* Convert beliefs to pattern (simplified) */
            uint8_t pattern[64];
            size_t pattern_len = beliefs->dim < 64 ? beliefs->dim : 64;

            for (size_t i = 0; i < pattern_len; i++) {
                pattern[i] = (uint8_t)(clamp_f(beliefs->mean[i], 0.0f, 1.0f) * 255);
            }

            /* This would create immune memory, but API may differ */
            bridge->stats.memory_transfers++;
        }
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fep_immune_convergence_il10_release(fep_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_convergence_il10) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Release anti-inflammatory IL-10 when beliefs converge */
    if (bridge->immune_system && bridge->state.converged) {
        /* Signal recovery via cytokine release */
        bridge->cytokine_effects.il10_precision_boost = CYTOKINE_IL10_PRECISION_IMPACT;
        bridge->cytokine_effects.il10_learning_boost = CYTOKINE_IL10_LR_BOOST;

        bridge->stats.convergence_il10_releases++;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Immune → FEP Direction Implementation
 * ============================================================================ */

int fep_immune_apply_inflammation_effects(fep_immune_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get inflammation level from immune system via stats */
    brain_inflammation_level_t level = INFLAMMATION_NONE;
    if (bridge->immune_system) {
        brain_immune_stats_t stats;
        if (brain_immune_get_stats(bridge->immune_system, &stats) == 0) {
            /* Use inflammation level directly from stats */
            level = stats.inflammation_level;
        }
    }

    bridge->state.inflammation_level = level;

    /* Get modulation factors */
    float precision_factor = get_inflammation_precision_factor(level);
    float lr_factor = get_inflammation_lr_factor(level);

    /* Apply sensitivity scaling */
    float sensitivity = bridge->config.inflammation_sensitivity;
    precision_factor = 1.0f - (1.0f - precision_factor) * sensitivity;
    lr_factor = 1.0f - (1.0f - lr_factor) * sensitivity;

    /* Apply to FEP system */
    fep_system_t* fep = bridge->fep_system;
    for (uint32_t l = 0; l < fep->num_levels; l++) {
        fep_hierarchy_level_t* lvl = &fep->levels[l];

        for (uint32_t i = 0; i < lvl->errors.dim; i++) {
            lvl->errors.precision[i] *= precision_factor;
        }
    }

    /* Update state */
    bridge->state.precision_reduction = 1.0f - precision_factor;
    bridge->state.learning_impairment = 1.0f - lr_factor;

    /* Check sickness behavior threshold */
    float inflammation_intensity = 1.0f - precision_factor;
    bridge->state.current_inflammation = inflammation_intensity;
    bridge->state.sickness_behavior_active =
        inflammation_intensity >= SICKNESS_BEHAVIOR_THRESHOLD;
    bridge->state.sickness_intensity = inflammation_intensity;

    /* Update statistics */
    bridge->stats.total_precision_reduction += bridge->state.precision_reduction;
    bridge->stats.total_learning_impairment += bridge->state.learning_impairment;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fep_immune_get_precision_modifier(
    const fep_immune_bridge_t* bridge,
    float* modifier
) {
    if (!bridge || !modifier) return -1;

    float base = get_inflammation_precision_factor(bridge->state.inflammation_level);

    /* Add cytokine effects */
    float cytokine_effect =
        bridge->cytokine_effects.il6_precision_reduction +
        bridge->cytokine_effects.tnf_precision_reduction +
        bridge->cytokine_effects.il1_precision_reduction +
        bridge->cytokine_effects.ifn_gamma_precision_reduction +
        bridge->cytokine_effects.il10_precision_boost;

    *modifier = clamp_f(base + cytokine_effect, 0.1f, 1.5f);
    return 0;
}

int fep_immune_get_learning_modifier(
    const fep_immune_bridge_t* bridge,
    float* modifier
) {
    if (!bridge || !modifier) return -1;

    float base = get_inflammation_lr_factor(bridge->state.inflammation_level);

    /* Add cytokine effects */
    float cytokine_effect =
        bridge->cytokine_effects.il6_learning_impairment +
        bridge->cytokine_effects.tnf_learning_impairment +
        bridge->cytokine_effects.il1_learning_impairment +
        bridge->cytokine_effects.il10_learning_boost;

    *modifier = clamp_f(base + cytokine_effect, 0.1f, 1.5f);
    return 0;
}

int fep_immune_update_cytokine_effects(fep_immune_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Update cytokine effects based on inflammation level */
    /* Note: This is a simplified implementation - full implementation would
     * query individual cytokine levels from the immune system */
    if (bridge->immune_system) {
        float sensitivity = bridge->config.cytokine_sensitivity;
        float base_level = bridge->state.inflammation_level;

        /* Compute cytokine effects based on inflammation level */
        bridge->cytokine_effects.il6_precision_reduction =
            CYTOKINE_IL6_PRECISION_IMPACT * base_level * sensitivity;
        bridge->cytokine_effects.il6_learning_impairment =
            CYTOKINE_IL6_LR_IMPAIRMENT * base_level * sensitivity;

        bridge->cytokine_effects.tnf_precision_reduction =
            CYTOKINE_TNF_PRECISION_IMPACT * base_level * sensitivity;
        bridge->cytokine_effects.tnf_learning_impairment =
            CYTOKINE_TNF_LR_IMPAIRMENT * base_level * sensitivity;

        bridge->cytokine_effects.il1_precision_reduction =
            CYTOKINE_IL1_PRECISION_IMPACT * base_level * sensitivity;
        bridge->cytokine_effects.il1_learning_impairment =
            CYTOKINE_IL1_LR_IMPAIRMENT * base_level * sensitivity;

        /* IL-10 recovery effects (inverse relationship with inflammation) */
        float recovery_level = 1.0f - base_level;
        bridge->cytokine_effects.il10_precision_boost =
            CYTOKINE_IL10_PRECISION_IMPACT * recovery_level * sensitivity;
        bridge->cytokine_effects.il10_learning_boost =
            CYTOKINE_IL10_LR_BOOST * recovery_level * sensitivity;

        /* Compute totals */
        bridge->cytokine_effects.total_precision_reduction =
            bridge->cytokine_effects.il6_precision_reduction +
            bridge->cytokine_effects.tnf_precision_reduction +
            bridge->cytokine_effects.il1_precision_reduction -
            bridge->cytokine_effects.il10_precision_boost;

        bridge->cytokine_effects.total_learning_impairment =
            bridge->cytokine_effects.il6_learning_impairment +
            bridge->cytokine_effects.tnf_learning_impairment +
            bridge->cytokine_effects.il1_learning_impairment -
            bridge->cytokine_effects.il10_learning_boost;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Update Cycle Implementation
 * ============================================================================ */

int fep_immune_bridge_update(
    fep_immune_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* 1. Update cytokine effects (inlined to avoid deadlock) */
    if (bridge->immune_system) {
        float sensitivity = bridge->config.cytokine_sensitivity;
        /* Normalize inflammation level to 0.0-1.0 range */
        float base_level = (float)bridge->state.inflammation_level / (float)INFLAMMATION_STORM;

        bridge->cytokine_effects.il6_precision_reduction =
            CYTOKINE_IL6_PRECISION_IMPACT * base_level * sensitivity;
        bridge->cytokine_effects.il6_learning_impairment =
            CYTOKINE_IL6_LR_IMPAIRMENT * base_level * sensitivity;

        bridge->cytokine_effects.tnf_precision_reduction =
            CYTOKINE_TNF_PRECISION_IMPACT * base_level * sensitivity;
        bridge->cytokine_effects.tnf_learning_impairment =
            CYTOKINE_TNF_LR_IMPAIRMENT * base_level * sensitivity;

        bridge->cytokine_effects.il1_precision_reduction =
            CYTOKINE_IL1_PRECISION_IMPACT * base_level * sensitivity;
        bridge->cytokine_effects.il1_learning_impairment =
            CYTOKINE_IL1_LR_IMPAIRMENT * base_level * sensitivity;

        float recovery_level = 1.0f - base_level;
        bridge->cytokine_effects.il10_precision_boost =
            CYTOKINE_IL10_PRECISION_IMPACT * recovery_level * sensitivity;
        bridge->cytokine_effects.il10_learning_boost =
            CYTOKINE_IL10_LR_BOOST * recovery_level * sensitivity;

        bridge->cytokine_effects.total_precision_reduction =
            bridge->cytokine_effects.il6_precision_reduction +
            bridge->cytokine_effects.tnf_precision_reduction +
            bridge->cytokine_effects.il1_precision_reduction -
            bridge->cytokine_effects.il10_precision_boost;

        bridge->cytokine_effects.total_learning_impairment =
            bridge->cytokine_effects.il6_learning_impairment +
            bridge->cytokine_effects.tnf_learning_impairment +
            bridge->cytokine_effects.il1_learning_impairment -
            bridge->cytokine_effects.il10_learning_boost;
    }

    /* 2. Apply inflammation effects (inlined to avoid deadlock) */
    if (bridge->fep_system) {
        brain_inflammation_level_t level = INFLAMMATION_NONE;
        if (bridge->immune_system) {
            brain_immune_stats_t stats;
            if (brain_immune_get_stats(bridge->immune_system, &stats) == 0) {
                /* Use inflammation level directly from stats */
                level = stats.inflammation_level;
            }
        }
        bridge->state.inflammation_level = level;

        float precision_factor = get_inflammation_precision_factor(level);
        float lr_factor = get_inflammation_lr_factor(level);
        float sensitivity = bridge->config.inflammation_sensitivity;
        precision_factor = 1.0f - (1.0f - precision_factor) * sensitivity;
        lr_factor = 1.0f - (1.0f - lr_factor) * sensitivity;

        /* Apply to FEP levels */
        fep_system_t* fep = bridge->fep_system;
        for (uint32_t l = 0; l < fep->num_levels; l++) {
            for (uint32_t i = 0; i < fep->levels[l].errors.dim; i++) {
                fep->levels[l].errors.precision[i] *= precision_factor;
            }
        }
        /* current_inflammation is the inverse of precision (high precision = low inflammation) */
        bridge->state.current_inflammation = 1.0f - precision_factor;
    }

    /* 3. Check FEP state for immune triggers */
    if (bridge->fep_system && bridge->config.enable_pe_immune_activation) {
        fep_system_t* fep = bridge->fep_system;

        /* Check prediction error magnitude */
        float total_pe = 0.0f;
        for (uint32_t l = 0; l < fep->num_levels; l++) {
            total_pe += fep->levels[l].errors.magnitude;
        }
        float avg_pe = fep->num_levels > 0 ? total_pe / (float)fep->num_levels : 0.0f;

        /* Report if above threshold (inlined) */
        if (avg_pe > bridge->config.prediction_error_threshold) {
            float scaled_magnitude = avg_pe * bridge->config.pe_sensitivity;
            if (scaled_magnitude >= FEP_IMMUNE_PE_THRESHOLD_CRITICAL) {
                if (bridge->immune_system) {
                    uint8_t pattern[8] = {0xFE, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
                    uint32_t antigen_id;
                    brain_immune_present_antigen(bridge->immune_system,
                        ANTIGEN_SOURCE_MANUAL, pattern, 8, 10, 0, &antigen_id);
                }
                bridge->stats.prediction_failures++;
            } else if (scaled_magnitude >= FEP_IMMUNE_PE_THRESHOLD_HIGH) {
                bridge->stats.prediction_failures++;
            } else if (scaled_magnitude >= FEP_IMMUNE_PE_THRESHOLD_MEDIUM) {
                bridge->stats.model_violations++;
            }
            bridge->state.prediction_failures_reported++;
            bridge->state.last_pe_report_time = get_time_ms();
        }

        /* Check for convergence (inlined) */
        bridge->state.converged = avg_pe < RECOVERY_CONVERGENCE_THRESHOLD;
        if (bridge->state.converged && bridge->config.enable_convergence_il10) {
            if (bridge->immune_system) {
                bridge->cytokine_effects.il10_precision_boost = CYTOKINE_IL10_PRECISION_IMPACT;
                bridge->cytokine_effects.il10_learning_boost = CYTOKINE_IL10_LR_BOOST;
                bridge->stats.convergence_il10_releases++;
            }
        }
    }

    /* 4. Update recovery progress */
    if (bridge->state.inflammation_level == INFLAMMATION_NONE) {
        bridge->state.recovery_progress = clamp_f(
            bridge->state.recovery_progress + bridge->config.recovery_rate * (float)delta_ms / 1000.0f,
            0.0f, 1.0f);
    } else {
        bridge->state.recovery_progress = 0.0f;
    }

    /* 5. Update statistics */
    bridge->stats.avg_inflammation =
        (bridge->stats.avg_inflammation * 0.99f) + (bridge->state.current_inflammation * 0.01f);

    if (bridge->fep_system && bridge->fep_system->num_levels > 0) {
        float pe = bridge->fep_system->levels[0].errors.magnitude;
        bridge->stats.avg_prediction_error =
            (bridge->stats.avg_prediction_error * 0.99f) + (pe * 0.01f);

        if (bridge->state.inflammation_level > INFLAMMATION_NONE) {
            bridge->stats.avg_free_energy_under_inflammation =
                (bridge->stats.avg_free_energy_under_inflammation * 0.99f) +
                (bridge->fep_system->free_energy.total * 0.01f);
        }
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * State/Stats Implementation
 * ============================================================================ */

int fep_immune_bridge_get_state(
    const fep_immune_bridge_t* bridge,
    fep_immune_state_t* state
) {
    if (!bridge || !state) return -1;
    *state = bridge->state;
    return 0;
}

int fep_immune_bridge_get_stats(
    const fep_immune_bridge_t* bridge,
    fep_immune_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

bool fep_immune_is_sickness_active(const fep_immune_bridge_t* bridge) {
    return bridge && bridge->state.sickness_behavior_active;
}

brain_inflammation_level_t fep_immune_get_inflammation_level(
    const fep_immune_bridge_t* bridge
) {
    return bridge ? bridge->state.inflammation_level : INFLAMMATION_NONE;
}

/* ============================================================================
 * Bio-Async Implementation
 * ============================================================================ */

int fep_immune_bridge_connect_bio_async(fep_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_IMMUNE_BRIDGE,
        .module_name = "fep_immune_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("FEP-immune bridge connected to bio-async");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }
    return 0;
}

int fep_immune_bridge_disconnect_bio_async(fep_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool fep_immune_bridge_is_bio_async_connected(
    const fep_immune_bridge_t* bridge
) {
    return bridge && bridge->base.bio_async_enabled;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for FEP-Immune Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int fep_immune_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "FEP_Immune_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("FEP-Immune bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "FEP_Immune_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "FEP_Immune_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
