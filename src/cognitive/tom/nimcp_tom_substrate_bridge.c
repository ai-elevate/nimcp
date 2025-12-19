/**
 * @file nimcp_tom_substrate_bridge.c
 * @brief Theory of Mind substrate integration implementation
 *
 * WHAT: Implements ToM-substrate bridge for metabolic modulation of mentalizing
 * WHY: ToM requires prefrontal and temporal-parietal networks with high metabolic demands
 * HOW: Monitors substrate metrics and computes ATP-based capacity factors
 */

#include "cognitive/tom/nimcp_tom_substrate_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_common.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/time/nimcp_time.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * Clamp value to range [min, max]
 *
 * WHAT: Constrains value to specified range
 * WHY: Ensures capacity factors stay in valid bounds
 * HOW: Returns min if below, max if above, value otherwise
 */
static inline float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * Compute mentalizing capacity from substrate state
 *
 * WHAT: Calculates ability to infer others' mental states
 * WHY: Mentalizing relies on mPFC/TPJ networks requiring high ATP
 * HOW: mentalizing_capacity = clamp(metabolic_capacity * atp_level, 0.2, 1.0)
 *
 * Biological basis:
 * - Medial prefrontal cortex (mPFC) and temporo-parietal junction (TPJ)
 *   are metabolically expensive regions
 * - ATP depletion impairs capacity to model complex mental states
 * - Minimum 0.2 represents basic reflex-level social responses
 */
static float compute_mentalizing_capacity(
    const substrate_metabolic_state_t* metabolic,
    float sensitivity
) {
    /* Guard: validate inputs */
    if (!metabolic) return 0.2f;

    float atp = metabolic->atp_level;
    float capacity = metabolic->metabolic_capacity;

    /* Clamp inputs to valid ranges */
    atp = clamp(atp, 0.0f, 1.0f);
    capacity = clamp(capacity, 0.0f, 1.0f);
    sensitivity = clamp(sensitivity, 0.5f, 2.0f);

    /* Base mentalizing: product of metabolic capacity and ATP level */
    float mentalizing = capacity * atp;

    /* Apply sensitivity scaling */
    mentalizing = powf(mentalizing, sensitivity);

    /* Clamp to [0.2, 1.0] - maintain minimal social cognition */
    return clamp(mentalizing, 0.2f, 1.0f);
}

/**
 * Compute perspective-taking accuracy
 *
 * WHAT: Calculates ability to adopt others' viewpoints
 * WHY: Perspective-taking is cognitively demanding, requires sustained attention
 * HOW: perspective_taking = clamp(atp_level * 1.1, 0.2, 1.0)
 *
 * Biological basis:
 * - Perspective-taking involves mental rotation and viewpoint transformation
 * - Requires sustained prefrontal activity
 * - 1.1 multiplier allows brief enhancement with optimal ATP
 */
static float compute_perspective_taking(
    const substrate_metabolic_state_t* metabolic,
    float sensitivity
) {
    /* Guard: validate inputs */
    if (!metabolic) return 0.2f;

    float atp = metabolic->atp_level;

    /* Clamp inputs */
    atp = clamp(atp, 0.0f, 1.0f);
    sensitivity = clamp(sensitivity, 0.5f, 2.0f);

    /* Perspective-taking scales directly with ATP, slight boost at optimal */
    float perspective = atp * 1.1f;

    /* Apply sensitivity */
    perspective = powf(perspective, sensitivity);

    /* Clamp to [0.2, 1.0] */
    return clamp(perspective, 0.2f, 1.0f);
}

/**
 * Compute belief tracking capacity
 *
 * WHAT: Calculates ability to track others' belief states
 * WHY: Tracking beliefs requires working memory and sustained attention
 * HOW: belief_tracking = clamp((atp_level + glucose_level) / 2.0, 0.2, 1.0)
 *
 * Biological basis:
 * - False belief tasks require maintaining multiple mental models
 * - Both ATP and glucose contribute to working memory capacity
 * - Combined metric reflects metabolic support for belief tracking
 */
static float compute_belief_tracking(
    const substrate_metabolic_state_t* metabolic,
    float sensitivity
) {
    /* Guard: validate inputs */
    if (!metabolic) return 0.2f;

    float atp = metabolic->atp_level;
    float glucose = metabolic->glucose_level;

    /* Clamp inputs */
    atp = clamp(atp, 0.0f, 1.0f);
    glucose = clamp(glucose, 0.0f, 1.0f);
    sensitivity = clamp(sensitivity, 0.5f, 2.0f);

    /* Average of ATP and glucose reflects sustained metabolic support */
    float belief = (atp + glucose) / 2.0f;

    /* Apply sensitivity */
    belief = powf(belief, sensitivity);

    /* Clamp to [0.2, 1.0] */
    return clamp(belief, 0.2f, 1.0f);
}

/**
 * Compute empathy factor
 *
 * WHAT: Calculates empathic processing capacity
 * WHY: Empathy involves mirror neurons and emotional processing
 * HOW: empathy_factor = clamp(metabolic_capacity, 0.3, 1.0)
 *
 * Biological basis:
 * - Mirror neuron systems require metabolic resources
 * - Emotional resonance depends on overall metabolic health
 * - Minimum 0.3 represents basic emotional contagion
 */
static float compute_empathy_factor(
    const substrate_metabolic_state_t* metabolic,
    float sensitivity
) {
    /* Guard: validate inputs */
    if (!metabolic) return 0.3f;

    float capacity = metabolic->metabolic_capacity;

    /* Clamp inputs */
    capacity = clamp(capacity, 0.0f, 1.0f);
    sensitivity = clamp(sensitivity, 0.5f, 2.0f);

    /* Empathy scales with overall metabolic capacity */
    float empathy = capacity;

    /* Apply sensitivity */
    empathy = powf(empathy, sensitivity);

    /* Clamp to [0.3, 1.0] - maintain basic emotional contagion */
    return clamp(empathy, 0.3f, 1.0f);
}

/**
 * Update statistics tracking
 *
 * WHAT: Updates running statistics for ToM substrate bridge
 * WHY: Enables monitoring and diagnostics of ToM metabolic state
 * HOW: Increments counters, tracks extrema, computes running averages
 */
static void update_statistics(
    tom_substrate_bridge_t* bridge,
    const substrate_metabolic_state_t* metabolic
) {
    /* Guard: validate inputs */
    if (!bridge || !metabolic) return;

    tom_substrate_stats_t* stats = &bridge->stats;
    const tom_substrate_effects_t* effects = &bridge->effects;

    /* Increment update counter */
    stats->total_updates++;

    /* Track impairment episodes */
    if (effects->is_impaired) {
        stats->impairment_episodes++;
        if (effects->mentalizing_capacity < 0.3f) {
            stats->severe_impairment_episodes++;
        }
    }

    /* Update current values */
    stats->current_mentalizing = effects->mentalizing_capacity;
    stats->current_perspective_taking = effects->perspective_taking;
    stats->current_belief_tracking = effects->belief_tracking;
    stats->current_empathy = effects->empathy_factor;

    /* Track extrema */
    if (stats->total_updates == 1) {
        /* First update: initialize extrema */
        stats->min_mentalizing = effects->mentalizing_capacity;
        stats->max_mentalizing = effects->mentalizing_capacity;
        stats->avg_mentalizing = effects->mentalizing_capacity;
        stats->avg_atp_level = metabolic->atp_level;
        /* Compute fatigue from inverse of metabolic capacity */
        stats->avg_fatigue_level = 1.0f - metabolic->metabolic_capacity;
    } else {
        /* Update extrema */
        if (effects->mentalizing_capacity < stats->min_mentalizing) {
            stats->min_mentalizing = effects->mentalizing_capacity;
        }
        if (effects->mentalizing_capacity > stats->max_mentalizing) {
            stats->max_mentalizing = effects->mentalizing_capacity;
        }

        /* Update running averages */
        float n = (float)stats->total_updates;
        stats->avg_mentalizing = ((stats->avg_mentalizing * (n - 1.0f)) +
                                  effects->mentalizing_capacity) / n;
        stats->avg_atp_level = ((stats->avg_atp_level * (n - 1.0f)) +
                               metabolic->atp_level) / n;
        /* Compute fatigue from inverse of metabolic capacity */
        float fatigue_level = 1.0f - metabolic->metabolic_capacity;
        stats->avg_fatigue_level = ((stats->avg_fatigue_level * (n - 1.0f)) +
                                    fatigue_level) / n;
    }
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

int tom_substrate_default_config(tom_substrate_config_t* config) {
    /* Guard: validate config pointer */
    if (!config) {
        NIMCP_LOGGING_ERROR("Cannot initialize default config: NULL pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Enable all modulations by default */
    config->enable_atp_modulation = true;
    config->enable_fatigue_modulation = true;
    config->enable_stress_modulation = true;
    config->enable_empathy_modulation = true;

    /* Set moderate sensitivities */
    config->atp_sensitivity = 1.0f;
    config->fatigue_sensitivity = 1.0f;
    config->stress_sensitivity = 1.0f;
    config->empathy_sensitivity = 1.0f;

    /* Impairment threshold: 60% capacity */
    config->impairment_threshold = 0.6f;

    /* Update every 100ms */
    config->update_interval_ms = 100;

    NIMCP_LOGGING_DEBUG("Initialized default ToM substrate config");
    return NIMCP_SUCCESS;
}

tom_substrate_bridge_t* tom_substrate_bridge_create(
    const tom_substrate_config_t* config,
    theory_of_mind_t tom,
    neural_substrate_t* substrate
) {
    /* Guard: validate required pointers */
    if (!tom) {
        NIMCP_LOGGING_ERROR("Cannot create ToM substrate bridge: NULL ToM module");
        return NULL;
    }

    if (!substrate) {
        NIMCP_LOGGING_ERROR("Cannot create ToM substrate bridge: NULL substrate");
        return NULL;
    }

    /* Allocate bridge structure */
    tom_substrate_bridge_t* bridge = (tom_substrate_bridge_t*)nimcp_malloc(
        sizeof(tom_substrate_bridge_t)
    );
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate ToM substrate bridge");
        return NULL;
    }

    /* Initialize pointers */
    bridge->tom = tom;
    bridge->substrate = substrate;

    /* Copy or use default configuration */
    if (config) {
        memcpy(&bridge->config, config, sizeof(tom_substrate_config_t));
    } else {
        tom_substrate_default_config(&bridge->config);
    }

    /* Initialize effects to neutral state */
    memset(&bridge->effects, 0, sizeof(tom_substrate_effects_t));
    bridge->effects.mentalizing_capacity = 1.0f;
    bridge->effects.perspective_taking = 1.0f;
    bridge->effects.belief_tracking = 1.0f;
    bridge->effects.empathy_factor = 1.0f;
    bridge->effects.is_impaired = false;

    /* Initialize statistics */
    memset(&bridge->stats, 0, sizeof(tom_substrate_stats_t));
    bridge->stats.min_mentalizing = 1.0f;
    bridge->stats.max_mentalizing = 1.0f;
    bridge->stats.avg_mentalizing = 1.0f;

    /* Initialize bio-async context */
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;

    /* Create mutex for thread safety */
    bridge->mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for ToM substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for ToM substrate bridge");
        nimcp_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize timestamp */
    bridge->last_update_time_ms = 0;

    NIMCP_LOGGING_INFO("Created ToM substrate bridge");
    return bridge;
}

void tom_substrate_bridge_destroy(tom_substrate_bridge_t* bridge) {
    /* Guard: NULL-safe */
    if (!bridge) {
        return;
    }

    /* Disconnect from bio-async if connected */
    if (bridge->bio_async_enabled) {
        tom_substrate_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free bridge structure */
    nimcp_free(bridge);

    NIMCP_LOGGING_DEBUG("Destroyed ToM substrate bridge");
}

int tom_substrate_connect_bio_async(tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot connect bio-async: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: check if already connected */
    if (bridge->bio_async_enabled) {
        NIMCP_LOGGING_DEBUG("ToM substrate bridge already connected to bio-async");
        return NIMCP_SUCCESS;
    }

    /* Register with bio-async router */
    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_TOM,
        .module_name = "tom_substrate_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected ToM substrate bridge to bio-async router");
        return NIMCP_SUCCESS;
    } else {
        NIMCP_LOGGING_DEBUG("Bio-async router not available, skipping registration");
        return NIMCP_SUCCESS;
    }
}

int tom_substrate_disconnect_bio_async(tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot disconnect bio-async: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: check if connected */
    if (!bridge->bio_async_enabled) {
        return NIMCP_SUCCESS;
    }

    /* Unregister from bio-async router */
    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }

    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_DEBUG("Disconnected ToM substrate bridge from bio-async");
    return NIMCP_SUCCESS;
}

bool tom_substrate_is_bio_async_connected(const tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        return false;
    }

    return bridge->bio_async_enabled;
}

int tom_substrate_update(tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot update ToM substrate effects: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Lock for thread safety */
    if (bridge->mutex) {
        nimcp_platform_mutex_lock(bridge->mutex);
    }

    /* Check update interval */
    uint64_t current_time = nimcp_time_get_ms();
    if (bridge->last_update_time_ms > 0 &&
        (current_time - bridge->last_update_time_ms) < bridge->config.update_interval_ms) {
        /* Too soon, skip update */
        if (bridge->mutex) {
            nimcp_platform_mutex_unlock(bridge->mutex);
        }
        return NIMCP_SUCCESS;
    }

    /* Get substrate metabolic state */
    substrate_metabolic_state_t metabolic;
    int ret = substrate_get_metabolic_state(bridge->substrate, &metabolic);
    if (ret != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to get substrate metabolic state");
        if (bridge->mutex) {
            nimcp_platform_mutex_unlock(bridge->mutex);
        }
        return ret;
    }

    /* Compute mentalizing capacity */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.mentalizing_capacity = compute_mentalizing_capacity(
            &metabolic,
            bridge->config.atp_sensitivity
        );
    } else {
        bridge->effects.mentalizing_capacity = 1.0f;
    }

    /* Compute perspective-taking */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.perspective_taking = compute_perspective_taking(
            &metabolic,
            bridge->config.fatigue_sensitivity
        );
    } else {
        bridge->effects.perspective_taking = 1.0f;
    }

    /* Compute belief tracking */
    if (bridge->config.enable_stress_modulation) {
        bridge->effects.belief_tracking = compute_belief_tracking(
            &metabolic,
            bridge->config.stress_sensitivity
        );
    } else {
        bridge->effects.belief_tracking = 1.0f;
    }

    /* Compute empathy factor */
    if (bridge->config.enable_empathy_modulation) {
        bridge->effects.empathy_factor = compute_empathy_factor(
            &metabolic,
            bridge->config.empathy_sensitivity
        );
    } else {
        bridge->effects.empathy_factor = 1.0f;
    }

    /* Determine impairment state */
    bridge->effects.is_impaired =
        (bridge->effects.mentalizing_capacity < bridge->config.impairment_threshold);

    /* Update statistics */
    update_statistics(bridge, &metabolic);

    /* Update timestamp */
    bridge->last_update_time_ms = current_time;

    /* Unlock */
    if (bridge->mutex) {
        nimcp_platform_mutex_unlock(bridge->mutex);
    }

    return NIMCP_SUCCESS;
}

float tom_substrate_get_mentalizing_capacity(const tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get mentalizing capacity: NULL bridge");
        return -1.0f;
    }

    return bridge->effects.mentalizing_capacity;
}

float tom_substrate_get_perspective_taking(const tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get perspective-taking: NULL bridge");
        return -1.0f;
    }

    return bridge->effects.perspective_taking;
}

float tom_substrate_get_belief_tracking(const tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get belief tracking: NULL bridge");
        return -1.0f;
    }

    return bridge->effects.belief_tracking;
}

float tom_substrate_get_empathy_factor(const tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get empathy factor: NULL bridge");
        return -1.0f;
    }

    return bridge->effects.empathy_factor;
}

int tom_substrate_get_effects(
    const tom_substrate_bridge_t* bridge,
    tom_substrate_effects_t* effects
) {
    /* Guard: validate pointers */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get effects: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!effects) {
        NIMCP_LOGGING_ERROR("Cannot get effects: NULL output pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Copy effects structure */
    memcpy(effects, &bridge->effects, sizeof(tom_substrate_effects_t));

    return NIMCP_SUCCESS;
}

bool tom_substrate_is_impaired(const tom_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        return false;
    }

    return bridge->effects.is_impaired;
}

int tom_substrate_get_stats(
    const tom_substrate_bridge_t* bridge,
    tom_substrate_stats_t* stats
) {
    /* Guard: validate pointers */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get stats: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!stats) {
        NIMCP_LOGGING_ERROR("Cannot get stats: NULL output pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Copy statistics structure */
    memcpy(stats, &bridge->stats, sizeof(tom_substrate_stats_t));

    return NIMCP_SUCCESS;
}
