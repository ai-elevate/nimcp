/**
 * @file nimcp_fep_consciousness.c
 * @brief Consciousness-Gated Action Selection for Free Energy Principle
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Implementation of consciousness-gated FEP action selection
 * WHY:  Consciousness gates action selection - high Φ enables deliberate actions
 * HOW:  IIT's Φ metric modulates precision and gates action selection
 */

#include "cognitive/free_energy/nimcp_fep_consciousness.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/error/nimcp_error_codes.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

struct fep_consciousness_bridge {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    fep_consciousness_config_t config;

    /* Connected systems */
    fep_system_t* fep_system;
    introspection_context_t introspection;

    /* Current state */
    fep_consciousness_state_t state;

    /* Habitual action cache */
    uint32_t* habitual_cache_actions;
    float* habitual_cache_states;
    uint32_t habitual_cache_size;
    uint32_t habitual_cache_count;

    /* Attention targets */
    uint32_t* attention_targets;
    float* attention_priorities;
    uint32_t attention_count;

};

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

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

void fep_consciousness_default_config(fep_consciousness_config_t* config) {
    if (!config) return;

    config->phi_threshold = FEP_CONSCIOUSNESS_DEFAULT_PHI_THRESHOLD;
    config->attention_gain = FEP_CONSCIOUSNESS_DEFAULT_ATTENTION_GAIN;
    config->coherence_weight = FEP_CONSCIOUSNESS_DEFAULT_COHERENCE_WEIGHT;
    config->metacognitive_threshold = FEP_CONSCIOUSNESS_DEFAULT_METACOGNITIVE_THRESHOLD;
    config->enable_global_workspace = true;
    config->enable_metacognitive_control = true;
    config->enable_habitual_cache = true;
    config->update_interval_ms = 100;
}

fep_consciousness_bridge_t* fep_consciousness_create(
    const fep_consciousness_config_t* config
) {
    fep_consciousness_bridge_t* bridge = (fep_consciousness_bridge_t*)nimcp_calloc(
        1, sizeof(fep_consciousness_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate consciousness bridge");
        return NULL;
    }

    /* Apply configuration */
    fep_consciousness_config_t default_cfg;
    if (!config) {
        fep_consciousness_default_config(&default_cfg);
        config = &default_cfg;
    }
    bridge->config = *config;

    /* Initialize state */
    bridge->state.current_phi = 0.0f;
    bridge->state.attention_level = 0.5f;
    bridge->state.action_coherence = 0.5f;
    bridge->state.metacognitive_confidence = 0.5f;
    bridge->state.conscious_access = false;
    bridge->state.consciousness_state = CONSCIOUSNESS_STATE_MINIMAL;
    bridge->state.last_phi_update_ms = get_time_ms();

    /* Allocate habitual cache if enabled */
    if (config->enable_habitual_cache) {
        bridge->habitual_cache_size = 256;
        bridge->habitual_cache_actions = (uint32_t*)nimcp_calloc(
            bridge->habitual_cache_size, sizeof(uint32_t));
        bridge->habitual_cache_states = (float*)nimcp_calloc(
            bridge->habitual_cache_size * 64, sizeof(float));  /* 64-dim state */
    }

    /* Allocate attention arrays */
    bridge->attention_targets = (uint32_t*)nimcp_calloc(32, sizeof(uint32_t));
    bridge->attention_priorities = (float*)nimcp_calloc(32, sizeof(float));

    /* Create mutex */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        fep_consciousness_destroy(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Consciousness bridge created");
    return bridge;
}

void fep_consciousness_destroy(fep_consciousness_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->base.bio_async_enabled) {
        fep_consciousness_disconnect_bio_async(bridge);
    }

    if (bridge->habitual_cache_actions) nimcp_free(bridge->habitual_cache_actions);
    if (bridge->habitual_cache_states) nimcp_free(bridge->habitual_cache_states);
    if (bridge->attention_targets) nimcp_free(bridge->attention_targets);
    if (bridge->attention_priorities) nimcp_free(bridge->attention_priorities);

    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Consciousness bridge destroyed");
}

/* ============================================================================
 * Connection Implementation
 * ============================================================================ */

int fep_consciousness_connect_fep(
    fep_consciousness_bridge_t* bridge,
    fep_system_t* fep
) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Consciousness bridge connected to FEP");
    return 0;
}

int fep_consciousness_connect_introspection(
    fep_consciousness_bridge_t* bridge,
    introspection_context_t introspection
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->introspection = introspection;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Consciousness bridge connected to introspection");
    return 0;
}

int fep_consciousness_disconnect(fep_consciousness_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->introspection = NULL;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Consciousness-Gated Operations Implementation
 * ============================================================================ */

int fep_consciousness_gate_action(
    fep_consciousness_bridge_t* bridge,
    uint32_t proposed_action,
    uint32_t* gated_action
) {
    if (!bridge || !gated_action) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Check consciousness level */
    if (bridge->state.current_phi < bridge->config.phi_threshold) {
        /* Unconscious processing - use habitual action if available */
        bridge->state.conscious_access = false;
        bridge->state.unconscious_actions++;

        /* Default to proposed action if no habitual cache hit */
        *gated_action = proposed_action;

        NIMCP_LOGGING_DEBUG("Unconscious action selection (Φ=%.3f < %.3f)",
                          bridge->state.current_phi, bridge->config.phi_threshold);
    } else {
        /* Conscious processing - full deliberation */
        bridge->state.conscious_access = true;
        bridge->state.conscious_actions++;

        /* Return proposed action for conscious execution */
        *gated_action = proposed_action;

        /* Cache for future habitual use */
        if (bridge->config.enable_habitual_cache &&
            bridge->habitual_cache_count < bridge->habitual_cache_size) {
            bridge->habitual_cache_actions[bridge->habitual_cache_count++] = proposed_action;
        }

        NIMCP_LOGGING_DEBUG("Conscious action selection (Φ=%.3f >= %.3f)",
                          bridge->state.current_phi, bridge->config.phi_threshold);
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fep_consciousness_modulate_precision(
    fep_consciousness_bridge_t* bridge,
    float* precision
) {
    if (!bridge || !precision) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* precision_out = precision_in × (1 + attention_gain × Φ) */
    float boost = 1.0f + bridge->config.attention_gain * bridge->state.current_phi;
    *precision *= boost;

    /* Clamp to valid range */
    *precision = clamp_f(*precision, FEP_CONSCIOUSNESS_MIN_PRECISION_FACTOR,
                        FEP_CONSCIOUSNESS_MAX_PRECISION_BOOST);

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fep_consciousness_evaluate_coherence(
    fep_consciousness_bridge_t* bridge,
    const fep_policy_t* policy,
    float* coherence
) {
    if (!bridge || !policy || !coherence) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Evaluate action sequence coherence */
    float c = 1.0f;

    if (policy->num_actions > 1 && policy->actions) {
        /* Check for conflicting consecutive actions */
        for (size_t i = 1; i < policy->num_actions; i++) {
            /* Simple heuristic: large action value changes indicate incoherence */
            float diff = fabsf(policy->actions[i] - policy->actions[i-1]);
            if (diff > 0.5f) {
                c *= 0.9f;
            }
        }
    }

    *coherence = clamp_f(c, 0.0f, 1.0f);
    bridge->state.action_coherence = *coherence;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Metacognitive Functions Implementation
 * ============================================================================ */

int fep_consciousness_assess_model_quality(
    fep_consciousness_bridge_t* bridge,
    float* quality
) {
    if (!bridge || !quality) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Assess quality based on prediction error and free energy */
    fep_system_t* fep = bridge->fep_system;

    float total_error = 0.0f;
    for (uint32_t l = 0; l < fep->num_levels; l++) {
        total_error += fep->levels[l].errors.magnitude;
    }
    float avg_error = total_error / (float)fep->num_levels;

    /* Quality is inverse of normalized error */
    *quality = 1.0f / (1.0f + avg_error);
    *quality = clamp_f(*quality, 0.0f, 1.0f);

    bridge->state.metacognitive_confidence = *quality;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fep_consciousness_request_attention(
    fep_consciousness_bridge_t* bridge,
    uint32_t target,
    float priority
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (priority < 0.0f || priority > 1.0f) return NIMCP_ERROR_INVALID_PARAM;
    if (!bridge->config.enable_global_workspace) return NIMCP_ERROR_INVALID_STATE;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    if (bridge->attention_count < 32) {
        bridge->attention_targets[bridge->attention_count] = target;
        bridge->attention_priorities[bridge->attention_count] = priority;
        bridge->attention_count++;
    }

    /* Update overall attention level */
    float max_priority = 0.0f;
    for (uint32_t i = 0; i < bridge->attention_count; i++) {
        if (bridge->attention_priorities[i] > max_priority) {
            max_priority = bridge->attention_priorities[i];
        }
    }
    bridge->state.attention_level = max_priority;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Update and State Implementation
 * ============================================================================ */

int fep_consciousness_update(
    fep_consciousness_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    uint64_t now = get_time_ms();
    uint64_t elapsed = now - bridge->state.last_phi_update_ms;

    /* Check if update interval has passed */
    if (elapsed >= bridge->config.update_interval_ms) {
        /* Update Φ from introspection if connected */
        if (bridge->introspection) {
            consciousness_phi_result_t* result = introspection_compute_phi(bridge->introspection, NULL);
            if (result) {
                bridge->state.current_phi = result->phi;
                bridge->state.consciousness_state = consciousness_classify_phi(result->phi);
            }
        } else {
            /* Simulate Φ decay toward baseline */
            bridge->state.current_phi *= 0.95f;
            bridge->state.current_phi += 0.05f * FEP_CONSCIOUSNESS_DEFAULT_PHI_THRESHOLD;
        }

        bridge->state.last_phi_update_ms = now;
    }

    /* Update conscious access flag */
    bridge->state.conscious_access =
        bridge->state.current_phi >= bridge->config.phi_threshold;

    /* Decay attention priorities */
    for (uint32_t i = 0; i < bridge->attention_count; i++) {
        bridge->attention_priorities[i] *= 0.99f;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fep_consciousness_get_state(
    const fep_consciousness_bridge_t* bridge,
    fep_consciousness_state_t* state
) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    *state = bridge->state;
    return 0;
}

/* ============================================================================
 * Bio-Async Implementation
 * ============================================================================ */

int fep_consciousness_connect_bio_async(fep_consciousness_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_CONSCIOUSNESS,
        .module_name = "fep_consciousness",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Consciousness bridge connected to bio-async");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }
    return 0;
}

int fep_consciousness_disconnect_bio_async(fep_consciousness_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool fep_consciousness_is_bio_async_connected(
    const fep_consciousness_bridge_t* bridge
) {
    if (!bridge) return false;
    return bridge->base.bio_async_enabled;
}

/* ============================================================================
 * KG Self-Awareness Integration
 * ============================================================================ */

int fep_consciousness_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "FEP_Consciousness");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("FEP Consciousness self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "FEP_Consciousness");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "FEP_Consciousness");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
