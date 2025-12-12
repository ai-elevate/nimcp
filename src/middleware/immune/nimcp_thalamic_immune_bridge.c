/**
 * @file nimcp_thalamic_immune_bridge.c
 * @brief Thalamic Router-Immune System Integration Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * WHAT: Bidirectional coupling between brain immune and thalamic routing
 * WHY:  Biological realism - cytokines affect gating, routing failures trigger immunity
 * HOW:  Monitor cytokines to modulate routing, monitor routing to trigger immune responses
 */

#include "middleware/immune/nimcp_thalamic_immune_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>
#include <pthread.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Clamp value to range
 *
 * WHAT: Constrain value to [min, max]
 * WHY:  Prevent overflow/underflow
 * HOW:  Return min if below, max if above, value otherwise
 */
static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Get inflammation intensity as normalized float
 *
 * WHAT: Convert inflammation level enum to [0-1] intensity
 * WHY:  Need continuous value for modulation calculations
 * HOW:  Map level to normalized range
 */
static float get_inflammation_intensity(brain_inflammation_level_t level) {
    switch (level) {
        case INFLAMMATION_NONE: return 0.0f;
        case INFLAMMATION_LOCAL: return 0.25f;
        case INFLAMMATION_REGIONAL: return 0.50f;
        case INFLAMMATION_SYSTEMIC: return 0.75f;
        case INFLAMMATION_STORM: return 1.0f;
        default: return 0.0f;
    }
}

/**
 * @brief Get max inflammation level from immune system
 *
 * WHAT: Query highest inflammation level across all sites
 * WHY:  Max inflammation determines routing modulation
 * HOW:  Query brain immune system state
 */
static brain_inflammation_level_t get_max_inflammation_level(
    const brain_immune_system_t* immune
) {
    if (!immune) return INFLAMMATION_NONE;

    /* Query immune system for inflammation state */
    brain_immune_stats_t stats;
    if (brain_immune_get_stats((brain_immune_system_t*)immune, &stats) != 0) {
        return INFLAMMATION_NONE;
    }

    /* Infer level from inflammation sites count (simplified) */
    if (stats.inflammation_sites == 0) return INFLAMMATION_NONE;
    if (stats.inflammation_sites < 5) return INFLAMMATION_LOCAL;
    if (stats.inflammation_sites < 15) return INFLAMMATION_REGIONAL;
    if (stats.inflammation_sites < 30) return INFLAMMATION_SYSTEMIC;
    return INFLAMMATION_STORM;
}

/**
 * @brief Compute cytokine level (simplified)
 *
 * WHAT: Estimate cytokine concentration from immune state
 * WHY:  Need cytokine levels for routing modulation
 * HOW:  Derive from inflammation level (proportional)
 */
static float compute_cytokine_level(
    const brain_immune_system_t* immune,
    brain_cytokine_type_t type
) {
    if (!immune) return 0.0f;

    brain_inflammation_level_t inflam = get_max_inflammation_level(immune);
    float base_level = get_inflammation_intensity(inflam);

    /* Scale by cytokine type */
    switch (type) {
        case BRAIN_CYTOKINE_IL6:
            return base_level * 0.8f;  /* IL-6 tracks inflammation closely */
        case BRAIN_CYTOKINE_IL1:
            return base_level * 0.7f;  /* IL-1β slightly lower */
        case BRAIN_CYTOKINE_TNF:
            return base_level * 0.9f;  /* TNF-α highest in inflammation */
        case BRAIN_CYTOKINE_IL10:
            return (1.0f - base_level) * 0.5f;  /* Anti-inflammatory inverse */
        default:
            return 0.0f;
    }
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int thalamic_immune_default_config(thalamic_immune_config_t* config) {
    if (!config) return -1;

    /* All features enabled by default */
    config->enable_cytokine_routing_modulation = true;
    config->enable_inflammation_hypervigilance = true;
    config->enable_routing_anomaly_detection = true;
    config->enable_health_feedback = true;
    config->enable_priority_escalation = true;

    /* Biologically-based default sensitivities */
    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->anomaly_sensitivity = 1.0f;

    /* Evidence-based thresholds */
    config->queue_anomaly_threshold = ROUTING_ANOMALY_QUEUE_THRESHOLD;
    config->drop_anomaly_threshold = ROUTING_ANOMALY_DROP_THRESHOLD;
    config->latency_anomaly_ms = ROUTING_ANOMALY_LATENCY_MS;

    return 0;
}

thalamic_immune_bridge_t* thalamic_immune_bridge_create(
    const thalamic_immune_config_t* config,
    brain_immune_system_t* immune_system,
    thalamic_router_t* thalamic_router
) {
    /* Guard: require both systems */
    if (!immune_system || !thalamic_router) {
        LOG_MODULE_ERROR("thalamic_immune_bridge",
                  "Cannot create bridge without immune and router systems");
        return NULL;
    }

    /* Allocate bridge */
    thalamic_immune_bridge_t* bridge = (thalamic_immune_bridge_t*)
        nimcp_malloc(sizeof(thalamic_immune_bridge_t));
    if (!bridge) {
        LOG_MODULE_ERROR("thalamic_immune_bridge", "Allocation failed");
        return NULL;
    }

    /* Initialize to zero */
    memset(bridge, 0, sizeof(thalamic_immune_bridge_t));

    /* Link systems */
    bridge->immune_system = immune_system;
    bridge->thalamic_router = thalamic_router;

    /* Apply configuration */
    if (config) {
        bridge->enable_cytokine_routing_modulation = config->enable_cytokine_routing_modulation;
        bridge->enable_inflammation_hypervigilance = config->enable_inflammation_hypervigilance;
        bridge->enable_routing_anomaly_detection = config->enable_routing_anomaly_detection;
        bridge->enable_health_feedback = config->enable_health_feedback;
        bridge->enable_priority_escalation = config->enable_priority_escalation;
    } else {
        /* Use defaults */
        thalamic_immune_config_t default_cfg;
        thalamic_immune_default_config(&default_cfg);
        bridge->enable_cytokine_routing_modulation = default_cfg.enable_cytokine_routing_modulation;
        bridge->enable_inflammation_hypervigilance = default_cfg.enable_inflammation_hypervigilance;
        bridge->enable_routing_anomaly_detection = default_cfg.enable_routing_anomaly_detection;
        bridge->enable_health_feedback = default_cfg.enable_health_feedback;
        bridge->enable_priority_escalation = default_cfg.enable_priority_escalation;
    }

    /* Create mutex */
    bridge->mutex = nimcp_malloc(sizeof(pthread_mutex_t));
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    pthread_mutex_init((pthread_mutex_t*)bridge->mutex, NULL);

    LOG_MODULE_INFO("thalamic_immune_bridge", "Bridge created successfully");
    return bridge;
}

void thalamic_immune_bridge_destroy(thalamic_immune_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free bridge (don't destroy linked systems - we don't own them) */
    nimcp_free(bridge);
    LOG_MODULE_INFO("thalamic_immune_bridge", "Bridge destroyed");
}

/* ============================================================================
 * Immune → Routing Implementation
 * ============================================================================ */

int thalamic_immune_apply_cytokine_effects(thalamic_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_cytokine_routing_modulation) return 0;
    if (!bridge->immune_system || !bridge->thalamic_router) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Compute cytokine effects */
    cytokine_routing_effects_t* effects = &bridge->cytokine_effects;

    /* Query cytokine levels */
    float il6_level = compute_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_IL6);
    float il1_level = compute_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_IL1);
    float tnf_level = compute_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_TNF);
    float il10_level = compute_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_IL10);

    /* Pro-inflammatory cytokines → priority boost */
    effects->il6_priority_boost = il6_level * CYTOKINE_IL6_PRIORITY_BOOST;
    effects->il1_priority_boost = il1_level * CYTOKINE_IL1_PRIORITY_BOOST;
    effects->tnf_priority_boost = tnf_level * CYTOKINE_TNF_PRIORITY_BOOST;

    /* Anti-inflammatory → gating restoration */
    effects->il10_gating_restoration = il10_level * CYTOKINE_IL10_GATING_RESTORE;

    /* Aggregate effects */
    effects->total_priority_modifier =
        effects->il6_priority_boost +
        effects->il1_priority_boost +
        effects->tnf_priority_boost;

    /* Gating threshold (lower = less filtering) */
    float proinflam_total = il6_level + il1_level + tnf_level;
    effects->gating_threshold_modifier = -proinflam_total * INFLAMMATION_GATING_REDUCTION;
    effects->gating_threshold_modifier += effects->il10_gating_restoration;
    effects->gating_threshold_modifier = clamp_f(effects->gating_threshold_modifier, -0.5f, 0.3f);

    /* Threat focus and social suppression */
    effects->threat_focus_level = clamp_f(proinflam_total * 0.8f, 0.0f, 1.0f);
    effects->social_suppression_level = clamp_f(proinflam_total * 0.6f, 0.0f, 1.0f);

    bridge->cytokine_modulations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int thalamic_immune_apply_inflammation_effects(thalamic_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_inflammation_hypervigilance) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    inflammation_routing_state_t* state = &bridge->inflammation_state;

    /* Get inflammation state */
    state->current_level = get_max_inflammation_level(bridge->immune_system);
    state->inflammation_intensity = get_inflammation_intensity(state->current_level);

    /* Hypervigilance increases with inflammation */
    state->hypervigilance_level = clamp_f(state->inflammation_intensity * 1.2f, 0.0f, 1.0f);

    /* Gating reduction (more signals pass through) */
    state->gating_reduction = state->inflammation_intensity * INFLAMMATION_GATING_REDUCTION;

    /* Threat priority boost */
    state->threat_priority_boost = 1.0f + (state->inflammation_intensity * INFLAMMATION_THREAT_PRIORITY);

    /* Social priority penalty (sickness behavior) */
    state->social_priority_penalty = 1.0f - (state->inflammation_intensity * INFLAMMATION_SOCIAL_DEGRADE);

    /* Sickness behavior active at regional+ inflammation */
    state->sickness_behavior_active = (state->current_level >= INFLAMMATION_REGIONAL);

    /* Attention bias toward threats */
    state->attention_bias = clamp_f(state->inflammation_intensity * 0.9f, 0.0f, 1.0f);

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int thalamic_immune_escalate_priority(
    thalamic_immune_bridge_t* bridge,
    uint32_t source_id,
    uint32_t dest_id,
    bool is_threat_signal
) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_priority_escalation) return 0;
    if (!bridge->thalamic_router) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Get current inflammation level */
    float intensity = bridge->inflammation_state.inflammation_intensity;

    /* Compute priority escalation */
    float base_attention = 0.5f;  /* Default attention */
    float escalated_attention = base_attention;

    if (is_threat_signal) {
        /* Threat signals get boosted by inflammation */
        escalated_attention = base_attention + (intensity * 0.4f);
    } else {
        /* Non-threat signals may be suppressed during sickness behavior */
        if (bridge->inflammation_state.sickness_behavior_active) {
            escalated_attention = base_attention * (1.0f - intensity * 0.3f);
        }
    }

    escalated_attention = clamp_f(escalated_attention, 0.1f, 1.0f);

    /* Apply to router */
    thalamic_router_set_attention(bridge->thalamic_router, source_id, dest_id, escalated_attention);

    bridge->priority_escalations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int thalamic_immune_restore_gating(thalamic_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->immune_system || !bridge->thalamic_router) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Check IL-10 level */
    float il10_level = compute_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_IL10);

    /* IL-10 restores gating threshold toward normal */
    float restoration = il10_level * CYTOKINE_IL10_GATING_RESTORE;

    /* Update gating effects */
    bridge->cytokine_effects.il10_gating_restoration = restoration;
    bridge->inflammation_state.gating_reduction *= (1.0f - restoration);
    bridge->inflammation_state.hypervigilance_level *= (1.0f - restoration);

    bridge->gating_adjustments++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

/* ============================================================================
 * Routing → Immune Implementation
 * ============================================================================ */

int thalamic_immune_detect_anomalies(thalamic_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_routing_anomaly_detection) return 0;
    if (!bridge->thalamic_router) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    routing_anomaly_state_t* state = &bridge->anomaly_state;

    /* Query router statistics */
    routing_stats_t router_stats;
    if (!thalamic_router_get_stats(bridge->thalamic_router, &router_stats)) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    /* Update metrics */
    state->signals_queued = router_stats.signals_routed;
    state->signals_dropped = router_stats.signals_dropped;
    state->priority_drops = router_stats.signals_bypassed;  /* Approximation */
    state->avg_latency_ms = router_stats.avg_latency_ms;
    state->throughput_hz = router_stats.throughput_hz;

    /* Compute derived metrics */
    uint64_t total_signals = state->signals_queued + state->signals_dropped;
    state->drop_rate = (total_signals > 0) ?
        (float)state->signals_dropped / (float)total_signals : 0.0f;

    state->queue_utilization = (float)router_stats.queue_depth /
        (float)THALAMIC_MAX_QUEUE_SIZE;

    /* Detect anomalies */
    state->queue_critical = (state->queue_utilization >= ROUTING_ANOMALY_QUEUE_THRESHOLD);
    state->excessive_drops = (state->drop_rate >= ROUTING_ANOMALY_DROP_THRESHOLD);
    state->high_latency = (state->avg_latency_ms >= ROUTING_ANOMALY_LATENCY_MS);
    state->priority_violations = (state->priority_drops >= ROUTING_IMMUNE_TRIGGER_DROPS);

    /* Count anomalies */
    uint32_t anomaly_count = 0;
    if (state->queue_critical) anomaly_count++;
    if (state->excessive_drops) anomaly_count++;
    if (state->high_latency) anomaly_count++;
    if (state->priority_violations) anomaly_count += 2;  /* Weighted higher */

    state->anomaly_count = anomaly_count;

    /* Compute threat severity */
    state->threat_severity = 0.0f;
    if (state->queue_critical) state->threat_severity += 0.2f;
    if (state->excessive_drops) state->threat_severity += 0.3f;
    if (state->high_latency) state->threat_severity += 0.2f;
    if (state->priority_violations) state->threat_severity += 0.4f;
    state->threat_severity = clamp_f(state->threat_severity, 0.0f, 1.0f);

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int thalamic_immune_trigger_from_anomaly(thalamic_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_routing_anomaly_detection) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    routing_anomaly_state_t* state = &bridge->anomaly_state;

    /* Trigger only if anomalies detected */
    if (state->anomaly_count == 0) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return 0;
    }

    /* Create antigen from routing anomaly */
    uint8_t epitope[BRAIN_IMMUNE_EPITOPE_SIZE];
    memset(epitope, 0, BRAIN_IMMUNE_EPITOPE_SIZE);

    /* Encode anomaly signature */
    epitope[0] = 0xFE;  /* Routing anomaly marker */
    epitope[1] = (uint8_t)(state->queue_utilization * 255);
    epitope[2] = (uint8_t)(state->drop_rate * 255);
    epitope[3] = (uint8_t)(state->avg_latency_ms / 100.0f * 255);
    epitope[4] = (uint8_t)state->anomaly_count;

    /* Severity based on threat level */
    uint32_t severity = (uint32_t)(state->threat_severity * 9.0f) + 1;  /* 1-10 */
    severity = (severity < 1) ? 1 : (severity > 10) ? 10 : severity;

    /* Present to immune system */
    uint32_t antigen_id;
    int result = brain_immune_present_antigen(
        bridge->immune_system,
        ANTIGEN_SOURCE_ANOMALY,
        epitope,
        5,  /* First 5 bytes encode anomaly */
        severity,
        0,  /* No specific node */
        &antigen_id
    );

    if (result == 0) {
        bridge->immune_triggered_anomalies++;
        LOG_MODULE_WARN("thalamic_immune_bridge",
                  "Routing anomaly triggered immune response (antigen %u, severity %u)",
                  antigen_id, severity);
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return result;
}

int thalamic_immune_boost_from_health(thalamic_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_health_feedback) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    routing_health_feedback_t* feedback = &bridge->health_feedback;
    routing_anomaly_state_t* anomaly = &bridge->anomaly_state;

    /* Compute health metrics */
    feedback->routing_efficiency = clamp_f(anomaly->throughput_hz / 1000.0f, 0.0f, 1.0f);
    feedback->queue_headroom = 1.0f - anomaly->queue_utilization;
    feedback->success_rate = 1.0f - anomaly->drop_rate;

    /* Overall health score */
    float health_score = (feedback->routing_efficiency +
                         feedback->queue_headroom +
                         feedback->success_rate) / 3.0f;

    /* Health above 0.7 triggers IL-10 boost */
    if (health_score > 0.7f) {
        feedback->il10_boost = (health_score - 0.7f) * 0.5f;
        feedback->inflammation_reduction = feedback->il10_boost * 0.8f;

        /* Trigger anti-inflammatory response in immune system */
        /* (Would call brain_immune_release_cytokine with IL-10) */

        bridge->health_boosts++;
    } else {
        feedback->il10_boost = 0.0f;
        feedback->inflammation_reduction = 0.0f;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

/* ============================================================================
 * Bidirectional Update Implementation
 * ============================================================================ */

int thalamic_immune_bridge_update(
    thalamic_immune_bridge_t* bridge,
    uint64_t delta_ms
) {
    /* Guard clauses */
    if (!bridge) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    bridge->total_updates++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    /* Immune → Routing */
    thalamic_immune_apply_cytokine_effects(bridge);
    thalamic_immune_apply_inflammation_effects(bridge);

    /* Routing → Immune */
    thalamic_immune_detect_anomalies(bridge);

    /* Trigger immune if anomalies severe */
    if (bridge->anomaly_state.threat_severity > 0.5f) {
        thalamic_immune_trigger_from_anomaly(bridge);
    }

    /* Boost from health */
    thalamic_immune_boost_from_health(bridge);

    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int thalamic_immune_get_cytokine_effects(
    const thalamic_immune_bridge_t* bridge,
    cytokine_routing_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    memcpy(effects, &bridge->cytokine_effects, sizeof(cytokine_routing_effects_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int thalamic_immune_get_inflammation_state(
    const thalamic_immune_bridge_t* bridge,
    inflammation_routing_state_t* state
) {
    if (!bridge || !state) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    memcpy(state, &bridge->inflammation_state, sizeof(inflammation_routing_state_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int thalamic_immune_get_anomaly_state(
    const thalamic_immune_bridge_t* bridge,
    routing_anomaly_state_t* state
) {
    if (!bridge || !state) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    memcpy(state, &bridge->anomaly_state, sizeof(routing_anomaly_state_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

bool thalamic_immune_is_hypervigilant(const thalamic_immune_bridge_t* bridge) {
    if (!bridge) return false;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    bool hypervigilant = (bridge->inflammation_state.hypervigilance_level > 0.5f) &&
                        (bridge->inflammation_state.current_level >= INFLAMMATION_REGIONAL);
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return hypervigilant;
}

float thalamic_immune_get_gating_threshold(const thalamic_immune_bridge_t* bridge) {
    if (!bridge) return 0.5f;  /* Default threshold */

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    float base_threshold = 0.5f;
    float threshold = base_threshold + bridge->cytokine_effects.gating_threshold_modifier;
    threshold -= bridge->inflammation_state.gating_reduction;
    threshold = clamp_f(threshold, 0.1f, 0.9f);

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return threshold;
}

float thalamic_immune_get_threat_priority_multiplier(
    const thalamic_immune_bridge_t* bridge
) {
    if (!bridge) return 1.0f;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    float multiplier = bridge->inflammation_state.threat_priority_boost;
    multiplier = clamp_f(multiplier, 1.0f, 2.0f);
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return multiplier;
}

/* ============================================================================
 * Bio-Async Integration Implementation
 * ============================================================================ */

#define THALAMIC_IMMUNE_MODULE_NAME "thalamic_immune_bridge"

/**
 * @brief Connect bridge to bio-async router
 */
int thalamic_immune_connect_bio_async(thalamic_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_IMMUNE_THALAMIC,
        .module_name = THALAMIC_IMMUNE_MODULE_NAME,
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("thalamic_immune_bridge connected to bio-async router");
    } else {
        NIMCP_LOGGING_INFO("Bio-async router not available, skipping registration");
    }

    return 0;
}

/**
 * @brief Disconnect from bio-async router
 */
int thalamic_immune_disconnect_bio_async(thalamic_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->bio_async_enabled) return 0;

    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }
    bridge->bio_async_enabled = false;

    NIMCP_LOGGING_DEBUG("thalamic_immune_bridge disconnected from bio-async router");
    return 0;
}

/**
 * @brief Check if bio-async is connected
 */
bool thalamic_immune_is_bio_async_connected(const thalamic_immune_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_enabled;
}
