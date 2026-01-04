/**
 * @file nimcp_hypothalamus_immune_bridge.c
 * @brief Hypothalamus-Immune System Bidirectional Bridge Implementation
 *
 * WHAT: Bidirectional integration between hypothalamus drives and brain immune system
 * WHY:  Implements neuroimmune crosstalk for sickness behavior and HPA axis regulation
 * HOW:  Cytokines modulate drive setpoints; HPA axis modulates immune intensity
 *
 * @version Phase 14: Brain Immune Integration
 * @date 2026-01-04
 */

#include "core/brain/regions/hypothalamus/nimcp_hypothalamus_immune_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Internal bridge state
 */
struct hypo_immune_bridge {
    /* External references */
    hypo_drive_system_handle_t* drives;
    brain_immune_system_t* immune;

    /* Configuration */
    hypo_immune_config_t config;

    /* State */
    hypo_cytokine_state_t cytokines;
    hypo_hpa_axis_t hpa;
    hypo_sickness_state_t sickness;
    hypo_circadian_immune_t circadian_phase;
    float circadian_modulation;

    /* Drive modulations (one per drive type) */
    hypo_immune_modulation_t modulations[HYPO_DRIVE_COUNT];

    /* Safety mode */
    bool in_safety_mode;
    float safety_mode_level;

    /* Bio-async */
    bio_module_context_t bio_ctx;
    bool bio_registered;

    /* Statistics */
    hypo_immune_bridge_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;
};

/*=============================================================================
 * FORWARD DECLARATIONS
 *===========================================================================*/

static nimcp_error_t immune_handle_cytokine_update(
    const void* msg, size_t msg_size, nimcp_bio_promise_t promise, void* ctx);
static nimcp_error_t immune_handle_inflammation(
    const void* msg, size_t msg_size, nimcp_bio_promise_t promise, void* ctx);
static nimcp_error_t immune_handle_antigen_detected(
    const void* msg, size_t msg_size, nimcp_bio_promise_t promise, void* ctx);

static void update_sickness_from_cytokines(hypo_immune_bridge_t* bridge);
static void compute_hpa_dynamics(hypo_immune_bridge_t* bridge, float dt);
static float compute_pro_inflammatory_total(const hypo_cytokine_state_t* cyt);

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

void hypo_immune_bridge_default_config(hypo_immune_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(*config));

    /* Cytokine sensitivity */
    config->fever_sensitivity = 1.0f;
    config->anorexia_sensitivity = 1.0f;
    config->fatigue_sensitivity = 1.0f;

    /* HPA axis parameters */
    config->cortisol_baseline = 0.2f;
    config->stress_threshold = 0.3f;
    config->recovery_rate = 0.05f;

    /* Circadian modulation */
    config->circadian_enabled = true;
    config->circadian_amplitude = 0.3f;

    /* Sickness behavior thresholds */
    config->mild_threshold = 0.2f;
    config->moderate_threshold = 0.4f;
    config->severe_threshold = 0.7f;
    config->storm_threshold = 0.9f;

    /* Alignment integration */
    config->use_as_safety_mode = true;
    config->safety_trigger = 0.6f;
}

hypo_immune_bridge_t* hypo_immune_bridge_create(
    hypo_drive_system_handle_t* drives,
    brain_immune_system_t* immune,
    const hypo_immune_config_t* config)
{
    if (!drives) {
        nimcp_log(LOG_LEVEL_ERROR, "hypo_immune_bridge_create: drives is NULL");
        return NULL;
    }

    hypo_immune_bridge_t* bridge = nimcp_calloc(1, sizeof(hypo_immune_bridge_t));
    if (!bridge) {
        nimcp_log(LOG_LEVEL_ERROR, "hypo_immune_bridge_create: allocation failed");
        return NULL;
    }

    bridge->drives = drives;
    bridge->immune = immune;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        hypo_immune_bridge_default_config(&bridge->config);
    }

    /* Initialize HPA axis */
    bridge->hpa.state = HPA_BASELINE;
    bridge->hpa.cortisol_level = bridge->config.cortisol_baseline;
    bridge->hpa.crh_level = 0.1f;
    bridge->hpa.acth_level = 0.1f;

    /* Initialize sickness state */
    bridge->sickness.level = SICKNESS_NONE;

    /* Initialize drive modulations */
    for (int i = 0; i < HYPO_DRIVE_COUNT; i++) {
        bridge->modulations[i].drive = (hypo_drive_type_t)i;
        bridge->modulations[i].baseline_setpoint = 0.5f;
        bridge->modulations[i].modulated_setpoint = 0.5f;
        bridge->modulations[i].modulation_factor = 1.0f;
        bridge->modulations[i].is_modulated = false;
    }

    /* Create mutex */
    mutex_attr_t attr;
    attr.type = MUTEX_TYPE_RECURSIVE;
    bridge->mutex = nimcp_mutex_create(&attr);
    if (!bridge->mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "hypo_immune_bridge_create: mutex creation failed");
        nimcp_free(bridge);
        return NULL;
    }

    nimcp_log(LOG_LEVEL_INFO, "hypo_immune_bridge: created successfully");
    return bridge;
}

void hypo_immune_bridge_destroy(hypo_immune_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->bio_registered) {
        hypo_immune_bridge_unregister_bio(bridge);
    }

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    nimcp_log(LOG_LEVEL_INFO, "hypo_immune_bridge: destroyed");
}

/*=============================================================================
 * CYTOKINE → HYPOTHALAMUS (Immune → Drive Modulation)
 *===========================================================================*/

int hypo_immune_bridge_update_cytokines(
    hypo_immune_bridge_t* bridge,
    const hypo_cytokine_state_t* cytokines)
{
    if (!bridge || !cytokines) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->cytokines = *cytokines;
    bridge->cytokines.last_update_us = nimcp_time_get_us();
    bridge->stats.cytokine_updates++;

    /* Update sickness state based on new cytokine levels */
    update_sickness_from_cytokines(bridge);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int hypo_immune_bridge_apply_cytokine_effects(hypo_immune_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    const hypo_cytokine_state_t* cyt = &bridge->cytokines;

    /* 1. Fever: IL-1β + IL-6 → Temperature setpoint ↑ */
    float fever_signal = (cyt->il1_beta + cyt->il6) * 0.5f;
    fever_signal *= bridge->config.fever_sensitivity;
    bridge->sickness.fever_magnitude = fminf(fever_signal * HYPO_IMMUNE_FEVER_FACTOR, 1.0f);

    if (bridge->sickness.fever_magnitude > 0.1f) {
        /* Increase temperature drive setpoint */
        hypo_drive_set_nucleus_input(bridge->drives,
            HYPO_NUCLEUS_ANTERIOR, bridge->sickness.fever_magnitude);
        bridge->stats.fever_episodes++;
    }

    /* 2. Anorexia: TNF-α → Hunger setpoint ↓ */
    float anorexia_signal = cyt->tnf_alpha * bridge->config.anorexia_sensitivity;
    bridge->sickness.anorexia_magnitude = fminf(anorexia_signal * HYPO_IMMUNE_ANOREXIA_FACTOR, 1.0f);

    if (bridge->sickness.anorexia_magnitude > 0.1f) {
        /* Decrease hunger drive via lateral hypothalamus */
        hypo_drive_set_nucleus_input(bridge->drives,
            HYPO_NUCLEUS_LATERAL, -bridge->sickness.anorexia_magnitude);
        bridge->stats.anorexia_episodes++;
    }

    /* 3. Fatigue: Pro-inflammatory cytokines → Fatigue ↑ */
    float pro_inflammatory = compute_pro_inflammatory_total(cyt);
    float fatigue_signal = pro_inflammatory * bridge->config.fatigue_sensitivity;
    bridge->sickness.fatigue_magnitude = fminf(fatigue_signal * HYPO_IMMUNE_FATIGUE_FACTOR, 1.0f);

    if (bridge->sickness.fatigue_magnitude > 0.1f) {
        /* Increase fatigue via preoptic area */
        hypo_drive_set_nucleus_input(bridge->drives,
            HYPO_NUCLEUS_PREOPTIC, bridge->sickness.fatigue_magnitude);
    }

    /* 4. Social withdrawal: High inflammation → Social drive ↓ */
    if (pro_inflammatory > 0.5f) {
        bridge->sickness.social_withdrawal = (pro_inflammatory - 0.5f) * HYPO_IMMUNE_SOCIAL_WITHDRAW_FACTOR;
        /* Suppress social via PVN (reduce oxytocin drive) */
        hypo_drive_set_nucleus_input(bridge->drives,
            HYPO_NUCLEUS_PARAVENTRICULAR, -bridge->sickness.social_withdrawal * 0.5f);
    }

    /* 5. Curiosity reduction: Sickness → Exploration ↓ */
    if (bridge->sickness.level >= SICKNESS_MODERATE) {
        bridge->sickness.curiosity_reduction = 0.3f + 0.2f * (bridge->sickness.level - SICKNESS_MODERATE);
        /* This would be applied via drive system direct modulation */
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float hypo_immune_bridge_get_fever(const hypo_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->sickness.fever_magnitude;
}

int hypo_immune_bridge_get_sickness_state(
    const hypo_immune_bridge_t* bridge,
    hypo_sickness_state_t* state)
{
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(((hypo_immune_bridge_t*)bridge)->mutex);
    *state = bridge->sickness;
    nimcp_mutex_unlock(((hypo_immune_bridge_t*)bridge)->mutex);
    return 0;
}

/*=============================================================================
 * HYPOTHALAMUS → IMMUNE (HPA Axis → Immune Suppression)
 *===========================================================================*/

int hypo_immune_bridge_update_hpa(hypo_immune_bridge_t* bridge, float stress_input) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    uint64_t now = nimcp_time_get_us();
    float dt = (bridge->hpa.last_update_us > 0) ?
        (now - bridge->hpa.last_update_us) / 1000000.0f : 0.0f;
    bridge->hpa.last_update_us = now;

    /* Update stress levels */
    bridge->hpa.acute_stress = stress_input;

    /* Accumulate chronic stress with decay */
    const float chronic_accumulation = 0.001f;
    const float chronic_decay = 0.0005f;
    if (stress_input > bridge->config.stress_threshold) {
        bridge->hpa.chronic_stress += chronic_accumulation * dt;
        bridge->hpa.chronic_stress = fminf(bridge->hpa.chronic_stress, 1.0f);
    } else {
        bridge->hpa.chronic_stress -= chronic_decay * dt;
        bridge->hpa.chronic_stress = fmaxf(bridge->hpa.chronic_stress, 0.0f);
    }

    /* State transitions based on stress */
    if (bridge->hpa.chronic_stress > 0.7f) {
        if (bridge->hpa.state != HPA_CHRONIC_STRESS) {
            bridge->hpa.state = HPA_CHRONIC_STRESS;
        }
    } else if (stress_input > bridge->config.stress_threshold) {
        if (bridge->hpa.state == HPA_BASELINE) {
            bridge->hpa.state = HPA_ACUTE_STRESS;
            bridge->hpa.stress_onset_us = now;
            bridge->stats.stress_activations++;
        } else if (bridge->hpa.state == HPA_ACUTE_STRESS) {
            /* Check for prolonged stress */
            if ((now - bridge->hpa.stress_onset_us) > 300000000ULL) { /* 5 minutes */
                bridge->hpa.state = HPA_PROLONGED_STRESS;
            }
        }
    } else if (bridge->hpa.state != HPA_BASELINE) {
        bridge->hpa.state = HPA_RECOVERY;
    }

    /* Compute HPA dynamics */
    compute_hpa_dynamics(bridge, dt);

    /* Recovery to baseline */
    if (bridge->hpa.state == HPA_RECOVERY) {
        if (bridge->hpa.cortisol_level <= bridge->config.cortisol_baseline * 1.1f) {
            bridge->hpa.state = HPA_BASELINE;
            bridge->stats.recovery_cycles++;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float hypo_immune_bridge_get_immune_suppression(const hypo_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;

    /* Cortisol above baseline causes immune suppression */
    float excess_cortisol = bridge->hpa.cortisol_level - bridge->config.cortisol_baseline;
    if (excess_cortisol <= 0.0f) return 0.0f;

    return fminf(excess_cortisol * HYPO_IMMUNE_CORTISOL_SUPPRESSION, 1.0f);
}

int hypo_immune_bridge_get_hpa_state(
    const hypo_immune_bridge_t* bridge,
    hypo_hpa_axis_t* hpa)
{
    if (!bridge || !hpa) return -1;

    nimcp_mutex_lock(((hypo_immune_bridge_t*)bridge)->mutex);
    *hpa = bridge->hpa;
    nimcp_mutex_unlock(((hypo_immune_bridge_t*)bridge)->mutex);
    return 0;
}

int hypo_immune_bridge_apply_cortisol_effects(hypo_immune_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float suppression = hypo_immune_bridge_get_immune_suppression(bridge);

    /* Update peak cortisol tracking */
    if (bridge->hpa.cortisol_level > bridge->stats.peak_cortisol) {
        bridge->stats.peak_cortisol = bridge->hpa.cortisol_level;
    }

    /* Broadcast cortisol state if significant */
    if (suppression > 0.1f && bridge->bio_registered) {
        hypo_immune_bridge_broadcast_cortisol(bridge);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

/*=============================================================================
 * CIRCADIAN → IMMUNE MODULATION
 *===========================================================================*/

int hypo_immune_bridge_update_circadian(
    hypo_immune_bridge_t* bridge,
    float scn_phase)
{
    if (!bridge) return -1;
    if (!bridge->config.circadian_enabled) return 0;

    nimcp_mutex_lock(bridge->mutex);

    /* Normalize phase to [0, 2π] */
    while (scn_phase < 0.0f) scn_phase += 2.0f * 3.14159265f;
    while (scn_phase >= 2.0f * 3.14159265f) scn_phase -= 2.0f * 3.14159265f;

    /* Determine circadian immune phase */
    float normalized = scn_phase / (2.0f * 3.14159265f);
    if (normalized < 0.25f) {
        bridge->circadian_phase = CIRCADIAN_IMMUNE_MORNING;
    } else if (normalized < 0.5f) {
        bridge->circadian_phase = CIRCADIAN_IMMUNE_DAY;
    } else if (normalized < 0.75f) {
        bridge->circadian_phase = CIRCADIAN_IMMUNE_EVENING;
    } else {
        bridge->circadian_phase = CIRCADIAN_IMMUNE_NIGHT;
    }

    /* Compute modulation factor */
    /* Peak inflammation potential in morning, lowest at night */
    bridge->circadian_modulation = 1.0f + bridge->config.circadian_amplitude *
        cosf(scn_phase - 0.5f * 3.14159265f);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

hypo_circadian_immune_t hypo_immune_bridge_get_circadian_phase(
    const hypo_immune_bridge_t* bridge)
{
    if (!bridge) return CIRCADIAN_IMMUNE_DAY;
    return bridge->circadian_phase;
}

float hypo_immune_bridge_get_circadian_modulation(const hypo_immune_bridge_t* bridge) {
    if (!bridge) return 1.0f;
    return bridge->circadian_modulation;
}

/*=============================================================================
 * SICKNESS BEHAVIOR
 *===========================================================================*/

hypo_sickness_level_t hypo_immune_bridge_compute_sickness_level(
    hypo_immune_bridge_t* bridge)
{
    if (!bridge) return SICKNESS_NONE;

    nimcp_mutex_lock(bridge->mutex);

    float inflammation = bridge->sickness.inflammation_level;
    hypo_sickness_level_t level;

    if (inflammation >= bridge->config.storm_threshold) {
        level = SICKNESS_CRITICAL;
        bridge->sickness.cytokine_storm = true;
    } else if (inflammation >= bridge->config.severe_threshold) {
        level = SICKNESS_SEVERE;
    } else if (inflammation >= bridge->config.moderate_threshold) {
        level = SICKNESS_MODERATE;
    } else if (inflammation >= bridge->config.mild_threshold) {
        level = SICKNESS_MILD;
    } else {
        level = SICKNESS_NONE;
    }

    if (level != bridge->sickness.level) {
        if (level > SICKNESS_NONE && bridge->sickness.level == SICKNESS_NONE) {
            bridge->sickness.onset_us = nimcp_time_get_us();
            bridge->stats.sickness_episodes++;
        }
        if (level == SICKNESS_CRITICAL) {
            bridge->stats.cytokine_storms++;
        }
        bridge->sickness.level = level;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return level;
}

int hypo_immune_bridge_apply_sickness_behavior(hypo_immune_bridge_t* bridge) {
    if (!bridge) return -1;

    /* First update sickness level */
    hypo_immune_bridge_compute_sickness_level(bridge);

    /* Then apply cytokine effects */
    return hypo_immune_bridge_apply_cytokine_effects(bridge);
}

bool hypo_immune_bridge_is_cytokine_storm(const hypo_immune_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->sickness.cytokine_storm;
}

int hypo_immune_bridge_enter_safety_mode(
    hypo_immune_bridge_t* bridge,
    float threat_level)
{
    if (!bridge) return -1;
    if (!bridge->config.use_as_safety_mode) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->in_safety_mode = true;
    bridge->safety_mode_level = fminf(threat_level, 1.0f);
    bridge->stats.safety_mode_entries++;

    /* Simulate sickness behavior for safety mode */
    /* Reduce exploration (curiosity) */
    bridge->sickness.curiosity_reduction = 0.5f + 0.5f * threat_level;

    /* Increase caution (safety drive up) */
    hypo_drive_set_nucleus_input(bridge->drives,
        HYPO_NUCLEUS_VENTROMEDIAL, threat_level * 0.5f);

    /* Reduce social engagement in severe cases */
    if (threat_level > 0.7f) {
        bridge->sickness.social_withdrawal = (threat_level - 0.5f) * 0.5f;
    }

    nimcp_log(LOG_LEVEL_WARN, "hypo_immune_bridge: entered safety mode (threat=%.2f)", threat_level);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int hypo_immune_bridge_exit_safety_mode(hypo_immune_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->in_safety_mode = false;
    bridge->safety_mode_level = 0.0f;

    /* Reset safety-induced modulations */
    bridge->sickness.curiosity_reduction = 0.0f;
    bridge->sickness.social_withdrawal = 0.0f;

    /* Reset nucleus inputs */
    hypo_drive_set_nucleus_input(bridge->drives, HYPO_NUCLEUS_VENTROMEDIAL, 0.0f);

    nimcp_log(LOG_LEVEL_INFO, "hypo_immune_bridge: exited safety mode");

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

/*=============================================================================
 * BIO-ASYNC INTEGRATION
 *===========================================================================*/

bool hypo_immune_bridge_register_bio(
    hypo_immune_bridge_t* bridge,
    bool use_kg_wiring)
{
    if (!bridge) return false;
    if (bridge->bio_registered) return true;

    (void)use_kg_wiring; /* Future KG wiring integration */

    bio_module_info_t info = {
        .module_id = HYPO_IMMUNE_BRIDGE_MODULE_ID,
        .module_name = "hypothalamus_immune_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (!bridge->bio_ctx) {
        nimcp_log(LOG_LEVEL_ERROR, "hypo_immune_bridge: failed to register with bio-router");
        return false;
    }

    /* Register handlers for immune messages */
    bio_router_register_handler(bridge->bio_ctx,
        BIO_MSG_CYTOKINE_UPDATE, immune_handle_cytokine_update);
    bio_router_register_handler(bridge->bio_ctx,
        BIO_MSG_INFLAMMATION_CHANGE, immune_handle_inflammation);
    bio_router_register_handler(bridge->bio_ctx,
        BIO_MSG_ANTIGEN_DETECTED, immune_handle_antigen_detected);

    bridge->bio_registered = true;
    nimcp_log(LOG_LEVEL_INFO, "hypo_immune_bridge: registered with bio-router");
    return true;
}

void hypo_immune_bridge_unregister_bio(hypo_immune_bridge_t* bridge) {
    if (!bridge || !bridge->bio_registered) return;

    bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_registered = false;
    nimcp_log(LOG_LEVEL_INFO, "hypo_immune_bridge: unregistered from bio-router");
}

nimcp_error_t hypo_immune_bridge_broadcast_fever(hypo_immune_bridge_t* bridge) {
    if (!bridge || !bridge->bio_registered) return NIMCP_ERROR_INVALID_PARAM;

    struct {
        bio_message_header_t header;
        float fever_magnitude;
        float temperature_setpoint_delta;
    } msg;

    msg.header.type = BIO_MSG_HYPO_IMMUNE_FEVER_STATE;
    msg.header.timestamp_us = nimcp_time_get_us();
    msg.header.source_module = HYPO_IMMUNE_BRIDGE_MODULE_ID;
    msg.header.target_module = 0;  /* Broadcast */
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.payload_size = sizeof(msg) - sizeof(bio_message_header_t);

    msg.fever_magnitude = bridge->sickness.fever_magnitude;
    msg.temperature_setpoint_delta = bridge->sickness.fever_magnitude * 2.0f; /* °C */

    bridge->stats.messages_sent++;
    return bio_router_broadcast(bridge->bio_ctx, &msg.header, sizeof(msg));
}

nimcp_error_t hypo_immune_bridge_broadcast_sickness(hypo_immune_bridge_t* bridge) {
    if (!bridge || !bridge->bio_registered) return NIMCP_ERROR_INVALID_PARAM;

    struct {
        bio_message_header_t header;
        uint8_t sickness_level;
        float inflammation;
        uint8_t cytokine_storm;
    } msg;

    msg.header.type = BIO_MSG_HYPO_IMMUNE_SICKNESS_STATE;
    msg.header.timestamp_us = nimcp_time_get_us();
    msg.header.source_module = HYPO_IMMUNE_BRIDGE_MODULE_ID;
    msg.header.target_module = 0;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.payload_size = sizeof(msg) - sizeof(bio_message_header_t);

    msg.sickness_level = (uint8_t)bridge->sickness.level;
    msg.inflammation = bridge->sickness.inflammation_level;
    msg.cytokine_storm = bridge->sickness.cytokine_storm ? 1 : 0;

    bridge->stats.messages_sent++;
    return bio_router_broadcast(bridge->bio_ctx, &msg.header, sizeof(msg));
}

nimcp_error_t hypo_immune_bridge_broadcast_cortisol(hypo_immune_bridge_t* bridge) {
    if (!bridge || !bridge->bio_registered) return NIMCP_ERROR_INVALID_PARAM;

    struct {
        bio_message_header_t header;
        float cortisol_level;
        float immune_suppression;
        uint8_t hpa_state;
    } msg;

    msg.header.type = BIO_MSG_HYPO_IMMUNE_CORTISOL_STATE;
    msg.header.timestamp_us = nimcp_time_get_us();
    msg.header.source_module = HYPO_IMMUNE_BRIDGE_MODULE_ID;
    msg.header.target_module = 0;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.payload_size = sizeof(msg) - sizeof(bio_message_header_t);

    msg.cortisol_level = bridge->hpa.cortisol_level;
    msg.immune_suppression = hypo_immune_bridge_get_immune_suppression(bridge);
    msg.hpa_state = (uint8_t)bridge->hpa.state;

    bridge->stats.messages_sent++;
    return bio_router_broadcast(bridge->bio_ctx, &msg.header, sizeof(msg));
}

/*=============================================================================
 * STATISTICS
 *===========================================================================*/

int hypo_immune_bridge_get_stats(
    const hypo_immune_bridge_t* bridge,
    hypo_immune_bridge_stats_t* stats)
{
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(((hypo_immune_bridge_t*)bridge)->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(((hypo_immune_bridge_t*)bridge)->mutex);
    return 0;
}

void hypo_immune_bridge_reset_stats(hypo_immune_bridge_t* bridge) {
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->mutex);
}

/*=============================================================================
 * INTERNAL HELPERS
 *===========================================================================*/

static float compute_pro_inflammatory_total(const hypo_cytokine_state_t* cyt) {
    if (!cyt) return 0.0f;

    /* Weight pro-inflammatory cytokines */
    return 0.3f * cyt->il1_beta +
           0.3f * cyt->tnf_alpha +
           0.2f * cyt->il6 +
           0.2f * cyt->ifn_gamma;
}

static void update_sickness_from_cytokines(hypo_immune_bridge_t* bridge) {
    float pro_inflam = compute_pro_inflammatory_total(&bridge->cytokines);
    float anti_inflam = bridge->cytokines.il10 + bridge->cytokines.tgf_beta;

    /* Net inflammation */
    bridge->sickness.inflammation_level = fmaxf(0.0f,
        pro_inflam - 0.5f * anti_inflam);
    bridge->sickness.inflammation_level = fminf(bridge->sickness.inflammation_level, 1.0f);
}

static void compute_hpa_dynamics(hypo_immune_bridge_t* bridge, float dt) {
    if (dt <= 0.0f) return;

    const float crh_tau = 5.0f;  /* CRH time constant (seconds) */
    const float acth_tau = 10.0f;
    const float cortisol_tau = 30.0f;

    float stress = bridge->hpa.acute_stress;

    /* CRH dynamics: driven by stress */
    float crh_target = stress;
    bridge->hpa.crh_level += (crh_target - bridge->hpa.crh_level) * (dt / crh_tau);

    /* ACTH dynamics: driven by CRH */
    float acth_target = bridge->hpa.crh_level * 0.8f;
    bridge->hpa.acth_level += (acth_target - bridge->hpa.acth_level) * (dt / acth_tau);

    /* Cortisol dynamics: driven by ACTH, with recovery */
    float cortisol_target = bridge->config.cortisol_baseline + bridge->hpa.acth_level * 0.6f;
    if (bridge->hpa.state == HPA_RECOVERY) {
        cortisol_target = bridge->config.cortisol_baseline;
    }
    bridge->hpa.cortisol_level += (cortisol_target - bridge->hpa.cortisol_level) *
        (dt / cortisol_tau * (bridge->hpa.state == HPA_RECOVERY ? 2.0f : 1.0f));

    /* Clamp values */
    bridge->hpa.crh_level = fmaxf(0.0f, fminf(bridge->hpa.crh_level, 1.0f));
    bridge->hpa.acth_level = fmaxf(0.0f, fminf(bridge->hpa.acth_level, 1.0f));
    bridge->hpa.cortisol_level = fmaxf(0.0f, fminf(bridge->hpa.cortisol_level, 1.0f));
}

/*=============================================================================
 * BIO-ASYNC HANDLERS
 *===========================================================================*/

static nimcp_error_t immune_handle_cytokine_update(
    const void* msg, size_t msg_size, nimcp_bio_promise_t promise, void* ctx)
{
    hypo_immune_bridge_t* bridge = (hypo_immune_bridge_t*)ctx;
    (void)msg_size;
    (void)promise;

    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    const struct {
        bio_message_header_t header;
        uint8_t cytokine_type;
        float concentration;
    }* cytokine_msg = msg;

    nimcp_mutex_lock(bridge->mutex);

    /* Update appropriate cytokine level */
    switch (cytokine_msg->cytokine_type) {
        case CYTOKINE_IL1B:
            bridge->cytokines.il1_beta = cytokine_msg->concentration;
            break;
        case CYTOKINE_IL6:
            bridge->cytokines.il6 = cytokine_msg->concentration;
            break;
        case CYTOKINE_TNFA:
            bridge->cytokines.tnf_alpha = cytokine_msg->concentration;
            break;
        case CYTOKINE_IL10:
            bridge->cytokines.il10 = cytokine_msg->concentration;
            break;
        case CYTOKINE_TGFB:
            bridge->cytokines.tgf_beta = cytokine_msg->concentration;
            break;
        default:
            break;
    }

    bridge->cytokines.last_update_us = nimcp_time_get_us();
    bridge->stats.messages_received++;

    update_sickness_from_cytokines(bridge);

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

static nimcp_error_t immune_handle_inflammation(
    const void* msg, size_t msg_size, nimcp_bio_promise_t promise, void* ctx)
{
    hypo_immune_bridge_t* bridge = (hypo_immune_bridge_t*)ctx;
    (void)msg_size;
    (void)promise;

    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    const struct {
        bio_message_header_t header;
        float inflammation_level;
        uint32_t region_id;
    }* inflam_msg = msg;

    nimcp_mutex_lock(bridge->mutex);

    /* Update inflammation level directly */
    bridge->sickness.inflammation_level = fmaxf(bridge->sickness.inflammation_level,
        inflam_msg->inflammation_level);
    bridge->stats.messages_received++;

    /* Recompute sickness level */
    hypo_immune_bridge_compute_sickness_level(bridge);

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

static nimcp_error_t immune_handle_antigen_detected(
    const void* msg, size_t msg_size, nimcp_bio_promise_t promise, void* ctx)
{
    hypo_immune_bridge_t* bridge = (hypo_immune_bridge_t*)ctx;
    (void)msg_size;
    (void)promise;

    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    const struct {
        bio_message_header_t header;
        uint32_t antigen_id;
        float severity;
    }* antigen_msg = msg;

    bridge->stats.messages_received++;

    /* High-severity antigens trigger acute stress response */
    if (antigen_msg->severity > 0.7f) {
        hypo_immune_bridge_update_hpa(bridge, antigen_msg->severity);
    }

    return NIMCP_SUCCESS;
}
