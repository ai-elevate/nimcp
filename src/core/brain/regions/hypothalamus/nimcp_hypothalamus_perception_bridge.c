/**
 * @file nimcp_hypothalamus_perception_bridge.c
 * @brief Implementation of Hypothalamus-Perception Bridge for Sensory Modulation
 *
 * WHAT: Bidirectional integration between hypothalamus drives and sensory cortices
 * WHY:  Arousal and drives modulate sensory processing - hungry animals see food more
 * HOW:  Arousal → sensory gain; drive urgency → stimulus salience boost
 *
 * @version Phase 17: Perception & Speech Modulation
 * @date 2026-01-04
 */

#include "core/brain/regions/hypothalamus/nimcp_hypothalamus_perception_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include <string.h>
#include <math.h>

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

struct hypo_perception_bridge {
    hypo_drive_system_handle_t* drives;
    hypo_perception_config_t config;

    /* Current modulation state */
    hypo_perception_modulation_t modulation;

    /* Arousal from external source (brainstem/LC) */
    float external_arousal;

    /* Anticipation state per drive */
    float anticipation[HYPO_DRIVE_COUNT];

    /* Recent detection history */
    hypo_sensory_detection_t last_detection;
    bool has_recent_detection;

    /* Bio-async registration */
    bool bio_registered;
    bio_module_context_t bio_ctx;
};

/*=============================================================================
 * HELPER FUNCTIONS
 *===========================================================================*/

static float clamp_01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static float clamp_range(float v, float min_v, float max_v) {
    if (v < min_v) return min_v;
    if (v > max_v) return max_v;
    return v;
}

/**
 * @brief Map stimulus category to corresponding drive
 */
static hypo_drive_type_t category_to_drive(hypo_stim_category_t category) {
    switch (category) {
        case HYPO_STIM_FOOD:    return HYPO_DRIVE_HUNGER;
        case HYPO_STIM_WATER:   return HYPO_DRIVE_THIRST;
        case HYPO_STIM_THREAT:  return HYPO_DRIVE_SAFETY;
        case HYPO_STIM_SOCIAL:  return HYPO_DRIVE_SOCIAL;
        case HYPO_STIM_NOVEL:   return HYPO_DRIVE_CURIOSITY;
        case HYPO_STIM_THERMAL: return HYPO_DRIVE_TEMPERATURE;
        case HYPO_STIM_PAIN:    return HYPO_DRIVE_SAFETY;
        case HYPO_STIM_REWARD:  return HYPO_DRIVE_COMPETENCE;
        default:                return HYPO_DRIVE_COUNT;  /* Invalid */
    }
}

/**
 * @brief Map drive to primary stimulus category
 */
static hypo_stim_category_t drive_to_category(hypo_drive_type_t drive) {
    switch (drive) {
        case HYPO_DRIVE_HUNGER:      return HYPO_STIM_FOOD;
        case HYPO_DRIVE_THIRST:      return HYPO_STIM_WATER;
        case HYPO_DRIVE_SAFETY:      return HYPO_STIM_THREAT;
        case HYPO_DRIVE_SOCIAL:      return HYPO_STIM_SOCIAL;
        case HYPO_DRIVE_CURIOSITY:   return HYPO_STIM_NOVEL;
        case HYPO_DRIVE_TEMPERATURE: return HYPO_STIM_THERMAL;
        case HYPO_DRIVE_COMPETENCE:  return HYPO_STIM_REWARD;
        default:                     return HYPO_STIM_NEUTRAL;
    }
}

/*=============================================================================
 * LIFECYCLE
 *===========================================================================*/

void hypo_perception_bridge_default_config(hypo_perception_config_t* config) {
    if (!config) return;

    config->arousal_gain_min = 0.7f;
    config->arousal_gain_max = 1.5f;
    config->drive_salience_weight = 0.8f;
    config->threat_priority_threshold = 0.5f;
    config->visual_weight = 1.0f;
    config->auditory_weight = 1.0f;
    config->enable_anticipation = true;
    config->anticipation_decay = 0.05f;
}

hypo_perception_bridge_t* hypo_perception_bridge_create(
    hypo_drive_system_handle_t* drives,
    const hypo_perception_config_t* config)
{
    if (!drives) {
        nimcp_log(LOG_LEVEL_ERROR, "hypo_perception_bridge_create: drives is NULL");
        return NULL;
    }

    hypo_perception_bridge_t* bridge = nimcp_calloc(1, sizeof(hypo_perception_bridge_t));
    if (!bridge) {
        nimcp_log(LOG_LEVEL_ERROR, "hypo_perception_bridge_create: allocation failed");
        return NULL;
    }

    bridge->drives = drives;

    if (config) {
        bridge->config = *config;
    } else {
        hypo_perception_bridge_default_config(&bridge->config);
    }

    /* Initialize modulation to baseline */
    bridge->modulation.global_gain = 1.0f;
    bridge->modulation.arousal_level = 0.5f;
    for (int i = 0; i < HYPO_SENSE_COUNT; i++) {
        bridge->modulation.modality_gains[i] = 1.0f;
    }
    for (int i = 0; i < HYPO_STIM_COUNT; i++) {
        bridge->modulation.category_salience[i] = 1.0f;
    }
    bridge->modulation.threat_priority = false;
    bridge->modulation.survival_mode = false;

    bridge->external_arousal = 0.5f;

    /* Initialize anticipation to zero */
    for (int i = 0; i < HYPO_DRIVE_COUNT; i++) {
        bridge->anticipation[i] = 0.0f;
    }

    bridge->has_recent_detection = false;
    bridge->bio_registered = false;
    bridge->bio_ctx = NULL;

    nimcp_log(LOG_LEVEL_INFO, "hypo_perception_bridge: created successfully");
    return bridge;
}

void hypo_perception_bridge_destroy(hypo_perception_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->bio_registered) {
        hypo_perception_bridge_unregister_bio(bridge);
    }

    nimcp_free(bridge);
    nimcp_log(LOG_LEVEL_INFO, "hypo_perception_bridge: destroyed");
}

/*=============================================================================
 * DRIVE → PERCEPTION MODULATION
 *===========================================================================*/

int hypo_perception_bridge_compute_modulation(hypo_perception_bridge_t* bridge) {
    if (!bridge || !bridge->drives) return -1;

    const hypo_perception_config_t* cfg = &bridge->config;
    hypo_perception_modulation_t* mod = &bridge->modulation;

    /* Get current drive states */
    hypo_drive_system_t drive_state;
    if (!hypo_drive_get_system_state(bridge->drives, &drive_state)) {
        return -1;
    }

    /*
     * AROUSAL → GLOBAL SENSORY GAIN:
     *
     * Arousal (from LC norepinephrine) increases overall sensory responsiveness.
     * Low arousal = reduced sensitivity (drowsy state)
     * High arousal = heightened sensitivity (alert state)
     */
    float arousal = clamp_01(bridge->external_arousal);
    mod->arousal_level = arousal;

    /* Linear interpolation between min and max gain */
    mod->global_gain = cfg->arousal_gain_min +
                       arousal * (cfg->arousal_gain_max - cfg->arousal_gain_min);

    /*
     * DRIVE URGENCY → CATEGORY SALIENCE:
     *
     * High hunger → food stimuli more salient
     * High thirst → water stimuli more salient
     * High safety → threat stimuli more salient (and prioritized)
     * etc.
     */

    /* Reset category salience to baseline */
    for (int i = 0; i < HYPO_STIM_COUNT; i++) {
        mod->category_salience[i] = 1.0f;
    }

    /* Boost salience for each drive's relevant category */
    for (int d = 0; d < HYPO_DRIVE_COUNT; d++) {
        float urgency = drive_state.drives[d].urgency;
        hypo_stim_category_t cat = drive_to_category((hypo_drive_type_t)d);

        if (cat < HYPO_STIM_COUNT) {
            /* Salience boost proportional to urgency */
            float boost = 1.0f + urgency * cfg->drive_salience_weight;
            mod->category_salience[cat] = clamp_range(boost, 1.0f, 2.0f);
        }
    }

    /* Safety drive special handling: threat priority */
    float safety_urgency = drive_state.drives[HYPO_DRIVE_SAFETY].urgency;
    mod->threat_priority = (safety_urgency > cfg->threat_priority_threshold);

    /* Pain stimuli also get boosted when safety is high */
    if (safety_urgency > 0.3f) {
        mod->category_salience[HYPO_STIM_PAIN] = clamp_range(
            1.0f + safety_urgency * 1.5f, 1.0f, 2.5f);
    }

    /*
     * SURVIVAL MODE:
     *
     * When any critical drive is extremely urgent, enter survival mode
     * with maximum sensory gain for relevant stimuli.
     */
    bool survival = false;
    for (int d = 0; d < HYPO_DRIVE_COUNT; d++) {
        if (drive_state.drives[d].urgency > 0.9f) {
            /* Critical drive = physiological or safety */
            if (d == HYPO_DRIVE_HUNGER || d == HYPO_DRIVE_THIRST ||
                d == HYPO_DRIVE_SAFETY || d == HYPO_DRIVE_TEMPERATURE) {
                survival = true;
                break;
            }
        }
    }
    mod->survival_mode = survival;

    if (survival) {
        /* Amplify global gain in survival mode */
        mod->global_gain = cfg->arousal_gain_max;
        nimcp_log(LOG_LEVEL_DEBUG,
            "hypo_perception_bridge: survival mode active, max sensory gain");
    }

    /*
     * MODALITY-SPECIFIC GAINS:
     *
     * Different modalities may be weighted differently based on context.
     * Visual and auditory are primary for most threat/food detection.
     */
    mod->modality_gains[HYPO_SENSE_VISUAL] = mod->global_gain * cfg->visual_weight;
    mod->modality_gains[HYPO_SENSE_AUDITORY] = mod->global_gain * cfg->auditory_weight;
    mod->modality_gains[HYPO_SENSE_SOMATOSENSORY] = mod->global_gain;
    mod->modality_gains[HYPO_SENSE_OLFACTORY] = mod->global_gain;
    mod->modality_gains[HYPO_SENSE_GUSTATORY] = mod->global_gain;

    /* Hunger boosts gustatory and olfactory */
    float hunger_urgency = drive_state.drives[HYPO_DRIVE_HUNGER].urgency;
    if (hunger_urgency > 0.3f) {
        mod->modality_gains[HYPO_SENSE_GUSTATORY] *= (1.0f + hunger_urgency * 0.5f);
        mod->modality_gains[HYPO_SENSE_OLFACTORY] *= (1.0f + hunger_urgency * 0.3f);
    }

    /* Decay anticipation over time */
    if (cfg->enable_anticipation) {
        for (int i = 0; i < HYPO_DRIVE_COUNT; i++) {
            bridge->anticipation[i] *= (1.0f - cfg->anticipation_decay);
        }
    }

    mod->timestamp_us = nimcp_time_get_us();

    return 0;
}

int hypo_perception_bridge_get_modulation(
    const hypo_perception_bridge_t* bridge,
    hypo_perception_modulation_t* modulation)
{
    if (!bridge || !modulation) return -1;

    *modulation = bridge->modulation;
    return 0;
}

int hypo_perception_bridge_set_arousal(
    hypo_perception_bridge_t* bridge,
    float arousal)
{
    if (!bridge) return -1;

    bridge->external_arousal = clamp_01(arousal);
    return 0;
}

/*=============================================================================
 * PERCEPTION → DRIVE FEEDBACK
 *===========================================================================*/

int hypo_perception_bridge_process_detection(
    hypo_perception_bridge_t* bridge,
    const hypo_sensory_detection_t* detection)
{
    if (!bridge || !detection) return -1;

    bridge->last_detection = *detection;
    bridge->has_recent_detection = true;

    /*
     * SENSORY DETECTION → DRIVE ANTICIPATION:
     *
     * Seeing food increases hunger anticipation (wanting)
     * Detecting threat boosts safety drive (fear response)
     * etc.
     */

    if (!bridge->config.enable_anticipation) {
        return 0;
    }

    hypo_drive_type_t relevant_drive = category_to_drive(detection->detected_category);
    if (relevant_drive >= HYPO_DRIVE_COUNT) {
        return 0;  /* No relevant drive for this category */
    }

    /* Boost anticipation based on detection confidence and intensity */
    float anticipation_boost = detection->confidence * detection->intensity * 0.5f;
    bridge->anticipation[relevant_drive] = clamp_01(
        bridge->anticipation[relevant_drive] + anticipation_boost
    );

    /* Threat detection gets immediate safety drive boost */
    if (detection->is_threat) {
        bridge->anticipation[HYPO_DRIVE_SAFETY] = clamp_01(
            bridge->anticipation[HYPO_DRIVE_SAFETY] + 0.3f
        );
        nimcp_log(LOG_LEVEL_DEBUG,
            "hypo_perception_bridge: threat detected, safety anticipation boosted");
    }

    return 0;
}

float hypo_perception_bridge_get_anticipation(
    const hypo_perception_bridge_t* bridge,
    hypo_drive_type_t drive_type)
{
    if (!bridge || drive_type >= HYPO_DRIVE_COUNT) {
        return 0.0f;
    }

    return bridge->anticipation[drive_type];
}

/*=============================================================================
 * BIO-ASYNC INTEGRATION
 *===========================================================================*/

static nimcp_error_t perception_handle_arousal_update(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t promise, void* user_data)
{
    (void)promise;
    hypo_perception_bridge_t* bridge = (hypo_perception_bridge_t*)user_data;
    if (!bridge || !msg || msg_size < sizeof(bio_message_header_t)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    if (msg_size >= sizeof(bio_message_header_t) + sizeof(float)) {
        const float* arousal = (const float*)(header + 1);
        hypo_perception_bridge_set_arousal(bridge, *arousal);
        hypo_perception_bridge_compute_modulation(bridge);
        hypo_perception_bridge_broadcast_modulation(bridge);
    }

    return NIMCP_SUCCESS;
}

static nimcp_error_t perception_handle_detection(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t promise, void* user_data)
{
    (void)promise;
    hypo_perception_bridge_t* bridge = (hypo_perception_bridge_t*)user_data;
    if (!bridge || !msg || msg_size < sizeof(bio_message_header_t)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    if (msg_size >= sizeof(bio_message_header_t) + sizeof(hypo_sensory_detection_t)) {
        const hypo_sensory_detection_t* detection =
            (const hypo_sensory_detection_t*)(header + 1);
        hypo_perception_bridge_process_detection(bridge, detection);
    }

    return NIMCP_SUCCESS;
}

static nimcp_error_t perception_handle_modulation_request(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t promise, void* user_data)
{
    (void)promise;
    (void)msg;
    (void)msg_size;
    hypo_perception_bridge_t* bridge = (hypo_perception_bridge_t*)user_data;
    if (!bridge) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    hypo_perception_bridge_compute_modulation(bridge);
    hypo_perception_bridge_broadcast_modulation(bridge);

    return NIMCP_SUCCESS;
}

bool hypo_perception_bridge_register_bio(hypo_perception_bridge_t* bridge, bool use_kg_wiring) {
    if (!bridge) return false;
    if (bridge->bio_registered) return true;

    (void)use_kg_wiring;  /* Future KG wiring integration */

    bio_module_info_t info = {
        .module_id = HYPO_PERCEPTION_BRIDGE_MODULE_ID,
        .module_name = "hypothalamus_perception_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (!bridge->bio_ctx) {
        nimcp_log(LOG_LEVEL_ERROR, "hypo_perception_bridge: failed to register with bio-router");
        return false;
    }

    /* Register handlers for perception messages */
    bio_router_register_handler(bridge->bio_ctx,
        BIO_MSG_HYPO_PERCEPTION_AROUSAL_UPDATE, perception_handle_arousal_update);
    bio_router_register_handler(bridge->bio_ctx,
        BIO_MSG_HYPO_PERCEPTION_DETECTION, perception_handle_detection);
    bio_router_register_handler(bridge->bio_ctx,
        BIO_MSG_HYPO_PERCEPTION_MODULATION_REQUEST, perception_handle_modulation_request);

    bridge->bio_registered = true;
    nimcp_log(LOG_LEVEL_INFO, "hypo_perception_bridge: registered with bio-router");
    return true;
}

void hypo_perception_bridge_unregister_bio(hypo_perception_bridge_t* bridge) {
    if (!bridge || !bridge->bio_registered) return;

    bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_registered = false;
    nimcp_log(LOG_LEVEL_INFO, "hypo_perception_bridge: unregistered from bio-router");
}

nimcp_error_t hypo_perception_bridge_broadcast_modulation(hypo_perception_bridge_t* bridge) {
    if (!bridge || !bridge->bio_registered) return NIMCP_ERROR_INVALID_PARAM;

    struct {
        bio_message_header_t header;
        hypo_perception_modulation_t modulation;
    } msg;

    msg.header.type = BIO_MSG_HYPO_PERCEPTION_MODULATION_OUTPUT;
    msg.header.timestamp_us = nimcp_time_get_us();
    msg.header.source_module = HYPO_PERCEPTION_BRIDGE_MODULE_ID;
    msg.header.target_module = 0;  /* Broadcast */
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.payload_size = sizeof(msg) - sizeof(bio_message_header_t);
    msg.header.sequence_id = 0;
    msg.header.channel = BIO_CHANNEL_NOREPINEPHRINE;  /* Arousal channel */

    msg.modulation = bridge->modulation;

    return bio_router_broadcast(bridge->bio_ctx, &msg, sizeof(msg));
}
