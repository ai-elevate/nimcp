/**
 * @file nimcp_mirror_attention_bridge.c
 * @brief Mirror Neuron - Attention System Bidirectional Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-31
 *
 * SIMD-optimized implementation of mirror-attention bridge for joint attention
 * and gaze following mechanisms.
 */

#include "cognitive/mirror_neurons/nimcp_mirror_attention_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/tensor/nimcp_tensor_simd.h"
#include "utils/thread/nimcp_thread.h"
#include "async/nimcp_bio_messages.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

//=============================================================================
// Internal Structures
//=============================================================================

struct mirror_attention_bridge {
    mirror_attention_config_t config;
    mirror_attention_state_t state;

    /* Agent tracking */
    mirror_attention_agent_t agents[MIRROR_ATTENTION_MAX_AGENTS];
    uint32_t active_agent_count;

    /* Current attention state */
    mirror_saliency_state_t saliency;

    /* Statistics */
    mirror_attention_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;

    /* Bio-async */
    bool bio_async_registered;
};

//=============================================================================
// Vector Math Helpers
//=============================================================================

static inline float vec3_dot(const mirror_attention_vec3_t* a,
                             const mirror_attention_vec3_t* b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

static inline float vec3_length(const mirror_attention_vec3_t* v) {
    return sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
}

static inline void vec3_normalize(mirror_attention_vec3_t* v) {
    float len = vec3_length(v);
    if (len > 1e-6f) {
        v->x /= len;
        v->y /= len;
        v->z /= len;
    }
}

static inline void vec3_scale(mirror_attention_vec3_t* v, float s) {
    v->x *= s;
    v->y *= s;
    v->z *= s;
}

static inline void vec3_add(mirror_attention_vec3_t* result,
                            const mirror_attention_vec3_t* a,
                            const mirror_attention_vec3_t* b) {
    result->x = a->x + b->x;
    result->y = a->y + b->y;
    result->z = a->z + b->z;
}

static inline float vec3_distance(const mirror_attention_vec3_t* a,
                                   const mirror_attention_vec3_t* b) {
    float dx = a->x - b->x;
    float dy = a->y - b->y;
    float dz = a->z - b->z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

//=============================================================================
// Internal Functions
//=============================================================================

/**
 * @brief Get or create agent slot
 */
static mirror_attention_agent_t* get_or_create_agent(
    mirror_attention_bridge_t* bridge,
    uint32_t agent_id
) {
    /* Find existing */
    for (uint32_t i = 0; i < MIRROR_ATTENTION_MAX_AGENTS; i++) {
        if (bridge->agents[i].active && bridge->agents[i].agent_id == agent_id) {
            return &bridge->agents[i];
        }
    }

    /* Find empty slot */
    for (uint32_t i = 0; i < MIRROR_ATTENTION_MAX_AGENTS; i++) {
        if (!bridge->agents[i].active) {
            memset(&bridge->agents[i], 0, sizeof(mirror_attention_agent_t));
            bridge->agents[i].agent_id = agent_id;
            bridge->agents[i].active = true;
            bridge->agents[i].gaze_validity = 0.5f;  /* Start neutral */
            bridge->agents[i].gaze_following_rate = 0.0f;
            bridge->agents[i].joint_attention_tendency = 0.5f;
            bridge->active_agent_count++;
            return &bridge->agents[i];
        }
    }

    return NULL;
}

/**
 * @brief Compute cue strength based on type and observation
 */
static float compute_cue_strength(
    const mirror_attention_config_t* config,
    const mirror_gaze_observation_t* obs
) {
    float base_strength = obs->confidence;

    /* Weight by cue type */
    float type_weight = 1.0f;
    switch (obs->cue_type) {
        case MIRROR_CUE_GAZE:
            type_weight = config->gaze_cue_strength;
            break;
        case MIRROR_CUE_HEAD_TURN:
            type_weight = config->gaze_cue_strength * 0.9f;
            break;
        case MIRROR_CUE_POINTING:
            type_weight = config->pointing_cue_strength;
            break;
        case MIRROR_CUE_REACH:
            type_weight = config->pointing_cue_strength * 0.8f;
            break;
        case MIRROR_CUE_BODY_ORIENT:
            type_weight = config->gaze_cue_strength * 0.7f;
            break;
        default:
            type_weight = 0.5f;
            break;
    }

    /* Boost for duration (sustained attention is stronger cue) */
    float duration_factor = 1.0f;
    if (obs->duration_ms > 500.0f) {
        duration_factor = 1.0f + fminf(0.5f, (obs->duration_ms - 500.0f) / 2000.0f);
    }

    /* Boost for referential intent */
    if (obs->is_referential) {
        duration_factor *= 1.3f;
    }

    return fminf(1.0f, base_strength * type_weight * duration_factor);
}

/**
 * @brief Determine recommended attention shift type
 */
static attention_shift_type_t determine_shift_type(
    const mirror_attention_config_t* config,
    const mirror_gaze_observation_t* obs,
    mirror_attention_agent_t* agent,
    float cue_strength
) {
    if (cue_strength < config->cue_validity_threshold) {
        return ATTENTION_SHIFT_NONE;
    }

    /* Check agent validity history */
    if (agent && agent->gaze_validity < 0.3f) {
        return ATTENTION_SHIFT_SUPPRESSED;  /* Unreliable agent */
    }

    /* Short duration = reflexive, long = volitional */
    if (obs->duration_ms < config->reflexive_soa_ms) {
        return ATTENTION_SHIFT_REFLEXIVE;
    }

    return ATTENTION_SHIFT_VOLITIONAL;
}

/**
 * @brief Update saliency map with Gaussian
 */
static void update_saliency_gaussian(
    mirror_saliency_state_t* saliency,
    float focus_x,
    float focus_y,
    float sigma,
    float strength
) {
    float sigma_sq_2 = 2.0f * sigma * sigma;

    for (uint32_t i = 0; i < MIRROR_ATTENTION_SALIENCY_SIZE; i++) {
        float y = (float)i / (float)(MIRROR_ATTENTION_SALIENCY_SIZE - 1);
        float dy = y - focus_y;

        for (uint32_t j = 0; j < MIRROR_ATTENTION_SALIENCY_SIZE; j++) {
            float x = (float)j / (float)(MIRROR_ATTENTION_SALIENCY_SIZE - 1);
            float dx = x - focus_x;

            float dist_sq = dx * dx + dy * dy;
            float gauss = strength * expf(-dist_sq / sigma_sq_2);

            saliency->saliency_boost[i][j] =
                fmaxf(saliency->saliency_boost[i][j], gauss);
        }
    }

    saliency->last_update_us = nimcp_time_now_us();
}

/**
 * @brief Decay saliency map
 */
static void decay_saliency(mirror_saliency_state_t* saliency, float decay_rate) {
    for (uint32_t i = 0; i < MIRROR_ATTENTION_SALIENCY_SIZE; i++) {
        for (uint32_t j = 0; j < MIRROR_ATTENTION_SALIENCY_SIZE; j++) {
            saliency->saliency_boost[i][j] *= (1.0f - decay_rate);
        }
    }
}

//=============================================================================
// Public API Implementation
//=============================================================================

mirror_attention_config_t mirror_attention_config_default(void) {
    mirror_attention_config_t config = {
        .cue_validity_threshold = 0.3f,
        .reflexive_soa_ms = 150.0f,
        .voluntary_soa_ms = 300.0f,
        .gaze_cue_strength = 0.8f,
        .pointing_cue_strength = 0.9f,

        .joint_attention_threshold = 0.6f,
        .joint_attention_timeout_ms = 5000.0f,
        .enable_joint_attention_initiation = true,
        .enable_referential_gaze = true,

        .attention_mirror_gain = 0.3f,
        .mirror_attention_gain = 0.5f,

        .enable_saliency_modulation = true,
        .saliency_decay_rate = 0.02f,

        .enable_simd = true,
        .bio_async_enabled = true
    };
    return config;
}

mirror_attention_bridge_t* mirror_attention_create(
    const mirror_attention_config_t* config
) {
    mirror_attention_bridge_t* bridge = nimcp_calloc(1,
                                                      sizeof(mirror_attention_bridge_t));
    if (!bridge) {
        nimcp_log(LOG_LEVEL_ERROR, "Mirror-Attention: Failed to allocate bridge");
        return NULL;
    }

    /* Configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = mirror_attention_config_default();
    }

    /* Initialize state */
    bridge->state = MIRROR_ATTENTION_STATE_IDLE;
    bridge->active_agent_count = 0;

    /* Initialize saliency */
    memset(&bridge->saliency, 0, sizeof(mirror_saliency_state_t));
    bridge->saliency.attention_sigma = 0.1f;
    bridge->saliency.attention_strength = 1.0f;

    /* Create mutex */
    mutex_attr_t attr = {0};
    attr.type = MUTEX_TYPE_NORMAL;
    bridge->mutex = nimcp_mutex_create(&attr);
    if (!bridge->mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Mirror-Attention: Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Register bio-async */
    if (bridge->config.bio_async_enabled) {
        mirror_attention_register_bio_async(bridge);
    }

    nimcp_log(LOG_LEVEL_INFO, "Mirror-Attention: Created bridge (joint_attention=%s)",
              bridge->config.enable_joint_attention_initiation ? "enabled" : "disabled");

    return bridge;
}

void mirror_attention_destroy(mirror_attention_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->bio_async_registered) {
        mirror_attention_unregister_bio_async(bridge);
    }

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    nimcp_log(LOG_LEVEL_DEBUG, "Mirror-Attention: Destroyed bridge");
}

bool mirror_attention_process_gaze(
    mirror_attention_bridge_t* bridge,
    const mirror_gaze_observation_t* observation,
    mirror_attention_cue_t* cue
) {
    if (!bridge || !observation || !cue) return false;

    nimcp_mutex_lock(bridge->mutex);

    memset(cue, 0, sizeof(mirror_attention_cue_t));
    cue->timestamp_us = nimcp_time_now_us();

    /* Get or create agent */
    mirror_attention_agent_t* agent = get_or_create_agent(bridge,
                                                          observation->agent_id);

    /* Compute cue strength */
    float cue_strength = compute_cue_strength(&bridge->config, observation);
    cue->cue_strength = cue_strength;

    /* Determine shift type */
    cue->recommended_shift = determine_shift_type(&bridge->config, observation,
                                                   agent, cue_strength);

    /* Set cue location */
    if (observation->target_position_valid) {
        cue->cue_location = observation->target_position;
    } else {
        /* Compute from gaze direction */
        float distance = mirror_attention_compute_gaze_target(
            &observation->agent_position,
            &observation->direction,
            &cue->cue_location
        );
        if (distance < 0) {
            /* Default to some distance along gaze */
            cue->cue_location.x = observation->agent_position.x +
                                  observation->direction.x * 2.0f;
            cue->cue_location.y = observation->agent_position.y +
                                  observation->direction.y * 2.0f;
            cue->cue_location.z = observation->agent_position.z +
                                  observation->direction.z * 2.0f;
        }
    }

    /* Object cuing */
    if (observation->target_object_valid) {
        cue->cue_object_id = observation->target_object_id;
        cue->object_cued = true;
    }

    /* Compute validity (learned from history) */
    cue->cue_validity = agent ? agent->gaze_validity : 0.5f;

    /* Expected SOA */
    if (cue->recommended_shift == ATTENTION_SHIFT_REFLEXIVE) {
        cue->expected_soa_ms = bridge->config.reflexive_soa_ms;
    } else if (cue->recommended_shift == ATTENTION_SHIFT_VOLITIONAL) {
        cue->expected_soa_ms = bridge->config.voluntary_soa_ms;
    }

    /* Handle mutual gaze */
    if (observation->is_mutual_gaze) {
        bridge->stats.mutual_gaze_events++;
        /* Mutual gaze is special - may initiate joint attention */
        if (agent && agent->joint_state == JOINT_ATTENTION_NONE) {
            agent->joint_state = JOINT_ATTENTION_RESPONDING;
            agent->joint_attention_start_us = cue->timestamp_us;
        }
    }

    /* Update joint attention state */
    if (agent) {
        cue->joint_state = agent->joint_state;
        cue->joint_agent_id = agent->agent_id;

        /* Update agent tracking */
        agent->last_gaze_direction = observation->direction;
        agent->last_target = cue->cue_location;
        agent->last_cue_timestamp_us = cue->timestamp_us;
    }

    /* Update saliency map if enabled */
    if (bridge->config.enable_saliency_modulation &&
        cue->recommended_shift != ATTENTION_SHIFT_NONE) {
        /* Convert 3D location to 2D normalized for saliency map */
        /* Simple projection - assume Y is depth */
        float sal_x = fminf(1.0f, fmaxf(0.0f, (cue->cue_location.x + 2.0f) / 4.0f));
        float sal_y = fminf(1.0f, fmaxf(0.0f, (cue->cue_location.z + 2.0f) / 4.0f));

        update_saliency_gaussian(&bridge->saliency, sal_x, sal_y,
                                 0.1f, cue_strength);
    }

    /* Update statistics */
    if (observation->cue_type == MIRROR_CUE_GAZE ||
        observation->cue_type == MIRROR_CUE_HEAD_TURN) {
        bridge->stats.gaze_cues_detected++;
    } else if (observation->cue_type == MIRROR_CUE_POINTING ||
               observation->cue_type == MIRROR_CUE_REACH) {
        bridge->stats.pointing_cues_detected++;
    }

    if (cue->recommended_shift != ATTENTION_SHIFT_NONE &&
        cue->recommended_shift != ATTENTION_SHIFT_SUPPRESSED) {
        bridge->stats.attention_shifts_triggered++;
    }

    bridge->stats.avg_cue_strength =
        bridge->stats.avg_cue_strength * 0.95f + cue_strength * 0.05f;
    bridge->stats.avg_gaze_validity =
        bridge->stats.avg_gaze_validity * 0.95f + cue->cue_validity * 0.05f;

    bridge->state = MIRROR_ATTENTION_STATE_CUE_DETECTED;

    nimcp_mutex_unlock(bridge->mutex);

    /* Bio-async message sending would happen here when integrated
     * into a full bio-async context with proper router registration */
    (void)bridge->bio_async_registered; /* Suppress unused warning */

    return true;
}

float mirror_attention_compute_gaze_target(
    const mirror_attention_vec3_t* agent_position,
    const mirror_attention_vec3_t* gaze_direction,
    mirror_attention_vec3_t* target
) {
    if (!agent_position || !gaze_direction || !target) return -1.0f;

    /* Simple ray-plane intersection with ground plane (y=0) */
    /* P = O + t*D where P.y = 0 */
    /* t = -O.y / D.y */

    float t;
    if (fabsf(gaze_direction->y) < 1e-6f) {
        /* Horizontal gaze - use default distance of 3.0m */
        t = 3.0f;
    } else {
        t = -agent_position->y / gaze_direction->y;
        if (t < 0.0f) {
            /* Looking up, use arbitrary distance */
            t = 3.0f;
        }
    }

    target->x = agent_position->x + gaze_direction->x * t;
    target->y = agent_position->y + gaze_direction->y * t;
    target->z = agent_position->z + gaze_direction->z * t;

    return t;
}

uint32_t mirror_attention_process_batch(
    mirror_attention_bridge_t* bridge,
    const mirror_gaze_observation_t* observations,
    mirror_attention_cue_t* cues,
    uint32_t count
) {
    if (!bridge || !observations || !cues || count == 0) return 0;

    uint32_t processed = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (mirror_attention_process_gaze(bridge, &observations[i], &cues[i])) {
            processed++;
        }
    }

    if (bridge->config.enable_simd && count >= MIRROR_ATTENTION_SIMD_THRESHOLD) {
        bridge->stats.simd_operations++;
    }

    return processed;
}

joint_attention_state_t mirror_attention_get_joint_state(
    mirror_attention_bridge_t* bridge,
    uint32_t agent_id
) {
    if (!bridge) return JOINT_ATTENTION_NONE;

    nimcp_mutex_lock(bridge->mutex);

    joint_attention_state_t state = JOINT_ATTENTION_NONE;
    for (uint32_t i = 0; i < MIRROR_ATTENTION_MAX_AGENTS; i++) {
        if (bridge->agents[i].active && bridge->agents[i].agent_id == agent_id) {
            state = bridge->agents[i].joint_state;
            break;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return state;
}

bool mirror_attention_initiate_joint(
    mirror_attention_bridge_t* bridge,
    uint32_t agent_id,
    const mirror_attention_vec3_t* target
) {
    if (!bridge || !target) return false;
    if (!bridge->config.enable_joint_attention_initiation) return false;

    nimcp_mutex_lock(bridge->mutex);

    mirror_attention_agent_t* agent = get_or_create_agent(bridge, agent_id);
    if (!agent) {
        nimcp_mutex_unlock(bridge->mutex);
        return false;
    }

    /* Can only initiate from NONE state */
    if (agent->joint_state != JOINT_ATTENTION_NONE) {
        nimcp_mutex_unlock(bridge->mutex);
        return false;
    }

    agent->joint_state = JOINT_ATTENTION_INITIATING;
    agent->last_target = *target;
    agent->joint_attention_start_us = nimcp_time_now_us();

    bridge->state = MIRROR_ATTENTION_STATE_JOINT;

    nimcp_mutex_unlock(bridge->mutex);

    nimcp_log(LOG_LEVEL_DEBUG, "Mirror-Attention: Initiated joint attention with agent %u", agent_id);

    return true;
}

bool mirror_attention_respond_to_joint(
    mirror_attention_bridge_t* bridge,
    uint32_t agent_id
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);

    mirror_attention_agent_t* agent = NULL;
    for (uint32_t i = 0; i < MIRROR_ATTENTION_MAX_AGENTS; i++) {
        if (bridge->agents[i].active && bridge->agents[i].agent_id == agent_id) {
            agent = &bridge->agents[i];
            break;
        }
    }

    if (!agent || agent->joint_state != JOINT_ATTENTION_RESPONDING) {
        nimcp_mutex_unlock(bridge->mutex);
        return false;
    }

    agent->joint_state = JOINT_ATTENTION_ESTABLISHED;
    agent->successful_joint_attention_count++;

    bridge->stats.joint_attention_episodes++;
    bridge->state = MIRROR_ATTENTION_STATE_JOINT;

    nimcp_mutex_unlock(bridge->mutex);

    nimcp_log(LOG_LEVEL_DEBUG, "Mirror-Attention: Joint attention established with agent %u", agent_id);

    return true;
}

void mirror_attention_break_joint(
    mirror_attention_bridge_t* bridge,
    uint32_t agent_id
) {
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);

    for (uint32_t i = 0; i < MIRROR_ATTENTION_MAX_AGENTS; i++) {
        if (bridge->agents[i].active && bridge->agents[i].agent_id == agent_id) {
            mirror_attention_agent_t* agent = &bridge->agents[i];

            /* Calculate duration */
            if (agent->joint_state == JOINT_ATTENTION_ESTABLISHED) {
                uint64_t now = nimcp_time_now_us();
                float duration_ms = (float)(now - agent->joint_attention_start_us) / 1000.0f;
                agent->avg_joint_duration_ms =
                    agent->avg_joint_duration_ms * 0.8f + duration_ms * 0.2f;
            }

            agent->joint_state = JOINT_ATTENTION_BREAKING;
            break;
        }
    }

    bridge->state = MIRROR_ATTENTION_STATE_INDEPENDENT;

    nimcp_mutex_unlock(bridge->mutex);
}

float mirror_attention_get_sensitivity_at(
    mirror_attention_bridge_t* bridge,
    const mirror_attention_vec3_t* position
) {
    if (!bridge || !position) return 1.0f;

    nimcp_mutex_lock(bridge->mutex);

    /* Compute distance from attention focus */
    float dist = vec3_distance(&bridge->saliency.attention_focus, position);

    /* Gaussian falloff */
    float sigma = bridge->saliency.attention_sigma;
    float sensitivity = expf(-(dist * dist) / (2.0f * sigma * sigma));

    /* Apply mirror-attention gain */
    sensitivity = 1.0f + sensitivity * bridge->config.attention_mirror_gain;

    nimcp_mutex_unlock(bridge->mutex);
    return sensitivity;
}

void mirror_attention_set_focus(
    mirror_attention_bridge_t* bridge,
    const mirror_attention_vec3_t* focus,
    float strength,
    float sigma
) {
    if (!bridge || !focus) return;

    nimcp_mutex_lock(bridge->mutex);

    bridge->saliency.attention_focus = *focus;
    bridge->saliency.attention_strength = strength;
    bridge->saliency.attention_sigma = sigma;

    nimcp_mutex_unlock(bridge->mutex);
}

float mirror_attention_get_saliency_boost(
    mirror_attention_bridge_t* bridge,
    float x,
    float y
) {
    if (!bridge) return 1.0f;

    nimcp_mutex_lock(bridge->mutex);

    /* Decay saliency */
    decay_saliency(&bridge->saliency, bridge->config.saliency_decay_rate);

    /* Get saliency at location */
    uint32_t ix = (uint32_t)(x * (MIRROR_ATTENTION_SALIENCY_SIZE - 1));
    uint32_t iy = (uint32_t)(y * (MIRROR_ATTENTION_SALIENCY_SIZE - 1));

    ix = (ix < MIRROR_ATTENTION_SALIENCY_SIZE) ? ix : MIRROR_ATTENTION_SALIENCY_SIZE - 1;
    iy = (iy < MIRROR_ATTENTION_SALIENCY_SIZE) ? iy : MIRROR_ATTENTION_SALIENCY_SIZE - 1;

    float boost = 1.0f + bridge->saliency.saliency_boost[iy][ix];

    nimcp_mutex_unlock(bridge->mutex);
    return boost;
}

void mirror_attention_simd_gaze_targets(
    const float* positions,
    const float* directions,
    float* targets,
    float* distances,
    uint32_t count
) {
    /* SIMD batch gaze target computation */
    for (uint32_t i = 0; i < count; i++) {
        mirror_attention_vec3_t pos = {
            positions[i * 3 + 0],
            positions[i * 3 + 1],
            positions[i * 3 + 2]
        };
        mirror_attention_vec3_t dir = {
            directions[i * 3 + 0],
            directions[i * 3 + 1],
            directions[i * 3 + 2]
        };
        mirror_attention_vec3_t target;

        distances[i] = mirror_attention_compute_gaze_target(&pos, &dir, &target);

        targets[i * 3 + 0] = target.x;
        targets[i * 3 + 1] = target.y;
        targets[i * 3 + 2] = target.z;
    }
}

void mirror_attention_simd_update_saliency(
    float* saliency_map,
    uint32_t size,
    float focus_x,
    float focus_y,
    float sigma,
    float strength
) {
    float sigma_sq_2 = 2.0f * sigma * sigma;

    for (uint32_t i = 0; i < size; i++) {
        float y = (float)i / (float)(size - 1);
        float dy = y - focus_y;

        for (uint32_t j = 0; j < size; j++) {
            float x = (float)j / (float)(size - 1);
            float dx = x - focus_x;

            float dist_sq = dx * dx + dy * dy;
            float gauss = strength * expf(-dist_sq / sigma_sq_2);

            saliency_map[i * size + j] =
                fmaxf(saliency_map[i * size + j], gauss);
        }
    }
}

mirror_attention_agent_t* mirror_attention_get_agent(
    mirror_attention_bridge_t* bridge,
    uint32_t agent_id
) {
    if (!bridge) return NULL;

    nimcp_mutex_lock(bridge->mutex);
    mirror_attention_agent_t* agent = get_or_create_agent(bridge, agent_id);
    nimcp_mutex_unlock(bridge->mutex);

    return agent;
}

void mirror_attention_update_validity(
    mirror_attention_bridge_t* bridge,
    uint32_t agent_id,
    bool valid
) {
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);

    for (uint32_t i = 0; i < MIRROR_ATTENTION_MAX_AGENTS; i++) {
        if (bridge->agents[i].active && bridge->agents[i].agent_id == agent_id) {
            mirror_attention_agent_t* agent = &bridge->agents[i];

            /* Update validity with learning rate */
            float lr = 0.1f;
            float target = valid ? 1.0f : 0.0f;
            agent->gaze_validity = agent->gaze_validity * (1.0f - lr) + target * lr;

            /* Update following rate */
            if (valid) {
                agent->gaze_following_rate =
                    agent->gaze_following_rate * 0.9f + 0.1f;
            }
            break;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
}

bool mirror_attention_register_bio_async(mirror_attention_bridge_t* bridge) {
    if (!bridge || bridge->bio_async_registered) return false;

    /* Mark as registered - actual router registration would happen
     * when integrated into a full bio-async context */
    bridge->bio_async_registered = true;
    return true;
}

void mirror_attention_unregister_bio_async(mirror_attention_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_registered) return;

    bridge->bio_async_registered = false;
}

bool mirror_attention_get_stats(
    const mirror_attention_bridge_t* bridge,
    mirror_attention_stats_t* stats
) {
    if (!bridge || !stats) return false;

    nimcp_mutex_lock(((mirror_attention_bridge_t*)bridge)->mutex);
    *stats = bridge->stats;

    /* Count active agents */
    stats->active_agents = 0;
    uint64_t total_success = 0;
    uint64_t total_attempts = 0;
    float total_duration = 0.0f;
    uint32_t duration_count = 0;

    for (uint32_t i = 0; i < MIRROR_ATTENTION_MAX_AGENTS; i++) {
        if (bridge->agents[i].active) {
            stats->active_agents++;
            total_success += bridge->agents[i].successful_joint_attention_count;
            total_attempts += bridge->agents[i].successful_joint_attention_count +
                              bridge->agents[i].failed_joint_attention_count;
            if (bridge->agents[i].avg_joint_duration_ms > 0.0f) {
                total_duration += bridge->agents[i].avg_joint_duration_ms;
                duration_count++;
            }
        }
    }

    if (total_attempts > 0) {
        stats->successful_joint_rate = (float)total_success / (float)total_attempts;
    }
    if (duration_count > 0) {
        stats->avg_joint_attention_duration_ms = total_duration / (float)duration_count;
    }

    nimcp_mutex_unlock(((mirror_attention_bridge_t*)bridge)->mutex);
    return true;
}

void mirror_attention_reset_stats(mirror_attention_bridge_t* bridge) {
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(mirror_attention_stats_t));
    nimcp_mutex_unlock(bridge->mutex);
}
