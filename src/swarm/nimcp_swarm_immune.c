/**
 * @file nimcp_swarm_immune.c
 * @brief Swarm Immune System Implementation
 *
 * Biologically-inspired adaptive immune response for swarm robotics.
 * Based on principles of:
 * - T-cell mediated immunity
 * - B-cell antibody production
 * - Clonal selection theory
 * - Immunological memory
 * - Self/non-self discrimination
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include "swarm/nimcp_swarm_immune.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform.h"
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Constants and Macros
 * ============================================================================ */

#define NIMCP_SWARM_IMMUNE_VERSION "1.0.0"
#define NIMCP_SWARM_IMMUNE_DEFAULT_MAX_MEMORY_CELLS 1000
#define NIMCP_SWARM_IMMUNE_DEFAULT_MAX_THREATS 100
#define NIMCP_SWARM_IMMUNE_DEFAULT_MAX_RESPONSES 50
#define NIMCP_SWARM_IMMUNE_DEFAULT_RECOGNITION_THRESHOLD 0.75f
#define NIMCP_SWARM_IMMUNE_DEFAULT_SELF_TOLERANCE 0.9f
#define NIMCP_SWARM_IMMUNE_DEFAULT_MEMORY_DECAY 0.99f
#define NIMCP_SWARM_IMMUNE_DEFAULT_CLONAL_EXPANSION 1.5f
#define NIMCP_SWARM_IMMUNE_DEFAULT_CONFIRMATION_THRESHOLD 3

/* Threat type names */
static const char* THREAT_TYPE_NAMES[] = {
    "MALICIOUS_DRONE",
    "JAMMING",
    "SPOOFING",
    "INJECTION",
    "REPLAY",
    "PHYSICAL",
    "DOS",
    "SYBIL",
    "BYZANTINE"
};

/* Response type names */
static const char* RESPONSE_TYPE_NAMES[] = {
    "NONE",
    "ISOLATION",
    "EVASION",
    "COUNTER_ATTACK",
    "ALERT",
    "COORDINATION",
    "RECONFIGURATION",
    "AUTHENTICATION"
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Calculate fuzzy match score between two patterns
 */
static float calculate_pattern_match(
    const uint8_t* pattern1,
    size_t len1,
    const uint8_t* pattern2,
    size_t len2
) {
    if (!pattern1 || !pattern2 || len1 == 0 || len2 == 0) {
        return 0.0f;
    }

    /* Use Hamming distance for same-length patterns */
    if (len1 == len2) {
        size_t matches = 0;
        for (size_t i = 0; i < len1; i++) {
            if (pattern1[i] == pattern2[i]) {
                matches++;
            }
        }
        return (float)matches / (float)len1;
    }

    /* Use subsequence matching for different lengths */
    size_t min_len = (len1 < len2) ? len1 : len2;
    size_t matches = 0;

    for (size_t i = 0; i < min_len; i++) {
        if (pattern1[i] == pattern2[i]) {
            matches++;
        }
    }

    return (float)matches / (float)((len1 + len2) / 2);
}

/**
 * @brief Calculate behavioral anomaly score
 */
static float calculate_anomaly_score(
    const NimcpSwarmBehaviorProfile* baseline,
    const NimcpSwarmBehaviorProfile* current
) {
    if (!baseline || !current) {
        return 0.0f;
    }

    float score = 0.0f;
    float weight = 0.25f; /* Equal weight for each metric */

    /* Message rate anomaly */
    float msg_rate_diff = fabsf(current->msg_rate - baseline->msg_rate);
    score += weight * (msg_rate_diff / (baseline->msg_rate + 1.0f));

    /* Movement pattern anomaly */
    float movement_diff = 0.0f;
    for (int i = 0; i < 3; i++) {
        float diff = fabsf(current->movement_pattern[i] - baseline->movement_pattern[i]);
        movement_diff += diff * diff;
    }
    movement_diff = sqrtf(movement_diff);
    score += weight * movement_diff;

    /* Energy usage anomaly */
    float energy_diff = fabsf(current->energy_usage - baseline->energy_usage);
    score += weight * (energy_diff / (baseline->energy_usage + 1.0f));

    /* Connection changes anomaly */
    float conn_diff = fabsf((float)current->connection_changes - (float)baseline->connection_changes);
    score += weight * (conn_diff / (float)(baseline->connection_changes + 1));

    return fminf(score, 1.0f);
}

/**
 * @brief Get current timestamp in milliseconds
 */
static uint64_t get_current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

/**
 * @brief Generate unique ID
 */
static uint32_t generate_unique_id(void) {
    static uint32_t counter = 0;
    return __sync_fetch_and_add(&counter, 1);
}

/* ============================================================================
 * Core API Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_immune_default_config(NimcpSwarmImmuneConfig* config) {
    if (!config) {
        NIMCP_LOG_ERROR("Invalid config pointer");
        return NIMCP_INVALID_PARAM;
    }

    config->max_memory_cells = NIMCP_SWARM_IMMUNE_DEFAULT_MAX_MEMORY_CELLS;
    config->max_active_threats = NIMCP_SWARM_IMMUNE_DEFAULT_MAX_THREATS;
    config->max_active_responses = NIMCP_SWARM_IMMUNE_DEFAULT_MAX_RESPONSES;
    config->recognition_threshold = NIMCP_SWARM_IMMUNE_DEFAULT_RECOGNITION_THRESHOLD;
    config->self_tolerance = NIMCP_SWARM_IMMUNE_DEFAULT_SELF_TOLERANCE;
    config->memory_decay_rate = NIMCP_SWARM_IMMUNE_DEFAULT_MEMORY_DECAY;
    config->clonal_expansion_rate = NIMCP_SWARM_IMMUNE_DEFAULT_CLONAL_EXPANSION;
    config->enable_sharing = true;
    config->enable_coordination = true;
    config->confirmation_threshold = NIMCP_SWARM_IMMUNE_DEFAULT_CONFIRMATION_THRESHOLD;

    return NIMCP_SUCCESS;
}

NimcpSwarmImmuneSystem* nimcp_swarm_immune_create(
    const NimcpSwarmImmuneConfig* config,
    bio_module_context_t* bio_ctx,
    uint32_t self_drone_id
) {
    if (!config) {
        NIMCP_LOG_ERROR("Invalid configuration");
        return NULL;
    }

    NimcpSwarmImmuneSystem* system = (NimcpSwarmImmuneSystem*)NIMCP_MALLOC(
        sizeof(NimcpSwarmImmuneSystem)
    );
    if (!system) {
        NIMCP_LOG_ERROR("Failed to allocate immune system");
        return NULL;
    }

    memset(system, 0, sizeof(NimcpSwarmImmuneSystem));
    system->config = *config;
    system->bio_ctx = bio_ctx;
    system->self_drone_id = self_drone_id;

    /* Generate self signature */
    for (size_t i = 0; i < 32; i++) {
        system->self_signature[i] = (uint8_t)(rand() % 256);
    }

    /* Allocate memory cells */
    system->memory_cell_capacity = config->max_memory_cells;
    system->memory_cells = (nimcp_swarm_memoryCell*)NIMCP_MALLOC(
        system->memory_cell_capacity * sizeof(nimcp_swarm_memoryCell)
    );
    if (!system->memory_cells) {
        NIMCP_LOG_ERROR("Failed to allocate memory cells");
        NIMCP_FREE(system);
        return NULL;
    }

    /* Allocate active threats */
    system->active_threat_capacity = config->max_active_threats;
    system->active_threats = (NimcpSwarmThreat*)NIMCP_MALLOC(
        system->active_threat_capacity * sizeof(NimcpSwarmThreat)
    );
    if (!system->active_threats) {
        NIMCP_LOG_ERROR("Failed to allocate threat storage");
        NIMCP_FREE(system->memory_cells);
        NIMCP_FREE(system);
        return NULL;
    }

    /* Allocate active responses */
    system->active_response_capacity = config->max_active_responses;
    system->active_responses = (NimcpSwarmResponse*)NIMCP_MALLOC(
        system->active_response_capacity * sizeof(NimcpSwarmResponse)
    );
    if (!system->active_responses) {
        NIMCP_LOG_ERROR("Failed to allocate response storage");
        NIMCP_FREE(system->active_threats);
        NIMCP_FREE(system->memory_cells);
        NIMCP_FREE(system);
        return NULL;
    }

    /* Allocate behavior profiles */
    system->behavior_profile_capacity = 100; /* Initial capacity */
    system->behavior_profiles = (NimcpSwarmBehaviorProfile*)NIMCP_MALLOC(
        system->behavior_profile_capacity * sizeof(NimcpSwarmBehaviorProfile)
    );
    if (!system->behavior_profiles) {
        NIMCP_LOG_ERROR("Failed to allocate behavior profiles");
        NIMCP_FREE(system->active_responses);
        NIMCP_FREE(system->active_threats);
        NIMCP_FREE(system->memory_cells);
        NIMCP_FREE(system);
        return NULL;
    }

    /* Create mutex for thread safety */
    system->mutex = nimcp_platform_mutex_create();
    if (!system->mutex) {
        NIMCP_LOG_ERROR("Failed to create mutex");
        NIMCP_FREE(system->behavior_profiles);
        NIMCP_FREE(system->active_responses);
        NIMCP_FREE(system->active_threats);
        NIMCP_FREE(system->memory_cells);
        NIMCP_FREE(system);
        return NULL;
    }

    NIMCP_LOG_INFO("Swarm immune system created (drone_id=%u, version=%s)",
                   self_drone_id, NIMCP_SWARM_IMMUNE_VERSION);

    return system;
}

void nimcp_swarm_immune_destroy(NimcpSwarmImmuneSystem* system) {
    if (!system) {
        return;
    }

    NIMCP_LOG_INFO("Destroying swarm immune system (drone_id=%u)",
                   system->self_drone_id);

    if (system->mutex) {
        nimcp_platform_mutex_destroy(system->mutex);
    }

    NIMCP_FREE(system->behavior_profiles);
    NIMCP_FREE(system->active_responses);
    NIMCP_FREE(system->active_threats);
    NIMCP_FREE(system->memory_cells);
    NIMCP_FREE(system);
}

/* ============================================================================
 * Antigen Recognition (Threat Detection)
 * ============================================================================ */

nimcp_result_t nimcp_swarm_immune_detect_threat(
    NimcpSwarmImmuneSystem* system,
    const uint8_t* data,
    size_t data_len,
    uint32_t source_drone_id,
    uint32_t* threat_id
) {
    if (!system || !data || data_len == 0) {
        NIMCP_LOG_ERROR("Invalid arguments for threat detection");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Check against known threat signatures in memory cells */
    float best_match = 0.0f;
    nimcp_swarm_memoryCell* matched_cell = NULL;

    for (size_t i = 0; i < system->memory_cell_count; i++) {
        nimcp_swarm_memoryCell* cell = &system->memory_cells[i];

        float match_score = calculate_pattern_match(
            cell->signature.pattern,
            cell->signature.pattern_len,
            data,
            data_len
        );

        if (match_score > best_match &&
            match_score >= cell->signature.match_threshold) {
            best_match = match_score;
            matched_cell = cell;
        }
    }

    /* If threat detected */
    if (matched_cell && best_match >= system->config.recognition_threshold) {
        /* Check if we have space for new threat */
        if (system->active_threat_count >= system->active_threat_capacity) {
            NIMCP_LOG_WARN("Active threat capacity reached");
            nimcp_platform_mutex_unlock(system->mutex);
            return NIMCP_CAPACITY_EXCEEDED;
        }

        /* Create new threat entry */
        NimcpSwarmThreat* threat = &system->active_threats[system->active_threat_count];
        threat->id = generate_unique_id();
        threat->type = matched_cell->signature.type;
        threat->source_drone_id = source_drone_id;
        threat->confidence = best_match;
        threat->detection_time = get_current_time_ms();
        threat->confirmed = false;
        threat->confirming_drones = 1;

        /* Determine severity based on confidence and threat type */
        if (best_match > 0.9f || threat->type == THREAT_MALICIOUS_DRONE) {
            threat->severity = SEVERITY_CRITICAL;
        } else if (best_match > 0.8f) {
            threat->severity = SEVERITY_HIGH;
        } else if (best_match > 0.7f) {
            threat->severity = SEVERITY_MEDIUM;
        } else {
            threat->severity = SEVERITY_LOW;
        }

        /* Copy threat data */
        threat->data_len = (data_len < 256) ? data_len : 256;
        memcpy(threat->data, data, threat->data_len);

        if (threat_id) {
            *threat_id = threat->id;
        }

        system->active_threat_count++;
        system->total_threats_detected++;

        /* Update memory cell activation */
        matched_cell->activation_count++;
        matched_cell->last_activation = get_current_time_ms();

        NIMCP_LOG_WARN("Threat detected: type=%s, confidence=%.2f, source=%u",
                       THREAT_TYPE_NAMES[threat->type], best_match, source_drone_id);

        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_THREAT_DETECTED;
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_immune_check_behavior(
    NimcpSwarmImmuneSystem* system,
    uint32_t drone_id,
    const NimcpSwarmBehaviorProfile* behavior,
    float* anomaly_score
) {
    if (!system || !behavior || !anomaly_score) {
        NIMCP_LOG_ERROR("Invalid arguments for behavior check");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Find baseline profile for this drone */
    NimcpSwarmBehaviorProfile* baseline = NULL;
    for (size_t i = 0; i < system->behavior_profile_count; i++) {
        if (system->behavior_profiles[i].drone_id == drone_id) {
            baseline = &system->behavior_profiles[i];
            break;
        }
    }

    /* If no baseline, create one */
    if (!baseline) {
        if (system->behavior_profile_count >= system->behavior_profile_capacity) {
            NIMCP_LOG_WARN("Behavior profile capacity reached");
            nimcp_platform_mutex_unlock(system->mutex);
            return NIMCP_CAPACITY_EXCEEDED;
        }

        baseline = &system->behavior_profiles[system->behavior_profile_count];
        memcpy(baseline, behavior, sizeof(NimcpSwarmBehaviorProfile));
        baseline->anomaly_score = 0.0f;
        system->behavior_profile_count++;

        *anomaly_score = 0.0f;
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_SUCCESS;
    }

    /* Calculate anomaly score */
    float score = calculate_anomaly_score(baseline, behavior);
    *anomaly_score = score;

    /* Update baseline with exponential moving average */
    float alpha = 0.1f; /* Smoothing factor */
    baseline->msg_rate = alpha * behavior->msg_rate + (1.0f - alpha) * baseline->msg_rate;
    baseline->energy_usage = alpha * behavior->energy_usage + (1.0f - alpha) * baseline->energy_usage;
    for (int i = 0; i < 3; i++) {
        baseline->movement_pattern[i] = alpha * behavior->movement_pattern[i] +
                                       (1.0f - alpha) * baseline->movement_pattern[i];
    }
    baseline->anomaly_score = alpha * score + (1.0f - alpha) * baseline->anomaly_score;
    baseline->last_update = get_current_time_ms();

    if (score > 0.7f) {
        NIMCP_LOG_WARN("High anomaly score detected: drone=%u, score=%.2f",
                       drone_id, score);
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_immune_verify_self(
    NimcpSwarmImmuneSystem* system,
    uint32_t drone_id,
    const uint8_t* signature,
    size_t signature_len,
    bool* is_self
) {
    if (!system || !signature || !is_self) {
        NIMCP_LOG_ERROR("Invalid arguments for self verification");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Check if this is our own drone ID */
    if (drone_id == system->self_drone_id) {
        *is_self = true;
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_SUCCESS;
    }

    /* Verify signature against known good signatures */
    /* In a real implementation, this would check against a whitelist */
    /* For now, use a simple tolerance check */

    float match_score = calculate_pattern_match(
        system->self_signature,
        32,
        signature,
        signature_len
    );

    *is_self = (match_score >= system->config.self_tolerance);

    if (!(*is_self)) {
        NIMCP_LOG_WARN("Non-self drone detected: id=%u, match=%.2f",
                       drone_id, match_score);
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Memory Cells (Learned Patterns)
 * ============================================================================ */

nimcp_result_t nimcp_swarm_immune_add_memory_cell(
    NimcpSwarmImmuneSystem* system,
    const NimcpSwarmThreatSignature* signature,
    NimcpSwarmResponseType response,
    float effectiveness,
    uint32_t* cell_id
) {
    if (!system || !signature) {
        NIMCP_LOG_ERROR("Invalid arguments for adding memory cell");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    if (system->memory_cell_count >= system->memory_cell_capacity) {
        NIMCP_LOG_WARN("Memory cell capacity reached");
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_CAPACITY_EXCEEDED;
    }

    nimcp_swarm_memoryCell* cell = &system->memory_cells[system->memory_cell_count];
    cell->id = generate_unique_id();
    cell->signature = *signature;
    cell->response = response;
    cell->effectiveness = effectiveness;
    cell->activation_count = 0;
    cell->created_time = get_current_time_ms();
    cell->last_activation = 0;
    cell->decay_factor = system->config.memory_decay_rate;
    cell->shared = false;

    if (cell_id) {
        *cell_id = cell->id;
    }

    system->memory_cell_count++;

    NIMCP_LOG_INFO("Memory cell added: id=%u, type=%s, response=%s",
                   cell->id,
                   THREAT_TYPE_NAMES[signature->type],
                   RESPONSE_TYPE_NAMES[response]);

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_immune_find_memory_cell(
    NimcpSwarmImmuneSystem* system,
    const uint8_t* data,
    size_t data_len,
    nimcp_swarm_memoryCell** cell
) {
    if (!system || !data || !cell) {
        NIMCP_LOG_ERROR("Invalid arguments for finding memory cell");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    float best_match = 0.0f;
    nimcp_swarm_memoryCell* best_cell = NULL;

    for (size_t i = 0; i < system->memory_cell_count; i++) {
        nimcp_swarm_memoryCell* current = &system->memory_cells[i];

        float match_score = calculate_pattern_match(
            current->signature.pattern,
            current->signature.pattern_len,
            data,
            data_len
        );

        if (match_score > best_match &&
            match_score >= current->signature.match_threshold) {
            best_match = match_score;
            best_cell = current;
        }
    }

    if (best_cell && best_match >= system->config.recognition_threshold) {
        *cell = best_cell;
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_NOT_FOUND;
}

nimcp_result_t nimcp_swarm_immune_update_effectiveness(
    NimcpSwarmImmuneSystem* system,
    uint32_t cell_id,
    float new_effectiveness
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    for (size_t i = 0; i < system->memory_cell_count; i++) {
        if (system->memory_cells[i].id == cell_id) {
            /* Update with exponential moving average */
            float alpha = 0.3f;
            system->memory_cells[i].effectiveness =
                alpha * new_effectiveness +
                (1.0f - alpha) * system->memory_cells[i].effectiveness;

            nimcp_platform_mutex_unlock(system->mutex);
            return NIMCP_SUCCESS;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_NOT_FOUND;
}

nimcp_result_t nimcp_swarm_immune_decay_memory(
    NimcpSwarmImmuneSystem* system,
    uint64_t current_time
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    size_t removed = 0;

    /* Apply decay and remove weak memory cells */
    for (size_t i = 0; i < system->memory_cell_count; ) {
        nimcp_swarm_memoryCell* cell = &system->memory_cells[i];

        /* Calculate time-based decay */
        uint64_t time_diff = current_time - cell->last_activation;
        if (time_diff > 0) {
            float time_factor = expf(-(float)time_diff / 3600000.0f); /* 1 hour decay */
            cell->effectiveness *= cell->decay_factor * time_factor;
        }

        /* Remove if effectiveness too low */
        if (cell->effectiveness < 0.1f && cell->activation_count < 5) {
            /* Swap with last element and reduce count */
            system->memory_cells[i] = system->memory_cells[system->memory_cell_count - 1];
            system->memory_cell_count--;
            removed++;
        } else {
            i++;
        }
    }

    if (removed > 0) {
        NIMCP_LOG_INFO("Memory decay removed %zu weak cells", removed);
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Antibody Production (Response Generation)
 * ============================================================================ */

NimcpSwarmResponseType nimcp_swarm_immune_get_response(
    NimcpSwarmImmuneSystem* system,
    NimcpSwarmThreatType threat_type,
    NimcpSwarmSeverity severity
) {
    /* Response strategy matrix */
    static const NimcpSwarmResponseType response_matrix[THREAT_COUNT][4] = {
        /* LOW             MEDIUM              HIGH                CRITICAL */
        {RESPONSE_ALERT,  RESPONSE_ALERT,     RESPONSE_ISOLATION, RESPONSE_ISOLATION},        /* MALICIOUS_DRONE */
        {RESPONSE_EVASION, RESPONSE_EVASION,  RESPONSE_EVASION,   RESPONSE_RECONFIGURATION}, /* JAMMING */
        {RESPONSE_ALERT,  RESPONSE_AUTHENTICATION, RESPONSE_AUTHENTICATION, RESPONSE_ISOLATION}, /* SPOOFING */
        {RESPONSE_ALERT,  RESPONSE_ALERT,     RESPONSE_ISOLATION, RESPONSE_COUNTER_ATTACK},  /* INJECTION */
        {RESPONSE_ALERT,  RESPONSE_AUTHENTICATION, RESPONSE_AUTHENTICATION, RESPONSE_ISOLATION}, /* REPLAY */
        {RESPONSE_EVASION, RESPONSE_EVASION,  RESPONSE_COORDINATION, RESPONSE_COUNTER_ATTACK}, /* PHYSICAL */
        {RESPONSE_ALERT,  RESPONSE_EVASION,   RESPONSE_RECONFIGURATION, RESPONSE_RECONFIGURATION}, /* DOS */
        {RESPONSE_ALERT,  RESPONSE_AUTHENTICATION, RESPONSE_ISOLATION, RESPONSE_ISOLATION},     /* SYBIL */
        {RESPONSE_ALERT,  RESPONSE_ISOLATION, RESPONSE_ISOLATION, RESPONSE_COORDINATION}       /* BYZANTINE */
    };

    if (threat_type >= THREAT_COUNT || severity > SEVERITY_CRITICAL) {
        return RESPONSE_ALERT;
    }

    return response_matrix[threat_type][severity];
}

nimcp_result_t nimcp_swarm_immune_generate_response(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id,
    uint32_t* response_id
) {
    if (!system || !response_id) {
        NIMCP_LOG_ERROR("Invalid arguments for response generation");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Find the threat */
    NimcpSwarmThreat* threat = NULL;
    for (size_t i = 0; i < system->active_threat_count; i++) {
        if (system->active_threats[i].id == threat_id) {
            threat = &system->active_threats[i];
            break;
        }
    }

    if (!threat) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_NOT_FOUND;
    }

    /* Check capacity */
    if (system->active_response_count >= system->active_response_capacity) {
        NIMCP_LOG_WARN("Active response capacity reached");
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_CAPACITY_EXCEEDED;
    }

    /* Generate response */
    NimcpSwarmResponse* response = &system->active_responses[system->active_response_count];
    response->id = generate_unique_id();
    response->threat_id = threat_id;
    response->target_drone_id = threat->source_drone_id;
    response->start_time = get_current_time_ms();

    /* Determine response type based on threat */
    response->type = nimcp_swarm_immune_get_response(
        system,
        threat->type,
        threat->severity
    );

    /* Set response parameters based on severity */
    response->intensity = (float)threat->severity / (float)SEVERITY_CRITICAL;
    response->duration = (uint64_t)(5000 * (1 + threat->severity)); /* 5-20 seconds */

    /* Determine if coordination needed */
    response->coordinated = (threat->severity >= SEVERITY_HIGH) ||
                           (response->type == RESPONSE_COORDINATION);
    response->participating_drones = response->coordinated ? 3 : 1;

    *response_id = response->id;
    system->active_response_count++;

    NIMCP_LOG_INFO("Response generated: id=%u, type=%s, threat=%u, intensity=%.2f",
                   response->id,
                   RESPONSE_TYPE_NAMES[response->type],
                   threat_id,
                   response->intensity);

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_immune_execute_response(
    NimcpSwarmImmuneSystem* system,
    uint32_t response_id
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Find the response */
    NimcpSwarmResponse* response = NULL;
    for (size_t i = 0; i < system->active_response_count; i++) {
        if (system->active_responses[i].id == response_id) {
            response = &system->active_responses[i];
            break;
        }
    }

    if (!response) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_NOT_FOUND;
    }

    /* Execute response based on type */
    switch (response->type) {
        case RESPONSE_ISOLATION:
            NIMCP_LOG_WARN("EXECUTING: Isolating drone %u", response->target_drone_id);
            /* In real implementation: remove drone from routing tables */
            break;

        case RESPONSE_EVASION:
            NIMCP_LOG_INFO("EXECUTING: Evading threat from drone %u", response->target_drone_id);
            /* In real implementation: adjust position/trajectory */
            break;

        case RESPONSE_COUNTER_ATTACK:
            NIMCP_LOG_WARN("EXECUTING: Counter-attack against drone %u", response->target_drone_id);
            /* In real implementation: active defense measures */
            break;

        case RESPONSE_ALERT:
            NIMCP_LOG_INFO("EXECUTING: Broadcasting alert for threat %u", response->threat_id);
            if (system->config.enable_sharing && system->bio_ctx) {
                nimcp_swarm_immune_broadcast_alert(system, response->threat_id, SEVERITY_MEDIUM);
            }
            break;

        case RESPONSE_COORDINATION:
            NIMCP_LOG_INFO("EXECUTING: Coordinating defensive response");
            /* In real implementation: multi-drone coordinated action */
            break;

        case RESPONSE_RECONFIGURATION:
            NIMCP_LOG_INFO("EXECUTING: Reconfiguring network topology");
            /* In real implementation: adjust communication topology */
            break;

        case RESPONSE_AUTHENTICATION:
            NIMCP_LOG_INFO("EXECUTING: Re-authenticating swarm members");
            /* In real implementation: challenge-response protocol */
            break;

        default:
            break;
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Clonal Selection (Adaptive Evolution)
 * ============================================================================ */

nimcp_result_t nimcp_swarm_immune_amplify_response(
    NimcpSwarmImmuneSystem* system,
    uint32_t response_id,
    float success_rate
) {
    if (!system || success_rate < 0.0f || success_rate > 1.0f) {
        NIMCP_LOG_ERROR("Invalid arguments for response amplification");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Find the response */
    NimcpSwarmResponse* response = NULL;
    for (size_t i = 0; i < system->active_response_count; i++) {
        if (system->active_responses[i].id == response_id) {
            response = &system->active_responses[i];
            break;
        }
    }

    if (!response) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_NOT_FOUND;
    }

    /* Find associated threat and memory cell */
    NimcpSwarmThreat* threat = NULL;
    for (size_t i = 0; i < system->active_threat_count; i++) {
        if (system->active_threats[i].id == response->threat_id) {
            threat = &system->active_threats[i];
            break;
        }
    }

    if (!threat) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_NOT_FOUND;
    }

    /* Update memory cell effectiveness (clonal selection) */
    for (size_t i = 0; i < system->memory_cell_count; i++) {
        nimcp_swarm_memoryCell* cell = &system->memory_cells[i];

        if (cell->signature.type == threat->type && cell->response == response->type) {
            /* Amplify effective responses */
            float amplification = system->config.clonal_expansion_rate * success_rate;
            cell->effectiveness = fminf(1.0f, cell->effectiveness * amplification);
            cell->activation_count++;

            NIMCP_LOG_INFO("Amplifying memory cell %u: effectiveness %.2f -> %.2f",
                          cell->id,
                          cell->effectiveness / amplification,
                          cell->effectiveness);

            /* Share successful patterns */
            if (success_rate > 0.8f && system->config.enable_sharing && !cell->shared) {
                nimcp_swarm_immune_share_memory_cell(system, cell->id);
            }

            break;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_immune_adapt_signature(
    NimcpSwarmImmuneSystem* system,
    uint32_t cell_id,
    const uint8_t* new_data,
    size_t new_data_len
) {
    if (!system || !new_data || new_data_len == 0) {
        NIMCP_LOG_ERROR("Invalid arguments for signature adaptation");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    for (size_t i = 0; i < system->memory_cell_count; i++) {
        if (system->memory_cells[i].id == cell_id) {
            nimcp_swarm_memoryCell* cell = &system->memory_cells[i];

            /* Blend existing pattern with new observation */
            size_t min_len = (cell->signature.pattern_len < new_data_len) ?
                            cell->signature.pattern_len : new_data_len;

            for (size_t j = 0; j < min_len; j++) {
                /* Weighted average */
                cell->signature.pattern[j] = (uint8_t)(
                    0.7f * cell->signature.pattern[j] + 0.3f * new_data[j]
                );
            }

            /* Slightly relax threshold for mutated threats */
            cell->signature.match_threshold *= 0.95f;
            cell->signature.match_threshold = fmaxf(0.5f, cell->signature.match_threshold);

            NIMCP_LOG_INFO("Adapted signature for memory cell %u", cell_id);

            nimcp_platform_mutex_unlock(system->mutex);
            return NIMCP_SUCCESS;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_NOT_FOUND;
}

nimcp_result_t nimcp_swarm_immune_affinity_maturation(
    NimcpSwarmImmuneSystem* system
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Sort memory cells by effectiveness and activation count */
    /* Keep most effective, remove least effective */
    size_t removed = 0;

    for (size_t i = 0; i < system->memory_cell_count; ) {
        nimcp_swarm_memoryCell* cell = &system->memory_cells[i];

        /* Remove low-performing cells */
        if (cell->activation_count > 10 && cell->effectiveness < 0.3f) {
            system->memory_cells[i] = system->memory_cells[system->memory_cell_count - 1];
            system->memory_cell_count--;
            removed++;
        } else {
            i++;
        }
    }

    if (removed > 0) {
        NIMCP_LOG_INFO("Affinity maturation: removed %zu low-affinity cells", removed);
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Swarm Coordination (via Bio-Async)
 * ============================================================================ */

nimcp_result_t nimcp_swarm_immune_share_threat(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    if (!system->config.enable_sharing || !system->bio_ctx) {
        return NIMCP_SUCCESS; /* Sharing disabled */
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Find threat */
    NimcpSwarmThreat* threat = NULL;
    for (size_t i = 0; i < system->active_threat_count; i++) {
        if (system->active_threats[i].id == threat_id) {
            threat = &system->active_threats[i];
            break;
        }
    }

    if (!threat) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_NOT_FOUND;
    }

    /* Send threat intelligence via bio-async */
    bio_message_t msg;
    msg.type = BIO_MSG_CUSTOM;
    msg.priority = (threat->severity >= SEVERITY_HIGH) ?
                   NIMCP_BIO_PRIORITY_HIGH : NIMCP_BIO_PRIORITY_NORMAL;
    msg.sender_id = system->self_drone_id;
    msg.timestamp = get_current_time_ms();

    /* Encode threat data in message payload */
    snprintf((char*)msg.payload.custom_data.data,
             sizeof(msg.payload.custom_data.data),
             "THREAT:type=%d,severity=%d,source=%u,confidence=%.2f",
             threat->type, threat->severity, threat->source_drone_id, threat->confidence);
    msg.payload.custom_data.size = strlen((char*)msg.payload.custom_data.data);

    nimcp_result_t result = nimcp_bio_async_send(system->bio_ctx, &msg);

    NIMCP_LOG_INFO("Shared threat %u with swarm: %s",
                   threat_id,
                   (result == NIMCP_SUCCESS) ? "success" : "failed");

    nimcp_platform_mutex_unlock(system->mutex);
    return result;
}

nimcp_result_t nimcp_swarm_immune_share_memory_cell(
    NimcpSwarmImmuneSystem* system,
    uint32_t cell_id
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    if (!system->config.enable_sharing || !system->bio_ctx) {
        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_lock(system->mutex);

    for (size_t i = 0; i < system->memory_cell_count; i++) {
        if (system->memory_cells[i].id == cell_id) {
            nimcp_swarm_memoryCell* cell = &system->memory_cells[i];

            /* Send memory cell via bio-async */
            bio_message_t msg;
            msg.type = BIO_MSG_CUSTOM;
            msg.priority = NIMCP_BIO_PRIORITY_NORMAL;
            msg.sender_id = system->self_drone_id;
            msg.timestamp = get_current_time_ms();

            snprintf((char*)msg.payload.custom_data.data,
                     sizeof(msg.payload.custom_data.data),
                     "MEMORY:type=%d,response=%d,effectiveness=%.2f",
                     cell->signature.type, cell->response, cell->effectiveness);
            msg.payload.custom_data.size = strlen((char*)msg.payload.custom_data.data);

            nimcp_result_t result = nimcp_bio_async_send(system->bio_ctx, &msg);

            if (result == NIMCP_SUCCESS) {
                cell->shared = true;
                NIMCP_LOG_INFO("Shared memory cell %u with swarm", cell_id);
            }

            nimcp_platform_mutex_unlock(system->mutex);
            return result;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_NOT_FOUND;
}

nimcp_result_t nimcp_swarm_immune_coordinate_response(
    NimcpSwarmImmuneSystem* system,
    uint32_t response_id,
    const uint32_t* participating_drones,
    size_t num_drones
) {
    if (!system || !participating_drones) {
        NIMCP_LOG_ERROR("Invalid arguments for response coordination");
        return NIMCP_INVALID_PARAM;
    }

    if (!system->config.enable_coordination || !system->bio_ctx) {
        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Find response */
    NimcpSwarmResponse* response = NULL;
    for (size_t i = 0; i < system->active_response_count; i++) {
        if (system->active_responses[i].id == response_id) {
            response = &system->active_responses[i];
            break;
        }
    }

    if (!response) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_NOT_FOUND;
    }

    /* Send coordination request */
    bio_message_t msg;
    msg.type = BIO_MSG_CUSTOM;
    msg.priority = NIMCP_BIO_PRIORITY_HIGH;
    msg.sender_id = system->self_drone_id;
    msg.timestamp = get_current_time_ms();

    snprintf((char*)msg.payload.custom_data.data,
             sizeof(msg.payload.custom_data.data),
             "COORDINATE:response=%u,type=%d,target=%u",
             response_id, response->type, response->target_drone_id);
    msg.payload.custom_data.size = strlen((char*)msg.payload.custom_data.data);

    nimcp_result_t result = nimcp_bio_async_send(system->bio_ctx, &msg);

    response->participating_drones = (uint32_t)num_drones;

    NIMCP_LOG_INFO("Coordinated response %u with %zu drones: %s",
                   response_id, num_drones,
                   (result == NIMCP_SUCCESS) ? "success" : "failed");

    nimcp_platform_mutex_unlock(system->mutex);
    return result;
}

nimcp_result_t nimcp_swarm_immune_broadcast_alert(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id,
    NimcpSwarmSeverity priority
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    if (!system->bio_ctx) {
        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Find threat */
    NimcpSwarmThreat* threat = NULL;
    for (size_t i = 0; i < system->active_threat_count; i++) {
        if (system->active_threats[i].id == threat_id) {
            threat = &system->active_threats[i];
            break;
        }
    }

    if (!threat) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_NOT_FOUND;
    }

    /* Broadcast alert */
    bio_message_t msg;
    msg.type = BIO_MSG_CUSTOM;
    msg.priority = (priority >= SEVERITY_HIGH) ?
                   NIMCP_BIO_PRIORITY_CRITICAL : NIMCP_BIO_PRIORITY_HIGH;
    msg.sender_id = system->self_drone_id;
    msg.timestamp = get_current_time_ms();

    snprintf((char*)msg.payload.custom_data.data,
             sizeof(msg.payload.custom_data.data),
             "ALERT:threat=%u,type=%s,severity=%d,source=%u",
             threat_id,
             THREAT_TYPE_NAMES[threat->type],
             threat->severity,
             threat->source_drone_id);
    msg.payload.custom_data.size = strlen((char*)msg.payload.custom_data.data);

    nimcp_result_t result = nimcp_bio_async_send(system->bio_ctx, &msg);

    NIMCP_LOG_WARN("Broadcast alert for threat %u: %s",
                   threat_id,
                   (result == NIMCP_SUCCESS) ? "success" : "failed");

    nimcp_platform_mutex_unlock(system->mutex);
    return result;
}

/* ============================================================================
 * Threat Management
 * ============================================================================ */

nimcp_result_t nimcp_swarm_immune_confirm_threat(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id,
    uint32_t confirming_drone_id
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    for (size_t i = 0; i < system->active_threat_count; i++) {
        if (system->active_threats[i].id == threat_id) {
            NimcpSwarmThreat* threat = &system->active_threats[i];

            threat->confirming_drones++;

            if (threat->confirming_drones >= system->config.confirmation_threshold) {
                threat->confirmed = true;
                NIMCP_LOG_WARN("Threat %u confirmed by %u drones",
                              threat_id, threat->confirming_drones);
            }

            nimcp_platform_mutex_unlock(system->mutex);
            return NIMCP_SUCCESS;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_NOT_FOUND;
}

nimcp_result_t nimcp_swarm_immune_neutralize_threat(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    for (size_t i = 0; i < system->active_threat_count; i++) {
        if (system->active_threats[i].id == threat_id) {
            /* Remove threat by swapping with last */
            system->active_threats[i] = system->active_threats[system->active_threat_count - 1];
            system->active_threat_count--;
            system->total_threats_neutralized++;

            NIMCP_LOG_INFO("Threat %u neutralized", threat_id);

            nimcp_platform_mutex_unlock(system->mutex);
            return NIMCP_SUCCESS;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_NOT_FOUND;
}

nimcp_result_t nimcp_swarm_immune_get_threat(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id,
    const NimcpSwarmThreat** threat
) {
    if (!system || !threat) {
        NIMCP_LOG_ERROR("Invalid arguments");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    for (size_t i = 0; i < system->active_threat_count; i++) {
        if (system->active_threats[i].id == threat_id) {
            *threat = &system->active_threats[i];
            nimcp_platform_mutex_unlock(system->mutex);
            return NIMCP_SUCCESS;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_NOT_FOUND;
}

/* ============================================================================
 * Statistics and Monitoring
 * ============================================================================ */

nimcp_result_t nimcp_swarm_immune_get_stats(
    NimcpSwarmImmuneSystem* system,
    uint64_t* total_threats,
    uint64_t* neutralized,
    uint64_t* false_positives
) {
    if (!system || !total_threats || !neutralized || !false_positives) {
        NIMCP_LOG_ERROR("Invalid arguments");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    *total_threats = system->total_threats_detected;
    *neutralized = system->total_threats_neutralized;
    *false_positives = system->false_positive_count;

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

const char* nimcp_swarm_threat_type_name(NimcpSwarmThreatType type) {
    if (type < THREAT_COUNT) {
        return THREAT_TYPE_NAMES[type];
    }
    return "UNKNOWN";
}

const char* nimcp_swarm_response_type_name(NimcpSwarmResponseType type) {
    if (type < RESPONSE_COUNT) {
        return RESPONSE_TYPE_NAMES[type];
    }
    return "UNKNOWN";
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

nimcp_result_t nimcp_swarm_immune_update(
    NimcpSwarmImmuneSystem* system,
    uint64_t current_time
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Process expired responses */
    for (size_t i = 0; i < system->active_response_count; ) {
        NimcpSwarmResponse* response = &system->active_responses[i];

        if (current_time - response->start_time >= response->duration) {
            /* Remove expired response */
            system->active_responses[i] = system->active_responses[system->active_response_count - 1];
            system->active_response_count--;
        } else {
            i++;
        }
    }

    /* Apply memory decay */
    nimcp_swarm_immune_decay_memory(system, current_time);

    /* Perform affinity maturation periodically */
    static uint64_t last_maturation = 0;
    if (current_time - last_maturation > 60000) { /* Every minute */
        nimcp_swarm_immune_affinity_maturation(system);
        last_maturation = current_time;
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_immune_reset(
    NimcpSwarmImmuneSystem* system,
    bool preserve_memory
) {
    if (!system) {
        NIMCP_LOG_ERROR("Invalid system pointer");
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(system->mutex);

    /* Clear active threats and responses */
    system->active_threat_count = 0;
    system->active_response_count = 0;

    /* Optionally clear memory cells */
    if (!preserve_memory) {
        system->memory_cell_count = 0;
    }

    /* Clear behavior profiles */
    system->behavior_profile_count = 0;

    /* Reset statistics */
    system->total_threats_detected = 0;
    system->total_threats_neutralized = 0;
    system->false_positive_count = 0;

    NIMCP_LOG_INFO("Immune system reset (preserve_memory=%s)",
                   preserve_memory ? "true" : "false");

    nimcp_platform_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}
