/**
 * @file nimcp_swarm_multi.c
 * @brief Multi-Swarm Coordination System Implementation
 */

#include "swarm/nimcp_swarm_multi.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/validation/nimcp_validate.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define CONFLICT_DETECTION_THRESHOLD 0.1  /**< Territory overlap threshold */
#define BRIDGE_QUALITY_THRESHOLD 0.3      /**< Minimum bridge quality */
#define RESOURCE_EXPIRY_TIME 60000        /**< Request expiry (ms) */
#define DISCOVERY_BROADCAST_INTERVAL 5000 /**< Discovery interval (ms) */

/* ============================================================================
 * Internal Message Types
 * ============================================================================ */

#define MSG_TYPE_SWARM_DISCOVERY    0x1001
#define MSG_TYPE_RESOURCE_REQUEST   0x1002
#define MSG_TYPE_RESOURCE_RESPONSE  0x1003
#define MSG_TYPE_TERRITORY_NEGOTIATE 0x1004
#define MSG_TYPE_MISSION_UPDATE     0x1005
#define MSG_TYPE_CONFLICT_ALERT     0x1006

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Generate unique ID
 */
static uint64_t generate_unique_id(void) {
    static uint64_t counter = 1;
    uint64_t timestamp = nimcp_get_time_ms();
    return (timestamp << 20) | (counter++ & 0xFFFFF);
}

/**
 * @brief Calculate health percentage from active agents
 */
static float calculate_health_percentage(uint32_t active, uint32_t total) {
    if (total == 0) return 0.0f;
    return (float)active / (float)total;
}

/**
 * @brief Determine health status from percentage
 */
static nimcp_swarm_health_t determine_health_status(float percentage) {
    if (percentage > 0.9f) return NIMCP_SWARM_HEALTH_EXCELLENT;
    if (percentage > 0.7f) return NIMCP_SWARM_HEALTH_GOOD;
    if (percentage > 0.5f) return NIMCP_SWARM_HEALTH_FAIR;
    if (percentage > 0.3f) return NIMCP_SWARM_HEALTH_POOR;
    return NIMCP_SWARM_HEALTH_CRITICAL;
}

/**
 * @brief Check if point is inside territory
 */
static bool point_in_territory(
    nimcp_coord3d_t point,
    const nimcp_territory_bounds_t* bounds
) {
    return (point.x >= bounds->min.x && point.x <= bounds->max.x &&
            point.y >= bounds->min.y && point.y <= bounds->max.y &&
            point.z >= bounds->min.z && point.z <= bounds->max.z);
}

/**
 * @brief Calculate territory overlap volume
 */
static double calculate_overlap_volume(
    const nimcp_territory_bounds_t* a,
    const nimcp_territory_bounds_t* b
) {
    double overlap_x = fmax(0.0, fmin(a->max.x, b->max.x) - fmax(a->min.x, b->min.x));
    double overlap_y = fmax(0.0, fmin(a->max.y, b->max.y) - fmax(a->min.y, b->min.y));
    double overlap_z = fmax(0.0, fmin(a->max.z, b->max.z) - fmax(a->min.z, b->min.z));

    return overlap_x * overlap_y * overlap_z;
}

/**
 * @brief Hash function for swarm IDs
 */
static uint64_t swarm_id_hash(const void* key) {
    uint64_t id = *(const uint64_t*)key;
    id ^= (id >> 33);
    id *= 0xff51afd7ed558ccdULL;
    id ^= (id >> 33);
    return id;
}

/**
 * @brief Compare function for swarm IDs
 */
static int swarm_id_compare(const void* a, const void* b) {
    uint64_t id_a = *(const uint64_t*)a;
    uint64_t id_b = *(const uint64_t*)b;
    if (id_a < id_b) return -1;
    if (id_a > id_b) return 1;
    return 0;
}

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

nimcp_multi_swarm_coordinator_t* nimcp_multi_swarm_create(
    nimcp_brain_t* brain,
    bio_router_t* router
) {
    NIMCP_LOG_INFO("Creating multi-swarm coordinator");

    nimcp_multi_swarm_coordinator_t* coordinator =
        (nimcp_multi_swarm_coordinator_t*)NIMCP_MALLOC(
            sizeof(nimcp_multi_swarm_coordinator_t));

    if (!coordinator) {
        NIMCP_LOG_ERROR("Failed to allocate multi-swarm coordinator");
        return NULL;
    }

    memset(coordinator, 0, sizeof(nimcp_multi_swarm_coordinator_t));

    coordinator->brain = brain;
    coordinator->router = router;

    /* Create registries */
    coordinator->swarm_registry = nimcp_hash_table_create(
        64, swarm_id_hash, swarm_id_compare);
    if (!coordinator->swarm_registry) {
        NIMCP_LOG_ERROR("Failed to create swarm registry");
        NIMCP_FREE(coordinator);
        return NULL;
    }

    coordinator->mission_registry = nimcp_hash_table_create(
        32, swarm_id_hash, swarm_id_compare);
    if (!coordinator->mission_registry) {
        NIMCP_LOG_ERROR("Failed to create mission registry");
        nimcp_hash_table_destroy(coordinator->swarm_registry);
        NIMCP_FREE(coordinator);
        return NULL;
    }

    /* Initialize locks */
    if (nimcp_rwlock_init(&coordinator->coordinator_lock) != NIMCP_SUCCESS) {
        NIMCP_LOG_ERROR("Failed to initialize coordinator lock");
        nimcp_hash_table_destroy(coordinator->mission_registry);
        nimcp_hash_table_destroy(coordinator->swarm_registry);
        NIMCP_FREE(coordinator);
        return NULL;
    }

    /* Initialize ID generators */
    coordinator->next_swarm_id = 1000;
    coordinator->next_mission_id = 1;
    coordinator->next_conflict_id = 1;

    /* Enable features by default */
    coordinator->enable_auto_negotiation = true;
    coordinator->enable_resource_sharing = true;
    coordinator->enable_bridge_formation = true;

    NIMCP_LOG_INFO("Multi-swarm coordinator created successfully");
    return coordinator;
}

void nimcp_multi_swarm_destroy(nimcp_multi_swarm_coordinator_t* coordinator) {
    if (!coordinator) return;

    NIMCP_LOG_INFO("Destroying multi-swarm coordinator");

    /* Destroy all super-swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        if (coordinator->super_swarms[i]) {
            nimcp_super_swarm_destroy(coordinator->super_swarms[i]);
        }
    }

    /* Clean up registries */
    if (coordinator->swarm_registry) {
        nimcp_hash_table_destroy(coordinator->swarm_registry);
    }
    if (coordinator->mission_registry) {
        nimcp_hash_table_destroy(coordinator->mission_registry);
    }

    /* Destroy lock */
    nimcp_rwlock_destroy(&coordinator->coordinator_lock);

    NIMCP_FREE(coordinator);
    NIMCP_LOG_INFO("Multi-swarm coordinator destroyed");
}

/* ============================================================================
 * Swarm Identity Management
 * ============================================================================ */

nimcp_swarm_identity_t* nimcp_swarm_identity_create(
    nimcp_multi_swarm_coordinator_t* coordinator,
    const char* name,
    uint32_t agent_count
) {
    if (!coordinator || !name) {
        NIMCP_LOG_ERROR("Invalid parameters for swarm identity creation");
        return NULL;
    }

    nimcp_swarm_identity_t* identity =
        (nimcp_swarm_identity_t*)NIMCP_MALLOC(sizeof(nimcp_swarm_identity_t));

    if (!identity) {
        NIMCP_LOG_ERROR("Failed to allocate swarm identity");
        return NULL;
    }

    memset(identity, 0, sizeof(nimcp_swarm_identity_t));

    /* Assign unique ID */
    nimcp_rwlock_write_lock(&coordinator->coordinator_lock);
    identity->swarm_id = coordinator->next_swarm_id++;
    nimcp_rwlock_write_unlock(&coordinator->coordinator_lock);

    /* Set basic properties */
    strncpy(identity->name, name, NIMCP_SWARM_NAME_MAX - 1);
    identity->name[NIMCP_SWARM_NAME_MAX - 1] = '\0';
    identity->agent_count = agent_count;
    identity->active_agents = agent_count;
    identity->formation_time = nimcp_get_time_ms();
    identity->last_contact = identity->formation_time;

    /* Initialize health */
    identity->health_percentage = 1.0f;
    identity->health = NIMCP_SWARM_HEALTH_EXCELLENT;

    /* Initialize territory to zero bounds */
    memset(&identity->territory, 0, sizeof(nimcp_territory_bounds_t));
    identity->territory.timestamp = identity->formation_time;

    NIMCP_LOG_INFO("Created swarm identity: ID=%lu, Name=%s, Agents=%u",
                   identity->swarm_id, identity->name, agent_count);

    return identity;
}

nimcp_result_t nimcp_swarm_register(
    nimcp_multi_swarm_coordinator_t* coordinator,
    nimcp_swarm_identity_t* identity
) {
    if (!coordinator || !identity) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    NIMCP_LOG_INFO("Registering swarm: ID=%lu, Name=%s",
                   identity->swarm_id, identity->name);

    nimcp_rwlock_write_lock(&coordinator->coordinator_lock);

    /* Check if already registered */
    if (nimcp_hash_table_get(coordinator->swarm_registry, &identity->swarm_id)) {
        NIMCP_LOG_WARN("Swarm already registered: ID=%lu", identity->swarm_id);
        nimcp_rwlock_write_unlock(&coordinator->coordinator_lock);
        return NIMCP_ERROR_ALREADY_EXISTS;
    }

    /* Register swarm */
    nimcp_result_t result = nimcp_hash_table_insert(
        coordinator->swarm_registry,
        &identity->swarm_id,
        identity
    );

    nimcp_rwlock_write_unlock(&coordinator->coordinator_lock);

    if (result == NIMCP_SUCCESS) {
        NIMCP_LOG_INFO("Swarm registered successfully: ID=%lu", identity->swarm_id);

        /* Broadcast discovery if router available */
        if (coordinator->router && coordinator->enable_bridge_formation) {
            nimcp_multi_swarm_broadcast_discovery(coordinator, identity->swarm_id);
        }
    } else {
        NIMCP_LOG_ERROR("Failed to register swarm: ID=%lu", identity->swarm_id);
    }

    return result;
}

nimcp_result_t nimcp_swarm_unregister(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t swarm_id
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    NIMCP_LOG_INFO("Unregistering swarm: ID=%lu", swarm_id);

    nimcp_rwlock_write_lock(&coordinator->coordinator_lock);

    nimcp_swarm_identity_t* identity =
        nimcp_hash_table_remove(coordinator->swarm_registry, &swarm_id);

    nimcp_rwlock_write_unlock(&coordinator->coordinator_lock);

    if (!identity) {
        NIMCP_LOG_WARN("Swarm not found for unregistration: ID=%lu", swarm_id);
        return NIMCP_ERROR_NOT_FOUND;
    }

    NIMCP_LOG_INFO("Swarm unregistered: ID=%lu", swarm_id);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_add_capability(
    nimcp_swarm_identity_t* identity,
    nimcp_swarm_capability_type_t type,
    float proficiency,
    uint32_t capacity,
    bool is_lendable
) {
    if (!identity) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    if (identity->capability_count >= NIMCP_MAX_SWARM_CAPABILITIES) {
        NIMCP_LOG_ERROR("Maximum capabilities reached for swarm: ID=%lu",
                        identity->swarm_id);
        return NIMCP_ERROR_OUT_OF_RANGE;
    }

    /* Clamp proficiency to [0, 1] */
    proficiency = fmaxf(0.0f, fminf(1.0f, proficiency));

    /* Add capability */
    nimcp_swarm_capability_t* cap =
        &identity->capabilities[identity->capability_count++];

    cap->type = type;
    cap->proficiency = proficiency;
    cap->capacity = capacity;
    cap->available = capacity;
    cap->is_lendable = is_lendable;

    NIMCP_LOG_DEBUG("Added capability to swarm ID=%lu: Type=%d, Prof=%.2f, Cap=%u",
                    identity->swarm_id, type, proficiency, capacity);

    return NIMCP_SUCCESS;
}

void nimcp_swarm_update_health(
    nimcp_swarm_identity_t* identity,
    uint32_t active_agents
) {
    if (!identity) return;

    identity->active_agents = active_agents;
    identity->health_percentage = calculate_health_percentage(
        active_agents, identity->agent_count);
    identity->health = determine_health_status(identity->health_percentage);
    identity->last_contact = nimcp_get_time_ms();

    NIMCP_LOG_DEBUG("Swarm health updated: ID=%lu, Active=%u/%u, Health=%.1f%%",
                    identity->swarm_id, active_agents, identity->agent_count,
                    identity->health_percentage * 100.0f);
}

void nimcp_swarm_identity_destroy(nimcp_swarm_identity_t* identity) {
    if (!identity) return;

    NIMCP_LOG_DEBUG("Destroying swarm identity: ID=%lu", identity->swarm_id);
    NIMCP_FREE(identity);
}

/* ============================================================================
 * Super-Swarm Management
 * ============================================================================ */

nimcp_super_swarm_t* nimcp_super_swarm_create(
    nimcp_multi_swarm_coordinator_t* coordinator,
    const char* name
) {
    if (!coordinator || !name) {
        NIMCP_LOG_ERROR("Invalid parameters for super-swarm creation");
        return NULL;
    }

    nimcp_super_swarm_t* super_swarm =
        (nimcp_super_swarm_t*)NIMCP_MALLOC(sizeof(nimcp_super_swarm_t));

    if (!super_swarm) {
        NIMCP_LOG_ERROR("Failed to allocate super-swarm");
        return NULL;
    }

    memset(super_swarm, 0, sizeof(nimcp_super_swarm_t));

    /* Assign ID */
    super_swarm->super_swarm_id = generate_unique_id();
    strncpy(super_swarm->name, name, NIMCP_SWARM_NAME_MAX - 1);
    super_swarm->name[NIMCP_SWARM_NAME_MAX - 1] = '\0';

    /* Create resource request table */
    super_swarm->resource_requests = nimcp_hash_table_create(
        16, swarm_id_hash, swarm_id_compare);
    if (!super_swarm->resource_requests) {
        NIMCP_LOG_ERROR("Failed to create resource request table");
        NIMCP_FREE(super_swarm);
        return NULL;
    }

    /* Create conflicts vector */
    super_swarm->conflicts = nimcp_vector_create(sizeof(nimcp_swarm_conflict_t));
    if (!super_swarm->conflicts) {
        NIMCP_LOG_ERROR("Failed to create conflicts vector");
        nimcp_hash_table_destroy(super_swarm->resource_requests);
        NIMCP_FREE(super_swarm);
        return NULL;
    }

    /* Initialize locks */
    if (nimcp_rwlock_init(&super_swarm->swarm_lock) != NIMCP_SUCCESS ||
        nimcp_rwlock_init(&super_swarm->mission_lock) != NIMCP_SUCCESS ||
        nimcp_rwlock_init(&super_swarm->bridge_lock) != NIMCP_SUCCESS) {
        NIMCP_LOG_ERROR("Failed to initialize super-swarm locks");
        nimcp_vector_destroy(super_swarm->conflicts);
        nimcp_hash_table_destroy(super_swarm->resource_requests);
        NIMCP_FREE(super_swarm);
        return NULL;
    }

    /* Add to coordinator */
    if (coordinator->super_swarm_count < NIMCP_MAX_SWARMS_PER_SUPER) {
        coordinator->super_swarms[coordinator->super_swarm_count++] = super_swarm;
    }

    NIMCP_LOG_INFO("Created super-swarm: ID=%lu, Name=%s",
                   super_swarm->super_swarm_id, super_swarm->name);

    return super_swarm;
}

nimcp_result_t nimcp_super_swarm_add_swarm(
    nimcp_super_swarm_t* super_swarm,
    nimcp_swarm_identity_t* identity
) {
    if (!super_swarm || !identity) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_rwlock_write_lock(&super_swarm->swarm_lock);

    if (super_swarm->swarm_count >= NIMCP_MAX_SWARMS_PER_SUPER) {
        NIMCP_LOG_ERROR("Super-swarm at maximum capacity");
        nimcp_rwlock_write_unlock(&super_swarm->swarm_lock);
        return NIMCP_ERROR_OUT_OF_RANGE;
    }

    super_swarm->swarms[super_swarm->swarm_count++] = identity;

    /* Update overall territory */
    if (super_swarm->swarm_count == 1) {
        super_swarm->overall_territory = identity->territory;
    } else {
        /* Expand to encompass new swarm */
        super_swarm->overall_territory.min.x = fmin(
            super_swarm->overall_territory.min.x, identity->territory.min.x);
        super_swarm->overall_territory.min.y = fmin(
            super_swarm->overall_territory.min.y, identity->territory.min.y);
        super_swarm->overall_territory.min.z = fmin(
            super_swarm->overall_territory.min.z, identity->territory.min.z);

        super_swarm->overall_territory.max.x = fmax(
            super_swarm->overall_territory.max.x, identity->territory.max.x);
        super_swarm->overall_territory.max.y = fmax(
            super_swarm->overall_territory.max.y, identity->territory.max.y);
        super_swarm->overall_territory.max.z = fmax(
            super_swarm->overall_territory.max.z, identity->territory.max.z);
    }

    nimcp_rwlock_write_unlock(&super_swarm->swarm_lock);

    NIMCP_LOG_INFO("Added swarm to super-swarm: Swarm=%lu, SuperSwarm=%lu",
                   identity->swarm_id, super_swarm->super_swarm_id);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_super_swarm_remove_swarm(
    nimcp_super_swarm_t* super_swarm,
    uint64_t swarm_id
) {
    if (!super_swarm) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_rwlock_write_lock(&super_swarm->swarm_lock);

    bool found = false;
    for (uint32_t i = 0; i < super_swarm->swarm_count; i++) {
        if (super_swarm->swarms[i]->swarm_id == swarm_id) {
            /* Shift remaining swarms */
            for (uint32_t j = i; j < super_swarm->swarm_count - 1; j++) {
                super_swarm->swarms[j] = super_swarm->swarms[j + 1];
            }
            super_swarm->swarm_count--;
            found = true;
            break;
        }
    }

    nimcp_rwlock_write_unlock(&super_swarm->swarm_lock);

    if (!found) {
        NIMCP_LOG_WARN("Swarm not found in super-swarm: ID=%lu", swarm_id);
        return NIMCP_ERROR_NOT_FOUND;
    }

    NIMCP_LOG_INFO("Removed swarm from super-swarm: Swarm=%lu", swarm_id);
    return NIMCP_SUCCESS;
}

void nimcp_super_swarm_destroy(nimcp_super_swarm_t* super_swarm) {
    if (!super_swarm) return;

    NIMCP_LOG_INFO("Destroying super-swarm: ID=%lu", super_swarm->super_swarm_id);

    /* Clean up resources */
    if (super_swarm->resource_requests) {
        nimcp_hash_table_destroy(super_swarm->resource_requests);
    }
    if (super_swarm->conflicts) {
        nimcp_vector_destroy(super_swarm->conflicts);
    }

    /* Destroy locks */
    nimcp_rwlock_destroy(&super_swarm->swarm_lock);
    nimcp_rwlock_destroy(&super_swarm->mission_lock);
    nimcp_rwlock_destroy(&super_swarm->bridge_lock);

    NIMCP_FREE(super_swarm);
}

/* ============================================================================
 * Territory Management
 * ============================================================================ */

nimcp_result_t nimcp_swarm_set_territory(
    nimcp_swarm_identity_t* identity,
    nimcp_coord3d_t min,
    nimcp_coord3d_t max,
    bool is_dynamic,
    float priority
) {
    if (!identity) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    /* Validate bounds */
    if (min.x > max.x || min.y > max.y || min.z > max.z) {
        NIMCP_LOG_ERROR("Invalid territory bounds: min > max");
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    identity->territory.min = min;
    identity->territory.max = max;
    identity->territory.is_dynamic = is_dynamic;
    identity->territory.priority = priority;
    identity->territory.timestamp = nimcp_get_time_ms();

    NIMCP_LOG_INFO("Set territory for swarm ID=%lu: [%.1f,%.1f,%.1f]-[%.1f,%.1f,%.1f]",
                   identity->swarm_id,
                   min.x, min.y, min.z, max.x, max.y, max.z);

    return NIMCP_SUCCESS;
}

bool nimcp_territory_overlaps(
    const nimcp_territory_bounds_t* bounds_a,
    const nimcp_territory_bounds_t* bounds_b
) {
    if (!bounds_a || !bounds_b) {
        return false;
    }

    double overlap = calculate_overlap_volume(bounds_a, bounds_b);
    return overlap > CONFLICT_DETECTION_THRESHOLD;
}

nimcp_result_t nimcp_territory_negotiate(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t swarm_a,
    uint64_t swarm_b
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_swarm_identity_t* identity_a = nimcp_swarm_get(coordinator, swarm_a);
    nimcp_swarm_identity_t* identity_b = nimcp_swarm_get(coordinator, swarm_b);

    if (!identity_a || !identity_b) {
        NIMCP_LOG_ERROR("One or both swarms not found for negotiation");
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Check if territories overlap */
    if (!nimcp_territory_overlaps(&identity_a->territory, &identity_b->territory)) {
        NIMCP_LOG_DEBUG("No overlap detected, negotiation not needed");
        return NIMCP_SUCCESS;
    }

    NIMCP_LOG_INFO("Negotiating territory between swarms %lu and %lu",
                   swarm_a, swarm_b);

    /* Simple negotiation: adjust based on priority and dynamic flags */
    if (identity_a->territory.is_dynamic && !identity_b->territory.is_dynamic) {
        /* A adjusts to avoid B */
        NIMCP_LOG_INFO("Swarm %lu adjusting territory (dynamic)", swarm_a);
        /* In a real implementation, adjust bounds here */
        return NIMCP_SUCCESS;
    } else if (!identity_a->territory.is_dynamic && identity_b->territory.is_dynamic) {
        /* B adjusts to avoid A */
        NIMCP_LOG_INFO("Swarm %lu adjusting territory (dynamic)", swarm_b);
        return NIMCP_SUCCESS;
    } else if (identity_a->territory.priority > identity_b->territory.priority) {
        /* B yields to A */
        NIMCP_LOG_INFO("Swarm %lu yields to higher priority swarm %lu",
                       swarm_b, swarm_a);
        return NIMCP_SUCCESS;
    } else {
        /* A yields to B */
        NIMCP_LOG_INFO("Swarm %lu yields to higher priority swarm %lu",
                       swarm_a, swarm_b);
        return NIMCP_SUCCESS;
    }
}

uint32_t nimcp_territory_detect_conflicts(
    nimcp_multi_swarm_coordinator_t* coordinator,
    nimcp_vector_t* conflicts
) {
    if (!coordinator || !conflicts) {
        return 0;
    }

    uint32_t conflict_count = 0;

    nimcp_rwlock_read_lock(&coordinator->coordinator_lock);

    /* Compare all pairs of swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_rwlock_read_lock(&super->swarm_lock);

        for (uint32_t j = 0; j < super->swarm_count; j++) {
            for (uint32_t k = j + 1; k < super->swarm_count; k++) {
                nimcp_swarm_identity_t* swarm_a = super->swarms[j];
                nimcp_swarm_identity_t* swarm_b = super->swarms[k];

                if (nimcp_territory_overlaps(&swarm_a->territory,
                                            &swarm_b->territory)) {
                    /* Create conflict record */
                    nimcp_swarm_conflict_t conflict = {0};
                    conflict.conflict_id = coordinator->next_conflict_id++;
                    conflict.swarm_ids[0] = swarm_a->swarm_id;
                    conflict.swarm_ids[1] = swarm_b->swarm_id;
                    conflict.swarm_count = 2;
                    conflict.detection_time = nimcp_get_time_ms();
                    conflict.is_resolved = false;

                    snprintf(conflict.description, sizeof(conflict.description),
                            "Territory overlap between swarms %lu and %lu",
                            swarm_a->swarm_id, swarm_b->swarm_id);

                    nimcp_vector_push_back(conflicts, &conflict);
                    conflict_count++;

                    NIMCP_LOG_WARN("Territory conflict detected: Swarms %lu and %lu",
                                  swarm_a->swarm_id, swarm_b->swarm_id);
                }
            }
        }

        nimcp_rwlock_read_unlock(&super->swarm_lock);
    }

    nimcp_rwlock_read_unlock(&coordinator->coordinator_lock);

    return conflict_count;
}

/* ============================================================================
 * Resource Sharing
 * ============================================================================ */

uint64_t nimcp_resource_request(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t requesting_swarm,
    uint64_t target_swarm,
    nimcp_resource_request_type_t type,
    uint32_t quantity,
    nimcp_mission_priority_t priority
) {
    if (!coordinator) {
        return 0;
    }

    /* Verify both swarms exist */
    if (!nimcp_swarm_get(coordinator, requesting_swarm) ||
        !nimcp_swarm_get(coordinator, target_swarm)) {
        NIMCP_LOG_ERROR("Invalid swarm IDs for resource request");
        return 0;
    }

    nimcp_resource_request_t* request =
        (nimcp_resource_request_t*)NIMCP_MALLOC(sizeof(nimcp_resource_request_t));

    if (!request) {
        NIMCP_LOG_ERROR("Failed to allocate resource request");
        return 0;
    }

    memset(request, 0, sizeof(nimcp_resource_request_t));

    request->request_id = generate_unique_id();
    request->requesting_swarm = requesting_swarm;
    request->target_swarm = target_swarm;
    request->type = type;
    request->quantity = quantity;
    request->priority = priority;
    request->expiry_time = nimcp_get_time_ms() + RESOURCE_EXPIRY_TIME;
    request->is_approved = false;

    /* Store in appropriate super-swarm */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        /* Check if both swarms are in this super-swarm */
        bool has_requesting = false, has_target = false;
        for (uint32_t j = 0; j < super->swarm_count; j++) {
            if (super->swarms[j]->swarm_id == requesting_swarm) has_requesting = true;
            if (super->swarms[j]->swarm_id == target_swarm) has_target = true;
        }

        if (has_requesting && has_target) {
            nimcp_hash_table_insert(super->resource_requests,
                                   &request->request_id, request);
            break;
        }
    }

    NIMCP_LOG_INFO("Created resource request: ID=%lu, From=%lu, To=%lu, Type=%d",
                   request->request_id, requesting_swarm, target_swarm, type);

    return request->request_id;
}

nimcp_result_t nimcp_resource_approve(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t request_id,
    float cost
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    /* Find request in super-swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_resource_request_t* request =
            nimcp_hash_table_get(super->resource_requests, &request_id);

        if (request) {
            request->is_approved = true;
            request->cost = cost;

            NIMCP_LOG_INFO("Approved resource request: ID=%lu, Cost=%.2f",
                          request_id, cost);
            return NIMCP_SUCCESS;
        }
    }

    NIMCP_LOG_WARN("Resource request not found: ID=%lu", request_id);
    return NIMCP_ERROR_NOT_FOUND;
}

nimcp_result_t nimcp_resource_deny(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t request_id
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    /* Find and remove request */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_resource_request_t* request =
            nimcp_hash_table_remove(super->resource_requests, &request_id);

        if (request) {
            NIMCP_LOG_INFO("Denied resource request: ID=%lu", request_id);
            NIMCP_FREE(request);
            return NIMCP_SUCCESS;
        }
    }

    NIMCP_LOG_WARN("Resource request not found: ID=%lu", request_id);
    return NIMCP_ERROR_NOT_FOUND;
}

uint32_t nimcp_resource_process_requests(
    nimcp_multi_swarm_coordinator_t* coordinator,
    nimcp_resource_allocator_fn allocator,
    void* user_data
) {
    if (!coordinator) {
        return 0;
    }

    uint32_t processed = 0;
    uint64_t current_time = nimcp_get_time_ms();

    /* Process requests in all super-swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super || !super->resource_requests) continue;

        /* This is a simplified implementation */
        /* In a real implementation, iterate through hash table */
        NIMCP_LOG_DEBUG("Processing resource requests for super-swarm %lu",
                       super->super_swarm_id);
    }

    return processed;
}

/* ============================================================================
 * Mission Management
 * ============================================================================ */

uint64_t nimcp_mission_create(
    nimcp_multi_swarm_coordinator_t* coordinator,
    const char* description,
    nimcp_mission_priority_t priority,
    nimcp_territory_bounds_t operation_area,
    uint64_t deadline
) {
    if (!coordinator || !description) {
        return 0;
    }

    nimcp_mission_assignment_t* mission =
        (nimcp_mission_assignment_t*)NIMCP_MALLOC(sizeof(nimcp_mission_assignment_t));

    if (!mission) {
        NIMCP_LOG_ERROR("Failed to allocate mission");
        return 0;
    }

    memset(mission, 0, sizeof(nimcp_mission_assignment_t));

    nimcp_rwlock_write_lock(&coordinator->coordinator_lock);
    mission->mission_id = coordinator->next_mission_id++;
    nimcp_rwlock_write_unlock(&coordinator->coordinator_lock);

    strncpy(mission->description, description, sizeof(mission->description) - 1);
    mission->priority = priority;
    mission->status = NIMCP_MISSION_STATUS_PENDING;
    mission->operation_area = operation_area;
    mission->start_time = nimcp_get_time_ms();
    mission->deadline = deadline;
    mission->progress = 0.0f;

    /* Store in mission registry */
    nimcp_hash_table_insert(coordinator->mission_registry,
                           &mission->mission_id, mission);

    NIMCP_LOG_INFO("Created mission: ID=%lu, Priority=%d, Desc='%s'",
                   mission->mission_id, priority, description);

    return mission->mission_id;
}

nimcp_result_t nimcp_mission_assign_swarms(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t mission_id,
    const uint64_t* swarm_ids,
    uint32_t swarm_count
) {
    if (!coordinator || !swarm_ids || swarm_count == 0) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_mission_assignment_t* mission =
        nimcp_hash_table_get(coordinator->mission_registry, &mission_id);

    if (!mission) {
        NIMCP_LOG_ERROR("Mission not found: ID=%lu", mission_id);
        return NIMCP_ERROR_NOT_FOUND;
    }

    if (swarm_count > NIMCP_MAX_SWARMS_PER_SUPER) {
        NIMCP_LOG_ERROR("Too many swarms for mission assignment");
        return NIMCP_ERROR_OUT_OF_RANGE;
    }

    /* Assign swarms */
    memcpy(mission->assigned_swarms, swarm_ids, swarm_count * sizeof(uint64_t));
    mission->swarm_count = swarm_count;
    mission->status = NIMCP_MISSION_STATUS_ASSIGNED;

    NIMCP_LOG_INFO("Assigned %u swarms to mission %lu", swarm_count, mission_id);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_mission_update_progress(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t mission_id,
    float progress
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_mission_assignment_t* mission =
        nimcp_hash_table_get(coordinator->mission_registry, &mission_id);

    if (!mission) {
        return NIMCP_ERROR_NOT_FOUND;
    }

    mission->progress = fmaxf(0.0f, fminf(1.0f, progress));

    if (mission->status == NIMCP_MISSION_STATUS_ASSIGNED) {
        mission->status = NIMCP_MISSION_STATUS_ACTIVE;
    }

    NIMCP_LOG_DEBUG("Mission progress updated: ID=%lu, Progress=%.1f%%",
                    mission_id, mission->progress * 100.0f);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_mission_complete(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t mission_id,
    bool success
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_mission_assignment_t* mission =
        nimcp_hash_table_get(coordinator->mission_registry, &mission_id);

    if (!mission) {
        return NIMCP_ERROR_NOT_FOUND;
    }

    mission->status = success ? NIMCP_MISSION_STATUS_COMPLETED :
                               NIMCP_MISSION_STATUS_FAILED;
    mission->progress = success ? 1.0f : mission->progress;

    NIMCP_LOG_INFO("Mission completed: ID=%lu, Success=%s",
                   mission_id, success ? "YES" : "NO");

    return NIMCP_SUCCESS;
}

nimcp_mission_assignment_t* nimcp_mission_get(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t mission_id
) {
    if (!coordinator) {
        return NULL;
    }

    return nimcp_hash_table_get(coordinator->mission_registry, &mission_id);
}

/* ============================================================================
 * Communication Bridges
 * ============================================================================ */

uint64_t nimcp_comm_bridge_create(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t swarm_a,
    uint64_t swarm_b,
    const uint32_t* relay_agents,
    uint32_t relay_count
) {
    if (!coordinator) {
        return 0;
    }

    /* Find appropriate super-swarm */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        bool has_a = false, has_b = false;
        for (uint32_t j = 0; j < super->swarm_count; j++) {
            if (super->swarms[j]->swarm_id == swarm_a) has_a = true;
            if (super->swarms[j]->swarm_id == swarm_b) has_b = true;
        }

        if (has_a && has_b) {
            nimcp_rwlock_write_lock(&super->bridge_lock);

            if (super->bridge_count >= NIMCP_MAX_COMM_BRIDGES) {
                NIMCP_LOG_ERROR("Maximum bridges reached for super-swarm");
                nimcp_rwlock_write_unlock(&super->bridge_lock);
                return 0;
            }

            nimcp_comm_bridge_t* bridge = &super->bridges[super->bridge_count++];
            memset(bridge, 0, sizeof(nimcp_comm_bridge_t));

            bridge->bridge_id = generate_unique_id();
            bridge->swarm_a = swarm_a;
            bridge->swarm_b = swarm_b;
            bridge->link_quality = 1.0f;
            bridge->is_active = true;
            bridge->last_message_time = nimcp_get_time_ms();

            if (relay_agents && relay_count > 0) {
                uint32_t count = (relay_count > 4) ? 4 : relay_count;
                memcpy(bridge->relay_agents, relay_agents, count * sizeof(uint32_t));
                bridge->relay_count = count;
            }

            nimcp_rwlock_write_unlock(&super->bridge_lock);

            NIMCP_LOG_INFO("Created communication bridge: ID=%lu, Between %lu and %lu",
                          bridge->bridge_id, swarm_a, swarm_b);

            return bridge->bridge_id;
        }
    }

    NIMCP_LOG_ERROR("Swarms not found in same super-swarm for bridge creation");
    return 0;
}

nimcp_result_t nimcp_comm_bridge_update_quality(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t bridge_id,
    float link_quality
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    link_quality = fmaxf(0.0f, fminf(1.0f, link_quality));

    /* Find bridge in super-swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_rwlock_write_lock(&super->bridge_lock);

        for (uint32_t j = 0; j < super->bridge_count; j++) {
            if (super->bridges[j].bridge_id == bridge_id) {
                super->bridges[j].link_quality = link_quality;

                /* Deactivate if quality too low */
                if (link_quality < BRIDGE_QUALITY_THRESHOLD) {
                    super->bridges[j].is_active = false;
                    NIMCP_LOG_WARN("Bridge deactivated due to low quality: ID=%lu",
                                  bridge_id);
                }

                nimcp_rwlock_write_unlock(&super->bridge_lock);
                return NIMCP_SUCCESS;
            }
        }

        nimcp_rwlock_write_unlock(&super->bridge_lock);
    }

    return NIMCP_ERROR_NOT_FOUND;
}

nimcp_result_t nimcp_comm_bridge_deactivate(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t bridge_id
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    /* Find and deactivate bridge */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_rwlock_write_lock(&super->bridge_lock);

        for (uint32_t j = 0; j < super->bridge_count; j++) {
            if (super->bridges[j].bridge_id == bridge_id) {
                super->bridges[j].is_active = false;
                nimcp_rwlock_write_unlock(&super->bridge_lock);

                NIMCP_LOG_INFO("Bridge deactivated: ID=%lu", bridge_id);
                return NIMCP_SUCCESS;
            }
        }

        nimcp_rwlock_write_unlock(&super->bridge_lock);
    }

    return NIMCP_ERROR_NOT_FOUND;
}

nimcp_result_t nimcp_comm_bridge_route_message(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t from_swarm,
    uint64_t to_swarm,
    bio_message_header_t* message
) {
    if (!coordinator || !message) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    if (!coordinator->router) {
        NIMCP_LOG_WARN("No bio-router available for message routing");
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    /* Find appropriate bridge */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_rwlock_read_lock(&super->bridge_lock);

        for (uint32_t j = 0; j < super->bridge_count; j++) {
            nimcp_comm_bridge_t* bridge = &super->bridges[j];

            if (bridge->is_active &&
                ((bridge->swarm_a == from_swarm && bridge->swarm_b == to_swarm) ||
                 (bridge->swarm_a == to_swarm && bridge->swarm_b == from_swarm))) {

                bridge->last_message_time = nimcp_get_time_ms();
                nimcp_rwlock_read_unlock(&super->bridge_lock);

                /* Route through bio-async system */
                nimcp_result_t result = nimcp_bio_router_route(
                    coordinator->router, message);

                NIMCP_LOG_DEBUG("Routed message via bridge %lu: From=%lu To=%lu",
                               bridge->bridge_id, from_swarm, to_swarm);

                return result;
            }
        }

        nimcp_rwlock_read_unlock(&super->bridge_lock);
    }

    NIMCP_LOG_WARN("No active bridge found between swarms %lu and %lu",
                   from_swarm, to_swarm);
    return NIMCP_ERROR_NOT_FOUND;
}

/* ============================================================================
 * Conflict Resolution
 * ============================================================================ */

uint32_t nimcp_conflict_detect(
    nimcp_multi_swarm_coordinator_t* coordinator
) {
    if (!coordinator) {
        return 0;
    }

    uint32_t total_conflicts = 0;

    /* Detect conflicts in all super-swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_vector_t* conflicts = super->conflicts;
        if (!conflicts) continue;

        /* Clear old conflicts */
        nimcp_vector_clear(conflicts);

        /* Detect new conflicts */
        uint32_t count = nimcp_territory_detect_conflicts(coordinator, conflicts);
        total_conflicts += count;

        NIMCP_LOG_INFO("Detected %u conflicts in super-swarm %lu",
                      count, super->super_swarm_id);
    }

    return total_conflicts;
}

nimcp_result_t nimcp_conflict_resolve(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t conflict_id,
    nimcp_conflict_resolution_t strategy,
    nimcp_conflict_resolver_fn resolver,
    void* user_data
) {
    if (!coordinator) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    /* Find conflict in super-swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super || !super->conflicts) continue;

        for (size_t j = 0; j < nimcp_vector_size(super->conflicts); j++) {
            nimcp_swarm_conflict_t* conflict =
                (nimcp_swarm_conflict_t*)nimcp_vector_at(super->conflicts, j);

            if (conflict && conflict->conflict_id == conflict_id) {
                conflict->strategy = strategy;

                /* Use custom resolver if provided */
                bool resolved = true;
                if (resolver) {
                    resolved = resolver(coordinator, conflict, user_data);
                } else {
                    /* Default resolution based on strategy */
                    switch (strategy) {
                        case NIMCP_CONFLICT_PRIORITY:
                            /* Handled by territory negotiation */
                            if (conflict->swarm_count >= 2) {
                                nimcp_territory_negotiate(coordinator,
                                    conflict->swarm_ids[0],
                                    conflict->swarm_ids[1]);
                            }
                            break;

                        case NIMCP_CONFLICT_NEGOTIATION:
                            NIMCP_LOG_INFO("Initiating negotiation for conflict %lu",
                                          conflict_id);
                            break;

                        default:
                            NIMCP_LOG_INFO("Resolving conflict %lu with strategy %d",
                                          conflict_id, strategy);
                            break;
                    }
                }

                if (resolved) {
                    conflict->is_resolved = true;
                    conflict->resolution_time = nimcp_get_time_ms();
                    NIMCP_LOG_INFO("Conflict resolved: ID=%lu", conflict_id);
                    return NIMCP_SUCCESS;
                }

                return NIMCP_ERROR_GENERAL;
            }
        }
    }

    return NIMCP_ERROR_NOT_FOUND;
}

uint32_t nimcp_conflict_auto_resolve(
    nimcp_multi_swarm_coordinator_t* coordinator,
    nimcp_conflict_resolver_fn resolver,
    void* user_data
) {
    if (!coordinator || !coordinator->enable_auto_negotiation) {
        return 0;
    }

    uint32_t resolved_count = 0;

    /* Auto-resolve conflicts in all super-swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super || !super->conflicts) continue;

        for (size_t j = 0; j < nimcp_vector_size(super->conflicts); j++) {
            nimcp_swarm_conflict_t* conflict =
                (nimcp_swarm_conflict_t*)nimcp_vector_at(super->conflicts, j);

            if (conflict && !conflict->is_resolved) {
                /* Try to resolve */
                nimcp_result_t result = nimcp_conflict_resolve(
                    coordinator,
                    conflict->conflict_id,
                    NIMCP_CONFLICT_PRIORITY,
                    resolver,
                    user_data
                );

                if (result == NIMCP_SUCCESS) {
                    resolved_count++;
                }
            }
        }
    }

    NIMCP_LOG_INFO("Auto-resolved %u conflicts", resolved_count);
    return resolved_count;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

uint32_t nimcp_multi_swarm_process_inbox(
    nimcp_multi_swarm_coordinator_t* coordinator
) {
    if (!coordinator || !coordinator->router) {
        return 0;
    }

    uint32_t processed = 0;

    /* Process messages from bio-async router */
    bio_message_header_t* message;
    while ((message = nimcp_bio_router_dequeue(coordinator->router)) != NULL) {
        /* Handle different message types */
        switch (message->type) {
            case MSG_TYPE_SWARM_DISCOVERY:
                NIMCP_LOG_DEBUG("Received swarm discovery message");
                break;

            case MSG_TYPE_RESOURCE_REQUEST:
                NIMCP_LOG_DEBUG("Received resource request message");
                break;

            case MSG_TYPE_TERRITORY_NEGOTIATE:
                NIMCP_LOG_DEBUG("Received territory negotiation message");
                break;

            case MSG_TYPE_MISSION_UPDATE:
                NIMCP_LOG_DEBUG("Received mission update message");
                break;

            case MSG_TYPE_CONFLICT_ALERT:
                NIMCP_LOG_DEBUG("Received conflict alert message");
                break;

            default:
                NIMCP_LOG_DEBUG("Received unknown message type: %u", message->type);
                break;
        }

        nimcp_bio_message_destroy(message);
        processed++;
    }

    return processed;
}

nimcp_result_t nimcp_multi_swarm_broadcast_discovery(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t swarm_id
) {
    if (!coordinator || !coordinator->router) {
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    nimcp_swarm_identity_t* identity = nimcp_swarm_get(coordinator, swarm_id);
    if (!identity) {
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Create discovery message */
    bio_message_header_t* message = nimcp_bio_message_create(
        MSG_TYPE_SWARM_DISCOVERY,
        identity,
        sizeof(nimcp_swarm_identity_t)
    );

    if (!message) {
        NIMCP_LOG_ERROR("Failed to create discovery message");
        return NIMCP_ERROR_OUT_OF_MEMORY;
    }

    /* Broadcast via router */
    nimcp_result_t result = nimcp_bio_router_broadcast(coordinator->router, message);

    nimcp_bio_message_destroy(message);

    NIMCP_LOG_INFO("Broadcasted discovery for swarm %lu", swarm_id);
    return result;
}

nimcp_result_t nimcp_multi_swarm_send_message(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t from_swarm,
    uint64_t to_swarm,
    uint32_t message_type,
    const void* payload,
    size_t payload_size
) {
    if (!coordinator || !payload) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    /* Create message */
    bio_message_header_t* message = nimcp_bio_message_create(
        message_type,
        payload,
        payload_size
    );

    if (!message) {
        return NIMCP_ERROR_OUT_OF_MEMORY;
    }

    /* Route through bridge */
    nimcp_result_t result = nimcp_comm_bridge_route_message(
        coordinator,
        from_swarm,
        to_swarm,
        message
    );

    nimcp_bio_message_destroy(message);

    return result;
}

/* ============================================================================
 * Query and Statistics
 * ============================================================================ */

nimcp_swarm_identity_t* nimcp_swarm_get(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint64_t swarm_id
) {
    if (!coordinator) {
        return NULL;
    }

    nimcp_rwlock_read_lock(&coordinator->coordinator_lock);
    nimcp_swarm_identity_t* identity =
        nimcp_hash_table_get(coordinator->swarm_registry, &swarm_id);
    nimcp_rwlock_read_unlock(&coordinator->coordinator_lock);

    return identity;
}

uint32_t nimcp_swarm_find_by_capability(
    nimcp_multi_swarm_coordinator_t* coordinator,
    nimcp_swarm_capability_type_t capability,
    float min_proficiency,
    nimcp_vector_t* results
) {
    if (!coordinator || !results) {
        return 0;
    }

    uint32_t found = 0;

    nimcp_rwlock_read_lock(&coordinator->coordinator_lock);

    /* Search through all super-swarms */
    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_rwlock_read_lock(&super->swarm_lock);

        for (uint32_t j = 0; j < super->swarm_count; j++) {
            nimcp_swarm_identity_t* swarm = super->swarms[j];

            /* Check capabilities */
            for (uint32_t k = 0; k < swarm->capability_count; k++) {
                if (swarm->capabilities[k].type == capability &&
                    swarm->capabilities[k].proficiency >= min_proficiency) {
                    nimcp_vector_push_back(results, &swarm->swarm_id);
                    found++;
                    break;
                }
            }
        }

        nimcp_rwlock_read_unlock(&super->swarm_lock);
    }

    nimcp_rwlock_read_unlock(&coordinator->coordinator_lock);

    return found;
}

uint32_t nimcp_swarm_find_in_territory(
    nimcp_multi_swarm_coordinator_t* coordinator,
    nimcp_territory_bounds_t territory,
    nimcp_vector_t* results
) {
    if (!coordinator || !results) {
        return 0;
    }

    uint32_t found = 0;

    nimcp_rwlock_read_lock(&coordinator->coordinator_lock);

    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_rwlock_read_lock(&super->swarm_lock);

        for (uint32_t j = 0; j < super->swarm_count; j++) {
            nimcp_swarm_identity_t* swarm = super->swarms[j];

            if (nimcp_territory_overlaps(&swarm->territory, &territory)) {
                nimcp_vector_push_back(results, &swarm->swarm_id);
                found++;
            }
        }

        nimcp_rwlock_read_unlock(&super->swarm_lock);
    }

    nimcp_rwlock_read_unlock(&coordinator->coordinator_lock);

    return found;
}

void nimcp_multi_swarm_get_stats(
    nimcp_multi_swarm_coordinator_t* coordinator,
    uint32_t* total_swarms,
    uint32_t* total_agents,
    uint32_t* active_missions,
    uint32_t* active_conflicts
) {
    if (!coordinator) {
        if (total_swarms) *total_swarms = 0;
        if (total_agents) *total_agents = 0;
        if (active_missions) *active_missions = 0;
        if (active_conflicts) *active_conflicts = 0;
        return;
    }

    uint32_t swarms = 0;
    uint32_t agents = 0;
    uint32_t missions = 0;
    uint32_t conflicts = 0;

    nimcp_rwlock_read_lock(&coordinator->coordinator_lock);

    for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
        nimcp_super_swarm_t* super = coordinator->super_swarms[i];
        if (!super) continue;

        nimcp_rwlock_read_lock(&super->swarm_lock);
        swarms += super->swarm_count;

        for (uint32_t j = 0; j < super->swarm_count; j++) {
            agents += super->swarms[j]->active_agents;
        }
        nimcp_rwlock_read_unlock(&super->swarm_lock);

        nimcp_rwlock_read_lock(&super->mission_lock);
        missions += super->active_mission_count;
        nimcp_rwlock_read_unlock(&super->mission_lock);

        if (super->conflicts) {
            conflicts += (uint32_t)nimcp_vector_size(super->conflicts);
        }
    }

    nimcp_rwlock_read_unlock(&coordinator->coordinator_lock);

    if (total_swarms) *total_swarms = swarms;
    if (total_agents) *total_agents = agents;
    if (active_missions) *active_missions = missions;
    if (active_conflicts) *active_conflicts = conflicts;
}

void nimcp_multi_swarm_print_status(
    nimcp_multi_swarm_coordinator_t* coordinator,
    bool verbose
) {
    if (!coordinator) {
        printf("Multi-Swarm Coordinator: NULL\n");
        return;
    }

    uint32_t total_swarms, total_agents, active_missions, active_conflicts;
    nimcp_multi_swarm_get_stats(coordinator, &total_swarms, &total_agents,
                                &active_missions, &active_conflicts);

    printf("\n=== Multi-Swarm Coordinator Status ===\n");
    printf("Super-Swarms:     %u\n", coordinator->super_swarm_count);
    printf("Total Swarms:     %u\n", total_swarms);
    printf("Total Agents:     %u\n", total_agents);
    printf("Active Missions:  %u\n", active_missions);
    printf("Active Conflicts: %u\n", active_conflicts);
    printf("Auto-Negotiation: %s\n", coordinator->enable_auto_negotiation ? "ON" : "OFF");
    printf("Resource Sharing: %s\n", coordinator->enable_resource_sharing ? "ON" : "OFF");
    printf("Bridge Formation: %s\n", coordinator->enable_bridge_formation ? "ON" : "OFF");

    if (verbose) {
        printf("\n--- Super-Swarm Details ---\n");
        for (uint32_t i = 0; i < coordinator->super_swarm_count; i++) {
            nimcp_super_swarm_t* super = coordinator->super_swarms[i];
            if (!super) continue;

            printf("\nSuper-Swarm %u: %s (ID=%lu)\n", i, super->name,
                   super->super_swarm_id);
            printf("  Swarms: %u\n", super->swarm_count);
            printf("  Missions: %u\n", super->active_mission_count);
            printf("  Bridges: %u\n", super->bridge_count);

            for (uint32_t j = 0; j < super->swarm_count; j++) {
                nimcp_swarm_identity_t* swarm = super->swarms[j];
                printf("    Swarm: %s (ID=%lu) - %u/%u agents (%.0f%% health)\n",
                       swarm->name, swarm->swarm_id,
                       swarm->active_agents, swarm->agent_count,
                       swarm->health_percentage * 100.0f);
            }
        }
    }

    printf("=====================================\n\n");
}
