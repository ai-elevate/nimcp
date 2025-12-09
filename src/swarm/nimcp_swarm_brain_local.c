/**
 * @file nimcp_swarm_brain_local.c
 * @brief Implementation of swarm brain local instantiation
 *
 * This module manages per-agent brain instances with distributed learning
 * and weight synchronization across the swarm.
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include "swarm/nimcp_swarm_brain_local.h"
#include "nimcp.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_core.h"
#include "core/brain/nimcp_brain_io.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Constants and Macros
 * ============================================================================ */

#define SWARM_BRAIN_MODULE "SwarmBrainLocal"
#define HASH_TABLE_SIZE 256
#define EPSILON 1e-6f

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Hash table entry for agent brains
 */
typedef struct agent_brain_entry_t {
    agent_brain_t brain_data;
    struct agent_brain_entry_t* next;
} agent_brain_entry_t;

/**
 * @brief Swarm brain manager internal state
 */
struct swarm_brain_manager {
    swarm_brain_config_t config;

    /* Agent brain storage (hash table) */
    agent_brain_entry_t** agent_table;
    uint32_t active_agents;

    /* Consensus weights */
    float* consensus_weights;
    uint32_t consensus_weight_count;
    bool consensus_dirty;

    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* Statistics */
    swarm_brain_stats_t stats;
    uint64_t creation_time_ms;

    /* Synchronization */
    nimcp_platform_mutex_t mutex;
};

/**
 * @brief Swarm brain bio-async message
 */
typedef struct {
    bio_message_header_t header;
    uint32_t agent_id;
    float divergence;
    uint32_t num_weights;
    float weights[];  /* Variable length */
} bio_msg_swarm_brain_t;

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Hash function for agent ID
 */
static uint32_t agent_hash(uint32_t agent_id) {
    return agent_id % HASH_TABLE_SIZE;
}

/**
 * @brief Find agent brain entry
 *
 * WHAT: Looks up agent brain in hash table
 * WHY:  Fast O(1) average case lookup
 * HOW:  Hash agent ID, traverse chain
 */
static agent_brain_entry_t* find_agent_entry(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
) {
    if (!mgr) {
        return NULL;
    }

    uint32_t hash = agent_hash(agent_id);
    agent_brain_entry_t* entry = mgr->agent_table[hash];

    while (entry) {
        if (entry->brain_data.agent_id == agent_id) {
            return entry;
        }
        entry = entry->next;
    }

    return NULL;
}

/**
 * @brief Calculate divergence score
 *
 * WHAT: Computes L2 distance between local and consensus weights
 * WHY:  Measure how different agent is from swarm
 * HOW:  Normalized Euclidean distance
 */
static float calculate_divergence(
    const float* local_weights,
    const float* consensus_weights,
    uint32_t num_weights
) {
    if (!local_weights || !consensus_weights || num_weights == 0) {
        return 0.0f;
    }

    float sum_sq_diff = 0.0f;
    float sum_sq_consensus = EPSILON;

    for (uint32_t i = 0; i < num_weights; i++) {
        float diff = local_weights[i] - consensus_weights[i];
        sum_sq_diff += diff * diff;
        sum_sq_consensus += consensus_weights[i] * consensus_weights[i];
    }

    /* Normalized divergence: ||w_local - w_consensus|| / ||w_consensus|| */
    return sqrtf(sum_sq_diff / sum_sq_consensus);
}

/**
 * @brief Update consensus weights
 *
 * WHAT: Recomputes consensus by averaging all active agent weights
 * WHY:  Maintain swarm coherence target
 * HOW:  Average weights across all agents
 */
static nimcp_result_t update_consensus_weights(swarm_brain_manager_t* mgr) {
    if (!mgr) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Guard clause: no active agents */
    if (mgr->active_agents == 0) {
        mgr->consensus_dirty = false;
        return NIMCP_SUCCESS;
    }

    /* Find maximum weight count across all agents */
    uint32_t max_weights = 0;
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        agent_brain_entry_t* entry = mgr->agent_table[i];
        while (entry) {
            if (entry->brain_data.active &&
                entry->brain_data.num_weights > max_weights) {
                max_weights = entry->brain_data.num_weights;
            }
            entry = entry->next;
        }
    }

    /* Guard clause: no weights to average */
    if (max_weights == 0) {
        mgr->consensus_dirty = false;
        return NIMCP_SUCCESS;
    }

    /* Allocate or resize consensus weights */
    if (mgr->consensus_weight_count != max_weights) {
        float* new_weights = (float*)nimcp_realloc(
            mgr->consensus_weights,
            max_weights * sizeof(float)
        );
        if (!new_weights) {
            LOG_ERROR("Failed to allocate consensus weights");
            return NIMCP_ERROR_MEMORY;
        }
        mgr->consensus_weights = new_weights;
        mgr->consensus_weight_count = max_weights;
    }

    /* Zero out consensus weights */
    memset(mgr->consensus_weights, 0, max_weights * sizeof(float));

    /* Sum all agent weights */
    uint32_t count = 0;
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        agent_brain_entry_t* entry = mgr->agent_table[i];
        while (entry) {
            if (entry->brain_data.active && entry->brain_data.local_weights) {
                for (uint32_t j = 0; j < entry->brain_data.num_weights; j++) {
                    mgr->consensus_weights[j] += entry->brain_data.local_weights[j];
                }
                count++;
            }
            entry = entry->next;
        }
    }

    /* Average weights */
    if (count > 0) {
        for (uint32_t i = 0; i < max_weights; i++) {
            mgr->consensus_weights[i] /= (float)count;
        }
    }

    mgr->consensus_dirty = false;
    return NIMCP_SUCCESS;
}

/**
 * @brief Bio-async message handler
 */
static nimcp_result_t bio_message_handler(
    void* context,
    const bio_message_header_t* msg,
    size_t msg_size
) {
    swarm_brain_manager_t* mgr = (swarm_brain_manager_t*)context;
    if (!mgr || !msg) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    switch (msg->type) {
        case BIO_MSG_BRAIN_WEIGHTS_REQUEST: {
            /* Request for consensus weights - trigger update */
            if (mgr->consensus_dirty) {
                update_consensus_weights(mgr);
            }
            break;
        }

        case BIO_MSG_BRAIN_WEIGHTS_UPDATE: {
            /* Received weight update from another agent */
            mgr->consensus_dirty = true;
            break;
        }

        default:
            break;
    }

    nimcp_platform_mutex_unlock(&mgr->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

swarm_brain_manager_t* swarm_brain_manager_create(
    const swarm_brain_config_t* config
) {
    /* Allocate manager */
    swarm_brain_manager_t* mgr = (swarm_brain_manager_t*)nimcp_calloc(
        1, sizeof(swarm_brain_manager_t)
    );
    if (!mgr) {
        LOG_ERROR("Failed to allocate swarm brain manager");
        return NULL;
    }

    /* Initialize config with defaults if not provided */
    if (config) {
        mgr->config = *config;
    } else {
        mgr->config = swarm_brain_local_default_config();
    }

    /* Allocate hash table */
    mgr->agent_table = (agent_brain_entry_t**)nimcp_calloc(
        HASH_TABLE_SIZE, sizeof(agent_brain_entry_t*)
    );
    if (!mgr->agent_table) {
        LOG_ERROR("Failed to allocate agent hash table");
        nimcp_free(mgr);
        return NULL;
    }

    /* Initialize mutex */
    if (nimcp_platform_mutex_init(&mgr->mutex, false) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to initialize mutex");
        nimcp_free(mgr->agent_table);
        nimcp_free(mgr);
        return NULL;
    }

    /* Initialize statistics */
    mgr->creation_time_ms = nimcp_time_get_us() / 1000;
    mgr->consensus_dirty = true;

    /* Register with bio-async if enabled */
    if (mgr->config.enable_bio_async) {
        bio_module_info_t info = {
            .module_name = SWARM_BRAIN_MODULE,
            .inbox_capacity = 64,
            .user_data = mgr
        };

        mgr->bio_ctx = bio_router_register_module(&info);
        if (mgr->bio_ctx) {
            mgr->bio_async_enabled = true;
            LOG_INFO("Swarm brain manager registered with bio-async");
        } else {
            LOG_WARN("Failed to register with bio-async");
        }
    }

    LOG_INFO("Created swarm brain manager (max_agents=%u, default_size=%u)",
             SWARM_BRAIN_MAX_AGENTS, mgr->config.default_brain_size);

    return mgr;
}

void swarm_brain_manager_destroy(swarm_brain_manager_t* mgr) {
    if (!mgr) {
        return;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    /* Destroy all agent brains */
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        agent_brain_entry_t* entry = mgr->agent_table[i];
        while (entry) {
            agent_brain_entry_t* next = entry->next;

            /* Free brain resources */
            if (entry->brain_data.brain) {
                nimcp_brain_destroy(entry->brain_data.brain);
            }
            if (entry->brain_data.local_weights) {
                nimcp_free(entry->brain_data.local_weights);
            }

            nimcp_free(entry);
            entry = next;
        }
    }

    /* Free consensus weights */
    if (mgr->consensus_weights) {
        nimcp_free(mgr->consensus_weights);
    }

    /* Unregister from bio-async */
    if (mgr->bio_async_enabled && mgr->bio_ctx) {
        bio_router_unregister_module(mgr->bio_ctx);
    }

    nimcp_platform_mutex_unlock(&mgr->mutex);
    nimcp_platform_mutex_destroy(&mgr->mutex);

    nimcp_free(mgr->agent_table);
    nimcp_free(mgr);

    LOG_INFO("Destroyed swarm brain manager");
}

/* ============================================================================
 * Agent Brain Management
 * ============================================================================ */

int swarm_brain_create_for_agent(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id,
    uint32_t brain_size
) {
    if (!mgr) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Use default if brain_size is 0 */
    if (brain_size == 0) {
        brain_size = mgr->config.default_brain_size;
    }

    /* Validate brain size */
    if (brain_size > mgr->config.max_local_neurons) {
        LOG_ERROR("Brain size %u exceeds max %u", brain_size,
                  mgr->config.max_local_neurons);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    /* Check if agent already exists */
    if (find_agent_entry(mgr, agent_id)) {
        LOG_WARN("Agent %u already has brain", agent_id);
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Check agent limit */
    if (mgr->active_agents >= SWARM_BRAIN_MAX_AGENTS) {
        LOG_ERROR("Maximum agents reached (%u)", SWARM_BRAIN_MAX_AGENTS);
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_MEMORY;
    }

    /* Create brain instance */
    char task_name[64];
    snprintf(task_name, sizeof(task_name), "swarm_agent_%u", agent_id);
    nimcp_brain_t brain = nimcp_brain_create(task_name, NIMCP_BRAIN_TINY, NIMCP_TASK_CLASSIFICATION,
                                              brain_size, brain_size / 2);
    if (!brain) {
        LOG_ERROR("Failed to create brain for agent %u", agent_id);
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_MEMORY;
    }

    /* Create hash table entry */
    agent_brain_entry_t* entry = (agent_brain_entry_t*)nimcp_calloc(
        1, sizeof(agent_brain_entry_t)
    );
    if (!entry) {
        LOG_ERROR("Failed to allocate agent entry");
        nimcp_brain_destroy(brain);
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_MEMORY;
    }

    /* Initialize agent brain data */
    entry->brain_data.agent_id = agent_id;
    entry->brain_data.brain = brain;
    entry->brain_data.brain_size = brain_size;
    entry->brain_data.num_weights = 0;  /* Will be set on first sync */
    entry->brain_data.local_weights = NULL;
    entry->brain_data.last_sync_ms = nimcp_time_get_us() / 1000;
    entry->brain_data.divergence_score = 0.0f;
    entry->brain_data.active = true;

    /* Insert into hash table */
    uint32_t hash = agent_hash(agent_id);
    entry->next = mgr->agent_table[hash];
    mgr->agent_table[hash] = entry;

    mgr->active_agents++;
    mgr->consensus_dirty = true;

    /* Update statistics */
    mgr->stats.num_agents = mgr->active_agents;
    mgr->stats.total_neurons += brain_size;

    nimcp_platform_mutex_unlock(&mgr->mutex);

    /* Send bio-async notification */
    if (mgr->bio_async_enabled && mgr->bio_ctx) {
        bio_message_header_t msg = {
            .type = BIO_MSG_BRAIN_AGENT_JOINED,
            .flags = BIO_MSG_FLAG_BROADCAST,
            .timestamp_us = nimcp_time_get_us()
        };
        bio_router_broadcast(mgr->bio_ctx, &msg, sizeof(msg));
    }

    LOG_INFO("Created brain for agent %u (size=%u)", agent_id, brain_size);
    return NIMCP_SUCCESS;
}

int swarm_brain_destroy_for_agent(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
) {
    if (!mgr) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    /* Find and remove entry */
    uint32_t hash = agent_hash(agent_id);
    agent_brain_entry_t** prev_ptr = &mgr->agent_table[hash];
    agent_brain_entry_t* entry = mgr->agent_table[hash];

    while (entry) {
        if (entry->brain_data.agent_id == agent_id) {
            /* Remove from chain */
            *prev_ptr = entry->next;

            /* Free resources */
            if (entry->brain_data.brain) {
                nimcp_brain_destroy(entry->brain_data.brain);
            }
            if (entry->brain_data.local_weights) {
                nimcp_free(entry->brain_data.local_weights);
            }

            /* Update statistics */
            mgr->active_agents--;
            mgr->stats.num_agents = mgr->active_agents;
            mgr->stats.total_neurons -= entry->brain_data.brain_size;
            mgr->consensus_dirty = true;

            nimcp_free(entry);

            nimcp_platform_mutex_unlock(&mgr->mutex);

            /* Send bio-async notification */
            if (mgr->bio_async_enabled && mgr->bio_ctx) {
                bio_message_header_t msg = {
                    .type = BIO_MSG_BRAIN_AGENT_LEFT,
                    .flags = BIO_MSG_FLAG_BROADCAST,
                    .timestamp_us = nimcp_time_get_us()
                };
                bio_router_broadcast(mgr->bio_ctx, &msg, sizeof(msg));
            }

            LOG_INFO("Destroyed brain for agent %u", agent_id);
            return NIMCP_SUCCESS;
        }

        prev_ptr = &entry->next;
        entry = entry->next;
    }

    nimcp_platform_mutex_unlock(&mgr->mutex);
    LOG_WARN("Agent %u not found", agent_id);
    return NIMCP_ERROR_INVALID_PARAM;
}

nimcp_brain_t swarm_brain_get(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
) {
    if (!mgr) {
        return NULL;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);
    agent_brain_entry_t* entry = find_agent_entry(mgr, agent_id);
    nimcp_brain_t brain = entry ? entry->brain_data.brain : NULL;
    nimcp_platform_mutex_unlock(&mgr->mutex);

    return brain;
}

/* ============================================================================
 * Local Learning and Processing
 * ============================================================================ */

int swarm_brain_local_learn(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id,
    const float* input,
    uint32_t input_size,
    const float* target,
    uint32_t target_size
) {
    if (!mgr || !input || !target) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    agent_brain_entry_t* entry = find_agent_entry(mgr, agent_id);
    if (!entry || !entry->brain_data.active) {
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Perform learning on brain using the public API */
    nimcp_training_result_t train_result;
    nimcp_status_t result = nimcp_brain_train_step(
        entry->brain_data.brain,
        input,
        input_size,
        target,
        target_size,
        &train_result
    );

    if (result == NIMCP_OK) {
        /* Mark consensus as dirty */
        mgr->consensus_dirty = true;
    }

    nimcp_platform_mutex_unlock(&mgr->mutex);
    return (result == NIMCP_OK) ? NIMCP_SUCCESS : NIMCP_ERROR;
}

int swarm_brain_local_process(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id,
    const float* input,
    uint32_t input_size,
    float* output,
    uint32_t* output_size
) {
    if (!mgr || !input || !output || !output_size) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    agent_brain_entry_t* entry = find_agent_entry(mgr, agent_id);
    if (!entry || !entry->brain_data.active) {
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Process through brain using the public API */
    nimcp_status_t result = nimcp_brain_infer(
        entry->brain_data.brain,
        input,
        input_size,
        output,
        *output_size
    );

    nimcp_platform_mutex_unlock(&mgr->mutex);
    return (result == NIMCP_OK) ? NIMCP_SUCCESS : NIMCP_ERROR;
}

/* ============================================================================
 * Weight Synchronization
 * ============================================================================ */

int swarm_brain_local_sync_weights(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
) {
    if (!mgr) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    agent_brain_entry_t* entry = find_agent_entry(mgr, agent_id);
    if (!entry || !entry->brain_data.active) {
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Update consensus if dirty */
    if (mgr->consensus_dirty) {
        update_consensus_weights(mgr);
    }

    /* Guard clause: no consensus weights yet */
    if (!mgr->consensus_weights || mgr->consensus_weight_count == 0) {
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_SUCCESS;
    }

    /* Get current agent weights (placeholder - would extract from brain) */
    /* In a real implementation, would need brain API to get/set weights */
    if (!entry->brain_data.local_weights) {
        entry->brain_data.local_weights = (float*)nimcp_calloc(
            mgr->consensus_weight_count, sizeof(float)
        );
        if (!entry->brain_data.local_weights) {
            nimcp_platform_mutex_unlock(&mgr->mutex);
            return NIMCP_ERROR_MEMORY;
        }
        entry->brain_data.num_weights = mgr->consensus_weight_count;
    }

    /* Calculate divergence before sync */
    float old_divergence = calculate_divergence(
        entry->brain_data.local_weights,
        mgr->consensus_weights,
        entry->brain_data.num_weights
    );

    /* Blend local weights with consensus (simple averaging) */
    for (uint32_t i = 0; i < entry->brain_data.num_weights; i++) {
        entry->brain_data.local_weights[i] =
            0.7f * entry->brain_data.local_weights[i] +
            0.3f * mgr->consensus_weights[i];
    }

    /* Calculate new divergence */
    entry->brain_data.divergence_score = calculate_divergence(
        entry->brain_data.local_weights,
        mgr->consensus_weights,
        entry->brain_data.num_weights
    );

    entry->brain_data.last_sync_ms = nimcp_time_get_us() / 1000;
    mgr->stats.sync_count++;

    nimcp_platform_mutex_unlock(&mgr->mutex);

    /* Send bio-async notification if divergence changed significantly */
    if (mgr->bio_async_enabled && mgr->bio_ctx &&
        fabsf(old_divergence - entry->brain_data.divergence_score) > 0.1f) {
        bio_message_header_t msg = {
            .type = BIO_MSG_BRAIN_SYNCED,
            .flags = BIO_MSG_FLAG_BROADCAST,
            .timestamp_us = nimcp_time_get_us()
        };
        bio_router_broadcast(mgr->bio_ctx, &msg, sizeof(msg));
    }

    return NIMCP_SUCCESS;
}

int swarm_brain_sync_all(swarm_brain_manager_t* mgr) {
    if (!mgr) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    /* Update consensus first */
    update_consensus_weights(mgr);

    /* Sync all active agents */
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        agent_brain_entry_t* entry = mgr->agent_table[i];
        while (entry) {
            if (entry->brain_data.active) {
                nimcp_platform_mutex_unlock(&mgr->mutex);
                swarm_brain_local_sync_weights(mgr, entry->brain_data.agent_id);
                nimcp_platform_mutex_lock(&mgr->mutex);
            }
            entry = entry->next;
        }
    }

    nimcp_platform_mutex_unlock(&mgr->mutex);
    return NIMCP_SUCCESS;
}

int swarm_brain_get_consensus_weights(
    swarm_brain_manager_t* mgr,
    float** weights,
    uint32_t* num_weights
) {
    if (!mgr || !weights || !num_weights) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    /* Update consensus if dirty */
    if (mgr->consensus_dirty) {
        update_consensus_weights(mgr);
    }

    /* Guard clause: no consensus */
    if (!mgr->consensus_weights || mgr->consensus_weight_count == 0) {
        *weights = NULL;
        *num_weights = 0;
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_SUCCESS;
    }

    /* Allocate and copy weights */
    *num_weights = mgr->consensus_weight_count;
    *weights = (float*)nimcp_malloc(mgr->consensus_weight_count * sizeof(float));
    if (!*weights) {
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_MEMORY;
    }

    memcpy(*weights, mgr->consensus_weights,
           mgr->consensus_weight_count * sizeof(float));

    nimcp_platform_mutex_unlock(&mgr->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Divergence Detection
 * ============================================================================ */

float swarm_brain_get_divergence(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
) {
    if (!mgr) {
        return -1.0f;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    agent_brain_entry_t* entry = find_agent_entry(mgr, agent_id);
    float divergence = entry ? entry->brain_data.divergence_score : -1.0f;

    nimcp_platform_mutex_unlock(&mgr->mutex);
    return divergence;
}

int swarm_brain_get_divergent_agents(
    swarm_brain_manager_t* mgr,
    uint32_t** agents,
    uint32_t* count
) {
    if (!mgr || !agents || !count) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    /* Count divergent agents */
    uint32_t divergent_count = 0;
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        agent_brain_entry_t* entry = mgr->agent_table[i];
        while (entry) {
            if (entry->brain_data.active &&
                entry->brain_data.divergence_score >
                    mgr->config.divergence_threshold) {
                divergent_count++;
            }
            entry = entry->next;
        }
    }

    /* Guard clause: no divergent agents */
    if (divergent_count == 0) {
        *agents = NULL;
        *count = 0;
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_SUCCESS;
    }

    /* Allocate array */
    *agents = (uint32_t*)nimcp_malloc(divergent_count * sizeof(uint32_t));
    if (!*agents) {
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_MEMORY;
    }

    /* Collect divergent agent IDs */
    uint32_t idx = 0;
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        agent_brain_entry_t* entry = mgr->agent_table[i];
        while (entry) {
            if (entry->brain_data.active &&
                entry->brain_data.divergence_score >
                    mgr->config.divergence_threshold) {
                (*agents)[idx++] = entry->brain_data.agent_id;
            }
            entry = entry->next;
        }
    }

    *count = divergent_count;
    nimcp_platform_mutex_unlock(&mgr->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Statistics and Monitoring
 * ============================================================================ */

int swarm_brain_local_get_stats(
    swarm_brain_manager_t* mgr,
    swarm_brain_stats_t* stats
) {
    if (!mgr || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    /* Calculate statistics */
    float sum_divergence = 0.0f;
    float max_divergence = 0.0f;
    uint32_t divergent_count = 0;

    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        agent_brain_entry_t* entry = mgr->agent_table[i];
        while (entry) {
            if (entry->brain_data.active) {
                sum_divergence += entry->brain_data.divergence_score;
                if (entry->brain_data.divergence_score > max_divergence) {
                    max_divergence = entry->brain_data.divergence_score;
                }
                if (entry->brain_data.divergence_score >
                    mgr->config.divergence_threshold) {
                    divergent_count++;
                }
            }
            entry = entry->next;
        }
    }

    /* Fill statistics */
    mgr->stats.avg_divergence =
        mgr->active_agents > 0 ? sum_divergence / mgr->active_agents : 0.0f;
    mgr->stats.max_divergence = max_divergence;
    mgr->stats.divergent_agents = divergent_count;
    mgr->stats.uptime_ms = (nimcp_time_get_us() / 1000) - mgr->creation_time_ms;

    *stats = mgr->stats;

    nimcp_platform_mutex_unlock(&mgr->mutex);
    return NIMCP_SUCCESS;
}

int swarm_brain_local_reset_stats(swarm_brain_manager_t* mgr) {
    if (!mgr) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    mgr->stats.sync_count = 0;
    mgr->creation_time_ms = nimcp_time_get_us() / 1000;

    nimcp_platform_mutex_unlock(&mgr->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

swarm_brain_config_t swarm_brain_local_default_config(void) {
    swarm_brain_config_t config = {
        .default_brain_size = SWARM_BRAIN_DEFAULT_SIZE,
        .max_local_neurons = SWARM_BRAIN_MAX_NEURONS,
        .sync_interval_ms = SWARM_BRAIN_DEFAULT_SYNC_INTERVAL,
        .divergence_threshold = SWARM_BRAIN_DEFAULT_DIVERGENCE_THRESHOLD,
        .enable_weight_sharing = true,
        .enable_bio_async = true
    };
    return config;
}

bool swarm_brain_has_agent(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
) {
    if (!mgr) {
        return false;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);
    bool has_agent = (find_agent_entry(mgr, agent_id) != NULL);
    nimcp_platform_mutex_unlock(&mgr->mutex);

    return has_agent;
}

uint32_t swarm_brain_get_agent_count(swarm_brain_manager_t* mgr) {
    if (!mgr) {
        return 0;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);
    uint32_t count = mgr->active_agents;
    nimcp_platform_mutex_unlock(&mgr->mutex);

    return count;
}

int swarm_brain_get_all_agents(
    swarm_brain_manager_t* mgr,
    uint32_t** agents,
    uint32_t* count
) {
    if (!mgr || !agents || !count) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&mgr->mutex);

    /* Guard clause: no agents */
    if (mgr->active_agents == 0) {
        *agents = NULL;
        *count = 0;
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_SUCCESS;
    }

    /* Allocate array */
    *agents = (uint32_t*)nimcp_malloc(mgr->active_agents * sizeof(uint32_t));
    if (!*agents) {
        nimcp_platform_mutex_unlock(&mgr->mutex);
        return NIMCP_ERROR_MEMORY;
    }

    /* Collect agent IDs */
    uint32_t idx = 0;
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        agent_brain_entry_t* entry = mgr->agent_table[i];
        while (entry) {
            if (entry->brain_data.active) {
                (*agents)[idx++] = entry->brain_data.agent_id;
            }
            entry = entry->next;
        }
    }

    *count = mgr->active_agents;
    nimcp_platform_mutex_unlock(&mgr->mutex);
    return NIMCP_SUCCESS;
}
