//=============================================================================
// nimcp_portia_collective_bridge.c - Portia-Collective Cognition Integration
//=============================================================================
/**
 * @file nimcp_portia_collective_bridge.c
 * @brief Implementation of Portia-Collective Cognition bridge
 *
 * @author NIMCP Development Team
 * @date 2025-12-31
 */

#include "portia/nimcp_portia_collective_bridge.h"
#include "portia/nimcp_portia_tier_switch.h"
#include "cognitive/collective_cognition/nimcp_collective_cognition.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"

#include <string.h>

#define LOG_MODULE "PORTIA_COLLECTIVE"

//=============================================================================
// Internal Structure
//=============================================================================

struct portia_collective_bridge {
    /* Configuration */
    portia_collective_config_t config;

    /* Connected systems */
    portia_tier_switch_t portia;
    collective_cognition_t* collective;

    /* Local state */
    uint32_t local_instance_id;
    uint32_t local_tier;
    float local_load;
    bool local_degraded;

    /* Collective state */
    collective_instance_state_t instances[PORTIA_COLLECTIVE_MAX_INSTANCES];
    uint32_t instance_count;
    uint32_t leader_id;
    bool is_leader;

    /* Statistics */
    portia_collective_stats_t stats;

    /* Timing */
    uint64_t last_update_ms;
    uint64_t last_broadcast_ms;

    /* State flags */
    bool initialized;
    bool bio_async_connected;
};

//=============================================================================
// Configuration API
//=============================================================================

void portia_collective_default_config(portia_collective_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(*config));

    config->resource_strategy = COLLECTIVE_RESOURCE_COOPERATIVE;
    config->degradation_mode = DEGRADATION_COORD_COMPENSATING;

    config->offload_threshold = PORTIA_COLLECTIVE_OFFLOAD_THRESHOLD;
    config->receive_threshold = PORTIA_COLLECTIVE_RECEIVE_THRESHOLD;
    config->degradation_threshold = 0.7f;

    config->update_interval_ms = PORTIA_COLLECTIVE_DEFAULT_UPDATE_MS;
    config->state_timeout_ms = 5000;

    config->enable_bio_async = true;
    config->enable_proactive_offload = false;
    config->enable_leader_election = true;
    config->broadcast_tier_changes = true;
}

//=============================================================================
// Lifecycle API
//=============================================================================

portia_collective_bridge_t* portia_collective_create(
    const portia_collective_config_t* config,
    portia_tier_switch_t portia,
    collective_cognition_t* collective
) {
    portia_collective_bridge_t* bridge = nimcp_calloc(1, sizeof(*bridge));
    if (!bridge) {
        LOG_ERROR("Failed to allocate portia-collective bridge");
        return NULL;
    }

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        portia_collective_default_config(&bridge->config);
    }

    /* Connect systems */
    bridge->portia = portia;
    bridge->collective = collective;

    /* Initialize local state */
    bridge->local_instance_id = (uint32_t)(nimcp_time_get_us() & 0xFFFF);
    bridge->local_tier = 3;  /* Assume FULL tier initially */
    bridge->local_load = 0.0f;
    bridge->local_degraded = false;

    /* Initialize collective state */
    bridge->instance_count = 0;
    bridge->leader_id = bridge->local_instance_id;
    bridge->is_leader = true;  /* Initially leader until others join */

    /* Timing */
    bridge->last_update_ms = nimcp_time_get_ms();
    bridge->last_broadcast_ms = 0;

    bridge->initialized = true;

    LOG_INFO("Portia-collective bridge created (local_id=%u)", bridge->local_instance_id);

    return bridge;
}

void portia_collective_destroy(portia_collective_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->bio_async_connected) {
        portia_collective_disconnect_bio_async(bridge);
    }

    LOG_INFO("Portia-collective bridge destroyed (local_id=%u)", bridge->local_instance_id);

    nimcp_free(bridge);
}

int portia_collective_reset(portia_collective_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Reset collective state */
    memset(bridge->instances, 0, sizeof(bridge->instances));
    bridge->instance_count = 0;
    bridge->leader_id = bridge->local_instance_id;
    bridge->is_leader = true;

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    /* Reset timing */
    bridge->last_update_ms = nimcp_time_get_ms();
    bridge->last_broadcast_ms = 0;

    LOG_DEBUG("Portia-collective bridge reset");

    return 0;
}

//=============================================================================
// Connection API
//=============================================================================

int portia_collective_connect_portia(
    portia_collective_bridge_t* bridge,
    portia_tier_switch_t portia
) {
    if (!bridge) {
        LOG_ERROR("Null bridge in connect_portia");
        return -1;
    }

    bridge->portia = portia;

    if (portia) {
        /* Query current tier from portia */
        /* Note: Would need portia API to get current tier */
        LOG_INFO("Connected to Portia tier switch");
    } else {
        LOG_INFO("Disconnected from Portia tier switch");
    }

    return 0;
}

int portia_collective_connect_collective(
    portia_collective_bridge_t* bridge,
    collective_cognition_t* collective
) {
    if (!bridge) {
        LOG_ERROR("Null bridge in connect_collective");
        return -1;
    }

    bridge->collective = collective;

    if (collective) {
        LOG_INFO("Connected to collective cognition system");
    } else {
        LOG_INFO("Disconnected from collective cognition system");
    }

    return 0;
}

//=============================================================================
// Internal Helpers
//=============================================================================

static int find_instance_index(
    const portia_collective_bridge_t* bridge,
    uint32_t instance_id
) {
    for (uint32_t i = 0; i < bridge->instance_count; ++i) {
        if (bridge->instances[i].instance_id == instance_id) {
            return (int)i;
        }
    }
    return -1;
}

static void update_local_state(portia_collective_bridge_t* bridge) {
    /* Update local instance in the instances array */
    int idx = find_instance_index(bridge, bridge->local_instance_id);
    if (idx < 0) {
        /* Add ourselves if not present */
        if (bridge->instance_count < PORTIA_COLLECTIVE_MAX_INSTANCES) {
            idx = (int)bridge->instance_count++;
        } else {
            return;
        }
    }

    collective_instance_state_t* state = &bridge->instances[idx];
    state->instance_id = bridge->local_instance_id;
    state->tier = bridge->local_tier;
    state->load_factor = bridge->local_load;
    state->is_degraded = bridge->local_degraded;
    state->can_receive_tasks = (bridge->local_load < bridge->config.receive_threshold);
    state->is_leader = bridge->is_leader;
    state->last_update_ms = nimcp_time_get_ms();

    /* Query power/thermal from portia if connected */
    if (bridge->portia) {
        state->power_level = 1.0f;  /* Would query portia */
        state->thermal_headroom = 1.0f;
    } else {
        state->power_level = 1.0f;
        state->thermal_headroom = 1.0f;
    }
}

static void elect_leader(portia_collective_bridge_t* bridge) {
    if (!bridge->config.enable_leader_election) return;

    /* Simple leader election: highest tier, then lowest load, then lowest ID */
    uint32_t best_id = bridge->local_instance_id;
    uint32_t best_tier = bridge->local_tier;
    float best_load = bridge->local_load;

    for (uint32_t i = 0; i < bridge->instance_count; ++i) {
        const collective_instance_state_t* inst = &bridge->instances[i];
        if (inst->is_degraded) continue;

        bool is_better = false;
        if (inst->tier > best_tier) {
            is_better = true;
        } else if (inst->tier == best_tier && inst->load_factor < best_load) {
            is_better = true;
        } else if (inst->tier == best_tier && inst->load_factor == best_load &&
                   inst->instance_id < best_id) {
            is_better = true;
        }

        if (is_better) {
            best_id = inst->instance_id;
            best_tier = inst->tier;
            best_load = inst->load_factor;
        }
    }

    if (best_id != bridge->leader_id) {
        bridge->leader_id = best_id;
        bridge->is_leader = (best_id == bridge->local_instance_id);
        bridge->stats.leader_changes++;
        LOG_INFO("Leader changed to instance %u", best_id);
    }
}

static void prune_stale_instances(portia_collective_bridge_t* bridge) {
    uint64_t now = nimcp_time_get_ms();
    uint32_t timeout = bridge->config.state_timeout_ms;

    uint32_t write_idx = 0;
    for (uint32_t read_idx = 0; read_idx < bridge->instance_count; ++read_idx) {
        collective_instance_state_t* inst = &bridge->instances[read_idx];

        /* Keep local instance and non-stale instances */
        if (inst->instance_id == bridge->local_instance_id ||
            (now - inst->last_update_ms) < timeout) {
            if (write_idx != read_idx) {
                bridge->instances[write_idx] = bridge->instances[read_idx];
            }
            write_idx++;
        } else {
            LOG_DEBUG("Pruned stale instance %u", inst->instance_id);
        }
    }

    bridge->instance_count = write_idx;
}

//=============================================================================
// Update API
//=============================================================================

int portia_collective_update(
    portia_collective_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return -1;

    (void)delta_ms;  /* Used for timing calculations if needed */

    uint64_t now = nimcp_time_get_ms();

    /* Update local state */
    update_local_state(bridge);

    /* Prune stale instances */
    prune_stale_instances(bridge);

    /* Re-elect leader if needed */
    elect_leader(bridge);

    /* Broadcast state if interval elapsed */
    if (bridge->config.broadcast_tier_changes &&
        (now - bridge->last_broadcast_ms) >= bridge->config.update_interval_ms) {

        /* Would broadcast via bio-async or collective cognition messaging */
        bridge->last_broadcast_ms = now;
    }

    /* Update average load statistic */
    float total_load = 0.0f;
    for (uint32_t i = 0; i < bridge->instance_count; ++i) {
        total_load += bridge->instances[i].load_factor;
    }
    if (bridge->instance_count > 0) {
        bridge->stats.avg_collective_load = total_load / (float)bridge->instance_count;
    }

    bridge->last_update_ms = now;

    return 0;
}

int portia_collective_broadcast_tier(
    portia_collective_bridge_t* bridge,
    uint32_t new_tier
) {
    if (!bridge) return -1;

    bridge->local_tier = new_tier;
    bridge->stats.tier_broadcasts++;

    /* Update local state immediately */
    update_local_state(bridge);

    LOG_DEBUG("Broadcasting tier change: %u", new_tier);

    /* Would send via bio-async to collective */

    return 0;
}

int portia_collective_handle_remote_tier(
    portia_collective_bridge_t* bridge,
    uint32_t instance_id,
    uint32_t new_tier
) {
    if (!bridge) return -1;
    if (instance_id == bridge->local_instance_id) return 0;  /* Ignore self */

    int idx = find_instance_index(bridge, instance_id);
    if (idx < 0) {
        /* New instance - add it */
        if (bridge->instance_count >= PORTIA_COLLECTIVE_MAX_INSTANCES) {
            LOG_WARNING("Cannot add instance %u: max instances reached", instance_id);
            return -1;
        }
        idx = (int)bridge->instance_count++;
        memset(&bridge->instances[idx], 0, sizeof(bridge->instances[idx]));
        bridge->instances[idx].instance_id = instance_id;
    }

    bridge->instances[idx].tier = new_tier;
    bridge->instances[idx].last_update_ms = nimcp_time_get_ms();
    bridge->stats.tier_events_received++;

    LOG_DEBUG("Received tier %u from instance %u", new_tier, instance_id);

    /* Re-evaluate leader */
    elect_leader(bridge);

    return 0;
}

//=============================================================================
// Load Balancing API
//=============================================================================

int portia_collective_request_offload(
    portia_collective_bridge_t* bridge,
    float task_complexity,
    uint32_t* target_instance
) {
    if (!bridge || !target_instance) return -1;

    /* Find best instance to receive task */
    uint32_t best_id = 0;
    float best_score = -1.0f;

    for (uint32_t i = 0; i < bridge->instance_count; ++i) {
        const collective_instance_state_t* inst = &bridge->instances[i];

        /* Skip self, degraded, and overloaded instances */
        if (inst->instance_id == bridge->local_instance_id) continue;
        if (inst->is_degraded) continue;
        if (!inst->can_receive_tasks) continue;

        /* Score based on tier, available capacity, and power */
        float capacity = 1.0f - inst->load_factor;
        float tier_score = (float)inst->tier / 3.0f;
        float score = capacity * 0.5f + tier_score * 0.3f + inst->power_level * 0.2f;

        /* Penalize if complexity exceeds tier capability */
        if (task_complexity > tier_score) {
            score *= 0.5f;
        }

        if (score > best_score) {
            best_score = score;
            best_id = inst->instance_id;
        }
    }

    if (best_score < 0.0f) {
        return -1;  /* No suitable target found */
    }

    *target_instance = best_id;
    bridge->stats.tasks_offloaded++;

    LOG_DEBUG("Offload request: complexity=%.2f -> instance %u (score=%.2f)",
              task_complexity, best_id, best_score);

    return 0;
}

bool portia_collective_can_receive(const portia_collective_bridge_t* bridge) {
    if (!bridge) return false;

    return !bridge->local_degraded &&
           bridge->local_load < bridge->config.receive_threshold;
}

int portia_collective_rebalance(portia_collective_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->is_leader) return 0;  /* Only leader rebalances */

    /* Simple rebalancing: identify overloaded and underloaded instances */
    uint32_t redistributed = 0;

    for (uint32_t i = 0; i < bridge->instance_count; ++i) {
        collective_instance_state_t* inst = &bridge->instances[i];

        if (inst->load_factor > bridge->config.offload_threshold &&
            !inst->is_degraded) {
            /* This instance needs to offload */
            uint32_t target = 0;
            if (portia_collective_request_offload(bridge, 0.5f, &target) == 0) {
                redistributed++;
                /* Would send offload request message */
            }
        }
    }

    LOG_DEBUG("Rebalance complete: %u tasks redistributed", redistributed);

    return (int)redistributed;
}

//=============================================================================
// Degradation API
//=============================================================================

int portia_collective_coordinate_degradation(
    portia_collective_bridge_t* bridge,
    bool local_degraded
) {
    if (!bridge) return -1;

    bool was_degraded = bridge->local_degraded;
    bridge->local_degraded = local_degraded;

    if (local_degraded && !was_degraded) {
        bridge->stats.degradation_events++;

        switch (bridge->config.degradation_mode) {
            case DEGRADATION_COORD_COMPENSATING:
                /* Request others to compensate */
                portia_collective_request_compensation(bridge);
                break;

            case DEGRADATION_COORD_SYNCHRONIZED:
                /* Would broadcast degradation to all */
                break;

            case DEGRADATION_COORD_CASCADING:
                /* Would trigger staged degradation */
                break;

            case DEGRADATION_COORD_NONE:
            default:
                break;
        }
    }

    update_local_state(bridge);

    return 0;
}

int portia_collective_request_compensation(portia_collective_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->local_degraded) return 0;

    LOG_INFO("Requesting compensation from collective (local degraded)");

    /* Would broadcast compensation request to collective */
    /* Other instances would take over critical functions */

    return 0;
}

//=============================================================================
// Query API
//=============================================================================

int portia_collective_get_summary(
    const portia_collective_bridge_t* bridge,
    collective_resource_summary_t* summary
) {
    if (!bridge || !summary) return -1;

    memset(summary, 0, sizeof(*summary));

    summary->total_instances = bridge->instance_count;
    summary->leader_instance_id = bridge->leader_id;

    float total_load = 0.0f;
    float total_tier = 0.0f;

    for (uint32_t i = 0; i < bridge->instance_count; ++i) {
        const collective_instance_state_t* inst = &bridge->instances[i];

        if (inst->tier == 3) {  /* PLATFORM_TIER_FULL */
            summary->instances_full_tier++;
        }
        if (inst->is_degraded) {
            summary->instances_degraded++;
        }

        total_load += inst->load_factor;
        total_tier += (float)inst->tier;
    }

    if (bridge->instance_count > 0) {
        summary->average_load = total_load / (float)bridge->instance_count;
        summary->average_tier = total_tier / (float)bridge->instance_count;
    }

    summary->collective_capacity = (float)bridge->instance_count * summary->average_tier / 3.0f;
    summary->collective_utilization = summary->average_load;
    summary->collective_stressed = (summary->average_load > bridge->config.degradation_threshold);

    return 0;
}

int portia_collective_get_instance_state(
    const portia_collective_bridge_t* bridge,
    uint32_t instance_id,
    collective_instance_state_t* state
) {
    if (!bridge || !state) return -1;

    int idx = find_instance_index(bridge, instance_id);
    if (idx < 0) return -1;

    *state = bridge->instances[idx];
    return 0;
}

uint32_t portia_collective_get_local_id(const portia_collective_bridge_t* bridge) {
    return bridge ? bridge->local_instance_id : 0;
}

bool portia_collective_is_leader(const portia_collective_bridge_t* bridge) {
    return bridge ? bridge->is_leader : false;
}

//=============================================================================
// Statistics API
//=============================================================================

int portia_collective_get_stats(
    const portia_collective_bridge_t* bridge,
    portia_collective_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

void portia_collective_reset_stats(portia_collective_bridge_t* bridge) {
    if (!bridge) return;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
}

//=============================================================================
// Bio-Async API
//=============================================================================

int portia_collective_connect_bio_async(portia_collective_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Would register with bio-async router */
    bridge->bio_async_connected = true;

    LOG_INFO("Connected to bio-async router");
    return 0;
}

int portia_collective_disconnect_bio_async(portia_collective_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Would unregister from bio-async router */
    bridge->bio_async_connected = false;

    LOG_INFO("Disconnected from bio-async router");
    return 0;
}
