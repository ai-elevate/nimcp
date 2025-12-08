/**
 * @file nimcp_swarm_energy_gossip.c
 * @brief Energy-Aware Gossip Protocol Implementation
 *
 * Biological inspiration: Metabolic efficiency in biological systems
 * - Honeybee energy conservation during foraging
 * - Bat torpor and hibernation strategies
 * - Neural energy management in the brain
 * - Cellular ATP management and regulation
 */

#include "swarm/nimcp_swarm_energy_gossip.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Calculate distance between two 3D points
 */
static float calculate_distance(const float pos1[3], const float pos2[3]) {
    float dx = pos1[0] - pos2[0];
    float dy = pos1[1] - pos2[1];
    float dz = pos1[2] - pos2[2];
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

/**
 * @brief Generate random float between 0 and 1
 */
static float random_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

/**
 * @brief Update energy state based on current level
 */
static void update_energy_state(nimcp_energy_gossip* gossip) {
    if (!gossip) return;

    NimcpEnergyState old_state = gossip->current_state;

    if (gossip->stats.harvest_rate > 0.0f &&
        gossip->stats.harvest_rate > gossip->stats.consumption_rate) {
        gossip->current_state = NIMCP_ENERGY_CHARGING;
    } else {
        gossip->current_state = nimcp_energy_level_to_state(
            gossip->stats.current_level,
            false
        );
    }

    if (old_state != gossip->current_state) {
        gossip->energy_state_changes++;
        NIMCP_LOG_INFO("Node %u energy state changed: %s -> %s (%.1f%%)",
            gossip->node_id,
            nimcp_energy_state_to_string(old_state),
            nimcp_energy_state_to_string(gossip->current_state),
            gossip->stats.current_level);
    }
}

/**
 * @brief Check if message is already in cache
 */
static bool is_message_cached(const nimcp_energy_gossip* gossip, uint64_t message_id) {
    if (!gossip || !gossip->message_cache) return false;

    for (uint32_t i = 0; i < gossip->message_cache_size; i++) {
        if (gossip->message_cache[i] &&
            gossip->message_cache[i]->header.message_id == message_id) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Add message to cache
 */
static nimcp_result_t add_to_cache(nimcp_energy_gossip* gossip, const NimcpGossipMessage* message) {
    if (!gossip || !message) return NIMCP_ERROR_NULL_POINTER;

    /* Check if cache is full */
    if (gossip->message_cache_size >= gossip->message_cache_capacity) {
        /* Remove oldest message (FIFO) */
        if (gossip->message_cache[0]) {
            if (gossip->message_cache[0]->payload) {
                nimcp_free(gossip->message_cache[0]->payload);
            }
            nimcp_free(gossip->message_cache[0]);
        }

        /* Shift messages */
        for (uint32_t i = 0; i < gossip->message_cache_size - 1; i++) {
            gossip->message_cache[i] = gossip->message_cache[i + 1];
        }
        gossip->message_cache_size--;
    }

    /* Allocate new message */
    NimcpGossipMessage* cached = (NimcpGossipMessage*)nimcp_malloc(sizeof(NimcpGossipMessage));
    if (!cached) return NIMCP_NO_MEMORY_ALLOCATION;

    memcpy(cached, message, sizeof(NimcpGossipMessage));

    /* Copy payload */
    if (message->payload && message->payload_size > 0) {
        cached->payload = nimcp_malloc(message->payload_size);
        if (!cached->payload) {
            nimcp_free(cached);
            return NIMCP_NO_MEMORY_ALLOCATION;
        }
        memcpy(cached->payload, message->payload, message->payload_size);
    }

    gossip->message_cache[gossip->message_cache_size++] = cached;
    return NIMCP_SUCCESS;
}

/**
 * @brief Find relay node by ID
 */
static NimcpRelayNode* find_relay_node(nimcp_energy_gossip* gossip, uint32_t node_id) {
    if (!gossip || !gossip->relay_nodes) return NULL;

    for (uint32_t i = 0; i < gossip->relay_node_count; i++) {
        if (gossip->relay_nodes[i].node_id == node_id) {
            return &gossip->relay_nodes[i];
        }
    }
    return NULL;
}

/**
 * @brief Calculate relay score (higher is better)
 */
static float calculate_relay_score(const NimcpRelayNode* relay) {
    if (!relay || !relay->is_available) return 0.0f;

    /* Scoring factors:
     * - Energy level (40% weight)
     * - Reliability (30% weight)
     * - Inverse distance (20% weight)
     * - Message count balance (10% weight)
     */
    float energy_score = relay->energy_level / 100.0f;
    float reliability_score = relay->reliability_score;
    float distance_score = (relay->distance > 0.0f) ? (1.0f / (1.0f + relay->distance)) : 1.0f;
    float load_score = (relay->message_count > 0) ? (1.0f / (1.0f + relay->message_count / 100.0f)) : 1.0f;

    return 0.4f * energy_score +
           0.3f * reliability_score +
           0.2f * distance_score +
           0.1f * load_score;
}

/**
 * @brief Find harvest opportunity by position
 */
static NimcpHarvestOpportunity* find_harvest_opportunity(
    nimcp_energy_gossip* gossip,
    const float position[3]
) {
    if (!gossip || !gossip->opportunities || !position) return NULL;

    const float tolerance = 1.0f; /* 1 unit tolerance */

    for (uint32_t i = 0; i < gossip->opportunity_count; i++) {
        float dist = calculate_distance(gossip->opportunities[i].position, position);
        if (dist < tolerance) {
            return &gossip->opportunities[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Core Functions Implementation
 * ============================================================================ */

void nimcp_energy_gossip_default_config(nimcp_energy_gossipConfig* config) {
    if (!config) return;

    memset(config, 0, sizeof(nimcp_energy_gossipConfig));

    /* Adaptive intervals (milliseconds) */
    config->intervals.interval_critical_ms = 10000;  /* 10 seconds */
    config->intervals.interval_low_ms = 5000;        /* 5 seconds */
    config->intervals.interval_normal_ms = 1000;     /* 1 second */
    config->intervals.interval_high_ms = 500;        /* 500 ms */
    config->intervals.interval_full_ms = 250;        /* 250 ms */
    config->intervals.interval_charging_ms = 2000;   /* 2 seconds */

    /* Forwarding probabilities */
    config->forwarding_probs.prob_critical = 0.05f;  /* 5% when critical */
    config->forwarding_probs.prob_low = 0.25f;       /* 25% when low */
    config->forwarding_probs.prob_normal = 0.75f;    /* 75% normal */
    config->forwarding_probs.prob_high = 0.95f;      /* 95% when high */
    config->forwarding_probs.prob_full = 1.0f;       /* 100% when full */
    config->forwarding_probs.prob_charging = 0.50f;  /* 50% while charging */

    /* Relay configuration */
    config->max_relay_candidates = 10;
    config->min_relay_energy = 30.0f;  /* 30% minimum */

    /* Sleep configuration */
    config->sleep_check_interval_ms = 1000;  /* Check every second */

    /* Feature flags */
    config->enable_harvest_awareness = true;
    config->enable_coordinated_sleep = true;

    /* Gossip parameters */
    config->gossip_fanout = 3;  /* Gossip to 3 nodes */
    config->max_message_cache = 100;
    config->convergence_threshold = 0.95f;
}

nimcp_energy_gossip* nimcp_energy_gossip_create(
    uint32_t node_id,
    const nimcp_energy_gossipConfig* config
) {
    nimcp_energy_gossip* gossip = (nimcp_energy_gossip*)nimcp_malloc(sizeof(nimcp_energy_gossip));
    if (!gossip) {
        NIMCP_LOG_ERROR("Failed to allocate energy gossip protocol");
        return NULL;
    }

    memset(gossip, 0, sizeof(nimcp_energy_gossip));
    gossip->node_id = node_id;

    /* Apply configuration */
    if (config) {
        memcpy(&gossip->config, config, sizeof(nimcp_energy_gossipConfig));
    } else {
        nimcp_energy_gossip_default_config(&gossip->config);
    }

    /* Initialize energy state */
    gossip->current_state = NIMCP_ENERGY_NORMAL;
    gossip->stats.current_level = 100.0f;
    gossip->stats.consumption_rate = 1.0f;  /* Default rate */
    gossip->stats.harvest_rate = 0.0f;
    gossip->stats.last_update = time(NULL);

    /* Initialize emergency reserve */
    gossip->reserve.reserve_percentage = 15.0f;
    gossip->reserve.auto_return_enabled = true;
    gossip->reserve.emergency_threshold = 10.0f;

    /* Allocate message cache */
    gossip->message_cache_capacity = gossip->config.max_message_cache;
    gossip->message_cache = (NimcpGossipMessage**)nimcp_calloc(
        gossip->message_cache_capacity,
        sizeof(NimcpGossipMessage*)
    );
    if (!gossip->message_cache) {
        NIMCP_LOG_ERROR("Failed to allocate message cache");
        nimcp_free(gossip);
        return NULL;
    }

    /* Allocate relay nodes array */
    gossip->relay_node_capacity = gossip->config.max_relay_candidates * 2;
    gossip->relay_nodes = (NimcpRelayNode*)nimcp_calloc(
        gossip->relay_node_capacity,
        sizeof(NimcpRelayNode)
    );
    if (!gossip->relay_nodes) {
        NIMCP_LOG_ERROR("Failed to allocate relay nodes");
        nimcp_free(gossip->message_cache);
        nimcp_free(gossip);
        return NULL;
    }

    /* Allocate harvest opportunities array */
    if (gossip->config.enable_harvest_awareness) {
        gossip->opportunities = (NimcpHarvestOpportunity*)nimcp_calloc(
            10,  /* Initial capacity */
            sizeof(NimcpHarvestOpportunity)
        );
    }

    /* Create mutex */
    gossip->mutex = nimcp_mutex_create();
    if (!gossip->mutex) {
        NIMCP_LOG_ERROR("Failed to create mutex");
        nimcp_free(gossip->opportunities);
        nimcp_free(gossip->relay_nodes);
        nimcp_free(gossip->message_cache);
        nimcp_free(gossip);
        return NULL;
    }

    gossip->next_message_id = 1;
    gossip->sequence_number = 0;
    gossip->is_initialized = true;

    NIMCP_LOG_INFO("Energy-aware gossip protocol created for node %u", node_id);
    return gossip;
}

void nimcp_energy_gossip_destroy(nimcp_energy_gossip* gossip) {
    if (!gossip) return;

    NIMCP_LOG_INFO("Destroying energy gossip protocol for node %u", gossip->node_id);

    /* Free message cache */
    if (gossip->message_cache) {
        for (uint32_t i = 0; i < gossip->message_cache_size; i++) {
            if (gossip->message_cache[i]) {
                if (gossip->message_cache[i]->payload) {
                    nimcp_free(gossip->message_cache[i]->payload);
                }
                nimcp_free(gossip->message_cache[i]);
            }
        }
        nimcp_free(gossip->message_cache);
    }

    /* Free relay nodes */
    nimcp_free(gossip->relay_nodes);

    /* Free sleep schedule */
    nimcp_free(gossip->sleep_schedule);

    /* Free harvest opportunities */
    nimcp_free(gossip->opportunities);

    /* Destroy mutex */
    if (gossip->mutex) {
        nimcp_mutex_destroy(gossip->mutex);
    }

    nimcp_free(gossip);
}

nimcp_result_t nimcp_energy_gossip_init_bio_async(
    nimcp_energy_gossip* gossip,
    bio_inbox_t* inbox
) {
    if (!gossip || !inbox) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);
    gossip->inbox = inbox;
    gossip->bio_async_enabled = true;
    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_INFO("Bio-async enabled for node %u", gossip->node_id);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Energy Management Implementation
 * ============================================================================ */

nimcp_result_t nimcp_energy_gossip_update_energy(
    nimcp_energy_gossip* gossip,
    float energy_level
) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;
    if (energy_level < 0.0f || energy_level > 100.0f) {
        return NIMCP_ERR_INVALID_ARGUMENT;
    }

    nimcp_mutex_lock(gossip->mutex);

    float old_level = gossip->stats.current_level;
    gossip->stats.current_level = energy_level;

    /* Update consumption tracking */
    time_t now = time(NULL);
    float time_delta = (float)difftime(now, gossip->stats.last_update);
    if (time_delta > 0.0f) {
        float energy_delta = old_level - energy_level;
        if (energy_delta > 0.0f) {
            gossip->stats.total_consumed += (uint64_t)(energy_delta * 1000);
            /* Update consumption rate (exponential moving average) */
            float current_rate = energy_delta / time_delta;
            gossip->stats.consumption_rate =
                0.8f * gossip->stats.consumption_rate + 0.2f * current_rate;
        } else if (energy_delta < 0.0f) {
            /* Harvesting energy */
            gossip->stats.total_harvested += (uint64_t)(-energy_delta * 1000);
        }
    }
    gossip->stats.last_update = now;

    /* Update predicted lifetime */
    if (gossip->stats.consumption_rate > gossip->stats.harvest_rate) {
        float net_consumption = gossip->stats.consumption_rate - gossip->stats.harvest_rate;
        gossip->stats.predicted_lifetime =
            (net_consumption > 0.0f) ? (energy_level / net_consumption) : INFINITY;
    } else {
        gossip->stats.predicted_lifetime = INFINITY;
    }

    /* Update energy state */
    update_energy_state(gossip);

    /* Check emergency return */
    if (gossip->reserve.auto_return_enabled &&
        nimcp_energy_gossip_needs_emergency_return(gossip)) {
        NIMCP_LOG_WARN("Node %u: Emergency return needed (%.1f%%)",
            gossip->node_id, energy_level);
    }

    nimcp_mutex_unlock(gossip->mutex);

    return NIMCP_SUCCESS;
}

NimcpEnergyState nimcp_energy_gossip_get_state(const nimcp_energy_gossip* gossip) {
    if (!gossip) return NIMCP_ENERGY_CRITICAL;
    return gossip->current_state;
}

nimcp_result_t nimcp_energy_gossip_set_consumption_rate(
    nimcp_energy_gossip* gossip,
    float rate
) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;
    if (rate < 0.0f) return NIMCP_ERR_INVALID_ARGUMENT;

    nimcp_mutex_lock(gossip->mutex);
    gossip->stats.consumption_rate = rate;
    nimcp_mutex_unlock(gossip->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_set_harvest_rate(
    nimcp_energy_gossip* gossip,
    float rate
) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;
    if (rate < 0.0f) return NIMCP_ERR_INVALID_ARGUMENT;

    nimcp_mutex_lock(gossip->mutex);
    gossip->stats.harvest_rate = rate;

    /* Update energy state in case we're now charging */
    update_energy_state(gossip);

    nimcp_mutex_unlock(gossip->mutex);

    return NIMCP_SUCCESS;
}

float nimcp_energy_gossip_predict_lifetime(const nimcp_energy_gossip* gossip) {
    if (!gossip) return 0.0f;
    return gossip->stats.predicted_lifetime;
}

nimcp_result_t nimcp_energy_gossip_get_stats(
    const nimcp_energy_gossip* gossip,
    NimcpEnergyStats* stats
) {
    if (!gossip || !stats) return NIMCP_ERROR_NULL_POINTER;

    memcpy(stats, &gossip->stats, sizeof(NimcpEnergyStats));
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Emergency Reserve Implementation
 * ============================================================================ */

nimcp_result_t nimcp_energy_gossip_configure_reserve(
    nimcp_energy_gossip* gossip,
    float reserve_percentage,
    float return_distance,
    float return_energy_cost
) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;
    if (reserve_percentage < 0.0f || reserve_percentage > 50.0f) {
        return NIMCP_ERR_INVALID_ARGUMENT;
    }

    nimcp_mutex_lock(gossip->mutex);
    gossip->reserve.reserve_percentage = reserve_percentage;
    gossip->reserve.return_distance = return_distance;
    gossip->reserve.return_energy_cost = return_energy_cost;
    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_INFO("Node %u: Emergency reserve configured (%.1f%%, cost: %.1f)",
        gossip->node_id, reserve_percentage, return_energy_cost);

    return NIMCP_SUCCESS;
}

bool nimcp_energy_gossip_needs_emergency_return(const nimcp_energy_gossip* gossip) {
    if (!gossip) return false;

    /* Need emergency return if:
     * 1. Energy below emergency threshold, OR
     * 2. Current energy < return cost + reserve
     */
    float required_energy = gossip->reserve.return_energy_cost +
                           gossip->reserve.reserve_percentage;

    return gossip->stats.current_level <= gossip->reserve.emergency_threshold ||
           gossip->stats.current_level < required_energy;
}

nimcp_result_t nimcp_energy_gossip_emergency_mode(nimcp_energy_gossip* gossip) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);
    gossip->reserve.emergency_mode = true;
    gossip->current_state = NIMCP_ENERGY_CRITICAL;
    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_WARN("Node %u: EMERGENCY MODE ACTIVATED", gossip->node_id);

    /* Broadcast emergency message if bio-async enabled */
    if (gossip->bio_async_enabled && gossip->inbox) {
        /* Emergency broadcast implementation would go here */
        NIMCP_LOG_INFO("Node %u: Broadcasting emergency status", gossip->node_id);
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Gossip Protocol Implementation
 * ============================================================================ */

uint64_t nimcp_energy_gossip_broadcast(
    nimcp_energy_gossip* gossip,
    const void* payload,
    size_t payload_size,
    NimcpMessagePriority priority,
    uint32_t ttl
) {
    if (!gossip || !payload || payload_size == 0) return 0;

    /* Check if we should send based on energy state */
    if (gossip->current_state == NIMCP_ENERGY_CRITICAL &&
        priority < NIMCP_MSG_PRIORITY_HIGH) {
        NIMCP_LOG_DEBUG("Node %u: Dropping non-critical message due to energy",
            gossip->node_id);
        gossip->messages_dropped++;
        return 0;
    }

    nimcp_mutex_lock(gossip->mutex);

    /* Create message */
    NimcpGossipMessage message;
    memset(&message, 0, sizeof(NimcpGossipMessage));

    message.header.message_id = gossip->next_message_id++;
    message.header.originator_id = gossip->node_id;
    message.header.hop_count = 0;
    message.header.ttl = ttl;
    message.header.priority = priority;
    message.header.timestamp = time(NULL);
    message.header.sequence_number = gossip->sequence_number++;
    message.payload_size = payload_size;
    message.payload = (void*)payload;  /* Casting away const for structure */
    message.forwarded = false;
    message.forward_count = 0;

    /* Add to cache */
    add_to_cache(gossip, &message);

    gossip->messages_sent++;

    uint64_t message_id = message.header.message_id;
    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_DEBUG("Node %u: Broadcast message %llu (priority: %d, ttl: %u)",
        gossip->node_id, (unsigned long long)message_id, priority, ttl);

    return message_id;
}

nimcp_result_t nimcp_energy_gossip_receive(
    nimcp_energy_gossip* gossip,
    const NimcpGossipMessage* message
) {
    if (!gossip || !message) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    /* Check if already received */
    if (is_message_cached(gossip, message->header.message_id)) {
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_SUCCESS;  /* Duplicate, ignore */
    }

    /* Check if we should process based on priority and energy */
    if (!nimcp_energy_gossip_should_process(gossip, message->header.priority)) {
        gossip->messages_dropped++;
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_ERR_SKIP;
    }

    /* Add to cache */
    add_to_cache(gossip, message);

    gossip->messages_received++;

    NIMCP_LOG_DEBUG("Node %u: Received message %llu from node %u (hop %u/%u)",
        gossip->node_id,
        (unsigned long long)message->header.message_id,
        message->header.originator_id,
        message->header.hop_count,
        message->header.ttl);

    nimcp_mutex_unlock(gossip->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_forward(
    nimcp_energy_gossip* gossip,
    NimcpGossipMessage* message
) {
    if (!gossip || !message) return NIMCP_ERROR_NULL_POINTER;

    /* Check TTL */
    if (message->header.hop_count >= message->header.ttl) {
        return NIMCP_ERR_SKIP;
    }

    /* Check if we're sleeping */
    if (gossip->is_sleeping && message->header.priority < NIMCP_MSG_PRIORITY_CRITICAL) {
        return NIMCP_ERR_SKIP;
    }

    /* Probabilistic forwarding based on energy state */
    if (!nimcp_energy_gossip_should_forward(gossip, message)) {
        return NIMCP_ERR_SKIP;
    }

    nimcp_mutex_lock(gossip->mutex);

    message->header.hop_count++;
    message->forwarded = true;
    message->forward_count++;
    gossip->messages_forwarded++;

    NIMCP_LOG_DEBUG("Node %u: Forwarding message %llu (hop %u/%u)",
        gossip->node_id,
        (unsigned long long)message->header.message_id,
        message->header.hop_count,
        message->header.ttl);

    nimcp_mutex_unlock(gossip->mutex);

    return NIMCP_SUCCESS;
}

float nimcp_energy_gossip_get_forward_probability(const nimcp_energy_gossip* gossip) {
    if (!gossip) return 0.0f;

    switch (gossip->current_state) {
        case NIMCP_ENERGY_CRITICAL:
            return gossip->config.forwarding_probs.prob_critical;
        case NIMCP_ENERGY_LOW:
            return gossip->config.forwarding_probs.prob_low;
        case NIMCP_ENERGY_NORMAL:
            return gossip->config.forwarding_probs.prob_normal;
        case NIMCP_ENERGY_HIGH:
            return gossip->config.forwarding_probs.prob_high;
        case NIMCP_ENERGY_FULL:
            return gossip->config.forwarding_probs.prob_full;
        case NIMCP_ENERGY_CHARGING:
            return gossip->config.forwarding_probs.prob_charging;
        default:
            return 0.5f;
    }
}

bool nimcp_energy_gossip_should_forward(
    const nimcp_energy_gossip* gossip,
    const NimcpGossipMessage* message
) {
    if (!gossip || !message) return false;

    /* Always forward critical messages */
    if (message->header.priority == NIMCP_MSG_PRIORITY_CRITICAL) {
        return true;
    }

    /* Never forward in emergency mode unless critical */
    if (gossip->reserve.emergency_mode) {
        return false;
    }

    /* Probabilistic forwarding based on energy state */
    float probability = nimcp_energy_gossip_get_forward_probability(gossip);
    return random_float() < probability;
}

/* ============================================================================
 * Relay Selection Implementation
 * ============================================================================ */

nimcp_result_t nimcp_energy_gossip_register_relay(
    nimcp_energy_gossip* gossip,
    uint32_t node_id,
    float energy_level,
    float distance
) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;
    if (node_id == gossip->node_id) return NIMCP_ERR_INVALID_ARGUMENT;

    nimcp_mutex_lock(gossip->mutex);

    /* Check if already registered */
    NimcpRelayNode* existing = find_relay_node(gossip, node_id);
    if (existing) {
        /* Update existing */
        existing->energy_level = energy_level;
        existing->energy_state = nimcp_energy_level_to_state(energy_level, false);
        existing->distance = distance;
        existing->last_seen = time(NULL);
        existing->is_available = (energy_level >= gossip->config.min_relay_energy);
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_SUCCESS;
    }

    /* Add new relay node */
    if (gossip->relay_node_count >= gossip->relay_node_capacity) {
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_ERR_BUFFER_OVERFLOW;
    }

    NimcpRelayNode* relay = &gossip->relay_nodes[gossip->relay_node_count++];
    memset(relay, 0, sizeof(NimcpRelayNode));

    relay->node_id = node_id;
    relay->energy_state = nimcp_energy_level_to_state(energy_level, false);
    relay->energy_level = energy_level;
    relay->reliability_score = 1.0f;  /* Initial score */
    relay->distance = distance;
    relay->message_count = 0;
    relay->last_seen = time(NULL);
    relay->is_available = (energy_level >= gossip->config.min_relay_energy);

    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_DEBUG("Node %u: Registered relay node %u (energy: %.1f%%, distance: %.1f)",
        gossip->node_id, node_id, energy_level, distance);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_update_relay(
    nimcp_energy_gossip* gossip,
    uint32_t node_id,
    float energy_level
) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    NimcpRelayNode* relay = find_relay_node(gossip, node_id);
    if (!relay) {
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_NOT_FOUND;
    }

    relay->energy_level = energy_level;
    relay->energy_state = nimcp_energy_level_to_state(energy_level, false);
    relay->last_seen = time(NULL);
    relay->is_available = (energy_level >= gossip->config.min_relay_energy);

    nimcp_mutex_unlock(gossip->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_select_relays(
    nimcp_energy_gossip* gossip,
    uint32_t max_relays,
    uint32_t* selected,
    uint32_t* selected_count
) {
    if (!gossip || !selected || !selected_count) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    *selected_count = 0;

    /* Calculate scores for all available relays */
    float* scores = (float*)nimcp_calloc(gossip->relay_node_count, sizeof(float));
    if (!scores) {
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_NO_MEMORY_ALLOCATION;
    }

    for (uint32_t i = 0; i < gossip->relay_node_count; i++) {
        scores[i] = calculate_relay_score(&gossip->relay_nodes[i]);
    }

    /* Select top N relays by score */
    for (uint32_t n = 0; n < max_relays && n < gossip->relay_node_count; n++) {
        /* Find highest scoring relay */
        float max_score = -1.0f;
        uint32_t max_idx = 0;

        for (uint32_t i = 0; i < gossip->relay_node_count; i++) {
            if (scores[i] > max_score) {
                max_score = scores[i];
                max_idx = i;
            }
        }

        if (max_score > 0.0f) {
            selected[*selected_count] = gossip->relay_nodes[max_idx].node_id;
            (*selected_count)++;
            scores[max_idx] = -1.0f;  /* Mark as selected */
        } else {
            break;  /* No more available relays */
        }
    }

    nimcp_free(scores);
    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_DEBUG("Node %u: Selected %u relay nodes", gossip->node_id, *selected_count);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_get_relay_info(
    const nimcp_energy_gossip* gossip,
    uint32_t node_id,
    NimcpRelayNode* relay
) {
    if (!gossip || !relay) return NIMCP_ERROR_NULL_POINTER;

    for (uint32_t i = 0; i < gossip->relay_node_count; i++) {
        if (gossip->relay_nodes[i].node_id == node_id) {
            memcpy(relay, &gossip->relay_nodes[i], sizeof(NimcpRelayNode));
            return NIMCP_SUCCESS;
        }
    }

    return NIMCP_NOT_FOUND;
}

/* ============================================================================
 * Sleep Scheduling Implementation
 * ============================================================================ */

nimcp_result_t nimcp_energy_gossip_schedule_sleep(
    nimcp_energy_gossip* gossip,
    time_t start_time,
    uint32_t duration_seconds,
    bool wake_on_emergency
) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    /* Expand schedule array if needed */
    if (!gossip->sleep_schedule) {
        gossip->sleep_schedule = (NimcpSleepSchedule*)nimcp_calloc(
            10, sizeof(NimcpSleepSchedule)
        );
        if (!gossip->sleep_schedule) {
            nimcp_mutex_unlock(gossip->mutex);
            return NIMCP_NO_MEMORY_ALLOCATION;
        }
    }

    NimcpSleepSchedule* schedule = &gossip->sleep_schedule[gossip->sleep_schedule_count++];
    schedule->sleep_start = start_time;
    schedule->sleep_end = start_time + duration_seconds;
    schedule->duration_seconds = duration_seconds;
    schedule->wake_on_emergency = wake_on_emergency;
    schedule->wake_events = 0;

    /* Estimate energy saved */
    float active_consumption = gossip->stats.consumption_rate;
    float sleep_consumption = active_consumption * 0.1f;  /* 10% consumption while sleeping */
    schedule->energy_saved = (active_consumption - sleep_consumption) * duration_seconds;

    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_INFO("Node %u: Sleep scheduled for %u seconds (saving ~%.1f energy units)",
        gossip->node_id, duration_seconds, schedule->energy_saved);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_sleep(
    nimcp_energy_gossip* gossip,
    uint32_t duration_seconds
) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    gossip->is_sleeping = true;
    gossip->sleep_until = time(NULL) + duration_seconds;
    gossip->sleep_cycles++;

    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_INFO("Node %u: Entering sleep mode for %u seconds",
        gossip->node_id, duration_seconds);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_wake(nimcp_energy_gossip* gossip) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    if (gossip->is_sleeping) {
        gossip->is_sleeping = false;
        gossip->sleep_until = 0;

        NIMCP_LOG_INFO("Node %u: Waking from sleep", gossip->node_id);
    }

    nimcp_mutex_unlock(gossip->mutex);

    return NIMCP_SUCCESS;
}

bool nimcp_energy_gossip_is_sleeping(const nimcp_energy_gossip* gossip) {
    if (!gossip) return false;

    /* Check if sleep period has expired */
    if (gossip->is_sleeping && gossip->sleep_until > 0) {
        time_t now = time(NULL);
        if (now >= gossip->sleep_until) {
            return false;  /* Sleep period expired */
        }
    }

    return gossip->is_sleeping;
}

nimcp_result_t nimcp_energy_gossip_coordinate_sleep(
    nimcp_energy_gossip* gossip,
    const NimcpSleepSchedule* node_schedules,
    uint32_t node_count
) {
    if (!gossip || !node_schedules) return NIMCP_ERROR_NULL_POINTER;

    /* Analyze other nodes' schedules to find optimal sleep time */
    time_t now = time(NULL);
    uint32_t sleeping_nodes[24] = {0};  /* Nodes sleeping per hour */

    for (uint32_t i = 0; i < node_count; i++) {
        time_t start = node_schedules[i].sleep_start;
        time_t end = node_schedules[i].sleep_end;

        for (time_t t = start; t < end; t += 3600) {
            uint32_t hour = (uint32_t)((t - now) / 3600) % 24;
            sleeping_nodes[hour]++;
        }
    }

    /* Find hour with minimum sleeping nodes */
    uint32_t min_sleeping = UINT32_MAX;
    uint32_t best_hour = 0;

    for (uint32_t h = 0; h < 24; h++) {
        if (sleeping_nodes[h] < min_sleeping) {
            min_sleeping = sleeping_nodes[h];
            best_hour = h;
        }
    }

    /* Schedule sleep at optimal time */
    time_t sleep_start = now + (best_hour * 3600);
    uint32_t sleep_duration = 3600;  /* 1 hour default */

    nimcp_result_t result = nimcp_energy_gossip_schedule_sleep(
        gossip, sleep_start, sleep_duration, true
    );

    NIMCP_LOG_INFO("Node %u: Coordinated sleep scheduled in %u hours (%u nodes sleeping)",
        gossip->node_id, best_hour, min_sleeping);

    return result;
}

/* ============================================================================
 * Harvest Awareness Implementation
 * ============================================================================ */

nimcp_result_t nimcp_energy_gossip_register_harvest(
    nimcp_energy_gossip* gossip,
    const float position[3],
    float harvest_rate,
    float quality_score
) {
    if (!gossip || !position) return NIMCP_ERROR_NULL_POINTER;
    if (!gossip->config.enable_harvest_awareness) return NIMCP_NOT_IMPLEMENTED;

    nimcp_mutex_lock(gossip->mutex);

    /* Check if already registered */
    NimcpHarvestOpportunity* existing = find_harvest_opportunity(gossip, position);
    if (existing) {
        existing->harvest_rate = harvest_rate;
        existing->quality_score = quality_score;
        existing->available = true;
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_SUCCESS;
    }

    /* Add new opportunity */
    if (gossip->opportunity_count >= 10) {  /* Max capacity */
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_ERR_BUFFER_OVERFLOW;
    }

    NimcpHarvestOpportunity* opp = &gossip->opportunities[gossip->opportunity_count++];
    memset(opp, 0, sizeof(NimcpHarvestOpportunity));

    opp->available = true;
    opp->harvest_rate = harvest_rate;
    memcpy(opp->position, position, sizeof(float) * 3);
    opp->congestion_level = 0;
    opp->quality_score = quality_score;
    opp->discovered_time = time(NULL);

    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_INFO("Node %u: Harvest opportunity registered (rate: %.2f, quality: %.2f)",
        gossip->node_id, harvest_rate, quality_score);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_select_harvest(
    nimcp_energy_gossip* gossip,
    NimcpHarvestOpportunity* opportunity
) {
    if (!gossip || !opportunity) return NIMCP_ERROR_NULL_POINTER;
    if (!gossip->config.enable_harvest_awareness) return NIMCP_NOT_IMPLEMENTED;

    nimcp_mutex_lock(gossip->mutex);

    /* Find best opportunity (highest score with low congestion) */
    float best_score = -1.0f;
    NimcpHarvestOpportunity* best = NULL;

    for (uint32_t i = 0; i < gossip->opportunity_count; i++) {
        NimcpHarvestOpportunity* opp = &gossip->opportunities[i];
        if (!opp->available) continue;

        /* Calculate combined score */
        float congestion_penalty = 1.0f / (1.0f + opp->congestion_level * 0.5f);
        float score = opp->quality_score * opp->harvest_rate * congestion_penalty;

        if (score > best_score) {
            best_score = score;
            best = opp;
        }
    }

    if (!best) {
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_NOT_FOUND;
    }

    memcpy(opportunity, best, sizeof(NimcpHarvestOpportunity));
    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_INFO("Node %u: Selected harvest opportunity (score: %.2f)",
        gossip->node_id, best_score);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_update_harvest_congestion(
    nimcp_energy_gossip* gossip,
    const float position[3],
    uint32_t congestion_level
) {
    if (!gossip || !position) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    NimcpHarvestOpportunity* opp = find_harvest_opportunity(gossip, position);
    if (!opp) {
        nimcp_mutex_unlock(gossip->mutex);
        return NIMCP_NOT_FOUND;
    }

    opp->congestion_level = congestion_level;
    nimcp_mutex_unlock(gossip->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_start_harvest(nimcp_energy_gossip* gossip) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    /* Select best harvest opportunity */
    NimcpHarvestOpportunity opportunity;
    nimcp_result_t result = nimcp_energy_gossip_select_harvest(gossip, &opportunity);
    if (result != NIMCP_SUCCESS) {
        nimcp_mutex_unlock(gossip->mutex);
        return result;
    }

    gossip->current_harvest = find_harvest_opportunity(gossip, opportunity.position);
    if (gossip->current_harvest) {
        gossip->stats.harvest_rate = gossip->current_harvest->harvest_rate;
        update_energy_state(gossip);
    }

    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_INFO("Node %u: Started harvesting (rate: %.2f)",
        gossip->node_id, opportunity.harvest_rate);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_stop_harvest(nimcp_energy_gossip* gossip) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(gossip->mutex);

    gossip->current_harvest = NULL;
    gossip->stats.harvest_rate = 0.0f;
    update_energy_state(gossip);

    nimcp_mutex_unlock(gossip->mutex);

    NIMCP_LOG_INFO("Node %u: Stopped harvesting", gossip->node_id);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Adaptive Interval Implementation
 * ============================================================================ */

uint32_t nimcp_energy_gossip_get_heartbeat_interval(const nimcp_energy_gossip* gossip) {
    if (!gossip) return 1000;  /* Default 1 second */

    switch (gossip->current_state) {
        case NIMCP_ENERGY_CRITICAL:
            return gossip->config.intervals.interval_critical_ms;
        case NIMCP_ENERGY_LOW:
            return gossip->config.intervals.interval_low_ms;
        case NIMCP_ENERGY_NORMAL:
            return gossip->config.intervals.interval_normal_ms;
        case NIMCP_ENERGY_HIGH:
            return gossip->config.intervals.interval_high_ms;
        case NIMCP_ENERGY_FULL:
            return gossip->config.intervals.interval_full_ms;
        case NIMCP_ENERGY_CHARGING:
            return gossip->config.intervals.interval_charging_ms;
        default:
            return gossip->config.intervals.interval_normal_ms;
    }
}

bool nimcp_energy_gossip_should_process(
    const nimcp_energy_gossip* gossip,
    NimcpMessagePriority priority
) {
    if (!gossip) return false;

    /* Always process critical messages */
    if (priority == NIMCP_MSG_PRIORITY_CRITICAL) {
        return true;
    }

    /* In emergency mode, only process high priority and above */
    if (gossip->reserve.emergency_mode) {
        return priority <= NIMCP_MSG_PRIORITY_HIGH;
    }

    /* Energy-based filtering */
    switch (gossip->current_state) {
        case NIMCP_ENERGY_CRITICAL:
            return priority <= NIMCP_MSG_PRIORITY_HIGH;
        case NIMCP_ENERGY_LOW:
            return priority <= NIMCP_MSG_PRIORITY_NORMAL;
        case NIMCP_ENERGY_NORMAL:
        case NIMCP_ENERGY_HIGH:
        case NIMCP_ENERGY_FULL:
        case NIMCP_ENERGY_CHARGING:
            return true;  /* Process all messages */
        default:
            return true;
    }
}

/* ============================================================================
 * Bio-Async Message Handlers
 * ============================================================================ */

nimcp_result_t nimcp_energy_gossip_handle_energy_broadcast(
    nimcp_energy_gossip* gossip,
    const bio_message_t* message
) {
    if (!gossip || !message) return NIMCP_ERROR_NULL_POINTER;

    NIMCP_LOG_DEBUG("Node %u: Handling energy broadcast from node %u",
        gossip->node_id, message->sender_id);

    /* Extract energy information from message payload */
    if (message->payload && message->size >= sizeof(float)) {
        float* energy_level = (float*)message->payload;

        /* Update relay information */
        nimcp_energy_gossip_update_relay(gossip, message->sender_id, *energy_level);
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_handle_sleep_coordination(
    nimcp_energy_gossip* gossip,
    const bio_message_t* message
) {
    if (!gossip || !message) return NIMCP_ERROR_NULL_POINTER;

    NIMCP_LOG_DEBUG("Node %u: Handling sleep coordination from node %u",
        gossip->node_id, message->sender_id);

    /* Process sleep schedule information */
    if (message->payload && message->size >= sizeof(NimcpSleepSchedule)) {
        /* Would coordinate with other node's schedule */
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_energy_gossip_handle_relay_request(
    nimcp_energy_gossip* gossip,
    const bio_message_t* message
) {
    if (!gossip || !message) return NIMCP_ERROR_NULL_POINTER;

    NIMCP_LOG_DEBUG("Node %u: Handling relay request from node %u",
        gossip->node_id, message->sender_id);

    /* Check if we can serve as relay */
    if (gossip->stats.current_level >= gossip->config.min_relay_energy &&
        !gossip->reserve.emergency_mode) {
        /* Accept relay request */
        NIMCP_LOG_INFO("Node %u: Accepting relay request from node %u",
            gossip->node_id, message->sender_id);
        return NIMCP_SUCCESS;
    } else {
        /* Decline relay request */
        NIMCP_LOG_INFO("Node %u: Declining relay request (energy: %.1f%%)",
            gossip->node_id, gossip->stats.current_level);
        return NIMCP_NOT_INITIALIZED;
    }
}

nimcp_result_t nimcp_energy_gossip_process_inbox(nimcp_energy_gossip* gossip) {
    if (!gossip) return NIMCP_ERROR_NULL_POINTER;
    if (!gossip->bio_async_enabled || !gossip->inbox) {
        return NIMCP_NOT_IMPLEMENTED;
    }

    /* Process all messages in inbox */
    bio_message_t message;
    while (nimcp_bio_inbox_try_receive(gossip->inbox, &message, 0) == NIMCP_SUCCESS) {
        /* Route message based on type */
        switch (message.type) {
            case BIO_MSG_CUSTOM:
                /* Check custom message subtype */
                if (message.subtype == 1) {  /* Energy broadcast */
                    nimcp_energy_gossip_handle_energy_broadcast(gossip, &message);
                } else if (message.subtype == 2) {  /* Sleep coordination */
                    nimcp_energy_gossip_handle_sleep_coordination(gossip, &message);
                } else if (message.subtype == 3) {  /* Relay request */
                    nimcp_energy_gossip_handle_relay_request(gossip, &message);
                }
                break;

            default:
                NIMCP_LOG_DEBUG("Node %u: Unhandled message type %d",
                    gossip->node_id, message.type);
                break;
        }

        /* Free message payload if needed */
        if (message.payload && message.size > 0) {
            nimcp_free(message.payload);
        }
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

NimcpEnergyState nimcp_energy_level_to_state(float energy_level, bool is_charging) {
    if (is_charging) {
        return NIMCP_ENERGY_CHARGING;
    }

    if (energy_level < NIMCP_ENERGY_CRITICAL_THRESHOLD) {
        return NIMCP_ENERGY_CRITICAL;
    } else if (energy_level < NIMCP_ENERGY_LOW_THRESHOLD) {
        return NIMCP_ENERGY_LOW;
    } else if (energy_level < NIMCP_ENERGY_NORMAL_THRESHOLD) {
        return NIMCP_ENERGY_NORMAL;
    } else if (energy_level < NIMCP_ENERGY_HIGH_THRESHOLD) {
        return NIMCP_ENERGY_HIGH;
    } else {
        return NIMCP_ENERGY_FULL;
    }
}

const char* nimcp_energy_state_to_string(NimcpEnergyState state) {
    switch (state) {
        case NIMCP_ENERGY_CRITICAL: return "CRITICAL";
        case NIMCP_ENERGY_LOW:      return "LOW";
        case NIMCP_ENERGY_NORMAL:   return "NORMAL";
        case NIMCP_ENERGY_HIGH:     return "HIGH";
        case NIMCP_ENERGY_FULL:     return "FULL";
        case NIMCP_ENERGY_CHARGING: return "CHARGING";
        default:                     return "UNKNOWN";
    }
}

void nimcp_energy_gossip_print_stats(const nimcp_energy_gossip* gossip) {
    if (!gossip) return;

    printf("\n========== Energy Gossip Statistics (Node %u) ==========\n", gossip->node_id);
    printf("Energy State: %s (%.1f%%)\n",
        nimcp_energy_state_to_string(gossip->current_state),
        gossip->stats.current_level);
    printf("Consumption Rate: %.2f units/sec\n", gossip->stats.consumption_rate);
    printf("Harvest Rate: %.2f units/sec\n", gossip->stats.harvest_rate);
    printf("Predicted Lifetime: %.1f seconds\n", gossip->stats.predicted_lifetime);
    printf("Total Consumed: %llu units\n",
        (unsigned long long)gossip->stats.total_consumed);
    printf("Total Harvested: %llu units\n",
        (unsigned long long)gossip->stats.total_harvested);
    printf("Charge Cycles: %u\n", gossip->stats.charge_cycles);

    printf("\nMessaging:\n");
    printf("  Sent: %llu\n", (unsigned long long)gossip->messages_sent);
    printf("  Received: %llu\n", (unsigned long long)gossip->messages_received);
    printf("  Forwarded: %llu\n", (unsigned long long)gossip->messages_forwarded);
    printf("  Dropped: %llu\n", (unsigned long long)gossip->messages_dropped);

    printf("\nRelay Nodes: %u registered\n", gossip->relay_node_count);
    printf("Energy State Changes: %llu\n",
        (unsigned long long)gossip->energy_state_changes);
    printf("Sleep Cycles: %llu\n", (unsigned long long)gossip->sleep_cycles);

    if (gossip->reserve.emergency_mode) {
        printf("\n*** EMERGENCY MODE ACTIVE ***\n");
    }

    if (gossip->is_sleeping) {
        printf("\n*** CURRENTLY SLEEPING ***\n");
    }

    if (gossip->current_harvest) {
        printf("\n*** CURRENTLY HARVESTING (rate: %.2f) ***\n",
            gossip->stats.harvest_rate);
    }

    printf("=====================================================\n\n");
}
