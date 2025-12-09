/**
 * @file nimcp_swarm_logic_bridge.c
 * @brief Implementation of Logic-Swarm Bridge
 *
 * Bridges neural logic gates with swarm intelligence for consensus,
 * validation, and distributed inference.
 */

#include "swarm/nimcp_swarm_logic_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/time/nimcp_time.h"
#include "security/nimcp_security.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/*=============================================================================
 * CONSTANTS
 *============================================================================*/

#define SWARM_LOGIC_DEFAULT_MAX_RULES 1000
#define SWARM_LOGIC_DEFAULT_CACHE_SIZE 256
#define SWARM_LOGIC_DEFAULT_MAX_AGENTS 100
#define SWARM_LOGIC_DEFAULT_INFERENCE_THRESHOLD 0.7f
#define SWARM_LOGIC_DEFAULT_TIMEOUT_MS 1000.0f
#define SWARM_LOGIC_CACHE_KEY_SIZE 64

/*=============================================================================
 * INTERNAL STRUCTURES
 *============================================================================*/

/**
 * @brief Cached evaluation entry
 */
typedef struct {
    uint32_t rule_id;
    uint64_t state_hash;
    swarm_logic_result_t result;
    uint64_t timestamp_us;
    bool valid;
} cache_entry_t;

/**
 * @brief Logic-swarm bridge internal structure
 */
struct swarm_logic_bridge {
    // Configuration
    swarm_logic_bridge_config_t config;

    // Rules
    swarm_logic_rule_t* rules;
    uint32_t rule_count;
    uint32_t rule_capacity;

    // Cache
    cache_entry_t* cache;
    uint32_t cache_size;
    uint32_t cache_next_index;

    // Neural logic network
    neural_logic_network_t logic_network;

    // Bio-async integration
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    // Statistics
    swarm_logic_bridge_stats_t stats;

    // Thread safety
    nimcp_platform_mutex_t* mutex;

    // Security
    bool security_registered;
};

/*=============================================================================
 * INTERNAL HELPERS
 *============================================================================*/

/**
 * @brief Calculate hash of agent states
 *
 * WHAT: Computes hash of agent states for cache lookup
 * WHY:  Enables fast cache hit detection
 * HOW:  Uses FNV-1a hash algorithm
 */
static uint64_t calculate_state_hash(const swarm_agent_state_t* states, uint32_t num_agents) {
    if (!states || num_agents == 0) {
        return 0;
    }

    // FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;

    for (uint32_t i = 0; i < num_agents; i++) {
        // Hash agent_id
        hash ^= states[i].agent_id;
        hash *= 1099511628211ULL;

        // Hash belief_value (convert to bits)
        uint32_t belief_bits;
        memcpy(&belief_bits, &states[i].belief_value, sizeof(uint32_t));
        hash ^= belief_bits;
        hash *= 1099511628211ULL;
    }

    return hash;
}

/**
 * @brief Find rule by ID
 *
 * WHAT: Finds rule in rules array
 * WHY:  Enables rule lookup
 * HOW:  Linear search through rules
 */
static swarm_logic_rule_t* find_rule(swarm_logic_bridge_t* bridge, uint32_t rule_id) {
    if (!bridge || !bridge->rules) {
        return NULL;
    }

    for (uint32_t i = 0; i < bridge->rule_count; i++) {
        if (bridge->rules[i].rule_id == rule_id) {
            return &bridge->rules[i];
        }
    }

    return NULL;
}

/**
 * @brief Check cache for evaluation result
 *
 * WHAT: Checks if evaluation result is cached
 * WHY:  Avoids redundant evaluations
 * HOW:  Compares rule_id and state_hash
 */
static bool check_cache(swarm_logic_bridge_t* bridge, uint32_t rule_id,
                       uint64_t state_hash, swarm_logic_result_t* result) {
    if (!bridge || !bridge->cache || !result) {
        return false;
    }

    for (uint32_t i = 0; i < bridge->cache_size; i++) {
        if (bridge->cache[i].valid &&
            bridge->cache[i].rule_id == rule_id &&
            bridge->cache[i].state_hash == state_hash) {

            // Check if cache entry is still fresh (< 1 second old)
            uint64_t now_us = nimcp_time_get_us();
            if (now_us - bridge->cache[i].timestamp_us < 1000000) {
                *result = bridge->cache[i].result;
                bridge->stats.cache_hits++;
                return true;
            }
        }
    }

    bridge->stats.cache_misses++;
    return false;
}

/**
 * @brief Add result to cache
 *
 * WHAT: Adds evaluation result to cache
 * WHY:  Speeds up repeated evaluations
 * HOW:  Ring buffer replacement
 */
static void add_to_cache(swarm_logic_bridge_t* bridge, uint32_t rule_id,
                        uint64_t state_hash, const swarm_logic_result_t* result) {
    if (!bridge || !bridge->cache || !result) {
        return;
    }

    uint32_t idx = bridge->cache_next_index;
    bridge->cache[idx].rule_id = rule_id;
    bridge->cache[idx].state_hash = state_hash;
    bridge->cache[idx].result = *result;
    bridge->cache[idx].timestamp_us = nimcp_time_get_us();
    bridge->cache[idx].valid = true;

    bridge->cache_next_index = (bridge->cache_next_index + 1) % bridge->cache_size;
}

/**
 * @brief Evaluate logic gate on agent states
 *
 * WHAT: Evaluates logic gate given agent belief values
 * WHY:  Core logic evaluation function
 * HOW:  Applies gate type to input values
 */
static float evaluate_logic_gate(logic_gate_type_t gate_type,
                                 const float* inputs,
                                 uint32_t num_inputs,
                                 float threshold) {
    if (!inputs || num_inputs == 0) {
        return 0.0f;
    }

    switch (gate_type) {
        case LOGIC_GATE_AND: {
            // AND: All inputs above threshold
            float min_value = inputs[0];
            for (uint32_t i = 1; i < num_inputs; i++) {
                if (inputs[i] < min_value) {
                    min_value = inputs[i];
                }
            }
            return min_value;
        }

        case LOGIC_GATE_OR: {
            // OR: Any input above threshold
            float max_value = inputs[0];
            for (uint32_t i = 1; i < num_inputs; i++) {
                if (inputs[i] > max_value) {
                    max_value = inputs[i];
                }
            }
            return max_value;
        }

        case LOGIC_GATE_NOT: {
            // NOT: Invert first input
            return 1.0f - inputs[0];
        }

        case LOGIC_GATE_XOR: {
            // XOR: Exactly one input above threshold
            uint32_t count_high = 0;
            float sum = 0.0f;
            for (uint32_t i = 0; i < num_inputs; i++) {
                if (inputs[i] >= threshold) {
                    count_high++;
                }
                sum += inputs[i];
            }

            // True if odd number of high inputs
            return (count_high % 2 == 1) ? sum / num_inputs : 0.0f;
        }

        case LOGIC_GATE_IMPLIES: {
            // IMPLIES: If A then B (¬A ∨ B)
            if (num_inputs < 2) {
                return 0.0f;
            }
            float a = inputs[0];
            float b = inputs[1];
            // (1 - A) OR B = max(1 - A, B)
            return fmaxf(1.0f - a, b);
        }

        default:
            return 0.0f;
    }
}

/**
 * @brief Bio-async message handler
 *
 * WHAT: Handles incoming bio-async messages
 * WHY:  Enables inter-module communication
 * HOW:  Processes message based on type
 */
static nimcp_error_t logic_bridge_message_handler(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data
) {
    if (!msg || msg_size < sizeof(bio_message_header_t) || !user_data) {
        return NIMCP_INVALID_PARAM;
    }

    swarm_logic_bridge_t* bridge = (swarm_logic_bridge_t*)user_data;
    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    (void)response_promise; // Not used yet

    LOG_DEBUG("Logic bridge received message type 0x%04X", header->type);

    // Handle different message types
    switch (header->type) {
        case BIO_MSG_LOGIC_GATE_EVALUATE:
            // Future: Handle logic gate evaluation request
            break;

        case BIO_MSG_SWARM_QUORUM_VOTE:
            // Future: Handle swarm vote
            break;

        default:
            LOG_DEBUG("Unhandled message type: 0x%04X", header->type);
            break;
    }

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *============================================================================*/

void swarm_logic_bridge_get_default_config(swarm_logic_bridge_config_t* config) {
    if (!config) {
        return;
    }

    memset(config, 0, sizeof(swarm_logic_bridge_config_t));

    config->max_rules = SWARM_LOGIC_DEFAULT_MAX_RULES;
    config->rule_cache_size = SWARM_LOGIC_DEFAULT_CACHE_SIZE;
    config->enable_bio_async = true;
    config->inference_threshold = SWARM_LOGIC_DEFAULT_INFERENCE_THRESHOLD;
    config->max_agents = SWARM_LOGIC_DEFAULT_MAX_AGENTS;
    config->timeout_ms = SWARM_LOGIC_DEFAULT_TIMEOUT_MS;
}

swarm_logic_bridge_t* swarm_logic_bridge_create(const swarm_logic_bridge_config_t* config) {
    LOG_INFO("Creating logic-swarm bridge");

    // Allocate bridge
    swarm_logic_bridge_t* bridge = (swarm_logic_bridge_t*)nimcp_calloc(1, sizeof(swarm_logic_bridge_t));
    if (!bridge) {
        LOG_ERROR("Failed to allocate logic bridge");
        return NULL;
    }

    // Set configuration
    if (config) {
        memcpy(&bridge->config, config, sizeof(swarm_logic_bridge_config_t));
    } else {
        swarm_logic_bridge_get_default_config(&bridge->config);
    }

    // Allocate rules array
    bridge->rule_capacity = bridge->config.max_rules;
    bridge->rules = (swarm_logic_rule_t*)nimcp_calloc(bridge->rule_capacity,
                                                       sizeof(swarm_logic_rule_t));
    if (!bridge->rules) {
        LOG_ERROR("Failed to allocate rules array");
        nimcp_free(bridge);
        return NULL;
    }

    // Allocate cache
    bridge->cache_size = bridge->config.rule_cache_size;
    bridge->cache = (cache_entry_t*)nimcp_calloc(bridge->cache_size,
                                                  sizeof(cache_entry_t));
    if (!bridge->cache) {
        LOG_ERROR("Failed to allocate cache");
        nimcp_free(bridge->rules);
        nimcp_free(bridge);
        return NULL;
    }

    // Create mutex
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        LOG_ERROR("Failed to create mutex");
        nimcp_free(bridge->cache);
        nimcp_free(bridge->rules);
        nimcp_free(bridge);
        return NULL;
    }

    // Create neural logic network
    neural_logic_config_t logic_config;
    memset(&logic_config, 0, sizeof(neural_logic_config_t));
    logic_config.max_logic_neurons = bridge->config.max_rules * 2;
    logic_config.max_variables = bridge->config.max_agents;
    logic_config.variable_pattern_dim = 16;
    logic_config.use_gpu = false; // CPU only for now
    logic_config.enable_bio_async = bridge->config.enable_bio_async;
    logic_config.timestep_us = 100.0f;
    logic_config.integration_window_ms = 10.0f;

    bridge->logic_network = neural_logic_create(&logic_config);
    if (!bridge->logic_network) {
        LOG_ERROR("Failed to create neural logic network");
        nimcp_platform_mutex_destroy(bridge->mutex);
        nimcp_free(bridge->cache);
        nimcp_free(bridge->rules);
        nimcp_free(bridge);
        return NULL;
    }

    // Register with bio-async if enabled
    if (bridge->config.enable_bio_async) {
        bio_module_info_t module_info;
        memset(&module_info, 0, sizeof(bio_module_info_t));
        module_info.module_id = BIO_MODULE_UNKNOWN; // Will be assigned
        module_info.module_name = "swarm_logic_bridge";
        module_info.inbox_capacity = 0; // Use default
        module_info.user_data = bridge;

        bridge->bio_ctx = bio_router_register_module(&module_info);
        if (bridge->bio_ctx) {
            // Register message handlers
            bio_router_register_handler(bridge->bio_ctx,
                                       BIO_MSG_LOGIC_GATE_EVALUATE,
                                       logic_bridge_message_handler);
            bio_router_register_handler(bridge->bio_ctx,
                                       BIO_MSG_SWARM_QUORUM_VOTE,
                                       logic_bridge_message_handler);

            bridge->bio_async_enabled = true;
            LOG_INFO("Logic bridge registered with bio-async router");
        } else {
            LOG_WARN("Failed to register with bio-async router");
        }
    }

    // Security integration (optional - can be added later with proper context)
    bridge->security_registered = false;

    // Initialize statistics
    memset(&bridge->stats, 0, sizeof(swarm_logic_bridge_stats_t));

    LOG_INFO("Logic-swarm bridge created successfully");
    return bridge;
}

void swarm_logic_bridge_destroy(swarm_logic_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    LOG_INFO("Destroying logic-swarm bridge");

    // Unregister from bio-async
    if (bridge->bio_async_enabled && bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
    }

    // Destroy neural logic network
    if (bridge->logic_network) {
        neural_logic_destroy(bridge->logic_network);
    }

    // Free rules and their input arrays
    if (bridge->rules) {
        for (uint32_t i = 0; i < bridge->rule_count; i++) {
            if (bridge->rules[i].input_agent_ids) {
                nimcp_free(bridge->rules[i].input_agent_ids);
            }
        }
        nimcp_free(bridge->rules);
    }

    // Free cache
    if (bridge->cache) {
        nimcp_free(bridge->cache);
    }

    // Destroy mutex
    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    LOG_INFO("Logic-swarm bridge destroyed");
}

/*=============================================================================
 * RULE MANAGEMENT
 *============================================================================*/

nimcp_error_t swarm_logic_bridge_add_rule(swarm_logic_bridge_t* bridge,
                                           const swarm_logic_rule_t* rule) {
    if (!bridge || !rule) {
        LOG_ERROR("Invalid parameters for add_rule");
        return NIMCP_INVALID_PARAM;
    }

    if (!rule->input_agent_ids || rule->num_inputs == 0) {
        LOG_ERROR("Rule must have at least one input");
        return NIMCP_INVALID_PARAM;
    }

    if (rule->gate_type >= LOGIC_GATE_COUNT) {
        LOG_ERROR("Invalid gate type: %d", rule->gate_type);
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    // Check capacity
    if (bridge->rule_count >= bridge->rule_capacity) {
        LOG_ERROR("Rule capacity exceeded: %u/%u", bridge->rule_count, bridge->rule_capacity);
        nimcp_platform_mutex_unlock(bridge->mutex);
        return NIMCP_NO_MEMORY;
    }

    // Check for duplicate rule ID
    if (find_rule(bridge, rule->rule_id) != NULL) {
        LOG_ERROR("Rule ID %u already exists", rule->rule_id);
        nimcp_platform_mutex_unlock(bridge->mutex);
        return NIMCP_INVALID_PARAM;
    }

    // Copy rule
    swarm_logic_rule_t* new_rule = &bridge->rules[bridge->rule_count];
    memcpy(new_rule, rule, sizeof(swarm_logic_rule_t));

    // Allocate and copy input agent IDs
    new_rule->input_agent_ids = (uint32_t*)nimcp_malloc(rule->num_inputs * sizeof(uint32_t));
    if (!new_rule->input_agent_ids) {
        LOG_ERROR("Failed to allocate input agent IDs");
        nimcp_platform_mutex_unlock(bridge->mutex);
        return NIMCP_NO_MEMORY;
    }

    memcpy(new_rule->input_agent_ids, rule->input_agent_ids,
           rule->num_inputs * sizeof(uint32_t));

    bridge->rule_count++;
    bridge->stats.active_rules = bridge->rule_count;

    nimcp_platform_mutex_unlock(bridge->mutex);

    LOG_INFO("Added rule %u: gate=%d, inputs=%u", rule->rule_id,
             rule->gate_type, rule->num_inputs);

    return NIMCP_SUCCESS;
}

nimcp_error_t swarm_logic_bridge_remove_rule(swarm_logic_bridge_t* bridge,
                                              uint32_t rule_id) {
    if (!bridge) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    // Find rule
    int index = -1;
    for (uint32_t i = 0; i < bridge->rule_count; i++) {
        if (bridge->rules[i].rule_id == rule_id) {
            index = (int)i;
            break;
        }
    }

    if (index < 0) {
        nimcp_platform_mutex_unlock(bridge->mutex);
        LOG_ERROR("Rule %u not found", rule_id);
        return NIMCP_NOT_FOUND;
    }

    // Free input array
    if (bridge->rules[index].input_agent_ids) {
        nimcp_free(bridge->rules[index].input_agent_ids);
    }

    // Shift remaining rules
    if ((uint32_t)index < bridge->rule_count - 1) {
        memmove(&bridge->rules[index], &bridge->rules[index + 1],
                (bridge->rule_count - index - 1) * sizeof(swarm_logic_rule_t));
    }

    bridge->rule_count--;
    bridge->stats.active_rules = bridge->rule_count;

    nimcp_platform_mutex_unlock(bridge->mutex);

    LOG_INFO("Removed rule %u", rule_id);
    return NIMCP_SUCCESS;
}

const swarm_logic_rule_t* swarm_logic_bridge_get_rule(swarm_logic_bridge_t* bridge,
                                                       uint32_t rule_id) {
    if (!bridge) {
        return NULL;
    }

    return find_rule(bridge, rule_id);
}

uint32_t swarm_logic_bridge_get_all_rules(swarm_logic_bridge_t* bridge,
                                           const swarm_logic_rule_t** rules,
                                           uint32_t max_rules) {
    if (!bridge || !rules || max_rules == 0) {
        return 0;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    uint32_t count = bridge->rule_count < max_rules ? bridge->rule_count : max_rules;
    for (uint32_t i = 0; i < count; i++) {
        rules[i] = &bridge->rules[i];
    }

    nimcp_platform_mutex_unlock(bridge->mutex);

    return count;
}

/*=============================================================================
 * EVALUATION FUNCTIONS
 *============================================================================*/

int swarm_logic_bridge_evaluate(swarm_logic_bridge_t* bridge,
                                 const swarm_agent_state_t* agent_states,
                                 uint32_t num_agents,
                                 swarm_logic_result_t* results,
                                 uint32_t max_results) {
    if (!bridge || !agent_states || num_agents == 0 || !results || max_results == 0) {
        LOG_ERROR("Invalid parameters for evaluate");
        return -1;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    uint64_t state_hash = calculate_state_hash(agent_states, num_agents);
    uint32_t result_count = 0;

    // Evaluate each rule
    for (uint32_t i = 0; i < bridge->rule_count && result_count < max_results; i++) {
        swarm_logic_rule_t* rule = &bridge->rules[i];

        // Check cache first
        if (check_cache(bridge, rule->rule_id, state_hash, &results[result_count])) {
            result_count++;
            continue;
        }

        // Collect input values from agent states
        float* input_values = (float*)nimcp_malloc(rule->num_inputs * sizeof(float));
        if (!input_values) {
            continue;
        }

        uint32_t valid_inputs = 0;
        for (uint32_t j = 0; j < rule->num_inputs; j++) {
            // Find agent with matching ID
            for (uint32_t k = 0; k < num_agents; k++) {
                if (agent_states[k].agent_id == rule->input_agent_ids[j] &&
                    agent_states[k].is_active) {
                    input_values[valid_inputs++] = agent_states[k].belief_value;
                    break;
                }
            }
        }

        if (valid_inputs == 0) {
            nimcp_free(input_values);
            continue;
        }

        // Evaluate logic gate
        uint64_t start_time = nimcp_time_get_us();
        float gate_output = evaluate_logic_gate(rule->gate_type, input_values,
                                               valid_inputs, rule->threshold);
        uint64_t eval_time = nimcp_time_get_us() - start_time;

        nimcp_free(input_values);

        // Create result
        swarm_logic_result_t* result = &results[result_count];
        memset(result, 0, sizeof(swarm_logic_result_t));
        result->rule_id = rule->rule_id;
        result->result = (gate_output >= rule->threshold);
        result->confidence = gate_output * rule->confidence_weight;
        result->num_inputs_used = valid_inputs;
        result->evaluation_time_us = eval_time;

        snprintf(result->explanation, sizeof(result->explanation),
                "Rule %u: gate=%d, inputs=%u/%u, output=%.3f",
                rule->rule_id, rule->gate_type, valid_inputs,
                rule->num_inputs, gate_output);

        // Add to cache
        add_to_cache(bridge, rule->rule_id, state_hash, result);

        result_count++;
        bridge->stats.total_evaluations++;
        bridge->stats.successful_evaluations++;
    }

    // Update statistics
    if (result_count > 0) {
        uint64_t total_time = 0;
        for (uint32_t i = 0; i < result_count; i++) {
            total_time += results[i].evaluation_time_us;
        }
        bridge->stats.avg_evaluation_time_us = (float)total_time / result_count;
    }

    nimcp_platform_mutex_unlock(bridge->mutex);

    LOG_DEBUG("Evaluated %u rules with %u agents", result_count, num_agents);
    return (int)result_count;
}

nimcp_error_t swarm_logic_bridge_evaluate_rule(swarm_logic_bridge_t* bridge,
                                                uint32_t rule_id,
                                                const swarm_agent_state_t* agent_states,
                                                uint32_t num_agents,
                                                swarm_logic_result_t* result) {
    if (!bridge || !agent_states || num_agents == 0 || !result) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    // Find rule
    swarm_logic_rule_t* rule = find_rule(bridge, rule_id);
    if (!rule) {
        nimcp_platform_mutex_unlock(bridge->mutex);
        return NIMCP_NOT_FOUND;
    }

    // Check cache
    uint64_t state_hash = calculate_state_hash(agent_states, num_agents);
    if (check_cache(bridge, rule_id, state_hash, result)) {
        nimcp_platform_mutex_unlock(bridge->mutex);
        return NIMCP_SUCCESS;
    }

    // Collect input values
    float* input_values = (float*)nimcp_malloc(rule->num_inputs * sizeof(float));
    if (!input_values) {
        nimcp_platform_mutex_unlock(bridge->mutex);
        return NIMCP_NO_MEMORY;
    }

    uint32_t valid_inputs = 0;
    for (uint32_t j = 0; j < rule->num_inputs; j++) {
        for (uint32_t k = 0; k < num_agents; k++) {
            if (agent_states[k].agent_id == rule->input_agent_ids[j] &&
                agent_states[k].is_active) {
                input_values[valid_inputs++] = agent_states[k].belief_value;
                break;
            }
        }
    }

    if (valid_inputs == 0) {
        nimcp_free(input_values);
        nimcp_platform_mutex_unlock(bridge->mutex);
        return NIMCP_NOT_FOUND;
    }

    // Evaluate
    uint64_t start_time = nimcp_time_get_us();
    float gate_output = evaluate_logic_gate(rule->gate_type, input_values,
                                           valid_inputs, rule->threshold);
    uint64_t eval_time = nimcp_time_get_us() - start_time;

    nimcp_free(input_values);

    // Fill result
    memset(result, 0, sizeof(swarm_logic_result_t));
    result->rule_id = rule_id;
    result->result = (gate_output >= rule->threshold);
    result->confidence = gate_output * rule->confidence_weight;
    result->num_inputs_used = valid_inputs;
    result->evaluation_time_us = eval_time;

    snprintf(result->explanation, sizeof(result->explanation),
            "Rule %u: gate=%d, inputs=%u/%u, output=%.3f",
            rule_id, rule->gate_type, valid_inputs, rule->num_inputs, gate_output);

    // Add to cache
    add_to_cache(bridge, rule_id, state_hash, result);

    bridge->stats.total_evaluations++;
    bridge->stats.successful_evaluations++;

    nimcp_platform_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * CONSENSUS VALIDATION
 *============================================================================*/

nimcp_error_t swarm_logic_bridge_validate_consensus(swarm_logic_bridge_t* bridge,
                                                     const float* votes,
                                                     uint32_t num_votes,
                                                     logic_gate_type_t consensus_type,
                                                     swarm_logic_result_t* result) {
    if (!bridge || !votes || num_votes == 0 || !result) {
        return NIMCP_INVALID_PARAM;
    }

    if (consensus_type != LOGIC_GATE_AND && consensus_type != LOGIC_GATE_OR) {
        LOG_ERROR("Invalid consensus type: %d (must be AND or OR)", consensus_type);
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    uint64_t start_time = nimcp_time_get_us();
    float gate_output = evaluate_logic_gate(consensus_type, votes, num_votes, 0.5f);
    uint64_t eval_time = nimcp_time_get_us() - start_time;

    memset(result, 0, sizeof(swarm_logic_result_t));
    result->rule_id = 0; // Consensus has no rule ID
    result->result = (gate_output >= 0.5f);
    result->confidence = gate_output;
    result->num_inputs_used = num_votes;
    result->evaluation_time_us = eval_time;

    const char* consensus_str = (consensus_type == LOGIC_GATE_AND) ? "unanimous" : "majority";
    snprintf(result->explanation, sizeof(result->explanation),
            "Consensus (%s): %u votes, result=%.3f",
            consensus_str, num_votes, gate_output);

    bridge->stats.total_evaluations++;
    bridge->stats.successful_evaluations++;

    nimcp_platform_mutex_unlock(bridge->mutex);

    LOG_INFO("Consensus validation: %s with %u votes = %s",
             consensus_str, num_votes, result->result ? "TRUE" : "FALSE");

    return NIMCP_SUCCESS;
}

int swarm_logic_bridge_detect_contradiction(swarm_logic_bridge_t* bridge,
                                            const swarm_agent_state_t* beliefs,
                                            uint32_t num_beliefs,
                                            uint32_t (*contradictions)[2],
                                            uint32_t max_contradictions) {
    if (!bridge || !beliefs || num_beliefs < 2 || !contradictions || max_contradictions == 0) {
        return -1;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    uint32_t contradiction_count = 0;

    // Check all pairs of beliefs
    for (uint32_t i = 0; i < num_beliefs && contradiction_count < max_contradictions; i++) {
        if (!beliefs[i].is_active) {
            continue;
        }

        for (uint32_t j = i + 1; j < num_beliefs && contradiction_count < max_contradictions; j++) {
            if (!beliefs[j].is_active) {
                continue;
            }

            // XOR: contradiction if beliefs are opposite
            float inputs[2] = { beliefs[i].belief_value, beliefs[j].belief_value };
            float xor_result = evaluate_logic_gate(LOGIC_GATE_XOR, inputs, 2, 0.5f);

            // High XOR result indicates contradiction
            if (xor_result >= bridge->config.inference_threshold) {
                contradictions[contradiction_count][0] = beliefs[i].agent_id;
                contradictions[contradiction_count][1] = beliefs[j].agent_id;
                contradiction_count++;

                LOG_DEBUG("Contradiction detected: agent %u (%.3f) vs agent %u (%.3f)",
                         beliefs[i].agent_id, beliefs[i].belief_value,
                         beliefs[j].agent_id, beliefs[j].belief_value);
            }
        }
    }

    nimcp_platform_mutex_unlock(bridge->mutex);

    return (int)contradiction_count;
}

nimcp_error_t swarm_logic_bridge_validate_implication(swarm_logic_bridge_t* bridge,
                                                       uint32_t antecedent_agent,
                                                       uint32_t consequent_agent,
                                                       const swarm_agent_state_t* agent_states,
                                                       uint32_t num_agents,
                                                       swarm_logic_result_t* result) {
    if (!bridge || !agent_states || num_agents == 0 || !result) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    // Find agents
    float antecedent_value = 0.0f;
    float consequent_value = 0.0f;
    bool found_antecedent = false;
    bool found_consequent = false;

    for (uint32_t i = 0; i < num_agents; i++) {
        if (agent_states[i].agent_id == antecedent_agent && agent_states[i].is_active) {
            antecedent_value = agent_states[i].belief_value;
            found_antecedent = true;
        }
        if (agent_states[i].agent_id == consequent_agent && agent_states[i].is_active) {
            consequent_value = agent_states[i].belief_value;
            found_consequent = true;
        }
    }

    if (!found_antecedent || !found_consequent) {
        nimcp_platform_mutex_unlock(bridge->mutex);
        return NIMCP_NOT_FOUND;
    }

    // Evaluate IMPLIES gate
    float inputs[2] = { antecedent_value, consequent_value };
    uint64_t start_time = nimcp_time_get_us();
    float gate_output = evaluate_logic_gate(LOGIC_GATE_IMPLIES, inputs, 2, 0.5f);
    uint64_t eval_time = nimcp_time_get_us() - start_time;

    memset(result, 0, sizeof(swarm_logic_result_t));
    result->rule_id = 0;
    result->result = (gate_output >= 0.5f);
    result->confidence = gate_output;
    result->num_inputs_used = 2;
    result->evaluation_time_us = eval_time;

    snprintf(result->explanation, sizeof(result->explanation),
            "Implication: agent %u (%.3f) => agent %u (%.3f) = %.3f",
            antecedent_agent, antecedent_value,
            consequent_agent, consequent_value, gate_output);

    bridge->stats.total_evaluations++;
    bridge->stats.successful_evaluations++;

    nimcp_platform_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * BIO-ASYNC INTEGRATION
 *============================================================================*/

int swarm_logic_bridge_process_inbox(swarm_logic_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) {
        return 0;
    }

    uint32_t processed = bio_router_process_inbox(bridge->bio_ctx, 10);
    return (int)processed;
}

nimcp_error_t swarm_logic_bridge_send_result(swarm_logic_bridge_t* bridge,
                                              bio_module_id_t target_module,
                                              const swarm_logic_result_t* result) {
    if (!bridge || !result || !bridge->bio_async_enabled) {
        return NIMCP_INVALID_PARAM;
    }

    // Create logic result message
    bio_msg_logic_gate_result_t msg;
    memset(&msg, 0, sizeof(msg));

    bio_msg_init_header(&msg.header, BIO_MSG_LOGIC_GATE_RESULT,
                       bio_module_context_get_id(bridge->bio_ctx),
                       target_module, sizeof(msg));

    msg.gate_id = result->rule_id;
    msg.gate_type = 0; // Unknown gate type in result
    msg.output = result->result ? 1.0f : 0.0f;
    msg.spiked = result->result;
    msg.spike_time_us = result->evaluation_time_us;
    msg.threshold_used = 0.5f;

    return bio_router_send(bridge->bio_ctx, &msg, sizeof(msg), 0);
}

nimcp_error_t swarm_logic_bridge_broadcast_consensus(swarm_logic_bridge_t* bridge,
                                                      const swarm_logic_result_t* result) {
    if (!bridge || !result || !bridge->bio_async_enabled) {
        return NIMCP_INVALID_PARAM;
    }

    // Create consensus message
    bio_msg_logic_gate_result_t msg;
    memset(&msg, 0, sizeof(msg));

    bio_msg_init_header(&msg.header, BIO_MSG_LOGIC_GATE_RESULT,
                       bio_module_context_get_id(bridge->bio_ctx),
                       BIO_MODULE_ALL, sizeof(msg));

    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.gate_id = result->rule_id;
    msg.output = result->confidence;
    msg.spiked = result->result;

    return bio_router_broadcast(bridge->bio_ctx, &msg, sizeof(msg));
}

/*=============================================================================
 * STATISTICS AND DEBUGGING
 *============================================================================*/

nimcp_error_t swarm_logic_bridge_get_stats(swarm_logic_bridge_t* bridge,
                                            swarm_logic_bridge_stats_t* stats) {
    if (!bridge || !stats) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(bridge->mutex);
    memcpy(stats, &bridge->stats, sizeof(swarm_logic_bridge_stats_t));
    nimcp_platform_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

void swarm_logic_bridge_reset_stats(swarm_logic_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    bridge->stats.total_evaluations = 0;
    bridge->stats.successful_evaluations = 0;
    bridge->stats.failed_evaluations = 0;
    bridge->stats.cache_hits = 0;
    bridge->stats.cache_misses = 0;
    bridge->stats.avg_evaluation_time_us = 0.0f;

    // Keep active counts

    nimcp_platform_mutex_unlock(bridge->mutex);
}

void swarm_logic_bridge_clear_cache(swarm_logic_bridge_t* bridge) {
    if (!bridge || !bridge->cache) {
        return;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    for (uint32_t i = 0; i < bridge->cache_size; i++) {
        bridge->cache[i].valid = false;
    }

    bridge->cache_next_index = 0;

    nimcp_platform_mutex_unlock(bridge->mutex);

    LOG_DEBUG("Logic bridge cache cleared");
}
