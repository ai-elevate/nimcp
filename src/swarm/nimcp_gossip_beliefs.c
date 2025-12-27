/**
 * @file nimcp_gossip_beliefs.c
 * @brief Implementation of Gossip-Based Belief Propagation System
 *
 * WHAT: Belief propagation through probabilistic gossip protocol
 * WHY:  Enable distributed consensus and information diffusion
 * HOW:  Probabilistic sharing with decay, credibility weighting, contradiction detection
 *
 * @version 1.0
 * @date 2025
 */

#include "swarm/nimcp_gossip_beliefs.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/validation/nimcp_common.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/containers/nimcp_hash_table.h"

#define LOG_MODULE "gossip_beliefs"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Constants and Configuration
 * ============================================================================ */

#define BELIEF_ID_BUFFER_SIZE 64
#define AGENT_ID_BUFFER_SIZE 64
#define MIN_CERTAINTY 0.0f
#define MAX_CERTAINTY 1.0f
#define MIN_CREDIBILITY 0.0f
#define MAX_CREDIBILITY 1.0f
#define DEFAULT_GOSSIP_PROBABILITY 0.2f
#define DEFAULT_MAX_GOSSIP_TARGETS 5
#define DEFAULT_BELIEF_DECAY 0.001f
#define DEFAULT_CREDIBILITY_WEIGHT 0.5f
#define CONSENSUS_THRESHOLD 0.5f  /**< 50% of agents */
#define CONTRADICTION_THRESHOLD -0.7f  /**< Cosine similarity threshold */

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Agent information in the gossip network
 */
typedef struct {
    uint32_t agent_id;
    float credibility;
    hash_table_t* beliefs;  /**< belief_id -> belief_t* */
    uint32_t belief_count;
} agent_info_t;

/**
 * @brief Main gossip beliefs system structure
 */
struct gossip_beliefs {
    /* Configuration */
    gossip_beliefs_config_t config;

    /* Storage */
    hash_table_t* agents;           /**< agent_id -> agent_info_t* */
    hash_table_t* all_beliefs;      /**< belief_id -> belief_t* */

    /* Statistics */
    uint32_t total_beliefs;
    uint32_t total_agents;
    uint32_t total_gossips;
    float avg_certainty;

    /* Bio-async integration */
    void* bio_ctx;
    bio_module_context_t bio_module;
    bool bio_async_enabled;

    /* State */
    uint32_t next_belief_id;
    uint64_t last_gossip_ms;
    uint64_t last_decay_ms;

    /* Synchronization */
    nimcp_platform_mutex_t* mutex;
    bool is_initialized;
};

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

static float vector_cosine_similarity(const float* v1, const float* v2, uint32_t size);
static void generate_belief_id(char* buffer, size_t size, uint32_t id);
static void generate_agent_id_str(char* buffer, size_t size, uint32_t id);
static agent_info_t* get_or_create_agent(gossip_beliefs_t* gb, uint32_t agent_id);
static agent_info_t* get_agent(gossip_beliefs_t* gb, uint32_t agent_id);
static nimcp_result_t send_gossip_message(gossip_beliefs_t* gb,
    const belief_t* belief, uint32_t target_agent);
static nimcp_result_t handle_gossip_spread(gossip_beliefs_t* gb, const void* msg);
static void apply_decay_to_agent(agent_info_t* agent, float decay_factor);

/* ============================================================================
 * Core API Implementation
 * ============================================================================ */

gossip_beliefs_t* gossip_beliefs_create(const gossip_beliefs_config_t* config)
{
    /* WHAT: Create and initialize gossip beliefs system
     * WHY:  Required before any operations
     * HOW:  Allocate structure, create containers, set defaults
     */

    if (!config) {
        LOG_ERROR("NULL configuration provided");
        return NULL;
    }

    LOG_INFO("Creating gossip beliefs system (gossip_prob=%.2f, max_targets=%u)",
             config->gossip_probability, config->max_gossip_targets);

    gossip_beliefs_t* gb = (gossip_beliefs_t*)nimcp_malloc(sizeof(gossip_beliefs_t));
    if (!gb) {
        LOG_ERROR("Failed to allocate gossip beliefs system");
        return NULL;
    }

    memset(gb, 0, sizeof(gossip_beliefs_t));
    memcpy(&gb->config, config, sizeof(gossip_beliefs_config_t));

    /* Create hash tables */
    hash_table_config_t ht_config = {
        .initial_buckets = 256,
        .key_type = HASH_KEY_STRING,
        .hash_algorithm = HASH_ALG_FNV1A,
        .value_destructor = NULL,
        .case_insensitive = false,
        .thread_safe = false
    };

    gb->agents = hash_table_create(&ht_config);
    if (!gb->agents) {
        LOG_ERROR("Failed to create agents hash table");
        nimcp_free(gb);
        return NULL;
    }

    gb->all_beliefs = hash_table_create(&ht_config);
    if (!gb->all_beliefs) {
        LOG_ERROR("Failed to create beliefs hash table");
        hash_table_destroy(gb->agents);
        nimcp_free(gb);
        return NULL;
    }

    /* Initialize state */
    gb->next_belief_id = 1;
    gb->last_gossip_ms = nimcp_time_get_us() / 1000;
    gb->last_decay_ms = gb->last_gossip_ms;

    /* Create mutex */
    gb->mutex = nimcp_platform_mutex_create();
    if (!gb->mutex) {
        LOG_ERROR("Failed to create mutex");
        hash_table_destroy(gb->all_beliefs);
        hash_table_destroy(gb->agents);
        nimcp_free(gb);
        return NULL;
    }

    gb->is_initialized = false;

    LOG_INFO("Gossip beliefs system created successfully");
    return gb;
}

void gossip_beliefs_destroy(gossip_beliefs_t* gb)
{
    /* WHAT: Free all resources and cleanup
     * WHY:  Prevent memory leaks
     * HOW:  Free beliefs, agents, containers, mutex
     */

    if (!gb) {
        return;
    }

    LOG_INFO("Destroying gossip beliefs system");

    if (gb->mutex) {
        nimcp_platform_mutex_lock(gb->mutex);
    }

    /* Free hash tables (TODO: implement proper cleanup iteration) */
    if (gb->all_beliefs) {
        hash_table_destroy(gb->all_beliefs);
    }

    if (gb->agents) {
        hash_table_destroy(gb->agents);
    }

    if (gb->mutex) {
        nimcp_platform_mutex_unlock(gb->mutex);
        nimcp_platform_mutex_destroy(gb->mutex);
    }

    nimcp_free(gb);

    LOG_INFO("Gossip beliefs system destroyed");
}

nimcp_result_t gossip_beliefs_init(gossip_beliefs_t* gb, void* bio_ctx)
{
    /* WHAT: Initialize bio-async integration
     * WHY:  Enable belief sharing via bio-router
     * HOW:  Register module and message handlers
     */

    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    LOG_INFO("Initializing gossip beliefs system");

    nimcp_platform_mutex_lock(gb->mutex);

    if (gb->is_initialized) {
        LOG_WARN("Gossip beliefs system already initialized");
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_SUCCESS;
    }

    gb->bio_ctx = bio_ctx;
    gb->bio_async_enabled = (bio_ctx != NULL) && gb->config.enable_bio_async;

    if (gb->bio_async_enabled && bio_router_is_initialized()) {
        /* Register with bio-router */
        bio_module_info_t module_info = {
            .module_id = BIO_MODULE_SWARM_MEMORY,
            .module_name = "gossip_beliefs",
            .inbox_capacity = NIMCP_INBOX_CAPACITY_XLARGE,
            .user_data = gb
        };

        gb->bio_module = bio_router_register_module(&module_info);
        if (gb->bio_module) {
            LOG_INFO("Registered with bio-router");

            /* Register message handlers */
            bio_router_register_handler(gb->bio_module, BIO_MSG_GOSSIP_SPREAD,
                (bio_message_handler_t)handle_gossip_spread);
        } else {
            LOG_WARN("Failed to register with bio-router");
        }
    }

    gb->is_initialized = true;
    nimcp_platform_mutex_unlock(gb->mutex);

    LOG_INFO("Gossip beliefs system initialized");
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Belief Management Implementation
 * ============================================================================ */

int gossip_introduce_belief(gossip_beliefs_t* gb, uint32_t agent_id, const belief_t* belief)
{
    /* WHAT: Add belief to agent's belief set
     * WHY:  Inject information into gossip network
     * HOW:  Create belief copy, add to agent's hash table
     */

    if (!gb || !belief) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(gb->mutex);

    /* Get or create agent */
    agent_info_t* agent = get_or_create_agent(gb, agent_id);
    if (!agent) {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NO_MEMORY;
    }

    /* Create belief copy */
    belief_t* new_belief = (belief_t*)nimcp_malloc(sizeof(belief_t));
    if (!new_belief) {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NO_MEMORY;
    }

    memcpy(new_belief, belief, sizeof(belief_t));
    new_belief->belief_id = gb->next_belief_id++;
    new_belief->propagation_count = 0;
    new_belief->first_heard_ms = nimcp_time_get_us() / 1000;

    /* Copy belief vector */
    new_belief->belief_vector = (float*)nimcp_malloc(sizeof(float) * belief->vector_size);
    if (!new_belief->belief_vector) {
        nimcp_free(new_belief);
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NO_MEMORY;
    }
    memcpy(new_belief->belief_vector, belief->belief_vector,
           sizeof(float) * belief->vector_size);

    /* Add to agent's beliefs */
    char belief_id[BELIEF_ID_BUFFER_SIZE];
    generate_belief_id(belief_id, sizeof(belief_id), new_belief->belief_id);

    if (hash_table_insert_string(agent->beliefs, belief_id,
                                  &new_belief, sizeof(void*))) {
        agent->belief_count++;
        gb->total_beliefs++;

        /* Also add to global beliefs table */
        hash_table_insert_string(gb->all_beliefs, belief_id,
                                 &new_belief, sizeof(void*));

        /* Update average certainty */
        gb->avg_certainty = (gb->avg_certainty * (gb->total_beliefs - 1) +
                             new_belief->certainty) / gb->total_beliefs;

        nimcp_platform_mutex_unlock(gb->mutex);

        LOG_DEBUG("Introduced belief %u to agent %u (topic: %s, certainty: %.3f)",
                  new_belief->belief_id, agent_id, new_belief->topic,
                  new_belief->certainty);

        return NIMCP_SUCCESS;
    } else {
        nimcp_free(new_belief->belief_vector);
        nimcp_free(new_belief);
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_ERROR;
    }
}

int gossip_update_belief(gossip_beliefs_t* gb, uint32_t agent_id, uint32_t belief_id,
                          float new_certainty)
{
    /* WHAT: Update belief certainty
     * WHY:  Beliefs strengthen/weaken with evidence
     * HOW:  Lookup belief, update certainty field
     */

    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    /* Clamp certainty */
    new_certainty = fmaxf(MIN_CERTAINTY, fminf(MAX_CERTAINTY, new_certainty));

    nimcp_platform_mutex_lock(gb->mutex);

    agent_info_t* agent = get_agent(gb, agent_id);
    if (!agent) {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NOT_FOUND;
    }

    char belief_id_str[BELIEF_ID_BUFFER_SIZE];
    generate_belief_id(belief_id_str, sizeof(belief_id_str), belief_id);

    belief_t** belief_ptr = (belief_t**)hash_table_lookup_string(
        agent->beliefs, belief_id_str);

    if (!belief_ptr || !*belief_ptr) {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NOT_FOUND;
    }

    float old_certainty = (*belief_ptr)->certainty;
    (*belief_ptr)->certainty = new_certainty;

    /* Update average */
    gb->avg_certainty = gb->avg_certainty - (old_certainty / gb->total_beliefs) +
                        (new_certainty / gb->total_beliefs);

    nimcp_platform_mutex_unlock(gb->mutex);

    LOG_DEBUG("Updated belief %u certainty: %.3f -> %.3f",
              belief_id, old_certainty, new_certainty);

    return NIMCP_SUCCESS;
}

int gossip_remove_belief(gossip_beliefs_t* gb, uint32_t agent_id, uint32_t belief_id)
{
    /* WHAT: Remove belief from agent
     * WHY:  Forget or reject belief
     * HOW:  Remove from hash table, free memory
     */

    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(gb->mutex);

    agent_info_t* agent = get_agent(gb, agent_id);
    if (!agent) {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NOT_FOUND;
    }

    char belief_id_str[BELIEF_ID_BUFFER_SIZE];
    generate_belief_id(belief_id_str, sizeof(belief_id_str), belief_id);

    if (hash_table_remove_string(agent->beliefs, belief_id_str)) {
        agent->belief_count--;
        gb->total_beliefs--;

        nimcp_platform_mutex_unlock(gb->mutex);

        LOG_DEBUG("Removed belief %u from agent %u", belief_id, agent_id);
        return NIMCP_SUCCESS;
    } else {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NOT_FOUND;
    }
}

/* ============================================================================
 * Gossip Propagation Implementation
 * ============================================================================ */

int gossip_propagate_round(gossip_beliefs_t* gb, uint64_t current_time_ms)
{
    /* WHAT: Execute one gossip round
     * WHY:  Drive belief diffusion
     * HOW:  For each agent, probabilistically gossip beliefs
     */

    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(gb->mutex);

    gb->last_gossip_ms = current_time_ms;
    uint32_t gossip_count = 0;

    /* TODO: Iterate through agents and gossip beliefs */
    /* This would require hash table iteration support */

    nimcp_platform_mutex_unlock(gb->mutex);

    LOG_DEBUG("Gossip round complete: %u gossips performed", gossip_count);

    return NIMCP_SUCCESS;
}

int gossip_get_agent_beliefs(gossip_beliefs_t* gb, uint32_t agent_id,
                              belief_t** beliefs, uint32_t* count)
{
    /* WHAT: Get all beliefs held by agent
     * WHY:  Query agent's belief set
     * HOW:  Copy beliefs from agent's hash table
     */

    if (!gb || !beliefs || !count) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(gb->mutex);

    agent_info_t* agent = get_agent(gb, agent_id);
    if (!agent) {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NOT_FOUND;
    }

    *count = agent->belief_count;

    /* TODO: Implement belief array extraction from hash table */

    nimcp_platform_mutex_unlock(gb->mutex);

    return NIMCP_SUCCESS;
}

int gossip_apply_decay(gossip_beliefs_t* gb, uint64_t current_time_ms)
{
    /* WHAT: Apply decay to all beliefs
     * WHY:  Beliefs fade without reinforcement
     * HOW:  Reduce certainty exponentially
     */

    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(gb->mutex);

    uint64_t time_delta = current_time_ms - gb->last_decay_ms;
    float decay_factor = expf(-gb->config.belief_decay_rate * time_delta / 1000.0F);

    /* TODO: Iterate through all agents and apply decay */

    gb->last_decay_ms = current_time_ms;

    nimcp_platform_mutex_unlock(gb->mutex);

    LOG_DEBUG("Applied belief decay (factor=%.4f)", decay_factor);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Consensus and Analysis Implementation
 * ============================================================================ */

int gossip_get_consensus_beliefs(gossip_beliefs_t* gb, belief_t** consensus, uint32_t* count)
{
    /* WHAT: Get beliefs held by majority
     * WHY:  Identify swarm consensus
     * HOW:  Count belief holders, filter by threshold
     */

    if (!gb || !consensus || !count) {
        return NIMCP_INVALID_PARAM;
    }

    /* TODO: Implement consensus detection */
    *count = 0;

    LOG_DEBUG("Consensus query not yet fully implemented");

    return NIMCP_SUCCESS;
}

int gossip_detect_contradictions(gossip_beliefs_t* gb, uint32_t** contradiction_pairs,
                                  uint32_t* num_contradictions)
{
    /* WHAT: Find contradictory belief pairs
     * WHY:  Identify conflicts for resolution
     * HOW:  Compare belief vectors, detect opposition
     */

    if (!gb || !contradiction_pairs || !num_contradictions) {
        return NIMCP_INVALID_PARAM;
    }

    if (!gb->config.enable_contradiction_detection) {
        *num_contradictions = 0;
        return NIMCP_SUCCESS;
    }

    /* TODO: Implement contradiction detection with vector comparison */
    *num_contradictions = 0;

    LOG_DEBUG("Contradiction detection not yet fully implemented");

    return NIMCP_SUCCESS;
}

float gossip_calculate_entropy(gossip_beliefs_t* gb)
{
    /* WHAT: Calculate belief diversity
     * WHY:  Measure consensus vs disagreement
     * HOW:  Shannon entropy over belief distribution
     */

    if (!gb) {
        return 0.0F;
    }

    /* TODO: Implement entropy calculation */

    return 0.0F;
}

/* ============================================================================
 * Agent Management Implementation
 * ============================================================================ */

int gossip_register_agent(gossip_beliefs_t* gb, uint32_t agent_id, float credibility)
{
    /* WHAT: Register agent in gossip network
     * WHY:  Agents must be registered to participate
     * HOW:  Create agent_info structure
     */

    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    credibility = fmaxf(MIN_CREDIBILITY, fminf(MAX_CREDIBILITY, credibility));

    nimcp_platform_mutex_lock(gb->mutex);

    agent_info_t* agent = get_or_create_agent(gb, agent_id);
    if (!agent) {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NO_MEMORY;
    }

    agent->credibility = credibility;

    nimcp_platform_mutex_unlock(gb->mutex);

    LOG_INFO("Registered agent %u (credibility=%.3f)", agent_id, credibility);

    return NIMCP_SUCCESS;
}

int gossip_unregister_agent(gossip_beliefs_t* gb, uint32_t agent_id)
{
    /* WHAT: Remove agent from network
     * WHY:  Agent has left swarm
     * HOW:  Remove from agents hash table
     */

    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(gb->mutex);

    char agent_id_str[AGENT_ID_BUFFER_SIZE];
    generate_agent_id_str(agent_id_str, sizeof(agent_id_str), agent_id);

    if (hash_table_remove_string(gb->agents, agent_id_str)) {
        gb->total_agents--;
        nimcp_platform_mutex_unlock(gb->mutex);

        LOG_INFO("Unregistered agent %u", agent_id);
        return NIMCP_SUCCESS;
    } else {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NOT_FOUND;
    }
}

int gossip_update_credibility(gossip_beliefs_t* gb, uint32_t agent_id, float credibility)
{
    /* WHAT: Update agent credibility
     * WHY:  Credibility changes with reliability
     * HOW:  Update credibility field
     */

    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    credibility = fmaxf(MIN_CREDIBILITY, fminf(MAX_CREDIBILITY, credibility));

    nimcp_platform_mutex_lock(gb->mutex);

    agent_info_t* agent = get_agent(gb, agent_id);
    if (!agent) {
        nimcp_platform_mutex_unlock(gb->mutex);
        return NIMCP_NOT_FOUND;
    }

    agent->credibility = credibility;

    nimcp_platform_mutex_unlock(gb->mutex);

    LOG_DEBUG("Updated agent %u credibility to %.3f", agent_id, credibility);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Integration Implementation
 * ============================================================================ */

nimcp_result_t gossip_beliefs_process_message(gossip_beliefs_t* gb, const void* msg)
{
    /* WHAT: Handle incoming gossip message
     * WHY:  Receive gossiped beliefs
     * HOW:  Parse, deserialize, integrate
     */

    if (!gb || !msg) {
        return NIMCP_INVALID_PARAM;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    switch (header->type) {
        case BIO_MSG_GOSSIP_SPREAD:
            return handle_gossip_spread(gb, msg);

        default:
            LOG_DEBUG("Unhandled message type: 0x%x", header->type);
            break;
    }

    return NIMCP_SUCCESS;
}

uint32_t gossip_beliefs_process_inbox(gossip_beliefs_t* gb, uint32_t max_messages)
{
    /* WHAT: Process pending messages from inbox
     * WHY:  Receive beliefs from other agents
     * HOW:  Poll bio-router inbox
     */

    if (!gb || !gb->bio_async_enabled || !gb->bio_module) {
        return 0;
    }

    return bio_router_process_inbox(gb->bio_module, max_messages);
}

/* ============================================================================
 * Statistics Implementation
 * ============================================================================ */

int gossip_get_stats(gossip_beliefs_t* gb, uint32_t* total_beliefs,
                     uint32_t* total_agents, float* avg_certainty,
                     uint32_t* total_gossips)
{
    if (!gb) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(gb->mutex);

    if (total_beliefs) *total_beliefs = gb->total_beliefs;
    if (total_agents) *total_agents = gb->total_agents;
    if (avg_certainty) *avg_certainty = gb->avg_certainty;
    if (total_gossips) *total_gossips = gb->total_gossips;

    nimcp_platform_mutex_unlock(gb->mutex);

    return NIMCP_SUCCESS;
}

void gossip_beliefs_print_status(const gossip_beliefs_t* gb, bool verbose)
{
    if (!gb) {
        return;
    }

    LOG_INFO("=== Gossip Beliefs Status ===");
    LOG_INFO("Total beliefs: %u", gb->total_beliefs);
    LOG_INFO("Total agents: %u", gb->total_agents);
    LOG_INFO("Average certainty: %.3f", gb->avg_certainty);
    LOG_INFO("Total gossips: %u", gb->total_gossips);
    LOG_INFO("Bio-async enabled: %s", gb->bio_async_enabled ? "yes" : "no");

    if (verbose) {
        LOG_INFO("Gossip probability: %.3f", gb->config.gossip_probability);
        LOG_INFO("Max gossip targets: %u", gb->config.max_gossip_targets);
        LOG_INFO("Belief decay rate: %.4f", gb->config.belief_decay_rate);
        LOG_INFO("Credibility weight: %.3f", gb->config.credibility_weight);
        LOG_INFO("Contradiction detection: %s",
                 gb->config.enable_contradiction_detection ? "enabled" : "disabled");
    }
}

/* ============================================================================
 * Utility Functions Implementation
 * ============================================================================ */

belief_t* belief_create(const char* topic, const float* belief_vector,
                        uint32_t vector_size, float certainty,
                        uint32_t source_agent_id)
{
    belief_t* belief = (belief_t*)nimcp_malloc(sizeof(belief_t));
    if (!belief) {
        return NULL;
    }

    belief->belief_id = 0; /* Set by system */
    strncpy(belief->topic, topic, sizeof(belief->topic) - 1);
    belief->topic[sizeof(belief->topic) - 1] = '\0';
    belief->certainty = fmaxf(MIN_CERTAINTY, fminf(MAX_CERTAINTY, certainty));
    belief->vector_size = vector_size;
    belief->source_agent_id = source_agent_id;
    belief->propagation_count = 0;
    belief->first_heard_ms = nimcp_time_get_us() / 1000;

    belief->belief_vector = (float*)nimcp_malloc(sizeof(float) * vector_size);
    if (!belief->belief_vector) {
        nimcp_free(belief);
        return NULL;
    }

    memcpy(belief->belief_vector, belief_vector, sizeof(float) * vector_size);

    return belief;
}

void belief_destroy(belief_t* belief)
{
    if (!belief) {
        return;
    }

    if (belief->belief_vector) {
        nimcp_free(belief->belief_vector);
    }

    nimcp_free(belief);
}

float belief_similarity(const belief_t* belief1, const belief_t* belief2)
{
    /* WHAT: Calculate semantic similarity
     * WHY:  Detect redundant/contradictory beliefs
     * HOW:  Cosine similarity of vectors
     */

    if (!belief1 || !belief2) {
        return 0.0F;
    }

    if (belief1->vector_size != belief2->vector_size) {
        return 0.0F;
    }

    return vector_cosine_similarity(belief1->belief_vector,
                                    belief2->belief_vector,
                                    belief1->vector_size);
}

/* ============================================================================
 * Helper Functions Implementation
 * ============================================================================ */

static float vector_cosine_similarity(const float* v1, const float* v2, uint32_t size)
{
    float dot_product = 0.0F;
    float norm1 = 0.0F;
    float norm2 = 0.0F;

    for (uint32_t i = 0; i < size; i++) {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }

    norm1 = sqrtf(norm1);
    norm2 = sqrtf(norm2);

    if (norm1 < 1e-8F || norm2 < 1e-8F) {
        return 0.0F;
    }

    return dot_product / (norm1 * norm2);
}

static void generate_belief_id(char* buffer, size_t size, uint32_t id)
{
    snprintf(buffer, size, "BELIEF_%08X", id);
}

static void generate_agent_id_str(char* buffer, size_t size, uint32_t id)
{
    snprintf(buffer, size, "AGENT_%08X", id);
}

static agent_info_t* get_or_create_agent(gossip_beliefs_t* gb, uint32_t agent_id)
{
    char agent_id_str[AGENT_ID_BUFFER_SIZE];
    generate_agent_id_str(agent_id_str, sizeof(agent_id_str), agent_id);

    agent_info_t** agent_ptr = (agent_info_t**)hash_table_lookup_string(
        gb->agents, agent_id_str);

    if (agent_ptr && *agent_ptr) {
        return *agent_ptr;
    }

    /* Create new agent */
    agent_info_t* agent = (agent_info_t*)nimcp_malloc(sizeof(agent_info_t));
    if (!agent) {
        return NULL;
    }

    agent->agent_id = agent_id;
    agent->credibility = 0.5F; /* Default credibility */
    agent->belief_count = 0;

    /* Create beliefs hash table for this agent */
    hash_table_config_t ht_config = {
        .initial_buckets = 64,
        .key_type = HASH_KEY_STRING,
        .hash_algorithm = HASH_ALG_FNV1A,
        .value_destructor = NULL,
        .case_insensitive = false,
        .thread_safe = false
    };

    agent->beliefs = hash_table_create(&ht_config);
    if (!agent->beliefs) {
        nimcp_free(agent);
        return NULL;
    }

    /* Add to agents table */
    if (hash_table_insert_string(gb->agents, agent_id_str,
                                  &agent, sizeof(void*))) {
        gb->total_agents++;
        return agent;
    } else {
        hash_table_destroy(agent->beliefs);
        nimcp_free(agent);
        return NULL;
    }
}

static agent_info_t* get_agent(gossip_beliefs_t* gb, uint32_t agent_id)
{
    char agent_id_str[AGENT_ID_BUFFER_SIZE];
    generate_agent_id_str(agent_id_str, sizeof(agent_id_str), agent_id);

    agent_info_t** agent_ptr = (agent_info_t**)hash_table_lookup_string(
        gb->agents, agent_id_str);

    return (agent_ptr && *agent_ptr) ? *agent_ptr : NULL;
}

static nimcp_result_t send_gossip_message(gossip_beliefs_t* gb,
    const belief_t* belief, uint32_t target_agent)
{
    if (!gb->bio_async_enabled || !gb->bio_module) {
        return NIMCP_ERROR;
    }

    /* Build message (simplified) */
    size_t msg_size = sizeof(bio_message_header_t) + sizeof(uint32_t);
    uint8_t* buffer = (uint8_t*)nimcp_malloc(msg_size);
    if (!buffer) {
        return NIMCP_NO_MEMORY;
    }

    bio_message_header_t* header = (bio_message_header_t*)buffer;
    header->type = BIO_MSG_GOSSIP_SPREAD;
    header->source_module = BIO_MODULE_SWARM_MEMORY;
    header->target_module = BIO_MODULE_SWARM_MEMORY;
    header->timestamp_us = nimcp_time_get_us();
    header->channel = BIO_CHANNEL_DOPAMINE;
    header->payload_size = sizeof(uint32_t);
    header->flags = 0;

    uint32_t* id_ptr = (uint32_t*)(buffer + sizeof(bio_message_header_t));
    *id_ptr = belief->belief_id;

    nimcp_result_t result = bio_router_send(gb->bio_module, buffer, msg_size, 0);

    nimcp_free(buffer);
    return result;
}

static nimcp_result_t handle_gossip_spread(gossip_beliefs_t* gb, const void* msg)
{
    /* WHAT: Handle received gossip
     * WHY:  Integrate gossiped belief
     * HOW:  Parse, deserialize, store
     */

    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    const uint32_t* belief_id = (const uint32_t*)((const uint8_t*)msg +
                                                    sizeof(bio_message_header_t));

    LOG_DEBUG("Received gossiped belief: id=%u, from=0x%x",
              *belief_id, header->source_module);

    /* TODO: Implement full belief deserialization and integration */

    return NIMCP_SUCCESS;
}

static void apply_decay_to_agent(agent_info_t* agent, float decay_factor)
{
    /* TODO: Iterate through agent's beliefs and apply decay */
}
