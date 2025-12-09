/**
 * @file nimcp_swarm_brain_local.h
 * @brief Swarm Brain Local Instantiation - Per-Agent Brain Management
 *
 * WHAT: Manages local brain instances for each swarm agent with distributed learning
 * WHY:  Enables each agent to have its own local brain while maintaining swarm coherence
 * HOW:  Per-agent brain allocation, weight synchronization, divergence tracking
 *
 * ARCHITECTURE:
 *
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │          Swarm Brain Manager                                 │
 *   │                                                              │
 *   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
 *   │  │  Agent 1     │  │  Agent 2     │  │  Agent N     │      │
 *   │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │      │
 *   │  │  │ Brain  │  │  │  │ Brain  │  │  │  │ Brain  │  │      │
 *   │  │  │Instance│  │  │  │Instance│  │  │  │Instance│  │      │
 *   │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │      │
 *   │  │  Weights     │  │  Weights     │  │  Weights     │      │
 *   │  │  Divergence  │  │  Divergence  │  │  Divergence  │      │
 *   │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
 *   │         │                 │                 │              │
 *   │         └─────────────────┴─────────────────┘              │
 *   │                           │                                │
 *   │                  ┌────────▼────────┐                       │
 *   │                  │ Weight Consensus │                      │
 *   │                  │    Aggregation   │                      │
 *   │                  └─────────────────┘                       │
 *   │                           │                                │
 *   │                  ┌────────▼────────┐                       │
 *   │                  │  Bio-Async      │                       │
 *   │                  │  Integration    │                       │
 *   │                  └─────────────────┘                       │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * FEATURES:
 * - Per-agent brain allocation with configurable sizes
 * - Local learning and processing
 * - Weight synchronization across swarm
 * - Divergence detection and management
 * - Consensus weight aggregation
 * - Bio-async message passing
 *
 * USAGE EXAMPLE:
 * ```c
 * // Configure swarm brain manager
 * swarm_brain_config_t config = {
 *     .default_brain_size = 100,
 *     .max_local_neurons = 500,
 *     .sync_interval_ms = 1000,
 *     .divergence_threshold = 0.3,
 *     .enable_weight_sharing = true,
 *     .enable_bio_async = true
 * };
 *
 * // Create manager
 * swarm_brain_manager_t* mgr = swarm_brain_manager_create(&config);
 *
 * // Create brain for agent
 * swarm_brain_create_for_agent(mgr, agent_id, 100);
 *
 * // Local learning
 * swarm_brain_local_learn(mgr, agent_id, input, 10, target, 5);
 *
 * // Synchronize with swarm
 * swarm_brain_local_sync_weights(mgr, agent_id);
 *
 * // Check divergence
 * float divergence = swarm_brain_get_divergence(mgr, agent_id);
 *
 * // Cleanup
 * swarm_brain_manager_destroy(mgr);
 * ```
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 * @version 1.0.0
 */

#ifndef NIMCP_SWARM_BRAIN_LOCAL_H
#define NIMCP_SWARM_BRAIN_LOCAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "utils/validation/nimcp_common.h"
#include "nimcp.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

/** Forward declare brain_t type */
typedef struct brain_struct* brain_t;

/** Opaque swarm brain manager handle */
typedef struct swarm_brain_manager swarm_brain_manager_t;

//=============================================================================
// Constants
//=============================================================================

/** Maximum number of agents that can be managed */
#define SWARM_BRAIN_MAX_AGENTS 256

/** Default brain size (neurons) */
#define SWARM_BRAIN_DEFAULT_SIZE 100

/** Default sync interval (ms) */
#define SWARM_BRAIN_DEFAULT_SYNC_INTERVAL 1000

/** Default divergence threshold */
#define SWARM_BRAIN_DEFAULT_DIVERGENCE_THRESHOLD 0.3f

/** Maximum neurons per agent brain */
#define SWARM_BRAIN_MAX_NEURONS 1000

//=============================================================================
// Bio-Async Message Types
//=============================================================================

/** Bio-async message types for swarm brain */
typedef enum {
    BIO_MSG_BRAIN_SYNCED = 0x0C00,          /**< Brain weights synchronized */
    BIO_MSG_BRAIN_DIVERGENCE = 0x0C01,      /**< Brain divergence detected */
    BIO_MSG_BRAIN_WEIGHTS_REQUEST = 0x0C02, /**< Request consensus weights */
    BIO_MSG_BRAIN_WEIGHTS_UPDATE = 0x0C03,  /**< Consensus weights update */
    BIO_MSG_BRAIN_AGENT_JOINED = 0x0C04,    /**< New agent joined swarm */
    BIO_MSG_BRAIN_AGENT_LEFT = 0x0C05       /**< Agent left swarm */
} swarm_brain_bio_msg_type_t;

//=============================================================================
// Data Structures
//=============================================================================

/**
 * @brief Per-agent brain instance
 *
 * Tracks individual agent's brain, weights, and synchronization state
 */
typedef struct {
    uint32_t agent_id;              /**< Unique agent identifier */
    nimcp_brain_t brain;            /**< Local brain instance */
    uint32_t brain_size;            /**< Number of neurons */
    float* local_weights;           /**< Agent-specific weights */
    uint32_t num_weights;           /**< Number of weights */
    uint64_t last_sync_ms;          /**< Last sync timestamp */
    float divergence_score;         /**< Divergence from consensus (0-1) */
    bool active;                    /**< Agent is active */
} agent_brain_t;

/**
 * @brief Swarm brain manager configuration
 */
typedef struct {
    uint32_t default_brain_size;    /**< Default neurons per brain */
    uint32_t max_local_neurons;     /**< Maximum neurons per agent */
    float sync_interval_ms;         /**< Sync interval (default: 1000ms) */
    float divergence_threshold;     /**< Divergence alert threshold (0-1) */
    bool enable_weight_sharing;     /**< Enable weight sharing between agents */
    bool enable_bio_async;          /**< Enable bio-async integration */
} swarm_brain_config_t;

/**
 * @brief Swarm brain statistics
 */
typedef struct {
    uint32_t num_agents;            /**< Number of active agents */
    uint32_t total_neurons;         /**< Total neurons across all agents */
    uint64_t sync_count;            /**< Total synchronizations performed */
    float avg_divergence;           /**< Average divergence across agents */
    float max_divergence;           /**< Maximum divergence */
    uint32_t divergent_agents;      /**< Number of divergent agents */
    uint64_t uptime_ms;             /**< Manager uptime */
} swarm_brain_stats_t;

//=============================================================================
// Lifecycle Functions
//=============================================================================

/**
 * @brief Create swarm brain manager
 *
 * WHAT: Initializes manager for coordinating per-agent brains
 * WHY:  Central coordination point for distributed brain instances
 * HOW:  Allocates state, sets up bio-async, initializes consensus tracking
 *
 * @param config Configuration (NULL for defaults)
 * @return Manager handle or NULL on failure
 */
swarm_brain_manager_t* swarm_brain_manager_create(const swarm_brain_config_t* config);

/**
 * @brief Destroy swarm brain manager
 *
 * WHAT: Cleans up all agent brains and manager state
 * WHY:  Release all resources and disconnect from bio-async
 * HOW:  Destroys all agent brains, frees memory, unregisters from bio-async
 *
 * @param mgr Manager handle (NULL safe)
 */
void swarm_brain_manager_destroy(swarm_brain_manager_t* mgr);

//=============================================================================
// Agent Brain Management
//=============================================================================

/**
 * @brief Create brain for specific agent
 *
 * WHAT: Allocates and initializes brain instance for agent
 * WHY:  Each agent needs its own local brain for processing
 * HOW:  Creates brain, allocates weights, registers with manager
 *
 * @param mgr Manager handle
 * @param agent_id Unique agent identifier
 * @param brain_size Number of neurons (0 = use default)
 * @return NIMCP_SUCCESS or error code
 */
int swarm_brain_create_for_agent(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id,
    uint32_t brain_size
);

/**
 * @brief Destroy brain for specific agent
 *
 * WHAT: Removes and cleans up agent's brain instance
 * WHY:  Agent leaving swarm or resource cleanup
 * HOW:  Destroys brain, frees weights, removes from manager
 *
 * @param mgr Manager handle
 * @param agent_id Agent identifier
 * @return NIMCP_SUCCESS or error code
 */
int swarm_brain_destroy_for_agent(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
);

/**
 * @brief Get brain instance for agent
 *
 * WHAT: Retrieves brain handle for direct access
 * WHY:  Allow direct brain operations if needed
 * HOW:  Looks up agent, returns brain handle
 *
 * @param mgr Manager handle
 * @param agent_id Agent identifier
 * @return Brain handle or NULL if not found
 */
nimcp_brain_t swarm_brain_get(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
);

//=============================================================================
// Local Learning and Processing
//=============================================================================

/**
 * @brief Perform local learning on agent's brain
 *
 * WHAT: Trains agent's brain on local data
 * WHY:  Each agent learns from its own experiences
 * HOW:  Forwards to brain learning API, tracks weight changes
 *
 * @param mgr Manager handle
 * @param agent_id Agent identifier
 * @param input Input data
 * @param input_size Input size
 * @param target Target output
 * @param target_size Target size
 * @return NIMCP_SUCCESS or error code
 */
int swarm_brain_local_learn(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id,
    const float* input,
    uint32_t input_size,
    const float* target,
    uint32_t target_size
);

/**
 * @brief Process input through agent's brain
 *
 * WHAT: Runs inference on agent's local brain
 * WHY:  Local processing for agent decisions
 * HOW:  Forwards to brain inference, returns output
 *
 * @param mgr Manager handle
 * @param agent_id Agent identifier
 * @param input Input data
 * @param input_size Input size
 * @param output Output buffer
 * @param output_size Output size (in/out)
 * @return NIMCP_SUCCESS or error code
 */
int swarm_brain_local_process(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id,
    const float* input,
    uint32_t input_size,
    float* output,
    uint32_t* output_size
);

//=============================================================================
// Weight Synchronization
//=============================================================================

/**
 * @brief Synchronize agent's weights with swarm consensus
 *
 * WHAT: Updates agent's weights based on swarm consensus
 * WHY:  Maintain coherence across distributed brains
 * HOW:  Averages with consensus weights, updates divergence
 *
 * @param mgr Manager handle
 * @param agent_id Agent identifier
 * @return NIMCP_SUCCESS or error code
 */
int swarm_brain_local_sync_weights(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
);

/**
 * @brief Synchronize all agents with consensus
 *
 * WHAT: Performs full swarm synchronization
 * WHY:  Periodic coherence maintenance
 * HOW:  Syncs each active agent sequentially
 *
 * @param mgr Manager handle
 * @return NIMCP_SUCCESS or error code
 */
int swarm_brain_sync_all(swarm_brain_manager_t* mgr);

/**
 * @brief Get consensus weights across all agents
 *
 * WHAT: Computes averaged weights from all active agents
 * WHY:  Calculate swarm consensus for synchronization
 * HOW:  Averages weights across all active brains
 *
 * @param mgr Manager handle
 * @param weights Output weights array (allocated by function)
 * @param num_weights Output weight count
 * @return NIMCP_SUCCESS or error code
 *
 * NOTE: Caller must free weights array with nimcp_free()
 */
int swarm_brain_get_consensus_weights(
    swarm_brain_manager_t* mgr,
    float** weights,
    uint32_t* num_weights
);

//=============================================================================
// Divergence Detection
//=============================================================================

/**
 * @brief Get divergence score for specific agent
 *
 * WHAT: Returns how different agent's weights are from consensus
 * WHY:  Track individual agent coherence with swarm
 * HOW:  Computes distance between local and consensus weights
 *
 * @param mgr Manager handle
 * @param agent_id Agent identifier
 * @return Divergence score (0-1), or -1 on error
 */
float swarm_brain_get_divergence(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
);

/**
 * @brief Get list of divergent agents
 *
 * WHAT: Returns agents exceeding divergence threshold
 * WHY:  Identify agents needing synchronization
 * HOW:  Scans all agents, collects IDs above threshold
 *
 * @param mgr Manager handle
 * @param agents Output array of agent IDs (allocated by function)
 * @param count Output count of divergent agents
 * @return NIMCP_SUCCESS or error code
 *
 * NOTE: Caller must free agents array with nimcp_free()
 */
int swarm_brain_get_divergent_agents(
    swarm_brain_manager_t* mgr,
    uint32_t** agents,
    uint32_t* count
);

//=============================================================================
// Statistics and Monitoring
//=============================================================================

/**
 * @brief Get swarm brain statistics
 *
 * @param mgr Manager handle
 * @param stats Output statistics structure
 * @return NIMCP_SUCCESS or error code
 */
int swarm_brain_local_get_stats(
    swarm_brain_manager_t* mgr,
    swarm_brain_stats_t* stats
);

/**
 * @brief Reset statistics counters
 *
 * @param mgr Manager handle
 * @return NIMCP_SUCCESS or error code
 */
int swarm_brain_local_reset_stats(swarm_brain_manager_t* mgr);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get default swarm brain configuration
 *
 * @return Default configuration structure
 */
swarm_brain_config_t swarm_brain_local_default_config(void);

/**
 * @brief Check if agent has active brain
 *
 * @param mgr Manager handle
 * @param agent_id Agent identifier
 * @return true if agent has active brain, false otherwise
 */
bool swarm_brain_has_agent(
    swarm_brain_manager_t* mgr,
    uint32_t agent_id
);

/**
 * @brief Get number of active agents
 *
 * @param mgr Manager handle
 * @return Number of active agents
 */
uint32_t swarm_brain_get_agent_count(swarm_brain_manager_t* mgr);

/**
 * @brief Get list of all active agent IDs
 *
 * @param mgr Manager handle
 * @param agents Output array of agent IDs (allocated by function)
 * @param count Output count
 * @return NIMCP_SUCCESS or error code
 *
 * NOTE: Caller must free agents array with nimcp_free()
 */
int swarm_brain_get_all_agents(
    swarm_brain_manager_t* mgr,
    uint32_t** agents,
    uint32_t* count
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_BRAIN_LOCAL_H */
