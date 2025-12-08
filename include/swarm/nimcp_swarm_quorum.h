/**
 * @file nimcp_swarm_quorum.h
 * @brief Quorum Sensing and Decision Threshold system for NIMCP swarms
 *
 * Implements distributed decision-making inspired by bacterial quorum sensing
 * and honeybee nest-site selection. Enables swarms to reach consensus through
 * chemical-like signal accumulation and threshold activation.
 *
 * Biological Basis:
 * - Bacterial quorum sensing (autoinducer concentration)
 * - Honeybee waggle dance and nest-site selection
 * - Ant trail pheromone-based decision making
 *
 * Key Mechanisms:
 * 1. Signal molecules accumulate as drones broadcast opinions
 * 2. Threshold activation triggers collective action
 * 3. Positive feedback amplifies winning options
 * 4. Cross-inhibition suppresses alternatives
 * 5. Hysteresis prevents oscillation
 */

#ifndef NIMCP_SWARM_QUORUM_H
#define NIMCP_SWARM_QUORUM_H

#include "async/nimcp_bio_router.h"
#include "utils/thread/nimcp_thread.h"
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"

/* Forward declarations */
struct nimcp_brain;

/**
 * @brief Signal molecule types (abstract chemical concentrations)
 */
typedef enum {
    NIMCP_SIGNAL_ATTACK = 0,    /**< Propose attack action */
    NIMCP_SIGNAL_RETREAT,       /**< Propose retreat */
    NIMCP_SIGNAL_EXPLORE,       /**< Propose exploration */
    NIMCP_SIGNAL_DEFEND,        /**< Propose defensive posture */
    NIMCP_SIGNAL_RESOURCE,      /**< Resource discovery */
    NIMCP_SIGNAL_ALERT,         /**< General alert */
    NIMCP_SIGNAL_FORMATION,     /**< Formation change proposal */
    NIMCP_SIGNAL_LEADER,        /**< Leader election signal */
    NIMCP_SIGNAL_COUNT          /**< Total number of signal types */
} nimcp_signal_type_t;

/**
 * @brief Commitment states for individual drones
 */
typedef enum {
    NIMCP_COMMIT_UNCOMMITTED = 0, /**< No preference yet */
    NIMCP_COMMIT_LEANING,         /**< Weak preference forming */
    NIMCP_COMMIT_COMMITTED,       /**< Strong commitment */
    NIMCP_COMMIT_AMPLIFYING       /**< Actively recruiting others */
} nimcp_commitment_state_t;

/**
 * @brief Decision types for quorum-based actions
 */
typedef enum {
    NIMCP_DECISION_TARGET_SELECT = 0, /**< Target selection */
    NIMCP_DECISION_FORMATION_CHANGE,  /**< Formation change */
    NIMCP_DECISION_RETREAT,           /**< Retreat decision */
    NIMCP_DECISION_RESOURCE_ALLOC,    /**< Resource allocation */
    NIMCP_DECISION_LEADER_ELECT,      /**< Leader election */
    NIMCP_DECISION_PATROL_ROUTE,      /**< Patrol route selection */
    NIMCP_DECISION_ATTACK_TIMING,     /**< Attack timing coordination */
    NIMCP_DECISION_COUNT              /**< Total decision types */
} nimcp_decision_type_t;

/**
 * @brief Signal molecule concentrations
 */
typedef struct {
    double concentration;        /**< Current concentration [0.0, 1.0] */
    double threshold;            /**< Activation threshold */
    double decay_rate;           /**< Decay per time step */
    double amplification;        /**< Amplification factor for committed drones */
    double inhibition;           /**< Cross-inhibition strength */
    uint64_t last_update_time;   /**< Last update timestamp */
    uint32_t committed_count;    /**< Number of committed drones */
    bool threshold_reached;      /**< Whether threshold was reached */
    double hysteresis_low;       /**< Lower threshold for deactivation */
    double hysteresis_high;      /**< Upper threshold for activation */
} nimcp_signal_molecule_t;

/**
 * @brief Commitment record for a single drone
 */
typedef struct {
    uint32_t drone_id;                    /**< Drone identifier */
    nimcp_signal_type_t signal;           /**< Signal type committed to */
    nimcp_commitment_state_t state;       /**< Current commitment state */
    double commitment_strength;           /**< Strength of commitment [0.0, 1.0] */
    uint64_t commitment_time;             /**< When commitment was made */
    uint32_t recruitment_count;           /**< Number of drones recruited */
    bool is_amplifying;                   /**< Whether actively amplifying */
} nimcp_drone_commitment_t;

/**
 * @brief Decision record for tracking consensus
 */
typedef struct {
    nimcp_decision_type_t type;           /**< Type of decision */
    nimcp_signal_type_t winning_signal;   /**< Signal that reached quorum */
    double consensus_strength;            /**< Strength of consensus [0.0, 1.0] */
    uint64_t decision_time;               /**< When quorum was reached */
    uint32_t participating_drones;        /**< Number of drones that participated */
    uint32_t committed_drones;            /**< Number of committed drones */
    bool is_final;                        /**< Whether decision is finalized */
    void* decision_data;                  /**< Decision-specific data */
} nimcp_quorum_decision_t;

/**
 * @brief Quorum sensing configuration
 */
typedef struct {
    double base_threshold;                /**< Base threshold for activation */
    double threshold_variance;            /**< Variance in thresholds */
    double decay_rate;                    /**< Signal decay rate */
    double amplification_factor;          /**< Amplification for committed drones */
    double inhibition_strength;           /**< Cross-inhibition strength */
    double hysteresis_width;              /**< Width of hysteresis region */
    double commitment_threshold_low;      /**< Threshold for leaning state */
    double commitment_threshold_high;     /**< Threshold for committed state */
    double amplification_threshold;       /**< Threshold for amplifying state */
    uint32_t min_quorum_size;             /**< Minimum drones for valid quorum */
    double cascade_speed;                 /**< Speed of commitment cascade */
    bool enable_cross_inhibition;         /**< Enable cross-inhibition */
    bool enable_positive_feedback;        /**< Enable positive feedback */
    bool enable_hysteresis;               /**< Enable hysteresis */
} nimcp_quorum_config_t;

/**
 * @brief Statistics for quorum sensing
 */
typedef struct {
    uint64_t total_decisions;             /**< Total decisions made */
    uint64_t successful_quorums;          /**< Successful quorum formations */
    uint64_t failed_quorums;              /**< Failed quorum attempts */
    uint64_t split_decisions;             /**< Cases of split decisions */
    double avg_decision_time;             /**< Average time to decision (ms) */
    double avg_consensus_strength;        /**< Average consensus strength */
    uint32_t max_committed_drones;        /**< Maximum committed drones */
    uint32_t min_committed_drones;        /**< Minimum committed drones */
    uint64_t total_signals_broadcast;     /**< Total signals broadcast */
    uint64_t total_commitments;           /**< Total commitment events */
    uint64_t cascade_events;              /**< Number of cascade events */
    double avg_cascade_time;              /**< Average cascade time (ms) */
} nimcp_quorum_stats_t;

/**
 * @brief Main quorum sensing system structure
 */
typedef struct nimcp_swarm_quorum {
    /* Configuration */
    nimcp_quorum_config_t config;         /**< Configuration parameters */

    /* Signal molecules */
    nimcp_signal_molecule_t signals[NIMCP_SIGNAL_COUNT]; /**< Signal concentrations */

    /* Drone commitments */
    nimcp_drone_commitment_t* commitments; /**< Array of drone commitments */
    uint32_t commitment_count;            /**< Number of commitments */
    uint32_t commitment_capacity;         /**< Capacity of commitment array */

    /* Decision tracking */
    nimcp_quorum_decision_t* decisions;   /**< Array of decisions */
    uint32_t decision_count;              /**< Number of decisions */
    uint32_t decision_capacity;           /**< Capacity of decision array */

    /* Statistics */
    nimcp_quorum_stats_t stats;           /**< Statistics tracking */

    /* Bio-async integration */
    struct nimcp_brain* brain;            /**< Associated brain for messaging */
    void* bio_ctx;                        /**< Bio-async router context */

    /* Synchronization */
    nimcp_platform_mutex_t* mutex;  /**< Mutex for thread safety */

    /* State */
    uint64_t creation_time;               /**< Creation timestamp */
    bool is_active;                       /**< Whether system is active */
} nimcp_swarm_quorum_t;

/**
 * @brief Message payload for signal broadcast
 */
typedef struct {
    nimcp_signal_type_t signal;           /**< Signal type */
    double strength;                      /**< Signal strength */
    uint32_t drone_id;                    /**< Broadcasting drone ID */
    nimcp_commitment_state_t commitment;  /**< Commitment state */
} nimcp_quorum_signal_msg_t;

/**
 * @brief Message payload for commitment update
 */
typedef struct {
    uint32_t drone_id;                    /**< Drone identifier */
    nimcp_signal_type_t signal;           /**< Signal committed to */
    nimcp_commitment_state_t state;       /**< New commitment state */
    double strength;                      /**< Commitment strength */
} nimcp_quorum_commitment_msg_t;

/**
 * @brief Message payload for decision announcement
 */
typedef struct {
    nimcp_decision_type_t decision_type;  /**< Type of decision */
    nimcp_signal_type_t winning_signal;   /**< Winning signal */
    double consensus_strength;            /**< Consensus strength */
    uint32_t participating_drones;        /**< Participants count */
} nimcp_quorum_decision_msg_t;

/* ============================================================================
 * Core API Functions
 * ============================================================================ */

/**
 * @brief Create a new quorum sensing system
 *
 * @param config Configuration parameters (NULL for defaults)
 * @param brain Associated brain for bio-async messaging (optional)
 * @return Pointer to new quorum system, or NULL on failure
 */
nimcp_swarm_quorum_t* nimcp_swarm_quorum_create(
    const nimcp_quorum_config_t* config,
    struct nimcp_brain* brain
);

/**
 * @brief Destroy a quorum sensing system
 *
 * @param quorum Quorum system to destroy
 */
void nimcp_swarm_quorum_destroy(nimcp_swarm_quorum_t* quorum);

/**
 * @brief Get default configuration
 *
 * @param config Output configuration structure
 */
void nimcp_swarm_quorum_default_config(nimcp_quorum_config_t* config);

/* ============================================================================
 * Signal Broadcasting
 * ============================================================================ */

/**
 * @brief Broadcast a signal molecule
 *
 * @param quorum Quorum system
 * @param drone_id Broadcasting drone ID
 * @param signal Signal type to broadcast
 * @param strength Signal strength [0.0, 1.0]
 * @return true on success, false on failure
 */
bool nimcp_quorum_broadcast_signal(
    nimcp_swarm_quorum_t* quorum,
    uint32_t drone_id,
    nimcp_signal_type_t signal,
    double strength
);

/**
 * @brief Receive and integrate signal from another drone
 *
 * @param quorum Quorum system
 * @param signal Signal type
 * @param strength Signal strength
 * @param source_drone Source drone ID
 * @return true on success, false on failure
 */
bool nimcp_quorum_receive_signal(
    nimcp_swarm_quorum_t* quorum,
    nimcp_signal_type_t signal,
    double strength,
    uint32_t source_drone
);

/**
 * @brief Update signal concentrations (decay, diffusion)
 *
 * @param quorum Quorum system
 * @param delta_time Time elapsed since last update (ms)
 */
void nimcp_quorum_update_signals(
    nimcp_swarm_quorum_t* quorum,
    double delta_time
);

/* ============================================================================
 * Commitment Management
 * ============================================================================ */

/**
 * @brief Update drone commitment state
 *
 * @param quorum Quorum system
 * @param drone_id Drone identifier
 * @param signal Signal to commit to
 * @param strength Commitment strength [0.0, 1.0]
 * @return true on success, false on failure
 */
bool nimcp_quorum_update_commitment(
    nimcp_swarm_quorum_t* quorum,
    uint32_t drone_id,
    nimcp_signal_type_t signal,
    double strength
);

/**
 * @brief Get drone's current commitment
 *
 * @param quorum Quorum system
 * @param drone_id Drone identifier
 * @return Pointer to commitment record, or NULL if not found
 */
const nimcp_drone_commitment_t* nimcp_quorum_get_commitment(
    const nimcp_swarm_quorum_t* quorum,
    uint32_t drone_id
);

/**
 * @brief Remove drone commitment (drone becomes uncommitted)
 *
 * @param quorum Quorum system
 * @param drone_id Drone identifier
 * @return true on success, false on failure
 */
bool nimcp_quorum_remove_commitment(
    nimcp_swarm_quorum_t* quorum,
    uint32_t drone_id
);

/**
 * @brief Trigger commitment cascade for a signal
 *
 * Initiates positive feedback loop that rapidly recruits uncommitted drones
 *
 * @param quorum Quorum system
 * @param signal Signal type for cascade
 * @return Number of newly committed drones
 */
uint32_t nimcp_quorum_trigger_cascade(
    nimcp_swarm_quorum_t* quorum,
    nimcp_signal_type_t signal
);

/* ============================================================================
 * Threshold and Decision Logic
 * ============================================================================ */

/**
 * @brief Check if quorum threshold is reached for a signal
 *
 * @param quorum Quorum system
 * @param signal Signal type to check
 * @return true if threshold reached, false otherwise
 */
bool nimcp_quorum_check_threshold(
    const nimcp_swarm_quorum_t* quorum,
    nimcp_signal_type_t signal
);

/**
 * @brief Apply cross-inhibition between competing signals
 *
 * @param quorum Quorum system
 * @param winning_signal Signal that suppresses others
 */
void nimcp_quorum_apply_cross_inhibition(
    nimcp_swarm_quorum_t* quorum,
    nimcp_signal_type_t winning_signal
);

/**
 * @brief Make a decision based on current signal state
 *
 * @param quorum Quorum system
 * @param decision_type Type of decision to make
 * @param decision_data Decision-specific data (optional)
 * @return true if decision was made, false if no quorum
 */
bool nimcp_quorum_make_decision(
    nimcp_swarm_quorum_t* quorum,
    nimcp_decision_type_t decision_type,
    void* decision_data
);

/**
 * @brief Get the most recent decision
 *
 * @param quorum Quorum system
 * @return Pointer to decision, or NULL if no decisions made
 */
const nimcp_quorum_decision_t* nimcp_quorum_get_last_decision(
    const nimcp_swarm_quorum_t* quorum
);

/**
 * @brief Finalize a decision (prevents further changes)
 *
 * @param quorum Quorum system
 * @param decision_index Index of decision to finalize
 * @return true on success, false on failure
 */
bool nimcp_quorum_finalize_decision(
    nimcp_swarm_quorum_t* quorum,
    uint32_t decision_index
);

/* ============================================================================
 * Positive Feedback
 * ============================================================================ */

/**
 * @brief Apply positive feedback from committed drones
 *
 * Amplifies signal strength based on number of committed drones
 *
 * @param quorum Quorum system
 * @param signal Signal type to amplify
 * @return New signal concentration after amplification
 */
double nimcp_quorum_apply_positive_feedback(
    nimcp_swarm_quorum_t* quorum,
    nimcp_signal_type_t signal
);

/**
 * @brief Recruit uncommitted drones to a signal
 *
 * @param quorum Quorum system
 * @param signal Signal to recruit for
 * @param recruiter_id ID of recruiting drone
 * @return Number of drones recruited
 */
uint32_t nimcp_quorum_recruit_drones(
    nimcp_swarm_quorum_t* quorum,
    nimcp_signal_type_t signal,
    uint32_t recruiter_id
);

/* ============================================================================
 * Query and Statistics
 * ============================================================================ */

/**
 * @brief Get signal concentration
 *
 * @param quorum Quorum system
 * @param signal Signal type
 * @return Current concentration [0.0, 1.0]
 */
double nimcp_quorum_get_signal_concentration(
    const nimcp_swarm_quorum_t* quorum,
    nimcp_signal_type_t signal
);

/**
 * @brief Get number of committed drones for a signal
 *
 * @param quorum Quorum system
 * @param signal Signal type
 * @return Number of committed drones
 */
uint32_t nimcp_quorum_get_committed_count(
    const nimcp_swarm_quorum_t* quorum,
    nimcp_signal_type_t signal
);

/**
 * @brief Get consensus strength for current state
 *
 * @param quorum Quorum system
 * @return Consensus strength [0.0, 1.0], 1.0 = unanimous
 */
double nimcp_quorum_get_consensus_strength(
    const nimcp_swarm_quorum_t* quorum
);

/**
 * @brief Get statistics
 *
 * @param quorum Quorum system
 * @return Pointer to statistics structure
 */
const nimcp_quorum_stats_t* nimcp_quorum_get_stats(
    const nimcp_swarm_quorum_t* quorum
);

/**
 * @brief Reset statistics
 *
 * @param quorum Quorum system
 */
void nimcp_quorum_reset_stats(nimcp_swarm_quorum_t* quorum);

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

/**
 * @brief Process incoming bio-async message
 *
 * @param quorum Quorum system
 * @param msg Message to process
 * @return true on success, false on failure
 */
bool nimcp_quorum_handle_message(
    nimcp_swarm_quorum_t* quorum,
    const bio_message_header_t* msg
);

/**
 * @brief Register quorum system with brain's bio-async messaging
 *
 * @param quorum Quorum system
 * @param brain Brain to register with
 * @return true on success, false on failure
 */
bool nimcp_quorum_register_handlers(
    nimcp_swarm_quorum_t* quorum,
    struct nimcp_brain* brain
);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Get signal type name as string
 *
 * @param signal Signal type
 * @return Signal name string
 */
const char* nimcp_quorum_signal_name(nimcp_signal_type_t signal);

/**
 * @brief Get decision type name as string
 *
 * @param decision Decision type
 * @return Decision name string
 */
const char* nimcp_quorum_decision_name(nimcp_decision_type_t decision);

/**
 * @brief Get commitment state name as string
 *
 * @param state Commitment state
 * @return State name string
 */
const char* nimcp_quorum_commitment_name(nimcp_commitment_state_t state);

/**
 * @brief Print quorum system state to stdout
 *
 * @param quorum Quorum system
 */
void nimcp_quorum_print_state(const nimcp_swarm_quorum_t* quorum);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_QUORUM_H */
