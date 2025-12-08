/**
 * @file nimcp_swarm_immune.h
 * @brief Swarm Immune System - Adaptive threat detection and response
 *
 * Biologically-inspired immune system for swarm robotics:
 * - Antigen recognition (threat detection)
 * - Memory cells (learned threat patterns)
 * - Antibody production (response strategies)
 * - Self/non-self discrimination
 * - Clonal selection (adaptive responses)
 *
 * @author NIMCP Development Team
 * @date 2025
 *
 * @copyright Copyright (c) 2025 NIMCP Project
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef NIMCP_SWARM_IMMUNE_H
#define NIMCP_SWARM_IMMUNE_H

#include "utils/validation/nimcp_common.h"
#include "async/nimcp_bio_messages.h"
#include <stdint.h>
#include <stdbool.h>

#include "async/nimcp_bio_router.h"
#include "utils/thread/nimcp_thread.h"
#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Type Definitions
 * ============================================================================ */

/**
 * @brief Threat types (antigens)
 */
typedef enum {
    THREAT_MALICIOUS_DRONE = 0,  /**< Compromised swarm member */
    THREAT_JAMMING,              /**< Communication jamming */
    THREAT_SPOOFING,             /**< Identity spoofing */
    THREAT_INJECTION,            /**< Message injection attack */
    THREAT_REPLAY,               /**< Replay attack */
    THREAT_PHYSICAL,             /**< Physical attack on drone */
    THREAT_DOS,                  /**< Denial of service */
    THREAT_SYBIL,                /**< Sybil attack */
    THREAT_BYZANTINE,            /**< Byzantine behavior */
    THREAT_COUNT
} NimcpSwarmThreatType;

/**
 * @brief Response strategy types (antibodies)
 */
typedef enum {
    RESPONSE_NONE = 0,           /**< No response needed */
    RESPONSE_ISOLATION,          /**< Quarantine compromised drone */
    RESPONSE_EVASION,            /**< Move away from threat */
    RESPONSE_COUNTER_ATTACK,     /**< Active defense measures */
    RESPONSE_ALERT,              /**< Warn swarm */
    RESPONSE_COORDINATION,       /**< Coordinate defensive action */
    RESPONSE_RECONFIGURATION,    /**< Reconfigure network topology */
    RESPONSE_AUTHENTICATION,     /**< Re-authenticate members */
    RESPONSE_COUNT
} NimcpSwarmResponseType;

/**
 * @brief Threat severity levels
 */
typedef enum {
    SEVERITY_LOW = 0,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL
} NimcpSwarmSeverity;

/**
 * @brief Threat signature for pattern matching
 */
typedef struct {
    uint8_t pattern[64];         /**< Signature pattern */
    size_t pattern_len;          /**< Pattern length */
    float match_threshold;       /**< Fuzzy match threshold (0.0-1.0) */
    NimcpSwarmThreatType type;   /**< Threat type */
    uint32_t detection_count;    /**< Times detected */
    uint64_t last_seen;          /**< Last detection timestamp */
} NimcpSwarmThreatSignature;

/**
 * @brief Memory cell - learned threat pattern
 */
typedef struct {
    uint32_t id;                            /**< Unique memory cell ID */
    NimcpSwarmThreatSignature signature;    /**< Threat signature */
    NimcpSwarmResponseType response;        /**< Associated response */
    float effectiveness;                    /**< Response effectiveness (0.0-1.0) */
    uint32_t activation_count;              /**< Times activated */
    uint64_t created_time;                  /**< Creation timestamp */
    uint64_t last_activation;               /**< Last activation time */
    float decay_factor;                     /**< Memory decay rate */
    bool shared;                            /**< Shared with swarm */
} NimcpSwarmMemoryCell;

/**
 * @brief Detected threat (antigen)
 */
typedef struct {
    uint32_t id;                     /**< Unique threat ID */
    NimcpSwarmThreatType type;       /**< Threat type */
    NimcpSwarmSeverity severity;     /**< Severity level */
    uint32_t source_drone_id;        /**< Source of threat */
    float confidence;                /**< Detection confidence (0.0-1.0) */
    uint64_t detection_time;         /**< When detected */
    uint8_t data[256];               /**< Threat data */
    size_t data_len;                 /**< Data length */
    bool confirmed;                  /**< Confirmed by multiple drones */
    uint32_t confirming_drones;      /**< Number of confirming drones */
} NimcpSwarmThreat;

/**
 * @brief Active immune response (antibody)
 */
typedef struct {
    uint32_t id;                     /**< Response ID */
    NimcpSwarmResponseType type;     /**< Response type */
    uint32_t threat_id;              /**< Target threat ID */
    uint32_t target_drone_id;        /**< Target drone (if applicable) */
    uint64_t start_time;             /**< Response start time */
    uint64_t duration;               /**< Response duration (ms) */
    float intensity;                 /**< Response intensity (0.0-1.0) */
    bool coordinated;                /**< Requires swarm coordination */
    uint32_t participating_drones;   /**< Number of participating drones */
} NimcpSwarmResponse;

/**
 * @brief Behavioral profile for anomaly detection
 */
typedef struct {
    uint32_t drone_id;               /**< Drone ID */
    float msg_rate;                  /**< Message rate (msgs/sec) */
    float movement_pattern[3];       /**< Movement vector */
    float energy_usage;              /**< Energy consumption */
    uint32_t connection_changes;     /**< Topology change count */
    uint64_t last_update;            /**< Last profile update */
    float anomaly_score;             /**< Cumulative anomaly score */
} NimcpSwarmBehaviorProfile;

/**
 * @brief Swarm immune system configuration
 */
typedef struct {
    size_t max_memory_cells;         /**< Maximum memory cells */
    size_t max_active_threats;       /**< Maximum tracked threats */
    size_t max_active_responses;     /**< Maximum concurrent responses */
    float recognition_threshold;     /**< Antigen recognition threshold */
    float self_tolerance;            /**< Self/non-self discrimination threshold */
    float memory_decay_rate;         /**< Memory cell decay rate */
    float clonal_expansion_rate;     /**< Clonal selection rate */
    bool enable_sharing;             /**< Enable threat intelligence sharing */
    bool enable_coordination;        /**< Enable coordinated responses */
    uint32_t confirmation_threshold; /**< Threat confirmation threshold */
} NimcpSwarmImmuneConfig;

/**
 * @brief Swarm immune system state
 */
typedef struct {
    NimcpSwarmImmuneConfig config;   /**< Configuration */

    /* Memory cells (learned threats) */
    NimcpSwarmMemoryCell* memory_cells;
    size_t memory_cell_count;
    size_t memory_cell_capacity;

    /* Active threats (detected antigens) */
    NimcpSwarmThreat* active_threats;
    size_t active_threat_count;
    size_t active_threat_capacity;

    /* Active responses (deployed antibodies) */
    NimcpSwarmResponse* active_responses;
    size_t active_response_count;
    size_t active_response_capacity;

    /* Behavioral profiles for anomaly detection */
    NimcpSwarmBehaviorProfile* behavior_profiles;
    size_t behavior_profile_count;
    size_t behavior_profile_capacity;

    /* Statistics */
    uint64_t total_threats_detected;
    uint64_t total_threats_neutralized;
    uint64_t false_positive_count;
    uint64_t false_negative_count;

    /* Bio-async integration */
    bio_module_context_t* bio_ctx;

    /* Self identification */
    uint32_t self_drone_id;
    uint8_t self_signature[32];

    /* Thread safety */
    void* mutex;
} NimcpSwarmImmuneSystem;

/* ============================================================================
 * Core API
 * ============================================================================ */

/**
 * @brief Create a swarm immune system
 *
 * @param config Configuration parameters
 * @param bio_ctx Bio-async context for swarm communication (optional)
 * @param self_drone_id This drone's ID for self/non-self discrimination
 * @return Newly created immune system or NULL on failure
 */
NimcpSwarmImmuneSystem* nimcp_swarm_immune_create(
    const NimcpSwarmImmuneConfig* config,
    bio_module_context_t* bio_ctx,
    uint32_t self_drone_id
);

/**
 * @brief Destroy immune system and free resources
 *
 * @param system Immune system to destroy
 */
void nimcp_swarm_immune_destroy(NimcpSwarmImmuneSystem* system);

/**
 * @brief Get default configuration
 *
 * @param config Output configuration
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_default_config(NimcpSwarmImmuneConfig* config);

/* ============================================================================
 * Antigen Recognition (Threat Detection)
 * ============================================================================ */

/**
 * @brief Detect threats from incoming data
 *
 * Scans data for known threat signatures and behavioral anomalies.
 *
 * @param system Immune system
 * @param data Input data to scan
 * @param data_len Data length
 * @param source_drone_id Source drone ID
 * @param threat_id Output: detected threat ID (if any)
 * @return NIMCP_OK if no threat, NIMCP_THREAT_DETECTED if threat found
 */
nimcp_result_t nimcp_swarm_immune_detect_threat(
    NimcpSwarmImmuneSystem* system,
    const uint8_t* data,
    size_t data_len,
    uint32_t source_drone_id,
    uint32_t* threat_id
);

/**
 * @brief Check if drone behavior is anomalous
 *
 * @param system Immune system
 * @param drone_id Drone to check
 * @param behavior Current behavior profile
 * @param anomaly_score Output: anomaly score (0.0-1.0)
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_check_behavior(
    NimcpSwarmImmuneSystem* system,
    uint32_t drone_id,
    const NimcpSwarmBehaviorProfile* behavior,
    float* anomaly_score
);

/**
 * @brief Verify if drone is part of self (legitimate swarm member)
 *
 * @param system Immune system
 * @param drone_id Drone ID to verify
 * @param signature Drone signature
 * @param signature_len Signature length
 * @param is_self Output: true if verified self
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_verify_self(
    NimcpSwarmImmuneSystem* system,
    uint32_t drone_id,
    const uint8_t* signature,
    size_t signature_len,
    bool* is_self
);

/* ============================================================================
 * Memory Cells (Learned Patterns)
 * ============================================================================ */

/**
 * @brief Add a memory cell (learned threat pattern)
 *
 * @param system Immune system
 * @param signature Threat signature
 * @param response Associated response strategy
 * @param effectiveness Initial effectiveness
 * @param cell_id Output: created memory cell ID
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_add_memory_cell(
    NimcpSwarmImmuneSystem* system,
    const NimcpSwarmThreatSignature* signature,
    NimcpSwarmResponseType response,
    float effectiveness,
    uint32_t* cell_id
);

/**
 * @brief Search for matching memory cell
 *
 * @param system Immune system
 * @param data Data to match
 * @param data_len Data length
 * @param cell Output: matched memory cell (if found)
 * @return NIMCP_OK if match found, NIMCP_NOT_FOUND otherwise
 */
nimcp_result_t nimcp_swarm_immune_find_memory_cell(
    NimcpSwarmImmuneSystem* system,
    const uint8_t* data,
    size_t data_len,
    NimcpSwarmMemoryCell** cell
);

/**
 * @brief Update memory cell effectiveness
 *
 * @param system Immune system
 * @param cell_id Memory cell ID
 * @param new_effectiveness Effectiveness score (0.0-1.0)
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_update_effectiveness(
    NimcpSwarmImmuneSystem* system,
    uint32_t cell_id,
    float new_effectiveness
);

/**
 * @brief Apply memory decay to all cells
 *
 * @param system Immune system
 * @param current_time Current timestamp
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_decay_memory(
    NimcpSwarmImmuneSystem* system,
    uint64_t current_time
);

/* ============================================================================
 * Antibody Production (Response Generation)
 * ============================================================================ */

/**
 * @brief Generate immune response to threat
 *
 * @param system Immune system
 * @param threat_id Threat ID
 * @param response_id Output: generated response ID
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_generate_response(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id,
    uint32_t* response_id
);

/**
 * @brief Execute active response
 *
 * @param system Immune system
 * @param response_id Response ID
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_execute_response(
    NimcpSwarmImmuneSystem* system,
    uint32_t response_id
);

/**
 * @brief Get recommended response for threat type
 *
 * @param system Immune system
 * @param threat_type Threat type
 * @param severity Threat severity
 * @return Recommended response type
 */
NimcpSwarmResponseType nimcp_swarm_immune_get_response(
    NimcpSwarmImmuneSystem* system,
    NimcpSwarmThreatType threat_type,
    NimcpSwarmSeverity severity
);

/* ============================================================================
 * Clonal Selection (Adaptive Evolution)
 * ============================================================================ */

/**
 * @brief Amplify successful responses (clonal expansion)
 *
 * @param system Immune system
 * @param response_id Response that was successful
 * @param success_rate Success rate (0.0-1.0)
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_amplify_response(
    NimcpSwarmImmuneSystem* system,
    uint32_t response_id,
    float success_rate
);

/**
 * @brief Adapt threat signature based on observed mutations
 *
 * @param system Immune system
 * @param cell_id Memory cell ID
 * @param new_data Mutated threat data
 * @param new_data_len Data length
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_adapt_signature(
    NimcpSwarmImmuneSystem* system,
    uint32_t cell_id,
    const uint8_t* new_data,
    size_t new_data_len
);

/**
 * @brief Perform affinity maturation on memory cells
 *
 * Improves threat recognition over time through selection.
 *
 * @param system Immune system
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_affinity_maturation(
    NimcpSwarmImmuneSystem* system
);

/* ============================================================================
 * Swarm Coordination (via Bio-Async)
 * ============================================================================ */

/**
 * @brief Share threat intelligence with swarm
 *
 * @param system Immune system
 * @param threat_id Threat to share
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_share_threat(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id
);

/**
 * @brief Share memory cell with swarm
 *
 * @param system Immune system
 * @param cell_id Memory cell to share
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_share_memory_cell(
    NimcpSwarmImmuneSystem* system,
    uint32_t cell_id
);

/**
 * @brief Coordinate response with other drones
 *
 * @param system Immune system
 * @param response_id Response to coordinate
 * @param participating_drones Array of drone IDs
 * @param num_drones Number of participating drones
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_coordinate_response(
    NimcpSwarmImmuneSystem* system,
    uint32_t response_id,
    const uint32_t* participating_drones,
    size_t num_drones
);

/**
 * @brief Broadcast alert to entire swarm
 *
 * @param system Immune system
 * @param threat_id Threat to alert about
 * @param priority Alert priority
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_broadcast_alert(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id,
    NimcpSwarmSeverity priority
);

/* ============================================================================
 * Threat Management
 * ============================================================================ */

/**
 * @brief Confirm threat from multiple sources
 *
 * @param system Immune system
 * @param threat_id Threat ID
 * @param confirming_drone_id Drone confirming the threat
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_confirm_threat(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id,
    uint32_t confirming_drone_id
);

/**
 * @brief Mark threat as neutralized
 *
 * @param system Immune system
 * @param threat_id Threat ID
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_neutralize_threat(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id
);

/**
 * @brief Get active threat information
 *
 * @param system Immune system
 * @param threat_id Threat ID
 * @param threat Output: threat information
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_get_threat(
    NimcpSwarmImmuneSystem* system,
    uint32_t threat_id,
    const NimcpSwarmThreat** threat
);

/* ============================================================================
 * Statistics and Monitoring
 * ============================================================================ */

/**
 * @brief Get immune system statistics
 *
 * @param system Immune system
 * @param total_threats Output: total threats detected
 * @param neutralized Output: threats neutralized
 * @param false_positives Output: false positive count
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_get_stats(
    NimcpSwarmImmuneSystem* system,
    uint64_t* total_threats,
    uint64_t* neutralized,
    uint64_t* false_positives
);

/**
 * @brief Get threat type name
 *
 * @param type Threat type
 * @return String name of threat type
 */
const char* nimcp_swarm_threat_type_name(NimcpSwarmThreatType type);

/**
 * @brief Get response type name
 *
 * @param type Response type
 * @return String name of response type
 */
const char* nimcp_swarm_response_type_name(NimcpSwarmResponseType type);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Update system time step
 *
 * @param system Immune system
 * @param current_time Current timestamp
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_update(
    NimcpSwarmImmuneSystem* system,
    uint64_t current_time
);

/**
 * @brief Reset immune system state
 *
 * @param system Immune system
 * @param preserve_memory If true, preserve memory cells
 * @return NIMCP_OK on success
 */
nimcp_result_t nimcp_swarm_immune_reset(
    NimcpSwarmImmuneSystem* system,
    bool preserve_memory
);

#include "async/nimcp_bio_router.h"
#include "utils/thread/nimcp_thread.h"
#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_IMMUNE_H */
