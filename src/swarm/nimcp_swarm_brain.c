/**
 * @file nimcp_swarm_brain.c
 * @brief NIMCP Swarm Brain Coordinator Implementation
 *
 * Integrates all swarm components into a cohesive distributed cognitive system.
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 * @version 1.0.0
 */

#include "swarm/nimcp_swarm_brain.h"
#include "swarm/nimcp_swarm_signal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_common.h"
#include "utils/time/nimcp_time.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Type alias for consistency with other modules
typedef nimcp_platform_mutex_t nimcp_mutex_t;

//=============================================================================
// Module Constants
//=============================================================================

#define MODULE_NAME "swarm_brain"

// Timing constants
#define PEER_TIMEOUT_MS 500           // Consider peer dead after 500ms
#define HEARTBEAT_JITTER_MS 10        // Random jitter for heartbeat
#define SYNC_JITTER_MS 5              // Random jitter for sync

// Vote constants
#define MAX_ACTIVE_VOTES 8            // Maximum concurrent votes
#define VOTE_QUORUM_RATIO 0.51f       // 51% quorum required

// Workspace constants
#define WORKSPACE_DECAY_RATE 0.95f    // Attention decay per second
#define WORKSPACE_MIN_ATTENTION 0.01f // Minimum attention threshold

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Active vote tracking
 */
typedef struct {
    vote_proposal_t proposal;         // The proposal being voted on
    uint32_t votes_for;               // Count of approve votes
    uint32_t votes_against;           // Count of reject votes
    uint32_t votes_abstain;           // Count of abstain votes
    uint64_t expiry_ms;               // When vote expires
    bool active;                      // Vote is active
    bool completed;                   // Vote has completed
    vote_decision_t result;           // Final result
} active_vote_t;

/**
 * @brief Collective workspace implementation
 */
typedef struct {
    workspace_entry_t* entries;       // Workspace entries
    uint32_t size;                    // Number of entries
    uint64_t last_update_ms;          // Last update time
    float coherence;                  // Overall coherence
    nimcp_mutex_t* lock;              // Thread safety
} collective_workspace_t;

/**
 * @brief Emergence context
 */
typedef struct {
    swarm_emergence_tier_t current_tier;  // Current tier
    uint32_t tier_change_count;           // Number of tier changes
    uint64_t tier_enter_time_ms;          // Time entered current tier
    float coherence_history[10];          // Recent coherence values
    uint32_t coherence_index;             // Circular buffer index
} emergence_context_t;

/**
 * @brief Consensus context
 */
typedef struct {
    active_vote_t votes[MAX_ACTIVE_VOTES]; // Active votes
    uint32_t vote_count;                   // Number of active votes
    uint32_t total_votes_completed;        // Total completed
    nimcp_mutex_t* lock;                   // Thread safety
} consensus_context_t;

/**
 * @brief Swarm brain implementation
 */
struct nimcp_swarm_brain {
    // Configuration
    swarm_brain_config_t config;

    // Core components
    struct brain* local_brain;                    // Local constrained brain
    nimcp_swarm_signal_adapter_t* signal_adapter; // Radio I/O
    collective_workspace_t* workspace;            // Shared workspace
    emergence_context_t* emergence;               // Emergence tracking
    consensus_context_t* consensus;               // Voting system

    // Peer tracking
    swarm_peer_info_t peers[SWARM_MAX_PEERS];    // Active peers
    uint32_t peer_count;                          // Number of active peers
    nimcp_mutex_t* peer_lock;                     // Peer list lock

    // State
    bool joined;                                  // Has joined swarm
    bool operational;                             // System is operational
    uint64_t creation_time_ms;                    // Creation timestamp
    uint64_t last_heartbeat_ms;                   // Last heartbeat sent
    uint64_t last_sync_ms;                        // Last neuromod sync

    // Statistics
    swarm_stats_t stats;                          // Runtime statistics
    nimcp_mutex_t* stats_lock;                    // Stats lock

    // Bio-async (optional)
    struct nimcp_bio_async_module_ctx* bio_ctx;   // Bio-async context
    bool bio_async_enabled;                       // Bio-async enabled

    // Thread safety
    nimcp_mutex_t* state_lock;                    // Overall state lock
};

//=============================================================================
// Forward Declarations
//=============================================================================

static bool process_incoming_messages(swarm_brain_t* swarm);
static bool send_heartbeat(swarm_brain_t* swarm);
static bool update_peers(swarm_brain_t* swarm);
static bool update_workspace(swarm_brain_t* swarm);
static bool process_votes(swarm_brain_t* swarm);
static bool update_emergence_tier(swarm_brain_t* swarm);
static void handle_message(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);
static void handle_heartbeat(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);
static void handle_perception(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);
static void handle_threat(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);
static void handle_vote_propose(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);
static void handle_vote_cast(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);
static void handle_neuromod_sync(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);
static void handle_workspace_update(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);
static void handle_goodbye(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get current time in milliseconds
 */
static uint64_t get_time_ms(void) {
    return nimcp_time_get_ms();
}

/**
 * @brief Simple random number generator (0-max)
 */
static uint32_t simple_rand(uint32_t max) {
    static uint32_t seed = 12345;
    seed = (1103515245 * seed + 12345) & 0x7fffffff;
    return seed % (max + 1);
}

/**
 * @brief Find peer by ID
 */
static swarm_peer_info_t* find_peer(swarm_brain_t* swarm, uint16_t drone_id) {
    if (!swarm) return NULL;

    for (uint32_t i = 0; i < swarm->peer_count; i++) {
        if (swarm->peers[i].active && swarm->peers[i].drone_id == drone_id) {
            return &swarm->peers[i];
        }
    }
    return NULL;
}

/**
 * @brief Add or update peer
 */
static bool update_peer(swarm_brain_t* swarm, uint16_t drone_id) {
    if (!swarm) return false;

    nimcp_platform_mutex_lock(swarm->peer_lock);

    swarm_peer_info_t* peer = find_peer(swarm, drone_id);
    if (peer) {
        // Update existing peer
        peer->last_seen_ms = get_time_ms();
        peer->message_count++;
    } else {
        // Add new peer if space available
        if (swarm->peer_count < SWARM_MAX_PEERS) {
            peer = &swarm->peers[swarm->peer_count];
            peer->drone_id = drone_id;
            peer->last_seen_ms = get_time_ms();
            peer->coherence = 0.5f;
            peer->message_count = 1;
            peer->active = true;
            swarm->peer_count++;

            LOG_INFO("New peer joined: drone_id=%u, total_peers=%u",
                     drone_id, swarm->peer_count);
        } else {
            LOG_WARN("Peer list full, cannot add drone_id=%u", drone_id);
            nimcp_platform_mutex_unlock(swarm->peer_lock);
            return false;
        }
    }

    nimcp_platform_mutex_unlock(swarm->peer_lock);
    return true;
}

/**
 * @brief Remove inactive peers
 */
static void remove_inactive_peers(swarm_brain_t* swarm) {
    if (!swarm) return;

    uint64_t now = get_time_ms();
    nimcp_platform_mutex_lock(swarm->peer_lock);

    uint32_t removed = 0;
    for (uint32_t i = 0; i < swarm->peer_count; ) {
        if (swarm->peers[i].active &&
            (now - swarm->peers[i].last_seen_ms) > PEER_TIMEOUT_MS) {
            LOG_INFO("Peer timeout: drone_id=%u", swarm->peers[i].drone_id);
            swarm->peers[i].active = false;
            removed++;

            // Compact array by moving last element here
            if (i < swarm->peer_count - 1) {
                swarm->peers[i] = swarm->peers[swarm->peer_count - 1];
            }
            swarm->peer_count--;
        } else {
            i++;
        }
    }

    if (removed > 0) {
        LOG_DEBUG("Removed %u inactive peers, remaining=%u", removed, swarm->peer_count);
    }

    nimcp_platform_mutex_unlock(swarm->peer_lock);
}

//=============================================================================
// Collective Workspace Functions
//=============================================================================

/**
 * @brief Create collective workspace
 */
static collective_workspace_t* create_workspace(uint32_t size) {
    collective_workspace_t* ws = (collective_workspace_t*)nimcp_malloc(sizeof(collective_workspace_t));
    if (!ws) return NULL;

    ws->entries = (workspace_entry_t*)nimcp_calloc(size, sizeof(workspace_entry_t));
    if (!ws->entries) {
        nimcp_free(ws);
        return NULL;
    }

    ws->size = size;
    ws->last_update_ms = get_time_ms();
    ws->coherence = 0.0f;

    // Allocate and initialize mutex
    ws->lock = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!ws->lock || nimcp_platform_mutex_init(ws->lock, false) != 0) {
        if (ws->lock) nimcp_free(ws->lock);
        nimcp_free(ws->entries);
        nimcp_free(ws);
        return NULL;
    }

    return ws;
}

/**
 * @brief Destroy collective workspace
 */
static void destroy_workspace(collective_workspace_t* ws) {
    if (!ws) return;

    if (ws->lock) {
        nimcp_platform_mutex_destroy(ws->lock);
        nimcp_free(ws->lock);
    }
    if (ws->entries) nimcp_free(ws->entries);
    nimcp_free(ws);
}

/**
 * @brief Update workspace entry
 */
static void workspace_update_entry(collective_workspace_t* ws, uint32_t concept_id, float attention) {
    if (!ws || !ws->entries) return;

    nimcp_platform_mutex_lock(ws->lock);

    // Find or create entry
    workspace_entry_t* entry = NULL;
    for (uint32_t i = 0; i < ws->size; i++) {
        if (ws->entries[i].concept_id == concept_id) {
            entry = &ws->entries[i];
            break;
        }
        if (ws->entries[i].attention < WORKSPACE_MIN_ATTENTION && !entry) {
            // Reuse low-attention entry
            entry = &ws->entries[i];
        }
    }

    if (entry) {
        entry->concept_id = concept_id;
        entry->attention = fminf(entry->attention + attention, 1.0f);
        entry->contributor_count++;
        entry->last_update_ms = get_time_ms();
    }

    ws->last_update_ms = get_time_ms();

    nimcp_platform_mutex_unlock(ws->lock);
}

/**
 * @brief Decay workspace attention over time
 */
static void workspace_decay(collective_workspace_t* ws) {
    if (!ws || !ws->entries) return;

    uint64_t now = get_time_ms();
    nimcp_platform_mutex_lock(ws->lock);

    float dt = (now - ws->last_update_ms) / 1000.0f; // Convert to seconds
    float decay = powf(WORKSPACE_DECAY_RATE, dt);

    for (uint32_t i = 0; i < ws->size; i++) {
        ws->entries[i].attention *= decay;
        if (ws->entries[i].attention < WORKSPACE_MIN_ATTENTION) {
            ws->entries[i].attention = 0.0f;
            ws->entries[i].contributor_count = 0;
        }
    }

    ws->last_update_ms = now;

    nimcp_platform_mutex_unlock(ws->lock);
}

/**
 * @brief Calculate workspace coherence
 */
static float workspace_calculate_coherence(collective_workspace_t* ws) {
    if (!ws || !ws->entries) return 0.0f;

    nimcp_platform_mutex_lock(ws->lock);

    float total_attention = 0.0f;
    float max_attention = 0.0f;

    for (uint32_t i = 0; i < ws->size; i++) {
        total_attention += ws->entries[i].attention;
        if (ws->entries[i].attention > max_attention) {
            max_attention = ws->entries[i].attention;
        }
    }

    // Coherence is ratio of max attention to total (high = focused)
    float coherence = (total_attention > 0.0f) ? (max_attention / total_attention) : 0.0f;
    ws->coherence = coherence;

    nimcp_platform_mutex_unlock(ws->lock);

    return coherence;
}

//=============================================================================
// Emergence Functions
//=============================================================================

/**
 * @brief Create emergence context
 */
static emergence_context_t* create_emergence_context(void) {
    emergence_context_t* ctx = (emergence_context_t*)nimcp_calloc(1, sizeof(emergence_context_t));
    if (!ctx) return NULL;

    ctx->current_tier = SWARM_TIER_0_DISCONNECTED;
    ctx->tier_enter_time_ms = get_time_ms();
    return ctx;
}

/**
 * @brief Destroy emergence context
 */
static void destroy_emergence_context(emergence_context_t* ctx) {
    if (ctx) nimcp_free(ctx);
}

/**
 * @brief Update emergence tier based on peer count and coherence
 */
static void emergence_update_tier(emergence_context_t* ctx, uint32_t peer_count, float coherence) {
    if (!ctx) return;

    swarm_emergence_tier_t old_tier = ctx->current_tier;
    swarm_emergence_tier_t new_tier = SWARM_TIER_0_DISCONNECTED;

    // Determine tier based on peer count
    if (peer_count >= 16) {
        new_tier = SWARM_TIER_4_SUPERORGANISM;
    } else if (peer_count >= 8) {
        new_tier = SWARM_TIER_3_SWARM;
    } else if (peer_count >= 4) {
        new_tier = SWARM_TIER_2_CLUSTER;
    } else if (peer_count >= 2) {
        new_tier = SWARM_TIER_1_PAIRED;
    } else {
        new_tier = SWARM_TIER_0_DISCONNECTED;
    }

    // Require high coherence for higher tiers
    if (new_tier >= SWARM_TIER_3_SWARM && coherence < 0.4f) {
        new_tier = SWARM_TIER_2_CLUSTER;
    }

    // Update if tier changed
    if (new_tier != old_tier) {
        ctx->current_tier = new_tier;
        ctx->tier_change_count++;
        ctx->tier_enter_time_ms = get_time_ms();

        LOG_INFO("Emergence tier changed: %s -> %s (peers=%u, coherence=%.3f)",
                 swarm_emergence_tier_string(old_tier),
                 swarm_emergence_tier_string(new_tier),
                 peer_count, coherence);
    }

    // Update coherence history
    ctx->coherence_history[ctx->coherence_index] = coherence;
    ctx->coherence_index = (ctx->coherence_index + 1) % 10;
}

//=============================================================================
// Consensus Functions
//=============================================================================

/**
 * @brief Create consensus context
 */
static consensus_context_t* create_consensus_context(void) {
    consensus_context_t* ctx = (consensus_context_t*)nimcp_calloc(1, sizeof(consensus_context_t));
    if (!ctx) return NULL;

    ctx->lock = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!ctx->lock || nimcp_platform_mutex_init(ctx->lock, false) != 0) {
        if (ctx->lock) nimcp_free(ctx->lock);
        nimcp_free(ctx);
        return NULL;
    }

    return ctx;
}

/**
 * @brief Destroy consensus context
 */
static void destroy_consensus_context(consensus_context_t* ctx) {
    if (!ctx) return;
    if (ctx->lock) {
        nimcp_platform_mutex_destroy(ctx->lock);
        nimcp_free(ctx->lock);
    }
    nimcp_free(ctx);
}

/**
 * @brief Start new vote
 */
static bool consensus_start_vote(consensus_context_t* ctx, const vote_proposal_t* proposal) {
    if (!ctx || !proposal) return false;

    nimcp_platform_mutex_lock(ctx->lock);

    // Find free vote slot
    active_vote_t* vote = NULL;
    for (uint32_t i = 0; i < MAX_ACTIVE_VOTES; i++) {
        if (!ctx->votes[i].active) {
            vote = &ctx->votes[i];
            break;
        }
    }

    if (!vote) {
        LOG_WARN("No free vote slots available");
        nimcp_platform_mutex_unlock(ctx->lock);
        return false;
    }

    // Initialize vote
    vote->proposal = *proposal;
    vote->votes_for = 0;
    vote->votes_against = 0;
    vote->votes_abstain = 0;
    vote->expiry_ms = proposal->expiry_ms;
    vote->active = true;
    vote->completed = false;
    ctx->vote_count++;

    LOG_DEBUG("Started vote: proposal_id=%u, action_type=%u",
              proposal->proposal_id, proposal->action_type);

    nimcp_platform_mutex_unlock(ctx->lock);
    return true;
}

/**
 * @brief Cast vote on proposal
 */
static bool consensus_cast_vote(consensus_context_t* ctx, uint32_t proposal_id, vote_decision_t decision) {
    if (!ctx) return false;

    nimcp_platform_mutex_lock(ctx->lock);

    // Find vote
    active_vote_t* vote = NULL;
    for (uint32_t i = 0; i < MAX_ACTIVE_VOTES; i++) {
        if (ctx->votes[i].active && ctx->votes[i].proposal.proposal_id == proposal_id) {
            vote = &ctx->votes[i];
            break;
        }
    }

    if (!vote) {
        nimcp_platform_mutex_unlock(ctx->lock);
        return false;
    }

    // Record vote
    switch (decision) {
        case VOTE_APPROVE:
            vote->votes_for++;
            break;
        case VOTE_REJECT:
            vote->votes_against++;
            break;
        case VOTE_ABSTAIN:
            vote->votes_abstain++;
            break;
    }

    nimcp_platform_mutex_unlock(ctx->lock);
    return true;
}

/**
 * @brief Process active votes (check timeouts, quorum)
 */
static void consensus_process_votes(consensus_context_t* ctx, uint32_t peer_count) {
    if (!ctx) return;

    uint64_t now = get_time_ms();
    nimcp_platform_mutex_lock(ctx->lock);

    for (uint32_t i = 0; i < MAX_ACTIVE_VOTES; i++) {
        active_vote_t* vote = &ctx->votes[i];
        if (!vote->active || vote->completed) continue;

        uint32_t total_votes = vote->votes_for + vote->votes_against + vote->votes_abstain;
        uint32_t quorum = (uint32_t)(peer_count * VOTE_QUORUM_RATIO);

        // Check timeout
        if (now >= vote->expiry_ms) {
            vote->completed = true;
            vote->active = false;
            ctx->vote_count--;
            ctx->total_votes_completed++;

            LOG_INFO("Vote expired: proposal_id=%u, votes=%u/%u",
                     vote->proposal.proposal_id, total_votes, peer_count);
            continue;
        }

        // Check quorum
        if (total_votes >= quorum) {
            vote->completed = true;
            vote->active = false;
            ctx->vote_count--;
            ctx->total_votes_completed++;

            // Determine result
            if (vote->votes_for > vote->votes_against) {
                vote->result = VOTE_APPROVE;
            } else if (vote->votes_against > vote->votes_for) {
                vote->result = VOTE_REJECT;
            } else {
                vote->result = VOTE_ABSTAIN;
            }

            LOG_INFO("Vote completed: proposal_id=%u, result=%d, for=%u, against=%u",
                     vote->proposal.proposal_id, vote->result,
                     vote->votes_for, vote->votes_against);
        }
    }

    nimcp_platform_mutex_unlock(ctx->lock);
}

//=============================================================================
// Message Handlers
//=============================================================================

static void handle_heartbeat(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm || !data) return;

    // Heartbeat just updates peer presence
    update_peer(swarm, source_id);
}

static void handle_perception(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm || !data || len < sizeof(perception_data_t)) return;

    perception_data_t* perception = (perception_data_t*)data;
    update_peer(swarm, source_id);

    LOG_DEBUG("Received perception from drone %u: sensor_type=%u, confidence=%.3f",
              source_id, perception->sensor_type, perception->confidence);

    // TODO: Integrate perception into local brain or workspace
}

static void handle_threat(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm || !data || len < sizeof(threat_data_t)) return;

    threat_data_t* threat = (threat_data_t*)data;
    update_peer(swarm, source_id);

    LOG_WARN("THREAT from drone %u: type=%u, severity=%.3f, desc=%s",
             source_id, threat->threat_type, threat->severity, threat->description);

    // TODO: Urgent processing - alert local brain, update workspace
}

static void handle_vote_propose(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm || !data || len < sizeof(vote_proposal_t)) return;

    vote_proposal_t* proposal = (vote_proposal_t*)data;
    update_peer(swarm, source_id);

    LOG_DEBUG("Vote proposal from drone %u: proposal_id=%u, action_type=%u",
              source_id, proposal->proposal_id, proposal->action_type);

    consensus_start_vote(swarm->consensus, proposal);

    // TODO: Evaluate proposal locally and cast vote
}

static void handle_vote_cast(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm || !data || len < 8) return;

    uint32_t proposal_id = *(uint32_t*)data;
    vote_decision_t decision = *(vote_decision_t*)(data + 4);
    update_peer(swarm, source_id);

    consensus_cast_vote(swarm->consensus, proposal_id, decision);
}

static void handle_neuromod_sync(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm || !data || len < sizeof(neuromod_state_t)) return;

    neuromod_state_t* state = (neuromod_state_t*)data;
    update_peer(swarm, source_id);

    LOG_DEBUG("Neuromod sync from drone %u: DA=%.3f, 5HT=%.3f, NE=%.3f, ACh=%.3f",
              source_id, state->dopamine, state->serotonin,
              state->norepinephrine, state->acetylcholine);

    // TODO: Integrate neuromodulator state with local brain
}

static void handle_workspace_update(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm || !data || len < 8) return;

    uint32_t concept_id = *(uint32_t*)data;
    float attention = *(float*)(data + 4);
    update_peer(swarm, source_id);

    workspace_update_entry(swarm->workspace, concept_id, attention * 0.5f); // Weight by 0.5 for peer input
}

static void handle_goodbye(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm) return;

    LOG_INFO("Peer leaving: drone_id=%u", source_id);

    // Mark peer as inactive immediately
    nimcp_platform_mutex_lock(swarm->peer_lock);
    swarm_peer_info_t* peer = find_peer(swarm, source_id);
    if (peer) {
        peer->active = false;
        // Remove from array
        for (uint32_t i = 0; i < swarm->peer_count; i++) {
            if (swarm->peers[i].drone_id == source_id) {
                if (i < swarm->peer_count - 1) {
                    swarm->peers[i] = swarm->peers[swarm->peer_count - 1];
                }
                swarm->peer_count--;
                break;
            }
        }
    }
    nimcp_platform_mutex_unlock(swarm->peer_lock);
}

/**
 * @brief Main message dispatcher
 */
static void handle_message(swarm_brain_t* swarm, uint8_t* data, uint32_t len, uint32_t source_id) {
    if (!swarm || !data || len < 1) return;

    swarm_message_type_t msg_type = (swarm_message_type_t)data[0];
    uint8_t* payload = data + 1;
    uint32_t payload_len = len - 1;

    // Dispatch based on message type
    switch (msg_type) {
        case SWARM_MSG_HEARTBEAT:
            handle_heartbeat(swarm, payload, payload_len, source_id);
            break;
        case SWARM_MSG_PERCEPTION:
            handle_perception(swarm, payload, payload_len, source_id);
            break;
        case SWARM_MSG_THREAT:
            handle_threat(swarm, payload, payload_len, source_id);
            break;
        case SWARM_MSG_VOTE_PROPOSE:
            handle_vote_propose(swarm, payload, payload_len, source_id);
            break;
        case SWARM_MSG_VOTE_CAST:
            handle_vote_cast(swarm, payload, payload_len, source_id);
            break;
        case SWARM_MSG_NEUROMOD_SYNC:
            handle_neuromod_sync(swarm, payload, payload_len, source_id);
            break;
        case SWARM_MSG_WORKSPACE_UPDATE:
            handle_workspace_update(swarm, payload, payload_len, source_id);
            break;
        case SWARM_MSG_GOODBYE:
            handle_goodbye(swarm, payload, payload_len, source_id);
            break;
        default:
            LOG_WARN("Unknown message type: %u from drone %u", msg_type, source_id);
            break;
    }

    // Update stats
    nimcp_platform_mutex_lock(swarm->stats_lock);
    swarm->stats.messages_received++;
    nimcp_platform_mutex_unlock(swarm->stats_lock);
}

//=============================================================================
// Processing Functions
//=============================================================================

static bool process_incoming_messages(swarm_brain_t* swarm) {
    if (!swarm || !swarm->signal_adapter) return false;

    uint8_t buffer[SWARM_MAX_MESSAGE_SIZE];
    uint32_t received_len;
    uint32_t source_id;

    // Process all available messages (non-blocking)
    while (swarm_signal_receive(swarm->signal_adapter, buffer, SWARM_MAX_MESSAGE_SIZE,
                                &received_len, &source_id)) {
        if (received_len > 0 && source_id != swarm->config.drone_id) {
            handle_message(swarm, buffer, received_len, source_id);
        }
    }

    return true;
}

static bool send_heartbeat(swarm_brain_t* swarm) {
    if (!swarm || !swarm->signal_adapter) return false;

    uint64_t now = get_time_ms();
    uint32_t jitter = simple_rand(HEARTBEAT_JITTER_MS);

    if ((now - swarm->last_heartbeat_ms) < (swarm->config.heartbeat_ms + jitter)) {
        return true; // Not time yet
    }

    uint8_t message[2];
    message[0] = SWARM_MSG_HEARTBEAT;
    message[1] = 0; // No payload

    bool success = swarm_signal_broadcast(swarm->signal_adapter, message, 2);
    if (success) {
        swarm->last_heartbeat_ms = now;

        nimcp_platform_mutex_lock(swarm->stats_lock);
        swarm->stats.messages_sent++;
        nimcp_platform_mutex_unlock(swarm->stats_lock);
    }

    return success;
}

static bool update_peers(swarm_brain_t* swarm) {
    if (!swarm) return false;

    remove_inactive_peers(swarm);

    nimcp_platform_mutex_lock(swarm->stats_lock);
    swarm->stats.peers_connected = swarm->peer_count;
    nimcp_platform_mutex_unlock(swarm->stats_lock);

    return true;
}

static bool update_workspace(swarm_brain_t* swarm) {
    if (!swarm || !swarm->workspace) return false;

    workspace_decay(swarm->workspace);
    float coherence = workspace_calculate_coherence(swarm->workspace);

    nimcp_platform_mutex_lock(swarm->stats_lock);
    swarm->stats.workspace_coherence = coherence;
    nimcp_platform_mutex_unlock(swarm->stats_lock);

    return true;
}

static bool process_votes(swarm_brain_t* swarm) {
    if (!swarm || !swarm->consensus) return false;

    consensus_process_votes(swarm->consensus, swarm->peer_count);

    nimcp_platform_mutex_lock(swarm->stats_lock);
    swarm->stats.votes_completed = swarm->consensus->total_votes_completed;
    nimcp_platform_mutex_unlock(swarm->stats_lock);

    return true;
}

static bool update_emergence_tier(swarm_brain_t* swarm) {
    if (!swarm || !swarm->emergence) return false;

    float coherence = swarm->workspace ? swarm->workspace->coherence : 0.0f;
    emergence_update_tier(swarm->emergence, swarm->peer_count, coherence);

    nimcp_platform_mutex_lock(swarm->stats_lock);
    swarm->stats.emergence_tier_changes = swarm->emergence->tier_change_count;
    nimcp_platform_mutex_unlock(swarm->stats_lock);

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

swarm_brain_config_t swarm_brain_default_config(void) {
    swarm_brain_config_t config;
    memset(&config, 0, sizeof(config));

    config.drone_id = 1;
    strncpy(config.swarm_name, "default_swarm", SWARM_MAX_NAME_LEN - 1);
    config.heartbeat_ms = SWARM_DEFAULT_HEARTBEAT_MS;
    config.sync_ms = SWARM_DEFAULT_SYNC_MS;
    config.vote_timeout_ms = SWARM_DEFAULT_VOTE_TIMEOUT_MS;
    config.coherence_threshold = SWARM_DEFAULT_COHERENCE_THRESHOLD;
    config.critical_mass = SWARM_DEFAULT_CRITICAL_MASS;
    config.workspace_size = SWARM_DEFAULT_WORKSPACE_SIZE;
    config.broadcast_threshold = SWARM_DEFAULT_BROADCAST_THRESHOLD;
    config.neuromod_diffusion = SWARM_DEFAULT_NEUROMOD_DIFFUSION;
    config.enable_reward_sharing = true;
    config.enable_bio_async = true;

    return config;
}

swarm_brain_t* swarm_brain_create(const swarm_brain_config_t* config) {
    if (!config) {
        LOG_ERROR("NULL configuration provided");
        return NULL;
    }

    LOG_INFO("Creating swarm brain: drone_id=%u, swarm=%s",
             config->drone_id, config->swarm_name);

    swarm_brain_t* swarm = (swarm_brain_t*)nimcp_calloc(1, sizeof(swarm_brain_t));
    if (!swarm) {
        LOG_ERROR("Failed to allocate swarm brain");
        return NULL;
    }

    // Copy configuration
    swarm->config = *config;
    swarm->creation_time_ms = get_time_ms();

    // Create and initialize mutexes
    swarm->state_lock = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    swarm->peer_lock = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    swarm->stats_lock = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));

    if (!swarm->state_lock || !swarm->peer_lock || !swarm->stats_lock) {
        LOG_ERROR("Failed to allocate mutexes");
        swarm_brain_destroy(swarm);
        return NULL;
    }

    if (nimcp_platform_mutex_init(swarm->state_lock, false) != 0 ||
        nimcp_platform_mutex_init(swarm->peer_lock, false) != 0 ||
        nimcp_platform_mutex_init(swarm->stats_lock, false) != 0) {
        LOG_ERROR("Failed to initialize mutexes");
        swarm_brain_destroy(swarm);
        return NULL;
    }

    // Create signal adapter (using simulation mode for now)
    swarm_signal_config_t signal_config = {
        .radio_type = SWARM_RADIO_SIMULATION,
        .max_packet_size = SWARM_MAX_MESSAGE_SIZE,
        .retry_count = 3,
        .timeout_ms = 50
    };
    swarm->signal_adapter = swarm_signal_adapter_create(&signal_config);
    if (!swarm->signal_adapter) {
        LOG_ERROR("Failed to create signal adapter");
        swarm_brain_destroy(swarm);
        return NULL;
    }

    // Create collective workspace
    swarm->workspace = create_workspace(config->workspace_size);
    if (!swarm->workspace) {
        LOG_ERROR("Failed to create workspace");
        swarm_brain_destroy(swarm);
        return NULL;
    }

    // Create emergence context
    swarm->emergence = create_emergence_context();
    if (!swarm->emergence) {
        LOG_ERROR("Failed to create emergence context");
        swarm_brain_destroy(swarm);
        return NULL;
    }

    // Create consensus context
    swarm->consensus = create_consensus_context();
    if (!swarm->consensus) {
        LOG_ERROR("Failed to create consensus context");
        swarm_brain_destroy(swarm);
        return NULL;
    }

    // TODO: Create constrained local brain (requires brain API)
    // For now, leave as NULL
    swarm->local_brain = NULL;

    // TODO: Setup bio-async if enabled
    if (config->enable_bio_async) {
        swarm->bio_async_enabled = true;
        // swarm->bio_ctx = setup bio-async...
    }

    swarm->operational = true;

    LOG_INFO("Swarm brain created successfully");
    return swarm;
}

void swarm_brain_destroy(swarm_brain_t* swarm) {
    if (!swarm) return;

    LOG_INFO("Destroying swarm brain: drone_id=%u", swarm->config.drone_id);

    // Leave swarm if joined
    if (swarm->joined) {
        swarm_brain_leave(swarm);
    }

    // Cleanup components
    if (swarm->consensus) destroy_consensus_context(swarm->consensus);
    if (swarm->emergence) destroy_emergence_context(swarm->emergence);
    if (swarm->workspace) destroy_workspace(swarm->workspace);
    if (swarm->signal_adapter) swarm_signal_adapter_destroy(swarm->signal_adapter);

    // TODO: Destroy local brain if created
    // TODO: Cleanup bio-async context

    // Destroy mutexes
    if (swarm->state_lock) {
        nimcp_platform_mutex_destroy(swarm->state_lock);
        nimcp_free(swarm->state_lock);
    }
    if (swarm->peer_lock) {
        nimcp_platform_mutex_destroy(swarm->peer_lock);
        nimcp_free(swarm->peer_lock);
    }
    if (swarm->stats_lock) {
        nimcp_platform_mutex_destroy(swarm->stats_lock);
        nimcp_free(swarm->stats_lock);
    }

    nimcp_free(swarm);
    LOG_INFO("Swarm brain destroyed");
}

bool swarm_brain_join(swarm_brain_t* swarm) {
    if (!swarm) return false;

    nimcp_platform_mutex_lock(swarm->state_lock);

    if (swarm->joined) {
        LOG_WARN("Already joined swarm");
        nimcp_platform_mutex_unlock(swarm->state_lock);
        return true;
    }

    LOG_INFO("Joining swarm: %s", swarm->config.swarm_name);

    // Send initial heartbeat to announce presence
    swarm->joined = true;
    swarm->last_heartbeat_ms = 0; // Force immediate heartbeat

    nimcp_platform_mutex_unlock(swarm->state_lock);

    send_heartbeat(swarm);

    LOG_INFO("Successfully joined swarm");
    return true;
}

bool swarm_brain_leave(swarm_brain_t* swarm) {
    if (!swarm) return false;

    nimcp_platform_mutex_lock(swarm->state_lock);

    if (!swarm->joined) {
        nimcp_platform_mutex_unlock(swarm->state_lock);
        return true;
    }

    LOG_INFO("Leaving swarm: %s", swarm->config.swarm_name);

    // Send goodbye message
    uint8_t message[2];
    message[0] = SWARM_MSG_GOODBYE;
    message[1] = 0;
    swarm_signal_broadcast(swarm->signal_adapter, message, 2);

    swarm->joined = false;

    nimcp_platform_mutex_unlock(swarm->state_lock);

    LOG_INFO("Successfully left swarm");
    return true;
}

bool swarm_brain_process(swarm_brain_t* swarm) {
    if (!swarm || !swarm->operational) return false;

    // Process incoming messages
    process_incoming_messages(swarm);

    // Send periodic heartbeat
    send_heartbeat(swarm);

    // Update peer list (remove timeouts)
    update_peers(swarm);

    // Update workspace (decay, coherence)
    update_workspace(swarm);

    // Process active votes
    process_votes(swarm);

    // Update emergence tier
    update_emergence_tier(swarm);

    // Update uptime stat
    nimcp_platform_mutex_lock(swarm->stats_lock);
    swarm->stats.uptime_ms = get_time_ms() - swarm->creation_time_ms;
    nimcp_platform_mutex_unlock(swarm->stats_lock);

    return true;
}

bool swarm_brain_broadcast_perception(swarm_brain_t* swarm, const perception_data_t* perception) {
    if (!swarm || !perception || !swarm->signal_adapter) return false;

    uint8_t message[SWARM_MAX_MESSAGE_SIZE];
    message[0] = SWARM_MSG_PERCEPTION;

    size_t payload_size = sizeof(perception_data_t);
    if (payload_size + 1 > SWARM_MAX_MESSAGE_SIZE) {
        LOG_ERROR("Perception data too large");
        return false;
    }

    memcpy(message + 1, perception, payload_size);

    bool success = swarm_signal_broadcast(swarm->signal_adapter, message, payload_size + 1);
    if (success) {
        nimcp_platform_mutex_lock(swarm->stats_lock);
        swarm->stats.messages_sent++;
        nimcp_platform_mutex_unlock(swarm->stats_lock);
    }

    return success;
}

bool swarm_brain_broadcast_threat(swarm_brain_t* swarm, const threat_data_t* threat) {
    if (!swarm || !threat || !swarm->signal_adapter) return false;

    uint8_t message[SWARM_MAX_MESSAGE_SIZE];
    message[0] = SWARM_MSG_THREAT;

    size_t payload_size = sizeof(threat_data_t);
    if (payload_size + 1 > SWARM_MAX_MESSAGE_SIZE) {
        LOG_ERROR("Threat data too large");
        return false;
    }

    memcpy(message + 1, threat, payload_size);

    bool success = swarm_signal_broadcast(swarm->signal_adapter, message, payload_size + 1);
    if (success) {
        LOG_WARN("Broadcast threat: type=%u, severity=%.3f", threat->threat_type, threat->severity);

        nimcp_platform_mutex_lock(swarm->stats_lock);
        swarm->stats.messages_sent++;
        nimcp_platform_mutex_unlock(swarm->stats_lock);
    }

    return success;
}

bool swarm_brain_propose_action(swarm_brain_t* swarm, const vote_proposal_t* proposal) {
    if (!swarm || !proposal || !swarm->signal_adapter) return false;

    // Start local vote tracking
    if (!consensus_start_vote(swarm->consensus, proposal)) {
        return false;
    }

    // Broadcast proposal to swarm
    uint8_t message[SWARM_MAX_MESSAGE_SIZE];
    message[0] = SWARM_MSG_VOTE_PROPOSE;

    size_t payload_size = sizeof(vote_proposal_t);
    if (payload_size + 1 > SWARM_MAX_MESSAGE_SIZE) {
        LOG_ERROR("Vote proposal too large");
        return false;
    }

    memcpy(message + 1, proposal, payload_size);

    bool success = swarm_signal_broadcast(swarm->signal_adapter, message, payload_size + 1);
    if (success) {
        LOG_INFO("Proposed action: proposal_id=%u, action_type=%u",
                 proposal->proposal_id, proposal->action_type);

        nimcp_platform_mutex_lock(swarm->stats_lock);
        swarm->stats.messages_sent++;
        nimcp_platform_mutex_unlock(swarm->stats_lock);
    }

    return success;
}

bool swarm_brain_sync_neuromodulators(swarm_brain_t* swarm, const neuromod_state_t* local_state) {
    if (!swarm || !local_state || !swarm->signal_adapter) return false;

    uint64_t now = get_time_ms();
    uint32_t jitter = simple_rand(SYNC_JITTER_MS);

    if ((now - swarm->last_sync_ms) < (swarm->config.sync_ms + jitter)) {
        return true; // Not time yet
    }

    uint8_t message[SWARM_MAX_MESSAGE_SIZE];
    message[0] = SWARM_MSG_NEUROMOD_SYNC;

    size_t payload_size = sizeof(neuromod_state_t);
    if (payload_size + 1 > SWARM_MAX_MESSAGE_SIZE) {
        LOG_ERROR("Neuromod state too large");
        return false;
    }

    memcpy(message + 1, local_state, payload_size);

    bool success = swarm_signal_broadcast(swarm->signal_adapter, message, payload_size + 1);
    if (success) {
        swarm->last_sync_ms = now;

        nimcp_platform_mutex_lock(swarm->stats_lock);
        swarm->stats.messages_sent++;
        nimcp_platform_mutex_unlock(swarm->stats_lock);
    }

    return success;
}

swarm_emergence_tier_t swarm_brain_get_emergence_tier(const swarm_brain_t* swarm) {
    if (!swarm || !swarm->emergence) return SWARM_TIER_0_DISCONNECTED;
    return swarm->emergence->current_tier;
}

const workspace_entry_t* swarm_brain_get_workspace(const swarm_brain_t* swarm, uint32_t* workspace_size) {
    if (!swarm || !swarm->workspace || !workspace_size) return NULL;

    *workspace_size = swarm->workspace->size;
    return swarm->workspace->entries;
}

const swarm_peer_info_t* swarm_brain_get_peers(const swarm_brain_t* swarm, uint32_t* peer_count) {
    if (!swarm || !peer_count) return NULL;

    *peer_count = swarm->peer_count;
    return swarm->peers;
}

bool swarm_brain_get_stats(const swarm_brain_t* swarm, swarm_stats_t* stats) {
    if (!swarm || !stats) return false;

    nimcp_platform_mutex_lock(swarm->stats_lock);
    *stats = swarm->stats;
    nimcp_platform_mutex_unlock(swarm->stats_lock);

    return true;
}

brain_t swarm_brain_get_local_brain(swarm_brain_t* swarm) {
    if (!swarm) return NULL;
    return (brain_t)swarm->local_brain;
}

const char* swarm_emergence_tier_string(swarm_emergence_tier_t tier) {
    switch (tier) {
        case SWARM_TIER_0_DISCONNECTED: return "DISCONNECTED";
        case SWARM_TIER_1_PAIRED: return "PAIRED";
        case SWARM_TIER_2_CLUSTER: return "CLUSTER";
        case SWARM_TIER_3_SWARM: return "SWARM";
        case SWARM_TIER_4_SUPERORGANISM: return "SUPERORGANISM";
        default: return "UNKNOWN";
    }
}

const char* swarm_message_type_string(swarm_message_type_t msg_type) {
    switch (msg_type) {
        case SWARM_MSG_HEARTBEAT: return "HEARTBEAT";
        case SWARM_MSG_PERCEPTION: return "PERCEPTION";
        case SWARM_MSG_THREAT: return "THREAT";
        case SWARM_MSG_VOTE_PROPOSE: return "VOTE_PROPOSE";
        case SWARM_MSG_VOTE_CAST: return "VOTE_CAST";
        case SWARM_MSG_NEUROMOD_SYNC: return "NEUROMOD_SYNC";
        case SWARM_MSG_WORKSPACE_UPDATE: return "WORKSPACE_UPDATE";
        case SWARM_MSG_GOODBYE: return "GOODBYE";
        default: return "UNKNOWN";
    }
}

bool swarm_brain_is_operational(const swarm_brain_t* swarm) {
    return swarm && swarm->operational;
}

bool swarm_brain_reset_stats(swarm_brain_t* swarm) {
    if (!swarm) return false;

    nimcp_platform_mutex_lock(swarm->stats_lock);
    memset(&swarm->stats, 0, sizeof(swarm_stats_t));
    nimcp_platform_mutex_unlock(swarm->stats_lock);

    return true;
}
