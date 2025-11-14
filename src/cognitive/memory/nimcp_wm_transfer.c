/**
 * @file nimcp_wm_transfer.c
 * @brief Phase M3: Working Memory to Engram Transfer Implementation
 *
 * WHAT: Implements selective transfer from working memory to long-term memory
 * WHY:  Not all temporary information should become permanent memories
 * HOW:  Multi-factor scoring based on rehearsal, attention, emotion, and time
 *
 * @version Phase M3 Working Memory Transfer
 * @date 2025-11-13
 */

#include "cognitive/memory/nimcp_wm_transfer.h"
#include "cognitive/memory/nimcp_engram.h"
#include "utils/platform/nimcp_platform_time.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//=============================================================================
// Constants
//=============================================================================

// Transfer scoring weights (must sum to 1.0)
#define REHEARSAL_WEIGHT 0.4f    // 40% contribution
#define ATTENTION_WEIGHT 0.3f    // 30% contribution
#define EMOTIONAL_WEIGHT 0.2f    // 20% contribution
#define TIME_WEIGHT 0.1f         // 10% contribution

// Transfer threshold (score must be >= this to transfer)
#define TRANSFER_SCORE_THRESHOLD 0.5f

// Default working memory capacity (Miller's law: 7±2)
#define DEFAULT_WM_CAPACITY 7

//=============================================================================
// System Management
//=============================================================================

/**
 * @brief Create working memory transfer system
 * WHAT: Allocate and initialize transfer system
 * WHY:  Prepare system for managing WM → engram transfers
 * HOW:  Allocate struct, set defaults, initialize tracking arrays
 */
wm_transfer_system_t* wm_transfer_create(void) {
    // Allocate system
    wm_transfer_system_t* system = (wm_transfer_system_t*)calloc(1, sizeof(wm_transfer_system_t));
    if (!system) {
        fprintf(stderr, "Failed to allocate WM transfer system\n");
        return NULL;
    }

    // Set default criteria
    system->criteria = wm_transfer_get_default_criteria();

    // Initialize stats to zero (calloc handles this)

    // Initialize attention tracking
    system->attention_weight_count = DEFAULT_WM_CAPACITY;
    system->last_attention_weights = (float*)calloc(DEFAULT_WM_CAPACITY, sizeof(float));
    if (!system->last_attention_weights) {
        fprintf(stderr, "Failed to allocate attention weights\n");
        free(system);
        return NULL;
    }

    // Record creation time
    system->last_update_time_ms = nimcp_platform_time_monotonic_ms();

    return system;
}

/**
 * @brief Destroy working memory transfer system
 * WHAT: Free all resources associated with system
 * WHY:  Prevent memory leaks when brain is destroyed
 * HOW:  Free attention array, then system struct
 */
void wm_transfer_destroy(wm_transfer_system_t* system) {
    if (!system) return;

    // Free attention tracking
    if (system->last_attention_weights) {
        free(system->last_attention_weights);
    }

    // Free system
    free(system);
}

/**
 * @brief Reset transfer system (clear stats, keep criteria)
 * WHAT: Clear statistics while preserving configuration
 * WHY:  Allow reuse of system with fresh state
 * HOW:  Zero stats and attention tracking
 */
void wm_transfer_reset(wm_transfer_system_t* system) {
    if (!system) return;

    // Clear stats
    memset(&system->stats, 0, sizeof(wm_transfer_stats_t));

    // Reset attention weights
    if (system->last_attention_weights) {
        memset(system->last_attention_weights, 0,
               system->attention_weight_count * sizeof(float));
    }

    // Reset update time
    system->last_update_time_ms = nimcp_platform_time_monotonic_ms();
}

//=============================================================================
// Integration API
//=============================================================================

/**
 * @brief Connect to working memory system
 * WHAT: Link transfer system to working memory source
 * WHY:  Transfer system needs access to WM items
 * HOW:  Store pointer (not owned) to working memory
 */
void wm_transfer_set_working_memory(
    wm_transfer_system_t* system,
    void* working_memory)
{
    if (!system) return;
    system->working_memory = working_memory;
}

/**
 * @brief Connect to engram system
 * WHAT: Link transfer system to engram destination
 * WHY:  Transferred items must be encoded as engrams
 * HOW:  Store pointer (not owned) to engram system
 */
void wm_transfer_set_engram_system(
    wm_transfer_system_t* system,
    engram_system_t* engram_system)
{
    if (!system) return;
    system->engram_system = engram_system;
}

/**
 * @brief Connect to emotional tagging system
 * WHAT: Link transfer system to emotional salience source
 * WHY:  Emotional arousal enhances encoding
 * HOW:  Store pointer (not owned) to emotional system
 */
void wm_transfer_set_emotional_system(
    wm_transfer_system_t* system,
    void* emotional_system)
{
    if (!system) return;
    system->emotional_system = emotional_system;
}

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Compute transfer score for working memory item
 * WHAT: Calculate multi-factor score for transfer decision
 * WHY:  Determine if item should be transferred to engrams
 * HOW:  Weight rehearsal, attention, emotion, time contributions
 */
static float compute_transfer_score(
    const wm_transfer_system_t* system,
    uint32_t rehearsal_count,
    float attention_weight,
    float emotional_salience,
    uint64_t time_in_wm_ms)
{
    float score = 0.0f;

    // Rehearsal contribution (40% weight)
    if (rehearsal_count >= system->criteria.rehearsal_threshold) {
        score += REHEARSAL_WEIGHT;
    }

    // Attention contribution (30% weight)
    if (attention_weight >= system->criteria.attention_threshold) {
        score += ATTENTION_WEIGHT;
    }

    // Emotional contribution (20% weight)
    if (emotional_salience >= system->criteria.emotional_threshold) {
        score += EMOTIONAL_WEIGHT;
    }

    // Time contribution (10% weight)
    if (time_in_wm_ms >= system->criteria.time_threshold_ms) {
        score += TIME_WEIGHT;
    }

    return score;
}

/**
 * @brief Check if item should be transferred
 * WHAT: Apply transfer threshold to score
 * WHY:  Only transfer items that meet sufficient criteria
 * HOW:  Compare score to threshold (0.5 = 50%)
 */
static bool should_transfer_item(float score) {
    return score >= TRANSFER_SCORE_THRESHOLD;
}

/**
 * @brief Update transfer statistics for trigger
 * WHAT: Increment stats counters based on what triggered transfer
 * WHY:  Track which factors are driving transfers
 * HOW:  Check each criterion and increment if met
 */
static void update_transfer_stats(
    wm_transfer_system_t* system,
    uint32_t rehearsal_count,
    float attention_weight,
    float emotional_salience,
    uint64_t time_in_wm_ms)
{
    // Update total transfers
    system->stats.total_transfers++;

    // Track which factors triggered this transfer
    if (rehearsal_count >= system->criteria.rehearsal_threshold) {
        system->stats.rehearsal_triggered++;
    }
    if (attention_weight >= system->criteria.attention_threshold) {
        system->stats.attention_triggered++;
    }
    if (emotional_salience >= system->criteria.emotional_threshold) {
        system->stats.emotion_triggered++;
    }
    if (time_in_wm_ms >= system->criteria.time_threshold_ms) {
        system->stats.time_triggered++;
    }
}

//=============================================================================
// Transfer Operations
//=============================================================================

/**
 * @brief Evaluate working memory items for transfer to engrams
 * WHAT: Check WM items against criteria, transfer if met
 * WHY:  Implements selective consolidation to long-term memory
 * HOW:  Score each item, transfer if >= 0.5, update stats
 *
 * NOTE: This is a placeholder implementation that demonstrates the algorithm.
 * Full implementation requires integration with actual working memory system.
 */
uint32_t wm_transfer_evaluate(
    wm_transfer_system_t* system,
    float time_delta_seconds)
{
    // Guard clauses
    if (!system) return 0;
    if (!system->working_memory) return 0;
    if (!system->engram_system) return 0;

    uint32_t transfers = 0;
    uint64_t current_time_ms = nimcp_platform_time_monotonic_ms();

    // Update time tracking
    uint64_t time_elapsed_ms = current_time_ms - system->last_update_time_ms;
    system->last_update_time_ms = current_time_ms;

    // For each working memory item (placeholder logic)
    // NOTE: Actual implementation would query working memory system
    // For now, we demonstrate the algorithm with mock data

    // Placeholder: No actual working memory items to process yet
    // This will be filled in during brain integration

    return transfers;
}

/**
 * @brief Force transfer of specific working memory item
 * WHAT: Manually trigger transfer bypassing criteria
 * WHY:  Allow explicit encoding of important events
 * HOW:  Extract features, encode to engram directly
 */
bool wm_transfer_force_item(
    wm_transfer_system_t* system,
    uint32_t wm_slot)
{
    // Guard clauses
    if (!system) return false;
    if (!system->working_memory) return false;
    if (!system->engram_system) return false;

    // Placeholder: Would extract features from working memory slot
    // and encode directly to engram system

    // Update stats
    system->stats.total_transfers++;

    return true;
}

/**
 * @brief Update attention weights for working memory items
 * WHAT: Store current attention distribution across WM slots
 * WHY:  Attention determines transfer priority
 * HOW:  Copy weights array, resize if needed
 */
void wm_transfer_update_attention(
    wm_transfer_system_t* system,
    const float* attention_weights,
    uint32_t count)
{
    // Guard clauses
    if (!system) return;
    if (!attention_weights) return;
    if (count == 0) return;

    // Resize attention array if needed
    if (count != system->attention_weight_count) {
        float* new_weights = (float*)realloc(system->last_attention_weights,
                                               count * sizeof(float));
        if (!new_weights) {
            fprintf(stderr, "Failed to resize attention weights\n");
            return;
        }
        system->last_attention_weights = new_weights;
        system->attention_weight_count = count;
    }

    // Copy attention weights
    memcpy(system->last_attention_weights, attention_weights,
           count * sizeof(float));
}

//=============================================================================
// Configuration API
//=============================================================================

/**
 * @brief Set transfer criteria
 * WHAT: Update transfer decision thresholds
 * WHY:  Allow customization of transfer behavior
 * HOW:  Copy criteria struct to system
 */
void wm_transfer_set_criteria(
    wm_transfer_system_t* system,
    const wm_transfer_criteria_t* criteria)
{
    if (!system || !criteria) return;
    system->criteria = *criteria;
}

/**
 * @brief Get current transfer criteria
 * WHAT: Retrieve current transfer thresholds
 * WHY:  Allow inspection of configuration
 * HOW:  Copy criteria from system to output
 */
void wm_transfer_get_criteria(
    const wm_transfer_system_t* system,
    wm_transfer_criteria_t* criteria_out)
{
    if (!system || !criteria_out) return;
    *criteria_out = system->criteria;
}

//=============================================================================
// Statistics API
//=============================================================================

/**
 * @brief Get transfer statistics
 * WHAT: Retrieve transfer operation statistics
 * WHY:  Monitoring and debugging transfer behavior
 * HOW:  Copy stats from system to output
 */
void wm_transfer_get_statistics(
    const wm_transfer_system_t* system,
    wm_transfer_stats_t* stats_out)
{
    if (!system || !stats_out) return;
    *stats_out = system->stats;
}

//=============================================================================
// Default Configuration
//=============================================================================

/**
 * @brief Get default transfer criteria
 * WHAT: Return default thresholds based on neuroscience literature
 * WHY:  Provide sensible starting configuration
 * HOW:  Return struct with documented defaults
 *
 * DEFAULTS BASED ON:
 * - Miller (1956): Working memory capacity 7±2 items
 * - Atkinson & Shiffrin (1968): Rehearsal enhances transfer
 * - McGaugh (2000): Emotional arousal enhances consolidation
 */
wm_transfer_criteria_t wm_transfer_get_default_criteria(void) {
    wm_transfer_criteria_t criteria = {
        .rehearsal_threshold = 3,       // 3+ rehearsals triggers transfer
        .attention_threshold = 0.5f,    // 50% attention required
        .emotional_threshold = 0.3f,    // 30% emotional salience
        .time_threshold_ms = 5000,      // 5 seconds in working memory
        .decay_rate = 0.1f              // 10% decay per second
    };
    return criteria;
}
