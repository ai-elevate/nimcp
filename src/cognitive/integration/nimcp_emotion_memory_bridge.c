/**
 * @file nimcp_emotion_memory_bridge.c
 * @brief Emotion-Memory Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-31
 *
 * WHAT: Bidirectional integration between emotion and memory systems
 * WHY:  Emotional experiences enhance memory consolidation and retrieval.
 *       Memory retrieval can trigger emotional responses.
 * HOW:  Emotions tag memories with valence/arousal; retrieval recreates
 *       emotional context; emotional intensity modulates consolidation strength.
 *
 * BIOLOGICAL BASIS:
 * - Amygdala-hippocampus interactions enhance emotional memory encoding
 * - Emotional arousal releases norepinephrine, strengthening consolidation
 * - Memory retrieval reactivates associated emotional circuits
 *
 * @author NIMCP Development Team
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/integration/nimcp_emotion_memory_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for emotion_memory_bridge module */
static nimcp_health_agent_t* g_emotion_memory_bridge_health_agent = NULL;

/**
 * @brief Set health agent for emotion_memory_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void emotion_memory_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_emotion_memory_bridge_health_agent = agent;
}

/** @brief Send heartbeat from emotion_memory_bridge module */
static inline void emotion_memory_bridge_heartbeat(const char* operation, float progress) {
    if (g_emotion_memory_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_emotion_memory_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from emotion_memory_bridge module (instance-level) */
static inline void emotion_memory_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_emotion_memory_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_emotion_memory_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_emotion_memory_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


#define LOG_MODULE "EMOTION_MEMORY_BRIDGE"


/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define INITIAL_TAG_CAPACITY  256

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Internal emotional tag structure
 */
typedef struct emotional_tag {
    uint64_t memory_id;          /**< ID of tagged memory */
    float valence;               /**< Positive/negative [-1, 1] */
    float arousal;               /**< Activation level [0, 1] */
    uint64_t timestamp;          /**< When tag was applied */
    bool valid;                  /**< Whether this slot is in use */
} emotional_tag_t;

/**
 * @brief Emotion-Memory bridge internal structure
 */
struct emotion_memory_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */
    emotion_memory_config_t config;      /**< Bridge configuration */
    emotional_tag_t* tags;               /**< Array of emotional tags */
    size_t tag_capacity;                 /**< Allocated capacity */
    size_t tag_count;                    /**< Number of valid tags */
    emotion_memory_stats_t stats;        /**< Bridge statistics */
    nimcp_platform_mutex_t* mutex;       /**< Thread synchronization */
    bool initialized;                    /**< Initialization flag */
};

/* ============================================================================
 * Internal Helper Functions (unlocked)
 * ============================================================================ */

/**
 * @brief Find an emotional tag by memory_id (unlocked version)
 *
 * WHAT: Search for existing emotional tag
 * WHY:  Avoid duplicate tags for same memory
 * HOW:  Linear search through valid tags
 *
 * @param bridge Bridge to search
 * @param memory_id Memory ID to find
 * @return Pointer to tag or NULL if not found
 */
static emotional_tag_t* find_tag_unlocked(emotion_memory_bridge_t* bridge,
                                          uint64_t memory_id) {
    if (!bridge || !bridge->tags) {
        return NULL;
    }

    for (size_t i = 0; i < bridge->tag_capacity; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->tag_capacity > 256) {
            emotion_memory_bridge_heartbeat("emotion_memo_loop",
                             (float)(i + 1) / (float)bridge->tag_capacity);
        }

        if (bridge->tags[i].valid && bridge->tags[i].memory_id == memory_id) {
            return &bridge->tags[i];
        }
    }
    return NULL;
}

/**
 * @brief Find an empty slot for a new tag (unlocked version)
 *
 * WHAT: Find available slot in tags array
 * WHY:  Reuse freed slots before expanding
 * HOW:  Linear search for invalid slot
 *
 * @param bridge Bridge to search
 * @return Pointer to empty slot or NULL if full
 */
static emotional_tag_t* find_empty_slot_unlocked(emotion_memory_bridge_t* bridge) {
    if (!bridge || !bridge->tags) {
        return NULL;
    }

    for (size_t i = 0; i < bridge->tag_capacity; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->tag_capacity > 256) {
            emotion_memory_bridge_heartbeat("emotion_memo_loop",
                             (float)(i + 1) / (float)bridge->tag_capacity);
        }

        if (!bridge->tags[i].valid) {
            return &bridge->tags[i];
        }
    }
    return NULL;
}

/**
 * @brief Get current timestamp in milliseconds
 *
 * WHAT: Provide consistent timestamp source
 * WHY:  Track when tags were created
 * HOW:  Use clock_gettime if available
 *
 * @return Current time in milliseconds
 */
static uint64_t get_timestamp_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
    }
    return 0;
}

/**
 * @brief Clamp a float value to a range
 *
 * @param value Value to clamp
 * @param min Minimum value
 * @param max Maximum value
 * @return Clamped value
 */
static float clamp_float(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Update running averages in stats (unlocked version)
 *
 * WHAT: Update average valence/arousal statistics
 * WHY:  Track emotional distribution of tagged memories
 * HOW:  Incremental mean calculation
 *
 * @param bridge Bridge to update
 * @param valence New valence value
 * @param arousal New arousal value
 */
static void update_averages_unlocked(emotion_memory_bridge_t* bridge,
                                     float valence, float arousal) {
    if (!bridge || bridge->stats.memories_tagged == 0) {
        return;
    }

    /* Incremental mean: new_avg = old_avg + (new_value - old_avg) / n */
    uint64_t n = bridge->stats.memories_tagged;
    bridge->stats.avg_valence += (valence - bridge->stats.avg_valence) / (float)n;
    bridge->stats.avg_arousal += (arousal - bridge->stats.avg_arousal) / (float)n;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int emotion_memory_bridge_default_config(emotion_memory_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    emotion_memory_bridge_heartbeat("emotion_memo_default_config", 0.0f);


    config->emotional_weight_factor = 0.5f;
    config->consolidation_threshold = 0.3f;
    config->valence_sensitivity = 1.0f;

    return 0;
}

emotion_memory_bridge_t* emotion_memory_bridge_create(
    const emotion_memory_config_t* config
) {
    /* Allocate bridge structure */
    /* Phase 8: Heartbeat at operation start */
    emotion_memory_bridge_heartbeat("emotion_memo_create", 0.0f);


    emotion_memory_bridge_t* bridge = nimcp_calloc(1, sizeof(emotion_memory_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;
    }

    /* Initialize configuration */
    if (config) {
        bridge->config = *config;
    } else {
        emotion_memory_bridge_default_config(&bridge->config);
    }

    /* Allocate tags array */
    bridge->tags = nimcp_calloc(INITIAL_TAG_CAPACITY, sizeof(emotional_tag_t));
    if (!bridge->tags) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->tag_capacity = INITIAL_TAG_CAPACITY;
    bridge->tag_count = 0;

    /* Create mutex for thread safety */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        nimcp_free(bridge->tags);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize stats to zero (already done by calloc, but explicit) */
    memset(&bridge->stats, 0, sizeof(emotion_memory_stats_t));

    bridge->initialized = true;

    return bridge;
}

void emotion_memory_bridge_destroy(emotion_memory_bridge_t* bridge) {
    if (!bridge) {
        return;
        NIMCP_LOGGING_DEBUG("Destroying %s bridge", "emotion_memory");
    }

    /* Destroy mutex */
    /* Phase 8: Heartbeat at operation start */
    emotion_memory_bridge_heartbeat("emotion_memo_destroy", 0.0f);


    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        bridge->base.mutex = NULL;
    }

    /* Free tags array */
    if (bridge->tags) {
        nimcp_free(bridge->tags);
        bridge->tags = NULL;
    }

    bridge->initialized = false;

    /* Free bridge structure */
    nimcp_free(bridge);
}

/* ============================================================================
 * Emotion -> Memory Direction
 * ============================================================================ */

int emotion_memory_tag_memory(
    emotion_memory_bridge_t* bridge,
    uint64_t memory_id,
    float valence,
    float arousal
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    /* Clamp values to valid ranges */
    /* Phase 8: Heartbeat at operation start */
    emotion_memory_bridge_heartbeat("emotion_memo_emotion_memory_tag_m", 0.0f);


    valence = clamp_float(valence, EMOTION_MEMORY_VALENCE_MIN, EMOTION_MEMORY_VALENCE_MAX);
    arousal = clamp_float(arousal, EMOTION_MEMORY_AROUSAL_MIN, EMOTION_MEMORY_AROUSAL_MAX);

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Check if memory already has a tag */
    emotional_tag_t* tag = find_tag_unlocked(bridge, memory_id);

    if (tag) {
        /* Update existing tag */
        tag->valence = valence;
        tag->arousal = arousal;
        tag->timestamp = get_timestamp_ms();
    } else {
        /* Find empty slot */
        tag = find_empty_slot_unlocked(bridge);

        if (!tag) {
            /* Check capacity limit */
            if (bridge->tag_count >= EMOTION_MEMORY_MAX_MEMORIES) {
                nimcp_platform_mutex_unlock(bridge->base.mutex);
                return -1;
            }

            /* Need to expand array (if we have room under max) */
            size_t new_capacity = bridge->tag_capacity * 2;
            if (new_capacity > EMOTION_MEMORY_MAX_MEMORIES) {
                new_capacity = EMOTION_MEMORY_MAX_MEMORIES;
            }

            emotional_tag_t* new_tags = nimcp_realloc(
                bridge->tags,
                new_capacity * sizeof(emotional_tag_t)
            );
            if (!new_tags) {
                nimcp_platform_mutex_unlock(bridge->base.mutex);
                return -1;
            }

            /* Zero the new portion */
            memset(&new_tags[bridge->tag_capacity], 0,
                   (new_capacity - bridge->tag_capacity) * sizeof(emotional_tag_t));

            bridge->tags = new_tags;
            tag = &bridge->tags[bridge->tag_capacity];
            bridge->tag_capacity = new_capacity;
        }

        /* Initialize new tag */
        tag->memory_id = memory_id;
        tag->valence = valence;
        tag->arousal = arousal;
        tag->timestamp = get_timestamp_ms();
        tag->valid = true;

        bridge->tag_count++;
    }

    /* Update statistics */
    bridge->stats.memories_tagged++;
    update_averages_unlocked(bridge, valence, arousal);

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

int emotion_memory_modulate_consolidation(
    emotion_memory_bridge_t* bridge,
    uint64_t memory_id,
    float emotional_intensity
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    /* Clamp intensity to valid range */
    /* Phase 8: Heartbeat at operation start */
    emotion_memory_bridge_heartbeat("emotion_memo_emotion_memory_modul", 0.0f);


    emotional_intensity = clamp_float(emotional_intensity, 0.0f, 1.0f);

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Find the tag for this memory */
    emotional_tag_t* tag = find_tag_unlocked(bridge, memory_id);

    if (!tag) {
        /* Memory not tagged - still return success, just no boost */
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return 0;
    }

    /* Check if intensity exceeds consolidation threshold */
    if (emotional_intensity > bridge->config.consolidation_threshold) {
        /* Consolidation boost would be: 1.0 + (intensity * weight_factor) */
        /* The actual consolidation factor is computed by the caller using:
         * consolidation_factor = 1.0 + (intensity * config.emotional_weight_factor)
         * Here we just track the boost event */
        bridge->stats.consolidation_boosts++;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Memory -> Emotion Direction
 * ============================================================================ */

int emotion_memory_on_retrieval(
    emotion_memory_bridge_t* bridge,
    uint64_t memory_id,
    emotion_memory_emotion_out_t* emotion_out
) {
    if (!bridge || !bridge->initialized || !emotion_out) {
        return -1;
    }

    /* Initialize output */
    /* Phase 8: Heartbeat at operation start */
    emotion_memory_bridge_heartbeat("emotion_memo_emotion_memory_on_re", 0.0f);


    memset(emotion_out, 0, sizeof(emotion_memory_emotion_out_t));
    emotion_out->has_emotion = false;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Find the tag for this memory */
    emotional_tag_t* tag = find_tag_unlocked(bridge, memory_id);

    if (!tag) {
        /* No emotional tag for this memory - not an error, just no emotion */
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return 0;  /* Success - emotion_out initialized with has_emotion = false */
    }

    /* Compute emotional response scaled by sensitivity */
    emotion_out->valence = tag->valence * bridge->config.valence_sensitivity;
    emotion_out->arousal = tag->arousal * bridge->config.valence_sensitivity;

    /* Clamp output values */
    emotion_out->valence = clamp_float(emotion_out->valence,
                                       EMOTION_MEMORY_VALENCE_MIN,
                                       EMOTION_MEMORY_VALENCE_MAX);
    emotion_out->arousal = clamp_float(emotion_out->arousal,
                                       EMOTION_MEMORY_AROUSAL_MIN,
                                       EMOTION_MEMORY_AROUSAL_MAX);

    /* Compute intensity as magnitude of emotional response */
    emotion_out->intensity = sqrtf(emotion_out->valence * emotion_out->valence +
                                   emotion_out->arousal * emotion_out->arousal) / sqrtf(2.0f);
    emotion_out->has_emotion = true;

    /* Update statistics */
    bridge->stats.retrievals_with_emotion++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int emotion_memory_get_emotional_memories(
    emotion_memory_bridge_t* bridge,
    float valence_min,
    float valence_max,
    uint64_t memory_ids[],
    size_t max_count
) {
    if (!bridge || !bridge->initialized || !memory_ids || max_count == 0) {
        return -1;
    }

    /* Clamp range values */
    /* Phase 8: Heartbeat at operation start */
    emotion_memory_bridge_heartbeat("emotion_memo_emotion_memory_get_e", 0.0f);


    valence_min = clamp_float(valence_min, EMOTION_MEMORY_VALENCE_MIN, EMOTION_MEMORY_VALENCE_MAX);
    valence_max = clamp_float(valence_max, EMOTION_MEMORY_VALENCE_MIN, EMOTION_MEMORY_VALENCE_MAX);

    /* Ensure min <= max */
    if (valence_min > valence_max) {
        float temp = valence_min;
        valence_min = valence_max;
        valence_max = temp;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);

    size_t found_count = 0;

    /* Iterate through all tags */
    for (size_t i = 0; i < bridge->tag_capacity && found_count < max_count; i++) {
        if (bridge->tags[i].valid) {
            float v = bridge->tags[i].valence;
            if (v >= valence_min && v <= valence_max) {
                memory_ids[found_count] = bridge->tags[i].memory_id;
                found_count++;
            }
        }
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return (int)found_count;
}

/* ============================================================================
 * Stats API
 * ============================================================================ */

int emotion_memory_bridge_get_stats(
    const emotion_memory_bridge_t* bridge,
    emotion_memory_stats_t* stats
) {
    if (!bridge || !bridge->initialized || !stats) {
        return -1;
    }

    /* Cast away const for mutex lock (safe - mutex protects data, not bridge ptr) */
    /* Phase 8: Heartbeat at operation start */
    emotion_memory_bridge_heartbeat("emotion_memo_get_stats", 0.0f);


    emotion_memory_bridge_t* mutable_bridge = (emotion_memory_bridge_t*)bridge;

    nimcp_platform_mutex_lock(mutable_bridge->base.mutex);

    *stats = bridge->stats;

    nimcp_platform_mutex_unlock(mutable_bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void emotion_memory_bridge_set_instance_health_agent(emotion_memory_bridge_t* bridge, nimcp_health_agent_t* agent) {
    if (!bridge) {
        NIMCP_THROW(NIMCP_ERROR_NULL_POINTER,
                    "emotion_memory_bridge_set_instance_health_agent: NULL bridge");
        return;
    }
    bridge->health_agent = agent;
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int emotion_memory_bridge_training_begin(emotion_memory_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "emotion_memory_bridge_training_begin: NULL argument");
        return -1;
    }
    emotion_memory_bridge_heartbeat_instance(bridge->health_agent, "emotion_memory_bridge_training_begin", 0.0f);
    return 0;
}

int emotion_memory_bridge_training_end(emotion_memory_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "emotion_memory_bridge_training_end: NULL argument");
        return -1;
    }
    emotion_memory_bridge_heartbeat_instance(bridge->health_agent, "emotion_memory_bridge_training_end", 1.0f);
    return 0;
}

int emotion_memory_bridge_training_step(emotion_memory_bridge_t* bridge, float progress) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "emotion_memory_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    emotion_memory_bridge_heartbeat_instance(bridge->health_agent, "emotion_memory_bridge_training_step", progress);
    return 0;
}
