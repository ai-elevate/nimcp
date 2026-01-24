/**
 * @file nimcp_curiosity_reasoning_bridge.c
 * @brief Curiosity-Reasoning Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-31
 *
 * WHAT: Bidirectional integration between curiosity and reasoning systems
 * WHY:  Curiosity drives exploration of reasoning space; novel conclusions
 *       generate curiosity; epistemic uncertainty guides inquiry.
 * HOW:  Curiosity level biases reasoning exploration; novel conclusions
 *       trigger curiosity signals; uncertainty is shared bidirectionally.
 *
 * BIOLOGICAL BASIS:
 * - Dopaminergic signals from VTA/SNc drive curiosity and exploration
 * - Prefrontal cortex integrates uncertainty with reasoning processes
 * - Information-seeking behavior optimizes epistemic value
 */

#include "cognitive/integration/nimcp_curiosity_reasoning_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Exploration topic tracking entry
 */
typedef struct exploration_topic {
    uint32_t topic_id;              /**< Topic identifier */
    float curiosity_level;          /**< Current curiosity level for topic */
    float uncertainty_level;        /**< Epistemic uncertainty level */
    float exploration_priority;     /**< Computed exploration priority */
    uint64_t last_exploration;      /**< Timestamp of last exploration */
    bool active;                    /**< Whether topic is actively tracked */
} exploration_topic_t;

/**
 * @brief Internal bridge structure
 */
struct curiosity_reasoning_bridge {
    bridge_base_t base;                     /**< MUST be first: base bridge infrastructure */
    curiosity_reasoning_config_t config;    /**< Bridge configuration */
    exploration_topic_t* topics;            /**< Topic tracking array */
    size_t topic_capacity;                  /**< Maximum topics */
    size_t topic_count;                     /**< Current topic count */
    curiosity_reasoning_stats_t stats;      /**< Bridge statistics */
    bool initialized;                       /**< Initialization flag */
};

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define DEFAULT_TOPIC_CAPACITY  128
#define CURIOSITY_BOOST_FACTOR  0.15f

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * WHAT: Clamp float to range
 * WHY:  Prevent numerical overflow/underflow
 * HOW:  Return min/max if out of bounds
 */
static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * WHAT: Find topic by ID
 * WHY:  Locate existing topic entry
 * HOW:  Linear search through active topics
 *
 * @note Caller must hold mutex
 */
static exploration_topic_t* find_topic_unlocked(
    curiosity_reasoning_bridge_t* bridge,
    uint32_t topic_id
) {
    for (size_t i = 0; i < bridge->topic_count; i++) {
        if (bridge->topics[i].active && bridge->topics[i].topic_id == topic_id) {
            return &bridge->topics[i];
        }
    }
    return NULL;
}

/**
 * WHAT: Find or create topic by ID
 * WHY:  Ensure topic entry exists for operations
 * HOW:  Search existing, create new if not found
 *
 * @note Caller must hold mutex
 */
static exploration_topic_t* find_or_create_topic_unlocked(
    curiosity_reasoning_bridge_t* bridge,
    uint32_t topic_id
) {
    /* Search existing */
    exploration_topic_t* topic = find_topic_unlocked(bridge, topic_id);
    if (topic) {
        return topic;
    }

    /* Check capacity */
    if (bridge->topic_count >= bridge->topic_capacity) {
        return NULL;  /* At capacity */
    }

    /* Find first inactive slot or use next slot */
    for (size_t i = 0; i < bridge->topic_capacity; i++) {
        if (!bridge->topics[i].active) {
            bridge->topics[i].topic_id = topic_id;
            bridge->topics[i].curiosity_level = 0.0f;
            bridge->topics[i].uncertainty_level = 0.0f;
            bridge->topics[i].exploration_priority = 0.0f;
            bridge->topics[i].last_exploration = 0;
            bridge->topics[i].active = true;
            bridge->topic_count++;
            return &bridge->topics[i];
        }
    }

    return NULL;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int curiosity_reasoning_bridge_default_config(curiosity_reasoning_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;
    }

    config->exploration_bias = 0.7f;
    config->novelty_threshold = 0.5f;
    config->uncertainty_weight = 0.3f;

    return 0;
}

curiosity_reasoning_bridge_t* curiosity_reasoning_bridge_create(
    const curiosity_reasoning_config_t* config
) {
    /* Allocate bridge structure */
    curiosity_reasoning_bridge_t* bridge = (curiosity_reasoning_bridge_t*)nimcp_calloc(
        1, sizeof(curiosity_reasoning_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        curiosity_reasoning_bridge_default_config(&bridge->config);
    }

    /* Allocate topics array */
    bridge->topic_capacity = DEFAULT_TOPIC_CAPACITY;
    bridge->topics = (exploration_topic_t*)nimcp_calloc(
        bridge->topic_capacity, sizeof(exploration_topic_t));
    if (!bridge->topics) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->topic_count = 0;

    /* Create mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "curiosity_reasoning") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge->topics);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize stats */
    memset(&bridge->stats, 0, sizeof(curiosity_reasoning_stats_t));

    bridge->initialized = true;
    return bridge;
}

void curiosity_reasoning_bridge_destroy(curiosity_reasoning_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
        bridge->base.mutex = NULL;
    }

    /* Free topics array */
    if (bridge->topics) {
        nimcp_free(bridge->topics);
        bridge->topics = NULL;
    }

    bridge->initialized = false;
    nimcp_free(bridge);
}

/* ============================================================================
 * Curiosity -> Reasoning Direction Implementation
 * ============================================================================ */

int curiosity_reasoning_drive_exploration(
    curiosity_reasoning_bridge_t* bridge,
    const curiosity_reasoning_context_t* context,
    float curiosity_level
) {
    if (!bridge || !bridge->initialized || !context) {
        return -1;
    }

    /* Clamp curiosity level to valid range */
    curiosity_level = clamp_f(curiosity_level,
                               CURIOSITY_REASONING_MIN_LEVEL,
                               CURIOSITY_REASONING_MAX_LEVEL);

    nimcp_mutex_lock(bridge->base.mutex);

    /* Use context_id as topic_id (truncate to 32-bit) */
    uint32_t topic_id = (uint32_t)(context->context_id & 0xFFFFFFFF);

    /* Find or create topic for this context */
    exploration_topic_t* topic = find_or_create_topic_unlocked(bridge, topic_id);
    if (!topic) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;  /* At capacity */
    }

    /* Update topic with curiosity level */
    topic->curiosity_level = curiosity_level;

    /* Compute exploration priority: curiosity_level * exploration_bias */
    topic->exploration_priority = curiosity_level * bridge->config.exploration_bias;

    /* Clamp to valid range */
    topic->exploration_priority = clamp_f(topic->exploration_priority,
                                           CURIOSITY_REASONING_MIN_LEVEL,
                                           CURIOSITY_REASONING_MAX_LEVEL);

    /* Update stats */
    bridge->stats.explorations_driven++;

    /* Update average curiosity level (exponential moving average) */
    float alpha = 0.1f;
    bridge->stats.avg_curiosity_level =
        bridge->stats.avg_curiosity_level * (1.0f - alpha) +
        curiosity_level * alpha;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int curiosity_reasoning_share_uncertainty(
    curiosity_reasoning_bridge_t* bridge,
    uint64_t topic_id,
    float uncertainty_level
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    /* Clamp uncertainty level to valid range */
    uncertainty_level = clamp_f(uncertainty_level,
                                 CURIOSITY_REASONING_MIN_LEVEL,
                                 CURIOSITY_REASONING_MAX_LEVEL);

    nimcp_mutex_lock(bridge->base.mutex);

    /* Find or create topic */
    uint32_t tid = (uint32_t)(topic_id & 0xFFFFFFFF);
    exploration_topic_t* topic = find_or_create_topic_unlocked(bridge, tid);
    if (!topic) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;  /* At capacity */
    }

    /* Update uncertainty level */
    topic->uncertainty_level = uncertainty_level;

    /* Adjust exploration priority: += uncertainty_level * uncertainty_weight */
    topic->exploration_priority += uncertainty_level * bridge->config.uncertainty_weight;

    /* Clamp to valid range */
    topic->exploration_priority = clamp_f(topic->exploration_priority,
                                           CURIOSITY_REASONING_MIN_LEVEL,
                                           CURIOSITY_REASONING_MAX_LEVEL);

    /* Update stats */
    bridge->stats.uncertainty_shared++;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Reasoning -> Curiosity Direction Implementation
 * ============================================================================ */

int curiosity_reasoning_on_novel_conclusion(
    curiosity_reasoning_bridge_t* bridge,
    uint64_t conclusion_id,
    float novelty_score
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    /* Clamp novelty score to valid range */
    novelty_score = clamp_f(novelty_score,
                             CURIOSITY_REASONING_MIN_LEVEL,
                             CURIOSITY_REASONING_MAX_LEVEL);

    nimcp_mutex_lock(bridge->base.mutex);

    int triggered = 0;

    /* Check if novelty exceeds threshold */
    if (novelty_score > bridge->config.novelty_threshold) {
        /* Boost curiosity for related topics */
        /* For simplicity, boost all active topics slightly */
        for (size_t i = 0; i < bridge->topic_count && i < bridge->topic_capacity; i++) {
            if (bridge->topics[i].active) {
                /* Boost curiosity based on novelty */
                bridge->topics[i].curiosity_level += novelty_score * CURIOSITY_BOOST_FACTOR;
                bridge->topics[i].curiosity_level = clamp_f(
                    bridge->topics[i].curiosity_level,
                    CURIOSITY_REASONING_MIN_LEVEL,
                    CURIOSITY_REASONING_MAX_LEVEL);

                /* Recalculate exploration priority */
                bridge->topics[i].exploration_priority =
                    bridge->topics[i].curiosity_level * bridge->config.exploration_bias +
                    bridge->topics[i].uncertainty_level * bridge->config.uncertainty_weight;
                bridge->topics[i].exploration_priority = clamp_f(
                    bridge->topics[i].exploration_priority,
                    CURIOSITY_REASONING_MIN_LEVEL,
                    CURIOSITY_REASONING_MAX_LEVEL);
            }
        }

        /* Update stats */
        bridge->stats.novel_conclusions++;

        /* Update average novelty score */
        float alpha = 0.1f;
        bridge->stats.avg_novelty_score =
            bridge->stats.avg_novelty_score * (1.0f - alpha) +
            novelty_score * alpha;

        triggered = 1;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return triggered;
}

/* ============================================================================
 * Query API Implementation
 * ============================================================================ */

float curiosity_reasoning_get_exploration_priority(
    curiosity_reasoning_bridge_t* bridge,
    uint64_t topic_id
) {
    if (!bridge || !bridge->initialized) {
        return -1.0f;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    uint32_t tid = (uint32_t)(topic_id & 0xFFFFFFFF);
    exploration_topic_t* topic = find_topic_unlocked(bridge, tid);

    float priority = 0.0f;
    if (topic && topic->active) {
        priority = topic->exploration_priority;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return priority;
}

/* ============================================================================
 * Stats API Implementation
 * ============================================================================ */

int curiosity_reasoning_bridge_get_stats(
    const curiosity_reasoning_bridge_t* bridge,
    curiosity_reasoning_stats_t* stats
) {
    if (!bridge || !bridge->initialized || !stats) {
        return -1;
    }

    /* Note: For const correctness, we cast away const for mutex lock.
     * This is acceptable since we're only reading. Alternatively,
     * a read-write lock could be used. */
    curiosity_reasoning_bridge_t* mutable_bridge =
        (curiosity_reasoning_bridge_t*)bridge;

    nimcp_mutex_lock(mutable_bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(mutable_bridge->base.mutex);

    return 0;
}
