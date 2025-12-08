/**
 * @file nimcp_rate_limiter.c
 * @brief Token bucket rate limiter implementation
 *
 * WHAT: Token bucket algorithm with penalty system and per-client tracking
 * WHY:  Protect BBB and other APIs from DoS and abuse
 * HOW:  Hash table of client buckets, refill on access, penalty state machine
 *
 * @author NIMCP Development Team
 * @date 2025-12-07
 */

#include "security/nimcp_rate_limiter.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "utils/validation/nimcp_common.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Per-client bucket state
 */
typedef struct client_bucket {
    char client_id[NIMCP_RATE_LIMIT_MAX_CLIENT_ID];
    float tokens;                   /**< Current tokens in bucket */
    uint64_t last_refill_time_ms;   /**< Last refill timestamp */
    nimcp_client_state_t state;     /**< Current state */
    uint32_t penalty_level;         /**< Current penalty level (0-4) */
    uint32_t violations;            /**< Total violations */
    uint32_t consecutive_good;      /**< Consecutive good requests */
    uint64_t block_until_ms;        /**< Block expiration time */
    nimcp_client_stats_t stats;     /**< Per-client statistics */
    struct client_bucket* next;     /**< Hash chain next */
} client_bucket_t;

/**
 * @brief Hash table for client buckets
 */
#define HASH_TABLE_SIZE 1024

typedef struct {
    client_bucket_t* buckets[HASH_TABLE_SIZE];
    nimcp_platform_mutex_t bucket_locks[HASH_TABLE_SIZE];
} client_hash_table_t;

/**
 * @brief Internal rate limiter structure
 */
struct nimcp_rate_limiter_impl {
    uint32_t magic;                             /**< Magic number */
    nimcp_rate_limit_config_t config;           /**< Configuration */
    nimcp_platform_mutex_t limiter_lock;        /**< Main limiter lock */
    client_hash_table_t* client_table;          /**< Per-client buckets */
    float global_tokens;                        /**< Global bucket tokens */
    uint64_t global_last_refill_ms;             /**< Global refill time */
    nimcp_rate_limit_stats_t stats;             /**< Statistics */
    nimcp_rate_limit_violation_callback_t violation_callback;
    void* violation_callback_data;
    bio_module_context_t bio_context;           /**< Bio-async context */
    bool bio_registered;                        /**< Bio-async registration */
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Get current time in milliseconds
 */
static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

/**
 * @brief Simple hash function for client IDs
 */
static uint32_t hash_client_id(const char* client_id) {
    if (!client_id) return 0;

    uint32_t hash = 5381;
    while (*client_id) {
        hash = ((hash << 5) + hash) + (uint32_t)(*client_id++);
    }
    return hash % HASH_TABLE_SIZE;
}

/**
 * @brief Validate limiter handle
 */
static inline bool is_valid_limiter(nimcp_rate_limiter_t limiter) {
    return limiter != NULL && limiter->magic == NIMCP_RATE_LIMITER_MAGIC;
}

/**
 * @brief Refill tokens based on elapsed time (token bucket algorithm)
 */
static void refill_tokens(float* tokens, uint64_t* last_refill_ms,
                          float rate_per_second, float max_tokens)
{
    uint64_t now = get_time_ms();
    uint64_t elapsed_ms = now - *last_refill_ms;

    if (elapsed_ms > 0) {
        float new_tokens = ((float)elapsed_ms / 1000.0f) * rate_per_second;
        *tokens += new_tokens;

        if (*tokens > max_tokens) {
            *tokens = max_tokens;
        }

        *last_refill_ms = now;
    }
}

/**
 * @brief Apply penalty to client
 */
static void apply_penalty(nimcp_rate_limiter_t limiter,
                          client_bucket_t* bucket)
{
    if (!limiter->config.penalty.enabled) {
        return;
    }

    bucket->violations++;

    // Get penalty action for current violation count
    uint32_t violation_idx = bucket->penalty_level;
    if (violation_idx >= 5) {
        violation_idx = 4; // Max penalty
    }

    nimcp_penalty_action_t action = limiter->config.penalty.actions[violation_idx];

    // Apply penalty based on action
    switch (action) {
        case PENALTY_NONE:
            break;

        case PENALTY_WARN:
            bucket->state = CLIENT_STATE_WARNING;
            LOG_MODULE_WARN("rate_limiter",
                "Client %s warning (violations=%u)",
                bucket->client_id, bucket->violations);
            break;

        case PENALTY_REDUCE_RATE_25:
            bucket->state = CLIENT_STATE_RATE_REDUCED;
            bucket->stats.current_rate *= 0.75f;
            bucket->penalty_level++;
            LOG_MODULE_WARN("rate_limiter",
                "Client %s rate reduced to 75%% (violations=%u)",
                bucket->client_id, bucket->violations);
            break;

        case PENALTY_REDUCE_RATE_50:
            bucket->state = CLIENT_STATE_RATE_REDUCED;
            bucket->stats.current_rate *= 0.50f;
            bucket->penalty_level++;
            LOG_MODULE_WARN("rate_limiter",
                "Client %s rate reduced to 50%% (violations=%u)",
                bucket->client_id, bucket->violations);
            break;

        case PENALTY_REDUCE_RATE_75:
            bucket->state = CLIENT_STATE_RATE_REDUCED;
            bucket->stats.current_rate *= 0.25f;
            bucket->penalty_level++;
            LOG_MODULE_WARN("rate_limiter",
                "Client %s rate reduced to 25%% (violations=%u)",
                bucket->client_id, bucket->violations);
            break;

        case PENALTY_BLOCK_TEMPORARY:
            bucket->state = CLIENT_STATE_BLOCKED_TEMP;
            bucket->block_until_ms = get_time_ms() + limiter->config.penalty.cooldown_ms;
            bucket->penalty_level++;
            nimcp_platform_mutex_lock(&limiter->limiter_lock);
            limiter->stats.clients_blocked_temp++;
            nimcp_platform_mutex_unlock(&limiter->limiter_lock);
            LOG_MODULE_WARN("rate_limiter",
                "Client %s temporarily blocked for %ums (violations=%u)",
                bucket->client_id, limiter->config.penalty.cooldown_ms,
                bucket->violations);
            break;

        case PENALTY_BLOCK_PERMANENT:
            bucket->state = CLIENT_STATE_BLOCKED_PERM;
            bucket->penalty_level++;
            nimcp_platform_mutex_lock(&limiter->limiter_lock);
            limiter->stats.clients_blocked_perm++;
            nimcp_platform_mutex_unlock(&limiter->limiter_lock);
            LOG_MODULE_ERROR("rate_limiter",
                "Client %s permanently blocked (violations=%u)",
                bucket->client_id, bucket->violations);
            break;
    }

    // Update statistics
    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    limiter->stats.penalties_applied++;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    // Fire callback if registered
    if (limiter->violation_callback) {
        limiter->violation_callback(bucket->client_id, bucket->violations,
                                    action, limiter->violation_callback_data);
    }
}

/**
 * @brief Check if penalty should be reduced (good behavior)
 */
static void check_good_behavior(nimcp_rate_limiter_t limiter,
                                 client_bucket_t* bucket)
{
    if (!limiter->config.penalty.enabled ||
        !limiter->config.penalty.reset_on_good_behavior) {
        return;
    }

    bucket->consecutive_good++;

    if (bucket->consecutive_good >= limiter->config.penalty.good_behavior_count &&
        bucket->penalty_level > 0) {
        // Reset one penalty level
        bucket->penalty_level--;
        bucket->consecutive_good = 0;

        // Restore some rate
        bucket->stats.current_rate = limiter->config.requests_per_second *
                                      powf(0.75f, (float)bucket->penalty_level);

        if (bucket->penalty_level == 0) {
            bucket->state = CLIENT_STATE_NORMAL;
            LOG_MODULE_INFO("rate_limiter",
                "Client %s penalties reset due to good behavior",
                bucket->client_id);
        }
    }
}

/**
 * @brief Find or create client bucket
 */
static client_bucket_t* get_client_bucket(nimcp_rate_limiter_t limiter,
                                           const char* client_id,
                                           bool create)
{
    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);

    // Search chain
    client_bucket_t* bucket = limiter->client_table->buckets[hash];
    while (bucket) {
        if (strcmp(bucket->client_id, client_id) == 0) {
            nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
            return bucket;
        }
        bucket = bucket->next;
    }

    // Not found
    if (!create) {
        nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
        return NULL;
    }

    // Create new bucket
    bucket = (client_bucket_t*)calloc(1, sizeof(client_bucket_t));
    if (!bucket) {
        nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
        return NULL;
    }

    strncpy(bucket->client_id, client_id, NIMCP_RATE_LIMIT_MAX_CLIENT_ID - 1);
    bucket->tokens = (float)limiter->config.burst_size;
    bucket->last_refill_time_ms = get_time_ms();
    bucket->state = CLIENT_STATE_NORMAL;
    bucket->stats.current_rate = limiter->config.requests_per_second;
    bucket->stats.tokens_available = (uint32_t)bucket->tokens;
    strncpy(bucket->stats.client_id, client_id, NIMCP_RATE_LIMIT_MAX_CLIENT_ID - 1);

    // Add to chain
    bucket->next = limiter->client_table->buckets[hash];
    limiter->client_table->buckets[hash] = bucket;

    // Update statistics
    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    limiter->stats.active_clients++;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
    return bucket;
}

//=============================================================================
// Lifecycle Implementation
//=============================================================================

nimcp_rate_limit_config_t nimcp_rate_limiter_default_config(void) {
    nimcp_rate_limit_config_t config = {
        .requests_per_second = NIMCP_RATE_LIMIT_DEFAULT_RPS,
        .burst_size = NIMCP_RATE_LIMIT_DEFAULT_BURST,
        .algorithm = RATE_LIMIT_TOKEN_BUCKET,
        .per_client = true,
        .per_resource = false,
        .max_tracked_clients = 10000,
        .penalty = {
            .enabled = true,
            .violation_threshold = 2,
            .actions = {
                PENALTY_WARN,
                PENALTY_REDUCE_RATE_25,
                PENALTY_REDUCE_RATE_50,
                PENALTY_BLOCK_TEMPORARY,
                PENALTY_BLOCK_PERMANENT
            },
            .cooldown_ms = NIMCP_RATE_LIMIT_DEFAULT_COOLDOWN_MS,
            .violation_window_ms = 60000,
            .reset_on_good_behavior = true,
            .good_behavior_count = 100
        },
        .enable_statistics = true,
        .enable_logging = true,
        .cleanup_interval_ms = 300000,
        .idle_timeout_ms = 600000
    };
    return config;
}

nimcp_rate_limiter_t nimcp_rate_limiter_create(
    const nimcp_rate_limit_config_t* config)
{
    // Use default if NULL
    nimcp_rate_limit_config_t actual_config;
    if (config == NULL) {
        actual_config = nimcp_rate_limiter_default_config();
    } else {
        actual_config = *config;
    }

    // Validate
    if (actual_config.requests_per_second <= 0.0f ||
        actual_config.burst_size == 0) {
        LOG_ERROR("Invalid rate limiter configuration");
        return NULL;
    }

    // Allocate limiter
    nimcp_rate_limiter_t limiter = (nimcp_rate_limiter_t)calloc(
        1, sizeof(struct nimcp_rate_limiter_impl));
    if (!limiter) {
        LOG_ERROR("Failed to allocate rate limiter");
        return NULL;
    }

    // Initialize
    limiter->magic = NIMCP_RATE_LIMITER_MAGIC;
    limiter->config = actual_config;
    limiter->global_tokens = (float)actual_config.burst_size;
    limiter->global_last_refill_ms = get_time_ms();
    limiter->bio_registered = false;
    memset(&limiter->stats, 0, sizeof(nimcp_rate_limit_stats_t));

    // Initialize main lock
    if (nimcp_platform_mutex_init(&limiter->limiter_lock, false) != 0) {
        LOG_ERROR("Failed to initialize limiter lock");
        free(limiter);
        return NULL;
    }

    // Allocate hash table
    limiter->client_table = (client_hash_table_t*)calloc(
        1, sizeof(client_hash_table_t));
    if (!limiter->client_table) {
        LOG_ERROR("Failed to allocate client hash table");
        nimcp_platform_mutex_destroy(&limiter->limiter_lock);
        free(limiter);
        return NULL;
    }

    // Initialize bucket locks
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        nimcp_platform_mutex_init(&limiter->client_table->bucket_locks[i], false);
    }

    LOG_MODULE_INFO("rate_limiter",
        "Created rate limiter (rate=%.1f req/s, burst=%u)",
        actual_config.requests_per_second, actual_config.burst_size);

    return limiter;
}

void nimcp_rate_limiter_destroy(nimcp_rate_limiter_t limiter) {
    if (!is_valid_limiter(limiter)) {
        return;
    }

    LOG_MODULE_DEBUG("rate_limiter", "Destroying rate limiter");

    // Cleanup client buckets
    if (limiter->client_table) {
        for (int i = 0; i < HASH_TABLE_SIZE; i++) {
            nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[i]);

            client_bucket_t* bucket = limiter->client_table->buckets[i];
            while (bucket) {
                client_bucket_t* next = bucket->next;
                free(bucket);
                bucket = next;
            }

            nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[i]);
            nimcp_platform_mutex_destroy(&limiter->client_table->bucket_locks[i]);
        }
        free(limiter->client_table);
    }

    // Unregister from bio-async
    if (limiter->bio_registered && limiter->bio_context) {
        bio_router_unregister_module(limiter->bio_context);
    }

    // Cleanup
    limiter->magic = 0;
    nimcp_platform_mutex_destroy(&limiter->limiter_lock);
    free(limiter);
}

//=============================================================================
// Core Rate Limiting Implementation
//=============================================================================

bool nimcp_rate_limiter_allow(
    nimcp_rate_limiter_t limiter,
    const char* client_id)
{
    if (!is_valid_limiter(limiter)) {
        return false;
    }

    // Update statistics
    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    limiter->stats.total_requests++;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    // Global limit if no client_id
    if (!limiter->config.per_client || client_id == NULL) {
        nimcp_platform_mutex_lock(&limiter->limiter_lock);

        refill_tokens(&limiter->global_tokens,
                      &limiter->global_last_refill_ms,
                      limiter->config.requests_per_second,
                      (float)limiter->config.burst_size);

        bool allowed = (limiter->global_tokens >= 1.0f);
        if (allowed) {
            limiter->global_tokens -= 1.0f;
            limiter->stats.requests_allowed++;
        } else {
            limiter->stats.requests_denied++;
        }

        nimcp_platform_mutex_unlock(&limiter->limiter_lock);
        return allowed;
    }

    // Per-client limiting
    client_bucket_t* bucket = get_client_bucket(limiter, client_id, true);
    if (!bucket) {
        LOG_MODULE_ERROR("rate_limiter",
            "Failed to get bucket for client %s", client_id);
        return false;
    }

    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);

    // Check if blocked
    if (bucket->state == CLIENT_STATE_BLOCKED_PERM) {
        nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
        nimcp_platform_mutex_lock(&limiter->limiter_lock);
        limiter->stats.requests_denied++;
        nimcp_platform_mutex_unlock(&limiter->limiter_lock);
        return false;
    }

    if (bucket->state == CLIENT_STATE_BLOCKED_TEMP) {
        if (get_time_ms() < bucket->block_until_ms) {
            nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
            nimcp_platform_mutex_lock(&limiter->limiter_lock);
            limiter->stats.requests_denied++;
            nimcp_platform_mutex_unlock(&limiter->limiter_lock);
            return false;
        } else {
            // Unblock
            bucket->state = CLIENT_STATE_NORMAL;
            LOG_MODULE_INFO("rate_limiter",
                "Client %s temporary block expired", client_id);
        }
    }

    // Refill tokens
    refill_tokens(&bucket->tokens, &bucket->last_refill_time_ms,
                  bucket->stats.current_rate,
                  (float)limiter->config.burst_size);

    // Check tokens
    bool allowed = (bucket->tokens >= 1.0f);

    if (allowed) {
        bucket->tokens -= 1.0f;
        bucket->stats.requests_allowed++;
        bucket->stats.last_request_time_ms = get_time_ms();
        bucket->stats.tokens_available = (uint32_t)bucket->tokens;
        bucket->consecutive_good++;

        check_good_behavior(limiter, bucket);

        nimcp_platform_mutex_lock(&limiter->limiter_lock);
        limiter->stats.requests_allowed++;
        limiter->stats.overall_allow_rate =
            (float)limiter->stats.requests_allowed /
            (float)limiter->stats.total_requests * 100.0f;
        nimcp_platform_mutex_unlock(&limiter->limiter_lock);
    } else {
        bucket->stats.requests_denied++;
        bucket->consecutive_good = 0;

        apply_penalty(limiter, bucket);

        nimcp_platform_mutex_lock(&limiter->limiter_lock);
        limiter->stats.requests_denied++;
        limiter->stats.overall_allow_rate =
            (float)limiter->stats.requests_allowed /
            (float)limiter->stats.total_requests * 100.0f;
        nimcp_platform_mutex_unlock(&limiter->limiter_lock);
    }

    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
    return allowed;
}

bool nimcp_rate_limiter_allow_resource(
    nimcp_rate_limiter_t limiter,
    const char* client_id,
    const char* resource_id)
{
    // For now, just use client_id (can extend with resource tracking)
    return nimcp_rate_limiter_allow(limiter, client_id);
}

bool nimcp_rate_limiter_acquire(
    nimcp_rate_limiter_t limiter,
    const char* client_id,
    uint32_t count)
{
    if (!is_valid_limiter(limiter) || count == 0) {
        return false;
    }

    client_bucket_t* bucket = get_client_bucket(limiter, client_id, true);
    if (!bucket) {
        return false;
    }

    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);

    refill_tokens(&bucket->tokens, &bucket->last_refill_time_ms,
                  bucket->stats.current_rate,
                  (float)limiter->config.burst_size);

    bool allowed = (bucket->tokens >= (float)count);

    if (allowed) {
        bucket->tokens -= (float)count;
        bucket->stats.requests_allowed += count;
    } else {
        bucket->stats.requests_denied++;
    }

    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
    return allowed;
}

bool nimcp_rate_limiter_check(
    nimcp_rate_limiter_t limiter,
    const char* client_id)
{
    if (!is_valid_limiter(limiter)) {
        return false;
    }

    client_bucket_t* bucket = get_client_bucket(limiter, client_id, false);
    if (!bucket) {
        return true; // New client, would be allowed
    }

    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);

    // Just check, don't consume
    refill_tokens(&bucket->tokens, &bucket->last_refill_time_ms,
                  bucket->stats.current_rate,
                  (float)limiter->config.burst_size);

    bool would_allow = (bucket->tokens >= 1.0f) &&
                       (bucket->state != CLIENT_STATE_BLOCKED_PERM) &&
                       (bucket->state != CLIENT_STATE_BLOCKED_TEMP ||
                        get_time_ms() >= bucket->block_until_ms);

    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
    return would_allow;
}

uint64_t nimcp_rate_limiter_time_until_ready(
    nimcp_rate_limiter_t limiter,
    const char* client_id)
{
    if (!is_valid_limiter(limiter)) {
        return 0;
    }

    client_bucket_t* bucket = get_client_bucket(limiter, client_id, false);
    if (!bucket) {
        return 0; // New client, ready now
    }

    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);

    refill_tokens(&bucket->tokens, &bucket->last_refill_time_ms,
                  bucket->stats.current_rate,
                  (float)limiter->config.burst_size);

    uint64_t wait_ms = 0;
    if (bucket->tokens < 1.0f) {
        float tokens_needed = 1.0f - bucket->tokens;
        wait_ms = (uint64_t)(tokens_needed / bucket->stats.current_rate * 1000.0f);
    }

    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);
    return wait_ms;
}

//=============================================================================
// Statistics and Management Implementation
//=============================================================================

nimcp_error_t nimcp_rate_limiter_get_stats(
    nimcp_rate_limiter_t limiter,
    nimcp_rate_limit_stats_t* stats)
{
    if (!is_valid_limiter(limiter) || stats == NULL) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    *stats = limiter->stats;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_rate_limiter_reset_stats(nimcp_rate_limiter_t limiter) {
    if (!is_valid_limiter(limiter)) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    memset(&limiter->stats, 0, sizeof(nimcp_rate_limit_stats_t));
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    LOG_MODULE_INFO("rate_limiter", "Statistics reset");
    return NIMCP_SUCCESS;
}

uint32_t nimcp_rate_limiter_get_active_clients(nimcp_rate_limiter_t limiter) {
    if (!is_valid_limiter(limiter)) {
        return 0;
    }

    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    uint32_t count = limiter->stats.active_clients;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    return count;
}

nimcp_error_t nimcp_rate_limiter_reset_client(
    nimcp_rate_limiter_t limiter,
    const char* client_id)
{
    if (!is_valid_limiter(limiter) || client_id == NULL) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    client_bucket_t* bucket = get_client_bucket(limiter, client_id, false);
    if (!bucket) {
        return NIMCP_ERROR_NOT_FOUND;
    }

    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);

    bucket->tokens = (float)limiter->config.burst_size;
    bucket->state = CLIENT_STATE_NORMAL;
    bucket->penalty_level = 0;
    bucket->violations = 0;
    bucket->consecutive_good = 0;
    bucket->stats.current_rate = limiter->config.requests_per_second;

    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);

    LOG_MODULE_INFO("rate_limiter", "Client %s reset", client_id);
    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_rate_limiter_block_client(
    nimcp_rate_limiter_t limiter,
    const char* client_id)
{
    if (!is_valid_limiter(limiter) || client_id == NULL) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    client_bucket_t* bucket = get_client_bucket(limiter, client_id, true);
    if (!bucket) {
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);

    bucket->state = CLIENT_STATE_BLOCKED_PERM;

    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    limiter->stats.clients_blocked_perm++;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);

    LOG_MODULE_WARN("rate_limiter", "Client %s permanently blocked", client_id);
    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_rate_limiter_unblock_client(
    nimcp_rate_limiter_t limiter,
    const char* client_id)
{
    if (!is_valid_limiter(limiter) || client_id == NULL) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    client_bucket_t* bucket = get_client_bucket(limiter, client_id, false);
    if (!bucket) {
        return NIMCP_ERROR_NOT_FOUND;
    }

    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);

    if (bucket->state == CLIENT_STATE_BLOCKED_PERM) {
        nimcp_platform_mutex_lock(&limiter->limiter_lock);
        limiter->stats.clients_blocked_perm--;
        nimcp_platform_mutex_unlock(&limiter->limiter_lock);
    }

    bucket->state = CLIENT_STATE_NORMAL;
    bucket->penalty_level = 0;

    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);

    LOG_MODULE_INFO("rate_limiter", "Client %s unblocked", client_id);
    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_rate_limiter_get_client_stats(
    nimcp_rate_limiter_t limiter,
    const char* client_id,
    nimcp_client_stats_t* stats)
{
    if (!is_valid_limiter(limiter) || client_id == NULL || stats == NULL) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    client_bucket_t* bucket = get_client_bucket(limiter, client_id, false);
    if (!bucket) {
        return NIMCP_ERROR_NOT_FOUND;
    }

    uint32_t hash = hash_client_id(client_id);
    nimcp_platform_mutex_lock(&limiter->client_table->bucket_locks[hash]);
    *stats = bucket->stats;
    nimcp_platform_mutex_unlock(&limiter->client_table->bucket_locks[hash]);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_rate_limiter_set_violation_callback(
    nimcp_rate_limiter_t limiter,
    nimcp_rate_limit_violation_callback_t callback,
    void* user_data)
{
    if (!is_valid_limiter(limiter)) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    limiter->violation_callback = callback;
    limiter->violation_callback_data = user_data;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_rate_limiter_set_rate(
    nimcp_rate_limiter_t limiter,
    float requests_per_second)
{
    if (!is_valid_limiter(limiter) || requests_per_second <= 0.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    limiter->config.requests_per_second = requests_per_second;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    LOG_MODULE_INFO("rate_limiter",
        "Rate updated to %.1f req/s", requests_per_second);
    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_rate_limiter_set_burst(
    nimcp_rate_limiter_t limiter,
    uint32_t burst_size)
{
    if (!is_valid_limiter(limiter) || burst_size == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(&limiter->limiter_lock);
    limiter->config.burst_size = burst_size;
    nimcp_platform_mutex_unlock(&limiter->limiter_lock);

    LOG_MODULE_INFO("rate_limiter", "Burst size updated to %u", burst_size);
    return NIMCP_SUCCESS;
}

//=============================================================================
// Bio-Async Integration Implementation
//=============================================================================

static nimcp_error_t rate_limiter_bio_handler(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    nimcp_rate_limiter_t limiter = (nimcp_rate_limiter_t)user_data;

    if (!is_valid_limiter(limiter) || msg == NULL) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    // Handle health check
    if (header->type == BIO_MSG_HEALTH_CHECK) {
        bio_msg_health_response_t response = {0};
        bio_msg_init_header(&response.header, BIO_MSG_HEALTH_RESPONSE,
            BIO_MODULE_SECURITY, header->source_module, sizeof(response));

        response.healthy = true;
        response.active_threads = limiter->stats.active_clients;

        if (response_promise) {
            nimcp_bio_promise_complete(response_promise, &response);
        }
        return NIMCP_SUCCESS;
    }

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_rate_limiter_register_bio_async(
    nimcp_rate_limiter_t limiter)
{
    if (!is_valid_limiter(limiter)) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (limiter->bio_registered) {
        return NIMCP_SUCCESS;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SECURITY,
        .module_name = "rate_limiter",
        .inbox_capacity = 64,
        .user_data = limiter
    };

    limiter->bio_context = bio_router_register_module(&info);
    if (!limiter->bio_context) {
        LOG_ERROR("Failed to register rate limiter with bio-router");
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    bio_router_register_handler(limiter->bio_context,
        BIO_MSG_HEALTH_CHECK, rate_limiter_bio_handler);

    limiter->bio_registered = true;

    LOG_MODULE_INFO("rate_limiter", "Registered with bio-async router");
    return NIMCP_SUCCESS;
}

uint32_t nimcp_rate_limiter_process_inbox(
    nimcp_rate_limiter_t limiter,
    uint32_t max_messages)
{
    if (!is_valid_limiter(limiter) || !limiter->bio_registered) {
        return 0;
    }

    return bio_router_process_inbox(limiter->bio_context, max_messages);
}
