/**
 * @file nimcp_dialect_learning.c
 * @brief Neural Protocol Dialect Learning Implementation
 *
 * @author NIMCP Team
 * @date 2025-12-08
 */

#include "networking/nlp/nimcp_dialect_learning.h"
#include "utils/validation/nimcp_common.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_messages.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

//=============================================================================
// Constants
//=============================================================================

#define DEFAULT_MAX_DIALECTS 100
#define DEFAULT_TRANSLATION_DIM 64
#define DEFAULT_LEARNING_RATE 0.01f
#define MAX_PATH_LENGTH 16
#define DIALECT_HASH_SIZE 256

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Dialect hash table entry
 */
typedef struct dialect_entry_struct {
    neural_dialect_t dialect;
    struct dialect_entry_struct* next;  // Collision chain
    uint64_t last_used_us;
    uint64_t update_count;
} dialect_entry_t;

/**
 * @brief Dialect learner instance
 */
typedef struct dialect_learner_struct {
    // Configuration
    dialect_learner_config_t config;

    // Dialect storage (hash table)
    dialect_entry_t* dialect_table[DIALECT_HASH_SIZE];
    uint32_t dialect_count;
    uint32_t next_dialect_id;

    // Statistics
    dialect_learner_stats_t stats;

    // Bio-async integration
    bio_module_context_t bio_ctx;

    // Thread safety
    pthread_mutex_t lock;
} dialect_learner_struct;

//=============================================================================
// Thread-local Error Storage
//=============================================================================

static __thread char error_buffer[256] = {0};

/**
 * @brief Set error message
 */
static void set_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(error_buffer, sizeof(error_buffer), format, args);
    va_end(args);
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get current time in microseconds
 */
static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/**
 * @brief Hash function for dialect lookup
 *
 * WHAT: Compute hash from source and target IDs
 * WHY:  Fast O(1) dialect lookup
 * HOW:  Combine IDs with XOR and modulo
 */
static uint32_t dialect_hash(uint32_t source, uint32_t target) {
    uint32_t hash = source ^ (target << 16) ^ (target >> 16);
    return hash % DIALECT_HASH_SIZE;
}

/**
 * @brief Find dialect in hash table
 *
 * WHAT: Locate dialect entry
 * WHY:  Retrieve existing dialect
 * HOW:  Hash lookup with collision chain traversal
 */
static dialect_entry_t* find_dialect_entry(
    dialect_learner_t dl,
    uint32_t source,
    uint32_t target
) {
    // Guard clauses
    if (!dl) return NULL;

    uint32_t hash = dialect_hash(source, target);
    dialect_entry_t* entry = dl->dialect_table[hash];

    while (entry) {
        if (entry->dialect.source_region == source &&
            entry->dialect.target_region == target) {
            return entry;
        }
        entry = entry->next;
    }

    return NULL;
}

/**
 * @brief Create new dialect entry
 *
 * WHAT: Allocate and initialize dialect
 * WHY:  Store new translation mapping
 * HOW:  Allocate entry and matrix
 */
static dialect_entry_t* create_dialect_entry(
    dialect_learner_t dl,
    uint32_t source,
    uint32_t target,
    uint32_t dim
) {
    // Guard clauses
    if (!dl || dim == 0) return NULL;

    // Allocate entry
    dialect_entry_t* entry = (dialect_entry_t*)calloc(1, sizeof(dialect_entry_t));
    if (!entry) {
        set_error("Failed to allocate dialect entry");
        return NULL;
    }

    // Initialize dialect
    entry->dialect.dialect_id = dl->next_dialect_id++;
    entry->dialect.source_region = source;
    entry->dialect.target_region = target;
    entry->dialect.matrix_rows = dim;
    entry->dialect.matrix_cols = dim;
    entry->dialect.compatibility_score = 0.0F;

    // Allocate translation matrix
    size_t matrix_size = dim * dim * sizeof(float);
    entry->dialect.translation_matrix = (float*)calloc(dim * dim, sizeof(float));
    if (!entry->dialect.translation_matrix) {
        set_error("Failed to allocate translation matrix");
        free(entry);
        return NULL;
    }

    // Initialize to identity matrix
    for (uint32_t i = 0; i < dim; i++) {
        entry->dialect.translation_matrix[i * dim + i] = 1.0F;
    }

    entry->last_used_us = get_time_us();
    entry->update_count = 0;

    return entry;
}

/**
 * @brief Insert dialect into hash table
 */
static void insert_dialect_entry(dialect_learner_t dl, dialect_entry_t* entry) {
    // Guard clauses
    if (!dl || !entry) return;

    uint32_t hash = dialect_hash(entry->dialect.source_region,
                                   entry->dialect.target_region);

    // Insert at head of collision chain
    entry->next = dl->dialect_table[hash];
    dl->dialect_table[hash] = entry;
    dl->dialect_count++;
}

/**
 * @brief Matrix-vector multiplication
 *
 * WHAT: Compute y = M * x
 * WHY:  Apply translation matrix to signal
 * HOW:  Standard matrix multiplication
 */
static void matrix_vector_mult(
    const float* matrix,
    const float* vector,
    float* result,
    uint32_t rows,
    uint32_t cols
) {
    // Guard clauses
    if (!matrix || !vector || !result || rows == 0 || cols == 0) return;

    for (uint32_t i = 0; i < rows; i++) {
        result[i] = 0.0F;
        for (uint32_t j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

/**
 * @brief Update translation matrix with gradient step
 *
 * WHAT: Apply one learning update
 * WHY:  Improve translation accuracy
 * HOW:  W += learning_rate * error * input^T
 */
static void update_matrix(
    float* matrix,
    const float* input,
    const float* target,
    const float* predicted,
    float learning_rate,
    uint32_t dim
) {
    // Guard clauses
    if (!matrix || !input || !target || !predicted || dim == 0) return;

    // Compute error
    float error[dim];
    for (uint32_t i = 0; i < dim; i++) {
        error[i] = target[i] - predicted[i];
    }

    // Update matrix: W[i,j] += lr * error[i] * input[j]
    for (uint32_t i = 0; i < dim; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            matrix[i * dim + j] += learning_rate * error[i] * input[j];
        }
    }
}

/**
 * @brief Compute mean squared error
 */
static float compute_mse(const float* predicted, const float* target, uint32_t dim) {
    // Guard clauses
    if (!predicted || !target || dim == 0) return -1.0F;

    float sum = 0.0F;
    for (uint32_t i = 0; i < dim; i++) {
        float diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / (float)dim;
}

//=============================================================================
// Bio-Async Message Handlers
//=============================================================================

/**
 * @brief Handle dialect learned messages
 */
static nimcp_error_t handle_dialect_learned(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data
) {
    // Implementation placeholder for bio-async integration
    (void)msg;
    (void)msg_size;
    (void)response_promise;
    (void)user_data;
    return NIMCP_SUCCESS;
}

//=============================================================================
// Core API Implementation
//=============================================================================

dialect_learner_t dialect_learner_create(const dialect_learner_config_t* config) {
    // Allocate structure
    dialect_learner_t dl = (dialect_learner_t)malloc(sizeof(dialect_learner_struct));
    if (!dl) {
        set_error("Failed to allocate dialect learner structure");
        return NULL;
    }
    memset(dl, 0, sizeof(dialect_learner_struct));

    // Set configuration
    if (config) {
        dl->config = *config;
    } else {
        dl->config.max_dialects = DEFAULT_MAX_DIALECTS;
        dl->config.translation_dim = DEFAULT_TRANSLATION_DIM;
        dl->config.learning_rate = DEFAULT_LEARNING_RATE;
        dl->config.enable_bidirectional = true;
        dl->config.enable_bio_async = false;
    }

    // Initialize mutex
    if (pthread_mutex_init(&dl->lock, NULL) != 0) {
        set_error("Failed to initialize mutex");
        free(dl);
        return NULL;
    }

    // Initialize hash table
    memset(dl->dialect_table, 0, sizeof(dl->dialect_table));
    dl->dialect_count = 0;
    dl->next_dialect_id = 1;

    // Bio-async registration
    if (dl->config.enable_bio_async && bio_router_is_initialized()) {
        bio_module_info_t info = {
            .module_id = BIO_MODULE_NLP,  // Using NLP module ID
            .module_name = "dialect_learner",
            .inbox_capacity = 64,
            .user_data = dl
        };
        dl->bio_ctx = bio_router_register_module(&info);

        if (dl->bio_ctx) {
            // Register handler for dialect messages
            bio_router_register_handler(
                dl->bio_ctx,
                BIO_MSG_NLP_NEURAL_ENCODE_COMPLETE,
                handle_dialect_learned
            );
        }
    }

    LOG_INFO("Dialect learner created (max_dialects=%u, dim=%u, lr=%.3f, bio_async=%s)",
             dl->config.max_dialects, dl->config.translation_dim,
             dl->config.learning_rate, dl->config.enable_bio_async ? "enabled" : "disabled");

    return dl;
}

void dialect_learner_destroy(dialect_learner_t dl) {
    // Guard clause
    if (!dl) return;

    LOG_INFO("Destroying dialect learner (total_dialects=%u)", dl->dialect_count);

    // Unregister from bio-async
    if (dl->bio_ctx) {
        bio_router_unregister_module(dl->bio_ctx);
        dl->bio_ctx = NULL;
    }

    // Free all dialects
    for (uint32_t i = 0; i < DIALECT_HASH_SIZE; i++) {
        dialect_entry_t* entry = dl->dialect_table[i];
        while (entry) {
            dialect_entry_t* next = entry->next;
            if (entry->dialect.translation_matrix) {
                free(entry->dialect.translation_matrix);
            }
            free(entry);
            entry = next;
        }
    }

    // Destroy mutex
    pthread_mutex_destroy(&dl->lock);

    // Free structure
    free(dl);
}

int dialect_learn_from_pairs(
    dialect_learner_t dl,
    uint32_t source,
    uint32_t target,
    const float** source_signals,
    const float** target_signals,
    uint32_t num_pairs,
    uint32_t signal_size
) {
    // Guard clauses
    if (!dl || !source_signals || !target_signals || num_pairs == 0 || signal_size == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    if (signal_size > dl->config.translation_dim) {
        set_error("Signal size exceeds translation dimension");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);

    // Find or create dialect entry
    dialect_entry_t* entry = find_dialect_entry(dl, source, target);
    if (!entry) {
        if (dl->dialect_count >= dl->config.max_dialects) {
            pthread_mutex_unlock(&dl->lock);
            set_error("Maximum dialects reached");
            return -1;
        }

        entry = create_dialect_entry(dl, source, target, signal_size);
        if (!entry) {
            pthread_mutex_unlock(&dl->lock);
            return -1;
        }
        insert_dialect_entry(dl, entry);
        dl->stats.total_dialects_learned++;
    }

    // Learning loop
    float predicted[signal_size];
    float total_error = 0.0F;

    for (uint32_t iter = 0; iter < num_pairs; iter++) {
        // Forward pass
        matrix_vector_mult(entry->dialect.translation_matrix,
                          source_signals[iter], predicted,
                          signal_size, signal_size);

        // Compute error
        float error = compute_mse(predicted, target_signals[iter], signal_size);
        total_error += error;

        // Update matrix
        update_matrix(entry->dialect.translation_matrix,
                     source_signals[iter], target_signals[iter], predicted,
                     dl->config.learning_rate, signal_size);
    }

    // Compute final compatibility score
    entry->dialect.compatibility_score = 1.0F / (1.0F + total_error / (float)num_pairs);
    entry->update_count += num_pairs;
    entry->last_used_us = get_time_us();

    // Learn bidirectional if enabled
    if (dl->config.enable_bidirectional) {
        dialect_entry_t* reverse = find_dialect_entry(dl, target, source);
        if (!reverse) {
            reverse = create_dialect_entry(dl, target, source, signal_size);
            if (reverse) {
                insert_dialect_entry(dl, reverse);
                dl->stats.total_dialects_learned++;

                // Learn reverse mapping
                for (uint32_t iter = 0; iter < num_pairs; iter++) {
                    matrix_vector_mult(reverse->dialect.translation_matrix,
                                      target_signals[iter], predicted,
                                      signal_size, signal_size);
                    update_matrix(reverse->dialect.translation_matrix,
                                 target_signals[iter], source_signals[iter], predicted,
                                 dl->config.learning_rate, signal_size);
                }
                reverse->update_count = num_pairs;
                reverse->last_used_us = get_time_us();
            }
        }
    }

    pthread_mutex_unlock(&dl->lock);

    LOG_DEBUG("Learned dialect %u->%u from %u pairs (compat=%.3f)",
              source, target, num_pairs, entry->dialect.compatibility_score);

    return 0;
}

int dialect_update_online(
    dialect_learner_t dl,
    uint32_t source,
    uint32_t target,
    const float* source_signal,
    const float* target_signal,
    uint32_t signal_size
) {
    // Guard clauses
    if (!dl || !source_signal || !target_signal || signal_size == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);

    dialect_entry_t* entry = find_dialect_entry(dl, source, target);
    if (!entry) {
        pthread_mutex_unlock(&dl->lock);
        set_error("Dialect not found");
        return -1;
    }

    // Forward pass
    float predicted[signal_size];
    matrix_vector_mult(entry->dialect.translation_matrix,
                      source_signal, predicted,
                      signal_size, signal_size);

    // Update
    update_matrix(entry->dialect.translation_matrix,
                 source_signal, target_signal, predicted,
                 dl->config.learning_rate, signal_size);

    entry->update_count++;
    entry->last_used_us = get_time_us();
    dl->stats.total_updates++;

    pthread_mutex_unlock(&dl->lock);

    return 0;
}

int dialect_translate(
    dialect_learner_t dl,
    uint32_t source,
    uint32_t target,
    const float* signal,
    uint32_t signal_size,
    float* translated,
    uint32_t* translated_size
) {
    // Guard clauses
    if (!dl || !signal || !translated || !translated_size || signal_size == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);

    dialect_entry_t* entry = find_dialect_entry(dl, source, target);
    if (!entry) {
        // Try multi-hop path
        uint32_t path[MAX_PATH_LENGTH];
        uint32_t path_len;
        int result = dialect_get_bridge_path(dl, source, target, path, &path_len);
        pthread_mutex_unlock(&dl->lock);

        if (result == 0) {
            return dialect_translate_path(dl, path, path_len, signal, signal_size,
                                        translated, translated_size);
        }

        set_error("No dialect or path found");
        return -1;
    }

    // Apply translation
    matrix_vector_mult(entry->dialect.translation_matrix, signal, translated,
                      entry->dialect.matrix_rows, entry->dialect.matrix_cols);
    *translated_size = entry->dialect.matrix_rows;

    entry->last_used_us = get_time_us();
    dl->stats.total_translations++;

    pthread_mutex_unlock(&dl->lock);

    return 0;
}

int dialect_translate_with(
    dialect_learner_t dl,
    const neural_dialect_t* dialect,
    const float* signal,
    uint32_t signal_size,
    float* translated,
    uint32_t* translated_size
) {
    // Guard clauses
    if (!dl || !dialect || !signal || !translated || !translated_size) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);

    matrix_vector_mult(dialect->translation_matrix, signal, translated,
                      dialect->matrix_rows, dialect->matrix_cols);
    *translated_size = dialect->matrix_rows;

    dl->stats.total_translations++;

    pthread_mutex_unlock(&dl->lock);

    return 0;
}

int dialect_get(
    dialect_learner_t dl,
    uint32_t source,
    uint32_t target,
    neural_dialect_t* dialect
) {
    // Guard clauses
    if (!dl || !dialect) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);

    dialect_entry_t* entry = find_dialect_entry(dl, source, target);
    if (!entry) {
        pthread_mutex_unlock(&dl->lock);
        set_error("Dialect not found");
        return -1;
    }

    // Copy dialect (shallow copy - matrix pointer shared)
    *dialect = entry->dialect;

    pthread_mutex_unlock(&dl->lock);

    return 0;
}

bool dialect_exists(dialect_learner_t dl, uint32_t source, uint32_t target) {
    // Guard clause
    if (!dl) return false;

    pthread_mutex_lock(&dl->lock);
    bool exists = (find_dialect_entry(dl, source, target) != NULL);
    pthread_mutex_unlock(&dl->lock);

    return exists;
}

float dialect_get_compatibility(dialect_learner_t dl, uint32_t source, uint32_t target) {
    // Guard clause
    if (!dl) return -1.0F;

    pthread_mutex_lock(&dl->lock);

    dialect_entry_t* entry = find_dialect_entry(dl, source, target);
    float compat = entry ? entry->dialect.compatibility_score : -1.0F;

    pthread_mutex_unlock(&dl->lock);

    return compat;
}

int dialect_get_bridge_path(
    dialect_learner_t dl,
    uint32_t source,
    uint32_t target,
    uint32_t* path,
    uint32_t* path_len
) {
    // Guard clauses
    if (!dl || !path || !path_len) {
        set_error("Invalid parameters");
        return -1;
    }

    // Simple BFS implementation (placeholder - full implementation would use queue)
    // For now, just check for direct connection
    pthread_mutex_lock(&dl->lock);

    dialect_entry_t* entry = find_dialect_entry(dl, source, target);
    if (entry) {
        path[0] = source;
        path[1] = target;
        *path_len = 2;
        pthread_mutex_unlock(&dl->lock);
        return 0;
    }

    pthread_mutex_unlock(&dl->lock);
    set_error("No path found (multi-hop search not yet implemented)");
    return -1;
}

int dialect_translate_path(
    dialect_learner_t dl,
    const uint32_t* path,
    uint32_t path_len,
    const float* signal,
    uint32_t signal_size,
    float* translated,
    uint32_t* translated_size
) {
    // Guard clauses
    if (!dl || !path || !signal || !translated || !translated_size || path_len < 2) {
        set_error("Invalid parameters");
        return -1;
    }

    // Allocate temporary buffers for intermediate translations
    float temp1[signal_size];
    float temp2[signal_size];
    const float* current = signal;
    float* next = temp1;
    uint32_t current_size = signal_size;

    // Translate through each hop
    for (uint32_t i = 0; i < path_len - 1; i++) {
        uint32_t hop_size;
        int result = dialect_translate(dl, path[i], path[i + 1],
                                       current, current_size, next, &hop_size);
        if (result != 0) {
            return result;
        }

        // Swap buffers
        if (next == temp1) {
            current = temp1;
            next = temp2;
        } else {
            current = temp2;
            next = temp1;
        }
        current_size = hop_size;
    }

    // Copy final result
    memcpy(translated, current, current_size * sizeof(float));
    *translated_size = current_size;

    pthread_mutex_lock(&dl->lock);
    dl->stats.multihop_translations++;
    pthread_mutex_unlock(&dl->lock);

    return 0;
}

int dialect_get_all(
    dialect_learner_t dl,
    neural_dialect_t* dialects,
    uint32_t max_count,
    uint32_t* count
) {
    // Guard clauses
    if (!dl || !dialects || !count) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);

    uint32_t num_copied = 0;
    for (uint32_t i = 0; i < DIALECT_HASH_SIZE && num_copied < max_count; i++) {
        dialect_entry_t* entry = dl->dialect_table[i];
        while (entry && num_copied < max_count) {
            dialects[num_copied++] = entry->dialect;
            entry = entry->next;
        }
    }

    *count = num_copied;

    pthread_mutex_unlock(&dl->lock);

    return 0;
}

int dialect_remove(dialect_learner_t dl, uint32_t source, uint32_t target) {
    // Guard clause
    if (!dl) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);

    uint32_t hash = dialect_hash(source, target);
    dialect_entry_t** prev = &dl->dialect_table[hash];
    dialect_entry_t* entry = *prev;

    while (entry) {
        if (entry->dialect.source_region == source &&
            entry->dialect.target_region == target) {
            *prev = entry->next;
            if (entry->dialect.translation_matrix) {
                free(entry->dialect.translation_matrix);
            }
            free(entry);
            dl->dialect_count--;
            pthread_mutex_unlock(&dl->lock);
            return 0;
        }
        prev = &entry->next;
        entry = entry->next;
    }

    pthread_mutex_unlock(&dl->lock);
    set_error("Dialect not found");
    return -1;
}

int dialect_clear_all(dialect_learner_t dl) {
    // Guard clause
    if (!dl) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);

    for (uint32_t i = 0; i < DIALECT_HASH_SIZE; i++) {
        dialect_entry_t* entry = dl->dialect_table[i];
        while (entry) {
            dialect_entry_t* next = entry->next;
            if (entry->dialect.translation_matrix) {
                free(entry->dialect.translation_matrix);
            }
            free(entry);
            entry = next;
        }
        dl->dialect_table[i] = NULL;
    }

    dl->dialect_count = 0;

    pthread_mutex_unlock(&dl->lock);

    LOG_INFO("Cleared all dialects");

    return 0;
}

int dialect_get_stats(dialect_learner_t dl, dialect_learner_stats_t* stats) {
    // Guard clauses
    if (!dl || !stats) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);
    *stats = dl->stats;
    stats->active_dialects = dl->dialect_count;
    pthread_mutex_unlock(&dl->lock);

    return 0;
}

int dialect_reset_stats(dialect_learner_t dl) {
    // Guard clause
    if (!dl) {
        set_error("Invalid parameters");
        return -1;
    }

    pthread_mutex_lock(&dl->lock);
    memset(&dl->stats, 0, sizeof(dialect_learner_stats_t));
    pthread_mutex_unlock(&dl->lock);

    return 0;
}

//=============================================================================
// Utility Functions
//=============================================================================

int dialect_clone(const neural_dialect_t* original, neural_dialect_t* clone) {
    // Guard clauses
    if (!original || !clone) {
        set_error("Invalid parameters");
        return -1;
    }

    // Copy metadata
    *clone = *original;

    // Deep copy matrix
    size_t matrix_size = original->matrix_rows * original->matrix_cols * sizeof(float);
    clone->translation_matrix = (float*)malloc(matrix_size);
    if (!clone->translation_matrix) {
        set_error("Failed to allocate cloned matrix");
        return -1;
    }

    memcpy(clone->translation_matrix, original->translation_matrix, matrix_size);

    return 0;
}

void dialect_free(neural_dialect_t* dialect) {
    // Guard clause
    if (!dialect) return;

    if (dialect->translation_matrix) {
        free(dialect->translation_matrix);
        dialect->translation_matrix = NULL;
    }
}

float dialect_compute_similarity(
    const neural_dialect_t* dialect1,
    const neural_dialect_t* dialect2
) {
    // Guard clauses
    if (!dialect1 || !dialect2) return -1.0F;

    if (dialect1->matrix_rows != dialect2->matrix_rows ||
        dialect1->matrix_cols != dialect2->matrix_cols) {
        return -1.0F;
    }

    // Compute Frobenius norm of difference
    float sum_sq_diff = 0.0F;
    float sum_sq_norm = 0.0F;
    uint32_t size = dialect1->matrix_rows * dialect1->matrix_cols;

    for (uint32_t i = 0; i < size; i++) {
        float diff = dialect1->translation_matrix[i] - dialect2->translation_matrix[i];
        sum_sq_diff += diff * diff;
        sum_sq_norm += dialect1->translation_matrix[i] * dialect1->translation_matrix[i];
    }

    if (sum_sq_norm < 1e-6F) return 0.0F;

    // Return normalized similarity (1.0 = identical, 0.0 = completely different)
    return 1.0F - sqrtf(sum_sq_diff) / sqrtf(sum_sq_norm);
}

const char* dialect_learner_get_last_error(void) {
    return error_buffer;
}
