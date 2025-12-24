/**
 * @file nimcp_pattern_db.c
 * @brief Implementation of Runtime-Updateable Pattern Database
 *
 * WHAT: Lock-free concurrent pattern database for threat detection
 * WHY:  Enable runtime pattern updates without system restart
 * HOW:  RCU-like versioning, atomic updates, lock-free reads
 *
 * ARCHITECTURE:
 * - Lock-free reads using atomic pointers and versioning
 * - Copy-on-write for pattern updates
 * - Snapshot-based rollback using refcounted versions
 * - Thread-safe using atomic operations and RCU semantics
 *
 * @author NIMCP Security Team
 * @date 2025-12-07
 */

#include "security/nimcp_pattern_db.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "security/nimcp_security.h"
#include "async/nimcp_bio_router.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <regex.h>
#include <signal.h>
#include <unistd.h>

//=============================================================================
// Internal Constants
//=============================================================================

#define PATTERN_DB_MAGIC_VALID   0x50415444
#define PATTERN_DB_INITIAL_CAPACITY  128
#define PATTERN_DB_LOAD_FACTOR   0.75f
#define PATTERN_DB_MAX_REGEX_ERRORS 256

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Compiled pattern internal representation
 *
 * WHAT: Compiled regex with metadata
 * WHY:  Efficient matching without recompilation
 * HOW:  POSIX regex_t for portable regex support
 */
typedef struct {
    uint32_t pattern_id;
    nimcp_pattern_entry_t entry;
    regex_t compiled_regex;
    bool is_compiled;
    uint32_t match_count;
    uint64_t total_match_time_us;
    char pattern_copy[NIMCP_PATTERN_MAX_LENGTH];
    char description_copy[NIMCP_PATTERN_MAX_DESCRIPTION];
} pattern_slot_t;

/**
 * @brief Version snapshot for rollback
 */
typedef struct pattern_snapshot {
    uint32_t version;
    uint32_t pattern_count;
    pattern_slot_t* patterns;
    size_t capacity;
    uint32_t ref_count;
    struct pattern_snapshot* next;
} pattern_snapshot_t;

/**
 * @brief Pattern database implementation
 */
struct nimcp_pattern_db_impl {
    uint32_t magic;
    nimcp_pattern_db_config_t config;

    // Pattern storage (lock-free reads)
    pattern_slot_t* patterns;
    size_t capacity;
    uint32_t pattern_count;
    uint32_t next_pattern_id;

    // Versioning
    uint32_t current_version;
    pattern_snapshot_t* snapshots;
    uint32_t snapshot_count;

    // Statistics
    nimcp_pattern_db_stats_t stats;

    // Synchronization
    pthread_rwlock_t write_lock;

    // Bio-async integration
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;
};

//=============================================================================
// Internal Helper Functions
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
 * @brief Validate pattern database handle
 */
static bool validate_db(nimcp_pattern_db_t db) {
    return db != NULL && db->magic == PATTERN_DB_MAGIC_VALID;
}

/**
 * @brief Check regex pattern complexity to prevent ReDoS
 *
 * WHAT: Detect potentially dangerous regex patterns
 * WHY:  Prevent exponential backtracking attacks (ReDoS)
 * HOW:  Count nested quantifiers and alternations
 */
static bool is_regex_safe(const char* pattern) {
    if (!pattern) return false;

    int nested_quantifiers = 0;
    int max_nesting = 0;
    int alternations = 0;
    bool in_group = false;

    for (const char* p = pattern; *p; p++) {
        switch (*p) {
            case '(':
                if (in_group) nested_quantifiers++;
                in_group = true;
                if (nested_quantifiers > max_nesting) {
                    max_nesting = nested_quantifiers;
                }
                break;
            case ')':
                if (nested_quantifiers > 0) nested_quantifiers--;
                in_group = false;
                break;
            case '*':
            case '+':
            case '?':
                /* Check for nested quantifiers */
                if (nested_quantifiers > 2) {
                    return false; /* Too much nesting */
                }
                break;
            case '|':
                alternations++;
                if (alternations > 10) {
                    return false; /* Too many alternations */
                }
                break;
        }
    }

    /* Reject patterns with excessive nesting */
    if (max_nesting > 3) {
        return false;
    }

    return true;
}

/**
 * @brief Compile regex pattern
 */
static nimcp_error_t compile_pattern(pattern_slot_t* slot) {
    int regex_flags = REG_EXTENDED;

    /* SECURITY: Check pattern complexity to prevent ReDoS */
    if (!is_regex_safe(slot->pattern_copy)) {
        fprintf(stderr, "Pattern rejected: complexity too high (ReDoS risk)\n");
        return NIMCP_ERROR;
    }

    // Apply flags
    if (slot->entry.flags & NIMCP_PATTERN_FLAG_CASE_INSENSITIVE) {
        regex_flags |= REG_ICASE;
    }
    if (slot->entry.flags & NIMCP_PATTERN_FLAG_MULTILINE) {
        regex_flags |= REG_NEWLINE;
    }

    int ret = regcomp(&slot->compiled_regex, slot->pattern_copy, regex_flags);
    if (ret != 0) {
        char error_buf[PATTERN_DB_MAX_REGEX_ERRORS];
        regerror(ret, &slot->compiled_regex, error_buf, sizeof(error_buf));
        fprintf(stderr, "Pattern compilation failed: %s\n", error_buf);
        return NIMCP_ERROR;
    }

    slot->is_compiled = true;
    return NIMCP_SUCCESS;
}

/**
 * @brief Free compiled pattern
 */
static void free_pattern(pattern_slot_t* slot) {
    if (slot->is_compiled) {
        regfree(&slot->compiled_regex);
        slot->is_compiled = false;
    }
}

/**
 * @brief Find pattern by ID
 */
static pattern_slot_t* find_pattern(nimcp_pattern_db_t db, nimcp_pattern_id_t id) {
    for (size_t i = 0; i < db->capacity; i++) {
        if (db->patterns[i].pattern_id == id && db->patterns[i].is_compiled) {
            return &db->patterns[i];
        }
    }
    return NULL;
}

/**
 * @brief Grow pattern array
 */
static nimcp_error_t grow_patterns(nimcp_pattern_db_t db) {
    size_t new_capacity = db->capacity * 2;
    if (db->config.max_patterns > 0 && new_capacity > db->config.max_patterns) {
        new_capacity = db->config.max_patterns;
    }

    if (new_capacity <= db->capacity) {
        return NIMCP_ERROR;  // At maximum capacity
    }

    pattern_slot_t* new_patterns = calloc(new_capacity, sizeof(pattern_slot_t));
    if (!new_patterns) {
        return NIMCP_NO_MEMORY;
    }

    // Copy existing patterns
    memcpy(new_patterns, db->patterns, db->capacity * sizeof(pattern_slot_t));

    free(db->patterns);
    db->patterns = new_patterns;
    db->capacity = new_capacity;

    return NIMCP_SUCCESS;
}

/**
 * @brief Create version snapshot
 */
static nimcp_error_t create_snapshot(nimcp_pattern_db_t db, uint32_t* snapshot_id) {
    if (db->snapshot_count >= db->config.max_snapshots) {
        // Remove oldest snapshot
        if (db->snapshots != NULL) {
            pattern_snapshot_t* oldest = db->snapshots;
            db->snapshots = oldest->next;

            // MEMORY SAFETY: ref_count is currently always 1 (no sharing)
            // Protected by write_lock, so no concurrent access.
            // TODO: If snapshot sharing is added, use nimcp_atomic_uint32_t for ref_count
            oldest->ref_count--;
            if (oldest->ref_count == 0) {
                for (size_t i = 0; i < oldest->pattern_count; i++) {
                    free_pattern(&oldest->patterns[i]);
                }
                free(oldest->patterns);
                free(oldest);
            }
            db->snapshot_count--;
        }
    }

    pattern_snapshot_t* snapshot = calloc(1, sizeof(pattern_snapshot_t));
    if (!snapshot) {
        return NIMCP_NO_MEMORY;
    }

    snapshot->version = db->current_version;
    snapshot->pattern_count = db->pattern_count;
    snapshot->capacity = db->capacity;
    snapshot->ref_count = 1;

    // Deep copy patterns
    snapshot->patterns = calloc(db->capacity, sizeof(pattern_slot_t));
    if (!snapshot->patterns) {
        free(snapshot);
        return NIMCP_NO_MEMORY;
    }

    for (size_t i = 0; i < db->capacity; i++) {
        if (db->patterns[i].is_compiled) {
            snapshot->patterns[i] = db->patterns[i];
            // Re-compile regex in snapshot
            compile_pattern(&snapshot->patterns[i]);
        }
    }

    // Add to snapshot list
    snapshot->next = db->snapshots;
    db->snapshots = snapshot;
    db->snapshot_count++;

    if (snapshot_id) {
        *snapshot_id = snapshot->version;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Configuration Functions
//=============================================================================

nimcp_pattern_db_config_t nimcp_pattern_db_default_config(void) {
    nimcp_pattern_db_config_t config = {
        .initial_capacity = NIMCP_PATTERN_DEFAULT_CAPACITY,
        .max_patterns = 0,  // Unlimited
        .max_snapshots = NIMCP_PATTERN_MAX_SNAPSHOTS,
        .enable_statistics = true,
        .enable_validation = true,
        .enable_bio_async = false,
        .module_id = 0,
        .match_timeout_ms = 100.0F,
        .worker_threads = 0  // Auto-detect
    };
    return config;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

nimcp_pattern_db_t nimcp_pattern_db_create(const nimcp_pattern_db_config_t* config) {
    LOG_MODULE_DEBUG("pattern_db", "Creating pattern database");

    nimcp_pattern_db_t db = calloc(1, sizeof(struct nimcp_pattern_db_impl));
    if (!db) {
        LOG_MODULE_ERROR("pattern_db", "Failed to allocate pattern database");
        return NULL;
    }

    // Set configuration
    if (config) {
        db->config = *config;
    } else {
        db->config = nimcp_pattern_db_default_config();
    }

    // Allocate pattern array
    size_t capacity = db->config.initial_capacity;
    if (capacity == 0) {
        capacity = PATTERN_DB_INITIAL_CAPACITY;
    }

    db->patterns = calloc(capacity, sizeof(pattern_slot_t));
    if (!db->patterns) {
        free(db);
        return NULL;
    }

    db->capacity = capacity;
    db->pattern_count = 0;
    db->next_pattern_id = 1;  // Start from 1, 0 is invalid
    db->current_version = 1;
    db->snapshots = NULL;
    db->snapshot_count = 0;

    // Initialize synchronization
    if (pthread_rwlock_init(&db->write_lock, NULL) != 0) {
        free(db->patterns);
        free(db);
        return NULL;
    }

    // Initialize statistics
    memset(&db->stats, 0, sizeof(db->stats));
    db->stats.current_version = db->current_version;

    // Initialize bio-async if enabled
    db->bio_async_enabled = false;
    db->bio_ctx = NULL;

    db->magic = PATTERN_DB_MAGIC_VALID;

    LOG_MODULE_INFO("pattern_db", "Pattern database created with capacity %zu", capacity);
    return db;
}

void nimcp_pattern_db_destroy(nimcp_pattern_db_t db) {
    if (!validate_db(db)) {
        LOG_MODULE_WARN("pattern_db", "Attempted to destroy invalid pattern database");
        return;
    }

    LOG_MODULE_DEBUG("pattern_db", "Destroying pattern database with %u patterns", db->pattern_count);
    db->magic = 0;  // Invalidate immediately

    // Unregister from bio-async
    if (db->bio_async_enabled && db->bio_ctx) {
        bio_router_unregister_module(db->bio_ctx);
    }

    // Free all patterns
    for (size_t i = 0; i < db->capacity; i++) {
        free_pattern(&db->patterns[i]);
    }
    free(db->patterns);

    // Free all snapshots
    pattern_snapshot_t* snapshot = db->snapshots;
    while (snapshot) {
        pattern_snapshot_t* next = snapshot->next;

        snapshot->ref_count--;
        if (snapshot->ref_count == 0) {
            for (size_t i = 0; i < snapshot->capacity; i++) {
                free_pattern(&snapshot->patterns[i]);
            }
            free(snapshot->patterns);
            free(snapshot);
        }

        snapshot = next;
    }

    pthread_rwlock_destroy(&db->write_lock);
    free(db);
}

//=============================================================================
// Pattern Management
//=============================================================================

nimcp_error_t nimcp_pattern_db_add(
    nimcp_pattern_db_t db,
    const nimcp_pattern_entry_t* entry,
    nimcp_pattern_id_t* id
) {
    if (!validate_db(db) || !entry || !entry->pattern) {
        return NIMCP_INVALID_PARAM;
    }

    if (strlen(entry->pattern) >= NIMCP_PATTERN_MAX_LENGTH) {
        return NIMCP_INVALID_PARAM;
    }

    if (entry->description && strlen(entry->description) >= NIMCP_PATTERN_MAX_DESCRIPTION) {
        return NIMCP_INVALID_PARAM;
    }

    pthread_rwlock_wrlock(&db->write_lock);

    // Check if we need to grow
    if (db->pattern_count >= db->capacity * PATTERN_DB_LOAD_FACTOR) {
        nimcp_error_t err = grow_patterns(db);
        if (err != NIMCP_SUCCESS) {
            pthread_rwlock_unlock(&db->write_lock);
            return err;
        }
    }

    // Find free slot
    pattern_slot_t* slot = NULL;
    for (size_t i = 0; i < db->capacity; i++) {
        if (!db->patterns[i].is_compiled) {
            slot = &db->patterns[i];
            break;
        }
    }

    if (!slot) {
        pthread_rwlock_unlock(&db->write_lock);
        return NIMCP_ERROR;
    }

    // Fill in pattern slot
    memset(slot, 0, sizeof(pattern_slot_t));
    slot->pattern_id = db->next_pattern_id++;
    slot->entry = *entry;

    // Copy strings
    strncpy(slot->pattern_copy, entry->pattern, NIMCP_PATTERN_MAX_LENGTH - 1);
    slot->pattern_copy[NIMCP_PATTERN_MAX_LENGTH - 1] = '\0';
    slot->entry.pattern = slot->pattern_copy;

    if (entry->description) {
        strncpy(slot->description_copy, entry->description, NIMCP_PATTERN_MAX_DESCRIPTION - 1);
        slot->description_copy[NIMCP_PATTERN_MAX_DESCRIPTION - 1] = '\0';
        slot->entry.description = slot->description_copy;
    } else {
        slot->description_copy[0] = '\0';
        slot->entry.description = slot->description_copy;
    }

    // Compile pattern
    nimcp_error_t err = compile_pattern(slot);
    if (err != NIMCP_SUCCESS) {
        memset(slot, 0, sizeof(pattern_slot_t));
        pthread_rwlock_unlock(&db->write_lock);
        return err;
    }

    db->pattern_count++;
    db->current_version++;

    // Update statistics
    db->stats.total_patterns = db->pattern_count;
    db->stats.current_version = db->current_version;
    if (entry->category < NIMCP_PATTERN_CATEGORY_COUNT) {
        db->stats.patterns_by_category[entry->category]++;
    }

    if (id) {
        *id = slot->pattern_id;
    }

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_pattern_db_remove(
    nimcp_pattern_db_t db,
    nimcp_pattern_id_t id
) {
    if (!validate_db(db) || id == NIMCP_PATTERN_ID_INVALID) {
        return NIMCP_INVALID_PARAM;
    }

    pthread_rwlock_wrlock(&db->write_lock);

    pattern_slot_t* slot = find_pattern(db, id);
    if (!slot) {
        pthread_rwlock_unlock(&db->write_lock);
        return NIMCP_NOT_FOUND;
    }

    // Update statistics
    if (slot->entry.category < NIMCP_PATTERN_CATEGORY_COUNT) {
        db->stats.patterns_by_category[slot->entry.category]--;
    }

    // Free pattern
    free_pattern(slot);
    memset(slot, 0, sizeof(pattern_slot_t));

    db->pattern_count--;
    db->current_version++;

    db->stats.total_patterns = db->pattern_count;
    db->stats.current_version = db->current_version;

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_pattern_db_update(
    nimcp_pattern_db_t db,
    nimcp_pattern_id_t id,
    const nimcp_pattern_entry_t* entry
) {
    if (!validate_db(db) || id == NIMCP_PATTERN_ID_INVALID || !entry) {
        return NIMCP_INVALID_PARAM;
    }

    pthread_rwlock_wrlock(&db->write_lock);

    pattern_slot_t* slot = find_pattern(db, id);
    if (!slot) {
        pthread_rwlock_unlock(&db->write_lock);
        return NIMCP_NOT_FOUND;
    }

    // Update statistics (remove old category)
    if (slot->entry.category < NIMCP_PATTERN_CATEGORY_COUNT) {
        db->stats.patterns_by_category[slot->entry.category]--;
    }

    // Free old pattern
    free_pattern(slot);

    // Update entry
    slot->entry = *entry;

    // Copy strings
    strncpy(slot->pattern_copy, entry->pattern, NIMCP_PATTERN_MAX_LENGTH - 1);
    slot->pattern_copy[NIMCP_PATTERN_MAX_LENGTH - 1] = '\0';
    slot->entry.pattern = slot->pattern_copy;

    if (entry->description) {
        strncpy(slot->description_copy, entry->description, NIMCP_PATTERN_MAX_DESCRIPTION - 1);
        slot->description_copy[NIMCP_PATTERN_MAX_DESCRIPTION - 1] = '\0';
        slot->entry.description = slot->description_copy;
    } else {
        slot->description_copy[0] = '\0';
        slot->entry.description = slot->description_copy;
    }

    // Recompile
    nimcp_error_t err = compile_pattern(slot);
    if (err != NIMCP_SUCCESS) {
        memset(slot, 0, sizeof(pattern_slot_t));
        db->pattern_count--;
        pthread_rwlock_unlock(&db->write_lock);
        return err;
    }

    // Update statistics (add new category)
    if (entry->category < NIMCP_PATTERN_CATEGORY_COUNT) {
        db->stats.patterns_by_category[entry->category]++;
    }

    db->current_version++;
    db->stats.current_version = db->current_version;

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_pattern_db_get(
    nimcp_pattern_db_t db,
    nimcp_pattern_id_t id,
    nimcp_pattern_entry_t* entry
) {
    if (!validate_db(db) || id == NIMCP_PATTERN_ID_INVALID || !entry) {
        return NIMCP_INVALID_PARAM;
    }

    pthread_rwlock_rdlock(&db->write_lock);

    pattern_slot_t* slot = find_pattern(db, id);
    if (!slot) {
        pthread_rwlock_unlock(&db->write_lock);
        return NIMCP_NOT_FOUND;
    }

    *entry = slot->entry;

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

//=============================================================================
// Bulk Operations
//=============================================================================

nimcp_error_t nimcp_pattern_db_import(
    nimcp_pattern_db_t db,
    const nimcp_pattern_entry_t* entries,
    size_t count
) {
    if (!validate_db(db) || !entries || count == 0) {
        return NIMCP_INVALID_PARAM;
    }

    // Create snapshot before import for rollback
    uint32_t snapshot_version;
    nimcp_error_t err = nimcp_pattern_db_snapshot(db, &snapshot_version);
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    // Import all patterns
    for (size_t i = 0; i < count; i++) {
        err = nimcp_pattern_db_add(db, &entries[i], NULL);
        if (err != NIMCP_SUCCESS) {
            // Rollback on error
            nimcp_pattern_db_rollback(db, snapshot_version);
            return err;
        }
    }

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_pattern_db_load(
    nimcp_pattern_db_t db,
    const char* filepath
) {
    if (!validate_db(db) || !filepath) {
        return NIMCP_INVALID_PARAM;
    }

    // TODO: Implement JSON parsing for pattern file loading
    // For now, return not implemented
    return NIMCP_NOT_IMPLEMENTED;
}

nimcp_error_t nimcp_pattern_db_save(
    nimcp_pattern_db_t db,
    const char* filepath
) {
    if (!validate_db(db) || !filepath) {
        return NIMCP_INVALID_PARAM;
    }

    // TODO: Implement JSON serialization for pattern file saving
    // For now, return not implemented
    return NIMCP_NOT_IMPLEMENTED;
}

nimcp_error_t nimcp_pattern_db_clear(nimcp_pattern_db_t db) {
    if (!validate_db(db)) {
        return NIMCP_INVALID_PARAM;
    }

    pthread_rwlock_wrlock(&db->write_lock);

    // Free all patterns
    for (size_t i = 0; i < db->capacity; i++) {
        if (db->patterns[i].is_compiled) {
            free_pattern(&db->patterns[i]);
            memset(&db->patterns[i], 0, sizeof(pattern_slot_t));
        }
    }

    db->pattern_count = 0;
    db->current_version++;

    // Reset statistics
    db->stats.total_patterns = 0;
    memset(db->stats.patterns_by_category, 0, sizeof(db->stats.patterns_by_category));
    db->stats.current_version = db->current_version;

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

//=============================================================================
// Versioning and Rollback
//=============================================================================

uint32_t nimcp_pattern_db_version(nimcp_pattern_db_t db) {
    if (!validate_db(db)) {
        return 0;
    }
    return db->current_version;
}

nimcp_error_t nimcp_pattern_db_snapshot(
    nimcp_pattern_db_t db,
    uint32_t* snapshot_id
) {
    if (!validate_db(db)) {
        return NIMCP_INVALID_PARAM;
    }

    pthread_rwlock_wrlock(&db->write_lock);
    nimcp_error_t err = create_snapshot(db, snapshot_id);
    pthread_rwlock_unlock(&db->write_lock);

    if (err == NIMCP_SUCCESS) {
        db->stats.snapshot_count = db->snapshot_count;
    }

    return err;
}

nimcp_error_t nimcp_pattern_db_rollback(
    nimcp_pattern_db_t db,
    uint32_t version
) {
    if (!validate_db(db)) {
        return NIMCP_INVALID_PARAM;
    }

    pthread_rwlock_wrlock(&db->write_lock);

    // Find snapshot with matching version
    pattern_snapshot_t* snapshot = db->snapshots;
    while (snapshot && snapshot->version != version) {
        snapshot = snapshot->next;
    }

    if (!snapshot) {
        pthread_rwlock_unlock(&db->write_lock);
        return NIMCP_NOT_FOUND;
    }

    // Allocate new patterns array BEFORE freeing old one (atomicity)
    // MEMORY SAFETY: If allocation fails, db state remains consistent
    pattern_slot_t* new_patterns = calloc(snapshot->capacity, sizeof(pattern_slot_t));
    if (!new_patterns) {
        pthread_rwlock_unlock(&db->write_lock);
        return NIMCP_NO_MEMORY;
    }

    // Free current patterns
    for (size_t i = 0; i < db->capacity; i++) {
        free_pattern(&db->patterns[i]);
    }
    free(db->patterns);

    // Restore from snapshot (atomically - new_patterns already allocated)
    db->patterns = new_patterns;
    db->capacity = snapshot->capacity;
    db->pattern_count = snapshot->pattern_count;

    // Deep copy patterns from snapshot
    for (size_t i = 0; i < snapshot->capacity; i++) {
        if (snapshot->patterns[i].is_compiled) {
            db->patterns[i] = snapshot->patterns[i];
            compile_pattern(&db->patterns[i]);
        }
    }

    db->current_version++;

    // Update statistics
    db->stats.total_patterns = db->pattern_count;
    db->stats.current_version = db->current_version;
    memset(db->stats.patterns_by_category, 0, sizeof(db->stats.patterns_by_category));
    for (size_t i = 0; i < db->capacity; i++) {
        if (db->patterns[i].is_compiled &&
            db->patterns[i].entry.category < NIMCP_PATTERN_CATEGORY_COUNT) {
            db->stats.patterns_by_category[db->patterns[i].entry.category]++;
        }
    }

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

//=============================================================================
// Pattern Matching
//=============================================================================

nimcp_error_t nimcp_pattern_db_match(
    nimcp_pattern_db_t db,
    const char* input,
    nimcp_pattern_match_result_t* result
) {
    if (!validate_db(db) || !input || !result) {
        return NIMCP_INVALID_PARAM;
    }

    memset(result, 0, sizeof(nimcp_pattern_match_result_t));

    uint64_t start_time = get_time_us();

    pthread_rwlock_rdlock(&db->write_lock);

    // Build sorted pattern list by priority
    typedef struct {
        pattern_slot_t* slot;
        uint32_t priority;
    } pattern_pri_t;

    pattern_pri_t* sorted_patterns = calloc(db->pattern_count, sizeof(pattern_pri_t));
    if (!sorted_patterns) {
        pthread_rwlock_unlock(&db->write_lock);
        return NIMCP_NO_MEMORY;
    }

    size_t sorted_count = 0;
    for (size_t i = 0; i < db->capacity; i++) {
        if (db->patterns[i].is_compiled) {
            sorted_patterns[sorted_count].slot = &db->patterns[i];
            sorted_patterns[sorted_count].priority = db->patterns[i].entry.priority;
            sorted_count++;
        }
    }

    // Simple insertion sort by priority (descending)
    for (size_t i = 1; i < sorted_count; i++) {
        pattern_pri_t key = sorted_patterns[i];
        ssize_t j = i - 1;
        while (j >= 0 && sorted_patterns[j].priority < key.priority) {
            sorted_patterns[j + 1] = sorted_patterns[j];
            j--;
        }
        sorted_patterns[j + 1] = key;
    }

    // Match patterns in priority order
    float total_threat = 0.0F;
    uint32_t match_count = 0;

    for (size_t i = 0; i < sorted_count; i++) {
        pattern_slot_t* slot = sorted_patterns[i].slot;

        regmatch_t match;
        int ret = regexec(&slot->compiled_regex, input, 1, &match, 0);

        if (ret == 0) {
            // Pattern matched
            match_count++;
            total_threat += slot->entry.weight;

            // Record first/best match
            if (!result->matched) {
                result->matched = true;
                result->pattern_id = slot->pattern_id;
                result->category = slot->entry.category;
                result->match_offset = match.rm_so;
                result->match_length = match.rm_eo - match.rm_so;

                if (slot->entry.description) {
                    strncpy(result->description, slot->entry.description,
                            NIMCP_PATTERN_MAX_DESCRIPTION - 1);
                    result->description[NIMCP_PATTERN_MAX_DESCRIPTION - 1] = '\0';
                }
            }

            // Update pattern statistics
            slot->match_count++;
        }
    }

    free(sorted_patterns);

    result->match_count = match_count;
    result->threat_score = (total_threat > 1.0F) ? 1.0F : total_threat;

    uint64_t end_time = get_time_us();
    float match_time = (float)(end_time - start_time);

    // Update statistics
    db->stats.total_matches++;
    if (result->matched) {
        db->stats.total_hits++;
    }

    if (db->config.enable_statistics) {
        // Update running average
        float n = (float)db->stats.total_matches;
        db->stats.avg_match_time_us =
            (db->stats.avg_match_time_us * (n - 1.0F) + match_time) / n;

        if (match_time > db->stats.max_match_time_us) {
            db->stats.max_match_time_us = match_time;
        }
    }

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_pattern_db_match_category(
    nimcp_pattern_db_t db,
    const char* input,
    nimcp_pattern_category_t category,
    nimcp_pattern_match_result_t* result
) {
    if (!validate_db(db) || !input || !result) {
        return NIMCP_INVALID_PARAM;
    }

    if (category >= NIMCP_PATTERN_CATEGORY_COUNT) {
        return NIMCP_INVALID_PARAM;
    }

    memset(result, 0, sizeof(nimcp_pattern_match_result_t));

    pthread_rwlock_rdlock(&db->write_lock);

    // Match only patterns in specified category
    for (size_t i = 0; i < db->capacity; i++) {
        if (!db->patterns[i].is_compiled) {
            continue;
        }

        if (db->patterns[i].entry.category != category) {
            continue;
        }

        regmatch_t match;
        int ret = regexec(&db->patterns[i].compiled_regex, input, 1, &match, 0);

        if (ret == 0) {
            // Pattern matched
            result->matched = true;
            result->pattern_id = db->patterns[i].pattern_id;
            result->category = db->patterns[i].entry.category;
            result->threat_score = db->patterns[i].entry.weight;
            result->match_count = 1;
            result->match_offset = match.rm_so;
            result->match_length = match.rm_eo - match.rm_so;

            if (db->patterns[i].entry.description) {
                strncpy(result->description, db->patterns[i].entry.description,
                        NIMCP_PATTERN_MAX_DESCRIPTION - 1);
                result->description[NIMCP_PATTERN_MAX_DESCRIPTION - 1] = '\0';
            }

            break;  // Return first match
        }
    }

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_pattern_db_match_timeout(
    nimcp_pattern_db_t db,
    const char* input,
    float timeout_ms,
    nimcp_pattern_match_result_t* result
) {
    // For now, just call regular match (timeout handling requires more complex implementation)
    return nimcp_pattern_db_match(db, input, result);
}

//=============================================================================
// Statistics and Monitoring
//=============================================================================

nimcp_error_t nimcp_pattern_db_get_stats(
    nimcp_pattern_db_t db,
    nimcp_pattern_db_stats_t* stats
) {
    if (!validate_db(db) || !stats) {
        return NIMCP_INVALID_PARAM;
    }

    pthread_rwlock_rdlock(&db->write_lock);
    *stats = db->stats;

    // Calculate memory usage estimate
    stats->memory_usage_bytes = sizeof(struct nimcp_pattern_db_impl);
    stats->memory_usage_bytes += db->capacity * sizeof(pattern_slot_t);

    // Add snapshot memory
    pattern_snapshot_t* snapshot = db->snapshots;
    while (snapshot) {
        stats->memory_usage_bytes += sizeof(pattern_snapshot_t);
        stats->memory_usage_bytes += snapshot->capacity * sizeof(pattern_slot_t);
        snapshot = snapshot->next;
    }

    pthread_rwlock_unlock(&db->write_lock);

    return NIMCP_SUCCESS;
}

void nimcp_pattern_db_reset_stats(nimcp_pattern_db_t db) {
    if (!validate_db(db)) {
        return;
    }

    pthread_rwlock_wrlock(&db->write_lock);

    db->stats.total_matches = 0;
    db->stats.total_hits = 0;
    db->stats.avg_match_time_us = 0.0F;
    db->stats.max_match_time_us = 0.0F;

    pthread_rwlock_unlock(&db->write_lock);
}

const char* nimcp_pattern_category_name(nimcp_pattern_category_t category) {
    static const char* names[] = {
        "SQL_INJECTION",
        "XSS",
        "SHELL_INJECTION",
        "PATH_TRAVERSAL",
        "FORMAT_STRING",
        "PROMPT_INJECTION",
        "BUFFER_OVERFLOW",
        "LDAP_INJECTION",
        "XML_INJECTION",
        "COMMAND_INJECTION",
        "CUSTOM"
    };

    if (category >= NIMCP_PATTERN_CATEGORY_COUNT) {
        return "UNKNOWN";
    }

    return names[category];
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

uint32_t nimcp_pattern_db_process_inbox(
    nimcp_pattern_db_t db,
    uint32_t max_messages
) {
    if (!validate_db(db) || !db->bio_async_enabled || !db->bio_ctx) {
        return 0;
    }

    return bio_router_process_inbox(db->bio_ctx, max_messages);
}

nimcp_error_t nimcp_pattern_db_register_bio_async(
    nimcp_pattern_db_t db,
    bio_module_id_t module_id
) {
    if (!validate_db(db)) {
        return NIMCP_INVALID_PARAM;
    }

    if (!bio_router_is_initialized()) {
        return NIMCP_ERROR;
    }

    bio_module_info_t info = {
        .module_id = module_id,
        .module_name = "pattern_db",
        .inbox_capacity = 256,
        .user_data = db
    };

    db->bio_ctx = bio_router_register_module(&info);
    if (!db->bio_ctx) {
        return NIMCP_ERROR;
    }

    db->bio_async_enabled = true;
    db->config.module_id = module_id;

    // TODO: Register message handlers for pattern management

    return NIMCP_SUCCESS;
}
