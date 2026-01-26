/**
 * @file nimcp_graph_dao.c
 * @brief Data Access Object Pattern Implementation for GPU Graphs
 *
 * WHAT: Concrete DAO implementation for GPU graph storage
 * WHY:  Efficient graph data management with GPU-CPU synchronization
 * HOW:  LRU cache + GPU storage with thread-safe operations
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#include "gpu/graph/nimcp_graph_dao.h"
#include "gpu/graph/nimcp_graph_gpu.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for graph_dao module */
static nimcp_health_agent_t* g_graph_dao_health_agent = NULL;

/**
 * @brief Set health agent for graph_dao heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void graph_dao_set_health_agent(nimcp_health_agent_t* agent) {
    g_graph_dao_health_agent = agent;
}

/** @brief Send heartbeat from graph_dao module */
static inline void graph_dao_heartbeat(const char* operation, float progress) {
    if (g_graph_dao_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_graph_dao_health_agent, operation, progress);
    }
}


//=============================================================================
// Internal Constants
//=============================================================================

#define HASH_TABLE_SIZE 256

//=============================================================================
// Forward Declarations (Internal)
//=============================================================================

// LRU Cache Operations
static nimcp_graph_lru_cache_t* lru_cache_create(size_t capacity);
static void lru_cache_destroy(nimcp_graph_lru_cache_t* cache);
static nimcp_graph_cache_entry_t* lru_cache_get(nimcp_graph_lru_cache_t* cache, int id);
static bool lru_cache_put(nimcp_graph_lru_cache_t* cache, int id, void* data);
static void lru_cache_remove(nimcp_graph_lru_cache_t* cache, int id);
static void lru_cache_touch(nimcp_graph_lru_cache_t* cache, nimcp_graph_cache_entry_t* entry);
static nimcp_graph_cache_entry_t* lru_cache_evict(nimcp_graph_lru_cache_t* cache);

// DAO Operations (vtable implementations)
static int dao_create_impl(void* dao, void* graph_data);
static int dao_read_impl(void* dao, int id, void** out_data);
static int dao_update_impl(void* dao, int id, void* graph_data);
static int dao_delete_impl(void* dao, int id);
static int dao_query_impl(void* dao, nimcp_graph_query_t* query, void** results);
static int dao_batch_insert_impl(void* dao, void** graphs, size_t count);
static int dao_sync_to_gpu_impl(void* dao);
static int dao_sync_from_gpu_impl(void* dao);

// Internal helpers
static uint64_t get_timestamp_us(void);
static int hash_id(int id);
static int find_gpu_slot(nimcp_gpu_graph_dao_t* dao, int id);
static int allocate_gpu_slot(nimcp_gpu_graph_dao_t* dao);

//=============================================================================
// Factory Functions
//=============================================================================

nimcp_gpu_graph_dao_t* nimcp_graph_dao_create_gpu(
    nimcp_gpu_context_t* gpu_context,
    size_t cache_size)
{
    if (!gpu_context) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "gpu_context is NULL");

        return NULL;
    }

    nimcp_gpu_graph_dao_t* dao = (nimcp_gpu_graph_dao_t*)calloc(
        1, sizeof(nimcp_gpu_graph_dao_t));
    if (!dao) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "dao is NULL");

        return NULL;
    }

    // Set up vtable
    dao->ops.create = dao_create_impl;
    dao->ops.read = dao_read_impl;
    dao->ops.update = dao_update_impl;
    dao->ops.delete_graph = dao_delete_impl;
    dao->ops.query = dao_query_impl;
    dao->ops.batch_insert = dao_batch_insert_impl;
    dao->ops.sync_to_gpu = dao_sync_to_gpu_impl;
    dao->ops.sync_from_gpu = dao_sync_from_gpu_impl;

    dao->gpu_ctx = gpu_context;

    // Initialize cache
    dao->cache_size = (cache_size > 0) ? cache_size : NIMCP_DAO_DEFAULT_CACHE_SIZE;
    dao->host_cache = lru_cache_create(dao->cache_size);
    if (!dao->host_cache) {
        free(dao);
        return NULL;
    }

    // Initialize GPU storage
    dao->gpu_capacity = NIMCP_DAO_DEFAULT_GPU_CAPACITY;
    dao->gpu_storage = (void**)calloc(dao->gpu_capacity, sizeof(void*));
    dao->gpu_ids = (int*)calloc(dao->gpu_capacity, sizeof(int));
    if (!dao->gpu_storage || !dao->gpu_ids) {
        lru_cache_destroy(dao->host_cache);
        free(dao->gpu_storage);
        free(dao->gpu_ids);
        free(dao);
        return NULL;
    }

    // Initialize IDs to invalid
    for (size_t i = 0; i < dao->gpu_capacity; i++) {
        dao->gpu_ids[i] = -1;
    }

    dao->gpu_count = 0;
    dao->next_id = 1;

    // Initialize mutex
    if (pthread_mutex_init(&dao->lock, NULL) != 0) {
        lru_cache_destroy(dao->host_cache);
        free(dao->gpu_storage);
        free(dao->gpu_ids);
        free(dao);
        return NULL;
    }

    dao->initialized = true;
    return dao;
}

nimcp_gpu_graph_dao_t* nimcp_graph_dao_create_hybrid(
    nimcp_gpu_context_t* gpu_context,
    size_t gpu_capacity)
{
    nimcp_gpu_graph_dao_t* dao = nimcp_graph_dao_create_gpu(
        gpu_context, NIMCP_DAO_DEFAULT_CACHE_SIZE * 4);  // Larger cache for hybrid

    if (dao && gpu_capacity > 0) {
        // Reallocate GPU storage with custom capacity
        free(dao->gpu_storage);
        free(dao->gpu_ids);

        dao->gpu_capacity = gpu_capacity;
        dao->gpu_storage = (void**)calloc(gpu_capacity, sizeof(void*));
        dao->gpu_ids = (int*)calloc(gpu_capacity, sizeof(int));

        if (!dao->gpu_storage || !dao->gpu_ids) {
            nimcp_graph_dao_destroy(dao);
            return NULL;
        }

        for (size_t i = 0; i < gpu_capacity; i++) {
            dao->gpu_ids[i] = -1;
        }
    }

    return dao;
}

void nimcp_graph_dao_destroy(nimcp_gpu_graph_dao_t* dao)
{
    if (!dao) {
        return;
    }

    pthread_mutex_lock(&dao->lock);

    // Free GPU-resident graphs
    for (size_t i = 0; i < dao->gpu_capacity; i++) {
        if (dao->gpu_storage[i]) {
            nimcp_gpu_graph_destroy((nimcp_gpu_graph_t*)dao->gpu_storage[i]);
        }
    }

    free(dao->gpu_storage);
    free(dao->gpu_ids);

    if (dao->host_cache) {
        lru_cache_destroy(dao->host_cache);
    }

    pthread_mutex_unlock(&dao->lock);
    pthread_mutex_destroy(&dao->lock);

    free(dao);
}

//=============================================================================
// CRUD Operations
//=============================================================================

int nimcp_graph_dao_create(
    nimcp_gpu_graph_dao_t* dao,
    nimcp_gpu_graph_t* graph)
{
    if (!dao || !graph || !dao->initialized) {
        return -1;
    }

    pthread_mutex_lock(&dao->lock);

    // Allocate ID
    int id = dao->next_id++;
    if (dao->next_id >= NIMCP_DAO_MAX_GRAPH_ID) {
        dao->next_id = 1;  // Wrap around
    }

    // Store in GPU storage if space available
    int slot = allocate_gpu_slot(dao);
    if (slot >= 0) {
        dao->gpu_storage[slot] = graph;
        dao->gpu_ids[slot] = id;
        dao->gpu_count++;
    } else {
        // Store in cache only
        if (!lru_cache_put(dao->host_cache, id, graph)) {
            pthread_mutex_unlock(&dao->lock);
            return -1;
        }
    }

    pthread_mutex_unlock(&dao->lock);
    return id;
}

nimcp_error_t nimcp_graph_dao_read(
    nimcp_gpu_graph_dao_t* dao,
    int id,
    nimcp_gpu_graph_t** out_graph)
{
    if (!dao || !out_graph || !dao->initialized) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    *out_graph = NULL;

    pthread_mutex_lock(&dao->lock);

    // Check cache first
    nimcp_graph_cache_entry_t* entry = lru_cache_get(dao->host_cache, id);
    if (entry) {
        *out_graph = (nimcp_gpu_graph_t*)entry->graph_data;
        dao->host_cache->hits++;
        pthread_mutex_unlock(&dao->lock);
        return NIMCP_SUCCESS;
    }

    dao->host_cache->misses++;

    // Check GPU storage
    int slot = find_gpu_slot(dao, id);
    if (slot >= 0) {
        *out_graph = (nimcp_gpu_graph_t*)dao->gpu_storage[slot];

        // Add to cache for faster future access
        lru_cache_put(dao->host_cache, id, *out_graph);

        pthread_mutex_unlock(&dao->lock);
        return NIMCP_SUCCESS;
    }

    pthread_mutex_unlock(&dao->lock);
    return NIMCP_ERROR_NOT_FOUND;
}

nimcp_error_t nimcp_graph_dao_update(
    nimcp_gpu_graph_dao_t* dao,
    int id,
    nimcp_gpu_graph_t* graph)
{
    if (!dao || !graph || !dao->initialized) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&dao->lock);

    // Update in GPU storage
    int slot = find_gpu_slot(dao, id);
    if (slot >= 0) {
        nimcp_gpu_graph_t* old = (nimcp_gpu_graph_t*)dao->gpu_storage[slot];
        if (old != graph) {
            nimcp_gpu_graph_destroy(old);
        }
        dao->gpu_storage[slot] = graph;
    }

    // Update in cache
    nimcp_graph_cache_entry_t* entry = lru_cache_get(dao->host_cache, id);
    if (entry) {
        if (entry->graph_data != graph && entry->graph_data != NULL) {
            // Only free if different and not NULL
            if (slot < 0) {  // Only if not in GPU storage
                nimcp_gpu_graph_destroy((nimcp_gpu_graph_t*)entry->graph_data);
            }
        }
        entry->graph_data = graph;
        entry->dirty = true;
    } else if (slot < 0) {
        // Not in GPU storage or cache - create new entry
        pthread_mutex_unlock(&dao->lock);
        return NIMCP_ERROR_NOT_FOUND;
    }

    pthread_mutex_unlock(&dao->lock);
    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_graph_dao_delete(
    nimcp_gpu_graph_dao_t* dao,
    int id)
{
    if (!dao || !dao->initialized) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&dao->lock);

    bool found = false;

    // Remove from GPU storage
    int slot = find_gpu_slot(dao, id);
    if (slot >= 0) {
        nimcp_gpu_graph_destroy((nimcp_gpu_graph_t*)dao->gpu_storage[slot]);
        dao->gpu_storage[slot] = NULL;
        dao->gpu_ids[slot] = -1;
        dao->gpu_count--;
        found = true;
    }

    // Remove from cache (don't free if was in GPU storage)
    nimcp_graph_cache_entry_t* entry = lru_cache_get(dao->host_cache, id);
    if (entry) {
        if (!found && entry->graph_data) {
            nimcp_gpu_graph_destroy((nimcp_gpu_graph_t*)entry->graph_data);
        }
        lru_cache_remove(dao->host_cache, id);
        found = true;
    }

    pthread_mutex_unlock(&dao->lock);
    return found ? NIMCP_SUCCESS : NIMCP_ERROR_NOT_FOUND;
}

//=============================================================================
// Batch Operations
//=============================================================================

size_t nimcp_graph_dao_batch_insert(
    nimcp_gpu_graph_dao_t* dao,
    nimcp_gpu_graph_t** graphs,
    size_t count,
    int* out_ids)
{
    if (!dao || !graphs || count == 0 || !dao->initialized) {
        return 0;
    }

    size_t inserted = 0;
    for (size_t i = 0; i < count; i++) {
        if (graphs[i]) {
            int id = nimcp_graph_dao_create(dao, graphs[i]);
            if (id > 0) {
                if (out_ids) {
                    out_ids[inserted] = id;
                }
                inserted++;
            }
        }
    }

    return inserted;
}

size_t nimcp_graph_dao_batch_read(
    nimcp_gpu_graph_dao_t* dao,
    const int* ids,
    size_t count,
    nimcp_gpu_graph_t** out_graphs)
{
    if (!dao || !ids || !out_graphs || count == 0 || !dao->initialized) {
        return 0;
    }

    size_t read_count = 0;
    for (size_t i = 0; i < count; i++) {
        nimcp_gpu_graph_t* graph = NULL;
        if (nimcp_graph_dao_read(dao, ids[i], &graph) == NIMCP_SUCCESS) {
            out_graphs[read_count++] = graph;
        }
    }

    return read_count;
}

size_t nimcp_graph_dao_batch_delete(
    nimcp_gpu_graph_dao_t* dao,
    const int* ids,
    size_t count)
{
    if (!dao || !ids || count == 0 || !dao->initialized) {
        return 0;
    }

    size_t deleted = 0;
    for (size_t i = 0; i < count; i++) {
        if (nimcp_graph_dao_delete(dao, ids[i]) == NIMCP_SUCCESS) {
            deleted++;
        }
    }

    return deleted;
}

//=============================================================================
// Query Operations
//=============================================================================

nimcp_error_t nimcp_graph_dao_query(
    nimcp_gpu_graph_dao_t* dao,
    nimcp_graph_query_t* query,
    nimcp_graphql_result_t** results)
{
    if (!dao || !query || !results || !dao->initialized) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    // Create GraphQL executor
    nimcp_graphql_executor_t* exec = nimcp_graphql_executor_create(dao->gpu_ctx);
    if (!exec) {
        return NIMCP_ERROR_NO_MEMORY;
    }

    // Execute query (simplified - would need to iterate over stored graphs)
    nimcp_error_t err = nimcp_graphql_execute_query(exec, query);

    if (err == NIMCP_SUCCESS && query->result) {
        *results = query->result;
        query->result = NULL;  // Transfer ownership
    }

    nimcp_graphql_executor_destroy(exec);
    return err;
}

size_t nimcp_graph_dao_find(
    nimcp_gpu_graph_dao_t* dao,
    const char* filter,
    int* out_ids,
    size_t max_results)
{
    if (!dao || !out_ids || max_results == 0 || !dao->initialized) {
        return 0;
    }

    (void)filter;  // TODO: Implement filter parsing and evaluation

    // For now, return all IDs up to max_results
    return nimcp_graph_dao_list_ids(dao, out_ids, max_results);
}

size_t nimcp_graph_dao_count(
    nimcp_gpu_graph_dao_t* dao,
    const char* filter)
{
    if (!dao || !dao->initialized) {
        return 0;
    }

    (void)filter;  // TODO: Implement filter counting

    pthread_mutex_lock(&dao->lock);
    size_t count = dao->gpu_count + dao->host_cache->count;
    pthread_mutex_unlock(&dao->lock);

    return count;
}

//=============================================================================
// Synchronization
//=============================================================================

nimcp_error_t nimcp_graph_dao_sync_to_gpu(nimcp_gpu_graph_dao_t* dao)
{
    if (!dao || !dao->initialized) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&dao->lock);

    // Sync dirty cache entries to GPU
    // This would involve copying modified graphs from host to device
    // Implementation depends on how graph data is structured

    pthread_mutex_unlock(&dao->lock);
    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_graph_dao_sync_from_gpu(nimcp_gpu_graph_dao_t* dao)
{
    if (!dao || !dao->initialized) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&dao->lock);

    // Sync GPU graphs to host cache
    // This would involve copying graph data from device to host
    // Implementation depends on how graph data is structured

    pthread_mutex_unlock(&dao->lock);
    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_graph_dao_flush(nimcp_gpu_graph_dao_t* dao)
{
    if (!dao || !dao->initialized) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    // Sync first, then clear
    nimcp_error_t err = nimcp_graph_dao_sync_to_gpu(dao);
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    return nimcp_graph_dao_cache_clear(dao, false);
}

//=============================================================================
// Cache Management
//=============================================================================

void nimcp_graph_dao_cache_stats(
    const nimcp_gpu_graph_dao_t* dao,
    uint64_t* out_hits,
    uint64_t* out_misses,
    size_t* out_count)
{
    if (!dao || !dao->host_cache) {
        if (out_hits) *out_hits = 0;
        if (out_misses) *out_misses = 0;
        if (out_count) *out_count = 0;
        return;
    }

    if (out_hits) *out_hits = dao->host_cache->hits;
    if (out_misses) *out_misses = dao->host_cache->misses;
    if (out_count) *out_count = dao->host_cache->count;
}

nimcp_error_t nimcp_graph_dao_cache_clear(
    nimcp_gpu_graph_dao_t* dao,
    bool sync_first)
{
    if (!dao || !dao->initialized) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (sync_first) {
        nimcp_error_t err = nimcp_graph_dao_sync_to_gpu(dao);
        if (err != NIMCP_SUCCESS) {
            return err;
        }
    }

    pthread_mutex_lock(&dao->lock);

    // Clear cache entries (don't free graphs that are in GPU storage)
    while (dao->host_cache->count > 0) {
        nimcp_graph_cache_entry_t* entry = lru_cache_evict(dao->host_cache);
        if (entry) {
            // Check if graph is in GPU storage
            int slot = find_gpu_slot(dao, entry->graph_id);
            if (slot < 0 && entry->graph_data) {
                // Not in GPU storage, safe to free
                nimcp_gpu_graph_destroy((nimcp_gpu_graph_t*)entry->graph_data);
            }
            free(entry);
        }
    }

    pthread_mutex_unlock(&dao->lock);
    return NIMCP_SUCCESS;
}

size_t nimcp_graph_dao_prefetch(
    nimcp_gpu_graph_dao_t* dao,
    const int* ids,
    size_t count)
{
    if (!dao || !ids || count == 0 || !dao->initialized) {
        return 0;
    }

    size_t prefetched = 0;
    for (size_t i = 0; i < count; i++) {
        nimcp_gpu_graph_t* graph = NULL;
        if (nimcp_graph_dao_read(dao, ids[i], &graph) == NIMCP_SUCCESS) {
            prefetched++;
        }
    }

    return prefetched;
}

//=============================================================================
// Utility Functions
//=============================================================================

bool nimcp_graph_dao_exists(
    const nimcp_gpu_graph_dao_t* dao,
    int id)
{
    if (!dao || !dao->initialized) {
        return false;
    }

    nimcp_gpu_graph_dao_t* mut_dao = (nimcp_gpu_graph_dao_t*)dao;
    pthread_mutex_lock(&mut_dao->lock);

    bool exists = false;

    // Check cache
    nimcp_graph_cache_entry_t* entry = lru_cache_get(mut_dao->host_cache, id);
    if (entry) {
        exists = true;
    } else {
        // Check GPU storage
        int slot = find_gpu_slot(mut_dao, id);
        exists = (slot >= 0);
    }

    pthread_mutex_unlock(&mut_dao->lock);
    return exists;
}

void nimcp_graph_dao_stats(
    const nimcp_gpu_graph_dao_t* dao,
    size_t* out_total,
    size_t* out_gpu,
    size_t* out_cached)
{
    if (!dao || !dao->initialized) {
        if (out_total) *out_total = 0;
        if (out_gpu) *out_gpu = 0;
        if (out_cached) *out_cached = 0;
        return;
    }

    if (out_gpu) *out_gpu = dao->gpu_count;
    if (out_cached) *out_cached = dao->host_cache ? dao->host_cache->count : 0;
    if (out_total) *out_total = dao->gpu_count +
                                (dao->host_cache ? dao->host_cache->count : 0);
}

size_t nimcp_graph_dao_list_ids(
    const nimcp_gpu_graph_dao_t* dao,
    int* out_ids,
    size_t max_count)
{
    if (!dao || !out_ids || max_count == 0 || !dao->initialized) {
        return 0;
    }

    nimcp_gpu_graph_dao_t* mut_dao = (nimcp_gpu_graph_dao_t*)dao;
    pthread_mutex_lock(&mut_dao->lock);

    size_t count = 0;

    // List GPU storage IDs
    for (size_t i = 0; i < dao->gpu_capacity && count < max_count; i++) {
        if (dao->gpu_ids[i] >= 0) {
            out_ids[count++] = dao->gpu_ids[i];
        }
    }

    pthread_mutex_unlock(&mut_dao->lock);
    return count;
}

//=============================================================================
// DAO vtable implementations
//=============================================================================

static int dao_create_impl(void* dao, void* graph_data)
{
    return nimcp_graph_dao_create((nimcp_gpu_graph_dao_t*)dao,
                                   (nimcp_gpu_graph_t*)graph_data);
}

static int dao_read_impl(void* dao, int id, void** out_data)
{
    nimcp_gpu_graph_t* graph = NULL;
    nimcp_error_t err = nimcp_graph_dao_read((nimcp_gpu_graph_dao_t*)dao, id, &graph);
    if (err == NIMCP_SUCCESS) {
        *out_data = graph;
        return NIMCP_SUCCESS;
    }
    return (int)err;
}

static int dao_update_impl(void* dao, int id, void* graph_data)
{
    return (int)nimcp_graph_dao_update((nimcp_gpu_graph_dao_t*)dao, id,
                                        (nimcp_gpu_graph_t*)graph_data);
}

static int dao_delete_impl(void* dao, int id)
{
    return (int)nimcp_graph_dao_delete((nimcp_gpu_graph_dao_t*)dao, id);
}

static int dao_query_impl(void* dao, nimcp_graph_query_t* query, void** results)
{
    nimcp_graphql_result_t* result = NULL;
    nimcp_error_t err = nimcp_graph_dao_query((nimcp_gpu_graph_dao_t*)dao,
                                               query, &result);
    if (err == NIMCP_SUCCESS) {
        *results = result;
    }
    return (int)err;
}

static int dao_batch_insert_impl(void* dao, void** graphs, size_t count)
{
    return (int)nimcp_graph_dao_batch_insert((nimcp_gpu_graph_dao_t*)dao,
                                              (nimcp_gpu_graph_t**)graphs,
                                              count, NULL);
}

static int dao_sync_to_gpu_impl(void* dao)
{
    return (int)nimcp_graph_dao_sync_to_gpu((nimcp_gpu_graph_dao_t*)dao);
}

static int dao_sync_from_gpu_impl(void* dao)
{
    return (int)nimcp_graph_dao_sync_from_gpu((nimcp_gpu_graph_dao_t*)dao);
}

//=============================================================================
// LRU Cache Implementation
//=============================================================================

static nimcp_graph_lru_cache_t* lru_cache_create(size_t capacity)
{
    nimcp_graph_lru_cache_t* cache = (nimcp_graph_lru_cache_t*)calloc(
        1, sizeof(nimcp_graph_lru_cache_t));
    if (!cache) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "cache is NULL");

        return NULL;
    }

    cache->entries = (nimcp_graph_cache_entry_t**)calloc(
        HASH_TABLE_SIZE, sizeof(nimcp_graph_cache_entry_t*));
    if (!cache->entries) {
        free(cache);
        return NULL;
    }

    cache->capacity = capacity;
    cache->count = 0;
    cache->head = NULL;
    cache->tail = NULL;
    cache->hits = 0;
    cache->misses = 0;

    return cache;
}

static void lru_cache_destroy(nimcp_graph_lru_cache_t* cache)
{
    if (!cache) {
        return;
    }

    // Free all entries
    nimcp_graph_cache_entry_t* entry = cache->head;
    while (entry) {
        nimcp_graph_cache_entry_t* next = entry->next;
        // Don't free graph_data here - it may be shared with GPU storage
        free(entry);
        entry = next;
    }

    free(cache->entries);
    free(cache);
}

static int hash_id(int id)
{
    return id % HASH_TABLE_SIZE;
}

static nimcp_graph_cache_entry_t* lru_cache_get(nimcp_graph_lru_cache_t* cache, int id)
{
    if (!cache) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "cache is NULL");

        return NULL;
    }

    int hash = hash_id(id);
    nimcp_graph_cache_entry_t* entry = cache->entries[hash];

    // Linear probing in hash chain
    while (entry) {
        if (entry->graph_id == id) {
            lru_cache_touch(cache, entry);
            return entry;
        }
        entry = entry->next;
    }

    return NULL;
}

static bool lru_cache_put(nimcp_graph_lru_cache_t* cache, int id, void* data)
{
    if (!cache) {
        return false;
    }

    // Check if already exists
    nimcp_graph_cache_entry_t* existing = lru_cache_get(cache, id);
    if (existing) {
        existing->graph_data = data;
        existing->dirty = true;
        return true;
    }

    // Evict if at capacity
    while (cache->count >= cache->capacity) {
        nimcp_graph_cache_entry_t* evicted = lru_cache_evict(cache);
        if (evicted) {
            // Note: caller is responsible for freeing graph_data if needed
            free(evicted);
        }
    }

    // Create new entry
    nimcp_graph_cache_entry_t* entry = (nimcp_graph_cache_entry_t*)calloc(
        1, sizeof(nimcp_graph_cache_entry_t));
    if (!entry) {
        return false;
    }

    entry->graph_id = id;
    entry->graph_data = data;
    entry->access_time = get_timestamp_us();
    entry->access_count = 1;
    entry->dirty = false;

    // Add to hash table
    int hash = hash_id(id);
    entry->next = cache->entries[hash];
    if (cache->entries[hash]) {
        cache->entries[hash]->prev = entry;
    }
    cache->entries[hash] = entry;

    // Add to LRU list (front)
    entry->prev = NULL;
    entry->next = cache->head;
    if (cache->head) {
        cache->head->prev = entry;
    }
    cache->head = entry;
    if (!cache->tail) {
        cache->tail = entry;
    }

    cache->count++;
    return true;
}

static void lru_cache_remove(nimcp_graph_lru_cache_t* cache, int id)
{
    if (!cache) {
        return;
    }

    int hash = hash_id(id);
    nimcp_graph_cache_entry_t* entry = cache->entries[hash];
    nimcp_graph_cache_entry_t* prev = NULL;

    while (entry) {
        if (entry->graph_id == id) {
            // Remove from hash chain
            if (prev) {
                prev->next = entry->next;
            } else {
                cache->entries[hash] = entry->next;
            }

            // Remove from LRU list
            if (entry->prev) {
                entry->prev->next = entry->next;
            } else {
                cache->head = entry->next;
            }

            if (entry->next) {
                entry->next->prev = entry->prev;
            } else {
                cache->tail = entry->prev;
            }

            cache->count--;
            free(entry);
            return;
        }
        prev = entry;
        entry = entry->next;
    }
}

static void lru_cache_touch(nimcp_graph_lru_cache_t* cache, nimcp_graph_cache_entry_t* entry)
{
    if (!cache || !entry) {
        return;
    }

    entry->access_time = get_timestamp_us();
    entry->access_count++;

    // Move to front of LRU list
    if (entry == cache->head) {
        return;  // Already at front
    }

    // Remove from current position
    if (entry->prev) {
        entry->prev->next = entry->next;
    }
    if (entry->next) {
        entry->next->prev = entry->prev;
    } else {
        cache->tail = entry->prev;
    }

    // Insert at front
    entry->prev = NULL;
    entry->next = cache->head;
    if (cache->head) {
        cache->head->prev = entry;
    }
    cache->head = entry;
}

static nimcp_graph_cache_entry_t* lru_cache_evict(nimcp_graph_lru_cache_t* cache)
{
    if (!cache || !cache->tail) {
        return NULL;
    }

    nimcp_graph_cache_entry_t* entry = cache->tail;

    // Remove from LRU list
    if (entry->prev) {
        entry->prev->next = NULL;
    } else {
        cache->head = NULL;
    }
    cache->tail = entry->prev;

    // Remove from hash table
    int hash = hash_id(entry->graph_id);
    nimcp_graph_cache_entry_t* curr = cache->entries[hash];
    nimcp_graph_cache_entry_t* prev = NULL;

    while (curr) {
        if (curr == entry) {
            if (prev) {
                prev->next = curr->next;
            } else {
                cache->entries[hash] = curr->next;
            }
            break;
        }
        prev = curr;
        curr = curr->next;
    }

    cache->count--;
    return entry;
}

//=============================================================================
// Internal Helpers
//=============================================================================

static uint64_t get_timestamp_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static int find_gpu_slot(nimcp_gpu_graph_dao_t* dao, int id)
{
    for (size_t i = 0; i < dao->gpu_capacity; i++) {
        if (dao->gpu_ids[i] == id) {
            return (int)i;
        }
    }
    return -1;
}

static int allocate_gpu_slot(nimcp_gpu_graph_dao_t* dao)
{
    for (size_t i = 0; i < dao->gpu_capacity; i++) {
        if (dao->gpu_ids[i] < 0) {
            return (int)i;
        }
    }
    return -1;  // No free slot
}
