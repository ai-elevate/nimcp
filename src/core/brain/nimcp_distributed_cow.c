//=============================================================================
// nimcp_distributed_cow.c - Distributed COW Implementation
//=============================================================================

#include "nimcp_distributed_cow.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>  // For compression
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "networking/protocol/nimcp_protocol.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/platform/nimcp_platform_rwlock.h"

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Internal brain structure extension for distributed COW
 *
 * NOTE: This is added to brain_struct in nimcp_brain.c
 * For now, we'll use a separate tracking structure
 */
typedef struct {
    brain_t brain;                       /**< Associated brain */
    distributed_cow_state_t* dcow_state; /**< Distributed COW state */
} distributed_cow_brain_t;

// Global registry for distributed COW brains
static distributed_cow_brain_t** g_dcow_brains = NULL;
static uint32_t g_num_dcow_brains = 0;
static nimcp_platform_mutex_t g_registry_mutex;
static bool g_registry_mutex_initialized = false;

//=============================================================================
// Registry Initialization
//=============================================================================

/**
 * @brief Ensure global registry mutex is initialized
 */
static void ensure_registry_mutex_initialized(void) {
    if (!g_registry_mutex_initialized) {
        nimcp_platform_mutex_init(&g_registry_mutex, false);
        g_registry_mutex_initialized = true;
    }
}

//=============================================================================
// Network Segment Serialization
//=============================================================================

/**
 * @brief Serialize network segment
 *
 * WHAT: Converts neurons and synapses to wire format
 * WHY:  Enable network transfer
 * HOW:  Packs neuron states, synapse weights, and connectivity
 *
 * @param network Adaptive network
 * @param start_neuron First neuron ID
 * @param num_neurons Number of neurons
 * @param buffer Output buffer
 * @param buffer_size Buffer size
 * @param enable_compression Compress output?
 * @return Bytes written, or -1 on error
 */
__attribute__((unused))
static int serialize_network_segment(
    adaptive_network_t network,
    uint32_t start_neuron,
    uint32_t num_neurons,
    uint8_t* buffer,
    size_t buffer_size,
    bool enable_compression
) {
    if (!network || !buffer) {
        return -1;
    }

    // Get network info
    // TODO: Use actual network API when available
    uint32_t total_neurons = 10000; // Stub
    if (start_neuron + num_neurons > total_neurons) {
        num_neurons = total_neurons - start_neuron;
    }
    (void)network;  // Suppress unused warning

    // Temporary buffer for uncompressed data
    size_t temp_buffer_size = num_neurons * (sizeof(float) * 2 + sizeof(uint32_t) * 10); // Estimate
    uint8_t* temp_buffer = nimcp_malloc(temp_buffer_size);
    if (!temp_buffer) {
        return -1;
    }

    size_t offset = 0;

    // Write header
    *(uint32_t*)(temp_buffer + offset) = start_neuron;
    offset += sizeof(uint32_t);
    *(uint32_t*)(temp_buffer + offset) = num_neurons;
    offset += sizeof(uint32_t);

    // For each neuron, write state and connections
    // TODO: Use actual network API when available
    for (uint32_t i = start_neuron; i < start_neuron + num_neurons; i++) {
        // Stub neuron data
        float activation = 0.0f;
        float membrane = 0.0f;

        *(float*)(temp_buffer + offset) = activation;
        offset += sizeof(float);
        *(float*)(temp_buffer + offset) = membrane;
        offset += sizeof(float);

        // Stub synapses
        uint32_t num_synapses = 10; // Stub

        *(uint32_t*)(temp_buffer + offset) = num_synapses;
        offset += sizeof(uint32_t);

        // Write synapses (target ID + weight)
        for (uint32_t s = 0; s < num_synapses; s++) {
            if (offset + sizeof(uint32_t) + sizeof(float) > temp_buffer_size) {
                // Realloc if needed
                temp_buffer_size *= 2;
                temp_buffer = nimcp_realloc(temp_buffer, temp_buffer_size);
            }

            *(uint32_t*)(temp_buffer + offset) = i + s; // Stub target
            offset += sizeof(uint32_t);
            *(float*)(temp_buffer + offset) = 0.5f; // Stub weight
            offset += sizeof(float);
        }
    }

    // Compress if requested
    if (enable_compression && offset > 1024) {
        uLongf compressed_size = compressBound(offset);
        uint8_t* compressed_buffer = nimcp_malloc(compressed_size);

        if (compressed_buffer) {
            int result = compress2(compressed_buffer, &compressed_size, temp_buffer, offset, Z_DEFAULT_COMPRESSION);

            if (result == Z_OK && compressed_size < offset) {
                // Compression successful and beneficial
                if (compressed_size + 1 <= buffer_size) {
                    buffer[0] = 1; // Compression flag
                    memcpy(buffer + 1, compressed_buffer, compressed_size);
                    nimcp_free(compressed_buffer);
                    nimcp_free(temp_buffer);
                    return compressed_size + 1;
                }
            }
            nimcp_free(compressed_buffer);
        }
    }

    // No compression or compression failed - use uncompressed
    if (offset + 1 <= buffer_size) {
        buffer[0] = 0; // No compression
        memcpy(buffer + 1, temp_buffer, offset);
        nimcp_free(temp_buffer);
        return offset + 1;
    }

    nimcp_free(temp_buffer);
    return -1; // Buffer too small
}

/**
 * @brief Deserialize network segment
 *
 * @param buffer Input buffer
 * @param buffer_size Buffer size
 * @param network Network to populate
 * @return true on success
 */
static bool deserialize_network_segment(
    const uint8_t* buffer,
    size_t buffer_size,
    adaptive_network_t network
) {
    if (!buffer || !network || buffer_size < 1) {
        return false;
    }

    bool is_compressed = (buffer[0] == 1);
    const uint8_t* data = buffer + 1;
    size_t data_size = buffer_size - 1;

    // Decompress if needed
    uint8_t* decompressed_buffer = NULL;
    if (is_compressed) {
        uLongf decompressed_size = data_size * 10; // Estimate
        decompressed_buffer = nimcp_malloc(decompressed_size);

        if (!decompressed_buffer) {
            return false;
        }

        int result = uncompress(decompressed_buffer, &decompressed_size, data, data_size);
        if (result != Z_OK) {
            nimcp_free(decompressed_buffer);
            return false;
        }

        data = decompressed_buffer;
        data_size = decompressed_size;
    }

    size_t offset = 0;

    // Read header
    uint32_t start_neuron = *(const uint32_t*)(data + offset);
    offset += sizeof(uint32_t);
    uint32_t num_neurons = *(const uint32_t*)(data + offset);
    offset += sizeof(uint32_t);

    // Read neuron data
    for (uint32_t i = 0; i < num_neurons; i++) {
        uint32_t neuron_id = start_neuron + i;

        // Read neuron state
        float activation = *(const float*)(data + offset);
        offset += sizeof(float);
        float membrane = *(const float*)(data + offset);
        offset += sizeof(float);

        // TODO: Set neuron state when API available
        (void)neuron_id;
        (void)activation;
        (void)membrane;

        // Read synapses
        uint32_t num_synapses = *(const uint32_t*)(data + offset);
        offset += sizeof(uint32_t);

        for (uint32_t s = 0; s < num_synapses; s++) {
            uint32_t target_id = *(const uint32_t*)(data + offset);
            offset += sizeof(uint32_t);
            float weight = *(const float*)(data + offset);
            offset += sizeof(float);

            // TODO: Set synapse weight when API available
            (void)target_id;
            (void)weight;
        }
    }

    if (decompressed_buffer) {
        nimcp_free(decompressed_buffer);
    }

    return true;
}

//=============================================================================
// Cache Management
//=============================================================================

/**
 * @brief Find cached segment
 */
static network_segment_t* find_cached_segment(
    distributed_cow_state_t* state,
    uint32_t start_neuron,
    uint32_t num_neurons
) {
    nimcp_platform_rwlock_rdlock(&state->cache_lock);

    for (uint32_t i = 0; i < state->num_cached_segments; i++) {
        network_segment_t* seg = state->cached_segments[i];
        if (seg->start_neuron_id == start_neuron && seg->num_neurons == num_neurons) {
            state->cache_hits++;
            nimcp_platform_rwlock_rdunlock(&state->cache_lock);
            return seg;
        }
    }

    state->cache_misses++;
    nimcp_platform_rwlock_rdunlock(&state->cache_lock);
    return NULL;
}

/**
 * @brief Add segment to cache
 */
static bool add_to_cache(
    distributed_cow_state_t* state,
    network_segment_t* segment
) {
    nimcp_platform_rwlock_wrlock(&state->cache_lock);

    // Check capacity
    if (state->num_cached_segments >= state->cache_capacity) {
        // Evict LRU segment
        // For simplicity, evict first segment (FIFO)
        network_segment_t* evicted = state->cached_segments[0];
        state->cache_size_bytes -= evicted->uncompressed_size;
        nimcp_free(evicted);

        // Shift remaining segments
        for (uint32_t i = 0; i < state->num_cached_segments - 1; i++) {
            state->cached_segments[i] = state->cached_segments[i + 1];
        }
        state->num_cached_segments--;
    }

    // Add new segment
    state->cached_segments[state->num_cached_segments++] = segment;
    state->cache_size_bytes += segment->uncompressed_size;

    nimcp_platform_rwlock_wrunlock(&state->cache_lock);
    return true;
}

//=============================================================================
// Network Protocol Handlers
//=============================================================================

/**
 * @brief Handle COW fetch segment request (master side)
 */
__attribute__((unused))
static bool handle_fetch_segment_request(
    brain_t brain,
    const cow_fetch_segment_request_t* request,
    uint8_t* response_buffer,
    size_t response_buffer_size,
    size_t* response_length
) {
    // Serialize network segment
    adaptive_network_t network = brain_get_network(brain);
    if (!network) {
        return false;
    }

    // Prepare response header
    cow_segment_data_response_t response = {
        .segment_id = nimcp_time_get_us(),
        .start_neuron_id = request->start_neuron_id,
        .num_neurons = request->num_neurons,
        .num_synapses = 0, // Will be calculated during serialization
        .is_compressed = request->enable_compression,
        .data_length = 0
    };

    // Serialize segment data
    uint8_t* data_buffer = response_buffer + sizeof(cow_segment_data_response_t);
    size_t data_buffer_size = response_buffer_size - sizeof(cow_segment_data_response_t);

    // TODO: Implement actual serialization
    // For now, return stub response
    (void)network;
    (void)request;
    (void)data_buffer;
    (void)data_buffer_size;

    int bytes_written = 1024; // Stub

    if (bytes_written < 0) {
        return false;
    }

    response.data_length = bytes_written;

    // Write response header
    memcpy(response_buffer, &response, sizeof(cow_segment_data_response_t));
    *response_length = sizeof(cow_segment_data_response_t) + bytes_written;

    return true;
}

//=============================================================================
// Distributed COW State Management
//=============================================================================

/**
 * @brief Create distributed COW state
 */
static distributed_cow_state_t* create_distributed_cow_state(
    const char* master_host,
    uint16_t master_port,
    p2p_node_t p2p_node,
    const distributed_cow_config_t* config
) {
    distributed_cow_state_t* state = nimcp_calloc(1, sizeof(distributed_cow_state_t));
    if (!state) {
        return NULL;
    }

    state->is_distributed = true;
    state->is_master = false;
    strncpy(state->master_host, master_host, sizeof(state->master_host) - 1);
    state->master_port = master_port;
    state->p2p_node = p2p_node;

    // Copy or use default config
    if (config) {
        state->config = *config;
    } else {
        state->config = distributed_cow_default_config();
    }

    // Allocate cache
    state->cache_capacity = (state->config.cache_capacity_mb * 1024 * 1024) /
                           (state->config.segment_size * 100); // Estimate
    state->cached_segments = nimcp_calloc(state->cache_capacity, sizeof(network_segment_t*));
    state->num_cached_segments = 0;
    state->cache_size_bytes = 0;

    // Generate unique clone ID
    state->clone_id = nimcp_time_get_us();
    state->local_refcount = 1;
    state->remote_refcount = 0;

    // Initialize locks
    nimcp_platform_rwlock_init(&state->cache_lock);
    nimcp_platform_mutex_init(&state->fetch_mutex, false);

    // Initialize stats
    state->total_fetches = 0;
    state->total_bytes_fetched = 0;
    state->cache_hits = 0;
    state->cache_misses = 0;
    state->avg_fetch_latency_ms = 0.0f;

    return state;
}

/**
 * @brief Destroy distributed COW state
 */
__attribute__((unused))
static void destroy_distributed_cow_state(distributed_cow_state_t* state) {
    if (!state) {
        return;
    }

    // Free cached segments
    nimcp_platform_rwlock_wrlock(&state->cache_lock);
    for (uint32_t i = 0; i < state->num_cached_segments; i++) {
        nimcp_free(state->cached_segments[i]);
    }
    nimcp_free(state->cached_segments);
    nimcp_platform_rwlock_wrunlock(&state->cache_lock);

    // Destroy locks
    nimcp_platform_rwlock_destroy(&state->cache_lock);
    nimcp_platform_mutex_destroy(&state->fetch_mutex);

    nimcp_free(state);
}

//=============================================================================
// Registry Management
//=============================================================================

/**
 * @brief Register distributed COW brain
 */
static bool register_distributed_cow_brain(brain_t brain, distributed_cow_state_t* state) {
    ensure_registry_mutex_initialized();
    nimcp_platform_mutex_lock(&g_registry_mutex);

    // Allocate registry if needed
    if (!g_dcow_brains) {
        g_dcow_brains = nimcp_calloc(16, sizeof(distributed_cow_brain_t*));
    }

    // Find empty slot or expand
    distributed_cow_brain_t* entry = nimcp_calloc(1, sizeof(distributed_cow_brain_t));
    entry->brain = brain;
    entry->dcow_state = state;

    g_dcow_brains[g_num_dcow_brains++] = entry;

    nimcp_platform_mutex_unlock(&g_registry_mutex);
    return true;
}

/**
 * @brief Find distributed COW state for brain
 */
static distributed_cow_state_t* find_distributed_cow_state(brain_t brain) {
    ensure_registry_mutex_initialized();
    nimcp_platform_mutex_lock(&g_registry_mutex);

    for (uint32_t i = 0; i < g_num_dcow_brains; i++) {
        if (g_dcow_brains[i]->brain == brain) {
            distributed_cow_state_t* state = g_dcow_brains[i]->dcow_state;
            nimcp_platform_mutex_unlock(&g_registry_mutex);
            return state;
        }
    }

    nimcp_platform_mutex_unlock(&g_registry_mutex);
    return NULL;
}

//=============================================================================
// Public API Implementation
//=============================================================================

/**
 * @brief Create distributed COW clone
 */
brain_t brain_clone_cow_distributed(
    brain_t original,
    const char* remote_host,
    uint16_t remote_port,
    const distributed_cow_config_t* config
) {
    if (!original || !remote_host) {
        return NULL;
    }

    // Create local COW clone first
    brain_t clone = brain_clone_cow(original);
    if (!clone) {
        return NULL;
    }

    // Create P2P node for communication
    node_config_t node_config = {
        .listen_port = 0, // Any available port
        .max_peers = 4,
        .ping_interval = 5000
    };
    p2p_node_t p2p_node = p2p_node_create(&node_config);
    if (!p2p_node) {
        brain_destroy(clone);
        return NULL;
    }

    // Start P2P node
    p2p_node_start(p2p_node);

    // Connect to master
    if (!p2p_node_connect_peer(p2p_node, remote_host, remote_port)) {
        p2p_node_destroy(p2p_node);
        brain_destroy(clone);
        return NULL;
    }

    // Create distributed COW state
    distributed_cow_state_t* dcow_state = create_distributed_cow_state(
        remote_host, remote_port, p2p_node, config
    );
    if (!dcow_state) {
        p2p_node_destroy(p2p_node);
        brain_destroy(clone);
        return NULL;
    }

    // Register clone
    register_distributed_cow_brain(clone, dcow_state);

    // Send create clone message to master
    // (Simplified - actual implementation would use protocol messages)

    return clone;
}

/**
 * @brief Enable distributed COW master
 */
bool brain_enable_distributed_cow_master(brain_t brain, p2p_node_t p2p_node) {
    if (!brain || !p2p_node) {
        return false;
    }

    // Create master state
    distributed_cow_state_t* state = nimcp_calloc(1, sizeof(distributed_cow_state_t));
    if (!state) {
        return false;
    }

    state->is_distributed = true;
    state->is_master = true;
    state->p2p_node = p2p_node;
    state->local_refcount = 1;
    state->remote_refcount = 0;

    nimcp_platform_rwlock_init(&state->cache_lock);
    nimcp_platform_mutex_init(&state->fetch_mutex, false);

    // Register as master
    register_distributed_cow_brain(brain, state);

    // Install P2P message handlers
    // (Simplified - actual implementation would register protocol handlers)

    return true;
}

/**
 * @brief Fetch network segment
 */
bool distributed_cow_fetch_segment(
    brain_t brain,
    uint32_t start_neuron_id,
    uint32_t num_neurons
) {
    distributed_cow_state_t* state = find_distributed_cow_state(brain);
    if (!state || !state->is_distributed || state->is_master) {
        return false;
    }

    // Check cache first
    network_segment_t* cached = find_cached_segment(state, start_neuron_id, num_neurons);
    if (cached) {
        return true; // Already cached
    }

    nimcp_platform_mutex_lock(&state->fetch_mutex);

    uint64_t start_time = nimcp_time_get_us();

    // Build fetch request
    // Build fetch request (unused in stub)
    (void)start_neuron_id;
    (void)num_neurons;

    // Send request via P2P
    // (Simplified - actual implementation would use p2p_node_send_message)

    // For demo purposes, simulate network fetch
    uint8_t response_buffer[65536];
    size_t response_length = 0;

    // Simulate response (in real implementation, this would come from network)
    // handle_fetch_segment_request(brain, &request, response_buffer, sizeof(response_buffer), &response_length);

    // Deserialize response
    adaptive_network_t network = brain_get_network(brain);
    if (network && response_length > sizeof(cow_segment_data_response_t)) {
        const cow_segment_data_response_t* response = (const cow_segment_data_response_t*)response_buffer;

        // Deserialize network data
        deserialize_network_segment(
            response_buffer + sizeof(cow_segment_data_response_t),
            response->data_length,
            network
        );

        // Create cache entry
        network_segment_t* segment = nimcp_calloc(1, sizeof(network_segment_t));
        segment->start_neuron_id = response->start_neuron_id;
        segment->num_neurons = response->num_neurons;
        segment->num_synapses = response->num_synapses;
        segment->segment_id = response->segment_id;
        segment->timestamp = nimcp_time_get_us();
        segment->is_compressed = response->is_compressed;
        segment->compressed_size = response->data_length;
        segment->uncompressed_size = num_neurons * 1024; // Estimate

        add_to_cache(state, segment);
    }

    // Update stats
    uint64_t latency_us = nimcp_time_get_us() - start_time;
    float latency_ms = latency_us / 1000.0f;

    state->total_fetches++;
    state->total_bytes_fetched += response_length;
    state->avg_fetch_latency_ms = (state->avg_fetch_latency_ms * (state->total_fetches - 1) + latency_ms) / state->total_fetches;

    nimcp_platform_mutex_unlock(&state->fetch_mutex);

    return true;
}

/**
 * @brief Prefetch segments
 */
uint32_t distributed_cow_prefetch_segments(brain_t brain, uint32_t current_neuron_id) {
    distributed_cow_state_t* state = find_distributed_cow_state(brain);
    if (!state || !state->config.enable_prefetch) {
        return 0;
    }

    uint32_t prefetched = 0;
    uint32_t segment_size = state->config.segment_size;
    uint32_t lookahead = state->config.prefetch_lookahead;

    // Prefetch next N segments
    for (uint32_t offset = segment_size; offset < lookahead; offset += segment_size) {
        uint32_t start_neuron = current_neuron_id + offset;
        if (distributed_cow_fetch_segment(brain, start_neuron, segment_size)) {
            prefetched++;
        }
    }

    return prefetched;
}

/**
 * @brief Fetch full network
 */
bool distributed_cow_fetch_full_network(brain_t brain) {
    distributed_cow_state_t* state = find_distributed_cow_state(brain);
    if (!state || !state->is_distributed) {
        return false;
    }

    adaptive_network_t network = brain_get_network(brain);
    // TODO: Use actual network API
    uint32_t total_neurons = 10000; // Stub
    (void)network;
    uint32_t segment_size = state->config.segment_size;

    // Fetch all segments
    for (uint32_t start = 0; start < total_neurons; start += segment_size) {
        uint32_t num_neurons = (start + segment_size > total_neurons) ?
                               (total_neurons - start) : segment_size;

        if (!distributed_cow_fetch_segment(brain, start, num_neurons)) {
            return false;
        }
    }

    // Transition to local COW
    state->is_distributed = false;

    return true;
}

/**
 * @brief Get distributed COW statistics
 */
bool brain_get_distributed_cow_stats(brain_t brain, distributed_cow_stats_t* stats) {
    if (!stats) {
        return false;
    }

    distributed_cow_state_t* state = find_distributed_cow_state(brain);
    if (!state) {
        memset(stats, 0, sizeof(distributed_cow_stats_t));
        stats->is_distributed = false;
        return false;
    }

    nimcp_platform_rwlock_rdlock(&state->cache_lock);

    stats->is_distributed = state->is_distributed;
    stats->is_master = state->is_master;
    stats->local_refcount = state->local_refcount;
    stats->remote_refcount = state->remote_refcount;
    stats->num_cached_segments = state->num_cached_segments;
    stats->cache_size_bytes = state->cache_size_bytes;
    stats->total_fetches = state->total_fetches;
    stats->total_bytes_fetched = state->total_bytes_fetched;
    stats->cache_hits = state->cache_hits;
    stats->cache_misses = state->cache_misses;

    uint64_t total_accesses = state->cache_hits + state->cache_misses;
    stats->cache_hit_rate = total_accesses > 0 ? (float)state->cache_hits / total_accesses : 0.0f;
    stats->avg_fetch_latency_ms = state->avg_fetch_latency_ms;

    // Calculate bandwidth (bytes/sec)
    if (state->avg_fetch_latency_ms > 0) {
        float bytes_per_ms = state->total_bytes_fetched / (state->total_fetches * state->avg_fetch_latency_ms);
        stats->network_bandwidth_mbps = (bytes_per_ms * 1000.0f * 8.0f) / (1024.0f * 1024.0f);
    } else {
        stats->network_bandwidth_mbps = 0.0f;
    }

    nimcp_platform_rwlock_rdunlock(&state->cache_lock);

    return true;
}

/**
 * @brief Check if brain is distributed COW
 */
bool brain_is_distributed_cow(brain_t brain) {
    distributed_cow_state_t* state = find_distributed_cow_state(brain);
    return state != NULL && state->is_distributed;
}

/**
 * @brief Clear cache
 */
size_t distributed_cow_clear_cache(brain_t brain, uint32_t target_size_mb) {
    distributed_cow_state_t* state = find_distributed_cow_state(brain);
    if (!state) {
        return 0;
    }

    size_t target_bytes = target_size_mb * 1024 * 1024;
    size_t bytes_freed = 0;

    nimcp_platform_rwlock_wrlock(&state->cache_lock);

    while (state->cache_size_bytes > target_bytes && state->num_cached_segments > 0) {
        // Evict oldest segment (FIFO)
        network_segment_t* evicted = state->cached_segments[0];
        bytes_freed += evicted->uncompressed_size;
        state->cache_size_bytes -= evicted->uncompressed_size;
        nimcp_free(evicted);

        // Shift remaining segments
        for (uint32_t i = 0; i < state->num_cached_segments - 1; i++) {
            state->cached_segments[i] = state->cached_segments[i + 1];
        }
        state->num_cached_segments--;
    }

    nimcp_platform_rwlock_wrunlock(&state->cache_lock);

    return bytes_freed;
}
