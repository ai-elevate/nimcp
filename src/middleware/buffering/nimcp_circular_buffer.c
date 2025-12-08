//=============================================================================
// nimcp_circular_buffer.c - Lock-Free Circular Buffer Implementation
//=============================================================================

#include "middleware/buffering/nimcp_circular_buffer.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "middleware_circular_buffer"

#include <string.h>
#include <stdatomic.h>

// Cache line size for alignment (typical x86-64)
#define CACHE_LINE_SIZE 64

// Align pointer to cache line boundary
#define ALIGN_TO_CACHE_LINE(ptr) \
    ((void*)(((uintptr_t)(ptr) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1)))

/**
 * @brief Circular buffer structure
 *
 * WHAT: Internal representation with lock-free indices
 * WHY:  Enable concurrent single-producer single-consumer access
 * HOW:  Atomic read/write indices, cache-aligned to prevent false sharing
 */
struct circular_buffer {
    // Configuration
    size_t element_size;           /**< Size of each element in bytes */
    size_t capacity;               /**< Maximum number of elements */
    overflow_strategy_t strategy;  /**< Overflow handling */

    // Cache-aligned atomic indices (prevent false sharing)
    _Alignas(CACHE_LINE_SIZE) atomic_size_t write_pos;  /**< Write position */
    _Alignas(CACHE_LINE_SIZE) atomic_size_t read_pos;   /**< Read position */

    // Statistics
    _Alignas(CACHE_LINE_SIZE) circular_buffer_stats_t stats;

    // Data storage (allocated separately for alignment)
    void* data;  /**< Element storage */
};

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

/**
 * @brief Update average utilization
 *
 * WHAT: Incremental average calculation
 * WHY:  Track buffer usage over time without storing all samples
 * HOW:  Running average: new_avg = old_avg + (sample - old_avg) / count
 */
static void update_avg_utilization(circular_buffer_t* buf, size_t current_size) {
    if (!buf) return;

    size_t total_ops = buf->stats.total_writes + buf->stats.total_reads;
    if (total_ops == 0) {
        buf->stats.avg_usage = 0.0f;
        return;
    }

    float current_pct = (buf->capacity > 0) ?
        (100.0f * current_size / buf->capacity) : 0.0f;

    // Incremental average
    buf->stats.avg_usage = buf->stats.avg_usage +
        (current_pct - buf->stats.avg_usage) / total_ops;
}

/**
 * @brief Update peak utilization
 *
 * WHAT: Track maximum buffer occupancy
 * WHY:  Identify if buffer size is adequate
 * HOW:  Update peak if current size exceeds recorded peak
 */
static void update_peak_utilization(circular_buffer_t* buf, size_t current_size) {
    if (!buf) return;
    if (current_size > buf->stats.peak_usage) {
        buf->stats.peak_usage = current_size;
    }
}

//=============================================================================
// LIFECYCLE
//=============================================================================

circular_buffer_t* circular_buffer_create(
    size_t element_size,
    size_t capacity,
    overflow_strategy_t strategy
) {
    // Guard: validate inputs
    if (element_size == 0 || capacity == 0) return NULL;

    // Allocate control structure (cache-aligned due to _Alignas members)
    circular_buffer_t* buf = nimcp_aligned_alloc(CACHE_LINE_SIZE, sizeof(circular_buffer_t));
    if (!buf) return NULL;

    // Zero-initialize the structure
    memset(buf, 0, sizeof(circular_buffer_t));

    // Allocate data storage (cache-aligned)
    size_t data_size = element_size * (capacity + 1);  // +1 for full/empty distinction

    // Round up to cache line boundary for nimcp_aligned_alloc
    size_t aligned_size = ((data_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

    buf->data = nimcp_aligned_alloc(CACHE_LINE_SIZE, aligned_size);
    if (!buf->data) {
        nimcp_free(buf);
        return NULL;
    }

    // Initialize configuration
    buf->element_size = element_size;
    buf->capacity = capacity + 1;  // Reserve one slot for full/empty distinction
    buf->strategy = strategy;

    // Initialize atomic indices
    atomic_store(&buf->write_pos, 0);
    atomic_store(&buf->read_pos, 0);

    // Initialize statistics
    memset(&buf->stats, 0, sizeof(circular_buffer_stats_t));

    return buf;
}

void circular_buffer_destroy(circular_buffer_t* buffer) {
    if (!buffer) return;

    // Free aligned data storage
    if (buffer->data) {
        nimcp_aligned_free(buffer->data);
    }

    // Free aligned buffer structure
    nimcp_aligned_free(buffer);
}

//=============================================================================
// CORE OPERATIONS
//=============================================================================

bool circular_buffer_push(circular_buffer_t* buffer, const void* element) {
    // Guard: validate inputs
    if (!buffer || !element) return false;

    // Load current positions
    size_t write = atomic_load(&buffer->write_pos);
    size_t read = atomic_load(&buffer->read_pos);

    // Calculate next write position
    size_t next_write = (write + 1) % buffer->capacity;

    // Check if buffer full
    if (next_write == read) {
        buffer->stats.overflows++;

        switch (buffer->strategy) {
            case OVERFLOW_OVERWRITE:
                // Advance read pointer to overwrite oldest
                atomic_store(&buffer->read_pos, (read + 1) % buffer->capacity);
                break;

            case OVERFLOW_BLOCK:
                // Wait for space (spin - use with care!)
                while (next_write == atomic_load(&buffer->read_pos)) {
                    // Could add yield here for better behavior
                }
                break;

            case OVERFLOW_ERROR:
                // Report error
                return false;
        }
    }

    // Copy element to buffer
    void* dest = (char*)buffer->data + (write * buffer->element_size);
    memcpy(dest, element, buffer->element_size);

    // Advance write pointer (release semantics for synchronization)
    atomic_store(&buffer->write_pos, next_write);

    // Update statistics
    buffer->stats.total_writes++;
    size_t current_size = circular_buffer_size(buffer);
    update_avg_utilization(buffer, current_size);
    update_peak_utilization(buffer, current_size);

    return true;
}

bool circular_buffer_pop(circular_buffer_t* buffer, void* element) {
    // Guard: validate inputs
    if (!buffer || !element) return false;

    // Load current positions
    size_t read = atomic_load(&buffer->read_pos);
    size_t write = atomic_load(&buffer->write_pos);

    // Check if buffer empty
    if (read == write) {
        buffer->stats.underflows++;
        return false;
    }

    // Copy element from buffer
    const void* src = (const char*)buffer->data + (read * buffer->element_size);
    memcpy(element, src, buffer->element_size);

    // Advance read pointer
    atomic_store(&buffer->read_pos, (read + 1) % buffer->capacity);

    // Update statistics
    buffer->stats.total_reads++;
    size_t current_size = circular_buffer_size(buffer);
    update_avg_utilization(buffer, current_size);

    return true;
}

bool circular_buffer_peek(
    const circular_buffer_t* buffer,
    size_t offset,
    void* element
) {
    // Guard: validate inputs
    if (!buffer || !element) return false;

    // Load current positions
    size_t read = atomic_load(&buffer->read_pos);
    size_t write = atomic_load(&buffer->write_pos);

    // Calculate current size
    size_t size = (write >= read) ?
        (write - read) : (buffer->capacity - read + write);

    // Check if offset valid
    if (offset >= size) return false;

    // Calculate peek position
    size_t peek_pos = (read + offset) % buffer->capacity;

    // Copy element
    const void* src = (const char*)buffer->data + (peek_pos * buffer->element_size);
    memcpy(element, src, buffer->element_size);

    return true;
}

//=============================================================================
// BATCH OPERATIONS
//=============================================================================

size_t circular_buffer_push_batch(
    circular_buffer_t* buffer,
    const void* elements,
    size_t count
) {
    // Guard: validate inputs
    if (!buffer || !elements || count == 0) return 0;

    size_t written = 0;
    const char* src = (const char*)elements;

    // Write elements one by one (could optimize with memcpy for contiguous space)
    for (size_t i = 0; i < count; i++) {
        if (!circular_buffer_push(buffer, src + (i * buffer->element_size))) {
            // Stop on overflow if strategy is ERROR
            if (buffer->strategy == OVERFLOW_ERROR) {
                break;
            }
        }
        written++;
    }

    return written;
}

size_t circular_buffer_pop_batch(
    circular_buffer_t* buffer,
    void* elements,
    size_t count
) {
    // Guard: validate inputs
    if (!buffer || !elements || count == 0) return 0;

    size_t read = 0;
    char* dest = (char*)elements;

    // Read elements one by one
    for (size_t i = 0; i < count; i++) {
        if (!circular_buffer_pop(buffer, dest + (i * buffer->element_size))) {
            break;  // Buffer empty
        }
        read++;
    }

    return read;
}

//=============================================================================
// QUERY OPERATIONS
//=============================================================================

size_t circular_buffer_size(const circular_buffer_t* buffer) {
    // Guard: validate input
    if (!buffer) return 0;

    size_t write = atomic_load(&buffer->write_pos);
    size_t read = atomic_load(&buffer->read_pos);

    // Calculate size handling wrap-around
    return (write >= read) ?
        (write - read) : (buffer->capacity - read + write);
}

size_t circular_buffer_capacity(const circular_buffer_t* buffer) {
    // Guard: validate input
    if (!buffer) return 0;

    return buffer->capacity - 1;  // Actual usable capacity (one slot reserved)
}

bool circular_buffer_is_empty(const circular_buffer_t* buffer) {
    // Guard: validate input
    if (!buffer) return true;

    return atomic_load(&buffer->read_pos) == atomic_load(&buffer->write_pos);
}

bool circular_buffer_is_full(const circular_buffer_t* buffer) {
    // Guard: validate input
    if (!buffer) return false;

    size_t write = atomic_load(&buffer->write_pos);
    size_t read = atomic_load(&buffer->read_pos);
    size_t next_write = (write + 1) % buffer->capacity;

    return next_write == read;
}

float circular_buffer_utilization(const circular_buffer_t* buffer) {
    // Guard: validate input
    if (!buffer || buffer->capacity <= 1) return 0.0f;

    size_t size = circular_buffer_size(buffer);
    size_t capacity = buffer->capacity - 1;

    return (100.0f * size) / capacity;
}

//=============================================================================
// MANAGEMENT
//=============================================================================

void circular_buffer_clear(circular_buffer_t* buffer) {
    // Guard: validate input
    if (!buffer) return;

    // Reset pointers
    atomic_store(&buffer->write_pos, 0);
    atomic_store(&buffer->read_pos, 0);
}

void circular_buffer_get_stats(
    const circular_buffer_t* buffer,
    circular_buffer_stats_t* stats
) {
    // Guard: validate inputs
    if (!buffer || !stats) return;

    // Copy statistics
    memcpy(stats, &buffer->stats, sizeof(circular_buffer_stats_t));
}

void circular_buffer_reset_stats(circular_buffer_t* buffer) {
    // Guard: validate input
    if (!buffer) return;

    // Reset all statistics
    memset(&buffer->stats, 0, sizeof(circular_buffer_stats_t));
}
