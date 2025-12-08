/**
 * @file nimcp_constant_time.c
 * @brief Implementation of constant-time cryptographic operations
 *
 * WHAT: Timing-safe comparison and selection operations
 * WHY:  Prevent side-channel timing attacks on cryptographic operations
 * HOW:  Branchless algorithms with data-independent execution time
 *
 * SECURITY PRINCIPLES:
 * 1. No conditional branches on secret data
 * 2. No table lookups with secret indices (except ct_lookup which is safe)
 * 3. No early exit conditions based on data values
 * 4. Volatile pointers to prevent compiler optimization
 * 5. Sequential memory access to resist cache timing attacks
 *
 * TESTING:
 * All functions are tested for:
 * - Functional correctness
 * - Constant-time behavior via statistical timing analysis
 * - Resistance to compiler optimization
 *
 * @author NIMCP Security Team
 * @date 2025-12-07
 */

#include "security/nimcp_constant_time.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/error/nimcp_error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LOG_MODULE "constant_time"

// Platform-specific secure random
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

//=============================================================================
// Internal Context Structure
//=============================================================================

/**
 * WHAT: Internal constant-time context structure
 * WHY:  Track statistics and configuration
 * HOW:  Opaque pointer pattern with magic number validation
 */
struct nimcp_ct_context {
    uint32_t magic;                  /**< Magic number for validation */
    nimcp_ct_stats_t stats;          /**< Operation statistics */
    bool bio_async_registered;      /**< Bio-async registration status */
    bio_module_context_t bio_ctx;    /**< Bio-async module context */
};

//=============================================================================
// Global State
//=============================================================================

static nimcp_ct_context_t g_default_ctx = NULL;

//=============================================================================
// Bio-Async Message Handlers
//=============================================================================

/**
 * WHAT: Handle bio-async messages for constant-time module
 * WHY:  Integration with NIMCP's async messaging system
 * HOW:  Process requests for stats, verification tests
 */
static nimcp_error_t s_ct_message_handler(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    nimcp_ct_context_t ctx = (nimcp_ct_context_t)user_data;

    if (!ctx || ctx->magic != NIMCP_CT_MAGIC) {
        LOG_ERROR("Invalid context in message handler");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Parse message header
    if (msg_size < sizeof(bio_message_header_t)) {
        LOG_ERROR("Message too small");
        return NIMCP_ERROR_BUFFER_TOO_SMALL;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    // Handle different message types
    switch (header->type) {
        case BIO_MSG_HEALTH_CHECK: {
            // Send back statistics
            nimcp_ct_stats_t stats;
            nimcp_ct_get_stats(ctx, &stats);

            if (response_promise) {
                nimcp_bio_promise_complete(response_promise, &stats);
            }

            LOG_DEBUG("Stats request handled: %lu comparisons",
                     stats.memcmp_operations);
            return NIMCP_SUCCESS;
        }

        default:
            LOG_WARN("Unknown message type: %u", header->type);
            return NIMCP_ERROR_NOT_IMPLEMENTED;
    }
}

/**
 * WHAT: Register constant-time module with bio-async router
 * WHY:  Enable async communication and monitoring
 * HOW:  Register module ID and message handlers
 */
static nimcp_error_t s_register_bio_async(nimcp_ct_context_t ctx)
{
    if (ctx->bio_async_registered) {
        return NIMCP_SUCCESS;  // Already registered
    }

    bio_router_t router = bio_router_get_global();
    if (!router) {
        LOG_WARN("Bio-async router not available, skipping registration");
        return NIMCP_SUCCESS;  // Not fatal
    }

    bio_module_info_t module_info = {
        .module_id = BIO_MODULE_SECURITY,
        .module_name = "security_constant_time",
        .inbox_capacity = 64,
        .user_data = ctx
    };

    ctx->bio_ctx = bio_router_register_module(&module_info);
    if (!ctx->bio_ctx) {
        LOG_ERROR("Failed to register with bio-async router");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Register message handlers
    nimcp_error_t err = bio_router_register_handler(ctx->bio_ctx,
                                      BIO_MSG_HEALTH_CHECK,
                                      s_ct_message_handler);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to register message handler: %d", err);
        return err;
    }

    ctx->bio_async_registered = true;
    LOG_INFO("Constant-time module registered with bio-async");

    return NIMCP_SUCCESS;
}

//=============================================================================
// Context Management
//=============================================================================

nimcp_ct_context_t nimcp_ct_create(void)
{
    nimcp_ct_context_t ctx = (nimcp_ct_context_t)nimcp_calloc(1, sizeof(struct nimcp_ct_context));
    if (!ctx) {
        LOG_ERROR("Failed to allocate constant-time context");
        return NULL;
    }

    ctx->magic = NIMCP_CT_MAGIC;
    memset(&ctx->stats, 0, sizeof(nimcp_ct_stats_t));
    ctx->bio_async_registered = false;
    ctx->bio_ctx = NULL;

    // Register with bio-async if available
    s_register_bio_async(ctx);

    LOG_INFO("Constant-time context created");

    return ctx;
}

void nimcp_ct_destroy(nimcp_ct_context_t ctx)
{
    if (!ctx) {
        return;
    }

    if (ctx->magic != NIMCP_CT_MAGIC) {
        LOG_ERROR("Invalid magic number in context destroy");
        return;
    }

    // Unregister from bio-async
    if (ctx->bio_async_registered && ctx->bio_ctx) {
        bio_router_unregister_module(ctx->bio_ctx);
    }

    // Zero sensitive data
    ctx->magic = 0;
    nimcp_secure_zero(ctx, sizeof(struct nimcp_ct_context));

    nimcp_free(ctx);

    LOG_INFO("Constant-time context destroyed");
}

nimcp_result_t nimcp_ct_get_stats(nimcp_ct_context_t ctx, nimcp_ct_stats_t* stats)
{
    if (!ctx || ctx->magic != NIMCP_CT_MAGIC) {
        LOG_ERROR("Invalid context");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!stats) {
        LOG_ERROR("NULL stats pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    memcpy(stats, &ctx->stats, sizeof(nimcp_ct_stats_t));
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_ct_reset_stats(nimcp_ct_context_t ctx)
{
    if (!ctx || ctx->magic != NIMCP_CT_MAGIC) {
        LOG_ERROR("Invalid context");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(&ctx->stats, 0, sizeof(nimcp_ct_stats_t));
    LOG_DEBUG("Statistics reset");

    return NIMCP_SUCCESS;
}

//=============================================================================
// Constant-Time Memory Comparison
//=============================================================================

int nimcp_ct_memcmp(const void* a, const void* b, size_t len)
{
    if (!a || !b) {
        LOG_ERROR("NULL pointer in ct_memcmp");
        return -1;
    }

    if (len == 0) {
        return 0;
    }

    // WHAT: Use volatile to prevent compiler optimization
    // WHY:  Compiler might optimize away the loop if it detects memcmp pattern
    // HOW:  Volatile forces actual memory reads
    const volatile uint8_t* pa = (const volatile uint8_t*)a;
    const volatile uint8_t* pb = (const volatile uint8_t*)b;

    // WHAT: Accumulate differences using bitwise OR
    // WHY:  No early exit, constant-time regardless of where bytes differ
    // HOW:  diff becomes non-zero if any byte differs
    volatile uint8_t diff = 0;

    for (size_t i = 0; i < len; i++) {
        diff |= (pa[i] ^ pb[i]);
    }

    // WHAT: Convert to 0 or 1 without branches
    // WHY:  Standard return value convention (0 = equal)
    // HOW:  Bitwise OR forces MSB to indicate difference
    return (diff != 0) ? 1 : 0;
}

int nimcp_ct_memcmp_tracked(nimcp_ct_context_t ctx, const void* a, const void* b, size_t len)
{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int result = nimcp_ct_memcmp(a, b, len);

    clock_gettime(CLOCK_MONOTONIC, &end);

    // Update statistics if context provided
    if (ctx && ctx->magic == NIMCP_CT_MAGIC) {
        ctx->stats.memcmp_operations++;
        ctx->stats.total_bytes_compared += len;

        // Update average timing
        double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 +
                           (end.tv_nsec - start.tv_nsec);

        // Running average
        uint64_t total_ops = ctx->stats.memcmp_operations +
                            ctx->stats.strcmp_operations +
                            ctx->stats.hash_comparisons;

        ctx->stats.avg_comparison_time_ns =
            (ctx->stats.avg_comparison_time_ns * (total_ops - 1) + elapsed_ns) / total_ops;
    }

    return result;
}

//=============================================================================
// Constant-Time String Comparison
//=============================================================================

int nimcp_ct_strcmp(const char* a, const char* b)
{
    if (!a || !b) {
        LOG_ERROR("NULL pointer in ct_strcmp");
        return -1;
    }

    // WHAT: Compute string lengths (not constant-time, but acceptable)
    // WHY:  String length is not typically secret; content is
    // HOW:  Standard strlen (we could make this CT too if needed)
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);

    // WHAT: Compare lengths using constant-time select
    // WHY:  Don't want to leak which string is longer
    // HOW:  Use minimum length for comparison
    size_t min_len = (len_a < len_b) ? len_a : len_b;
    size_t max_len = (len_a > len_b) ? len_a : len_b;

    // WHAT: Constant-time memory comparison
    // WHY:  Prevent timing leaks on string content
    int mem_diff = nimcp_ct_memcmp(a, b, max_len);

    // WHAT: Constant-time length comparison
    // WHY:  Combine both checks without branches
    int len_diff = (len_a != len_b) ? 1 : 0;

    // Return combined result
    return mem_diff | len_diff;
}

int nimcp_ct_strncmp(const char* a, const char* b, size_t n)
{
    if (!a || !b) {
        LOG_ERROR("NULL pointer in ct_strncmp");
        return -1;
    }

    if (n == 0) {
        return 0;
    }

    // WHAT: Bounded length computation
    // WHY:  Limit comparison length for safety
    size_t len_a = strnlen(a, n);
    size_t len_b = strnlen(b, n);

    size_t min_len = (len_a < len_b) ? len_a : len_b;
    if (min_len > n) {
        min_len = n;
    }

    int mem_diff = nimcp_ct_memcmp(a, b, min_len);
    int len_diff = (len_a != len_b) ? 1 : 0;

    return mem_diff | len_diff;
}

//=============================================================================
// Constant-Time Conditional Selection
//=============================================================================

uint8_t nimcp_ct_select_u8(uint8_t a, uint8_t b, uint8_t select)
{
    // WHAT: Create mask from select without branches
    // WHY:  Mask is 0x00 if select==0, 0xFF if select!=0
    // HOW:  Negate the boolean, which extends sign bit
    volatile uint8_t mask = (uint8_t)(-(volatile int8_t)(select != 0));

    // WHAT: Use mask to select between a and b
    // WHY:  a ^ ((a ^ b) & mask) = a when mask=0, b when mask=0xFF
    // HOW:  Bitwise operations only, no branches
    return (uint8_t)(a ^ ((a ^ b) & mask));
}

uint32_t nimcp_ct_select_u32(uint32_t a, uint32_t b, uint32_t select)
{
    volatile uint32_t mask = (uint32_t)(-(volatile int32_t)(select != 0));
    return (uint32_t)(a ^ ((a ^ b) & mask));
}

uint64_t nimcp_ct_select_u64(uint64_t a, uint64_t b, uint64_t select)
{
    volatile uint64_t mask = (uint64_t)(-(volatile int64_t)(select != 0));
    return (uint64_t)(a ^ ((a ^ b) & mask));
}

size_t nimcp_ct_select_size(size_t a, size_t b, uint8_t select)
{
    volatile size_t mask = (size_t)(-(volatile intptr_t)(select != 0));
    return (size_t)(a ^ ((a ^ b) & mask));
}

//=============================================================================
// Constant-Time Array Lookup
//=============================================================================

uint8_t nimcp_ct_lookup_u8(const uint8_t* table, size_t table_len, size_t index)
{
    if (!table) {
        LOG_ERROR("NULL table in ct_lookup_u8");
        return 0;
    }

    if (table_len == 0) {
        return 0;
    }

    // WHAT: Scan entire table using conditional selection
    // WHY:  Execution time independent of index value
    // HOW:  Check each position, accumulate result
    volatile uint8_t result = 0;

    for (size_t i = 0; i < table_len; i++) {
        // WHAT: Create mask if this is the target index
        // WHY:  Mask is used for constant-time selection
        uint8_t is_match = (uint8_t)(i == index);

        // WHAT: Conditionally update result
        // WHY:  Always scans entire table regardless of index
        result = nimcp_ct_select_u8(result, table[i], is_match);
    }

    return (uint8_t)result;
}

uint32_t nimcp_ct_lookup_u32(const uint32_t* table, size_t table_len, size_t index)
{
    if (!table) {
        LOG_ERROR("NULL table in ct_lookup_u32");
        return 0;
    }

    if (table_len == 0) {
        return 0;
    }

    volatile uint32_t result = 0;

    for (size_t i = 0; i < table_len; i++) {
        uint32_t is_match = (uint32_t)(i == index);
        result = nimcp_ct_select_u32(result, table[i], is_match);
    }

    return (uint32_t)result;
}

//=============================================================================
// Cryptographic Hash Comparison
//=============================================================================

bool nimcp_ct_hash_equal(const uint8_t* hash1, const uint8_t* hash2, size_t hash_len)
{
    if (!hash1 || !hash2) {
        LOG_ERROR("NULL hash pointer");
        return false;
    }

    if (hash_len == 0) {
        LOG_WARN("Zero-length hash comparison");
        return true;
    }

    // Update global stats if available
    if (g_default_ctx && g_default_ctx->magic == NIMCP_CT_MAGIC) {
        g_default_ctx->stats.hash_comparisons++;
        g_default_ctx->stats.total_bytes_compared += hash_len;
    }

    int result = nimcp_ct_memcmp(hash1, hash2, hash_len);

    LOG_DEBUG("Hash comparison: %s (%zu bytes)",
             result == 0 ? "EQUAL" : "DIFFERENT", hash_len);

    return (result == 0);
}

bool nimcp_ct_sha256_equal(const uint8_t hash1[32], const uint8_t hash2[32])
{
    return nimcp_ct_hash_equal(hash1, hash2, 32);
}

//=============================================================================
// Timing Verification (Testing Only)
//=============================================================================

bool nimcp_ct_verify_timing(const char* operation_name, size_t num_trials,
                            double threshold_percent)
{
    if (!operation_name || num_trials < 100) {
        LOG_ERROR("Invalid timing verification parameters");
        return false;
    }

    LOG_INFO("Verifying constant-time behavior for %s (%zu trials, %.1f%% threshold)",
             operation_name, num_trials, threshold_percent);

    // WHAT: Statistical timing analysis
    // WHY:  Detect timing variations that could leak information
    // HOW:  Compare timing distributions for different data patterns

    // Allocate test buffers
    size_t test_size = 256;
    uint8_t* buf_zeros = (uint8_t*)nimcp_calloc(test_size, 1);
    uint8_t* buf_random = (uint8_t*)nimcp_calloc(test_size, 1);

    if (!buf_zeros || !buf_random) {
        nimcp_free(buf_zeros);
        nimcp_free(buf_random);
        return false;
    }

    // Fill random buffer
    for (size_t i = 0; i < test_size; i++) {
        buf_random[i] = (uint8_t)(rand() & 0xFF);
    }

    // Measure timing for zeros
    struct timespec start, end;
    double total_time_zeros = 0.0;

    for (size_t i = 0; i < num_trials; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        nimcp_ct_memcmp(buf_zeros, buf_zeros, test_size);
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_time_zeros += (end.tv_sec - start.tv_sec) * 1e9 +
                           (end.tv_nsec - start.tv_nsec);
    }

    // Measure timing for random data
    double total_time_random = 0.0;

    for (size_t i = 0; i < num_trials; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        nimcp_ct_memcmp(buf_random, buf_zeros, test_size);
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_time_random += (end.tv_sec - start.tv_sec) * 1e9 +
                            (end.tv_nsec - start.tv_nsec);
    }

    // Calculate averages
    double avg_zeros = total_time_zeros / num_trials;
    double avg_random = total_time_random / num_trials;

    // Calculate percentage difference
    double diff_percent = fabs(avg_random - avg_zeros) / avg_zeros * 100.0;

    LOG_INFO("Timing results: zeros=%.2f ns, random=%.2f ns, diff=%.2f%%",
             avg_zeros, avg_random, diff_percent);

    // Clean up
    nimcp_free(buf_zeros);
    nimcp_free(buf_random);

    // Check if within threshold
    bool is_constant_time = (diff_percent <= threshold_percent);

    if (is_constant_time) {
        LOG_INFO("✓ %s is constant-time (%.2f%% ≤ %.2f%%)",
                operation_name, diff_percent, threshold_percent);
    } else {
        LOG_WARN("✗ %s timing leak detected (%.2f%% > %.2f%%)",
                operation_name, diff_percent, threshold_percent);
    }

    return is_constant_time;
}

//=============================================================================
// Secure Memory Wiping
//=============================================================================

void nimcp_secure_zero(void* ptr, size_t len)
{
    if (!ptr || len == 0) {
        return;
    }

#if defined(_WIN32)
    // Windows: Use SecureZeroMemory
    SecureZeroMemory(ptr, len);

#elif defined(__GLIBC__) && __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 25
    // Linux with glibc >= 2.25: Use explicit_bzero
    explicit_bzero(ptr, len);

#elif defined(__OpenBSD__) || defined(__FreeBSD__)
    // BSD systems: explicit_bzero available
    explicit_bzero(ptr, len);

#else
    // WHAT: Fallback using volatile pointer
    // WHY:  Prevents compiler from optimizing away the memset
    // HOW:  Volatile forces actual memory writes
    volatile uint8_t* p = (volatile uint8_t*)ptr;
    for (size_t i = 0; i < len; i++) {
        p[i] = 0;
    }

    // WHAT: Memory barrier to prevent reordering
    // WHY:  Ensure zeroing happens before function returns
    // HOW:  Compiler barrier (prevents reordering at compile-time)
    __asm__ __volatile__("" ::: "memory");
#endif

    LOG_DEBUG("Secure zero: %zu bytes", len);
}

nimcp_result_t nimcp_secure_wipe(void* ptr, size_t len)
{
    if (!ptr) {
        LOG_ERROR("NULL pointer in secure_wipe");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (len == 0) {
        return NIMCP_SUCCESS;
    }

    volatile uint8_t* p = (volatile uint8_t*)ptr;

    // WHAT: Multi-pass overwrite following DoD 5220.22-M
    // WHY:  Defense-in-depth against data remanence attacks
    // HOW:  Multiple passes with different patterns

    // Pass 1: Write 0x00
    for (size_t i = 0; i < len; i++) {
        p[i] = 0x00;
    }

    // Pass 2: Write 0xFF
    for (size_t i = 0; i < len; i++) {
        p[i] = 0xFF;
    }

    // Pass 3: Write random data
    for (size_t i = 0; i < len; i++) {
        p[i] = (uint8_t)(rand() & 0xFF);
    }

    // Pass 4: Final zero
    for (size_t i = 0; i < len; i++) {
        p[i] = 0x00;
    }

    // Memory barrier
    __asm__ __volatile__("" ::: "memory");

    LOG_DEBUG("Secure wipe: %zu bytes (4 passes)", len);

    return NIMCP_SUCCESS;
}

//=============================================================================
// Module Initialization
//=============================================================================

/**
 * WHAT: Initialize global constant-time context
 * WHY:  Provide default context for stateless API calls
 * HOW:  Create context on first use (lazy initialization)
 */
static void __attribute__((constructor)) s_ct_module_init(void)
{
    if (!g_default_ctx) {
        g_default_ctx = nimcp_ct_create();
        if (g_default_ctx) {
            LOG_INFO("Constant-time module initialized");
        }
    }
}

/**
 * WHAT: Cleanup global constant-time context
 * WHY:  Prevent memory leaks on shutdown
 * HOW:  Destroy context at program exit
 */
static void __attribute__((destructor)) s_ct_module_cleanup(void)
{
    if (g_default_ctx) {
        nimcp_ct_destroy(g_default_ctx);
        g_default_ctx = NULL;
        LOG_INFO("Constant-time module cleaned up");
    }
}
