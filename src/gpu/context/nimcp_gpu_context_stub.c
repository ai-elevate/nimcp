/**
 * @file nimcp_gpu_context_stub.c
 * @brief Stub implementation of GPU context for non-CUDA builds
 *
 * WHAT: Provides no-op/fallback implementations when CUDA is not available
 * WHY:  Allows building and linking without CUDA dependencies
 * HOW:  Returns NULL/error for functions that require GPU
 *
 * @author NIMCP Development Team
 * @date 2025-12-31
 */

#include "gpu/context/nimcp_gpu_context.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for gpu_context_stub module */
static nimcp_health_agent_t* g_gpu_context_stub_health_agent = NULL;

/**
 * @brief Set health agent for gpu_context_stub heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void gpu_context_stub_set_health_agent(nimcp_health_agent_t* agent) {
    g_gpu_context_stub_health_agent = agent;
}

/** @brief Send heartbeat from gpu_context_stub module */
static inline void gpu_context_stub_heartbeat(const char* operation, float progress) {
    if (g_gpu_context_stub_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_gpu_context_stub_health_agent, operation, progress);
    }
}


//=============================================================================
// Context Lifecycle API - Stub Implementations
//=============================================================================

nimcp_gpu_context_t* nimcp_gpu_context_create(int device_id) {
    (void)device_id;
    // No GPU available in non-CUDA build
    return NULL;
}

nimcp_gpu_context_t* nimcp_gpu_context_create_auto(void) {
    // No GPU available in non-CUDA build
    return NULL;
}

void nimcp_gpu_context_destroy(nimcp_gpu_context_t* ctx) {
    (void)ctx;
    // Nothing to do
}

bool nimcp_gpu_context_is_valid(const nimcp_gpu_context_t* ctx) {
    return ctx != NULL && ctx->initialized;
}

//=============================================================================
// Device Management API - Stub Implementations
//=============================================================================

int nimcp_gpu_context_set_device(nimcp_gpu_context_t* ctx) {
    (void)ctx;
    return -1; // No GPU
}

int nimcp_gpu_context_synchronize(nimcp_gpu_context_t* ctx) {
    (void)ctx;
    return 0; // Nothing to synchronize
}

const char* nimcp_gpu_context_get_error(const nimcp_gpu_context_t* ctx) {
    if (ctx) {
        return ctx->last_error_msg;
    }
    return "No GPU context";
}

//=============================================================================
// Memory Management API - Stub Implementations
//=============================================================================

void* nimcp_gpu_malloc(nimcp_gpu_context_t* ctx, size_t size_bytes) {
    (void)ctx;
    (void)size_bytes;
    return NULL; // No GPU memory available
}

void nimcp_gpu_free(nimcp_gpu_context_t* ctx, void* dev_ptr) {
    (void)ctx;
    (void)dev_ptr;
    // Nothing to do
}

int nimcp_gpu_memcpy(nimcp_gpu_context_t* ctx,
                      void* dst, const void* src,
                      size_t size_bytes,
                      nimcp_gpu_memcpy_kind_t kind) {
    (void)ctx;
    (void)kind;
    // Fallback: use CPU memcpy for host-to-host
    if (dst && src) {
        memcpy(dst, src, size_bytes);
        return 0;
    }
    return -1;
}

int nimcp_gpu_memcpy_async(nimcp_gpu_context_t* ctx,
                            void* dst, const void* src,
                            size_t size_bytes,
                            nimcp_gpu_memcpy_kind_t kind,
                            bool use_transfer_stream) {
    (void)use_transfer_stream;
    // Just use synchronous copy
    return nimcp_gpu_memcpy(ctx, dst, src, size_bytes, kind);
}

int nimcp_gpu_memset(nimcp_gpu_context_t* ctx,
                      void* dev_ptr, int value,
                      size_t size_bytes) {
    (void)ctx;
    if (dev_ptr) {
        memset(dev_ptr, value, size_bytes);
        return 0;
    }
    return -1;
}

void nimcp_gpu_memory_stats(const nimcp_gpu_context_t* ctx,
                             size_t* allocated_out,
                             size_t* peak_out,
                             size_t* free_out) {
    if (allocated_out) *allocated_out = ctx ? ctx->allocated_memory : 0;
    if (peak_out) *peak_out = ctx ? ctx->peak_memory : 0;
    if (free_out) *free_out = 0; // No GPU memory
}

//=============================================================================
// Stream Management API - Stub Implementations
//=============================================================================

nimcp_cuda_stream_t nimcp_gpu_get_compute_stream(nimcp_gpu_context_t* ctx) {
    return ctx ? ctx->compute_stream : NULL;
}

nimcp_cuda_stream_t nimcp_gpu_get_transfer_stream(nimcp_gpu_context_t* ctx) {
    return ctx ? ctx->transfer_stream : NULL;
}

int nimcp_gpu_stream_synchronize(nimcp_gpu_context_t* ctx) {
    (void)ctx;
    return 0; // Nothing to synchronize
}

//=============================================================================
// Library Handles API - Stub Implementations
//=============================================================================

nimcp_cublas_handle_t nimcp_gpu_get_cublas(nimcp_gpu_context_t* ctx) {
    return ctx ? ctx->cublas_handle : NULL;
}

nimcp_cufft_handle_t nimcp_gpu_get_cufft_1d(nimcp_gpu_context_t* ctx, int n) {
    (void)n;
    return ctx ? ctx->cufft_plan_1d : 0;
}

//=============================================================================
// Kernel Configuration API - Stub Implementations
//=============================================================================

void nimcp_gpu_calc_launch_config(const nimcp_gpu_context_t* ctx,
                                   uint32_t num_elements,
                                   uint32_t* block_size_out,
                                   uint32_t* grid_size_out) {
    (void)num_elements;
    if (block_size_out) *block_size_out = ctx ? ctx->default_block_size : 256;
    if (grid_size_out) *grid_size_out = 1;
}

uint32_t nimcp_gpu_get_optimal_block_size(const nimcp_gpu_context_t* ctx,
                                           size_t shared_mem_per_block) {
    (void)shared_mem_per_block;
    return ctx ? ctx->default_block_size : 256;
}

//=============================================================================
// Utility Functions - Stub Implementations
//=============================================================================

void nimcp_gpu_context_print_info(const nimcp_gpu_context_t* ctx) {
    if (!ctx) {
        printf("GPU Context: NULL (no GPU available)\n");
    } else {
        printf("GPU Context: Device %d\n", ctx->device_id);
    }
}

int nimcp_gpu_context_get_info_string(const nimcp_gpu_context_t* ctx,
                                       char* buffer, size_t size) {
    if (!buffer || size == 0) return 0;

    if (!ctx) {
        return snprintf(buffer, size, "No GPU context (CUDA not enabled)");
    }
    return snprintf(buffer, size, "GPU Device %d: %s",
                    ctx->device_id,
                    ctx->device_info.name[0] ? ctx->device_info.name : "Unknown");
}
