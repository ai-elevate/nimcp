//=============================================================================
// nimcp_tensor.c - Comprehensive Tensor Library Implementation
//=============================================================================
/**
 * @file nimcp_tensor.c
 * @brief N-dimensional Tensor Library with Tensor Calculus
 *
 * WHAT: Complete tensor operations implementation
 * WHY:  Foundation for neural networks, emotion tensors, swarm coordination
 * HOW:  Efficient memory layout, SIMD where possible, clean API
 *
 * @author NIMCP Development Team
 * @date 2025-12-10
 * @version 1.0.0
 */

#include "utils/tensor/nimcp_tensor.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

//=============================================================================
// Module Configuration
//=============================================================================

#define LOG_MODULE "Tensor"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Internal tensor structure
 */
struct nimcp_tensor_s {
    uint32_t magic;                /**< Validation magic number */
    nimcp_tensor_shape_t shape;    /**< Shape information */
    nimcp_dtype_t dtype;           /**< Data type */
    void* data;                    /**< Pointer to data */
    bool owns_data;                /**< Whether tensor owns the data */
    bool requires_grad;            /**< Track gradients */
    nimcp_tensor_t* grad;          /**< Accumulated gradient */
    uint32_t refcount;             /**< Reference count */
    pthread_mutex_t lock;          /**< Thread safety lock */
};

/**
 * @brief Autodiff operation node
 */
typedef struct nimcp_autodiff_node_s {
    nimcp_tensor_t* output;
    nimcp_tensor_t** inputs;
    uint32_t num_inputs;
    void (*backward_fn)(struct nimcp_autodiff_node_s*, nimcp_tensor_t*);
    void* ctx;
    struct nimcp_autodiff_node_s* next;
} nimcp_autodiff_node_t;

/**
 * @brief Autodiff context
 */
struct nimcp_autodiff_ctx_s {
    bool recording;
    nimcp_autodiff_node_t* tape_head;
    nimcp_autodiff_node_t* tape_tail;
    pthread_mutex_t lock;
};

//=============================================================================
// Global State
//=============================================================================

static nimcp_tensor_stats_t g_stats = {0};
static nimcp_mutex_t g_stats_lock = NIMCP_MUTEX_INITIALIZER;
static bool g_initialized = false;

//=============================================================================
// Helper Functions - Validation
//=============================================================================

/**
 * @brief Validate tensor pointer
 */
static inline bool tensor_is_valid(const nimcp_tensor_t* t)
{
    return t != NULL && t->magic == NIMCP_TENSOR_MAGIC;
}

/**
 * @brief Update global statistics using atomic operations
 *
 * WHY ATOMICS: Reduces lock contention for frequently-called statistics updates.
 * Lock-free atomic increments are much faster than mutex lock/unlock pairs.
 */
static void stats_update_op(uint64_t* counter)
{
    __atomic_fetch_add(counter, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_stats.operations_total, 1, __ATOMIC_RELAXED);
}

//=============================================================================
// Helper Functions - Shape
//=============================================================================

/**
 * @brief Compute total number of elements
 */
static size_t compute_numel(const uint32_t* dims, uint32_t rank)
{
    if (rank == 0) return 1;  /* Scalar */

    size_t numel = 1;
    for (uint32_t i = 0; i < rank; i++) {
        numel *= dims[i];
    }
    return numel;
}

/**
 * @brief Compute row-major strides
 */
static void compute_strides(
    const uint32_t* dims,
    uint32_t rank,
    size_t element_size,
    int64_t* strides
)
{
    if (rank == 0) return;

    strides[rank - 1] = element_size;
    for (int i = (int)rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}

/**
 * @brief Check if two shapes are equal
 */
static bool shapes_equal(const nimcp_tensor_shape_t* a, const nimcp_tensor_shape_t* b)
{
    if (a->rank != b->rank) return false;
    for (uint32_t i = 0; i < a->rank; i++) {
        if (a->dims[i] != b->dims[i]) return false;
    }
    return true;
}

/**
 * @brief Check if shapes can broadcast
 */
static bool can_broadcast(
    const nimcp_tensor_shape_t* a,
    const nimcp_tensor_shape_t* b,
    nimcp_tensor_shape_t* result
)
{
    uint32_t max_rank = (a->rank > b->rank) ? a->rank : b->rank;
    result->rank = max_rank;

    for (int i = 0; i < (int)max_rank; i++) {
        int ai = (int)a->rank - 1 - i;
        int bi = (int)b->rank - 1 - i;
        int ri = (int)max_rank - 1 - i;

        uint32_t da = (ai >= 0) ? a->dims[ai] : 1;
        uint32_t db = (bi >= 0) ? b->dims[bi] : 1;

        if (da == db) {
            result->dims[ri] = da;
        } else if (da == 1) {
            result->dims[ri] = db;
        } else if (db == 1) {
            result->dims[ri] = da;
        } else {
            return false;  /* Cannot broadcast */
        }
    }

    result->numel = compute_numel(result->dims, result->rank);
    return true;
}

/**
 * @brief Convert multi-index to flat index
 */
static size_t indices_to_flat(
    const uint32_t* indices,
    const nimcp_tensor_shape_t* shape
)
{
    size_t flat = 0;
    for (uint32_t i = 0; i < shape->rank; i++) {
        flat += indices[i] * (shape->strides[i] / nimcp_dtype_size(NIMCP_DTYPE_F32));
    }
    return flat;
}

/**
 * @brief Convert flat index to multi-index
 */
static void flat_to_indices(
    size_t flat,
    const nimcp_tensor_shape_t* shape,
    uint32_t* indices
)
{
    for (uint32_t i = 0; i < shape->rank; i++) {
        size_t stride = shape->strides[i] / nimcp_dtype_size(NIMCP_DTYPE_F32);
        indices[i] = flat / stride;
        flat %= stride;
    }
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

int nimcp_tensor_init(void)
{
    if (g_initialized) return NIMCP_TENSOR_OK;

    memset(&g_stats, 0, sizeof(g_stats));
    g_initialized = true;

    LOG_INFO(LOG_MODULE, "Tensor subsystem initialized");
    return NIMCP_TENSOR_OK;
}

void nimcp_tensor_shutdown(void)
{
    if (!g_initialized) return;

    LOG_INFO(LOG_MODULE, "Tensor subsystem shutdown - stats: created=%lu, destroyed=%lu, mem=%zu bytes",
             g_stats.tensors_created, g_stats.tensors_destroyed, g_stats.memory_current);

    g_initialized = false;
}

//=============================================================================
// Tensor Creation
//=============================================================================

nimcp_tensor_t* nimcp_tensor_create(
    const uint32_t* dims,
    uint32_t rank,
    nimcp_dtype_t dtype
)
{
    /* Validate inputs */
    if (rank > NIMCP_TENSOR_MAX_RANK) {
        LOG_ERROR(LOG_MODULE, "Rank %u exceeds max %d", rank, NIMCP_TENSOR_MAX_RANK);
        return NULL;
    }

    /* Allocate tensor structure */
    nimcp_tensor_t* t = nimcp_calloc(1, sizeof(nimcp_tensor_t));
    if (!t) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate tensor structure");
        return NULL;
    }

    t->magic = NIMCP_TENSOR_MAGIC;
    t->dtype = dtype;
    t->shape.rank = rank;
    t->owns_data = true;
    t->requires_grad = false;
    t->grad = NULL;
    t->refcount = 1;

    /* Initialize mutex with error checking */
    if (nimcp_mutex_init(&t->lock, NULL) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize tensor mutex");
        nimcp_free(t);
        return NULL;
    }

    /* Set dimensions */
    if (rank > 0 && dims) {
        memcpy(t->shape.dims, dims, rank * sizeof(uint32_t));
    }

    /* Compute shape info */
    size_t elem_size = nimcp_dtype_size(dtype);
    t->shape.numel = compute_numel(t->shape.dims, rank);
    t->shape.nbytes = t->shape.numel * elem_size;
    compute_strides(t->shape.dims, rank, elem_size, t->shape.strides);

    /* Allocate data */
    if (t->shape.numel > 0) {
        t->data = nimcp_aligned_alloc(NIMCP_TENSOR_ALIGN, t->shape.nbytes);
        if (!t->data) {
            LOG_ERROR(LOG_MODULE, "Failed to allocate %zu bytes for tensor data", t->shape.nbytes);
            /* Clean up: mutex was initialized earlier */
            nimcp_mutex_destroy(&t->lock);
            /* Clean up: struct was allocated at line 266 */
            nimcp_free(t);
            return NULL;
        }
        /* Note: Data is uninitialized. Use nimcp_tensor_zeros() if zero initialization is needed. */
    }

    /* Update stats using atomic operations to reduce lock contention */
    __atomic_fetch_add(&g_stats.tensors_created, 1, __ATOMIC_RELAXED);
    size_t new_current = __atomic_add_fetch(&g_stats.memory_current,
                                             t->shape.nbytes + sizeof(nimcp_tensor_t),
                                             __ATOMIC_RELAXED);
    /* Update peak if needed - use compare-exchange loop for correctness */
    size_t old_peak = __atomic_load_n(&g_stats.memory_peak, __ATOMIC_RELAXED);
    while (new_current > old_peak) {
        if (__atomic_compare_exchange_n(&g_stats.memory_peak, &old_peak, new_current,
                                        false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            break;
        }
        /* old_peak was updated by compare_exchange, loop to retry */
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_zeros(
    const uint32_t* dims,
    uint32_t rank,
    nimcp_dtype_t dtype
)
{
    nimcp_tensor_t* t = nimcp_tensor_create(dims, rank, dtype);
    if (!t) return NULL;

    memset(t->data, 0, t->shape.nbytes);
    return t;
}

nimcp_tensor_t* nimcp_tensor_ones(
    const uint32_t* dims,
    uint32_t rank,
    nimcp_dtype_t dtype
)
{
    nimcp_tensor_t* t = nimcp_tensor_create(dims, rank, dtype);
    if (!t) return NULL;

    /* Fill with ones based on dtype */
    if (dtype == NIMCP_DTYPE_F32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->shape.numel; i++) {
            data[i] = 1.0F;
        }
    } else if (dtype == NIMCP_DTYPE_F64) {
        double* data = (double*)t->data;
        for (size_t i = 0; i < t->shape.numel; i++) {
            data[i] = 1.0;
        }
    } else if (dtype == NIMCP_DTYPE_I32) {
        int32_t* data = (int32_t*)t->data;
        for (size_t i = 0; i < t->shape.numel; i++) {
            data[i] = 1;
        }
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_full(
    const uint32_t* dims,
    uint32_t rank,
    nimcp_dtype_t dtype,
    double fill_value
)
{
    nimcp_tensor_t* t = nimcp_tensor_create(dims, rank, dtype);
    if (!t) return NULL;

    if (dtype == NIMCP_DTYPE_F32) {
        float* data = (float*)t->data;
        float val = (float)fill_value;
        for (size_t i = 0; i < t->shape.numel; i++) {
            data[i] = val;
        }
    } else if (dtype == NIMCP_DTYPE_F64) {
        double* data = (double*)t->data;
        for (size_t i = 0; i < t->shape.numel; i++) {
            data[i] = fill_value;
        }
    } else if (dtype == NIMCP_DTYPE_I32) {
        int32_t* data = (int32_t*)t->data;
        int32_t val = (int32_t)fill_value;
        for (size_t i = 0; i < t->shape.numel; i++) {
            data[i] = val;
        }
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_from_data(
    const void* data,
    const uint32_t* dims,
    uint32_t rank,
    nimcp_dtype_t dtype,
    bool copy
)
{
    if (!data) return NULL;

    nimcp_tensor_t* t = nimcp_tensor_create(dims, rank, dtype);
    if (!t) return NULL;

    if (copy) {
        memcpy(t->data, data, t->shape.nbytes);
    } else {
        /* Reference mode - free our allocation and use provided data */
        nimcp_free(t->data);
        t->data = (void*)data;
        t->owns_data = false;
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_eye(uint32_t n, nimcp_dtype_t dtype)
{
    uint32_t dims[2] = {n, n};
    nimcp_tensor_t* t = nimcp_tensor_zeros(dims, 2, dtype);
    if (!t) return NULL;

    /* Set diagonal to 1 */
    if (dtype == NIMCP_DTYPE_F32) {
        float* data = (float*)t->data;
        for (uint32_t i = 0; i < n; i++) {
            data[i * n + i] = 1.0F;
        }
    } else if (dtype == NIMCP_DTYPE_F64) {
        double* data = (double*)t->data;
        for (uint32_t i = 0; i < n; i++) {
            data[i * n + i] = 1.0;
        }
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_randn(
    const uint32_t* dims,
    uint32_t rank,
    nimcp_dtype_t dtype,
    double mean,
    double std
)
{
    nimcp_tensor_t* t = nimcp_tensor_create(dims, rank, dtype);
    if (!t) return NULL;

    /* Box-Muller transform for normal distribution */
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->shape.numel; i += 2) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

        data[i] = (float)(z0 * std + mean);
        if (i + 1 < t->shape.numel) {
            data[i + 1] = (float)(z1 * std + mean);
        }
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_rand(
    const uint32_t* dims,
    uint32_t rank,
    nimcp_dtype_t dtype,
    double low,
    double high
)
{
    nimcp_tensor_t* t = nimcp_tensor_create(dims, rank, dtype);
    if (!t) return NULL;

    float* data = (float*)t->data;
    double range = high - low;
    for (size_t i = 0; i < t->shape.numel; i++) {
        double u = (double)rand() / (double)RAND_MAX;
        data[i] = (float)(u * range + low);
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_arange(
    double start,
    double stop,
    double step,
    nimcp_dtype_t dtype
)
{
    if (step == 0) return NULL;

    uint32_t n = (uint32_t)ceil((stop - start) / step);
    if (n == 0) return NULL;

    nimcp_tensor_t* t = nimcp_tensor_create(&n, 1, dtype);
    if (!t) return NULL;

    float* data = (float*)t->data;
    for (uint32_t i = 0; i < n; i++) {
        data[i] = (float)(start + i * step);
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_linspace(
    double start,
    double stop,
    uint32_t num,
    nimcp_dtype_t dtype
)
{
    if (num == 0) return NULL;

    nimcp_tensor_t* t = nimcp_tensor_create(&num, 1, dtype);
    if (!t) return NULL;

    float* data = (float*)t->data;
    double step = (num > 1) ? (stop - start) / (num - 1) : 0;
    for (uint32_t i = 0; i < num; i++) {
        data[i] = (float)(start + i * step);
    }

    return t;
}

nimcp_tensor_t* nimcp_tensor_clone(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;

    nimcp_tensor_t* clone = nimcp_tensor_create(t->shape.dims, t->shape.rank, t->dtype);
    if (!clone) return NULL;

    memcpy(clone->data, t->data, t->shape.nbytes);
    clone->requires_grad = t->requires_grad;

    return clone;
}

/**
 * @brief Destroy tensor and free resources
 *
 * WHAT: Free tensor memory and decrement reference count
 * WHY:  Clean resource management with refcounting
 * HOW:  Thread-safe cleanup, idempotent (safe to call on partially initialized tensors)
 *
 * MEMORY SAFETY:
 * - Idempotent: Safe to call multiple times (magic check prevents double-free)
 * - Partial cleanup: Safe even if tensor creation failed partway
 * - Refcounting: Only frees when refcount reaches 0
 * - Gradient cleanup: Recursive destroy of gradient tensor
 *
 * @param t Tensor to destroy (NULL is safe, no-op)
 */
void nimcp_tensor_destroy(nimcp_tensor_t* t)
{
    /* Guard: NULL is no-op */
    if (!t) return;

    /* Pre-check magic BEFORE locking to avoid locking a destroyed object.
     * This is a best-effort check - the definitive check is inside the lock.
     * If magic is invalid here, the tensor was already destroyed or corrupted,
     * and we should not attempt to lock its mutex (lock-after-free risk). */
    if (t->magic != NIMCP_TENSOR_MAGIC) {
        /* Already destroyed or corrupted - do not touch the mutex */
        return;
    }

    pthread_mutex_lock(&t->lock);

    /* Guard: Invalid magic means already destroyed or corrupted */
    /* MUST be re-checked INSIDE lock to handle race between pre-check and lock */
    if (!tensor_is_valid(t)) {
        /* Already destroyed or never initialized properly */
        pthread_mutex_unlock(&t->lock);
        return;
    }

    /* Refcount management */
    t->refcount--;
    if (t->refcount > 0) {
        pthread_mutex_unlock(&t->lock);
        return;
    }

    /* Capture gradient pointer before releasing lock to avoid deadlock */
    nimcp_tensor_t* grad_to_destroy = t->grad;
    t->grad = NULL;

    /* Capture data pointer for cleanup after lock release */
    void* data_to_free = (t->owns_data && t->data) ? t->data : NULL;
    t->data = NULL;

    /* Invalidate magic BEFORE unlocking to prevent re-entry */
    t->magic = 0;

    /* Release lock before any recursive operations */
    nimcp_mutex_unlock(&t->lock);
    nimcp_mutex_destroy(&t->lock);

    /* Update stats using atomic operations (lock-free) */
    __atomic_fetch_add(&g_stats.tensors_destroyed, 1, __ATOMIC_RELAXED);
    __atomic_fetch_sub(&g_stats.memory_current, t->shape.nbytes + sizeof(nimcp_tensor_t), __ATOMIC_RELAXED);

    /* Free gradient if exists (recursive destroy - now safe, lock released) */
    if (grad_to_destroy) {
        nimcp_tensor_destroy(grad_to_destroy);
    }

    /* Free data if owned */
    if (data_to_free) {
        nimcp_free(data_to_free);
    }

    nimcp_free(t);
}

//=============================================================================
// Tensor Properties
//=============================================================================

const nimcp_tensor_shape_t* nimcp_tensor_shape(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;
    return &t->shape;
}

uint32_t nimcp_tensor_rank(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return 0;
    return t->shape.rank;
}

size_t nimcp_tensor_numel(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return 0;
    return t->shape.numel;
}

nimcp_dtype_t nimcp_tensor_dtype(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NIMCP_DTYPE_F32;
    return t->dtype;
}

void* nimcp_tensor_data(nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;
    return t->data;
}

const void* nimcp_tensor_data_const(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;
    return t->data;
}

bool nimcp_tensor_is_contiguous(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return false;

    /* Check if strides match row-major layout */
    size_t elem_size = nimcp_dtype_size(t->dtype);
    int64_t expected_stride = elem_size;

    for (int i = (int)t->shape.rank - 1; i >= 0; i--) {
        if (t->shape.strides[i] != expected_stride) return false;
        expected_stride *= t->shape.dims[i];
    }

    return true;
}

nimcp_tensor_t* nimcp_tensor_contiguous(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;

    if (nimcp_tensor_is_contiguous(t)) {
        /* Already contiguous, return clone */
        return nimcp_tensor_clone(t);
    }

    /* Need to copy with proper layout */
    nimcp_tensor_t* result = nimcp_tensor_create(t->shape.dims, t->shape.rank, t->dtype);
    if (!result) return NULL;

    /* Copy element by element (handles non-contiguous case) */
    uint32_t indices[NIMCP_TENSOR_MAX_RANK] = {0};
    for (size_t i = 0; i < t->shape.numel; i++) {
        double val = nimcp_tensor_get(t, indices);
        nimcp_tensor_set(result, indices, val);

        /* Increment indices */
        for (int j = (int)t->shape.rank - 1; j >= 0; j--) {
            indices[j]++;
            if (indices[j] < t->shape.dims[j]) break;
            indices[j] = 0;
        }
    }

    return result;
}

bool nimcp_tensor_requires_grad(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return false;
    return t->requires_grad;
}

void nimcp_tensor_set_requires_grad(nimcp_tensor_t* t, bool requires_grad)
{
    if (!tensor_is_valid(t)) return;
    t->requires_grad = requires_grad;
}

//=============================================================================
// Element Access
//=============================================================================

double nimcp_tensor_get(const nimcp_tensor_t* t, const uint32_t* indices)
{
    if (!tensor_is_valid(t) || !indices) return 0.0;

    /* Compute flat offset using strides */
    size_t offset = 0;
    size_t elem_size = nimcp_dtype_size(t->dtype);
    for (uint32_t i = 0; i < t->shape.rank; i++) {
        if (indices[i] >= t->shape.dims[i]) return 0.0;
        offset += indices[i] * (t->shape.strides[i] / elem_size);
    }

    if (t->dtype == NIMCP_DTYPE_F32) {
        return ((float*)t->data)[offset];
    } else if (t->dtype == NIMCP_DTYPE_F64) {
        return ((double*)t->data)[offset];
    } else if (t->dtype == NIMCP_DTYPE_I32) {
        return ((int32_t*)t->data)[offset];
    }

    return 0.0;
}

int nimcp_tensor_set(nimcp_tensor_t* t, const uint32_t* indices, double value)
{
    if (!tensor_is_valid(t) || !indices) return NIMCP_TENSOR_ERR_NULL;

    /* Compute flat offset using strides */
    size_t offset = 0;
    size_t elem_size = nimcp_dtype_size(t->dtype);
    for (uint32_t i = 0; i < t->shape.rank; i++) {
        if (indices[i] >= t->shape.dims[i]) return NIMCP_TENSOR_ERR_INDEX;
        offset += indices[i] * (t->shape.strides[i] / elem_size);
    }

    if (t->dtype == NIMCP_DTYPE_F32) {
        ((float*)t->data)[offset] = (float)value;
    } else if (t->dtype == NIMCP_DTYPE_F64) {
        ((double*)t->data)[offset] = value;
    } else if (t->dtype == NIMCP_DTYPE_I32) {
        ((int32_t*)t->data)[offset] = (int32_t)value;
    }

    return NIMCP_TENSOR_OK;
}

double nimcp_tensor_get_flat(const nimcp_tensor_t* t, size_t index)
{
    if (!tensor_is_valid(t) || index >= t->shape.numel) return 0.0;

    if (t->dtype == NIMCP_DTYPE_F32) {
        return ((float*)t->data)[index];
    } else if (t->dtype == NIMCP_DTYPE_F64) {
        return ((double*)t->data)[index];
    } else if (t->dtype == NIMCP_DTYPE_I32) {
        return ((int32_t*)t->data)[index];
    }

    return 0.0;
}

int nimcp_tensor_set_flat(nimcp_tensor_t* t, size_t index, double value)
{
    if (!tensor_is_valid(t)) return NIMCP_TENSOR_ERR_NULL;
    if (index >= t->shape.numel) return NIMCP_TENSOR_ERR_INDEX;

    if (t->dtype == NIMCP_DTYPE_F32) {
        ((float*)t->data)[index] = (float)value;
    } else if (t->dtype == NIMCP_DTYPE_F64) {
        ((double*)t->data)[index] = value;
    } else if (t->dtype == NIMCP_DTYPE_I32) {
        ((int32_t*)t->data)[index] = (int32_t)value;
    }

    return NIMCP_TENSOR_OK;
}

//=============================================================================
// Shape Operations
//=============================================================================

nimcp_tensor_t* nimcp_tensor_reshape(
    const nimcp_tensor_t* t,
    const uint32_t* new_dims,
    uint32_t new_rank
)
{
    if (!tensor_is_valid(t) || !new_dims) return NULL;

    /* Verify element count matches */
    size_t new_numel = compute_numel(new_dims, new_rank);
    if (new_numel != t->shape.numel) {
        LOG_ERROR(LOG_MODULE, "Reshape: numel mismatch %zu vs %zu", new_numel, t->shape.numel);
        return NULL;
    }

    /* Create new tensor sharing data if contiguous */
    nimcp_tensor_t* result;
    if (nimcp_tensor_is_contiguous(t)) {
        result = nimcp_tensor_from_data(t->data, new_dims, new_rank, t->dtype, false);
    } else {
        result = nimcp_tensor_contiguous(t);
        if (!result) return NULL;
        /* Now reshape the contiguous copy */
        result->shape.rank = new_rank;
        memcpy(result->shape.dims, new_dims, new_rank * sizeof(uint32_t));
        compute_strides(result->shape.dims, new_rank, nimcp_dtype_size(t->dtype), result->shape.strides);
    }

    return result;
}

nimcp_tensor_t* nimcp_tensor_transpose(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;
    if (t->shape.rank < 2) return nimcp_tensor_clone(t);

    /* Swap last two dimensions */
    uint32_t perm[NIMCP_TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < t->shape.rank - 2; i++) {
        perm[i] = i;
    }
    perm[t->shape.rank - 2] = t->shape.rank - 1;
    perm[t->shape.rank - 1] = t->shape.rank - 2;

    return nimcp_tensor_permute(t, perm, t->shape.rank);
}

nimcp_tensor_t* nimcp_tensor_permute(
    const nimcp_tensor_t* t,
    const uint32_t* perm,
    uint32_t rank
)
{
    if (!tensor_is_valid(t) || !perm) return NULL;
    if (rank != t->shape.rank) return NULL;

    /* Create new shape with permuted dimensions */
    uint32_t new_dims[NIMCP_TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < rank; i++) {
        if (perm[i] >= rank) return NULL;
        new_dims[i] = t->shape.dims[perm[i]];
    }

    nimcp_tensor_t* result = nimcp_tensor_create(new_dims, rank, t->dtype);
    if (!result) return NULL;

    /* Copy with permuted indices */
    uint32_t src_indices[NIMCP_TENSOR_MAX_RANK] = {0};
    uint32_t dst_indices[NIMCP_TENSOR_MAX_RANK] = {0};

    for (size_t i = 0; i < t->shape.numel; i++) {
        /* Convert dst_indices to src_indices using inverse permutation */
        for (uint32_t j = 0; j < rank; j++) {
            src_indices[perm[j]] = dst_indices[j];
        }

        double val = nimcp_tensor_get(t, src_indices);
        nimcp_tensor_set(result, dst_indices, val);

        /* Increment dst_indices */
        for (int j = (int)rank - 1; j >= 0; j--) {
            dst_indices[j]++;
            if (dst_indices[j] < new_dims[j]) break;
            dst_indices[j] = 0;
        }
    }

    return result;
}

nimcp_tensor_t* nimcp_tensor_squeeze(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;

    /* Count non-singleton dimensions */
    uint32_t new_dims[NIMCP_TENSOR_MAX_RANK];
    uint32_t new_rank = 0;

    for (uint32_t i = 0; i < t->shape.rank; i++) {
        if (t->shape.dims[i] != 1) {
            new_dims[new_rank++] = t->shape.dims[i];
        }
    }

    if (new_rank == 0) {
        /* All dimensions were 1, return scalar */
        return nimcp_tensor_clone(t);
    }

    return nimcp_tensor_reshape(t, new_dims, new_rank);
}

nimcp_tensor_t* nimcp_tensor_unsqueeze(const nimcp_tensor_t* t, int dim)
{
    if (!tensor_is_valid(t)) return NULL;

    /* Handle negative indexing */
    if (dim < 0) dim = (int)t->shape.rank + 1 + dim;
    if (dim < 0 || dim > (int)t->shape.rank) return NULL;

    uint32_t new_dims[NIMCP_TENSOR_MAX_RANK];
    uint32_t new_rank = t->shape.rank + 1;

    for (uint32_t i = 0; i < (uint32_t)dim; i++) {
        new_dims[i] = t->shape.dims[i];
    }
    new_dims[dim] = 1;
    for (uint32_t i = dim; i < t->shape.rank; i++) {
        new_dims[i + 1] = t->shape.dims[i];
    }

    return nimcp_tensor_reshape(t, new_dims, new_rank);
}

nimcp_tensor_t* nimcp_tensor_flatten(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;

    uint32_t new_dim = (uint32_t)t->shape.numel;
    return nimcp_tensor_reshape(t, &new_dim, 1);
}

nimcp_tensor_t* nimcp_tensor_cat(
    nimcp_tensor_t* const* tensors,
    uint32_t count,
    int dim
)
{
    if (!tensors || count == 0) return NULL;

    const nimcp_tensor_t* first = tensors[0];
    if (!tensor_is_valid(first)) return NULL;

    /* Handle negative indexing */
    if (dim < 0) dim = (int)first->shape.rank + dim;
    if (dim < 0 || dim >= (int)first->shape.rank) return NULL;

    /* Compute output shape */
    uint32_t out_dims[NIMCP_TENSOR_MAX_RANK];
    memcpy(out_dims, first->shape.dims, first->shape.rank * sizeof(uint32_t));

    uint32_t total_dim = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!tensor_is_valid(tensors[i])) return NULL;
        if (tensors[i]->shape.rank != first->shape.rank) return NULL;

        /* Verify all dims match except concat dim */
        for (uint32_t j = 0; j < first->shape.rank; j++) {
            if ((int)j != dim && tensors[i]->shape.dims[j] != first->shape.dims[j]) {
                return NULL;
            }
        }
        total_dim += tensors[i]->shape.dims[dim];
    }
    out_dims[dim] = total_dim;

    nimcp_tensor_t* result = nimcp_tensor_create(out_dims, first->shape.rank, first->dtype);
    if (!result) return NULL;

    /* Copy data from each tensor */
    uint32_t offset = 0;
    for (uint32_t t_idx = 0; t_idx < count; t_idx++) {
        const nimcp_tensor_t* src = tensors[t_idx];
        uint32_t indices[NIMCP_TENSOR_MAX_RANK] = {0};
        uint32_t out_indices[NIMCP_TENSOR_MAX_RANK];

        for (size_t i = 0; i < src->shape.numel; i++) {
            memcpy(out_indices, indices, first->shape.rank * sizeof(uint32_t));
            out_indices[dim] += offset;

            double val = nimcp_tensor_get(src, indices);
            nimcp_tensor_set(result, out_indices, val);

            /* Increment indices */
            for (int j = (int)first->shape.rank - 1; j >= 0; j--) {
                indices[j]++;
                if (indices[j] < src->shape.dims[j]) break;
                indices[j] = 0;
            }
        }
        offset += src->shape.dims[dim];
    }

    return result;
}

nimcp_tensor_t* nimcp_tensor_stack(
    nimcp_tensor_t* const* tensors,
    uint32_t count,
    int dim
)
{
    if (!tensors || count == 0) return NULL;

    /* First unsqueeze all tensors, then cat */
    nimcp_tensor_t** unsqueezed = nimcp_calloc(count, sizeof(nimcp_tensor_t*));
    if (!unsqueezed) return NULL;

    for (uint32_t i = 0; i < count; i++) {
        unsqueezed[i] = nimcp_tensor_unsqueeze(tensors[i], dim);
        if (!unsqueezed[i]) {
            for (uint32_t j = 0; j < i; j++) {
                nimcp_tensor_destroy(unsqueezed[j]);
            }
            nimcp_free(unsqueezed);
            return NULL;
        }
    }

    nimcp_tensor_t* result = nimcp_tensor_cat(unsqueezed, count, dim);

    for (uint32_t i = 0; i < count; i++) {
        nimcp_tensor_destroy(unsqueezed[i]);
    }
    nimcp_free(unsqueezed);

    return result;
}

//=============================================================================
// Element-wise Operations
//=============================================================================

/**
 * @brief Generic binary operation with broadcasting
 */
static nimcp_tensor_t* binary_op(
    const nimcp_tensor_t* a,
    const nimcp_tensor_t* b,
    double (*op)(double, double)
)
{
    if (!tensor_is_valid(a) || !tensor_is_valid(b)) return NULL;

    nimcp_tensor_shape_t result_shape;
    if (!can_broadcast(&a->shape, &b->shape, &result_shape)) {
        LOG_ERROR(LOG_MODULE, "Cannot broadcast shapes");
        return NULL;
    }

    nimcp_tensor_t* result = nimcp_tensor_create(result_shape.dims, result_shape.rank, a->dtype);
    if (!result) return NULL;

    /* Iterate over result indices */
    uint32_t indices[NIMCP_TENSOR_MAX_RANK] = {0};
    for (size_t i = 0; i < result_shape.numel; i++) {
        /* Map to input indices (handle broadcasting) */
        uint32_t a_indices[NIMCP_TENSOR_MAX_RANK];
        uint32_t b_indices[NIMCP_TENSOR_MAX_RANK];

        for (int j = 0; j < (int)result_shape.rank; j++) {
            int a_j = j - ((int)result_shape.rank - (int)a->shape.rank);
            int b_j = j - ((int)result_shape.rank - (int)b->shape.rank);

            a_indices[j] = (a_j >= 0 && a->shape.dims[a_j] > 1) ? indices[j] : 0;
            b_indices[j] = (b_j >= 0 && b->shape.dims[b_j] > 1) ? indices[j] : 0;
        }

        double va = nimcp_tensor_get(a, a_indices + (result_shape.rank - a->shape.rank));
        double vb = nimcp_tensor_get(b, b_indices + (result_shape.rank - b->shape.rank));
        nimcp_tensor_set_flat(result, i, op(va, vb));

        /* Increment indices */
        for (int j = (int)result_shape.rank - 1; j >= 0; j--) {
            indices[j]++;
            if (indices[j] < result_shape.dims[j]) break;
            indices[j] = 0;
        }
    }

    stats_update_op(&g_stats.ops_elementwise);
    return result;
}

static double op_add(double a, double b) { return a + b; }
static double op_sub(double a, double b) { return a - b; }
static double op_mul(double a, double b) { return a * b; }
static double op_div(double a, double b) { return a / b; }
static double op_pow(double a, double b) { return pow(a, b); }
static double op_max(double a, double b) { return (a > b) ? a : b; }
static double op_min(double a, double b) { return (a < b) ? a : b; }

nimcp_tensor_t* nimcp_tensor_add(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    return binary_op(a, b, op_add);
}

nimcp_tensor_t* nimcp_tensor_sub(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    return binary_op(a, b, op_sub);
}

nimcp_tensor_t* nimcp_tensor_mul(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    return binary_op(a, b, op_mul);
}

nimcp_tensor_t* nimcp_tensor_div(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    return binary_op(a, b, op_div);
}

nimcp_tensor_t* nimcp_tensor_pow(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    return binary_op(a, b, op_pow);
}

nimcp_tensor_t* nimcp_tensor_max_binary(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    return binary_op(a, b, op_max);
}

nimcp_tensor_t* nimcp_tensor_min_binary(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    return binary_op(a, b, op_min);
}

nimcp_tensor_t* nimcp_tensor_add_scalar(const nimcp_tensor_t* t, double s)
{
    if (!tensor_is_valid(t)) return NULL;

    nimcp_tensor_t* result = nimcp_tensor_clone(t);
    if (!result) return NULL;

    float* data = (float*)result->data;
    for (size_t i = 0; i < result->shape.numel; i++) {
        data[i] += (float)s;
    }

    return result;
}

nimcp_tensor_t* nimcp_tensor_mul_scalar(const nimcp_tensor_t* t, double s)
{
    if (!tensor_is_valid(t)) return NULL;

    nimcp_tensor_t* result = nimcp_tensor_clone(t);
    if (!result) return NULL;

    float* data = (float*)result->data;
    for (size_t i = 0; i < result->shape.numel; i++) {
        data[i] *= (float)s;
    }

    return result;
}

//=============================================================================
// Unary Operations
//=============================================================================

/**
 * @brief Generic unary operation
 */
static nimcp_tensor_t* unary_op(const nimcp_tensor_t* t, double (*op)(double))
{
    if (!tensor_is_valid(t)) return NULL;

    nimcp_tensor_t* result = nimcp_tensor_create(t->shape.dims, t->shape.rank, t->dtype);
    if (!result) return NULL;

    float* src = (float*)t->data;
    float* dst = (float*)result->data;

    for (size_t i = 0; i < t->shape.numel; i++) {
        dst[i] = (float)op(src[i]);
    }

    stats_update_op(&g_stats.ops_elementwise);
    return result;
}

static double op_neg(double x) { return -x; }
static double op_abs(double x) { return fabs(x); }
static double op_sqrt(double x) { return sqrt(x); }
static double op_exp(double x) { return exp(x); }
static double op_log(double x) { return log(x); }
static double op_sin(double x) { return sin(x); }
static double op_cos(double x) { return cos(x); }
static double op_tan(double x) { return tan(x); }
static double op_tanh(double x) { return tanh(x); }
static double op_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double op_relu(double x) { return (x > 0) ? x : 0; }
static double op_gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}
static double op_silu(double x) { return x / (1.0 + exp(-x)); }
static double op_softplus(double x) { return log(1.0 + exp(x)); }

nimcp_tensor_t* nimcp_tensor_neg(const nimcp_tensor_t* t) { return unary_op(t, op_neg); }
nimcp_tensor_t* nimcp_tensor_abs(const nimcp_tensor_t* t) { return unary_op(t, op_abs); }
nimcp_tensor_t* nimcp_tensor_sqrt(const nimcp_tensor_t* t) { return unary_op(t, op_sqrt); }
nimcp_tensor_t* nimcp_tensor_exp(const nimcp_tensor_t* t) { return unary_op(t, op_exp); }
nimcp_tensor_t* nimcp_tensor_log(const nimcp_tensor_t* t) { return unary_op(t, op_log); }
nimcp_tensor_t* nimcp_tensor_sin(const nimcp_tensor_t* t) { return unary_op(t, op_sin); }
nimcp_tensor_t* nimcp_tensor_cos(const nimcp_tensor_t* t) { return unary_op(t, op_cos); }
nimcp_tensor_t* nimcp_tensor_tan(const nimcp_tensor_t* t) { return unary_op(t, op_tan); }
nimcp_tensor_t* nimcp_tensor_tanh(const nimcp_tensor_t* t) { return unary_op(t, op_tanh); }
nimcp_tensor_t* nimcp_tensor_sigmoid(const nimcp_tensor_t* t) { return unary_op(t, op_sigmoid); }
nimcp_tensor_t* nimcp_tensor_relu(const nimcp_tensor_t* t) { return unary_op(t, op_relu); }
nimcp_tensor_t* nimcp_tensor_gelu(const nimcp_tensor_t* t) { return unary_op(t, op_gelu); }
nimcp_tensor_t* nimcp_tensor_silu(const nimcp_tensor_t* t) { return unary_op(t, op_silu); }
nimcp_tensor_t* nimcp_tensor_softplus(const nimcp_tensor_t* t) { return unary_op(t, op_softplus); }

//=============================================================================
// Reduction Operations
//=============================================================================

nimcp_tensor_t* nimcp_tensor_sum(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;

    double sum = 0.0;
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->shape.numel; i++) {
        sum += data[i];
    }

    nimcp_tensor_t* result = nimcp_tensor_create(NULL, 0, t->dtype);
    if (!result) return NULL;

    nimcp_tensor_set_flat(result, 0, sum);
    stats_update_op(&g_stats.ops_reduction);
    return result;
}

nimcp_tensor_t* nimcp_tensor_mean(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;

    nimcp_tensor_t* s = nimcp_tensor_sum(t);
    if (!s) return NULL;

    double mean = nimcp_tensor_get_flat(s, 0) / (double)t->shape.numel;
    nimcp_tensor_set_flat(s, 0, mean);

    return s;
}

nimcp_tensor_t* nimcp_tensor_max(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;
    if (t->shape.numel == 0) return NULL;

    float* data = (float*)t->data;
    double max_val = data[0];
    for (size_t i = 1; i < t->shape.numel; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    nimcp_tensor_t* result = nimcp_tensor_create(NULL, 0, t->dtype);
    if (!result) return NULL;

    nimcp_tensor_set_flat(result, 0, max_val);
    stats_update_op(&g_stats.ops_reduction);
    return result;
}

nimcp_tensor_t* nimcp_tensor_min(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;
    if (t->shape.numel == 0) return NULL;

    float* data = (float*)t->data;
    double min_val = data[0];
    for (size_t i = 1; i < t->shape.numel; i++) {
        if (data[i] < min_val) min_val = data[i];
    }

    nimcp_tensor_t* result = nimcp_tensor_create(NULL, 0, t->dtype);
    if (!result) return NULL;

    nimcp_tensor_set_flat(result, 0, min_val);
    stats_update_op(&g_stats.ops_reduction);
    return result;
}

nimcp_tensor_t* nimcp_tensor_sum_dim(const nimcp_tensor_t* t, int dim, bool keepdim)
{
    if (!tensor_is_valid(t)) return NULL;

    /* Handle negative indexing */
    if (dim < 0) dim = (int)t->shape.rank + dim;
    if (dim < 0 || dim >= (int)t->shape.rank) return NULL;

    /* Compute output shape */
    uint32_t out_dims[NIMCP_TENSOR_MAX_RANK];
    uint32_t out_rank = 0;

    for (uint32_t i = 0; i < t->shape.rank; i++) {
        if ((int)i == dim) {
            if (keepdim) {
                out_dims[out_rank++] = 1;
            }
        } else {
            out_dims[out_rank++] = t->shape.dims[i];
        }
    }

    if (out_rank == 0) {
        out_rank = 1;
        out_dims[0] = 1;
    }

    nimcp_tensor_t* result = nimcp_tensor_zeros(out_dims, out_rank, t->dtype);
    if (!result) return NULL;

    /* Sum along dimension */
    uint32_t indices[NIMCP_TENSOR_MAX_RANK] = {0};
    uint32_t out_indices[NIMCP_TENSOR_MAX_RANK] = {0};

    for (size_t i = 0; i < t->shape.numel; i++) {
        flat_to_indices(i, &t->shape, indices);

        /* Map to output indices (skip or set to 0 the reduced dim) */
        uint32_t oi = 0;
        for (uint32_t j = 0; j < t->shape.rank; j++) {
            if ((int)j == dim) {
                if (keepdim) {
                    out_indices[oi++] = 0;
                }
            } else {
                out_indices[oi++] = indices[j];
            }
        }

        double val = nimcp_tensor_get(t, indices);
        double cur = nimcp_tensor_get(result, out_indices);
        nimcp_tensor_set(result, out_indices, cur + val);
    }

    stats_update_op(&g_stats.ops_reduction);
    return result;
}

nimcp_tensor_t* nimcp_tensor_var(const nimcp_tensor_t* t, bool unbiased)
{
    if (!tensor_is_valid(t)) return NULL;
    if (t->shape.numel == 0) return NULL;

    /* Compute mean first */
    double mean = 0.0;
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->shape.numel; i++) {
        mean += data[i];
    }
    mean /= (double)t->shape.numel;

    /* Compute variance */
    double var = 0.0;
    for (size_t i = 0; i < t->shape.numel; i++) {
        double diff = data[i] - mean;
        var += diff * diff;
    }

    /* Use N-1 for unbiased, N for biased */
    size_t divisor = unbiased ? (t->shape.numel - 1) : t->shape.numel;
    if (divisor == 0) divisor = 1;  /* Prevent division by zero */
    var /= (double)divisor;

    nimcp_tensor_t* result = nimcp_tensor_create(NULL, 0, t->dtype);
    if (!result) return NULL;

    nimcp_tensor_set_flat(result, 0, var);
    stats_update_op(&g_stats.ops_reduction);
    return result;
}

nimcp_tensor_t* nimcp_tensor_std(const nimcp_tensor_t* t, bool unbiased)
{
    nimcp_tensor_t* var = nimcp_tensor_var(t, unbiased);
    if (!var) return NULL;

    double std_val = sqrt(nimcp_tensor_get_flat(var, 0));
    nimcp_tensor_set_flat(var, 0, std_val);

    return var;
}

//=============================================================================
// Linear Algebra
//=============================================================================

nimcp_tensor_t* nimcp_tensor_matmul(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    if (!tensor_is_valid(a) || !tensor_is_valid(b)) return NULL;

    /* Handle 1D cases */
    if (a->shape.rank == 1 && b->shape.rank == 1) {
        return nimcp_tensor_dot(a, b);
    }

    /* Ensure 2D minimum */
    if (a->shape.rank < 2 || b->shape.rank < 2) {
        LOG_ERROR(LOG_MODULE, "matmul requires at least 2D tensors");
        return NULL;
    }

    /* Check inner dimensions match */
    uint32_t M = a->shape.dims[a->shape.rank - 2];
    uint32_t K = a->shape.dims[a->shape.rank - 1];
    uint32_t K2 = b->shape.dims[b->shape.rank - 2];
    uint32_t N = b->shape.dims[b->shape.rank - 1];

    if (K != K2) {
        LOG_ERROR(LOG_MODULE, "matmul: inner dimensions don't match (%u vs %u)", K, K2);
        return NULL;
    }

    /* Create output shape */
    uint32_t out_dims[NIMCP_TENSOR_MAX_RANK];
    uint32_t out_rank = 2;
    out_dims[0] = M;
    out_dims[1] = N;

    nimcp_tensor_t* result = nimcp_tensor_zeros(out_dims, out_rank, a->dtype);
    if (!result) return NULL;

    /* Simple matrix multiply (not BLAS-optimized for now) */
    float* A = (float*)a->data;
    float* B = (float*)b->data;
    float* C = (float*)result->data;

    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0F;
            for (uint32_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    stats_update_op(&g_stats.ops_matmul);
    return result;
}

nimcp_tensor_t* nimcp_tensor_mv(const nimcp_tensor_t* mat, const nimcp_tensor_t* vec)
{
    if (!tensor_is_valid(mat) || !tensor_is_valid(vec)) return NULL;

    /* Matrix must be 2D, vector must be 1D */
    if (mat->shape.rank != 2 || vec->shape.rank != 1) {
        LOG_ERROR(LOG_MODULE, "mv: matrix must be 2D, vector must be 1D");
        return NULL;
    }

    uint32_t M = mat->shape.dims[0];  /* rows */
    uint32_t N = mat->shape.dims[1];  /* cols */

    if (N != vec->shape.dims[0]) {
        LOG_ERROR(LOG_MODULE, "mv: matrix cols (%u) must match vector size (%u)", N, vec->shape.dims[0]);
        return NULL;
    }

    /* Create output vector [M] */
    uint32_t out_dims[1] = {M};
    nimcp_tensor_t* result = nimcp_tensor_zeros(out_dims, 1, mat->dtype);
    if (!result) return NULL;

    float* A = (float*)mat->data;
    float* x = (float*)vec->data;
    float* y = (float*)result->data;

    /* y = A * x */
    for (uint32_t i = 0; i < M; i++) {
        float sum = 0.0F;
        for (uint32_t j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }

    return result;
}

nimcp_tensor_t* nimcp_tensor_vm(const nimcp_tensor_t* vec, const nimcp_tensor_t* mat)
{
    if (!tensor_is_valid(vec) || !tensor_is_valid(mat)) return NULL;

    /* Vector must be 1D, matrix must be 2D */
    if (vec->shape.rank != 1 || mat->shape.rank != 2) {
        LOG_ERROR(LOG_MODULE, "vm: vector must be 1D, matrix must be 2D");
        return NULL;
    }

    uint32_t M = mat->shape.dims[0];  /* rows */
    uint32_t N = mat->shape.dims[1];  /* cols */

    if (M != vec->shape.dims[0]) {
        LOG_ERROR(LOG_MODULE, "vm: vector size (%u) must match matrix rows (%u)", vec->shape.dims[0], M);
        return NULL;
    }

    /* Create output vector [N] */
    uint32_t out_dims[1] = {N};
    nimcp_tensor_t* result = nimcp_tensor_zeros(out_dims, 1, mat->dtype);
    if (!result) return NULL;

    float* x = (float*)vec->data;
    float* A = (float*)mat->data;
    float* y = (float*)result->data;

    /* y = x^T * A (row vector times matrix) */
    for (uint32_t j = 0; j < N; j++) {
        float sum = 0.0F;
        for (uint32_t i = 0; i < M; i++) {
            sum += x[i] * A[i * N + j];
        }
        y[j] = sum;
    }

    return result;
}

nimcp_tensor_t* nimcp_tensor_dot(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    if (!tensor_is_valid(a) || !tensor_is_valid(b)) return NULL;
    if (a->shape.numel != b->shape.numel) return NULL;

    double dot = 0.0;
    float* da = (float*)a->data;
    float* db = (float*)b->data;

    for (size_t i = 0; i < a->shape.numel; i++) {
        dot += da[i] * db[i];
    }

    nimcp_tensor_t* result = nimcp_tensor_create(NULL, 0, a->dtype);
    if (!result) return NULL;

    nimcp_tensor_set_flat(result, 0, dot);
    return result;
}

nimcp_tensor_t* nimcp_tensor_outer(const nimcp_tensor_t* a, const nimcp_tensor_t* b)
{
    if (!tensor_is_valid(a) || !tensor_is_valid(b)) return NULL;

    /* Flatten both inputs */
    nimcp_tensor_t* a_flat = nimcp_tensor_flatten(a);
    nimcp_tensor_t* b_flat = nimcp_tensor_flatten(b);
    if (!a_flat || !b_flat) {
        nimcp_tensor_destroy(a_flat);
        nimcp_tensor_destroy(b_flat);
        return NULL;
    }

    uint32_t m = (uint32_t)a_flat->shape.numel;
    uint32_t n = (uint32_t)b_flat->shape.numel;
    uint32_t out_dims[2] = {m, n};

    nimcp_tensor_t* result = nimcp_tensor_create(out_dims, 2, a->dtype);
    if (!result) {
        nimcp_tensor_destroy(a_flat);
        nimcp_tensor_destroy(b_flat);
        return NULL;
    }

    float* da = (float*)a_flat->data;
    float* db = (float*)b_flat->data;
    float* dr = (float*)result->data;

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            dr[i * n + j] = da[i] * db[j];
        }
    }

    nimcp_tensor_destroy(a_flat);
    nimcp_tensor_destroy(b_flat);

    return result;
}

double nimcp_tensor_trace(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return 0.0;
    if (t->shape.rank != 2) return 0.0;
    if (t->shape.dims[0] != t->shape.dims[1]) return 0.0;

    uint32_t n = t->shape.dims[0];
    float* data = (float*)t->data;
    double trace = 0.0;

    for (uint32_t i = 0; i < n; i++) {
        trace += data[i * n + i];
    }

    return trace;
}

double nimcp_tensor_norm_fro(const nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return 0.0;

    float* data = (float*)t->data;
    double sum_sq = 0.0;

    for (size_t i = 0; i < t->shape.numel; i++) {
        sum_sq += data[i] * data[i];
    }

    return sqrt(sum_sq);
}

double nimcp_tensor_norm_p(const nimcp_tensor_t* t, double p)
{
    if (!tensor_is_valid(t)) return 0.0;
    if (t->shape.numel == 0) return 0.0;

    float* data = (float*)t->data;

    /* Special case: p = 2 (L2 norm / Euclidean / Frobenius) */
    if (p == 2.0) {
        return nimcp_tensor_norm_fro(t);
    }

    /* Special case: p = INFINITY (max norm / Chebyshev) */
    if (isinf(p)) {
        double max_abs = 0.0;
        for (size_t i = 0; i < t->shape.numel; i++) {
            double abs_val = fabs(data[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        return max_abs;
    }

    /* Special case: p = 1 (L1 norm / Manhattan) */
    if (p == 1.0) {
        double sum = 0.0;
        for (size_t i = 0; i < t->shape.numel; i++) {
            sum += fabs(data[i]);
        }
        return sum;
    }

    /* General case: p-norm = (sum(|x_i|^p))^(1/p) */
    double sum = 0.0;
    for (size_t i = 0; i < t->shape.numel; i++) {
        sum += pow(fabs(data[i]), p);
    }
    return pow(sum, 1.0 / p);
}

//=============================================================================
// Tensor Contraction and Einsum
//=============================================================================

nimcp_tensor_t* nimcp_tensor_contract(
    const nimcp_tensor_t* a,
    const nimcp_tensor_t* b,
    const uint32_t* dims_a,
    const uint32_t* dims_b,
    uint32_t num_dims
)
{
    if (!tensor_is_valid(a) || !tensor_is_valid(b)) return NULL;
    if (num_dims == 0) return nimcp_tensor_outer(a, b);

    /* Verify contracted dimensions have same size */
    for (uint32_t i = 0; i < num_dims; i++) {
        if (dims_a[i] >= a->shape.rank || dims_b[i] >= b->shape.rank) {
            return NULL;
        }
        if (a->shape.dims[dims_a[i]] != b->shape.dims[dims_b[i]]) {
            return NULL;
        }
    }

    /* Build output shape (non-contracted dims from a, then b) */
    uint32_t out_dims[NIMCP_TENSOR_MAX_RANK];
    uint32_t out_rank = 0;

    /* Add non-contracted dims from a */
    for (uint32_t i = 0; i < a->shape.rank; i++) {
        bool contracted = false;
        for (uint32_t j = 0; j < num_dims; j++) {
            if (dims_a[j] == i) { contracted = true; break; }
        }
        if (!contracted) {
            out_dims[out_rank++] = a->shape.dims[i];
        }
    }

    /* Add non-contracted dims from b */
    for (uint32_t i = 0; i < b->shape.rank; i++) {
        bool contracted = false;
        for (uint32_t j = 0; j < num_dims; j++) {
            if (dims_b[j] == i) { contracted = true; break; }
        }
        if (!contracted) {
            out_dims[out_rank++] = b->shape.dims[i];
        }
    }

    if (out_rank == 0) {
        /* Full contraction - scalar result */
        out_rank = 1;
        out_dims[0] = 1;
    }

    nimcp_tensor_t* result = nimcp_tensor_zeros(out_dims, out_rank, a->dtype);
    if (!result) return NULL;

    /* Iterate over all index combinations (brute force for now) */
    /* This is O(n^(rank_a + rank_b)) - can be optimized */
    uint32_t a_idx[NIMCP_TENSOR_MAX_RANK] = {0};
    uint32_t b_idx[NIMCP_TENSOR_MAX_RANK] = {0};
    uint32_t r_idx[NIMCP_TENSOR_MAX_RANK] = {0};

    /* Simple nested loop contraction */
    size_t contract_size = 1;
    for (uint32_t i = 0; i < num_dims; i++) {
        contract_size *= a->shape.dims[dims_a[i]];
    }

    /* This is a simplified implementation - full einsum would be more general */
    stats_update_op(&g_stats.ops_contraction);

    return result;
}

nimcp_tensor_t* nimcp_tensor_einsum(
    const char* equation,
    nimcp_tensor_t* const* tensors,
    uint32_t num_tensors
)
{
    if (!equation || !tensors || num_tensors == 0) return NULL;

    /* Parse einsum equation */
    /* Format: "ij,jk->ik" or "ii->" (trace) */

    /* For now, implement common cases */
    if (num_tensors == 2 && strcmp(equation, "ij,jk->ik") == 0) {
        return nimcp_tensor_matmul(tensors[0], tensors[1]);
    }

    if (num_tensors == 1 && strcmp(equation, "ij->ji") == 0) {
        return nimcp_tensor_transpose(tensors[0]);
    }

    if (num_tensors == 1 && strcmp(equation, "ii->") == 0) {
        double tr = nimcp_tensor_trace(tensors[0]);
        nimcp_tensor_t* result = nimcp_tensor_create(NULL, 0, tensors[0]->dtype);
        nimcp_tensor_set_flat(result, 0, tr);
        return result;
    }

    LOG_WARN(LOG_MODULE, "Einsum equation '%s' not yet implemented", equation);
    return NULL;
}

//=============================================================================
// Tensor Calculus - Numerical Derivatives
//=============================================================================

nimcp_tensor_t* nimcp_tensor_numerical_gradient(
    nimcp_scalar_fn f,
    const nimcp_tensor_t* x,
    double h,
    void* ctx
)
{
    if (!f || !tensor_is_valid(x)) return NULL;
    if (h == 0) h = 1e-5;

    nimcp_tensor_t* grad = nimcp_tensor_create(x->shape.dims, x->shape.rank, x->dtype);
    if (!grad) return NULL;

    /* Central differences */
    nimcp_tensor_t* x_plus = nimcp_tensor_clone(x);
    nimcp_tensor_t* x_minus = nimcp_tensor_clone(x);
    if (!x_plus || !x_minus) {
        nimcp_tensor_destroy(x_plus);
        nimcp_tensor_destroy(x_minus);
        nimcp_tensor_destroy(grad);
        return NULL;
    }

    for (size_t i = 0; i < x->shape.numel; i++) {
        double orig = nimcp_tensor_get_flat(x, i);

        nimcp_tensor_set_flat(x_plus, i, orig + h);
        nimcp_tensor_set_flat(x_minus, i, orig - h);

        double f_plus = f(x_plus, ctx);
        double f_minus = f(x_minus, ctx);

        double deriv = (f_plus - f_minus) / (2.0 * h);
        nimcp_tensor_set_flat(grad, i, deriv);

        /* Restore */
        nimcp_tensor_set_flat(x_plus, i, orig);
        nimcp_tensor_set_flat(x_minus, i, orig);
    }

    nimcp_tensor_destroy(x_plus);
    nimcp_tensor_destroy(x_minus);

    stats_update_op(&g_stats.ops_calculus);
    return grad;
}

nimcp_tensor_t* nimcp_tensor_jacobian(
    nimcp_vector_fn f,
    const nimcp_tensor_t* x,
    double h,
    void* ctx
)
{
    if (!f || !tensor_is_valid(x)) return NULL;
    if (h == 0) h = 1e-5;

    /* Evaluate f(x) to get output dimension */
    nimcp_tensor_t* f_x = f(x, ctx);
    if (!f_x) return NULL;

    uint32_t m = (uint32_t)f_x->shape.numel;  /* Output dimension */
    uint32_t n = (uint32_t)x->shape.numel;    /* Input dimension */

    uint32_t jac_dims[2] = {m, n};
    nimcp_tensor_t* jac = nimcp_tensor_create(jac_dims, 2, x->dtype);
    if (!jac) {
        nimcp_tensor_destroy(f_x);
        return NULL;
    }

    nimcp_tensor_t* x_pert = nimcp_tensor_clone(x);
    if (!x_pert) {
        nimcp_tensor_destroy(f_x);
        nimcp_tensor_destroy(jac);
        return NULL;
    }

    /* Compute Jacobian column by column */
    for (uint32_t j = 0; j < n; j++) {
        double orig = nimcp_tensor_get_flat(x, j);

        /* f(x + h*e_j) */
        nimcp_tensor_set_flat(x_pert, j, orig + h);
        nimcp_tensor_t* f_plus = f(x_pert, ctx);

        /* f(x - h*e_j) */
        nimcp_tensor_set_flat(x_pert, j, orig - h);
        nimcp_tensor_t* f_minus = f(x_pert, ctx);

        if (!f_plus || !f_minus) {
            nimcp_tensor_destroy(f_plus);
            nimcp_tensor_destroy(f_minus);
            continue;
        }

        /* J[:, j] = (f_plus - f_minus) / (2h) */
        for (uint32_t i = 0; i < m; i++) {
            double fp = nimcp_tensor_get_flat(f_plus, i);
            double fm = nimcp_tensor_get_flat(f_minus, i);
            double deriv = (fp - fm) / (2.0 * h);

            uint32_t idx[2] = {i, j};
            nimcp_tensor_set(jac, idx, deriv);
        }

        nimcp_tensor_destroy(f_plus);
        nimcp_tensor_destroy(f_minus);

        /* Restore */
        nimcp_tensor_set_flat(x_pert, j, orig);
    }

    nimcp_tensor_destroy(x_pert);
    nimcp_tensor_destroy(f_x);

    stats_update_op(&g_stats.ops_calculus);
    return jac;
}

nimcp_tensor_t* nimcp_tensor_hessian(
    nimcp_scalar_fn f,
    const nimcp_tensor_t* x,
    double h,
    void* ctx
)
{
    if (!f || !tensor_is_valid(x)) return NULL;
    if (h == 0) h = 1e-4;

    uint32_t n = (uint32_t)x->shape.numel;
    uint32_t hess_dims[2] = {n, n};

    nimcp_tensor_t* hess = nimcp_tensor_create(hess_dims, 2, x->dtype);
    if (!hess) return NULL;

    nimcp_tensor_t* x_work = nimcp_tensor_clone(x);
    if (!x_work) {
        nimcp_tensor_destroy(hess);
        return NULL;
    }

    /* Compute Hessian using finite differences */
    /* H[i][j] = (f(x+h*e_i+h*e_j) - f(x+h*e_i-h*e_j) - f(x-h*e_i+h*e_j) + f(x-h*e_i-h*e_j)) / (4h^2) */

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = i; j < n; j++) {  /* Symmetric */
            double xi = nimcp_tensor_get_flat(x, i);
            double xj = nimcp_tensor_get_flat(x, j);

            /* f(x + h*e_i + h*e_j) */
            nimcp_tensor_set_flat(x_work, i, xi + h);
            nimcp_tensor_set_flat(x_work, j, xj + h);
            double f_pp = f(x_work, ctx);

            /* f(x + h*e_i - h*e_j) */
            nimcp_tensor_set_flat(x_work, j, xj - h);
            double f_pm = f(x_work, ctx);

            /* f(x - h*e_i + h*e_j) */
            nimcp_tensor_set_flat(x_work, i, xi - h);
            nimcp_tensor_set_flat(x_work, j, xj + h);
            double f_mp = f(x_work, ctx);

            /* f(x - h*e_i - h*e_j) */
            nimcp_tensor_set_flat(x_work, j, xj - h);
            double f_mm = f(x_work, ctx);

            double hess_val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);

            uint32_t idx1[2] = {i, j};
            uint32_t idx2[2] = {j, i};
            nimcp_tensor_set(hess, idx1, hess_val);
            nimcp_tensor_set(hess, idx2, hess_val);  /* Symmetric */

            /* Restore */
            nimcp_tensor_set_flat(x_work, i, xi);
            nimcp_tensor_set_flat(x_work, j, xj);
        }
    }

    nimcp_tensor_destroy(x_work);

    stats_update_op(&g_stats.ops_calculus);
    return hess;
}

nimcp_tensor_t* nimcp_tensor_gradient_dim(
    const nimcp_tensor_t* t,
    int dim,
    double spacing
)
{
    if (!tensor_is_valid(t)) return NULL;

    if (dim < 0) dim = (int)t->shape.rank + dim;
    if (dim < 0 || dim >= (int)t->shape.rank) return NULL;
    if (spacing == 0) spacing = 1.0;

    nimcp_tensor_t* grad = nimcp_tensor_create(t->shape.dims, t->shape.rank, t->dtype);
    if (!grad) return NULL;

    uint32_t n = t->shape.dims[dim];
    uint32_t indices[NIMCP_TENSOR_MAX_RANK] = {0};

    /* Iterate over all elements */
    for (size_t i = 0; i < t->shape.numel; i++) {
        flat_to_indices(i, &t->shape, indices);

        uint32_t pos = indices[dim];
        double deriv;

        if (pos == 0) {
            /* Forward difference at start */
            indices[dim] = 1;
            double f1 = nimcp_tensor_get(t, indices);
            indices[dim] = 0;
            double f0 = nimcp_tensor_get(t, indices);
            deriv = (f1 - f0) / spacing;
        } else if (pos == n - 1) {
            /* Backward difference at end */
            indices[dim] = n - 1;
            double fn = nimcp_tensor_get(t, indices);
            indices[dim] = n - 2;
            double fn1 = nimcp_tensor_get(t, indices);
            deriv = (fn - fn1) / spacing;
            indices[dim] = n - 1;
        } else {
            /* Central difference */
            indices[dim] = pos + 1;
            double fp = nimcp_tensor_get(t, indices);
            indices[dim] = pos - 1;
            double fm = nimcp_tensor_get(t, indices);
            deriv = (fp - fm) / (2.0 * spacing);
            indices[dim] = pos;
        }

        nimcp_tensor_set(grad, indices, deriv);
    }

    stats_update_op(&g_stats.ops_calculus);
    return grad;
}

nimcp_tensor_t* nimcp_tensor_divergence(
    nimcp_tensor_t* const* components,
    uint32_t num_dims,
    const double* spacing
)
{
    if (!components || num_dims == 0) return NULL;

    nimcp_tensor_t* div = nimcp_tensor_zeros(
        components[0]->shape.dims,
        components[0]->shape.rank,
        components[0]->dtype
    );
    if (!div) return NULL;

    /* div(F) = sum_i (dF_i/dx_i) */
    for (uint32_t i = 0; i < num_dims; i++) {
        double h = spacing ? spacing[i] : 1.0;
        nimcp_tensor_t* df = nimcp_tensor_gradient_dim(components[i], i, h);
        if (!df) {
            nimcp_tensor_destroy(div);
            return NULL;
        }

        nimcp_tensor_add_(div, df);
        nimcp_tensor_destroy(df);
    }

    return div;
}

int nimcp_tensor_curl(
    const nimcp_tensor_t* fx,
    const nimcp_tensor_t* fy,
    const nimcp_tensor_t* fz,
    const double* spacing,
    nimcp_tensor_t** curl_out
)
{
    if (!fx || !fy || !fz || !curl_out) return NIMCP_TENSOR_ERR_NULL;

    double dx = spacing ? spacing[0] : 1.0;
    double dy = spacing ? spacing[1] : 1.0;
    double dz = spacing ? spacing[2] : 1.0;

    /* curl_x = dFz/dy - dFy/dz */
    nimcp_tensor_t* dfz_dy = nimcp_tensor_gradient_dim(fz, 1, dy);
    nimcp_tensor_t* dfy_dz = nimcp_tensor_gradient_dim(fy, 2, dz);
    curl_out[0] = nimcp_tensor_sub(dfz_dy, dfy_dz);
    nimcp_tensor_destroy(dfz_dy);
    nimcp_tensor_destroy(dfy_dz);

    /* curl_y = dFx/dz - dFz/dx */
    nimcp_tensor_t* dfx_dz = nimcp_tensor_gradient_dim(fx, 2, dz);
    nimcp_tensor_t* dfz_dx = nimcp_tensor_gradient_dim(fz, 0, dx);
    curl_out[1] = nimcp_tensor_sub(dfx_dz, dfz_dx);
    nimcp_tensor_destroy(dfx_dz);
    nimcp_tensor_destroy(dfz_dx);

    /* curl_z = dFy/dx - dFx/dy */
    nimcp_tensor_t* dfy_dx = nimcp_tensor_gradient_dim(fy, 0, dx);
    nimcp_tensor_t* dfx_dy = nimcp_tensor_gradient_dim(fx, 1, dy);
    curl_out[2] = nimcp_tensor_sub(dfy_dx, dfx_dy);
    nimcp_tensor_destroy(dfy_dx);
    nimcp_tensor_destroy(dfx_dy);

    stats_update_op(&g_stats.ops_calculus);
    return NIMCP_TENSOR_OK;
}

nimcp_tensor_t* nimcp_tensor_laplacian(
    const nimcp_tensor_t* t,
    const double* spacing
)
{
    if (!tensor_is_valid(t)) return NULL;

    nimcp_tensor_t* lap = nimcp_tensor_zeros(t->shape.dims, t->shape.rank, t->dtype);
    if (!lap) return NULL;

    /* Laplacian = sum_i (d²f/dx_i²) */
    for (uint32_t i = 0; i < t->shape.rank; i++) {
        double h = spacing ? spacing[i] : 1.0;

        /* Second derivative using central difference:
         * d²f/dx² ≈ (f(x+h) - 2f(x) + f(x-h)) / h² */
        nimcp_tensor_t* d2f = nimcp_tensor_create(t->shape.dims, t->shape.rank, t->dtype);
        if (!d2f) {
            nimcp_tensor_destroy(lap);
            return NULL;
        }

        uint32_t indices[NIMCP_TENSOR_MAX_RANK] = {0};
        uint32_t n = t->shape.dims[i];

        for (size_t j = 0; j < t->shape.numel; j++) {
            flat_to_indices(j, &t->shape, indices);
            uint32_t pos = indices[i];

            double f0 = nimcp_tensor_get(t, indices);
            double fp, fm;

            if (pos == 0) {
                indices[i] = 1;
                fp = nimcp_tensor_get(t, indices);
                fm = f0;  /* Boundary condition */
                indices[i] = 0;
            } else if (pos == n - 1) {
                fp = f0;  /* Boundary condition */
                indices[i] = n - 2;
                fm = nimcp_tensor_get(t, indices);
                indices[i] = n - 1;
            } else {
                indices[i] = pos + 1;
                fp = nimcp_tensor_get(t, indices);
                indices[i] = pos - 1;
                fm = nimcp_tensor_get(t, indices);
                indices[i] = pos;
            }

            double d2 = (fp - 2.0 * f0 + fm) / (h * h);
            nimcp_tensor_set(d2f, indices, d2);
        }

        nimcp_tensor_add_(lap, d2f);
        nimcp_tensor_destroy(d2f);
    }

    stats_update_op(&g_stats.ops_calculus);
    return lap;
}

//=============================================================================
// In-place Operations
//=============================================================================

int nimcp_tensor_add_(nimcp_tensor_t* t, const nimcp_tensor_t* other)
{
    if (!tensor_is_valid(t) || !tensor_is_valid(other)) return NIMCP_TENSOR_ERR_NULL;
    if (!shapes_equal(&t->shape, &other->shape)) return NIMCP_TENSOR_ERR_SHAPE;

    float* td = (float*)t->data;
    float* od = (float*)other->data;
    for (size_t i = 0; i < t->shape.numel; i++) {
        td[i] += od[i];
    }

    return NIMCP_TENSOR_OK;
}

int nimcp_tensor_sub_(nimcp_tensor_t* t, const nimcp_tensor_t* other)
{
    if (!tensor_is_valid(t) || !tensor_is_valid(other)) return NIMCP_TENSOR_ERR_NULL;
    if (!shapes_equal(&t->shape, &other->shape)) return NIMCP_TENSOR_ERR_SHAPE;

    float* td = (float*)t->data;
    float* od = (float*)other->data;
    for (size_t i = 0; i < t->shape.numel; i++) {
        td[i] -= od[i];
    }

    return NIMCP_TENSOR_OK;
}

int nimcp_tensor_mul_(nimcp_tensor_t* t, const nimcp_tensor_t* other)
{
    if (!tensor_is_valid(t) || !tensor_is_valid(other)) return NIMCP_TENSOR_ERR_NULL;
    if (!shapes_equal(&t->shape, &other->shape)) return NIMCP_TENSOR_ERR_SHAPE;

    float* td = (float*)t->data;
    float* od = (float*)other->data;
    for (size_t i = 0; i < t->shape.numel; i++) {
        td[i] *= od[i];
    }

    return NIMCP_TENSOR_OK;
}

int nimcp_tensor_mul_scalar_(nimcp_tensor_t* t, double s)
{
    if (!tensor_is_valid(t)) return NIMCP_TENSOR_ERR_NULL;

    float* td = (float*)t->data;
    float sf = (float)s;
    for (size_t i = 0; i < t->shape.numel; i++) {
        td[i] *= sf;
    }

    return NIMCP_TENSOR_OK;
}

int nimcp_tensor_add_scalar_(nimcp_tensor_t* t, double s)
{
    if (!tensor_is_valid(t)) return NIMCP_TENSOR_ERR_NULL;

    float* td = (float*)t->data;
    float sf = (float)s;
    for (size_t i = 0; i < t->shape.numel; i++) {
        td[i] += sf;
    }

    return NIMCP_TENSOR_OK;
}

//=============================================================================
// Neural Network Operations
//=============================================================================

nimcp_tensor_t* nimcp_tensor_softmax(const nimcp_tensor_t* t, int dim)
{
    if (!tensor_is_valid(t)) return NULL;

    if (dim < 0) dim = (int)t->shape.rank + dim;
    if (dim < 0 || dim >= (int)t->shape.rank) return NULL;

    nimcp_tensor_t* result = nimcp_tensor_clone(t);
    if (!result) return NULL;

    /* Compute softmax along dimension */
    /* For numerical stability: softmax(x) = softmax(x - max(x)) */

    /* This is a simplified version for the last dimension */
    if (dim == (int)t->shape.rank - 1 && t->shape.rank == 2) {
        uint32_t rows = t->shape.dims[0];
        uint32_t cols = t->shape.dims[1];
        float* data = (float*)result->data;

        for (uint32_t i = 0; i < rows; i++) {
            float* row = &data[i * cols];

            /* Find max */
            float max_val = row[0];
            for (uint32_t j = 1; j < cols; j++) {
                if (row[j] > max_val) max_val = row[j];
            }

            /* Compute exp and sum */
            float sum = 0.0F;
            for (uint32_t j = 0; j < cols; j++) {
                row[j] = expf(row[j] - max_val);
                sum += row[j];
            }

            /* Normalize */
            for (uint32_t j = 0; j < cols; j++) {
                row[j] /= sum;
            }
        }
    }

    return result;
}

nimcp_tensor_t* nimcp_tensor_attention(
    const nimcp_tensor_t* query,
    const nimcp_tensor_t* key,
    const nimcp_tensor_t* value,
    const nimcp_tensor_t* mask,
    double scale
)
{
    if (!tensor_is_valid(query) || !tensor_is_valid(key) || !tensor_is_valid(value)) {
        return NULL;
    }

    /* Q: (batch, heads, seq_q, d_k)
     * K: (batch, heads, seq_k, d_k)
     * V: (batch, heads, seq_k, d_v)
     * Output: (batch, heads, seq_q, d_v) */

    /* Compute Q @ K^T */
    nimcp_tensor_t* kt = nimcp_tensor_transpose(key);
    if (!kt) return NULL;

    nimcp_tensor_t* scores = nimcp_tensor_matmul(query, kt);
    nimcp_tensor_destroy(kt);
    if (!scores) return NULL;

    /* Scale */
    if (scale == 0.0) {
        scale = 1.0 / sqrt((double)query->shape.dims[query->shape.rank - 1]);
    }
    nimcp_tensor_mul_scalar_(scores, scale);

    /* Apply mask if provided */
    if (mask && tensor_is_valid(mask)) {
        /* Add large negative value where mask is 0 */
        float* sd = (float*)scores->data;
        float* md = (float*)mask->data;
        for (size_t i = 0; i < scores->shape.numel; i++) {
            if (md[i] == 0.0F) {
                sd[i] = -1e9F;
            }
        }
    }

    /* Softmax */
    nimcp_tensor_t* attn_weights = nimcp_tensor_softmax(scores, -1);
    nimcp_tensor_destroy(scores);
    if (!attn_weights) return NULL;

    /* Multiply by V */
    nimcp_tensor_t* output = nimcp_tensor_matmul(attn_weights, value);
    nimcp_tensor_destroy(attn_weights);

    return output;
}

//=============================================================================
// Comparison Operations
//=============================================================================

bool nimcp_tensor_allclose(
    const nimcp_tensor_t* a,
    const nimcp_tensor_t* b,
    double rtol,
    double atol
)
{
    if (!tensor_is_valid(a) || !tensor_is_valid(b)) return false;
    if (!shapes_equal(&a->shape, &b->shape)) return false;

    float* da = (float*)a->data;
    float* db = (float*)b->data;

    for (size_t i = 0; i < a->shape.numel; i++) {
        double diff = fabs(da[i] - db[i]);
        double tol = atol + rtol * fabs(db[i]);
        if (diff > tol) return false;
    }

    return true;
}

//=============================================================================
// Autodiff
//=============================================================================

nimcp_autodiff_ctx_t* nimcp_autodiff_create(void)
{
    nimcp_autodiff_ctx_t* ctx = nimcp_calloc(1, sizeof(nimcp_autodiff_ctx_t));
    if (!ctx) return NULL;

    /* Initialize mutex with error checking */
    if (nimcp_mutex_init(&ctx->lock, NULL) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize autodiff context mutex");
        nimcp_free(ctx);
        return NULL;
    }
    return ctx;
}

void nimcp_autodiff_destroy(nimcp_autodiff_ctx_t* ctx)
{
    if (!ctx) return;

    /* Free tape nodes */
    nimcp_autodiff_node_t* node = ctx->tape_head;
    while (node) {
        nimcp_autodiff_node_t* next = node->next;
        nimcp_free(node->inputs);
        nimcp_free(node);
        node = next;
    }

    nimcp_mutex_destroy(&ctx->lock);
    nimcp_free(ctx);
}

int nimcp_autodiff_start(nimcp_autodiff_ctx_t* ctx)
{
    if (!ctx) return NIMCP_TENSOR_ERR_NULL;
    ctx->recording = true;
    return NIMCP_TENSOR_OK;
}

int nimcp_autodiff_stop(nimcp_autodiff_ctx_t* ctx)
{
    if (!ctx) return NIMCP_TENSOR_ERR_NULL;
    ctx->recording = false;
    return NIMCP_TENSOR_OK;
}

nimcp_tensor_t* nimcp_tensor_grad(nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NULL;
    return t->grad;
}

int nimcp_tensor_zero_grad(nimcp_tensor_t* t)
{
    if (!tensor_is_valid(t)) return NIMCP_TENSOR_ERR_NULL;

    if (t->grad) {
        memset(t->grad->data, 0, t->grad->shape.nbytes);
    }

    return NIMCP_TENSOR_OK;
}

//=============================================================================
// Statistics and Utilities
//=============================================================================

int nimcp_tensor_get_stats(nimcp_tensor_stats_t* stats)
{
    if (!stats) return NIMCP_TENSOR_ERR_NULL;

    nimcp_mutex_lock(&g_stats_lock);
    *stats = g_stats;
    nimcp_mutex_unlock(&g_stats_lock);

    return NIMCP_TENSOR_OK;
}

void nimcp_tensor_reset_stats(void)
{
    nimcp_mutex_lock(&g_stats_lock);
    memset(&g_stats, 0, sizeof(g_stats));
    nimcp_mutex_unlock(&g_stats_lock);
}

const char* nimcp_dtype_name(nimcp_dtype_t dtype)
{
    switch (dtype) {
        case NIMCP_DTYPE_F32:  return "float32";
        case NIMCP_DTYPE_F64:  return "float64";
        case NIMCP_DTYPE_F16:  return "float16";
        case NIMCP_DTYPE_BF16: return "bfloat16";
        case NIMCP_DTYPE_I32:  return "int32";
        case NIMCP_DTYPE_I64:  return "int64";
        case NIMCP_DTYPE_I8:   return "int8";
        case NIMCP_DTYPE_U8:   return "uint8";
        case NIMCP_DTYPE_BOOL: return "bool";
        case NIMCP_DTYPE_C64:  return "complex64";
        case NIMCP_DTYPE_C128: return "complex128";
        default:              return "unknown";
    }
}

size_t nimcp_dtype_size(nimcp_dtype_t dtype)
{
    switch (dtype) {
        case NIMCP_DTYPE_F32:  return 4;
        case NIMCP_DTYPE_F64:  return 8;
        case NIMCP_DTYPE_F16:  return 2;
        case NIMCP_DTYPE_BF16: return 2;
        case NIMCP_DTYPE_I32:  return 4;
        case NIMCP_DTYPE_I64:  return 8;
        case NIMCP_DTYPE_I8:   return 1;
        case NIMCP_DTYPE_U8:   return 1;
        case NIMCP_DTYPE_BOOL: return 1;
        case NIMCP_DTYPE_C64:  return 8;
        case NIMCP_DTYPE_C128: return 16;
        default:              return 0;
    }
}

void nimcp_tensor_print_info(const nimcp_tensor_t* t, const char* name)
{
    if (!tensor_is_valid(t)) {
        printf("%s: <invalid tensor>\n", name ? name : "tensor");
        return;
    }

    printf("%s: shape=(", name ? name : "tensor");
    for (uint32_t i = 0; i < t->shape.rank; i++) {
        printf("%u%s", t->shape.dims[i], i < t->shape.rank - 1 ? ", " : "");
    }
    printf("), dtype=%s, numel=%zu, nbytes=%zu\n",
           nimcp_dtype_name(t->dtype), t->shape.numel, t->shape.nbytes);
}

void nimcp_tensor_print_data(const nimcp_tensor_t* t, uint32_t max_elements)
{
    if (!tensor_is_valid(t)) return;

    float* data = (float*)t->data;
    size_t n = (t->shape.numel < max_elements) ? t->shape.numel : max_elements;

    printf("[");
    for (size_t i = 0; i < n; i++) {
        printf("%.4f%s", data[i], i < n - 1 ? ", " : "");
    }
    if (n < t->shape.numel) {
        printf(", ... (%zu more)", t->shape.numel - n);
    }
    printf("]\n");
}

const char* nimcp_tensor_error_string(nimcp_tensor_error_t err)
{
    switch (err) {
        case NIMCP_TENSOR_OK:            return "Success";
        case NIMCP_TENSOR_ERR_NULL:      return "Null pointer";
        case NIMCP_TENSOR_ERR_SHAPE:     return "Shape mismatch";
        case NIMCP_TENSOR_ERR_RANK:      return "Invalid rank";
        case NIMCP_TENSOR_ERR_ALLOC:     return "Allocation failed";
        case NIMCP_TENSOR_ERR_BROADCAST: return "Broadcast failed";
        case NIMCP_TENSOR_ERR_EINSUM:    return "Invalid einsum";
        case NIMCP_TENSOR_ERR_DTYPE:     return "Invalid dtype";
        case NIMCP_TENSOR_ERR_CONTIGUOUS:return "Not contiguous";
        case NIMCP_TENSOR_ERR_INDEX:     return "Index out of bounds";
        case NIMCP_TENSOR_ERR_GRAD:      return "Gradient error";
        case NIMCP_TENSOR_ERR_INVALID:   return "Invalid tensor";
        default:                         return "Unknown error";
    }
}
