/**
 * @file nimcp_cognitive_dispatch.h
 * @brief Parallel cognitive module dispatch using actor pattern
 *
 * WHAT: Enables concurrent execution of cognitive training and inference modules
 * WHY:  Sequential execution of 12+ independent modules wastes ~40s per training step
 * HOW:  Actor pattern — each module owns its state, receives input via task arg,
 *       returns results via promise. Coordinator merges results single-threaded.
 *       Zero shared mutable state between modules. No mutexes between actors.
 *
 * ARCHITECTURE:
 *   Coordinator (main thread)
 *     ├── creates promise[0..N-1]
 *     ├── submits task[0..N-1] to thread pool
 *     ├── nimcp_future_all() barrier
 *     ├── collects results from futures
 *     └── updates cognitive_stats single-threaded
 */

#ifndef NIMCP_COGNITIVE_DISPATCH_H
#define NIMCP_COGNITIVE_DISPATCH_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>  /* NAN */

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct brain_struct* brain_t;

/*=============================================================================
 * Training Module IDs
 *=============================================================================*/

typedef enum {
    COG_MODULE_GROUNDED_LANG = 0,
    COG_MODULE_KNOWLEDGE,
    COG_MODULE_VAE,
    COG_MODULE_FEP_PARIETAL,
    COG_MODULE_PHYSICS_NN,
    COG_MODULE_PRED_HIERARCHY,
    COG_MODULE_JEPA,
    COG_MODULE_CREATIVE,
    COG_MODULE_SELF_HEAL,
    COG_MODULE_INTUITION,
    COG_MODULE_FEP_ORCHESTRATOR,
    COG_MODULE_SNN_FNO,
    COG_MODULE_COUNT  /* 12 */
} cognitive_module_id_t;

/*=============================================================================
 * Result Structs
 *=============================================================================*/

/**
 * @brief Result from a single cognitive training module task
 *
 * Returned via promise after the module completes its training step.
 * The coordinator reads these single-threaded — no contention.
 */
typedef struct {
    cognitive_module_id_t module_id;
    bool     executed;      /**< true if module was non-NULL and ran */
    bool     success;       /**< true if no error */
    float    loss;          /**< loss value (NAN if module doesn't produce one) */
    float    aux_value;     /**< secondary metric (vae free_energy, fep surprise, etc.) */
    uint64_t elapsed_us;    /**< wall-clock time for this module */
} cognitive_task_result_t;

/**
 * @brief Batch result from parallel cognitive training dispatch
 */
typedef struct {
    cognitive_task_result_t results[COG_MODULE_COUNT];
    uint32_t               num_executed;    /**< count of modules that actually ran */
    uint64_t               total_elapsed_us;/**< wall-clock for entire parallel batch */
} cognitive_batch_result_t;

/*=============================================================================
 * API
 *=============================================================================*/

/**
 * @brief Train all cognitive subsystems in parallel using actor pattern
 *
 * Dispatches 12 independent training modules to the brain's thread pool.
 * Each module runs as an actor: reads brain state, trains own weights,
 * returns result via promise. Coordinator collects results and updates
 * cognitive_stats single-threaded after all modules complete.
 *
 * Falls back to sequential execution if brain->inference_pool is NULL.
 *
 * @param brain     Internal brain handle
 * @param features  Input features from current training example
 * @param num_features Feature count
 * @param target    Target output
 * @param target_size Target size
 * @param label     Text label (for language/knowledge modules)
 * @param loss      Current training loss (for feedback-based modules)
 * @return Batch result with per-module timing and loss values
 */
cognitive_batch_result_t brain_train_cognitive_parallel(
    brain_t brain,
    const float* features,
    uint32_t num_features,
    const float* target,
    uint32_t target_size,
    const char* label,
    float loss);

/**
 * @brief Sequential fallback for cognitive training (original implementation)
 */
void brain_train_cognitive_subsystems_sequential(
    brain_t brain,
    const float* features,
    uint32_t num_features,
    const float* target,
    uint32_t target_size,
    const char* label,
    float loss);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_COGNITIVE_DISPATCH_H */
