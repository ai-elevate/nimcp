# Key API Patterns

## Tensor Library (`include/utils/tensor/nimcp_tensor.h`)

```c
// Create: 3 args — dims array, rank (not ndims), dtype
nimcp_tensor_t* t = nimcp_tensor_create(dims, rank, NIMCP_DTYPE_F32);

// Sum returns tensor*, NOT scalar
nimcp_tensor_t* sum_t = nimcp_tensor_sum(t);
double val = nimcp_tensor_get_flat(sum_t, 0);    // Extract scalar from tensor

// Norms and in-place ops
double norm = nimcp_tensor_norm_p(t, 2.0);       // Returns double directly
nimcp_tensor_mul_scalar_(t, 0.5f);               // In-place multiplication

nimcp_tensor_destroy(t);
```

**GOTCHA**: `nimcp_tensor_create` requires 3 args: `(dims, rank, NIMCP_DTYPE_F32)`. The dtype parameter is mandatory.

**GOTCHA**: `nimcp_tensor_sum()` returns `nimcp_tensor_t*`, must extract with `nimcp_tensor_get_flat()`

**GOTCHA**: `op_div` uses epsilon clamping (1e-7) instead of returning 0. No LOG_WARN emitted.

---

## Mutex API Pattern (IMPORTANT)

```c
// Option 1: Allocate + init (heap)
nimcp_mutex_t* m = nimcp_mutex_create(attr);  // Returns nimcp_mutex_t*, NOT error code
// ... use mutex ...
nimcp_mutex_free(m);  // destroy + free (CORRECT for heap-allocated mutexes)

// Option 2: Init existing struct
nimcp_mutex_t m;
nimcp_mutex_init(&m, NULL);
// ... use mutex ...
nimcp_mutex_destroy(&m);  // destroy only, no free
```

**GOTCHA**: `nimcp_mutex_create(attr)` returns `nimcp_mutex_t*`, NOT an error code.

**GOTCHA**: `nimcp_mutex_free()` = destroy + free. This IS correct for heap-allocated mutexes from `nimcp_mutex_create()`. This is NOT a bug.

**GOTCHA**: `mutex_attr_t` supports `MUTEX_TYPE_NORMAL`, `MUTEX_TYPE_RECURSIVE`, `MUTEX_TYPE_ERRORCHECK`.

**Deadlock prevention**: Never call public mutex-locking functions from within locked code. Create `*_unlocked()` helpers instead.

---

## Brain API

```c
// Public handle vs internal pointer
nimcp_brain_t* handle;          // Public handle
brain_t* brain = handle->internal_brain;  // Internal pointer

// Inference and training
brain_decide(brain, features, num_features);   // Inference
brain_apply_reward_learning(brain, reward);    // Training

// Decision lifecycle
copy_decision_deep(src);         // Deep copy for cache (not CoW)
brain_free_decision(decision);   // Cleanup (not nimcp_free())
```

**Brain init modes**: `brain_init_mode_t` enum — FULL / FAST / MINIMAL. FAST skips 60+ subsystems, runs 6 of 27 dependency waves. 1.5M neurons: ~14s (FAST) vs 10+ min (FULL). Python: `Brain(..., init_mode='fast')`.

---

## FEP Bridges

**GOTCHA**: FEP bridges return `0` for success, `-1` for errors. They do NOT use NIMCP_OK/NIMCP_ERROR_* codes.

**GOTCHA**: `metabolic_compute_effects()` also returns `0`/`-1`, not NIMCP error codes.

---

## Bio-Async API

**GOTCHA**: `nimcp_bio_promise_complete(promise, result)` takes 2 args, NOT 3.

---

## Brain Immune System

```c
// Thread-safe antigen access (PREFERRED)
brain_immune_get_antigen_copy(immune);  // Returns struct copy under mutex

// AVOID: returns pointer that may dangle
brain_immune_get_antigen(immune);
```

**GOTCHA**: B cells must be in PLASMA state to produce antibodies. State progression: NAIVE -> ACTIVATED -> PLASMA. Use `brain_immune_t_help_b()` to transition.

---

## GPU Stream Pool

```c
cudaStream_t stream = nimcp_gpu_get_pool_stream(ctx);  // Round-robin from 8 streams
nimcp_gpu_sync_pool(ctx);  // Sync all pool streams
```

CPU fallback stubs provided for non-CUDA builds.

---

## Synapse Metadata Pool

Chunked block allocator with 64K entries per block. Pointers are stable across growth (no realloc invalidation).

---

## Atomic Operations

**GOTCHA**: `_Atomic double` requires `-latomic` on GCC/Linux. NVCC does not support C11 `_Atomic` — use `volatile` + GCC `__atomic_*` builtins in headers shared with `.cu` files.

---

## Positional Encoding (`include/utils/encoding/nimcp_positional_encoding.h`)

- Types: `NIMCP_POS_SINUSOIDAL`, `NIMCP_POS_LEARNED`, `NIMCP_POS_ROTARY`, `NIMCP_POS_ALIBI`, `NIMCP_POS_RELATIVE`
- API: `nimcp_pos_encoder_create()`, `nimcp_pos_encode_position()`, `nimcp_pos_rope_apply()`, `nimcp_pos_alibi_get_bias()`

---

## Logging Macros

```c
NIMCP_LOGGING_ERROR("message");
NIMCP_LOGGING_WARN("message");
NIMCP_LOGGING_INFO("message");
NIMCP_LOGGING_DEBUG("message");
NIMCP_LOGGING_TRACE("message");  // Verbose, for hot paths
```

---

## Error Codes

```c
// Correct error codes
NIMCP_ERROR_NULL_POINTER
NIMCP_ERROR_INVALID_STATE
NIMCP_ERROR_INVALID_PARAMETER
NIMCP_ERROR_OPERATION_FAILED
NIMCP_ERROR_NO_MEMORY

// Wrong (don't use)
NIMCP_ERR_*, NIMCP_ERROR_INVALID_ARGUMENT, NIMCP_ERROR_RESOURCE_EXHAUSTED
```

---

## Platform Tiers

**GOTCHA**: Platform tiers use `PLATFORM_TIER_FULL/MEDIUM/CONSTRAINED/MINIMAL`, not `PLATFORM_TIER_0/1/2/3`.

---

## Cytokine Enum Naming

```c
// Base cytokine types (from nimcp_swarm_immune.h)
CYTOKINE_IL1B, CYTOKINE_IL6, CYTOKINE_IL10, CYTOKINE_TNFA

// Brain-specific wrapper (from nimcp_brain_immune.h)
BRAIN_CYTOKINE_IL1 = CYTOKINE_IL1B
BRAIN_CYTOKINE_IL6 = CYTOKINE_IL6
BRAIN_CYTOKINE_IL10 = CYTOKINE_IL10
BRAIN_CYTOKINE_TNF = CYTOKINE_TNFA
BRAIN_CYTOKINE_IFN_GAMMA = 5          // Brain-specific (quarantine)
BRAIN_CYTOKINE_COUNT = 6
```

---

## Metabolic Modulation API (`include/cognitive/common/nimcp_metabolic_modulation.h`)

```c
float nimcp_clamp_f(float value, float min_val, float max_val);

metabolic_modulation_config_t cfg = metabolic_modulation_default_config();

metabolic_input_t input = { .atp_level = atp, .metabolic_capacity = cap };
metabolic_effects_t effects;
metabolic_effects_init_full(&effects);
metabolic_compute_effects(&input, &cfg, &effects);  // Returns 0 on success, -1 on error
```

**GOTCHA**: Always call `metabolic_effects_init_full()` before `metabolic_compute_effects()` to ensure defaults.
