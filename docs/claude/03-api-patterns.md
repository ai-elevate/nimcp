# Key API Patterns

## Tensor Library (`include/utils/tensor/nimcp_tensor.h`)

```c
nimcp_tensor_t* t = nimcp_tensor_create(dims, ndims);
nimcp_tensor_t* sum_t = nimcp_tensor_sum(t);           // Returns tensor*, NOT scalar
double val = nimcp_tensor_get_flat(sum_t, 0);          // Extract scalar from tensor
double norm = nimcp_tensor_norm_p(t, 2.0);             // Returns double directly
nimcp_tensor_destroy(t);
```

**GOTCHA**: `nimcp_tensor_sum()` returns `nimcp_tensor_t*`, must extract with `nimcp_tensor_get_flat()`

**GOTCHA**: `nimcp_tensor_create` requires 3 args: `(dims, ndims, NIMCP_DTYPE_F32)`. The dtype parameter is mandatory.

---

## Positional Encoding (`include/utils/encoding/nimcp_positional_encoding.h`)

- Types: `NIMCP_POS_SINUSOIDAL`, `NIMCP_POS_LEARNED`, `NIMCP_POS_ROTARY`, `NIMCP_POS_ALIBI`, `NIMCP_POS_RELATIVE`
- API: `nimcp_pos_encoder_create()`, `nimcp_pos_encode_position()`, `nimcp_pos_rope_apply()`, `nimcp_pos_alibi_get_bias()`

---

## Mutex API Pattern (IMPORTANT)

```c
// Correct: Allocate + init
bridge->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
nimcp_mutex_init(bridge->mutex, NULL);

// Correct: Destroy + free
nimcp_mutex_destroy(bridge->mutex);
nimcp_free(bridge->mutex);

// WRONG: nimcp_mutex_create() does not exist
```

---

## Logging Macros

```c
// Correct: NIMCP_LOGGING_* macros
NIMCP_LOGGING_ERROR("message");
NIMCP_LOGGING_WARN("message");
NIMCP_LOGGING_INFO("message");
NIMCP_LOGGING_DEBUG("message");

// Wrong: nimcp_log_* (doesn't exist)
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

## Cytokine Enum Naming

```c
// Base cytokine types (from nimcp_swarm_immune.h)
CYTOKINE_IL1B, CYTOKINE_IL6, CYTOKINE_IL10, CYTOKINE_TNFA

// Brain-specific wrapper (from nimcp_brain_immune.h)
BRAIN_CYTOKINE_IL1 = CYTOKINE_IL1B    // Use BRAIN_CYTOKINE_IL1 in code
BRAIN_CYTOKINE_IL6 = CYTOKINE_IL6
BRAIN_CYTOKINE_IL10 = CYTOKINE_IL10
BRAIN_CYTOKINE_TNF = CYTOKINE_TNFA
BRAIN_CYTOKINE_IFN_GAMMA = 5          // Brain-specific (quarantine)
BRAIN_CYTOKINE_COUNT = 6

// Module-specific constants use CYTOKINE_IL1_ prefix (not IL1B):
CYTOKINE_IL1_ATTENTION_IMPACT, CYTOKINE_IL1_LTP_IMPAIRMENT, etc.
CYTOKINE_IFN_GAMMA_* (not BRAIN_CYTOKINE_IFN_GAMMA_*)
```

---

## Platform Tiers

**GOTCHA**: Platform tiers use `PLATFORM_TIER_FULL/MEDIUM/CONSTRAINED/MINIMAL`, not `PLATFORM_TIER_0/1/2/3`.

---

## Brain API

**GOTCHA**: `brain_t` creation is heavyweight (initializes 50+ subsystems). Hemisphere tests are resource-intensive.

**GOTCHA**: Use `brain_decide()` for inference, `brain_apply_reward_learning()` for training. Simple `brain_update/infer/train` don't exist.
