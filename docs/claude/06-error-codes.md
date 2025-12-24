# Error Code Strategy

NIMCP uses a **stratified error code system** with module-specific ranges. See `/home/bbrelin/nimcp/docs/ERROR_CODE_STRATEGY.md` for full documentation.

## Error Code Ranges

| Range | Purpose | Examples |
|-------|---------|----------|
| **-1 to -99** | **Module-local** | Tensor (-1 to -11), LNN (-1 to -15), Packet validation (-40 to -54) |
| **1000-9999** | **Core NIMCP** | INVALID_PARAMETER (1002), NO_MEMORY (2000), BRAIN_CREATION (3000) |
| **20000+** | **Module-specific** | Portia (20000+), Swarm (21000+), Security (22000+) |

## When to Use Each Range

| Use Case | Recommended Range | Reasoning |
|----------|-------------------|-----------|
| **Low-level utility** (tensor, LNN) | Negative (-1 to -99) | Fast, local error handling; no global collision risk |
| **High-level subsystem** (Portia, Swarm) | Positive (20000+) | Clear separation from core codes; module identity |
| **Core infrastructure** (memory, threading) | Core (1000-9999) | Centralized, cross-module concerns |

## Common Error Codes

```c
// Core codes (most common)
NIMCP_SUCCESS                  0     // Success
NIMCP_ERROR_INVALID_PARAMETER  1002  // Invalid argument
NIMCP_ERROR_NULL_POINTER       1003  // Unexpected NULL
NIMCP_ERROR_INVALID_STATE      1005  // Object state error
NIMCP_ERROR_NO_MEMORY          2000  // Allocation failed

// Tensor module
NIMCP_TENSOR_OK                0     // Success
NIMCP_TENSOR_ERR_NULL          -1    // NULL pointer
NIMCP_TENSOR_ERR_SHAPE         -2    // Shape mismatch
NIMCP_TENSOR_ERR_ALLOC         -4    // Allocation failed

// Portia module
NIMCP_PORTIA_ERROR_NOT_INITIALIZED      20001  // Not initialized
NIMCP_PORTIA_ERROR_TIER_LOCKED          20003  // Tier change blocked

// LNN module
LNN_ERROR_NONE = 0
LNN_ERROR_NULL_POINTER = -1
LNN_ERROR_INVALID_PARAM = -13
LNN_ERROR_OPERATION_FAILED = -12
LNN_ERROR_OUT_OF_MEMORY = -3
```

## Tensor Memory Safety

The tensor library provides strong memory safety guarantees. See `/home/bbrelin/nimcp/docs/TENSOR_MEMORY_SAFETY_FIXES.md` for details.

**Key guarantees**:
- `nimcp_tensor_destroy()` is **idempotent** (safe to call multiple times)
- `nimcp_tensor_destroy(NULL)` is a **no-op** (NULL-safe)
- Partial cleanup is safe (if creation fails partway)
- No double-free (NULL guards after freeing)
- Refcounting protects shared tensors

**Example**:
```c
nimcp_tensor_t* t = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
if (!t) {
    // Creation failed - check logs for NIMCP_TENSOR_ERR_RANK or NIMCP_TENSOR_ERR_ALLOC
    return;
}
// ... use tensor ...
nimcp_tensor_destroy(t);  // Always safe, even if creation was partial
```
