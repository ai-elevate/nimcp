# Resource Optimization (Tier-Based Memory Optimization)

NIMCP supports resource-constrained platforms through the Portia tier system. The tier optimization header provides compile-time and runtime optimizations.

## Platform Tiers

| Tier | Hardware | Memory Budget | Use Case |
|------|----------|---------------|----------|
| `PLATFORM_TIER_FULL` | ≥8 cores, ≥8GB RAM | 8GB | Research, training |
| `PLATFORM_TIER_MEDIUM` | ≥4 cores, ≥2GB RAM | 2GB | Development, edge AI |
| `PLATFORM_TIER_CONSTRAINED` | ≥2 cores, ≥256MB RAM | 256MB | Drones, IoT gateways |
| `PLATFORM_TIER_MINIMAL` | ≥1 core, ≥64MB RAM | 64MB | Sensors, MCUs |

## Building for Constrained Platforms

```bash
# Build for MINIMAL tier (64MB target)
cmake .. -DNIMCP_BUILD_TIER=PLATFORM_TIER_MINIMAL

# Build for CONSTRAINED tier (256MB target)
cmake .. -DNIMCP_BUILD_TIER=PLATFORM_TIER_CONSTRAINED
```

## Key Optimizations (~400-500KB savings on MINIMAL tier)

| Optimization | Header | Savings | Description |
|--------------|--------|---------|-------------|
| Bio-async inbox | `nimcp_tier_optimization.h` | 150KB | Tier-based inbox capacity (4-32 messages) |
| History buffers | `nimcp_tier_optimization.h` | 50KB | Tier-scaled history sizes (4-128 entries) |
| Mutex pooling | `nimcp_mutex_pool.h` | 5.6KB | Shared mutex pool for bridges |
| Fixed arrays | `nimcp_tier_optimization.h` | 100KB | Tier-based dendrite/synapse arrays |
| Statistics | `NIMCP_ENABLE_STATISTICS` | 10KB | Conditional statistics collection |

## Tier Optimization Constants

```c
#include "utils/platform/nimcp_tier_optimization.h"

// Bio-async inbox capacity (MINIMAL=4, FULL=32)
.inbox_capacity = NIMCP_BIO_INBOX_CAPACITY

// History buffer sizes
NIMCP_HISTORY_SIZE_SMALL   // MINIMAL=4, FULL=32
NIMCP_HISTORY_SIZE_MEDIUM  // MINIMAL=8, FULL=64
NIMCP_HISTORY_SIZE_LARGE   // MINIMAL=16, FULL=128

// Fixed array sizes
NIMCP_MAX_SPINE_IDS        // MINIMAL=8, FULL=64
NIMCP_MAX_SYNAPSE_IDS      // MINIMAL=32, FULL=256

// Feature flags
NIMCP_ENABLE_STATISTICS    // 0 on MINIMAL, 1 otherwise
NIMCP_ENABLE_DETAILED_HISTORY
NIMCP_USE_MUTEX_POOL       // 1 on MINIMAL for memory savings
```

## Mutex Pool (Bridge Memory Optimization)

```c
#include "utils/thread/nimcp_mutex_pool.h"

// Instead of per-bridge mutex allocation:
// bridge->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));

// Use the shared pool (on MINIMAL tier via macros):
NIMCP_BRIDGE_MUTEX_FIELD              // Declares mutex_slot or mutex*
NIMCP_BRIDGE_MUTEX_INIT(bridge, name) // Acquire slot or allocate
NIMCP_BRIDGE_MUTEX_LOCK(bridge)       // Lock
NIMCP_BRIDGE_MUTEX_UNLOCK(bridge)     // Unlock
NIMCP_BRIDGE_MUTEX_DESTROY(bridge)    // Release slot or free
```

## Tier-Aware Allocation Helpers

```c
// Scale buffer size by tier
size_t buf_size = nimcp_tier_scale_size(1024);  // MINIMAL=128, FULL=1024

// Scale array count by tier
uint32_t count = nimcp_tier_scale_count(256);   // MINIMAL=32, FULL=256

// Get tier-appropriate thread count
uint32_t threads = nimcp_tier_thread_count();   // MINIMAL=1, FULL=8

// Get memory budget in bytes
size_t budget = nimcp_tier_memory_budget_bytes();
```

## Files

- Header: `include/utils/platform/nimcp_tier_optimization.h`
- Mutex pool: `include/utils/thread/nimcp_mutex_pool.h`
- Implementation: `src/utils/thread/nimcp_mutex_pool.c`
