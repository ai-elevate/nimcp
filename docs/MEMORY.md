# NIMCP Memory Management and Monitoring

**Version:** 2.6.1
**Date:** 2025-11-04
**Status:** Production

## Overview

NIMCP provides a comprehensive memory management system with leak detection, usage tracking, and integration with the metrics collection system. This document covers memory allocation APIs, monitoring, debugging, and best practices.

## Table of Contents

- [Memory Allocation API](#memory-allocation-api)
- [Memory Tracking and Debugging](#memory-tracking-and-debugging)
- [Memory Metrics](#memory-metrics)
- [Memory Leak Detection](#memory-leak-detection)
- [Performance Considerations](#performance-considerations)
- [Best Practices](#best-practices)
- [Integration with Metrics System](#integration-with-metrics-system)

## Memory Allocation API

### Core Functions

NIMCP provides drop-in replacements for standard allocation functions with tracking and debugging capabilities:

```c
#include "utils/memory/nimcp_memory.h"

// Basic allocation
void* nimcp_malloc(size_t size);
void* nimcp_calloc(size_t count, size_t size);
void* nimcp_realloc(void* ptr, size_t new_size);
void nimcp_free(void* ptr);

// String duplication
char* nimcp_strdup(const char* str);

// Aligned allocation (for SIMD, cache alignment)
void* nimcp_aligned_malloc(size_t size, size_t alignment);
void nimcp_aligned_free(void* ptr);
```

### Basic Usage

```c
// Initialize memory tracking at startup
nimcp_memory_init();
nimcp_memory_enable_tracking(true);

// Allocate memory
int* numbers = nimcp_malloc(100 * sizeof(int));
if (!numbers) {
    // Handle allocation failure
    return;
}

// Use memory
for (int i = 0; i < 100; i++) {
    numbers[i] = i;
}

// Free memory
nimcp_free(numbers);

// Cleanup at shutdown
nimcp_memory_check_leaks();
nimcp_memory_cleanup();
```

### Aligned Memory Allocation

For performance-critical code using SIMD or requiring cache alignment:

```c
// Allocate 1024 bytes aligned to 64-byte boundary (cache line)
float* vector = nimcp_aligned_malloc(1024 * sizeof(float), 64);

// Use aligned memory for SIMD operations
// ... SIMD code ...

// Free aligned memory
nimcp_aligned_free(vector);
```

## Memory Tracking and Debugging

### Initialization

```c
// Initialize tracking system
nimcp_memory_init();

// Enable tracking (recommended for debug builds)
nimcp_memory_enable_tracking(true);

// Enable verbose debug output
nimcp_memory_enable_debug_output(true);
```

### Getting Statistics

```c
nimcp_memory_stats_t stats;
if (nimcp_memory_get_stats(&stats)) {
    printf("Total allocated: %zu bytes\n", stats.total_allocated);
    printf("Currently in use: %zu bytes\n", stats.current_allocated);
    printf("Peak usage: %zu bytes\n", stats.peak_allocated);
    printf("Allocations: %zu\n", stats.allocation_count);
    printf("Frees: %zu\n", stats.free_count);
    printf("Failed allocations: %zu\n", stats.failed_allocations);
}
```

### Memory Statistics Structure

```c
typedef struct {
    size_t total_allocated;    // Total bytes ever allocated
    size_t current_allocated;  // Current bytes in use
    size_t peak_allocated;     // Peak memory usage
    size_t allocation_count;   // Number of allocations
    size_t free_count;         // Number of frees
    size_t failed_allocations; // Failed allocation attempts
} nimcp_memory_stats_t;
```

### Debugging Functions

```c
// Dump all current allocations
nimcp_memory_dump_allocations();

// Check for memory leaks
nimcp_memory_check_leaks();

// Analyze allocation patterns
nimcp_memory_analyze_patterns();

// Clear statistics (for profiling specific sections)
nimcp_memory_clear_stats();
```

## Memory Metrics

### Brain-Level Memory Metrics

The brain probe API (`nimcp_brain_probe()`) includes memory usage:

```c
nimcp_brain_probe_t probe;
nimcp_brain_probe(brain, &probe);

printf("Brain memory usage: %.2f MB\n",
       probe.memory_bytes / (1024.0 * 1024.0));
```

See [BRAIN_PROBE.md](BRAIN_PROBE.md) for complete details.

### System-Level Memory Metrics

Use the metrics collector to track memory usage:

```c
#include "utils/metrics/nimcp_metrics.h"

nimcp_metrics_collector_t metrics = nimcp_metrics_create();

// Record memory metrics
nimcp_memory_stats_t stats;
nimcp_memory_get_stats(&stats);

nimcp_metrics_record_gauge(metrics, "memory.current_bytes",
                           stats.current_allocated,
                           NIMCP_METRIC_CATEGORY_MEMORY);

nimcp_metrics_record_gauge(metrics, "memory.peak_bytes",
                           stats.peak_allocated,
                           NIMCP_METRIC_CATEGORY_MEMORY);

nimcp_metrics_record_counter(metrics, "memory.allocations",
                             stats.allocation_count,
                             NIMCP_METRIC_CATEGORY_MEMORY);
```

### Standard Memory Metrics

| Metric Name | Type | Category | Description | Unit |
|-------------|------|----------|-------------|------|
| `memory.current_bytes` | Gauge | Memory | Current bytes allocated | bytes |
| `memory.peak_bytes` | Gauge | Memory | Peak memory usage | bytes |
| `memory.total_allocated` | Counter | Memory | Total bytes ever allocated | bytes |
| `memory.allocation_count` | Counter | Memory | Number of allocations | count |
| `memory.free_count` | Counter | Memory | Number of frees | count |
| `memory.failed_allocations` | Counter | Memory | Failed allocation attempts | count |
| `brain.memory_bytes` | Gauge | Memory | Per-brain memory usage | bytes |

See [METRICS_CATALOG.md](METRICS_CATALOG.md) for complete metrics catalog.

## Memory Leak Detection

### Detection Features

NIMCP's memory tracking system provides:

1. **Canary Guards**: Detect buffer overflows
2. **Allocation Tracking**: Track all allocations with file/line info
3. **Double-Free Detection**: Prevent crashes from double frees
4. **Leak Reporting**: Detailed reports of unfreed memory

### Checking for Leaks

```c
int main(void) {
    nimcp_memory_init();
    nimcp_memory_enable_tracking(true);

    // Application code
    run_application();

    // Check for leaks before exit
    nimcp_memory_check_leaks();

    nimcp_memory_cleanup();
    return 0;
}
```

### Leak Report Example

```
=== Memory Leak Report ===
Found 2 leaked allocations:

Leak #1: 1024 bytes
  Allocated at: src/cognitive/knowledge/nimcp_knowledge.c:145
  Address: 0x7f1234567000

Leak #2: 512 bytes
  Allocated at: src/core/brain/nimcp_brain.c:89
  Address: 0x7f1234568000

Total leaked: 1536 bytes
==========================
```

### Automated Leak Testing

The test suite includes memory leak tests:

```bash
# Run memory leak tests
cd build
make utility_tests
./src/tests/utility_tests --gtest_filter="*Memory*"

# Run with Valgrind
valgrind --leak-check=full --show-leak-kinds=all \
         ./src/tests/utility_tests
```

## Performance Considerations

### Overhead

When tracking is enabled:

- **Memory Overhead**: ~40 bytes per allocation (tracking metadata + canaries)
- **Time Overhead**: ~2-5% (mutex locks, linked list updates)
- **Thread Safety**: All operations are thread-safe via mutexes

### Build Configuration

**Debug Builds:**
```cmake
# Enable tracking in debug builds
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DNIMCP_MEMORY_TRACKING")
```

**Release Builds:**
```cmake
# Disable tracking in release builds for performance
# (tracking functions become no-ops)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -DNDEBUG")
```

### Conditional Tracking

Enable/disable tracking at runtime:

```c
#ifdef DEBUG
    nimcp_memory_enable_tracking(true);
    nimcp_memory_enable_debug_output(true);
#else
    nimcp_memory_enable_tracking(false);
#endif
```

## Best Practices

### 1. Initialize Early, Cleanup Late

```c
int main(int argc, char** argv) {
    // Initialize first
    nimcp_memory_init();
    nimcp_init();

    // Application code
    // ...

    // Cleanup in reverse order
    nimcp_shutdown();
    nimcp_memory_check_leaks();
    nimcp_memory_cleanup();

    return 0;
}
```

### 2. Use RAII Pattern in C

```c
// Define cleanup functions
void brain_destroy_auto(nimcp_brain_t* brain) {
    if (brain && *brain) {
        nimcp_brain_destroy(*brain);
    }
}

// Use with cleanup attribute (GCC/Clang)
void example(void) {
    __attribute__((cleanup(brain_destroy_auto)))
    nimcp_brain_t brain = nimcp_brain_create("test", ...);

    // Automatic cleanup on scope exit
}
```

### 3. Check Allocation Failures

```c
void* ptr = nimcp_malloc(large_size);
if (!ptr) {
    // Handle failure - log error, use fallback, etc.
    nimcp_log_error("Failed to allocate %zu bytes", large_size);
    return NIMCP_ERROR_OUT_OF_MEMORY;
}
```

### 4. Profile Memory Usage

```c
// Clear stats before profiling
nimcp_memory_clear_stats();

// Run code to profile
train_brain(brain, dataset);

// Get statistics
nimcp_memory_stats_t stats;
nimcp_memory_get_stats(&stats);
printf("Training used: %zu bytes\n", stats.current_allocated);
```

### 5. Use Aligned Allocation for Performance

```c
// For SIMD operations
float* simd_data = nimcp_aligned_malloc(count * sizeof(float), 32);

// For cache line alignment
struct cache_aligned* data = nimcp_aligned_malloc(
    sizeof(struct cache_aligned), 64
);
```

### 6. Integrate with Metrics System

```c
// Create periodic memory monitoring
void monitor_memory(nimcp_metrics_collector_t metrics) {
    nimcp_memory_stats_t stats;
    nimcp_memory_get_stats(&stats);

    nimcp_metrics_record_gauge(metrics, "memory.current_mb",
                               stats.current_allocated / (1024.0 * 1024.0),
                               NIMCP_METRIC_CATEGORY_MEMORY);

    nimcp_metrics_record_gauge(metrics, "memory.peak_mb",
                               stats.peak_allocated / (1024.0 * 1024.0),
                               NIMCP_METRIC_CATEGORY_MEMORY);
}
```

## Integration with Metrics System

### Complete Example

```c
#include "include/nimcp.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/metrics/nimcp_metrics.h"

int main(void) {
    // Initialize
    nimcp_memory_init();
    nimcp_memory_enable_tracking(true);
    nimcp_init();

    // Create metrics collector
    nimcp_metrics_collector_t metrics = nimcp_metrics_create();
    nimcp_metrics_set_directory(metrics, "./memory_metrics");

    // Create brain
    nimcp_brain_t brain = nimcp_brain_create("classifier",
                                               NIMCP_BRAIN_MEDIUM,
                                               NIMCP_TASK_CLASSIFICATION,
                                               100, 10);

    // Record memory metrics periodically
    for (int i = 0; i < 1000; i++) {
        // Training step
        nimcp_brain_learn_example(brain, features, 100, label, 0.9f);

        // Record metrics every 100 steps
        if (i % 100 == 0) {
            nimcp_memory_stats_t stats;
            nimcp_memory_get_stats(&stats);

            nimcp_metrics_record_gauge(metrics, "memory.current_mb",
                                       stats.current_allocated / (1024.0 * 1024.0),
                                       NIMCP_METRIC_CATEGORY_MEMORY);

            // Probe brain memory
            nimcp_brain_probe_t probe;
            nimcp_brain_probe(brain, &probe);
            nimcp_metrics_record_gauge(metrics, "brain.memory_mb",
                                       probe.memory_bytes / (1024.0 * 1024.0),
                                       NIMCP_METRIC_CATEGORY_MEMORY);
        }
    }

    // Export metrics
    nimcp_metrics_export_tableau_csv(metrics, "./memory_metrics.csv");
    nimcp_metrics_export_powerbi_json(metrics, "./memory_metrics.json");

    // Cleanup
    nimcp_brain_destroy(brain);
    nimcp_metrics_destroy(metrics);
    nimcp_shutdown();

    // Check for leaks
    nimcp_memory_check_leaks();
    nimcp_memory_cleanup();

    return 0;
}
```

### Visualization in Tableau/PowerBI

After exporting metrics, visualize in dashboards:

**Tableau:**
1. Import `memory_metrics.csv`
2. Create line chart with timestamp on X-axis
3. Plot `memory.current_mb` and `memory.peak_mb`
4. Add calculated field for memory growth rate

**PowerBI:**
1. Import `memory_metrics.json`
2. Create time series visualization
3. Add alert for memory threshold exceeded
4. Create memory leak detection visual (allocations - frees over time)

## Testing

### Unit Tests

```bash
# Run memory-specific tests
cd build
make utility_tests
./src/tests/utility_tests --gtest_filter="MemoryTest.*"
```

### Integration Tests

```bash
# Run full test suite with leak detection
./src/tests/integration_tests
```

### Valgrind Analysis

```bash
# Full leak check
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         ./examples/brain_probe_demo
```

## Troubleshooting

### High Memory Usage

1. Use `nimcp_memory_dump_allocations()` to find large allocations
2. Check `nimcp_memory_analyze_patterns()` for allocation hotspots
3. Profile with `nimcp_memory_get_stats()` periodically
4. Use brain probe to check per-brain memory usage

### Memory Leaks

1. Enable tracking: `nimcp_memory_enable_tracking(true)`
2. Run application
3. Call `nimcp_memory_check_leaks()` before exit
4. Review leak report with file/line numbers
5. Fix leaks and retest

### Performance Issues

1. Disable tracking in production: `nimcp_memory_enable_tracking(false)`
2. Use aligned allocation for SIMD code
3. Batch allocations to reduce overhead
4. Consider memory pools for frequently allocated objects

## See Also

- [BRAIN_PROBE.md](BRAIN_PROBE.md) - Brain state monitoring including memory
- [METRICS_IMPLEMENTATION.md](METRICS_IMPLEMENTATION.md) - Metrics collection system
- [METRICS_CATALOG.md](METRICS_CATALOG.md) - Complete metrics catalog
- [nimcp_memory.h](../src/utils/memory/nimcp_memory.h) - Memory API reference

## License

Part of NIMCP project. See main LICENSE file.
