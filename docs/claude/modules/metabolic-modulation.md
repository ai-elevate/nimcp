# Shared Metabolic Modulation Utilities (Dec 2024)

Shared utilities for ATP/fatigue-based modulation across substrate bridges.

## Purpose

Eliminates code duplication from 80+ substrate bridge files that had identical:
- `clamp_f()` utility functions
- Metabolic effect computation logic
- Magic number multipliers (1.0, 1.1, 0.9, etc.)

## Files

| File | Description |
|------|-------------|
| `include/cognitive/common/nimcp_metabolic_modulation.h` | Header with types and inline utilities |
| `src/cognitive/common/nimcp_metabolic_modulation.c` | Implementation of effect computation |

## Bridges Using This API

| Bridge | File | Custom Multipliers |
|--------|------|-------------------|
| Grief | `src/cognitive/grief/nimcp_grief_substrate_bridge.c` | Default |
| Joy | `src/cognitive/joy/nimcp_joy_substrate_bridge.c` | Yes (0.95 fatigue) |
| Knowledge | `src/cognitive/knowledge/nimcp_knowledge_substrate_bridge.c` | TBD |
| Personality | `src/cognitive/personality/nimcp_personality_substrate_bridge.c` | TBD |
| Salience | `src/cognitive/salience/nimcp_salience_substrate_bridge.c` | TBD |

## Core Types

### `metabolic_effect_multipliers_t`

Configures the effect computation multipliers:

```c
typedef struct metabolic_effect_multipliers {
    float atp_primary_mult;       // Default: 1.0
    float atp_secondary_mult;     // Default: 1.1
    float fatigue_primary_mult;   // Default: 1.0
    float fatigue_secondary_mult; // Default: 0.9
} metabolic_effect_multipliers_t;
```

### `metabolic_modulation_config_t`

Full configuration for metabolic modulation:

```c
typedef struct metabolic_modulation_config {
    bool enable_atp_modulation;      // Enable ATP-based effects
    bool enable_fatigue_modulation;  // Enable fatigue-based effects
    bool enable_bio_async;           // Enable bio-async messaging
    float atp_sensitivity;           // ATP sensitivity (default 1.0)
    float fatigue_sensitivity;       // Fatigue sensitivity (default 1.0)
    float min_capacity;              // Minimum capacity floor (default 0.2)
    metabolic_effect_multipliers_t multipliers;
} metabolic_modulation_config_t;
```

### `metabolic_effects_t`

Generic output effects structure:

```c
typedef struct metabolic_effects {
    float primary_atp;       // Primary ATP-modulated effect
    float secondary_atp;     // Secondary ATP-modulated effect
    float primary_fatigue;   // Primary fatigue-modulated effect
    float secondary_fatigue; // Secondary fatigue-modulated effect
    float overall_capacity;  // Average of all 4 effects
} metabolic_effects_t;
```

### `metabolic_input_t`

Input state from substrate:

```c
typedef struct metabolic_input {
    float atp_level;          // Current ATP level [0, 1]
    float metabolic_capacity; // Current metabolic capacity [0, 1]
} metabolic_input_t;
```

## Key Functions

### Utility Clamp Functions

```c
// Inline clamp utilities (replace static clamp_f in each file)
float nimcp_clamp_f(float value, float min_val, float max_val);
double nimcp_clamp_d(double value, double min_val, double max_val);
int nimcp_clamp_i(int value, int min_val, int max_val);
```

### Configuration Functions

```c
// Get default configuration
metabolic_modulation_config_t metabolic_modulation_default_config(void);

// Get default multipliers
metabolic_effect_multipliers_t metabolic_default_multipliers(void);

// Build config from individual fields (for bridge-specific configs)
metabolic_modulation_config_t metabolic_config_from_fields(
    bool enable_atp,
    bool enable_fatigue,
    bool enable_bio_async,
    float atp_sensitivity,
    float fatigue_sensitivity,
    float min_capacity,
    const metabolic_effect_multipliers_t* multipliers  // NULL for defaults
);
```

### Effect Computation Functions

```c
// Main computation function
int metabolic_compute_effects(
    const metabolic_input_t* input,
    const metabolic_modulation_config_t* config,
    metabolic_effects_t* effects
);  // Returns 0 on success, -1 on error

// Individual effect computation
float metabolic_compute_atp_effect(
    float atp_level,
    float sensitivity,
    float effect_multiplier,
    float min_capacity
);

float metabolic_compute_fatigue_effect(
    float metabolic_capacity,
    float sensitivity,
    float effect_multiplier,
    float min_capacity
);

// Initialize effects to full (1.0)
void metabolic_effects_init_full(metabolic_effects_t* effects);

// Compute overall capacity from effects
float metabolic_compute_overall_capacity(const metabolic_effects_t* effects);
```

## Usage Pattern

### Bridge Creation (with default multipliers)

```c
#include "cognitive/common/nimcp_metabolic_modulation.h"

struct my_substrate_bridge {
    my_substrate_config_t config;
    metabolic_modulation_config_t metabolic_config;  // Add this
    my_substrate_effects_t effects;
    // ...
};

my_bridge_t* my_bridge_create(..., const my_config_t* config) {
    // Initialize shared metabolic config from bridge-specific config
    bridge->metabolic_config = metabolic_config_from_fields(
        bridge->config.enable_atp_modulation,
        bridge->config.enable_fatigue_modulation,
        bridge->config.enable_bio_async,
        bridge->config.atp_sensitivity,
        bridge->config.fatigue_sensitivity,
        bridge->config.min_capacity,
        NULL  // Use default multipliers
    );
}
```

### Bridge Creation (with custom multipliers)

```c
my_bridge_t* my_bridge_create(...) {
    // Custom multipliers for this cognitive function
    metabolic_effect_multipliers_t custom_mult = {
        .atp_primary_mult = 1.0f,
        .atp_secondary_mult = 1.1f,
        .fatigue_primary_mult = 1.0f,
        .fatigue_secondary_mult = 0.95f  // Custom value
    };

    bridge->metabolic_config = metabolic_config_from_fields(
        ...,
        &custom_mult  // Pass custom multipliers
    );
}
```

### Bridge Update

```c
int my_bridge_update(my_bridge_t* bridge) {
    // Get metabolic state from substrate
    substrate_metabolic_state_t metabolic;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);

    // Prepare input
    metabolic_input_t input = {
        .atp_level = metabolic.atp_level,
        .metabolic_capacity = metabolic.metabolic_capacity
    };

    // Compute effects
    metabolic_effects_t generic_effects;
    metabolic_effects_init_full(&generic_effects);

    if (metabolic_compute_effects(&input, &bridge->metabolic_config, &generic_effects) == 0) {
        // Map generic effects to bridge-specific named effects
        bridge->effects.my_effect_1 = generic_effects.primary_atp;
        bridge->effects.my_effect_2 = generic_effects.secondary_atp;
        bridge->effects.my_effect_3 = generic_effects.primary_fatigue;
        bridge->effects.my_effect_4 = generic_effects.secondary_fatigue;
        bridge->effects.overall_capacity = generic_effects.overall_capacity;
    }
    return 0;
}
```

## Effect Mapping Convention

Generic effects map to cognitive-domain-specific effects:

| Generic Effect | Typical Meaning | Example (Joy) | Example (Grief) |
|----------------|-----------------|---------------|-----------------|
| `primary_atp` | Main ATP-dependent capacity | hedonic_capacity | processing_capacity |
| `secondary_atp` | Secondary ATP effect (1.1x) | joy_intensity | resilience_level |
| `primary_fatigue` | Main fatigue-dependent capacity | savoring_ability | emotion_regulation |
| `secondary_fatigue` | Secondary fatigue effect (0.9x) | positive_anticipation | adaptation_rate |
| `overall_capacity` | Average of all 4 | overall_capacity | overall_capacity |

## Default Values

| Parameter | Default | Notes |
|-----------|---------|-------|
| `enable_atp_modulation` | `true` | |
| `enable_fatigue_modulation` | `true` | |
| `enable_bio_async` | `false` | |
| `atp_sensitivity` | `1.0` | |
| `fatigue_sensitivity` | `1.0` | |
| `min_capacity` | `0.2` | Floor for all effects |
| `atp_primary_mult` | `1.0` | |
| `atp_secondary_mult` | `1.1` | 10% boost |
| `fatigue_primary_mult` | `1.0` | |
| `fatigue_secondary_mult` | `0.9` | 10% reduction |

## Migration Guide

To migrate an existing substrate bridge to use shared utilities:

1. Add include: `#include "cognitive/common/nimcp_metabolic_modulation.h"`

2. Add to bridge struct: `metabolic_modulation_config_t metabolic_config;`

3. In `*_create()`: Initialize using `metabolic_config_from_fields()`

4. In `*_update()`:
   - Remove local `clamp_f()` function
   - Replace manual ATP/fatigue calculation with `metabolic_compute_effects()`
   - Map generic effects to bridge-specific effect names

5. Remove duplicated code:
   - Static `clamp_f()` function
   - Manual effect computation loops
   - Magic number multipliers

## GOTCHAs

- `metabolic_compute_effects()` returns `0` for success, `-1` for errors (not NIMCP_OK/NIMCP_ERROR_*)
- Always call `metabolic_effects_init_full()` before `metabolic_compute_effects()` to ensure defaults
- Effects are clamped to `[min_capacity, 1.0]`, not `[0, 1]`
- Pass `NULL` for multipliers parameter to use defaults (not an empty struct)

---

## Tensor-Based API (v2.7.0)

For batch processing across multiple brain regions, use the tensor-based API which provides SIMD-accelerated operations via `nimcp_tensor_t`.

### Tensor Types

#### `metabolic_effect_index_t`

Indices for accessing effect values in tensors:

```c
typedef enum {
    METABOLIC_IDX_PRIMARY_ATP = 0,
    METABOLIC_IDX_SECONDARY_ATP = 1,
    METABOLIC_IDX_PRIMARY_FATIGUE = 2,
    METABOLIC_IDX_SECONDARY_FATIGUE = 3,
    METABOLIC_IDX_OVERALL_CAPACITY = 4,
    METABOLIC_EFFECT_COUNT = 5
} metabolic_effect_index_t;
```

#### `metabolic_effects_tensor_t`

Tensor-based effects for batch processing:

```c
typedef struct metabolic_effects_tensor {
    nimcp_tensor_t* effects;  // Shape: [batch, 5] or [5]
    uint32_t batch_size;      // Number of regions
    bool owns_tensor;         // True if struct owns tensor memory
} metabolic_effects_tensor_t;
```

#### `metabolic_input_tensor_t`

Tensor-based input for batch computation:

```c
typedef struct metabolic_input_tensor {
    nimcp_tensor_t* atp_levels;         // Shape: [batch]
    nimcp_tensor_t* metabolic_capacities; // Shape: [batch]
    uint32_t batch_size;
} metabolic_input_tensor_t;
```

#### `metabolic_multipliers_tensor_t`

Per-region multipliers:

```c
typedef struct metabolic_multipliers_tensor {
    nimcp_tensor_t* multipliers;  // Shape: [batch, 4] or [4]
    uint32_t batch_size;
} metabolic_multipliers_tensor_t;
```

### Tensor Functions

#### Creation/Destruction

```c
// Create structures
metabolic_effects_tensor_t* metabolic_effects_tensor_create(uint32_t batch_size);
metabolic_input_tensor_t* metabolic_input_tensor_create(uint32_t batch_size);
metabolic_multipliers_tensor_t* metabolic_multipliers_tensor_create(uint32_t batch_size);

// Destroy structures (NULL-safe)
void metabolic_effects_tensor_destroy(metabolic_effects_tensor_t* effects);
void metabolic_input_tensor_destroy(metabolic_input_tensor_t* input);
void metabolic_multipliers_tensor_destroy(metabolic_multipliers_tensor_t* mult);
```

#### Initialization

```c
// Initialize effects to 1.0
int metabolic_effects_tensor_init_full(metabolic_effects_tensor_t* effects);

// Initialize multipliers to defaults [1.0, 1.1, 1.0, 0.9]
int metabolic_multipliers_tensor_init_default(metabolic_multipliers_tensor_t* mult);

// Set multipliers for specific region
int metabolic_multipliers_tensor_set_region(
    metabolic_multipliers_tensor_t* mult,
    uint32_t region_idx,
    float atp_primary,
    float atp_secondary,
    float fatigue_primary,
    float fatigue_secondary
);
```

#### Computation

```c
// Batch compute effects for all regions
int metabolic_compute_effects_tensor(
    const metabolic_input_tensor_t* input,
    const metabolic_modulation_config_t* config,
    const metabolic_multipliers_tensor_t* multipliers,  // NULL for defaults
    metabolic_effects_tensor_t* effects
);

// Compute overall capacity (mean of first 4 effects)
int metabolic_compute_overall_capacity_tensor(metabolic_effects_tensor_t* effects);
```

#### Tensor/Scalar Conversion

```c
// Extract single region to scalar struct
int metabolic_effects_tensor_to_scalar(
    const metabolic_effects_tensor_t* tensor_effects,
    uint32_t region_idx,
    metabolic_effects_t* scalar_effects
);

// Copy scalar to tensor region
int metabolic_effects_scalar_to_tensor(
    metabolic_effects_tensor_t* tensor_effects,
    uint32_t region_idx,
    const metabolic_effects_t* scalar_effects
);
```

#### Element Access

```c
// Get/set individual effect values
float metabolic_effects_tensor_get(
    const metabolic_effects_tensor_t* effects,
    uint32_t region_idx,
    metabolic_effect_index_t effect_idx
);

int metabolic_effects_tensor_set(
    metabolic_effects_tensor_t* effects,
    uint32_t region_idx,
    metabolic_effect_index_t effect_idx,
    float value
);

// Convenience: create input from scalar values
metabolic_input_tensor_t* metabolic_input_tensor_from_scalar(
    float atp_level,
    float metabolic_capacity
);

// Set input for specific region
int metabolic_input_tensor_set_region(
    metabolic_input_tensor_t* input,
    uint32_t region_idx,
    float atp_level,
    float metabolic_capacity
);
```

### Tensor Usage Pattern

#### Batch Processing Multiple Brain Regions

```c
#include "cognitive/common/nimcp_metabolic_modulation.h"

// Example: Process 8 brain regions in batch
int process_brain_regions(brain_t* brain) {
    const uint32_t NUM_REGIONS = 8;

    // Create tensor structures
    metabolic_input_tensor_t* input = metabolic_input_tensor_create(NUM_REGIONS);
    metabolic_effects_tensor_t* effects = metabolic_effects_tensor_create(NUM_REGIONS);
    metabolic_multipliers_tensor_t* mult = metabolic_multipliers_tensor_create(NUM_REGIONS);

    if (!input || !effects || !mult) {
        metabolic_input_tensor_destroy(input);
        metabolic_effects_tensor_destroy(effects);
        metabolic_multipliers_tensor_destroy(mult);
        return -1;
    }

    // Initialize multipliers with defaults, then customize specific regions
    metabolic_multipliers_tensor_init_default(mult);
    metabolic_multipliers_tensor_set_region(mult, 3, 1.0f, 1.15f, 1.0f, 0.85f);  // Region 3: custom

    // Populate input from each region
    for (uint32_t i = 0; i < NUM_REGIONS; i++) {
        brain_region_t* region = brain_get_region(brain, i);
        metabolic_input_tensor_set_region(input, i,
            region->atp_level,
            region->metabolic_capacity
        );
    }

    // Batch compute all effects (SIMD-accelerated)
    metabolic_modulation_config_t config = metabolic_modulation_default_config();
    metabolic_compute_effects_tensor(input, &config, mult, effects);

    // Extract results for each region
    for (uint32_t i = 0; i < NUM_REGIONS; i++) {
        brain_region_t* region = brain_get_region(brain, i);
        metabolic_effects_t scalar;
        metabolic_effects_tensor_to_scalar(effects, i, &scalar);

        // Apply effects to region
        region->processing_capacity = scalar.overall_capacity;
    }

    // Cleanup
    metabolic_input_tensor_destroy(input);
    metabolic_effects_tensor_destroy(effects);
    metabolic_multipliers_tensor_destroy(mult);

    return 0;
}
```

#### Single-Region with Tensor API (Migration Path)

```c
// Easy migration from scalar to tensor API
int my_bridge_update_tensor(my_bridge_t* bridge) {
    // Create single-region input from scalar values
    metabolic_input_tensor_t* input = metabolic_input_tensor_from_scalar(
        metabolic.atp_level,
        metabolic.metabolic_capacity
    );

    metabolic_effects_tensor_t* effects = metabolic_effects_tensor_create(1);

    if (!input || !effects) {
        metabolic_input_tensor_destroy(input);
        metabolic_effects_tensor_destroy(effects);
        return -1;
    }

    // Compute using tensor API
    metabolic_compute_effects_tensor(input, &bridge->metabolic_config, NULL, effects);

    // Extract to scalar for existing code
    metabolic_effects_t scalar;
    metabolic_effects_tensor_to_scalar(effects, 0, &scalar);

    bridge->effects.my_effect_1 = scalar.primary_atp;
    bridge->effects.my_effect_2 = scalar.secondary_atp;
    // ...

    metabolic_input_tensor_destroy(input);
    metabolic_effects_tensor_destroy(effects);

    return 0;
}
```

### Tensor API GOTCHAs

- All tensor functions return `0` for success, `-1` for errors (consistent with scalar API)
- `metabolic_effects_tensor_get()` returns `NAN` on error
- Batch size must match between input and effects tensors
- When `batch_size == 1`, tensors use rank-1 shape `[5]` instead of `[1, 5]`
- Pass `NULL` for multipliers to use defaults across all regions
- Always destroy tensor structures after use to avoid memory leaks
