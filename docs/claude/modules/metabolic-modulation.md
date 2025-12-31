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
