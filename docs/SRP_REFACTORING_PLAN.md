# SRP Refactoring Plan for nimcp_spatial_neuromod.c

## Current State
- **File size**: 1552 lines
- **Functions**: 32 total
- **SRP violations**: 8 functions >50 lines
- **Longest function**: 157 lines (spatial_neuromod_update)

## Goals
1. All functions <50 lines
2. Single Responsibility Principle
3. High cohesion, low coupling
4. Maintain backward compatibility
5. No breaking changes to public API

## Refactoring Strategy

### 1. spatial_neuromod_update() [157 lines → 3 functions]

**Current responsibilities:**
- Classical diffusion
- Quantum-Shannon diffusion
- Concentration updates
- Statistics tracking

**Refactored:**
```c
// Helper: Apply classical diffusion (< 30 lines)
static bool apply_classical_diffusion(
    spatial_neuromod_field_t* field,
    neural_network_t network,
    float dt);

// Helper: Apply quantum-Shannon diffusion (< 30 lines)
static bool apply_quantum_shannon_diffusion(
    spatial_neuromod_field_t* field,
    float dt);

// Main: Orchestrate update (< 40 lines)
bool spatial_neuromod_update(
    spatial_neuromod_field_t* field,
    neural_network_t network,
    float dt);
```

### 2. spatial_neuromod_select_pareto_optimal() [144 lines → 4 functions]

**Current responsibilities:**
- Score all neurons
- Find dominated solutions
- Count Pareto front
- Select K from front

**Refactored:**
```c
// Helper: Score all neurons (< 30 lines)
static bool score_all_neurons_multi_objective(
    const spatial_neuromod_field_t* field,
    neural_network_t network,
    const spatial_neuromod_config_t* config,
    float* all_scores);

// Helper: Find Pareto front (< 40 lines)
static uint32_t find_pareto_front(
    const float* all_scores,
    uint32_t num_neurons,
    uint32_t num_objectives,
    float epsilon,
    bool* is_dominated);

// Helper: Select K from front using weights (< 40 lines)
static bool select_from_pareto_front(
    const float* all_scores,
    const bool* is_dominated,
    uint32_t num_neurons,
    uint32_t front_size,
    const spatial_neuromod_config_t* config,
    uint32_t* selected_ids,
    uint32_t* num_selected);

// Main: Orchestrate Pareto selection (< 30 lines)
bool spatial_neuromod_select_pareto_optimal(...);
```

### 3. spatial_neuromod_select_optimal_sources() [69 lines → 3 functions]

**Current responsibilities:**
- Score all neurons
- Sort by score
- Select top K

**Refactored:**
```c
// Helper: Score and collect (< 30 lines)
static bool score_and_collect_neurons(
    const spatial_neuromod_field_t* field,
    neural_network_t network,
    const spatial_neuromod_config_t* config,
    neuron_score_t* scores,
    uint32_t num_neurons);

// Helper: Select top K (< 25 lines)
static uint32_t select_top_k_neurons(
    const neuron_score_t* scores,
    uint32_t num_neurons,
    uint32_t K,
    float min_score,
    uint32_t* selected_ids,
    float* selected_scores);

// Main: Orchestrate selection (< 30 lines)
bool spatial_neuromod_select_optimal_sources(...);
```

### 4. spatial_neuromod_update_dynamic_adaptation() [76 lines → 3 functions]

**Current responsibilities:**
- Update EMA
- Check cooldown
- Adapt K

**Refactored:**
```c
// Helper: Update EMA (< 20 lines)
static void update_efficiency_ema(
    spatial_neuromod_field_t* field,
    const spatial_neuromod_config_t* config);

// Helper: Attempt K adaptation (< 30 lines)
static bool attempt_k_adaptation(
    spatial_neuromod_field_t* field,
    const spatial_neuromod_config_t* config);

// Main: Orchestrate dynamic adaptation (< 35 lines)
bool spatial_neuromod_update_dynamic_adaptation(...);
```

### 5. Other Functions

**spatial_neuromod_release_adaptive() [67 lines]** - Already close to limit, minor cleanup
**spatial_neuromod_compute_laplacian() [63 lines]** - Already close, acceptable
**spatial_neuromod_score_neuron() [62 lines]** - Already close, acceptable
**spatial_neuromod_score_neuron_multi_objective() [60 lines]** - Already close, acceptable

## Implementation Order

1. ✅ Identify violations (DONE)
2. Create helper functions for spatial_neuromod_update()
3. Create helper functions for spatial_neuromod_select_pareto_optimal()
4. Create helper functions for spatial_neuromod_select_optimal_sources()
5. Create helper functions for spatial_neuromod_update_dynamic_adaptation()
6. Test all changes
7. Verify backward compatibility

## Success Criteria

- ✅ All functions <50 lines
- ✅ Each function has single responsibility
- ✅ All tests pass (no regressions)
- ✅ No breaking changes to public API
- ✅ Clean build (zero warnings)

## Notes

- All helper functions are `static` (internal only)
- Public API remains unchanged
- Performance should be identical (inlining)
- Code readability improved
