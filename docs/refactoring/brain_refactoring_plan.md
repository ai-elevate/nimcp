# NIMCP Brain Refactoring Plan
**Version**: 1.0
**Date**: 2026-02-16
**Author**: Claude (Sonnet 4.5)
**Status**: PLAN - NOT YET EXECUTED

## Executive Summary

Refactor `/home/bbrelin/nimcp/src/core/brain/nimcp_brain.c` (6,150 lines, 12+ concerns) into 6 focused modules following the Single Responsibility Principle. Decompose `brain_struct` (100+ fields, 20+ concerns) into cohesive sub-structs.

**Effort Estimate**: 16-24 hours across 3-4 days
**Risk Level**: HIGH (core module, 472 tests depend on it)
**Approach**: Incremental with continuous test verification

---

## Phase 1: Analysis & Preparation (2-3 hours)

### 1.1 Analyze Function Distribution

Read `nimcp_brain.c` and categorize all functions:

```bash
cd /home/bbrelin/nimcp
grep -n "^[a-z_]*(" src/core/brain/nimcp_brain.c | head -100
```

**Categories**:
- **Lifecycle**: `brain_create`, `brain_destroy`, `allocate_brain`, `init_*_subsystem`
- **Inference**: `brain_process`, `brain_decide`, `brain_learn_example`
- **Training**: Training pipeline, callbacks, loss computation
- **Serialization**: `brain_save`, `brain_load`, `from_pretrained`
- **Features**: `brain_resize`, oscillations, COW cloning
- **State**: Accessors (`brain_get_*`), statistics

### 1.2 Map brain_struct Fields

Read `/home/bbrelin/nimcp/include/core/brain/nimcp_brain_internal.h` and categorize all 100+ fields:

```
NETWORK STATE (15-20 fields):
- adaptive_network_t network
- snn_network_t* snn_network
- lnn_network_t* lnn_network
- cnn_trainer_t* cnn_trainer
- active_network_type
- owns_specialized_network
- output_labels
- num_output_labels
- last_input
- cached_decision
- input_size
- cache_mutex
- etc.

TRAINING STATE (10-15 fields):
- training_ctx
- snn_training_ctx
- lnn_training_ctx
- enable_training_integration
- plasticity_bridge
- enable_plasticity_bridge
- event_driven_plasticity
- enable_event_driven_plasticity
- loss_history[10]
- loss_history_index
- loss_history_count
- base_learning_rate
- etc.

OSCILLATION STATE (5-10 fields):
- oscillations
- enable_cortical_columns
- hypercolumns
- num_hypercolumns
- feature_hypercolumns
- num_feature_hypercolumns
- etc.

IMMUNE STATE (3-5 fields):
- health_agent (if present)
- immune_system (if present)
- bbb_system
- bbb_enabled
- etc.

MEMORY STATE (10-15 fields):
- working_memory
- global_workspace
- engram_system
- systems_consolidation
- wm_transfer_system
- semantic_memory
- longterm_memory
- longterm_capacity
- longterm_count
- etc.

INTEGRATION STATE (20-30 fields):
- distributed
- bio_async_enabled
- visual_cortex
- audio_cortex
- speech_cortex
- multimodal
- nlp_network
- visual_feature_buffer
- audio_feature_buffer
- speech_feature_buffer
- integrated_feature_buffer
- multihead_attention
- brain_regions
- glial
- myelin_sheath
- axon_network
- dendrite_network
- neuromodulator_system
- pink_noise
- quantum_annealer
- quantum_shannon_diffusion
- cross_modal_graph
- etc.

COGNITIVE STATE (15-20 fields):
- introspection
- ethics
- salience
- consolidation
- curiosity
- knowledge
- logic
- symbolic_logic
- epistemic
- theory_of_mind
- explanation_gen
- meta_learner
- predictive_network
- mirror_neurons
- emotional_system
- executive
- sleep_system
- mental_health_monitor
- autobio
- self_model
- personality
- etc.

EMOTION STATE (5-10 fields):
- shadow_emotions
- bias_detection
- grief_system
- joy_system
- remorse_system
- social_bond_system
- empathy_network
- empathetic_response_engine
- etc.

COW/SNAPSHOT STATE (8-10 fields):
- is_cow_clone
- owns_network
- original_network
- network_is_cached
- network_refcount_atomic
- can_use_readonly
- is_snapshot
- snapshot_stats
- etc.

SECURITY STATE (5-8 fields):
- security_integration
- sec_module_id
- sec_region_ids
- num_sec_regions
- security_bridge
- bbb_system
- bbb_memory_region_id
- etc.

TOPOLOGY/ANALYSIS STATE (5-8 fields):
- functional_modules
- network_hubs
- topology_metrics
- auto_detect_communities
- community_detection_interval
- network_analyzer
- etc.

CONFIG/STATS/MISC (10-15 fields):
- config
- stats
- strategy
- current_time_us
- last_glial_update_us
- wellbeing_monitoring_enabled
- wellbeing_check_interval_ms
- last_wellbeing_check_time
- last_distress
- last_curiosity_drive
- last_novelty_score
- spike_feature_extractor
- population_analyzer
- event_bus
- etc.
```

### 1.3 Identify Dependencies

Map function call graphs:
- Which functions call which?
- Which functions modify which struct fields?
- Which functions are public API vs internal helpers?

---

## Phase 2: Create Decomposed Internal Header (3-4 hours)

### 2.1 Define Sub-Structs

Create `/home/bbrelin/nimcp/src/core/brain/nimcp_brain_internal_decomposed.h`:

```c
#ifndef NIMCP_BRAIN_INTERNAL_DECOMPOSED_H
#define NIMCP_BRAIN_INTERNAL_DECOMPOSED_H

#include "core/brain/nimcp_brain_internal.h"

//=============================================================================
// DECOMPOSED BRAIN STATE SUB-STRUCTURES
//=============================================================================

/**
 * @brief Network state - neural network topology, connections, activations
 *
 * RESPONSIBILITY: Manage neural network instances and their state
 */
typedef struct {
    // Core network
    adaptive_network_t network;

    // Specialized network implementations
    struct snn_network_s* snn_network;
    struct lnn_network_s* lnn_network;
    struct cnn_trainer_s* cnn_trainer;
    uint8_t active_network_type;
    bool owns_specialized_network;

    // Output labels
    char** output_labels;
    uint32_t num_output_labels;

    // Decision caching
    float* last_input;
    brain_decision_t* cached_decision;
    uint32_t input_size;
    nimcp_platform_mutex_t cache_mutex;

    // COW state
    bool is_cow_clone;
    bool owns_network;
    adaptive_network_t original_network;
    bool network_is_cached;
    _Atomic(uint32_t)* network_refcount_atomic;
    bool can_use_readonly;
    bool is_snapshot;
    brain_stats_t snapshot_stats;

} brain_network_state_t;

/**
 * @brief Training state - training context, optimizers, loss tracking
 *
 * RESPONSIBILITY: Coordinate training and learning
 */
typedef struct {
    // Training contexts
    nimcp_brain_training_ctx_t* training_ctx;
    struct snn_training_ctx_s* snn_training_ctx;
    struct lnn_training_ctx_s* lnn_training_ctx;
    bool enable_training_integration;

    // Plasticity bridge
    tpb_context_t* plasticity_bridge;
    bool enable_plasticity_bridge;

    // Event-driven plasticity
    edp_context_t* event_driven_plasticity;
    bool enable_event_driven_plasticity;

    // Adaptive learning rate
    float loss_history[10];
    uint32_t loss_history_index;
    uint32_t loss_history_count;
    float base_learning_rate;

    // Curiosity-driven modulation
    float last_curiosity_drive;
    float last_novelty_score;

    // Biological frameworks
    homeostatic_controller_t homeostatic;
    dendritic_tree_t dendritic;
    pc_hierarchy_t predictive_coding;
    second_messenger_system_t* second_messengers;
    bool enable_second_messengers;

} brain_training_state_t;

/**
 * @brief Oscillation state - brain waves, cortical columns, topographic maps
 *
 * RESPONSIBILITY: Manage oscillatory dynamics and columnar organization
 */
typedef struct {
    // Oscillation analysis
    brain_oscillation_analyzer_t* oscillations;

    // Cortical columns
    bool enable_cortical_columns;
    cortical_column_pool_t* cortical_column_pool;
    hypercolumn_t** hypercolumns;
    uint32_t num_hypercolumns;
    feature_hypercolumn_t** feature_hypercolumns;
    uint32_t num_feature_hypercolumns;
    orientation_hypercolumn_t** orientation_hypercolumns;
    uint32_t num_orientation_hypercolumns;

    // Topographic maps
    topographic_map_t* visual_topographic_map;
    topographic_map_t* auditory_topographic_map;
    topographic_map_t* somatosensory_topographic_map;

    // Laminar structure
    laminar_structure_t* laminar_system;
    columnar_connectivity_t* columnar_connectivity;

} brain_oscillation_state_t;

/**
 * @brief Immune state - health monitoring, blood-brain barrier, security
 *
 * RESPONSIBILITY: Protect brain from attacks and monitor health
 */
typedef struct {
    // Blood-brain barrier
    bbb_system_t* bbb_system;
    uint32_t bbb_memory_region_id;
    bool bbb_enabled;

    // Security integration
    nimcp_security_integration_t* security_integration;
    uint32_t sec_module_id;
    uint32_t* sec_region_ids;
    uint32_t num_sec_regions;
    nimcp_security_recovery_bridge_t* security_bridge;

} brain_immune_state_t;

/**
 * @brief Memory state - working memory, workspace, engrams, consolidation
 *
 * RESPONSIBILITY: Manage short-term and long-term memory systems
 */
typedef struct {
    // Working memory
    working_memory_t* working_memory;

    // Global workspace
    global_workspace_t* global_workspace;

    // Engram system
    engram_system_t* engram_system;
    systems_consolidation_system_t* systems_consolidation;
    wm_transfer_system_t* wm_transfer_system;
    semantic_memory_system_t* semantic_memory;
    consolidation_handle_t consolidation;

    // Long-term memory buffer
    struct {
        float* features;
        uint32_t num_features;
        float salience;
        uint64_t timestamp_ms;
    } *longterm_memory;
    uint32_t longterm_capacity;
    uint32_t longterm_count;

} brain_memory_state_t;

/**
 * @brief Integration state - multimodal, bio-async, distributed cognition
 *
 * RESPONSIBILITY: Integrate multiple brain subsystems and external interfaces
 */
typedef struct {
    // Distributed cognition
    distrib_cognition_t distributed;

    // Bio-async
    bool bio_async_enabled;

    // Multimodal processing
    visual_cortex_t visual_cortex;
    audio_cortex_t audio_cortex;
    speech_cortex_t speech_cortex;
    multimodal_integration_t multimodal;
    nlp_network_t nlp_network;
    float* visual_feature_buffer;
    float* audio_feature_buffer;
    float* speech_feature_buffer;
    float* integrated_feature_buffer;

    // Attention
    multihead_attention_t multihead_attention;

    // Brain regions
    struct brain_module_struct* brain_regions;

    // Biological subsystems
    glial_integration_t* glial;
    myelin_sheath_network_t* myelin_sheath;
    axon_network_t* axon_network;
    dendrite_network_t* dendrite_network;
    neuromodulator_system_t neuromodulator_system;
    neuromod_pink_noise_t* pink_noise;

    // Quantum/Shannon
    quantum_annealer_t quantum_annealer;
    quantum_shannon_diffusion_t* quantum_shannon_diffusion;
    cross_modal_routing_graph_t* cross_modal_graph;

    // Analysis
    spike_feature_extractor_t spike_feature_extractor;
    population_analyzer_t population_analyzer;
    network_analyzer_t* network_analyzer;

    // Topology
    community_structure_t* functional_modules;
    hub_structure_t* network_hubs;
    network_metrics_t* topology_metrics;
    bool auto_detect_communities;
    float community_detection_interval;

    // Event bus
    event_bus_t event_bus;

    // Memory pools
    memory_pool_t decision_struct_pool;
    memory_pool_t output_vector_pool;
    memory_pool_t active_neuron_ids_pool;

} brain_integration_state_t;

/**
 * @brief Cognitive state - high-level cognition, emotions, personality
 *
 * RESPONSIBILITY: Manage conscious and unconscious cognitive processes
 */
typedef struct {
    // Core cognition
    introspection_context_t introspection;
    ethics_engine_t ethics;
    salience_evaluator_t salience;
    curiosity_engine_t curiosity;
    knowledge_system_t knowledge;

    // Logic and reasoning
    neural_logic_network_t logic;
    symbolic_logic_t* symbolic_logic;
    epistemic_filter_t epistemic;

    // Theory of mind
    theory_of_mind_t theory_of_mind;
    explanation_generator_t explanation_gen;
    meta_learner_t meta_learner;
    predictive_network_t predictive_network;
    mirror_neurons_t mirror_neurons;

    // Emotional systems
    emotional_system_t* emotional_system;
    shadow_system_t shadow_emotions;
    bias_system_t bias_detection;
    grief_system_t grief_system;
    joy_system_t joy_system;
    remorse_regret_system_t remorse_system;
    social_bond_system_t social_bond_system;
    empathy_network_t empathy_network;
    void* empathetic_response_engine;

    // Executive functions
    executive_controller_t* executive;
    sleep_system_t sleep_system;
    mental_health_monitor_t* mental_health_monitor;
    struct mental_health_guardian* mental_health_guardian;

    // Self-awareness
    autobiographical_memory_t autobio;
    self_model_system_t self_model;
    personality_profile_t* personality;

    // Wellbeing
    distress_assessment_t last_distress;
    bool wellbeing_monitoring_enabled;
    uint64_t wellbeing_check_interval_ms;
    uint64_t last_wellbeing_check_time;

} brain_cognitive_state_t;

/**
 * @brief Complete brain structure with decomposed sub-structs
 *
 * MIGRATION PLAN:
 * 1. Create this new struct alongside existing brain_struct
 * 2. Add conversion functions (old -> new, new -> old)
 * 3. Gradually migrate modules to use new struct
 * 4. Remove old struct once all modules migrated
 */
typedef struct brain_struct_decomposed {
    // Core configuration and strategy
    brain_config_t config;
    task_strategy_t* strategy;
    brain_stats_t stats;

    // Decomposed state sub-structures
    brain_network_state_t network_state;
    brain_training_state_t training_state;
    brain_oscillation_state_t oscillation_state;
    brain_immune_state_t immune_state;
    brain_memory_state_t memory_state;
    brain_integration_state_t integration_state;
    brain_cognitive_state_t cognitive_state;

    // Timing (shared across all subsystems)
    uint64_t current_time_us;
    uint64_t last_glial_update_us;

} brain_struct_decomposed_t;

//=============================================================================
// INTERNAL FUNCTION PROTOTYPES
//=============================================================================

// Lifecycle
brain_t brain_allocate(void);
bool brain_init_subsystems(brain_t brain);
void brain_cleanup_subsystems(brain_t brain);

// Inference
brain_decision_t* brain_process_internal(brain_t brain, const float* features, uint32_t num_features);
brain_decision_t* brain_decide_internal(brain_t brain, const float* features, uint32_t num_features);
float brain_learn_example_internal(brain_t brain, const float* features, uint32_t num_features, const char* label, float confidence);

// Training
bool brain_training_begin(brain_t brain);
float brain_training_step(brain_t brain, const float* features, uint32_t num_features, const float* target, uint32_t num_outputs);
void brain_training_end(brain_t brain);

// Serialization
bool brain_save_internal(brain_t brain, const char* filepath);
brain_t brain_load_internal(const char* filepath);

// Features
bool brain_resize_internal(brain_t brain, uint32_t new_neuron_count);
bool brain_enable_oscillations_internal(brain_t brain);
brain_t brain_clone_cow_internal(brain_t original);

// State
void brain_update_stats(brain_t brain);
bool brain_ensure_writable_network(brain_t brain);

// Caching
bool brain_is_cached_input(brain_t brain, const float* features, uint32_t num_features);
void brain_cache_decision(brain_t brain, const float* features, uint32_t num_features, brain_decision_t* decision);
void brain_clear_cache(brain_t brain);

// Helpers
void brain_heartbeat(brain_t brain, const char* operation, float progress);
void brain_adapt_learning_rate_from_loss(brain_t brain, float current_loss);

#endif // NIMCP_BRAIN_INTERNAL_DECOMPOSED_H
```

### 2.2 Create Conversion Helpers

Add functions to convert between old and new structs (temporary, for migration):

```c
// Convert old struct to new decomposed struct
void brain_migrate_to_decomposed(brain_t old_brain, brain_struct_decomposed_t* new_brain);

// Convert new decomposed struct back to old struct
void brain_migrate_from_decomposed(const brain_struct_decomposed_t* new_brain, brain_t old_brain);
```

---

## Phase 3: Create Split Source Files (8-12 hours)

For each new source file, follow this process:

1. **Extract functions** from `nimcp_brain.c`
2. **Add proper includes** and logging module define
3. **Implement internal helpers** if needed
4. **Verify compilation** with `make nimcp -j4`

### 3.1 nimcp_brain_lifecycle.c

**Functions to extract**:
- `allocate_brain()` (existing)
- `brain_destroy()` (existing)
- `init_*_subsystem()` (all initialization functions)
- `build_*_config()` (configuration builders)
- `get_neuron_count()`, `get_default_sparsity()`
- `create_brain_network()`
- `init_output_labels()`

**New functions**:
- `brain_init_subsystems()` - calls all `init_*_subsystem()` in correct order
- `brain_cleanup_subsystems()` - reverses initialization order

**Test file**: `tests/core/brain/test_brain_lifecycle.cpp`

### 3.2 nimcp_brain_inference.c

**Functions to extract**:
- `brain_process()` (from existing location)
- `brain_decide()` (from existing location)
- `brain_learn_example()` (from learning module)
- `is_cached_input()`, `cache_decision()`, `clear_cache()`
- Strategy transformation functions
- Loss computation functions

**New functions**:
- `brain_process_internal()` - core inference logic
- `brain_decide_internal()` - decision-making logic

**Test file**: `tests/core/brain/test_brain_inference.cpp`

### 3.3 nimcp_brain_training.c

**Functions to extract**:
- Training pipeline coordination
- Callback management
- Loss tracking
- `adapt_learning_rate_from_loss()`
- Quantum annealing integration

**New functions**:
- `brain_training_begin()`
- `brain_training_step()`
- `brain_training_end()`

**Test file**: `tests/core/brain/test_brain_training.cpp`

### 3.4 nimcp_brain_serialization.c

**Functions to extract**:
- `brain_save()` (existing)
- `brain_load()` (existing)
- `brain_from_pretrained()` (existing)
- Snapshot save/restore
- Checkpoint support

**New functions**:
- `brain_save_internal()`
- `brain_load_internal()`

**Test file**: `tests/core/brain/test_brain_serialization.cpp`

### 3.5 nimcp_brain_features.c

**Functions to extract**:
- `brain_resize()` (existing)
- `brain_auto_resize()` (existing)
- Oscillation enable/disable
- Phase coherence/PAC getters
- `ensure_writable_network()` (existing)
- `brain_clone_cow()` (existing)

**New functions**:
- `brain_resize_internal()`
- `brain_enable_oscillations_internal()`

**Test file**: `tests/core/brain/test_brain_features.cpp`

### 3.6 nimcp_brain_state.c

**Functions to extract**:
- All `brain_get_*()` accessor functions (20+)
- `brain_update_stats()`
- Statistics management
- State validation

**New functions**:
- `brain_validate_state()` - check state consistency

**Test file**: `tests/core/brain/test_brain_integration.cpp` (integration tests)

---

## Phase 4: Write Tests (4-6 hours)

For each test file:

1. **Use BRAIN_TINY** (not hemispheric_brain_create) to avoid 10-60GB RAM
2. **Test core functionality** for that module
3. **Mock dependencies** where appropriate
4. **Verify against existing tests** (ensure no regressions)

### Example: test_brain_lifecycle.cpp

```cpp
#include <gtest/gtest.h>
#include "core/brain/nimcp_brain.h"

class BrainLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        brain = nullptr;
    }

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
            brain = nullptr;
        }
    }

    brain_t brain;
};

TEST_F(BrainLifecycleTest, CreateDestroyTinyBrain) {
    brain = brain_create("test", BRAIN_SIZE_TINY, BRAIN_TASK_CLASSIFICATION, 10, 3);
    ASSERT_NE(brain, nullptr);
    EXPECT_EQ(brain_get_neuron_count(brain), 100);  // BRAIN_SIZE_TINY = 100 neurons
}

TEST_F(BrainLifecycleTest, CustomConfiguration) {
    brain_config_t config = brain_config_default();
    config.size = BRAIN_SIZE_SMALL;
    config.num_inputs = 20;
    config.num_outputs = 5;
    config.enable_working_memory = true;
    config.enable_global_workspace = true;

    brain = brain_create_custom(&config);
    ASSERT_NE(brain, nullptr);

    // Verify subsystems initialized
    EXPECT_NE(brain_get_working_memory(brain), nullptr);
    EXPECT_NE(brain_get_global_workspace(brain), nullptr);
}

TEST_F(BrainLifecycleTest, InitializeOptionalSubsystems) {
    // Test selective subsystem initialization
    // ...
}
```

---

## Phase 5: Integration & Verification (3-4 hours)

### 5.1 Update CMakeLists.txt

Add new source files to `src/lib/CMakeLists.txt`:

```cmake
# Brain core (decomposed)
src/core/brain/nimcp_brain_lifecycle.c
src/core/brain/nimcp_brain_inference.c
src/core/brain/nimcp_brain_training.c
src/core/brain/nimcp_brain_serialization.c
src/core/brain/nimcp_brain_features.c
src/core/brain/nimcp_brain_state.c
```

### 5.2 Verify Build

```bash
cd /home/bbrelin/nimcp/build
cmake ..
make nimcp -j4
```

### 5.3 Run Test Suite

```bash
# Run new tests
ctest -R "BrainLifecycle|BrainInference|BrainTraining|BrainSerialization|BrainFeatures|BrainIntegration" -V

# Run full regression suite
ctest -R "regression" -j3 --timeout 600
```

**Success criteria**:
- All new tests pass
- All 472 existing regression tests pass
- No memory leaks (valgrind clean)
- Build time not significantly increased

### 5.4 Delete Original File

Once verified:

```bash
git rm src/core/brain/nimcp_brain.c
```

---

## Phase 6: Documentation & Cleanup (2-3 hours)

### 6.1 Update Documentation

- Update `/home/bbrelin/nimcp/docs/claude/04-file-organization.md`
- Add section on brain module decomposition
- Document new internal header

### 6.2 Update CLAUDE.md

Add to trigger-action rules:

```markdown
## Brain Module Structure

The brain module is decomposed into 6 focused files following SRP:

1. **nimcp_brain_lifecycle.c** - Creation, destruction, initialization
2. **nimcp_brain_inference.c** - Forward pass, decision-making
3. **nimcp_brain_training.c** - Training pipeline, callbacks
4. **nimcp_brain_serialization.c** - Save, load, checkpoints
5. **nimcp_brain_features.c** - Resize, oscillations, COW cloning
6. **nimcp_brain_state.c** - Accessors, statistics

Internal header: `nimcp_brain_internal_decomposed.h` defines sub-structs.
```

### 6.3 Commit Changes

```bash
git add -A
git commit -m "refactor: Decompose nimcp_brain.c into 6 SRP-compliant modules

- Split 6,150-line monolith into focused modules
- Decompose brain_struct into 7 coherent sub-structs
- Add 6 new test files with full coverage
- All 472 regression tests pass
- Ref: docs/refactoring/brain_refactoring_plan.md"
git push
```

---

## Risk Mitigation

### Risk: Breaking Existing Tests

**Mitigation**:
- Run regression suite after EVERY change
- Use feature branches for incremental work
- Keep old code until new code verified

### Risk: Performance Regression

**Mitigation**:
- Benchmark before/after with `test_performance`
- Profile with `perf` or `valgrind --tool=callgrind`
- Optimize hot paths if degradation >5%

### Risk: Memory Leaks

**Mitigation**:
- Run `valgrind --leak-check=full` on all tests
- Use AddressSanitizer (`-fsanitize=address`)
- Verify with `heaptrack` for large allocations

---

## Success Criteria

- ✅ `nimcp_brain.c` deleted
- ✅ 6 new source files created and tested
- ✅ All 472 regression tests pass
- ✅ No memory leaks detected
- ✅ Build time not increased >10%
- ✅ Code coverage maintained or improved
- ✅ Documentation updated

---

## Notes

- **DO NOT attempt this in a single session** - requires multiple days
- **Verify incrementally** - test after each file creation
- **Use BRAIN_TINY in tests** - avoid 10-60GB RAM allocations
- **Keep public API unchanged** - only internal refactoring
- **Ask for help** - if stuck, consult previous Claude or user

---

## Checklist

Phase 1: Analysis & Preparation
- [ ] 1.1 Analyze function distribution
- [ ] 1.2 Map brain_struct fields
- [ ] 1.3 Identify dependencies

Phase 2: Create Decomposed Internal Header
- [ ] 2.1 Define sub-structs
- [ ] 2.2 Create conversion helpers

Phase 3: Create Split Source Files
- [ ] 3.1 nimcp_brain_lifecycle.c
- [ ] 3.2 nimcp_brain_inference.c
- [ ] 3.3 nimcp_brain_training.c
- [ ] 3.4 nimcp_brain_serialization.c
- [ ] 3.5 nimcp_brain_features.c
- [ ] 3.6 nimcp_brain_state.c

Phase 4: Write Tests
- [ ] 4.1 test_brain_lifecycle.cpp
- [ ] 4.2 test_brain_inference.cpp
- [ ] 4.3 test_brain_training.cpp
- [ ] 4.4 test_brain_serialization.cpp
- [ ] 4.5 test_brain_features.cpp
- [ ] 4.6 test_brain_integration.cpp

Phase 5: Integration & Verification
- [ ] 5.1 Update CMakeLists.txt
- [ ] 5.2 Verify build
- [ ] 5.3 Run test suite (all pass)
- [ ] 5.4 Delete original file

Phase 6: Documentation & Cleanup
- [ ] 6.1 Update documentation
- [ ] 6.2 Update CLAUDE.md
- [ ] 6.3 Commit changes

---

**END OF REFACTORING PLAN**
