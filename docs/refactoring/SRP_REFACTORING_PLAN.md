# SRP Refactoring Plan: 5 Cognitive Modules

**Date**: 2026-02-16
**Version**: 2.6.3
**Author**: Claude Code

## Overview

Refactoring 5 cognitive P2 files (~11,000 lines total) to comply with the Single Responsibility Principle (SRP). Each module will be split into focused implementation files with shared internal headers.

## Refactoring Strategy

### Core Principles
1. **Keep public headers UNCHANGED** - No API breaking changes
2. **Create internal headers** for shared types/functions between split files
3. **Move functions** to split files based on responsibility
4. **DO NOT modify** CMakeLists.txt (already compiles all .c files)
5. **Create manifest** files documenting the split

### File Naming Convention
- Internal headers: `nimcp_<module>_internal.h`
- Split files: `nimcp_<module>_<concern>.c`
- Manifest: `NEW_FILES_MANIFEST.txt` in each source directory

## Module 1: Global Workspace (2,342 lines → 5 files)

### Source Directory
`/home/bbrelin/nimcp/src/cognitive/global_workspace/`

### Split Plan

#### Internal Header: `nimcp_gw_internal.h`
**Contains**:
- `struct global_workspace_struct` - complete definition
- Helper function declarations (internal API)
- Internal constants and macros

#### 1. `nimcp_gw_competition.c` - Competition Resolution
**Responsibility**: Winner-take-all, decay, competitor management
**Functions**:
- `resolve_winner_take_all()` - Find strongest signal
- `resolve_priority_based()` - Priority-based resolution
- `resolve_round_robin()` - Round-robin selection
- `apply_competition_decay()` - Exponential decay to stale signals
- `add_to_competition_pool()` - Add/update competitor
- `clear_competition_pool()` - Reset pool after broadcast

**Lines**: ~400

#### 2. `nimcp_gw_broadcast.c` - Broadcast Mechanism
**Responsibility**: Broadcasting winners, content management
**Functions**:
- `global_workspace_compete()` - PUBLIC: Submit and compete
- `global_workspace_submit()` - PUBLIC: Add to pool
- `global_workspace_resolve()` - PUBLIC: Resolve and broadcast
- `global_workspace_read_broadcast()` - PUBLIC: Read current broadcast
- `broadcast_winner()` - Internal: Execute broadcast
- `copy_broadcast_content()` - Copy content with CoW support

**Lines**: ~500

#### 3. `nimcp_gw_subscribers.c` - Subscription Management
**Responsibility**: Subscriber list, notifications
**Functions**:
- `global_workspace_subscribe()` - PUBLIC: Add subscriber
- `global_workspace_unsubscribe()` - PUBLIC: Remove subscriber
- `global_workspace_get_subscriber_count()` - PUBLIC: Query count
- `notify_subscribers()` - Internal: Notify all (if push model)
- `is_subscribed()` - Check if module subscribed

**Lines**: ~300

#### 4. `nimcp_gw_history.c` - History Operations
**Responsibility**: Circular buffer history tracking
**Functions**:
- `global_workspace_get_history()` - PUBLIC: Retrieve history
- `global_workspace_time_since_broadcast()` - PUBLIC: Time queries
- `add_to_history()` - Append to circular buffer
- `clear_history()` - Reset history

**Lines**: ~300

#### 5. `nimcp_gw_core.c` - Lifecycle & Configuration
**Responsibility**: Create/destroy, config, stats, bio-async, memory pools
**Functions**:
- `global_workspace_create()` - PUBLIC: Default creation
- `global_workspace_create_custom()` - PUBLIC: Custom config
- `global_workspace_destroy()` - PUBLIC: Cleanup
- `global_workspace_default_config()` - PUBLIC: Default config
- `global_workspace_set_ignition_threshold()` - PUBLIC: Config update
- `global_workspace_get_statistics()` - PUBLIC: Stats access
- `global_workspace_print_state()` - PUBLIC: Debug output
- `get_time_ms()` - Internal: Time helper
- `clamp_float()` - Internal: Math helper
- `cognitive_module_to_string()` - PUBLIC: Utility
- Bio-async message handlers

**Lines**: ~842 (largest - lifecycle code)

---

## Module 2: Working Memory (2,634 lines → 7 files)

### Source Directory
`/home/bbrelin/nimcp/src/cognitive/working_memory/`

### Split Plan

#### Internal Header: `nimcp_wm_internal.h`
**Contains**:
- `struct working_memory` - complete definition
- Helper function declarations
- Internal constants

#### 1. `nimcp_wm_storage.c` - Item Storage
**Responsibility**: Add/remove/get items, capacity management
**Functions**:
- `working_memory_add()` - PUBLIC: Add item
- `working_memory_add_with_emotion()` - PUBLIC: Add with emotion
- `working_memory_get()` - PUBLIC: Get item
- `working_memory_remove()` - PUBLIC: Remove item
- `working_memory_clear()` - PUBLIC: Clear all

**Lines**: ~400

#### 2. `nimcp_wm_decay.c` - Temporal Decay
**Responsibility**: Exponential decay calculations
**Functions**:
- `working_memory_decay()` - PUBLIC: Apply decay
- `compute_decay_factor()` - Internal: exp(-t/τ)
- `should_decay_item()` - Check if decay applies

**Lines**: ~200

#### 3. `nimcp_wm_eviction.c` - Salience-Based Eviction
**Responsibility**: Eviction priority, lowest-salience search
**Functions**:
- `find_lowest_salience_index()` - Find eviction target
- `evict_item_at_index()` - Remove and compact
- `working_memory_get_effective_capacity()` - Capacity after inflammation

**Lines**: ~250

#### 4. `nimcp_wm_emotional.c` - Emotional Tagging
**Responsibility**: Emotional context, emotion-based operations
**Functions**:
- `working_memory_get_emotion()` - PUBLIC: Get emotion tag
- `working_memory_get_total_salience()` - PUBLIC: Salience with emotion boost
- `working_memory_find_by_emotion()` - Find items by emotion
- `emotional_compute_salience_boost()` - Compute boost factor

**Lines**: ~250

#### 5. `nimcp_wm_encoding.c` - Positional Encoding
**Responsibility**: Position encoding integration, serial position effects
**Functions**:
- `working_memory_apply_position_encoding()` - Apply PE to item
- `working_memory_get_position_encoding()` - Get PE vector

**Lines**: ~200

#### 6. `nimcp_wm_integration.c` - System Integration
**Responsibility**: Bio-async, second messengers, immune, workspace
**Functions**:
- `working_memory_integrate_second_messengers()` - PUBLIC: SM integration
- `working_memory_integrate_global_workspace()` - PUBLIC: GW integration
- `working_memory_connect_immune()` - PUBLIC: Immune integration
- `working_memory_set_sleep_state()` - PUBLIC: Sleep modulation
- Bio-async message handlers
- Quantum bridge operations

**Lines**: ~600

#### 7. `nimcp_wm_core.c` - Lifecycle & Configuration
**Responsibility**: Create/destroy, config, stats
**Functions**:
- `working_memory_create()` - PUBLIC
- `working_memory_create_custom()` - PUBLIC
- `working_memory_destroy()` - PUBLIC
- `working_memory_default_config()` - PUBLIC
- `working_memory_get_statistics()` - PUBLIC
- `working_memory_get_last_error()` - PUBLIC
- `get_current_time_ms()` - Internal

**Lines**: ~734

---

## Module 3: Introspection (2,673 lines → 6 files)

### Source Directory
`/home/bbrelin/nimcp/src/cognitive/introspection/`

### Split Plan

#### Internal Header: `nimcp_introspection_internal.h`
**Contains**:
- `struct introspection_context_struct` - complete definition
- `pattern_entry_t`, `pattern_registry_t` - internal types
- Helper declarations

#### 1. `nimcp_introspection_state.c` - Neural State Extraction
**Responsibility**: Extract state, neuron populations, state vectors
**Functions**:
- `introspection_get_neuron_state()` - PUBLIC
- `introspection_get_layer_activation()` - PUBLIC
- `introspection_extract_state_vector()` - PUBLIC
- `compute_population_statistics()` - Internal

**Lines**: ~450

#### 2. `nimcp_introspection_uncertainty.c` - Uncertainty Estimation
**Responsibility**: Ensemble methods, confidence intervals
**Functions**:
- `introspection_estimate_uncertainty()` - PUBLIC
- `introspection_get_prediction_confidence()` - PUBLIC
- `ensemble_compute_variance()` - Internal
- `compute_entropy_from_predictions()` - Internal

**Lines**: ~400

#### 3. `nimcp_introspection_patterns.c` - Pattern Tracking
**Responsibility**: Pattern registry, pattern tracking
**Functions**:
- `introspection_register_pattern()` - PUBLIC
- `introspection_get_pattern_activity()` - PUBLIC
- `pattern_registry_lookup()` - Internal
- `pattern_registry_update()` - Internal
- `hash_string()` - Internal

**Lines**: ~350

#### 4. `nimcp_introspection_history.c` - Activity History
**Responsibility**: History snapshots, auto-sampling
**Functions**:
- `introspection_add_to_history()` - PUBLIC
- `introspection_get_activity_history()` - PUBLIC
- `introspection_enable_auto_sampling()` - PUBLIC
- `auto_sample_callback()` - Internal
- `should_trigger_sample()` - Internal

**Lines**: ~400

#### 5. `nimcp_introspection_topology.c` - Network Topology
**Responsibility**: Topology caching, analysis
**Functions**:
- `introspection_get_network_topology()` - PUBLIC
- `introspection_compute_connectivity()` - PUBLIC
- `cache_topology()` - Internal
- `invalidate_topology_cache()` - Internal

**Lines**: ~300

#### 6. `nimcp_introspection_core.c` - Lifecycle & Integration
**Responsibility**: Create/destroy, config, stats, bio-async, KG
**Functions**:
- `introspection_context_create()` - PUBLIC
- `introspection_context_destroy()` - PUBLIC
- `introspection_default_config()` - PUBLIC
- `introspection_get_statistics()` - PUBLIC
- `introspection_connect_immune()` - PUBLIC
- `introspection_set_ensemble()` - PUBLIC
- Bio-async handlers
- SNN/Plasticity bridge management

**Lines**: ~773

---

## Module 4: Recursive Cognition Orchestrator (2,247 lines → 6 files)

### Source Directory
`/home/bbrelin/nimcp/src/cognitive/recursive/`

### Split Plan

#### Internal Header: `nimcp_rcog_orch_internal.h`
**Contains**:
- Internal orchestrator structures
- Task queue types
- Helper declarations

#### 1. `nimcp_rcog_decomposition.c` - Task Decomposition
**Responsibility**: Decompose tasks, dependency graphs, cycle detection
**Functions**:
- `rcog_orch_decompose_task()` - PUBLIC
- `rcog_orch_validate_decomposition()` - PUBLIC
- `build_dependency_graph()` - Internal
- `detect_cycles()` - Internal: Tarjan's algorithm

**Lines**: ~450

#### 2. `nimcp_rcog_depth.c` - Recursion Depth Control
**Responsibility**: Depth limiting, recursion tracking
**Functions**:
- `rcog_orch_push_depth()` - PUBLIC
- `rcog_orch_pop_depth()` - PUBLIC
- `rcog_orch_get_current_depth()` - PUBLIC
- `check_depth_limit()` - Internal

**Lines**: ~200

#### 3. `nimcp_rcog_batch.c` - Batch Execution
**Responsibility**: Batch task execution, tracking
**Functions**:
- `rcog_orch_execute_batch()` - PUBLIC
- `rcog_orch_get_batch_status()` - PUBLIC
- `schedule_batch_tasks()` - Internal
- `collect_batch_results()` - Internal

**Lines**: ~350

#### 4. `nimcp_rcog_mcts.c` - MCTS Strategy
**Responsibility**: Monte Carlo Tree Search for strategy selection
**Functions**:
- `rcog_mcts_select_strategy()` - PUBLIC
- `rcog_mcts_expand_node()` - Internal
- `rcog_mcts_simulate()` - Internal
- `rcog_mcts_backpropagate()` - Internal

**Lines**: ~400

#### 5. `nimcp_rcog_trace.c` - Execution Tracing
**Responsibility**: Trace recording, trace management
**Functions**:
- `rcog_orch_enable_tracing()` - PUBLIC
- `rcog_orch_get_trace()` - PUBLIC
- `record_trace_event()` - Internal
- `clear_trace_buffer()` - Internal

**Lines**: ~250

#### 6. `nimcp_rcog_orch_core.c` - Lifecycle & Stats
**Responsibility**: Create/destroy, immune modulation, answer refinement, stats
**Functions**:
- `rcog_orch_create()` - PUBLIC
- `rcog_orch_destroy()` - PUBLIC
- `rcog_orch_get_statistics()` - PUBLIC
- `rcog_orch_refine_answer()` - PUBLIC
- `rcog_orch_apply_immune_modulation()` - PUBLIC
- Bio-async integration

**Lines**: ~597

---

## Module 5: Recursive Cognition Engine (1,789 lines → 6 files)

### Source Directory
`/home/bbrelin/nimcp/src/cognitive/recursive/`

### Split Plan

#### Internal Header: `nimcp_rcog_engine_internal.h`
**Contains**:
- Engine internal structures
- Search state types
- Helper declarations

#### 1. `nimcp_rcog_engine_goals.c` - Goal Processing
**Responsibility**: Process goals, goal context management
**Functions**:
- `rcog_engine_process()` - PUBLIC: Main entry point
- `rcog_engine_set_goal()` - PUBLIC
- `create_goal_context()` - Internal
- `destroy_goal_context()` - Internal

**Lines**: ~300

#### 2. `nimcp_rcog_engine_tools.c` - Tool Management
**Responsibility**: Tool selection, invocation, result handling
**Functions**:
- `rcog_engine_select_tool()` - PUBLIC
- `rcog_engine_invoke_tool()` - PUBLIC
- `rank_tools_for_goal()` - Internal
- `parse_tool_result()` - Internal

**Lines**: ~350

#### 3. `nimcp_rcog_engine_search.c` - Search Strategy
**Responsibility**: Search strategy, state management, tree exploration
**Functions**:
- `rcog_engine_search()` - PUBLIC
- `rcog_engine_expand_state()` - PUBLIC
- `explore_search_tree()` - Internal
- `evaluate_search_path()` - Internal

**Lines**: ~350

#### 4. `nimcp_rcog_engine_thoughts.c` - Thought Generation
**Responsibility**: Generate thoughts, evaluate, rank
**Functions**:
- `rcog_engine_generate_thought()` - PUBLIC
- `rcog_engine_evaluate_thought()` - PUBLIC
- `rank_thoughts()` - Internal
- `prune_weak_thoughts()` - Internal

**Lines**: ~300

#### 5. `nimcp_rcog_engine_cache.c` - Result Caching
**Responsibility**: Cache management, lookup, eviction
**Functions**:
- `rcog_engine_cache_result()` - PUBLIC
- `rcog_engine_lookup_cache()` - PUBLIC
- `cache_evict_lru()` - Internal
- `cache_hash_goal()` - Internal

**Lines**: ~200

#### 6. `nimcp_rcog_engine_core.c` - Lifecycle & Learning
**Responsibility**: Create/destroy, learning, consolidation, stats
**Functions**:
- `rcog_engine_create()` - PUBLIC
- `rcog_engine_destroy()` - PUBLIC
- `rcog_engine_learn_from_trace()` - PUBLIC
- `rcog_engine_consolidate()` - PUBLIC
- `rcog_engine_get_statistics()` - PUBLIC

**Lines**: ~289

---

## Testing Strategy

### Test Organization
For each split module, create unit tests in:
```
/home/bbrelin/nimcp/test/unit/cognitive/<module>/<concern>/
```

### Test File Naming
```
test_<module>_<concern>.cpp
```

### Test Coverage Requirements
1. **Unit tests** for each split file's public functions
2. **Integration tests** for interactions between split files
3. **Regression tests** to ensure no behavioral changes

### Example: Global Workspace Tests
```
test/unit/cognitive/global_workspace/
├── competition/
│   └── test_gw_competition.cpp
├── broadcast/
│   └── test_gw_broadcast.cpp
├── subscribers/
│   └── test_gw_subscribers.cpp
├── history/
│   └── test_gw_history.cpp
└── core/
    └── test_gw_core.cpp
```

---

## Implementation Checklist

### Per Module
- [ ] Create internal header with complete struct definition
- [ ] Create split implementation files
- [ ] Move functions to appropriate split files
- [ ] Verify all functions have access to needed internals
- [ ] Create `NEW_FILES_MANIFEST.txt`
- [ ] Write unit tests for each split file
- [ ] Run `make nimcp -j4` to verify build
- [ ] Run regression tests to verify behavior

### Global Verification
- [ ] All 5 modules refactored
- [ ] Public headers unchanged
- [ ] No CMakeLists.txt modifications needed
- [ ] All tests pass (472/472)
- [ ] No performance degradation

---

## Manifest File Template

```
NEW_FILES_MANIFEST.txt
=====================

Module: <Module Name>
Date: 2026-02-16
Refactored by: Claude Code
Version: 2.6.3

Original File: nimcp_<module>.c (X lines)
Split into: N files

FILES CREATED:
==============

1. nimcp_<module>_internal.h (Y lines)
   - Complete struct definition
   - Internal helper declarations
   - Lines from original: N/A (new header)

2. nimcp_<module>_<concern1>.c (Z lines)
   - Responsibility: <description>
   - Functions: <list>
   - Lines from original: XXX-YYY

3. nimcp_<module>_<concern2>.c (Z lines)
   ...

TESTS CREATED:
==============

1. test/unit/cognitive/<module>/<concern1>/test_<module>_<concern1>.cpp
   - Tests for <concern1> module
   - Coverage: <list of test cases>

...

VERIFICATION:
=============

Build: PASS (make nimcp -j4)
Tests: PASS (ctest -R unit_cognitive_<module>)
Regression: PASS (ctest -R regression_cognitive_<module>)

NOTES:
======

- Public API unchanged
- Internal structs now in internal header
- All split files include internal header
- CMakeLists.txt unchanged (auto-compiles all .c)
```

---

## Benefits of This Refactoring

### Code Quality
1. **Single Responsibility**: Each file has one clear purpose
2. **Easier Maintenance**: Smaller files easier to understand/modify
3. **Better Testability**: Isolated concerns easier to test
4. **Improved Readability**: Clear separation of concerns

### Development Velocity
1. **Parallel Development**: Multiple devs can work on different concerns
2. **Faster Compilation**: Changes to one concern don't recompile others (in future if split into separate compilation units)
3. **Easier Debugging**: Smaller scope to search for bugs

### Architecture
1. **Clearer Dependencies**: Internal headers make dependencies explicit
2. **Refactoring Foundation**: Easier to extract concerns into separate libraries later
3. **Documentation**: File names self-document module organization

---

## Next Steps

1. **Implement Global Workspace split** (establishes pattern)
2. **Review with team** (ensure approach is sound)
3. **Apply pattern to remaining 4 modules**
4. **Create comprehensive test suites**
5. **Update documentation** (add to INDEX.md)
6. **Run full regression suite** (472/472 PASS)

---

## Questions for Review

1. Is the concern separation appropriate for each module?
2. Should we create a standard "internal.h" pattern for all modules?
3. Should we add compilation guards to prevent direct inclusion of internal headers?
4. Should we add static inline helpers to internal headers or keep them in .c files?

