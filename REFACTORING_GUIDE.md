# Refactoring Guide: Splitting Large Cognitive Files

**Date**: 2026-02-16
**Status**: In Progress
**Goal**: Split 2 large files (~8000 lines total) into 14 smaller focused files

## Overview

This document provides the systematic approach for splitting:
1. `src/cognitive/omni/nimcp_omni_world_model.c` (4307 lines) → 7 files
2. `src/cognitive/immune/nimcp_brain_immune.c` (3846 lines) → 7 files

## Internal Headers Created

✅ **Completed**:
- `/home/bbrelin/nimcp/src/cognitive/omni/nimcp_omni_wm_internal.h`
- `/home/bbrelin/nimcp/src/cognitive/immune/nimcp_brain_immune_internal.h`

These headers define:
- Shared internal structures
- Forward declarations
- Helper function prototypes
- Module boundaries

## Split Plan

### Omni World Model Splits

| File | Responsibility | Key Functions | Lines |
|------|---------------|---------------|-------|
| `nimcp_omni_wm_core.c` | Main lifecycle, config, stats, MDN, bio-async | `omni_wm_create`, `omni_wm_destroy`, `omni_wm_get_stats`, `omni_wm_connect_bio_async` | ~800 |
| `nimcp_omni_wm_state.c` | State/RSSM/latent lifecycle | `omni_wm_state_create`, `omni_wm_rssm_state_create`, `omni_wm_latent_create` | ~500 |
| `nimcp_omni_wm_dynamics.c` | RSSM dynamics, step functions | `omni_wm_step_dynamics_forward`, `dynamics_create`, symlog functions | ~600 |
| `nimcp_omni_wm_replay_buffer.c` | Experience replay | `replay_buffer_create`, `omni_wm_add_experience`, `omni_wm_sample_experiences` | ~400 |
| `nimcp_omni_wm_serialization.c` | Save/load, CRC32, serialization | `omni_wm_save`, `omni_wm_load`, `serialize_config`, CRC32 | ~900 |
| `nimcp_omni_wm_checkpoint.c` | Checkpoint management | `omni_wm_checkpoint`, `omni_wm_restore_checkpoint` | ~400 |
| `nimcp_omni_wm_counterfactual.c` | Counterfactual queries, rollouts | `omni_wm_counterfactual`, `omni_wm_what_if`, `omni_wm_rollout` | ~700 |

### Brain Immune Splits

| File | Responsibility | Key Functions | Lines |
|------|---------------|---------------|-------|
| `nimcp_brain_immune_stats.c` | Statistics, monitoring, KG, training | `brain_immune_get_stats`, `brain_immune_get_checkpoint_state`, training callbacks | ~600 |
| `nimcp_brain_immune_antigens.c` | Antigen detection, intake, processing | `brain_immune_present_*`, `find_antigen_by_id`, `process_pending_antigens` | ~500 |
| `nimcp_brain_immune_cells.c` | B/T cell lifecycle, state machines | `brain_immune_activate_b_cell`, `brain_immune_activate_*_t`, find functions | ~700 |
| `nimcp_brain_immune_antibodies.c` | Antibody production, execution | `brain_immune_produce_antibody`, `brain_immune_execute_antibody`, `decay_antibodies` | ~500 |
| `nimcp_brain_immune_signaling.c` | Cytokine release, bio-async messaging | `brain_immune_release_cytokine`, `brain_immune_broadcast_alert` | ~400 |
| `nimcp_brain_immune_inflammation.c` | Inflammation sites, escalation, resolution | `brain_immune_initiate_inflammation`, `brain_immune_escalate_inflammation` | ~500 |
| `nimcp_brain_immune_orchestrator.c` | Integration layer (BBB, BFT, swarm), phase management | `brain_immune_create`, `brain_immune_destroy`, `brain_immune_update`, connect functions | ~700 |

## File Template Structure

Each split file MUST follow this structure:

```c
/**
 * @file <filename>
 * @brief <Brief description of responsibility>
 * @version 1.0.0
 * @date 2026-02-16
 *
 * WHAT: <What this module does>
 * WHY:  <Why it's separated - SRP justification>
 * HOW:  <How it implements its responsibility>
 */

#include "<module>_internal.h"

/* Additional includes if needed */

/* Static helpers (if any) */

/* Public API implementations */
```

## Key Rules

### 1. **Include Internal Header First**
```c
#include "cognitive/omni/nimcp_omni_wm_internal.h"  // or
#include "cognitive/immune/nimcp_brain_immune_internal.h"
```

### 2. **Function Ownership**
- **Public API functions**: Keep original signature exactly
- **Static helpers**: Can be converted to internal helpers in header if shared
- **Internal functions**: Suffix with `_internal` or `_unlocked` if mutex-aware

### 3. **Mutex Discipline**
- Functions ending `_unlocked`: Assume mutex already held by caller
- Public API functions: Must acquire mutex themselves
- Internal helpers: Document mutex requirements in header

### 4. **Error Handling**
- Use `NIMCP_THROW_TO_IMMUNE` for errors
- Guard clause pattern: braces AND return
- Return `0` for success, `-1` for errors (FEP bridges & immune)
- Return `NIMCP_ERROR_*` codes for core API functions

### 5. **Memory Management**
- Use `nimcp_alloc`/`nimcp_calloc`/`nimcp_free` exclusively
- Never use raw `malloc/free` except in:
  - `nimcp_memory.c`
  - `nimcp_unified_memory.c`
  - `nimcp_constant_time.c`

## Example: Complete State Module

See `nimcp_omni_wm_state.c` (next section) for a complete example.

## Testing Strategy

For each split file, create a corresponding test file:

### Omni WM Tests
- `tests/cognitive/omni/test_omni_wm_state.cpp`
- `tests/cognitive/omni/test_omni_wm_dynamics.cpp`
- `tests/cognitive/omni/test_omni_wm_replay.cpp`
- `tests/cognitive/omni/test_omni_wm_serialization.cpp`
- `tests/cognitive/omni/test_omni_wm_checkpoint.cpp`
- `tests/cognitive/omni/test_omni_wm_counterfactual.cpp`

### Brain Immune Tests
- `tests/cognitive/immune/test_brain_immune_antigens.cpp`
- `tests/cognitive/immune/test_brain_immune_cells.cpp`
- `tests/cognitive/immune/test_brain_immune_antibodies.cpp`
- `tests/cognitive/immune/test_brain_immune_signaling.cpp`
- `tests/cognitive/immune/test_brain_immune_inflammation.cpp`
- `tests/cognitive/immune/test_brain_immune_integration.cpp`

### Test Template
```cpp
#include <gtest/gtest.h>
#include "cognitive/<module>/<header>.h"

class <Module>Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test fixtures
    }

    void TearDown() override {
        // Cleanup
    }
};

TEST_F(<Module>Test, BasicCreation) {
    // Test basic functionality
}

TEST_F(<Module>Test, ErrorHandling) {
    // Test error cases with NULL, invalid params
}

TEST_F(<Module>Test, StateTransitions) {
    // Test state machine progressions (for cells, inflammation, etc.)
}
```

## Extraction Process

### Step 1: Identify Functions
```bash
grep -n "^[a-z_A-Z].*{$" src/cognitive/omni/nimcp_omni_world_model.c | grep "function_pattern"
```

### Step 2: Extract Function Block
- Find function start
- Find matching closing brace
- Include all helper functions it calls
- Copy to new file

### Step 3: Update Internal Header
- Add prototype if called from other modules
- Keep static if only used within module

### Step 4: Test Build
```bash
cd /home/bbrelin/nimcp/build
cmake ..
make nimcp -j4
```

### Step 5: Create Unit Test
- Test public API
- Test error handling
- Test edge cases
- Run: `ctest -R <test_name>`

## Manifest Files

Create `NEW_FILES_MANIFEST.txt` in each directory:

### `/home/bbrelin/nimcp/src/cognitive/omni/NEW_FILES_MANIFEST.txt`
```
# Omni World Model Refactoring Manifest
# Date: 2026-02-16
# Original: nimcp_omni_world_model.c (4307 lines)
#
# Split Files:
nimcp_omni_wm_core.c - Main lifecycle, config, stats, bio-async
nimcp_omni_wm_state.c - State/RSSM/latent management
nimcp_omni_wm_dynamics.c - RSSM dynamics, step functions
nimcp_omni_wm_replay_buffer.c - Experience replay buffer
nimcp_omni_wm_serialization.c - Save/load, CRC32, serialization
nimcp_omni_wm_checkpoint.c - Checkpoint management
nimcp_omni_wm_counterfactual.c - Counterfactual queries, rollouts
nimcp_omni_wm_internal.h - Shared internal structures
```

### `/home/bbrelin/nimcp/src/cognitive/immune/NEW_FILES_MANIFEST.txt`
```
# Brain Immune System Refactoring Manifest
# Date: 2026-02-16
# Original: nimcp_brain_immune.c (3846 lines)
#
# Split Files:
nimcp_brain_immune_orchestrator.c - Main lifecycle, integration
nimcp_brain_immune_antigens.c - Antigen detection, processing
nimcp_brain_immune_cells.c - B/T cell state machines
nimcp_brain_immune_antibodies.c - Antibody production, execution
nimcp_brain_immune_signaling.c - Cytokine release, messaging
nimcp_brain_immune_inflammation.c - Inflammation management
nimcp_brain_immune_stats.c - Statistics, monitoring, checkpointing
nimcp_brain_immune_internal.h - Shared internal structures
```

## CMakeLists.txt

⚠️ **DO NOT MODIFY** `CMakeLists.txt` files during this refactoring.

The build system will automatically detect and compile all `.c` files in the directories. If issues arise, we'll handle them in a separate build configuration pass.

## Progress Tracking

| Module | Internal Header | State Module | Dynamics | Replay | Serial | Checkpoint | CF/Stats |
|--------|----------------|--------------|----------|--------|--------|------------|----------|
| Omni WM | ✅ | ⏳ | 🔲 | 🔲 | 🔲 | 🔲 | 🔲 |
| Immune | ✅ | 🔲 | 🔲 | 🔲 | 🔲 | 🔲 | 🔲 |

Legend:
- ✅ Complete
- ⏳ In Progress
- 🔲 Not Started

## Next Steps

1. **Complete state module example** (nimcp_omni_wm_state.c + test)
2. **User approval of pattern**
3. **Apply pattern to remaining 12 modules**
4. **Create all 14 test files**
5. **Build and test full regression suite**
6. **Create manifests**
7. **Final verification**

## Important Notes

### B Cell State Machine
**CRITICAL**: B cells must be in **PLASMA** state to produce antibodies.

State progression: `NAIVE → ACTIVATED → PLASMA → MEMORY`

### Return Value Conventions
- FEP bridges: `0` success, `-1` error
- Metabolic modulation: `0` success, `-1` error
- Standard NIMCP: `nimcp_error_t` codes

### Guard Clause Pattern
Both braces AND return required:
```c
if (!ptr) {
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ptr is NULL");
    return -1;
}
```

`NIMCP_THROW_TO_IMMUNE` alone doesn't halt execution.

## Questions?

Before proceeding with the remaining modules, please review:
1. Internal headers structure
2. Split plan and file responsibilities
3. Template structure
4. Testing approach

Ready to proceed with implementation after approval.
