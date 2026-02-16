# Refactoring Completion Checklist

**Date**: 2026-02-16
**Goal**: Split 8153 lines across 2 files into 14 focused SRP-compliant modules
**Status**: Foundation Complete - Awaiting User Approval to Proceed

---

## Phase 1: Foundation ✅ COMPLETE

- [x] Create internal headers (2/2)
  - [x] `src/cognitive/omni/nimcp_omni_wm_internal.h`
  - [x] `src/cognitive/immune/nimcp_brain_immune_internal.h`

- [x] Create example module (1/1)
  - [x] `src/cognitive/omni/nimcp_omni_wm_state.c` (404 lines, 17 functions)

- [x] Create example test (1/1)
  - [x] `tests/cognitive/omni/test_omni_wm_state.cpp` (426 lines, 18 test cases)

- [x] Create documentation (3/3)
  - [x] `REFACTORING_GUIDE.md` - Complete process guide
  - [x] `REFACTORING_STATUS.md` - Detailed status tracking
  - [x] `REFACTORING_SUMMARY.md` - Executive summary

- [x] Verify build pattern works
  - [x] State module compiles cleanly
  - [x] Public API unchanged
  - [x] Test pattern established

---

## Phase 2: Omni World Model (⏳ PENDING USER APPROVAL)

### Implementation (6 modules)
- [ ] `nimcp_omni_wm_core.c` - Main lifecycle, config, stats, bio-async (~800 lines)
  - Functions: `omni_wm_create/destroy`, `omni_wm_get_default_config`, `omni_wm_get_stats/reset_stats`, bio-async connect/disconnect, training callbacks, string converters

- [ ] `nimcp_omni_wm_dynamics.c` - RSSM dynamics, step functions (~600 lines)
  - Functions: `dynamics_create/destroy`, `omni_wm_rssm_step`, `omni_wm_predict_forward/backward/lateral/hierarchical`, `omni_wm_symlog/symexp`, step implementations

- [ ] `nimcp_omni_wm_replay_buffer.c` - Experience replay (~400 lines)
  - Functions: `replay_buffer_create/destroy`, `omni_wm_add_experience`, `omni_wm_sample_experiences`, `omni_wm_get_replay_size/clear_replay`, `omni_wm_experience_create/destroy`

- [ ] `nimcp_omni_wm_serialization.c` - Save/load, CRC32 (~900 lines)
  - Functions: `omni_wm_save/load/serialize/deserialize`, `crc32_compute`, read/write helpers, serialize/deserialize_config/state/rssm/dynamics/stats

- [ ] `nimcp_omni_wm_checkpoint.c` - Checkpoint management (~400 lines)
  - Functions: `checkpoint_store_create/destroy`, `omni_wm_checkpoint/restore/delete_checkpoint`, `omni_wm_get_checkpoint_count/clear_checkpoints`

- [ ] `nimcp_omni_wm_counterfactual.c` - Counterfactuals, rollouts, MDN (~700 lines)
  - Functions: `omni_wm_counterfactual/what_if`, `omni_wm_cf_query_create/destroy`, `omni_wm_cf_result_destroy`, `omni_wm_rollout_create/destroy/rollout`, `omni_wm_evaluate_efe`, MDN functions

### Tests (6 test files)
- [ ] `test_omni_wm_core.cpp` - Lifecycle, config, stats
- [ ] `test_omni_wm_dynamics.cpp` - RSSM dynamics, predictions
- [ ] `test_omni_wm_replay.cpp` - Experience replay operations
- [ ] `test_omni_wm_serialization.cpp` - Save/load, serialization
- [ ] `test_omni_wm_checkpoint.cpp` - Checkpoint management
- [ ] `test_omni_wm_counterfactual.cpp` - Counterfactuals, rollouts

---

## Phase 3: Brain Immune System (⏳ PENDING USER APPROVAL)

### Implementation (7 modules)
- [ ] `nimcp_brain_immune_orchestrator.c` - Main lifecycle, integration (~700 lines)
  - Functions: `brain_immune_create/destroy/start/stop/update`, `brain_immune_connect_*`, `brain_immune_default_config`, phase management

- [ ] `nimcp_brain_immune_antigens.c` - Antigen detection, processing (~500 lines)
  - Functions: `brain_immune_present_*`, `find_antigen_by_id`, `process_pending_antigens`, `brain_immune_add_antigen/get_antigen/is_neutralized`

- [ ] `nimcp_brain_immune_cells.c` - B/T cell state machines (~700 lines)
  - Functions: `brain_immune_activate_b_cell/helper_t/killer_t`, `brain_immune_b_cell_to_memory`, `brain_immune_t_help_b/t_cell_kill`, find_*_by_id, state updates

- [ ] `nimcp_brain_immune_antibodies.c` - Antibody production (~500 lines)
  - Functions: `brain_immune_produce_antibody/execute_antibody/neutralize`, `brain_immune_trigger_swarm_response`, `decay_antibodies`, `find_antibody_by_id`

- [ ] `nimcp_brain_immune_signaling.c` - Cytokine release (~400 lines)
  - Functions: `brain_immune_release_cytokine/broadcast_alert`, `brain_immune_get_cytokine_level`, cytokine updates, imagination modulation

- [ ] `nimcp_brain_immune_inflammation.c` - Inflammation management (~500 lines)
  - Functions: `brain_immune_initiate/escalate/resolve_inflammation`, `brain_immune_get_inflammation_level`, `find_inflammation_by_id`, site updates

- [ ] `nimcp_brain_immune_stats.c` - Statistics, checkpointing (~600 lines)
  - Functions: `brain_immune_get_stats/get_checkpoint_state/get_phase`, training callbacks, KG integration, `brain_immune_compute_affinity`

### Tests (7 test files)
- [ ] `test_brain_immune_antigens.cpp` - Antigen presentation, processing
- [ ] `test_brain_immune_cells.cpp` - B/T cell lifecycle, state progression
- [ ] `test_brain_immune_antibodies.cpp` - Antibody production, execution
- [ ] `test_brain_immune_signaling.cpp` - Cytokine release, messaging
- [ ] `test_brain_immune_inflammation.cpp` - Inflammation escalation, resolution
- [ ] `test_brain_immune_integration.cpp` - BBB/BFT/swarm integration
- [ ] `test_brain_immune_stats.cpp` - Statistics, checkpointing

---

## Phase 4: Documentation & Cleanup (⏳ PENDING COMPLETION)

- [ ] Create manifests (2 files)
  - [ ] `src/cognitive/omni/NEW_FILES_MANIFEST.txt`
  - [ ] `src/cognitive/immune/NEW_FILES_MANIFEST.txt`

- [ ] Verify all builds pass
  - [ ] `make nimcp -j4` builds cleanly
  - [ ] No warnings or errors
  - [ ] All new modules compiled

- [ ] Run all tests
  - [ ] All 18 new test files pass
  - [ ] Full regression suite (472/472 PASS)
  - [ ] No new test failures introduced

- [ ] Code quality verification
  - [ ] All functions < 50 lines
  - [ ] Guard clause pattern everywhere
  - [ ] Proper error handling
  - [ ] Mutex discipline correct
  - [ ] B cell state machine enforced

- [ ] Backup and archive
  - [ ] Original files backed up
  - [ ] Git commits for each module
  - [ ] Tag release with `refactor-srp-v1`

---

## Build & Test Commands Reference

### Build Single Module
```bash
cd /home/bbrelin/nimcp/build
cmake ..
make nimcp -j4
```

### Build and Run Test
```bash
make test_omni_wm_state
ctest -R test_omni_wm_state -V
```

### Full Regression
```bash
ctest -R "regression" -j3 --timeout 600
```

---

## Git Commit Pattern

After each module + test passes:
```bash
git add src/cognitive/<module>/nimcp_<module>_<split>.c
git add tests/cognitive/<module>/test_<module>_<split>.cpp
git commit -m "refactor: Split <module> <split> module - SRP compliance

- Extract <split> responsibilities to dedicated file
- Add comprehensive unit tests (N test cases)
- Maintain public API compatibility
- All functions < 50 lines

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Critical Reminders

### B Cell State Machine
```
NAIVE → ACTIVATED → PLASMA → MEMORY
         ↓            ↓         ↓
    (recognizes)  (produces) (stores)
                  antibodies
```
**ONLY PLASMA can produce antibodies!**

### Return Conventions
- FEP bridges: `0` success, `-1` error
- Immune functions: `0` success, `-1` error
- Core API: `nimcp_error_t` codes

### Mutex Discipline
- Public API: Acquire mutex
- Internal `_internal`: Caller holds mutex
- Helpers `_unlocked`: Document mutex requirement

### Memory Management
- Use `nimcp_alloc/nimcp_calloc/nimcp_free`
- Exception: `nimcp_memory.c`, `nimcp_unified_memory.c`, `nimcp_constant_time.c`

---

## Success Metrics

**Target**: 100% completion of all checkboxes above

**Verification**:
- Build passes: `make nimcp -j4` exits 0
- Tests pass: 472/472 regression suite + 18 new tests
- Code quality: All functions < 50 lines, guard clauses everywhere
- SRP compliance: Each module has single, clear responsibility

---

## Time Estimate

| Phase | Tasks | Est. Time | Status |
|-------|-------|-----------|--------|
| Foundation | 7 items | 4 hours | ✅ DONE |
| Omni WM | 6 modules + 6 tests | 6 hours | ⏳ PENDING |
| Brain Immune | 7 modules + 7 tests | 7 hours | ⏳ PENDING |
| Documentation | 2 manifests + validation | 2 hours | ⏳ PENDING |
| **TOTAL** | **29 deliverables** | **19 hours** | **10% complete** |

---

## Next Action

**🎯 USER DECISION REQUIRED:**

1. Review the foundation work (internal headers, example module, test, docs)
2. Approve the pattern and approach
3. Decide on execution strategy:
   - Option A: Full automation - Claude applies pattern to all 12 remaining modules
   - Option B: Incremental - Claude does one module at a time, user approves each
   - Option C: Collaborative - User extracts some, Claude does others

**Once approved**: Claude will systematically apply the proven 6-step process to each remaining module.

---

**Last Updated**: 2026-02-16
**Next Review**: After user approval to proceed
