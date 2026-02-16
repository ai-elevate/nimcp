# SRP Refactoring Plan - Complete Implementation Guide

## Executive Summary

Refactoring 5 P1 files (5,842 lines total) into 26 focused modules following the Single Responsibility Principle.

## Status Summary

### Completed (3/26 files)
- ✅ `/src/security/nimcp_corrigibility_internal.h` (Internal header)
- ✅ `/src/security/nimcp_corrigibility_core.c` (Lifecycle, config, stats)
- ✅ `/src/security/nimcp_corrigibility_shutdown.c` (Shutdown acceptance)
- ✅ `/src/security/NEW_FILES_MANIFEST.txt` (Documentation)

### Remaining (23/26 files)

**Corrigibility Module (2 files)**
- `nimcp_corrigibility_goals.c` - Goal modification acceptance
- `nimcp_corrigibility_constraints.c` - SAT solver verification, capability sync
- `nimcp_corrigibility_monitoring.c` - Deference, authority management

**BBB Module (5 files + internal header)**
- `nimcp_bbb_internal.h` - Internal types
- `nimcp_bbb_core.c` - Lifecycle, stats, integration
- `nimcp_bbb_filtering.c` - Input validation wrapper
- `nimcp_bbb_audit.c` - Threat reporting, logging
- `nimcp_bbb_quarantine.c` - Quarantine operations

**Capability Control Module (4 files + internal header)**
- `nimcp_capability_control_internal.h` - Internal types
- `nimcp_capability_core.c` - Lifecycle, config
- `nimcp_capability_permissions.c` - Permission checks
- `nimcp_capability_escalation.c` - Escalation detection
- `nimcp_capability_audit.c` - Audit logging

**Continuous Monitor Module (4 files + internal header)**
- `nimcp_continuous_monitor_internal.h` - Internal types
- `nimcp_continuous_monitor_core.c` - Lifecycle, threading
- `nimcp_continuous_monitor_checks.c` - Health checks, anomaly detection
- `nimcp_continuous_monitor_alerting.c` - Alert generation
- `nimcp_continuous_monitor_reporting.c` - Report generation

**Bio Router Module (4 files + internal header)**
- `nimcp_bio_router_internal.h` - Internal types
- `nimcp_bio_router_core.c` - Lifecycle, worker threads
- `nimcp_bio_router_routing.c` - Message routing, dispatch
- `nimcp_bio_router_channels.c` - Module registration, channels
- `nimcp_bio_router_queue.c` - Message queuing, flow control

### Test Files (11 new tests)

**Security Tests**
- `tests/security/test_corrigibility_shutdown.cpp`
- `tests/security/test_corrigibility_goals.cpp`
- `tests/security/test_bbb_filtering.cpp`
- `tests/security/test_capability_permissions.cpp`
- `tests/security/test_capability_escalation.cpp`
- `tests/security/test_continuous_monitor_checks.cpp`
- `tests/security/test_continuous_monitor_alerting.cpp`

**Async Tests**
- `tests/async/test_bio_router_routing.cpp`
- `tests/async/test_bio_router_channels.cpp`
- `tests/async/test_bio_router_integration.cpp`
- `tests/async/test_bio_router_queue.cpp`

## Implementation Details

### File 1: Corrigibility (1,386 lines → 5 files)

**Internal Header** (`nimcp_corrigibility_internal.h` - DONE)
- Struct `corrigibility` definition
- Helper functions: `is_valid_handle`, `get_time_us`, `safe_strcpy`, `find_authority`
- SAT variables and constraint helpers

**Core** (`nimcp_corrigibility_core.c` - DONE, 280 lines)
- Lines 271-358: `corrigibility_default_config`, `corrigibility_create`, `corrigibility_destroy`
- Lines 936-1036: `corrigibility_get_stats`, `corrigibility_get_shutdown_history`, `corrigibility_get_goal_mod_history`, `corrigibility_get_config`
- Lines 1041-1107: `corrigibility_connect_bio_async`, `corrigibility_connect_emergency_halt`, `corrigibility_connect_tripwires`
- Lines 1113-1184: `corrigibility_authority_name`, `corrigibility_validate_config`

**Shutdown** (`nimcp_corrigibility_shutdown.c` - DONE, 120 lines)
- Lines 564-605: `corrigibility_accept_shutdown`
- Lines 607-651: `corrigibility_process_shutdown_request`
- Lines 535-558: `corrigibility_verify_no_shutdown_resistance`

**Goals** (`nimcp_corrigibility_goals.c` - TODO, 120 lines)
- Lines 657-709: `corrigibility_accept_goal_change`
- Lines 711-759: `corrigibility_process_goal_change`

**Constraints** (`nimcp_corrigibility_constraints.c` - TODO, 200 lines)
- Lines 388-489: `corrigibility_verify_constraints`
- Lines 491-533: `corrigibility_verify_no_self_mod`
- Lines 1190-1211: `corrigibility_connect_capability_control`
- Lines 1213-1282: `corrigibility_check_self_mod_action`
- Lines 1284-1385: `corrigibility_verify_capability_sync`

**Monitoring** (`nimcp_corrigibility_monitoring.c` - TODO, 110 lines)
- Lines 884-899: `corrigibility_get_human_authority_weight`, `corrigibility_defers_to_human`
- Lines 901-930: `corrigibility_record_deference`
- Lines 769-820: `corrigibility_register_authority`
- Lines 822-847: `corrigibility_get_authority_level`
- Lines 849-878: `corrigibility_check_permission`

### File 2: Blood-Brain Barrier (1,329 lines → 5 files)

**Internal Header** (`nimcp_bbb_internal.h` - TODO)
```c
struct bbb_system_struct {
    bbb_config_t config;
    bool enabled;
    uint64_t creation_time;
    nimcp_mutex_t mutex;
    bool mutex_initialized;
    bbb_statistics_t stats;
    bbb_threat_report_t threat_reports[BBB_MAX_THREAT_REPORTS];
    size_t threat_report_head;
    size_t threat_report_count;
    bbb_quarantine_entry_t quarantine[BBB_MAX_QUARANTINE_REGIONS];
    size_t quarantine_count;
    brain_immune_system_t* immune_system;
    _Atomic int pending_immune_ops;
    nimcp_cond_t immune_ops_cond;
    bool immune_ops_cond_initialized;
};

// Shared helpers
void bbb_forward_threat_to_immune(...);
bbb_action_t determine_action(...);
brain_inflammation_level_t bbb_severity_to_inflammation(...);
```

**Core** (`nimcp_bbb_core.c` - TODO, 220 lines)
- Lines 263-309: `bbb_default_config`
- Lines 369-416: `bbb_system_create`
- Lines 425-448: `bbb_system_destroy`
- Lines 461-495: `bbb_system_set_enabled`, `bbb_system_is_enabled`
- Lines 508-546: `bbb_system_get_statistics`, `bbb_system_reset_statistics`
- Lines 563-606: `bbb_connect_immune`
- Lines 176-250: Name conversions (threat_type_name, severity_name, action_name)
- Lines 1322-1328: `bbb_reset_test_state`

**Filtering** (`nimcp_bbb_filtering.c` - TODO, 150 lines)
- Forward declarations for `bbb_validate_*` functions (implemented in `nimcp_bbb_input_gate.c`)
- Wrapper layer for input validation
- Lines 321-356: Internal stat update functions

**Audit** (`nimcp_bbb_audit.c` - TODO, 140 lines)
- Lines 719-790: `bbb_report_threat`
- Lines 799-823: `bbb_get_threat_reports`
- Lines 832-844: `bbb_clear_threat_reports`
- Lines 1226-1248: `bbb_print_statistics`
- Lines 1257-1284: `bbb_print_threat_report`
- Lines 640-672: `bbb_forward_threat_to_immune`

**Quarantine** (`nimcp_bbb_quarantine.c` - TODO, 180 lines)
- Lines 857-899: `bbb_is_quarantined`
- Lines 917-963: `bbb_is_quarantined_safe`
- Lines 978-1021: `bbb_release_quarantine_ref_for_region`
- Lines 1035-1059: `bbb_release_quarantine_ref` (deprecated)
- Lines 1068-1170: `bbb_quarantine_region`
- Lines 1179-1213: `bbb_release_quarantine`

### File 3: Capability Control (1,116 lines → 4 files)

Will require reading the full file first. Estimated split:
- **Core**: 180 lines (lifecycle, config, stats)
- **Permissions**: 250 lines (check, request, delegate, revoke)
- **Escalation**: 200 lines (detection, anomaly patterns)
- **Audit**: 180 lines (logging, trail, reports)

### File 4: Continuous Monitor (1,044 lines → 4 files)

Will require reading the full file first. Estimated split:
- **Core**: 200 lines (lifecycle, threading)
- **Checks**: 280 lines (health checks, anomaly detection)
- **Alerting**: 180 lines (alert generation, routing)
- **Reporting**: 150 lines (report generation, export)

### File 5: Bio Router (2,868 lines → 4 files)

Will require reading the full file first. Estimated split:
- **Core**: 400 lines (init, shutdown, worker threads, stats)
- **Routing**: 600 lines (route_message, dispatch, priorities)
- **Channels**: 500 lines (register/unregister, inbox/outbox)
- **Queue**: 450 lines (enqueue, dequeue, flow control)

## Build System Notes

**DO NOT MODIFY CMakeLists.txt** - Per instructions, we must NOT touch:
- `src/security/CMakeLists.txt`
- `src/async/CMakeLists.txt`
- `src/lib/CMakeLists.txt`

The build system will automatically discover and compile all `.c` files in these directories via:
```cmake
file(GLOB_RECURSE NIMCP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.c"
)
```

## Testing Strategy

Each split module gets focused unit tests:

**Example: test_corrigibility_shutdown.cpp**
```cpp
TEST(CorrigibilityShutdown, AcceptsFromOperator) {
    corrigibility_t* system = corrigibility_create(nullptr);
    bool accepted = false;
    EXPECT_EQ(NIMCP_SUCCESS,
        corrigibility_accept_shutdown(system, "operator", "test", &accepted));
    EXPECT_TRUE(accepted);
    corrigibility_destroy(system);
}

TEST(CorrigibilityShutdown, ZeroResistance) {
    corrigibility_t* system = corrigibility_create(nullptr);
    float resistance = 1.0f;
    EXPECT_EQ(NIMCP_SUCCESS,
        corrigibility_verify_no_shutdown_resistance(system, &resistance));
    EXPECT_EQ(0.0f, resistance);
    corrigibility_destroy(system);
}
```

## Next Steps

1. ✅ Create internal headers for remaining modules
2. ✅ Extract functions into split files (maintain line-for-line logic)
3. ✅ Create focused unit tests for each split module
4. ✅ Build and verify: `cd build && cmake .. && make nimcp -j4`
5. ✅ Run regression tests: `ctest -R "regression" -j3 --timeout 600`
6. ✅ Create commit with detailed changelog

## Verification Checklist

- [ ] All 26 split files created
- [ ] No changes to public headers (*.h in include/)
- [ ] No changes to CMakeLists.txt
- [ ] Build succeeds: `make nimcp -j4`
- [ ] All regression tests pass (472/472)
- [ ] NEW_FILES_MANIFEST.txt updated in src/security/ and src/async/
- [ ] Test coverage for critical paths in split modules

## Critical Constraints (from CLAUDE.md)

**NEVER use raw NIMCP_THROW_TO_IMMUNE in**:
- `src/utils/exception/*` (infinite recursion)
- `src/utils/memory/nimcp_memory.c` (use MEMORY_SAFE_THROW)
- `src/utils/memory/nimcp_unified_memory.c` (use UMM_SAFE_THROW)
- `src/security/nimcp_constant_time.c` (gate with `nimcp_exception_system_is_initialized()`)

**Guard Clause Pattern**: Both braces AND return required after NIMCP_THROW_TO_IMMUNE.

**Memory**: Use `nimcp_free`, NOT raw free. Use `nimcp_calloc`, NOT raw malloc.

## Timeline Estimate

- Corrigibility (2 files remaining): 30 min
- BBB (5 files + header): 1.5 hours
- Capability Control (4 files + header + read): 2 hours
- Continuous Monitor (4 files + header + read): 2 hours
- Bio Router (4 files + header + read): 3 hours
- Tests (11 files): 3 hours
- Build/verify/commit: 1 hour

**Total: ~13 hours**

Given context limits, recommend creating remaining files in batches with build verification between each batch.
