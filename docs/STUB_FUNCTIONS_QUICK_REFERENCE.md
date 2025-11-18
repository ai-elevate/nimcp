# NIMCP Stub Functions - Quick Reference

**Last Updated:** 2025-11-18

## Critical Stubs (Fix First)

### Network Analysis Module ⚠️ BLOCKING 14 TESTS
**File:** `src/cognitive/analysis/nimcp_network_analysis.c`

| Function | Line | Status | Fix Effort |
|----------|------|--------|------------|
| `network_analyzer_run()` | 100 | Returns fake data | HARD |
| `network_analyzer_detect_communities()` | 187 | Returns 3 fake communities | MODERATE |
| `network_analyzer_detect_hubs()` | 244 | Returns 5 fake hubs | MODERATE |
| `network_analyzer_compute_metrics()` | 283 | Returns hardcoded metrics | EASY |

**Fix:** Integrate existing Louvain and centrality algorithms.

---

## Medium Priority Stubs

### Hierarchical Brain Module
**File:** `src/lib/cognitive/nimcp_hierarchical.c`

| Function | Line | What's Missing | Effort |
|----------|------|----------------|--------|
| `hierarchical_forward()` | 373 | Forward pass logic | HARD |
| `hierarchical_get_output()` | 393 | Output extraction | MODERATE |
| `hierarchical_learn()` | 425 | Learning logic | HARD |
| `hierarchical_save()` | 760 | Serialization | MODERATE |
| `hierarchical_load()` | 771 | Deserialization | MODERATE |

---

### Astrocyte Network Integration
**File:** `src/glial/astrocytes/nimcp_astrocytes.c`

| Function | Line | What's Missing | Effort |
|----------|------|----------------|--------|
| `astrocyte_network_assign_synapses()` | 803 | Spatial synapse assignment | MODERATE |
| `astrocyte_network_step()` | 815 | Full network dynamics | MODERATE |

**Note:** Core astrocyte functions (calcium, glutamate, etc.) are FULLY IMPLEMENTED.

---

### Graph Metrics
**File:** `src/utils/algorithms/nimcp_graph_metrics.c`

| Function | Line | Issue | Effort |
|----------|------|-------|--------|
| `compute_graph_metrics()` | 434 | Uses trivial community assignment | EASY |

**Fix:** Call existing `louvain_detect_communities()` instead of stub.

---

## Low Priority (Backend Systems)

### Data I/O - Not Implemented Backends
**File:** `src/io/dataio/nimcp_dataio.c`

- PostgreSQL backend (line 405) - requires libpq
- JSON format (line 515) - requires cJSON
- Parquet format (line 519) - requires Apache Arrow
- SQLite format (line 533) - requires libsqlite3

### Replication - Not Implemented Backends
**File:** `src/networking/replication/nimcp_replication.c`

- Redis backend (line 652)
- PostgreSQL replication (line 662)

---

## Glial Cells (TDD RED Phase)

| Module | File | Status | Lines | Priority |
|--------|------|--------|-------|----------|
| Microglia | `src/glial/microglia/nimcp_microglia.c` | Stub | 400 | LOW |
| Oligodendrocytes | `src/glial/oligodendrocytes/nimcp_oligodendrocytes.c` | Stub | 405 | LOW |
| Integration | `src/glial/integration/nimcp_glial_integration.c` | Stub | - | MEDIUM |

---

## Partial Implementations

### Brain Module
**File:** `src/core/brain/nimcp_brain.c`

| Feature | Line | Status |
|---------|------|--------|
| JSON Export/Import | 11630 | Stub (returns minimal JSON) |
| Pretrained Model Load | 10601 | Works, but limited features |

### Neural Logic
**File:** `src/core/neuron_types/nimcp_neural_logic.c`

| Feature | Line | Status |
|---------|------|--------|
| Variable Binding | 993 | Partial (pattern storage incomplete) |

---

## Quick Stats

- **Total TODO/FIXME/STUB markers:** 164
- **Files with explicit stub implementations:** 7
- **Critical blocking stubs:** 4 functions (network analysis)
- **Backend systems not implemented:** 9
- **Test coverage:** 82% (673 passing tests)

---

## Recommended Fix Order

1. **Week 1:** Fix network analysis stubs → Unblock 14 tests
2. **Week 2:** Complete astrocyte integration
3. **Week 3:** Implement hierarchical brain logic
4. **Week 4:** Complete neural logic variable binding
5. **Later:** Implement backend systems as needed

---

## Tests Affected by Stubs

### Network Analysis (14 tests)
```
test/integration/core/brain/test_brain_community_detection.cpp
test/integration/core/brain/test_brain_community_integration.cpp
test/integration/test_network_analyzer_quantum_routing.cpp
test/integration/cognitive/test_network_analysis.cpp
test/regression/algorithms/test_community_detection_regression.cpp
test/regression/test_routing_efficiency_regression.cpp
test/regression/test_quantum_routing_efficiency.cpp
test/unit/core/brain/test_brain_network_analyzer.cpp
test/unit/utils/algorithms/test_centrality.cpp
test/unit/utils/algorithms/test_louvain.cpp
test/unit/utils/algorithms/test_community_detection.cpp
test/unit/utils/algorithms/test_modularity.cpp
test/unit/utils/quantum/test_quantum_adaptive_routing.cpp
test/unit/cognitive/analysis/test_network_analyzer_integration.cpp
```

---

## Files to Modify

### Critical
- `src/cognitive/analysis/nimcp_network_analysis.c`

### Important
- `src/lib/cognitive/nimcp_hierarchical.c`
- `src/glial/astrocytes/nimcp_astrocytes.c`
- `src/utils/algorithms/nimcp_graph_metrics.c`
- `src/core/neuron_types/nimcp_neural_logic.c`

### Optional
- All backend systems
- Glial cell implementations (microglia, oligodendrocytes)
- JSON export/import
- Training API wrapper
