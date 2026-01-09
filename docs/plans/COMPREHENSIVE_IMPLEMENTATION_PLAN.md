# NIMCP Comprehensive Implementation Plan
## Complete Feature Implementation with Full Integration

**Version**: 1.1
**Date**: 2026-01-09
**Status**: Planning Document
**Scope**: All incomplete features from Master Schedule, Advanced Concepts, and Brain Regions

---

## Executive Summary

This plan implements **ALL remaining incomplete features** across NIMCP with:
- **Bio-async messaging** for all inter-module communication
- **Bridge architecture** connecting every feature to the cognitive ecosystem
- **Immune system integration** for threat detection and self-healing
- **Brain factory initialization** for proper lifecycle management
- **Comprehensive logging** at all levels
- **Complete test coverage** (unit, integration, regression, e2e)
- **Intra-layer integration** for modules within the same architectural tier
- **Inter-layer integration** for modules across different architectural tiers

### Scope Summary

| Category | Items | Est. LOC | Tests |
|----------|-------|----------|-------|
| Advanced Concepts (Incomplete) | 9 concepts | ~45,000 | ~900 |
| Brain Regions (New) | 17 regions | ~85,000 | ~1,700 |
| Extrapolation (Remaining 15%) | 3 modules | ~4,500 | ~90 |
| Embodiment (Remaining 40%) | 4 modules | ~2,800 | ~56 |
| Superhuman (Remaining 50%) | 7 modules | ~15,000 | ~300 |
| Integration Layers (All) | 40 modules | ~274,000 | ~3,010 |
| **TOTAL** | **~40 modules + full integration** | **~426,300** | **~6,056** |

---

## Architecture Requirements

### 1. Bio-Async Integration Pattern

Every new module MUST implement:

```c
// Required bio-async message handlers
typedef struct {
    nimcp_bio_async_handler_t msg_handler;
    nimcp_bio_async_subscription_t* subscriptions;
    uint32_t subscription_count;
    nimcp_bio_router_t* router_ref;
} module_bio_async_t;

// Standard message types per module
typedef enum {
    MSG_TYPE_MODULE_STATE_UPDATE,
    MSG_TYPE_MODULE_REQUEST,
    MSG_TYPE_MODULE_RESPONSE,
    MSG_TYPE_MODULE_ERROR,
    MSG_TYPE_MODULE_METRICS
} module_msg_type_t;
```

### 2. Bridge Architecture Pattern

Every module requires these bridges:

| Bridge Type | Purpose | File Pattern |
|-------------|---------|--------------|
| SNN Bridge | Spiking neural network integration | `nimcp_<module>_snn_bridge.h/c` |
| Plasticity Bridge | Learning/adaptation | `nimcp_<module>_plasticity_bridge.h/c` |
| FEP Bridge | Free energy principle integration | `nimcp_<module>_fep_bridge.h/c` |
| Substrate Bridge | Bio-async messaging | `nimcp_<module>_substrate_bridge.h/c` |
| Thalamic Bridge | Attention gating | `nimcp_<module>_thalamic_bridge.h/c` |
| Immune Bridge | Threat detection/healing | `nimcp_<module>_immune_bridge.h/c` |
| Hub Bridge | Cognitive integration hub | `nimcp_<module>_hub_bridge.h/c` |

### 3. Immune System Integration Pattern

```c
// Every module must register with immune system
typedef struct {
    nimcp_immune_sensor_t sensor;
    nimcp_antibody_receptor_t receptor;
    nimcp_threat_callback_t on_threat;
    nimcp_healing_callback_t on_heal;
} module_immune_integration_t;

nimcp_status_t module_immune_register(
    module_t* module,
    nimcp_brain_immune_t* immune_system);
```

### 4. Brain Factory Integration Pattern

```c
// Every module must be initializable via brain factory
nimcp_status_t nimcp_brain_init_<module>(
    nimcp_brain_t* brain,
    const nimcp_brain_config_t* config);

// Factory registration
NIMCP_BRAIN_FACTORY_REGISTER(<module>, nimcp_brain_init_<module>);
```

### 5. Logging Integration Pattern

```c
// Standard logging macros for all modules
#define MODULE_LOG_TRACE(module, fmt, ...) \
    NIMCP_LOG_TRACE("[%s] " fmt, module->name, ##__VA_ARGS__)
#define MODULE_LOG_DEBUG(module, fmt, ...) \
    NIMCP_LOG_DEBUG("[%s] " fmt, module->name, ##__VA_ARGS__)
#define MODULE_LOG_INFO(module, fmt, ...) \
    NIMCP_LOG_INFO("[%s] " fmt, module->name, ##__VA_ARGS__)
#define MODULE_LOG_WARN(module, fmt, ...) \
    NIMCP_LOG_WARN("[%s] " fmt, module->name, ##__VA_ARGS__)
#define MODULE_LOG_ERROR(module, fmt, ...) \
    NIMCP_LOG_ERROR("[%s] " fmt, module->name, ##__VA_ARGS__)

// Metrics logging
#define MODULE_LOG_METRIC(module, metric, value) \
    nimcp_metrics_record(module->metrics, metric, value)
```

### 6. Intra-Layer Integration Pattern

**CRITICAL**: All modules within the same architectural layer MUST communicate bidirectionally.

#### 6.1 Layer Definitions

| Layer | Modules | Integration Type |
|-------|---------|-----------------|
| **Physics Layer** | Ephaptic, Info Geometry, HH Dynamics, Thermodynamics | Physical coupling |
| **Chemistry Layer** | pH, NO Signaling, Neurovascular | Chemical signaling |
| **Biology Layer** | Epigenetics, Neurogenesis, Gene Expression | Biological cascades |
| **Neuromodulatory Layer** | LC, VTA, Raphe, Habenula | Neuromodulator cross-talk |
| **Memory Layer** | Entorhinal, Perirhinal, Parahippocampal, Mammillary | Memory circuit |
| **Sensory Layer** | Somatosensory, Olfactory, Gustatory | Multi-sensory integration |
| **Executive Layer** | OFC, Retrosplenial, PFC | Executive coordination |
| **Integration Layer** | Claustrum, PAG, Red Nucleus, Reticular | Global binding |
| **Superhuman Layer** | Eagle Vision, Echolocation, Time Dilation, etc. | Enhanced perception |

#### 6.2 Intra-Layer Bridge Pattern

```c
/**
 * Every module must implement intra-layer bridges to ALL sibling modules
 * within its architectural layer.
 */

// Intra-layer integration structure
typedef struct {
    // Sibling module references (same layer)
    void** sibling_modules;
    uint32_t num_siblings;

    // Intra-layer message types
    nimcp_msg_type_t* intra_msg_types;

    // Bidirectional channels
    nimcp_bio_channel_t** intra_channels;

    // Synchronization
    nimcp_sync_barrier_t layer_barrier;
    float layer_coherence;  // 0-1 synchronization level

    // Layer-specific coordinator
    nimcp_layer_coordinator_t* coordinator;
} module_intra_layer_t;

// Required intra-layer API
nimcp_status_t module_intra_layer_init(
    module_t* module,
    nimcp_layer_t* layer);

nimcp_status_t module_intra_layer_sync(
    module_t* module,
    float* coherence);

nimcp_status_t module_intra_layer_broadcast(
    module_t* module,
    nimcp_msg_t* msg);

nimcp_status_t module_intra_layer_receive(
    module_t* module,
    nimcp_msg_t** msgs,
    uint32_t* count);
```

#### 6.3 Intra-Layer Integration Matrix

**Neuromodulatory Layer Example**:
```
              LC      VTA     Raphe   Habenula
    LC        -       ↔       ↔       ↔
    VTA       ↔       -       ↔       ←
    Raphe     ↔       ↔       -       ↔
    Habenula  ↔       →       ↔       -

Legend: ↔ = bidirectional, → = sends to, ← = receives from
```

**Memory Layer Example**:
```
              EC      PRC     PHC     MB
    EC        -       ↔       ↔       ↔
    PRC       ↔       -       ↔       ↔
    PHC       ↔       ↔       -       ↔
    MB        ↔       ↔       ↔       -

All connections bidirectional for memory circuit coherence
```

#### 6.4 Intra-Layer Bridge Files

For each layer, create:
- `include/integration/intra/<layer>/nimcp_<layer>_intra_coordinator.h`
- `src/integration/intra/<layer>/nimcp_<layer>_intra_coordinator.c`
- Individual cross-module bridges within the layer

**Example - Neuromodulatory Layer**:
```
include/integration/intra/neuromodulatory/
├── nimcp_neuromod_intra_coordinator.h
├── nimcp_lc_vta_bridge.h
├── nimcp_lc_raphe_bridge.h
├── nimcp_lc_habenula_bridge.h
├── nimcp_vta_raphe_bridge.h
├── nimcp_vta_habenula_bridge.h
└── nimcp_raphe_habenula_bridge.h

src/integration/intra/neuromodulatory/
├── nimcp_neuromod_intra_coordinator.c
├── nimcp_lc_vta_bridge.c
├── nimcp_lc_raphe_bridge.c
├── nimcp_lc_habenula_bridge.c
├── nimcp_vta_raphe_bridge.c
├── nimcp_vta_habenula_bridge.c
└── nimcp_raphe_habenula_bridge.c
```

### 7. Inter-Layer Integration Pattern

**CRITICAL**: All layers MUST communicate with adjacent and non-adjacent layers via defined pathways.

#### 7.1 Layer Hierarchy and Connections

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTER-LAYER INTEGRATION MAP                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                           │
│  │ SUPERHUMAN   │ ←──────────────────────────────────────────┐              │
│  │    LAYER     │                                             │              │
│  └──────┬───────┘                                             │              │
│         │ Enhanced perception feeds                           │              │
│         ↓                                                     │              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐  │              │
│  │ INTEGRATION  │ ←→  │  EXECUTIVE   │ ←→  │   MEMORY     │  │              │
│  │    LAYER     │     │    LAYER     │     │    LAYER     │  │              │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘  │              │
│         │                    │                    │           │              │
│         └────────────────────┼────────────────────┘           │              │
│                              │                                │              │
│                              ↓                                │              │
│  ┌──────────────┐     ┌──────────────┐                       │              │
│  │   SENSORY    │ ←→  │ NEUROMOD     │ ←─────────────────────┘              │
│  │    LAYER     │     │    LAYER     │                                       │
│  └──────┬───────┘     └──────┬───────┘                                       │
│         │                    │                                               │
│         └─────────┬──────────┘                                               │
│                   │                                                          │
│                   ↓                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   BIOLOGY    │ ←→  │  CHEMISTRY   │ ←→  │   PHYSICS    │                 │
│  │    LAYER     │     │    LAYER     │     │    LAYER     │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 7.2 Inter-Layer Bridge Pattern

```c
/**
 * Every module must implement inter-layer bridges to modules in
 * connected layers (both adjacent and non-adjacent as defined).
 */

// Inter-layer integration structure
typedef struct {
    // Layer identification
    nimcp_layer_id_t own_layer;
    nimcp_layer_id_t* connected_layers;
    uint32_t num_connected_layers;

    // Inter-layer channels (one per connected layer)
    nimcp_inter_layer_channel_t** channels;

    // Hierarchical routing
    nimcp_layer_router_t* layer_router;

    // Bottom-up signals (lower → higher layers)
    nimcp_signal_buffer_t bottom_up_buffer;

    // Top-down signals (higher → lower layers)
    nimcp_signal_buffer_t top_down_buffer;

    // Cross-layer synchronization
    float inter_layer_coherence;
    nimcp_phase_lock_t* phase_locks;
} module_inter_layer_t;

// Required inter-layer API
nimcp_status_t module_inter_layer_init(
    module_t* module,
    nimcp_layer_registry_t* registry);

nimcp_status_t module_inter_layer_send_bottom_up(
    module_t* module,
    nimcp_layer_id_t target_layer,
    nimcp_msg_t* msg);

nimcp_status_t module_inter_layer_send_top_down(
    module_t* module,
    nimcp_layer_id_t target_layer,
    nimcp_msg_t* msg);

nimcp_status_t module_inter_layer_receive(
    module_t* module,
    nimcp_layer_id_t source_layer,
    nimcp_msg_t** msgs,
    uint32_t* count);

nimcp_status_t module_inter_layer_sync_phase(
    module_t* module,
    nimcp_layer_id_t target_layer,
    float* phase_diff);
```

#### 7.3 Inter-Layer Connection Definitions

| Source Layer | Target Layer | Connection Type | Purpose |
|--------------|--------------|-----------------|---------|
| Physics | Chemistry | Bottom-up | Physical constraints on chemistry |
| Chemistry | Biology | Bottom-up | Chemical signals trigger biology |
| Biology | Neuromodulatory | Bottom-up | Gene expression affects neuromodulators |
| Neuromodulatory | Sensory | Bidirectional | Gain modulation |
| Neuromodulatory | Memory | Bidirectional | Plasticity modulation |
| Neuromodulatory | Executive | Bidirectional | Decision modulation |
| Sensory | Memory | Bottom-up | Sensory encoding |
| Sensory | Executive | Bottom-up | Sensory-driven decisions |
| Memory | Executive | Bidirectional | Memory-guided decisions |
| Memory | Integration | Bottom-up | Memory binding |
| Executive | Integration | Bidirectional | Executive control |
| Integration | Superhuman | Bidirectional | Enhanced integration |
| Superhuman | Neuromodulatory | Top-down | Enhanced perception modulates arousal |

#### 7.4 Inter-Layer Bridge Files

```
include/integration/inter/
├── nimcp_inter_layer_router.h          # Central inter-layer routing
├── nimcp_layer_registry.h              # Layer registration
├── nimcp_bottom_up_pathway.h           # Bottom-up signal handling
├── nimcp_top_down_pathway.h            # Top-down signal handling
│
├── physics_chemistry/
│   └── nimcp_physics_chemistry_bridge.h
├── chemistry_biology/
│   └── nimcp_chemistry_biology_bridge.h
├── biology_neuromod/
│   └── nimcp_biology_neuromod_bridge.h
├── neuromod_sensory/
│   └── nimcp_neuromod_sensory_bridge.h
├── neuromod_memory/
│   └── nimcp_neuromod_memory_bridge.h
├── neuromod_executive/
│   └── nimcp_neuromod_executive_bridge.h
├── sensory_memory/
│   └── nimcp_sensory_memory_bridge.h
├── sensory_executive/
│   └── nimcp_sensory_executive_bridge.h
├── memory_executive/
│   └── nimcp_memory_executive_bridge.h
├── memory_integration/
│   └── nimcp_memory_integration_bridge.h
├── executive_integration/
│   └── nimcp_executive_integration_bridge.h
├── integration_superhuman/
│   └── nimcp_integration_superhuman_bridge.h
└── superhuman_neuromod/
    └── nimcp_superhuman_neuromod_bridge.h

src/integration/inter/
├── nimcp_inter_layer_router.c
├── nimcp_layer_registry.c
├── nimcp_bottom_up_pathway.c
├── nimcp_top_down_pathway.c
└── [corresponding .c files for each bridge]
```

### 8. Layer Coordinator System

#### 8.1 Central Layer Coordinator

```c
/**
 * The Layer Coordinator manages all intra-layer and inter-layer
 * communication, ensuring coherent operation across the entire system.
 */

typedef struct {
    // All registered layers
    nimcp_layer_t* layers[NIMCP_MAX_LAYERS];
    uint32_t num_layers;

    // Inter-layer routing table
    nimcp_routing_table_t inter_layer_routes;

    // Global synchronization
    nimcp_global_sync_t global_sync;
    float global_coherence;

    // Bio-async integration
    nimcp_bio_router_t* bio_router;

    // Immune integration
    nimcp_brain_immune_t* immune_system;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_layer_coordinator_t;

// Layer Coordinator API
nimcp_status_t nimcp_layer_coordinator_init(
    nimcp_layer_coordinator_t* coord,
    nimcp_brain_t* brain);

nimcp_status_t nimcp_layer_coordinator_register_layer(
    nimcp_layer_coordinator_t* coord,
    nimcp_layer_t* layer);

nimcp_status_t nimcp_layer_coordinator_register_module(
    nimcp_layer_coordinator_t* coord,
    nimcp_layer_id_t layer_id,
    void* module,
    nimcp_module_interface_t* interface);

nimcp_status_t nimcp_layer_coordinator_update(
    nimcp_layer_coordinator_t* coord,
    float dt);

nimcp_status_t nimcp_layer_coordinator_route_message(
    nimcp_layer_coordinator_t* coord,
    nimcp_layer_id_t source,
    nimcp_layer_id_t target,
    nimcp_msg_t* msg);

float nimcp_layer_coordinator_get_coherence(
    nimcp_layer_coordinator_t* coord);
```

#### 8.2 Brain Factory Integration for Layer Coordinator

```c
// In nimcp_brain_init.c
nimcp_status_t nimcp_brain_init_layer_system(nimcp_brain_t* brain) {
    NIMCP_LOG_INFO("Initializing layer coordination system");

    // Initialize central coordinator
    nimcp_layer_coordinator_init(&brain->layer_coordinator, brain);

    // Register all layers
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->physics_layer);
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->chemistry_layer);
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->biology_layer);
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->neuromod_layer);
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->sensory_layer);
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->memory_layer);
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->executive_layer);
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->integration_layer);
    nimcp_layer_coordinator_register_layer(&brain->layer_coordinator,
                                            &brain->superhuman_layer);

    // Initialize inter-layer bridges
    nimcp_inter_layer_bridges_init(brain);

    // Initialize intra-layer bridges for each layer
    nimcp_intra_layer_bridges_init(brain);

    NIMCP_LOG_INFO("Layer system initialized with %d layers, %d inter-layer bridges",
                   brain->layer_coordinator.num_layers,
                   brain->layer_coordinator.inter_layer_routes.num_routes);

    return NIMCP_OK;
}
```

### 9. Integration Test Requirements

#### 9.1 Intra-Layer Integration Tests

Every layer MUST have comprehensive intra-layer integration tests:

| Layer | Test File | Test Count |
|-------|-----------|------------|
| Physics | `test/integration/intra/physics/test_physics_intra_integration.cpp` | 30 |
| Chemistry | `test/integration/intra/chemistry/test_chemistry_intra_integration.cpp` | 25 |
| Biology | `test/integration/intra/biology/test_biology_intra_integration.cpp` | 25 |
| Neuromodulatory | `test/integration/intra/neuromod/test_neuromod_intra_integration.cpp` | 40 |
| Sensory | `test/integration/intra/sensory/test_sensory_intra_integration.cpp` | 30 |
| Memory | `test/integration/intra/memory/test_memory_intra_integration.cpp` | 40 |
| Executive | `test/integration/intra/executive/test_executive_intra_integration.cpp` | 30 |
| Integration | `test/integration/intra/integration/test_integration_intra_integration.cpp` | 40 |
| Superhuman | `test/integration/intra/superhuman/test_superhuman_intra_integration.cpp` | 35 |
| **Total** | | **295** |

#### 9.2 Inter-Layer Integration Tests

Every inter-layer connection MUST have integration tests:

| Connection | Test File | Test Count |
|------------|-----------|------------|
| Physics↔Chemistry | `test/integration/inter/test_physics_chemistry_integration.cpp` | 20 |
| Chemistry↔Biology | `test/integration/inter/test_chemistry_biology_integration.cpp` | 20 |
| Biology↔Neuromod | `test/integration/inter/test_biology_neuromod_integration.cpp` | 20 |
| Neuromod↔Sensory | `test/integration/inter/test_neuromod_sensory_integration.cpp` | 25 |
| Neuromod↔Memory | `test/integration/inter/test_neuromod_memory_integration.cpp` | 25 |
| Neuromod↔Executive | `test/integration/inter/test_neuromod_executive_integration.cpp` | 25 |
| Sensory↔Memory | `test/integration/inter/test_sensory_memory_integration.cpp` | 25 |
| Sensory↔Executive | `test/integration/inter/test_sensory_executive_integration.cpp` | 20 |
| Memory↔Executive | `test/integration/inter/test_memory_executive_integration.cpp` | 30 |
| Memory↔Integration | `test/integration/inter/test_memory_integration_layer_integration.cpp` | 25 |
| Executive↔Integration | `test/integration/inter/test_executive_integration_layer_integration.cpp` | 25 |
| Integration↔Superhuman | `test/integration/inter/test_integration_superhuman_integration.cpp` | 20 |
| Superhuman↔Neuromod | `test/integration/inter/test_superhuman_neuromod_integration.cpp` | 20 |
| **Total** | | **300** |

#### 9.3 Full-Stack Integration Tests

End-to-end tests that verify complete layer traversal:

| Test | Description | Test Count |
|------|-------------|------------|
| Bottom-Up Full Stack | Physics → Integration | 15 |
| Top-Down Full Stack | Executive → Physics | 15 |
| Circular Integration | Full loop through all layers | 10 |
| Layer Coherence | Global synchronization | 10 |
| **Total** | | **50** |

### 10. Cognitive Layer Integration

**CRITICAL**: All new modules MUST integrate with the Cognitive Integration Hub for event-driven communication.

#### 10.1 Cognitive Hub Registration

Every module must register with `cognitive_integration_hub`:

```c
// Required cognitive hub integration
nimcp_status_t module_cognitive_hub_init(
    module_t* module,
    cognitive_integration_hub_t hub) {

    // Register module with hub
    cognitive_hub_register_module(hub,
                                   module->id,
                                   module->category,
                                   module->name,
                                   module);

    // Subscribe to relevant events
    cognitive_hub_subscribe(hub, module->id,
                            COGNITIVE_EVENT_ATTENTION_SHIFT,
                            module_on_attention_shift, module);
    cognitive_hub_subscribe(hub, module->id,
                            COGNITIVE_EVENT_MEMORY_CONSOLIDATION,
                            module_on_memory_consolidation, module);
    cognitive_hub_subscribe(hub, module->id,
                            COGNITIVE_EVENT_EXECUTIVE_DECISION,
                            module_on_executive_decision, module);

    // Register query handler
    cognitive_hub_register_query_handler(hub, module->id,
                                          module_query_handler);

    return NIMCP_OK;
}
```

#### 10.2 Cognitive Categories for New Modules

| Module | Cognitive Category | Events Subscribed | Events Published |
|--------|-------------------|-------------------|------------------|
| LC | COGNITIVE_CATEGORY_NEUROMOD | Attention, Stress | Arousal, Novelty |
| VTA | COGNITIVE_CATEGORY_REWARD | Action, Outcome | RPE, Motivation |
| Raphe | COGNITIVE_CATEGORY_NEUROMOD | Stress, Circadian | Mood, Patience |
| Habenula | COGNITIVE_CATEGORY_REWARD | Outcome, Expectation | Disappointment, Avoidance |
| Entorhinal | COGNITIVE_CATEGORY_MEMORY | Position, Sensory | Grid, Border |
| Claustrum | COGNITIVE_CATEGORY_INTEGRATION | All Categories | Binding, Salience |
| All Physics | COGNITIVE_CATEGORY_SUBSTRATE | State Changes | Biophysical Updates |

### 11. Omnidirectional Module Integration

**CRITICAL**: All new modules MUST integrate with the Omnidirectional inference system (JEPA bidirectional + predictive hierarchy).

#### 11.1 Omnidirectional Bridge Requirements

Every new module must implement an omnidirectional bridge:

```c
// Required omnidirectional integration structure
typedef struct {
    // JEPA bidirectional connection
    nimcp_bio_channel_t* jepa_channel;

    // Predictive hierarchy connection
    nimcp_bio_channel_t* pred_hierarchy_channel;

    // Hopfield memory connection (for pattern retrieval)
    nimcp_bio_channel_t* hopfield_channel;

    // Message handlers
    void (*on_omni_predict_request)(const nimcp_bio_msg_t* msg, void* ctx);
    void (*on_omni_predict_result)(const nimcp_bio_msg_t* msg, void* ctx);
    void (*on_pred_hier_forward)(const nimcp_bio_msg_t* msg, void* ctx);
    void (*on_pred_hier_backward)(const nimcp_bio_msg_t* msg, void* ctx);
    void (*on_pred_hier_error_propagate)(const nimcp_bio_msg_t* msg, void* ctx);

    // Precision weighting from free energy
    float precision_weight;
    float free_energy_estimate;
} module_omni_integration_t;

// Required omnidirectional API
nimcp_status_t module_omni_init(module_t* module, nimcp_brain_t* brain);
nimcp_status_t module_omni_forward_predict(module_t* module, const float* input);
nimcp_status_t module_omni_backward_infer(module_t* module, const float* target);
nimcp_status_t module_omni_update_precision(module_t* module, float precision);
```

#### 11.2 Omnidirectional Bridge Files per Module

Each new module requires:
- `include/cognitive/<module>/nimcp_<module>_omni_bridge.h`
- `src/cognitive/<module>/nimcp_<module>_omni_bridge.c`

#### 11.3 Omnidirectional Message Types

| Message Type | Direction | Purpose |
|--------------|-----------|---------|
| BIO_MSG_OMNI_PREDICT_REQUEST | Module → JEPA | Request forward prediction |
| BIO_MSG_OMNI_PREDICT_RESULT | JEPA → Module | Prediction result |
| BIO_MSG_OMNI_DIRECTION_SWITCH | Coordinator → All | Switch forward/backward mode |
| BIO_MSG_OMNI_PRECISION_UPDATE | FEP → Module | Update precision weighting |
| BIO_MSG_OMNI_FREE_ENERGY_REPORT | Module → FEP | Report free energy |
| BIO_MSG_PRED_HIER_FORWARD | Lower → Higher | Bottom-up prediction |
| BIO_MSG_PRED_HIER_BACKWARD | Higher → Lower | Top-down expectation |
| BIO_MSG_PRED_HIER_ERROR_PROPAGATE | Any → Any | Prediction error |

### 12. Hypothalamus Integration

**CRITICAL**: All new modules MUST integrate bidirectionally with the Hypothalamus for homeostatic regulation.

#### 12.1 Hypothalamus Bridge Requirements

The Hypothalamus is the master homeostatic regulator. Every new module must:

1. **Receive** circadian phase signals
2. **Receive** stress/cortisol signals
3. **Receive** autonomic state signals
4. **Send** energy/metabolic demands
5. **Send** homeostatic perturbations

```c
// Required hypothalamus integration
typedef struct {
    // Hypothalamus reference
    hypothalamus_adapter_t* hypothalamus;

    // Incoming signals
    float circadian_modulation;     // From SCN
    float cortisol_level;           // From HPA axis
    float autonomic_balance;        // Sympathetic/parasympathetic

    // Outgoing signals
    float metabolic_demand;         // Energy requirements
    float homeostatic_perturbation; // Effect on homeostasis

    // Bio-async channel
    nimcp_bio_channel_t* hypo_channel;

    // Message handlers
    void (*on_circadian_phase)(hypo_circadian_phase_t phase, void* ctx);
    void (*on_stress_response)(float cortisol, void* ctx);
    void (*on_homeostatic_alert)(uint32_t alert_type, float urgency, void* ctx);
} module_hypothalamus_integration_t;

// Required hypothalamus API per module
nimcp_status_t module_hypothalamus_init(module_t* module, nimcp_brain_t* brain);
nimcp_status_t module_hypothalamus_update(module_t* module);
float module_get_circadian_modulation(module_t* module);
float module_get_stress_modulation(module_t* module);
```

#### 12.2 Hypothalamus Bridge Files per Module

Each new module requires:
- `include/core/brain/regions/hypothalamus/nimcp_hypothalamus_<module>_bridge.h`
- `src/core/brain/regions/hypothalamus/nimcp_hypothalamus_<module>_bridge.c`

#### 12.3 Required Hypothalamus Connections

| Module | Hypothalamus Input | Hypothalamus Output |
|--------|-------------------|---------------------|
| Locus Coeruleus | Circadian, Stress | Arousal modulation |
| VTA | Circadian, Hunger | Reward sensitivity |
| Raphe | Circadian, Stress | Mood modulation |
| Habenula | Stress | Depression signal |
| Entorhinal | Circadian | Memory consolidation timing |
| All Sensory | Circadian, Autonomic | Sensory gain |
| All Executive | Stress, Circadian | Decision urgency |
| Claustrum | All | Global state broadcast |

#### 12.4 Hypothalamus Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/hypothalamus/test_hypo_neuromod_integration.cpp` | 30 |
| `test/integration/hypothalamus/test_hypo_memory_integration.cpp` | 20 |
| `test/integration/hypothalamus/test_hypo_sensory_integration.cpp` | 20 |
| `test/integration/hypothalamus/test_hypo_executive_integration.cpp` | 20 |
| `test/integration/hypothalamus/test_hypo_omnidirectional_integration.cpp` | 15 |
| **Total** | **105** |

### 13. Thalamic/Middleware Layer Integration

All modules must integrate with the thalamic system for attention gating and the middleware layer for routing, encoding, and training coordination.

#### 13.1 Thalamus Integration Requirements

The thalamus (`include/core/brain/subcortical/nimcp_thalamus.h`) provides sensory relay and cortical gating with:

- **Nuclei Types**: LGN (visual), MGN (auditory), VPL/VPM (somatosensory), VA/VL (motor), Pulvinar (attention), MD (executive), TRN (inhibitory gating)
- **Firing Modes**: Tonic (faithful relay), Burst (reduced attention), Inhibited (TRN suppression)
- **Relay Order**: First-order (subcortical input) vs Higher-order (cortical input)

```c
// Required thalamic integration per module
typedef struct {
    // Thalamic relay configuration
    thal_nucleus_type_t primary_nucleus;
    thal_relay_order_t relay_order;

    // Attention gating
    float attention_weight;
    float trn_inhibition_threshold;

    // Firing mode sensitivity
    float tonic_mode_gain;
    float burst_mode_gain;

    // Bio-async channel to thalamus
    nimcp_bio_channel_t* thalamus_channel;

    // Callbacks
    void (*on_attention_change)(float new_attention, void* ctx);
    void (*on_mode_change)(thal_firing_mode_t mode, void* ctx);
    void (*on_relay_complete)(const float* output, uint32_t size, void* ctx);
} module_thalamic_integration_t;

// Required thalamic API per module
nimcp_status_t module_thalamic_init(module_t* module, thalamus_t* thalamus);
nimcp_status_t module_thalamic_relay(module_t* module, const float* input, uint32_t size);
nimcp_status_t module_set_thalamic_attention(module_t* module, float attention);
float module_get_thalamic_gating(module_t* module);
```

#### 13.2 Middleware Layer Integration Requirements

The middleware layer provides routing, encoding, positional encoding, and training coordination:

| Middleware Component | Header | Purpose |
|---------------------|--------|---------|
| Thalamic Router | `middleware/thalamic/nimcp_thalamic_router.h` | Signal routing |
| Encoding Layer | `middleware/encoding/nimcp_encoding_layer.h` | Feature encoding |
| Positional Encoding | `middleware/encoding/nimcp_positional_encoding.h` | Position-aware processing |
| Training Middleware | `middleware/training/nimcp_training_middleware.h` | Learning coordination |
| Compression | `middleware/compression/nimcp_middleware_compression.h` | Data compression |

```c
// Required middleware integration per module
typedef struct {
    // Router integration
    nimcp_thalamic_router_t* router;
    uint32_t routing_priority;

    // Encoding integration
    nimcp_encoding_layer_t* encoder;
    uint32_t embedding_dim;

    // Positional encoding
    nimcp_positional_encoding_t* pos_encoder;
    bool use_rotary_encoding;

    // Training middleware
    nimcp_training_middleware_t* training_mw;
    bool training_enabled;

    // Compression (for large tensors)
    bool compression_enabled;
    float compression_ratio;
} module_middleware_integration_t;

// Required middleware API per module
nimcp_status_t module_middleware_init(module_t* module, nimcp_middleware_context_t* mw_ctx);
nimcp_status_t module_encode_input(module_t* module, const float* input, uint32_t size, float* encoded);
nimcp_status_t module_route_output(module_t* module, const float* output, uint32_t size, uint32_t target_id);
nimcp_status_t module_apply_positional_encoding(module_t* module, float* tensor, uint32_t seq_len);
```

#### 13.3 Thalamic Bridge Files per Module

Each new module requires:
- `include/cognitive/<module>/nimcp_<module>_thalamic_bridge.h`
- `src/cognitive/<module>/nimcp_<module>_thalamic_bridge.c`

#### 13.4 Required Thalamic Nucleus Connections

| Module | Primary Nucleus | Relay Order | Attention Gating |
|--------|----------------|-------------|------------------|
| Locus Coeruleus | Pulvinar | Higher | Global arousal |
| VTA | MD | Higher | Reward attention |
| Raphe | MD | Higher | Mood modulation |
| Habenula | MD | Higher | Aversive attention |
| Entorhinal | Anterior | Higher | Memory gating |
| Perirhinal | Pulvinar | Higher | Object attention |
| Parahippocampal | Pulvinar | Higher | Scene attention |
| Somatosensory | VPL/VPM | First | Tactile gating |
| Olfactory | MD | First | Smell gating |
| Gustatory | VPM | First | Taste gating |
| Orbitofrontal | MD | Higher | Value attention |
| Retrosplenial | Anterior | Higher | Spatial gating |
| Claustrum | Pulvinar | Higher | Global binding |
| PAG | Brainstem relay | First | Survival gating |
| All Visual | LGN | First | Visual attention |
| All Auditory | MGN | First | Auditory attention |
| All Motor | VA/VL | Higher | Action selection |

#### 13.5 Thalamic/Middleware Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/thalamic/test_thal_neuromod_integration.cpp` | 25 |
| `test/integration/thalamic/test_thal_memory_integration.cpp` | 20 |
| `test/integration/thalamic/test_thal_sensory_integration.cpp` | 25 |
| `test/integration/thalamic/test_thal_executive_integration.cpp` | 20 |
| `test/integration/middleware/test_mw_routing_integration.cpp` | 20 |
| `test/integration/middleware/test_mw_encoding_integration.cpp` | 15 |
| `test/integration/middleware/test_mw_training_integration.cpp` | 15 |
| **Total** | **140** |

### 14. Neural Substrate Layer Integration

All modules must integrate with the neural substrate layer for metabolic modeling, physical constraints, and biophysical simulation.

#### 14.1 Neural Substrate Integration Requirements

The neural substrate (`include/core/neural_substrate/nimcp_neural_substrate.h`) provides:

- **Metabolic State**: ATP level, oxygen saturation, glucose level
- **Physical State**: Temperature, membrane integrity, ion balance
- **Modulation Factors**: Firing rate, transmission efficiency, conduction velocity, plasticity capacity
- **Health Levels**: Optimal, Stressed, Compromised, Critical, Failing
- **Alerts**: Low ATP, Hypoxia, Hypoglycemia, Hyperthermia, Hypothermia, Ion imbalance, Membrane damage

```c
// Required neural substrate integration per module
typedef struct {
    // Substrate reference
    neural_substrate_t* substrate;

    // Module-specific metabolic costs
    float cost_per_activation;      // ATP cost per activation event
    float cost_per_spike;           // ATP cost per spike
    float cost_per_transmission;    // ATP cost per synaptic event
    float baseline_cost;            // Resting metabolic cost

    // Substrate modulation sensitivity
    float atp_sensitivity;          // How much ATP affects module
    float temperature_sensitivity;  // Q10 coefficient
    float oxygen_sensitivity;       // Hypoxia sensitivity

    // Module capacity thresholds
    float min_capacity_threshold;   // Below this, module enters low-power mode
    float critical_capacity;        // Below this, module shuts down

    // Bio-async channel to substrate
    nimcp_bio_channel_t* substrate_channel;

    // Callbacks
    void (*on_capacity_change)(float new_capacity, void* ctx);
    void (*on_alert)(substrate_alert_type_t alert, void* ctx);
    void (*on_health_change)(substrate_health_level_t health, void* ctx);
} module_substrate_integration_t;

// Required substrate API per module
nimcp_status_t module_substrate_init(module_t* module, neural_substrate_t* substrate);
nimcp_status_t module_record_activity(module_t* module, uint32_t spike_count, uint32_t transmission_count);
float module_get_substrate_modulation(module_t* module);
substrate_health_level_t module_get_substrate_health(module_t* module);
bool module_check_substrate_capacity(module_t* module);
```

#### 14.2 Substrate Bridge Files per Module

Each new module requires:
- `include/core/neural_substrate/bridges/nimcp_substrate_<module>_bridge.h`
- `src/core/neural_substrate/bridges/nimcp_substrate_<module>_bridge.c`

#### 14.3 Module-Specific Metabolic Parameters

| Module | Cost/Activation | ATP Sensitivity | Temperature Q10 | Min Capacity |
|--------|-----------------|-----------------|-----------------|--------------|
| Locus Coeruleus | 0.002 | High (0.9) | 2.5 | 0.4 |
| VTA | 0.003 | High (0.9) | 2.5 | 0.4 |
| Raphe | 0.002 | Medium (0.7) | 2.3 | 0.3 |
| Habenula | 0.001 | Medium (0.7) | 2.2 | 0.3 |
| Entorhinal | 0.005 | High (0.85) | 2.5 | 0.5 |
| Hippocampal regions | 0.006 | Very High (0.95) | 2.8 | 0.5 |
| Somatosensory | 0.004 | Medium (0.7) | 2.0 | 0.4 |
| Sensory regions | 0.003 | Medium (0.6) | 2.0 | 0.3 |
| Executive regions | 0.008 | Very High (0.95) | 2.5 | 0.5 |
| Claustrum | 0.007 | High (0.85) | 2.4 | 0.5 |
| PAG | 0.002 | Low (0.5) | 1.8 | 0.2 |

#### 14.4 Substrate Health Response per Module

| Health Level | Module Response |
|--------------|-----------------|
| Optimal | Full operation |
| Stressed | Reduce firing rate by 10%, increase energy efficiency |
| Compromised | Reduce firing rate by 30%, disable non-essential functions |
| Critical | Minimal operation, only essential signals |
| Failing | Emergency shutdown, broadcast distress signal |

#### 14.5 Neural Substrate Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/substrate/test_substrate_neuromod_integration.cpp` | 25 |
| `test/integration/substrate/test_substrate_memory_integration.cpp` | 20 |
| `test/integration/substrate/test_substrate_sensory_integration.cpp` | 20 |
| `test/integration/substrate/test_substrate_executive_integration.cpp` | 20 |
| `test/integration/substrate/test_substrate_metabolic_cascade.cpp` | 15 |
| `test/integration/substrate/test_substrate_health_recovery.cpp` | 15 |
| **Total** | **115** |

### 15. Motor Cortex Integration

All modules must integrate with the motor cortex for action output and sensorimotor coordination.

#### 15.1 Motor Cortex Integration Requirements

The motor cortex adapter (`include/core/brain/regions/motor/nimcp_motor_adapter.h`) provides:

- **Body Regions**: Face, hands, arms, trunk, legs, feet, eyes (somatotopic homunculus)
- **Movement Types**: Discrete, Serial, Continuous, Ballistic, Corrective
- **Processing Stages**: Planning, Preparing, Executing, Correcting
- **Motor Programs**: Stored learned movement sequences
- **Integration Points**: Basal ganglia (action selection), Cerebellum (coordination), Thalamus VA/VL (routing)

```c
// Required motor cortex integration per module
typedef struct {
    // Motor adapter reference
    motor_adapter_t* motor;

    // Module's motor influence
    bool can_initiate_movement;     // Module can request movements
    bool receives_motor_feedback;   // Module gets efference copy
    bool modulates_motor_vigor;     // Module affects movement amplitude

    // Supported body regions (bitmask)
    uint32_t influenced_regions;

    // Motor modulation parameters
    float vigor_modulation;         // How much module affects vigor [0-2]
    float timing_modulation;        // How much module affects timing [0.5-2]
    float precision_modulation;     // How much module affects precision [0-2]

    // Bio-async channel to motor
    nimcp_bio_channel_t* motor_channel;

    // Callbacks
    void (*on_movement_start)(uint32_t movement_id, motor_region_t region, void* ctx);
    void (*on_movement_complete)(const motor_result_t* result, void* ctx);
    void (*on_motor_feedback)(uint32_t effector_id, const motor_effector_state_t* state, void* ctx);
} module_motor_integration_t;

// Required motor API per module
nimcp_status_t module_motor_init(module_t* module, motor_adapter_t* motor);
nimcp_status_t module_request_movement(module_t* module, const motor_goal_t* goal);
nimcp_status_t module_modulate_vigor(module_t* module, float vigor_factor);
nimcp_status_t module_receive_efference_copy(module_t* module, const motor_command_t* cmd);
```

#### 15.2 Motor Bridge Files per Module

Each new module requires:
- `include/core/brain/regions/motor/bridges/nimcp_motor_<module>_bridge.h`
- `src/core/brain/regions/motor/bridges/nimcp_motor_<module>_bridge.c`

#### 15.3 Motor Cortex Connection Roles

| Module | Motor Role | Influenced Regions | Primary Function |
|--------|------------|-------------------|------------------|
| Locus Coeruleus | Arousal modulation | All | Global motor readiness |
| VTA | Vigor/motivation | All | Movement initiation |
| Raphe | Impulse control | All | Movement timing |
| Habenula | Inhibition | All | Movement suppression |
| Basal Ganglia | Action selection | All | Movement gating |
| Cerebellum | Coordination | All | Movement timing |
| Somatosensory | Proprioception | All | Sensory feedback |
| Entorhinal | Navigation | Legs, Eyes | Path integration |
| Orbitofrontal | Value-based action | All | Action value |
| Retrosplenial | Spatial action | Legs, Arms, Eyes | Oriented movement |
| PAG | Survival motor | All | Fight/flight response |
| Superior Colliculus | Eye movement | Eyes | Saccades |
| Claustrum | Global coordination | All | Multi-effector binding |

#### 15.4 Motor-Cognitive Feedback Loops

```c
// Efference copy integration (motor → cognitive)
typedef struct {
    // Predicted sensory consequence
    float* predicted_proprioception;
    float* predicted_visual_pos;

    // Motor command copy
    motor_command_t current_command;

    // Timing information
    double movement_onset_ms;
    double expected_duration_ms;

    // For sensory prediction
    bool expecting_reafference;
} motor_efference_copy_t;

// Corollary discharge pathway
nimcp_status_t motor_send_efference_copy(
    motor_adapter_t* motor,
    nimcp_bio_router_t* router,
    const motor_command_t* command
);

// Receiving module processes efference copy
nimcp_status_t module_process_efference_copy(
    module_t* module,
    const motor_efference_copy_t* copy
);
```

#### 15.5 Motor Cortex Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/motor/test_motor_neuromod_integration.cpp` | 25 |
| `test/integration/motor/test_motor_bg_integration.cpp` | 25 |
| `test/integration/motor/test_motor_cerebellum_integration.cpp` | 25 |
| `test/integration/motor/test_motor_sensory_integration.cpp` | 20 |
| `test/integration/motor/test_motor_cognitive_integration.cpp` | 20 |
| `test/integration/motor/test_motor_efference_copy.cpp` | 15 |
| `test/integration/motor/test_motor_planning_execution.cpp` | 20 |
| **Total** | **150** |

### 16. Portia Module Integration

All modules must integrate with the Portia adaptive intelligence system for resource-aware operation and graceful degradation.

#### 16.1 Portia System Overview

The Portia module (`include/portia/nimcp_portia.h`) provides Portia fimbriata-inspired adaptive intelligence:

- **Tier Manager**: Dynamic platform tier switching (FULL → MEDIUM → CONSTRAINED → MINIMAL)
- **Power Monitor**: Battery/energy awareness with state tracking
- **Resource Tracker**: CPU/memory/thermal monitoring
- **Degradation Controller**: Graceful feature reduction
- **Accelerator Detector**: GPU/NPU/TPU/FPGA discovery and offload
- **Sensor Fusion**: Multi-metric decision making
- **Planning Engine**: Strategic resource allocation

#### 16.2 Portia Integration Requirements

```c
// Required Portia integration per module
typedef struct {
    // Portia context reference
    portia_context_t* portia;

    // Module resource profile
    float cpu_weight;               // Module's CPU usage weight [0-1]
    float memory_weight;            // Module's memory usage weight [0-1]
    float power_weight;             // Module's power consumption weight [0-1]

    // Tier-dependent behavior
    bool supports_degradation;      // Module can operate at reduced capacity
    portia_degradation_level_t min_degradation;  // Minimum acceptable degradation
    portia_degradation_level_t max_degradation;  // Maximum tolerable degradation

    // Resource thresholds
    float min_cpu_required;         // Minimum CPU for operation
    float min_memory_required;      // Minimum memory for operation

    // Accelerator preferences
    portia_accelerator_type_t preferred_accelerator;
    bool can_offload;               // Module can offload to accelerator

    // Bio-async channel to Portia
    nimcp_bio_channel_t* portia_channel;

    // Callbacks
    void (*on_tier_change)(platform_tier_t new_tier, void* ctx);
    void (*on_degradation_change)(portia_degradation_level_t level, void* ctx);
    void (*on_power_state_change)(portia_power_state_t state, void* ctx);
    void (*on_thermal_alert)(portia_thermal_state_t state, void* ctx);
} module_portia_integration_t;

// Required Portia API per module
nimcp_status_t module_portia_init(module_t* module, portia_context_t* portia);
nimcp_status_t module_adapt_to_tier(module_t* module, platform_tier_t tier);
nimcp_status_t module_apply_degradation(module_t* module, portia_degradation_level_t level);
uint32_t module_get_recommended_neurons(module_t* module);
bool module_can_offload_to_accelerator(module_t* module, portia_accelerator_type_t accel);
```

#### 16.3 Portia Bridge Files per Module

Each new module requires:
- `include/portia/bridges/nimcp_portia_<module>_bridge.h`
- `src/portia/bridges/nimcp_portia_<module>_bridge.c`

#### 16.4 Module Degradation Profiles

| Module | Min Degradation | Max Degradation | Accelerator Support |
|--------|-----------------|-----------------|---------------------|
| Locus Coeruleus | NONE | MODERATE | CPU only |
| VTA | NONE | MODERATE | CPU only |
| Raphe | NONE | SEVERE | CPU only |
| Habenula | NONE | SEVERE | CPU only |
| Entorhinal | NONE | MODERATE | GPU optional |
| Hippocampal regions | NONE | MINOR | GPU preferred |
| Somatosensory | MINOR | SEVERE | CPU only |
| Sensory regions | MINOR | SEVERE | GPU optional |
| Executive regions | NONE | MODERATE | GPU optional |
| Claustrum | NONE | MINOR | GPU optional |
| PAG | NONE | EMERGENCY | CPU only (survival-critical) |
| All Memory | NONE | MODERATE | GPU preferred |
| All Visual | MINOR | SEVERE | GPU preferred |

#### 16.5 Tier-Dependent Module Behavior

| Tier | Neuron Scaling | Feature Set | Precision |
|------|---------------|-------------|-----------|
| FULL | 100% | All features | float32 |
| MEDIUM | 50-75% | Most features | float32 |
| CONSTRAINED | 25-50% | Essential features | float16/int8 |
| MINIMAL | 10-25% | Critical only | int8 |

#### 16.6 Portia Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/portia/test_portia_neuromod_integration.cpp` | 25 |
| `test/integration/portia/test_portia_memory_integration.cpp` | 20 |
| `test/integration/portia/test_portia_sensory_integration.cpp` | 20 |
| `test/integration/portia/test_portia_degradation_cascade.cpp` | 25 |
| `test/integration/portia/test_portia_tier_switching.cpp` | 20 |
| `test/integration/portia/test_portia_accelerator_offload.cpp` | 15 |
| **Total** | **125** |

### 17. Swarm Module Integration

All modules must integrate with the Swarm collective intelligence system for distributed cognition and emergent behavior.

#### 17.1 Swarm System Overview

The Swarm module (`include/swarm/nimcp_swarm_brain.h`) provides distributed cognitive processing:

- **Local Brain**: Resource-constrained NIMCP brain per agent
- **Signal Adapter**: Radio communication abstraction (LoRa, WiFi, etc.)
- **Collective Workspace**: Shared attention and goals
- **Emergence Detection**: Tier tracking (Disconnected → Superorganism)
- **Consensus Engine**: Voting and decision-making
- **Neuromodulator Sync**: Emotional state sharing across swarm
- **Bio-async Integration**: Internal message coordination

#### 17.2 Emergence Tiers

| Tier | Agents | Capability |
|------|--------|------------|
| TIER_0_DISCONNECTED | 1 | Solo operation |
| TIER_1_PAIRED | 2-3 | Basic coordination |
| TIER_2_CLUSTER | 4-7 | Group behaviors |
| TIER_3_SWARM | 8-15 | Emergent intelligence |
| TIER_4_SUPERORGANISM | 16+ | Full collective cognition |

#### 17.3 Swarm Integration Requirements

```c
// Required swarm integration per module
typedef struct {
    // Swarm brain reference
    swarm_brain_t* swarm;

    // Module's swarm role
    bool participates_in_consensus;     // Module votes on decisions
    bool shares_perception;             // Module broadcasts observations
    bool receives_threats;              // Module receives threat alerts
    bool syncs_neuromodulators;         // Module shares emotional state

    // Emergence-dependent behavior
    swarm_emergence_tier_t min_tier;    // Minimum tier for module activation
    float coherence_threshold;          // Required swarm coherence [0-1]

    // Collective workspace
    uint32_t workspace_concepts[8];     // Concepts this module attends to
    uint32_t concept_count;

    // Weight synchronization
    bool supports_weight_sync;          // Module can sync neural weights
    uint32_t sync_layers[8];            // Layers to synchronize
    uint32_t sync_layer_count;

    // Bio-async channel to swarm
    nimcp_bio_channel_t* swarm_channel;

    // Callbacks
    void (*on_emergence_change)(swarm_emergence_tier_t tier, void* ctx);
    void (*on_peer_perception)(const perception_data_t* data, void* ctx);
    void (*on_threat_alert)(const threat_data_t* threat, void* ctx);
    void (*on_consensus_result)(uint32_t proposal_id, vote_decision_t result, void* ctx);
    void (*on_neuromod_sync)(const neuromod_state_t* collective_state, void* ctx);
} module_swarm_integration_t;

// Required swarm API per module
nimcp_status_t module_swarm_init(module_t* module, swarm_brain_t* swarm);
nimcp_status_t module_broadcast_observation(module_t* module, const perception_data_t* data);
nimcp_status_t module_propose_action(module_t* module, const vote_proposal_t* proposal);
nimcp_status_t module_cast_vote(module_t* module, uint32_t proposal_id, vote_decision_t vote);
nimcp_status_t module_sync_weights_to_swarm(module_t* module);
swarm_emergence_tier_t module_get_swarm_tier(module_t* module);
```

#### 17.4 Swarm Bridge Files per Module

Each new module requires:
- `include/swarm/bridges/nimcp_swarm_<module>_bridge.h`
- `src/swarm/bridges/nimcp_swarm_<module>_bridge.c`

#### 17.5 Module Swarm Roles

| Module | Consensus | Perception Share | Threat Receive | Neuromod Sync |
|--------|-----------|------------------|----------------|---------------|
| Locus Coeruleus | No | No | Yes | Yes (arousal) |
| VTA | No | No | No | Yes (reward) |
| Raphe | No | No | Yes | Yes (mood) |
| Habenula | No | No | Yes | Yes (aversion) |
| Entorhinal | Yes | Yes (spatial) | Yes | No |
| Hippocampal regions | Yes | Yes (memory) | Yes | No |
| Somatosensory | No | Yes (touch) | Yes | No |
| Visual regions | No | Yes (vision) | Yes | No |
| Auditory regions | No | Yes (sound) | Yes | No |
| Executive regions | Yes | No | Yes | Yes |
| Claustrum | Yes | Yes (binding) | Yes | Yes |
| PAG | No | No | Yes | Yes (survival) |
| Orbitofrontal | Yes | No | No | Yes |

#### 17.6 Swarm Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/swarm/test_swarm_neuromod_integration.cpp` | 25 |
| `test/integration/swarm/test_swarm_memory_integration.cpp` | 20 |
| `test/integration/swarm/test_swarm_sensory_integration.cpp` | 25 |
| `test/integration/swarm/test_swarm_consensus_integration.cpp` | 25 |
| `test/integration/swarm/test_swarm_emergence_cascade.cpp` | 20 |
| `test/integration/swarm/test_swarm_weight_sync.cpp` | 15 |
| `test/integration/swarm/test_swarm_collective_learning.cpp` | 20 |
| **Total** | **150** |

### 18. Dragonfly Module Integration

All modules must integrate with the Dragonfly target tracking system for rapid pursuit and interception capabilities.

#### 18.1 Dragonfly System Overview

The Dragonfly module (`include/dragonfly/nimcp_dragonfly.h`) provides biological target tracking:

- **TSDN**: Target-Selective Descending Neurons (16-neuron population coding)
- **CSTMD1**: Centrifugal Small-Target Motion Detector (winner-take-all attention)
- **Prediction**: IMM filter for trajectory estimation
- **Interception**: Proportional Navigation guidance for optimal pursuit

#### 18.2 Operating Modes

| Mode | Description |
|------|-------------|
| IDLE | Waiting for targets |
| SCANNING | Actively scanning for targets |
| TRACKING | Locked onto target |
| PURSUING | Active pursuit |
| INTERCEPTING | Final interception phase |

#### 18.3 Dragonfly Integration Requirements

```c
// Required dragonfly integration per module
typedef struct {
    // Dragonfly system reference
    dragonfly_system_t* dragonfly;

    // Module's dragonfly role
    bool provides_detections;           // Module sends visual detections
    bool receives_motor_commands;       // Module receives pursuit commands
    bool modulates_tracking;            // Module affects tracking behavior
    bool receives_state_updates;        // Module gets tracking state

    // Detection parameters (if provides_detections)
    float min_contrast;                 // Minimum contrast for valid detection
    float max_distance;                 // Maximum detection distance
    uint32_t max_targets_per_frame;     // Max detections per update

    // Tracking modulation
    float attention_weight;             // How much module affects target selection
    float urgency_weight;               // How much module affects pursuit urgency

    // Energy awareness
    bool energy_aware_pursuit;          // Consider energy in pursuit decisions
    float min_energy_for_pursuit;       // Minimum energy to initiate pursuit

    // Bio-async channel to dragonfly
    nimcp_bio_channel_t* dragonfly_channel;

    // Callbacks
    void (*on_mode_change)(dragonfly_mode_t mode, void* ctx);
    void (*on_target_lock)(const dragonfly_target_info_t* target, void* ctx);
    void (*on_hunt_result)(dragonfly_hunt_result_t result, void* ctx);
    void (*on_motor_command)(const dragonfly_motor_cmd_t* cmd, void* ctx);
    void (*on_evasion_detected)(evasion_type_t evasion, void* ctx);
} module_dragonfly_integration_t;

// Required dragonfly API per module
nimcp_status_t module_dragonfly_init(module_t* module, dragonfly_system_t* dragonfly);
nimcp_status_t module_send_detection(module_t* module, const dragonfly_detection_t* detection);
nimcp_status_t module_modulate_tracking(module_t* module, float attention_mod, float urgency_mod);
dragonfly_mode_t module_get_dragonfly_mode(module_t* module);
nimcp_status_t module_request_pursuit_abort(module_t* module);
```

#### 18.4 Dragonfly Bridge Files per Module

Each new module requires:
- `include/dragonfly/bridges/nimcp_dragonfly_<module>_bridge.h`
- `src/dragonfly/bridges/nimcp_dragonfly_<module>_bridge.c`

#### 18.5 Module Dragonfly Roles

| Module | Detections | Motor Cmds | Tracking Mod | State Updates |
|--------|------------|------------|--------------|---------------|
| Visual Cortex (V1-V5) | Yes | No | Yes | Yes |
| Superior Colliculus | Yes | Yes | Yes | Yes |
| Locus Coeruleus | No | No | Yes (arousal) | Yes |
| VTA | No | No | Yes (motivation) | Yes |
| Habenula | No | No | Yes (inhibition) | Yes |
| Motor Cortex | No | Yes | No | Yes |
| Cerebellum | No | Yes | No | Yes |
| PAG | No | No | Yes (survival) | Yes |
| Orbitofrontal | No | No | Yes (value) | Yes |
| Attention System | No | No | Yes | Yes |
| Executive | No | No | Yes | Yes |

#### 18.6 Dragonfly-Cognitive Integration

```c
// Integration with cognitive systems
typedef struct {
    // Attention modulation
    float salience_boost;               // Boost target salience in attention system
    float distractor_suppression;       // Suppress non-targets

    // Emotional modulation
    float fear_response;                // Fear affects pursuit (e.g., predator detection)
    float reward_anticipation;          // Reward affects pursuit vigor

    // Executive control
    bool allow_pursuit_override;        // Executive can abort pursuit
    float abort_threshold;              // Confidence threshold for abort

    // Working memory
    bool track_target_history;          // Maintain target history in WM
    uint32_t max_history_targets;       // Maximum tracked target history
} dragonfly_cognitive_params_t;

// Bidirectional communication
nimcp_status_t dragonfly_receive_attention_modulation(
    dragonfly_system_t* dragonfly,
    const float* attention_weights,
    uint32_t weight_count
);

nimcp_status_t dragonfly_send_to_working_memory(
    dragonfly_system_t* dragonfly,
    const dragonfly_target_info_t* target
);
```

#### 18.7 Dragonfly Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/dragonfly/test_dragonfly_visual_integration.cpp` | 25 |
| `test/integration/dragonfly/test_dragonfly_motor_integration.cpp` | 25 |
| `test/integration/dragonfly/test_dragonfly_attention_integration.cpp` | 20 |
| `test/integration/dragonfly/test_dragonfly_executive_integration.cpp` | 20 |
| `test/integration/dragonfly/test_dragonfly_emotion_integration.cpp` | 15 |
| `test/integration/dragonfly/test_dragonfly_pursuit_pipeline.cpp` | 25 |
| **Total** | **130** |

### 19. Sleep Module Integration

All modules must integrate with the Sleep-Wake cycle system for memory consolidation, synaptic homeostasis, and energy management.

#### 19.1 Sleep System Overview

The Sleep module (`include/cognitive/nimcp_sleep_wake.h`) provides biologically-inspired sleep-wake cycling:

- **Sleep States**: AWAKE (Beta/Gamma 13-100Hz) → DROWSY (Alpha 8-13Hz) → LIGHT_NREM (Theta 4-8Hz) → DEEP_NREM (Delta 0.5-4Hz) → REM (Theta + atonia)
- **Sleep Pressure**: Adenosine accumulation model for metabolic cost of learning
- **Memory Consolidation**: Prioritized replay during deep sleep
- **Synaptic Homeostasis**: Weight downscaling (85%) and pruning (threshold 0.01)
- **REM Creativity**: Random activation for creative recombination

#### 19.2 Sleep State Characteristics

| State | Frequency | Duration | Function |
|-------|-----------|----------|----------|
| AWAKE | 13-100Hz (β/γ) | Active | Normal processing |
| DROWSY | 8-13Hz (α) | 2 min | Transition, relaxation |
| LIGHT_NREM | 4-8Hz (θ) | 15 min | Memory sorting |
| DEEP_NREM | 0.5-4Hz (δ) | 30 min | Consolidation, homeostasis |
| REM | θ + atonia | 10 min | Creativity, emotional processing |

#### 19.3 Sleep Integration Requirements

```c
// Required sleep integration per module
typedef struct {
    // Sleep system reference
    sleep_system_t sleep;

    // Module's sleep behavior
    bool suspends_during_sleep;         // Module pauses in sleep states
    bool participates_in_replay;        // Module memories are replayed
    bool requires_homeostasis;          // Module needs synaptic scaling
    bool active_during_rem;             // Module participates in REM

    // State-dependent behavior
    sleep_state_t min_active_state;     // Minimum state for operation
    float awake_activity_level;         // Activity during awake [0-1]
    float drowsy_activity_level;        // Activity during drowsy [0-1]
    float nrem_activity_level;          // Activity during NREM [0-1]
    float rem_activity_level;           // Activity during REM [0-1]

    // Memory consolidation
    float consolidation_priority;       // Priority for replay [0-1]
    bool emotional_priority_boost;      // Emotional memories get priority
    uint32_t replay_batch_contribution; // Memories per replay batch

    // Synaptic homeostasis
    float downscaling_sensitivity;      // How much to downscale [0-1]
    float pruning_threshold;            // Threshold for synapse removal
    bool exempt_from_pruning;           // Critical connections exempt

    // Sleep pressure contribution
    float activity_pressure_rate;       // Pressure added per activation
    float learning_pressure_rate;       // Pressure added per learning step

    // Bio-async channel to sleep
    nimcp_bio_channel_t* sleep_channel;

    // Callbacks
    void (*on_state_change)(sleep_state_t new_state, void* ctx);
    void (*on_consolidation_start)(void* ctx);
    void (*on_consolidation_complete)(uint32_t memories_replayed, void* ctx);
    void (*on_homeostasis_complete)(const homeostasis_stats_t* stats, void* ctx);
} module_sleep_integration_t;

// Required sleep API per module
nimcp_status_t module_sleep_init(module_t* module, sleep_system_t sleep);
nimcp_status_t module_on_sleep_state_change(module_t* module, sleep_state_t state);
nimcp_status_t module_contribute_to_replay(module_t* module, void** memories, uint32_t* count);
nimcp_status_t module_apply_homeostasis(module_t* module, float scaling_factor, float pruning_threshold);
float module_get_sleep_pressure_contribution(module_t* module);
bool module_is_sleep_critical(module_t* module);  // Must stay active during sleep
```

#### 19.4 Sleep Bridge Files per Module

Each new module requires:
- `include/cognitive/<module>/nimcp_<module>_sleep_bridge.h`
- `src/cognitive/<module>/nimcp_<module>_sleep_bridge.c`

#### 19.5 Module Sleep Behavior Profiles

| Module | Suspends | Replay | Homeostasis | REM Active | Priority |
|--------|----------|--------|-------------|------------|----------|
| Locus Coeruleus | Partial | No | Yes | Yes (arousal) | N/A |
| VTA | Partial | No | Yes | Yes (reward) | N/A |
| Raphe | Partial | No | Yes | No | N/A |
| Habenula | Yes | No | Yes | No | N/A |
| Entorhinal | Yes | Yes | Yes | Yes | High (spatial) |
| Perirhinal | Yes | Yes | Yes | No | High (object) |
| Parahippocampal | Yes | Yes | Yes | Yes | High (context) |
| Hippocampal regions | No (active) | Yes | Yes | Yes | Highest |
| Somatosensory | Yes | Yes | Yes | No | Medium |
| Sensory regions | Yes | Yes | Yes | No | Medium |
| Executive regions | Yes | Yes | Yes | Yes | High |
| Claustrum | Partial | No | Yes | Yes | N/A |
| PAG | No (survival) | No | No | No | N/A |
| Orbitofrontal | Yes | Yes | Yes | Yes | High (value) |

#### 19.6 Sleep Cycle Pipeline

```c
// Sleep cycle integration flow
typedef struct {
    // Pre-sleep phase
    float pre_sleep_pressure;           // Pressure before sleep
    uint32_t awake_learning_steps;      // Learning during awake

    // Drowsy phase
    float oscillation_slowdown;         // Frequency reduction rate
    bool sensory_gating_active;         // Reduce sensory input

    // Light NREM phase
    uint32_t memories_sorted;           // Memories sorted by importance
    float sorting_threshold;            // Threshold for keeping memory

    // Deep NREM phase (consolidation)
    uint32_t total_replays;             // Total replay events
    uint32_t emotional_replays;         // Emotional memory replays
    uint32_t novel_replays;             // Novel memory replays
    float consolidation_efficiency;     // Success rate [0-1]

    // Synaptic homeostasis (during deep NREM)
    float total_weight_before;          // Pre-scaling total
    float total_weight_after;           // Post-scaling total
    uint32_t synapses_pruned;           // Removed weak synapses
    float energy_saved;                 // Estimated energy savings

    // REM phase
    uint32_t creative_activations;      // Random activations
    uint32_t novel_associations;        // New associations formed
    float emotional_processing;         // Emotional regulation

    // Post-wake
    float post_wake_pressure;           // Pressure after sleep
    float cognitive_refresh;            // Performance improvement
} sleep_cycle_metrics_t;

// Module participation in sleep phases
nimcp_status_t module_participate_in_drowsy(module_t* module);
nimcp_status_t module_participate_in_light_nrem(module_t* module, float* sort_scores, uint32_t count);
nimcp_status_t module_participate_in_deep_nrem(module_t* module, const void** replay_memories, uint32_t count);
nimcp_status_t module_participate_in_rem(module_t* module, float creativity_noise);
```

#### 19.7 Cross-System Sleep Coordination

```c
// Sleep coordination with other systems
typedef struct {
    // Hypothalamus coordination (circadian)
    bool sync_to_circadian;             // Align sleep with circadian rhythm
    float circadian_sleep_drive;        // Circadian contribution to sleep pressure

    // Immune system coordination
    bool immune_enhanced_during_sleep;  // Boost immune during sleep
    float cytokine_modulation;          // Sleep affects immune signaling

    // Portia coordination (power management)
    bool reduce_power_during_sleep;     // Lower power consumption
    platform_tier_t sleep_tier;         // Tier during sleep (typically CONSTRAINED)

    // Swarm coordination (collective sleep)
    bool sync_swarm_sleep;              // Coordinate sleep across swarm
    float swarm_sleep_coherence;        // How synchronized sleep is

    // Dragonfly coordination
    bool disable_pursuit_during_sleep;  // No hunting while asleep
    dragonfly_mode_t wake_on_threat;    // Wake if threat detected
} sleep_coordination_t;
```

#### 19.8 Sleep Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/sleep/test_sleep_neuromod_integration.cpp` | 25 |
| `test/integration/sleep/test_sleep_memory_integration.cpp` | 30 |
| `test/integration/sleep/test_sleep_consolidation_pipeline.cpp` | 25 |
| `test/integration/sleep/test_sleep_homeostasis_integration.cpp` | 25 |
| `test/integration/sleep/test_sleep_rem_integration.cpp` | 20 |
| `test/integration/sleep/test_sleep_circadian_integration.cpp` | 15 |
| `test/integration/sleep/test_sleep_immune_integration.cpp` | 15 |
| `test/integration/sleep/test_sleep_portia_integration.cpp` | 15 |
| **Total** | **170** |

### 20. Glial Module Integration

Every module must integrate with the glial system (astrocytes, microglia, oligodendrocytes) for:
- **Synaptic modulation** via astrocyte tripartite synapse support
- **Homeostatic plasticity** via astrocyte BCM threshold modulation
- **Activity-dependent pruning** via microglia complement cascade (C1q/C3)
- **Immune signaling** via microglia cytokines (IL-1β, TNF-α, IL-6, IL-10, TGF-β)
- **Conduction optimization** via oligodendrocyte myelination and G-ratio tuning

**Est. LOC**: ~22,000 | **Tests**: ~190

#### 20.1 Astrocyte Integration Pattern

```c
// Astrocyte integration for tripartite synapse support
typedef struct {
    // Calcium dynamics integration
    float intracellular_calcium;        // μM (resting ~0.1, active ~1.0)
    float ip3_concentration;            // IP3 for calcium release
    bool calcium_wave_active;           // Propagating wave state

    // Glutamate/D-serine release
    float glutamate_released;           // mM released from astrocyte
    float d_serine_concentration;       // NMDA co-agonist level
    float gliotransmitter_threshold;    // Calcium threshold for release

    // Synaptic modulation
    float synaptic_gain_modulation;     // Multiplicative factor [0.5-2.0]
    float extrasynaptic_glutamate;      // Spillover glutamate
    float perisynaptic_coverage;        // % of synapse covered [0-1]

    // BCM threshold modulation
    float bcm_threshold;                // Sliding threshold for plasticity
    float bcm_modulation_rate;          // How fast threshold adapts
    bool enable_homeostatic_plasticity; // Use astrocyte BCM control

    // Metabolic support
    float glucose_delivery_rate;        // mmol/s to neurons
    float lactate_shuttle_rate;         // Lactate to neurons
    float atp_support_level;            // ATP provided [0-1]

    // Network reference
    astrocyte_network_t* astro_network; // Parent network
    uint32_t assigned_astrocyte_id;     // Specific astrocyte
} module_astrocyte_integration_t;

// Module astrocyte callbacks
typedef void (*astrocyte_calcium_callback_t)(void* module, float calcium, bool wave_active);
typedef void (*astrocyte_gliotransmitter_callback_t)(void* module, float glutamate, float d_serine);
typedef void (*astrocyte_bcm_callback_t)(void* module, float new_threshold);

// Integration functions
nimcp_status_t module_register_with_astrocyte(
    void* module,
    astrocyte_network_t* network,
    const module_astrocyte_integration_t* config);

nimcp_status_t module_request_calcium_wave(
    void* module,
    float propagation_speed,      // μm/s (typical: 15-25)
    float wave_amplitude);        // Peak calcium [μM]

nimcp_status_t module_receive_gliotransmitter(
    void* module,
    float* glutamate_out,
    float* d_serine_out);

nimcp_status_t module_get_bcm_threshold(
    void* module,
    float* threshold_out);
```

#### 20.2 Microglia Integration Pattern

```c
// Microglia integration for immune surveillance and synaptic pruning
typedef struct {
    // State tracking
    microglia_state_t current_state;    // RAMIFIED, ACTIVATED, PHAGOCYTIC
    float activation_level;             // [0-1] activation intensity
    float surveillance_rate;            // Process extension frequency

    // Synaptic pruning (complement cascade)
    bool c1q_tagged;                    // "Eat me" signal present
    bool c3_opsonized;                  // Ready for phagocytosis
    float pruning_probability;          // [0-1] chance of pruning
    uint32_t synapses_tagged;           // Count of tagged synapses
    uint32_t synapses_pruned;           // Count of pruned synapses

    // Cytokine signaling
    float il1_beta;                     // Pro-inflammatory (pg/mL)
    float tnf_alpha;                    // Pro-inflammatory (pg/mL)
    float il6;                          // Dual-role (pg/mL)
    float il10;                         // Anti-inflammatory (pg/mL)
    float tgf_beta;                     // Neuroprotective (pg/mL)

    // Neuroprotection
    bool neuroprotective_mode;          // Anti-inflammatory phenotype
    float bdnf_secretion;               // Neurotrophic factor level
    float phagocytic_activity;          // Debris clearance rate

    // Network reference
    microglia_network_t* microglia_net; // Parent network
    uint32_t patrolling_region_id;      // Region being monitored
} module_microglia_integration_t;

// Microglia callbacks
typedef void (*microglia_state_callback_t)(void* module, microglia_state_t new_state);
typedef void (*microglia_cytokine_callback_t)(void* module, const microglia_cytokine_state_t* cytokines);
typedef void (*microglia_pruning_callback_t)(void* module, uint32_t synapse_id, bool pruned);

// Integration functions
nimcp_status_t module_register_with_microglia(
    void* module,
    microglia_network_t* network,
    const module_microglia_integration_t* config);

nimcp_status_t module_request_synaptic_tagging(
    void* module,
    uint32_t synapse_id,
    float activity_level,         // Low activity increases C1q tagging
    float importance_score);      // High importance protects from pruning

nimcp_status_t module_receive_cytokine_signal(
    void* module,
    microglia_cytokine_state_t* cytokines_out);

nimcp_status_t module_request_debris_clearance(
    void* module,
    const void* debris_data,
    size_t debris_size);
```

#### 20.3 Oligodendrocyte Integration Pattern

```c
// Oligodendrocyte integration for myelination and conduction optimization
typedef struct {
    // Myelination status
    bool is_myelinated;                 // Whether axon has myelin
    float myelin_thickness;             // μm (typical: 0.5-2.0)
    float g_ratio;                      // Axon/fiber ratio (optimal: 0.6-0.8)
    float internode_length;             // μm between nodes of Ranvier

    // Conduction properties
    float unmyelinated_velocity;        // m/s (0.5-2.0)
    float myelinated_velocity;          // m/s (50-100, 10-100x faster)
    float saltatory_efficiency;         // Energy savings from jumping
    uint32_t node_count;                // Nodes of Ranvier count

    // Metabolic support
    float lactate_provision;            // Energy to axon
    float metabolic_coupling;           // How tightly coupled [0-1]
    bool mitochondria_transfer;         // Transfer to neurons

    // Plasticity
    float activity_dependent_myelination; // Myelination based on activity
    float remyelination_rate;           // Recovery after damage
    float myelin_turnover;              // Natural replacement rate

    // Network reference
    oligodendrocyte_network_t* oligo_net; // Parent network
    uint32_t assigned_oligo_id;         // Specific oligodendrocyte
} module_oligodendrocyte_integration_t;

// Oligodendrocyte callbacks
typedef void (*oligo_myelination_callback_t)(void* module, bool myelinated, float g_ratio);
typedef void (*oligo_velocity_callback_t)(void* module, float new_velocity);
typedef void (*oligo_metabolic_callback_t)(void* module, float lactate_level);

// Integration functions
nimcp_status_t module_register_with_oligodendrocyte(
    void* module,
    oligodendrocyte_network_t* network,
    const module_oligodendrocyte_integration_t* config);

nimcp_status_t module_request_myelination(
    void* module,
    uint32_t axon_id,
    float activity_level);        // High activity promotes myelination

nimcp_status_t module_get_conduction_velocity(
    void* module,
    uint32_t axon_id,
    float* velocity_out);

nimcp_status_t module_optimize_g_ratio(
    void* module,
    uint32_t axon_id,
    float target_g_ratio);        // Optimize toward 0.6-0.8
```

#### 20.4 Tripartite Synapse Integration

```c
// Complete tripartite synapse model (neuron + astrocyte + microglia)
typedef struct {
    // Pre-synaptic neuron
    uint32_t presynaptic_neuron_id;
    float presynaptic_activity;
    float glutamate_release;

    // Post-synaptic neuron
    uint32_t postsynaptic_neuron_id;
    float postsynaptic_response;
    float synaptic_strength;

    // Astrocyte component
    uint32_t perisynaptic_astrocyte_id;
    float astrocyte_calcium;
    float gliotransmitter_release;
    float synaptic_gain;

    // Microglia surveillance
    uint32_t surveilling_microglia_id;
    bool tagged_for_pruning;
    float complement_level;

    // Oligodendrocyte (for axon health)
    uint32_t axon_oligo_id;
    float conduction_speed;

    // Integrated synapse dynamics
    float total_synaptic_efficacy;    // Combined modulation
    float long_term_stability;         // Survival probability
    float metabolic_cost;              // Energy requirements
} tripartite_synapse_t;

// Tripartite synapse operations
nimcp_status_t tripartite_synapse_create(
    glial_integration_t* glial,
    uint32_t pre_id,
    uint32_t post_id,
    tripartite_synapse_t* synapse_out);

nimcp_status_t tripartite_synapse_update(
    glial_integration_t* glial,
    tripartite_synapse_t* synapse,
    float dt);                         // Time step

nimcp_status_t tripartite_synapse_get_efficacy(
    const tripartite_synapse_t* synapse,
    float* efficacy_out);              // Combined strength
```

#### 20.5 Module Glial Behavior Profile

| Module | Astrocyte Role | Microglia Role | Oligo Role |
|--------|---------------|----------------|------------|
| **Attention** | BCM threshold for focus | Prune unused attention paths | Fast attention switching |
| **Memory** | Consolidation support | Prune weak memories | Quick memory retrieval |
| **Emotion** | Emotional calcium waves | Prune maladaptive circuits | Fast emotional response |
| **Reasoning** | Energy for computation | Prune inefficient paths | Fast logical chains |
| **Executive** | Sustained attention energy | Prune distractors | Fast executive commands |
| **Theory of Mind** | Social context calcium | Prune social errors | Fast social inference |
| **Ethics** | Value stability | Prune contradictions | Fast moral judgments |
| **Self Model** | Self-coherence | Prune self-inconsistencies | Fast self-updates |
| **Introspection** | Meta-awareness energy | Prune meta-errors | Fast introspective access |
| **Curiosity** | Novelty-driven waves | Prune stale interests | Fast curiosity signals |
| **Wellbeing** | Homeostatic balance | Clear stress markers | Stable wellbeing signals |
| **Salience** | Salience modulation | Prune low-salience | Fast salience detection |
| **Epistemic** | Confidence calibration | Prune overconfidence | Fast belief updates |
| **Meta Learning** | Learning rate modulation | Prune bad strategies | Fast strategy switching |
| **Mirror Neurons** | Imitation support | Prune poor matches | Fast mirror responses |
| **Bias Detection** | Bias awareness | Prune systematic errors | Fast bias detection |
| **Working Memory** | WM maintenance energy | Prune old WM items | Fast WM access |

#### 20.6 Glial Bio-Async Messages

```c
// Glial-specific bio-async message types
typedef enum {
    // Astrocyte messages
    BIO_MSG_ASTRO_CALCIUM_WAVE,         // Calcium wave propagation
    BIO_MSG_ASTRO_GLIOTRANSMITTER,      // Glutamate/D-serine release
    BIO_MSG_ASTRO_BCM_UPDATE,           // BCM threshold change
    BIO_MSG_ASTRO_METABOLIC_SUPPORT,    // Energy delivery
    BIO_MSG_ASTRO_NETWORK_SYNC,         // Astrocyte network synchrony

    // Microglia messages
    BIO_MSG_MICRO_STATE_CHANGE,         // Ramified/activated/phagocytic
    BIO_MSG_MICRO_C1Q_TAG,              // Synapse tagged for pruning
    BIO_MSG_MICRO_C3_OPSONIZE,          // Ready for phagocytosis
    BIO_MSG_MICRO_PRUNE_SYNAPSE,        // Pruning executed
    BIO_MSG_MICRO_CYTOKINE_RELEASE,     // IL-1β, TNF-α, IL-6, IL-10, TGF-β
    BIO_MSG_MICRO_DEBRIS_CLEARED,       // Cleanup complete

    // Oligodendrocyte messages
    BIO_MSG_OLIGO_MYELINATE,            // Myelination request
    BIO_MSG_OLIGO_MYELINATION_COMPLETE, // Myelination done
    BIO_MSG_OLIGO_G_RATIO_UPDATE,       // G-ratio optimization
    BIO_MSG_OLIGO_VELOCITY_CHANGE,      // Conduction speed update
    BIO_MSG_OLIGO_METABOLIC_SUPPORT,    // Lactate shuttle

    // Integrated glial messages
    BIO_MSG_GLIAL_TRIPARTITE_UPDATE,    // Full tripartite synapse update
    BIO_MSG_GLIAL_NETWORK_STATUS,       // Overall glial health
    BIO_MSG_GLIAL_INFLAMMATION,         // Inflammatory state
    BIO_MSG_GLIAL_NEUROPROTECTION       // Protective mode active
} glial_bio_msg_type_t;

// Message routing
nimcp_status_t glial_route_to_module(
    glial_integration_t* glial,
    void* module,
    glial_bio_msg_type_t msg_type,
    const void* payload,
    size_t payload_size);
```

#### 20.7 Glial-Sleep Interaction

```c
// Glial behavior during sleep states
typedef struct {
    // Awake state
    float awake_astro_activity;         // High - support active processing
    float awake_micro_surveillance;     // Moderate - routine monitoring
    float awake_oligo_activity;         // Normal myelination maintenance

    // Drowsy state
    float drowsy_astro_slowdown;        // Reduce calcium dynamics
    float drowsy_micro_reduction;       // Decrease surveillance

    // Light NREM state
    float light_nrem_astro_sorting;     // Help sort memories
    float light_nrem_micro_tagging;     // Tag synapses for review

    // Deep NREM state (critical for glial function!)
    float deep_nrem_astro_clearance;    // Glymphatic clearance boost
    float deep_nrem_micro_pruning;      // Maximum pruning activity
    float deep_nrem_oligo_repair;       // Myelin repair window

    // REM state
    float rem_astro_creativity;         // Support novel associations
    float rem_micro_quiescence;         // Reduce pruning
} glial_sleep_profile_t;

// Deep NREM is critical for glymphatic system!
// - Astrocytes shrink by ~60%, opening perivascular spaces
// - CSF flushes through, clearing metabolic waste
// - Amyloid-beta and tau clearance accelerated
```

#### 20.8 Glial Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/glial/test_astrocyte_module_integration.cpp` | 30 |
| `test/integration/glial/test_microglia_module_integration.cpp` | 30 |
| `test/integration/glial/test_oligodendrocyte_module_integration.cpp` | 25 |
| `test/integration/glial/test_tripartite_synapse_integration.cpp` | 25 |
| `test/integration/glial/test_glial_bioasync_integration.cpp` | 25 |
| `test/integration/glial/test_glial_sleep_integration.cpp` | 20 |
| `test/integration/glial/test_glial_immune_integration.cpp` | 20 |
| `test/integration/glial/test_glial_plasticity_integration.cpp` | 15 |
| **Total** | **190** |

### 21. Quantum Module Integration

Every module must integrate with the quantum subsystem for:
- **Quantum Walk** for accelerated neuromodulator diffusion (√N speedup)
- **Quantum Annealing** for escaping local minima in weight optimization
- **Quantum Reasoning** for Grover-inspired inference and SAT solving
- **Quantum Monte Carlo** for efficient sampling and measurement
- **Quantum Consensus** for swarm decision-making (Byzantine fault-tolerant)

**Est. LOC**: ~24,000 | **Tests**: ~210

#### 21.1 Quantum Walk Integration Pattern

```c
// Quantum walk integration for neuromodulator diffusion
typedef struct {
    // Quantum walker configuration
    quantum_walk_config_t walker_config;
    quantum_walker_t* walker;               // Active walker instance

    // Neuromodulator targets
    bool enable_dopamine_diffusion;         // Use QW for dopamine
    bool enable_serotonin_diffusion;        // Use QW for serotonin
    bool enable_acetylcholine_diffusion;    // Use QW for ACh
    bool enable_norepinephrine_diffusion;   // Use QW for NE

    // Quantum walk parameters
    quantum_coin_type_t coin_type;          // HADAMARD, GROVER, FOURIER, IDENTITY
    uint32_t walk_steps;                    // Evolution steps
    float hybrid_mixing;                    // Quantum-classical mixing [0-1]
    float decoherence_rate;                 // Rate of quantum → classical

    // Performance metrics
    float speedup_achieved;                 // Measured speedup vs classical
    float spreading_efficiency;             // Distance/steps ratio
    uint64_t total_walk_steps;              // Total evolution steps

    // Network reference
    neural_network_t network;               // Neural network graph
} module_quantum_walk_integration_t;

// Module quantum walk callbacks
typedef void (*quantum_diffusion_callback_t)(void* module, const float* concentration, uint32_t num_nodes);
typedef void (*quantum_measurement_callback_t)(void* module, uint32_t measured_node);

// Integration functions
nimcp_status_t module_init_quantum_walk(
    void* module,
    neural_network_t network,
    const quantum_walk_config_t* config);

nimcp_status_t module_diffuse_neuromodulator(
    void* module,
    uint32_t source_node,
    neuromodulator_type_t neuromod,
    float* concentration_out);

nimcp_status_t module_quantum_walk_step(
    void* module,
    uint32_t num_steps);

nimcp_status_t module_quantum_walk_measure(
    void* module,
    uint32_t* result_node);
```

#### 21.2 Quantum Annealing Integration Pattern

```c
// Quantum annealing integration for weight optimization
typedef struct {
    // Annealer configuration
    quantum_annealing_config_t anneal_config;
    quantum_annealer_t annealer;            // Active annealer instance

    // Optimization targets
    bool enable_weight_optimization;         // Optimize synaptic weights
    bool enable_threshold_optimization;      // Optimize neuron thresholds
    bool enable_structure_optimization;      // Optimize connectivity

    // Annealing parameters
    cooling_schedule_t cooling_schedule;     // EXPONENTIAL, LINEAR, LOGARITHMIC, ADAPTIVE
    float initial_temperature;               // Starting exploration (typical: 1.0)
    float final_temperature;                 // Final exploitation (typical: 0.01)
    uint32_t num_iterations;                 // Annealing steps
    float quantum_strength;                  // Tunneling probability [0-1]
    bool enable_tunneling;                   // Quantum tunneling enabled

    // Trigger conditions
    uint32_t optimization_interval;          // Steps between optimizations
    float loss_plateau_threshold;            // Trigger on loss plateau
    float gradient_noise_threshold;          // Trigger on gradient noise

    // Results
    float best_energy_found;                 // Lowest energy achieved
    float improvement_over_classical;        // % improvement vs SA
    uint32_t tunneling_events;               // Count of successful tunnels
} module_quantum_annealing_integration_t;

// Energy function for module-specific optimization
typedef float (*module_energy_fn_t)(const float* weights, uint32_t dim, void* module);

// Integration functions
nimcp_status_t module_init_quantum_annealer(
    void* module,
    const quantum_annealing_config_t* config);

nimcp_status_t module_optimize_weights_quantum(
    void* module,
    const float* initial_weights,
    float* optimized_weights,
    uint32_t num_weights,
    module_energy_fn_t energy_fn);

nimcp_status_t module_should_anneal(
    void* module,
    float current_loss,
    float gradient_norm,
    bool* should_anneal_out);
```

#### 21.3 Quantum Reasoning Integration Pattern

```c
// Quantum reasoning integration for logical inference
typedef struct {
    // Reasoner configuration
    qreason_config_t reason_config;
    qreason_t reasoner;                      // Active reasoner instance

    // Inference modes
    bool enable_sat_solving;                 // Grover-inspired SAT
    bool enable_forward_chaining;            // Rule-based inference
    bool enable_ternary_logic;               // 3-valued logic
    bool enable_belief_propagation;          // Ternary BP

    // Grover parameters
    uint32_t max_grover_iterations;          // Grover iteration limit
    bool use_classical_fallback;             // Fall back to DPLL for large problems
    uint32_t quantum_variable_limit;         // Max vars for quantum (16)

    // Ternary belief propagation
    ternary_bp_config_t bp_config;
    float believe_threshold;                 // >= this is BELIEVE (0.7)
    float disbelieve_threshold;              // <= this is DISBELIEVE (0.3)

    // Knowledge base
    uint32_t num_facts;                      // Facts in KB
    uint32_t num_rules;                      // Rules in KB
    float min_confidence;                    // Minimum confidence threshold

    // Statistics
    uint64_t queries_performed;
    uint64_t satisfiable_count;
    float avg_grover_iterations;
} module_quantum_reasoning_integration_t;

// Integration functions
nimcp_status_t module_init_quantum_reasoner(
    void* module,
    const qreason_config_t* config);

nimcp_status_t module_add_fact(
    void* module,
    uint32_t variable,
    qreason_truth_t value,
    float confidence);

nimcp_status_t module_add_rule(
    void* module,
    const uint32_t* antecedents,
    uint32_t n_antecedents,
    uint32_t consequent,
    float confidence);

nimcp_status_t module_solve_sat(
    void* module,
    const qreason_cnf_t* cnf,
    qreason_result_t* result);

nimcp_status_t module_query_belief(
    void* module,
    uint32_t variable,
    qreason_truth_t* value_out,
    float* confidence_out);
```

#### 21.4 Quantum Monte Carlo Integration Pattern

```c
// Quantum Monte Carlo integration for sampling
typedef struct {
    // QMC configuration
    uint32_t default_num_shots;              // Finite measurement shots
    uint32_t default_amplitude_samples;      // Amplitude estimation samples
    uint32_t default_burnin;                 // MCMC burn-in period
    float target_acceptance_rate;            // Adaptive M-H target (0.234)

    // Sampling modes
    bool enable_importance_sampling;         // Use importance sampling
    bool enable_stratified_sampling;         // Use stratified sampling
    bool enable_adaptive_proposals;          // Learn proposal distribution

    // Entropy estimation
    bool enable_entropy_estimation;          // Estimate Shannon entropy
    bool enable_renyi_entropy;               // Estimate Rényi entropy
    uint32_t entropy_samples;                // Samples for entropy

    // Partition function
    bool enable_partition_estimation;        // Estimate Z
    float partition_temperature;             // Temperature for Z
    float partition_variance;                // Variance of Z estimate
} module_quantum_monte_carlo_integration_t;

// Integration functions
nimcp_status_t module_init_quantum_mc(
    void* module,
    const module_quantum_monte_carlo_integration_t* config);

nimcp_status_t module_sample_quantum_state(
    void* module,
    const float* amplitudes,
    uint32_t num_states,
    uint32_t num_shots,
    qmc_measurement_result_t* result);

nimcp_status_t module_estimate_entropy(
    void* module,
    const float* probabilities,
    uint32_t num_states,
    qmc_entropy_result_t* result);

nimcp_status_t module_adaptive_anneal(
    void* module,
    qmc_energy_fn energy_fn,
    const float* initial_state,
    float* optimized_state,
    uint32_t dim,
    qmc_anneal_result_t* result);
```

#### 21.5 Quantum Consensus Integration Pattern

```c
// Quantum consensus integration for swarm voting
typedef struct {
    // Consensus configuration
    quantum_consensus_config_t consensus_config;
    quantum_consensus_t consensus;           // Active consensus instance

    // Algorithm selection
    quantum_consensus_algo_t algorithm;      // GROVER, ANNEALING, WALK, HYBRID
    uint32_t grover_iterations;              // Grover iterations (0=auto √N)
    float agreement_threshold;               // Pass threshold (0.667)

    // Byzantine fault tolerance
    float bft_threshold;                     // Fault tolerance (0.333)
    bool enable_collusion_detection;         // Detect voting collusion
    bool enable_amplitude_weighting;         // Weight by confidence

    // Vote states (ternary)
    // AGREE (+1), ABSTAIN (0), DISAGREE (-1)
    float min_vote_confidence;               // Minimum confidence to count

    // Statistics
    uint64_t total_proposals;
    uint64_t proposals_passed;
    float avg_convergence_rounds;
    uint64_t collusions_detected;
} module_quantum_consensus_integration_t;

// Integration functions
nimcp_status_t module_init_quantum_consensus(
    void* module,
    const quantum_consensus_config_t* config);

nimcp_status_t module_create_proposal(
    void* module,
    const char* topic,
    float value,
    uint32_t* proposal_id_out);

nimcp_status_t module_cast_vote(
    void* module,
    uint32_t proposal_id,
    quantum_vote_choice_t choice,
    float confidence);

nimcp_status_t module_run_consensus(
    void* module,
    uint32_t proposal_id,
    quantum_consensus_result_t* result);
```

#### 21.6 Module Quantum Behavior Profile

| Module | Quantum Walk | Quantum Annealing | Quantum Reasoning | Quantum MC | Quantum Consensus |
|--------|-------------|-------------------|-------------------|------------|-------------------|
| **Attention** | Focus spread | Attention optimization | Priority logic | Salience sampling | - |
| **Memory** | Memory retrieval | Memory consolidation | Recall inference | Replay sampling | - |
| **Emotion** | Mood diffusion | Emotional regulation | Emotional reasoning | Affect sampling | - |
| **Reasoning** | Inference spread | Weight optimization | SAT/CNF solving | Hypothesis sampling | - |
| **Executive** | Control signals | Strategy optimization | Decision logic | Option sampling | - |
| **Theory of Mind** | Belief diffusion | Social optimization | Belief inference | Mental state sampling | - |
| **Ethics** | Value diffusion | Moral optimization | Ethical reasoning | Value sampling | Ethical voting |
| **Self Model** | Self-awareness spread | Self optimization | Self inference | State sampling | - |
| **Introspection** | Meta spread | Meta optimization | Meta reasoning | Meta sampling | - |
| **Curiosity** | Interest spread | Exploration optimization | Novelty logic | Exploration sampling | - |
| **Wellbeing** | Homeostatic spread | Balance optimization | Wellbeing logic | State sampling | - |
| **Swarm** | Collective spread | Collective optimization | Collective reasoning | Collective sampling | Swarm voting |
| **Collective** | Inter-node spread | Group optimization | Group inference | Group sampling | Group consensus |

#### 21.7 Quantum Bio-Async Messages

```c
// Quantum-specific bio-async message types
typedef enum {
    // Quantum Walk messages
    BIO_MSG_QUANTUM_WALK_INIT,              // Initialize walker on network
    BIO_MSG_QUANTUM_WALK_STEP,              // Evolve walker one step
    BIO_MSG_QUANTUM_WALK_MEASURE,           // Measure walker position
    BIO_MSG_QUANTUM_WALK_DISTRIBUTION,      // Get probability distribution
    BIO_MSG_QUANTUM_DIFFUSION_RESULT,       // Neuromodulator diffusion result

    // Quantum Annealing messages
    BIO_MSG_QUANTUM_ANNEAL_START,           // Begin annealing
    BIO_MSG_QUANTUM_ANNEAL_STEP,            // Single annealing step
    BIO_MSG_QUANTUM_ANNEAL_RESULT,          // Optimization result
    BIO_MSG_QUANTUM_TUNNELING_EVENT,        // Successful tunneling

    // Quantum Reasoning messages
    BIO_MSG_QUANTUM_REASON_QUERY,           // Query knowledge base
    BIO_MSG_QUANTUM_REASON_RESULT,          // Query result
    BIO_MSG_QUANTUM_SAT_SOLVE,              // Solve SAT problem
    BIO_MSG_QUANTUM_SAT_RESULT,             // SAT solution
    BIO_MSG_QUANTUM_BELIEF_UPDATE,          // Belief propagation update

    // Quantum Monte Carlo messages
    BIO_MSG_QUANTUM_MC_SAMPLE,              // Request sampling
    BIO_MSG_QUANTUM_MC_RESULT,              // Sampling result
    BIO_MSG_QUANTUM_ENTROPY_ESTIMATE,       // Entropy estimation
    BIO_MSG_QUANTUM_PARTITION_ESTIMATE,     // Partition function estimate

    // Quantum Consensus messages
    BIO_MSG_QUANTUM_PROPOSE,                // Create proposal
    BIO_MSG_QUANTUM_VOTE,                   // Cast vote
    BIO_MSG_QUANTUM_CONSENSUS_RUN,          // Run consensus algorithm
    BIO_MSG_QUANTUM_CONSENSUS_RESULT,       // Consensus outcome
    BIO_MSG_QUANTUM_COLLUSION_DETECTED      // Collusion alert
} quantum_bio_msg_type_t;

// Message routing
nimcp_status_t quantum_route_to_module(
    void* quantum_system,
    void* module,
    quantum_bio_msg_type_t msg_type,
    const void* payload,
    size_t payload_size);
```

#### 21.8 Quantum-Cognitive Synergies

```c
// Special quantum-cognitive integration patterns
typedef struct {
    // Quantum Walk + Attention
    // Use quantum walk to spread attention across neural network
    float attention_quantum_speedup;         // Speedup from QW

    // Quantum Annealing + Plasticity
    // Use QA to escape local minima during learning
    float plasticity_improvement;            // % improvement from QA

    // Quantum Reasoning + Executive
    // Use Grover for fast decision search
    uint32_t decision_grover_iterations;     // Grover iterations for decisions

    // Quantum MC + Introspection
    // Use QMC for efficient self-state sampling
    float introspection_sampling_efficiency; // Efficiency vs classical

    // Quantum Consensus + Collective
    // Use QC for distributed swarm decisions
    float consensus_convergence_speedup;     // √N speedup from Grover
} quantum_cognitive_synergy_t;

// Synergy initialization
nimcp_status_t quantum_init_cognitive_synergies(
    void* brain,
    quantum_cognitive_synergy_t* synergies);
```

#### 21.9 Quantum Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/quantum/test_quantum_walk_module_integration.cpp` | 30 |
| `test/integration/quantum/test_quantum_annealing_module_integration.cpp` | 30 |
| `test/integration/quantum/test_quantum_reasoning_module_integration.cpp` | 30 |
| `test/integration/quantum/test_quantum_mc_module_integration.cpp` | 25 |
| `test/integration/quantum/test_quantum_consensus_module_integration.cpp` | 25 |
| `test/integration/quantum/test_quantum_bioasync_integration.cpp` | 25 |
| `test/integration/quantum/test_quantum_cognitive_synergy_integration.cpp` | 20 |
| `test/integration/quantum/test_quantum_ternary_integration.cpp` | 25 |
| **Total** | **210** |

### 22. Training Module Integration

**Priority**: Critical
**Dependencies**: All cognitive modules, SNN system, Plasticity system
**Est. LOC**: 26,000 | **Tests**: 230

The Training Module provides comprehensive learning infrastructure that must integrate with all cognitive modules to enable biologically-plausible learning across the entire system.

#### 22.1 Training Module Components

| Component | File | Purpose |
|-----------|------|---------|
| Training Module Core | `nimcp_training_module.h/c` | Core training infrastructure with phases T1-T4 |
| Gradient Manager | `nimcp_gradient_manager.h/c` | Accumulation, scaling, health checking |
| Optimizers | `nimcp_optimizers.h/c` | SGD, Adam, AdamW, RMSprop, AdaGrad, Nadam |
| Loss Functions | `nimcp_loss_functions.h/c` | MSE, Cross-Entropy, KL, Huber, Focal, Triplet |
| LR Scheduler | `nimcp_lr_scheduler.h/c` | Step, Cosine, OneCycle, ReduceOnPlateau |
| Curriculum Learning | `nimcp_curriculum_learning.h/c` | Self-Paced, Teacher-Guided, Uncertainty |
| Regularization | `nimcp_regularization.h/c` | L1, L2, Dropout, Weight Decay |

#### 22.2 Training Types

```c
// Training types from nimcp_training_module.h
typedef enum {
    NIMCP_TRAINING_STDP = 0,          // Spike-timing dependent plasticity
    NIMCP_TRAINING_DENDRITIC,         // Dendritic compartment learning
    NIMCP_TRAINING_PREDICTIVE,        // Predictive coding learning
    NIMCP_TRAINING_BCM,               // Bienenstock-Cooper-Munro rule
    NIMCP_TRAINING_HOMEOSTATIC,       // Homeostatic plasticity
    NIMCP_TRAINING_ELIGIBILITY,       // Eligibility trace-based learning
    NIMCP_TRAINING_BRAIN_LEARNING,    // Combined brain-inspired learning
    NIMCP_TRAINING_BIOLOGICAL,        // Full biological realism
    NIMCP_TRAINING_TYPE_COUNT
} nimcp_training_type_t;

// Training phases
typedef enum {
    NIMCP_TRAINING_PHASE_T1,          // Homeostatic (stabilization)
    NIMCP_TRAINING_PHASE_T2,          // Dendritic (compartment learning)
    NIMCP_TRAINING_PHASE_T3,          // Predictive (error minimization)
    NIMCP_TRAINING_PHASE_T4,          // Meta-learning (learning to learn)
    NIMCP_TRAINING_PHASE_COUNT
} nimcp_training_phase_t;
```

#### 22.3 Optimizer Integration

```c
// Optimizer types
typedef enum {
    NIMCP_OPTIMIZER_SGD = 0,          // Vanilla SGD
    NIMCP_OPTIMIZER_SGD_MOMENTUM,     // SGD with momentum
    NIMCP_OPTIMIZER_NESTEROV,         // Nesterov accelerated gradient
    NIMCP_OPTIMIZER_ADAGRAD,          // Adaptive gradient
    NIMCP_OPTIMIZER_RMSPROP,          // Root mean square propagation
    NIMCP_OPTIMIZER_ADAM,             // Adaptive moment estimation
    NIMCP_OPTIMIZER_ADAMW,            // Adam with decoupled weight decay
    NIMCP_OPTIMIZER_NADAM,            // Nesterov Adam
    NIMCP_OPTIMIZER_CUSTOM,           // User-defined optimizer
    NIMCP_OPTIMIZER_TYPE_COUNT
} nimcp_optimizer_type_t;

// Adam configuration example
typedef struct {
    float learning_rate;     // Default: 0.001
    float beta1;             // Default: 0.9
    float beta2;             // Default: 0.999
    float epsilon;           // Default: 1e-8
    float weight_decay;      // L2 regularization
    bool amsgrad;            // Use AMSGrad variant
} nimcp_adam_config_t;
```

#### 22.4 Loss Function Integration

```c
// Loss function types
typedef enum {
    NIMCP_LOSS_MSE = 0,               // Mean squared error
    NIMCP_LOSS_MAE,                   // Mean absolute error
    NIMCP_LOSS_CROSS_ENTROPY,         // Cross-entropy loss
    NIMCP_LOSS_BINARY_CROSS_ENTROPY,  // Binary cross-entropy
    NIMCP_LOSS_KL_DIVERGENCE,         // Kullback-Leibler divergence
    NIMCP_LOSS_HUBER,                 // Huber loss (smooth L1)
    NIMCP_LOSS_HINGE,                 // Hinge loss (SVM)
    NIMCP_LOSS_LOG_COSH,              // Log-cosh loss
    NIMCP_LOSS_FOCAL,                 // Focal loss (class imbalance)
    NIMCP_LOSS_CONTRASTIVE,           // Contrastive loss (similarity)
    NIMCP_LOSS_TRIPLET,               // Triplet loss (metric learning)
    NIMCP_LOSS_CUSTOM,                // User-defined loss
    NIMCP_LOSS_TYPE_COUNT
} nimcp_loss_type_t;

// Reduction modes for loss computation
typedef enum {
    NIMCP_LOSS_REDUCE_MEAN = 0,       // Average over batch
    NIMCP_LOSS_REDUCE_SUM,            // Sum over batch
    NIMCP_LOSS_REDUCE_NONE            // Return per-sample loss
} nimcp_loss_reduction_t;
```

#### 22.5 Learning Rate Scheduler Integration

```c
// Scheduler types
typedef enum {
    NIMCP_LR_CONSTANT = 0,            // Fixed learning rate
    NIMCP_LR_STEP,                    // Step decay
    NIMCP_LR_EXPONENTIAL,             // Exponential decay
    NIMCP_LR_COSINE_ANNEALING,        // Cosine annealing
    NIMCP_LR_LINEAR_WARMUP,           // Linear warmup
    NIMCP_LR_MULTI_STEP,              // Multi-step decay
    NIMCP_LR_REDUCE_ON_PLATEAU,       // Reduce on plateau
    NIMCP_LR_CYCLIC,                  // Cyclic learning rate
    NIMCP_LR_ONE_CYCLE,               // One-cycle policy
    NIMCP_LR_COSINE_WARMUP,           // Cosine with warmup
    NIMCP_LR_POLYNOMIAL,              // Polynomial decay
    NIMCP_LR_CUSTOM,                  // User-defined scheduler
    NIMCP_LR_TYPE_COUNT
} nimcp_lr_scheduler_type_t;
```

#### 22.6 Curriculum Learning Integration

```c
// Curriculum learning strategies
typedef enum {
    NIMCP_CURRICULUM_NONE = 0,        // No curriculum
    NIMCP_CURRICULUM_SELF_PACED,      // Self-paced learning
    NIMCP_CURRICULUM_TEACHER_GUIDED,  // Teacher-guided curriculum
    NIMCP_CURRICULUM_TRANSFER,        // Transfer curriculum
    NIMCP_CURRICULUM_UNCERTAINTY,     // Uncertainty-based sampling
    NIMCP_CURRICULUM_LOSS_BASED,      // Loss-based difficulty
    NIMCP_CURRICULUM_GRADIENT_NORM,   // Gradient norm difficulty
    NIMCP_CURRICULUM_CONFIDENCE,      // Confidence-based selection
    NIMCP_CURRICULUM_ANTI,            // Anti-curriculum (hard first)
    NIMCP_CURRICULUM_HYBRID,          // Combined strategies
    NIMCP_CURRICULUM_STRATEGY_COUNT
} nimcp_curriculum_strategy_t;

// Curriculum states
typedef enum {
    NIMCP_CURRICULUM_WARMUP = 0,      // Initial warmup phase
    NIMCP_CURRICULUM_EASY,            // Easy samples only
    NIMCP_CURRICULUM_MEDIUM,          // Medium difficulty
    NIMCP_CURRICULUM_FULL             // Full difficulty range
} nimcp_curriculum_state_t;
```

#### 22.7 Gradient Management Integration

```c
// Gradient accumulation modes
typedef enum {
    NIMCP_GRAD_ACCUM_SUM = 0,         // Sum gradients
    NIMCP_GRAD_ACCUM_MEAN             // Average gradients
} nimcp_grad_accum_mode_t;

// Gradient scaling strategies
typedef enum {
    NIMCP_GRAD_SCALE_NONE = 0,        // No scaling
    NIMCP_GRAD_SCALE_FIXED,           // Fixed scale factor
    NIMCP_GRAD_SCALE_DYNAMIC          // Dynamic loss scaling
} nimcp_grad_scale_strategy_t;

// Gradient health status
typedef enum {
    NIMCP_GRAD_HEALTHY = 0,           // All gradients valid
    NIMCP_GRAD_HAS_NAN,               // NaN detected
    NIMCP_GRAD_HAS_INF,               // Infinity detected
    NIMCP_GRAD_HAS_ZERO,              // All-zero gradients
    NIMCP_GRAD_OVERFLOW,              // Overflow detected
    NIMCP_GRAD_UNDERFLOW              // Underflow detected
} nimcp_grad_health_t;
```

#### 22.8 Training Bridge Files

**New Bridge Files**:
- `include/training/bridges/nimcp_training_cognitive_bridge.h`
- `include/training/bridges/nimcp_training_snn_bridge.h`
- `include/training/bridges/nimcp_training_fep_bridge.h`
- `include/training/bridges/nimcp_training_immune_bridge.h`
- `include/training/bridges/nimcp_training_bioasync_bridge.h`
- `src/training/bridges/nimcp_training_cognitive_bridge.c`
- `src/training/bridges/nimcp_training_snn_bridge.c`
- `src/training/bridges/nimcp_training_fep_bridge.c`
- `src/training/bridges/nimcp_training_immune_bridge.c`
- `src/training/bridges/nimcp_training_bioasync_bridge.c`

#### 22.9 Module Training Behavior

| Cognitive Module | Training Type | Optimizer | Loss Function | Curriculum |
|-----------------|---------------|-----------|---------------|------------|
| Attention | STDP | Adam | Cross-Entropy | Self-Paced |
| Memory | Dendritic | AdamW | Contrastive | Teacher-Guided |
| Emotion | BCM | SGD+Momentum | MSE | Uncertainty |
| Reasoning | Predictive | Adam | KL-Divergence | Loss-Based |
| Executive | Eligibility | AdamW | Huber | Confidence |
| Ethics | Homeostatic | Adam | Focal | Teacher-Guided |
| Curiosity | STDP | RMSprop | Triplet | Anti-Curriculum |
| Self-Model | Brain Learning | AdamW | MSE | Self-Paced |
| Introspection | Predictive | Adam | Cross-Entropy | Uncertainty |
| Wellbeing | BCM | SGD | MSE | Hybrid |
| Salience | STDP | Adam | Binary-CE | Confidence |
| Bias | Homeostatic | AdamW | KL-Divergence | Teacher-Guided |
| Epistemic | Predictive | Adam | Cross-Entropy | Loss-Based |
| Global Workspace | Eligibility | AdamW | MSE | Self-Paced |
| Meta-Learning | Biological | Adam | Triplet | Anti-Curriculum |
| ToM | Dendritic | AdamW | Cross-Entropy | Uncertainty |
| Mirror Neurons | STDP | RMSprop | Contrastive | Teacher-Guided |

#### 22.10 Training Bio-Async Messages

```c
// Training-related bio-async message types
typedef enum {
    BIO_MSG_TRAINING_START,             // Start training session
    BIO_MSG_TRAINING_STOP,              // Stop training session
    BIO_MSG_TRAINING_STEP,              // Execute training step
    BIO_MSG_TRAINING_BATCH_COMPLETE,    // Batch completed
    BIO_MSG_TRAINING_EPOCH_COMPLETE,    // Epoch completed
    BIO_MSG_TRAINING_CHECKPOINT,        // Save checkpoint
    BIO_MSG_TRAINING_RESTORE,           // Restore checkpoint
    BIO_MSG_TRAINING_LR_UPDATE,         // Learning rate updated
    BIO_MSG_TRAINING_CURRICULUM_ADVANCE,// Curriculum state advanced
    BIO_MSG_TRAINING_GRADIENT_HEALTH,   // Gradient health report
    BIO_MSG_TRAINING_LOSS_REPORT,       // Loss value report
    BIO_MSG_TRAINING_METRICS_UPDATE,    // Training metrics update
    BIO_MSG_TRAINING_PHASE_TRANSITION,  // Phase T1->T2->T3->T4
    BIO_MSG_TRAINING_OPTIMIZER_SWITCH,  // Dynamic optimizer change
    BIO_MSG_TRAINING_REGULARIZATION,    // Regularization event
    BIO_MSG_TRAINING_COUNT
} nimcp_training_msg_type_t;
```

#### 22.11 Training-Cognitive Integration API

```c
// Training integration context per module
typedef struct {
    // Training configuration
    nimcp_training_type_t training_type;
    nimcp_training_phase_t current_phase;

    // Optimizer settings
    nimcp_optimizer_context_t* optimizer;
    nimcp_optimizer_type_t optimizer_type;

    // Loss function
    nimcp_loss_context_t* loss_ctx;
    nimcp_loss_type_t loss_type;

    // Learning rate scheduler
    nimcp_lr_scheduler_t* lr_scheduler;
    nimcp_lr_scheduler_type_t scheduler_type;

    // Curriculum learning
    nimcp_curriculum_ctx_t* curriculum;
    nimcp_curriculum_strategy_t curriculum_strategy;

    // Gradient management
    nimcp_gradient_manager_ctx_t* grad_manager;

    // Bio-async integration
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Metrics
    nimcp_training_stats_t stats;
} nimcp_module_training_integration_t;

// Initialize training for a cognitive module
nimcp_status_t nimcp_training_init_for_module(
    nimcp_module_training_integration_t* training,
    const char* module_name,
    nimcp_training_type_t type,
    nimcp_optimizer_type_t optimizer,
    nimcp_loss_type_t loss,
    nimcp_curriculum_strategy_t curriculum,
    nimcp_bio_router_t* router);

// Execute training step for module
nimcp_status_t nimcp_training_module_step(
    nimcp_module_training_integration_t* training,
    float* params,
    const float* gradients,
    size_t param_count);

// Advance curriculum based on performance
nimcp_status_t nimcp_training_curriculum_advance(
    nimcp_module_training_integration_t* training,
    float performance_metric);

// Transition training phase
nimcp_status_t nimcp_training_phase_transition(
    nimcp_module_training_integration_t* training,
    nimcp_training_phase_t new_phase);
```

#### 22.12 FEP-Training Integration

```c
// Free energy training bridge
typedef struct {
    // FEP components
    float free_energy;
    float precision_weights[NIMCP_MAX_MODULES];
    float prediction_error;

    // Training adaptation
    float learning_rate_modulation;   // FEP modulates LR
    float gradient_precision;         // Precision-weighted gradients
    float surprise_signal;            // Drives curriculum

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
} nimcp_fep_training_bridge_t;

// FEP-guided learning rate adjustment
nimcp_status_t nimcp_fep_modulate_learning_rate(
    nimcp_fep_training_bridge_t* bridge,
    nimcp_module_training_integration_t* training);

// Precision-weighted gradient update
nimcp_status_t nimcp_fep_precision_gradient_update(
    nimcp_fep_training_bridge_t* bridge,
    float* gradients,
    size_t count);

// Surprise-driven curriculum advancement
nimcp_status_t nimcp_fep_surprise_curriculum(
    nimcp_fep_training_bridge_t* bridge,
    nimcp_curriculum_ctx_t* curriculum);
```

#### 22.13 Immune-Training Integration

```c
// Immune system training integration
typedef struct {
    // Immune monitoring
    nimcp_immune_sensor_t sensor;
    float gradient_health_score;      // 0-1
    float loss_stability_score;       // 0-1
    float learning_progress_score;    // 0-1

    // Threat responses
    bool gradient_nan_detected;
    bool loss_explosion_detected;
    bool catastrophic_forgetting_detected;

    // Healing actions
    nimcp_checkpoint_t* last_stable_checkpoint;
    uint32_t rollback_count;
} nimcp_immune_training_integration_t;

// Monitor training health
nimcp_status_t nimcp_immune_monitor_training(
    nimcp_immune_training_integration_t* immune,
    nimcp_module_training_integration_t* training);

// Trigger training rollback on catastrophic failure
nimcp_status_t nimcp_immune_training_rollback(
    nimcp_immune_training_integration_t* immune,
    nimcp_module_training_integration_t* training);

// Adaptive learning rate reduction on instability
nimcp_status_t nimcp_immune_lr_intervention(
    nimcp_immune_training_integration_t* immune,
    nimcp_module_training_integration_t* training,
    float reduction_factor);
```

#### 22.14 Distributed Training Integration

```c
// Distributed training configuration
typedef struct {
    uint32_t world_size;              // Number of workers
    uint32_t rank;                    // This worker's rank
    bool use_gradient_allreduce;      // Enable gradient sync
    bool use_mixed_precision;         // FP16/BF16 training
    uint32_t gradient_sync_interval;  // Steps between sync

    // Bio-async distributed messaging
    nimcp_bio_async_handler_t distributed_handler;
} nimcp_distributed_training_config_t;

// All-reduce gradients across workers
nimcp_status_t nimcp_training_allreduce_gradients(
    nimcp_distributed_training_config_t* dist_config,
    float* gradients,
    size_t count);

// Synchronize curriculum state across workers
nimcp_status_t nimcp_training_sync_curriculum(
    nimcp_distributed_training_config_t* dist_config,
    nimcp_curriculum_ctx_t* curriculum);
```

#### 22.15 Training Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/training/test_training_cognitive_bridge.cpp` | 25 |
| `test/integration/training/test_training_snn_bridge.cpp` | 25 |
| `test/integration/training/test_training_fep_bridge.cpp` | 25 |
| `test/integration/training/test_training_immune_bridge.cpp` | 20 |
| `test/integration/training/test_training_bioasync_bridge.cpp` | 20 |
| `test/integration/training/test_optimizer_cognitive_integration.cpp` | 25 |
| `test/integration/training/test_loss_module_integration.cpp` | 20 |
| `test/integration/training/test_scheduler_adaptation_integration.cpp` | 20 |
| `test/integration/training/test_curriculum_cognitive_integration.cpp` | 25 |
| `test/integration/training/test_gradient_health_integration.cpp` | 15 |
| `test/integration/training/test_distributed_training_integration.cpp` | 10 |
| **Total** | **230** |

### 23. Language Module Integration

**Priority**: Critical
**Dependencies**: Perception layer, Cognitive layer, Motor cortex, Training system
**Est. LOC**: 28,000 | **Tests**: 250

The Language Module provides comprehensive speech comprehension (Wernicke's area) and production (Broca's area) capabilities that must integrate with all cognitive modules to enable natural language understanding and generation.

#### 23.1 Language Module Architecture

```
   Perception Layer          Cognitive Layer           Training Layer
         │                         │                         │
         ▼                         ▼                         ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                  LANGUAGE ORCHESTRATOR                          │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
   │  │ Wernicke │◄─►│ Broca    │  │ NLP Core │  │ Multimodal/Spike││
   │  │ (Compre- │  │ (Produc- │  │ (Attn,   │  │ (Cross-modal,   ││
   │  │  hension)│  │  tion)   │  │ Embed)   │  │  Spiking NLP)   ││
   │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
   │         │              │            │               │          │
   │         └──────────────┴────────────┴───────────────┘          │
   │                              │                                  │
   │                    State Machine                                │
   │     IDLE → LISTENING → COMPREHENDING → INTEGRATING → OUTPUT    │
   └─────────────────────────────────────────────────────────────────┘
```

#### 23.2 Language Layer Components

| Component | File | Purpose |
|-----------|------|---------|
| Language Orchestrator | `nimcp_language_orchestrator.h/c` | Central coordinator for all language processing |
| Language Types | `nimcp_language_types.h` | Shared types (phonemes, words, concepts, syntax) |
| Language Config | `nimcp_language_config.h` | Configuration structures |
| Language Bio-Async | `nimcp_language_bio_async.h` | Bio-async messaging integration |
| Wernicke Adapter | `nimcp_wernicke_adapter.h/c` | Speech comprehension (BA22) |
| Broca Adapter | `nimcp_broca_adapter.h/c` | Speech production (BA44/45) |

#### 23.3 Wernicke's Area (Comprehension)

```c
// Processing layers: Phonological → Lexical → Semantic → Syntactic
typedef struct {
    phonological_analyzer_t* phonological;  // Phoneme recognition (STG)
    lexical_access_t* lexical;              // Word recognition (MTG)
    semantic_integrator_t* semantic;        // Meaning retrieval (ATL)
    syntactic_comprehension_t* syntactic;   // Phrase structure (BA45)
} wernicke_layers_t;

// Comprehension result
typedef struct {
    language_word_t* words;           // Recognized words
    uint32_t word_count;
    language_concept_t* concepts;     // Activated concepts
    uint32_t concept_count;
    language_parse_node_t* parse_tree;// Syntactic parse
    float* semantic_vector;           // Sentence embedding
    float semantic_anomaly;           // N400-like signal
    float syntactic_anomaly;          // P600-like signal
} language_comprehension_result_t;
```

#### 23.4 Broca's Area (Production)

```c
// Processing pipeline: Semantic → Syntactic → Phonological → Motor
typedef struct {
    syntax_processor_t* syntax;          // Syntactic structure building
    phonological_processor_t* phonological; // Phoneme sequence planning
    speech_motor_planner_t* motor;       // Motor command generation
} broca_layers_t;

// Production plan
typedef struct {
    language_word_t* words;              // Planned word sequence
    language_phoneme_t* phonemes;        // Planned phoneme sequence
    language_motor_command_t* commands;  // Motor commands for articulation
    language_prosody_t prosody;          // Prosodic contour
    float fluency_score;
} language_production_plan_t;

// Broca processing status
typedef enum {
    BROCA_STATUS_IDLE,
    BROCA_STATUS_LEXICAL_ACCESS,
    BROCA_STATUS_SYNTACTIC,
    BROCA_STATUS_PHONOLOGICAL,
    BROCA_STATUS_MOTOR_PLANNING,
    BROCA_STATUS_READY,
    BROCA_STATUS_ERROR
} broca_status_t;
```

#### 23.5 Language Processing States

```c
// Language layer state machine
typedef enum {
    LANGUAGE_STATE_IDLE = 0,          // No active processing
    LANGUAGE_STATE_LISTENING,         // Receiving perceptual input
    LANGUAGE_STATE_COMPREHENDING,     // Processing through Wernicke
    LANGUAGE_STATE_INTEGRATING,       // Cross-module integration
    LANGUAGE_STATE_GENERATING,        // Generating through Broca
    LANGUAGE_STATE_PRODUCING,         // Motor output preparation
    LANGUAGE_STATE_ERROR
} language_state_t;

// Language processing modes
typedef enum {
    LANGUAGE_MODE_COMPREHENSION,      // Input processing only
    LANGUAGE_MODE_PRODUCTION,         // Output generation only
    LANGUAGE_MODE_DIALOGUE,           // Full bidirectional
    LANGUAGE_MODE_REPETITION,         // Direct echo (arcuate fasciculus)
    LANGUAGE_MODE_TRANSLATION         // Cross-language processing
} language_mode_t;
```

#### 23.6 Phonological Types

```c
// Phoneme categories (IPA-based)
typedef enum {
    PHONEME_CAT_VOWEL,               // Vowel sounds
    PHONEME_CAT_STOP,                // Stop consonants (p, b, t, d, k, g)
    PHONEME_CAT_FRICATIVE,           // Fricatives (f, v, s, z, etc.)
    PHONEME_CAT_AFFRICATE,           // Affricates (ch, j)
    PHONEME_CAT_NASAL,               // Nasals (m, n, ng)
    PHONEME_CAT_APPROXIMANT,         // Approximants (l, r, w, y)
    PHONEME_CAT_SILENCE
} phoneme_category_t;

// Phoneme representation
typedef struct {
    uint32_t id;                     // IPA code
    phoneme_category_t category;
    float confidence;                // Recognition confidence [0-1]
    float duration_ms;
    float formants[4];               // F1-F4 frequencies
    float pitch_hz;                  // Fundamental frequency (F0)
    bool is_stressed;                // Prosodic stress
    bool is_word_boundary;
} language_phoneme_t;
```

#### 23.7 Semantic/Syntactic Types

```c
// Thematic roles (Case Grammar)
typedef enum {
    THEMATIC_ROLE_AGENT,             // Doer of action
    THEMATIC_ROLE_PATIENT,           // Affected by action
    THEMATIC_ROLE_THEME,             // Entity moved/changed
    THEMATIC_ROLE_EXPERIENCER,       // Perceiver/feeler
    THEMATIC_ROLE_BENEFICIARY,       // Recipient of benefit
    THEMATIC_ROLE_INSTRUMENT,        // Tool used
    THEMATIC_ROLE_LOCATION,
    THEMATIC_ROLE_SOURCE,
    THEMATIC_ROLE_GOAL,
    THEMATIC_ROLE_TIME,
    THEMATIC_ROLE_MANNER,
    THEMATIC_ROLE_CAUSE
} thematic_role_t;

// Part of speech
typedef enum {
    POS_NOUN, POS_VERB, POS_ADJECTIVE, POS_ADVERB,
    POS_DETERMINER, POS_PREPOSITION, POS_CONJUNCTION,
    POS_PRONOUN, POS_AUXILIARY, POS_COMPLEMENTIZER
} part_of_speech_t;

// Phrase types for parsing
typedef enum {
    PHRASE_NP,   // Noun phrase
    PHRASE_VP,   // Verb phrase
    PHRASE_PP,   // Prepositional phrase
    PHRASE_AP,   // Adjective phrase
    PHRASE_S,    // Sentence
    PHRASE_SBAR, // Subordinate clause
    PHRASE_CP    // Complementizer phrase
} phrase_type_t;
```

#### 23.8 Language Layer Bridges

| Bridge | File | Purpose |
|--------|------|---------|
| Perception Bridge | `nimcp_language_perception_bridge.h` | Audio/visual input from cortices |
| Cognitive Bridge | `nimcp_language_cognitive_bridge.h` | Working memory, attention, reasoning |
| Training Bridge | `nimcp_language_training_bridge.h` | Language learning (vocabulary, grammar) |
| Omni Bridge | `nimcp_language_omni_bridge.h` | Predictive processing (word prediction) |
| Immune Bridge | `nimcp_language_immune_bridge.h` | Inflammation effects (aphasia modeling) |
| GPU Bridge | `nimcp_language_gpu_bridge.h` | GPU acceleration |
| Thalamic Bridge | `nimcp_language_thalamic_bridge.h` | Attention gating |
| Substrate Bridge | `nimcp_language_substrate_bridge.h` | Bio-async messaging |
| Logic Bridge | `nimcp_language_logic_bridge.h` | Logical reasoning integration |
| Motor Bridge | `nimcp_language_motor_bridge.h` | Motor cortex for speech output |
| Prefrontal Bridge | `nimcp_language_prefrontal_bridge.h` | Executive control |
| Temporal Bridge | `nimcp_language_temporal_bridge.h` | Temporal lobe integration |
| Hippocampus Bridge | `nimcp_language_hippocampus_bridge.h` | Memory consolidation |
| Parietal Bridge | `nimcp_language_parietal_bridge.h` | Spatial language integration |
| Cingulate Bridge | `nimcp_language_cingulate_bridge.h` | Error monitoring |
| Insula Bridge | `nimcp_language_insula_bridge.h` | Speech motor coordination |
| Cerebellum Bridge | `nimcp_language_cerebellum_bridge.h` | Speech timing |

#### 23.9 Language-Cognitive Module Integration

| Cognitive Module | Language Integration | Direction |
|-----------------|---------------------|-----------|
| Attention | Focus on spoken/written input | Bidirectional |
| Working Memory | Phonological loop, semantic buffer | Bidirectional |
| Emotion | Prosodic modulation, emotional language | Bidirectional |
| Reasoning | Logical discourse, argumentation | Language → Cognitive |
| Executive | Speech planning, turn-taking | Bidirectional |
| Ethics | Appropriate language use | Cognitive → Language |
| Curiosity | Question generation | Cognitive → Language |
| Self-Model | Inner speech, self-talk | Bidirectional |
| Introspection | Verbalization of mental states | Cognitive → Language |
| ToM | Pragmatics, indirect speech acts | Bidirectional |
| Mirror Neurons | Articulatory mirroring | Bidirectional |
| Semantic Memory | Lexicon, conceptual knowledge | Bidirectional |
| Episodic Memory | Narrative comprehension/production | Bidirectional |

#### 23.10 Language Bio-Async Messages

```c
// Language-related bio-async message types
typedef enum {
    BIO_MSG_LANG_PHONEME_INPUT,           // Phoneme received from perception
    BIO_MSG_LANG_WORD_RECOGNIZED,         // Word recognized by Wernicke
    BIO_MSG_LANG_CONCEPT_ACTIVATED,       // Concept activated in semantic memory
    BIO_MSG_LANG_COMPREHENSION_COMPLETE,  // Utterance comprehended
    BIO_MSG_LANG_PRODUCTION_REQUEST,      // Request to produce language
    BIO_MSG_LANG_PRODUCTION_READY,        // Production plan ready
    BIO_MSG_LANG_MOTOR_COMMAND,           // Motor command for articulation
    BIO_MSG_LANG_STATE_CHANGE,            // State machine transition
    BIO_MSG_LANG_ANOMALY_DETECTED,        // Semantic/syntactic anomaly (N400/P600)
    BIO_MSG_LANG_AMBIGUITY_DETECTED,      // Ambiguous input detected
    BIO_MSG_LANG_WERNICKE_BROCA_SYNC,     // Arcuate fasciculus communication
    BIO_MSG_LANG_PROSODY_UPDATE,          // Prosodic information
    BIO_MSG_LANG_CONTEXT_UPDATE,          // Discourse context update
    BIO_MSG_LANG_TRAINING_UPDATE,         // Learning update
    BIO_MSG_LANG_COUNT
} nimcp_language_msg_type_t;
```

#### 23.11 Language-Cognitive Integration API

```c
// Language integration context for cognitive modules
typedef struct {
    // Language orchestrator reference
    language_orchestrator_t* orchestrator;

    // Comprehension interface
    bool (*process_utterance)(const char* text);
    bool (*process_phonemes)(const language_phoneme_t* phonemes, uint32_t count);
    bool (*get_comprehension)(language_comprehension_result_t* result);

    // Production interface
    bool (*generate_response)(const float* semantic_input, uint32_t dim);
    bool (*get_production_plan)(language_production_plan_t* plan);

    // Semantic interface
    bool (*activate_concept)(uint32_t concept_id, float activation);
    bool (*query_semantic)(const char* query, float* result, uint32_t dim);

    // Context interface
    bool (*update_context)(const language_context_t* context);
    bool (*get_current_context)(language_context_t* context);

    // Bio-async integration
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;
} nimcp_language_cognitive_integration_t;

// Initialize language integration for cognitive module
nimcp_status_t nimcp_language_init_cognitive_integration(
    nimcp_language_cognitive_integration_t* integration,
    language_orchestrator_t* orchestrator,
    nimcp_bio_router_t* router);

// Process natural language query through reasoning
nimcp_status_t nimcp_language_reason_over_text(
    nimcp_language_cognitive_integration_t* integration,
    const char* query,
    char* response,
    size_t max_response_len);
```

#### 23.12 Wernicke-Broca Integration (Arcuate Fasciculus)

```c
// Models the bidirectional arcuate fasciculus
typedef struct {
    // Forward: Wernicke → Broca (comprehension → production)
    float* forward_weights;
    uint32_t forward_dim;

    // Backward: Broca → Wernicke (production monitoring)
    float* backward_weights;
    uint32_t backward_dim;

    // Repetition pathway (direct bypass)
    bool repetition_mode;
    float repetition_strength;

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
} nimcp_arcuate_fasciculus_t;

// Send comprehension result to Broca for response generation
nimcp_status_t nimcp_arcuate_forward(
    nimcp_arcuate_fasciculus_t* arcuate,
    const language_comprehension_result_t* comprehension,
    broca_adapter_t* broca);

// Send production feedback to Wernicke for monitoring
nimcp_status_t nimcp_arcuate_backward(
    nimcp_arcuate_fasciculus_t* arcuate,
    const language_production_plan_t* production,
    wernicke_adapter_t* wernicke);

// Direct repetition (bypasses semantic processing)
nimcp_status_t nimcp_arcuate_repeat(
    nimcp_arcuate_fasciculus_t* arcuate,
    const language_phoneme_t* input_phonemes,
    uint32_t input_count,
    language_phoneme_t* output_phonemes,
    uint32_t* output_count);
```

#### 23.13 Aphasia Modeling (Immune Integration)

```c
// Models language disorders from neural damage/inflammation
typedef struct {
    // Wernicke's aphasia (comprehension deficit)
    float wernicke_damage;           // 0-1 damage level
    bool fluent_but_meaningless;     // Word salad
    bool poor_comprehension;
    bool neologisms;

    // Broca's aphasia (production deficit)
    float broca_damage;              // 0-1 damage level
    bool non_fluent_speech;
    bool preserved_comprehension;
    bool agrammatism;

    // Conduction aphasia (arcuate damage)
    float arcuate_damage;            // 0-1 damage level
    bool impaired_repetition;

    // Immune influence
    float il1b_level;                // Pro-inflammatory
    float tnf_alpha_level;           // Pro-inflammatory
    float il10_level;                // Anti-inflammatory

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
} nimcp_aphasia_model_t;

// Apply inflammation effects to language processing
nimcp_status_t nimcp_language_apply_inflammation(
    language_orchestrator_t* orchestrator,
    const nimcp_aphasia_model_t* aphasia);

// Recover from aphasia through training
nimcp_status_t nimcp_language_aphasia_recovery(
    language_orchestrator_t* orchestrator,
    nimcp_aphasia_model_t* aphasia,
    float recovery_rate);
```

#### 23.14 Language Training Integration

```c
// Language learning configuration
typedef struct {
    // Vocabulary learning
    bool enable_vocabulary_learning;
    float vocabulary_learning_rate;

    // Grammar learning
    bool enable_grammar_learning;
    float grammar_learning_rate;

    // Phoneme learning
    bool enable_phoneme_learning;
    float phoneme_learning_rate;

    // Semantic learning
    bool enable_semantic_learning;
    float semantic_learning_rate;

    // Curriculum
    nimcp_curriculum_strategy_t curriculum;
    nimcp_curriculum_state_t curriculum_state;

    // Training stats
    uint64_t vocabulary_size;
    uint64_t grammar_rules_learned;
    float comprehension_accuracy;
    float production_fluency;
} nimcp_language_training_config_t;

// Initialize language learning
nimcp_status_t nimcp_language_init_training(
    language_orchestrator_t* orchestrator,
    const nimcp_language_training_config_t* config);

// Learn from comprehension feedback
nimcp_status_t nimcp_language_learn_from_feedback(
    language_orchestrator_t* orchestrator,
    const language_comprehension_result_t* expected,
    const language_comprehension_result_t* actual);

// Learn from production feedback
nimcp_status_t nimcp_language_learn_production(
    language_orchestrator_t* orchestrator,
    const language_production_plan_t* plan,
    float feedback_score);
```

#### 23.15 Language Integration Tests

| Test File | Test Count |
|-----------|------------|
| `test/integration/language/test_language_orchestrator_integration.cpp` | 30 |
| `test/integration/language/test_wernicke_broca_integration.cpp` | 25 |
| `test/integration/language/test_language_perception_bridge.cpp` | 20 |
| `test/integration/language/test_language_cognitive_bridge.cpp` | 25 |
| `test/integration/language/test_language_training_bridge.cpp` | 20 |
| `test/integration/language/test_language_omni_bridge.cpp` | 20 |
| `test/integration/language/test_language_immune_bridge.cpp` | 15 |
| `test/integration/language/test_language_motor_bridge.cpp` | 15 |
| `test/integration/language/test_arcuate_fasciculus_integration.cpp` | 20 |
| `test/integration/language/test_aphasia_modeling.cpp` | 15 |
| `test/integration/language/test_language_bioasync_integration.cpp` | 20 |
| `test/integration/language/test_language_semantic_memory_integration.cpp` | 25 |
| **Total** | **250** |

### 24. Updated Scope Summary

| Category | Items | Est. LOC | Tests |
|----------|-------|----------|-------|
| Advanced Concepts (Incomplete) | 9 concepts | ~45,000 | ~900 |
| Brain Regions (New) | 17 regions | ~85,000 | ~1,700 |
| Extrapolation (Remaining 15%) | 3 modules | ~4,500 | ~90 |
| Embodiment (Remaining 40%) | 4 modules | ~2,800 | ~56 |
| Superhuman (Remaining 50%) | 7 modules | ~15,000 | ~300 |
| **Intra-Layer Integration** | **9 layers** | **~18,000** | **~295** |
| **Inter-Layer Integration** | **13 connections** | **~13,000** | **~300** |
| **Layer Coordinator** | **1 system** | **~3,000** | **~50** |
| **Cognitive Hub Integration** | **40 modules** | **~8,000** | **~160** |
| **Omnidirectional Integration** | **40 modules** | **~12,000** | **~240** |
| **Hypothalamus Integration** | **40 modules** | **~8,000** | **~105** |
| **Thalamic/Middleware Integration** | **40 modules** | **~16,000** | **~140** |
| **Neural Substrate Integration** | **40 modules** | **~12,000** | **~115** |
| **Motor Cortex Integration** | **40 modules** | **~15,000** | **~150** |
| **Portia Integration** | **40 modules** | **~14,000** | **~125** |
| **Swarm Integration** | **40 modules** | **~18,000** | **~150** |
| **Dragonfly Integration** | **40 modules** | **~16,000** | **~130** |
| **Sleep Integration** | **40 modules** | **~20,000** | **~170** |
| **Glial Integration** | **40 modules** | **~22,000** | **~190** |
| **Quantum Integration** | **40 modules** | **~24,000** | **~210** |
| **Training Integration** | **40 modules** | **~26,000** | **~230** |
| **Language Integration** | **40 modules** | **~28,000** | **~250** |
| **TOTAL** | **~40 modules + full integration** | **~426,300** | **~6,056** |

---

## PART I: ADVANCED CONCEPTS IMPLEMENTATION

### Phase AC-1: Electromagnetic Fields (Ephaptic Coupling)

**Priority**: Medium
**Dependencies**: SNN system, Attention system
**Est. LOC**: 5,000 | **Tests**: 100

#### 1.1 Core Module

**Files**:
- `include/physics/ephaptic/nimcp_ephaptic.h`
- `include/physics/ephaptic/nimcp_local_field_potential.h`
- `include/physics/ephaptic/nimcp_field_synchronization.h`
- `src/physics/ephaptic/nimcp_ephaptic.c`
- `src/physics/ephaptic/nimcp_local_field_potential.c`
- `src/physics/ephaptic/nimcp_field_synchronization.c`

**API**:
```c
typedef struct {
    float field_strength[3];         // V/m (x, y, z)
    float field_potential;           // mV
    float magnetic_field[3];         // Tesla
    float field_induced_current;     // pA
    float membrane_polarization;
    float phase_locking_strength;    // 0-1
    uint32_t synchronized_neurons;

    // Bio-async integration
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune integration
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_ephaptic_system_t;

nimcp_status_t nimcp_ephaptic_init(nimcp_ephaptic_system_t* sys,
                                    nimcp_brain_t* brain);
nimcp_status_t nimcp_ephaptic_compute_lfp(nimcp_ephaptic_system_t* sys,
                                           nimcp_snn_population_t* population);
nimcp_status_t nimcp_ephaptic_synchronize(nimcp_ephaptic_system_t* sys,
                                           float target_phase);
```

#### 1.2 Bridges

| Bridge | File | Purpose |
|--------|------|---------|
| SNN Bridge | `nimcp_ephaptic_snn_bridge.h/c` | Population field effects |
| Plasticity Bridge | `nimcp_ephaptic_plasticity_bridge.h/c` | Field-dependent LTP/LTD |
| FEP Bridge | `nimcp_ephaptic_fep_bridge.h/c` | Precision weighting via fields |
| Substrate Bridge | `nimcp_ephaptic_substrate_bridge.h/c` | Bio-async messaging |
| Thalamic Bridge | `nimcp_ephaptic_thalamic_bridge.h/c` | Attention gating |
| Immune Bridge | `nimcp_ephaptic_immune_bridge.h/c` | Abnormal field detection |
| Hub Bridge | `nimcp_ephaptic_hub_bridge.h/c` | Cognitive hub integration |

#### 1.3 Brain Factory Integration

**File**: `src/core/brain/factory/init/nimcp_brain_init_ephaptic.c`

```c
nimcp_status_t nimcp_brain_init_ephaptic(nimcp_brain_t* brain,
                                          const nimcp_brain_config_t* config) {
    NIMCP_LOG_INFO("Initializing ephaptic coupling system");

    // Allocate from brain's memory pool
    brain->ephaptic = nimcp_pool_alloc(brain->pool, sizeof(nimcp_ephaptic_system_t));

    // Initialize core system
    nimcp_ephaptic_init(brain->ephaptic, brain);

    // Register with bio-async router
    nimcp_bio_router_register(brain->router,
                              NIMCP_MSG_TYPE_EPHAPTIC,
                              ephaptic_msg_handler,
                              brain->ephaptic);

    // Register with immune system
    nimcp_immune_register_sensor(brain->immune,
                                  &brain->ephaptic->immune_sensor,
                                  IMMUNE_SENSOR_EPHAPTIC);

    // Initialize all bridges
    nimcp_ephaptic_snn_bridge_init(&brain->ephaptic->snn_bridge, brain->snn);
    nimcp_ephaptic_plasticity_bridge_init(&brain->ephaptic->plasticity_bridge, brain->plasticity);
    nimcp_ephaptic_fep_bridge_init(&brain->ephaptic->fep_bridge, brain->fep);
    nimcp_ephaptic_substrate_bridge_init(&brain->ephaptic->substrate_bridge, brain->router);
    nimcp_ephaptic_thalamic_bridge_init(&brain->ephaptic->thalamic_bridge, brain->thalamus);
    nimcp_ephaptic_immune_bridge_init(&brain->ephaptic->immune_bridge, brain->immune);
    nimcp_ephaptic_hub_bridge_init(&brain->ephaptic->hub_bridge, brain->cognitive_hub);

    NIMCP_LOG_INFO("Ephaptic system initialized with %d bridges", 7);
    return NIMCP_OK;
}
```

#### 1.4 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/physics/ephaptic/test_ephaptic_core.cpp` | 20 |
| Unit | `test/unit/physics/ephaptic/test_lfp.cpp` | 15 |
| Unit | `test/unit/physics/ephaptic/test_synchronization.cpp` | 15 |
| Unit | `test/unit/physics/ephaptic/test_ephaptic_snn_bridge.cpp` | 10 |
| Unit | `test/unit/physics/ephaptic/test_ephaptic_plasticity_bridge.cpp` | 10 |
| Unit | `test/unit/physics/ephaptic/test_ephaptic_fep_bridge.cpp` | 10 |
| Integration | `test/integration/physics/ephaptic/test_ephaptic_snn_integration.cpp` | 10 |
| Integration | `test/integration/physics/ephaptic/test_ephaptic_immune_integration.cpp` | 5 |
| Regression | `test/regression/physics/ephaptic/test_ephaptic_regression.cpp` | 5 |
| E2E | `test/e2e/e2e_test_ephaptic_pipeline.cpp` | 5 |

---

### Phase AC-2: Information Geometry

**Priority**: Medium
**Dependencies**: Learning systems, Plasticity
**Est. LOC**: 5,500 | **Tests**: 110

#### 2.1 Core Module

**Files**:
- `include/physics/geometry/nimcp_information_geometry.h`
- `include/physics/geometry/nimcp_fisher_information.h`
- `include/physics/geometry/nimcp_natural_gradient.h`
- `include/physics/geometry/nimcp_neural_manifold.h`
- `src/physics/geometry/nimcp_information_geometry.c`
- `src/physics/geometry/nimcp_fisher_information.c`
- `src/physics/geometry/nimcp_natural_gradient.c`
- `src/physics/geometry/nimcp_neural_manifold.c`

**API**:
```c
typedef struct {
    float embedding[LATENT_DIM];
    float tangent_vectors[LATENT_DIM][AMBIENT_DIM];
    float fisher_matrix[LATENT_DIM][LATENT_DIM];
    float fisher_matrix_inverse[LATENT_DIM][LATENT_DIM];
    float ricci_curvature;
    float natural_gradient[LATENT_DIM];

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_info_geometry_t;

nimcp_status_t nimcp_info_geometry_init(nimcp_info_geometry_t* geom,
                                         nimcp_brain_t* brain);
nimcp_status_t nimcp_info_geometry_compute_fisher(nimcp_info_geometry_t* geom,
                                                   const float* distribution);
nimcp_status_t nimcp_info_geometry_natural_gradient(nimcp_info_geometry_t* geom,
                                                     const float* gradient,
                                                     float* natural_grad);
nimcp_status_t nimcp_info_geometry_update(nimcp_info_geometry_t* geom,
                                           float learning_rate);
```

#### 2.2 Bridges

| Bridge | File | Purpose |
|--------|------|---------|
| SNN Bridge | `nimcp_info_geometry_snn_bridge.h/c` | Neural code geometry |
| Plasticity Bridge | `nimcp_info_geometry_plasticity_bridge.h/c` | Natural gradient learning |
| FEP Bridge | `nimcp_info_geometry_fep_bridge.h/c` | Belief geometry |
| Substrate Bridge | `nimcp_info_geometry_substrate_bridge.h/c` | Bio-async messaging |
| Thalamic Bridge | `nimcp_info_geometry_thalamic_bridge.h/c` | Attention manifolds |
| Immune Bridge | `nimcp_info_geometry_immune_bridge.h/c` | Anomaly detection |
| Hub Bridge | `nimcp_info_geometry_hub_bridge.h/c` | Cognitive integration |

#### 2.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/physics/geometry/test_fisher_information.cpp` | 20 |
| Unit | `test/unit/physics/geometry/test_natural_gradient.cpp` | 20 |
| Unit | `test/unit/physics/geometry/test_neural_manifold.cpp` | 20 |
| Unit | `test/unit/physics/geometry/test_info_geometry_bridges.cpp` | 20 |
| Integration | `test/integration/physics/geometry/test_info_geometry_plasticity_integration.cpp` | 15 |
| Regression | `test/regression/physics/geometry/test_info_geometry_regression.cpp` | 10 |
| E2E | `test/e2e/e2e_test_info_geometry_pipeline.cpp` | 5 |

---

### Phase AC-3: Hodgkin-Huxley Dynamics

**Priority**: High (biophysical realism)
**Dependencies**: Neuron models, Ion channels
**Est. LOC**: 6,000 | **Tests**: 120

#### 3.1 Core Module

**Files**:
- `include/physics/biophysics/nimcp_hodgkin_huxley.h`
- `include/physics/biophysics/nimcp_ion_channels.h`
- `include/physics/biophysics/nimcp_gating_variables.h`
- `include/physics/biophysics/nimcp_action_potential.h`
- `src/physics/biophysics/nimcp_hodgkin_huxley.c`
- `src/physics/biophysics/nimcp_ion_channels.c`
- `src/physics/biophysics/nimcp_gating_variables.c`
- `src/physics/biophysics/nimcp_action_potential.c`

**API**:
```c
typedef struct {
    // Membrane potential
    float V;  // mV

    // Gating variables (0-1)
    float m;  // Sodium activation
    float h;  // Sodium inactivation
    float n;  // Potassium activation

    // Conductances (mS/cm²)
    float g_Na;  // Sodium (default: 120)
    float g_K;   // Potassium (default: 36)
    float g_L;   // Leak (default: 0.3)

    // Reversal potentials (mV)
    float E_Na;  // +55 mV
    float E_K;   // -77 mV
    float E_L;   // -54.3 mV

    // Membrane capacitance
    float C_m;   // µF/cm² (default: 1.0)

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_hh_neuron_t;

nimcp_status_t nimcp_hh_neuron_init(nimcp_hh_neuron_t* neuron,
                                     nimcp_brain_t* brain);
nimcp_status_t nimcp_hh_neuron_update(nimcp_hh_neuron_t* neuron,
                                       float I_ext, float dt);
nimcp_status_t nimcp_hh_neuron_get_spike(nimcp_hh_neuron_t* neuron,
                                          bool* spiked);
```

#### 3.2 Bridges

| Bridge | File | Purpose |
|--------|------|---------|
| SNN Bridge | `nimcp_hh_snn_bridge.h/c` | Replace simple neurons |
| Plasticity Bridge | `nimcp_hh_plasticity_bridge.h/c` | Channel-dependent plasticity |
| FEP Bridge | `nimcp_hh_fep_bridge.h/c` | Precision from ion dynamics |
| Substrate Bridge | `nimcp_hh_substrate_bridge.h/c` | Bio-async messaging |
| Thalamic Bridge | `nimcp_hh_thalamic_bridge.h/c` | Gating modulation |
| Immune Bridge | `nimcp_hh_immune_bridge.h/c` | Channelopathy detection |
| Hub Bridge | `nimcp_hh_hub_bridge.h/c` | Cognitive integration |

#### 3.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/physics/biophysics/test_hodgkin_huxley.cpp` | 25 |
| Unit | `test/unit/physics/biophysics/test_ion_channels.cpp` | 25 |
| Unit | `test/unit/physics/biophysics/test_gating_variables.cpp` | 20 |
| Unit | `test/unit/physics/biophysics/test_action_potential.cpp` | 20 |
| Unit | `test/unit/physics/biophysics/test_hh_bridges.cpp` | 15 |
| Integration | `test/integration/physics/biophysics/test_hh_snn_integration.cpp` | 10 |
| Regression | `test/regression/physics/biophysics/test_hh_regression.cpp` | 5 |
| E2E | `test/e2e/e2e_test_hh_pipeline.cpp` | 5 |

---

### Phase AC-4: pH Dynamics

**Priority**: Low
**Dependencies**: Metabolic systems
**Est. LOC**: 3,000 | **Tests**: 60

#### 4.1 Core Module

**Files**:
- `include/chemistry/ph/nimcp_ph_dynamics.h`
- `include/chemistry/ph/nimcp_proton_pumps.h`
- `include/chemistry/ph/nimcp_buffer_systems.h`
- `src/chemistry/ph/nimcp_ph_dynamics.c`
- `src/chemistry/ph/nimcp_proton_pumps.c`
- `src/chemistry/ph/nimcp_buffer_systems.c`

**API**:
```c
typedef struct {
    float extracellular_pH;        // Typically 7.4
    float intracellular_pH;        // Typically 7.2
    float synaptic_vesicle_pH;     // Typically 5.5

    // Proton pumps
    float h_atpase_activity;       // V-ATPase in vesicles
    float na_h_exchanger_activity; // NHE at membrane

    // Buffering capacity
    float bicarbonate_buffer;      // HCO3- system
    float protein_buffer;

    // Effects
    float ph_conductance_modifier;

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
} nimcp_ph_system_t;
```

#### 4.2 Bridges

Standard 7-bridge pattern (SNN, Plasticity, FEP, Substrate, Thalamic, Immune, Hub)

#### 4.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/chemistry/ph/test_ph_dynamics.cpp` | 20 |
| Unit | `test/unit/chemistry/ph/test_proton_pumps.cpp` | 15 |
| Unit | `test/unit/chemistry/ph/test_buffer_systems.cpp` | 10 |
| Unit | `test/unit/chemistry/ph/test_ph_bridges.cpp` | 10 |
| Integration | `test/integration/chemistry/ph/test_ph_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_ph_pipeline.cpp` | 5 |

---

### Phase AC-5: Nitric Oxide Signaling

**Priority**: Medium (important for LTP)
**Dependencies**: Calcium dynamics, Plasticity
**Est. LOC**: 4,000 | **Tests**: 80

#### 5.1 Core Module

**Files**:
- `include/chemistry/gasotransmitters/nimcp_nitric_oxide.h`
- `include/chemistry/gasotransmitters/nimcp_nos_system.h`
- `include/chemistry/gasotransmitters/nimcp_cgmp_pathway.h`
- `include/chemistry/gasotransmitters/nimcp_retrograde_signaling.h`
- `src/chemistry/gasotransmitters/nimcp_nitric_oxide.c`
- `src/chemistry/gasotransmitters/nimcp_nos_system.c`
- `src/chemistry/gasotransmitters/nimcp_cgmp_pathway.c`
- `src/chemistry/gasotransmitters/nimcp_retrograde_signaling.c`

**API**:
```c
typedef struct {
    // NO production
    float nos_activity;            // Nitric oxide synthase
    float no_concentration;        // nM
    float no_diffusion_radius;     // µm

    // Effects
    float cgmp_concentration;      // Cyclic GMP
    float vasodilation_factor;     // Blood flow increase

    // Retrograde plasticity
    float presynaptic_potentiation;

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_no_system_t;
```

#### 5.2 Bridges

Standard 7-bridge pattern

#### 5.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/chemistry/gasotransmitters/test_nitric_oxide.cpp` | 20 |
| Unit | `test/unit/chemistry/gasotransmitters/test_nos_system.cpp` | 15 |
| Unit | `test/unit/chemistry/gasotransmitters/test_cgmp_pathway.cpp` | 15 |
| Unit | `test/unit/chemistry/gasotransmitters/test_retrograde.cpp` | 10 |
| Unit | `test/unit/chemistry/gasotransmitters/test_no_bridges.cpp` | 10 |
| Integration | `test/integration/chemistry/gasotransmitters/test_no_plasticity_integration.cpp` | 5 |
| Regression | `test/regression/chemistry/gasotransmitters/test_no_regression.cpp` | 5 |
| E2E | `test/e2e/e2e_test_no_pipeline.cpp` | 5 |

---

### Phase AC-6: Epigenetics

**Priority**: Medium (trauma, addiction modeling)
**Dependencies**: Gene expression, Second messengers
**Est. LOC**: 5,000 | **Tests**: 100

#### 6.1 Core Module

**Files**:
- `include/biology/epigenetics/nimcp_epigenetics.h`
- `include/biology/epigenetics/nimcp_dna_methylation.h`
- `include/biology/epigenetics/nimcp_histone_modification.h`
- `include/biology/epigenetics/nimcp_chromatin_state.h`
- `src/biology/epigenetics/nimcp_epigenetics.c`
- `src/biology/epigenetics/nimcp_dna_methylation.c`
- `src/biology/epigenetics/nimcp_histone_modification.c`
- `src/biology/epigenetics/nimcp_chromatin_state.c`

**API**:
```c
typedef struct {
    // DNA methylation
    float promoter_methylation[NUM_GENES];  // 0-1

    // Histone modifications
    float histone_acetylation[NUM_GENES];   // Open chromatin
    float histone_methylation[NUM_GENES];   // Closed chromatin

    // Enzymes
    float dnmt_activity;           // DNA methyltransferase
    float hdac_activity;           // Histone deacetylase
    float hat_activity;            // Histone acetyltransferase

    // Gene accessibility
    float gene_expression_potential[NUM_GENES];

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_epigenetic_system_t;
```

#### 6.2 Bridges

Standard 7-bridge pattern

#### 6.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/biology/epigenetics/test_epigenetics.cpp` | 25 |
| Unit | `test/unit/biology/epigenetics/test_dna_methylation.cpp` | 20 |
| Unit | `test/unit/biology/epigenetics/test_histone_modification.cpp` | 20 |
| Unit | `test/unit/biology/epigenetics/test_chromatin_state.cpp` | 15 |
| Unit | `test/unit/biology/epigenetics/test_epigenetics_bridges.cpp` | 10 |
| Integration | `test/integration/biology/epigenetics/test_epigenetics_stress_integration.cpp` | 5 |
| Regression | `test/regression/biology/epigenetics/test_epigenetics_regression.cpp` | 5 |
| E2E | `test/e2e/e2e_test_epigenetics_pipeline.cpp` | 5 |

---

### Phase AC-7: Neurogenesis

**Priority**: Low (pattern separation, depression)
**Dependencies**: Hippocampus, BDNF
**Est. LOC**: 4,500 | **Tests**: 90

#### 7.1 Core Module

**Files**:
- `include/biology/neurogenesis/nimcp_neurogenesis.h`
- `include/biology/neurogenesis/nimcp_neural_stem_cells.h`
- `include/biology/neurogenesis/nimcp_neuron_maturation.h`
- `include/biology/neurogenesis/nimcp_neuron_integration.h`
- `src/biology/neurogenesis/nimcp_neurogenesis.c`
- `src/biology/neurogenesis/nimcp_neural_stem_cells.c`
- `src/biology/neurogenesis/nimcp_neuron_maturation.c`
- `src/biology/neurogenesis/nimcp_neuron_integration.c`

**API**:
```c
typedef struct {
    // Neural stem cells
    uint32_t num_stem_cells;
    float proliferation_rate;      // Cells/day

    // Newborn neurons
    uint32_t num_newborn_neurons;
    float* neuron_ages;            // Days since birth
    float* integration_progress;   // 0-1

    // Survival factors
    float bdnf_level;
    float running_activity;        // Exercise
    float learning_activity;       // Cognitive stimulation

    // Survival rate
    float survival_probability;    // ~50% die in first month

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_neurogenesis_t;
```

#### 7.2 Bridges

Standard 7-bridge pattern

#### 7.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/biology/neurogenesis/test_neurogenesis.cpp` | 25 |
| Unit | `test/unit/biology/neurogenesis/test_neural_stem_cells.cpp` | 20 |
| Unit | `test/unit/biology/neurogenesis/test_neuron_maturation.cpp` | 15 |
| Unit | `test/unit/biology/neurogenesis/test_neuron_integration.cpp` | 15 |
| Unit | `test/unit/biology/neurogenesis/test_neurogenesis_bridges.cpp` | 10 |
| Integration | `test/integration/biology/neurogenesis/test_neurogenesis_hippocampus_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_neurogenesis_pipeline.cpp` | 5 |

---

### Phase AC-8: Neurovascular Coupling

**Priority**: Low (fMRI validation)
**Dependencies**: Astrocytes, NO signaling
**Est. LOC**: 5,000 | **Tests**: 100

#### 8.1 Core Module

**Files**:
- `include/biology/neurovascular/nimcp_neurovascular.h`
- `include/biology/neurovascular/nimcp_blood_flow.h`
- `include/biology/neurovascular/nimcp_bold_signal.h`
- `include/biology/neurovascular/nimcp_hemodynamic_response.h`
- `src/biology/neurovascular/nimcp_neurovascular.c`
- `src/biology/neurovascular/nimcp_blood_flow.c`
- `src/biology/neurovascular/nimcp_bold_signal.c`
- `src/biology/neurovascular/nimcp_hemodynamic_response.c`

**API**:
```c
typedef struct {
    // Vascular state
    float vessel_diameter;         // µm
    float blood_flow_rate;         // mL/min/100g
    float blood_oxygen_level;      // %

    // BOLD signal components
    float deoxyhemoglobin;         // Paramagnetic
    float cerebral_blood_volume;   // CBV
    float cerebral_blood_flow;     // CBF
    float oxygen_extraction;       // OEF

    // Coupling mechanisms
    float astrocyte_calcium;       // Links to astrocytes
    float nitric_oxide;            // Vasodilator
    float prostaglandins;          // Vasodilator

    // BOLD response
    float bold_signal;             // % change
    float hemodynamic_response[20]; // HRF kernel

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_neurovascular_t;
```

#### 8.2 Bridges

Standard 7-bridge pattern

#### 8.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/biology/neurovascular/test_neurovascular.cpp` | 25 |
| Unit | `test/unit/biology/neurovascular/test_blood_flow.cpp` | 20 |
| Unit | `test/unit/biology/neurovascular/test_bold_signal.cpp` | 20 |
| Unit | `test/unit/biology/neurovascular/test_hemodynamic_response.cpp` | 15 |
| Unit | `test/unit/biology/neurovascular/test_neurovascular_bridges.cpp` | 10 |
| Integration | `test/integration/biology/neurovascular/test_neurovascular_astrocyte_integration.cpp` | 5 |
| Regression | `test/regression/biology/neurovascular/test_neurovascular_regression.cpp` | 5 |
| E2E | `test/e2e/e2e_test_neurovascular_pipeline.cpp` | 5 |

---

### Phase AC-9: Non-Equilibrium Thermodynamics (Complete)

**Priority**: Medium
**Dependencies**: FEP, Metabolic systems
**Est. LOC**: 4,000 | **Tests**: 80

#### 9.1 Core Module

**Files**:
- `include/physics/thermodynamics/nimcp_thermodynamics.h`
- `include/physics/thermodynamics/nimcp_energy_budget.h`
- `include/physics/thermodynamics/nimcp_entropy_production.h`
- `include/physics/thermodynamics/nimcp_landauer_cost.h`
- `src/physics/thermodynamics/nimcp_thermodynamics.c`
- `src/physics/thermodynamics/nimcp_energy_budget.c`
- `src/physics/thermodynamics/nimcp_entropy_production.c`
- `src/physics/thermodynamics/nimcp_landauer_cost.c`

**API**:
```c
typedef struct {
    // Energy tracking
    float total_energy_consumed;   // Joules
    float power_consumption;       // Watts
    float heat_dissipation;        // W/m³

    // Entropy production
    float entropy_production_rate; // J/(K·s)
    float free_energy_dissipated;

    // Metabolic constraints
    float atp_available;           // Mol
    float atp_consumption_rate;    // Mol/s
    float oxygen_delivery_rate;    // Mol/s

    // Efficiency metrics
    float computational_efficiency; // Ops/Joule
    float thermodynamic_efficiency; // Useful work / total energy

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_thermodynamic_state_t;
```

#### 9.2 Bridges

Standard 7-bridge pattern

#### 9.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/physics/thermodynamics/test_thermodynamics.cpp` | 20 |
| Unit | `test/unit/physics/thermodynamics/test_energy_budget.cpp` | 20 |
| Unit | `test/unit/physics/thermodynamics/test_entropy_production.cpp` | 15 |
| Unit | `test/unit/physics/thermodynamics/test_landauer_cost.cpp` | 10 |
| Unit | `test/unit/physics/thermodynamics/test_thermodynamics_bridges.cpp` | 10 |
| Integration | `test/integration/physics/thermodynamics/test_thermodynamics_fep_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_thermodynamics_pipeline.cpp` | 5 |

---

## PART II: NEW BRAIN REGIONS IMPLEMENTATION

### Phase BR-1: Locus Coeruleus (Norepinephrine Center)

**Priority**: P1 - Critical for arousal/exploration
**Dependencies**: Neuromodulators, Attention
**Est. LOC**: 5,000 | **Tests**: 100

#### 1.1 Core Module

**Files**:
- `include/core/brain/regions/locus_coeruleus/nimcp_locus_coeruleus.h`
- `include/core/brain/regions/locus_coeruleus/nimcp_lc_adapter.h`
- `include/core/brain/regions/locus_coeruleus/nimcp_norepinephrine_release.h`
- `include/core/brain/regions/locus_coeruleus/nimcp_arousal_modulation.h`
- `include/core/brain/regions/locus_coeruleus/nimcp_novelty_detection.h`
- `src/core/brain/regions/locus_coeruleus/nimcp_locus_coeruleus.c`
- `src/core/brain/regions/locus_coeruleus/nimcp_lc_adapter.c`
- `src/core/brain/regions/locus_coeruleus/nimcp_norepinephrine_release.c`
- `src/core/brain/regions/locus_coeruleus/nimcp_arousal_modulation.c`
- `src/core/brain/regions/locus_coeruleus/nimcp_novelty_detection.c`

**API**:
```c
typedef struct {
    // NE release parameters
    float tonic_firing_rate;       // Hz (baseline ~1-3 Hz)
    float phasic_firing_rate;      // Hz (burst up to 20 Hz)
    float ne_concentration;        // nM
    float ne_release_rate;         // nM/s

    // Global arousal
    float arousal_level;           // 0-1
    float alertness;               // 0-1
    float vigilance;               // 0-1

    // Novelty/surprise detection
    float novelty_signal;          // 0-1
    float surprise_magnitude;      // 0-1
    float exploration_drive;       // 0-1

    // Mode switching
    bool phasic_mode;              // true = focused, false = exploratory
    float mode_switch_threshold;

    // Projections (volume transmission)
    nimcp_projection_t cortical_projections;
    nimcp_projection_t hippocampal_projections;
    nimcp_projection_t amygdala_projections;

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_locus_coeruleus_t;

nimcp_status_t nimcp_lc_init(nimcp_locus_coeruleus_t* lc, nimcp_brain_t* brain);
nimcp_status_t nimcp_lc_update(nimcp_locus_coeruleus_t* lc, float dt);
nimcp_status_t nimcp_lc_detect_novelty(nimcp_locus_coeruleus_t* lc,
                                        const float* sensory_input,
                                        float* novelty_score);
nimcp_status_t nimcp_lc_modulate_arousal(nimcp_locus_coeruleus_t* lc,
                                          float target_arousal);
nimcp_status_t nimcp_lc_trigger_attention_reset(nimcp_locus_coeruleus_t* lc);
```

#### 1.2 Bridges

| Bridge | File | Purpose |
|--------|------|---------|
| SNN Bridge | `nimcp_lc_snn_bridge.h/c` | NE modulation of SNN |
| Plasticity Bridge | `nimcp_lc_plasticity_bridge.h/c` | NE-dependent plasticity |
| FEP Bridge | `nimcp_lc_fep_bridge.h/c` | Precision modulation |
| Substrate Bridge | `nimcp_lc_substrate_bridge.h/c` | Bio-async messaging |
| Thalamic Bridge | `nimcp_lc_thalamic_bridge.h/c` | Arousal gating |
| Immune Bridge | `nimcp_lc_immune_bridge.h/c` | Stress response |
| Hub Bridge | `nimcp_lc_hub_bridge.h/c` | Cognitive integration |
| Quantum Bridge | `nimcp_lc_quantum_bridge.h/c` | Quantum coherence modulation |

#### 1.3 Brain Factory Integration

**File**: `src/core/brain/factory/init/nimcp_brain_init_locus_coeruleus.c`

#### 1.4 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/locus_coeruleus/test_lc_core.cpp` | 25 |
| Unit | `test/unit/core/brain/regions/locus_coeruleus/test_ne_release.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/locus_coeruleus/test_novelty_detection.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/locus_coeruleus/test_arousal_modulation.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/locus_coeruleus/test_lc_bridges.cpp` | 15 |
| Integration | `test/integration/core/brain/regions/locus_coeruleus/test_lc_integration.cpp` | 10 |
| Regression | `test/regression/core/brain/regions/locus_coeruleus/test_lc_regression.cpp` | 5 |
| E2E | `test/e2e/e2e_test_locus_coeruleus_pipeline.cpp` | 5 |

---

### Phase BR-2: Ventral Tegmental Area (Dopamine/Reward)

**Priority**: P1 - Critical for reward learning
**Dependencies**: Basal Ganglia, PFC
**Est. LOC**: 5,500 | **Tests**: 110

#### 2.1 Core Module

**Files**:
- `include/core/brain/regions/vta/nimcp_vta.h`
- `include/core/brain/regions/vta/nimcp_vta_adapter.h`
- `include/core/brain/regions/vta/nimcp_dopamine_release.h`
- `include/core/brain/regions/vta/nimcp_reward_prediction_error.h`
- `include/core/brain/regions/vta/nimcp_incentive_salience.h`
- `src/core/brain/regions/vta/nimcp_vta.c`
- `src/core/brain/regions/vta/nimcp_vta_adapter.c`
- `src/core/brain/regions/vta/nimcp_dopamine_release.c`
- `src/core/brain/regions/vta/nimcp_reward_prediction_error.c`
- `src/core/brain/regions/vta/nimcp_incentive_salience.c`

**API**:
```c
typedef struct {
    // Dopamine dynamics
    float da_firing_rate;          // Hz
    float da_concentration;        // nM
    float da_release_rate;         // nM/s

    // Reward prediction error
    float expected_reward;
    float actual_reward;
    float prediction_error;        // RPE = actual - expected

    // Incentive salience
    float wanting_signal;          // Motivation
    float liking_signal;           // Hedonic impact

    // Goal-directed behavior
    float goal_value;
    float effort_cost;
    float net_utility;

    // Projections
    nimcp_projection_t nac_projection;    // Nucleus accumbens
    nimcp_projection_t pfc_projection;    // Prefrontal cortex
    nimcp_projection_t hippocampal_projection;

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_vta_t;

nimcp_status_t nimcp_vta_init(nimcp_vta_t* vta, nimcp_brain_t* brain);
nimcp_status_t nimcp_vta_update(nimcp_vta_t* vta, float dt);
nimcp_status_t nimcp_vta_compute_rpe(nimcp_vta_t* vta,
                                      float reward,
                                      float* rpe);
nimcp_status_t nimcp_vta_modulate_motivation(nimcp_vta_t* vta,
                                              float goal_value,
                                              float* motivation);
```

#### 2.2 Bridges

Standard 8-bridge pattern (including Quantum Bridge)

#### 2.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/vta/test_vta_core.cpp` | 25 |
| Unit | `test/unit/core/brain/regions/vta/test_dopamine_release.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/vta/test_reward_prediction_error.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/vta/test_incentive_salience.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/vta/test_vta_bridges.cpp` | 15 |
| Integration | `test/integration/core/brain/regions/vta/test_vta_basal_ganglia_integration.cpp` | 10 |
| Regression | `test/regression/core/brain/regions/vta/test_vta_regression.cpp` | 5 |
| E2E | `test/e2e/e2e_test_vta_pipeline.cpp` | 5 |

---

### Phase BR-3: Raphe Nuclei (Serotonin/Mood)

**Priority**: P1 - Critical for mood regulation
**Dependencies**: Emotional system, Sleep-wake
**Est. LOC**: 5,000 | **Tests**: 100

#### 3.1 Core Module

**Files**:
- `include/core/brain/regions/raphe/nimcp_raphe.h`
- `include/core/brain/regions/raphe/nimcp_raphe_adapter.h`
- `include/core/brain/regions/raphe/nimcp_serotonin_release.h`
- `include/core/brain/regions/raphe/nimcp_mood_regulation.h`
- `include/core/brain/regions/raphe/nimcp_impulse_control.h`
- `include/core/brain/regions/raphe/nimcp_temporal_discounting.h`
- `src/core/brain/regions/raphe/nimcp_raphe.c`
- `src/core/brain/regions/raphe/nimcp_raphe_adapter.c`
- `src/core/brain/regions/raphe/nimcp_serotonin_release.c`
- `src/core/brain/regions/raphe/nimcp_mood_regulation.c`
- `src/core/brain/regions/raphe/nimcp_impulse_control.c`
- `src/core/brain/regions/raphe/nimcp_temporal_discounting.c`

**API**:
```c
typedef struct {
    // 5-HT dynamics
    float serotonin_firing_rate;   // Hz
    float serotonin_concentration; // nM
    float serotonin_release_rate;  // nM/s

    // Mood state
    float mood_valence;            // -1 to +1
    float emotional_stability;     // 0-1
    float anxiety_level;           // 0-1

    // Impulse control
    float inhibition_strength;     // 0-1
    float patience_factor;         // Delay tolerance
    float risk_aversion;           // 0-1

    // Temporal discounting
    float discount_rate;           // Steepness of discounting
    float future_orientation;      // 0-1

    // Sleep-wake coupling
    float sleep_pressure;
    float wake_promotion;

    // Projections
    nimcp_projection_t cortical_projections;
    nimcp_projection_t limbic_projections;
    nimcp_projection_t hypothalamic_projections;

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_raphe_t;
```

#### 3.2 Bridges

Standard 8-bridge pattern

#### 3.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/raphe/test_raphe_core.cpp` | 25 |
| Unit | `test/unit/core/brain/regions/raphe/test_serotonin_release.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/raphe/test_mood_regulation.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/raphe/test_impulse_control.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/raphe/test_temporal_discounting.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/raphe/test_raphe_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/raphe/test_raphe_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_raphe_pipeline.cpp` | 5 |

---

### Phase BR-4: Habenula (Aversive Learning)

**Priority**: P1 - Critical for balanced learning
**Dependencies**: VTA (inhibitory), Raphe
**Est. LOC**: 4,500 | **Tests**: 90

#### 4.1 Core Module

**Files**:
- `include/core/brain/regions/habenula/nimcp_habenula.h`
- `include/core/brain/regions/habenula/nimcp_habenula_adapter.h`
- `include/core/brain/regions/habenula/nimcp_negative_rpe.h`
- `include/core/brain/regions/habenula/nimcp_avoidance_learning.h`
- `include/core/brain/regions/habenula/nimcp_disappointment_signal.h`
- `src/core/brain/regions/habenula/nimcp_habenula.c`
- `src/core/brain/regions/habenula/nimcp_habenula_adapter.c`
- `src/core/brain/regions/habenula/nimcp_negative_rpe.c`
- `src/core/brain/regions/habenula/nimcp_avoidance_learning.c`
- `src/core/brain/regions/habenula/nimcp_disappointment_signal.c`

**API**:
```c
typedef struct {
    // Activity
    float firing_rate;             // Hz
    float activation_level;        // 0-1

    // Negative reward prediction error
    float negative_rpe;            // Disappointment signal
    float expected_punishment;
    float actual_punishment;

    // Avoidance learning
    float avoidance_strength;      // 0-1
    float learned_helplessness;    // 0-1 (pathological)

    // VTA inhibition
    float vta_inhibition_signal;   // Blocks dopamine

    // Depression modeling
    float anhedonia_level;         // 0-1
    float motivational_deficit;    // 0-1

    // Projections
    nimcp_projection_t vta_projection;     // Inhibitory
    nimcp_projection_t raphe_projection;   // Modulates 5-HT

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_habenula_t;
```

#### 4.2 Bridges

Standard 8-bridge pattern

#### 4.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/habenula/test_habenula_core.cpp` | 25 |
| Unit | `test/unit/core/brain/regions/habenula/test_negative_rpe.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/habenula/test_avoidance_learning.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/habenula/test_disappointment.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/habenula/test_habenula_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/habenula/test_habenula_vta_integration.cpp` | 10 |
| E2E | `test/e2e/e2e_test_habenula_pipeline.cpp` | 5 |

---

### Phase BR-5: Entorhinal Cortex (Grid Cells/Memory Gateway)

**Priority**: P2 - Critical for memory circuit
**Dependencies**: Hippocampus
**Est. LOC**: 6,000 | **Tests**: 120

#### 5.1 Core Module

**Files**:
- `include/core/brain/regions/entorhinal/nimcp_entorhinal.h`
- `include/core/brain/regions/entorhinal/nimcp_entorhinal_adapter.h`
- `include/core/brain/regions/entorhinal/nimcp_grid_cells.h`
- `include/core/brain/regions/entorhinal/nimcp_border_cells.h`
- `include/core/brain/regions/entorhinal/nimcp_head_direction_cells.h`
- `include/core/brain/regions/entorhinal/nimcp_path_integration.h`
- `include/core/brain/regions/entorhinal/nimcp_memory_gateway.h`
- `src/core/brain/regions/entorhinal/nimcp_entorhinal.c`
- `src/core/brain/regions/entorhinal/nimcp_entorhinal_adapter.c`
- `src/core/brain/regions/entorhinal/nimcp_grid_cells.c`
- `src/core/brain/regions/entorhinal/nimcp_border_cells.c`
- `src/core/brain/regions/entorhinal/nimcp_head_direction_cells.c`
- `src/core/brain/regions/entorhinal/nimcp_path_integration.c`
- `src/core/brain/regions/entorhinal/nimcp_memory_gateway.c`

**API**:
```c
typedef struct {
    // Grid cells
    nimcp_grid_cell_t* grid_cells;
    uint32_t num_grid_cells;
    float grid_scale;              // Grid spacing
    float grid_orientation;        // Grid orientation

    // Border cells
    nimcp_border_cell_t* border_cells;
    uint32_t num_border_cells;

    // Head direction cells
    nimcp_hd_cell_t* hd_cells;
    uint32_t num_hd_cells;
    float current_heading;         // Degrees

    // Path integration
    float position[3];             // x, y, z
    float velocity[3];
    float accumulated_error;

    // Memory gateway
    float encoding_gate;           // 0-1
    float retrieval_gate;          // 0-1
    float memory_binding_strength;

    // Bio-async
    nimcp_bio_async_handler_t bio_handler;
    nimcp_bio_router_t* router;

    // Immune
    nimcp_immune_sensor_t immune_sensor;

    // Logging
    nimcp_logger_t* logger;
    nimcp_metrics_t* metrics;
} nimcp_entorhinal_t;
```

#### 5.2 Bridges

Standard 8-bridge pattern

#### 5.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/entorhinal/test_entorhinal_core.cpp` | 25 |
| Unit | `test/unit/core/brain/regions/entorhinal/test_grid_cells.cpp` | 25 |
| Unit | `test/unit/core/brain/regions/entorhinal/test_border_cells.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/entorhinal/test_hd_cells.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/entorhinal/test_path_integration.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/entorhinal/test_memory_gateway.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/entorhinal/test_entorhinal_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/entorhinal/test_entorhinal_hippocampus_integration.cpp` | 10 |
| E2E | `test/e2e/e2e_test_entorhinal_pipeline.cpp` | 5 |

---

### Phase BR-6: Perirhinal Cortex (Object Recognition)

**Priority**: P2
**Dependencies**: Entorhinal, Visual cortex
**Est. LOC**: 4,000 | **Tests**: 80

#### 6.1 Core Module

**Files**:
- `include/core/brain/regions/perirhinal/nimcp_perirhinal.h`
- `include/core/brain/regions/perirhinal/nimcp_perirhinal_adapter.h`
- `include/core/brain/regions/perirhinal/nimcp_object_recognition.h`
- `include/core/brain/regions/perirhinal/nimcp_familiarity_signal.h`
- `include/core/brain/regions/perirhinal/nimcp_item_memory.h`
- `src/core/brain/regions/perirhinal/nimcp_perirhinal.c`
- `src/core/brain/regions/perirhinal/nimcp_perirhinal_adapter.c`
- `src/core/brain/regions/perirhinal/nimcp_object_recognition.c`
- `src/core/brain/regions/perirhinal/nimcp_familiarity_signal.c`
- `src/core/brain/regions/perirhinal/nimcp_item_memory.c`

#### 6.2 Bridges

Standard 8-bridge pattern

#### 6.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/perirhinal/test_perirhinal_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/perirhinal/test_object_recognition.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/perirhinal/test_familiarity.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/perirhinal/test_item_memory.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/perirhinal/test_perirhinal_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/perirhinal/test_perirhinal_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_perirhinal_pipeline.cpp` | 5 |

---

### Phase BR-7: Parahippocampal Cortex (Scene/Context)

**Priority**: P2
**Dependencies**: Entorhinal, Visual cortex
**Est. LOC**: 4,000 | **Tests**: 80

#### 7.1 Core Module

**Files**:
- `include/core/brain/regions/parahippocampal/nimcp_parahippocampal.h`
- `include/core/brain/regions/parahippocampal/nimcp_parahippocampal_adapter.h`
- `include/core/brain/regions/parahippocampal/nimcp_scene_recognition.h`
- `include/core/brain/regions/parahippocampal/nimcp_context_processing.h`
- `include/core/brain/regions/parahippocampal/nimcp_spatial_layout.h`
- `src/core/brain/regions/parahippocampal/nimcp_parahippocampal.c`
- `src/core/brain/regions/parahippocampal/nimcp_parahippocampal_adapter.c`
- `src/core/brain/regions/parahippocampal/nimcp_scene_recognition.c`
- `src/core/brain/regions/parahippocampal/nimcp_context_processing.c`
- `src/core/brain/regions/parahippocampal/nimcp_spatial_layout.c`

#### 7.2 Bridges

Standard 8-bridge pattern

#### 7.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/parahippocampal/test_parahippocampal_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/parahippocampal/test_scene_recognition.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/parahippocampal/test_context_processing.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/parahippocampal/test_spatial_layout.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/parahippocampal/test_parahippocampal_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/parahippocampal/test_parahippocampal_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_parahippocampal_pipeline.cpp` | 5 |

---

### Phase BR-8: Mammillary Bodies (Memory Relay)

**Priority**: P2
**Dependencies**: Hippocampus, Thalamus
**Est. LOC**: 3,000 | **Tests**: 60

#### 8.1 Core Module

**Files**:
- `include/core/brain/regions/mammillary/nimcp_mammillary.h`
- `include/core/brain/regions/mammillary/nimcp_mammillary_adapter.h`
- `include/core/brain/regions/mammillary/nimcp_papez_circuit.h`
- `include/core/brain/regions/mammillary/nimcp_spatial_memory_relay.h`
- `src/core/brain/regions/mammillary/nimcp_mammillary.c`
- `src/core/brain/regions/mammillary/nimcp_mammillary_adapter.c`
- `src/core/brain/regions/mammillary/nimcp_papez_circuit.c`
- `src/core/brain/regions/mammillary/nimcp_spatial_memory_relay.c`

#### 8.2 Bridges

Standard 8-bridge pattern

#### 8.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/mammillary/test_mammillary_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/mammillary/test_papez_circuit.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/mammillary/test_spatial_memory_relay.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/mammillary/test_mammillary_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/mammillary/test_mammillary_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_mammillary_pipeline.cpp` | 5 |

---

### Phase BR-9: Somatosensory Cortex (Touch/Proprioception)

**Priority**: P3
**Dependencies**: Thalamus, Motor cortex
**Est. LOC**: 5,000 | **Tests**: 100

#### 9.1 Core Module

**Files**:
- `include/core/brain/regions/somatosensory/nimcp_somatosensory.h`
- `include/core/brain/regions/somatosensory/nimcp_somatosensory_adapter.h`
- `include/core/brain/regions/somatosensory/nimcp_body_map.h`
- `include/core/brain/regions/somatosensory/nimcp_tactile_processing.h`
- `include/core/brain/regions/somatosensory/nimcp_proprioception.h`
- `include/core/brain/regions/somatosensory/nimcp_pain_processing.h`
- `src/core/brain/regions/somatosensory/nimcp_somatosensory.c`
- `src/core/brain/regions/somatosensory/nimcp_somatosensory_adapter.c`
- `src/core/brain/regions/somatosensory/nimcp_body_map.c`
- `src/core/brain/regions/somatosensory/nimcp_tactile_processing.c`
- `src/core/brain/regions/somatosensory/nimcp_proprioception.c`
- `src/core/brain/regions/somatosensory/nimcp_pain_processing.c`

#### 9.2 Bridges

Standard 8-bridge pattern

#### 9.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/somatosensory/test_somatosensory_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/somatosensory/test_body_map.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/somatosensory/test_tactile_processing.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/somatosensory/test_proprioception.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/somatosensory/test_pain_processing.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/somatosensory/test_somatosensory_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/somatosensory/test_somatosensory_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_somatosensory_pipeline.cpp` | 5 |

---

### Phase BR-10: Olfactory/Piriform Cortex (Smell)

**Priority**: P3
**Dependencies**: Amygdala, Entorhinal
**Est. LOC**: 3,500 | **Tests**: 70

#### 10.1 Core Module

**Files**:
- `include/core/brain/regions/olfactory/nimcp_olfactory.h`
- `include/core/brain/regions/olfactory/nimcp_olfactory_adapter.h`
- `include/core/brain/regions/olfactory/nimcp_odor_identification.h`
- `include/core/brain/regions/olfactory/nimcp_olfactory_memory.h`
- `include/core/brain/regions/olfactory/nimcp_odor_pattern_completion.h`
- `src/core/brain/regions/olfactory/nimcp_olfactory.c`
- `src/core/brain/regions/olfactory/nimcp_olfactory_adapter.c`
- `src/core/brain/regions/olfactory/nimcp_odor_identification.c`
- `src/core/brain/regions/olfactory/nimcp_olfactory_memory.c`
- `src/core/brain/regions/olfactory/nimcp_odor_pattern_completion.c`

#### 10.2 Bridges

Standard 8-bridge pattern

#### 10.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/olfactory/test_olfactory_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/olfactory/test_odor_identification.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/olfactory/test_olfactory_memory.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/olfactory/test_olfactory_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/olfactory/test_olfactory_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_olfactory_pipeline.cpp` | 5 |

---

### Phase BR-11: Gustatory Cortex (Taste)

**Priority**: P3
**Dependencies**: Insula, Orbitofrontal
**Est. LOC**: 3,000 | **Tests**: 60

#### 11.1 Core Module

**Files**:
- `include/core/brain/regions/gustatory/nimcp_gustatory.h`
- `include/core/brain/regions/gustatory/nimcp_gustatory_adapter.h`
- `include/core/brain/regions/gustatory/nimcp_taste_processing.h`
- `include/core/brain/regions/gustatory/nimcp_food_reward.h`
- `include/core/brain/regions/gustatory/nimcp_disgust_response.h`
- `src/core/brain/regions/gustatory/nimcp_gustatory.c`
- `src/core/brain/regions/gustatory/nimcp_gustatory_adapter.c`
- `src/core/brain/regions/gustatory/nimcp_taste_processing.c`
- `src/core/brain/regions/gustatory/nimcp_food_reward.c`
- `src/core/brain/regions/gustatory/nimcp_disgust_response.c`

#### 11.2 Bridges

Standard 8-bridge pattern

#### 11.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/gustatory/test_gustatory_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/gustatory/test_taste_processing.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/gustatory/test_food_reward.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/gustatory/test_gustatory_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/gustatory/test_gustatory_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_gustatory_pipeline.cpp` | 5 |

---

### Phase BR-12: Orbitofrontal Cortex (Value/Decision)

**Priority**: P4
**Dependencies**: PFC, Amygdala
**Est. LOC**: 5,000 | **Tests**: 100

#### 12.1 Core Module

**Files**:
- `include/core/brain/regions/orbitofrontal/nimcp_orbitofrontal.h`
- `include/core/brain/regions/orbitofrontal/nimcp_ofc_adapter.h`
- `include/core/brain/regions/orbitofrontal/nimcp_value_representation.h`
- `include/core/brain/regions/orbitofrontal/nimcp_reversal_learning.h`
- `include/core/brain/regions/orbitofrontal/nimcp_expected_outcome.h`
- `include/core/brain/regions/orbitofrontal/nimcp_social_reward.h`
- `src/core/brain/regions/orbitofrontal/nimcp_orbitofrontal.c`
- `src/core/brain/regions/orbitofrontal/nimcp_ofc_adapter.c`
- `src/core/brain/regions/orbitofrontal/nimcp_value_representation.c`
- `src/core/brain/regions/orbitofrontal/nimcp_reversal_learning.c`
- `src/core/brain/regions/orbitofrontal/nimcp_expected_outcome.c`
- `src/core/brain/regions/orbitofrontal/nimcp_social_reward.c`

#### 12.2 Bridges

Standard 8-bridge pattern

#### 12.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/orbitofrontal/test_ofc_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/orbitofrontal/test_value_representation.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/orbitofrontal/test_reversal_learning.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/orbitofrontal/test_expected_outcome.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/orbitofrontal/test_ofc_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/orbitofrontal/test_ofc_integration.cpp` | 10 |
| Regression | `test/regression/core/brain/regions/orbitofrontal/test_ofc_regression.cpp` | 5 |
| E2E | `test/e2e/e2e_test_orbitofrontal_pipeline.cpp` | 5 |

---

### Phase BR-13: Retrosplenial Cortex (Spatial Reasoning)

**Priority**: P4
**Dependencies**: Hippocampus, Parietal
**Est. LOC**: 4,000 | **Tests**: 80

#### 13.1 Core Module

**Files**:
- `include/core/brain/regions/retrosplenial/nimcp_retrosplenial.h`
- `include/core/brain/regions/retrosplenial/nimcp_retrosplenial_adapter.h`
- `include/core/brain/regions/retrosplenial/nimcp_viewpoint_transformation.h`
- `include/core/brain/regions/retrosplenial/nimcp_landmark_navigation.h`
- `include/core/brain/regions/retrosplenial/nimcp_spatial_memory_consolidation.h`
- `src/core/brain/regions/retrosplenial/nimcp_retrosplenial.c`
- `src/core/brain/regions/retrosplenial/nimcp_retrosplenial_adapter.c`
- `src/core/brain/regions/retrosplenial/nimcp_viewpoint_transformation.c`
- `src/core/brain/regions/retrosplenial/nimcp_landmark_navigation.c`
- `src/core/brain/regions/retrosplenial/nimcp_spatial_memory_consolidation.c`

#### 13.2 Bridges

Standard 8-bridge pattern

#### 13.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/retrosplenial/test_retrosplenial_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/retrosplenial/test_viewpoint_transformation.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/retrosplenial/test_landmark_navigation.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/retrosplenial/test_retrosplenial_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/retrosplenial/test_retrosplenial_integration.cpp` | 10 |
| E2E | `test/e2e/e2e_test_retrosplenial_pipeline.cpp` | 5 |

---

### Phase BR-14: Claustrum (Consciousness Integration)

**Priority**: P5
**Dependencies**: All cortical regions
**Est. LOC**: 5,000 | **Tests**: 100

#### 14.1 Core Module

**Files**:
- `include/core/brain/regions/claustrum/nimcp_claustrum.h`
- `include/core/brain/regions/claustrum/nimcp_claustrum_adapter.h`
- `include/core/brain/regions/claustrum/nimcp_cross_modal_binding.h`
- `include/core/brain/regions/claustrum/nimcp_salience_detection.h`
- `include/core/brain/regions/claustrum/nimcp_attention_coordination.h`
- `include/core/brain/regions/claustrum/nimcp_unified_experience.h`
- `src/core/brain/regions/claustrum/nimcp_claustrum.c`
- `src/core/brain/regions/claustrum/nimcp_claustrum_adapter.c`
- `src/core/brain/regions/claustrum/nimcp_cross_modal_binding.c`
- `src/core/brain/regions/claustrum/nimcp_salience_detection.c`
- `src/core/brain/regions/claustrum/nimcp_attention_coordination.c`
- `src/core/brain/regions/claustrum/nimcp_unified_experience.c`

#### 14.2 Bridges

Standard 8-bridge pattern

#### 14.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/claustrum/test_claustrum_core.cpp` | 25 |
| Unit | `test/unit/core/brain/regions/claustrum/test_cross_modal_binding.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/claustrum/test_salience_detection.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/claustrum/test_attention_coordination.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/claustrum/test_unified_experience.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/claustrum/test_claustrum_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/claustrum/test_claustrum_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_claustrum_pipeline.cpp` | 5 |

---

### Phase BR-15: Periaqueductal Gray (Pain/Defense)

**Priority**: P5
**Dependencies**: Amygdala, Hypothalamus
**Est. LOC**: 4,000 | **Tests**: 80

#### 15.1 Core Module

**Files**:
- `include/core/brain/regions/pag/nimcp_pag.h`
- `include/core/brain/regions/pag/nimcp_pag_adapter.h`
- `include/core/brain/regions/pag/nimcp_analgesia.h`
- `include/core/brain/regions/pag/nimcp_defensive_behaviors.h`
- `include/core/brain/regions/pag/nimcp_vocalization_control.h`
- `include/core/brain/regions/pag/nimcp_autonomic_regulation.h`
- `src/core/brain/regions/pag/nimcp_pag.c`
- `src/core/brain/regions/pag/nimcp_pag_adapter.c`
- `src/core/brain/regions/pag/nimcp_analgesia.c`
- `src/core/brain/regions/pag/nimcp_defensive_behaviors.c`
- `src/core/brain/regions/pag/nimcp_vocalization_control.c`
- `src/core/brain/regions/pag/nimcp_autonomic_regulation.c`

#### 15.2 Bridges

Standard 8-bridge pattern

#### 15.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/pag/test_pag_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/pag/test_analgesia.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/pag/test_defensive_behaviors.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/pag/test_vocalization.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/pag/test_pag_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/pag/test_pag_integration.cpp` | 10 |
| E2E | `test/e2e/e2e_test_pag_pipeline.cpp` | 5 |

---

### Phase BR-16: Red Nucleus (Motor Refinement)

**Priority**: P5
**Dependencies**: Cerebellum, Motor cortex
**Est. LOC**: 3,000 | **Tests**: 60

#### 16.1 Core Module

**Files**:
- `include/core/brain/regions/red_nucleus/nimcp_red_nucleus.h`
- `include/core/brain/regions/red_nucleus/nimcp_red_nucleus_adapter.h`
- `include/core/brain/regions/red_nucleus/nimcp_limb_coordination.h`
- `include/core/brain/regions/red_nucleus/nimcp_motor_learning_support.h`
- `src/core/brain/regions/red_nucleus/nimcp_red_nucleus.c`
- `src/core/brain/regions/red_nucleus/nimcp_red_nucleus_adapter.c`
- `src/core/brain/regions/red_nucleus/nimcp_limb_coordination.c`
- `src/core/brain/regions/red_nucleus/nimcp_motor_learning_support.c`

#### 16.2 Bridges

Standard 8-bridge pattern

#### 16.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/red_nucleus/test_red_nucleus_core.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/red_nucleus/test_limb_coordination.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/red_nucleus/test_motor_learning.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/red_nucleus/test_red_nucleus_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/red_nucleus/test_red_nucleus_integration.cpp` | 5 |
| E2E | `test/e2e/e2e_test_red_nucleus_pipeline.cpp` | 5 |

---

### Phase BR-17: Reticular Formation (Arousal/Consciousness)

**Priority**: P5
**Dependencies**: Medulla, Thalamus
**Est. LOC**: 4,500 | **Tests**: 90

#### 17.1 Core Module

**Files**:
- `include/core/brain/regions/reticular/nimcp_reticular.h`
- `include/core/brain/regions/reticular/nimcp_reticular_adapter.h`
- `include/core/brain/regions/reticular/nimcp_sleep_wake_control.h`
- `include/core/brain/regions/reticular/nimcp_ascending_arousal.h`
- `include/core/brain/regions/reticular/nimcp_attention_gating.h`
- `include/core/brain/regions/reticular/nimcp_vital_reflexes.h`
- `src/core/brain/regions/reticular/nimcp_reticular.c`
- `src/core/brain/regions/reticular/nimcp_reticular_adapter.c`
- `src/core/brain/regions/reticular/nimcp_sleep_wake_control.c`
- `src/core/brain/regions/reticular/nimcp_ascending_arousal.c`
- `src/core/brain/regions/reticular/nimcp_attention_gating.c`
- `src/core/brain/regions/reticular/nimcp_vital_reflexes.c`

#### 17.2 Bridges

Standard 8-bridge pattern

#### 17.3 Tests

| Test Type | File | Test Count |
|-----------|------|------------|
| Unit | `test/unit/core/brain/regions/reticular/test_reticular_core.cpp` | 25 |
| Unit | `test/unit/core/brain/regions/reticular/test_sleep_wake_control.cpp` | 20 |
| Unit | `test/unit/core/brain/regions/reticular/test_ascending_arousal.cpp` | 15 |
| Unit | `test/unit/core/brain/regions/reticular/test_attention_gating.cpp` | 10 |
| Unit | `test/unit/core/brain/regions/reticular/test_reticular_bridges.cpp` | 10 |
| Integration | `test/integration/core/brain/regions/reticular/test_reticular_integration.cpp` | 10 |
| E2E | `test/e2e/e2e_test_reticular_pipeline.cpp` | 5 |

---

## PART III: REMAINING TIER COMPLETION

### Phase T5-7: Extrapolation Capabilities (Remaining 15%)

**Est. LOC**: 4,500 | **Tests**: 90

#### Modules to Complete

| Module | Status | Remaining Work |
|--------|--------|----------------|
| World Model Enhancement | 90% | Multi-modal integration, long-horizon prediction |
| Compositional Generalization | 80% | Systematic compositionality tests |
| Causal Reasoning Enhancement | 85% | Counterfactual imagination integration |

**Files**:
- `include/cognitive/extrapolation/nimcp_world_model_multimodal.h/c`
- `include/cognitive/extrapolation/nimcp_compositional_systematic.h/c`
- `include/cognitive/extrapolation/nimcp_counterfactual_imagination.h/c`

**Tests**: 90 tests across unit/integration/regression/e2e

---

### Phase T11: Embodiment (Remaining 40%)

**Est. LOC**: 2,800 | **Tests**: 56

#### Modules to Complete

| Module | Status | Remaining Work |
|--------|--------|----------------|
| Body Schema | 60% | Affordance processing, body ownership |
| Pragmatics | 50% | Context integration, discourse management |
| Embodied Simulation | 0% | Action simulation, motor imagery |
| Interoceptive Prediction | 70% | Visceral prediction errors |

**Files**:
- `include/embodiment/nimcp_affordance_processing.h/c`
- `include/embodiment/nimcp_body_ownership.h/c`
- `include/embodiment/nimcp_embodied_simulation.h/c`
- `include/language/pragmatics/nimcp_discourse_management.h/c`

**Tests**: 56 tests across unit/integration/regression/e2e

---

### Phase T12: Superhuman Enhancements (Remaining 50%)

**Est. LOC**: 15,000 | **Tests**: 300

#### Modules to Complete

| Module | Status | Remaining Work |
|--------|--------|----------------|
| Eagle Vision | 50% | Long-range pattern detection |
| Echolocation | 30% | 3D spatial mapping from echoes |
| Time Dilation | 40% | Subjective time manipulation |
| Savant Abilities | 0% | Pattern recognition savant mode |
| Hyperthymesia | 60% | Perfect autobiographical recall |
| Synesthesia | 0% | Cross-modal perception |
| Precognition | 70% | Advanced pattern prediction |

**Files**:
- `include/superhuman/nimcp_eagle_vision.h/c`
- `include/superhuman/nimcp_echolocation.h/c`
- `include/superhuman/nimcp_time_dilation.h/c`
- `include/superhuman/nimcp_savant_mode.h/c`
- `include/superhuman/nimcp_hyperthymesia.h/c`
- `include/superhuman/nimcp_synesthesia.h/c`
- `include/superhuman/nimcp_precognition.h/c`

**Tests**: 300 tests across unit/integration/regression/e2e

---

## PART IV: IMPLEMENTATION SCHEDULE

### Phase Order (Dependency-Based)

```
PHASE 1: Advanced Concepts Foundation
├── AC-3: Hodgkin-Huxley Dynamics (biophysical foundation)
├── AC-9: Thermodynamics (energy constraints)
└── AC-1: Ephaptic Coupling (synchronization)

PHASE 2: Chemistry Layer
├── AC-5: Nitric Oxide (retrograde signaling)
├── AC-4: pH Dynamics (metabolic effects)
└── AC-8: Neurovascular Coupling (blood flow)

PHASE 3: Biology Layer
├── AC-6: Epigenetics (long-term changes)
├── AC-7: Neurogenesis (new neurons)
└── AC-2: Information Geometry (learning optimization)

PHASE 4: Neuromodulatory Centers (Critical)
├── BR-1: Locus Coeruleus (NE)
├── BR-2: Ventral Tegmental Area (DA)
├── BR-3: Raphe Nuclei (5-HT)
└── BR-4: Habenula (aversive)

PHASE 5: Memory Circuit
├── BR-5: Entorhinal Cortex
├── BR-6: Perirhinal Cortex
├── BR-7: Parahippocampal Cortex
└── BR-8: Mammillary Bodies

PHASE 6: Sensory Processing
├── BR-9: Somatosensory Cortex
├── BR-10: Olfactory Cortex
└── BR-11: Gustatory Cortex

PHASE 7: Executive Enhancement
├── BR-12: Orbitofrontal Cortex
└── BR-13: Retrosplenial Cortex

PHASE 8: Integration Layer
├── BR-14: Claustrum
├── BR-15: Periaqueductal Gray
├── BR-16: Red Nucleus
└── BR-17: Reticular Formation

PHASE 9: Tier Completion
├── T5-7: Extrapolation (15% remaining)
├── T11: Embodiment (40% remaining)
└── T12: Superhuman (50% remaining)
```

---

## PART V: TEST SUMMARY

### Total Test Count by Type

| Test Type | Count |
|-----------|-------|
| Unit Tests | 2,100 |
| Integration Tests | 500 |
| Regression Tests | 250 |
| E2E Tests | 196 |
| **TOTAL** | **3,046** |

### Test Coverage Requirements

- **Line Coverage**: >= 85%
- **Branch Coverage**: >= 80%
- **Function Coverage**: >= 95%

### Test Execution

```bash
# Run all new tests
cd /home/bbrelin/nimcp/build
ctest -R "physics|chemistry|biology|locus_coeruleus|vta|raphe|habenula|entorhinal|perirhinal|parahippocampal|mammillary|somatosensory|olfactory|gustatory|orbitofrontal|retrosplenial|claustrum|pag|red_nucleus|reticular" --output-on-failure

# Run with parallel execution
ctest -j$(nproc) --output-on-failure
```

---

## PART VI: SUMMARY

### Total Implementation Scope

| Category | Modules | Est. LOC | Tests |
|----------|---------|----------|-------|
| Advanced Concepts | 9 | 42,000 | 840 |
| Brain Regions | 17 | 73,500 | 1,470 |
| Tier Completion | 3 tiers | 22,300 | 446 |
| Bridges (8 per module) | 208 bridges | 14,500 | 290 |
| **Intra-Layer Integration** | **9 layers** | **18,000** | **295** |
| **Inter-Layer Integration** | **13 connections** | **13,000** | **300** |
| **Layer Coordinator System** | **1 system** | **3,000** | **50** |
| **Cognitive Hub Integration** | **40 modules** | **8,000** | **160** |
| **Omnidirectional Integration** | **40 modules** | **12,000** | **240** |
| **Hypothalamus Integration** | **40 modules** | **8,000** | **105** |
| **Full-Stack Integration** | **4 test suites** | **-** | **50** |
| **GRAND TOTAL** | **~40 modules + full integration** | **~215,300** | **~4,246** |

### Key Requirements Checklist

#### Core Module Requirements
- [ ] All modules communicate via bio-async
- [ ] All modules have 8 standard bridges (SNN, Plasticity, FEP, Substrate, Thalamic, Immune, Hub, Quantum)
- [ ] All modules integrate with immune system
- [ ] All modules initialized via brain factory
- [ ] All modules have comprehensive logging

#### Testing Requirements
- [ ] All modules have unit tests (>=20 per module)
- [ ] All modules have integration tests (>=5 per module)
- [ ] All modules have regression tests (>=5 per module)
- [ ] All modules have e2e tests (>=5 per module)

#### Layer Integration Requirements
- [ ] **All modules implement intra-layer integration with sibling modules**
- [ ] **All modules implement inter-layer integration with connected layers**
- [ ] **All layers registered with central Layer Coordinator**
- [ ] **Bottom-up and top-down pathways verified for all inter-layer connections**
- [ ] **Intra-layer coherence tests pass (>=0.85 coherence)**
- [ ] **Inter-layer synchronization tests pass**
- [ ] **Full-stack integration tests pass (Physics → Integration and back)**

#### Cognitive Hub Integration Requirements
- [ ] **All modules registered with cognitive_integration_hub**
- [ ] **All modules subscribe to relevant cognitive events**
- [ ] **All modules publish appropriate cognitive events**
- [ ] **All modules implement query handler for inter-module queries**
- [ ] **Cognitive category correctly assigned to each module**

#### Omnidirectional Integration Requirements
- [ ] **All modules have omni bridge (nimcp_<module>_omni_bridge.h/c)**
- [ ] **All modules connect to JEPA bidirectional predictor**
- [ ] **All modules integrate with predictive hierarchy**
- [ ] **All modules handle BIO_MSG_OMNI_* message types**
- [ ] **All modules handle BIO_MSG_PRED_HIER_* message types**
- [ ] **Precision weighting updated from free energy estimates**
- [ ] **Forward prediction and backward inference implemented**

#### Hypothalamus Integration Requirements
- [ ] **All modules have hypothalamus bridge (nimcp_hypothalamus_<module>_bridge.h/c)**
- [ ] **All modules receive circadian phase signals**
- [ ] **All modules receive stress/cortisol signals**
- [ ] **All modules receive autonomic state signals**
- [ ] **All modules report metabolic demands**
- [ ] **All modules report homeostatic perturbations**
- [ ] **Circadian modulation affects module behavior**
- [ ] **Stress response modulates module activity**

### Integration Architecture Verification

```
VERIFICATION CHECKLIST FOR EACH MODULE:

□ Intra-Layer Integration
  □ module_intra_layer_t struct implemented
  □ module_intra_layer_init() implemented
  □ module_intra_layer_sync() implemented
  □ module_intra_layer_broadcast() implemented
  □ module_intra_layer_receive() implemented
  □ All sibling bridges created
  □ Layer coordinator registration complete

□ Inter-Layer Integration
  □ module_inter_layer_t struct implemented
  □ module_inter_layer_init() implemented
  □ module_inter_layer_send_bottom_up() implemented
  □ module_inter_layer_send_top_down() implemented
  □ module_inter_layer_receive() implemented
  □ module_inter_layer_sync_phase() implemented
  □ All inter-layer bridges created
  □ Routing table entries verified

□ Test Coverage
  □ Intra-layer unit tests
  □ Intra-layer integration tests
  □ Inter-layer unit tests
  □ Inter-layer integration tests
  □ Full-stack traversal tests
```

---

**Document Version**: 1.1
**Created**: 2026-01-09
**Updated**: 2026-01-09
**Author**: Claude Code
**Status**: Ready for Implementation
