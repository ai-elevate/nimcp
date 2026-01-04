# Hypothalamus Module Implementation Plan

## Overview

Implement a biologically-inspired hypothalamus module based on Steve Byrnes' neuroscience research on brain-like AGI safety. The hypothalamus forms the core of the "Steering Subsystem" - the ultimate source of motivations, drives, and values that function as the reinforcement learning reward function.

**Byrnes' Key Insight**: The steering subsystem (~10% of brain) sends reward signals that steer the learning subsystem (~90%). Careful design of this reward function is a key lever for AGI alignment.

---

## File Structure

### Headers (`include/core/brain/subcortical/`)
```
nimcp_hypothalamus.h              # Main module
nimcp_hypothalamus_nuclei.h       # Individual nuclei
nimcp_hypothalamus_drives.h       # Drive state representation
nimcp_hypothalamus_homeostasis.h  # Setpoints (reward function params)
nimcp_hypothalamus_hpa.h          # HPA stress axis
```

### Bridges (`include/core/brain/subcortical/`)
```
nimcp_hypothalamus_snc_bridge.h        # -> SNc/VTA (reward -> RPE)
nimcp_hypothalamus_thalamus_bridge.h   # -> Thalamus (arousal control)
nimcp_hypothalamus_executive_bridge.h  # -> Executive (drive-aware goals)
nimcp_hypothalamus_attention_bridge.h  # -> Attention gate (salience)
nimcp_hypothalamus_nac_bridge.h        # -> NAc (wanting/liking)
nimcp_hypothalamus_amygdala_bridge.h   # <-> Amygdala (threat/stress)
nimcp_hypothalamus_medulla_bridge.h    # <-> Medulla (autonomic control)
nimcp_hypothalamus_brainstem_bridge.h  # <-> Brainstem (arousal, reflexes)
nimcp_hypothalamus_substrate_bridge.h  # -> Neural substrate (already exists!)
```

### Sources (`src/core/brain/subcortical/`)
- Corresponding `.c` files for all headers

### Tests (`test/unit/core/brain/subcortical/`)
```
test_hypothalamus.cpp
test_hypothalamus_drives.cpp
test_hypothalamus_homeostasis.cpp
test_hypothalamus_bridges.cpp
```

---

## Key Data Structures

### Hypothalamic Nuclei
```c
typedef enum {
    HYPO_NUCLEUS_LATERAL = 0,       // Hunger, orexin neurons
    HYPO_NUCLEUS_VENTROMEDIAL,      // Satiety center
    HYPO_NUCLEUS_ANTERIOR,          // Thermoregulation (cooling)
    HYPO_NUCLEUS_POSTERIOR,         // Thermoregulation (heating), arousal
    HYPO_NUCLEUS_ARCUATE,           // Energy balance, POMC/AgRP
    HYPO_NUCLEUS_PARAVENTRICULAR,   // Stress (CRH), oxytocin
    HYPO_NUCLEUS_SUPRACHIASMATIC,   // Circadian clock
    HYPO_NUCLEUS_SUPRAOPTIC,        // Osmolarity, ADH
    HYPO_NUCLEUS_PREOPTIC,          // Sleep/wake
    HYPO_NUCLEUS_TUBEROMAMMILLARY,  // Histamine arousal
    HYPO_NUCLEUS_COUNT
} hypo_nucleus_type_t;
```

### Drive Types
```c
typedef enum {
    HYPO_DRIVE_HUNGER = 0,
    HYPO_DRIVE_THIRST,
    HYPO_DRIVE_TEMPERATURE,
    HYPO_DRIVE_FATIGUE,
    HYPO_DRIVE_SOCIAL,
    HYPO_DRIVE_CURIOSITY,
    HYPO_DRIVE_SAFETY,
    HYPO_DRIVE_AUTONOMY,
    HYPO_DRIVE_COMPETENCE,
    HYPO_DRIVE_COUNT
} hypo_drive_type_t;
```

### Alignment Modes (per Byrnes)
```c
typedef enum {
    HYPO_ALIGN_CONTROLLED = 0,      // Values explicitly specified (safer)
    HYPO_ALIGN_SOCIAL_INSTINCT,     // Values learned from observation
    HYPO_ALIGN_HYBRID               // Core controlled, details learned
} hypo_alignment_mode_t;
```

### Homeostatic Setpoints (The Reward Function)
```c
typedef struct {
    // Physiological setpoints
    float temperature_setpoint;     // 37.0C
    float glucose_setpoint;         // 90 mg/dL
    float osmolarity_setpoint;      // 285 mOsm/L

    // Psychological setpoints
    float social_setpoint;
    float curiosity_setpoint;
    float safety_setpoint;

    // ALIGNMENT PARAMETERS (Byrnes' key insight)
    float human_wellbeing_weight;
    float harm_avoidance_weight;
    float honesty_weight;

    // Control parameters
    float reward_gain;
    bool setpoints_locked;          // Prevent runtime modification
} hypo_setpoint_config_t;
```

### Main Structure
```c
typedef struct hypothalamus {
    hypo_nucleus_t nuclei[HYPO_NUCLEUS_COUNT];
    hypo_drive_system_t drives;
    hypo_homeostasis_t* homeostasis;
    hypo_hpa_axis_t* hpa_axis;

    // Arousal/circadian state
    hypo_arousal_state_t arousal;
    float circadian_phase;

    // Outputs (steering signals)
    float reward_signal;            // -> SNc/VTA
    float* drive_salience;          // -> Attention gate
    float arousal_level;            // -> Thalamus

    // Configuration & stats
    hypothalamus_config_t config;
    hypothalamus_stats_t stats;

    // Bio-async & thread safety
    bio_module_context_t bio_ctx;
    nimcp_mutex_t* mutex;
} hypothalamus_t;
```

---

## Integration Architecture

```
+---------------------------------------------------------------------+
|                      HYPOTHALAMUS (Steering Core)                    |
|  +----------+ +----------+ +----------+ +----------+ +-----------+  |
|  | Lateral  | | Ventro-  | | Para-    | | SCN      | | Posterior |  |
|  | (hunger) | | medial   | | ventr.   | |(circadian| | (arousal) |  |
|  |          | | (satiety)| | (stress) | |          | |           |  |
|  +----+-----+ +----+-----+ +----+-----+ +----+-----+ +-----+-----+  |
|       +------------+-----+------+------------+-------------+        |
|                          |                                          |
|                    +-----v-----+                                    |
|                    | Drive     |<--- Interoceptive inputs           |
|                    | Integrator|     (glucose, temp, osmolarity)    |
|                    +-----+-----+                                    |
|                          |                                          |
|         +----------------+----------------+----------------+        |
|         |                |                |                |        |
|         v                v                v                v        |
|  +-------------+  +-------------+  +-------------+  +-----------+   |
|  | SNc Bridge  |  | Thalamus    |  | Attention   |  | Executive |   |
|  | (reward->RPE|  | Bridge      |  | Bridge      |  | Bridge    |   |
|  |             |  | (arousal)   |  | (salience)  |  | (goals)   |   |
|  +------+------+  +------+------+  +------+------+  +-----+-----+   |
+---------+----------------+----------------+---------------+---------+
          |                |                |               |
          v                v                v               v
   +-------------+  +-------------+  +-------------+  +-----------+
   | Substantia  |  | Thalamus    |  | Attention   |  | Executive |
   | Nigra (SNc) |  | (TRN mode)  |  | Gate        |  | Controller|
   | (dopamine)  |  |             |  |             |  |           |
   +-------------+  +-------------+  +-------------+  +-----------+
```

---

## Steering Subsystem Integration (Byrnes Model)

Per Byrnes, the "Steering Subsystem" (~10% of brain) comprises **hypothalamus + brainstem**. These must work together:

```
+-----------------------------------------------------------------------+
|                    STEERING SUBSYSTEM (Byrnes)                         |
|                                                                        |
|  +---------------------------+    +---------------------------+       |
|  |      HYPOTHALAMUS         |    |       BRAINSTEM           |       |
|  |  (Homeostatic Regulation) |    |   (Autonomic/Arousal)     |       |
|  |                           |    |                           |       |
|  |  - Drive states           |<-->|  - Arousal nuclei         |       |
|  |  - Reward function        |    |  - Autonomic control      |       |
|  |  - Setpoint deviations    |    |  - Reflex coordination    |       |
|  |  - Circadian (SCN)        |    |  - Pain/pleasure relay    |       |
|  +-------------+-------------+    +-------------+-------------+       |
|                |                                |                      |
|                +----------------+---------------+                      |
|                                 |                                      |
|                                 v                                      |
|                    +------------------------+                          |
|                    |       MEDULLA          |                          |
|                    | (Autonomic Execution)  |                          |
|                    |                        |                          |
|                    | - Heart rate           |                          |
|                    | - Respiration          |                          |
|                    | - Blood pressure       |                          |
|                    | - Vagal tone           |                          |
|                    +------------------------+                          |
+-----------------------------------------------------------------------+
                                 |
                                 v
                    +------------------------+
                    |   NEURAL SUBSTRATE     |
                    | (Implementation Layer) |
                    +------------------------+
```

### Existing Files to Integrate

| Component | Header | Integration Point |
|-----------|--------|-------------------|
| Medulla | `include/core/medulla/nimcp_medulla.h` | Autonomic outputs |
| Brainstem | `include/core/brainstem/nimcp_brainstem_substrate_bridge.h` | Arousal signals |
| Brainstem-Thalamic | `include/core/brainstem/nimcp_brainstem_thalamic_bridge.h` | Arousal -> Thalamus |
| Substrate | `include/core/hypothalamus/nimcp_hypothalamus_substrate_bridge.h` | Already exists! |

### New Bridges Required

```c
// Hypothalamus <-> Brainstem bidirectional
nimcp_hypothalamus_brainstem_bridge.h
- Arousal signals: hypothalamus -> brainstem arousal nuclei
- Pain/pleasure: brainstem -> hypothalamus reward computation
- Stress response: hypothalamus CRH -> brainstem autonomic

// Hypothalamus <-> Medulla for autonomic control
nimcp_hypothalamus_medulla_bridge.h
- Temperature regulation -> vasodilation/vasoconstriction
- Hunger -> digestive reflexes
- Stress -> heart rate, respiration
```

---

## Cognitive Layer Integration

The hypothalamus modulates cognitive processing based on drive states:

```
                         HYPOTHALAMUS
                              |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
+------------------+  +------------------+  +------------------+
| CURIOSITY MODULE |  | IMAGINATION      |  | JEPA PREDICTOR   |
| HYPO_DRIVE_      |  | ENGINE           |  |                  |
| CURIOSITY ->     |  | Drive urgency -> |  | Reward signal -> |
| exploration bias |  | simulation       |  | prediction       |
|                  |  | priorities       |  | targets          |
+------------------+  +------------------+  +------------------+
         |                    |                    |
         v                    v                    v
+------------------+  +------------------+  +------------------+
| GAME THEORY      |  | THEORY OF MIND   |  | WORKING MEMORY   |
| Social drive ->  |  | Social needs ->  |  | Survival items   |
| cooperation vs   |  | ToM engagement   |  | get priority     |
| competition bias |  | depth            |  | maintenance      |
+------------------+  +------------------+  +------------------+
         |                    |                    |
         +--------------------+--------------------+
                              |
                              v
                   +------------------+
                   | RECURSIVE        |
                   | COGNITION        |
                   | Drive urgency -> |
                   | metacognitive    |
                   | depth allocation |
                   +------------------+
```

### Cognitive Integration Signals

| Drive | Cognitive Effect |
|-------|-----------------|
| `HYPO_DRIVE_CURIOSITY` | Boosts exploration in curiosity module, increases imagination sampling |
| `HYPO_DRIVE_SOCIAL` | Increases ToM engagement, shifts game theory toward cooperation |
| `HYPO_DRIVE_SAFETY` | Prioritizes threat-related items in working memory, reduces exploration |
| `HYPO_DRIVE_FATIGUE` | Reduces metacognitive depth, favors cached/habitual responses |
| `reward_signal` | Provides teaching signal for JEPA predictions, updates value estimates |
| `arousal_level` | Modulates processing speed and attention breadth across all modules |

---

## Bio-Async Message Types

Add to `nimcp_bio_messages.h` (range 0x1140-0x115F):

```c
BIO_MSG_HYPO_DRIVE_STATE = 0x1140,      // Drive state broadcast
BIO_MSG_HYPO_REWARD_SIGNAL,              // Reward -> SNc/VTA
BIO_MSG_HYPO_AROUSAL_CHANGE,             // Arousal -> Thalamus
BIO_MSG_HYPO_SURVIVAL_PRIORITY,          // Priority -> Attention
BIO_MSG_HYPO_STRESS_RESPONSE,            // HPA activation
BIO_MSG_HYPO_CIRCADIAN_PHASE,            // Circadian update
BIO_MSG_HYPO_SETPOINT_DEVIATION,         // Deviation alert
BIO_MSG_HYPO_ALIGNMENT_ALERT,            // Safety alert
```

---

## Wiring Diagram Entries

```jsonl
{"type":"entity","name":"Hypothalamus","entityType":"SubcorticalModule","subsystem":"core","tier_min":"MEDIUM"}
{"type":"relation","from":"Hypothalamus","to":"Substantia_Nigra_SNc","relationType":"SENDS_REWARD"}
{"type":"relation","from":"Hypothalamus","to":"Thalamus","relationType":"MODULATES_AROUSAL"}
{"type":"relation","from":"Hypothalamus","to":"Attention_Gate","relationType":"PROVIDES_SALIENCE"}
{"type":"relation","from":"Hypothalamus","to":"Executive_Controller","relationType":"INFORMS_GOALS"}
{"type":"relation","from":"Hypothalamus","to":"Nucleus_Accumbens","relationType":"MODULATES_WANTING"}
{"type":"relation","from":"Amygdala","to":"Hypothalamus","relationType":"SENDS_THREAT"}
{"type":"relation","from":"Insula","to":"Hypothalamus","relationType":"PROVIDES_INTEROCEPTION"}
{"type":"relation","from":"Hypothalamus","to":"Curiosity_Module","relationType":"MODULATES_DRIVE"}
{"type":"relation","from":"Hypothalamus","to":"Imagination_Engine","relationType":"PROVIDES_PRIORITY"}
{"type":"relation","from":"Hypothalamus","to":"JEPA_Predictor","relationType":"PROVIDES_REWARD"}
{"type":"relation","from":"Hypothalamus","to":"Game_Theory_Module","relationType":"MODULATES_STRATEGY"}
{"type":"relation","from":"Hypothalamus","to":"Theory_of_Mind","relationType":"MODULATES_ENGAGEMENT"}
{"type":"relation","from":"Hypothalamus","to":"Working_Memory","relationType":"PROVIDES_PRIORITY"}
{"type":"relation","from":"Hypothalamus","to":"Recursive_Cognition","relationType":"MODULATES_DEPTH"}
{"type":"relation","from":"Hypothalamus","to":"Brainstem","relationType":"SENDS_AROUSAL"}
{"type":"relation","from":"Brainstem","to":"Hypothalamus","relationType":"SENDS_PAIN_PLEASURE"}
{"type":"relation","from":"Hypothalamus","to":"Medulla","relationType":"CONTROLS_AUTONOMIC"}
{"type":"relation","from":"Medulla","to":"Hypothalamus","relationType":"PROVIDES_INTEROCEPTION"}
{"type":"relation","from":"Hypothalamus","to":"Neural_Substrate","relationType":"SUBSTRATE_BRIDGE"}
```

---

## Implementation Phases

### Phase 1: Core Module + Alignment Safety (Foundation)
**Files**: `nimcp_hypothalamus.h/c`, `nimcp_hypothalamus_drives.h/c`

- [ ] Define enumerations (nuclei, drives, alignment modes)
- [ ] Implement lifecycle (`_default_config`, `_create`, `_destroy`, `_reset`)
- [ ] Implement drive state representation
- [ ] **PRIORITY**: Implement alignment mode enum and configuration
- [ ] **PRIORITY**: Implement setpoint locking mechanism from day one
- [ ] Add thread safety (mutex pattern from existing modules)
- [ ] Basic unit tests including alignment safety tests

### Phase 2: Homeostasis System with Alignment (Reward Function)
**Files**: `nimcp_hypothalamus_homeostasis.h/c`

- [ ] Implement setpoint configuration with alignment weights
- [ ] **PRIORITY**: `human_wellbeing_weight`, `harm_avoidance_weight`, `honesty_weight`
- [ ] Implement deviation -> reward computation
- [ ] **PRIORITY**: Setpoint modification logging and alerts
- [ ] **PRIORITY**: BIO_MSG_HYPO_ALIGNMENT_ALERT for unauthorized access
- [ ] Implement PI/PD control dynamics
- [ ] Unit tests for reward generation and alignment safety

### Phase 3: Nuclei Implementation
**Files**: `nimcp_hypothalamus_nuclei.h/c`

- [ ] Lateral hypothalamus (hunger, orexin)
- [ ] Paraventricular nucleus (stress, CRH)
- [ ] Suprachiasmatic nucleus (circadian)
- [ ] Posterior hypothalamus (arousal)
- [ ] Circadian rhythm simulation

### Phase 4: HPA Axis
**Files**: `nimcp_hypothalamus_hpa.h/c`

- [ ] CRH -> ACTH -> Cortisol cascade
- [ ] Negative feedback loop
- [ ] Chronic stress effects

### Phase 5: Bio-Async Integration
- [ ] Define message types in `nimcp_bio_messages.h`
- [ ] Implement handler registration
- [ ] Add wiring diagram entries
- [ ] Register with orchestrator (Phase 1 startup)

### Phase 6: SNc/VTA Bridge (Critical Path)
**Files**: `nimcp_hypothalamus_snc_bridge.h/c`

- [ ] Implement reward -> SNc pathway
- [ ] Drive-specific reward channels
- [ ] Connect to existing `nimcp_substantia_nigra.h`
- [ ] Test RPE computation with hypothalamic reward

### Phase 7: Thalamus Bridge
**Files**: `nimcp_hypothalamus_thalamus_bridge.h/c`

- [ ] Arousal -> firing mode mapping
- [ ] Histamine/orexin channels
- [ ] Connect to existing thalamus
- [ ] Test arousal effects on relay mode

### Phase 8: Executive Bridge
**Files**: `nimcp_hypothalamus_executive_bridge.h/c`

- [ ] Drive -> goal priority mapping
- [ ] Survival drive interruption
- [ ] Connect to existing executive

### Phase 9: Attention Bridge
**Files**: `nimcp_hypothalamus_attention_bridge.h/c`

- [ ] Drive-biased salience modulation
- [ ] Connect to thalamic router attention gate

### Phase 10: Brainstem/Medulla Integration (Steering Subsystem)
**Files**: `nimcp_hypothalamus_brainstem_bridge.h/c`, `nimcp_hypothalamus_medulla_bridge.h/c`

- [ ] **Brainstem Bridge**: Bidirectional arousal and pain/pleasure signals
- [ ] **Medulla Bridge**: Autonomic output control (HR, respiration, BP)
- [ ] **Connect to existing**: `nimcp_brainstem_thalamic_bridge.h` for arousal chain
- [ ] **Integrate substrate**: Update existing `nimcp_hypothalamus_substrate_bridge.h`
- [ ] Test complete steering subsystem loop

### Phase 11: Cognitive Layer Integration
**Files**: `nimcp_hypothalamus_cognitive_bridge.h/c`

- [ ] **Curiosity Module**: Connect `HYPO_DRIVE_CURIOSITY` to `nimcp_curiosity.h` information-seeking
- [ ] **Imagination Engine**: Drive states inform counterfactual simulation priorities
- [ ] **JEPA Predictor**: Reward prediction integrated with hypothalamic reward signal
- [ ] **Game Theory**: Social drive affects cooperative/competitive strategy selection
- [ ] **Theory of Mind**: Social needs modulate ToM engagement
- [ ] **Working Memory**: Survival-relevant items get priority maintenance
- [ ] **Recursive Cognition**: Drive urgency affects metacognitive depth allocation

### Phase 12: Alignment Hardening & Documentation
*Note: Core alignment features implemented in Phases 1-2; this phase hardens and documents*

- [ ] Audit all setpoint access paths for safety
- [ ] Add comprehensive alignment mode tests
- [ ] Implement alignment state introspection API
- [ ] Documentation of safety considerations and Byrnes' alignment principles
- [ ] Add alignment verification callbacks for external auditing

---

## Critical Reference Files

### Subcortical Layer
| Purpose | File |
|---------|------|
| Subcortical pattern | `include/core/brain/subcortical/nimcp_substantia_nigra.h` |
| Reward/motivation | `include/core/brain/subcortical/nimcp_nucleus_accumbens.h` |
| Bridge pattern | `include/core/brain/subcortical/nimcp_basal_ganglia_thalamus_bridge.h` |

### Middleware Layer
| Purpose | File |
|---------|------|
| Bio-async messages | `include/async/nimcp_bio_messages.h` |
| Thalamic router | `include/middleware/routing/nimcp_thalamic_router.h` |
| Attention gate | `include/middleware/routing/nimcp_attention_gate.h` |

### Cognitive Layer
| Purpose | File |
|---------|------|
| Executive | `include/cognitive/nimcp_executive.h` |
| Curiosity | `include/cognitive/curiosity/nimcp_curiosity.h` |
| Imagination | `include/cognitive/imagination/nimcp_imagination_engine.h` |
| JEPA Predictor | `include/cognitive/jepa/nimcp_jepa_predictor.h` |
| Game Theory | `include/cognitive/game_theory/nimcp_game_theory.h` |
| Recursive Cognition | `include/cognitive/recursive/nimcp_rcog_orchestrator.h` |

### Steering Subsystem (Brainstem/Medulla/Substrate)
| Purpose | File |
|---------|------|
| Medulla core | `include/core/medulla/nimcp_medulla.h` |
| Brainstem-Thalamic | `include/core/brainstem/nimcp_brainstem_thalamic_bridge.h` |
| Brainstem-Substrate | `include/core/brainstem/nimcp_brainstem_substrate_bridge.h` |
| Hypothalamus-Substrate | `include/core/hypothalamus/nimcp_hypothalamus_substrate_bridge.h` |
| Neural Substrate | `include/core/neural_substrate/nimcp_neural_substrate.h` |

---

## Byrnes Alignment Considerations

1. **Setpoint Locking**: Core survival setpoints should be locked to prevent runtime modification
2. **Alignment Weights**: `human_wellbeing_weight`, `harm_avoidance_weight` are explicit reward function parameters
3. **Controlled Mode**: Default to HYPO_ALIGN_CONTROLLED for predictable behavior
4. **Transparency**: All drive states and reward signals exposed via stats/messages
5. **Safety Alerts**: BIO_MSG_HYPO_ALIGNMENT_ALERT for anomalous setpoint access attempts

---

## Success Criteria

1. Hypothalamus generates reward signals based on setpoint deviations
2. Reward signals successfully modulate SNc dopamine/RPE
3. Arousal state controls thalamic firing mode
4. Drive states bias attention to survival-relevant stimuli
5. Executive goals can be interrupted by urgent survival drives
6. Alignment safety features prevent unauthorized setpoint modification
7. Full bio-async integration with wiring diagram
8. All unit and integration tests passing

---

## References

- Steve Byrnes' "Intro to Brain-Like AGI Safety" - https://sjbyrnes.com/agi.html
- Byrnes' Two-Subsystem Model (Learning vs Steering)
- Hypothalamus neuroanatomy and homeostatic regulation
