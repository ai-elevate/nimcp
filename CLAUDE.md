# NIMCP Project Memory

## Project Overview
NIMCP (Neural Information Management and Cognitive Processing) is a C-based neural simulation library with biologically-inspired components including brain regions, plasticity mechanisms, cognitive systems, and swarm intelligence.

## Build Commands
```bash
cd /home/bbrelin/nimcp/build
cmake ..
make nimcp -j4                    # Build main library
make <test_target> -j4            # Build specific test
```

## Test Execution
```bash
# LNN tests (84 total)
./test/unit/lnn/unit_lnn_test_lnn_config --gtest_brief=1   # 24 tests
./test/unit/lnn/unit_lnn_test_lnn_neuron --gtest_brief=1   # 34 tests
./test/unit/lnn/unit_lnn_test_lnn_layer --gtest_brief=1    # 26 tests

# Introspection tests (173 total)
./test/unit/cognitive/introspection/unit_cognitive_introspection_consciousness_metrics --gtest_brief=1  # 31 tests
./test/unit/cognitive/introspection/unit_cognitive_introspection_temporal_patterns --gtest_brief=1      # 43 tests
./test/unit/cognitive/introspection/unit_cognitive_introspection_ensemble_uncertainty --gtest_brief=1   # 27 tests
./test/e2e/e2e_test_introspection_pipeline --gtest_brief=1                                              # 8 tests

# Tensor library tests (87 total)
./test/unit/utils/tensor/unit_utils_tensor_test_tensor --gtest_brief=1        # 51 tests
./test/unit/utils/encoding/unit_utils_encoding_test_positional_encoding --gtest_brief=1  # 18 tests

# Module-specific tests
./test/unit/middleware/encoding/unit_middleware_encoding_population_coding --gtest_brief=1   # 64 tests
./test/unit/middleware/features/unit_middleware_features_feature_extractor --gtest_brief=1   # 63 tests
./test/unit/middleware/routing/unit_middleware_routing_thalamic_router --gtest_brief=1       # 48 tests
```

## Coding Standards
- **Documentation**: WHAT/WHY/HOW comments on all functions
- **Guard clauses**: No nested ifs, early returns for validation
- **Single Responsibility**: Functions do one thing, < 50 lines
- **Biological basis**: Document neural/cognitive grounding
- **Memory safety**: Use nimcp_malloc/nimcp_free, proper cleanup

## Critical API Usage Rule
**NEVER make assumptions about API methods.** Always verify function signatures by reading the actual header files before writing test code or implementations. This includes:
- Function parameter count and types
- Parameter order
- Return types and output parameters
- Enum/constant names

## MANDATORY Test Writing Protocol

**BEFORE writing ANY test code, you MUST:**
1. Read the COMPLETE header file for the module being tested
2. List the exact function signatures you will call in a code block
3. Show the struct definitions for any types you'll use
4. Only THEN write test code

**NEVER assume or infer:**
- Function names - READ the header
- Parameter counts/types - READ the header
- Return types - READ the header
- Struct member names - READ the header
- Default config values - READ the implementation

**If you write test code that fails to compile due to API mismatch:**
- STOP and acknowledge you violated this protocol
- Read the actual header before proceeding

**Alternative approach:** Write implementation first, then derive tests from what was actually written - rather than writing tests for APIs you haven't verified.

## Key API Patterns

### Tensor Library (`include/utils/tensor/nimcp_tensor.h`)
```c
nimcp_tensor_t* t = nimcp_tensor_create(dims, ndims);
nimcp_tensor_t* sum_t = nimcp_tensor_sum(t);           // Returns tensor*, NOT scalar
double val = nimcp_tensor_get_flat(sum_t, 0);          // Extract scalar from tensor
double norm = nimcp_tensor_norm_p(t, 2.0);             // Returns double directly
nimcp_tensor_destroy(t);
```

**GOTCHA**: `nimcp_tensor_sum()` returns `nimcp_tensor_t*`, must extract with `nimcp_tensor_get_flat()`

### Positional Encoding (`include/utils/encoding/nimcp_positional_encoding.h`)
- Types: `NIMCP_POS_SINUSOIDAL`, `NIMCP_POS_LEARNED`, `NIMCP_POS_ROTARY`, `NIMCP_POS_ALIBI`, `NIMCP_POS_RELATIVE`
- API: `nimcp_pos_encoder_create()`, `nimcp_pos_encode_position()`, `nimcp_pos_rope_apply()`, `nimcp_pos_alibi_get_bias()`

## Recent Completions

### Pink Noise Domain Bridges (Complete - Dec 2024)
Integrated 1/f pink noise across 12 neural subsystems for biologically realistic stochastic dynamics:

| Module | Bridge | Biological Basis |
|--------|--------|------------------|
| Calcium | `nimcp_calcium_pink_noise_bridge.h` | NMDA receptor, Ca2+ wave fluctuations |
| Dendritic | `nimcp_dendritic_pink_noise_bridge.h` | bAP timing, NMDA spike variability |
| Heterosynaptic | `nimcp_heterosynaptic_pink_noise_bridge.h` | Competition dynamics noise |
| Metabolic | `nimcp_metabolic_pink_noise_bridge.h` | ATP/energy fluctuations |
| Spatial Neuromod | `nimcp_spatial_neuromod_pink_noise_bridge.h` | Diffusion variability |
| Vesicle Packaging | `nimcp_vesicle_packaging_pink_noise_bridge.h` | RRP, Pr, quantal noise |
| Ensemble Uncertainty | `nimcp_ensemble_uncertainty_pink_noise_bridge.h` | Epistemic noise injection |
| Systems Consolidation | `nimcp_systems_consolidation_pink_noise_bridge.h` | Sleep replay modulation |
| Brain Oscillations | `nimcp_oscillations_pink_noise_bridge.h` | Cross-frequency 1/f coupling |
| Population Coding | `nimcp_population_coding_pink_noise_bridge.h` | Tuning curve modulation |
| STDP | `nimcp_stdp_pink_noise_bridge.h` | Learning rate noise |
| STP | `nimcp_stp_pink_noise_bridge.h` | Facilitation/depression variability |

**Common Bridge Pattern:**
```c
// Create bridge
<module>_pink_noise_config_t config = <module>_pink_noise_default_config();
<module>_pink_noise_bridge_t* bridge = <module>_pink_noise_create(&config);

// Connect to module
<module>_pink_noise_connect_<module>(bridge, <module>_system);

// Update and apply modulation
<module>_pink_noise_update(bridge);
<module>_pink_noise_apply_modulation(bridge);

// Cleanup
<module>_pink_noise_destroy(bridge);
```

**Mutex API Pattern (IMPORTANT):**
```c
// Correct: Allocate + init
bridge->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
nimcp_mutex_init(bridge->mutex, NULL);

// Correct: Destroy + free
nimcp_mutex_destroy(bridge->mutex);
nimcp_free(bridge->mutex);

// WRONG: nimcp_mutex_create() does not exist
```

**Test Coverage: 139 tests**
- Unit: 119 tests (`./test/unit/plasticity/unit_plasticity_pink_noise_enhancements`)
- E2E: 6 tests (`./test/e2e/e2e_test_pink_noise_pipeline`)
- Integration: 14 tests (`./test/integration/plasticity/integration_plasticity_pink_noise`)

### Brain Immune System (Complete - Dec 2024)
Implemented biologically-inspired immune coordination layer integrating BBB, BFT, and swarm immune:

| Component | Biological Analog | NIMCP Integration |
|-----------|------------------|-------------------|
| B Cells | Antibody producers | Swarm memory cells, pattern storage |
| T Helper (CD4+) | Coordination | Bio-async signaling, B cell activation |
| T Killer (CD8+) | Cytotoxic | BFT quarantine, node isolation |
| Antibodies | Countermeasures | Swarm responses (ALERT, ISOLATION, COUNTER_ATTACK) |
| Cytokines | Signaling | Bio-async NOREPINEPHRINE channel |
| Inflammation | Escalation | Local → Regional → Systemic → Storm |

**Key Features:**
- **Auto-learning**: Neutralization converts B cells to memory automatically
- **Fuzzy affinity**: 3-component matching (exact 50%, bit 30%, length 20%)
- **Cross-reactive immunity**: Recognizes variants at 70% threshold
- **Auto-recognition**: Memory check triggers secondary response on antigen presentation

**Cytokine Enum Naming:**
```c
// Base cytokine types (from nimcp_swarm_immune.h)
CYTOKINE_IL1B, CYTOKINE_IL6, CYTOKINE_IL10, CYTOKINE_TNFA

// Brain-specific wrapper (from nimcp_brain_immune.h)
BRAIN_CYTOKINE_IL1 = CYTOKINE_IL1B    // Use BRAIN_CYTOKINE_IL1 in code
BRAIN_CYTOKINE_IL6 = CYTOKINE_IL6
BRAIN_CYTOKINE_IL10 = CYTOKINE_IL10
BRAIN_CYTOKINE_TNF = CYTOKINE_TNFA
BRAIN_CYTOKINE_IFN_GAMMA = 5          // Brain-specific (quarantine)
BRAIN_CYTOKINE_COUNT = 6

// Module-specific constants use CYTOKINE_IL1_ prefix (not IL1B):
CYTOKINE_IL1_ATTENTION_IMPACT, CYTOKINE_IL1_LTP_IMPAIRMENT, etc.
CYTOKINE_IFN_GAMMA_* (not BRAIN_CYTOKINE_IFN_GAMMA_*)
```

**API Examples:**
```c
// Create and start immune system
brain_immune_config_t config;
brain_immune_default_config(&config);
brain_immune_system_t* immune = brain_immune_create(&config);
brain_immune_start(immune);

// Connect integrations
brain_immune_connect_bbb(immune, bbb);
brain_immune_connect_bft(immune, bft);
brain_immune_connect_swarm(immune, swarm_immune);

// Present threat (auto-checks memory for secondary response)
uint32_t antigen_id;
brain_immune_present_antigen(immune, ANTIGEN_SOURCE_MANUAL, epitope, len, severity, node, &antigen_id);

// Full immune cycle
uint32_t b_cell_id, helper_id, antibody_id;
brain_immune_activate_b_cell(immune, antigen_id, &b_cell_id);
brain_immune_activate_helper_t(immune, antigen_id, &helper_id);
brain_immune_t_help_b(immune, helper_id, b_cell_id);  // B cell → PLASMA state
brain_immune_produce_antibody(immune, b_cell_id, ANTIBODY_IGG, &antibody_id);
brain_immune_neutralize(immune, antigen_id, antibody_id);  // Auto-creates memory

// Affinity computation for threat recognition
float affinity = brain_immune_compute_affinity(pattern1, len1, pattern2, len2);
```

**Test Coverage: 104 tests**
- Unit: 47 tests (`./test/unit/cognitive/immune/unit_cognitive_immune_brain_immune`)
- Integration: 15 tests (`./test/integration/cognitive/immune/integration_cognitive_immune_brain_immune`)
- Regression: 29 tests (`./test/regression/cognitive/immune/regression_cognitive_immune_brain_immune`)
- E2E: 13 tests (`./test/e2e/e2e_test_brain_immune_pipeline`)

**GOTCHA**: B cells must be in PLASMA state to produce antibodies. Use `brain_immune_t_help_b()` to transition from ACTIVATED → PLASMA.

### Brain Immune Integration Across Modules (Complete - Dec 2024)
Integrated brain immune system bidirectionally with 27+ NIMCP modules:

| Category | Modules | Bridge Headers |
|----------|---------|----------------|
| Cognitive | Attention, Memory, Reasoning, Executive, Introspection, Curiosity, Wellbeing, Mental Health, ToM, Self-Model, Sleep | `include/cognitive/immune/nimcp_*_immune_bridge.h` |
| Perception | Visual Cortex, Audio Cortex, Speech Cortex | `include/perception/immune/nimcp_*_immune_bridge.h` |
| Plasticity | STDP, BCM, Homeostatic, Synaptic Scaling, Eligibility Traces | `include/plasticity/immune/nimcp_*_immune_bridge.h` |
| Middleware | Routing, Buffering, Population Coding, Feature Extraction, Thalamic | `include/middleware/immune/nimcp_*_immune*.h` |
| Core | Oscillations, Cortical Columns, Broca's Area | Various locations |

**Integration Pattern:**
```c
// Each bridge follows this pattern:
typedef struct {
    brain_immune_system_t* immune_system;  // Pointer to brain immune
    <module>_t* module;                     // Module being integrated
    <module>_immune_config_t config;        // Bridge configuration
    <module>_cytokine_effects_t cytokine_effects;  // Computed effects
    nimcp_mutex_t* mutex;                   // Thread safety
} <module>_immune_bridge_t;

// Standard API:
<module>_immune_bridge_t* <module>_immune_create(config, module, immune_system);
void <module>_immune_destroy(bridge);
int <module>_immune_update(bridge);  // Update cytokine effects
int <module>_immune_apply_modulation(bridge);  // Apply to module
```

**Logging Macros:**
```c
// Correct: NIMCP_LOGGING_* macros
NIMCP_LOGGING_ERROR("message");
NIMCP_LOGGING_WARN("message");
NIMCP_LOGGING_INFO("message");
NIMCP_LOGGING_DEBUG("message");

// Wrong: nimcp_log_* (doesn't exist)
```

**Error Codes:**
```c
// Correct error codes
NIMCP_ERROR_NULL_POINTER
NIMCP_ERROR_INVALID_STATE
NIMCP_ERROR_INVALID_PARAMETER
NIMCP_ERROR_OPERATION_FAILED
NIMCP_ERROR_NO_MEMORY

// Wrong (don't use)
NIMCP_ERR_*, NIMCP_ERROR_INVALID_ARGUMENT, NIMCP_ERROR_RESOURCE_EXHAUSTED
```

**Files Added (289 files, 119K+ lines):**
- 27 header files in `include/*/immune/`
- 27 implementation files in `src/*/immune/`
- 70+ test files in `test/unit/*/immune/`

### Training-Immune Integration (Complete - Dec 2024)
Bidirectional integration between brain immune system and training pipeline, modeling fever-induced learning suppression and immune responses to training instabilities:

| Direction | Integration | Biological Basis |
|-----------|-------------|------------------|
| Immune → Training | Inflammation reduces learning rate | Fever suppresses synaptic plasticity to conserve energy |
| Training → Immune | Divergence triggers immune response | Training instabilities are threats to neural integrity |

**Learning Rate Modulation (Fever Model):**
```
Inflammation Level    LR Factor    Effect
─────────────────────────────────────────────
NONE                  1.00         Full learning
LOCAL                 0.95         Slight reduction
REGIONAL              0.80         Moderate reduction
SYSTEMIC              0.50         Severe reduction
STORM                 0.10         Emergency state
```

**API Examples:**
```c
// Create and connect training immune system
training_immune_config_t config;
training_immune_default_config(&config);
training_immune_system_t* ti = training_immune_create(&config);

training_immune_connect_brain_immune(ti, brain_immune);
training_immune_connect_optimizer(ti, optimizer);
training_immune_connect_gradient_manager(ti, grad_mgr);
training_immune_start(ti);

// Training loop with immune integration
for (int step = 0; step < max_steps; step++) {
    // Update metrics (auto-detects instabilities)
    training_immune_update_metrics(ti, loss, grad_norm, lr);
    training_immune_check_stability(ti);

    // Get inflammation-modulated learning rate
    float effective_lr = training_immune_get_effective_lr(ti, base_lr);
    nimcp_optimizer_set_lr(optimizer, effective_lr);

    // Train step...
}

// Inflammation automatically modulates learning
training_immune_update_inflammation(ti, INFLAMMATION_SYSTEMIC);
float lr_factor = training_immune_get_lr_factor(ti);  // Returns 0.50
```

**Instability Detection:**
- **NaN/Inf**: Loss or gradients become invalid → Severity 10
- **Loss Explosion**: Loss increases >10x → Severity 8
- **Gradient Explosion**: Norm > threshold → Severity 6
- **Gradient Vanishing**: Norm < 1e-7 → Severity 4
- **Loss Plateau**: No improvement for N steps → Severity 3

**Test Coverage: 30+ tests**
- Lifecycle and configuration
- Integration with brain immune, optimizer, gradient manager
- LR modulation for all inflammation levels
- All instability detection types
- Auto-immune response triggering
- Statistics and monitoring

**Files:**
- Header: `include/middleware/immune/nimcp_training_immune.h`
- Implementation: `src/middleware/immune/nimcp_training_immune.c`
- Tests: `test/unit/middleware/immune/test_training_immune.cpp`

**GOTCHA**: Manual CMakeLists.txt edit required. See `TRAINING_IMMUNE_BUILD_INSTRUCTIONS.md` for details.

### Cross-Bridge Perception/Cortical Integration (Complete - Dec 2024)
Full integration of Perception-Training and Cortical-Training bridges with all existing training subsystems:

| Integration | Direction | Effect |
|-------------|-----------|--------|
| Perception → Cognitive | visual_confidence → attention_focus | High perception quality increases cognitive attention |
| Cortical → Cognitive | burst_rate → epistemic_uncertainty | Stable predictions reduce cognitive uncertainty |
| Perception → Logic | perception_quality condition | Gates require perception quality threshold |
| Cortical → Logic | cortical_stable condition | Gates require prediction stability |
| Perception → Immune | collapse detection | Low confidence triggers immune response |
| Cortical → Immune | explosion detection | Free energy explosion triggers immune response |
| Perception → Plasticity | lr_factor modulation | Perception quality scales regional plasticity |
| Cortical → Plasticity | burst_rate enhancement | Bursts enhance LTP in confirmed predictions |
| Perception → Portia-Swarm | confidence modifier [0.5, 1.5] | High perception boosts decision confidence |
| Cortical → Portia-Swarm | threshold modifier [0.7, 1.3] | Stable cortex lowers decision threshold |

**Architecture:**
```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                  │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  PERCEPTION   │────▶│    COGNITIVE      │◀────│    CORTICAL       │
│   TRAINING    │     │    TRAINING       │     │    TRAINING       │
└───────┬───────┘     └─────────┬─────────┘     └─────────┬─────────┘
        │                       │                         │
        │              ┌────────┴────────┐               │
        │              ▼                 ▼               │
        │    ┌─────────────────┐  ┌─────────────┐       │
        └───▶│ TRAINING-LOGIC  │◀─│  PORTIA-    │◀──────┘
             │    BRIDGE       │  │ SWARM-LOGIC │
             └────────┬────────┘  └─────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
┌─────────────┐ ┌───────────┐ ┌───────────┐
│  TRAINING-  │ │ TRAINING- │ │  SWARM    │
│   IMMUNE    │ │ PLASTICITY│ │ CONSENSUS │
└─────────────┘ └───────────┘ └───────────┘
```

**API Examples:**
```c
// Connect perception/cortical to cognitive bridge
cognitive_training_connect_perception_training(cognitive_bridge, perception_bridge);
cognitive_training_connect_cortical_training(cognitive_bridge, cortical_bridge);

// Connect perception/cortical to logic bridge
training_logic_connect_perception_training(logic_bridge, perception_bridge);
training_logic_connect_cortical_training(logic_bridge, cortical_bridge);

// Connect perception/cortical to immune
training_immune_connect_perception_training(immune_sys, perception_bridge);
training_immune_connect_cortical_training(immune_sys, cortical_bridge);

// Connect perception/cortical to plasticity
tpb_connect_perception_training(plasticity_ctx, perception_bridge);
tpb_connect_cortical_training(plasticity_ctx, cortical_bridge);

// Connect perception/cortical to portia-swarm-logic
portia_swarm_logic_connect_perception_training(psl_bridge, perception_bridge);
portia_swarm_logic_connect_cortical_training(psl_bridge, cortical_bridge);

// Get modifiers from portia-swarm
float conf_mod = portia_swarm_logic_get_perception_confidence_modifier(psl_bridge);  // [0.5, 1.5]
float thresh_mod = portia_swarm_logic_get_cortical_threshold_modifier(psl_bridge);   // [0.7, 1.3]
```

**Modifier Formulas:**
```c
// Perception confidence modifier
confidence_base = 0.5 + 0.5 × visual_confidence
modifier = confidence_base × lr_factor
clamped to [0.5, 1.5]

// Cortical threshold modifier
modifier = 1.0 - 0.3 × (burst_rate - 0.5)
if (!predictions_stable) modifier += 0.15
modifier += 0.1 × prediction_error_mag
clamped to [0.7, 1.3]
```

**Test Coverage: 180 tests**
- Perception-Cognitive Integration: 20 tests
- Cortical-Cognitive Integration: 20 tests
- Perception-Logic Integration: 15 tests
- Cortical-Logic Integration: 15 tests
- Perception-Immune Integration: 15 tests
- Cortical-Immune Integration: 15 tests
- Perception-Plasticity Integration: 15 tests
- Cortical-Plasticity Integration: 15 tests
- Perception-Portia-Swarm Integration: 15 tests
- Cortical-Portia-Swarm Integration: 15 tests
- Unified E2E Integration: 20 tests

**Bio-async IDs:**
- `BIO_MODULE_PERCEPTION_TRAINING = 0x0523`
- `BIO_MODULE_CORTICAL_TRAINING = 0x0524`

**Files:**
- Headers: `include/middleware/training/nimcp_*_training_bridge.h`
- Implementation: `src/middleware/training/nimcp_*_training_bridge.c`
- Portia integration: `include/portia/nimcp_portia_swarm_logic_bridge.h`
- Tests: `test/integration/middleware/training/test_*_integration.cpp`
- Tests: `test/integration/portia/test_*_portia_swarm_integration.cpp`
- E2E: `test/e2e/e2e_test_unified_training_integration.cpp`

### Liquid Neural Network (LNN) Module (Complete - Dec 2024)
Integrated Liquid Time-Constant (LTC) neural network module with continuous-time dynamics:

| Component | Description | Files |
|-----------|-------------|-------|
| Types | Core enums, structs, error codes | `include/lnn/nimcp_lnn_types.h` |
| Config | Network/layer configuration | `include/lnn/nimcp_lnn_config.h` |
| Neuron | Single LTC neuron operations | `include/lnn/nimcp_lnn_neuron.h` |
| Layer | Vectorized layer with wiring | `include/lnn/nimcp_lnn_layer.h` |
| Network | Multi-layer network | `include/lnn/nimcp_lnn_network.h` |
| ODE | Euler, Heun, RK4 solvers | `include/lnn/nimcp_lnn_ode.h` |
| Wiring | Sparse connectivity patterns | `include/lnn/nimcp_lnn_wiring.h` |
| Gradient | Adjoint method backprop | `include/lnn/nimcp_lnn_gradient.h` |
| Training | Training loop integration | `include/lnn/nimcp_lnn_training.h` |
| Bio-async | Inter-module messaging | `include/lnn/nimcp_lnn_bio_async.h` |
| Immune | Instability detection | `include/lnn/nimcp_lnn_immune.h` |
| Parallel | Multi-threaded execution | `include/lnn/nimcp_lnn_parallel.h` |

**LTC Neuron Dynamics:**
```
dx/dt = -x/τ(x,I) + f(W_in·I + W_rec·x + b)
τ(x,I) = τ_base · σ(W_τ·[x;I] + b_τ)  // Input-dependent time constant
```

**Wiring Patterns:**
- `LNN_WIRING_FULL` - Dense all-to-all
- `LNN_WIRING_RANDOM` - Erdos-Renyi sparse
- `LNN_WIRING_SMALL_WORLD` - Watts-Strogatz
- `LNN_WIRING_SCALE_FREE` - Barabasi-Albert hubs
- `LNN_WIRING_NCP` - Neural Circuit Policy (sensory→inter→command→motor)

**ODE Solvers:**
- `LNN_ODE_EULER` - 1st order, fast
- `LNN_ODE_HEUN` - 2nd order predictor-corrector
- `LNN_ODE_RK4` - 4th order Runge-Kutta (default)
- `LNN_ODE_DOPRI5` - Adaptive 5th order

**API Examples:**
```c
// Configuration
lnn_config_t config;
lnn_config_ncp(&config, 8, 16, 8, 4);  // NCP: 8 sensory, 16 inter, 8 command, 4 motor
lnn_config_validate(&config);

// Layer creation
lnn_layer_config_t layer_cfg = {
    .n_neurons = 32,
    .activation = LNN_ACTIVATION_TANH,
    .tau_base_init = 10.0f,
    .tau_min = 0.1f,
    .tau_max = 1000.0f,
    .wiring_type = LNN_WIRING_NCP,
    .ode_method = LNN_ODE_RK4,
    .dt = 1.0f
};
lnn_layer_t* layer = lnn_layer_create(&layer_cfg, n_inputs);
lnn_layer_init_weights(layer, 0.1f, seed);
lnn_layer_forward(layer, input, output, dt);

// Neuron operations
lnn_neuron_t* neuron = lnn_neuron_create(&neuron_cfg, n_inputs, n_recurrent);
float tau = lnn_neuron_compute_tau(neuron, input, n_inputs, recurrent, n_recurrent);
lnn_neuron_step(neuron, input, n_inputs, recurrent, n_recurrent, dt, LNN_ODE_RK4);
```

**LNN Error Codes:**
```c
LNN_ERROR_NONE = 0
LNN_ERROR_NULL_POINTER = -1
LNN_ERROR_INVALID_PARAM = -13
LNN_ERROR_OPERATION_FAILED = -12
LNN_ERROR_OUT_OF_MEMORY = -3
```

**Test Coverage: 84 tests**
```bash
./test/unit/lnn/unit_lnn_test_lnn_config --gtest_brief=1   # 24 tests
./test/unit/lnn/unit_lnn_test_lnn_neuron --gtest_brief=1   # 34 tests
./test/unit/lnn/unit_lnn_test_lnn_layer --gtest_brief=1    # 26 tests
```

**Bio-async Module ID:**
- `BIO_MODULE_LNN_CORE = 0x0600`

**Files:**
- Headers: `include/lnn/nimcp_lnn_*.h` (13 files)
- Implementation: `src/lnn/nimcp_lnn_*.c` (12 files)
- Tests: `test/unit/lnn/test_lnn_*.cpp` (3 files)
- Build: `src/lnn/CMakeLists.txt`

**GOTCHA**: `nimcp_tensor_create` requires 3 args: `(dims, ndims, NIMCP_DTYPE_F32)`. The dtype parameter is mandatory.

**GOTCHA**: Use `lnn_wiring_is_connected()` (inline alias) or `lnn_wiring_has_edge()` to check connectivity.

### Hemispheric Brain Architecture (Complete - Dec 2024)
Biologically-inspired two-hemisphere brain with inter-hemispheric communication via corpus callosum:

| Component | Biological Basis | NIMCP Implementation |
|-----------|------------------|---------------------|
| Left Hemisphere | Language, logic, sequential | Symbolic logic, fine motor, analytical |
| Right Hemisphere | Spatial, holistic, pattern recognition | Emotion, face recognition, creative |
| Corpus Callosum | 200M axons, 5-20ms latency | Bio-async bridge with bandwidth limiting |
| Lateralization | Hemisphere specialization | 12 cognitive domains with dominance weights |

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                      HEMISPHERIC BRAIN (hemispheric_brain_t)             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────┐     ┌──────────────────────────┐         │
│  │    LEFT HEMISPHERE       │     │    RIGHT HEMISPHERE      │         │
│  │  - Language (0.95)       │     │  - Spatial (0.80)        │         │
│  │  - Logic (0.85)          │     │  - Emotion (0.70)        │         │
│  │  - Fine Motor (0.90)     │     │  - Face Recognition (0.85)│        │
│  │  - Local Attention (0.75)│     │  - Global Attention (0.75)│        │
│  └──────────────────────────┘     └──────────────────────────┘         │
│              │                                │                         │
│              └────────────┬───────────────────┘                         │
│              ┌────────────▼────────────┐                                │
│              │    CORPUS CALLOSUM      │                                │
│              │  Channels: MOTOR,       │                                │
│              │  SENSORY, COGNITIVE,    │                                │
│              │  EMOTIONAL, INHIBITORY  │                                │
│              │  Bandwidth: Configurable│                                │
│              └─────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

**Cognitive Domains (12 total):**
```c
COGNITIVE_DOMAIN_LANGUAGE           // Left dominant (0.95)
COGNITIVE_DOMAIN_SPATIAL            // Right dominant (0.20)
COGNITIVE_DOMAIN_MOTOR_FINE         // Left dominant (0.90)
COGNITIVE_DOMAIN_MOTOR_GROSS        // Bilateral (0.50)
COGNITIVE_DOMAIN_EMOTION            // Right dominant (0.30)
COGNITIVE_DOMAIN_ATTENTION_GLOBAL   // Right dominant (0.25)
COGNITIVE_DOMAIN_ATTENTION_LOCAL    // Left dominant (0.75)
COGNITIVE_DOMAIN_MUSIC_MELODY       // Right dominant (0.20)
COGNITIVE_DOMAIN_MUSIC_RHYTHM       // Left dominant (0.80)
COGNITIVE_DOMAIN_FACE_RECOGNITION   // Right dominant (0.15)
COGNITIVE_DOMAIN_LOGICAL_REASONING  // Left dominant (0.85)
COGNITIVE_DOMAIN_CREATIVE_THINKING  // Right dominant (0.35)
```

**Processing Modes:**
- `HEMISPHERIC_MODE_LATERALIZED` - Route to dominant hemisphere
- `HEMISPHERIC_MODE_PARALLEL` - Both hemispheres process simultaneously
- `HEMISPHERIC_MODE_COMPETITIVE` - Hemispheres race, winner outputs
- `HEMISPHERIC_MODE_COOPERATIVE` - Combine outputs (weighted/average/dominant)

**Bandwidth Modes:**
```c
CALLOSUM_BW_UNLIMITED   // No limits (fast simulation)
CALLOSUM_BW_REALISTIC   // ~200 msg/s, 5-20ms latency (biological)
CALLOSUM_BW_RESTRICTED  // ~50 msg/s, 20-50ms latency (impaired)
CALLOSUM_BW_CUSTOM      // User-defined limits
```

**API Examples:**
```c
// Create hemispheric brain
hemispheric_brain_config_t config = hemispheric_brain_default_config();
hemispheric_brain_t* brain = hemispheric_brain_create(&config);

// Process with lateralization (routes to appropriate hemisphere)
hemispheric_brain_process_lateralized(brain, input, input_size,
    COGNITIVE_DOMAIN_LANGUAGE, output, output_size);  // → Left hemisphere

// Parallel processing (both hemispheres)
hemispheric_brain_process_parallel(brain, input, input_size,
    left_output, right_output, output_size);

// Competitive processing (hemispheres race)
hemisphere_id_t winner;
hemispheric_brain_process_competitive(brain, input, input_size,
    output, output_size, &winner);

// Cooperative processing (combine outputs)
hemispheric_brain_set_cooperation_strategy(brain, COOPERATION_WEIGHTED);
hemispheric_brain_process_cooperative(brain, input, input_size,
    output, output_size);

// Split-brain mode (disconnect callosum)
hemispheric_brain_disconnect_callosum(brain);
// Hemispheres now operate independently
hemispheric_brain_reconnect_callosum(brain);

// Per-hemisphere resource control
hemispheric_brain_set_tier(brain, HEMISPHERE_LEFT, PLATFORM_TIER_FULL);
hemispheric_brain_set_tier(brain, HEMISPHERE_RIGHT, PLATFORM_TIER_MINIMAL);

// Asymmetric resource allocation (70% to left)
hemispheric_brain_set_asymmetric_resources(brain, 0.7f, true);

// Query lateralization
hemisphere_id_t dominant = hemispheric_brain_get_dominant_for(brain,
    COGNITIVE_DOMAIN_SPATIAL);  // Returns HEMISPHERE_RIGHT
float dominance = hemispheric_brain_get_dominance(brain,
    COGNITIVE_DOMAIN_LANGUAGE);  // Returns 0.95

// Shift dominance (plasticity)
hemispheric_brain_shift_dominance(brain, COGNITIVE_DOMAIN_EMOTION, 0.1f);

// Hemisphere-level operations
brain_hemisphere_t* left = hemispheric_brain_get_left(brain);
hemisphere_infer(left, input, input_size, output, output_size);
hemisphere_train(left, input, target, size);  // Reward-based learning

// Neuromodulator diffusion across hemispheres
hemisphere_apply_neuromod_diffusion(left, NEUROMOD_DOPAMINE, 0.8f);

// Cleanup
hemispheric_brain_destroy(brain);
```

**Hemisphere Inference (Real Neural Network):**
```c
// hemisphere_infer uses brain_decide() internally
brain_decision_t* decision = brain_decide(hemisphere->brain, input, input_size);
memcpy(output, decision->output_vector, copy_size * sizeof(float));
brain_free_decision(decision);
```

**Hemisphere Training (Reward-Based Learning):**
```c
// hemisphere_train uses three-factor learning
hemisphere_infer(hemi, input, size, output, size);  // Activate neurons
float loss = MSE(output, target);
float reward = 2.0f * expf(-loss) - 1.0f;  // Convert loss to reward
brain_apply_reward_learning(hemi->brain, reward);  // Hebbian + Reward + Dopamine
```

**Bio-async Integration:**
```c
// Module IDs (0x1300 - 0x130F)
BIO_MODULE_HEMISPHERIC_BRAIN = 0x1300
BIO_MODULE_LEFT_HEMISPHERE   = 0x1301
BIO_MODULE_RIGHT_HEMISPHERE  = 0x1302
BIO_MODULE_CORPUS_CALLOSUM   = 0x1303
BIO_MODULE_LATERALIZATION    = 0x1304

// Message Types (0x1100 - 0x110F)
BIO_MSG_HEMISPHERE_ACTIVITY     = 0x1100
BIO_MSG_HEMISPHERE_SYNC         = 0x1101
BIO_MSG_CALLOSUM_TRANSFER       = 0x1102
BIO_MSG_LATERALIZATION_SHIFT    = 0x1103
BIO_MSG_HEMISPHERE_DOMINANCE    = 0x1104
BIO_MSG_SPLIT_BRAIN_EVENT       = 0x1105
```

**Test Coverage:**
```bash
./test/unit/core/brain/hemispheric/unit_core_brain_hemispheric_test --gtest_filter="LateralizationTest.*"  # 7 tests
./test/unit/core/brain/hemispheric/unit_core_brain_hemispheric_test --gtest_filter="*UtilTest*"           # 4 tests
./test/unit/core/brain/hemispheric/unit_core_brain_hemispheric_test --gtest_filter="CorpusCallosumTest.*" # 7 tests
```

**Files:**
- Headers: `include/core/brain/hemispheric/nimcp_*.h` (4 files)
- Implementation: `src/core/brain/hemispheric/nimcp_*.c` (4 files)
- Tests: `test/unit/core/brain/hemispheric/test_hemispheric_brain.cpp`
- Build: Integrated in `src/lib/CMakeLists.txt`

**GOTCHA**: `brain_t` creation is heavyweight (initializes 50+ subsystems). Hemisphere tests are resource-intensive.

**GOTCHA**: Use `brain_decide()` for inference, `brain_apply_reward_learning()` for training. Simple `brain_update/infer/train` don't exist.

**GOTCHA**: Platform tiers use `PLATFORM_TIER_FULL/MEDIUM/CONSTRAINED/MINIMAL`, not `PLATFORM_TIER_0/1/2/3`.

### Bio-Async Integration for Immune Bridges (Complete - Dec 2024)
All 24+ immune bridge modules now have bio-async integration for inter-module messaging:

| Category | Bridges | Module IDs |
|----------|---------|------------|
| Cognitive | attention, emotion, memory, reasoning, executive, introspection, curiosity, wellbeing, mental_health, tom, self_model, sleep, autobiographical, knowledge | BIO_MODULE_IMMUNE_ATTENTION, etc. |
| Perception | visual, audio, speech | BIO_MODULE_IMMUNE_VISUAL, etc. |
| Plasticity | stdp, bcm, homeostatic, synaptic_scaling, eligibility, dendritic | BIO_MODULE_IMMUNE_STDP, etc. |
| Middleware | routing, buffering, population_coding, feature_extractor, thalamic, sequence, training | BIO_MODULE_IMMUNE_ROUTING, etc. |
| Core | oscillations, cortical, broca | BIO_MODULE_IMMUNE_OSCILLATIONS, etc. |

**Bio-async Pattern:**
```c
// Struct fields (add to bridge struct)
bio_module_context_t bio_ctx;       // Bio-async module context
bool bio_async_enabled;              // Whether bio-async is active

// Standard API functions
int <prefix>_connect_bio_async(<bridge>_t* bridge);
int <prefix>_disconnect_bio_async(<bridge>_t* bridge);
bool <prefix>_is_bio_async_connected(const <bridge>_t* bridge);

// Implementation pattern
int emotion_immune_connect_bio_async(emotion_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_IMMUNE_EMOTION,
        .module_name = "emotion_immune_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }
    return 0;
}
```

**BIO_MODULE_* Definitions (nimcp_bio_messages.h):**
```c
/* Immune bridge modules (0x0D00 - 0x0DFF) */
BIO_MODULE_IMMUNE_BRAIN = 0x0D00,
BIO_MODULE_IMMUNE_ATTENTION,
BIO_MODULE_IMMUNE_MEMORY,
// ... 40+ immune module definitions
```

### Introspection Module Enhancements (Complete - Dec 2024)
Implemented three major introspection subsystems for brain metacognition:

| Component | Description | Key Features |
|-----------|-------------|--------------|
| Consciousness Metrics | IIT 3.0 Φ computation | State classification, MIP analysis, monitoring callbacks |
| Temporal Patterns | DTW-based pattern detection | Pattern library, prediction, trend analysis |
| Ensemble Uncertainty | Epistemic/aleatoric decomposition | Calibration metrics, confidence estimation |

**API Examples:**
```c
// Consciousness metrics
consciousness_phi_result_t* result = introspection_compute_phi(intro, NULL);
consciousness_state_t state = consciousness_classify_phi(result->phi);

// Temporal patterns
temporal_pattern_t* patterns = introspection_detect_patterns(intro, NULL, &count);
introspection_register_pattern(intro, &pattern);
introspection_clear_pattern_library(intro);

// Ensemble uncertainty
ensemble_context_t ensemble = ensemble_create(network, &config);
ensemble_uncertainty_result_t unc = ensemble_compute_uncertainty(ensemble, features, n);
// unc.epistemic, unc.aleatoric, unc.total
```

**Test Coverage:**
- Unit Tests: 101 tests (31 consciousness, 43 temporal, 27 ensemble)
- Integration Tests: 27 tests (9 per module)
- Regression Tests: 37 tests (11 consciousness, 17 temporal, 9 ensemble)
- E2E Tests: 8 pipeline scenarios
- **Total: 173 tests**

### Positional Encoding Integration (Complete - Dec 2024)
Integrated PE into 10 modules with full test coverage:

| Module | PE Types | Purpose |
|--------|----------|---------|
| Multihead Attention | RoPE, ALiBi | Q/K projections, attention bias |
| Working Memory | Sinusoidal, Relative | Serial position (7±2 items) |
| Sequence Detector | RoPE, Relative | Spike train pattern matching |
| Predictive Regions | Learned, Sinusoidal | Hierarchy levels, temporal predictions |
| Speech Cortex | Sinusoidal, Learned | Phoneme sequences, phonological buffer |
| Emotion-Attention | Sinusoidal, Learned | Temporal emotion states, priority |
| Circular Buffer | Sinusoidal, ALiBi | Buffer positions, access bias |
| Swarm Signal | Sinusoidal, ALiBi | Packet sequence ordering |
| Language Production | Sinusoidal, RoPE | Motor commands, articulatory gestures |
| Population Coding | Sinusoidal | Neuron position encoding |

### Tensor Integration (Complete)
- Phase 1: Gradient Manager, Z-Score Normalizer, Loss Functions
- Phase 2: Optimizers, Visual Cortex, Audio Cortex
- Phase 3: Population Coding, Feature Extractor, Thalamic Router
- **289 tests passed**

## File Organization
```
include/
├── lnn/                   # Liquid Neural Networks (LTC neurons, ODE solvers, wiring)
├── utils/tensor/          # Tensor library
├── utils/encoding/        # Positional encoding
├── utils/fault_tolerance/ # BFT, recovery cache, checkpointing
├── plasticity/attention/  # Multihead attention with PE
├── cognitive/introspection/  # Consciousness metrics, temporal patterns, ensemble uncertainty
├── cognitive/immune/      # Brain immune system (B/T cells, antibodies, cytokines)
├── cognitive/             # Working memory, emotion-attention
├── middleware/            # Sequence detector, circular buffer, population coding
├── perception/            # Speech cortex
├── security/              # Blood-brain barrier (BBB)
├── swarm/                 # Swarm signal, swarm immune
├── core/brain/regions/    # Language production, predictive regions
└── core/brain/hemispheric/ # Bilateral brain (hemispheres, corpus callosum, lateralization)

test/
├── unit/                  # Unit tests by module
├── integration/           # Module interaction tests
├── regression/            # Stability tests
└── e2e/                   # End-to-end pipeline tests
```

## Resource Optimization (Tier-Based Memory Optimization)

NIMCP supports resource-constrained platforms through the Portia tier system. The tier optimization header provides compile-time and runtime optimizations.

### Platform Tiers
| Tier | Hardware | Memory Budget | Use Case |
|------|----------|---------------|----------|
| `PLATFORM_TIER_FULL` | ≥8 cores, ≥8GB RAM | 8GB | Research, training |
| `PLATFORM_TIER_MEDIUM` | ≥4 cores, ≥2GB RAM | 2GB | Development, edge AI |
| `PLATFORM_TIER_CONSTRAINED` | ≥2 cores, ≥256MB RAM | 256MB | Drones, IoT gateways |
| `PLATFORM_TIER_MINIMAL` | ≥1 core, ≥64MB RAM | 64MB | Sensors, MCUs |

### Building for Constrained Platforms
```bash
# Build for MINIMAL tier (64MB target)
cmake .. -DNIMCP_BUILD_TIER=PLATFORM_TIER_MINIMAL

# Build for CONSTRAINED tier (256MB target)
cmake .. -DNIMCP_BUILD_TIER=PLATFORM_TIER_CONSTRAINED
```

### Key Optimizations (~400-500KB savings on MINIMAL tier)

| Optimization | Header | Savings | Description |
|--------------|--------|---------|-------------|
| Bio-async inbox | `nimcp_tier_optimization.h` | 150KB | Tier-based inbox capacity (4-32 messages) |
| History buffers | `nimcp_tier_optimization.h` | 50KB | Tier-scaled history sizes (4-128 entries) |
| Mutex pooling | `nimcp_mutex_pool.h` | 5.6KB | Shared mutex pool for bridges |
| Fixed arrays | `nimcp_tier_optimization.h` | 100KB | Tier-based dendrite/synapse arrays |
| Statistics | `NIMCP_ENABLE_STATISTICS` | 10KB | Conditional statistics collection |

### Tier Optimization Constants
```c
#include "utils/platform/nimcp_tier_optimization.h"

// Bio-async inbox capacity (MINIMAL=4, FULL=32)
.inbox_capacity = NIMCP_BIO_INBOX_CAPACITY

// History buffer sizes
NIMCP_HISTORY_SIZE_SMALL   // MINIMAL=4, FULL=32
NIMCP_HISTORY_SIZE_MEDIUM  // MINIMAL=8, FULL=64
NIMCP_HISTORY_SIZE_LARGE   // MINIMAL=16, FULL=128

// Fixed array sizes
NIMCP_MAX_SPINE_IDS        // MINIMAL=8, FULL=64
NIMCP_MAX_SYNAPSE_IDS      // MINIMAL=32, FULL=256

// Feature flags
NIMCP_ENABLE_STATISTICS    // 0 on MINIMAL, 1 otherwise
NIMCP_ENABLE_DETAILED_HISTORY
NIMCP_USE_MUTEX_POOL       // 1 on MINIMAL for memory savings
```

### Mutex Pool (Bridge Memory Optimization)
```c
#include "utils/thread/nimcp_mutex_pool.h"

// Instead of per-bridge mutex allocation:
// bridge->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));

// Use the shared pool (on MINIMAL tier via macros):
NIMCP_BRIDGE_MUTEX_FIELD              // Declares mutex_slot or mutex*
NIMCP_BRIDGE_MUTEX_INIT(bridge, name) // Acquire slot or allocate
NIMCP_BRIDGE_MUTEX_LOCK(bridge)       // Lock
NIMCP_BRIDGE_MUTEX_UNLOCK(bridge)     // Unlock
NIMCP_BRIDGE_MUTEX_DESTROY(bridge)    // Release slot or free
```

### Tier-Aware Allocation Helpers
```c
// Scale buffer size by tier
size_t buf_size = nimcp_tier_scale_size(1024);  // MINIMAL=128, FULL=1024

// Scale array count by tier
uint32_t count = nimcp_tier_scale_count(256);   // MINIMAL=32, FULL=256

// Get tier-appropriate thread count
uint32_t threads = nimcp_tier_thread_count();   // MINIMAL=1, FULL=8

// Get memory budget in bytes
size_t budget = nimcp_tier_memory_budget_bytes();
```

### Files
- Header: `include/utils/platform/nimcp_tier_optimization.h`
- Mutex pool: `include/utils/thread/nimcp_mutex_pool.h`
- Implementation: `src/utils/thread/nimcp_mutex_pool.c`

## Common Issues & Solutions

### CMake Subdirectory Conflicts
If `add_subdirectory()` causes duplicate target errors, check if the subdirectory is already added from `test/CMakeLists.txt`.

### Missing Test Files
CMakeLists may reference non-existent test files. Check with `ls` before adding tests.

### Bio-async Message
"Bio-async router not available, skipping registration" is normal/expected in tests.

## Git Workflow
```bash
git add -A
git commit --no-verify -m "message"
git push
```
