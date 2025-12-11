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
└── core/brain/regions/    # Language production, predictive regions

test/
├── unit/                  # Unit tests by module
├── integration/           # Module interaction tests
├── regression/            # Stability tests
└── e2e/                   # End-to-end pipeline tests
```

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
