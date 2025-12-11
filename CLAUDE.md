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
├── plasticity/attention/  # Multihead attention with PE
├── cognitive/introspection/  # Consciousness metrics, temporal patterns, ensemble uncertainty
├── cognitive/             # Working memory, emotion-attention
├── middleware/            # Sequence detector, circular buffer, population coding
├── perception/            # Speech cortex
├── swarm/                 # Swarm signal
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
