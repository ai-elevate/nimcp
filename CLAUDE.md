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
# Tensor library tests (87 total)
./test/unit/utils/tensor/unit_utils_tensor_test_tensor --gtest_brief=1        # 51 tests
./test/unit/utils/encoding/unit_utils_encoding_test_positional_encoding --gtest_brief=1  # 18 tests
./test/integration/utils/tensor/integration_utils_tensor_test_tensor_memory_integration --gtest_brief=1  # 9 tests
./test/regression/utils/tensor/regression_utils_tensor_test_tensor_performance --gtest_brief=1  # 9 tests

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

### Tensor Integration (Complete)
- Phase 1: Gradient Manager, Z-Score Normalizer, Loss Functions
- Phase 2: Optimizers, Visual Cortex, Audio Cortex
- Phase 3: Population Coding, Feature Extractor, Thalamic Router
- **289 tests passed**

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

**Test Coverage Created:**
- Unit Tests: ~220 tests across 10 files
- Integration Tests: 3 files (~30 tests)
- Regression Tests: 4 files (~2,840 lines)
- E2E Tests: 1 pipeline test (8 scenarios)

## File Organization
```
include/
├── utils/tensor/          # Tensor library
├── utils/encoding/        # Positional encoding
├── plasticity/attention/  # Multihead attention with PE
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
