# Build and Test Commands

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

## Git Workflow

```bash
git add -A
git commit --no-verify -m "message"
git push
```
