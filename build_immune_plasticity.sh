#!/bin/bash
# Build script for immune-plasticity integration

cd /home/bbrelin/nimcp/build

echo "=== Configuring CMake ==="
cmake ..

echo ""
echo "=== Building nimcp library ==="
make nimcp -j4

echo ""
echo "=== Building unit test ==="
make unit_cognitive_immune_plasticity_modulation -j4

echo ""
echo "=== Building E2E test ==="
make e2e_test_brain_immune_plasticity_pipeline -j4

echo ""
echo "=== Running unit tests ==="
./test/unit/cognitive/immune/unit_cognitive_immune_plasticity_modulation --gtest_brief=1

echo ""
echo "=== Running E2E test ==="
./test/e2e/e2e_test_brain_immune_plasticity_pipeline --gtest_brief=1

echo ""
echo "=== All tests complete ==="
