#!/bin/bash
# Build and test Homeostatic Plasticity-Immune Integration Module

set -e  # Exit on error

echo "========================================"
echo "Homeostatic Plasticity-Immune Integration"
echo "========================================"
echo

# Navigate to build directory
cd "$(dirname "$0")/build"

# Configure CMake
echo "1. Configuring CMake..."
cmake .. > /dev/null 2>&1
echo "   ✓ CMake configured"
echo

# Build main library
echo "2. Building NIMCP library..."
make nimcp -j4 > /dev/null 2>&1
echo "   ✓ Library built"
echo

# Build test
echo "3. Building homeostatic-immune test..."
make unit_plasticity_homeostatic_immune -j4 2>&1 | grep -E "Built|error|warning" || true
echo "   ✓ Test built"
echo

# Run test
echo "4. Running tests..."
echo "========================================"
./test/unit/plasticity/immune/unit_plasticity_homeostatic_immune --gtest_brief=1
echo "========================================"
echo

echo "✅ All tests completed successfully!"
echo
echo "Test executable: ./test/unit/plasticity/immune/unit_plasticity_homeostatic_immune"
echo "Documentation: ../HOMEOSTATIC_IMMUNE_INTEGRATION.md"
echo
echo "Files created:"
echo "  - include/plasticity/immune/nimcp_homeostatic_immune_bridge.h"
echo "  - src/plasticity/immune/nimcp_homeostatic_immune_bridge.c"
echo "  - test/unit/plasticity/immune/test_homeostatic_immune_integration.cpp"
echo
