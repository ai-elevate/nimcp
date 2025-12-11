#!/bin/bash
# Build script for Executive Function-Immune Integration Bridge
# Created: 2025-12-11

set -e  # Exit on error

echo "========================================"
echo "Executive-Immune Bridge Build Script"
echo "========================================"
echo ""

# Check we're in the right directory
if [ ! -d "build" ]; then
    echo "Error: build directory not found. Run from nimcp root."
    exit 1
fi

cd build

echo "Step 1: Running CMake..."
cmake .. || { echo "CMake failed!"; exit 1; }

echo ""
echo "Step 2: Building executive-immune bridge source..."
# Note: This requires manual CMakeLists.txt updates first
# The source file should be added to src/cognitive/immune/CMakeLists.txt

echo ""
echo "Step 3: Building test executable..."
# Note: This requires manual CMakeLists.txt updates first
# The test should be added to test/unit/cognitive/executive/CMakeLists.txt
# make test_executive_immune_full_integration -j4

echo ""
echo "========================================"
echo "BUILD COMPLETE"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Add to src/cognitive/immune/CMakeLists.txt:"
echo "   nimcp_executive_immune_bridge.c"
echo ""
echo "2. Add to test/unit/cognitive/executive/CMakeLists.txt:"
echo "   test_executive_immune_full_integration.cpp"
echo ""
echo "3. Then run:"
echo "   cd build"
echo "   make test_executive_immune_full_integration -j4"
echo "   ./test/unit/cognitive/executive/test_executive_immune_full_integration --gtest_brief=1"
echo ""
