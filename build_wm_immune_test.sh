#!/bin/bash
# Build script for working memory - immune integration test
# Usage: ./build_wm_immune_test.sh

set -e  # Exit on error

echo "========================================"
echo "Building WM-Immune Integration Test"
echo "========================================"

cd /home/bbrelin/nimcp/build

# Configure if needed
if [ ! -f CMakeCache.txt ]; then
    echo "Configuring CMake..."
    cmake .. -DCMAKE_BUILD_TYPE=Debug
fi

# Build nimcp library
echo ""
echo "Building nimcp library..."
make nimcp -j4

# Build the test
echo ""
echo "Building unit_cognitive_test_working_memory_immune..."
make unit_cognitive_test_working_memory_immune -j4

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "Run tests with:"
echo "  ./test/unit/cognitive/unit_cognitive_test_working_memory_immune --gtest_brief=1"
echo ""
