#!/bin/bash
# Build script for attention-immune integration
# Usage: ./build_attention_immune.sh

set -e

echo "===== Building NIMCP with Attention-Immune Integration ====="

cd /home/bbrelin/nimcp/build

# Configure
echo "Configuring CMake..."
cmake .. > /dev/null 2>&1 || {
    echo "CMake configuration failed"
    cmake .. 2>&1 | tail -30
    exit 1
}

# Build library
echo "Building nimcp library..."
make nimcp -j4 > /dev/null 2>&1 || {
    echo "Library build failed"
    make nimcp 2>&1 | tail -50
    exit 1
}

# Build test
echo "Building attention-immune bridge test..."
make unit_cognitive_immune_attention_bridge -j4 2>&1 | tail -20 || {
    echo "Test build failed"
    make unit_cognitive_immune_attention_bridge 2>&1 | tail -50
    exit 1
}

echo ""
echo "===== Running Attention-Immune Integration Tests ====="
echo ""

# Run test
./test/unit/cognitive/immune/unit_cognitive_immune_attention_bridge --gtest_brief=1

echo ""
echo "===== Build and Test Complete ====="
