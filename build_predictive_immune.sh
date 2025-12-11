#!/bin/bash
# Build script for predictive-immune integration

set -e

cd /home/bbrelin/nimcp/build

echo "Configuring CMake..."
cmake .. 2>&1 | tail -30

echo ""
echo "Building nimcp library..."
make nimcp -j4 2>&1 | tail -30

echo ""
echo "Building tests..."
make unit_cognitive_predictive_immune_integration -j4

echo ""
echo "Running tests..."
./test/unit/cognitive/predictive_immune/unit_cognitive_predictive_immune_integration --gtest_brief=1

echo ""
echo "Build and test complete!"
