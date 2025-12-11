#!/bin/bash
# Build and test script for perception-immune integration

set -e  # Exit on error

echo "==================================================================="
echo "Building NIMCP with Perception-Immune Integration"
echo "==================================================================="

cd /home/bbrelin/nimcp/build

echo ""
echo "Step 1: Running CMake configuration..."
cmake ..

echo ""
echo "Step 2: Building main library..."
make nimcp -j4

echo ""
echo "Step 3: Building unit tests..."
make unit_cognitive_immune_perception_immune -j4

echo ""
echo "Step 4: Building integration tests..."
make integration_cognitive_immune_perception_immune -j4

echo ""
echo "==================================================================="
echo "Running Tests"
echo "==================================================================="

echo ""
echo "Test 1: Unit Tests - Perception Immune Integration"
echo "-------------------------------------------------------------------"
./test/unit/cognitive/immune/unit_cognitive_immune_perception_immune --gtest_brief=1

echo ""
echo "Test 2: Integration Tests - Perception Immune Pipeline"
echo "-------------------------------------------------------------------"
./test/integration/cognitive/immune/integration_cognitive_immune_perception_immune --gtest_brief=1

echo ""
echo "==================================================================="
echo "All Tests Completed Successfully!"
echo "==================================================================="
