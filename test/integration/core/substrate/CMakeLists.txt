# CMakeLists.txt for Core Substrate Bridges Integration Tests
# Tests integration between neuron, synapse, and axon/dendrite substrate bridges

# Test executable: integration_core_substrate_bridges
add_executable(integration_core_substrate_bridges
    test_core_substrate_bridges_integration.cpp
)

# Include directories
target_include_directories(integration_core_substrate_bridges PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/test
)

# Link libraries
target_link_libraries(integration_core_substrate_bridges PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    pthread
    m
)

# Compiler flags
target_compile_options(integration_core_substrate_bridges PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-unused-variable
)

# Add to test suite
add_test(
    NAME integration_core_substrate_bridges
    COMMAND integration_core_substrate_bridges --gtest_brief=1
)

# Test properties
set_tests_properties(integration_core_substrate_bridges PROPERTIES
    TIMEOUT 120
    LABELS "integration;core;substrate;bridges"
)
