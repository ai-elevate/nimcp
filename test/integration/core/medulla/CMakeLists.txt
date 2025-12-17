# ============================================================================
# Medulla Oblongata Integration Tests
# ============================================================================
#
# WHAT: Integration tests for Medulla module interactions
# WHY:  Verify arousal, protection, circadian, and coupling work together
# HOW:  Test cross-module behavior with realistic scenarios
#

# Test: Medulla Integration (all subsystems)
add_executable(integration_core_medulla
    integration_core_medulla.cpp
)

target_link_libraries(integration_core_medulla
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)

target_include_directories(integration_core_medulla
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(integration_core_medulla PRIVATE cxx_std_17)

# Register test with CTest
add_test(NAME integration_core_medulla
         COMMAND integration_core_medulla --gtest_brief=1)

set_tests_properties(integration_core_medulla
    PROPERTIES
        TIMEOUT 120
        LABELS "integration;core;medulla;brainstem;autonomic"
)
