# ============================================================================
# Parietal Lobe Integration Tests
# Tests cross-module interactions within the Parietal subsystem
# ============================================================================

# Main Parietal Integration Test
add_executable(integration_cognitive_parietal
    test_parietal_integration.cpp
)

target_link_libraries(integration_cognitive_parietal
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_parietal
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(integration_cognitive_parietal PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_parietal
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;parietal"
        TIMEOUT 180
)
