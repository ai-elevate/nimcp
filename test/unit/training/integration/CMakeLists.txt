# ============================================================================
# Training Integration Hub Unit Tests
# ============================================================================
# Tests for the training integration hub functionality including:
# - Hub lifecycle (create/destroy)
# - Module registration/unregistration
# - Event subscription and publishing
# - Inter-module queries
# - Statistics tracking
# - Convenience helper functions
# ============================================================================

add_executable(unit_test_training_integration_hub
    test_training_integration_hub.cpp
)

target_link_libraries(unit_test_training_integration_hub
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_test_training_integration_hub
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_test_training_integration_hub PRIVATE cxx_std_17)

gtest_discover_tests(unit_test_training_integration_hub
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;training;integration;hub"
        TIMEOUT 60
)
