# ============================================================================
# Cognitive Integration Regression Tests
# ============================================================================
# WHAT: Regression tests for Cognitive Integration system
# WHY:  Ensure event ordering, no event loss, thread safety, and performance bounds
# HOW:  Test edge cases, stress conditions, and performance thresholds
# ============================================================================

# Hub Regression Tests
add_executable(regression_cognitive_test_hub_regression
    test_hub_regression.cpp
)

target_link_libraries(regression_cognitive_test_hub_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_cognitive_test_hub_regression
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(regression_cognitive_test_hub_regression PRIVATE cxx_std_17)

gtest_discover_tests(regression_cognitive_test_hub_regression
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;cognitive;integration;hub"
        TIMEOUT 300
)

# Bridge Regression Tests
add_executable(regression_cognitive_test_bridge_regression
    test_bridge_regression.cpp
)

target_link_libraries(regression_cognitive_test_bridge_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_cognitive_test_bridge_regression
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(regression_cognitive_test_bridge_regression PRIVATE cxx_std_17)

gtest_discover_tests(regression_cognitive_test_bridge_regression
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;cognitive;integration;bridge"
        TIMEOUT 300
)

# Performance Regression Tests
add_executable(regression_cognitive_test_integration_performance_regression
    test_integration_performance_regression.cpp
)

target_link_libraries(regression_cognitive_test_integration_performance_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_cognitive_test_integration_performance_regression
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(regression_cognitive_test_integration_performance_regression PRIVATE cxx_std_17)

gtest_discover_tests(regression_cognitive_test_integration_performance_regression
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;cognitive;integration;performance"
        TIMEOUT 300
)
