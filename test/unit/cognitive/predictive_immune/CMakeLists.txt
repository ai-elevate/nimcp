# Unit tests for predictive-immune integration

# Test executable
add_executable(unit_cognitive_predictive_immune_integration
    test_predictive_immune_integration.cpp
)

target_include_directories(unit_cognitive_predictive_immune_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(unit_cognitive_predictive_immune_integration
    nimcp
    GTest::gtest
    GTest::gtest_main
    pthread
)

# Add test to CTest
add_test(
    NAME unit_cognitive_predictive_immune_integration
    COMMAND unit_cognitive_predictive_immune_integration --gtest_brief=1
)

set_tests_properties(unit_cognitive_predictive_immune_integration PROPERTIES
    LABELS "unit;cognitive;predictive;immune;integration"
    TIMEOUT 30
)
