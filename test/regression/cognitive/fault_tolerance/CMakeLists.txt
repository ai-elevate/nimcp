# Cognitive Fault Tolerance Regression Tests

# Emotional Tagging Regression Tests
add_executable(regression_cognitive_fault_tolerance_emotional_tagging
    test_emotional_tagging_regression.cpp
)
target_link_libraries(regression_cognitive_fault_tolerance_emotional_tagging
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_cognitive_fault_tolerance_emotional_tagging
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME regression_cognitive_fault_tolerance_emotional_tagging
         COMMAND regression_cognitive_fault_tolerance_emotional_tagging)
set_tests_properties(regression_cognitive_fault_tolerance_emotional_tagging PROPERTIES LABELS "regression")
