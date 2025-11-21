# ============================================================================
# Brain Factory Regression Tests
# ============================================================================

# Test 1: Factory Regression - Stability and consistency
add_executable(regression_core_brain_factory_test_factory_regression
    test_factory_regression.cpp
)
target_link_libraries(regression_core_brain_factory_test_factory_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_core_brain_factory_test_factory_regression
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME regression_core_brain_factory_test_factory_regression
         COMMAND regression_core_brain_factory_test_factory_regression)
set_tests_properties(regression_core_brain_factory_test_factory_regression PROPERTIES LABELS "regression")
