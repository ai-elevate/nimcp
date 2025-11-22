# Brain Regression Tests
# Test core brain functionality regressions

# Note: factory and processing subdirectories are added separately by main CMakeLists.txt

# Oscillation Backward Compatibility Test
add_executable(regression_core_brain_oscillation_backward_compat
    ../test_oscillation_backward_compat.cpp
)
target_link_libraries(regression_core_brain_oscillation_backward_compat
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
    )
add_test(NAME regression_core_brain_oscillation_backward_compat
         COMMAND regression_core_brain_oscillation_backward_compat)

# Brain JSON Regression Test
add_executable(regression_core_brain_json_regression
    test_brain_json_regression.cpp
)
target_link_libraries(regression_core_brain_json_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
    )
add_test(NAME regression_core_brain_json_regression
         COMMAND regression_core_brain_json_regression)

# Distributed COW Regression Test
add_executable(regression_core_brain_distributed_cow_regression
    test_distributed_cow_regression.cpp
)
target_link_libraries(regression_core_brain_distributed_cow_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
    )
add_test(NAME regression_core_brain_distributed_cow_regression
         COMMAND regression_core_brain_distributed_cow_regression)

# Memory Tracking Regression Test
add_executable(regression_core_brain_memory_tracking_regression
    test_memory_tracking_regression.cpp
)
target_link_libraries(regression_core_brain_memory_tracking_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
    )
add_test(NAME regression_core_brain_memory_tracking_regression
         COMMAND regression_core_brain_memory_tracking_regression)
