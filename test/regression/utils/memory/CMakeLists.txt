# Memory Utils Regression Tests
# Phase 2: Brain Substrate Pools Performance

add_executable(regression_utils_memory_test_brain_pools_regression
    test_brain_pools_regression.cpp
)
target_link_libraries(regression_utils_memory_test_brain_pools_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        m
        pthread
)
add_test(NAME regression_utils_memory_test_brain_pools_regression
         COMMAND regression_utils_memory_test_brain_pools_regression)

# Phase 3 Layer Pools Regression Tests
add_executable(regression_utils_memory_test_layer_pools_regression
    test_layer_pools_regression.cpp
)
target_link_libraries(regression_utils_memory_test_layer_pools_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        m
        pthread
)
add_test(NAME regression_utils_memory_test_layer_pools_regression
         COMMAND regression_utils_memory_test_layer_pools_regression)

# Unified Memory Regression Tests
add_executable(regression_utils_memory_test_unified_memory_regression
    test_unified_memory_regression.cpp
)
target_link_libraries(regression_utils_memory_test_unified_memory_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        m
        pthread
)
add_test(NAME regression_utils_memory_test_unified_memory_regression
         COMMAND regression_utils_memory_test_unified_memory_regression)
