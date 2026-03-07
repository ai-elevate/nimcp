# ============================================================================
# 40-Watt Brain Optimization Regression Tests
# ============================================================================
add_executable(regression_optimization
    test_optimization_regression.cpp
)
target_link_libraries(regression_optimization
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(regression_optimization
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(regression_optimization PRIVATE cxx_std_17)
gtest_discover_tests(regression_optimization
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "regression;optimization" TIMEOUT 300
)

# Hot Path Regression Tests
add_executable(regression_hotpath
    test_hotpath_regression.cpp
)
target_link_libraries(regression_hotpath
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(regression_hotpath
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(regression_hotpath PRIVATE cxx_std_17)
gtest_discover_tests(regression_hotpath
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "regression;optimization;hot_path" TIMEOUT 300
)
