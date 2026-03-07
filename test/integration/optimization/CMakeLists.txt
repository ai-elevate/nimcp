# ============================================================================
# 40-Watt Brain Optimization Integration Tests
# ============================================================================
add_executable(integration_optimization
    test_optimization_integration.cpp
)
target_link_libraries(integration_optimization
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(integration_optimization
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(integration_optimization PRIVATE cxx_std_17)
gtest_discover_tests(integration_optimization
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "integration;optimization" TIMEOUT 300
)

# Hot Path Integration Tests
add_executable(integration_hotpath
    test_hotpath_integration.cpp
)
target_link_libraries(integration_hotpath
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(integration_hotpath
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(integration_hotpath PRIVATE cxx_std_17)
gtest_discover_tests(integration_hotpath
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "integration;optimization;hot_path" TIMEOUT 300
)
