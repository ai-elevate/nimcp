# ============================================================================
# 40-Watt Brain Optimization E2E Tests
# ============================================================================
add_executable(e2e_optimization
    test_optimization_e2e.cpp
)
target_link_libraries(e2e_optimization
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(e2e_optimization
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(e2e_optimization PRIVATE cxx_std_17)
gtest_discover_tests(e2e_optimization
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "e2e;optimization" TIMEOUT 300
)
