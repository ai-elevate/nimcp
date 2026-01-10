# ============================================================================
# Medulla Oblongata Regression Tests
# ============================================================================
#
# WHAT: Regression tests for Medulla module stability
# WHY:  Prevent regressions in arousal, protection, circadian behavior
# HOW:  Test known edge cases, boundary conditions, and historical bugs
#

# Test: Medulla Regression (boundary conditions, state consistency)
add_executable(regression_core_medulla
    regression_core_medulla.cpp
)

target_link_libraries(regression_core_medulla
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)

target_include_directories(regression_core_medulla
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(regression_core_medulla PRIVATE cxx_std_17)

# Register test with CTest
add_test(NAME regression_core_medulla
         COMMAND regression_core_medulla --gtest_brief=1)

set_tests_properties(regression_core_medulla
    PROPERTIES
        TIMEOUT 180
        LABELS "regression;core;medulla;brainstem;stability"
)

# ============================================================================
# Medulla-Cerebellum Bridge Regression Tests
# ============================================================================
#
# WHAT: Regression tests for medulla-cerebellum bridge via inferior olive
# WHY:  Ensure bridge stability, performance, numerical accuracy, correctness
# HOW:  Test performance benchmarks, memory stability, numerical precision,
#       boundary conditions, and state consistency
#
# TEST CATEGORIES:
# 1. PERF  - Performance benchmarks (throughput, latency)
# 2. MEM   - Memory stability (create/destroy cycles, long-running ops)
# 3. NUM   - Numerical accuracy (Yerkes-Dodson, modulation, oscillation)
# 4. BOUND - Boundary conditions (error types, limits, extremes)
# 5. STATE - State consistency (reset, statistics, connections)

add_executable(test_medulla_cerebellum_bridge_regression
    test_medulla_cerebellum_bridge_regression.cpp
)

target_link_libraries(test_medulla_cerebellum_bridge_regression
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)

target_include_directories(test_medulla_cerebellum_bridge_regression
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(test_medulla_cerebellum_bridge_regression PRIVATE cxx_std_17)

# Register test with CTest
add_test(NAME test_medulla_cerebellum_bridge_regression
         COMMAND test_medulla_cerebellum_bridge_regression --gtest_brief=1)

set_tests_properties(test_medulla_cerebellum_bridge_regression
    PROPERTIES
        TIMEOUT 300
        LABELS "regression;core;medulla;cerebellum;bridge;performance;stability"
)
