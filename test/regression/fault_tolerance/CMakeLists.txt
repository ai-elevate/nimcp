# ============================================================================
# Fault Tolerance Regression Tests
# ============================================================================
# WHAT: Regression tests for health system stability
# WHY:  Phase 8 health integration requires stability testing
# HOW:  GoogleTest-based tests for edge cases and boundary conditions
# ============================================================================

# Main Fault Tolerance Regression Tests
add_executable(regression_fault_tolerance
    test_fault_tolerance_regression.cpp
)

target_include_directories(regression_fault_tolerance PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(regression_fault_tolerance
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(regression_fault_tolerance PRIVATE cxx_std_17)
target_compile_definitions(regression_fault_tolerance PRIVATE NIMCP_TESTING)

gtest_discover_tests(regression_fault_tolerance
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;fault_tolerance"
        TIMEOUT 180
)

# Health Stability Regression Tests (Phase 8)
add_executable(regression_health_stability
    test_health_stability_regression.cpp
)

target_include_directories(regression_health_stability PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(regression_health_stability
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(regression_health_stability PRIVATE cxx_std_17)
target_compile_definitions(regression_health_stability PRIVATE NIMCP_TESTING)

gtest_discover_tests(regression_health_stability
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;fault_tolerance;health;phase8"
        TIMEOUT 180
)

# Heartbeat B01 Memory Core Regression Tests (Phase 8: Per-Batch Heartbeat Testing)
add_executable(regression_heartbeat_B01_memory_core
    test_heartbeat_B01_memory_core_regression.cpp
)

target_include_directories(regression_heartbeat_B01_memory_core PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(regression_heartbeat_B01_memory_core
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(regression_heartbeat_B01_memory_core PRIVATE cxx_std_17)
target_compile_definitions(regression_heartbeat_B01_memory_core PRIVATE NIMCP_TESTING)

gtest_discover_tests(regression_heartbeat_B01_memory_core
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;fault_tolerance;heartbeat;B01;memory_core"
        TIMEOUT 180
)

# Heartbeat B02 Memory (non-core) Regression Tests (Phase 8: Per-Batch Heartbeat Testing)
add_executable(regression_heartbeat_B02_memory
    test_heartbeat_B02_memory_regression.cpp
)

target_include_directories(regression_heartbeat_B02_memory PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(regression_heartbeat_B02_memory
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(regression_heartbeat_B02_memory PRIVATE cxx_std_17)
target_compile_definitions(regression_heartbeat_B02_memory PRIVATE NIMCP_TESTING)

gtest_discover_tests(regression_heartbeat_B02_memory
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;fault_tolerance;heartbeat;B02;memory"
        TIMEOUT 180
)

# Heartbeat B03 Immune Regression Tests (Phase 8: Per-Batch Heartbeat Testing)
add_executable(regression_heartbeat_B03_immune
    test_heartbeat_B03_immune_regression.cpp
)

target_include_directories(regression_heartbeat_B03_immune PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(regression_heartbeat_B03_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(regression_heartbeat_B03_immune PRIVATE cxx_std_17)
target_compile_definitions(regression_heartbeat_B03_immune PRIVATE NIMCP_TESTING)

gtest_discover_tests(regression_heartbeat_B03_immune
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;fault_tolerance;heartbeat;B03;immune"
        TIMEOUT 180
)

message(STATUS "Added fault tolerance regression tests")
