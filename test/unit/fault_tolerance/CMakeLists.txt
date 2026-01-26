# ============================================================================
# Fault Tolerance Unit Tests
# ============================================================================
# WHAT: Unit tests for health agent, state manager, and recovery systems
# WHY:  Phase 8 health integration requires comprehensive testing
# HOW:  GoogleTest-based tests for individual components
# ============================================================================

# Health Agent Heartbeat Unit Tests
add_executable(unit_health_agent_heartbeat
    test_health_agent_heartbeat.cpp
)

target_include_directories(unit_health_agent_heartbeat PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(unit_health_agent_heartbeat
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(unit_health_agent_heartbeat PRIVATE cxx_std_17)
target_compile_definitions(unit_health_agent_heartbeat PRIVATE NIMCP_TESTING)

gtest_discover_tests(unit_health_agent_heartbeat
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;fault_tolerance;health;heartbeat"
        TIMEOUT 120
)

# State Manager Unit Tests
add_executable(unit_state_manager
    test_state_manager.cpp
)

target_include_directories(unit_state_manager PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(unit_state_manager
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(unit_state_manager PRIVATE cxx_std_17)
target_compile_definitions(unit_state_manager PRIVATE NIMCP_TESTING)

gtest_discover_tests(unit_state_manager
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;fault_tolerance;state_manager"
        TIMEOUT 120
)

# Training State Manager Unit Tests (Phase 8)
add_executable(unit_training_state_manager
    test_training_state_manager.cpp
)

target_include_directories(unit_training_state_manager PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(unit_training_state_manager
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(unit_training_state_manager PRIVATE cxx_std_17)
target_compile_definitions(unit_training_state_manager PRIVATE NIMCP_TESTING)

gtest_discover_tests(unit_training_state_manager
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;fault_tolerance;training;state_manager"
        TIMEOUT 120
)

# Heartbeat B01 Memory Core Unit Tests (Phase 8: Per-Batch Heartbeat Testing)
add_executable(unit_heartbeat_B01_memory_core
    test_heartbeat_B01_memory_core_functions.cpp
)

target_include_directories(unit_heartbeat_B01_memory_core PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(unit_heartbeat_B01_memory_core
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(unit_heartbeat_B01_memory_core PRIVATE cxx_std_17)
target_compile_definitions(unit_heartbeat_B01_memory_core PRIVATE NIMCP_TESTING)

gtest_discover_tests(unit_heartbeat_B01_memory_core
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;fault_tolerance;heartbeat;B01;memory_core"
        TIMEOUT 120
)

# Heartbeat B02 Memory (non-core) Unit Tests (Phase 8: Per-Batch Heartbeat Testing)
add_executable(unit_heartbeat_B02_memory
    test_heartbeat_B02_memory_functions.cpp
)

target_include_directories(unit_heartbeat_B02_memory PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(unit_heartbeat_B02_memory
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(unit_heartbeat_B02_memory PRIVATE cxx_std_17)
target_compile_definitions(unit_heartbeat_B02_memory PRIVATE NIMCP_TESTING)

gtest_discover_tests(unit_heartbeat_B02_memory
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;fault_tolerance;heartbeat;B02;memory"
        TIMEOUT 120
)

# Heartbeat B03 Immune Unit Tests (Phase 8: Per-Batch Heartbeat Testing)
add_executable(unit_heartbeat_B03_immune
    test_heartbeat_B03_immune_functions.cpp
)

target_include_directories(unit_heartbeat_B03_immune PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(unit_heartbeat_B03_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(unit_heartbeat_B03_immune PRIVATE cxx_std_17)
target_compile_definitions(unit_heartbeat_B03_immune PRIVATE NIMCP_TESTING)

gtest_discover_tests(unit_heartbeat_B03_immune
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;fault_tolerance;heartbeat;B03;immune"
        TIMEOUT 120
)

message(STATUS "Added fault tolerance unit tests")
