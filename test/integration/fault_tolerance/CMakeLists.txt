# Fault Tolerance Integration Tests

# Main Fault Tolerance Integration Tests
add_executable(integration_fault_tolerance
    test_fault_tolerance_integration.cpp
)
target_link_libraries(integration_fault_tolerance
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
)
target_include_directories(integration_fault_tolerance
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME integration_fault_tolerance
         COMMAND integration_fault_tolerance)
set_tests_properties(integration_fault_tolerance PROPERTIES
    LABELS "integration;fault_tolerance"
    TIMEOUT 120
)

# Self-Repair Integration Tests
add_executable(integration_self_repair
    test_self_repair_integration.cpp
)
target_link_libraries(integration_self_repair
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
)
target_include_directories(integration_self_repair
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME integration_self_repair
         COMMAND integration_self_repair)
set_tests_properties(integration_self_repair PROPERTIES
    LABELS "integration;fault_tolerance;self-repair"
    TIMEOUT 180
)

# Self-Repair Bio-Async Integration Tests
add_executable(integration_self_repair_bio_async
    test_self_repair_bio_async_integration.cpp
)
target_link_libraries(integration_self_repair_bio_async
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
)
target_include_directories(integration_self_repair_bio_async
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME integration_self_repair_bio_async
         COMMAND integration_self_repair_bio_async)
set_tests_properties(integration_self_repair_bio_async PROPERTIES
    LABELS "integration;fault_tolerance;self-repair;bio-async"
    TIMEOUT 120
)

# Health Module Integration Tests (Phase 8: System-Wide Health Integration)
add_executable(integration_health_module
    test_health_module_integration.cpp
)
target_link_libraries(integration_health_module
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
)
target_include_directories(integration_health_module
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME integration_health_module
         COMMAND integration_health_module)
set_tests_properties(integration_health_module PROPERTIES
    LABELS "integration;fault_tolerance;health;phase8"
    TIMEOUT 180
)

# Heartbeat B01 Memory Core Integration Tests (Phase 8: Per-Batch Heartbeat Testing)
add_executable(integration_heartbeat_B01_memory_core
    test_heartbeat_B01_memory_core_integration.cpp
)
target_link_libraries(integration_heartbeat_B01_memory_core
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
)
target_include_directories(integration_heartbeat_B01_memory_core
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME integration_heartbeat_B01_memory_core
         COMMAND integration_heartbeat_B01_memory_core)
set_tests_properties(integration_heartbeat_B01_memory_core PROPERTIES
    LABELS "integration;fault_tolerance;heartbeat;B01;memory_core"
    TIMEOUT 180
)

# Heartbeat B02 Memory (non-core) Integration Tests (Phase 8: Per-Batch Heartbeat Testing)
add_executable(integration_heartbeat_B02_memory
    test_heartbeat_B02_memory_integration.cpp
)
target_link_libraries(integration_heartbeat_B02_memory
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
)
target_include_directories(integration_heartbeat_B02_memory
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME integration_heartbeat_B02_memory
         COMMAND integration_heartbeat_B02_memory)
set_tests_properties(integration_heartbeat_B02_memory PROPERTIES
    LABELS "integration;fault_tolerance;heartbeat;B02;memory"
    TIMEOUT 180
)
