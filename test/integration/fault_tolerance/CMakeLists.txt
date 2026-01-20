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
