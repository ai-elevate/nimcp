# ============================================================================
# Brain Module Bio-Async Integration Tests
# ============================================================================
#
# WHAT: Integration tests for brain module bio-async communication
# WHY:  Verify brain creation, processing, and message routing via bio-async
# HOW:  Create brain instances, send messages, verify proper integration
#

# Test 1: Brain Bio-Async Integration (Cross-module communication)
add_executable(integration_brain_bio_async_test
    test_brain_bio_async_integration.cpp
)

target_link_libraries(integration_brain_bio_async_test
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)

target_include_directories(integration_brain_bio_async_test
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)

gtest_discover_tests(integration_brain_bio_async_test)

# Test 2: Brain Persistence Integration (Bio-async event notifications)
add_executable(integration_brain_persistence_test
    test_brain_persistence_integration.cpp
)

target_link_libraries(integration_brain_persistence_test
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)

target_include_directories(integration_brain_persistence_test
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)

gtest_discover_tests(integration_brain_persistence_test)

# Test 3: Brain Coordinator Integration (All 6 coordinators working together)
add_executable(integration_brain_coordinator_test
    test_brain_coordinator_integration.cpp
)

target_link_libraries(integration_brain_coordinator_test
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)

target_include_directories(integration_brain_coordinator_test
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)

gtest_discover_tests(integration_brain_coordinator_test)
