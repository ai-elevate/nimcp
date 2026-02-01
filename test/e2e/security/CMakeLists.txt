# ============================================================================
# Security E2E Tests - AI Safety Enhancement Pipeline
# ============================================================================
# WHAT: End-to-end tests for complete AI safety infrastructure
# WHY:  Verify all 10 safety modules work together as unified system
# HOW:  Test full lifecycle: initialization, monitoring, threat response,
#       value preservation, recovery, and graceful shutdown
# ============================================================================

# Safety Enhancement E2E Test
add_executable(e2e_security_test_safety_enhancement
    test_safety_enhancement_e2e.cpp
)
target_link_libraries(e2e_security_test_safety_enhancement
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(e2e_security_test_safety_enhancement
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(e2e_security_test_safety_enhancement PRIVATE cxx_std_17)
add_test(NAME e2e_security_test_safety_enhancement
         COMMAND e2e_security_test_safety_enhancement)
set_tests_properties(e2e_security_test_safety_enhancement PROPERTIES
    LABELS "e2e;security;safety;enhancement"
    TIMEOUT 600
)

message(STATUS "Added Security Safety Enhancement E2E tests")
