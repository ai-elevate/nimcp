# ============================================================================
# Security Unit Tests - Blood-Brain Barrier (Phase IS-1)
# ============================================================================
# WHAT: Unit tests for Neural Immune System perimeter defense
# WHY:  Verify individual BBB module functionality and security guarantees
# HOW:  Test each module in isolation: input gate, code signing, memory, access
# ============================================================================

# Blood-Brain Barrier Unit Tests (79 tests)
add_executable(unit_security_test_blood_brain_barrier
    test_blood_brain_barrier.cpp
)
target_link_libraries(unit_security_test_blood_brain_barrier
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_blood_brain_barrier
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_security_test_blood_brain_barrier
         COMMAND unit_security_test_blood_brain_barrier)
set_tests_properties(unit_security_test_blood_brain_barrier PROPERTIES
    LABELS "unit"
    TIMEOUT 60
)

# General Security Unit Tests (legacy)
add_executable(unit_security_test_security
    test_security.cpp
)
target_link_libraries(unit_security_test_security
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_security
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_security_test_security
         COMMAND unit_security_test_security)
set_tests_properties(unit_security_test_security PROPERTIES LABELS "unit")

# Comprehensive Security Unit Tests (legacy)
add_executable(unit_security_test_security_comprehensive
    test_security_comprehensive.cpp
)
target_link_libraries(unit_security_test_security_comprehensive
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_security_comprehensive
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_security_test_security_comprehensive
         COMMAND unit_security_test_security_comprehensive)
set_tests_properties(unit_security_test_security_comprehensive PROPERTIES
    LABELS "unit"
    TIMEOUT 60
)

# Security Coverage Framework Unit Tests (Phase SC-1)
add_executable(unit_security_test_security_coverage
    test_security_coverage.cpp
)
target_link_libraries(unit_security_test_security_coverage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_security_coverage
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_security_test_security_coverage
         COMMAND unit_security_test_security_coverage)
set_tests_properties(unit_security_test_security_coverage PROPERTIES
    LABELS "unit;security;sc1"
    TIMEOUT 120
)

# ============================================================================
# Security Integration Framework Unit Tests (Phase SC-4)
# ============================================================================
# WHAT: Unit tests for Universal Security Integration Framework
# WHY:  Verify module registration, trust management, DP queries, events
# HOW:  Test integration API for all module categories
# ============================================================================

add_executable(unit_security_test_security_integration
    test_security_integration.cpp
)
target_link_libraries(unit_security_test_security_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(unit_security_test_security_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_security_test_security_integration
         COMMAND unit_security_test_security_integration)
set_tests_properties(unit_security_test_security_integration PROPERTIES
    LABELS "unit;security;sc4"
    TIMEOUT 120
)
