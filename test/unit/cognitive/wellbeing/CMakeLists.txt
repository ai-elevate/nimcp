# Cognitive Wellbeing Unit Tests

# Basic Wellbeing Tests
add_executable(unit_cognitive_wellbeing_basic
    test_wellbeing.cpp
)
target_link_libraries(unit_cognitive_wellbeing_basic
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_compile_definitions(unit_cognitive_wellbeing_basic PRIVATE NIMCP_TESTING)
target_include_directories(unit_cognitive_wellbeing_basic
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/wellbeing
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_cognitive_wellbeing_basic
         COMMAND unit_cognitive_wellbeing_basic --gtest_brief=1)
set_tests_properties(unit_cognitive_wellbeing_basic PROPERTIES LABELS "unit;wellbeing")

# Comprehensive Wellbeing Tests
add_executable(unit_cognitive_wellbeing_comprehensive
    test_wellbeing_comprehensive.cpp
)
target_link_libraries(unit_cognitive_wellbeing_comprehensive
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_compile_definitions(unit_cognitive_wellbeing_comprehensive PRIVATE NIMCP_TESTING)
target_include_directories(unit_cognitive_wellbeing_comprehensive
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/wellbeing
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_cognitive_wellbeing_comprehensive
         COMMAND unit_cognitive_wellbeing_comprehensive --gtest_brief=1)
set_tests_properties(unit_cognitive_wellbeing_comprehensive PROPERTIES LABELS "unit;wellbeing")

# Coverage Wellbeing Tests
add_executable(unit_cognitive_wellbeing_coverage
    test_wellbeing_coverage.cpp
)
target_link_libraries(unit_cognitive_wellbeing_coverage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_compile_definitions(unit_cognitive_wellbeing_coverage PRIVATE NIMCP_TESTING)
target_include_directories(unit_cognitive_wellbeing_coverage
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/wellbeing
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_cognitive_wellbeing_coverage
         COMMAND unit_cognitive_wellbeing_coverage --gtest_brief=1)
set_tests_properties(unit_cognitive_wellbeing_coverage PROPERTIES LABELS "unit;wellbeing")

# Extended Wellbeing Tests
add_executable(unit_cognitive_wellbeing_extended
    test_wellbeing_extended.cpp
)
target_link_libraries(unit_cognitive_wellbeing_extended
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_compile_definitions(unit_cognitive_wellbeing_extended PRIVATE NIMCP_TESTING)
target_include_directories(unit_cognitive_wellbeing_extended
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/wellbeing
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_cognitive_wellbeing_extended
         COMMAND unit_cognitive_wellbeing_extended --gtest_brief=1)
set_tests_properties(unit_cognitive_wellbeing_extended PROPERTIES LABELS "unit;wellbeing")

# Real Wellbeing Tests
add_executable(unit_cognitive_wellbeing_real
    test_wellbeing_real.cpp
)
target_link_libraries(unit_cognitive_wellbeing_real
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_compile_definitions(unit_cognitive_wellbeing_real PRIVATE NIMCP_TESTING)
target_include_directories(unit_cognitive_wellbeing_real
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/wellbeing
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_cognitive_wellbeing_real
         COMMAND unit_cognitive_wellbeing_real --gtest_brief=1)
set_tests_properties(unit_cognitive_wellbeing_real PROPERTIES LABELS "unit;wellbeing")

# Immune Integration Tests
add_executable(unit_cognitive_wellbeing_immune_integration
    test_wellbeing_immune_integration.cpp
)
target_link_libraries(unit_cognitive_wellbeing_immune_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_compile_definitions(unit_cognitive_wellbeing_immune_integration PRIVATE NIMCP_TESTING)
target_include_directories(unit_cognitive_wellbeing_immune_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/wellbeing
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_cognitive_wellbeing_immune_integration
         COMMAND unit_cognitive_wellbeing_immune_integration --gtest_brief=1)
set_tests_properties(unit_cognitive_wellbeing_immune_integration PROPERTIES
    LABELS "unit;wellbeing;immune;integration"
    TIMEOUT 120
)

# Full Wellbeing-Immune Integration Tests
add_executable(unit_cognitive_wellbeing_immune_full_integration
    test_wellbeing_immune_full_integration.cpp
)
target_link_libraries(unit_cognitive_wellbeing_immune_full_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_compile_definitions(unit_cognitive_wellbeing_immune_full_integration PRIVATE NIMCP_TESTING)
target_include_directories(unit_cognitive_wellbeing_immune_full_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/wellbeing
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_cognitive_wellbeing_immune_full_integration
         COMMAND unit_cognitive_wellbeing_immune_full_integration --gtest_brief=1)
set_tests_properties(unit_cognitive_wellbeing_immune_full_integration PROPERTIES
    LABELS "unit;wellbeing;immune;integration"
    TIMEOUT 120
)

# Enhanced Wellbeing Tests
add_executable(unit_cognitive_wellbeing_enhanced
    test_wellbeing_enhanced.cpp
)
target_link_libraries(unit_cognitive_wellbeing_enhanced
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_compile_definitions(unit_cognitive_wellbeing_enhanced PRIVATE NIMCP_TESTING)
target_include_directories(unit_cognitive_wellbeing_enhanced
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/wellbeing
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_cognitive_wellbeing_enhanced
         COMMAND unit_cognitive_wellbeing_enhanced --gtest_brief=1)
set_tests_properties(unit_cognitive_wellbeing_enhanced PROPERTIES
    LABELS "unit;wellbeing;enhanced"
    TIMEOUT 180
)
