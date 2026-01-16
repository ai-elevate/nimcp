# LGSS Unit Tests

# Test files
set(LGSS_UNIT_TESTS
    test_lgss_safety_kb.cpp
    test_lgss_context.cpp
)

# Create test executable
add_executable(test_lgss_unit ${LGSS_UNIT_TESTS})

target_link_libraries(test_lgss_unit
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(test_lgss_unit
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Add test
add_test(NAME lgss_unit_tests COMMAND test_lgss_unit)

# Set test properties
set_tests_properties(lgss_unit_tests PROPERTIES
    TIMEOUT 60
    LABELS "unit;security;lgss"
)
