# Security Immune System Unit Tests

# Pattern Database Immune Unit Tests
add_executable(unit_security_immune_pattern_db
    test_pattern_db_immune.cpp
)
target_link_libraries(unit_security_immune_pattern_db
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_immune_pattern_db
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME unit_security_immune_pattern_db
         COMMAND unit_security_immune_pattern_db)
set_tests_properties(unit_security_immune_pattern_db PROPERTIES LABELS "unit")
