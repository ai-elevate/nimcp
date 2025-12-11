# Brain Immune System Unit Tests

# Brain Immune Unit Tests
add_executable(unit_cognitive_immune_brain_immune
    test_brain_immune.cpp
)
target_link_libraries(unit_cognitive_immune_brain_immune
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_brain_immune
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
)
add_test(NAME unit_cognitive_immune_brain_immune
         COMMAND unit_cognitive_immune_brain_immune)
set_tests_properties(unit_cognitive_immune_brain_immune PROPERTIES LABELS "unit")
