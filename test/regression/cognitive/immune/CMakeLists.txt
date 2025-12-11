# Brain Immune System Regression Tests

# Brain Immune Regression Tests
add_executable(regression_cognitive_immune_brain_immune
    test_brain_immune_regression.cpp
)
target_link_libraries(regression_cognitive_immune_brain_immune
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_cognitive_immune_brain_immune
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/fault_tolerance
        ${CMAKE_SOURCE_DIR}/src/swarm
        ${CMAKE_SOURCE_DIR}/src/async
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
)
add_test(NAME regression_cognitive_immune_brain_immune
         COMMAND regression_cognitive_immune_brain_immune)
set_tests_properties(regression_cognitive_immune_brain_immune PROPERTIES LABELS "regression")
