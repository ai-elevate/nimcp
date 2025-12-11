# Brain Immune System Integration Tests

# Integration tests for brain immune with FT/Security modules
add_executable(integration_cognitive_immune_brain_immune
    test_brain_immune_integration.cpp
)
target_link_libraries(integration_cognitive_immune_brain_immune
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_cognitive_immune_brain_immune
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
add_test(NAME integration_cognitive_immune_brain_immune
         COMMAND integration_cognitive_immune_brain_immune)
set_tests_properties(integration_cognitive_immune_brain_immune PROPERTIES LABELS "integration")
