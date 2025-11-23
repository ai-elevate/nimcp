# Cognitive Introspection Integration Tests (Phase 1.5.4)

# Connectivity Health Integration Tests
add_executable(integration_cognitive_introspection_connectivity_health
    test_connectivity_health_integration.cpp
)
target_link_libraries(integration_cognitive_introspection_connectivity_health
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_cognitive_introspection_connectivity_health
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/core/brain_regions
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
)
add_test(NAME integration_cognitive_introspection_connectivity_health
         COMMAND integration_cognitive_introspection_connectivity_health)
set_tests_properties(integration_cognitive_introspection_connectivity_health PROPERTIES LABELS "integration")
