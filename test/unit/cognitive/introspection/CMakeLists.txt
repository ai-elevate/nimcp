# Cognitive Introspection Unit Tests (Phase 1.5.4)

# Connectivity Health Unit Tests
add_executable(unit_cognitive_introspection_connectivity_health
    test_connectivity_health.cpp
)
target_link_libraries(unit_cognitive_introspection_connectivity_health
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_introspection_connectivity_health
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/core/brain_regions
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
)
add_test(NAME unit_cognitive_introspection_connectivity_health
         COMMAND unit_cognitive_introspection_connectivity_health)
set_tests_properties(unit_cognitive_introspection_connectivity_health PROPERTIES LABELS "unit")
