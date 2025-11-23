# Cognitive Introspection Regression Tests (Phase 1.5.4)

# Connectivity Health Regression Tests
add_executable(regression_cognitive_introspection_connectivity_health
    test_connectivity_health_regression.cpp
)
target_link_libraries(regression_cognitive_introspection_connectivity_health
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_cognitive_introspection_connectivity_health
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/core/brain_regions
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
)
add_test(NAME regression_cognitive_introspection_connectivity_health
         COMMAND regression_cognitive_introspection_connectivity_health)
set_tests_properties(regression_cognitive_introspection_connectivity_health PROPERTIES LABELS "regression")
