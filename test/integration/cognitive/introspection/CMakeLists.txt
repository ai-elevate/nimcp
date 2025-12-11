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

# Auto Activity History Integration Tests
add_executable(integration_cognitive_introspection_auto_activity_history
    test_auto_activity_history_integration.cpp
)
target_link_libraries(integration_cognitive_introspection_auto_activity_history
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_cognitive_introspection_auto_activity_history
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
)
add_test(NAME integration_cognitive_introspection_auto_activity_history
         COMMAND integration_cognitive_introspection_auto_activity_history)
set_tests_properties(integration_cognitive_introspection_auto_activity_history PROPERTIES LABELS "integration")

# Consciousness Metrics Integration Tests
add_executable(integration_cognitive_introspection_consciousness_metrics
    test_consciousness_metrics_integration.cpp
)
target_link_libraries(integration_cognitive_introspection_consciousness_metrics
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_cognitive_introspection_consciousness_metrics
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
        ${CMAKE_SOURCE_DIR}/src/async
)
add_test(NAME integration_cognitive_introspection_consciousness_metrics
         COMMAND integration_cognitive_introspection_consciousness_metrics)
set_tests_properties(integration_cognitive_introspection_consciousness_metrics PROPERTIES LABELS "integration")

# Ensemble Uncertainty Integration Tests
add_executable(integration_cognitive_introspection_ensemble_uncertainty
    test_ensemble_uncertainty_integration.cpp
)
target_link_libraries(integration_cognitive_introspection_ensemble_uncertainty
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_cognitive_introspection_ensemble_uncertainty
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
)
add_test(NAME integration_cognitive_introspection_ensemble_uncertainty
         COMMAND integration_cognitive_introspection_ensemble_uncertainty)
set_tests_properties(integration_cognitive_introspection_ensemble_uncertainty PROPERTIES LABELS "integration")

# Temporal Patterns Integration Tests
add_executable(integration_cognitive_introspection_temporal_patterns
    test_temporal_patterns_integration.cpp
)
target_link_libraries(integration_cognitive_introspection_temporal_patterns
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_cognitive_introspection_temporal_patterns
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/cognitive/introspection
)
add_test(NAME integration_cognitive_introspection_temporal_patterns
         COMMAND integration_cognitive_introspection_temporal_patterns)
set_tests_properties(integration_cognitive_introspection_temporal_patterns PROPERTIES LABELS "integration")
