# ============================================================================
# Brain Factory Integration Tests
# ============================================================================

# Test 1: Factory Integration - Complete workflows
add_executable(integration_core_brain_factory_test_factory_integration
    test_factory_integration.cpp
)
target_link_libraries(integration_core_brain_factory_test_factory_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_core_brain_factory_test_factory_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME integration_core_brain_factory_test_factory_integration
         COMMAND integration_core_brain_factory_test_factory_integration)
set_tests_properties(integration_core_brain_factory_test_factory_integration PROPERTIES LABELS "integration")
