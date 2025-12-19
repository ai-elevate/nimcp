# Cognitive Substrate Bridges Integration Tests
# Tests integration of all 8 cognitive substrate bridges with shared neural substrate

add_executable(integration_cognitive_substrate_bridges
    test_cognitive_substrate_bridges_integration.cpp
)

target_link_libraries(integration_cognitive_substrate_bridges
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)

target_include_directories(integration_cognitive_substrate_bridges
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/core/neural_substrate
        ${CMAKE_SOURCE_DIR}/include/cognitive/attention
        ${CMAKE_SOURCE_DIR}/include/cognitive/emotion
        ${CMAKE_SOURCE_DIR}/include/cognitive/executive
        ${CMAKE_SOURCE_DIR}/include/cognitive/introspection
        ${CMAKE_SOURCE_DIR}/include/cognitive/memory
        ${CMAKE_SOURCE_DIR}/include/cognitive/reasoning
        ${CMAKE_SOURCE_DIR}/include/cognitive/tom
        ${CMAKE_SOURCE_DIR}/include/cognitive/working_memory
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/include/utils/thread
)

add_test(NAME integration_cognitive_substrate_bridges
         COMMAND integration_cognitive_substrate_bridges)

set_tests_properties(integration_cognitive_substrate_bridges
    PROPERTIES
        LABELS "integration"
        TIMEOUT 120
)
