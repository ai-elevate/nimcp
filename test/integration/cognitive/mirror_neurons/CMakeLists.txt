# Cognitive Mirror Neurons Integration Tests
# ============================================================================
# Phase 10.11 Mirror Neurons - Integration Testing
# ============================================================================

# Mirror Substrate Integration Tests
add_executable(integration_cognitive_mirror_neurons_substrate
    test_mirror_substrate_integration.cpp
)
target_link_libraries(integration_cognitive_mirror_neurons_substrate
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_cognitive_mirror_neurons_substrate
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/core/brain
        ${CMAKE_SOURCE_DIR}/src/core/axon
        ${CMAKE_SOURCE_DIR}/src/core/dendrite
        ${CMAKE_SOURCE_DIR}/src/glial/myelin_sheath
        ${CMAKE_SOURCE_DIR}/src/glial/astrocytes
        ${CMAKE_SOURCE_DIR}/src/glial/oligodendrocytes
        ${CMAKE_SOURCE_DIR}/src/glial/microglia
        ${CMAKE_SOURCE_DIR}/src/cognitive/mirror_neurons
)
add_test(NAME integration_cognitive_mirror_neurons_substrate
         COMMAND integration_cognitive_mirror_neurons_substrate)
set_tests_properties(integration_cognitive_mirror_neurons_substrate PROPERTIES LABELS "integration")

