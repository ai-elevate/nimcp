# Ethics Unit Tests
# ============================================================================
# Tests for Ethics System Bridges and Integration
# ============================================================================

# ============================================================================
# Ethics-SNN Bridge Tests
# ============================================================================
# Tests for bidirectional ethics - SNN module integration

add_executable(unit_cognitive_ethics_test_ethics_snn_bridge
    test_ethics_snn_bridge.cpp
)
target_link_libraries(unit_cognitive_ethics_test_ethics_snn_bridge
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        nimcp
        m
        pthread
)
target_include_directories(unit_cognitive_ethics_test_ethics_snn_bridge
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/cognitive/ethics
        ${CMAKE_SOURCE_DIR}/src/snn
)
target_compile_features(unit_cognitive_ethics_test_ethics_snn_bridge PRIVATE cxx_std_17)
add_test(NAME unit_cognitive_ethics_test_ethics_snn_bridge
         COMMAND unit_cognitive_ethics_test_ethics_snn_bridge)
set_tests_properties(unit_cognitive_ethics_test_ethics_snn_bridge PROPERTIES
    TIMEOUT 120
    LABELS "unit;cognitive;ethics;snn;bridge"
)

# ============================================================================
# Ethics-Plasticity Bridge Tests
# ============================================================================
# Tests for bidirectional ethics - plasticity module integration

add_executable(unit_cognitive_ethics_test_ethics_plasticity_bridge
    test_ethics_plasticity_bridge.cpp
)
target_link_libraries(unit_cognitive_ethics_test_ethics_plasticity_bridge
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        nimcp
        m
        pthread
)
target_include_directories(unit_cognitive_ethics_test_ethics_plasticity_bridge
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/cognitive/ethics
        ${CMAKE_SOURCE_DIR}/src/plasticity
)
target_compile_features(unit_cognitive_ethics_test_ethics_plasticity_bridge PRIVATE cxx_std_17)
add_test(NAME unit_cognitive_ethics_test_ethics_plasticity_bridge
         COMMAND unit_cognitive_ethics_test_ethics_plasticity_bridge)
set_tests_properties(unit_cognitive_ethics_test_ethics_plasticity_bridge PROPERTIES
    TIMEOUT 120
    LABELS "unit;cognitive;ethics;plasticity;bridge"
)
