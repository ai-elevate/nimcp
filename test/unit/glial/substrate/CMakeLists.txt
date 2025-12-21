# ============================================================================
# Glial Substrate Bridge Unit Tests
# ============================================================================
# WHAT: Unit tests for glial substrate bridge module
# WHY:  Ensure glial substrate correctly integrates with substrate mechanics
# HOW:  Test lifecycle, effects, substrate-glial interactions
# ============================================================================

add_executable(unit_glial_substrate_bridge
    test_glial_substrate_bridge.cpp
)

target_link_libraries(unit_glial_substrate_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_glial_substrate_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_glial_substrate_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_glial_substrate_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;glial;substrate;bridge"
        TIMEOUT 120
)
