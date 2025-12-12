# Glial-Immune Bridge Unit Tests

# Glial Immune Bridges Unit Tests (Astrocyte + Microglia)
add_executable(unit_glial_immune_bridges
    test_glial_immune_bridges.cpp
)
target_link_libraries(unit_glial_immune_bridges
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_glial_immune_bridges
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/glial
)
add_test(NAME unit_glial_immune_bridges
         COMMAND unit_glial_immune_bridges)
set_tests_properties(unit_glial_immune_bridges PROPERTIES LABELS "unit")
