# VCS Integration Unit Tests

# VCS Integration Unit Tests
add_executable(unit_utils_vcs_integration
    test_vcs_integration.cpp
)
target_link_libraries(unit_utils_vcs_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_utils_vcs_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_utils_vcs_integration
         COMMAND unit_utils_vcs_integration)
set_tests_properties(unit_utils_vcs_integration PROPERTIES LABELS "unit;vcs;self-repair")
