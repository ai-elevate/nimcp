# ============================================================================
# pH Dynamics Unit Tests
# ============================================================================
# WHAT: Unit tests for pH dynamics module
# WHY:  Verify pH regulation, buffer systems, and pump operations
# HOW:  GTest-based tests for lifecycle, update, and effects

message(STATUS "Adding pH dynamics unit tests...")

# pH Dynamics test
add_executable(unit_chemistry_ph_test_ph_dynamics
    test_ph_dynamics.cpp
)

target_include_directories(unit_chemistry_ph_test_ph_dynamics PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(unit_chemistry_ph_test_ph_dynamics
    nimcp
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME unit_chemistry_ph_test_ph_dynamics
         COMMAND unit_chemistry_ph_test_ph_dynamics)
