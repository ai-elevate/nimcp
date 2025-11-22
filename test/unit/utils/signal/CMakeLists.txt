#=============================================================================
# CMakeLists.txt - Signal Filter Unit Tests
#=============================================================================

# Signal Filter Unit Tests
add_executable(unit_utils_signal_test_signal_filter
    test_signal_filter.cpp
)

target_link_libraries(unit_utils_signal_test_signal_filter
    nimcp
    GTest::gtest
    GTest::gtest_main
)

add_test(
    NAME unit_utils_signal_test_signal_filter
    COMMAND unit_utils_signal_test_signal_filter
)

set_tests_properties(unit_utils_signal_test_signal_filter
    PROPERTIES
    LABELS "unit;utils;signal"
    TIMEOUT 60
)
