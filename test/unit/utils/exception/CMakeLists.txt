#=============================================================================
# CMakeLists.txt - Exception Handling Unit Tests
#=============================================================================
# WHAT: Build tests for exception handling system
# WHY:  Verify exception creation, handlers, and immune integration
# HOW:  Link with Check framework and nimcp library
#=============================================================================

# Find Check library using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(CHECK REQUIRED check)

if(NOT CHECK_FOUND)
    message(STATUS "Check library not found - skipping exception tests")
    return()
endif()

# Exception handling test
add_executable(unit_utils_exception_test_exception
    test_exception.c
)

target_include_directories(unit_utils_exception_test_exception PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CHECK_INCLUDE_DIRS}
)

target_link_libraries(unit_utils_exception_test_exception
    nimcp
    ${CHECK_LIBRARIES}
    pthread
    m
)

add_test(
    NAME unit_utils_exception_test_exception
    COMMAND unit_utils_exception_test_exception
)

set_tests_properties(unit_utils_exception_test_exception
    PROPERTIES
    LABELS "unit;utils;exception"
    TIMEOUT 120
)
