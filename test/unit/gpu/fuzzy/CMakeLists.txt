# ============================================================================
# Fuzzy GPU Unit Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for fuzzy GPU unit tests
# WHY:  Enable testing of fuzzy logic GPU kernels
# HOW:  Link against Check framework and nimcp library
# ============================================================================

# Guard: Check if Check framework is available
find_package(Check QUIET)
if(NOT Check_FOUND)
    message(STATUS "Check framework not found - skipping fuzzy GPU unit tests")
    return()
endif()

# ============================================================================
# Test: Fuzzy GPU Membership Functions
# ============================================================================
add_executable(test_fuzzy_gpu_mf
    test_fuzzy_gpu_mf.c
)

target_link_libraries(test_fuzzy_gpu_mf
    PRIVATE
        nimcp
        Check::check
        m
        pthread
)

target_include_directories(test_fuzzy_gpu_mf
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_fuzzy_gpu_mf COMMAND test_fuzzy_gpu_mf)
set_tests_properties(unit_fuzzy_gpu_mf PROPERTIES
    TIMEOUT 60
    LABELS "unit;gpu;fuzzy"
)

# ============================================================================
# Test: Fuzzy GPU Defuzzification
# ============================================================================
add_executable(test_fuzzy_gpu_defuzz
    test_fuzzy_gpu_defuzz.c
)

target_link_libraries(test_fuzzy_gpu_defuzz
    PRIVATE
        nimcp
        Check::check
        m
        pthread
)

target_include_directories(test_fuzzy_gpu_defuzz
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_fuzzy_gpu_defuzz COMMAND test_fuzzy_gpu_defuzz)
set_tests_properties(unit_fuzzy_gpu_defuzz PROPERTIES
    TIMEOUT 60
    LABELS "unit;gpu;fuzzy"
)
